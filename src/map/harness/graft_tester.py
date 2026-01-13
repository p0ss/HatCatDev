"""
Graft tester - end-to-end workflow for testing grafts.

Orchestrates evaluation, meld creation, scion training, and re-evaluation.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from ..registry import Concept, load_concept_pack
from ..graft import Scion, ScionConfig, ScionTrainer, Cleft, CleftRegion, merge_clefts
from .config import HarnessConfig
from .models import TargetModel, JudgeModel
from .evaluator import ConceptEvaluator, EvaluationResult
from .reporter import HarnessReporter, HarnessReport
from .calibration import JudgeCalibrator, CalibrationReport
from .meld_designer import MeldDesigner, MeldData

logger = logging.getLogger(__name__)


@dataclass
class GraftResult:
    """Result of grafting a single concept."""
    concept_id: str
    concept_term: str

    # Pre-graft evaluation
    pre_score: float
    pre_knew: bool
    pre_response: str

    # Training
    training_data_source: str  # "pack", "meld", or "generated"
    n_positive_examples: int
    n_negative_examples: int
    scion_id: Optional[str]
    training_loss: Optional[float]

    # Post-graft evaluation
    post_score: float
    post_knew: bool
    post_response: str

    # Analysis
    score_improvement: float
    learned: bool  # Did it go from not knowing to knowing?

    def to_dict(self) -> dict:
        return {
            "concept_id": self.concept_id,
            "concept_term": self.concept_term,
            "pre_score": self.pre_score,
            "pre_knew": self.pre_knew,
            "pre_response": self.pre_response,
            "training_data_source": self.training_data_source,
            "n_positive_examples": self.n_positive_examples,
            "n_negative_examples": self.n_negative_examples,
            "scion_id": self.scion_id,
            "training_loss": self.training_loss,
            "post_score": self.post_score,
            "post_knew": self.post_knew,
            "post_response": self.post_response,
            "score_improvement": self.score_improvement,
            "learned": self.learned,
        }


@dataclass
class GraftTestReport:
    """Complete graft test report."""
    timestamp: str
    config: dict

    # Calibration
    calibration: Optional[dict]

    # Baseline evaluation
    baseline_report: dict

    # Graft results
    concepts_grafted: int
    concepts_learned: int
    learning_rate: float  # Fraction that learned
    mean_improvement: float
    graft_results: List[dict]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "config": self.config,
            "calibration": self.calibration,
            "baseline_report": self.baseline_report,
            "summary": {
                "concepts_grafted": self.concepts_grafted,
                "concepts_learned": self.concepts_learned,
                "learning_rate": self.learning_rate,
                "mean_improvement": self.mean_improvement,
            },
            "graft_results": self.graft_results,
        }


def create_synthetic_cleft(
    concept_id: str,
    hidden_dim: int,
    layer: int,
    top_k_percent: float = 15.0,
) -> Cleft:
    """
    Create a synthetic cleft for testing when no lens is available.

    Uses random but deterministic selection of dimensions based on concept ID.
    """
    import numpy as np

    np.random.seed(hash(concept_id) % (2**32))
    k = int(hidden_dim * top_k_percent / 100)

    # Select random dimensions
    col_indices = sorted(np.random.choice(hidden_dim, k, replace=False).tolist())
    row_indices = sorted(np.random.choice(hidden_dim, k, replace=False).tolist())

    # MLP dimensions are typically 4x hidden_dim
    mlp_dim = hidden_dim * 4

    regions = [
        CleftRegion(
            concept_id=concept_id,
            layer_index=layer,
            component="mlp.up_proj",
            row_indices=list(range(mlp_dim)),  # All MLP outputs
            col_indices=col_indices,  # Selected hidden dims
        ),
        CleftRegion(
            concept_id=concept_id,
            layer_index=layer,
            component="mlp.down_proj",
            row_indices=row_indices,  # Selected hidden dims
            col_indices=list(range(mlp_dim)),  # All MLP outputs
        ),
    ]

    return Cleft(
        concept_id=concept_id,
        regions=regions,
        hidden_dim=hidden_dim,
    )


class GraftTester:
    """
    End-to-end graft testing workflow.

    Evaluates baseline, identifies unknown concepts, grafts them,
    and measures improvement.
    """

    def __init__(self, config: HarnessConfig):
        self.config = config
        self.target: Optional[TargetModel] = None
        self.judge: Optional[JudgeModel] = None
        self.evaluator: Optional[ConceptEvaluator] = None
        self.reporter: Optional[HarnessReporter] = None
        self.meld_designer: Optional[MeldDesigner] = None

    def setup(self) -> None:
        """Initialize models and components."""
        logger.info("Setting up graft tester...")

        # Load models
        self.target = TargetModel(
            model_id=self.config.target_model_id,
            device=self.config.device,
            dtype=self.config.target_dtype,
        )

        self.judge = JudgeModel(
            model_id=self.config.judge_model_id,
            device=self.config.device,
            dtype=self.config.judge_dtype,
            max_retries=self.config.judge_max_retries,
        )

        # Create components
        self.evaluator = ConceptEvaluator(
            target=self.target,
            judge=self.judge,
            config=self.config,
        )

        self.reporter = HarnessReporter(config=self.config)

        self.meld_designer = MeldDesigner(
            judge=self.judge,
            melds_path=self.config.melds_path,
        )

        logger.info("Setup complete")

    def calibrate_judge(self) -> CalibrationReport:
        """Run judge calibration."""
        if not self.judge:
            raise RuntimeError("Call setup() first")

        logger.info("Running judge calibration...")
        calibrator = JudgeCalibrator(
            judge=self.judge,
            accuracy_threshold=0.75,
        )
        return calibrator.run_calibration()

    def evaluate_baseline(
        self,
        max_concepts: Optional[int] = None,
        with_training_data_only: bool = True,
    ) -> Tuple[HarnessReport, List[EvaluationResult]]:
        """Evaluate baseline concept knowledge."""
        if not self.evaluator:
            raise RuntimeError("Call setup() first")

        logger.info("Evaluating baseline concept knowledge...")
        results = self.evaluator.evaluate_all(
            max_concepts=max_concepts,
            with_training_data_only=with_training_data_only,
        )

        report = self.reporter.create_report(results)
        return report, results

    def train_scion(
        self,
        concept: Concept,
        meld_data: MeldData,
    ) -> Optional[Scion]:
        """Train a scion for a concept."""
        if not self.target:
            raise RuntimeError("Call setup() first")

        logger.info(f"Training scion for: {concept.term}")

        # Create synthetic cleft (since we don't have lenses for all concepts)
        layer = concept.layer if hasattr(concept, 'layer') else self.config.layers_to_graft[0]
        cleft = create_synthetic_cleft(
            concept_id=meld_data.concept_id,
            hidden_dim=self.target.hidden_size,
            layer=layer,
        )

        # Create union cleft (single concept)
        union_cleft = merge_clefts([cleft])

        # Prepare dataset
        dataset = {
            "positive": meld_data.positive_examples,
            "negative": meld_data.negative_examples,
        }

        # Training config
        scion_config = ScionConfig(
            learning_rate=self.config.scion_learning_rate,
            epochs=self.config.scion_epochs,
            batch_size=self.config.scion_batch_size,
            injection_layers=self.config.layers_to_graft,
        )

        try:
            trainer = ScionTrainer(
                model=self.target.model,
                tokenizer=self.target.tokenizer,
                union_cleft=union_cleft,
                config=scion_config,
                device=self.config.device,
            )

            scion = trainer.train(
                dataset=dataset,
                concept_id=meld_data.concept_id,
                verbose=True,
            )

            return scion

        except Exception as e:
            logger.error(f"Failed to train scion: {e}")
            return None

    def graft_concept(
        self,
        concept: Concept,
        pre_result: EvaluationResult,
    ) -> GraftResult:
        """
        Graft a single concept and measure improvement.

        Args:
            concept: The concept to graft.
            pre_result: Pre-graft evaluation result.

        Returns:
            GraftResult with before/after comparison.
        """
        # Get training data
        meld_data = self.meld_designer.get_or_create_training_data(
            concept=concept,
            min_examples=5,
        )

        if not meld_data:
            logger.warning(f"No training data for {concept.term}, skipping")
            return GraftResult(
                concept_id=pre_result.concept_id,
                concept_term=concept.term,
                pre_score=pre_result.score,
                pre_knew=pre_result.knows_concept,
                pre_response=pre_result.target_response,
                training_data_source="none",
                n_positive_examples=0,
                n_negative_examples=0,
                scion_id=None,
                training_loss=None,
                post_score=pre_result.score,
                post_knew=pre_result.knows_concept,
                post_response=pre_result.target_response,
                score_improvement=0,
                learned=False,
            )

        # Train scion
        scion = self.train_scion(concept, meld_data)

        if not scion:
            logger.warning(f"Scion training failed for {concept.term}")
            return GraftResult(
                concept_id=pre_result.concept_id,
                concept_term=concept.term,
                pre_score=pre_result.score,
                pre_knew=pre_result.knows_concept,
                pre_response=pre_result.target_response,
                training_data_source=meld_data.source,
                n_positive_examples=len(meld_data.positive_examples),
                n_negative_examples=len(meld_data.negative_examples),
                scion_id=None,
                training_loss=None,
                post_score=pre_result.score,
                post_knew=pre_result.knows_concept,
                post_response=pre_result.target_response,
                score_improvement=0,
                learned=False,
            )

        # Apply scion
        self.target.apply_scion(scion, mode="delta")

        # Re-evaluate
        post_result = self.evaluator.evaluate_concept(concept)

        # Calculate improvement
        score_improvement = post_result.score - pre_result.score
        learned = not pre_result.knows_concept and post_result.knows_concept

        return GraftResult(
            concept_id=pre_result.concept_id,
            concept_term=concept.term,
            pre_score=pre_result.score,
            pre_knew=pre_result.knows_concept,
            pre_response=pre_result.target_response,
            training_data_source=meld_data.source,
            n_positive_examples=len(meld_data.positive_examples),
            n_negative_examples=len(meld_data.negative_examples),
            scion_id=scion.scion_id,
            training_loss=scion.metrics.get("final_loss") if scion.metrics else None,
            post_score=post_result.score,
            post_knew=post_result.knows_concept,
            post_response=post_result.target_response,
            score_improvement=score_improvement,
            learned=learned,
        )

    def run_full_test(
        self,
        max_baseline_concepts: Optional[int] = None,
        max_graft_concepts: int = 5,
        run_calibration: bool = True,
        save_results: bool = True,
    ) -> GraftTestReport:
        """
        Run the complete graft testing workflow.

        1. Calibrate judge (optional)
        2. Evaluate baseline
        3. Select unknown concepts
        4. Graft each and measure improvement
        5. Generate report

        Args:
            max_baseline_concepts: Max concepts for baseline evaluation.
            max_graft_concepts: Max concepts to graft.
            run_calibration: Whether to run judge calibration.
            save_results: Whether to save results to disk.

        Returns:
            GraftTestReport with full results.
        """
        self.setup()

        # Calibration
        calibration_report = None
        if run_calibration:
            calibration_report = self.calibrate_judge()
            if not calibration_report.is_calibrated:
                logger.warning(calibration_report.recommendation)

        # Baseline evaluation
        baseline_report, baseline_results = self.evaluate_baseline(
            max_concepts=max_baseline_concepts,
            with_training_data_only=True,
        )

        # Find unknown concepts with training data
        unknown = self.evaluator.find_unknown_concepts(
            baseline_results,
            with_training_data=True,
        )

        logger.info(f"Found {len(unknown)} unknown concepts with training data")

        if not unknown:
            logger.warning("No unknown concepts to graft")
            return GraftTestReport(
                timestamp=datetime.now().isoformat(),
                config=self.config.to_dict(),
                calibration=calibration_report.to_dict() if calibration_report else None,
                baseline_report=baseline_report.to_dict(),
                concepts_grafted=0,
                concepts_learned=0,
                learning_rate=0.0,
                mean_improvement=0.0,
                graft_results=[],
            )

        # Graft selected concepts
        graft_results = []
        concepts_to_graft = unknown[:max_graft_concepts]

        for i, pre_result in enumerate(concepts_to_graft):
            logger.info(f"Grafting concept {i+1}/{len(concepts_to_graft)}: {pre_result.concept_term}")

            # Get concept object
            concept = self.evaluator.pack.get_concept(pre_result.concept_id)
            if not concept:
                # Try by term
                for c in self.evaluator.pack.concepts.values():
                    if c.term == pre_result.concept_term:
                        concept = c
                        break

            if not concept:
                logger.warning(f"Concept not found: {pre_result.concept_id}")
                continue

            result = self.graft_concept(concept, pre_result)
            graft_results.append(result)

            logger.info(
                f"  Score: {result.pre_score:.1f} -> {result.post_score:.1f} "
                f"({'+' if result.score_improvement >= 0 else ''}{result.score_improvement:.1f})"
            )

        # Compute summary statistics
        concepts_grafted = len(graft_results)
        concepts_learned = sum(1 for r in graft_results if r.learned)
        learning_rate = concepts_learned / concepts_grafted if concepts_grafted > 0 else 0
        improvements = [r.score_improvement for r in graft_results]
        mean_improvement = sum(improvements) / len(improvements) if improvements else 0

        report = GraftTestReport(
            timestamp=datetime.now().isoformat(),
            config=self.config.to_dict(),
            calibration=calibration_report.to_dict() if calibration_report else None,
            baseline_report=baseline_report.to_dict(),
            concepts_grafted=concepts_grafted,
            concepts_learned=concepts_learned,
            learning_rate=learning_rate,
            mean_improvement=mean_improvement,
            graft_results=[r.to_dict() for r in graft_results],
        )

        # Save results
        if save_results:
            self._save_report(report)

        return report

    def run_single_graft(
        self,
        concept_id: str,
        save_results: bool = True,
    ) -> GraftResult:
        """
        Graft a single specific concept.

        Args:
            concept_id: The concept ID or term to graft.
            save_results: Whether to save results.

        Returns:
            GraftResult for the concept.
        """
        self.setup()

        # Get concept
        concept = self.evaluator.pack.get_concept(concept_id)
        if not concept:
            # Try by term
            for c in self.evaluator.pack.concepts.values():
                if c.term == concept_id:
                    concept = c
                    break

        if not concept:
            raise ValueError(f"Concept not found: {concept_id}")

        # Evaluate pre-graft
        pre_result = self.evaluator.evaluate_concept(concept)
        logger.info(f"Pre-graft score: {pre_result.score:.1f}")

        # Graft
        result = self.graft_concept(concept, pre_result)

        logger.info(
            f"Result: {result.pre_score:.1f} -> {result.post_score:.1f} "
            f"(learned: {result.learned})"
        )

        if save_results:
            output_dir = self.config.output_dir / "single_grafts"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"graft_{concept_id}_{timestamp}.json"

            with open(output_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)

            logger.info(f"Saved result to {output_path}")

        return result

    def _save_report(self, report: GraftTestReport) -> Path:
        """Save the full test report."""
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"graft_test_{timestamp}.json"

        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Saved report to {output_path}")

        # Also save markdown summary
        md_path = output_dir / f"graft_test_{timestamp}.md"
        self._save_markdown_summary(report, md_path)

        return output_path

    def _save_markdown_summary(
        self,
        report: GraftTestReport,
        output_path: Path,
    ) -> None:
        """Save a markdown summary of the graft test."""
        lines = [
            "# Graft Test Report",
            "",
            f"**Timestamp:** {report.timestamp}",
            f"**Target Model:** {report.config['target_model_id']}",
            f"**Judge Model:** {report.config['judge_model_id']}",
            "",
            "## Summary",
            "",
            f"- **Concepts Grafted:** {report.concepts_grafted}",
            f"- **Concepts Learned:** {report.concepts_learned}",
            f"- **Learning Rate:** {report.learning_rate:.1%}",
            f"- **Mean Improvement:** {report.mean_improvement:+.2f}",
            "",
            "## Graft Results",
            "",
            "| Concept | Pre | Post | Improvement | Learned |",
            "|---------|-----|------|-------------|---------|",
        ]

        for r in report.graft_results:
            learned_str = "Yes" if r["learned"] else "No"
            lines.append(
                f"| {r['concept_term']} | {r['pre_score']:.1f} | "
                f"{r['post_score']:.1f} | {r['score_improvement']:+.1f} | {learned_str} |"
            )

        if report.calibration:
            lines.extend([
                "",
                "## Judge Calibration",
                "",
                f"- **Accuracy:** {report.calibration['accuracy']:.1%}",
                f"- **Calibrated:** {'Yes' if report.calibration['is_calibrated'] else 'No'}",
            ])

        with open(output_path, 'w') as f:
            f.write("\n".join(lines))

        logger.info(f"Saved markdown summary to {output_path}")
