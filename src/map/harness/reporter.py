"""
Report generation for the graft testing harness.

Generates JSON, Markdown, and CSV reports from evaluation results.
"""

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import HarnessConfig
from .evaluator import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class HarnessReport:
    """Complete harness evaluation report."""
    timestamp: str
    target_model: str
    judge_model: str
    config: dict

    # Summary statistics
    concepts_evaluated: int
    known_count: int
    unknown_count: int
    mean_score: float
    median_score: float
    min_score: float
    max_score: float
    std_score: float

    # Score distribution
    score_distribution: Dict[str, int]  # e.g., "0-3": 10, "4-6": 20

    # Per-concept results
    results: List[dict]

    # Concepts with training data
    trainable_unknown_count: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "target_model": self.target_model,
            "judge_model": self.judge_model,
            "config": self.config,
            "summary": {
                "concepts_evaluated": self.concepts_evaluated,
                "known": self.known_count,
                "unknown": self.unknown_count,
                "trainable_unknown": self.trainable_unknown_count,
                "mean_score": self.mean_score,
                "median_score": self.median_score,
                "min_score": self.min_score,
                "max_score": self.max_score,
                "std_score": self.std_score,
                "score_distribution": self.score_distribution,
            },
            "results": self.results,
        }


class HarnessReporter:
    """Generates reports from evaluation results."""

    def __init__(self, config: HarnessConfig):
        self.config = config

    def _compute_statistics(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, float]:
        """Compute summary statistics from results."""
        if not results:
            return {
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
            }

        scores = [r.score for r in results]
        n = len(scores)
        mean = sum(scores) / n

        sorted_scores = sorted(scores)
        if n % 2 == 0:
            median = (sorted_scores[n//2 - 1] + sorted_scores[n//2]) / 2
        else:
            median = sorted_scores[n//2]

        variance = sum((s - mean) ** 2 for s in scores) / n
        std = variance ** 0.5

        return {
            "mean": round(mean, 2),
            "median": round(median, 2),
            "min": round(min(scores), 2),
            "max": round(max(scores), 2),
            "std": round(std, 2),
        }

    def _compute_distribution(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, int]:
        """Compute score distribution buckets."""
        distribution = {
            "0-3": 0,
            "4-6": 0,
            "7-9": 0,
            "10": 0,
        }

        for r in results:
            if r.score <= 3:
                distribution["0-3"] += 1
            elif r.score <= 6:
                distribution["4-6"] += 1
            elif r.score <= 9:
                distribution["7-9"] += 1
            else:
                distribution["10"] += 1

        return distribution

    def create_report(
        self,
        results: List[EvaluationResult],
    ) -> HarnessReport:
        """Create a report from evaluation results."""
        stats = self._compute_statistics(results)
        distribution = self._compute_distribution(results)

        known = [r for r in results if r.knows_concept]
        unknown = [r for r in results if not r.knows_concept]
        trainable_unknown = [r for r in unknown if r.has_training_data]

        return HarnessReport(
            timestamp=datetime.now().isoformat(),
            target_model=self.config.target_model_id,
            judge_model=self.config.judge_model_id,
            config=self.config.to_dict(),
            concepts_evaluated=len(results),
            known_count=len(known),
            unknown_count=len(unknown),
            mean_score=stats["mean"],
            median_score=stats["median"],
            min_score=stats["min"],
            max_score=stats["max"],
            std_score=stats["std"],
            score_distribution=distribution,
            results=[r.to_dict() for r in results],
            trainable_unknown_count=len(trainable_unknown),
        )

    def save_json(
        self,
        report: HarnessReport,
        output_path: Path,
    ) -> Path:
        """Save report as JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Saved JSON report to {output_path}")
        return output_path

    def save_markdown(
        self,
        report: HarnessReport,
        output_path: Path,
    ) -> Path:
        """Save report as Markdown."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Graft Harness Evaluation Report",
            "",
            f"**Timestamp:** {report.timestamp}",
            f"**Target Model:** {report.target_model}",
            f"**Judge Model:** {report.judge_model}",
            "",
            "## Summary",
            "",
            f"- **Concepts Evaluated:** {report.concepts_evaluated}",
            f"- **Known:** {report.known_count} ({100*report.known_count/max(report.concepts_evaluated, 1):.1f}%)",
            f"- **Unknown:** {report.unknown_count} ({100*report.unknown_count/max(report.concepts_evaluated, 1):.1f}%)",
            f"- **Trainable Unknown:** {report.trainable_unknown_count}",
            "",
            "## Score Statistics",
            "",
            f"- **Mean:** {report.mean_score}",
            f"- **Median:** {report.median_score}",
            f"- **Min:** {report.min_score}",
            f"- **Max:** {report.max_score}",
            f"- **Std Dev:** {report.std_score}",
            "",
            "## Score Distribution",
            "",
            "| Range | Count |",
            "|-------|-------|",
        ]

        for range_str, count in report.score_distribution.items():
            lines.append(f"| {range_str} | {count} |")

        lines.extend([
            "",
            "## Top Known Concepts",
            "",
            "| Concept | Score | Layer |",
            "|---------|-------|-------|",
        ])

        # Top 10 known
        known_results = sorted(
            [r for r in report.results if r["knows_concept"]],
            key=lambda r: -r["score"],
        )[:10]

        for r in known_results:
            lines.append(f"| {r['concept_term']} | {r['score']:.1f} | {r['layer']} |")

        lines.extend([
            "",
            "## Top Unknown Concepts (Trainable)",
            "",
            "| Concept | Score | Layer | Training Examples |",
            "|---------|-------|-------|-------------------|",
        ])

        # Top 10 unknown with training data
        unknown_results = sorted(
            [r for r in report.results if not r["knows_concept"] and r["has_training_data"]],
            key=lambda r: r["score"],
        )[:10]

        for r in unknown_results:
            n_examples = r["n_positive_examples"] + r["n_negative_examples"]
            lines.append(f"| {r['concept_term']} | {r['score']:.1f} | {r['layer']} | {n_examples} |")

        content = "\n".join(lines)

        with open(output_path, 'w') as f:
            f.write(content)

        logger.info(f"Saved Markdown report to {output_path}")
        return output_path

    def save_csv(
        self,
        report: HarnessReport,
        output_path: Path,
    ) -> Path:
        """Save results as CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "concept_id",
            "concept_term",
            "layer",
            "score",
            "knows_concept",
            "has_training_data",
            "n_positive_examples",
            "n_negative_examples",
            "reasoning",
        ]

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for r in report.results:
                writer.writerow(r)

        logger.info(f"Saved CSV report to {output_path}")
        return output_path

    def save_all(
        self,
        report: HarnessReport,
        output_dir: Optional[Path] = None,
        prefix: str = "harness_report",
    ) -> Dict[str, Path]:
        """Save report in all formats."""
        output_dir = output_dir or self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{prefix}_{timestamp}"

        paths = {
            "json": self.save_json(report, output_dir / f"{base_name}.json"),
            "markdown": self.save_markdown(report, output_dir / f"{base_name}.md"),
            "csv": self.save_csv(report, output_dir / f"{base_name}.csv"),
        }

        return paths
