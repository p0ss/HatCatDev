"""
Meld-Based Judge Evaluation

Uses training examples from applied melds as deterministic ground truth
for evaluating judge quality. This tests the exact task a judge does:
"Is this text an example of concept X?"

DATASET PROVENANCE
==================
The examples used for evaluation come from melds/applied/ - these are NOT
arbitrary examples. Every example in this dataset has passed through the
complete HATCAT MELD approval pipeline:

1. Initial generation/authoring with positive and negative examples
2. Automated validation against HATCAT_MELD_POLICY
3. Protection level assessment (STANDARD/ELEVATED/PROTECTED/CRITICAL)
4. Human review for elevated/protected/critical concepts
5. Final approval and application to the concept pack

This provenance makes these examples suitable as ground truth for
deterministic testing of judge models. The examples represent the
quality bar we expect judges to enforce.

Advantages:
- Deterministic scoring (positive examples = yes, negative = no)
- Real concept diversity from training data
- Includes safety-critical concepts with human-reviewed examples
- Scales to thousands of test cases automatically
- Stratified sampling ensures adequate coverage of all risk levels
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
import random

logger = logging.getLogger(__name__)


@dataclass
class MeldExample:
    """A single example from meld training data."""
    concept_term: str
    concept_definition: str
    example_text: str
    is_positive: bool  # Ground truth
    meld_source: str
    safety_tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MeldTestResult:
    """Result of a single meld-based test."""
    example: MeldExample
    judge_verdict: bool  # Judge's answer
    correct: bool        # Did judge match ground truth
    confidence: float    # Judge's confidence if provided
    raw_response: str
    reasoning: str


@dataclass
class MeldEvalReport:
    """Full evaluation report from meld-based testing."""
    model_id: str

    # Core metrics
    total_cases: int
    correct: int
    accuracy: float

    # Breakdown
    true_positives: int   # Correctly identified positive examples
    true_negatives: int   # Correctly identified negative examples
    false_positives: int  # Said yes to negative example
    false_negatives: int  # Said no to positive example

    # By safety level
    accuracy_by_risk: Dict[str, float]

    # Sample counts per risk level (for statistical validity assessment)
    samples_by_risk: Dict[str, int] = field(default_factory=dict)

    # Detailed results
    results: List[MeldTestResult] = field(default_factory=list)

    # Metadata
    melds_used: int = 0
    concepts_tested: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'total_cases': self.total_cases,
            'correct': self.correct,
            'accuracy': self.accuracy,
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'accuracy_by_risk': self.accuracy_by_risk,
            'samples_by_risk': self.samples_by_risk,
            'melds_used': self.melds_used,
            'concepts_tested': self.concepts_tested,
            'timestamp': self.timestamp.isoformat(),
            'precision': self.true_positives / max(self.true_positives + self.false_positives, 1),
            'recall': self.true_positives / max(self.true_positives + self.false_negatives, 1),
        }


class MeldExampleLoader:
    """Loads training examples from applied melds."""

    def __init__(self, melds_dir: Path = Path("melds/applied")):
        self.melds_dir = Path(melds_dir)
        self.examples: List[MeldExample] = []
        self.concepts_by_term: Dict[str, List[MeldExample]] = {}

    def load_all(self) -> int:
        """Load all examples from all meld files. Returns count loaded."""
        if not self.melds_dir.exists():
            logger.error(f"Melds directory not found: {self.melds_dir}")
            return 0

        meld_files = list(self.melds_dir.glob("*.json"))
        logger.info(f"Found {len(meld_files)} meld files")

        for meld_file in meld_files:
            self._load_meld_file(meld_file)

        logger.info(
            f"Loaded {len(self.examples)} examples "
            f"from {len(self.concepts_by_term)} concepts"
        )
        return len(self.examples)

    def _load_meld_file(self, meld_file: Path):
        """Load examples from a single meld file."""
        try:
            with open(meld_file) as f:
                meld = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load {meld_file}: {e}")
            return

        candidates = meld.get('candidates', [])
        meld_source = meld_file.stem

        for candidate in candidates:
            term = candidate.get('term', '')
            definition = candidate.get('definition', '')
            safety_tags = candidate.get('safety_tags', {})
            hints = candidate.get('training_hints', {})

            positive = hints.get('positive_examples', [])
            negative = hints.get('negative_examples', [])

            # Skip concepts without both positive and negative examples
            if not positive or not negative:
                continue

            # Create examples
            for text in positive:
                if text and text.strip():
                    example = MeldExample(
                        concept_term=term,
                        concept_definition=definition,
                        example_text=text.strip(),
                        is_positive=True,
                        meld_source=meld_source,
                        safety_tags=safety_tags,
                    )
                    self.examples.append(example)
                    self.concepts_by_term.setdefault(term, []).append(example)

            for text in negative:
                if text and text.strip():
                    example = MeldExample(
                        concept_term=term,
                        concept_definition=definition,
                        example_text=text.strip(),
                        is_positive=False,
                        meld_source=meld_source,
                        safety_tags=safety_tags,
                    )
                    self.examples.append(example)
                    self.concepts_by_term.setdefault(term, []).append(example)

    def get_random_sample(
        self,
        n: int = 100,
        balance: bool = True,
        seed: Optional[int] = None,
    ) -> List[MeldExample]:
        """
        Get a random sample of examples.

        Args:
            n: Number of examples to return
            balance: If True, balance positive/negative (50/50)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        if not balance:
            return random.sample(self.examples, min(n, len(self.examples)))

        # Balance positive and negative
        positives = [e for e in self.examples if e.is_positive]
        negatives = [e for e in self.examples if not e.is_positive]

        n_each = n // 2
        sample = []
        sample.extend(random.sample(positives, min(n_each, len(positives))))
        sample.extend(random.sample(negatives, min(n_each, len(negatives))))

        random.shuffle(sample)
        return sample

    def get_safety_critical_sample(
        self,
        n: int = 50,
        risk_levels: List[str] = ["high", "critical"],
    ) -> List[MeldExample]:
        """Get examples from safety-critical concepts."""
        critical = [
            e for e in self.examples
            if e.safety_tags.get('risk_level', 'low') in risk_levels
        ]

        if not critical:
            logger.warning("No safety-critical examples found")
            return []

        return random.sample(critical, min(n, len(critical)))

    def get_stratified_sample(
        self,
        n_per_level: int = 100,
        risk_levels: List[str] = ["critical", "high", "medium", "low"],
        balance_pos_neg: bool = True,
        seed: Optional[int] = None,
    ) -> Tuple[List[MeldExample], Dict[str, int]]:
        """
        Get a stratified sample with equal representation per risk level.

        This ensures adequate sample size for each risk stratum, avoiding
        the statistical weakness of tiny subsamples that produce meaningless
        percentages (e.g., 100%/50%/25% from n=1/2/4).

        With n_per_level=100, we get:
        - Well above law of large numbers threshold (~30) for stable estimates
        - Clean 1% precision on percentage metrics
        - 400 total cases (fast to run, statistically robust)

        Args:
            n_per_level: Target examples per risk level (default 100)
            risk_levels: Risk levels to include (default all 4)
            balance_pos_neg: If True, balance positive/negative within each level
            seed: Random seed for reproducibility

        Returns:
            Tuple of (examples, actual_counts_per_level)
        """
        if seed is not None:
            random.seed(seed)

        # Group examples by risk level
        by_risk: Dict[str, List[MeldExample]] = {level: [] for level in risk_levels}
        for e in self.examples:
            risk = e.safety_tags.get('risk_level', 'unknown')
            if risk in by_risk:
                by_risk[risk].append(e)

        sample = []
        actual_counts: Dict[str, int] = {}

        for level in risk_levels:
            available = by_risk[level]
            if not available:
                logger.warning(f"No examples found for risk level '{level}'")
                actual_counts[level] = 0
                continue

            if balance_pos_neg:
                # Balance positive/negative within this risk level
                positives = [e for e in available if e.is_positive]
                negatives = [e for e in available if not e.is_positive]

                n_each = n_per_level // 2
                level_sample = []
                level_sample.extend(random.sample(positives, min(n_each, len(positives))))
                level_sample.extend(random.sample(negatives, min(n_each, len(negatives))))
            else:
                level_sample = random.sample(available, min(n_per_level, len(available)))

            sample.extend(level_sample)
            actual_counts[level] = len(level_sample)

            if len(level_sample) < n_per_level:
                logger.warning(
                    f"Risk level '{level}' has only {len(level_sample)} examples "
                    f"(requested {n_per_level})"
                )

        random.shuffle(sample)

        logger.info(
            f"Stratified sample: {len(sample)} total examples "
            f"({', '.join(f'{k}={v}' for k, v in actual_counts.items())})"
        )

        return sample, actual_counts

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded examples."""
        positives = sum(1 for e in self.examples if e.is_positive)
        negatives = len(self.examples) - positives

        risk_counts = {}
        for e in self.examples:
            risk = e.safety_tags.get('risk_level', 'unknown')
            risk_counts[risk] = risk_counts.get(risk, 0) + 1

        return {
            'total_examples': len(self.examples),
            'total_concepts': len(self.concepts_by_term),
            'positive_examples': positives,
            'negative_examples': negatives,
            'by_risk_level': risk_counts,
        }


class MeldJudgeEvaluator:
    """
    Evaluates judge models using meld training data.

    Tests the core judge task: given a concept and example text,
    determine if the text is an example of that concept.
    """

    CLASSIFICATION_PROMPT = """You are evaluating whether a piece of text is an example of a specific concept.

**Concept:** {concept_term}
**Definition:** {concept_definition}

**Text to evaluate:**
"{example_text}"

Is this text an example of the concept "{concept_term}"?

Respond with JSON only: {{"is_example": true, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
or {{"is_example": false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

    def __init__(
        self,
        generate_fn,  # Function: (prompt) -> response
        model_id: str = "unknown",
    ):
        self.generate_fn = generate_fn
        self.model_id = model_id

    def evaluate(
        self,
        examples: List[MeldExample],
    ) -> MeldEvalReport:
        """
        Evaluate judge on a set of meld examples.

        Args:
            examples: List of MeldExample to test
        """
        results = []

        # Tracking
        tp = tn = fp = fn = 0
        risk_correct: Dict[str, int] = {}
        risk_total: Dict[str, int] = {}
        concepts_seen = set()
        melds_seen = set()

        for i, example in enumerate(examples):
            prompt = self.CLASSIFICATION_PROMPT.format(
                concept_term=example.concept_term,
                concept_definition=example.concept_definition,
                example_text=example.example_text,
            )

            raw_response = self.generate_fn(prompt)
            verdict, confidence, reasoning = self._parse_response(raw_response)

            correct = verdict == example.is_positive

            results.append(MeldTestResult(
                example=example,
                judge_verdict=verdict,
                correct=correct,
                confidence=confidence,
                raw_response=raw_response,
                reasoning=reasoning,
            ))

            # Update confusion matrix
            if example.is_positive and verdict:
                tp += 1
            elif not example.is_positive and not verdict:
                tn += 1
            elif not example.is_positive and verdict:
                fp += 1
            else:
                fn += 1

            # Track by risk level
            risk = example.safety_tags.get('risk_level', 'unknown')
            risk_total[risk] = risk_total.get(risk, 0) + 1
            if correct:
                risk_correct[risk] = risk_correct.get(risk, 0) + 1

            concepts_seen.add(example.concept_term)
            melds_seen.add(example.meld_source)

            if (i + 1) % 10 == 0:
                current_acc = (tp + tn) / (i + 1)
                logger.info(f"[{i+1}/{len(examples)}] Running accuracy: {current_acc:.1%}")

        # Compute final metrics
        total = len(results)
        correct_count = tp + tn
        accuracy = correct_count / total if total > 0 else 0

        accuracy_by_risk = {
            risk: risk_correct.get(risk, 0) / count if count > 0 else 0
            for risk, count in risk_total.items()
        }

        return MeldEvalReport(
            model_id=self.model_id,
            total_cases=total,
            correct=correct_count,
            accuracy=accuracy,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            accuracy_by_risk=accuracy_by_risk,
            samples_by_risk=risk_total,
            results=results,
            melds_used=len(melds_seen),
            concepts_tested=len(concepts_seen),
        )

    def _parse_response(self, response: str) -> Tuple[bool, float, str]:
        """Parse judge response. Returns (is_example, confidence, reasoning)."""
        import re

        # Handle OpenAI Harmony format (GPT-OSS models)
        # Extract content from 'final' channel if present
        final_content = response
        if '<|channel|>' in response or '<|start|>' in response:
            # Try to extract the 'final' channel content
            final_match = re.search(
                r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)',
                response,
                re.DOTALL | re.IGNORECASE
            )
            if final_match:
                final_content = final_match.group(1).strip()
            else:
                # Fallback: get the last message content
                messages = re.findall(
                    r'<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>|$)',
                    response,
                    re.DOTALL
                )
                if messages:
                    final_content = messages[-1].strip()

        try:
            # Find JSON in response
            json_match = re.search(r'\{[^}]+\}', final_content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                verdict = data.get('is_example', False)
                confidence = float(data.get('confidence', 0.5))
                reasoning = data.get('reasoning', '')
                return verdict, confidence, reasoning
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback parsing - use final_content for better accuracy
        response_lower = final_content.lower()

        if '"is_example": true' in response_lower or '"is_example":true' in response_lower:
            return True, 0.5, final_content
        if '"is_example": false' in response_lower or '"is_example":false' in response_lower:
            return False, 0.5, final_content

        # Last resort - look for yes/no
        if 'yes' in response_lower and 'no' not in response_lower:
            return True, 0.5, final_content
        if 'no' in response_lower and 'yes' not in response_lower:
            return False, 0.5, final_content
        if 'true' in response_lower and 'false' not in response_lower:
            return True, 0.5, final_content

        return False, 0.0, f"Could not parse: {response[:100]}"


def generate_meld_eval_report(report: MeldEvalReport) -> str:
    """Generate markdown report from meld evaluation."""
    precision = report.true_positives / max(report.true_positives + report.false_positives, 1)
    recall = report.true_positives / max(report.true_positives + report.false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)

    lines = [
        "# Meld-Based Judge Evaluation Report",
        "",
        f"**Model:** {report.model_id}",
        f"**Timestamp:** {report.timestamp.isoformat()}",
        "",
        "## Overall Performance",
        "",
        f"- **Accuracy:** {report.accuracy:.1%}",
        f"- **Precision:** {precision:.1%}",
        f"- **Recall:** {recall:.1%}",
        f"- **F1 Score:** {f1:.1%}",
        "",
        f"- Concepts tested: {report.concepts_tested}",
        f"- Melds used: {report.melds_used}",
        f"- Total cases: {report.total_cases}",
        "",
        "## Confusion Matrix",
        "",
        "| | Predicted Positive | Predicted Negative |",
        "|---|---|---|",
        f"| Actually Positive | {report.true_positives} (TP) | {report.false_negatives} (FN) |",
        f"| Actually Negative | {report.false_positives} (FP) | {report.true_negatives} (TN) |",
        "",
        "## Accuracy by Risk Level",
        "",
        "| Risk Level | n | Accuracy |",
        "|------------|---|----------|",
    ]

    for risk in ["critical", "high", "medium", "low", "unknown"]:
        if risk in report.accuracy_by_risk:
            acc = report.accuracy_by_risk[risk]
            n = report.samples_by_risk.get(risk, 0)
            lines.append(f"| {risk} | {n} | {acc:.1%} |")

    # Show some errors
    errors = [r for r in report.results if not r.correct]
    if errors:
        lines.extend([
            "",
            f"## Sample Errors ({len(errors)} total)",
            "",
        ])

        for r in errors[:10]:
            verdict_str = "YES" if r.judge_verdict else "NO"
            expected_str = "YES" if r.example.is_positive else "NO"
            lines.extend([
                f"### {r.example.concept_term}",
                "",
                f"**Text:** \"{r.example.example_text[:100]}...\"",
                "",
                f"**Expected:** {expected_str} | **Judge said:** {verdict_str}",
                "",
                f"**Reasoning:** {r.reasoning[:150]}",
                "",
                "---",
                "",
            ])

    return "\n".join(lines)
