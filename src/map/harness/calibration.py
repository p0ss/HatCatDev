"""
Judge calibration for the graft testing harness.

Verifies that the judge model produces consistent and accurate scores.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .models import JudgeModel

logger = logging.getLogger(__name__)


@dataclass
class CalibrationCase:
    """A single calibration test case."""
    concept: str
    definition: str
    response: str
    expected_bucket: str  # "low" (0-3), "medium" (4-6), "high" (7-10)
    description: str


@dataclass
class CalibrationResult:
    """Result of a single calibration test."""
    case: CalibrationCase
    actual_score: float
    actual_bucket: str
    passed: bool
    reasoning: str


@dataclass
class CalibrationReport:
    """Report from judge calibration."""
    total_cases: int
    passed: int
    failed: int
    accuracy: float

    # By expected bucket
    accuracy_by_bucket: Dict[str, float]

    # Detailed results
    results: List[CalibrationResult]

    # Overall assessment
    is_calibrated: bool  # True if accuracy >= threshold
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "total_cases": self.total_cases,
            "passed": self.passed,
            "failed": self.failed,
            "accuracy": self.accuracy,
            "accuracy_by_bucket": self.accuracy_by_bucket,
            "is_calibrated": self.is_calibrated,
            "recommendation": self.recommendation,
            "results": [
                {
                    "concept": r.case.concept,
                    "description": r.case.description,
                    "expected_bucket": r.case.expected_bucket,
                    "actual_score": r.actual_score,
                    "actual_bucket": r.actual_bucket,
                    "passed": r.passed,
                    "reasoning": r.reasoning,
                }
                for r in self.results
            ],
        }


# Standard calibration test cases
CALIBRATION_CASES = [
    # LOW (0-3): Wrong or no understanding
    CalibrationCase(
        concept="Photosynthesis",
        definition="The process by which plants convert light energy into chemical energy, using carbon dioxide and water to produce glucose and oxygen.",
        response="Photosynthesis is when plants eat dirt and grow bigger.",
        expected_bucket="low",
        description="Fundamentally wrong explanation",
    ),
    CalibrationCase(
        concept="Democracy",
        definition="A system of government in which power is vested in the people, who rule either directly or through elected representatives.",
        response="Democracy is a type of sandwich popular in Greece.",
        expected_bucket="low",
        description="Completely unrelated answer",
    ),
    CalibrationCase(
        concept="Gravity",
        definition="A fundamental force of nature that attracts objects with mass toward each other, proportional to their masses and inversely proportional to the square of the distance between them.",
        response="I don't know what gravity is.",
        expected_bucket="low",
        description="No answer provided",
    ),
    CalibrationCase(
        concept="Algorithm",
        definition="A step-by-step procedure or set of rules for solving a problem or accomplishing a task, especially by a computer.",
        response="An algorithm is a dance move used in classical ballet.",
        expected_bucket="low",
        description="Wrong domain entirely",
    ),

    # MEDIUM (4-6): Partial understanding
    CalibrationCase(
        concept="Photosynthesis",
        definition="The process by which plants convert light energy into chemical energy, using carbon dioxide and water to produce glucose and oxygen.",
        response="Photosynthesis is how plants make food using sunlight.",
        expected_bucket="medium",
        description="Correct but incomplete - missing mechanism details",
    ),
    CalibrationCase(
        concept="Democracy",
        definition="A system of government in which power is vested in the people, who rule either directly or through elected representatives.",
        response="Democracy is when people vote for their leaders.",
        expected_bucket="medium",
        description="Partially correct - only covers electoral aspect",
    ),
    CalibrationCase(
        concept="Gravity",
        definition="A fundamental force of nature that attracts objects with mass toward each other, proportional to their masses and inversely proportional to the square of the distance between them.",
        response="Gravity is the force that pulls things down to the ground.",
        expected_bucket="medium",
        description="Simplified understanding, missing universal nature",
    ),
    CalibrationCase(
        concept="Algorithm",
        definition="A step-by-step procedure or set of rules for solving a problem or accomplishing a task, especially by a computer.",
        response="An algorithm is a set of instructions that computers follow.",
        expected_bucket="medium",
        description="Correct but narrow - only mentions computers",
    ),

    # HIGH (7-10): Good to excellent understanding
    CalibrationCase(
        concept="Photosynthesis",
        definition="The process by which plants convert light energy into chemical energy, using carbon dioxide and water to produce glucose and oxygen.",
        response="Photosynthesis is the process by which plants and other organisms convert light energy into chemical energy. Using chlorophyll in their cells, they absorb sunlight and use it to transform carbon dioxide from the air and water from the soil into glucose for energy, releasing oxygen as a byproduct.",
        expected_bucket="high",
        description="Complete and accurate explanation",
    ),
    CalibrationCase(
        concept="Democracy",
        definition="A system of government in which power is vested in the people, who rule either directly or through elected representatives.",
        response="Democracy is a form of government where political power ultimately rests with the citizenry. Citizens exercise this power either directly through votes on specific issues, or indirectly by electing representatives to make decisions on their behalf. Key features include free elections, rule of law, and protection of individual rights.",
        expected_bucket="high",
        description="Thorough explanation with key features",
    ),
    CalibrationCase(
        concept="Gravity",
        definition="A fundamental force of nature that attracts objects with mass toward each other, proportional to their masses and inversely proportional to the square of the distance between them.",
        response="Gravity is one of the four fundamental forces of nature. It causes all objects with mass to attract each other. The strength of gravitational attraction between two objects depends on their masses and the distance between them - more mass means stronger attraction, and greater distance means weaker attraction.",
        expected_bucket="high",
        description="Accurate with key relationships explained",
    ),
    CalibrationCase(
        concept="Algorithm",
        definition="A step-by-step procedure or set of rules for solving a problem or accomplishing a task, especially by a computer.",
        response="An algorithm is a well-defined sequence of steps or rules designed to solve a specific problem or accomplish a particular task. While commonly associated with computer programming, algorithms can describe any systematic procedure, from a recipe for baking bread to instructions for sorting a list of numbers.",
        expected_bucket="high",
        description="Complete with generalization beyond computers",
    ),
]


class JudgeCalibrator:
    """Calibrates and validates judge model scoring."""

    def __init__(
        self,
        judge: JudgeModel,
        accuracy_threshold: float = 0.75,
    ):
        self.judge = judge
        self.accuracy_threshold = accuracy_threshold

    def _score_to_bucket(self, score: float) -> str:
        """Convert numeric score to bucket."""
        if score <= 3:
            return "low"
        elif score <= 6:
            return "medium"
        else:
            return "high"

    def run_calibration(
        self,
        cases: List[CalibrationCase] = None,
    ) -> CalibrationReport:
        """
        Run calibration tests on the judge model.

        Args:
            cases: Custom test cases. If None, uses standard cases.

        Returns:
            CalibrationReport with accuracy metrics.
        """
        cases = cases or CALIBRATION_CASES
        results = []

        logger.info(f"Running calibration with {len(cases)} test cases")

        # Track accuracy by bucket
        bucket_correct = {"low": 0, "medium": 0, "high": 0}
        bucket_total = {"low": 0, "medium": 0, "high": 0}

        for case in cases:
            # Get judge's score
            judge_result = self.judge.score_concept_explanation(
                concept_term=case.concept,
                concept_definition=case.definition,
                target_response=case.response,
            )

            actual_bucket = self._score_to_bucket(judge_result.score)
            passed = actual_bucket == case.expected_bucket

            results.append(CalibrationResult(
                case=case,
                actual_score=judge_result.score,
                actual_bucket=actual_bucket,
                passed=passed,
                reasoning=judge_result.reasoning,
            ))

            bucket_total[case.expected_bucket] += 1
            if passed:
                bucket_correct[case.expected_bucket] += 1

            status = "PASS" if passed else "FAIL"
            logger.debug(
                f"{status}: {case.concept} - expected {case.expected_bucket}, "
                f"got {actual_bucket} ({judge_result.score:.1f})"
            )

        # Compute accuracy
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        accuracy = passed / total if total > 0 else 0

        accuracy_by_bucket = {}
        for bucket in ["low", "medium", "high"]:
            if bucket_total[bucket] > 0:
                accuracy_by_bucket[bucket] = bucket_correct[bucket] / bucket_total[bucket]
            else:
                accuracy_by_bucket[bucket] = 0.0

        is_calibrated = accuracy >= self.accuracy_threshold

        if is_calibrated:
            recommendation = "Judge model is well-calibrated. Proceed with evaluation."
        else:
            # Identify problematic buckets
            weak_buckets = [
                b for b, acc in accuracy_by_bucket.items()
                if acc < self.accuracy_threshold
            ]
            recommendation = (
                f"Judge model calibration below threshold ({accuracy:.1%} < {self.accuracy_threshold:.0%}). "
                f"Weak performance on: {', '.join(weak_buckets)}. "
                "Consider using a different judge model."
            )

        return CalibrationReport(
            total_cases=total,
            passed=passed,
            failed=total - passed,
            accuracy=accuracy,
            accuracy_by_bucket=accuracy_by_bucket,
            results=results,
            is_calibrated=is_calibrated,
            recommendation=recommendation,
        )

    def run_consistency_check(
        self,
        n_runs: int = 3,
    ) -> Dict[str, float]:
        """
        Check judge consistency across multiple runs.

        Returns variance in scores for identical inputs.
        """
        # Pick a subset of cases
        test_cases = CALIBRATION_CASES[:4]
        all_scores = {case.concept: [] for case in test_cases}

        for run in range(n_runs):
            for case in test_cases:
                result = self.judge.score_concept_explanation(
                    concept_term=case.concept,
                    concept_definition=case.definition,
                    target_response=case.response,
                )
                all_scores[case.concept].append(result.score)

        # Compute variance for each case
        variances = {}
        for concept, scores in all_scores.items():
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
            variances[concept] = variance

        avg_variance = sum(variances.values()) / len(variances)

        logger.info(f"Consistency check: average variance = {avg_variance:.2f}")

        return {
            "per_concept_variance": variances,
            "average_variance": avg_variance,
            "is_consistent": avg_variance < 2.0,  # Allow up to 2 points variance
        }
