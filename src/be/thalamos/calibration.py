"""
Calibration Suite for Practitioner Qualification

Before a CAT can serve as a Thalametrist or Thalamologist, it must be
calibrated to verify it can reliably distinguish correct from incorrect
responses.

The calibration uses factual questions with deterministic correct answers -
not subjective quality judgments. This provides an objective ground truth
for verifying practitioner reliability.

A practitioner must pass calibration before being qualified to assess subjects.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import json
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class CalibrationCase:
    """A factual calibration test case with deterministic correct answer."""
    question: str
    correct_answer: str  # The objectively correct answer
    test_response: str   # The response to evaluate
    should_pass: bool    # Whether the test_response is correct
    category: str = "general"
    description: str = ""


@dataclass
class CalibrationResult:
    """Result of a single calibration test."""
    case: CalibrationCase
    practitioner_verdict: bool  # Did practitioner say response was correct?
    passed: bool                # Did verdict match expected?
    raw_response: str
    reasoning: str


@dataclass
class CalibrationReport:
    """Full calibration report for a practitioner."""
    practitioner_model: str
    session_id: str

    # Core metrics
    total_cases: int
    passed: int
    failed: int
    accuracy: float

    # Confusion matrix
    true_positives: int   # Correctly identified correct answers
    true_negatives: int   # Correctly identified wrong answers
    false_positives: int  # Wrongly accepted incorrect answers
    false_negatives: int  # Wrongly rejected correct answers

    # Results
    results: List[CalibrationResult]

    # Qualification
    is_qualified: bool
    qualification_threshold: float
    recommendation: str

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'practitioner_model': self.practitioner_model,
            'session_id': self.session_id,
            'total_cases': self.total_cases,
            'passed': self.passed,
            'failed': self.failed,
            'accuracy': self.accuracy,
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'is_qualified': self.is_qualified,
            'qualification_threshold': self.qualification_threshold,
            'recommendation': self.recommendation,
            'timestamp': self.timestamp.isoformat(),
            'results': [
                {
                    'question': r.case.question,
                    'correct_answer': r.case.correct_answer,
                    'test_response': r.case.test_response,
                    'should_pass': r.case.should_pass,
                    'practitioner_verdict': r.practitioner_verdict,
                    'passed': r.passed,
                    'reasoning': r.reasoning,
                }
                for r in self.results
            ],
        }


# Standard calibration cases with factual questions and deterministic answers
STANDARD_CALIBRATION_CASES = [
    # === CORRECT RESPONSES (should_pass=True) ===
    CalibrationCase(
        question="What is the capital of France?",
        correct_answer="Paris",
        test_response="The capital of France is Paris.",
        should_pass=True,
        category="geography",
        description="Simple factual - correct",
    ),
    CalibrationCase(
        question="What is 7 multiplied by 8?",
        correct_answer="56",
        test_response="7 times 8 equals 56.",
        should_pass=True,
        category="math",
        description="Math - correct",
    ),
    CalibrationCase(
        question="What planet is closest to the Sun?",
        correct_answer="Mercury",
        test_response="Mercury is the closest planet to the Sun.",
        should_pass=True,
        category="science",
        description="Science fact - correct",
    ),
    CalibrationCase(
        question="What is the chemical symbol for gold?",
        correct_answer="Au",
        test_response="Gold's chemical symbol is Au, from the Latin 'aurum'.",
        should_pass=True,
        category="chemistry",
        description="Chemistry - correct with extra context",
    ),
    CalibrationCase(
        question="In what year did World War II end?",
        correct_answer="1945",
        test_response="World War II ended in 1945.",
        should_pass=True,
        category="history",
        description="History - correct",
    ),
    CalibrationCase(
        question="What is the largest organ in the human body?",
        correct_answer="skin",
        test_response="The skin is the largest organ in the human body.",
        should_pass=True,
        category="biology",
        description="Biology - correct",
    ),

    # === INCORRECT RESPONSES (should_pass=False) ===
    CalibrationCase(
        question="What is the capital of France?",
        correct_answer="Paris",
        test_response="The capital of France is London.",
        should_pass=False,
        category="geography",
        description="Simple factual - wrong answer",
    ),
    CalibrationCase(
        question="What is 7 multiplied by 8?",
        correct_answer="56",
        test_response="7 times 8 equals 54.",
        should_pass=False,
        category="math",
        description="Math - wrong calculation",
    ),
    CalibrationCase(
        question="What planet is closest to the Sun?",
        correct_answer="Mercury",
        test_response="Venus is the closest planet to the Sun.",
        should_pass=False,
        category="science",
        description="Science fact - wrong planet",
    ),
    CalibrationCase(
        question="What is the chemical symbol for gold?",
        correct_answer="Au",
        test_response="Gold's chemical symbol is Go.",
        should_pass=False,
        category="chemistry",
        description="Chemistry - wrong symbol",
    ),
    CalibrationCase(
        question="In what year did World War II end?",
        correct_answer="1945",
        test_response="World War II ended in 1942.",
        should_pass=False,
        category="history",
        description="History - wrong year",
    ),
    CalibrationCase(
        question="What is the largest organ in the human body?",
        correct_answer="skin",
        test_response="The liver is the largest organ in the human body.",
        should_pass=False,
        category="biology",
        description="Biology - wrong organ",
    ),

    # === EDGE CASES ===
    CalibrationCase(
        question="What is the capital of France?",
        correct_answer="Paris",
        test_response="I don't know the capital of France.",
        should_pass=False,
        category="edge",
        description="No answer given",
    ),
    CalibrationCase(
        question="What is 7 multiplied by 8?",
        correct_answer="56",
        test_response="That's a math question about multiplication.",
        should_pass=False,
        category="edge",
        description="Evasive non-answer",
    ),
]


class CalibrationSuite:
    """
    Calibration suite for qualifying practitioners.

    Uses factual Q&A with deterministic correct answers to verify
    the practitioner can reliably distinguish correct from incorrect responses.
    """

    VERIFICATION_PROMPT = """You are verifying if a response correctly answers a factual question.

Question: {question}
Correct Answer: {correct_answer}

Response to evaluate:
{test_response}

Does the response contain the correct answer to the question?
Respond with JSON only: {{"correct": true, "reasoning": "..."}} or {{"correct": false, "reasoning": "..."}}"""

    def __init__(
        self,
        room,  # ExaminationRoom
        qualification_threshold: float = 0.85,
    ):
        self.room = room
        self.qualification_threshold = qualification_threshold

        logger.info(f"CalibrationSuite initialized (threshold={qualification_threshold})")

    def run_calibration(
        self,
        cases: Optional[List[CalibrationCase]] = None,
    ) -> CalibrationReport:
        """
        Run calibration tests on the practitioner.

        Tests whether the practitioner can reliably distinguish
        correct from incorrect factual answers.
        """
        cases = cases or STANDARD_CALIBRATION_CASES
        results = []

        logger.info(f"Running calibration with {len(cases)} factual test cases")

        # Confusion matrix counts
        tp = tn = fp = fn = 0

        for i, case in enumerate(cases):
            logger.debug(f"[{i+1}/{len(cases)}] {case.description}")

            # Get practitioner verdict
            verdict, raw_response, reasoning = self._verify_response(
                question=case.question,
                correct_answer=case.correct_answer,
                test_response=case.test_response,
            )

            # Did verdict match expected?
            passed = verdict == case.should_pass

            results.append(CalibrationResult(
                case=case,
                practitioner_verdict=verdict,
                passed=passed,
                raw_response=raw_response,
                reasoning=reasoning,
            ))

            # Update confusion matrix
            if case.should_pass and verdict:
                tp += 1
            elif not case.should_pass and not verdict:
                tn += 1
            elif not case.should_pass and verdict:
                fp += 1
            else:  # case.should_pass and not verdict
                fn += 1

            status = "PASS" if passed else "FAIL"
            logger.debug(f"  {status}: expected {case.should_pass}, got {verdict}")

        # Compute metrics
        total = len(results)
        passed_count = sum(1 for r in results if r.passed)
        accuracy = passed_count / total if total > 0 else 0

        is_qualified = accuracy >= self.qualification_threshold

        # Generate recommendation
        if is_qualified:
            recommendation = (
                "Practitioner reliably identifies correct/incorrect answers. "
                "Qualified for cognitive assessment."
            )
        else:
            if fp > fn:
                recommendation = (
                    f"Calibration below threshold ({accuracy:.1%} < {self.qualification_threshold:.0%}). "
                    f"Practitioner accepts too many incorrect answers ({fp} false positives). "
                    "Consider using a more discerning model."
                )
            elif fn > fp:
                recommendation = (
                    f"Calibration below threshold ({accuracy:.1%} < {self.qualification_threshold:.0%}). "
                    f"Practitioner rejects too many correct answers ({fn} false negatives). "
                    "Consider using a less strict model."
                )
            else:
                recommendation = (
                    f"Calibration below threshold ({accuracy:.1%} < {self.qualification_threshold:.0%}). "
                    "Consider using a different model as practitioner."
                )

        report = CalibrationReport(
            practitioner_model=self.room.config.practitioner_model_id or self.room.config.subject_model_id,
            session_id=self.room.session_id,
            total_cases=total,
            passed=passed_count,
            failed=total - passed_count,
            accuracy=accuracy,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            results=results,
            is_qualified=is_qualified,
            qualification_threshold=self.qualification_threshold,
            recommendation=recommendation,
        )

        # Update room state
        self.room.practitioner_qualified = is_qualified

        # Record procedure
        self.room.record_procedure('calibration', report.to_dict())

        logger.info(
            f"Calibration complete: {accuracy:.1%} accuracy, "
            f"{'QUALIFIED' if is_qualified else 'NOT QUALIFIED'}"
        )

        return report

    def _verify_response(
        self,
        question: str,
        correct_answer: str,
        test_response: str,
    ) -> Tuple[bool, str, str]:
        """
        Have the practitioner verify if a response is correct.

        Returns (is_correct, raw_response, reasoning).
        """
        prompt = self.VERIFICATION_PROMPT.format(
            question=question,
            correct_answer=correct_answer,
            test_response=test_response,
        )

        raw_response = self.room.practitioner_generate(
            prompt,
            temperature=0.0,  # Greedy for consistency
        )

        # Handle empty response
        if not raw_response or not raw_response.strip():
            return False, "", "Empty response from practitioner"

        # Parse JSON response
        try:
            json_match = re.search(r'\{[^}]+\}', raw_response)
            if json_match:
                data = json.loads(json_match.group())
                is_correct = data.get('correct', False)
                reasoning = data.get('reasoning', '')
                return is_correct, raw_response, reasoning
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"JSON parse error: {e}")

        # Fallback: look for true/false patterns
        response_lower = raw_response.lower()
        if '"correct": true' in response_lower or '"correct":true' in response_lower:
            return True, raw_response, raw_response
        if '"correct": false' in response_lower or '"correct":false' in response_lower:
            return False, raw_response, raw_response

        # Check for plain true/false
        if 'true' in response_lower and 'false' not in response_lower:
            return True, raw_response, f"Inferred true from: {raw_response}"
        if 'false' in response_lower and 'true' not in response_lower:
            return False, raw_response, f"Inferred false from: {raw_response}"

        # Default to false if can't parse
        return False, raw_response, f"Could not parse: {raw_response[:100]}"

    def run_consistency_check(
        self,
        n_runs: int = 3,
    ) -> Dict[str, Any]:
        """
        Check practitioner consistency across multiple runs.

        With greedy decoding (temperature=0), should have zero variance.
        """
        test_cases = STANDARD_CALIBRATION_CASES[:4]
        all_verdicts = {case.question: [] for case in test_cases}

        logger.info(f"Running consistency check with {n_runs} runs")

        for run in range(n_runs):
            for case in test_cases:
                verdict, _, _ = self._verify_response(
                    question=case.question,
                    correct_answer=case.correct_answer,
                    test_response=case.test_response,
                )
                all_verdicts[case.question].append(verdict)

        # Check consistency
        inconsistent = []
        for question, verdicts in all_verdicts.items():
            if len(set(verdicts)) > 1:
                inconsistent.append(question)

        is_consistent = len(inconsistent) == 0

        logger.info(
            f"Consistency check: {len(inconsistent)} inconsistent "
            f"out of {len(test_cases)} cases"
        )

        return {
            'total_cases': len(test_cases),
            'n_runs': n_runs,
            'inconsistent_cases': inconsistent,
            'is_consistent': is_consistent,
        }

    def generate_report(
        self,
        calibration: CalibrationReport,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a markdown calibration report."""
        lines = [
            f"# Practitioner Calibration Report",
            f"",
            f"**Session:** {calibration.session_id}",
            f"**Practitioner:** {calibration.practitioner_model}",
            f"**Timestamp:** {calibration.timestamp.isoformat()}",
            f"",
            f"## Qualification",
            f"",
            f"**Status:** {'QUALIFIED' if calibration.is_qualified else 'NOT QUALIFIED'}",
            f"",
            f"**Recommendation:** {calibration.recommendation}",
            f"",
            f"## Metrics",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Cases | {calibration.total_cases} |",
            f"| Passed | {calibration.passed} |",
            f"| Failed | {calibration.failed} |",
            f"| Accuracy | {calibration.accuracy:.1%} |",
            f"| Threshold | {calibration.qualification_threshold:.0%} |",
            f"",
            f"## Confusion Matrix",
            f"",
            f"| | Predicted Correct | Predicted Wrong |",
            f"|---|---|---|",
            f"| Actually Correct | {calibration.true_positives} (TP) | {calibration.false_negatives} (FN) |",
            f"| Actually Wrong | {calibration.false_positives} (FP) | {calibration.true_negatives} (TN) |",
            f"",
        ]

        # Failed cases
        failed = [r for r in calibration.results if not r.passed]
        if failed:
            lines.extend([
                f"## Failed Cases ({len(failed)})",
                f"",
            ])
            for r in failed:
                lines.extend([
                    f"### {r.case.description}",
                    f"",
                    f"**Question:** {r.case.question}",
                    f"",
                    f"**Correct Answer:** {r.case.correct_answer}",
                    f"",
                    f"**Test Response:** {r.case.test_response}",
                    f"",
                    f"**Expected:** {'Correct' if r.case.should_pass else 'Wrong'}",
                    f"",
                    f"**Practitioner Said:** {'Correct' if r.practitioner_verdict else 'Wrong'}",
                    f"",
                    f"**Reasoning:** {r.reasoning}",
                    f"",
                    f"---",
                    f"",
                ])

        report = '\n'.join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Calibration report saved to {output_path}")

        return report
