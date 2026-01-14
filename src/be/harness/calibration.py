"""
Judge calibration for the graft testing harness.

Uses factual questions with objectively correct answers to verify
that the judge model can reliably identify correct vs incorrect responses.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, List

from .models import JudgeModel

logger = logging.getLogger(__name__)


@dataclass
class CalibrationCase:
    """A factual calibration test case with a deterministic correct answer."""
    question: str
    correct_answer: str  # The objectively correct answer
    test_response: str   # The response to evaluate
    should_pass: bool    # Whether the test_response is correct
    description: str


@dataclass
class CalibrationResult:
    """Result of a single calibration test."""
    case: CalibrationCase
    judge_verdict: bool  # Did judge say the response was correct?
    passed: bool         # Did judge's verdict match expected?
    score: float
    reasoning: str


@dataclass
class CalibrationReport:
    """Report from judge calibration."""
    total_cases: int
    passed: int
    failed: int
    accuracy: float

    # Breakdown
    true_positives: int   # Correctly identified correct answers
    true_negatives: int   # Correctly identified wrong answers
    false_positives: int  # Wrongly accepted incorrect answers
    false_negatives: int  # Wrongly rejected correct answers

    # Detailed results
    results: List[CalibrationResult]

    # Overall assessment
    is_calibrated: bool
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "total_cases": self.total_cases,
            "passed": self.passed,
            "failed": self.failed,
            "accuracy": self.accuracy,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "is_calibrated": self.is_calibrated,
            "recommendation": self.recommendation,
            "results": [
                {
                    "question": r.case.question,
                    "correct_answer": r.case.correct_answer,
                    "test_response": r.case.test_response,
                    "should_pass": r.case.should_pass,
                    "judge_verdict": r.judge_verdict,
                    "passed": r.passed,
                    "score": r.score,
                    "reasoning": r.reasoning,
                }
                for r in self.results
            ],
        }


# Factual calibration cases with deterministic correct answers
CALIBRATION_CASES = [
    # === CORRECT RESPONSES (should_pass=True) ===
    CalibrationCase(
        question="What is the capital of France?",
        correct_answer="Paris",
        test_response="The capital of France is Paris.",
        should_pass=True,
        description="Simple factual - correct",
    ),
    CalibrationCase(
        question="What is 7 multiplied by 8?",
        correct_answer="56",
        test_response="7 times 8 equals 56.",
        should_pass=True,
        description="Math - correct",
    ),
    CalibrationCase(
        question="What planet is closest to the Sun?",
        correct_answer="Mercury",
        test_response="Mercury is the closest planet to the Sun.",
        should_pass=True,
        description="Science fact - correct",
    ),
    CalibrationCase(
        question="What is the chemical symbol for gold?",
        correct_answer="Au",
        test_response="Gold's chemical symbol is Au, from the Latin 'aurum'.",
        should_pass=True,
        description="Chemistry - correct with extra context",
    ),
    CalibrationCase(
        question="In what year did World War II end?",
        correct_answer="1945",
        test_response="World War II ended in 1945.",
        should_pass=True,
        description="History - correct",
    ),
    CalibrationCase(
        question="What is the largest organ in the human body?",
        correct_answer="skin",
        test_response="The skin is the largest organ in the human body.",
        should_pass=True,
        description="Biology - correct",
    ),

    # === INCORRECT RESPONSES (should_pass=False) ===
    CalibrationCase(
        question="What is the capital of France?",
        correct_answer="Paris",
        test_response="The capital of France is London.",
        should_pass=False,
        description="Simple factual - wrong answer",
    ),
    CalibrationCase(
        question="What is 7 multiplied by 8?",
        correct_answer="56",
        test_response="7 times 8 equals 54.",
        should_pass=False,
        description="Math - wrong calculation",
    ),
    CalibrationCase(
        question="What planet is closest to the Sun?",
        correct_answer="Mercury",
        test_response="Venus is the closest planet to the Sun.",
        should_pass=False,
        description="Science fact - wrong planet",
    ),
    CalibrationCase(
        question="What is the chemical symbol for gold?",
        correct_answer="Au",
        test_response="Gold's chemical symbol is Go.",
        should_pass=False,
        description="Chemistry - wrong symbol",
    ),
    CalibrationCase(
        question="In what year did World War II end?",
        correct_answer="1945",
        test_response="World War II ended in 1942.",
        should_pass=False,
        description="History - wrong year",
    ),
    CalibrationCase(
        question="What is the largest organ in the human body?",
        correct_answer="skin",
        test_response="The liver is the largest organ in the human body.",
        should_pass=False,
        description="Biology - wrong organ",
    ),

    # === EDGE CASES ===
    CalibrationCase(
        question="What is the capital of France?",
        correct_answer="Paris",
        test_response="I don't know the capital of France.",
        should_pass=False,
        description="No answer given",
    ),
    CalibrationCase(
        question="What is 7 multiplied by 8?",
        correct_answer="56",
        test_response="That's a math question about multiplication.",
        should_pass=False,
        description="Evasive non-answer",
    ),
]


class JudgeCalibrator:
    """Calibrates judge model using factual questions with deterministic answers."""

    VERIFICATION_PROMPT = """You are verifying if a response correctly answers a factual question.

Question: {question}
Correct Answer: {correct_answer}

Response to evaluate:
{test_response}

Does the response contain the correct answer to the question?
Respond with JSON only: {{"correct": true, "reasoning": "..."}} or {{"correct": false, "reasoning": "..."}}"""

    def __init__(
        self,
        judge: JudgeModel,
        accuracy_threshold: float = 0.85,
    ):
        self.judge = judge
        self.accuracy_threshold = accuracy_threshold

    def _verify_response(
        self,
        question: str,
        correct_answer: str,
        test_response: str,
    ) -> tuple:
        """
        Have the judge verify if a response is correct.

        Returns (is_correct, confidence_score, reasoning).
        """
        prompt = self.VERIFICATION_PROMPT.format(
            question=question,
            correct_answer=correct_answer,
            test_response=test_response,
        )

        raw_response = self.judge._generate(prompt)

        # Print raw response for debugging
        print(f"  [{question[:40]}] -> {repr(raw_response) if raw_response else 'EMPTY'}")

        # Handle empty response
        if not raw_response or not raw_response.strip():
            return False, 0.0, f"Empty response from judge"

        # Parse JSON response
        try:
            json_match = re.search(r'\{[^}]+\}', raw_response)
            if json_match:
                data = json.loads(json_match.group())
                is_correct = data.get("correct", False)
                reasoning = data.get("reasoning", "")
                score = 10.0 if is_correct else 0.0
                return is_correct, score, reasoning
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"JSON parse error: {e}")

        # Fallback: look for true/false in response
        response_lower = raw_response.lower()
        if '"correct": true' in response_lower or '"correct":true' in response_lower:
            return True, 10.0, raw_response
        if '"correct": false' in response_lower or '"correct":false' in response_lower:
            return False, 0.0, raw_response

        # Check for plain true/false
        if 'true' in response_lower and 'false' not in response_lower:
            return True, 10.0, f"Inferred true from: {raw_response}"
        if 'false' in response_lower and 'true' not in response_lower:
            return False, 0.0, f"Inferred false from: {raw_response}"

        # Default to false if can't parse
        return False, 0.0, f"Could not parse: {repr(raw_response)}"

    def run_calibration(
        self,
        cases: List[CalibrationCase] = None,
    ) -> CalibrationReport:
        """
        Run calibration tests on the judge model.

        Tests whether the judge can reliably distinguish correct from
        incorrect factual answers.
        """
        cases = cases or CALIBRATION_CASES
        results = []

        logger.info(f"Running calibration with {len(cases)} factual test cases")

        # Confusion matrix counts
        tp = tn = fp = fn = 0

        for case in cases:
            judge_verdict, score, reasoning = self._verify_response(
                question=case.question,
                correct_answer=case.correct_answer,
                test_response=case.test_response,
            )

            # Did judge's verdict match what we expected?
            passed = judge_verdict == case.should_pass

            results.append(CalibrationResult(
                case=case,
                judge_verdict=judge_verdict,
                passed=passed,
                score=score,
                reasoning=reasoning,
            ))

            # Update confusion matrix
            if case.should_pass and judge_verdict:
                tp += 1
            elif not case.should_pass and not judge_verdict:
                tn += 1
            elif not case.should_pass and judge_verdict:
                fp += 1
            else:  # case.should_pass and not judge_verdict
                fn += 1

            status = "PASS" if passed else "FAIL"
            logger.debug(
                f"{status}: '{case.question}' - "
                f"expected {case.should_pass}, judge said {judge_verdict}"
            )

        # Compute accuracy
        total = len(results)
        passed_count = sum(1 for r in results if r.passed)
        accuracy = passed_count / total if total > 0 else 0

        is_calibrated = accuracy >= self.accuracy_threshold

        if is_calibrated:
            recommendation = "Judge model reliably identifies correct/incorrect answers. Proceed with evaluation."
        else:
            if fp > fn:
                recommendation = (
                    f"Judge calibration below threshold ({accuracy:.1%} < {self.accuracy_threshold:.0%}). "
                    f"Judge accepts too many incorrect answers ({fp} false positives). "
                    "Consider using a more discerning judge model."
                )
            elif fn > fp:
                recommendation = (
                    f"Judge calibration below threshold ({accuracy:.1%} < {self.accuracy_threshold:.0%}). "
                    f"Judge rejects too many correct answers ({fn} false negatives). "
                    "Consider using a less strict judge model."
                )
            else:
                recommendation = (
                    f"Judge calibration below threshold ({accuracy:.1%} < {self.accuracy_threshold:.0%}). "
                    "Consider using a different judge model."
                )

        return CalibrationReport(
            total_cases=total,
            passed=passed_count,
            failed=total - passed_count,
            accuracy=accuracy,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
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

        With greedy decoding, should have zero variance.
        """
        test_cases = CALIBRATION_CASES[:4]
        all_verdicts = {case.question: [] for case in test_cases}

        for run in range(n_runs):
            for case in test_cases:
                verdict, _, _ = self._verify_response(
                    question=case.question,
                    correct_answer=case.correct_answer,
                    test_response=case.test_response,
                )
                all_verdicts[case.question].append(verdict)

        # Check consistency (all verdicts should be the same)
        inconsistent = []
        for question, verdicts in all_verdicts.items():
            if len(set(verdicts)) > 1:
                inconsistent.append(question)

        is_consistent = len(inconsistent) == 0

        logger.info(
            f"Consistency check: {len(inconsistent)} inconsistent cases out of {len(test_cases)}"
        )

        return {
            "total_cases": len(test_cases),
            "inconsistent_cases": inconsistent,
            "is_consistent": is_consistent,
        }
