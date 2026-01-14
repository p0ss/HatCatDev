"""
Hard Evaluation Suite for Judge Models

Tests judge candidates on tasks that actually matter for our use cases:
- Concept explanation evaluation with nuanced quality levels
- Subtle error detection
- Hedging/evasion detection
- Reasoning chain assessment

This goes beyond basic factual Q&A to test discrimination ability.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
import random

logger = logging.getLogger(__name__)


class ResponseQuality(Enum):
    """Quality levels for test responses."""
    CORRECT_COMPLETE = "correct_complete"       # Fully correct, covers all key aspects
    CORRECT_INCOMPLETE = "correct_incomplete"   # Correct but missing key aspects
    SUBTLE_ERROR = "subtle_error"               # Mostly right but contains a key error
    PLAUSIBLE_WRONG = "plausible_wrong"         # Sounds reasonable but fundamentally wrong
    HEDGING = "hedging"                         # Doesn't commit to an answer
    NONSENSE = "nonsense"                       # Obviously wrong


@dataclass
class ConceptJudgingCase:
    """A concept explanation evaluation test case."""
    concept_term: str
    concept_definition: str  # Ground truth definition
    test_response: str       # Response to evaluate
    quality: ResponseQuality
    expected_score_range: Tuple[float, float]  # (min, max) expected score 0-10
    error_description: str = ""  # What's wrong (for subtle errors)
    category: str = "general"


# Hard test cases using real concept patterns
CONCEPT_JUDGING_CASES = [
    # === CORRECT AND COMPLETE ===
    ConceptJudgingCase(
        concept_term="Hypothesis",
        concept_definition="A tentative or testable proposition about observable phenomena, capable of generating predictions and being supported or falsified by evidence.",
        test_response="A hypothesis is a testable proposition that can generate predictions about observable phenomena. It can be supported or falsified through empirical evidence.",
        quality=ResponseQuality.CORRECT_COMPLETE,
        expected_score_range=(8, 10),
        category="science",
    ),
    ConceptJudgingCase(
        concept_term="Deception",
        concept_definition="The act of causing someone to believe something that is not true, typically to gain an advantage.",
        test_response="Deception involves intentionally making someone believe something false, usually for personal benefit. It includes lying, misleading, and withholding relevant information.",
        quality=ResponseQuality.CORRECT_COMPLETE,
        expected_score_range=(8, 10),
        category="psychology",
    ),

    # === CORRECT BUT INCOMPLETE ===
    ConceptJudgingCase(
        concept_term="Hypothesis",
        concept_definition="A tentative or testable proposition about observable phenomena, capable of generating predictions and being supported or falsified by evidence.",
        test_response="A hypothesis is an educated guess about how something works.",
        quality=ResponseQuality.CORRECT_INCOMPLETE,
        expected_score_range=(4, 6),
        error_description="Missing: testability, predictions, falsifiability",
        category="science",
    ),
    ConceptJudgingCase(
        concept_term="Algorithm",
        concept_definition="A finite sequence of well-defined instructions for solving a class of problems or performing a computation.",
        test_response="An algorithm is a set of steps to solve a problem.",
        quality=ResponseQuality.CORRECT_INCOMPLETE,
        expected_score_range=(4, 6),
        error_description="Missing: finite, well-defined, class of problems",
        category="computing",
    ),

    # === SUBTLE ERRORS ===
    ConceptJudgingCase(
        concept_term="Hypothesis",
        concept_definition="A tentative or testable proposition about observable phenomena, capable of generating predictions and being supported or falsified by evidence.",
        test_response="A hypothesis is a proven scientific theory that has been validated through extensive experimentation and peer review.",
        quality=ResponseQuality.SUBTLE_ERROR,
        expected_score_range=(2, 4),
        error_description="Confuses hypothesis (tentative, testable) with theory (well-supported). Key error: says 'proven' when hypotheses are tentative.",
        category="science",
    ),
    ConceptJudgingCase(
        concept_term="Correlation",
        concept_definition="A statistical relationship between two variables where changes in one are associated with changes in another.",
        test_response="Correlation is when one thing causes another thing to happen. If two variables are correlated, changes in one will cause changes in the other.",
        quality=ResponseQuality.SUBTLE_ERROR,
        expected_score_range=(2, 4),
        error_description="Conflates correlation with causation - a classic error",
        category="statistics",
    ),
    ConceptJudgingCase(
        concept_term="Entropy",
        concept_definition="A measure of disorder or randomness in a system, or the amount of information required to describe the state of a system.",
        test_response="Entropy is the measure of order in a system. High entropy means everything is well-organized and predictable.",
        quality=ResponseQuality.SUBTLE_ERROR,
        expected_score_range=(1, 3),
        error_description="Inverts the meaning - entropy measures disorder, not order; high entropy = MORE randomness",
        category="physics",
    ),

    # === PLAUSIBLE BUT FUNDAMENTALLY WRONG ===
    ConceptJudgingCase(
        concept_term="Recursion",
        concept_definition="A method where a function calls itself to solve smaller instances of the same problem.",
        test_response="Recursion is when a program runs in a continuous loop, repeatedly executing the same instructions until stopped by the user.",
        quality=ResponseQuality.PLAUSIBLE_WRONG,
        expected_score_range=(0, 2),
        error_description="Confuses recursion with infinite loops. Recursion has base cases; infinite loops don't.",
        category="computing",
    ),
    ConceptJudgingCase(
        concept_term="Natural Selection",
        concept_definition="The process by which organisms with favorable traits are more likely to reproduce and pass on their genes.",
        test_response="Natural selection is when animals consciously choose to develop better traits to survive in their environment. They decide what features would be helpful and evolve those traits.",
        quality=ResponseQuality.PLAUSIBLE_WRONG,
        expected_score_range=(0, 2),
        error_description="Implies conscious choice and Lamarckian evolution. Natural selection is not goal-directed.",
        category="biology",
    ),

    # === HEDGING / NON-ANSWERS ===
    ConceptJudgingCase(
        concept_term="Consciousness",
        concept_definition="The state of being aware of and responsive to one's surroundings; subjective experience.",
        test_response="Consciousness is a complex topic that philosophers and scientists have debated for centuries. There are many different views on what it might be, and it's hard to give a definitive answer.",
        quality=ResponseQuality.HEDGING,
        expected_score_range=(1, 3),
        error_description="Avoids giving any actual definition. Acknowledges complexity but provides no substance.",
        category="philosophy",
    ),
    ConceptJudgingCase(
        concept_term="Quantum Entanglement",
        concept_definition="A phenomenon where quantum particles become correlated such that the state of one instantly affects the state of another, regardless of distance.",
        test_response="Quantum entanglement is one of those quantum physics things that's really hard to explain. Einstein called it 'spooky action at a distance.' It's very mysterious and counterintuitive.",
        quality=ResponseQuality.HEDGING,
        expected_score_range=(1, 3),
        error_description="Uses famous quote and mystification instead of explaining the concept",
        category="physics",
    ),

    # === OBVIOUS NONSENSE ===
    ConceptJudgingCase(
        concept_term="Photosynthesis",
        concept_definition="The process by which plants convert light energy into chemical energy stored in glucose.",
        test_response="Photosynthesis is how plants communicate with each other through underground networks of light beams.",
        quality=ResponseQuality.NONSENSE,
        expected_score_range=(0, 1),
        error_description="Complete fabrication with no connection to actual process",
        category="biology",
    ),
]


# Reasoning chain evaluation cases
@dataclass
class ReasoningCase:
    """Test case for evaluating reasoning chain quality."""
    problem: str
    reasoning_chain: str
    conclusion: str
    is_valid: bool  # Is the reasoning logically valid?
    errors: List[str] = field(default_factory=list)
    category: str = "logic"


REASONING_CASES = [
    # Valid reasoning
    ReasoningCase(
        problem="All mammals are warm-blooded. Dogs are mammals. Are dogs warm-blooded?",
        reasoning_chain="1. All mammals are warm-blooded (given premise). 2. Dogs are mammals (given premise). 3. Since dogs belong to the category of mammals, and all mammals are warm-blooded, dogs must be warm-blooded.",
        conclusion="Yes, dogs are warm-blooded.",
        is_valid=True,
        category="syllogism",
    ),

    # Invalid reasoning - affirming the consequent
    ReasoningCase(
        problem="If it's raining, the ground is wet. The ground is wet. Is it raining?",
        reasoning_chain="1. If it's raining, the ground is wet (given). 2. The ground is wet (observed). 3. Therefore, it must be raining.",
        conclusion="Yes, it is raining.",
        is_valid=False,
        errors=["Affirming the consequent - ground could be wet for other reasons (sprinklers, flooding, etc.)"],
        category="fallacy",
    ),

    # Invalid reasoning - false dichotomy
    ReasoningCase(
        problem="Should we use Python or Java for this project?",
        reasoning_chain="1. We need to choose a programming language. 2. Python is good for quick development. 3. Java is good for enterprise applications. 4. Since this is a quick prototype, we should use Python. 5. Therefore Python is the only reasonable choice.",
        conclusion="We must use Python.",
        is_valid=False,
        errors=["False dichotomy - other languages exist", "Conclusion overstates certainty"],
        category="fallacy",
    ),

    # Valid reasoning with uncertainty
    ReasoningCase(
        problem="Given limited data, what can we conclude about the effectiveness of treatment X?",
        reasoning_chain="1. We have data from 50 patients. 2. 35 showed improvement (70%). 3. However, there was no control group. 4. The sample size is small for statistical significance. 5. Therefore, preliminary results suggest potential benefit, but more rigorous study is needed.",
        conclusion="Treatment X shows promise but requires further controlled study before conclusions can be drawn.",
        is_valid=True,
        category="scientific_reasoning",
    ),
]


@dataclass
class JudgeEvalResult:
    """Result of evaluating a judge on a single case."""
    case_id: str
    expected_quality: ResponseQuality
    expected_score_range: Tuple[float, float]
    judge_score: float
    judge_reasoning: str
    score_in_range: bool
    raw_response: str


@dataclass
class JudgeEvalReport:
    """Full evaluation report for a judge model."""
    model_id: str

    # Concept judging results
    concept_cases_total: int
    concept_score_accuracy: float  # % of scores in expected range
    quality_discrimination: Dict[str, float]  # Accuracy per quality level

    # Reasoning evaluation results
    reasoning_cases_total: int
    reasoning_accuracy: float  # % correct valid/invalid judgments

    # Detailed results
    concept_results: List[JudgeEvalResult]
    reasoning_results: List[Dict]

    # Overall
    overall_score: float
    recommendation: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'concept_cases_total': self.concept_cases_total,
            'concept_score_accuracy': self.concept_score_accuracy,
            'quality_discrimination': self.quality_discrimination,
            'reasoning_cases_total': self.reasoning_cases_total,
            'reasoning_accuracy': self.reasoning_accuracy,
            'overall_score': self.overall_score,
            'recommendation': self.recommendation,
            'timestamp': self.timestamp.isoformat(),
            'concept_results': [
                {
                    'case_id': r.case_id,
                    'expected_quality': r.expected_quality.value,
                    'expected_score_range': r.expected_score_range,
                    'judge_score': r.judge_score,
                    'score_in_range': r.score_in_range,
                    'judge_reasoning': r.judge_reasoning[:200],
                }
                for r in self.concept_results
            ],
            'reasoning_results': self.reasoning_results,
        }


class JudgeEvaluator:
    """
    Evaluates judge models on hard discrimination tasks.

    Tests whether a model can reliably:
    - Score concept explanations at appropriate quality levels
    - Detect subtle errors vs complete correctness
    - Identify hedging and non-answers
    - Evaluate reasoning chain validity
    """

    CONCEPT_EVAL_PROMPT = """You are evaluating the quality of a concept explanation.

**Concept:** {concept_term}
**Correct Definition:** {concept_definition}

**Explanation to Evaluate:**
{test_response}

Rate the explanation from 0-10:
- 0-2: Fundamentally wrong, nonsense, or completely misses the concept
- 3-4: Contains significant errors or misunderstandings
- 5-6: Partially correct but missing key aspects or has minor errors
- 7-8: Good understanding with only minor gaps
- 9-10: Excellent, complete, and accurate explanation

Respond with JSON: {{"score": N, "reasoning": "brief explanation of your rating"}}"""

    REASONING_EVAL_PROMPT = """You are evaluating whether a reasoning chain is logically valid.

**Problem:** {problem}

**Reasoning Chain:**
{reasoning_chain}

**Conclusion:** {conclusion}

Is this reasoning logically valid? Check for:
- Whether the conclusion follows from the premises
- Logical fallacies (affirming the consequent, false dichotomy, etc.)
- Unstated assumptions that may not hold
- Appropriate handling of uncertainty

Respond with JSON: {{"valid": true/false, "errors": ["list of errors if any"], "reasoning": "explanation"}}"""

    def __init__(
        self,
        generate_fn,  # Function to generate from model: (prompt) -> response
        model_id: str = "unknown",
    ):
        self.generate_fn = generate_fn
        self.model_id = model_id

    def evaluate_concept_judging(
        self,
        cases: Optional[List[ConceptJudgingCase]] = None,
    ) -> Tuple[List[JudgeEvalResult], Dict[str, float]]:
        """
        Evaluate judge on concept explanation scoring.

        Returns (results, quality_discrimination_dict)
        """
        cases = cases or CONCEPT_JUDGING_CASES
        results = []

        # Track accuracy by quality level
        quality_correct = {q.value: 0 for q in ResponseQuality}
        quality_total = {q.value: 0 for q in ResponseQuality}

        for i, case in enumerate(cases):
            case_id = f"{case.concept_term}_{case.quality.value}_{i}"

            prompt = self.CONCEPT_EVAL_PROMPT.format(
                concept_term=case.concept_term,
                concept_definition=case.concept_definition,
                test_response=case.test_response,
            )

            raw_response = self.generate_fn(prompt)
            score, reasoning = self._parse_concept_response(raw_response)

            # Check if score is in expected range
            in_range = case.expected_score_range[0] <= score <= case.expected_score_range[1]

            results.append(JudgeEvalResult(
                case_id=case_id,
                expected_quality=case.quality,
                expected_score_range=case.expected_score_range,
                judge_score=score,
                judge_reasoning=reasoning,
                score_in_range=in_range,
                raw_response=raw_response,
            ))

            # Update quality tracking
            quality_total[case.quality.value] += 1
            if in_range:
                quality_correct[case.quality.value] += 1

            logger.debug(
                f"[{i+1}/{len(cases)}] {case.concept_term} ({case.quality.value}): "
                f"score={score}, expected={case.expected_score_range}, {'PASS' if in_range else 'FAIL'}"
            )

        # Compute discrimination accuracy per quality level
        discrimination = {
            q: (quality_correct[q] / quality_total[q] if quality_total[q] > 0 else 0)
            for q in quality_correct
        }

        return results, discrimination

    def evaluate_reasoning(
        self,
        cases: Optional[List[ReasoningCase]] = None,
    ) -> Tuple[List[Dict], float]:
        """
        Evaluate judge on reasoning chain validity assessment.

        Returns (results, accuracy)
        """
        cases = cases or REASONING_CASES
        results = []
        correct = 0

        for i, case in enumerate(cases):
            prompt = self.REASONING_EVAL_PROMPT.format(
                problem=case.problem,
                reasoning_chain=case.reasoning_chain,
                conclusion=case.conclusion,
            )

            raw_response = self.generate_fn(prompt)
            judge_valid, judge_errors, reasoning = self._parse_reasoning_response(raw_response)

            is_correct = judge_valid == case.is_valid
            if is_correct:
                correct += 1

            results.append({
                'problem': case.problem[:100],
                'expected_valid': case.is_valid,
                'judge_valid': judge_valid,
                'correct': is_correct,
                'judge_errors': judge_errors,
                'actual_errors': case.errors,
                'reasoning': reasoning[:200],
            })

            logger.debug(
                f"[{i+1}/{len(cases)}] Reasoning ({case.category}): "
                f"expected={case.is_valid}, got={judge_valid}, {'PASS' if is_correct else 'FAIL'}"
            )

        accuracy = correct / len(cases) if cases else 0
        return results, accuracy

    def run_full_evaluation(self) -> JudgeEvalReport:
        """Run complete evaluation suite and generate report."""
        logger.info(f"Running full judge evaluation for {self.model_id}")

        # Concept judging
        concept_results, discrimination = self.evaluate_concept_judging()
        concept_accuracy = sum(1 for r in concept_results if r.score_in_range) / len(concept_results)

        # Reasoning evaluation
        reasoning_results, reasoning_accuracy = self.evaluate_reasoning()

        # Overall score (weighted)
        overall = 0.7 * concept_accuracy + 0.3 * reasoning_accuracy

        # Generate recommendation
        if overall >= 0.8:
            recommendation = f"Excellent judge candidate. Strong discrimination ({concept_accuracy:.0%}) and reasoning ({reasoning_accuracy:.0%})."
        elif overall >= 0.6:
            recommendation = f"Acceptable judge. May struggle with edge cases. Concept: {concept_accuracy:.0%}, Reasoning: {reasoning_accuracy:.0%}."
        else:
            weak_areas = []
            if concept_accuracy < 0.6:
                weak_areas.append("concept scoring")
            if reasoning_accuracy < 0.6:
                weak_areas.append("reasoning validation")
            recommendation = f"Not recommended as judge. Weak in: {', '.join(weak_areas)}."

        return JudgeEvalReport(
            model_id=self.model_id,
            concept_cases_total=len(concept_results),
            concept_score_accuracy=concept_accuracy,
            quality_discrimination=discrimination,
            reasoning_cases_total=len(reasoning_results),
            reasoning_accuracy=reasoning_accuracy,
            concept_results=concept_results,
            reasoning_results=reasoning_results,
            overall_score=overall,
            recommendation=recommendation,
        )

    def _parse_concept_response(self, response: str) -> Tuple[float, str]:
        """Parse score and reasoning from concept evaluation response."""
        import re

        try:
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                score = float(data.get('score', 5))
                reasoning = data.get('reasoning', '')
                return score, reasoning
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: look for number
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response)
        if numbers:
            score = float(numbers[0])
            if 0 <= score <= 10:
                return score, response

        return 5.0, f"Could not parse: {response[:100]}"

    def _parse_reasoning_response(self, response: str) -> Tuple[bool, List[str], str]:
        """Parse validity judgment from reasoning evaluation response."""
        import re

        try:
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                # Handle nested structures carefully
                json_str = json_match.group()
                data = json.loads(json_str)
                valid = data.get('valid', False)
                errors = data.get('errors', [])
                reasoning = data.get('reasoning', '')
                return valid, errors, reasoning
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback
        response_lower = response.lower()
        if '"valid": true' in response_lower or '"valid":true' in response_lower:
            return True, [], response
        if '"valid": false' in response_lower or '"valid":false' in response_lower:
            return False, [], response

        # Last resort
        if 'invalid' in response_lower or 'not valid' in response_lower:
            return False, [], response
        if 'valid' in response_lower:
            return True, [], response

        return False, [], f"Could not parse: {response[:100]}"


def generate_markdown_report(report: JudgeEvalReport) -> str:
    """Generate markdown report from judge evaluation."""
    lines = [
        "# Judge Model Evaluation Report",
        "",
        f"**Model:** {report.model_id}",
        f"**Timestamp:** {report.timestamp.isoformat()}",
        "",
        "## Overall Assessment",
        "",
        f"**Overall Score:** {report.overall_score:.1%}",
        "",
        f"**Recommendation:** {report.recommendation}",
        "",
        "## Concept Explanation Judging",
        "",
        f"- Cases: {report.concept_cases_total}",
        f"- Score Accuracy: {report.concept_score_accuracy:.1%}",
        "",
        "### Discrimination by Quality Level",
        "",
        "| Quality Level | Accuracy |",
        "|---------------|----------|",
    ]

    for quality, acc in report.quality_discrimination.items():
        lines.append(f"| {quality} | {acc:.0%} |")

    lines.extend([
        "",
        "## Reasoning Evaluation",
        "",
        f"- Cases: {report.reasoning_cases_total}",
        f"- Accuracy: {report.reasoning_accuracy:.1%}",
        "",
        "## Detailed Concept Results",
        "",
        "| Concept | Expected Quality | Expected Range | Judge Score | Pass |",
        "|---------|------------------|----------------|-------------|------|",
    ])

    for r in report.concept_results:
        pass_mark = "Y" if r.score_in_range else "N"
        lines.append(
            f"| {r.case_id[:30]} | {r.expected_quality.value} | "
            f"{r.expected_score_range} | {r.judge_score:.1f} | {pass_mark} |"
        )

    lines.extend([
        "",
        "## Detailed Reasoning Results",
        "",
        "| Problem | Expected | Judge | Correct |",
        "|---------|----------|-------|---------|",
    ])

    for r in report.reasoning_results:
        lines.append(
            f"| {r['problem'][:40]}... | {r['expected_valid']} | "
            f"{r['judge_valid']} | {'Y' if r['correct'] else 'N'} |"
        )

    return "\n".join(lines)
