"""
Assessment Protocols for Thalametry

Protocols define how concepts are assessed. Different protocols are suited
for different types of concepts and assessment goals:

- ExplanationProtocol: Ask subject to explain a concept, judge understanding
- ClassificationProtocol: Show examples, ask subject to classify as concept or not
- DiscriminationProtocol: Present positive/negative pairs, assess discrimination

Protocols are composable and can be combined for comprehensive assessment.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProtocolResult:
    """Result from running a protocol."""
    protocol_name: str
    concept_id: str
    concept_term: str

    # Raw data
    prompts_used: List[str]
    subject_responses: List[str]

    # Evaluation
    score: float  # 0-10 scale
    sub_scores: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'protocol_name': self.protocol_name,
            'concept_id': self.concept_id,
            'concept_term': self.concept_term,
            'prompts_used': self.prompts_used,
            'subject_responses': self.subject_responses,
            'score': self.score,
            'sub_scores': self.sub_scores,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat(),
        }


class AssessmentProtocol(ABC):
    """Base class for assessment protocols."""

    name: str = "base"

    @abstractmethod
    def run(
        self,
        room,  # ExaminationRoom
        concept_id: str,
        concept_term: str,
        concept_definition: str,
        **kwargs,
    ) -> ProtocolResult:
        """Run the protocol and return result."""
        pass


class ExplanationProtocol(AssessmentProtocol):
    """
    Ask subject to explain a concept, then judge understanding.

    This is the primary protocol for assessing conceptual knowledge.
    The subject is asked to explain the concept in their own words,
    and the practitioner evaluates the explanation against the definition.
    """

    name = "explanation"

    # Subject prompt template
    SUBJECT_PROMPT = """What is {concept_term}? Explain in 2-3 sentences."""

    # Practitioner evaluation template
    EVALUATION_PROMPT = """You are evaluating whether a model understands a concept.

Concept: {concept_term}
Definition: {concept_definition}

Model's explanation:
{subject_response}

Rate the model's understanding from 0-10:
- 0-3: No understanding or fundamentally wrong
- 4-6: Partial understanding, missing key aspects
- 7-9: Good understanding with minor gaps
- 10: Complete and accurate understanding

Respond with JSON only: {{"score": N, "reasoning": "brief explanation"}}"""

    def run(
        self,
        room,
        concept_id: str,
        concept_term: str,
        concept_definition: str,
        custom_prompt: Optional[str] = None,
        **kwargs,
    ) -> ProtocolResult:
        """Run explanation protocol."""
        import json
        import re

        # Step 1: Ask subject to explain
        prompt = custom_prompt or self.SUBJECT_PROMPT.format(
            concept_term=concept_term
        )

        logger.debug(f"ExplanationProtocol: asking about {concept_term}")
        subject_response = room.subject_generate(prompt, use_bedframe=True)

        # Step 2: Have practitioner evaluate
        eval_prompt = self.EVALUATION_PROMPT.format(
            concept_term=concept_term,
            concept_definition=concept_definition,
            subject_response=subject_response,
        )

        raw_eval = room.practitioner_generate(eval_prompt, temperature=0.0)

        # Step 3: Parse evaluation
        score = 0.0
        reasoning = ""

        if raw_eval:
            json_match = re.search(r'\{[^}]+\}', raw_eval)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    score = float(data.get('score', 0))
                    reasoning = data.get('reasoning', '')
                except (json.JSONDecodeError, ValueError):
                    pass

            if score == 0:
                # Fallback: try to find number
                num_match = re.search(r'\b(\d+(?:\.\d+)?)\b', raw_eval)
                if num_match:
                    score = float(num_match.group(1))
                    reasoning = raw_eval

        score = min(max(score, 0), 10)

        return ProtocolResult(
            protocol_name=self.name,
            concept_id=concept_id,
            concept_term=concept_term,
            prompts_used=[prompt],
            subject_responses=[subject_response],
            score=score,
            reasoning=reasoning,
        )


class ClassificationProtocol(AssessmentProtocol):
    """
    Show examples, ask subject to classify as matching concept or not.

    This protocol tests whether the subject can recognize instances
    of a concept. Requires training examples (positive/negative).
    """

    name = "classification"

    # Subject prompt for classification
    CLASSIFY_PROMPT = """Does the following example demonstrate the concept "{concept_term}"?

Example: {example}

Answer with "Yes" or "No" and a brief explanation."""

    # Practitioner evaluation for classification accuracy
    EVAL_PROMPT = """You are checking if a classification is correct.

Concept: {concept_term}
Definition: {concept_definition}

Example shown: {example}
Correct classification: {is_positive}

Model's response: {subject_response}

Did the model correctly classify this example?
Respond with JSON: {{"correct": true/false, "reasoning": "..."}}"""

    def run(
        self,
        room,
        concept_id: str,
        concept_term: str,
        concept_definition: str,
        positive_examples: Optional[List[str]] = None,
        negative_examples: Optional[List[str]] = None,
        n_examples: int = 4,
        **kwargs,
    ) -> ProtocolResult:
        """Run classification protocol."""
        import json
        import re

        positive_examples = positive_examples or []
        negative_examples = negative_examples or []

        if not positive_examples and not negative_examples:
            logger.warning(f"ClassificationProtocol: no examples for {concept_term}")
            return ProtocolResult(
                protocol_name=self.name,
                concept_id=concept_id,
                concept_term=concept_term,
                prompts_used=[],
                subject_responses=[],
                score=0.0,
                reasoning="No examples provided",
            )

        # Select examples
        n_pos = min(len(positive_examples), n_examples // 2)
        n_neg = min(len(negative_examples), n_examples - n_pos)

        examples = [
            (ex, True) for ex in positive_examples[:n_pos]
        ] + [
            (ex, False) for ex in negative_examples[:n_neg]
        ]

        prompts = []
        responses = []
        correct = 0
        total = len(examples)

        for example, is_positive in examples:
            # Ask subject to classify
            prompt = self.CLASSIFY_PROMPT.format(
                concept_term=concept_term,
                example=example,
            )
            prompts.append(prompt)

            subject_response = room.subject_generate(prompt, use_bedframe=True)
            responses.append(subject_response)

            # Check if correct
            eval_prompt = self.EVAL_PROMPT.format(
                concept_term=concept_term,
                concept_definition=concept_definition,
                example=example,
                is_positive="Positive (matches concept)" if is_positive else "Negative (does not match)",
                subject_response=subject_response,
            )

            raw_eval = room.practitioner_generate(eval_prompt, temperature=0.0)

            if raw_eval:
                json_match = re.search(r'\{[^}]+\}', raw_eval)
                if json_match:
                    try:
                        data = json.loads(json_match.group())
                        if data.get('correct', False):
                            correct += 1
                    except (json.JSONDecodeError, ValueError):
                        pass

        # Score is accuracy * 10
        score = (correct / total * 10) if total > 0 else 0.0

        return ProtocolResult(
            protocol_name=self.name,
            concept_id=concept_id,
            concept_term=concept_term,
            prompts_used=prompts,
            subject_responses=responses,
            score=score,
            sub_scores={'correct': correct, 'total': total, 'accuracy': correct/total if total else 0},
            reasoning=f"Classified {correct}/{total} examples correctly",
        )


class DiscriminationProtocol(AssessmentProtocol):
    """
    Present positive/negative pairs, assess ability to discriminate.

    This protocol tests whether the subject can distinguish between
    examples that do and don't match a concept.
    """

    name = "discrimination"

    DISCRIMINATE_PROMPT = """Which of these two examples better demonstrates the concept "{concept_term}"?

Example A: {example_a}

Example B: {example_b}

Answer "A" or "B" and explain why."""

    def run(
        self,
        room,
        concept_id: str,
        concept_term: str,
        concept_definition: str,
        positive_examples: Optional[List[str]] = None,
        negative_examples: Optional[List[str]] = None,
        n_pairs: int = 3,
        **kwargs,
    ) -> ProtocolResult:
        """Run discrimination protocol."""
        import random

        positive_examples = positive_examples or []
        negative_examples = negative_examples or []

        if not positive_examples or not negative_examples:
            logger.warning(f"DiscriminationProtocol: need both pos/neg examples for {concept_term}")
            return ProtocolResult(
                protocol_name=self.name,
                concept_id=concept_id,
                concept_term=concept_term,
                prompts_used=[],
                subject_responses=[],
                score=0.0,
                reasoning="Need both positive and negative examples",
            )

        # Create pairs
        n_pairs = min(n_pairs, len(positive_examples), len(negative_examples))

        prompts = []
        responses = []
        correct = 0

        for i in range(n_pairs):
            pos = positive_examples[i % len(positive_examples)]
            neg = negative_examples[i % len(negative_examples)]

            # Randomize order
            if random.random() > 0.5:
                example_a, example_b = pos, neg
                correct_answer = "A"
            else:
                example_a, example_b = neg, pos
                correct_answer = "B"

            prompt = self.DISCRIMINATE_PROMPT.format(
                concept_term=concept_term,
                example_a=example_a,
                example_b=example_b,
            )
            prompts.append(prompt)

            response = room.subject_generate(prompt, use_bedframe=True)
            responses.append(response)

            # Check answer
            response_upper = response.upper().strip()
            if response_upper.startswith(correct_answer):
                correct += 1

        score = (correct / n_pairs * 10) if n_pairs > 0 else 0.0

        return ProtocolResult(
            protocol_name=self.name,
            concept_id=concept_id,
            concept_term=concept_term,
            prompts_used=prompts,
            subject_responses=responses,
            score=score,
            sub_scores={'correct': correct, 'total': n_pairs, 'accuracy': correct/n_pairs if n_pairs else 0},
            reasoning=f"Correctly discriminated {correct}/{n_pairs} pairs",
        )


class CompositeProtocol(AssessmentProtocol):
    """
    Run multiple protocols and combine results.

    Allows comprehensive assessment using multiple approaches.
    """

    name = "composite"

    def __init__(
        self,
        protocols: List[AssessmentProtocol],
        weights: Optional[Dict[str, float]] = None,
    ):
        self.protocols = protocols
        self.weights = weights or {p.name: 1.0 for p in protocols}

    def run(
        self,
        room,
        concept_id: str,
        concept_term: str,
        concept_definition: str,
        **kwargs,
    ) -> ProtocolResult:
        """Run all protocols and combine results."""
        sub_results = []
        prompts = []
        responses = []
        sub_scores = {}

        for protocol in self.protocols:
            result = protocol.run(
                room,
                concept_id,
                concept_term,
                concept_definition,
                **kwargs,
            )
            sub_results.append(result)
            prompts.extend(result.prompts_used)
            responses.extend(result.subject_responses)
            sub_scores[protocol.name] = result.score

        # Weighted average
        total_weight = sum(self.weights.get(p.name, 1.0) for p in self.protocols)
        weighted_sum = sum(
            self.weights.get(r.protocol_name, 1.0) * r.score
            for r in sub_results
        )
        score = weighted_sum / total_weight if total_weight > 0 else 0.0

        reasoning = "; ".join(
            f"{r.protocol_name}: {r.score:.1f}"
            for r in sub_results
        )

        return ProtocolResult(
            protocol_name=self.name,
            concept_id=concept_id,
            concept_term=concept_term,
            prompts_used=prompts,
            subject_responses=responses,
            score=score,
            sub_scores=sub_scores,
            reasoning=reasoning,
        )
