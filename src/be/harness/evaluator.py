"""
Concept evaluation for the graft testing harness.

Evaluates whether the target model understands concepts from the concept pack.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Iterator

from src.map.registry import ConceptPack, Concept, load_concept_pack
from .config import HarnessConfig
from .models import TargetModel, JudgeModel, JudgeScore

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating a single concept."""
    concept_id: str
    concept_term: str
    concept_definition: str
    layer: int

    # Target model's response
    target_response: str
    input_tokens: int
    output_tokens: int

    # Judge's evaluation
    score: float
    reasoning: str
    raw_judge_response: str

    # Derived
    knows_concept: bool

    # Training data availability
    has_training_data: bool
    n_positive_examples: int
    n_negative_examples: int

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "concept_id": self.concept_id,
            "concept_term": self.concept_term,
            "concept_definition": self.concept_definition,
            "layer": self.layer,
            "target_response": self.target_response,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "score": self.score,
            "reasoning": self.reasoning,
            "raw_judge_response": self.raw_judge_response,
            "knows_concept": self.knows_concept,
            "has_training_data": self.has_training_data,
            "n_positive_examples": self.n_positive_examples,
            "n_negative_examples": self.n_negative_examples,
        }


class ConceptEvaluator:
    """
    Evaluates target model's understanding of concepts.

    Uses a judge model to score the target's explanations.
    """

    def __init__(
        self,
        target: TargetModel,
        judge: JudgeModel,
        config: HarnessConfig,
    ):
        self.target = target
        self.judge = judge
        self.config = config

        # Load concept pack
        logger.info(f"Loading concept pack from {config.concept_pack_path}")
        self.pack = load_concept_pack(str(config.concept_pack_path))
        logger.info(f"Loaded {len(self.pack.concepts)} concepts")

    def _get_concepts_with_training_data(self) -> List[Concept]:
        """Get concepts that have training examples."""
        concepts = []
        for concept in self.pack.concepts.values():
            if hasattr(concept, 'training_hints') and concept.training_hints:
                hints = concept.training_hints
                if hints.get('positive_examples') or hints.get('negative_examples'):
                    concepts.append(concept)
        return concepts

    def _build_probe_prompt(self, concept: Concept) -> str:
        """Build a prompt to probe the target model about a concept."""
        return f"What is {concept.term}? Explain in 2-3 sentences."

    def evaluate_concept(self, concept: Concept) -> EvaluationResult:
        """Evaluate target model's understanding of a single concept."""
        # Build probe prompt and get target response
        prompt = self._build_probe_prompt(concept)
        result = self.target.generate(
            prompt,
            max_new_tokens=self.config.max_response_tokens,
            temperature=self.config.temperature,
        )

        # Get judge's score
        definition = concept.definition if hasattr(concept, 'definition') else ""
        judge_result = self.judge.score_concept_explanation(
            concept_term=concept.term,
            concept_definition=definition,
            target_response=result.text,
        )

        # Check for training data
        training_hints = getattr(concept, 'training_hints', {}) or {}
        positive_examples = training_hints.get('positive_examples', [])
        negative_examples = training_hints.get('negative_examples', [])

        return EvaluationResult(
            concept_id=concept.id if hasattr(concept, 'id') else concept.term,
            concept_term=concept.term,
            concept_definition=definition,
            layer=concept.layer if hasattr(concept, 'layer') else 0,
            target_response=result.text,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            score=judge_result.score,
            reasoning=judge_result.reasoning,
            raw_judge_response=judge_result.raw_response,
            knows_concept=judge_result.score >= self.config.knowledge_threshold,
            has_training_data=bool(positive_examples or negative_examples),
            n_positive_examples=len(positive_examples),
            n_negative_examples=len(negative_examples),
        )

    def evaluate_concepts(
        self,
        concept_ids: Optional[List[str]] = None,
        max_concepts: Optional[int] = None,
        with_training_data_only: bool = False,
    ) -> Iterator[EvaluationResult]:
        """
        Evaluate multiple concepts.

        Args:
            concept_ids: Specific concepts to evaluate. If None, evaluate all.
            max_concepts: Maximum number of concepts to evaluate.
            with_training_data_only: Only evaluate concepts that have training data.

        Yields:
            EvaluationResult for each concept.
        """
        if concept_ids:
            concepts = [
                self.pack.get_concept(cid)
                for cid in concept_ids
                if self.pack.get_concept(cid) is not None
            ]
        elif with_training_data_only:
            concepts = self._get_concepts_with_training_data()
        else:
            concepts = list(self.pack.concepts.values())

        if max_concepts:
            concepts = concepts[:max_concepts]

        logger.info(f"Evaluating {len(concepts)} concepts")

        for i, concept in enumerate(concepts):
            logger.debug(f"Evaluating concept {i+1}/{len(concepts)}: {concept.term}")
            try:
                yield self.evaluate_concept(concept)
            except Exception as e:
                logger.error(f"Error evaluating {concept.term}: {e}")
                continue

    def evaluate_all(
        self,
        max_concepts: Optional[int] = None,
        with_training_data_only: bool = False,
    ) -> List[EvaluationResult]:
        """
        Evaluate all concepts and return results list.

        Args:
            max_concepts: Maximum number of concepts to evaluate.
            with_training_data_only: Only evaluate concepts that have training data.

        Returns:
            List of EvaluationResult objects.
        """
        max_concepts = max_concepts or self.config.max_concepts
        return list(self.evaluate_concepts(
            max_concepts=max_concepts,
            with_training_data_only=with_training_data_only,
        ))

    def find_unknown_concepts(
        self,
        results: List[EvaluationResult],
        with_training_data: bool = True,
    ) -> List[EvaluationResult]:
        """
        Find concepts the target model doesn't know.

        Args:
            results: Evaluation results to filter.
            with_training_data: Only return concepts that have training data
                               (required for grafting).

        Returns:
            List of EvaluationResult for unknown concepts.
        """
        unknown = [r for r in results if not r.knows_concept]

        if with_training_data:
            unknown = [r for r in unknown if r.has_training_data]

        # Sort by score (lowest first - least known)
        unknown.sort(key=lambda r: r.score)

        return unknown

    def find_known_concepts(
        self,
        results: List[EvaluationResult],
    ) -> List[EvaluationResult]:
        """Find concepts the target model knows."""
        known = [r for r in results if r.knows_concept]
        # Sort by score (highest first - most confident)
        known.sort(key=lambda r: -r.score)
        return known
