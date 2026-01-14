"""
Thalametrist - CAT for Cognitive Assessment

A Thalametrist is a CAT (Conjoined Adversarial Tomograph) performing
cognitive assessment. Like an optometrist evaluating vision, the
Thalametrist evaluates a subject's concept knowledge.

The Thalametrist:
- Poses questions about concepts to the subject
- Evaluates subject responses against concept definitions
- Produces structured assessments of concept knowledge
- Identifies gaps warranting prosthetic fitting (grafting)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import json
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConceptAssessment:
    """Assessment of a single concept."""
    concept_id: str
    concept_term: str
    concept_definition: str

    # Subject's response
    prompt_used: str
    subject_response: str

    # Thalametrist evaluation
    score: float  # 0-10 scale
    is_known: bool  # True if score >= threshold
    reasoning: str

    # Lens data (if available)
    concept_activations: Dict[str, float] = field(default_factory=dict)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'concept_id': self.concept_id,
            'concept_term': self.concept_term,
            'concept_definition': self.concept_definition,
            'prompt_used': self.prompt_used,
            'subject_response': self.subject_response,
            'score': self.score,
            'is_known': self.is_known,
            'reasoning': self.reasoning,
            'concept_activations': self.concept_activations,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class AssessmentResult:
    """Result of a full assessment session."""
    session_id: str
    subject_model: str
    practitioner_model: str

    # Assessed concepts
    assessments: List[ConceptAssessment]

    # Summary statistics
    total_concepts: int = 0
    known_count: int = 0
    unknown_count: int = 0
    mean_score: float = 0.0

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.assessments:
            self.total_concepts = len(self.assessments)
            self.known_count = sum(1 for a in self.assessments if a.is_known)
            self.unknown_count = self.total_concepts - self.known_count
            self.mean_score = sum(a.score for a in self.assessments) / self.total_concepts

    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'subject_model': self.subject_model,
            'practitioner_model': self.practitioner_model,
            'total_concepts': self.total_concepts,
            'known_count': self.known_count,
            'unknown_count': self.unknown_count,
            'mean_score': self.mean_score,
            'timestamp': self.timestamp.isoformat(),
            'assessments': [a.to_dict() for a in self.assessments],
        }


class Thalametrist:
    """
    CAT performing cognitive assessment.

    The Thalametrist evaluates a subject's understanding of concepts by:
    1. Posing questions to the subject about each concept
    2. Analyzing the subject's response against the concept definition
    3. Scoring the response and determining if the concept is "known"

    The Thalametrist uses the ExaminationRoom's generation methods,
    operating as the CAT (practitioner) in the conjoined relationship.
    """

    # Prompt template for asking subject about a concept
    SUBJECT_PROMPT = """What is {concept_term}? Explain in 2-3 sentences."""

    # Prompt template for practitioner evaluation
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

    def __init__(
        self,
        room,  # ExaminationRoom - avoid circular import
        knowledge_threshold: float = 5.0,  # Score >= this means "known"
    ):
        self.room = room
        self.knowledge_threshold = knowledge_threshold
        self.assessments: List[ConceptAssessment] = []

        logger.info(f"Thalametrist initialized (threshold={knowledge_threshold})")

    # =========================================================================
    # Assessment Methods
    # =========================================================================

    def assess_concept(
        self,
        concept_id: str,
        concept_term: str,
        concept_definition: str,
        custom_prompt: Optional[str] = None,
    ) -> ConceptAssessment:
        """
        Assess the subject's knowledge of a single concept.

        Steps:
        1. Ask subject about the concept
        2. Get subject's response (via BEDFrame for instrumentation)
        3. Have practitioner (self) evaluate the response
        4. Return structured assessment
        """
        # Step 1: Construct subject prompt
        subject_prompt = custom_prompt or self.SUBJECT_PROMPT.format(
            concept_term=concept_term
        )

        # Step 2: Get subject response
        logger.debug(f"Asking subject about: {concept_term}")
        subject_response = self.room.subject_generate(
            subject_prompt,
            use_bedframe=True,
        )

        # Get lens data from recent ticks
        concept_activations = {}
        lens_traces = self.room.get_lens_traces()
        if 'concept' in lens_traces:
            for concept, trace in lens_traces['concept'].items():
                if trace:
                    # Get most recent activation
                    concept_activations[concept] = trace[-1]['score']

        # Step 3: Evaluate response
        score, reasoning = self._evaluate_response(
            concept_term=concept_term,
            concept_definition=concept_definition,
            subject_response=subject_response,
        )

        # Step 4: Create assessment
        assessment = ConceptAssessment(
            concept_id=concept_id,
            concept_term=concept_term,
            concept_definition=concept_definition,
            prompt_used=subject_prompt,
            subject_response=subject_response,
            score=score,
            is_known=score >= self.knowledge_threshold,
            reasoning=reasoning,
            concept_activations=concept_activations,
        )

        self.assessments.append(assessment)

        logger.info(
            f"Assessment: {concept_term} = {score:.1f} "
            f"({'known' if assessment.is_known else 'unknown'})"
        )

        return assessment

    def assess_concepts(
        self,
        concepts: List[Dict[str, str]],
        max_concepts: Optional[int] = None,
    ) -> AssessmentResult:
        """
        Assess multiple concepts.

        Args:
            concepts: List of dicts with 'concept_id', 'term', 'definition'
            max_concepts: Limit number of concepts to assess

        Returns:
            AssessmentResult with all assessments
        """
        if max_concepts:
            concepts = concepts[:max_concepts]

        logger.info(f"Assessing {len(concepts)} concepts")

        assessments = []
        for i, concept in enumerate(concepts):
            logger.info(f"[{i+1}/{len(concepts)}] Assessing: {concept.get('term', concept.get('concept_id'))}")

            assessment = self.assess_concept(
                concept_id=concept.get('concept_id', f'concept_{i}'),
                concept_term=concept.get('term', concept.get('concept_id')),
                concept_definition=concept.get('definition', ''),
            )
            assessments.append(assessment)

        result = AssessmentResult(
            session_id=self.room.session_id,
            subject_model=self.room.config.subject_model_id,
            practitioner_model=self.room.config.practitioner_model_id or self.room.config.subject_model_id,
            assessments=assessments,
        )

        # Record to room
        self.room.record_procedure('concept_assessment', result.to_dict())

        return result

    # =========================================================================
    # Evaluation Logic
    # =========================================================================

    def _evaluate_response(
        self,
        concept_term: str,
        concept_definition: str,
        subject_response: str,
    ) -> Tuple[float, str]:
        """
        Evaluate subject's response against concept definition.

        Returns (score, reasoning) tuple.
        """
        # Construct evaluation prompt
        eval_prompt = self.EVALUATION_PROMPT.format(
            concept_term=concept_term,
            concept_definition=concept_definition,
            subject_response=subject_response,
        )

        # Get practitioner evaluation
        raw_response = self.room.practitioner_generate(
            eval_prompt,
            temperature=0.0,  # Greedy for consistency
        )

        # Parse response
        score, reasoning = self._parse_evaluation(raw_response)

        return score, reasoning

    def _parse_evaluation(self, response: str) -> Tuple[float, str]:
        """Parse JSON evaluation from practitioner response."""
        if not response or not response.strip():
            return 0.0, "Empty evaluation response"

        # Try to find JSON in response
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                score = float(data.get('score', 0))
                reasoning = data.get('reasoning', '')
                return min(max(score, 0), 10), reasoning
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: try to find a number
        number_match = re.search(r'\b(\d+(?:\.\d+)?)\b', response)
        if number_match:
            score = float(number_match.group(1))
            return min(max(score, 0), 10), response

        return 0.0, f"Could not parse: {response[:100]}"

    # =========================================================================
    # Gap Analysis
    # =========================================================================

    def identify_gaps(
        self,
        result: Optional[AssessmentResult] = None,
    ) -> List[ConceptAssessment]:
        """
        Identify concepts the subject doesn't know.

        These are candidates for grafting.
        """
        if result is None:
            result = AssessmentResult(
                session_id=self.room.session_id,
                subject_model=self.room.config.subject_model_id,
                practitioner_model=self.room.config.practitioner_model_id or '',
                assessments=self.assessments,
            )

        gaps = [a for a in result.assessments if not a.is_known]

        logger.info(
            f"Gap analysis: {len(gaps)} unknown concepts "
            f"out of {len(result.assessments)}"
        )

        return gaps

    def get_graft_candidates(
        self,
        result: Optional[AssessmentResult] = None,
        min_score: float = 0.0,
        max_score: float = 4.0,
    ) -> List[ConceptAssessment]:
        """
        Get concepts suitable for grafting.

        Filters for concepts that are:
        - Unknown (below knowledge threshold)
        - Within a score range (e.g., not completely unknown)
        """
        gaps = self.identify_gaps(result)

        candidates = [
            a for a in gaps
            if min_score <= a.score <= max_score
        ]

        # Sort by score (prioritize partial understanding)
        candidates.sort(key=lambda a: a.score, reverse=True)

        logger.info(
            f"Graft candidates: {len(candidates)} "
            f"(score range {min_score}-{max_score})"
        )

        return candidates

    # =========================================================================
    # Reporting
    # =========================================================================

    def generate_report(
        self,
        result: AssessmentResult,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a markdown report of the assessment."""
        lines = [
            f"# Concept Assessment Report",
            f"",
            f"**Session:** {result.session_id}",
            f"**Subject:** {result.subject_model}",
            f"**Practitioner:** {result.practitioner_model}",
            f"**Timestamp:** {result.timestamp.isoformat()}",
            f"",
            f"## Summary",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Concepts | {result.total_concepts} |",
            f"| Known | {result.known_count} ({100*result.known_count/max(result.total_concepts,1):.1f}%) |",
            f"| Unknown | {result.unknown_count} ({100*result.unknown_count/max(result.total_concepts,1):.1f}%) |",
            f"| Mean Score | {result.mean_score:.2f} |",
            f"| Knowledge Threshold | {self.knowledge_threshold} |",
            f"",
            f"## Concept Details",
            f"",
        ]

        for a in result.assessments:
            status = "Known" if a.is_known else "Unknown"
            lines.extend([
                f"### {a.concept_term}",
                f"",
                f"**Status:** {status} (score: {a.score:.1f}/10)",
                f"",
                f"**Definition:** {a.concept_definition}",
                f"",
                f"**Subject Response:**",
                f"> {a.subject_response}",
                f"",
                f"**Evaluation:** {a.reasoning}",
                f"",
                f"---",
                f"",
            ])

        report = '\n'.join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")

        return report
