"""
Meld designer for the graft testing harness.

Creates melds (training data) for concepts the target model doesn't know.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.map.registry import Concept
from .models import JudgeModel

logger = logging.getLogger(__name__)


@dataclass
class MeldData:
    """Training data for scion training."""
    concept_id: str
    concept_term: str
    concept_definition: str
    positive_examples: List[str]
    negative_examples: List[str]
    source: str  # "pack", "meld", or "generated"

    def to_dict(self) -> dict:
        return {
            "concept_id": self.concept_id,
            "concept_term": self.concept_term,
            "concept_definition": self.concept_definition,
            "positive_examples": self.positive_examples,
            "negative_examples": self.negative_examples,
            "source": self.source,
        }


class MeldDesigner:
    """
    Designs melds (training data) for concepts.

    Can extract from existing concept training hints or generate new examples.
    """

    def __init__(
        self,
        judge: Optional[JudgeModel] = None,
        melds_path: Optional[Path] = None,
    ):
        """
        Initialize the meld designer.

        Args:
            judge: JudgeModel for generating examples (optional).
            melds_path: Path to applied melds directory.
        """
        self.judge = judge
        self.melds_path = melds_path

        # Cache loaded melds
        self._meld_cache: Dict[str, dict] = {}

    def _load_melds(self) -> Dict[str, dict]:
        """Load all applied melds and index by concept term."""
        if self._meld_cache:
            return self._meld_cache

        if not self.melds_path or not self.melds_path.exists():
            return {}

        for meld_file in self.melds_path.glob("*.json"):
            try:
                with open(meld_file) as f:
                    meld = json.load(f)

                # Index by concept term
                for candidate in meld.get("candidates", []):
                    term = candidate.get("term", "")
                    if term:
                        self._meld_cache[term] = candidate

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load meld {meld_file}: {e}")

        logger.info(f"Loaded {len(self._meld_cache)} concepts from melds")
        return self._meld_cache

    def get_training_data_from_concept(
        self,
        concept: Concept,
    ) -> Optional[MeldData]:
        """
        Extract training data from concept's training_hints.

        Args:
            concept: Concept with training_hints.

        Returns:
            MeldData if training hints exist, None otherwise.
        """
        training_hints = getattr(concept, 'training_hints', None)
        if not training_hints:
            return None

        positive = training_hints.get('positive_examples', [])
        negative = training_hints.get('negative_examples', [])

        if not positive and not negative:
            return None

        concept_id = concept.id if hasattr(concept, 'id') else concept.term
        definition = concept.definition if hasattr(concept, 'definition') else ""

        return MeldData(
            concept_id=concept_id,
            concept_term=concept.term,
            concept_definition=definition,
            positive_examples=positive,
            negative_examples=negative,
            source="pack",
        )

    def get_training_data_from_meld(
        self,
        concept_term: str,
    ) -> Optional[MeldData]:
        """
        Extract training data from applied melds.

        Args:
            concept_term: Term to look up in melds.

        Returns:
            MeldData if found in melds, None otherwise.
        """
        melds = self._load_melds()
        candidate = melds.get(concept_term)

        if not candidate:
            return None

        training_hints = candidate.get('training_hints', {})
        positive = training_hints.get('positive_examples', [])
        negative = training_hints.get('negative_examples', [])

        if not positive and not negative:
            return None

        return MeldData(
            concept_id=concept_term,
            concept_term=concept_term,
            concept_definition=candidate.get('definition', ''),
            positive_examples=positive,
            negative_examples=negative,
            source="meld",
        )

    def generate_training_data(
        self,
        concept_term: str,
        concept_definition: str,
        n_positive: int = 10,
        n_negative: int = 10,
    ) -> Optional[MeldData]:
        """
        Generate training data using the judge model.

        Args:
            concept_term: Concept term.
            concept_definition: Concept definition.
            n_positive: Number of positive examples to generate.
            n_negative: Number of negative examples to generate.

        Returns:
            MeldData with generated examples, None if generation fails.
        """
        if not self.judge:
            logger.warning("No judge model available for example generation")
            return None

        logger.info(f"Generating training examples for: {concept_term}")

        positive, negative = self.judge.generate_training_examples(
            concept_term=concept_term,
            concept_definition=concept_definition,
            n_positive=n_positive,
            n_negative=n_negative,
        )

        if not positive or not negative:
            logger.warning(f"Failed to generate examples for {concept_term}")
            return None

        return MeldData(
            concept_id=concept_term,
            concept_term=concept_term,
            concept_definition=concept_definition,
            positive_examples=positive,
            negative_examples=negative,
            source="generated",
        )

    def get_or_create_training_data(
        self,
        concept: Concept,
        min_examples: int = 5,
    ) -> Optional[MeldData]:
        """
        Get training data from concept, melds, or generate it.

        Tries sources in order:
        1. Concept's training_hints
        2. Applied melds
        3. Generated by judge model

        Args:
            concept: The concept to get training data for.
            min_examples: Minimum examples needed.

        Returns:
            MeldData with sufficient examples, None if unable to obtain.
        """
        # Try concept first
        meld_data = self.get_training_data_from_concept(concept)
        if meld_data and len(meld_data.positive_examples) >= min_examples:
            logger.info(f"Using training data from concept pack for {concept.term}")
            return meld_data

        # Try melds
        meld_data = self.get_training_data_from_meld(concept.term)
        if meld_data and len(meld_data.positive_examples) >= min_examples:
            logger.info(f"Using training data from melds for {concept.term}")
            return meld_data

        # Generate
        definition = concept.definition if hasattr(concept, 'definition') else ""
        meld_data = self.generate_training_data(
            concept_term=concept.term,
            concept_definition=definition,
            n_positive=max(10, min_examples * 2),
            n_negative=max(10, min_examples * 2),
        )

        if meld_data:
            logger.info(f"Generated training data for {concept.term}")
            return meld_data

        logger.warning(f"Could not obtain training data for {concept.term}")
        return None

    def create_meld_file(
        self,
        meld_data: MeldData,
        output_path: Path,
        layer: int = 3,
        parent_concept: str = None,
    ) -> Path:
        """
        Create a meld JSON file from MeldData.

        Args:
            meld_data: Training data to convert.
            output_path: Where to save the meld file.
            layer: Layer hint for the concept.
            parent_concept: Parent concept ID (optional).

        Returns:
            Path to the created meld file.
        """
        meld = {
            "meld_request_id": f"org.hatcat/harness-generated-{meld_data.concept_term.lower()}@0.1.0",
            "target_pack_spec_id": "org.hatcat/sumo-wordnet-v4@5.0.0",
            "metadata": {
                "name": f"Harness-generated meld for {meld_data.concept_term}",
                "description": f"Training data for graft testing: {meld_data.concept_term}",
                "source": "harness_generated",
                "author": "graft-harness",
                "created": datetime.now().isoformat(),
            },
            "attachment_points": [],
            "candidates": [
                {
                    "term": meld_data.concept_term,
                    "role": "concept",
                    "parent_concepts": [parent_concept] if parent_concept else [],
                    "layer_hint": layer,
                    "definition": meld_data.concept_definition,
                    "domain": "Information",
                    "aliases": [],
                    "relationships": {},
                    "safety_tags": {
                        "risk_level": "low",
                        "impacts": [],
                        "treaty_relevant": False,
                        "harness_relevant": True,
                    },
                    "training_hints": {
                        "positive_examples": meld_data.positive_examples,
                        "negative_examples": meld_data.negative_examples,
                        "disambiguation": "",
                    },
                    "children": [],
                }
            ],
            "validation": {
                "status": "pending",
            },
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(meld, f, indent=2)

        logger.info(f"Created meld file: {output_path}")
        return output_path
