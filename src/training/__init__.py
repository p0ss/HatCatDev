"""
Training module: SUMO-aware concept classifier training.

This module provides tools for:
- Training binary classifiers to detect SUMO concepts in activations
- Training text probes for fast token→concept mapping
- Generating SUMO-aware training prompts (with WordNet and category relationships)
- Extracting activations from language models
- Computing concept centroids from name embeddings

NOTE: src/training/data_generation.py contains legacy training functions.
      Use sumo_data_generation.py for new training (SUMO + WordNet hierarchy aware).
"""

from .classifier import train_binary_classifier, BinaryClassifier
from .activations import get_mean_activation
from .sumo_classifiers import train_sumo_classifiers
from .sumo_data_generation import (
    create_sumo_training_dataset,
    build_sumo_negative_pool,
    split_camel_case,
)
from .probe_validation import (
    validate_probe_calibration,
    validate_probe_set,
    infer_concept_domain,
)

__all__ = [
    "train_binary_classifier",
    "BinaryClassifier",
    "get_mean_activation",
    "train_sumo_classifiers",
    "create_sumo_training_dataset",
    "build_sumo_negative_pool",
    "split_camel_case",
    "validate_probe_calibration",
    "validate_probe_set",
    "infer_concept_domain",
]
