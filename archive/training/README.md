# Archived Training Code

This directory contains legacy training code that has been superseded by newer implementations.

## Files

### data_generation.py
**Status**: DEPRECATED
**Reason**: Replaced by `src/training/sumo_data_generation.py`
**Date Archived**: 2025-11-13

Simple prompt generation ("What is X?") without SUMO hierarchy awareness.
Superseded by SUMO-aware generation with:
- Category relationship prompts
- WordNet relationship prompts
- CamelCase splitting for better training

### text_probes.py
**Status**: DEPRECATED
**Reason**: Replaced by `src/training/embedding_centroids.py`
**Date Archived**: 2025-11-13

TF-IDF + LogisticRegression text classifiers.
Superseded by embedding centroid approach:
- Uses model embeddings directly instead of TF-IDF
- Cosine similarity at inference (no sklearn dependency)
- Better semantic matching

## Migration Notes

If you need to use this code for backward compatibility:
- Import directly: `from archive.training.data_generation import ...`
- Not exported from `src/training/__init__.py`
- Tests may not be maintained
