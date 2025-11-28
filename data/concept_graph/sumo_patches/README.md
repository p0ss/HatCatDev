# SUMO Patches

This directory contains patches to the SUMO ontology in KIF format.

## Purpose

These patches preserve:
- Parent relationships derived from WordNet hypernyms
- Manual parent assignments for orphaned concepts
- Semantic inference-based parent assignments
- Geographic hierarchy reorganization

## Structure

Each `.patch.kif` file corresponds to a source `.kif` file in `sumo_source/`.
The build script loads both the source and patch files when generating layers.

## Format

Patches use standard KIF syntax:
```
(subclass ChildConcept ParentConcept)  ; Reason for patch
```

Last updated: 2025-11-21
