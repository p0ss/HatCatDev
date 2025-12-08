# Training Data Quality Report

Generated: 2025-12-06T10:22:02.677447

## Configuration

- Concept Pack: sumo-wordnet-v4
- Model: swiss-ai/Apertus-8B-2509
- Layer: 3
- Judge Model: claude-3-5-haiku-latest

## Summary by Quadrant

### Quadrant A

Low positives (1-2 synsets), Low negatives (0-3 siblings)

- Concepts: BayesianRiskReasoning, AerobicExerciseDevice, MedicalClinic
- Avg Positive Relevance: 2.83/5
- Avg Negative Relevance: 1.40/5
- Relevance Gap: 1.43
- Avg Lens F1: 0.663

### Quadrant B

Low positives (1-2 synsets), High negatives (10+ siblings)

- Concepts: Vodka, PyridostigmineBromide, EndTimesNarrative
- Avg Positive Relevance: 2.57/5
- Avg Negative Relevance: 1.27/5
- Relevance Gap: 1.30
- Avg Lens F1: 0.317

### Quadrant C

High positives (5+ synsets), High negatives (10+ siblings)

- Concepts: SportsGround, ChestOrCabinet, SwitchDevice
- Avg Positive Relevance: 1.97/5
- Avg Negative Relevance: 1.27/5
- Relevance Gap: 0.70
- Avg Lens F1: 0.333

### Quadrant D

High positives (5+ synsets), Low negatives (0-3 siblings)

- Concepts: StringInstrument, LegislativeChamber, Demonstrating
- Avg Positive Relevance: 2.80/5
- Avg Negative Relevance: 1.30/5
- Relevance Gap: 1.50
- Avg Lens F1: 0.250

## Overall Analysis

- Total samples analyzed: 240
- Overall Positive Relevance: 2.54/5
- Overall Negative Relevance: 1.31/5
- Overall Relevance Gap: 1.23
- Overall Lens F1: 0.391

## Interpretation

The relevance gap between positive and negative samples indicates how distinguishable our training data is. A larger gap suggests cleaner separation and higher achievable lens accuracy.

**Recommended F1 Target**: 0.70 (based on relevance gap of 1.23)
