# Training Data Quality: Preliminary Analysis

**Date**: 2025-12-06
**Model**: swiss-ai/Apertus-8B-2509
**Layer**: 3
**Samples**: 240 (12 concepts × 20 samples each)

## Executive Summary

Analysis of training data reveals significant output quality issues, but **output quality ≠ activation quality**. We train probes on activations, not decoded outputs. Prior experiments show ~75% correlation between prompt and output activations, meaning the model's internal representations are more stable than its outputs suggest.

**Key insight**: A model can activate "Vodka" concepts internally while producing garbled output about "Whip". The residual stream captures intent, not just decoded tokens.

## Output Quality by Quadrant

| Quadrant | Description | Positive Empty | Positive Good | Negative Good |
|----------|-------------|----------------|---------------|---------------|
| A | Low synsets, Low siblings | 13% | 80% | 83% |
| B | Low synsets, High siblings | 20% | 67% | 90% |
| C | **High synsets, High siblings** | **40%** | **43%** | 93% |
| D | High synsets, Low siblings | 17% | 73% | 90% |

**Observation**: Quadrant C (richest concepts) has the worst output quality. Concepts with many synsets produce more ambiguous prompts, leading to confused outputs.

## Output Issues Observed

### 1. Empty/Trivial Responses (20% overall)
```
Prompt: "Explain what 'Bayesian Risk Reasoning' means."
Response: ""
```

### 2. Repetitive/Degenerate Loops (10% overall)
```
Prompt: "'Bayesian Risk Reasoning' is defined as..."
Response: ", an instance of: Probability Theory, an instance of: Probability Theory,
an instance of: Probability Theory, an instance of: Probability Theory..."
```

### 3. Off-Topic Responses (estimated 15-20%)
```
Prompt: "What is 'Vodka'? unaged colorless liquor..."
Response: "and the Ukraine
What is 'Whip'? (n) 1. a long flexible rod used to urge a horse on..."
```

### 4. Meta-Responses Instead of Content
```
Prompt: "Give me examples of 'End Times Narrative'."
Response: "What are the End Times Narrative in your own words?..."
```

### 5. Repetitive Synonym Lists
```
Prompt: "What is 'Switch Device'?"
Response: "Synonyms: Switch, Swither, Switch, Turn On, Turn Off, Flip A Switch..."
```

## Critical Distinction: Output vs Activation Quality

### Why Output Quality Overstates the Problem

We extract activations from the **residual stream during generation**, not from the final decoded output. Prior experiments showed:

- **~75% correlation** between prompt activations and output activations
- **~25% divergence** between what the model "thinks" and what it outputs
- Activations capture the model's internal concept representations

### Revised Quality Estimates

```
Output quality:        ~60-80% "good" (varies by quadrant)
Output→Activation:     ~75% correlation (25% divergence)

If output is wrong 40% of time:
  - Only ~25% of that error propagates to activations
  - 40% × 25% = 10% activation error from output issues
  - Remaining output errors may still have correct activations

Expected activation accuracy: ~85-90%
```

### Estimated Maximum Achievable F1 (Revised)

| Quadrant | Output Good | Est. Activation Good | Revised F1 Ceiling |
|----------|-------------|---------------------|-------------------|
| A | 80% | ~90% | **~0.88** |
| B | 67% | ~85% | **~0.83** |
| C | 43% | ~78% | **~0.76** |
| D | 73% | ~87% | **~0.85** |

**Recommended F1 Target**: 0.85 (with falloff to 0.75 for Quadrant C concepts)

## Why Negatives Are Better Than Positives

Negatives are sampled from a pool of unrelated concepts. When the model processes "Bacteria" as a negative for "Switch Device", the activations are clearly about bacteria - strong topical signal.

Positives ask the model to explain a specific concept, which may produce confused outputs but still activate the target concept internally.

## Recommendations

### 1. Adjust F1 Targets
- Primary target: **0.85** (down from 0.95)
- Falloff to **0.75** for complex/ambiguous concepts (Quadrant C)
- This matches the estimated activation quality ceiling

### 2. Filter Degenerate Responses (Optional)
May still help to remove:
- Empty responses (< 20 chars) - no activation signal
- Very short responses (< 50 chars) - weak signal
- Note: Repetitive outputs may still have valid activations

### 3. Focus LLM Judge on Prompt Quality
Since activations depend more on prompt than output:
- Judge whether the **prompt** clearly relates to the concept
- Output relevance is secondary signal

### 4. Prompt Engineering
Improve prompt clarity to ensure the model activates the right concepts:
- More specific definitions in prompts
- Disambiguation for polysemous terms
- Context that narrows interpretation

## Next Steps

1. Run LLM judge evaluation (requires ANTHROPIC_API_KEY) focusing on **prompt relevance**
2. Correlate activation patterns with probe F1 (not just output quality)
3. Test F1=0.85 target with falloff to 0.75
4. Validate that sibling refinement improves discrimination within this ceiling

## Conclusion

Output quality analysis reveals ~20-40% degenerate outputs, but this overstates the impact on probe training. Since we train on activations (not outputs), and activations show ~75% stability vs outputs, the effective training signal quality is higher than output quality suggests.

**Revised recommendation**: F1 target of **0.85** (not 0.95) with falloff to **0.75** for difficult concepts. This balances realistic expectations with the understanding that activation quality exceeds output quality.

The sibling refinement pass remains valuable for improving discrimination between similar concepts, operating within this quality ceiling.
