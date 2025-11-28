# Extraction Strategy Comparison - Experimental Design Issue

## Summary

The extraction strategy comparison experiment produced identical F1 scores (0.9523809523809523) for baseline-20 and combined-20 on the Attribute concept. This appears to be due to a **fundamental experimental design flaw**.

## The Problem

### Current Test Methodology (FLAWED)

All three strategies use **baseline-20 extraction for testing** (line 194 in `compare_extraction_strategies.py`):

```python
# ALL strategies test with this:
test_pos_acts = extract_generation_only(model, tokenizer, test_pos, max_tokens=20, device=device)
test_neg_acts = extract_generation_only(model, tokenizer, test_neg, max_tokens=20, device=device)
```

But the training data comes from different distributions:
- **baseline-20**: Trains on generation-only, 20 tokens
- **combined-20**: Trains on prompt+generation (2x samples), 20 tokens
- **long-40**: Trains on generation-only, 40 tokens

### What This Tests (Not What We Want)

The current experiment tests: **"Do probes trained on different data distributions generalize to the same baseline-20 test distribution?"**

What we actually want to test: **"Do different extraction strategies produce better concept probes?"**

## Why This Matters

1. **Distribution Mismatch**:
   - combined-20 probe is trained on activations from BOTH prompt processing and generation
   - But tested only on generation activations
   - This tests transfer learning, not probe quality

2. **Ceiling Effect**:
   - With only 20 test samples, F1 = 0.952 means exactly 1 error
   - Too granular to detect meaningful differences

3. **Unfair Comparison**:
   - long-40 probe learns from 40-token context windows
   - But tested on 20-token windows
   - Artificially penalizes this strategy

## Correct Experimental Design

### Option A: Match Test to Training (Recommended)

Each strategy should be tested with its own extraction method:

```python
if strategy == 'baseline-20':
    test_acts = extract_generation_only(model, tokenizer, test_prompts, max_tokens=20)
elif strategy == 'combined-20':
    test_acts = extract_prompt_and_generation(model, tokenizer, test_prompts, max_tokens=20)
elif strategy == 'long-40':
    test_acts = extract_generation_only(model, tokenizer, test_prompts, max_tokens=40)
```

**Pros**: Fair comparison of probe quality for each strategy's deployment scenario
**Cons**: Different test conditions make comparison less direct

### Option B: Use ALL Extraction Methods for Testing

Test each probe against all three extraction methods:

```python
# Create 3 test sets with different extraction methods
test_baseline20 = extract_generation_only(..., max_tokens=20)
test_combined20 = extract_prompt_and_generation(..., max_tokens=20)
test_long40 = extract_generation_only(..., max_tokens=40)

# Test each probe on all three
for probe_strategy in ['baseline-20', 'combined-20', 'long-40']:
    for test_strategy in ['baseline-20', 'combined-20', 'long-40']:
        f1 = test_probe(probe_strategy, test_strategy)
```

**Pros**: Shows which probes generalize best across different extraction methods
**Cons**: More complex, harder to interpret

### Option C: Larger Shared Test Set

Keep current design but use much larger test set (e.g., 100+100 samples):

**Pros**: Simpler to implement, easier to detect real differences
**Cons**: Still has distribution mismatch problem

## Current Results Analysis

Given the experimental design issue, the identical F1 scores for baseline-20 and combined-20 could mean:

1. **Genuine result**: Both strategies produce equally good probes that transfer to baseline-20 extraction
2. **Ceiling effect**: Test set too small (20 samples) to detect differences
3. **Shared random state**: Test sets might be identical due to unseeded random generation

The perfect F1 (1.000) for Carnivore across all strategies confirms ceiling effect - this concept is too easy to classify.

## Recommendation

Rerun experiment with **Option A** - match test extraction to training extraction. This gives fair comparison of each strategy in its intended deployment scenario.

Also:
- Increase test set to 50+50 samples minimum
- Set explicit random seed for reproducibility
- Add more concepts beyond just Attribute and Carnivore
- Save test predictions to verify they're actually different

## Code Location

Issue is in `/home/poss/Documents/Code/HatCat/scripts/compare_extraction_strategies.py` lines 181-195.
