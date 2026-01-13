# Root Cause Analysis: Layer 0 Abstraction Problem

## Problem (Initial Misdiagnosis - INCORRECT)

~~The HatCat concept detection system using `gemma-3-4b-pt_sumo-wordnet-v2` lens pack is showing **lens saturation** - all detected concepts have probability 1.0, making them useless for distinguishing between different semantic content.~~

**This diagnosis was WRONG.** The lenses work fine in other tests with the same lens pack.

## Actual Root Cause

The issue was **loading layer 0 concepts** which are foundational SUMO concepts that are **maximally abstract** and apply to virtually all text.

## Evidence

### Run 20251117_141036 (base_layers=[1])
- Loaded 270 lenses from layer 1
- All detected concepts have probability 1.0
- Top concepts: BinaryRelation, BiologicalProcess, BoardOrBlock, CausingPain, etc.
- These are mid-level abstractions from layer 1

### Run 20251117_141930 (base_layers=[0, 1, 2])
- Loaded 1357 lenses from layers 0, 1, and 2
- All detected concepts have probability 1.0
- Top concepts: SetOrClass, PhysicalSystem, Proposition, Entity, Object, Attribute, List
- These are even MORE abstract - foundational concepts from layer 0
- Only 10 concepts scored per activation (matching top_k parameter)

## Why Layer 0 Caused the Problem

Layer 0 in SUMO contains the most abstract, foundational concepts (Entity, Object, SetOrClass, Attribute, Relation, Proposition, etc). These concepts are:

1. **Always applicable** - Almost any text will contain entities, objects, and attributes
2. **Maximally abstract** - Provide no specific semantic information that distinguishes between different prompts
3. **Universally high confidence** - The lenses correctly detect that these abstract concepts apply to nearly everything

When we configured `base_layers=[0, 1, 2]`:
- We loaded 1357 lenses instead of 270
- Layer 0 concepts (Entity, Object, SetOrClass, etc.) were scored alongside layer 1 and 2 concepts
- Since they apply to virtually all text, they scored high confidence (approaching 1.0)
- They dominated the top-K results because there's no parent-hiding logic in `detect_and_expand()`

## Missing Feature: Parent Concept Hiding

The `detect_and_expand()` method:
1. Scores all currently loaded lenses
2. Dynamically loads children of high-confidence parents
3. Scores the newly loaded children
4. Returns ALL scored concepts sorted by probability

**What it does NOT do:** Hide abstract parent concepts when more specific child concepts are detected with high confidence.

This means layer 0 concepts like "Entity" and "Object" appear in results alongside their more specific descendants, drowning out meaningful semantic differences.

## Implications for the Experiment

The current lens pack **cannot be used** to analyze concept clustering across behavioral vs definitional prompts because:

1. All concepts score 1.0, so we can't measure differential activation
2. The top concepts are generic abstractions that don't capture semantic differences
3. There's no signal to train classifiers on

## Potential Solutions

### Option 1: Use gemma-3-4b-pt_sumo-wordnet-v1 lens pack
- Includes AI safety concepts
- May have better calibration (needs testing)
- Has text classifiers in addition to activation lenses

### Option 2: Retrain lenses with better calibration
- Current lenses may have been trained without proper threshold calibration
- Need temperature scaling or proper probability calibration
- This is a fundamental lens pack quality issue

### Option 3: Use raw activation patterns instead of lenses
- Skip concept detection entirely
- Train classifiers directly on the activation vectors
- This is what we're already doing successfully with the main experiment
- The HatCat concept layer was meant to provide interpretability, not core functionality

### Option 4: Filter out saturated lenses
- Detect which lenses always output 1.0
- Exclude them from scoring
- Only use lenses that show variance

## Solution

**Use mid-level layers only for base concepts.** Changed configuration from:
```python
base_layers=[0, 1, 2]  # BAD - includes overly abstract layer 0
```

To:
```python
base_layers=[2, 3]  # GOOD - mid-level concepts that differentiate meaningfully
```

This provides:
- **Specificity**: Layer 2-3 concepts are specific enough to distinguish between different scenarios
- **Coverage**: Dynamic expansion will still load more specific children as needed
- **No abstraction pollution**: Avoids layer 0 concepts that apply universally

## Alternative Solutions Considered

1. **Implement parent-hiding logic** in `detect_and_expand()`:
   - When a child concept scores > threshold, remove its parents from results
   - Complex to implement correctly (requires traversing hierarchy)
   - May still not solve the issue if parent and child both score ~1.0

2. **Use threshold filtering** (like test_temporal_monitoring.py does):
   - Only keep concepts with prob > threshold
   - Doesn't solve the abstraction problem - layer 0 concepts would still pass threshold

3. **Use only layer 3+ concepts**:
   - Too specific - may miss broader patterns
   - Dynamic expansion from layers 2-3 provides better balance

## Impact on Experiment

The **core experimental results are unaffected** because:
- We train classifiers directly on activation vectors, not concept labels
- Activation similarities and cross-test results remain valid
- HatCat concept detection is for interpretability, not the main analysis

However, for **concept clustering analysis**, using base_layers=[2, 3] should provide meaningful semantic differentiation between behavioral vs definitional prompts.
