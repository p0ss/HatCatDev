# Jacobian vs Classifier Alignment Analysis

## Summary

Tested alignment between Jacobian-based concept vectors and trained MLP classifiers for 5 Layer 0 concepts.

**Result: Near-zero alignment (mean=-0.019, std=0.016)**

## Hypothesis

- **Jacobian**: Local sensitivity direction—how model output changes with respect to input in the context of a specific prompt ("The concept of X means")
- **Classifier**: Learned separating hyperplane—trained on contrastive data (positive examples, hard negatives, random negatives)

## Results

| Concept | Alignment | Jacobian Time |
|---------|-----------|---------------|
| Abstract | 0.0052 | 1.60s |
| Process | -0.0062 | 1.57s |
| Attribute | -0.0229 | 1.60s |
| Entity | -0.0318 | 1.65s |
| Physical | -0.0377 | 1.84s |

**Statistics:**
- Mean: -0.0187
- Median: -0.0229
- Std: 0.0160
- Range: [-0.0377, 0.0052]

**All alignments fall in [0.0-0.3) bin** — essentially orthogonal directions.

## Interpretation

### What This Means

1. **Jacobian ≠ Classifier Direction**
   - The two vectors are nearly orthogonal (cosine similarity ~0)
   - They capture fundamentally different aspects of the concept

2. **Why This Makes Sense**

   **Jacobian captures:**
   - Local gradient of model output w.r.t. activations
   - Specific to prompt: "The concept of {Physical} means"
   - Context-dependent, single-example sensitivity
   - Token position dependent (last token)

   **Classifier captures:**
   - Separating hyperplane from contrastive training
   - Learned from many examples (positive + hard negatives + random)
   - Invariant across contexts and token positions
   - Optimized for discrimination, not generation

3. **Geometric Interpretation**
   - Jacobian: "Which way to nudge activations to complete this prompt"
   - Classifier: "Which way distinguishes Physical from non-Physical across all contexts"

   These are different questions, so orthogonal answers are not surprising.

### Implications for Your Original hypothesis

You suggested based on Golden's Linear LLM paper:
> "maybe we can consider jacobian is basically the true centroid, but not the broader conceptual boundary"
> "classifier is... the broader conceptual landscape (contextual, relational)"

**The data suggests:** Jacobian and classifier are capturing **different geometric objects**, not nested ones (center vs boundary).

- Jacobian: **Task-specific sensitivity** (generation completion)
- Classifier: **Task-agnostic discrimination** (concept detection)

## Should We Use Jacobians?

### ❌ Not as "Ground Truth" or "Center"

The near-zero alignment shows Jacobians don't represent the same concept structure that classifiers learn. They're solving different problems:
- Jacobian: "How to generate concept-related text"
- Classifier: "How to detect concept presence"

### ✅ Potential Uses

1. **Multi-objective validation**
   - Low Jacobian alignment might indicate classifier is learning spurious features
   - Could be a sanity check: "Is this at least somewhat related to generation?"

2. **Complementary steering**
   - Jacobian steering: Push toward generative context
   - Classifier steering: Push toward discriminative boundary
   - Might be useful for different use cases

3. **Concept quality metric**
   - Concepts with higher J-C alignment might be "cleaner" (linear, unambiguous)
   - Low alignment might indicate multi-dimensional or context-dependent concepts

### 🤔 The Real Question

**What's the "right" concept vector for steering?**

Your classifiers are trained to **discriminate**, which seems aligned with steering goals:
- Enhance Physical-ness → move toward Physical classifier's positive region
- Suppress Physical-ness → move away from it

Jacobian is trained for **generation**, which may not transfer to steering contexts.

## Next Steps

### Option 1: Trust the Classifier
Your contrastive training with hard negatives and relational context is likely more aligned with steering objectives. The ~0 alignment with Jacobians doesn't invalidate this—they're just measuring different things.

### Option 2: Test Steering Quality Directly
Rather than validating against Jacobians, validate against **steering outcomes**:
- Apply classifier-based steering
- Measure output quality (does it exhibit concept?)
- Iterate on training data if steering fails

### Option 3: Explore Why Alignment is Zero
- Try different prompts for Jacobian ("This is Physical", "Physical object", etc.)
- Average Jacobians over multiple contexts
- Compare Jacobian at different layers
- Check if higher-dimensional Jacobian subspace (multiple SVD components) aligns better

## Technical Notes

- **Classifier Architecture**: 3-layer MLP (2560 → 128 → 64 → 1)
- **Alignment Method**: SVD of first layer to get principal input direction
- **Jacobian Extraction**: Layer 6, prompt "The concept of {X} means", BF16 model
- **Timing**: ~1.6s per Jacobian (fast!)

## Files Generated

- `results/jacobian_alignment_test.json` — Full test results with metadata
- `scripts/test_jacobian_vs_classifier.py` — Reusable alignment test script
