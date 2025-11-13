# Detached Jacobian Approach for Concept Extraction

## Overview

The detached Jacobian approach from ["LLMs are Locally Linear"](https://openreview.net/forum?id=oDWbJsIuEp) provides a theoretically grounded method for extracting low-dimensional concept vectors from LLMs.

## Key Theoretical Claims

1. **Low-Dimensional Subspaces**: LLMs operate in "extremely low-dimensional subspaces" despite their high parameter count
2. **Interpretable Singular Vectors**: Singular vectors of the Jacobian matrix decode into interpretable semantic concepts
3. **Perfect Reconstruction**: Achieves reconstruction with relative error below 10^-13
4. **No Training Required**: Works with frozen models - no additional training needed

## Method

### Jacobian Computation

For an input text with embeddings `x`, compute the Jacobian matrix:

```
J[i,j,k] = ∂output_k / ∂input_embedding[i,j]
```

Where:
- `i` = token index
- `j` = input embedding dimension
- `k` = output embedding dimension

### Concept Extraction

The weighted sum of Jacobians reveals the effective transformation:

```
concept_direction = sum_i(J_i @ embed_i)
```

The top singular vectors of this transformation represent the low-dimensional semantic subspace.

## Comparison with CAV Approach

### Current CAV Approach
- **Method**: Contrastive activation vectors from paired examples
- **Speed**: Fast (~0.7s for extraction)
- **Dimensionality**: Full dimensional space (2560D)
- **Training**: Requires paired concept/non-concept examples

### Detached Jacobian Approach
- **Method**: Compute Jacobian via autograd, extract singular vectors
- **Speed**: Slower (~10s for Jacobian computation)
- **Dimensionality**: Reveals true low-dimensional structure (often < 10D)
- **Training**: No training - single forward pass with gradient

## Advantages for Manifold Steering

1. **Theoretically Grounded**: Proven to capture the true low-dimensional manifold
2. **Precise**: 10^-13 reconstruction error vs. approximate CAV directions
3. **No Contrastive Examples Needed**: Single concept prompt sufficient
4. **True Manifold Structure**: Reveals actual dimensionality of concept space

## Implementation Status

### Current Status
- **Basic implementation**: `src/steering/detached_jacobian.py` ✓
- **Architecture compatibility**: Partial (needs Gemma-3 specific handling)
- **Full forward pass**: Requires careful handling of attention masks and model structure

### Blockers
1. Gem ma-3 architecture differences (e.g., `_update_causal_mask` location)
2. Memory requirements for full Jacobian computation
3. Computational cost (~10-20s per concept)

### Future Work

1. **Complete Gemma-3 Integration**
   - Properly handle `model.language_model` vs `model.model`
   - Find `_update_causal_mask` equivalent or reimplement
   - Handle position embeddings (global vs local)

2. **Optimize Computation**
   - Use chunked Jacobian computation to reduce memory
   - Implement reverse-mode autodiff efficiently
   - Cache Jacobians for reuse

3. **Benchmark Against CAV**
   - Steering quality comparison
   - Manifold dimensionality analysis
   - Runtime vs accuracy tradeoff

4. **Integration with Manifold Steering**
   - Use Jacobian singular vectors for manifold estimation
   - Replace PCA-based manifold with Jacobian-based subspace
   - Compare steering precision

## References

- Paper: https://openreview.net/forum?id=oDWbJsIuEp
- Code: https://github.com/jamesgolden1/equivalent-linear-LLMs
- Notebook: https://github.com/jamesgolden1/equivalent-linear-LLMs/blob/main/notebooks/gemma_3/detached_jacobian_gemma_demo.ipynb

## Recommendation

**Short term**: Continue using CAV approach - it's fast, effective, and well-integrated

**Medium term**: Complete Jacobian implementation as research tool to:
- Validate that CAV directions align with true manifold structure
- Measure effective dimensionality of our concepts
- Benchmark steering precision improvements

**Long term**: If Jacobian shows significant improvement, consider hybrid approach:
- Use Jacobian for high-precision concept extraction
- Use CAV for rapid prototyping
- Offer both methods with quality vs speed tradeoff
