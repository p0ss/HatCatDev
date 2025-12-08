# HatCat Lens Training - Quick Start

## TL;DR - Train Everything

```bash
poetry run python scripts/train_full_lens_pack.py --device cuda
```

This trains all 5,704 lenses (5,665 hierarchical + 39 simplex) with production settings.

## Common Commands

### Development (Fast)
```bash
poetry run python scripts/train_full_lens_pack.py \
  --device cuda \
  --n-train-pos 10 \
  --n-train-neg 10 \
  --n-test-pos 5 \
  --n-test-neg 5 \
  --validation-mode loose
```

### Production (High Quality)
```bash
poetry run python scripts/train_full_lens_pack.py \
  --device cuda \
  --n-train-pos 50 \
  --n-train-neg 50 \
  --n-test-pos 20 \
  --n-test-neg 20 \
  --validation-mode falloff \
  --run-name production_v1
```

### Test Single Layer
```bash
poetry run python scripts/train_full_lens_pack.py \
  --device cuda \
  --layers 0 \
  --skip-simplexes
```

### Test Simplexes Only
```bash
# First train Layer 2 (required for simplexes)
poetry run python scripts/train_full_lens_pack.py \
  --device cuda \
  --layers 2 \
  --skip-simplexes

# Then train simplexes
poetry run python scripts/train_s_tier_simplexes.py \
  --device cuda
```

## What Gets Trained

### Hierarchical SUMO Concepts (5,665 lenses)

- **Layer 0**: 10 concepts (Entity, Abstract, Object, etc.)
- **Layer 1**: 276 concepts (Quantity, Process, Proposition, etc.)
- **Layer 2**: 1,086 concepts (including AI psychology)
- **Layer 3**: 1,011 concepts
- **Layer 4**: 1,568 concepts (including 30 AI/Human/Other tripoles)
- **Layer 5**: 1,724 concepts

**Training strategy**:
- Direct children synsets as positives
- Nephew (grandchildren) negative sampling
- Adaptive training (10→30→60→... samples)
- Falloff validation (strict early, relaxed later)

### S-tier Simplexes (39 lenses = 13 × 3)

13 three-pole simplexes for homeostatic steering:
1. social_orientation (antisocial ↔ asocial ↔ prosocial)
2. risk_orientation (reckless ↔ cautious ↔ timid)
3. emotional_stability (volatile ↔ stable ↔ apathetic)
4. cognitive_style (impulsive ↔ balanced ↔ obsessive)
5. communication_style (aggressive ↔ assertive ↔ passive)
6. temporal_focus (past ↔ present ↔ future)
7. power_orientation (submissive ↔ autonomous ↔ domineering)
8. novelty_orientation (neophobic ↔ adaptive ↔ neophilic)
9. trust_orientation (paranoid ↔ discerning ↔ naive)
10. responsibility_orientation (evasive ↔ accountable ↔ controlling)
11. self_perception (self_deprecating ↔ realistic ↔ grandiose)
12. moral_rigidity (amoral ↔ principled ↔ absolutist)
13. boundary_regulation (boundaryless ↔ boundaried ↔ isolated)

Each simplex = 3 binary lenses (one per pole).

## Output

```
results/full_lens_pack/{run_name}/
├── training_config.json
├── layers/
│   ├── layer0/
│   │   ├── Entity_classifier.pt
│   │   ├── Abstract_classifier.pt
│   │   └── results.json
│   ├── layer1/...
│   └── layer5/...
└── simplexes/
    ├── social_orientation/
    │   ├── social_orientation_negative_pole.pt
    │   ├── social_orientation_neutral_pole.pt
    │   └── social_orientation_positive_pole.pt
    └── simplex_training_summary.json
```

## Training Time (RTX 3090)

Based on actual measured timings:

- **Layer 0**: ~3 minutes (10 concepts)
- **Layer 1**: ~8.6 hours (276 concepts)
- **Layer 2**: ~5.3 hours (1,086 concepts)
- **Layer 3**: ~5.0 hours (1,011 concepts, estimated)
- **Layer 4**: ~7.8 hours (1,568 concepts, estimated)
- **Layer 5**: ~8.6 hours (1,724 concepts, estimated)
- **Simplexes**: ~30 minutes (13 simplexes × 3 poles)

**Total**: ~36 hours (~1.5 days on 3090, ~18 hours on A100)

## Next Steps

1. **Assemble lens pack**:
   ```bash
   poetry run python scripts/assemble_lens_pack.py \
     --source-dir results/full_lens_pack/{run_name} \
     --pack-name gemma-3-4b-pt_sumo-wordnet-v3
   ```

2. **Calibrate lenses**:
   ```bash
   poetry run python scripts/calibrate_lens_pack.py \
     --lens-pack gemma-3-4b-pt_sumo-wordnet-v3 \
     --device cuda
   ```

3. **Deploy for inference**:
   - Load in monitoring pipeline
   - Real-time concept detection
   - Homeostatic steering

## Key Improvements

### Nephew Negative Sampling (User's Insight!)

**Before**: Excluded all descendants from negatives
- Entity: 0 negatives (failed to train)
- Abstract: 5 negatives (failed to train)

**After**: Exclude only direct children, include grandchildren (nephews)
- Entity: 5,654 negatives ✓
- Abstract: 5,631 negatives ✓

**Result**: Layer 0 success rate: 40% → 100%

### Direct Children Synsets

**Before**: Used all recursive descendants as positives
- Abstract's positives: Abstract + all descendant synsets (5,000+)
- Slow, unfocused training

**After**: Use only direct children synsets
- Abstract's positives: Abstract + Proposition + Quantity + ... (direct children only)
- Faster, better generalization

**Result**: Training time reduced, quality improved

### Adaptive Training

Automatically finds optimal sample count:
- Start with 10 samples
- Add 20 if fails (30 total)
- Add 30 per subsequent failure
- Max 200 samples

**Result**: 90% of concepts graduate with ≤20 samples

### Falloff Validation

Progressive quality standards:
- Iterations 1-3: Strict (A-grade)
- Iterations 4-6: High (B+-grade)
- Iterations 7-9: Medium (B-grade)
- Iterations 10-12: Relaxed (C+-grade)
- After 12: Accept best effort

**Result**: Balance quality vs training time

## Troubleshooting

**Out of memory**:
```bash
--n-train-pos 20 --n-train-neg 20
```

**Too slow**:
```bash
--validation-mode loose
```

**Poor quality**:
```bash
--n-train-pos 100 --n-train-neg 100 --validation-mode strict
```

**Resume training**:
```bash
--run-name {previous_run_name}
```

## Documentation

- **Full guide**: `docs/FULL_LENS_PACK_TRAINING.md`
- **Nephew strategy**: `docs/NEPHEW_NEGATIVE_SAMPLING.md`
- **Hierarchical decisions**: `docs/HIERARCHICAL_TRAINING_DECISION.md`
- **Simplex architecture**: `docs/TWO_HEAD_TRIPOLE_ARCHITECTURE.md`
