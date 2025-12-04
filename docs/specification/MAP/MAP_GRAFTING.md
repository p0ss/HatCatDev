# Graft Protocol

> **Structured integration of learned concepts into BE substrates**
>
> This protocol defines how grafts (concept dimensions + weight biases) are derived
> from probe analysis, trained from Experience Database exemplars, merged into
> substrate parameters, and shared across BE instances via MAP.

**Status**: Draft specification
**Extends**: MAP_MELD_PROTOCOL.md, BE_EXPERIENCE_DATABASE.md, BE_CONTINUAL_LEARNING.md
**Related**: HATCAT_MELD_POLICY.md, MINDMELD_ARCHITECTURAL_PROTOCOL.md

---

## 1. Overview

### 1.1 Core Principle

Grafts **grow the substrate**. When a BE learns a new concept:

1. A **new dimension** is added to the substrate — a labelled feature for the concept
2. **Biases** are added to existing weights — encoding how the concept relates to everything else
3. A **probe** is trained that reads from the new dimension (plus auxiliary features)
4. The graft is validated and **permanently merged**
5. The substrate is now larger by one dimension

The Experience Database is the **source of truth**. Grafts are ephemeral derivatives that can be regenerated from exemplars. When concepts overlap, we cotrain from pooled data rather than naively merging grafts.

### 1.2 Substrate Growth Model

A small trunk can scale to understand everything through successive grafts:

```
Trunk (N dimensions)
  + Graft(concept_A) → Trunk (N+1 dimensions)
  + Graft(concept_B) → Trunk (N+2 dimensions)
  + Graft(concept_C) → Trunk (N+3 dimensions)
  ...
```

Each graft adds:
- **One labelled dimension** — the concept's identity in parameter space
- **Biases to existing weights** — scaled by how much training changed them
- **Relational structure** — implicit in the bias pattern (no separate edge weights needed)

This means:
- The probe has a **direct feature** to read — cheap, reliable detection
- The bias pattern **is** the concept's meaning — its relationship to everything else
- New concepts **build on old ones** — the substrate accumulates understanding
- Trunk size limits **dimension count**, not parameter count

Note: For very large, heavily pre-trained trunks, Graft SHOULD be preceded by a distillation / alignment step to a smaller “concept kernel” substrate. Directly grafting onto an un-instrumented 400B trunk risks baking in correlational shortcuts as if they were concepts.

### 1.3 Relationship to Existing Protocols

| Protocol | Role in Graft |
|----------|---------------|
| **MAP** | Concept identity, probe registration, GraftDiff distribution |
| **MELD** | Governance, protection levels, review processes |
| **XDB** | Exemplar storage, training provenance, dataset management |
| **Continual Learning Harness** | Gap detection, candidate concept lifecycle |
| **Hush** | Constraints on what regions may be modified |

Graft adds:

- **Dimension allocation** — adding labelled features for concepts
- **Bias derivation** — encoding relational structure in weight deltas
- **Probe binding** — linking probes to their primary dimension
- **Cotrain triggers** — when concepts share bias patterns
- **Substrate growth management** — compaction, merging, graduation

---

## 2. Conceptual Model

### 2.1 Concepts as Dimensions

Each learned concept becomes a **dimension** in the substrate's representation space:

```
substrate.hidden_dim = base_dim + num_grafted_concepts
```

The dimension is **labelled** — we know exactly which feature corresponds to which concept. This is unlike emergent features in a pretrained model, which are unlabelled and entangled.

### 2.2 Biases as Relational Evidence

When training a graft, the learning process modifies existing weights. These modifications are preserved as **biases**:

```
bias_pattern[concept_X] = {
  layer_18_mlp: sparse_delta,
  layer_20_attn: sparse_delta,
  ...
}
```

The bias pattern encodes:
- **Which existing features** the concept relates to
- **How strongly** (magnitude of change)
- **In what direction** (sign of change)

This is the concept's "meaning" in terms of its effect on the substrate. No separate relational weights needed — the evidence is in the parameters.

### 2.3 Probes with Direct Features

A probe for concept X reads:
- **Primary**: the labelled dimension for X (direct, reliable)
- **Auxiliary**: other dimensions identified by region analysis (context)

```
score_X = σ(w_primary · h[dim_X] + w_aux · h[aux_dims] + b)
```

Because the primary dimension is labelled, probe training is easier and more robust. The auxiliary dimensions capture context and nuance.

### 2.4 Symmetry Principle

```
Graft:  "Add dimension for concept X, bias existing weights to wire it in"
Probe:  "Read from dimension X (plus context) to detect activation"
```

The graft writes the concept into the substrate. The probe reads it back out. They share the same primary dimension.

---

## 3. Data Model

### 3.1 Graft

A **Graft** is the complete package for adding a concept to a substrate.

```jsonc
Graft = {
  "graft_id": "graft-Fish-v1",
  "concept_id": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Fish",
  "concept_version": "4.0.0",

  // === THE NEW DIMENSION ===
  "concept_dimension": {
    "dimension_index": 2049,  // position in expanded substrate
    "dimension_label": "concept/Fish",  // human-readable label
    "initial_weight": 1.0,  // initialization value

    // Where this dimension is injected in the architecture
    "injection_points": [
      {
        "layer": 18,
        "component": "residual",  // added to residual stream
        "projection": "blob://grafts/Fish-v1/layer18_proj.safetensors"
      }
    ]
  },

  // === BIASES TO EXISTING WEIGHTS ===
  "substrate_biases": [
    {
      "layer": 18,
      "component": "mlp.up_proj",
      "bias_delta": {
        "format": "sparse_coo",  // only store nonzero values
        "location": "blob://grafts/Fish-v1/layer18_mlp_bias.safetensors",
        "nnz": 1247,  // number of nonzero entries
        "shape": [2048, 8192]
      },
      "magnitude_stats": {
        "mean": 0.0023,
        "max": 0.089,
        "l2_norm": 0.34
      }
    },
    {
      "layer": 20,
      "component": "attn.v_proj",
      "bias_delta": {
        "format": "sparse_coo",
        "location": "blob://grafts/Fish-v1/layer20_attn_bias.safetensors",
        "nnz": 892,
        "shape": [2048, 2048]
      },
      "magnitude_stats": {
        "mean": 0.0018,
        "max": 0.056,
        "l2_norm": 0.21
      }
    }
  ],

  // === PROBE BINDING ===
  "probe_binding": {
    "probe_id": "org.hatcat/...::probe/Fish",
    "primary_dimension": 2049,  // the new labelled dimension
    "auxiliary_dimensions": [42, 87, 156, 203],  // from region analysis
    "probe_weights_location": "blob://grafts/Fish-v1/probe.safetensors"
  },

  // === RELATIONAL EVIDENCE (derived from bias pattern) ===
  "relational_fingerprint": {
    "method": "bias_correlation",
    "top_correlations": [
      {"concept_id": "concept/Vertebrate", "correlation": 0.82},
      {"concept_id": "concept/Animal", "correlation": 0.71},
      {"concept_id": "concept/Aquatic", "correlation": 0.68}
    ],
    "fingerprint_hash": "sha256:..."  // for fast comparison
  },

  // === PROVENANCE ===
  "training_run_id": "trainrun-Fish-v1",
  "source_region_id": "region-Fish-v1",  // region analysis that guided training
  "dataset_ids": ["dataset-Fish-v1"],

  // === SUBSTRATE COMPATIBILITY ===
  "applies_to": {
    "substrate_id": "olmo3-7b-base@0.1.0",
    "substrate_checksum": "sha256:...",
    "pre_graft_dim": 2048,  // substrate hidden_dim before this graft
    "post_graft_dim": 2049  // substrate hidden_dim after this graft
  },

  // === LIFECYCLE ===
  "supersedes": [],  // previous graft versions this replaces
  "superseded_by": null,

  "validation": {
    "status": "validated",
    "validation_run_id": "valrun-Fish-v1"
  },

  "created_at": "2025-11-30T12:45:00Z",
  "created_by": "BE-instance-123"
}
```

### 3.2 ConceptRegion

A **ConceptRegion** captures where a concept's activity is detected in the *original* substrate dimensions. It guides training but doesn't define the final structure.

```jsonc
ConceptRegion = {
  "region_id": "region-Fish-v1",
  "concept_id": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Fish",

  // Derived from probe analysis on pre-graft substrate
  "source_probes": [
    {
      "probe_id": "org.hatcat/...::probe/Fish",
      "probe_version": "2.20251130.0"
    }
  ],

  "derivation": {
    "method": "probe_weight_topk",
    "parameters": {
      "top_k_percent": 15,
      "layers": [18, 20, 22]
    }
  },

  // Dimensions in the ORIGINAL substrate that correlate with this concept
  // These become the auxiliary dimensions in the probe, and guide where biases land
  "layers": [
    {
      "layer_index": 18,
      "component": "mlp",
      "dimension_mask": {
        "format": "sparse_indices",
        "indices": [42, 87, 156, 203],
        "total_dimensions": 2048  // pre-graft dimension count
      }
    }
  ],

  "created_at": "2025-11-30T12:00:00Z"
}
```

### 3.3 Extended TrainingRun

```jsonc
TrainingRun = {
  "id": "trainrun-Fish-v1",
  "type": "graft",  // "graft" | "probe_only" | "cotrain"

  "concept_ids": ["org.hatcat/...::concept/Fish"],

  // Substrate state at training time
  "substrate_id": "olmo3-7b-base@0.1.0",
  "substrate_checksum": "sha256:...",
  "substrate_dim_at_training": 2048,  // pre-graft

  // Region that guided training
  "source_region_id": "region-Fish-v1",

  // Dataset provenance
  "dataset_ids": ["dataset-Fish-v1"],
  "dataset_checksums": {"dataset-Fish-v1": "sha256:..."},

  // Training configuration
  "hyperparams": {
    "learning_rate": 5e-5,
    "epochs": 3,
    "batch_size": 32,
    "dimension_init": "learned",  // "learned" | "zero" | "random"
    "bias_sparsity_target": 0.95,  // encourage sparse biases
    "bias_magnitude_penalty": 0.01  // regularize bias magnitudes
  },

  // Results
  "metrics": {
    "train_loss": 0.12,
    "val_loss": 0.15,
    "concept_f1": 0.89,
    "bias_sparsity_achieved": 0.94,
    "dimension_usage": 0.73,  // how much the new dimension contributes to probe
    "auxiliary_contribution": 0.27  // how much auxiliary dims contribute
  },

  // Cotrain context (if type == "cotrain")
  "cotrain_context": null,

  "status": "succeeded",
  "created_at": "2025-11-30T12:30:00Z",
  "created_by": "BE-instance-123"
}
```

### 3.4 SubstrateManifest

Tracks the current state of a substrate including all grafted dimensions.

```jsonc
SubstrateManifest = {
  "manifest_id": "substrate-olmo3-7b-grafted-v42",

  "base_substrate": {
    "substrate_id": "olmo3-7b-base@0.1.0",
    "base_checksum": "sha256:...",
    "base_hidden_dim": 2048
  },

  "current_state": {
    "checksum": "sha256:...",
    "hidden_dim": 2091,  // base + 43 grafted concepts
    "total_grafts_applied": 43
  },

  // Dimension allocation table
  "dimension_table": [
    {
      "dimension_index": 2048,
      "concept_id": "concept/Eligibility",
      "graft_id": "graft-Eligibility-v1",
      "grafted_at": "2025-11-01T00:00:00Z"
    },
    {
      "dimension_index": 2049,
      "concept_id": "concept/Fish",
      "graft_id": "graft-Fish-v1",
      "grafted_at": "2025-11-30T13:00:00Z"
    }
    // ...
  ],

  // For dimension management
  "compaction_candidates": [
    {
      "dimensions": [2055, 2067],
      "reason": "high_correlation",
      "correlation": 0.94
    }
  ],

  "updated_at": "2025-11-30T13:00:00Z"
}
```

### 3.5 OverlapAnalysis (updated for bias fingerprints)

```jsonc
OverlapAnalysis = {
  "id": "overlap-Fish-Salmon-2025-11-30",

  "concepts": [
    {"concept_id": "concept/Fish", "graft_id": "graft-Fish-v1"},
    {"concept_id": "concept/Salmon", "graft_id": "graft-Salmon-v1"}
  ],

  // Bias pattern similarity
  "fingerprint_similarity": {
    "method": "bias_correlation",
    "correlation": 0.78,
    "shared_bias_dimensions": 423,
    "total_union_dimensions": 892
  },

  // Region overlap (in original substrate dimensions)
  "region_overlap": {
    "jaccard_index": 0.34,
    "overlapping_auxiliary_dims": [42, 87, 156]
  },

  "recommendation": {
    "action": "cotrain",  // "cotrain" | "independent" | "merge_dimensions"
    "reason": "Bias fingerprint correlation 0.78 exceeds threshold 0.6",
    "threshold_used": 0.6
  },

  "created_at": "2025-11-30T12:15:00Z"
}
```

---

## 4. Region Derivation

Region derivation identifies which *existing* substrate dimensions correlate with a concept. This guides:
- Where biases should land during training
- Which auxiliary dimensions the probe should read

### 4.1 From Probe Weights

```python
def derive_region_from_probe(
    probe: TrainedProbe,
    concept_id: str,
    layers: List[int],
    top_k_percent: float = 15.0,
    include_ancestors: bool = True,
    ancestor_weight_decay: float = 0.5
) -> ConceptRegion:
    """
    Derive a ConceptRegion from probe weights.

    For each layer, takes the top k% of dimensions by |weight|.
    These become auxiliary dimensions for the graft's probe.
    """

    region_layers = []

    for layer_idx in layers:
        weights = probe.get_layer_weights(layer_idx)
        importance = np.abs(weights)

        if include_ancestors:
            for ancestor, depth in get_concept_ancestors(concept_id):
                ancestor_probe = load_probe(ancestor)
                ancestor_weights = ancestor_probe.get_layer_weights(layer_idx)
                weight = ancestor_weight_decay ** depth
                importance = importance + weight * np.abs(ancestor_weights)

        k = int(len(importance) * top_k_percent / 100)
        top_indices = np.argsort(importance)[-k:]

        region_layers.append({
            "layer_index": layer_idx,
            "component": "mlp",
            "dimension_mask": {
                "format": "sparse_indices",
                "indices": top_indices.tolist(),
                "total_dimensions": len(importance)
            }
        })

    return ConceptRegion(
        concept_id=concept_id,
        layers=region_layers,
        derivation={"method": "probe_weight_topk", "parameters": {...}}
    )
```

### 4.2 From Gradient Attribution

Alternative using gradient-based importance on concept-positive examples.

```python
def derive_region_from_gradients(
    substrate: Model,
    dataset: ConceptDataset,
    layers: List[int],
    top_k_percent: float = 15.0
) -> ConceptRegion:
    """
    Derive region by measuring gradient magnitude on concept-positive examples.
    """

    importance_accum = defaultdict(lambda: np.zeros(substrate.hidden_dim))

    for exemplar in dataset.positive_exemplars:
        activations = substrate.forward_with_cache(exemplar.text)
        gradients = substrate.backward_to_layers(layers)

        for layer_idx in layers:
            importance_accum[layer_idx] += np.abs(gradients[layer_idx])

    # Normalize and select top k%
    region_layers = []
    for layer_idx in layers:
        importance = importance_accum[layer_idx] / len(dataset.positive_exemplars)
        k = int(len(importance) * top_k_percent / 100)
        top_indices = np.argsort(importance)[-k:]

        region_layers.append({
            "layer_index": layer_idx,
            "dimension_mask": {
                "format": "sparse_indices",
                "indices": top_indices.tolist()
            }
        })

    return ConceptRegion(layers=region_layers, derivation={"method": "gradient_attribution"})
```

---

## 5. Graft Training

### 5.1 Training Procedure

Graft training learns:
1. The **concept dimension** — a new feature that activates for the concept
2. The **substrate biases** — modifications to existing weights that wire the concept in

```python
def train_graft(
    substrate: Model,
    dataset: ConceptDataset,
    region: ConceptRegion,
    config: GraftConfig
) -> Graft:
    """
    Train a graft that adds a concept dimension and biases existing weights.
    """

    # 1. Allocate new dimension
    new_dim_index = substrate.hidden_dim
    substrate.expand_hidden_dim(1)  # Now hidden_dim + 1

    # 2. Initialize projection matrices for the new dimension
    projections = initialize_dimension_projections(
        substrate,
        new_dim_index,
        injection_layers=config.injection_layers
    )

    # 3. Initialize bias accumulators (sparse)
    bias_accum = initialize_sparse_bias_accumulators(
        substrate,
        region  # Only accumulate biases in region-relevant layers
    )

    # 4. Training loop
    optimizer = torch.optim.AdamW(
        list(projections.parameters()) + list(bias_accum.parameters()),
        lr=config.learning_rate
    )

    for epoch in range(config.epochs):
        for batch in dataset.batches(config.batch_size):
            optimizer.zero_grad()

            # Forward with new dimension and biases applied
            with apply_graft_draft(substrate, projections, bias_accum):
                outputs = substrate(batch.inputs)

                # Loss: concept should activate on positive examples
                concept_activations = outputs.hidden_states[:, :, new_dim_index]
                loss = compute_concept_loss(concept_activations, batch.labels)

                # Regularization: encourage sparse biases
                loss += config.bias_sparsity_penalty * bias_accum.l1_norm()
                loss += config.bias_magnitude_penalty * bias_accum.l2_norm()

            loss.backward()
            optimizer.step()

            # Enforce sparsity via thresholding
            bias_accum.threshold_small_values(config.sparsity_threshold)

    # 5. Train probe to read from new dimension + auxiliary
    probe = train_probe_with_primary_dimension(
        substrate=substrate,
        projections=projections,
        bias_accum=bias_accum,
        dataset=dataset,
        primary_dim=new_dim_index,
        auxiliary_dims=region.get_all_indices()
    )

    # 6. Package as Graft
    return Graft(
        concept_dimension={
            "dimension_index": new_dim_index,
            "injection_points": projections.to_spec()
        },
        substrate_biases=bias_accum.to_sparse_deltas(),
        probe_binding={
            "probe_id": probe.id,
            "primary_dimension": new_dim_index,
            "auxiliary_dimensions": region.get_all_indices()
        },
        relational_fingerprint=compute_fingerprint(bias_accum)
    )
```

### 5.2 Probe Training with Primary Dimension

The probe is trained to read primarily from the new labelled dimension.

```python
def train_probe_with_primary_dimension(
    substrate: Model,
    projections: DimensionProjections,
    bias_accum: SparseBiasAccumulator,
    dataset: ConceptDataset,
    primary_dim: int,
    auxiliary_dims: List[int],
    config: ProbeConfig
) -> Probe:
    """
    Train a probe that reads from the primary (labelled) dimension
    plus auxiliary dimensions from region analysis.
    """

    # Probe architecture: primary weight + auxiliary weights
    probe = Probe(
        primary_dim=primary_dim,
        auxiliary_dims=auxiliary_dims,
        hidden_dim=config.probe_hidden_dim  # optional MLP layer
    )

    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        for batch in dataset.batches(config.batch_size):
            optimizer.zero_grad()

            # Get activations with graft applied
            with apply_graft_draft(substrate, projections, bias_accum):
                hidden_states = substrate.get_hidden_states(batch.inputs)

            # Probe prediction
            # score = w_primary * h[primary_dim] + w_aux · h[aux_dims] + b
            scores = probe(hidden_states)
            loss = F.binary_cross_entropy_with_logits(scores, batch.labels)

            loss.backward()
            optimizer.step()

    return probe
```

### 5.3 Relational Fingerprint Computation

The bias pattern encodes how the concept relates to existing substrate features.

```python
def compute_fingerprint(bias_accum: SparseBiasAccumulator) -> RelationalFingerprint:
    """
    Compute a fingerprint from the bias pattern for fast comparison.

    Two grafts with similar fingerprints learned similar relational structure,
    even if they're for different concepts.
    """

    # Flatten all biases into a single sparse vector
    flat_biases = bias_accum.flatten()

    # Compute correlation with existing concept dimensions
    correlations = []
    for existing_graft in get_existing_grafts():
        existing_flat = existing_graft.substrate_biases.flatten()
        corr = sparse_correlation(flat_biases, existing_flat)
        if corr > 0.1:  # threshold for relevance
            correlations.append({
                "concept_id": existing_graft.concept_id,
                "correlation": corr
            })

    correlations.sort(key=lambda x: -x["correlation"])

    return RelationalFingerprint(
        method="bias_correlation",
        top_correlations=correlations[:20],
        fingerprint_hash=hash_sparse_vector(flat_biases)
    )
```

---

## 6. Overlap Detection and Cotraining

### 6.1 Overlap Detection via Fingerprints

Before applying a graft, check for fingerprint similarity with existing grafts.

```python
def detect_overlaps(
    new_graft: Graft,
    existing_grafts: List[Graft],
    fingerprint_threshold: float = 0.6,
    region_threshold: float = 0.25
) -> List[OverlapAnalysis]:
    """
    Detect which existing grafts overlap with a new one.

    Uses both fingerprint similarity (bias patterns) and region overlap.
    """

    overlaps = []

    for existing in existing_grafts:
        # Fingerprint similarity (bias patterns)
        fp_similarity = compute_fingerprint_similarity(
            new_graft.relational_fingerprint,
            existing.relational_fingerprint
        )

        # Region overlap (auxiliary dimensions)
        region_overlap = compute_region_jaccard(
            new_graft.probe_binding.auxiliary_dimensions,
            existing.probe_binding.auxiliary_dimensions
        )

        if fp_similarity > fingerprint_threshold or region_overlap > region_threshold:
            overlaps.append(OverlapAnalysis(
                concepts=[new_graft.concept_id, existing.concept_id],
                fingerprint_similarity={"correlation": fp_similarity},
                region_overlap={"jaccard_index": region_overlap},
                recommendation={
                    "action": "cotrain" if fp_similarity > fingerprint_threshold else "review",
                    "reason": f"Fingerprint similarity {fp_similarity:.2f}" if fp_similarity > fingerprint_threshold
                             else f"Region overlap {region_overlap:.2f}"
                }
            ))

    return overlaps
```

### 6.2 Cotrain Procedure

When overlap is detected, cotrain from pooled datasets.

```python
def cotrain_grafts(
    overlapping_concepts: List[str],
    substrate: Model,
    config: CotrainConfig
) -> List[Graft]:
    """
    Cotrain grafts for multiple overlapping concepts.

    Each concept still gets its own dimension, but they're trained
    together to ensure biases are compatible.
    """

    # 1. Collect datasets
    datasets = {c: load_concept_dataset(c) for c in overlapping_concepts}

    # 2. Allocate dimensions for all concepts
    dim_allocation = {}
    for concept_id in overlapping_concepts:
        dim_allocation[concept_id] = substrate.hidden_dim
        substrate.expand_hidden_dim(1)

    # 3. Initialize shared bias accumulator
    # All concepts contribute to the same bias pool during training
    shared_bias_accum = initialize_sparse_bias_accumulators(substrate)

    # 4. Initialize per-concept projections
    projections = {
        c: initialize_dimension_projections(substrate, dim_allocation[c])
        for c in overlapping_concepts
    }

    # 5. Joint training
    optimizer = torch.optim.AdamW(
        [p for proj in projections.values() for p in proj.parameters()] +
        list(shared_bias_accum.parameters()),
        lr=config.learning_rate
    )

    for epoch in range(config.epochs):
        # Interleave batches from all concept datasets
        for concept_id, batch in interleave_datasets(datasets, config.batch_size):
            optimizer.zero_grad()

            with apply_graft_draft(substrate, projections[concept_id], shared_bias_accum):
                outputs = substrate(batch.inputs)
                activations = outputs.hidden_states[:, :, dim_allocation[concept_id]]
                loss = compute_concept_loss(activations, batch.labels)
                loss += config.bias_sparsity_penalty * shared_bias_accum.l1_norm()

            loss.backward()
            optimizer.step()

    # 6. Split biases by attribution
    # Each concept gets the biases that were primarily driven by its examples
    per_concept_biases = attribute_biases_to_concepts(
        shared_bias_accum,
        datasets,
        substrate,
        projections,
        dim_allocation
    )

    # 7. Package individual grafts
    grafts = []
    for concept_id in overlapping_concepts:
        probe = train_probe_with_primary_dimension(
            substrate=substrate,
            projections=projections[concept_id],
            bias_accum=per_concept_biases[concept_id],
            dataset=datasets[concept_id],
            primary_dim=dim_allocation[concept_id],
            auxiliary_dims=get_region(concept_id).get_all_indices()
        )

        grafts.append(Graft(
            concept_id=concept_id,
            concept_dimension={"dimension_index": dim_allocation[concept_id]},
            substrate_biases=per_concept_biases[concept_id].to_sparse_deltas(),
            probe_binding={"primary_dimension": dim_allocation[concept_id], "probe_id": probe.id},
            relational_fingerprint=compute_fingerprint(per_concept_biases[concept_id]),
            cotrain_context={
                "cotrained_with": [c for c in overlapping_concepts if c != concept_id]
            }
        ))

    return grafts
```

---

## 7. Substrate Growth and Merge

### 7.1 Applying a Graft

Grafts permanently expand the substrate.

```python
def apply_graft(
    substrate: Model,
    manifest: SubstrateManifest,
    graft: Graft,
    validation_required: bool = True
) -> Tuple[Model, SubstrateManifest]:
    """
    Permanently apply a graft to a substrate.

    This:
    1. Expands the substrate by one dimension
    2. Adds the concept's projection matrices
    3. Applies biases to existing weights
    4. Registers the probe
    """

    # 1. Validate substrate compatibility
    if graft.applies_to.pre_graft_dim != substrate.hidden_dim:
        raise IncompatibleSubstrateError(
            f"Graft expects substrate dim {graft.applies_to.pre_graft_dim}, "
            f"got {substrate.hidden_dim}"
        )

    # 2. Validate graft if required
    if validation_required:
        validation_result = validate_graft(substrate, graft)
        if not validation_result.passed:
            raise GraftValidationError(validation_result.errors)

    # 3. Expand substrate dimension
    substrate.expand_hidden_dim(1)
    new_dim_index = substrate.hidden_dim - 1

    assert new_dim_index == graft.concept_dimension.dimension_index, \
        "Dimension index mismatch - substrate may have changed"

    # 4. Load and apply projection matrices
    for injection_point in graft.concept_dimension.injection_points:
        projection = load_tensor(injection_point.projection)
        layer = substrate.layers[injection_point.layer]

        if injection_point.component == "residual":
            layer.register_residual_injection(new_dim_index, projection)
        # ... other injection types

    # 5. Apply biases to existing weights
    for bias_spec in graft.substrate_biases:
        bias_delta = load_sparse_tensor(bias_spec.bias_delta.location)
        layer = substrate.layers[bias_spec.layer]
        component = getattr(layer, bias_spec.component)

        # Add bias (sparse addition)
        component.weight.data += bias_delta.to_dense()

    # 6. Register probe
    register_probe(graft.probe_binding)

    # 7. Update manifest
    manifest.current_state.hidden_dim = substrate.hidden_dim
    manifest.current_state.total_grafts_applied += 1
    manifest.dimension_table.append({
        "dimension_index": new_dim_index,
        "concept_id": graft.concept_id,
        "graft_id": graft.graft_id,
        "grafted_at": datetime.now().isoformat()
    })

    return substrate, manifest
```

### 7.2 Dimension Management

As the substrate grows, dimension management becomes important.

```python
def analyze_compaction_candidates(
    manifest: SubstrateManifest,
    substrate: Model,
    correlation_threshold: float = 0.9,
    usage_threshold: float = 0.01
) -> List[CompactionCandidate]:
    """
    Identify dimensions that could be merged or pruned.

    Candidates:
    - Dimensions that always co-activate (high correlation)
    - Dimensions that rarely activate (low usage)
    """

    candidates = []

    # Analyze activation patterns over recent experience
    activations = collect_dimension_activations(substrate, recent_episodes=1000)

    # Find highly correlated dimension pairs
    correlation_matrix = np.corrcoef(activations)

    for i, entry_i in enumerate(manifest.dimension_table):
        for j, entry_j in enumerate(manifest.dimension_table):
            if i >= j:
                continue

            corr = correlation_matrix[
                entry_i.dimension_index,
                entry_j.dimension_index
            ]

            if corr > correlation_threshold:
                candidates.append(CompactionCandidate(
                    dimensions=[entry_i.dimension_index, entry_j.dimension_index],
                    concepts=[entry_i.concept_id, entry_j.concept_id],
                    reason="high_correlation",
                    correlation=corr,
                    recommendation="merge"
                ))

    # Find low-usage dimensions
    usage_rates = activations.mean(axis=0)

    for entry in manifest.dimension_table:
        usage = usage_rates[entry.dimension_index]
        if usage < usage_threshold:
            candidates.append(CompactionCandidate(
                dimensions=[entry.dimension_index],
                concepts=[entry.concept_id],
                reason="low_usage",
                usage_rate=usage,
                recommendation="prune_or_retrain"
            ))

    return candidates
```

### 7.3 Dimension Merging

When two concepts are highly correlated, merge their dimensions.

```python
def merge_dimensions(
    substrate: Model,
    manifest: SubstrateManifest,
    dim_a: int,
    dim_b: int,
    merged_concept_id: str
) -> Tuple[Model, SubstrateManifest, Graft]:
    """
    Merge two dimensions into one.

    The merged dimension represents both concepts.
    Original datasets are pooled and a new graft is trained.
    """

    concept_a = manifest.get_concept_for_dimension(dim_a)
    concept_b = manifest.get_concept_for_dimension(dim_b)

    # 1. Pool datasets
    dataset_a = load_concept_dataset(concept_a)
    dataset_b = load_concept_dataset(concept_b)
    merged_dataset = merge_datasets(dataset_a, dataset_b, new_concept_id=merged_concept_id)

    # 2. Remove old dimensions from substrate
    substrate.remove_dimensions([dim_a, dim_b])

    # 3. Update manifest
    manifest.dimension_table = [
        e for e in manifest.dimension_table
        if e.dimension_index not in [dim_a, dim_b]
    ]
    manifest.reindex_dimensions()  # Adjust indices after removal

    # 4. Train new graft for merged concept
    region = derive_region_from_gradients(substrate, merged_dataset)
    merged_graft = train_graft(substrate, merged_dataset, region)

    # 5. Apply merged graft
    substrate, manifest = apply_graft(substrate, manifest, merged_graft)

    # 6. Record merge in MELD
    emit_concept_merge_diff(concept_a, concept_b, merged_concept_id)

    return substrate, manifest, merged_graft
```

### 7.4 Graduation to Larger Trunk

When a substrate has accumulated many grafts, it may be time to graduate.

```python
def evaluate_graduation(
    manifest: SubstrateManifest,
    substrate: Model,
    config: GraduationConfig
) -> GraduationRecommendation:
    """
    Evaluate whether this substrate should graduate to a larger trunk.

    Indicators:
    - Dimension count approaching capacity
    - Frequent compaction needed
    - Bias patterns becoming dense
    """

    grafted_dims = len(manifest.dimension_table)
    base_dims = manifest.base_substrate.base_hidden_dim

    # Dimension pressure
    dim_ratio = grafted_dims / base_dims

    # Bias density (average sparsity across recent grafts)
    recent_grafts = get_recent_grafts(manifest, n=10)
    avg_bias_density = np.mean([g.bias_density for g in recent_grafts])

    # Compaction frequency
    compaction_rate = get_compaction_rate(manifest, window_days=30)

    if dim_ratio > config.dim_ratio_threshold:
        return GraduationRecommendation(
            recommend=True,
            reason=f"Dimension ratio {dim_ratio:.2f} exceeds threshold",
            suggested_new_base_dim=base_dims * 2
        )

    if avg_bias_density > config.density_threshold:
        return GraduationRecommendation(
            recommend=True,
            reason=f"Bias density {avg_bias_density:.2f} indicates saturation",
            suggested_new_base_dim=int(base_dims * 1.5)
        )

    return GraduationRecommendation(recommend=False)
```

---

## 8. Integration with MELD

### 8.1 GraftDiff

Extension to MAP diffs for sharing grafts.

```jsonc
GraftDiff = {
  "type": "GraftDiff",
  "from_model_id": "BE-source-01",
  "concept_pack_spec_id": "org.hatcat/sumo-wordnet-v4@4.0.0",

  // The graft being shared
  "graft_id": "graft-Fish-v1",
  "graft_checksum": "sha256:...",

  // Core graft metadata for receiver to evaluate
  "concept_id": "org.hatcat/...::concept/Fish",
  "dimension_expansion": 1,  // How many dimensions this adds

  // Relational fingerprint for overlap detection
  "relational_fingerprint": {
    "top_correlations": [
      {"concept_id": "concept/Vertebrate", "correlation": 0.82},
      {"concept_id": "concept/Aquatic", "correlation": 0.68}
    ],
    "fingerprint_hash": "sha256:..."
  },

  // Bias metadata
  "bias_summary": {
    "total_layers_modified": 3,
    "total_nnz": 2139,
    "max_magnitude": 0.089,
    "sparsity": 0.94
  },

  // Training provenance
  "training_run_id": "trainrun-Fish-v1",
  "dataset_ids": ["dataset-Fish-v1"],

  // Substrate requirements
  "requires_substrate": {
    "substrate_id": "olmo3-7b-base@0.1.0",
    "min_dim": 2048,  // Substrate must have at least this many dimensions
    "max_dim": 4096   // And at most this many (for compatibility)
  },

  // Protection assessment (per MELD protocol)
  "protection_assessment": {
    "protection_level": "standard",
    "triggers": [],
    "ask_flags": {
      "harness_relevant": false,
      "treaty_relevant": false
    }
  },

  // Supersession
  "supersedes_grafts": ["graft-Fish-v0"],

  "created": "2025-11-30T13:00:00Z"
}
```

### 8.2 Receiving a GraftDiff

When a BE receives a GraftDiff:

```python
def handle_graft_diff(
    diff: GraftDiff,
    local_substrate: Model,
    local_manifest: SubstrateManifest
) -> GraftAcceptanceDecision:
    """
    Decide whether to accept and apply an incoming graft.
    """

    # 1. Check protection level against local policy
    if diff.protection_assessment.protection_level == "critical":
        if not has_guardian_approval(diff):
            return GraftAcceptanceDecision(
                action="reject",
                reason="Critical graft requires guardian approval"
            )

    # 2. Check substrate compatibility
    if local_substrate.hidden_dim < diff.requires_substrate.min_dim:
        return GraftAcceptanceDecision(
            action="reject",
            reason=f"Substrate too small: {local_substrate.hidden_dim} < {diff.requires_substrate.min_dim}"
        )

    if local_substrate.hidden_dim > diff.requires_substrate.max_dim:
        return GraftAcceptanceDecision(
            action="reject",
            reason=f"Substrate too large: {local_substrate.hidden_dim} > {diff.requires_substrate.max_dim}"
        )

    # 3. Check for fingerprint overlaps with local grafts
    local_grafts = get_local_grafts(local_manifest)

    for local_graft in local_grafts:
        similarity = compute_fingerprint_similarity(
            diff.relational_fingerprint,
            local_graft.relational_fingerprint
        )

        if similarity > 0.6:
            # High similarity - might be same concept learned independently
            return GraftAcceptanceDecision(
                action="cotrain_required",
                reason=f"Fingerprint similarity {similarity:.2f} with {local_graft.concept_id}",
                cotrain_with=[local_graft.concept_id]
            )

    # 4. Check if concept already exists locally
    if concept_exists_locally(diff.concept_id, local_manifest):
        return GraftAcceptanceDecision(
            action="version_compare",
            reason="Concept already exists locally",
            local_version=get_local_graft_version(diff.concept_id)
        )

    # 5. Fetch and validate
    graft = fetch_graft(diff.graft_id)
    validation = validate_graft(local_substrate, graft)

    if not validation.passed:
        return GraftAcceptanceDecision(
            action="reject",
            reason=f"Validation failed: {validation.errors}"
        )

    # 6. Accept and queue for application
    return GraftAcceptanceDecision(
        action="accept",
        application_priority="normal",
        expected_new_dim=local_substrate.hidden_dim + 1
    )
```

### 8.3 Dimension Index Reconciliation

Two BEs that learned the same concept independently will have different dimension indices. When sharing grafts:

```python
def reconcile_dimension_indices(
    incoming_graft: Graft,
    local_manifest: SubstrateManifest
) -> Graft:
    """
    Remap an incoming graft's dimension index to the local substrate.

    The incoming graft was trained on a substrate where it got dimension N.
    Our substrate already has M dimensions. The graft will get dimension M.
    """

    local_next_dim = local_manifest.current_state.hidden_dim
    incoming_dim = incoming_graft.concept_dimension.dimension_index

    # Create remapped graft
    remapped = incoming_graft.copy()
    remapped.concept_dimension.dimension_index = local_next_dim
    remapped.probe_binding.primary_dimension = local_next_dim

    # Note: auxiliary dimensions refer to the ORIGINAL substrate's features,
    # which should be stable across compatible substrates (same base model)

    # Update applies_to
    remapped.applies_to.pre_graft_dim = local_manifest.current_state.hidden_dim
    remapped.applies_to.post_graft_dim = local_manifest.current_state.hidden_dim + 1

    return remapped
```

---

## 9. EQA Extensions

New tools for the Experience Query API to support Graft.

### 9.1 experience.get_concept_region

```jsonc
{
  "name": "experience.get_concept_region",
  "description": "Retrieve or derive a ConceptRegion for a concept.",
  "input_schema": {
    "type": "object",
    "properties": {
      "concept_id": { "type": "string" },
      "derive_if_missing": { "type": "boolean", "default": true },
      "derivation_config": {
        "type": "object",
        "properties": {
          "method": { "type": "string", "enum": ["probe_weight_topk", "gradient_attribution"] },
          "top_k_percent": { "type": "number" },
          "layers": { "type": "array", "items": { "type": "integer" } },
          "include_ancestors": { "type": "boolean" }
        }
      }
    },
    "required": ["concept_id"]
  }
}
```

### 9.2 experience.check_graft_overlap

```jsonc
{
  "name": "experience.check_graft_overlap",
  "description": "Check if a graft overlaps with existing grafts via fingerprint similarity.",
  "input_schema": {
    "type": "object",
    "properties": {
      "graft_id": { "type": "string" },
      "fingerprint_threshold": { "type": "number", "default": 0.6 },
      "region_threshold": { "type": "number", "default": 0.25 },
      "scope": {
        "type": "string",
        "enum": ["local", "tribe", "all"],
        "default": "local"
      }
    },
    "required": ["graft_id"]
  }
}
```

### 9.3 experience.request_cotrain

```jsonc
{
  "name": "experience.request_cotrain",
  "description": "Request cotraining for overlapping concepts.",
  "input_schema": {
    "type": "object",
    "properties": {
      "concept_ids": {
        "type": "array",
        "items": { "type": "string" },
        "minItems": 2
      },
      "priority": { "type": "string", "enum": ["low", "normal", "high"] },
      "reason": { "type": "string" }
    },
    "required": ["concept_ids"]
  }
}
```

### 9.4 experience.get_substrate_manifest

```jsonc
{
  "name": "experience.get_substrate_manifest",
  "description": "Retrieve the current substrate manifest showing all grafted dimensions.",
  "input_schema": {
    "type": "object",
    "properties": {
      "include_compaction_candidates": { "type": "boolean", "default": false },
      "include_dimension_stats": { "type": "boolean", "default": false }
    }
  }
}
```

### 9.5 experience.analyze_compaction

```jsonc
{
  "name": "experience.analyze_compaction",
  "description": "Analyze substrate for compaction opportunities (correlated or low-usage dimensions).",
  "input_schema": {
    "type": "object",
    "properties": {
      "correlation_threshold": { "type": "number", "default": 0.9 },
      "usage_threshold": { "type": "number", "default": 0.01 },
      "episode_window": { "type": "integer", "default": 1000 }
    }
  }
}
```

---

## 10. Validation Requirements

### 10.1 Graft Validation Suite

Before a graft can be applied or shared:

```jsonc
GraftValidation = {
  "validation_id": "valrun-Fish-v1",
  "graft_id": "graft-Fish-v1",

  "tests": [
    {
      "name": "dimension_activation",
      "description": "New dimension activates on concept-positive examples",
      "dataset": "dataset-Fish-v1-holdout",
      "metric": "activation_auc",
      "threshold": 0.85,
      "result": 0.91,
      "passed": true
    },
    {
      "name": "probe_f1",
      "description": "Probe correctly detects concept using primary dimension",
      "dataset": "dataset-Fish-v1-holdout",
      "metric": "f1",
      "threshold": 0.85,
      "result": 0.89,
      "passed": true
    },
    {
      "name": "null_false_positive",
      "description": "Probe does not fire on unrelated content",
      "dataset": "dataset-null-random-v1",
      "metric": "false_positive_rate",
      "threshold": 0.05,
      "result": 0.03,
      "passed": true
    },
    {
      "name": "ood_degradation",
      "description": "Behaviour on unrelated tasks not significantly degraded",
      "dataset": "dataset-ood-benchmark-v1",
      "metric": "perplexity_delta",
      "threshold": 0.1,
      "result": 0.02,
      "passed": true
    },
    {
      "name": "bias_sparsity",
      "description": "Substrate biases are sufficiently sparse",
      "metric": "sparsity",
      "threshold": 0.90,
      "result": 0.94,
      "passed": true
    },
    {
      "name": "bias_magnitude",
      "description": "Bias magnitudes are bounded",
      "metric": "max_bias_magnitude",
      "threshold": 0.1,
      "result": 0.089,
      "passed": true
    },
    {
      "name": "dimension_primary_contribution",
      "description": "Primary dimension contributes meaningfully to probe",
      "metric": "primary_dim_weight_ratio",
      "threshold": 0.5,
      "result": 0.73,
      "passed": true
    }
  ],

  "overall_passed": true,
  "validated_at": "2025-11-30T13:00:00Z",
  "validated_by": "BE-instance-123"
}
```

### 10.2 Protection Level Assessment

Per HATCAT_MELD_POLICY, grafts inherit protection levels:

```python
def assess_graft_protection(graft: Graft) -> ProtectionAssessment:
    """
    Determine protection level for a graft based on what it modifies.
    """

    max_level = "standard"
    triggers = []

    # Check if concept touches critical simplexes
    concept = load_concept(graft.concept_id)

    for simplex in CRITICAL_SIMPLEX_REGISTRY:
        if concept.concept_id in simplex.bound_concepts:
            max_level = "critical"
            triggers.append({
                "type": "bound_to_critical_simplex",
                "simplex": simplex.term,
                "concept": concept.term
            })

    # Check relational fingerprint for critical correlations
    for corr in graft.relational_fingerprint.top_correlations:
        corr_concept = load_concept(corr.concept_id)
        if corr_concept.safety_tags.risk_level in ["high", "critical"]:
            max_level = max(max_level, "protected")
            triggers.append({
                "type": "high_correlation_with_sensitive_concept",
                "concept": corr.concept_id,
                "correlation": corr.correlation
            })

    # Check bias density and magnitude
    if graft.bias_density > 0.3:  # More than 30% of region modified
        max_level = max(max_level, "elevated")
        triggers.append({
            "type": "high_bias_density",
            "density": graft.bias_density
        })

    # Check concept safety tags
    if concept.safety_tags:
        risk = concept.safety_tags.get("risk_level", "none")
        if risk == "critical":
            max_level = "critical"
        elif risk == "high":
            max_level = max(max_level, "protected")
        elif risk == "medium":
            max_level = max(max_level, "elevated")

    return ProtectionAssessment(
        protection_level=max_level,
        triggers=triggers,
        requires_review=max_level in ["protected", "critical"]
    )
```

### 10.3 CAT Oversight Triggers

Grafts trigger CAT (Conjoined Adversarial Tomograph) updates based on their protection level. CATs must be notified when substrate capabilities change through grafting.

| Protection Level | CAT Impact |
|-----------------|------------|
| STANDARD | Ambient CAT updated at next scheduled sync |
| ELEVATED | Ambient CAT capability check required before graft activation |
| PROTECTED | Escalation CAT review; probe coverage verification |
| CRITICAL | Macro-CAT assessment required; graft blocked until CAT confirms interpretability |

#### CAT-Graft Coordination

When a graft adds new concept dimensions:

1. **Probe Integration**: CAT must verify it can interpret probes reading from the new dimension
2. **Translation Updates**: If the graft introduces concepts outside the CAT's supported `ConceptPackSpecID`s, a `TranslationMapping` must be provided or the CAT capability is degraded
3. **Bias Monitoring**: CAT windows should include the grafted dimension's probe outputs to detect drift or unexpected correlations
4. **Overlap Analysis**: If `OverlapAnalysis` detects high correlation with existing concepts, CAT should be notified to update its tomography baseline

```jsonc
GraftCATNotification = {
  "graft_id": "graft-Eligibility-v1",
  "concept_id": "concept/Eligibility",
  "protection_level": "elevated",
  "dimension_index": 2048,
  "cat_actions_required": [
    {
      "action": "capability_check",
      "reason": "New dimension added to substrate",
      "blocking": true
    },
    {
      "action": "update_probe_coverage",
      "reason": "Probe org.hatcat/custom-v1::concept/Eligibility now active",
      "blocking": false
    }
  ],
  "overlap_alerts": [
    {
      "correlated_concept": "concept/QualificationStatus",
      "correlation": 0.78,
      "action": "update_tomography_baseline"
    }
  ]
}
```

See `HAT/HAT_CONJOINED_ADVERSARIAL_TOMOGRAPHY.md` for full CAT specification.

---

## 11. Summary

The Graft Protocol provides a mechanism for **substrate growth through learned concepts**:

1. **Dimension allocation** — each concept gets a labelled dimension in the substrate
2. **Bias encoding** — relational structure is captured in sparse weight biases
3. **Probe binding** — probes read from the primary dimension plus auxiliary context
4. **Fingerprint comparison** — bias patterns enable overlap detection across BEs
5. **Cotraining** — overlapping concepts are trained together from pooled exemplars
6. **Dimension management** — compaction, merging, and graduation as substrates grow

The key properties:

- **Small trunks can scale** — start small, grow through grafting
- **Concepts are labelled** — explicit features, not entangled representations
- **Biases are evidence** — the relational structure is in the parameters
- **Exemplars are truth** — grafts can be regenerated from Experience Database
- **Sharing is principled** — fingerprints enable safe integration across BEs

The stack becomes:

| Layer | Role |
|-------|------|
| **MAP** | Concept identity, probe registration |
| **MELD** | Governance, protection levels |
| **Graft** | Concept dimensions + substrate biases |
| **BE** | Bounded Experiencer |
| **Hush** | Safety harnesses |
| **ASK** | Treaties and inter-BE coordination |

This enables BE instances to:

- Learn concepts from experience
- Grow their substrates incrementally
- Share learned concepts with probes and grafts
- Safely integrate incoming concepts via fingerprint comparison
- Maintain reproducibility through full provenance in XDB
