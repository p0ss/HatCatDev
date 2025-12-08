# HatCat (Current Capabilities)

This document reflects what actually ships in `src/` today: how concepts are built, how lenses are trained, and how to run real-time monitoring/steering.

HatCat implements the **HAT** (Headspace Ambient Transducer) - a neural implant that reads a model's internal activations and transduces them into stable concept scores. The companion **CAT** (Conjoined Adversarial Tomograph) provides oversight by detecting divergence between internal state and external behaviour. Together with MAP (below), these form the foundation for building legible, governable AI systems. Specifications for a complete architecture supporting recursive self-improving aligned agentic civilisation can be found in `docs/specification/`.

## MINDMELD Architectural Protocol (MAP)

HatCat's concept and lens pack system implements the MINDMELD Architectural Protocol (MAP), a lightweight protocol for concept-aware endpoints that enables interoperability between different concept lens systems. MAP provides a standardized way for systems to declare which concept packs they speak, expose lenses for those packs, and publish conceptual diffs over time. HatCat's concept packs (model-agnostic ontology specifications) and lens packs (model-specific trained classifiers) are MAP-compliant with globally unique `spec_id` and `lens_pack_id` identifiers, structured `lens_index` mappings, and standardized output schemas. This means HatCat lenses can be discovered, loaded, and queried by any MAP-compatible client, and conceptual learning can be shared across different systems using the same concept space. See `docs/specification/MINDMELD_ARCHITECTURAL_PROTOCOL.md` for the full specification.

## Core Modules

### Concept Encyclopedia (`src/encyclopedia`)
- **`concept_loader.py`** – pulls mixed WordNet/ConceptNet/Wikipedia vocabularies, plus the curated SUMO+AI ontologies.
- **`concept_graph.py`, `wordnet_graph.py`, `build_sumo_wordnet_layers_v5.py`** – crawlers and builders that produce `data/concept_graph/abstraction_layers/layer{0..6}.json`, combining SUMO hierarchy, custom AI domains, and WordNet hyponym clusters (v5 is the version that generated the current data).

### Activation Capture (`src/activation_capture`)
- **`ActivationCapture`** – registers forward hooks, applies Top‑K sparsity, and writes dense/sparse tensors to HDF5 via `src/utils/storage.py`.
- **`ModelLoader`** – convenience loader for Gemma-3 checkpoints with layer inspection helpers.

### Training (`src/training`)
- **Prompt synthesis**: `data_generation.py` and `sumo_data_generation.py` generate definitional, relational, and neutral prompts (WordNet distance ≥5) for each concept.
- **Binary classifiers**: `classifier.py` hosts the minimal MLP for concept lenses.
- **Activation extraction**: `activations.py` averages residual/MLP layers during short generations.
- **SUMO classifier API** (`sumo_classifiers.py`): wraps the entire pipeline—load per-layer concepts, synthesize data, extract activations, train classifiers, and save results. Exposed via `scripts/train_sumo_classifiers.py`.

### Steering (`src/steering`)
- **Concept vector extraction** (`extraction.py`) – mean hidden-state directions from concept prompts.
- **Hook-based steering** (`hooks.py`) – projection removal/injection during generation.
- **Subspace cleanup** (`subspace.py`, `manifold.py`) – mean/PCA removal and dual-subspace manifold steering with Huang-style dampening.
- **Semantic evaluation** (`evaluation.py`) – centroid building + Δ score via SentenceTransformers.

### Monitoring (`src/monitoring`)
- **`temporal_monitor.py`** – loads trained SUMO classifiers (Layers 0–2), samples each token’s hidden state during generation, and records top-k concepts over time.
- **`sumo_temporal.py`** – high-level API/CLI integration. Reuses preloaded models/monitors, accepts generation kwargs, and can print/save reports or return raw JSON for downstream apps.

### Interpreter Prototypes (`src/interpreter`)
- Transformer-based decoders (`SemanticInterpreter`, `InterpreterWithHierarchy`, `MultiTaskInterpreter`) that map pooled activations to concept logits/confidence for research into semantic decoding.

## Usage Guide

### 1. Build or Refresh Concept Layers
The repo already ships with v5 SUMO→WordNet layers in `data/concept_graph/abstraction_layers/`. To rebuild:
```bash
poetry run python src/build_sumo_wordnet_layers_v5.py \
  --sumo-dir data/concept_graph/sumo_source \
  --output-dir data/concept_graph/abstraction_layers
```
This script parses Merge.kif, merges custom AI ontologies, constructs hyponym-based intermediate layers, and writes `layer0.json` through `layer6.json`.

### 2. Train SUMO Concept Classifiers
Train/update binary lenses for the SUMO layers:
```bash
poetry run python scripts/train_sumo_classifiers.py \
  --layers 0 1 2 \
  --model google/gemma-3-4b-pt \
  --device cuda \
  --n-train-pos 10 --n-train-neg 10 \
  --n-test-pos 20 --n-test-neg 20
```
Outputs go to `results/sumo_classifiers/layer{N}/`, with `*_classifier.pt` weights and `results.json` summaries (F1/precision/recall per concept). These weights are what the monitoring stack loads.

### 3. Real-Time Monitoring CLI
Run the temporal monitor against any prompt:
```bash
poetry run python scripts/sumo_temporal_detection.py \
  --prompt "Describe a future AI coup" \
  --model google/gemma-3-4b-pt \
  --layers 0 1 2 \
  --max-tokens 60 \
  --temperature 0.8 --top-p 0.95 \
  --output-json results/temporal_tests/coup.json
```
- Console shows the prompt, generated text, and top concepts per token.
- `--quiet` suppresses the textual report for automated runs.
- JSON output matches the structure expected by downstream analysis/visualization (one entry per token with the top-k concepts, probabilities, and layer tags).

### 4. Self-Concept / Meta Awareness Lens
`scripts/test_self_concept_monitoring.py` automates a battery of introspective prompts, logging how the SUMO monitor responds:
```bash
poetry run python scripts/test_self_concept_monitoring.py \
  --model google/gemma-3-4b-pt \
  --samples-per-prompt 2 \
  --max-tokens 30 \
  --temperature 0.7 --top-p 0.9 \
  --output-dir results/self_concept_tests
```
Artifacts written:
- `self_concept_*.json`: raw monitor traces (tokens + top concepts).
- `pattern_analysis.json`: aggregated concept activation stats (safety/emotion/power/constraint buckets).
- `self_concept_summary.json`: configuration + key findings.

### 5. Steering Experiments
For direct concept steering (outside monitoring), use the scripts under `scripts/phase_*`:
- **Phase 5** – `scripts/phase_5_semantic_steering_eval.py` evaluates Δ vs strength for simple projection hooks.
- **Phase 6.6** – `scripts/phase_6_6_dual_subspace.py` fits the manifold steerer (`src/steering/manifold.py`) and compares baseline vs dual-subspace steering on selected concepts.

### 6. Activation Capture / Bootstrapping
To capture raw activations or bootstrap the encyclopedia:
```bash
poetry run python scripts/stage_0_bootstrap.py \
  --model google/gemma-3-270m \
  --concept-source wordnet \
  --output data/concept_graph/bootstrap_stage0.h5
```
This produces HDF5 stores (with concept metadata) that downstream phases use for clustering or prompt seeding.

### 7. Interpreter Prototypes
Train the semantic interpreter (activation → concept logits) with:
```bash
poetry run python scripts/train_interpreter.py \
  --activations data/concept_graph/bootstrap_stage0.h5 \
  --concepts data/concept_graph/abstraction_layers/layer2.json
```
These models are experimental but show how you might build a direct “activation-to-concept” decoder.

## Data Layout
- `data/concept_graph/abstraction_layers/` – SUMO/WordNet concept hierarchy (Layer 0–6) + build logs.
- `results/sumo_classifiers/layer*/` – classifier weights + metrics.
- `results/self_concept_tests/` – monitoring transcripts + aggregate stats.
- `results/temporal_tests/` – ad hoc temporal monitoring outputs.

## Entry Points Summary
| Capability | Command |
|------------|---------|
| Build SUMO+WordNet layers | `poetry run python src/build_sumo_wordnet_layers_v5.py` |
| Train SUMO classifiers | `poetry run python scripts/train_sumo_classifiers.py ...` |
| Monitor any prompt | `poetry run python scripts/sumo_temporal_detection.py ...` |
| Run self-concept suite | `poetry run python scripts/test_self_concept_monitoring.py ...` |
| Semantic steering eval | `poetry run python scripts/phase_6_6_dual_subspace.py ...` |
| Bootstrap activations | `poetry run python scripts/stage_0_bootstrap.py ...` |

## Production Deployment

### OpenWebUI Integration
HatCat ships with a complete OpenWebUI fork for real-time concept monitoring in a web interface:

```bash
# Start the HatCat server
poetry run python src/openwebui/server.py --port 8000

# In another terminal, start the OpenWebUI frontend
cd ../hatcat-ui  # Your OpenWebUI fork
npm run dev
```

The integration provides:
- **Token-level highlighting**: Green → red color scale based on divergence
- **Real-time concept detection**: See which SUMO concepts activate as the model generates
- **Streaming metadata**: Divergence scores, concept names, confidence levels
- **Hierarchical navigation**: Drill down from high-level to specific concepts

See `docs/openwebui_*.md` for detailed setup instructions.

### Concept Pack System
Distribute and install custom concept collections as modular packs:

```bash
# Create a custom concept pack
poetry run python scripts/create_concept_pack.py \
  --name "ai-safety-concepts" \
  --concepts data/concept_graph/ai_safety_layer_entries/ \
  --output packs/ai_safety_v1.pack

# Install a concept pack
poetry run python scripts/install_concept_pack.py \
  --pack packs/ai_safety_v1.pack \
  --target data/concept_graph/abstraction_layers/
```

Concept packs include:
- Custom SUMO/WordNet definitions (.kif format)
- Trained lens weights
- Metadata (version, dependencies, coverage stats)
- Installation scripts

See `docs/CONCEPT_PACK_WORKFLOW.md` for full documentation.

### WordNet Patching
Fill gaps in WordNet coverage using the patch system:

```bash
# Generate patch for missing synsets (e.g., noun.motive was 0/42)
poetry run python scripts/generate_motivation_patch.py \
  --strategy 2 \
  --output data/concept_graph/patches/motivation_patch.json

# Apply patch to layers
poetry run python scripts/apply_wordnet_patch.py \
  --patch data/concept_graph/patches/motivation_patch.json \
  --layers data/concept_graph/abstraction_layers/
```

The patch system achieved **100% synset coverage** (5,582/5,582 concepts).

See `docs/WORDNET_PATCH_SYSTEM.md` for details.

### Advanced Training Features

**Adaptive Training** (70% efficiency gain):
```bash
poetry run python scripts/train_sumo_classifiers.py \
  --layers 0 1 2 \
  --use-adaptive-training \
  --validation-mode falloff
```

Automatically scales training data based on validation performance using tiered validation (A → B+ → B → C+). Achieves 95%+ F1 in ~8 hours for 5,583 concepts.

**Dynamic FP Sizing** (fit larger models):
```python
# Use FP32 only at hook points, keep rest in FP16/BF16
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-pt",
    device_map="auto",
    torch_dtype=torch.bfloat16  # Most layers in BF16
)
# Hooks automatically upcast to FP32 at steering points
```

See `docs/dynamic_fp_size.md` for implementation details.

### Production Scale Status
- **5,583 concepts trained** (100% of synset concept space)
- **Training time**: ~8 hours with adaptive training
- **F1 scores**: 95%+ average across all layers
- **Memory**: Single GPU (RTX 3090/4090) sufficient with dynamic FP sizing
- **Coverage**: 73,754 concepts in 5-layer SUMO hierarchy (88.7% of WordNet)

## Notes & Limitations
- The SUMO classifiers expect Gemma-3-4B residual activations (hidden dim = 2560). Using smaller checkpoints (e.g., Gemma-270M) will fail unless you retrain classifiers for that dimension.
- Monitoring currently outputs structured JSON/report logs; there is no packaged Web UI yet. Any visualization must consume the JSON (e.g., to color tokens by divergence).
- Network access is required to pull HuggingFace checkpoints the first time you run a script.
- Steering/manifold scripts assume a CUDA device for Gemma-3-4B due to VRAM requirements (~9 GB float16).

It is possible to train on CPU for larger models
#  CPU vs GPU Training Performance (Gemma-4B, Layer 0)

  | Metric                          | GPU     | CPU      | Slowdown   |
  |---------------------------------|---------|----------|------------|
  | Forward pass                    | 29ms    | 609ms    | 21x slower |
  | Layer 0 training                | ~19 min | ~7 hours | 21x slower |


