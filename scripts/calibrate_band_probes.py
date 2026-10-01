#!/usr/bin/env python3
"""
Calibrate every band probe in a lens pack on its own background.

Each probe (`<concept>@L<model_layer>.pt`) is scored on background text it was
not trained on, and its score distribution is stored as quantiles. A runtime
maps a raw probe score to the fraction of background it exceeds (a percentile,
i.e. one minus its false-positive rate), so probes are comparable across layers
and concepts, and a lens can combine its probes as the max of their calibrated
scores: it fires when any layer's evidence stands out against that layer's own
background.

Calibrate in the same unit the runtime reads. HAT scores one token's hidden
state at a time during generation, and single-token states are more extreme
than mean-pooled ones: a pooled calibration saturates at runtime (most lenses
read 1.0). --unit tokens (default) samples per-token states from the
background texts; --unit pooled matches eval_lens_confusion's pooled test.

With pooled calibration, on the university ontology this beat the raw mean and raw max of probes on
cross-lens ranking (University top-1 17.6% -> 19.8% with contrasts, 11.3% ->
18.8% without); z-scoring instead was worse, as heavy-tailed probes dominate
the max.

Writes `probe_calibration.json` into the lens pack:
    {"method": "percentile", "background": {...},
     "probes": {"layer1/UrbanPlanning@L22": {"model_layer": 22, "quantiles": [...]}, ...}}

Usage:
    python scripts/calibrate_band_probes.py \\
        --lens-pack lens_packs/gemma-4-e4b-it_university-v3-contrasts-bands \\
        --concept-pack university-v3-contrasts --model google/gemma-4-E4B-it
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_lens_confusion import collect_split
from src.hat.classifiers.classifier import load_classifier
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.map.training.sumo_classifiers import extract_activations


def split_background(concept_pack: str, skeleton: Path, per_concept: int):
    """Even-indexed half of the pack's held-out split: text no lens trained on.
    The odd half stays free for evaluation (eval_lens_confusion --split-half test)."""
    hierarchy = PROJECT_ROOT / "concept_packs" / concept_pack / "hierarchy"
    builder = json.loads((hierarchy.parent / "pack.json").read_text())["ontology_stack"]["hierarchy_builder"]
    split = builder.get("holdout")
    if not split:
        raise SystemExit(f"{concept_pack} has no held-out split; pass --background-file instead")
    layers = sorted(hierarchy.glob("layer*.json"))
    trained = [json.loads(p.read_text()) for p in layers]
    trained = [d for d in trained if not all(c.get("definition_only") for c in d["concepts"])]
    deepest = {c["sumo_term"]: c for c in max(trained, key=lambda d: d["layer"])["concepts"]}
    held = defaultdict(list)
    collect_split(json.loads(skeleton.read_text()), deepest, split["level"], split["fraction"], held, per_concept)
    texts = [t for ts in held.values() for i, t in enumerate(ts) if i % 2 == 0]
    return texts, {"source": "held-out split, even half", "concept_pack": concept_pack, **split}


def token_states(model, tokenizer, texts, layers, per_text, batch_size=16, seed=0):
    """Hidden states at sampled token positions (skipping the first few), for each needed layer.

    hidden_states[L + 1] is model layer L; each row is one token's state, as HAT
    scores it during generation."""
    rng = np.random.default_rng(seed)
    tokenizer.padding_side = "right"
    rows = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], return_tensors="pt", padding=True,
                              truncation=True, max_length=128).to("cuda")
            hidden = model(**batch, output_hidden_states=True).hidden_states
            lengths = batch.attention_mask.sum(1).tolist()
            for i, n in enumerate(lengths):
                positions = np.arange(3, n) if n > 4 else np.arange(n)
                positions = rng.choice(positions, min(per_text, len(positions)), replace=False)
                rows.append(torch.stack([hidden[l + 1][i, positions] for l in layers], dim=1).float().cpu())
    return torch.cat(rows)  # [n_states, n_layers, hidden]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lens-pack", type=Path, required=True)
    parser.add_argument("--concept-pack", default=None, help="Concept pack whose held-out split is the background")
    parser.add_argument("--background-file", type=Path, default=None, help="Or: one background text per line")
    parser.add_argument("--skeleton", type=Path, default=PROJECT_ROOT / "results/ontology_skeleton_v2.json")
    parser.add_argument("--per-concept", type=int, default=20, help="Held-out texts drawn per concept (half are used)")
    parser.add_argument("--quantiles", type=int, default=201)
    parser.add_argument("--unit", choices=["tokens", "pooled"], default="tokens",
                        help="tokens: per-token hidden states, as HAT reads during generation (default); "
                             "pooled: one mean-pooled state per text")
    parser.add_argument("--tokens-per-text", type=int, default=8)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="Default <lens-pack>/probe_calibration.json (what HAT loads)")
    args = parser.parse_args()

    if args.background_file:
        texts = [t for t in args.background_file.read_text().splitlines() if t.strip()]
        background = {"source": str(args.background_file)}
    elif args.concept_pack:
        texts, background = split_background(args.concept_pack, args.skeleton, args.per_concept)
    else:
        raise SystemExit("Pass --concept-pack (held-out split) or --background-file")
    background["n_texts"] = len(texts)
    print(f"Background: {len(texts)} texts ({background['source']})")

    probe_files = sorted(args.lens_pack.glob("layer*/*@L*.pt"))
    needed = sorted({int(p.stem.rpartition("@L")[2]) for p in probe_files})
    print(f"{len(probe_files)} probes reading {len(needed)} model layers")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda", local_files_only=True).eval()
    if args.unit == "pooled":
        X = extract_activations(model, tokenizer, texts, "cuda", extraction_mode="prompt", layer_idx=needed)
        X = torch.tensor(X, dtype=torch.float32).reshape(len(texts), len(needed), -1)
    else:
        X = token_states(model, tokenizer, texts, needed, args.tokens_per_text)
    del model
    torch.cuda.empty_cache()
    X = X.cuda()
    X = (X - X.mean(-1, keepdim=True)) / (X.std(-1, keepdim=True) + 1e-8)
    background["unit"] = args.unit
    background["n_states"] = int(X.shape[0])
    column = {layer: i for i, layer in enumerate(needed)}

    qs = np.linspace(0, 1, args.quantiles)
    probes = {}
    with torch.no_grad():
        for path in probe_files:
            model_layer = int(path.stem.rpartition("@L")[2])
            clf = load_classifier(path, device="cuda", classifier_type="mlp").eval()
            scores = torch.sigmoid(clf(X[:, column[model_layer], :]).squeeze(-1)).cpu().numpy()
            probes[f"{path.parent.name}/{path.stem}"] = {
                "model_layer": model_layer,
                "quantiles": [round(float(v), 6) for v in np.quantile(scores, qs)],
            }

    out = args.output or args.lens_pack / "probe_calibration.json"
    out.write_text(json.dumps({"method": "percentile", "background": background, "probes": probes}))
    print(f"Wrote {len(probes)} probe calibrations to {out}")


if __name__ == "__main__":
    main()
