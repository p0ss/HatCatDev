#!/usr/bin/env python3
"""
Scaling curves for band-probe lenses: held-out quality vs training data.

Varies the concept pack (definition depth, cross-links on/off) and the number
of training samples per class, trains fixed-size band probes for a subset of
concepts, and scores each lens on held-out text it never saw: the topic
descriptions of Departments held out of every pack by the builder's
deterministic split (build_skeleton_concept_pack.py --holdout-level 4).

Metric per lens is AUROC, rank-based so it needs no calibration and no other
lenses: positives are held-out Departments under the concept; negatives are
held-out Departments under all other Universities ("all"), or only under its
siblings and cross-linked concepts ("hard", the disambiguation that matters).

Band layers are selected once per concept and shared by every configuration,
so the curves measure data, not layer-selection noise. Each concept's training
text is extracted once for all three bands.

Two budgets:
- content (default): real text only, sized by what the concept has - its MELD
  examples, its descendants' definitions (up to --descendant-cap) and, with
  --contrasts, both sides of every "differs because" boundary. Negatives are
  filled to balance from other concepts' real text.
- fixed (--budget fixed --samples ...): HatCat's templated generator at N
  samples per class, which pads with rewordings once real content runs out.

Usage:
    python scripts/scaling_experiment.py \\
        --packs university-exp-d1-nolinks university-exp-d2-nochildlinks \\
        --contrasts results/contrasts/university-exp-d1.jsonl \\
        --bands-from results/scaling/university_gemma4_d0_d1.json \\
        --model google/gemma-4-E4B-it \\
        --output results/scaling/university_gemma4_contrasts.json
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_skeleton_concept_pack import clean_scope, is_held_out
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.map.training.sumo_classifiers import (
    extract_activations,
    load_all_concepts,
    load_layer_concepts,
    select_layers_for_concept,
    train_simple_classifier,
)
from src.map.training.sumo_data_generation import (
    build_sumo_negative_pool,
    create_content_dataset,
    create_sumo_training_dataset,
)

LAYER = 1  # Universities


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Per-sample standardisation, as train_simple_classifier applies to its inputs."""
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-8)


def pick_subset(concepts, per_field, seed):
    by_field = defaultdict(list)
    for c in concepts:
        by_field[c["parent_concepts"][0]].append(c["sumo_term"])
    rng = random.Random(seed)
    return sorted(t for terms in by_field.values() for t in rng.sample(sorted(terms), min(per_field, len(terms))))


def held_out_texts(skeleton, reference_pack, fraction, max_per_concept, seed):
    """Held-out Department descriptions, attributed to their University (by node path)."""
    path_to_term = {c["node_path"]: t for t, c in reference_pack.items()}
    texts, owners = [], []
    rng = random.Random(seed)
    for root in skeleton["roots"]:
        for uni in root["children"]:
            term = path_to_term.get(f"{root['id']}/{uni['id']}")
            if term is None:
                continue
            candidates = [
                clean_scope(dept["scope"])
                for school in uni["children"] for dept in school.get("children", [])
                if dept.get("scope") and is_held_out(f"{root['id']}/{uni['id']}/{school['id']}/{dept['id']}", fraction)
            ]
            for text in rng.sample(candidates, min(max_per_concept, len(candidates))):
                texts.append(text)
                owners.append(term)
    return texts, np.array(owners)


def load_contrasts(path):
    """term -> list of (own-side examples, other-side examples, partner)."""
    by_term = defaultdict(list)
    if path:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            by_term[r["a"]].append((r["a_not_b"], r["b_not_a"], r["b"]))
            by_term[r["b"]].append((r["b_not_a"], r["a_not_b"], r["a"]))
    return by_term


def content_dataset(term, all_concepts, contrasts, cap, rng, diagonal_cap=0, diagonal_extra=False):
    """The trainer's content budget, with contrasts from a JSONL attached in memory."""
    records = {c["sumo_term"]: c for c in all_concepts}
    concept = dict(records[term])
    for own, other, _ in contrasts.get(term, []):
        concept["boundary_positives"] = concept.get("boundary_positives", []) + own
        concept["boundary_negatives"] = concept.get("boundary_negatives", []) + other
    return create_content_dataset(concept, all_concepts, descendant_cap=cap, rng=rng,
                                  diagonal_cap=diagonal_cap, diagonal_extra=diagonal_extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packs", nargs="+", required=True, help="Concept pack ids (same skeleton, same holdout)")
    parser.add_argument("--samples", nargs="+", type=int, default=[20, 40, 80, 160], help="Samples per class")
    parser.add_argument("--model", required=True)
    parser.add_argument("--skeleton", type=Path, default=Path("results/ontology_skeleton_v2.json"))
    parser.add_argument("--links-pack", default="university-exp-d1",
                        help="Pack whose siblings + cross-links define the 'hard' negatives for every config")
    parser.add_argument("--per-field", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="Train only the first N subset concepts (smoke test)")
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--held-out-per-concept", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bands-from", type=Path, default=None,
                        help="Reuse subset and band layers from an earlier run's output, so runs stay comparable")
    parser.add_argument("--budget", choices=["fixed", "content"], default="content")
    parser.add_argument("--contrasts", type=Path, default=None,
                        help="Contrast JSONL from generate_contrasts.py (content budget only)")
    parser.add_argument("--descendant-cap", type=int, default=100,
                        help="Most descendant definitions used as positives (content budget)")
    parser.add_argument("--diagonal-cap", type=int, default=0,
                        help="Content budget: negatives first from up to N texts of the concept's hard neighbours")
    parser.add_argument("--diagonal-extra", action="store_true",
                        help="Add the hard-neighbour texts on top of a full balancing fill instead of replacing it")
    parser.add_argument("--label", default="", help="Suffix for config names, e.g. '+contrasts'")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    hierarchies = {p: PROJECT_ROOT / "concept_packs" / p / "hierarchy" for p in args.packs + [args.links_pack]}
    links = load_layer_concepts(LAYER, hierarchies[args.links_pack])[1]
    subset = pick_subset(links.values(), args.per_field, args.seed)[: args.limit]
    earlier = json.loads(args.bands_from.read_text()) if args.bands_from else None
    if earlier:
        subset = earlier["subset"][: args.limit]
    print(f"Subset: {len(subset)} Universities: {subset}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda", local_files_only=True).eval()
    n_layers = getattr(model.config, "text_config", model.config).num_hidden_layers

    # Held-out text: extracted once, every layer kept on CPU
    skeleton = json.loads(args.skeleton.read_text())
    texts, owners = held_out_texts(skeleton, links, args.holdout_fraction, args.held_out_per_concept, args.seed)
    print(f"Held-out Department descriptions: {len(texts)} under {len(set(owners))} Universities")
    t = time.time()
    H = extract_activations(model, tokenizer, texts, "cuda", extraction_mode="prompt", layer_idx=None)
    H = torch.tensor(H, dtype=torch.float32).reshape(len(texts), n_layers, -1)
    print(f"  extracted in {time.time() - t:.0f}s")

    # Band layers, once per concept, from its MELD-based dataset in the links pack
    all_links = load_all_concepts(hierarchies[args.links_pack])
    bands = {t: earlier["bands"][t] for t in subset} if earlier else {}
    for term in [t for t in subset if t not in bands]:
        concept = links[term]
        prompts, labels = create_sumo_training_dataset(
            concept=concept, all_concepts=links, negative_pool=build_sumo_negative_pool(all_links, concept),
            n_positives=20, n_negatives=20, use_category_relationships=True, use_wordnet_relationships=True)
        pos = [p for p, l in zip(prompts, labels) if l == 1]
        neg = [p for p, l in zip(prompts, labels) if l == 0]
        bands[term], _ = select_layers_for_concept(model, tokenizer, pos, neg, device="cuda", n_model_layers=n_layers)

    def masks(term):
        c = links[term]
        hard = {t for t, x in links.items() if x["parent_concepts"] == c["parent_concepts"] and t != term}
        hard |= set(c.get("related_concepts", [])) | set(c.get("equivalent_concepts", []))
        pos = owners == term
        return pos, ~pos, np.isin(owners, sorted(hard))

    contrasts = load_contrasts(args.contrasts)
    rng = random.Random(args.seed)
    results = {"subset": subset, "bands": bands, "n_held_out": len(texts), "configs": []}
    for pack in args.packs:
        concepts = load_layer_concepts(LAYER, hierarchies[pack])[1]
        all_concepts = load_all_concepts(hierarchies[pack])
        for n in (args.samples if args.budget == "fixed" else [None]):
            start = time.time()
            per_concept, sizes = {}, []
            for term in subset:
                concept = concepts[term]
                if args.budget == "content":
                    prompts, labels = content_dataset(term, all_concepts, contrasts, args.descendant_cap, rng,
                                                      diagonal_cap=args.diagonal_cap,
                                                      diagonal_extra=args.diagonal_extra)
                else:
                    prompts, labels = create_sumo_training_dataset(
                        concept=concept, all_concepts=concepts,
                        negative_pool=build_sumo_negative_pool(all_concepts, concept),
                        n_positives=n, n_negatives=n, use_category_relationships=True, use_wordnet_relationships=True)
                sizes.append(sum(labels))
                X = extract_activations(model, tokenizer, prompts, "cuda", layer_idx=bands[term])
                y = np.array(labels)
                if X.shape[0] == 2 * len(y):  # combined extraction: prompt + generation per input
                    y = np.repeat(y, 2)
                X = X.reshape(X.shape[0], len(bands[term]), -1)
                split = np.random.permutation(len(y))
                cut = int(0.8 * len(y))
                tr, va = split[:cut], split[cut:]

                scores = []
                for b, layer in enumerate(bands[term]):
                    clf, _ = train_simple_classifier(X[tr, b], y[tr], X[va, b], y[va], dtype=torch.float32)
                    clf.eval()
                    with torch.no_grad():  # the classifier ends in a sigmoid: outputs are probabilities
                        dev = next(clf.parameters()).device
                        probs = clf(normalize(H[:, layer, :]).to(dev)).squeeze(-1)
                        scores.append(probs.float().cpu().numpy())
                scores = np.stack(scores)

                pos, neg_all, neg_hard = masks(term)
                row = {}
                for name, s in [("mean", scores.mean(0)), ("max", scores.max(0))] + \
                        [(f"band{b}", scores[b]) for b in range(len(scores))]:
                    for kind, neg in (("all", neg_all), ("hard", neg_hard)):
                        if pos.any() and neg.any():
                            row[f"{name}_{kind}"] = float(roc_auc_score(
                                np.r_[np.ones(pos.sum()), np.zeros(neg.sum())], np.r_[s[pos], s[neg]]))
                per_concept[term] = row

            keys = sorted({k for r in per_concept.values() for k in r})
            summary = {k: float(np.mean([r[k] for r in per_concept.values() if k in r])) for k in keys}
            n = n if n is not None else round(float(np.mean(sizes)))
            config = {"pack": pack + args.label, "budget": args.budget, "samples": n, "seconds": time.time() - start,
                      "summary": summary, "per_concept": per_concept}
            results["configs"].append(config)
            print(f"{pack + args.label:28s} n={n:4d}  AUROC mean-all {summary.get('mean_all', 0):.3f}  "
                  f"mean-hard {summary.get('mean_hard', 0):.3f}  max-hard {summary.get('max_hard', 0):.3f}  "
                  f"[{config['seconds']:.0f}s]", flush=True)
            args.output.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
