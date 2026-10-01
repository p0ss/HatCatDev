#!/usr/bin/env python3
"""
Held-out confusion check for a band-probe lens pack built from an ontology skeleton.

The held-out text is the level below everything the concept pack contains, so
the probes never saw it: a descendant's text should make its own ancestor's
lens win among all lenses at that level (e.g. a Department's topic description
should pick out its University among all Universities, and its Field among all
Fields).

- Packs with definition-only children (built with node paths): held-out text is
  the cleaned topic description of nodes two skeleton levels below the pack's
  deepest layer, sampled up to --max-per-concept per trained concept.
- Older packs without them: held-out text is the MELD positive examples of the
  children of the deepest trained level.
- Packs built with a held-out split (--holdout-level): held-out text is the
  split's nodes, so packs of any depth - and older lens packs scored against
  such a concept pack - are compared on the same unseen text.

Besides ranking every lens against the others (which mixes discrimination with
how hot each lens runs), each lens gets its own AUROC: its held-out texts
against everyone else's ("all"), and against its siblings' and cross-linked
concepts' only ("hard").

Each lens combines its band probes, each reading its own model layer
(per-sample normalised, as in training), by max and by mean.

Usage:
    python scripts/eval_lens_confusion.py \\
        --lens-pack lens_packs/gemma-4-e4b-it_university-l2-bands \\
        --concept-pack university-l2 \\
        --skeleton results/ontology_skeleton_v2.json \\
        --melds results/context_aware_melds \\
        --model google/gemma-4-E4B-it
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import roc_auc_score

from scripts.build_skeleton_concept_pack import clean_scope, is_held_out, term_for
from src.hat.classifiers.classifier import load_classifier
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.map.training.sibling_ranking import find_probe_paths
from src.map.training.sumo_classifiers import extract_activations


def collect_meld_examples(skeleton, melds_dir, layers, deepest, held_out, per_child):
    """Older packs: MELD examples of the children of the deepest trained level
    (hierarchy layer L is skeleton level L + 1, so those are level deepest + 2)."""
    meld_by_id = {}
    for path in (melds_dir / f"L{deepest + 2}").glob("*.json"):
        data = json.loads(path.read_text())
        meld_by_id[data["node"]["id"]] = data["meld_data"]

    def collect(node, parent_term=None):
        term = term_for(node["id"])
        if node["level"] == deepest + 2 and parent_term in layers[deepest]:
            meld = meld_by_id.get(node["id"])
            if meld:
                held_out[parent_term] += meld.get("positive_examples", [])[:per_child]
            return
        for child in node.get("children", []):
            collect(child, term)

    for root in skeleton["roots"]:
        collect(root)


def collect_descriptions(skeleton, hierarchy, deepest_concepts, held_out, max_per_concept):
    """Packs with definition-only levels: topic descriptions of skeleton nodes one
    level below everything in the pack, attributed to their trained ancestor."""
    term_by_path, pack_depth = {}, 0
    for path in hierarchy.glob("layer*.json"):
        data = json.loads(path.read_text())
        pack_depth = max(pack_depth, data["layer"])
        for c in data["concepts"]:
            term_by_path[c["node_path"]] = c["sumo_term"]
    held_level = pack_depth + 2  # skeleton levels are hierarchy layers + 1
    candidates = defaultdict(list)

    def collect(node, parent=None, owner=None):
        path = f"{parent}/{node['id']}" if parent else node["id"]
        if term_by_path.get(path) in deepest_concepts:
            owner = term_by_path[path]
        if node["level"] == held_level:
            if owner and node.get("scope"):
                candidates[owner].append(clean_scope(node["scope"]))
            return
        for child in node.get("children", []):
            collect(child, path, owner)

    for root in skeleton["roots"]:
        collect(root)
    rng = random.Random(0)
    for owner, texts in candidates.items():
        held_out[owner] += rng.sample(texts, min(max_per_concept, len(texts)))


def collect_split(skeleton, deepest_concepts, level, fraction, held_out, max_per_concept, half="all"):
    """Nodes held out of the pack by the builder's split, attributed to their trained ancestor."""
    owner_of_path = {c["node_path"]: t for t, c in deepest_concepts.items()}
    candidates = defaultdict(list)

    def collect(node, parent=None, owner=None):
        path = f"{parent}/{node['id']}" if parent else node["id"]
        owner = owner_of_path.get(path, owner)
        if node["level"] == level:
            if owner and node.get("scope") and is_held_out(path, fraction):
                candidates[owner].append(clean_scope(node["scope"]))
            return
        for child in node.get("children", []):
            collect(child, path, owner)

    for root in skeleton["roots"]:
        collect(root)
    rng = random.Random(0)
    for owner, texts in candidates.items():
        texts = rng.sample(texts, min(max_per_concept, len(texts)))
        if half != "all":  # even half calibrates probes (calibrate_band_probes.py), odd half tests
            texts = texts[1::2] if half == "test" else texts[0::2]
        held_out[owner] += texts


def per_lens_auroc(level, M, names, prompts, parent_of, layers) -> dict:
    """Each lens's own AUROC: its held-out texts vs everyone else's, and vs its hard neighbours'."""
    targets = []
    for term, _ in prompts:
        while term in parent_of and term not in names:
            term = parent_of[term]
        targets.append(term)
    targets = np.array(targets)
    scores = {"all": [], "hard": []}
    for i, name in enumerate(names):
        concept = layers[level][name]
        pos = targets == name
        if not pos.any():
            continue
        hard = {t for t, c in layers[level].items()
                if t != name and c["parent_concepts"] == concept["parent_concepts"]}
        hard |= set(concept.get("related_concepts", [])) | set(concept.get("equivalent_concepts", []))
        for kind, neg in (("all", ~pos), ("hard", np.isin(targets, sorted(hard)))):
            if neg.any():
                y = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
                scores[kind].append(roc_auc_score(y, np.r_[M[i, pos], M[i, neg]]))
    out = {f"auroc_{k}": float(np.mean(v)) for k, v in scores.items() if v}
    print(f"  per-lens AUROC: all {out.get('auroc_all', 0):.3f}, hard {out.get('auroc_hard', 0):.3f} "
          f"({len(scores['all'])} lenses)")
    return out


def report(level, mode, M, names, prompts, parent_of, layers) -> dict:
    """Rank each held-out prompt's target lens among the lenses at this level."""
    correct, ranks, confusions = 0, [], defaultdict(int)
    for i, (term, _) in enumerate(prompts):
        target = term
        while target in parent_of and target not in names:
            target = parent_of[target]
        if target not in names:
            continue
        order = np.argsort(-M[:, i])
        rank = int(np.where(np.array(names)[order] == target)[0][0]) + 1
        ranks.append(rank)
        correct += rank == 1
        if rank > 1:
            confusions[(target, names[order[0]])] += 1

    n = len(ranks)
    print(f"\nLevel {level} {mode} ({len(names)} lenses): top-1 {correct / n:.1%}, "
          f"top-3 {np.mean(np.array(ranks) <= 3):.1%}, median rank {np.median(ranks):.0f}  (n={n})")
    for (target, winner), count in sorted(confusions.items(), key=lambda x: -x[1])[:10]:
        concept = layers[level][target]
        if parent_of.get(target) == parent_of.get(winner):
            tag = "sibling"
        elif winner in concept.get("related_concepts", []) + concept.get("equivalent_concepts", []):
            tag = "cross-linked"
        else:
            tag = "cousin"
        print(f"  {count:3d}x  {target} -> {winner}  [{tag}]")

    return {"top1": correct / n, "top3": float(np.mean(np.array(ranks) <= 3)),
            "median_rank": float(np.median(ranks)), "n": n,
            "confusions": [[t, w, c] for (t, w), c in confusions.items()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lens-pack", type=Path, required=True)
    parser.add_argument("--concept-pack", required=True)
    parser.add_argument("--skeleton", type=Path, required=True)
    parser.add_argument("--melds", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--examples-per-child", type=int, default=3)
    parser.add_argument("--max-per-concept", type=int, default=6,
                        help="Held-out descriptions sampled per trained concept (node-path packs)")
    parser.add_argument("--held-out", choices=["auto", "split", "descendants", "meld"], default="auto",
                        help="auto: the pack's held-out split if it has one, else descendants or MELD examples")
    parser.add_argument("--probe-calibration", type=Path, default=None,
                        help="Per-probe calibration to score calibrated_max with (default <lens-pack>/probe_calibration.json). "
                             "Use one calibrated in the same unit as the held-out text: pooled.")
    parser.add_argument("--split-half", choices=["all", "test"], default="all",
                        help="test: only the odd half of the held-out split, leaving the even half to calibration")
    parser.add_argument("--output", type=Path, default=None, help="Write per-concept results as JSON")
    args = parser.parse_args()

    hierarchy = PROJECT_ROOT / "concept_packs" / args.concept_pack / "hierarchy"
    layers = {}
    for path in sorted(hierarchy.glob("layer*.json")):
        data = json.loads(path.read_text())
        layers[data["layer"]] = {c["sumo_term"]: c for c in data["concepts"]}
    # Layers with lenses; definition-only layers only describe their parents
    layers = {l: c for l, c in layers.items() if not all(x.get("definition_only") for x in c.values())}
    deepest = max(layers)
    parent_of = {t: c["parent_concepts"][0] for l in layers.values() for t, c in l.items() if c["parent_concepts"]}

    skeleton = json.loads(args.skeleton.read_text())
    held_out = defaultdict(list)  # deepest-level term -> prompts
    builder = json.loads((hierarchy.parent / "pack.json").read_text())["ontology_stack"]["hierarchy_builder"]
    mode = args.held_out
    if mode == "auto":
        mode = "split" if builder.get("holdout") else (
            "descendants" if any("node_path" in c for c in layers[0].values()) else "meld")
    if mode == "split":
        split = builder["holdout"]
        collect_split(skeleton, layers[deepest], split["level"], split["fraction"], held_out, args.max_per_concept,
                      half=args.split_half)
    elif mode == "descendants":
        collect_descriptions(skeleton, hierarchy, layers[deepest], held_out, args.max_per_concept)
    else:
        collect_meld_examples(skeleton, args.melds, layers, deepest, held_out, args.examples_per_child)
    print(f"Held-out prompts for {len(held_out)} of {len(layers[deepest])} level-{deepest} concepts "
          f"({sum(map(len, held_out.values()))} prompts)")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda", local_files_only=True)
    model.eval()
    n_model_layers = getattr(model.config, "text_config", model.config).num_hidden_layers

    # Load every lens as {model_layer: probe}
    lenses = {}
    for layer, concepts in layers.items():
        for term, concept in concepts.items():
            probes = find_probe_paths(args.lens_pack / f"layer{layer}", term)
            if not probes and "node_id" in concept:
                # Lens packs trained before names were cleaned use the node id's term
                probes = find_probe_paths(args.lens_pack / f"layer{layer}", term_for(concept["node_id"]))
            if probes:
                lenses[(term, layer)] = {l: load_classifier(p, device="cuda", classifier_type="mlp").eval()
                                         for l, p in probes.items()}

    prompts = [(term, p) for term, ps in held_out.items() for p in ps]
    X = extract_activations(model, tokenizer, [p for _, p in prompts], "cuda",
                            extraction_mode="prompt", layer_idx=None)
    X = torch.tensor(X, dtype=torch.float32).reshape(len(prompts), n_model_layers, -1).cuda()
    X = (X - X.mean(-1, keepdim=True)) / (X.std(-1, keepdim=True) + 1e-8)

    with torch.no_grad():
        band_scores = {key: torch.stack([torch.sigmoid(probe(X[:, l, :]).squeeze(-1)) for l, probe in probes.items()])
                       .cpu().numpy()
                       for key, probes in lenses.items()}  # [n_bands, n_prompts]

    # Per-probe calibration (calibrate_band_probes.py): raw score -> fraction of background exceeded
    calibrated = {}
    cal_path = args.probe_calibration or args.lens_pack / "probe_calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text())["probes"]
        for key, probes in lenses.items():
            term, layer = key
            rows = []
            for i, model_layer in enumerate(probes):  # same order as band_scores
                curve = cal.get(f"layer{layer}/{term}@L{model_layer}")
                if curve is None:
                    break
                q = np.array(curve["quantiles"])
                rows.append(np.interp(band_scores[key][i], q, np.linspace(0, 1, len(q))))
            else:
                calibrated[key] = np.stack(rows)
        print(f"Calibrated probes for {len(calibrated)} of {len(lenses)} lenses")

    results = {}
    for level in sorted(layers):
        keys = [k for k in lenses if k[1] == level]
        names = [k[0] for k in keys]
        modes = ["max", "mean"] + (["calibrated_max"] if all(k in calibrated for k in keys) else [])
        for mode in modes:
            if mode == "calibrated_max":
                M = np.stack([calibrated[k].max(axis=0) for k in keys])
            else:
                M = np.stack([getattr(band_scores[k], mode)(axis=0) for k in keys])  # [n_lenses, n_prompts]
            results[f"{level}_{mode}"] = report(level, mode, M, names, prompts, parent_of, layers)
            results[f"{level}_{mode}"].update(per_lens_auroc(level, M, names, prompts, parent_of, layers))

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
