#!/usr/bin/env python3
"""
Build a concept pack from an ontology skeleton and its context-aware MELDs.

Takes the output of generate_ontology_skeleton.py (the "university builder")
and generate_melds_with_context.py and writes a concept pack the lens trainer
can read, cut off at a chosen depth. Skeleton level 1 becomes hierarchy
layer 0, level 2 becomes layer 1, and so on.

Names and descriptions are about concepts, not institutions: "School of X"
becomes "X", and each node's scope ("This university investigates ...") becomes
a topic description of what its author expected it to cover, extended with the
names of its children. A concept is defined by what it contains, so the pack
also carries the level below the trained levels as definition-only nodes: they
describe their parents in training but get no lenses of their own.

A skeleton is a tree, so concepts that mean nearly the same thing can end up
in different branches with nothing linking them. The cross-linking pass embeds
every concept and records its close neighbours in other branches:
`related_concepts` become explicit hard negatives, so the probes learn to tell
them apart; `equivalent_concepts` are near-duplicates, kept out of each other's
negatives so neither probe is trained to reject the other.

The definition-only levels are cross-linked too, and their overlaps become
their trained ancestors' disambiguation work:
- a child nearly identical to one in another branch (the same class taught in
  two faculties) is a shared topic: positive for both owners, never a negative
  for either owner or their ancestors;
- a similar-but-distinct child is a diagonal hard negative for the other owner;
- owners with several such links between their children become related.
Children that link to many branches are generic (e.g. "Research Methods") and
don't count as evidence.

Usage:
    python scripts/build_skeleton_concept_pack.py \\
        --skeleton results/ontology_skeleton_v2.json \\
        --melds results/context_aware_melds \\
        --max-level 2 \\
        --pack-id university-l2-v2
"""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def term_for(text: str) -> str:
    """`Nonprofit Management` or `nonprofit-management` -> `NonprofitManagement`: filename-safe."""
    return "".join(part[:1].upper() + part[1:] for part in re.split(r"[^A-Za-z0-9]+", text) if part)


# Words that name the building rather than the concept
INSTITUTION = (r"(?:school|university|college|institute|institution|faculty|department|center|centre|"
               r"laboratory|lab|academy|atelier|program|programme|field|track)")
FRAMING_VERBS = (
    r"(?:(?:primarily|specifically|broadly)\s+)?"
    r"(?:focuses on|explores|examines|investigates|studies|analy[sz]es|speciali[sz]es in|delves into|covers|"
    r"is dedicated to|dedicates itself to|cent(?:er|re)s on|concentrates on|is devoted to|deals with|encompasses|"
    r"teaches(?:\s+students)?(?:\s+how)?(?:\s+to)?|trains\s+(?:[\w-]+\s+){0,3}?(?:in|to)|"
    r"provides (?:training|instruction|education) (?:in|on))\s+"
)
STUDENTS = (r"^students\s+(?:will\s+)?(?:learn|gain|develop|explore|study|master|acquire|examine|analy[sz]e|"
            r"investigate|engage with|delve into|practice)(?:\s+how)?(?:\s+to|\s+about|\s+in)?\s+")


def is_held_out(path: str, fraction: float) -> bool:
    """Deterministic split by node path, identical across every pack built from a skeleton."""
    return fraction > 0 and int(hashlib.md5(path.encode()).hexdigest(), 16) % 1000 < fraction * 1000


def clean_label(label: str) -> str:
    """'School of Military Intelligence' -> 'Military Intelligence'; 'for Visual Arts' -> 'Visual Arts'."""
    s = label.strip()
    s = re.sub(r"^(?:for|the)\s+", "", s, flags=re.I)
    s = re.sub(rf"^{INSTITUTION}s?\s+(?:of|for|in|on)\s+(?:the\s+)?", "", s, flags=re.I)
    s = re.sub(rf"\s+(?:research\s+)?{INSTITUTION}s?\s+(?=(?:of|for)\b)", " ", s, flags=re.I)
    s = re.sub(rf"\s+(?:research\s+)?{INSTITUTION}s?$", "", s, flags=re.I)
    return s.strip(" ,&-")


def clean_scope(scope: str) -> str:
    """'This university investigates the causes of X' -> 'The causes of X'."""
    s = re.sub(rf"^(?:this|the)\s+(?:[\w-]+\s+)?{INSTITUTION}\s+", "", scope.strip(), flags=re.I)
    s = re.sub(rf"^{FRAMING_VERBS}", "", s, flags=re.I)
    s = re.sub(r"^and\s+", "", s, flags=re.I)
    s = re.sub(STUDENTS, "", s, flags=re.I)
    return s[:1].upper() + s[1:]


def needs_name_review(original: str, cleaned: str) -> str | None:
    """Names a regex can't turn into a concept: metaphors and discipline names."""
    if original.lower().startswith("the ") or len(cleaned.split()) == 1:
        return "metaphor or single word"
    if re.search(r"\b(studies|research)\b", cleaned, re.I):
        return "names the discipline, not the concept"
    return None


def load_melds(melds_dir: Path, max_level: int) -> dict:
    """MELDs keyed by (level, node id): ids are label slugs and repeat across levels."""
    melds = {}
    for level in range(1, max_level + 1):
        if not (melds_dir / f"L{level}").is_dir():
            continue
        for path in sorted((melds_dir / f"L{level}").glob("*.json")):
            data = json.loads(path.read_text())
            if data.get("review", {}).get("passed", True):
                melds[(level, data["node"]["id"])] = data["meld_data"]
    return melds


def cross_link(layers: dict, model_name: str, related_threshold: float, equivalent_threshold: float) -> dict:
    """Link each concept to similar concepts under a different parent in the same layer."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    stats = {"related_pairs": 0, "equivalent_pairs": 0}
    for concepts in layers.values():
        texts = [f"{c['label']}: {c['topic_description']}" for c in concepts]
        emb = model.encode(texts, normalize_embeddings=True)
        sim = emb @ emb.T
        for i, a in enumerate(concepts):
            a["related_concepts"], a["equivalent_concepts"] = [], []
            for j in np.argsort(-sim[i]):
                b = concepts[j]
                if j == i or sim[i, j] < related_threshold:
                    continue
                if set(a["parent_concepts"]) & set(b["parent_concepts"]) or not a["parent_concepts"]:
                    continue  # siblings (or roots) are already contrasted by the tree
                key = "equivalent_concepts" if sim[i, j] >= equivalent_threshold else "related_concepts"
                a[key].append(b["sumo_term"])
                if i < j:
                    stats[key.replace("_concepts", "_pairs")] += 1
    return stats


def link_children(layers, trained_layers, definition_layers, model_name, related_threshold,
                   shared_threshold, hub_limit, min_child_links, owner_top_k) -> dict:
    """Turn overlaps between definition-only descendants into work for their trained owners."""
    from collections import Counter, defaultdict

    import numpy as np
    from sentence_transformers import SentenceTransformer

    records = {c["sumo_term"]: c for concepts in layers.values() for c in concepts}
    deepest_trained = max(trained_layers)

    def owner(term):
        """The descendant's ancestor at the deepest trained layer."""
        c = records[term]
        while c["layer"] > deepest_trained:
            c = records[c["parent_concepts"][0]]
        return c["sumo_term"]

    def ancestors(term):
        c, out = records[term], []
        while c["parent_concepts"]:
            c = records[c["parent_concepts"][0]]
            out.append(c["sumo_term"])
        return out

    children = [c for l in definition_layers for c in layers[l]]
    owners = np.array([owner(c["sumo_term"]) for c in children])
    model = SentenceTransformer(model_name, device="cpu")
    emb = model.encode([f"{c['label']}: {c['definition']}" for c in children],
                       normalize_embeddings=True, batch_size=256)

    pairs = []
    for start in range(0, len(children), 2048):
        sim = emb[start:start + 2048] @ emb.T
        for i, j in zip(*np.nonzero(sim >= related_threshold)):
            i += start
            if i < j and owners[i] != owners[j]:
                pairs.append((i, int(j), float(sim[i - start, j])))

    degree = np.zeros(len(children), int)
    for i, j, _ in pairs:
        degree[i] += 1
        degree[j] += 1

    stats = {"child_pairs": len(pairs), "shared": 0, "diagonal": 0, "hub_children": int((degree > hub_limit).sum()),
             "owner_links": 0}
    evidence = {}
    for c in (records[o] for o in set(owners)):
        c.setdefault("shared_topics", [])
        c.setdefault("diagonal_negatives", [])
    for i, j, sim in pairs:
        a, b = children[i], children[j]
        oa, ob = owners[i], owners[j]
        if sim >= shared_threshold:
            # One class taught in two places: each owner (and its ancestors) claims it
            for child, other_owner in ((a, ob), (b, oa)):
                for t in [other_owner] + ancestors(other_owner):
                    records[t].setdefault("shared_topics", []).append(child["sumo_term"])
                    records[t].setdefault("child_definitions", {})[child["sumo_term"]] = child["definition"]
            stats["shared"] += 1
        else:
            records[ob]["diagonal_negatives"].append(a["sumo_term"])
            records[oa]["diagonal_negatives"].append(b["sumo_term"])
            stats["diagonal"] += 1
        if degree[i] <= hub_limit and degree[j] <= hub_limit:
            key = tuple(sorted((oa, ob)))
            evidence[key] = evidence.get(key, 0) + 1

    # Deeper levels produce far more child links, so a fixed count stops meaning
    # anything: score each owner pair by links per sqrt(children x children) and
    # keep each owner's strongest few
    n_children = Counter(owners)
    candidates = defaultdict(list)
    for (oa, ob), n in evidence.items():
        a, b = records[oa], records[ob]
        if n < min_child_links or set(a["parent_concepts"]) & set(b["parent_concepts"]):
            continue  # too little evidence, or siblings (already contrasted)
        score = n / np.sqrt(n_children[oa] * n_children[ob])
        candidates[oa].append((score, ob))
        candidates[ob].append((score, oa))
    keep = {(o, other) for o, cands in candidates.items() for _, other in sorted(cands, reverse=True)[:owner_top_k]}
    for oa, ob in keep:
        if (ob, oa) not in keep or oa > ob:
            continue  # both owners must rank the pair in their top k; count each pair once
        for x, y in ((records[oa], records[ob]), (records[ob], records[oa])):
            if y["sumo_term"] not in x.get("related_concepts", []) + x.get("equivalent_concepts", []):
                x.setdefault("related_concepts", []).append(y["sumo_term"])
        stats["owner_links"] += 1
    for c in records.values():
        for key in ("shared_topics", "diagonal_negatives"):
            if key in c:
                c[key] = sorted(set(c[key]))
    return stats


def attach_contrasts(layers, paths) -> int:
    """Boundary examples from "differs because" pairs: own side positive, other side negative."""
    records = {c["sumo_term"]: c for concepts in layers.values() for c in concepts}
    n = 0
    for path in paths:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            a, b = records.get(r["a"]), records.get(r["b"])
            if a is None or b is None:
                continue
            for own, other, mine, theirs in ((a, b, "a_not_b", "b_not_a"), (b, a, "b_not_a", "a_not_b")):
                own.setdefault("boundary_positives", []).extend(r[mine])
                own.setdefault("boundary_negatives", []).extend(r[theirs])
                own.setdefault("distinctions", {})[other["sumo_term"]] = r["distinction"]
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skeleton", type=Path, required=True)
    parser.add_argument("--melds", type=Path, required=True, help="Directory with L1/, L2/, ... MELD files")
    parser.add_argument("--max-level", type=int, default=2, help="Deepest skeleton level to train")
    parser.add_argument("--definition-levels", type=int, default=1,
                        help="Levels below --max-level to include as definition-only children (default 1)")
    parser.add_argument("--pack-id", required=True)
    parser.add_argument("--description", default=None)
    parser.add_argument("--no-cross-link", action="store_true", help="Skip the cross-branch linking pass")
    parser.add_argument("--contrasts", type=Path, action="append", default=[],
                        help="'Differs because' JSONL from generate_contrasts.py; each pair's boundary "
                             "examples become its concepts' boundary positives and negatives (repeatable)")
    parser.add_argument("--no-child-links", action="store_true",
                        help="Link trained concepts only; don't derive shared/diagonal/owner links from children")
    parser.add_argument("--holdout-level", type=int, default=None,
                        help="Skeleton level to split for held-out evaluation (default: none)")
    parser.add_argument("--holdout-fraction", type=float, default=0.3,
                        help="Fraction of --holdout-level nodes (and their subtrees) kept out of the pack")
    parser.add_argument("--link-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--related-threshold", type=float, default=0.6,
                        help="Similarity above which cross-branch concepts are contrasted (default 0.6)")
    parser.add_argument("--equivalent-threshold", type=float, default=0.85,
                        help="Similarity above which they are treated as near-duplicates (default 0.85)")
    parser.add_argument("--child-related-threshold", type=float, default=0.7,
                        help="Similarity above which definition-only children in different branches link (default 0.7)")
    parser.add_argument("--child-shared-threshold", type=float, default=0.85,
                        help="Similarity above which linked children are one shared topic (default 0.85)")
    parser.add_argument("--child-hub-limit", type=int, default=20,
                        help="Children with more cross-branch links than this are generic, not evidence (default 20)")
    parser.add_argument("--min-child-links", type=int, default=3,
                        help="Child links needed before their owners become related (default 3)")
    parser.add_argument("--owner-top-k", type=int, default=5,
                        help="Most related owners each concept keeps from child evidence (default 5)")
    args = parser.parse_args()

    skeleton = json.loads(args.skeleton.read_text())
    deepest = args.max_level + args.definition_levels
    melds = load_melds(args.melds, deepest)

    layers: dict[int, list] = {}
    skipped, name_review = [], []

    # Assign terms level by level, so shallower (trained) concepts get the plain
    # name when a deeper class shares it; parentheticals stay in the label only
    # Nodes are identified by their path: ids are label slugs, so the same id can
    # appear at several levels, and occasionally twice under one parent
    by_level, parent_path, duplicates = {}, {}, []

    def held_out(node, path):
        return node["level"] == args.holdout_level and is_held_out(path, args.holdout_fraction)

    def collect(node, parent=None):
        path = f"{parent}/{node['id']}" if parent else node["id"]
        if held_out(node, path):
            return
        if path in parent_path:
            duplicates.append(path)  # the generator repeated a child; keep the first
            return
        if node["level"] <= deepest:
            by_level.setdefault(node["level"], []).append((path, node))
            parent_path[path] = parent
            for child in node.get("children", []):
                collect(child, path)

    for root in skeleton["roots"]:
        collect(root)
    term_of, owner = {}, {}
    for level in sorted(by_level):
        for path, node in by_level[level]:
            term = term_for(re.sub(r"\s*\([^)]*\)", "", clean_label(node["label"])))
            if term in owner:
                term = f"{term}In{term_of[parent_path[path]]}"
            base, n = term, 2
            while term in owner:  # siblings whose names clean to the same thing
                term, n = f"{base}{n}", n + 1
            term_of[path], owner[term] = term, path

    visited = set()

    def visit(node, parent=None, parent_term=None):
        level = node["level"]
        path = f"{parent}/{node['id']}" if parent else node["id"]
        if level > deepest or path in visited or held_out(node, path):
            return
        visited.add(path)
        trained = level <= args.max_level
        meld = melds.get((level, node["id"]))
        if trained and meld is None:
            skipped.append(node["id"])
            return

        label = clean_label(node["label"])
        term = term_of[path]
        reason = needs_name_review(node["label"], label)
        if reason:
            name_review.append({"term": term, "label": node["label"], "cleaned": label, "level": level, "reason": reason})

        children = [c for c in node.get("children", []) if c["level"] <= deepest
                    and (c["level"] > args.max_level or (c["level"], c["id"]) in melds)]
        scope = clean_scope(node.get("scope", ""))
        record = {
            "sumo_term": term,
            "label": label,
            "node_id": node["id"],
            "node_path": path,
            "layer": level - 1,
            "definition": (meld or {}).get("definition") or scope,
            "topic_description": scope,
            "parent_concepts": [parent_term] if parent_term else [],
            "category_children": [],
            "child_concepts": [],
            "is_category_lens": False,
            "is_pseudo_sumo": True,
            "definition_only": not trained,
            "synsets": [],
            "synset_count": 0,
            "sumo_depth": level - 1,
        }
        if trained:
            record.update({
                "positive_examples": meld.get("positive_examples", []),
                "negative_examples": meld.get("negative_examples", []),
                "contrast_concepts": meld.get("contrast_concepts", []),
                "training_hints": meld.get("training_hints", {}),
                "safety_tags": meld.get("safety_tags", {}),
            })
        layers.setdefault(level - 1, []).append(record)

        child_terms = [visit(child, path, term) for child in children]
        child_terms = [t for t in child_terms if t]
        record["category_children"] = record["child_concepts"] = child_terms
        record["is_category_lens"] = bool(child_terms)
        if child_terms:
            children_records = [c for c in layers[level] if c["sumo_term"] in child_terms]
            record["topic_description"] = f"{scope} Covers: {'; '.join(c['label'] for c in children_records)}."
            # What the concept contains, as content the trainer can use for positives
            # (the trainer only sees one layer at a time, so it can't look children up).
            # Descendants' definitions are inherited, so a Field draws on its Schools too.
            record["child_definitions"] = {c["sumo_term"]: c["definition"] for c in children_records}
            for c in children_records:
                record["child_definitions"].update(c.get("child_definitions", {}))
        return term

    for root in skeleton["roots"]:
        visit(root)

    trained_layers = sorted(l for l in layers if l < args.max_level)
    definition_layers = sorted(l for l in layers if l >= args.max_level)

    link_stats = None
    if not args.no_cross_link:
        link_stats = cross_link({l: layers[l] for l in trained_layers}, args.link_model,
                                args.related_threshold, args.equivalent_threshold)
        if definition_layers and not args.no_child_links:
            link_stats.update(link_children(
                layers, trained_layers, definition_layers, args.link_model,
                args.child_related_threshold, args.child_shared_threshold,
                args.child_hub_limit, args.min_child_links, args.owner_top_k,
            ))

    n_contrasts = attach_contrasts(layers, args.contrasts)

    pack_dir = PROJECT_ROOT / "concept_packs" / args.pack_id
    hierarchy_dir = pack_dir / "hierarchy"
    hierarchy_dir.mkdir(parents=True, exist_ok=True)

    # hierarchy.json keys concepts as "Term:layer", like the other concept packs
    layer_of = {c["sumo_term"]: layer for layer, concepts in layers.items() for c in concepts}
    key = lambda term: f"{term}:{layer_of[term]}"
    parent_to_children, child_to_parent, leaves, roots = {}, {}, [], []
    related, equivalent = {}, {}
    for layer, concepts in sorted(layers.items()):
        (hierarchy_dir / f"layer{layer}.json").write_text(
            json.dumps({"layer": layer, "concepts": concepts}, indent=1)
        )
        for c in concepts:
            if c["category_children"]:
                parent_to_children[key(c["sumo_term"])] = [key(t) for t in c["category_children"]]
            else:
                leaves.append(key(c["sumo_term"]))
            if c["parent_concepts"]:
                child_to_parent[key(c["sumo_term"])] = key(c["parent_concepts"][0])
            else:
                roots.append(key(c["sumo_term"]))
            if c.get("related_concepts"):
                related[key(c["sumo_term"])] = [key(t) for t in c["related_concepts"]]
            if c.get("equivalent_concepts"):
                equivalent[key(c["sumo_term"])] = [key(t) for t in c["equivalent_concepts"]]
    (hierarchy_dir / "hierarchy.json").write_text(json.dumps({
        "child_to_parent": child_to_parent,
        "leaf_concepts": leaves,
        "parent_to_children": parent_to_children,
        "related_concepts": related,
        "equivalent_concepts": equivalent,
        "root_concepts": roots,
        "total_concepts": len(layer_of),
        "total_leaves": len(leaves),
        "total_parents": len(parent_to_children),
        "total_roots": len(roots),
    }, indent=1))

    total = sum(len(c) for c in layers.values())
    pack = {
        "pack_id": args.pack_id,
        "spec_id": f"org.hatcat/{args.pack_id}@0.1.0",
        "version": "0.1.0",
        "created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "description": args.description or (
            f"Top {args.max_level} levels of the university-builder ontology "
            f"({skeleton.get('generator_model', 'unknown model')}), with context-aware MELDs "
            f"and {args.definition_levels} definition-only level(s) below."
        ),
        "ontology_stack": {
            "base_ontology": {
                "name": "Generated",
                "source": skeleton.get("generator_model"),
                "note": "Fields -> Universities -> ... from generate_ontology_skeleton.py",
            },
            "hierarchy_builder": {
                "script": "scripts/build_skeleton_concept_pack.py",
                "skeleton": str(args.skeleton),
                "melds": str(args.melds),
                "max_level": args.max_level,
                "definition_levels": args.definition_levels,
                "contrasts": [str(p) for p in args.contrasts],
                "holdout": None if args.holdout_level is None else {
                    "level": args.holdout_level, "fraction": args.holdout_fraction},
                "cross_link": None if link_stats is None else {
                    "model": args.link_model,
                    "related_threshold": args.related_threshold,
                    "equivalent_threshold": args.equivalent_threshold,
                    "child_related_threshold": args.child_related_threshold,
                    "child_shared_threshold": args.child_shared_threshold,
                    "child_hub_limit": args.child_hub_limit,
                    "min_child_links": args.min_child_links,
                    "owner_top_k": args.owner_top_k,
                    **link_stats,
                },
            },
        },
        "concept_metadata": {
            "total_concepts": total,
            "layers": trained_layers,
            "definition_layers": definition_layers,
            "layer_distribution": {str(k): len(v) for k, v in sorted(layers.items())},
            "hierarchy_file": "hierarchy/",
        },
    }
    (pack_dir / "pack.json").write_text(json.dumps(pack, indent=2))

    if name_review:
        (pack_dir / "name_review.json").write_text(json.dumps(name_review, indent=1))

    print(f"Wrote {total} concepts to {pack_dir}")
    for layer, concepts in sorted(layers.items()):
        print(f"  layer{layer}: {len(concepts)}{' (definition only)' if layer in definition_layers else ''}")
    if name_review:
        print(f"  {len(name_review)} names flagged for review in name_review.json")
    if n_contrasts:
        print(f"  Attached {n_contrasts} 'differs because' contrasts as boundary examples")
    if duplicates:
        print(f"  Dropped {len(duplicates)} duplicate skeleton nodes: {duplicates[:5]}")
    if link_stats:
        print(f"  Cross-links: {link_stats['related_pairs']} related pairs, "
              f"{link_stats['equivalent_pairs']} equivalent pairs")
        if "child_pairs" in link_stats:
            print(f"  Child links: {link_stats['child_pairs']} pairs -> {link_stats['shared']} shared topics, "
                  f"{link_stats['diagonal']} diagonal negatives; {link_stats['hub_children']} generic hub children; "
                  f"{link_stats['owner_links']} owner pairs linked from child evidence")
    if skipped:
        print(f"  Skipped {len(skipped)} nodes without a passing MELD: {skipped[:5]}...")


if __name__ == "__main__":
    main()
