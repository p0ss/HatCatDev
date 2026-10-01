#!/usr/bin/env python3
"""
Generate "differs because" contrasts between confusable concepts.

For each pair of concepts a lens must tell apart (siblings, and cross-branch
related concepts), the model writes where the boundary lies: one sentence on
what separates them, and boundary examples on both sides - text that is clearly
A but could be mistaken for B, and the reverse. These are the samples that
define a lens's outer bounds: B-not-A examples are A's hardest negatives, and
A-not-B examples show A where it stops.

Output is one JSON line per pair, resumable (pairs already written are skipped).

Usage:
    python scripts/generate_contrasts.py \\
        --concept-pack university-exp-d1 \\
        --concepts-from results/scaling/university_gemma4_d0_d1.json \\
        --model google/gemma-4-E4B-it \\
        --output results/contrasts/university-exp-d1.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = """You are marking the boundary between two closely related topic areas, so that a classifier reading someone's text can tell which one it is about.

Topic A: {a_label}
{a_desc}

Topic B: {b_label}
{b_desc}

Write:
1. "distinction": one sentence saying what separates A from B.
2. "a_not_b": {n} short passages (one or two sentences each) that are clearly about A, but that someone might mistake for B.
3. "b_not_a": {n} short passages that are clearly about B, but that someone might mistake for A.

Rules for the passages:
- Concrete statements, questions or scenarios someone might actually write, not definitions.
- Each one sits close to the boundary, yet a careful reader could tell which topic it belongs to.
- Never name either topic, and never mention universities, schools, departments, courses, fields or "the study of".
- Vary the situations; don't repeat one template.

Answer with JSON only: {{"distinction": "...", "a_not_b": ["..."], "b_not_a": ["..."]}}"""


def describe(concept) -> str:
    return concept.get("topic_description") or concept.get("definition", "")


def contrast_pairs(concepts, only):
    """Unordered pairs a lens must separate: siblings and cross-branch related concepts."""
    pairs = set()
    for term, c in concepts.items():
        if only and term not in only:
            continue
        partners = {t for t, x in concepts.items() if t != term and x["parent_concepts"] == c["parent_concepts"]}
        partners |= {t for t in c.get("related_concepts", []) if t in concepts}
        pairs |= {tuple(sorted((term, p))) for p in partners}
    return sorted(pairs)


def parse(text):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data.get("a_not_b"), list) or not isinstance(data.get("b_not_a"), list):
        return None
    clean = lambda xs: [x.strip() for x in xs if isinstance(x, str) and len(x.split()) >= 5]
    return {"distinction": str(data.get("distinction", "")).strip(),
            "a_not_b": clean(data["a_not_b"]), "b_not_a": clean(data["b_not_a"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--concept-pack", required=True)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--concepts-from", type=Path, default=None,
                        help="JSON with a 'subset' list: only pairs involving these concepts")
    parser.add_argument("--model", required=True)
    parser.add_argument("--examples", type=int, default=5, help="Boundary examples per side")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--limit", type=int, default=None, help="Generate only the first N pending pairs (for a check)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    hierarchy = PROJECT_ROOT / "concept_packs" / args.concept_pack / "hierarchy"
    concepts = {c["sumo_term"]: c for c in json.loads((hierarchy / f"layer{args.layer}.json").read_text())["concepts"]}
    only = set(json.loads(args.concepts_from.read_text())["subset"]) if args.concepts_from else None
    pairs = contrast_pairs(concepts, only)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            r = json.loads(line)
            done.add((r["a"], r["b"]))
    todo = [p for p in pairs if p not in done][: args.limit]
    print(f"{len(pairs)} contrast pairs, {len(todo)} to generate")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda", local_files_only=True).eval()

    def prompt_for(a, b):
        text = PROMPT.format(a_label=concepts[a]["label"], a_desc=describe(concepts[a]),
                             b_label=concepts[b]["label"], b_desc=describe(concepts[b]), n=args.examples)
        return tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                             tokenize=False, add_generation_prompt=True)

    failed = 0
    with open(args.output, "a") as out:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            for attempt in range(2):
                inputs = tokenizer([prompt_for(a, b) for a, b in batch], return_tensors="pt",
                                   padding=True, add_special_tokens=False).to("cuda")
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                                               temperature=0.7, top_p=0.95, pad_token_id=tokenizer.pad_token_id)
                texts = tokenizer.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                retry = []
                for (a, b), text in zip(batch, texts):
                    result = parse(text)
                    if result and result["a_not_b"] and result["b_not_a"]:
                        out.write(json.dumps({"a": a, "b": b, **result}) + "\n")
                    else:
                        retry.append((a, b))
                out.flush()
                batch = retry
                if not batch:
                    break
            failed += len(batch)
            print(f"  {min(start + args.batch_size, len(todo))}/{len(todo)} pairs ({failed} failed)", flush=True)


if __name__ == "__main__":
    main()
