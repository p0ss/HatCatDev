#!/usr/bin/env python3
"""
Populate missing training examples from source files.

Transfers positive_examples and negative_examples from:
1. Individual concept JSON files
2. Meld files (applied melds)

Into the hierarchy layer files.
"""

import argparse
import json
from pathlib import Path


def extract_concept_file_examples(concept_pack_dir: Path) -> dict[str, dict]:
    """Extract training_hints from individual concept JSON files."""
    examples = {}
    concepts_dir = concept_pack_dir / "concepts"

    if not concepts_dir.exists():
        return examples

    for concept_file in concepts_dir.glob("**/*.json"):
        try:
            with open(concept_file) as f:
                data = json.load(f)
            term = data.get('term') or data.get('sumo_term')
            hints = data.get('training_hints', {})
            if term and (hints.get('positive_examples') or hints.get('negative_examples')):
                examples[term] = hints
        except (json.JSONDecodeError, KeyError):
            continue

    return examples


def extract_meld_examples(melds_dir: Path) -> dict[str, dict]:
    """Extract training_hints from meld candidate files."""
    examples = {}

    applied_dir = melds_dir / "applied"
    if not applied_dir.exists():
        return examples

    for meld_file in applied_dir.glob("*.json"):
        try:
            with open(meld_file) as f:
                data = json.load(f)
            for candidate in data.get('candidates', []):
                term = candidate.get('term')
                hints = candidate.get('training_hints', {})
                if term and (hints.get('positive_examples') or hints.get('negative_examples')):
                    examples[term] = hints
        except (json.JSONDecodeError, KeyError):
            continue

    return examples


def find_missing_examples(concept_pack_dir: Path) -> tuple[list[dict], list[dict]]:
    """Find concepts missing training examples."""
    missing = []
    has_examples = []
    hierarchy_dir = concept_pack_dir / "hierarchy"

    for layer_file in sorted(hierarchy_dir.glob("layer*.json")):
        with open(layer_file) as f:
            data = json.load(f)

        for concept in data.get('concepts', []):
            hints = concept.get('training_hints', {})
            has_pos = bool(hints.get('positive_examples'))
            has_neg = bool(hints.get('negative_examples'))

            info = {
                'term': concept.get('sumo_term', concept.get('term')),
                'layer': concept.get('layer'),
                'layer_file': str(layer_file),
                'has_positive': has_pos,
                'has_negative': has_neg,
            }

            if not has_pos and not has_neg:
                missing.append(info)
            else:
                has_examples.append(info)

    return missing, has_examples


def update_concept_pack(concept_pack_dir: Path, all_examples: dict[str, dict], dry_run: bool = True):
    """Update concept pack with found training examples."""
    hierarchy_dir = concept_pack_dir / "hierarchy"
    updated = 0
    still_missing = []

    for layer_file in sorted(hierarchy_dir.glob("layer*.json")):
        with open(layer_file) as f:
            data = json.load(f)

        modified = False
        for concept in data.get('concepts', []):
            hints = concept.get('training_hints', {})
            has_pos = bool(hints.get('positive_examples'))
            has_neg = bool(hints.get('negative_examples'))

            if not has_pos or not has_neg:
                term = concept.get('sumo_term', concept.get('term'))
                if term in all_examples:
                    source_hints = all_examples[term]

                    # Initialize training_hints if missing
                    if 'training_hints' not in concept:
                        concept['training_hints'] = {}

                    # Only update missing fields
                    if not has_pos and source_hints.get('positive_examples'):
                        concept['training_hints']['positive_examples'] = source_hints['positive_examples']
                        modified = True
                    if not has_neg and source_hints.get('negative_examples'):
                        concept['training_hints']['negative_examples'] = source_hints['negative_examples']
                        modified = True
                    if source_hints.get('disambiguation') and not concept['training_hints'].get('disambiguation'):
                        concept['training_hints']['disambiguation'] = source_hints['disambiguation']

                    if modified:
                        updated += 1
                        print(f"  Updated: {term}")
                else:
                    if not has_pos and not has_neg:
                        still_missing.append(term)

        if modified and not dry_run:
            with open(layer_file, 'w') as f:
                json.dump(data, f, indent=2)

    return updated, still_missing


def main():
    parser = argparse.ArgumentParser(description='Populate missing training examples')
    parser.add_argument('--concept-pack', type=str,
                        default='/home/poss/Documents/Code/HatCatDev/concept_packs/first-light',
                        help='Path to concept pack')
    parser.add_argument('--melds-dir', type=str,
                        default='/home/poss/Documents/Code/HatCatDev/melds',
                        help='Path to melds directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be updated without writing')
    parser.add_argument('--apply', action='store_true',
                        help='Actually apply the changes')

    args = parser.parse_args()

    concept_pack_dir = Path(args.concept_pack)
    melds_dir = Path(args.melds_dir)

    # Find current state
    print("Analyzing current training examples in hierarchy...")
    missing, has_examples = find_missing_examples(concept_pack_dir)
    print(f"  With examples: {len(has_examples)}")
    print(f"  Missing examples: {len(missing)}")

    # Extract from concept files
    print(f"\nExtracting from concept files...")
    concept_examples = extract_concept_file_examples(concept_pack_dir)
    print(f"  Found {len(concept_examples)} concepts with examples")

    # Extract from melds
    print(f"\nExtracting from meld files: {melds_dir}")
    meld_examples = extract_meld_examples(melds_dir)
    print(f"  Found {len(meld_examples)} concepts with examples")

    # Merge (concept files take precedence)
    all_examples = {**meld_examples, **concept_examples}
    print(f"  Total unique: {len(all_examples)}")

    # Check coverage
    missing_terms = {m['term'] for m in missing}
    found = missing_terms & set(all_examples.keys())
    print(f"\n  Can fill {len(found)} of {len(missing_terms)} missing ({100*len(found)/len(missing_terms):.1f}%)")

    # Show what we can't fill
    cant_fill = missing_terms - set(all_examples.keys())
    if cant_fill:
        print(f"  Cannot fill: {len(cant_fill)} concepts")

    # Update
    if args.apply:
        print("\nApplying updates...")
        updated, still_missing = update_concept_pack(concept_pack_dir, all_examples, dry_run=False)
        print(f"\nUpdated {updated} concepts")
        print(f"Still missing: {len(still_missing)}")
    else:
        print("\nDry run - showing what would be updated:")
        updated, still_missing = update_concept_pack(concept_pack_dir, all_examples, dry_run=True)
        print(f"\nWould update {updated} concepts")
        print(f"Would still be missing: {len(still_missing)}")
        print("\nRun with --apply to make changes")


if __name__ == '__main__':
    main()
