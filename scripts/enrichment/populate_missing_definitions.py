#!/usr/bin/env python3
"""
Populate missing concept definitions from SUMO source files.

Scans concept pack for concepts missing definitions, then extracts
definitions from SUMO .kif files.
"""

import argparse
import json
import re
from pathlib import Path


def extract_sumo_definitions(sumo_dir: Path) -> dict[str, str]:
    """Extract all documentation strings from SUMO .kif files."""
    definitions = {}

    # Pattern: (documentation TermName EnglishLanguage "definition text")
    # Can span multiple lines
    doc_pattern = re.compile(
        r'\(documentation\s+(\w+)\s+EnglishLanguage\s+"([^"]+)"',
        re.MULTILINE | re.DOTALL
    )

    for kif_file in sumo_dir.glob("**/*.kif"):
        content = kif_file.read_text(errors='ignore')

        for match in doc_pattern.finditer(content):
            term = match.group(1)
            definition = match.group(2).strip()
            # Clean up definition - remove SUMO markup
            definition = re.sub(r'&%\w+', lambda m: m.group(0)[2:], definition)
            definition = re.sub(r'\s+', ' ', definition)
            definitions[term] = definition

    return definitions


def extract_concept_file_definitions(concept_pack_dir: Path) -> dict[str, str]:
    """Extract definitions from individual concept JSON files."""
    definitions = {}
    concepts_dir = concept_pack_dir / "concepts"

    if not concepts_dir.exists():
        return definitions

    for concept_file in concepts_dir.glob("**/*.json"):
        try:
            with open(concept_file) as f:
                data = json.load(f)
            term = data.get('term') or data.get('sumo_term')
            definition = data.get('definition', '').strip()
            if term and definition:
                definitions[term] = definition
        except (json.JSONDecodeError, KeyError):
            continue

    return definitions


def find_missing_definitions(concept_pack_dir: Path) -> list[dict]:
    """Find concepts missing definitions."""
    missing = []
    hierarchy_dir = concept_pack_dir / "hierarchy"

    for layer_file in sorted(hierarchy_dir.glob("layer*.json")):
        with open(layer_file) as f:
            data = json.load(f)

        for concept in data.get('concepts', []):
            has_def = bool(concept.get('definition', '').strip())
            has_sumo = bool(concept.get('sumo_definition', '').strip())

            if not has_def and not has_sumo:
                missing.append({
                    'term': concept.get('sumo_term', concept.get('term')),
                    'layer': concept.get('layer'),
                    'layer_file': str(layer_file),
                    'parents': concept.get('parent_concepts', []),
                })

    return missing


def update_concept_pack(concept_pack_dir: Path, definitions: dict[str, str], dry_run: bool = True):
    """Update concept pack with found definitions."""
    hierarchy_dir = concept_pack_dir / "hierarchy"
    updated = 0
    still_missing = []

    for layer_file in sorted(hierarchy_dir.glob("layer*.json")):
        with open(layer_file) as f:
            data = json.load(f)

        modified = False
        for concept in data.get('concepts', []):
            has_def = bool(concept.get('definition', '').strip())
            has_sumo = bool(concept.get('sumo_definition', '').strip())

            if not has_def and not has_sumo:
                term = concept.get('sumo_term', concept.get('term'))
                if term in definitions:
                    concept['sumo_definition'] = definitions[term]
                    concept['definition'] = definitions[term]
                    modified = True
                    updated += 1
                    print(f"  Found: {term}")
                else:
                    still_missing.append(term)

        if modified and not dry_run:
            with open(layer_file, 'w') as f:
                json.dump(data, f, indent=2)

    return updated, still_missing


def main():
    parser = argparse.ArgumentParser(description='Populate missing definitions from SUMO')
    parser.add_argument('--concept-pack', type=str,
                        default='/home/poss/Documents/Code/HatCatDev/concept_packs/first-light',
                        help='Path to concept pack')
    parser.add_argument('--sumo-dir', type=str,
                        default='/home/poss/Documents/Code/HatCatDev/data/concept_graph/sumo_source',
                        help='Path to SUMO .kif files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be updated without writing')
    parser.add_argument('--apply', action='store_true',
                        help='Actually apply the changes')

    args = parser.parse_args()

    concept_pack_dir = Path(args.concept_pack)
    sumo_dir = Path(args.sumo_dir)

    # Find missing
    print("Finding concepts missing definitions...")
    missing = find_missing_definitions(concept_pack_dir)
    print(f"  Found {len(missing)} concepts missing definitions")

    # Extract from SUMO
    print(f"\nExtracting definitions from SUMO source: {sumo_dir}")
    sumo_defs = extract_sumo_definitions(sumo_dir)
    print(f"  Extracted {len(sumo_defs)} definitions from SUMO")

    # Extract from individual concept files
    print(f"\nExtracting definitions from concept files...")
    concept_defs = extract_concept_file_definitions(concept_pack_dir)
    print(f"  Extracted {len(concept_defs)} definitions from concept files")

    # Merge (concept files take precedence as they're more specific)
    all_defs = {**sumo_defs, **concept_defs}
    print(f"  Total unique definitions: {len(all_defs)}")

    # Check coverage
    missing_terms = {m['term'] for m in missing}
    found = missing_terms & set(all_defs.keys())
    print(f"  Can fill {len(found)} of {len(missing_terms)} missing ({100*len(found)/len(missing_terms):.1f}%)")

    # Update
    if args.apply:
        print("\nApplying updates...")
        updated, still_missing = update_concept_pack(concept_pack_dir, all_defs, dry_run=False)
        print(f"\nUpdated {updated} concepts")
        print(f"Still missing: {len(still_missing)}")
        if still_missing:
            print(f"  Examples: {still_missing[:10]}")
    else:
        print("\nDry run - showing what would be updated:")
        updated, still_missing = update_concept_pack(concept_pack_dir, all_defs, dry_run=True)
        print(f"\nWould update {updated} concepts")
        print(f"Would still be missing: {len(still_missing)}")
        if still_missing:
            print(f"  Examples: {still_missing[:10]}")
        print("\nRun with --apply to make changes")


if __name__ == '__main__':
    main()
