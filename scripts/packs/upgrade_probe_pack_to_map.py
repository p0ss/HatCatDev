#!/usr/bin/env python3
"""
Upgrade a probe pack to MAP (MINDMELD Architectural Protocol) compliance.

This script:
1. Builds probe_index from classifier files in hierarchy/
2. Adds MAP fields to probe_pack.json:
   - probe_pack_id
   - concept_pack_spec_id
   - probe_output_schema
   - probe_index

Usage:
    python scripts/upgrade_probe_pack_to_map.py \\
        --probe-pack probe_packs/gemma-3-4b-pt_sumo-wordnet-v3 \\
        --concept-pack concept_packs/sumo-wordnet-v4 \\
        --model-name google/gemma-3-4b-pt

"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent


def load_concept_pack_metadata(concept_pack_dir):
    """Load concept pack metadata to get spec_id and layer information."""
    pack_json_path = concept_pack_dir / "pack.json"

    if not pack_json_path.exists():
        raise ValueError(f"Concept pack not found at {concept_pack_dir}")

    with open(pack_json_path) as f:
        pack_metadata = json.load(f)

    # Extract spec_id (or build it if missing)
    spec_id = pack_metadata.get("spec_id")
    if not spec_id:
        # Build spec_id from pack_id and version
        pack_id = pack_metadata["pack_id"]
        version = pack_metadata["version"]
        spec_id = f"org.hatcat/{pack_id}@{version}"
        print(f"WARNING: spec_id not found in concept pack, using: {spec_id}")

    return pack_metadata, spec_id


def load_layer_metadata(concept_pack_dir):
    """Load layer metadata from hierarchy/ to get concept info."""
    hierarchy_dir = concept_pack_dir / "hierarchy"

    if not hierarchy_dir.exists():
        raise ValueError(f"Hierarchy directory not found: {hierarchy_dir}")

    concept_map = {}  # Map concept name to layer info

    # Load all layer files
    for layer_file in sorted(hierarchy_dir.glob("layer*.json")):
        with open(layer_file) as f:
            layer_data = json.load(f)

        layer_num = layer_data.get("layer", 0)

        for concept in layer_data.get("concepts", []):
            sumo_term = concept.get("sumo_term")
            if sumo_term:
                concept_map[sumo_term] = {
                    "layer": layer_num,
                    "domain": concept.get("domain", "Unknown"),
                    "definition": concept.get("definition", "")
                }

    return concept_map


def build_probe_index(probe_pack_dir, concept_map, probe_pack_id, concept_pack_spec_id):
    """Build probe_index by scanning hierarchy/ for classifier files."""
    hierarchy_dir = probe_pack_dir / "hierarchy"

    if not hierarchy_dir.exists():
        raise ValueError(f"Hierarchy directory not found in probe pack: {hierarchy_dir}")

    probe_index = {}
    classifier_files = list(hierarchy_dir.glob("*_classifier.pt"))

    print(f"Found {len(classifier_files)} classifier files in {hierarchy_dir}")

    for classifier_file in classifier_files:
        # Extract concept name from filename (remove _classifier.pt suffix)
        concept_name = classifier_file.stem.replace("_classifier", "")

        # Get layer info from concept_map
        concept_info = concept_map.get(concept_name, {})
        layer = concept_info.get("layer", 0)

        # Build probe_id and concept_id
        probe_id = f"{probe_pack_id}::probe/{concept_name}"
        concept_id = f"{concept_pack_spec_id}::concept/{concept_name}"

        # Relative path from probe pack root
        relative_file = f"hierarchy/{classifier_file.name}"

        probe_index[concept_name] = {
            "probe_id": probe_id,
            "concept_id": concept_id,
            "layer": layer,
            "file": relative_file,
            "output_schema": {
                "$ref": "#/probe_output_schema"
            }
        }

    return probe_index


def upgrade_probe_pack(probe_pack_dir, concept_pack_dir, model_name):
    """Upgrade probe pack to MAP compliance."""

    print("=" * 80)
    print("UPGRADING PROBE PACK TO MAP COMPLIANCE")
    print("=" * 80)
    print()

    # Load concept pack metadata
    print(f"Loading concept pack metadata from {concept_pack_dir}...")
    concept_pack_metadata, concept_pack_spec_id = load_concept_pack_metadata(concept_pack_dir)
    print(f"  Concept Pack Spec ID: {concept_pack_spec_id}")
    print()

    # Load layer metadata
    print("Loading layer metadata...")
    concept_map = load_layer_metadata(concept_pack_dir)
    print(f"  Loaded {len(concept_map)} concepts from layer files")
    print()

    # Load existing probe_pack.json
    probe_pack_json_path = probe_pack_dir / "probe_pack.json"

    if not probe_pack_json_path.exists():
        raise ValueError(f"probe_pack.json not found at {probe_pack_json_path}")

    print(f"Loading existing probe_pack.json from {probe_pack_json_path}...")
    with open(probe_pack_json_path) as f:
        probe_pack_data = json.load(f)
    print()

    # Extract model short name for probe_pack_id
    model_short = model_name.replace("google/", "").replace("swiss-ai/", "")
    pack_version = probe_pack_data.get("version", "1.0.0")

    # Build probe_pack_id
    concept_pack_name = concept_pack_metadata["pack_id"]
    concept_pack_version = concept_pack_metadata["version"]
    probe_pack_id = f"org.hatcat/{model_short}__{concept_pack_name}@{concept_pack_version}__v{pack_version.split('.')[0]}"

    print(f"Building probe pack ID: {probe_pack_id}")
    print()

    # Build probe_index
    print("Building probe_index from classifier files...")
    probe_index = build_probe_index(probe_pack_dir, concept_map, probe_pack_id, concept_pack_spec_id)
    print(f"  Built index for {len(probe_index)} probes")
    print()

    # Add MAP fields to probe_pack_data
    print("Adding MAP fields to probe_pack.json...")

    # Set probe_pack_id
    probe_pack_data["probe_pack_id"] = probe_pack_id

    # Set concept_pack_spec_id
    probe_pack_data["concept_pack_spec_id"] = concept_pack_spec_id

    # Add probe_output_schema
    probe_pack_data["probe_output_schema"] = {
        "type": "object",
        "properties": {
            "score": {
                "type": "number",
                "description": "Concept activation score"
            },
            "null_pole": {
                "type": "number",
                "description": "Null/absence pole score"
            },
            "entropy": {
                "type": "number",
                "description": "Prediction entropy"
            }
        },
        "required": ["score"]
    }

    # Add probe_index
    probe_pack_data["probe_index"] = probe_index

    # Update timestamp
    probe_pack_data["updated"] = datetime.now().isoformat()

    # Write updated probe_pack.json
    print(f"Writing upgraded probe_pack.json to {probe_pack_json_path}...")
    with open(probe_pack_json_path, 'w') as f:
        json.dump(probe_pack_data, f, indent=2)

    print()
    print("=" * 80)
    print("UPGRADE COMPLETE")
    print("=" * 80)
    print()
    print(f"Probe Pack ID: {probe_pack_id}")
    print(f"Concept Pack Spec ID: {concept_pack_spec_id}")
    print(f"Total Probes in Index: {len(probe_index)}")
    print()
    print("The probe pack is now MAP-compliant!")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Upgrade probe pack to MAP compliance"
    )

    parser.add_argument('--probe-pack', required=True,
                        help='Path to probe pack directory (e.g., probe_packs/gemma-3-4b-pt_sumo-wordnet-v3)')
    parser.add_argument('--concept-pack', required=True,
                        help='Path to concept pack directory (e.g., concept_packs/sumo-wordnet-v4)')
    parser.add_argument('--model-name', required=True,
                        help='Full model name (e.g., google/gemma-3-4b-pt)')

    args = parser.parse_args()

    # Convert to Path objects
    probe_pack_dir = PROJECT_ROOT / args.probe_pack
    concept_pack_dir = PROJECT_ROOT / args.concept_pack

    # Validate paths
    if not probe_pack_dir.exists():
        print(f"Error: Probe pack directory not found: {probe_pack_dir}")
        sys.exit(1)

    if not concept_pack_dir.exists():
        print(f"Error: Concept pack directory not found: {concept_pack_dir}")
        sys.exit(1)

    # Run upgrade
    try:
        upgrade_probe_pack(probe_pack_dir, concept_pack_dir, args.model_name)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
