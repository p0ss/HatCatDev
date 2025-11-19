#!/usr/bin/env python3
"""
Create v3 probe pack by merging:
- Retrained layers 0-1 with falloff validation (from results/sumo_classifiers_layer01_falloff)
- Existing layers 2-5 from v2 probe pack (from concept_packs/gemma-3-4b-pt_sumo-wordnet-v2)

This ensures all layers use consistent training methodology (adaptive + falloff validation).
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any

def load_metadata(pack_dir: Path) -> Dict[str, Any]:
    """Load metadata from probe pack."""
    metadata_path = pack_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    return {}

def create_v3_probe_pack(
    layer01_dir: str = "results/sumo_classifiers_layer01_falloff",
    v2_pack_dir: str = "concept_packs/gemma-3-4b-pt_sumo-wordnet-v2",
    output_dir: str = "concept_packs/gemma-3-4b-pt_sumo-wordnet-v3"
):
    """
    Merge retrained layers 0-1 with existing v2 layers 2-5 to create v3 pack.

    Args:
        layer01_dir: Directory with retrained layers 0-1
        v2_pack_dir: Directory with v2 probe pack (has layers 2-5)
        output_dir: Output directory for v3 pack
    """
    layer01_path = Path(layer01_dir)
    v2_pack_path = Path(v2_pack_dir)
    output_path = Path(output_dir)

    print("=" * 80)
    print("Creating v3 Probe Pack")
    print("=" * 80)
    print()

    # Verify source directories exist
    if not layer01_path.exists():
        raise FileNotFoundError(f"Layer 0-1 directory not found: {layer01_path}")
    if not v2_pack_path.exists():
        raise FileNotFoundError(f"V2 pack directory not found: {v2_pack_path}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    hierarchy_dir = output_path / "hierarchy"
    hierarchy_dir.mkdir(exist_ok=True)

    print(f"Source directories:")
    print(f"  Layer 0-1 (retrained): {layer01_path}")
    print(f"  V2 pack (layers 2-5): {v2_pack_path}")
    print(f"Output: {output_path}")
    print()

    # Copy retrained layers 0-1
    print("Copying retrained layers 0-1...")
    layer01_hierarchy = layer01_path / "hierarchy"
    if layer01_hierarchy.exists():
        copied_count = 0
        for probe_file in layer01_hierarchy.glob("*.pt"):
            # Check if this is a layer 0 or 1 probe
            # Format: ConceptName_layerN.pt
            if "_layer0.pt" in probe_file.name or "_layer1.pt" in probe_file.name:
                dest = hierarchy_dir / probe_file.name
                shutil.copy2(probe_file, dest)
                copied_count += 1
        print(f"  ✓ Copied {copied_count} probes from layers 0-1")
    else:
        print(f"  ✗ Warning: No hierarchy directory found in {layer01_path}")

    # Copy layers 2-5 from v2 pack
    print("Copying layers 2-5 from v2 pack...")
    v2_hierarchy = v2_pack_path / "hierarchy"
    if v2_hierarchy.exists():
        copied_count = 0
        for probe_file in v2_hierarchy.glob("*.pt"):
            # Only copy layers 2-5
            if any(f"_layer{i}.pt" in probe_file.name for i in [2, 3, 4, 5]):
                dest = hierarchy_dir / probe_file.name
                shutil.copy2(probe_file, dest)
                copied_count += 1
        print(f"  ✓ Copied {copied_count} probes from layers 2-5")
    else:
        print(f"  ✗ Warning: No hierarchy directory found in {v2_pack_path}")

    # Copy text classifiers from v2 pack if they exist
    v2_text_dir = v2_pack_path / "text_classifiers"
    if v2_text_dir.exists():
        print("Copying text classifiers from v2 pack...")
        output_text_dir = output_path / "text_classifiers"
        shutil.copytree(v2_text_dir, output_text_dir, dirs_exist_ok=True)
        text_count = len(list(output_text_dir.glob("*.pt")))
        print(f"  ✓ Copied {text_count} text classifiers")

    # Create metadata for v3 pack
    print("Creating metadata...")

    # Load metadata from sources
    layer01_meta = load_metadata(layer01_path)
    v2_meta = load_metadata(v2_pack_path)

    # Count probes by layer
    layer_counts = {}
    for probe_file in hierarchy_dir.glob("*.pt"):
        for layer in range(6):
            if f"_layer{layer}.pt" in probe_file.name:
                layer_counts[layer] = layer_counts.get(layer, 0) + 1

    metadata = {
        "pack_id": "gemma-3-4b-pt_sumo-wordnet-v3",
        "description": "SUMO-WordNet probes with consistent falloff validation across all layers",
        "model": "google/gemma-3-4b-pt",
        "training_method": "adaptive + falloff validation",
        "layers": {
            "0-1": "Retrained with falloff validation for consistency",
            "2-5": "From v2 pack (already trained with falloff)"
        },
        "layer_counts": layer_counts,
        "total_probes": sum(layer_counts.values()),
        "created_from": {
            "layers_0_1": str(layer01_path),
            "layers_2_5": str(v2_pack_path)
        },
        "creation_date": layer01_meta.get("creation_date", "unknown"),
        "v2_metadata": v2_meta
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Created metadata.json")
    print()

    # Summary
    print("=" * 80)
    print("V3 Probe Pack Summary")
    print("=" * 80)
    print(f"Total probes: {metadata['total_probes']}")
    print()
    print("Probes by layer:")
    for layer in sorted(layer_counts.keys()):
        count = layer_counts[layer]
        source = "retrained (falloff)" if layer <= 1 else "from v2 (falloff)"
        print(f"  Layer {layer}: {count:4d} probes  ({source})")
    print()
    print(f"Output location: {output_path}")
    print()
    print("✓ V3 probe pack created successfully!")
    print()
    print("Next steps:")
    print("  1. Wait for layer 0-1 training to complete")
    print("  2. Run this script to merge the packs")
    print("  3. Test the v3 pack with base_layers=[0, 1, 2]")
    print("     to verify consistent calibration")
    print()

if __name__ == "__main__":
    create_v3_probe_pack()
