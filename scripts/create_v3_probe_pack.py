#!/usr/bin/env python3
"""
Create v3 probe pack with:
- All hierarchy probes (layers 0-5) trained with combined-20 extraction
- S-tier simplexes for fine-grained emotional/motivational monitoring
- Consistent falloff validation across all probes

V3 improvements over V2:
1. Combined-20 extraction (prompt+generation) for 2x training data at zero cost
2. S-tier simplexes for nuanced psychological state detection
3. Unified training methodology across all layers
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
    hierarchy_dir: str = "results/sumo_classifiers_v3",
    simplexes_dir: str = "results/s_tier_simplexes/run_20251117_090018",
    output_dir: str = "probe_packs/gemma-3-4b-pt_sumo-wordnet-v3"
):
    """
    Create v3 probe pack with hierarchy probes and simplexes.

    Args:
        hierarchy_dir: Directory with all layers 0-5 trained with combined-20
        simplexes_dir: Directory with S-tier simplex probes
        output_dir: Output directory for v3 pack
    """
    hierarchy_path = Path(hierarchy_dir)
    simplexes_path = Path(simplexes_dir)
    output_path = Path(output_dir)

    print("=" * 80)
    print("Creating v3 Probe Pack")
    print("=" * 80)
    print()

    # Verify source directories exist
    if not hierarchy_path.exists():
        raise FileNotFoundError(f"Hierarchy directory not found: {hierarchy_path}")

    # Simplexes are optional for now
    has_simplexes = simplexes_path.exists()

    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    output_hierarchy = output_path / "hierarchy"
    output_hierarchy.mkdir(exist_ok=True)

    if has_simplexes:
        output_simplexes = output_path / "simplexes"
        output_simplexes.mkdir(exist_ok=True)

    print(f"Source directories:")
    print(f"  Hierarchy (layers 0-5): {hierarchy_path}")
    print(f"  Simplexes: {simplexes_path} {'✓ Found' if has_simplexes else '✗ Not found (skipping)'}")
    print(f"Output: {output_path}")
    print()

    # Copy hierarchy probes from all layers
    print("Copying hierarchy probes (layers 0-5)...")

    # Try both possible structures:
    # 1. Flat structure: hierarchy_dir/*.pt
    # 2. Layered structure: hierarchy_dir/layer0/*.pt, hierarchy_dir/layer1/*.pt, etc.

    hierarchy_probes = list(hierarchy_path.glob("**/*.pt"))

    if not hierarchy_probes:
        # Try alternative: results/sumo_classifiers_v3/layer0/ConceptName_classifier.pt
        print("  Trying layered directory structure...")
        layer_counts = {}
        for layer in range(6):
            layer_dir = hierarchy_path / f"layer{layer}"
            if layer_dir.exists():
                copied = 0
                for probe_file in layer_dir.glob("*_classifier.pt"):
                    # Rename to standard format: ConceptName_layerN.pt
                    concept_name = probe_file.stem.replace("_classifier", "")
                    dest = output_hierarchy / f"{concept_name}_layer{layer}.pt"
                    shutil.copy2(probe_file, dest)
                    copied += 1
                layer_counts[layer] = copied
                print(f"  Layer {layer}: {copied:4d} probes")
        total_hierarchy = sum(layer_counts.values())
    else:
        # Flat structure - just copy all
        layer_counts = {}
        for probe_file in hierarchy_probes:
            dest = output_hierarchy / probe_file.name
            shutil.copy2(probe_file, dest)
            # Count by layer
            for layer in range(6):
                if f"_layer{layer}.pt" in probe_file.name:
                    layer_counts[layer] = layer_counts.get(layer, 0) + 1
        total_hierarchy = len(hierarchy_probes)
        print(f"  ✓ Copied {total_hierarchy} hierarchy probes")

    # Copy simplexes if available
    simplex_count = 0
    simplex_dimensions = []
    if has_simplexes:
        print("\nCopying simplex probes...")
        # Structure: simplexes_dir/dimension_name/pole/probe.pt
        for dimension_dir in simplexes_path.iterdir():
            if dimension_dir.is_dir():
                dimension_name = dimension_dir.name
                simplex_dimensions.append(dimension_name)

                # Create dimension directory in output
                output_dim = output_simplexes / dimension_name
                output_dim.mkdir(exist_ok=True)

                # Copy all pole subdirectories
                for pole_dir in dimension_dir.iterdir():
                    if pole_dir.is_dir():
                        pole_name = pole_dir.name
                        output_pole = output_dim / pole_name

                        # Copy all .pt files from this pole
                        pt_files = list(pole_dir.glob("*.pt"))
                        if pt_files:
                            output_pole.mkdir(exist_ok=True)
                            for probe_file in pt_files:
                                dest = output_pole / probe_file.name
                                shutil.copy2(probe_file, dest)
                                simplex_count += 1

                print(f"  {dimension_name}: 3 poles")

        if simplex_count > 0:
            print(f"  ✓ Copied {simplex_count} simplex probes across {len(simplex_dimensions)} dimensions")
        else:
            print(f"  ⚠️  Warning: Found {len(simplex_dimensions)} dimensions but no .pt files")
            print(f"     Simplexes may need to be retrained with probe saving enabled")

    # Create metadata
    print("\nCreating metadata...")

    metadata = {
        "pack_id": "gemma-3-4b-pt_sumo-wordnet-v3",
        "version": "3.0",
        "description": "SUMO-WordNet hierarchy + S-tier simplexes with combined-20 extraction",
        "model": "google/gemma-3-4b-pt",
        "training_features": {
            "extraction_strategy": "combined-20 (prompt+generation)",
            "adaptive_training": True,
            "validation_mode": "falloff",
            "extraction_benefit": "2x training data at zero additional cost"
        },
        "components": {
            "hierarchy": {
                "layers": list(range(6)),
                "layer_counts": layer_counts,
                "total_probes": total_hierarchy,
                "description": "SUMO ontology hierarchy probes (Layers 0-5)"
            },
            "simplexes": {
                "dimensions": simplex_dimensions,
                "total_probes": simplex_count,
                "description": "S-tier emotional/motivational simplex probes",
                "available": has_simplexes and simplex_count > 0
            }
        },
        "total_probes": total_hierarchy + simplex_count,
        "improvements_over_v2": [
            "Combined-20 extraction (prompt+generation) for better generalization",
            "S-tier simplexes for nuanced psychological state detection",
            "Unified training methodology across all layers",
            "2x training samples at same computational cost"
        ]
    }

    with open(output_path / "pack.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Created pack.json")

    # Summary
    print("\n" + "=" * 80)
    print("V3 Probe Pack Summary")
    print("=" * 80)
    print(f"\nTotal probes: {metadata['total_probes']}")
    print(f"  Hierarchy: {total_hierarchy}")
    print(f"  Simplexes: {simplex_count}")
    print()
    print("Hierarchy probes by layer:")
    for layer in sorted(layer_counts.keys()):
        count = layer_counts.get(layer, 0)
        print(f"  Layer {layer}: {count:4d} probes")

    if has_simplexes and simplex_dimensions:
        print()
        print("Simplex dimensions:")
        for dim in simplex_dimensions:
            print(f"  - {dim}")

    print()
    print(f"Output location: {output_path}")
    print()
    print("✓ V3 probe pack created successfully!")
    print()

    if simplex_count == 0 and has_simplexes:
        print("⚠️  Note: Simplex directories found but no .pt files.")
        print("   You may need to retrain simplexes with probe saving enabled.")
        print()

    print("Next steps:")
    print("  1. Test the v3 pack with dynamic_probe_manager")
    print("  2. Run calibration tests to verify probe quality")
    print("  3. Deploy to production monitoring")
    print()

if __name__ == "__main__":
    create_v3_probe_pack()
