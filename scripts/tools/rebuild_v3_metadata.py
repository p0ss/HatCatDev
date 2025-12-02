#!/usr/bin/env python3
"""
Rebuild v3 probe pack metadata by scanning trained probes.

This script:
1. Scans all .pt files in the v3 hierarchy folder
2. Loads the concept pack to get hierarchy information
3. Rebuilds the probe_pack.json with correct metadata
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
from collections import defaultdict

from src.registry.concept_pack_registry import ConceptPackRegistry


def scan_probe_files(probe_dir: Path):
    """Scan hierarchy directory for all probe files."""
    print(f"\nScanning probe directory: {probe_dir}")

    probes_by_layer = defaultdict(list)

    for probe_file in probe_dir.glob("*.pt"):
        # Extract concept name from filename
        # Format: ConceptName_classifier.pt or ConceptName_Simplex_classifier.pt
        filename = probe_file.stem

        if filename.endswith("_classifier"):
            concept_name = filename[:-11]  # Remove "_classifier"
        else:
            print(f"  Warning: Unexpected filename format: {filename}")
            continue

        # Load probe to get layer info
        try:
            probe = torch.load(probe_file, map_location='cpu')
            # Probes don't store layer info, we'll infer from directory structure
            # For now, assume all are in the hierarchy folder (mixed layers)
            probes_by_layer['unknown'].append(concept_name)
        except Exception as e:
            print(f"  Error loading {probe_file}: {e}")
            continue

    print(f"  Found {sum(len(v) for v in probes_by_layer.values())} probe files")
    return probes_by_layer


def load_concept_hierarchy():
    """Load concept pack to get hierarchy information."""
    print("\nLoading concept pack hierarchy...")

    registry = ConceptPackRegistry()
    concept_pack = registry.get_pack("sumo-wordnet-v1")

    print(f"  Loaded {len(concept_pack.concepts)} concepts")

    # Build hierarchy from parent-child relationships
    hierarchy = {}
    for concept_name, concept in concept_pack.concepts.items():
        for parent in concept.parents:
            if parent not in hierarchy:
                hierarchy[parent] = []
            hierarchy[parent].append(concept_name)

    print(f"  Found {len(hierarchy)} hierarchy relationships")

    return concept_pack, hierarchy


def infer_probe_layers(probe_dir: Path):
    """
    Infer which layer each probe belongs to.

    Since all probes are in a flat hierarchy directory, we need to
    check the probe architecture or metadata to determine layer.
    """
    print("\nInferring probe layers...")

    probe_layers = {}
    layer_counts = defaultdict(int)

    for probe_file in probe_dir.glob("*.pt"):
        filename = probe_file.stem

        if filename.endswith("_classifier"):
            concept_name = filename[:-11]
        else:
            continue

        try:
            probe = torch.load(probe_file, map_location='cpu')

            # Check input dimension to infer layer
            # Gemma-3-4b-pt hidden_dim = 2560
            if hasattr(probe, 'classifier') and len(probe.classifier) > 0:
                input_dim = probe.classifier[0].in_features
            elif hasattr(probe, 'fc1'):
                input_dim = probe.fc1.in_features
            else:
                # Try to get from state_dict
                state_dict = probe.state_dict() if hasattr(probe, 'state_dict') else probe
                for key in state_dict.keys():
                    if 'weight' in key:
                        input_dim = state_dict[key].shape[1]
                        break
                else:
                    print(f"  Warning: Could not determine input dim for {concept_name}")
                    continue

            # All should be 2560 for gemma-3-4b-pt
            # We can't infer layer from architecture alone since they're all trained on same model
            # We'll need to check training metadata or use v2 as reference
            probe_layers[concept_name] = None  # Unknown for now

        except Exception as e:
            print(f"  Error loading {concept_name}: {e}")
            continue

    return probe_layers


def use_v2_layer_mapping(v3_probe_dir: Path, v2_probe_dir: Path):
    """
    Use v2 pack metadata to map v3 probes to layers.
    Assumes v3 has same concepts in same layers as v2.
    """
    print("\nUsing v2 layer mapping as reference...")

    # Load v2 metadata
    v2_metadata_file = v2_probe_dir.parent / "probe_pack.json"
    if not v2_metadata_file.exists():
        print(f"  Error: v2 metadata not found at {v2_metadata_file}")
        return {}

    with open(v2_metadata_file, 'r') as f:
        v2_metadata = json.load(f)

    # Build concept -> layer mapping from v2
    v2_layer_map = {}
    for concept_name, concept_data in v2_metadata.get('concepts', {}).items():
        layers = concept_data.get('layers', [])
        if layers:
            v2_layer_map[concept_name] = layers

    print(f"  Loaded layer mappings for {len(v2_layer_map)} concepts from v2")

    # Map v3 probes to layers based on v2
    v3_probe_layers = defaultdict(lambda: defaultdict(list))

    for probe_file in v3_probe_dir.glob("*.pt"):
        filename = probe_file.stem

        if filename.endswith("_classifier"):
            concept_name = filename[:-11]
        else:
            continue

        # Get layers from v2 mapping
        if concept_name in v2_layer_map:
            for layer in v2_layer_map[concept_name]:
                v3_probe_layers[layer][concept_name] = {
                    'file': f"hierarchy/{probe_file.name}",
                    'type': 'activation'
                }
        else:
            # New concept in v3, try to infer or put in default layer
            print(f"  Warning: {concept_name} not found in v2, assigning to layer 3")
            v3_probe_layers[3][concept_name] = {
                'file': f"hierarchy/{probe_file.name}",
                'type': 'activation'
            }

    return v3_probe_layers


def rebuild_metadata(probe_pack_dir: Path):
    """Rebuild the probe_pack.json metadata file."""
    print("\n" + "=" * 80)
    print("REBUILDING V3 METADATA")
    print("=" * 80)

    v3_probe_dir = probe_pack_dir / "hierarchy"
    v2_probe_dir = probe_pack_dir.parent / "gemma-3-4b-pt_sumo-wordnet-v2" / "hierarchy"

    # Get layer mappings from v2
    probe_layers = use_v2_layer_mapping(v3_probe_dir, v2_probe_dir)

    # Load concept pack for hierarchy
    concept_pack, hierarchy = load_concept_hierarchy()

    # Build concepts dict with layer information
    concepts = {}
    for layer, layer_concepts in probe_layers.items():
        for concept_name, probe_info in layer_concepts.items():
            if concept_name not in concepts:
                concepts[concept_name] = {
                    'layers': [],
                    'probes': {}
                }

            concepts[concept_name]['layers'].append(layer)
            concepts[concept_name]['probes'][f'layer{layer}_activation'] = probe_info['file']

            # Add metadata from concept pack
            if concept_name in concept_pack.concepts:
                concept = concept_pack.concepts[concept_name]
                concepts[concept_name]['parents'] = concept.parents
                if hasattr(concept, 'description') and concept.description:
                    concepts[concept_name]['description'] = concept.description

    # Filter hierarchy to only include concepts we have probes for
    filtered_hierarchy = {}
    for parent, children in hierarchy.items():
        if parent in concepts:
            filtered_hierarchy[parent] = [child for child in children if child in concepts]

    # Create metadata structure
    metadata = {
        "probe_pack_id": "gemma-3-4b-pt_sumo-wordnet-v3",
        "version": "3.0.0",
        "description": "SUMO concept classifiers with enriched AI safety concepts",
        "created": "2025-11-23",
        "model_info": {
            "model_id": "gemma-3-4b-pt",
            "model_name": "google/gemma-3-4b-pt",
            "model_family": "gemma",
            "hidden_dim": 2560,
            "num_layers": 26
        },
        "concept_pack": {
            "pack_id": "sumo-wordnet-v1",
            "version": "1.0.0",
            "total_concepts": len(concepts)
        },
        "training_info": {
            "training_date": "2025-11-23",
            "training_method": "dual_adaptive_falloff",
            "layers_trained": sorted(list(probe_layers.keys())),
            "probe_types": ["activation"],
            "validation_mode": "falloff"
        },
        "concepts": concepts,
        "hierarchy": filtered_hierarchy,
        "probe_paths": {
            "activation_probes": "hierarchy",
            "text_probes": "text_probes",
            "metadata": "metadata"
        },
        "compatibility": {
            "min_version": "0.1.0",
            "max_version": "1.0.0"
        }
    }

    # Write metadata
    output_file = probe_pack_dir / "probe_pack.json"
    print(f"\nWriting metadata to {output_file}")
    print(f"  Total concepts: {len(concepts)}")
    print(f"  Total layers: {len(probe_layers)}")
    print(f"  Hierarchy relationships: {len(filtered_hierarchy)}")

    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Metadata rebuilt successfully")

    return metadata


def main():
    print("Rebuilding v3 probe pack metadata...")

    probe_pack_dir = Path("probe_packs/gemma-3-4b-pt_sumo-wordnet-v3")

    if not probe_pack_dir.exists():
        print(f"Error: Probe pack directory not found: {probe_pack_dir}")
        return 1

    metadata = rebuild_metadata(probe_pack_dir)

    print("\n" + "=" * 80)
    print("REBUILD COMPLETE")
    print("=" * 80)
    print(f"\nRebuilt metadata for {metadata['concept_pack']['total_concepts']} concepts")
    print(f"Layers: {metadata['training_info']['layers_trained']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
