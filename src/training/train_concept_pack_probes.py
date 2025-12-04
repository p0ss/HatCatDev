#!/usr/bin/env python3
"""
Train probes from a concept pack.

Usage:
    python scripts/train_concept_pack_probes.py \
        --concept-pack sumo-wordnet-v4 \
        --model swiss-ai/Apertus-8B-2509 \
        --output-dir probe_packs/apertus-8b_sumo-wordnet-v4

Generates version_manifest.json for diff-based distribution per MAP_MELD_PROTOCOL.md §8.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path - script is now at src/training/train_concept_pack_probes.py
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import and patch the layer data directory before importing training functions
from training import sumo_classifiers as sumo_classifiers_module
from data.version_manifest import ProbeManifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train probe pack from concept pack"
    )

    # Concept pack
    parser.add_argument('--concept-pack', required=True,
                        help='Concept pack ID (e.g., sumo-wordnet-v4)')

    # Model configuration
    parser.add_argument('--model', required=True,
                        help='Model name (e.g., swiss-ai/Apertus-8B-2509)')
    parser.add_argument('--device', default="cuda",
                        help='Device (default: cuda)')

    # Layer selection
    parser.add_argument('--layers', nargs='+', type=int, default=None,
                        help='Which layers to train (default: all layers in pack)')

    # Training configuration
    parser.add_argument('--n-train-pos', type=int, default=50,
                        help='Positive training samples per concept (default: 50)')
    parser.add_argument('--n-train-neg', type=int, default=50,
                        help='Negative training samples per concept (default: 50)')
    parser.add_argument('--n-test-pos', type=int, default=20,
                        help='Positive test samples per concept (default: 20)')
    parser.add_argument('--n-test-neg', type=int, default=20,
                        help='Negative test samples per concept (default: 20)')

    # Adaptive training
    parser.add_argument('--validation-mode', type=str, default='falloff',
                        choices=['loose', 'falloff', 'strict'],
                        help='Validation mode (default: falloff)')

    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for probe pack')

    return parser.parse_args()


def main():
    args = parse_args()

    # Load concept pack metadata
    pack_dir = PROJECT_ROOT / "concept_packs" / args.concept_pack
    pack_json_path = pack_dir / "pack.json"

    if not pack_json_path.exists():
        print(f"Error: Concept pack not found at {pack_dir}")
        print(f"Looking for: {pack_json_path}")
        sys.exit(1)

    with open(pack_json_path) as f:
        pack_metadata = json.load(f)

    print("=" * 80)
    print(f"TRAINING PROBE PACK FROM CONCEPT PACK")
    print("=" * 80)
    print()
    print(f"Concept Pack: {pack_metadata['pack_id']} v{pack_metadata['version']}")
    print(f"Description: {pack_metadata['description']}")
    print(f"Total Concepts: {pack_metadata['concept_metadata']['total_concepts']}")
    print(f"Model: {args.model}")
    print()

    # Determine layers to train
    available_layers = pack_metadata['concept_metadata']['layers']
    if args.layers is None:
        layers_to_train = available_layers
    else:
        layers_to_train = args.layers
        # Validate requested layers exist
        invalid_layers = [l for l in layers_to_train if l not in available_layers]
        if invalid_layers:
            print(f"Error: Requested layers {invalid_layers} not found in pack")
            print(f"Available layers: {available_layers}")
            sys.exit(1)

    print(f"Training Layers: {layers_to_train}")
    print()

    # Get concept pack hierarchy directory
    hierarchy_dir = pack_dir / pack_metadata['concept_metadata']['hierarchy_file']

    if not hierarchy_dir.exists():
        print(f"Error: Hierarchy directory not found at {hierarchy_dir}")
        sys.exit(1)

    print(f"Loading concepts from: {hierarchy_dir}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save pack metadata to output for reference
    pack_reference = {
        "source_pack": args.concept_pack,
        "pack_version": pack_metadata['version'],
        "model": args.model,
        "trained_layers": layers_to_train,
        "training_config": {
            "n_train_pos": args.n_train_pos,
            "n_train_neg": args.n_train_neg,
            "n_test_pos": args.n_test_pos,
            "n_test_neg": args.n_test_neg,
            "validation_mode": args.validation_mode
        },
        "trained_at": datetime.now().isoformat() + "Z"
    }

    with open(output_dir / "pack_info.json", 'w') as f:
        json.dump(pack_reference, f, indent=2)

    # Train probes using the concept pack hierarchy
    # NOTE: hierarchy_dir is passed directly to train_sumo_classifiers to avoid
    # the Python default argument bug where module-level patching doesn't work
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"Using hierarchy: {hierarchy_dir}")
    print()

    sumo_classifiers_module.train_sumo_classifiers(
        layers=layers_to_train,
        hierarchy_dir=hierarchy_dir,
        model_name=args.model,
        device=args.device,
        n_train_pos=args.n_train_pos,
        n_train_neg=args.n_train_neg,
        n_test_pos=args.n_test_pos,
        n_test_neg=args.n_test_neg,
        output_dir=str(output_dir),
        use_adaptive_training=True,
        validation_mode=args.validation_mode
    )

    # Generate version manifest from training results
    print()
    print("=" * 80)
    print("GENERATING VERSION MANIFEST")
    print("=" * 80)
    print()

    manifest = ProbeManifest(
        probe_pack_id=f"{args.concept_pack}_{args.model.replace('/', '_')}",
        model=args.model,
        source_pack=args.concept_pack,
    )
    manifest.current_version = pack_metadata['version']
    manifest._path = output_dir / "version_manifest.json"

    # Collect training results from layer results.json files
    trained_concepts = []
    for layer in layers_to_train:
        layer_results_path = output_dir / f"layer{layer}" / "results.json"
        if layer_results_path.exists():
            with open(layer_results_path) as f:
                layer_results = json.load(f)

            for result in layer_results.get("results", []):
                concept = result.get("concept", "")
                if concept:
                    manifest.update_probe(
                        concept=concept,
                        version=pack_metadata['version'],
                        layer=layer,
                        metrics={
                            "f1": result.get("test_f1", 0),
                            "precision": result.get("test_precision", 0),
                            "recall": result.get("test_recall", 0),
                        },
                        training_samples=result.get("n_train_samples", 0),
                        probe_file=f"layer{layer}/{concept}_classifier.pt"
                    )
                    trained_concepts.append(concept)

    # Record training run in version history
    manifest.record_training_run(
        version=pack_metadata['version'],
        trained_concepts=trained_concepts
    )

    manifest.save()
    print(f"Version manifest saved: {len(trained_concepts)} probes tracked")
    print(f"  Version: {manifest.current_version}")
    print(f"  Layers: {sorted(set(p.layer for p in manifest.probes.values()))}")

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print()
    print(f"Probe pack saved to: {output_dir}")
    print(f"Version manifest: {output_dir / 'version_manifest.json'}")
    print()


if __name__ == "__main__":
    main()
