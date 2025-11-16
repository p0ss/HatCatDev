#!/usr/bin/env python3
"""
Train S-tier three-pole simplex probes for homeostatic steering.

This script trains 3 binary classifiers per simplex:
- μ− (negative pole) detector
- μ0 (neutral homeostasis) detector
- μ+ (positive pole) detector

These enable homeostatic steering: detecting current pole and steering toward μ0.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.sumo_data_generation import create_simplex_pole_training_dataset
from training.dual_adaptive_trainer import DualAdaptiveTrainer

# Paths
LAYER2_PATH = PROJECT_ROOT / "data" / "concept_graph" / "abstraction_layers" / "layer2.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "s_tier_simplexes"

# Training configuration
BEHAVIORAL_RATIO = 0.6  # 60% behavioral, 40% definitional
N_POSITIVES = 30
N_NEGATIVES = 70


def load_s_tier_simplexes():
    """Load all S-tier simplexes from layer2.json"""
    with open(LAYER2_PATH) as f:
        layer2 = json.load(f)

    simplexes = []
    for concept in layer2['concepts']:
        if concept.get('s_tier') and concept.get('simplex_dimension'):
            simplexes.append(concept)

    return simplexes


def train_simplex_pole(
    simplex: dict,
    pole_name: str,
    trainer: DualAdaptiveTrainer,
    run_dir: Path
):
    """
    Train a single pole detector for a simplex.

    Args:
        simplex: Simplex concept dict from layer2.json
        pole_name: "negative_pole", "neutral_homeostasis", or "positive_pole"
        trainer: DualAdaptiveTrainer instance
        run_dir: Output directory for this simplex
    """
    dimension = simplex['simplex_dimension']
    three_pole = simplex['three_pole_simplex']

    # Get pole data
    pole_data = three_pole[pole_name]
    pole_type = pole_name.split('_')[0]  # "negative", "neutral", or "positive"

    # Get other poles for hard negatives
    other_pole_names = [p for p in ['negative_pole', 'neutral_homeostasis', 'positive_pole'] if p != pole_name]
    other_poles_data = [
        {**three_pole[p], 'pole_type': p.split('_')[0]}
        for p in other_pole_names
    ]

    print(f"\n  [{pole_type.upper()}] Training {pole_type} pole detector")
    print(f"    Synset: {pole_data.get('synset', 'custom SUMO')}")

    # Generate training data
    prompts, labels = create_simplex_pole_training_dataset(
        pole_data=pole_data,
        pole_type=pole_type,
        dimension=dimension,
        other_poles_data=other_poles_data,
        n_positives=N_POSITIVES,
        n_negatives=N_NEGATIVES,
        behavioral_ratio=BEHAVIORAL_RATIO
    )

    print(f"    Generated {len(prompts)} prompts ({sum(labels)} positive, {len(labels)-sum(labels)} negative)")

    # Train with adaptive falloff
    # Use same configuration as production training
    results = trainer.train_with_adaptive_falloff(
        prompts=prompts,
        labels=labels,
        concept_name=f"{dimension}_{pole_type}",
        validation_mode="falloff_strict"
    )

    # Save results
    pole_output_dir = run_dir / pole_type
    pole_output_dir.mkdir(parents=True, exist_ok=True)

    results_file = pole_output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"    ✓ Results saved to {results_file}")

    return results


def main():
    print("=" * 80)
    print("S+ THREE-POLE SIMPLEX TRAINING")
    print("=" * 80)

    # Load simplexes
    print("\n1. Loading S-tier simplexes from layer2.json...")
    simplexes = load_s_tier_simplexes()
    print(f"   Found {len(simplexes)} S-tier simplexes")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n2. Output directory: {run_dir}")

    # Load model
    print("\n3. Loading model...")
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
        local_files_only=True
    )
    model.eval()
    print(f"   ✓ Model loaded on {device}")

    # Initialize trainer
    print("\n4. Initializing trainer...")
    trainer = DualAdaptiveTrainer(
        model=model,
        tokenizer=tokenizer,
        validation_layer_idx=12,
        validate_probes=True,
        validation_mode="falloff_strict",
        train_activation=True,
        train_text=False
    )
    print("   ✓ Trainer ready")

    # Train each simplex
    print(f"\n5. Training {len(simplexes)} simplexes ({len(simplexes) * 3} probes total)...")

    all_results = []
    failed = []

    for i, simplex in enumerate(simplexes, 1):
        dimension = simplex['simplex_dimension']

        print(f"\n[{i}/{len(simplexes)}] {dimension}")
        print("─" * 60)

        simplex_dir = run_dir / dimension
        simplex_dir.mkdir(parents=True, exist_ok=True)

        simplex_results = {
            'dimension': dimension,
            'poles': {}
        }

        # Train all 3 poles
        for pole_name in ['negative_pole', 'neutral_homeostasis', 'positive_pole']:
            try:
                results = train_simplex_pole(
                    simplex=simplex,
                    pole_name=pole_name,
                    trainer=trainer,
                    run_dir=simplex_dir
                )

                pole_type = pole_name.split('_')[0]
                simplex_results['poles'][pole_type] = {
                    'success': True,
                    'test_f1': results.get('activation', {}).get('test_f1', 0.0),
                    'tier': results.get('activation', {}).get('tier', 'unknown')
                }

            except Exception as e:
                print(f"    ✗ Failed: {e}")
                pole_type = pole_name.split('_')[0]
                simplex_results['poles'][pole_type] = {
                    'success': False,
                    'error': str(e)
                }
                failed.append(f"{dimension}/{pole_type}")

        all_results.append(simplex_results)

        # Save intermediate results
        with open(run_dir / "results.json", 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'total_simplexes': len(simplexes),
                'completed': i,
                'failed_probes': failed,
                'simplexes': all_results
            }, f, indent=2)

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    total_probes = len(simplexes) * 3
    successful_probes = sum(
        sum(1 for p in s['poles'].values() if p.get('success'))
        for s in all_results
    )

    print(f"\nTotal simplexes: {len(simplexes)}")
    print(f"Total probes: {total_probes}")
    print(f"Successful: {successful_probes}/{total_probes}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed probes:")
        for probe in failed:
            print(f"  - {probe}")

    # Tier distribution
    tiers = {}
    for simplex in all_results:
        for pole_type, pole_results in simplex['poles'].items():
            if pole_results.get('success'):
                tier = pole_results.get('tier', 'unknown')
                tiers[tier] = tiers.get(tier, 0) + 1

    print("\nTier distribution:")
    for tier in ['A', 'B+', 'B', 'C+', 'C', 'F']:
        if tier in tiers:
            print(f"  {tier}: {tiers[tier]}")

    # Average test F1
    test_f1s = [
        pole_results.get('test_f1', 0.0)
        for simplex in all_results
        for pole_results in simplex['poles'].values()
        if pole_results.get('success')
    ]

    if test_f1s:
        avg_f1 = sum(test_f1s) / len(test_f1s)
        print(f"\nAverage test F1: {avg_f1:.3f}")

    print(f"\n✓ Results saved to: {run_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
