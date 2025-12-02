#!/usr/bin/env python3
"""
Post-hoc debiasing experiment for sibling confusion.

The hypothesis: probes learn a shared "concept-like" direction that fires on
all related concepts (siblings, parents, children). By identifying and
subtracting this common direction, we might improve discrimination.

Approach:
1. Load all probes from a layer
2. Compute the mean of first-layer weights (the "common direction")
3. For each probe, subtract a fraction of the common direction from its weights
4. Evaluate calibration before/after on benchmark probes
"""

import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

# Benchmark probes selected from calibration results
BENCHMARK = {
    'well_calibrated': [
        ('CriminalAction', 1), ('InductiveArgument', 1),
        ('MeasuringDevice', 2), ('Herbivore', 2),
        ('CurrencyCoin', 3), ('SelfImprovement', 3),
        ('Phanerozoic', 4), ('Aikido', 4),
    ],
    'marginal': [
        ('Entity', 0), ('Relation', 0),
        ('Indoors', 1), ('Mathematics', 1),
        ('Projectile', 2), ('Insoluble', 2),
        ('PurchaseOrder', 3), ('NightTime', 3),
        ('ArousalLow_HumanAgent', 4), ('CoverRecording', 4),
    ],
    'broken': [
        ('Process', 0), ('Attribute', 0),
        ('SpatialRelation', 1), ('BedFrame', 1),
        ('ContestAttribute', 2), ('PsychologicalAttribute', 2),
        ('Pasta', 3), ('EyeIris', 3),
        ('Episcopalian', 4), ('SwahiliLanguage', 4),
    ]
}


class ProbeClassifier(nn.Module):
    """MLP classifier matching the saved probe structure."""
    def __init__(self, input_dim: int = 4096):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def load_probe(probe_path: Path) -> ProbeClassifier:
    """Load a probe from disk."""
    state_dict = torch.load(probe_path, map_location='cpu')
    probe = ProbeClassifier()
    probe.model.load_state_dict(state_dict)
    return probe


def get_first_layer_weights(probe: ProbeClassifier) -> torch.Tensor:
    """Extract the first layer weights (shape: 128 x 4096)."""
    return probe.model[0].weight.data.clone()


def set_first_layer_weights(probe: ProbeClassifier, weights: torch.Tensor):
    """Set the first layer weights."""
    probe.model[0].weight.data = weights


def compute_common_direction(probes: Dict[str, ProbeClassifier]) -> torch.Tensor:
    """Compute the mean first-layer weight direction across all probes."""
    all_weights = []
    for name, probe in probes.items():
        weights = get_first_layer_weights(probe)  # 128 x 4096
        # Normalize each row (each hidden unit's input weights)
        normalized = weights / (weights.norm(dim=1, keepdim=True) + 1e-8)
        all_weights.append(normalized)

    # Stack and compute mean
    stacked = torch.stack(all_weights, dim=0)  # N x 128 x 4096
    mean_direction = stacked.mean(dim=0)  # 128 x 4096
    return mean_direction


def debias_probe(probe: ProbeClassifier, common_direction: torch.Tensor,
                 strength: float = 1.0) -> ProbeClassifier:
    """
    Remove the common direction from a probe's first layer weights.

    For each hidden unit, project out the component along the common direction.
    """
    weights = get_first_layer_weights(probe)  # 128 x 4096

    # For each hidden unit, project out the common direction
    debiased = weights.clone()
    for i in range(weights.shape[0]):
        w = weights[i]  # 4096
        c = common_direction[i]  # 4096

        # Normalize common direction
        c_norm = c / (c.norm() + 1e-8)

        # Project out the component along c
        projection = (w @ c_norm) * c_norm
        debiased[i] = w - strength * projection

    # Create new probe with debiased weights
    new_probe = ProbeClassifier()
    new_probe.model.load_state_dict(probe.model.state_dict())
    set_first_layer_weights(new_probe, debiased)

    return new_probe


def load_hierarchy(layers_dir: Path) -> Dict[int, Dict]:
    """Load concept hierarchy for sibling information."""
    hierarchy = {}
    for layer in range(5):
        layer_file = layers_dir / f'layer{layer}.json'
        if layer_file.exists():
            with open(layer_file) as f:
                hierarchy[layer] = json.load(f)
    return hierarchy


def main():
    parser = argparse.ArgumentParser(description='Post-hoc debiasing experiment')
    parser.add_argument('--probe-pack', default='apertus-8b_sumo-wordnet-v4',
                        help='Probe pack name')
    parser.add_argument('--layers-dir', default='data/concept_graph/abstraction_layers',
                        help='Directory containing layer hierarchy files')
    parser.add_argument('--strengths', type=float, nargs='+', default=[0.1, 0.25, 0.5, 0.75, 1.0],
                        help='Debiasing strengths to test')
    parser.add_argument('--output-dir', default='results/debiasing_experiment',
                        help='Output directory')

    args = parser.parse_args()

    probe_pack_dir = Path('probe_packs') / args.probe_pack
    layers_dir = Path(args.layers_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading probes from {probe_pack_dir}...")

    # Load all probes by layer
    probes_by_layer: Dict[int, Dict[str, ProbeClassifier]] = defaultdict(dict)
    for layer in range(5):
        layer_dir = probe_pack_dir / f'layer{layer}'
        if layer_dir.exists():
            for probe_file in layer_dir.glob('*_classifier.pt'):
                concept = probe_file.stem.replace('_classifier', '')
                probes_by_layer[layer][concept] = load_probe(probe_file)
            print(f"  Layer {layer}: {len(probes_by_layer[layer])} probes")

    # Compute common direction for each layer
    print("\nComputing common directions...")
    common_directions = {}
    for layer in range(5):
        if probes_by_layer[layer]:
            common_directions[layer] = compute_common_direction(probes_by_layer[layer])
            print(f"  Layer {layer}: computed from {len(probes_by_layer[layer])} probes")

    # Analyze the common direction
    print("\nAnalyzing common directions...")
    for layer, direction in common_directions.items():
        # Measure how similar individual probes are to the common direction
        similarities = []
        for concept, probe in probes_by_layer[layer].items():
            weights = get_first_layer_weights(probe)
            # Cosine similarity for each hidden unit
            sim = torch.nn.functional.cosine_similarity(
                weights.view(-1), direction.view(-1), dim=0
            ).item()
            similarities.append(sim)

        print(f"  Layer {layer}: mean similarity to common direction: {np.mean(similarities):.4f} "
              f"(std: {np.std(similarities):.4f})")

    # Report on benchmark probe availability
    print("\nBenchmark probe availability:")
    for category, probes in BENCHMARK.items():
        available = []
        missing = []
        for concept, layer in probes:
            if concept in probes_by_layer[layer]:
                available.append((concept, layer))
            else:
                missing.append((concept, layer))
        print(f"  {category}: {len(available)}/{len(probes)} available")
        if missing:
            print(f"    Missing: {missing}")

    # Save common directions for later analysis
    print(f"\nSaving results to {output_dir}...")
    results = {
        'common_direction_stats': {},
        'probe_counts': {str(k): len(v) for k, v in probes_by_layer.items()},
        'debiasing_strengths': args.strengths,
    }

    for layer in common_directions:
        results['common_direction_stats'][str(layer)] = {
            'shape': list(common_directions[layer].shape),
            'norm': common_directions[layer].norm().item(),
            'mean': common_directions[layer].mean().item(),
            'std': common_directions[layer].std().item(),
        }

    with open(output_dir / 'common_direction_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save the common directions as tensors
    for layer, direction in common_directions.items():
        torch.save(direction, output_dir / f'common_direction_layer{layer}.pt')

    print("\nTo test debiased probes, run the calibration script with:")
    print(f"  --debiased-probes-dir {output_dir}")
    print("\nNext step: Generate debiased probe copies and run calibration")


if __name__ == '__main__':
    main()
