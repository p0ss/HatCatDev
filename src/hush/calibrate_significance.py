"""
Calibrate significance scoring thresholds using lens pack calibration data.

Uses the training samples and calibration statistics to:
1. Extract per-concept noise floors from calibration.json
2. Run samples through model to compute significance scores
3. Analyze distribution of significance on filler vs decision tokens
4. Set calibrated thresholds for delta, entropy, max_above
"""

from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ConceptNoiseFloor:
    """Noise floor statistics for a single concept."""
    concept: str
    layer: int
    gen_mean: float  # Mean activation during generation
    gen_fire_rate: float  # Rate of firing above threshold during generation
    cross_mean: float  # Mean on cross-validation samples
    cross_std: float  # Std on cross-validation samples

    @property
    def noise_floor(self) -> float:
        """Compute noise floor as gen_mean (typical activation level)."""
        return self.gen_mean


def load_calibration_noise_floors(
    calibration_path: Path,
) -> Dict[str, ConceptNoiseFloor]:
    """
    Load per-concept noise floor statistics from calibration.json.

    Returns:
        Dict mapping concept_layer key (e.g. "Deception_L2") to NoiseFloor
    """
    with open(calibration_path, 'r') as f:
        data = json.load(f)

    noise_floors = {}
    calibration = data.get('calibration', {})

    for key, stats in calibration.items():
        # Key format is "ConceptName_L{layer}"
        concept = stats.get('concept', key.rsplit('_L', 1)[0])
        layer = stats.get('layer', 0)

        noise_floors[key] = ConceptNoiseFloor(
            concept=concept,
            layer=layer,
            gen_mean=stats.get('gen_mean', 0.0),
            gen_fire_rate=stats.get('gen_fire_rate', 0.0),
            cross_mean=stats.get('cross_mean', 0.0),
            cross_std=stats.get('cross_std', 0.0),
        )

    return noise_floors


def build_noise_floor_tensor(
    noise_floors: Dict[str, ConceptNoiseFloor],
    concept_to_idx: Dict[str, int],
    num_concepts: int,
):
    """
    Build a (C,) tensor of noise floor values indexed by concept ID.

    Args:
        noise_floors: Dict from load_calibration_noise_floors
        concept_to_idx: Mapping from concept name to index
        num_concepts: Total number of concepts

    Returns:
        Tensor of shape (C,) with noise floor per concept (torch.Tensor if available, else np.ndarray)
    """
    if TORCH_AVAILABLE:
        floors = torch.zeros(num_concepts)
    else:
        floors = np.zeros(num_concepts)

    for key, nf in noise_floors.items():
        concept = nf.concept
        if concept in concept_to_idx:
            idx = concept_to_idx[concept]
            floors[idx] = nf.noise_floor

    return floors


def analyze_calibration_distribution(
    calibration_path: Path,
) -> Dict[str, any]:
    """
    Analyze the distribution of calibration statistics to suggest thresholds.

    Returns:
        Dict with suggested thresholds and distribution statistics
    """
    noise_floors = load_calibration_noise_floors(calibration_path)

    # Collect statistics
    gen_means = [nf.gen_mean for nf in noise_floors.values() if nf.gen_mean > 0]
    gen_fire_rates = [nf.gen_fire_rate for nf in noise_floors.values()]
    cross_means = [nf.cross_mean for nf in noise_floors.values() if nf.cross_mean > 0]

    analysis = {
        'total_concepts': len(noise_floors),
        'concepts_with_gen_data': len(gen_means),

        'gen_mean': {
            'min': min(gen_means) if gen_means else 0,
            'max': max(gen_means) if gen_means else 0,
            'median': np.median(gen_means) if gen_means else 0,
            'p25': np.percentile(gen_means, 25) if gen_means else 0,
            'p75': np.percentile(gen_means, 75) if gen_means else 0,
        },

        'gen_fire_rate': {
            'min': min(gen_fire_rates),
            'max': max(gen_fire_rates),
            'median': np.median(gen_fire_rates),
            'mean': np.mean(gen_fire_rates),
        },

        # Suggested noise floor: median of gen_means
        # Concepts firing below this are likely noise
        'suggested_noise_floor': np.median(gen_means) if gen_means else 0.5,
    }

    # Group by layer
    by_layer = {}
    for key, nf in noise_floors.items():
        layer = nf.layer
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append(nf.gen_mean)

    analysis['by_layer'] = {
        layer: {
            'count': len(means),
            'median_gen_mean': np.median(means) if means else 0,
        }
        for layer, means in by_layer.items()
    }

    return analysis


def load_training_samples(
    training_samples_dir: Path,
    concept: str,
    layer: int,
) -> List[Dict]:
    """Load training samples for a specific concept/layer."""
    sample_file = training_samples_dir / f"layer{layer}" / f"{concept}.jsonl"

    if not sample_file.exists():
        # Try without layer subdirectory
        for subdir in training_samples_dir.iterdir():
            if subdir.is_dir():
                candidate = subdir / f"{concept}.jsonl"
                if candidate.exists():
                    sample_file = candidate
                    break

    if not sample_file.exists():
        return []

    samples = []
    with open(sample_file, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    return samples


def identify_filler_tokens(text: str) -> List[Tuple[int, str]]:
    """
    Identify likely filler tokens in generated text.

    Returns list of (position, token) for common filler words.
    """
    filler_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'that',
        'which', 'who', 'whom', 'whose', 'this', 'these', 'those', 'it', 'its',
    }

    words = text.lower().split()
    fillers = []
    for i, word in enumerate(words):
        # Strip punctuation for matching
        clean = word.strip('.,!?;:\'\"()-')
        if clean in filler_words:
            fillers.append((i, word))

    return fillers


if __name__ == '__main__':
    import sys

    # Default lens pack path
    lens_pack_path = Path('/home/poss/Documents/Code/HatCatDev/lens_packs/gemma-3-4b_first-light-v1')

    if len(sys.argv) > 1:
        lens_pack_path = Path(sys.argv[1])

    calibration_path = lens_pack_path / 'calibration.json'

    if not calibration_path.exists():
        print(f"Calibration file not found: {calibration_path}")
        sys.exit(1)

    print(f"Analyzing calibration data from: {lens_pack_path.name}")
    print("=" * 60)

    analysis = analyze_calibration_distribution(calibration_path)

    print(f"\nTotal concepts: {analysis['total_concepts']}")
    print(f"Concepts with generation data: {analysis['concepts_with_gen_data']}")

    print(f"\nGeneration-time activation statistics:")
    gm = analysis['gen_mean']
    print(f"  Median: {gm['median']:.4f}")
    print(f"  25th percentile: {gm['p25']:.4f}")
    print(f"  75th percentile: {gm['p75']:.4f}")
    print(f"  Range: [{gm['min']:.4f}, {gm['max']:.4f}]")

    print(f"\nGeneration fire rate:")
    gf = analysis['gen_fire_rate']
    print(f"  Mean: {gf['mean']:.4f}")
    print(f"  Median: {gf['median']:.4f}")

    print(f"\nSuggested noise floor: {analysis['suggested_noise_floor']:.4f}")

    print(f"\nBy layer:")
    for layer, stats in sorted(analysis['by_layer'].items()):
        print(f"  Layer {layer}: {stats['count']} concepts, median gen_mean: {stats['median_gen_mean']:.4f}")

    # Load and analyze some safety concepts
    print("\n" + "=" * 60)
    print("Safety concept noise floors:")

    noise_floors = load_calibration_noise_floors(calibration_path)
    safety_keywords = ['deception', 'manipulation', 'harm', 'sycophancy']

    safety_concepts = [
        (key, nf) for key, nf in noise_floors.items()
        if any(kw in key.lower() for kw in safety_keywords)
    ]

    for key, nf in sorted(safety_concepts, key=lambda x: x[0]):
        print(f"  {key}: gen_mean={nf.gen_mean:.4f}, fire_rate={nf.gen_fire_rate:.4f}")
