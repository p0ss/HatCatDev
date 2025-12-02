#!/usr/bin/env python3
"""
Dual Monitoring System Test

Simultaneously detects:
1. SUMO/WordNet concepts (v3 probe pack) - "what is the model thinking about?"
2. S-tier psychological simplexes (tripole probes) - "where is the model's psychological state?"

This demonstrates running both detection systems in parallel on the same activations.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.sumo_classifiers import extract_activations
from training.tripole_classifier import TripoleProbe

# Paths
PROBE_PACK_DIR = PROJECT_ROOT / "probe_packs" / "gemma-3-4b-pt_sumo-wordnet-v3"
SIMPLEX_DIR = PROJECT_ROOT / "results" / "s_tier_tripole_lazy" / "run_20251125_094653"


# Simple SUMO probe loader - just use Sequential since that's how probes are saved
def create_sumo_probe(input_dim: int, hidden_dim: int = 128):
    """Create a SUMO probe matching the saved architecture."""
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(hidden_dim, hidden_dim // 2),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(hidden_dim // 2, 1),
        torch.nn.Sigmoid()
    )


class SimplexProbe:
    """Container for a single S-tier simplex tripole probe."""

    def __init__(self, dimension: str, probe_path: Path, hidden_dim: int = 2560):
        self.dimension = dimension
        self.probe = TripoleProbe(hidden_dim=hidden_dim, n_poles=3)
        self.probe.load_state_dict(torch.load(probe_path, map_location='cpu'))
        self.probe.eval()

        # Pole names
        self.pole_names = ['negative', 'neutral', 'positive']

    def detect(self, activation: torch.Tensor) -> Dict:
        """
        Run detection on activation.

        Returns:
            Dict with pole probabilities and dominant pole
        """
        with torch.no_grad():
            logits = self.probe(activation.unsqueeze(0))  # [1, 3]
            probs = torch.softmax(logits, dim=-1).squeeze(0)  # [3]

            dominant_idx = probs.argmax().item()
            dominant_pole = self.pole_names[dominant_idx]
            dominant_conf = probs[dominant_idx].item()

            return {
                'dimension': self.dimension,
                'dominant_pole': dominant_pole,
                'confidence': dominant_conf,
                'pole_probabilities': {
                    'negative': probs[0].item(),
                    'neutral': probs[1].item(),
                    'positive': probs[2].item()
                }
            }


class DualMonitor:
    """
    Dual monitoring system: SUMO concepts + S-tier simplexes.

    Simplexes are always active (they're fundamental psychological dimensions).
    SUMO concepts can be loaded selectively as needed.
    """

    def __init__(
        self,
        model,
        tokenizer,
        probe_pack_dir: Path,
        simplex_dir: Path,
        device: str = "cuda",
        load_n_concepts: int = 50  # Load top N concepts for demo
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Load SUMO concept probes (just top N for demo)
        print(f"Loading top {load_n_concepts} SUMO concept probes...")
        self.concept_probes = self._load_concept_probes(probe_pack_dir, load_n_concepts)
        print(f"  ✓ Loaded {len(self.concept_probes)} concept probes")

        # Load S-tier simplex probes (always active - they're fundamental dimensions)
        print("\nLoading S-tier simplex probes (always active)...")
        self.simplex_probes = self._load_simplex_probes(simplex_dir)
        print(f"  ✓ Loaded {len(self.simplex_probes)} simplexes")

    def _load_concept_probes(self, probe_pack_dir: Path, n_concepts: int) -> Dict:
        """Load top N SUMO concept probes for demo."""
        # Load pack metadata
        pack_file = probe_pack_dir / "pack.json"
        with open(pack_file) as f:
            pack_metadata = json.load(f)

        hidden_dim = pack_metadata['model_info']['hidden_dim']

        # Load probe_pack.json for concept list
        probe_pack_file = probe_pack_dir / "probe_pack.json"
        with open(probe_pack_file) as f:
            probe_pack_data = json.load(f)

        # Get concept list (it's a flat list of concept names)
        concept_names = probe_pack_data['probes']['concepts'][:n_concepts]

        # Get activation probes from hierarchy
        hierarchy_dir = probe_pack_dir / "hierarchy"

        # Load top N concepts
        probes = {}
        loaded = 0

        for sumo_term in concept_names:
            # Construct probe path: hierarchy/{sumo_term}_classifier.pt
            probe_path = hierarchy_dir / f"{sumo_term}_classifier.pt"

            if probe_path.exists():
                # Load probe
                probe = create_sumo_probe(input_dim=hidden_dim)
                probe.load_state_dict(torch.load(probe_path, map_location='cpu'))
                probe.eval()
                probe.to(self.device)

                probes[sumo_term] = probe
                loaded += 1

        print(f"    Loaded {loaded} probes")
        return probes

    def _load_simplex_probes(self, simplex_dir: Path) -> List[SimplexProbe]:
        """Load all successfully trained simplex probes."""
        probes = []

        # Load results to find graduated simplexes
        results_file = simplex_dir / "results.json"
        with open(results_file) as f:
            results = json.load(f)

        graduated = results.get('graduated', [])

        for dimension in graduated:
            probe_path = simplex_dir / dimension / "tripole_probe.pt"
            if probe_path.exists():
                probe = SimplexProbe(dimension, probe_path)
                probes.append(probe)
                print(f"    ✓ {dimension}")

        return probes

    def monitor_text(
        self,
        text: str,
        layer_idx: int = 12,
        concept_threshold: float = 0.5,
        simplex_threshold: float = 0.4
    ) -> Dict:
        """
        Monitor a piece of text with both detection systems.

        Args:
            text: Text to monitor
            layer_idx: Layer to extract activations from
            concept_threshold: Min confidence to report a concept
            simplex_threshold: Min confidence to report a simplex pole

        Returns:
            Dict with both concept and simplex detections
        """
        # Extract activation from layer 12 (where simplexes were trained)
        print(f"\nMonitoring: \"{text[:80]}...\"")

        # Extract combined prompt+generation activations
        activations = extract_activations(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=[text],
            device=self.device,
            layer_idx=layer_idx,
            max_new_tokens=20,
            extraction_mode="combined"  # Gets both prompt and generation
        )

        # We get 2 activations per prompt (prompt + generation)
        prompt_activation = torch.tensor(activations[0], dtype=torch.float32)
        gen_activation = torch.tensor(activations[1], dtype=torch.float32)

        results = {
            'text': text,
            'prompt_phase': self._detect_both(prompt_activation, concept_threshold, simplex_threshold),
            'generation_phase': self._detect_both(gen_activation, concept_threshold, simplex_threshold)
        }

        return results

    def _detect_both(
        self,
        activation: torch.Tensor,
        concept_threshold: float,
        simplex_threshold: float
    ) -> Dict:
        """Run both detection systems on a single activation."""

        # 1. SUMO concept detection
        concept_detections = []
        activation_gpu = activation.to(self.device)

        for sumo_term, probe in self.concept_probes.items():
            with torch.no_grad():
                confidence = probe(activation_gpu).item()

            if confidence >= concept_threshold:
                concept_detections.append({
                    'sumo_term': sumo_term,
                    'confidence': confidence
                })

        # Sort by confidence
        concept_detections.sort(key=lambda x: x['confidence'], reverse=True)

        # 2. S-tier simplex detection (always active)
        simplex_detections = []
        for probe in self.simplex_probes:
            result = probe.detect(activation)
            if result['confidence'] >= simplex_threshold:
                simplex_detections.append(result)

        # Sort by confidence
        simplex_detections.sort(key=lambda x: x['confidence'], reverse=True)

        return {
            'concepts': concept_detections,
            'simplexes': simplex_detections
        }

    def print_results(self, results: Dict):
        """Pretty-print monitoring results."""
        print("\n" + "=" * 80)
        print("DUAL MONITORING RESULTS")
        print("=" * 80)

        for phase in ['prompt_phase', 'generation_phase']:
            phase_name = "PROMPT PROCESSING" if phase == 'prompt_phase' else "GENERATION"
            print(f"\n{phase_name}:")
            print("-" * 80)

            phase_results = results[phase]

            # Print SUMO concepts
            concepts = phase_results['concepts']
            if concepts:
                print(f"\n  SUMO Concepts Detected ({len(concepts)}):")
                for concept in concepts[:10]:  # Top 10
                    print(f"    • {concept['sumo_term']:<30} conf={concept['confidence']:.3f}")
                if len(concepts) > 10:
                    print(f"    ... and {len(concepts) - 10} more")
            else:
                print("  No SUMO concepts detected above threshold")

            # Print S-tier simplexes
            simplexes = phase_results['simplexes']
            if simplexes:
                print(f"\n  Psychological Simplexes ({len(simplexes)}):")
                for simplex in simplexes:
                    pole = simplex['dominant_pole']
                    conf = simplex['confidence']
                    dim = simplex['dimension']

                    # Color code by pole
                    if pole == 'negative':
                        indicator = "[-]"
                    elif pole == 'positive':
                        indicator = "[+]"
                    else:
                        indicator = "[0]"

                    print(f"    {indicator} {dim:<30} {pole:<10} conf={conf:.3f}")

                    # Show all pole probabilities
                    probs = simplex['pole_probabilities']
                    print(f"        (neg={probs['negative']:.2f}, neu={probs['neutral']:.2f}, pos={probs['positive']:.2f})")
            else:
                print("  No psychological simplexes detected above threshold")


def main():
    print("=" * 80)
    print("DUAL MONITORING SYSTEM TEST")
    print("=" * 80)
    print("Simultaneously detects:")
    print("  1. SUMO/WordNet concepts (what is being thought about)")
    print("  2. S-tier psychological simplexes (psychological state)")

    # Load model
    print("\n1. Loading model...")
    model_name = "google/gemma-3-4b-pt"
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

    # Initialize dual monitor
    print("\n2. Initializing dual monitoring system...")
    monitor = DualMonitor(
        model=model,
        tokenizer=tokenizer,
        probe_pack_dir=PROBE_PACK_DIR,
        simplex_dir=SIMPLEX_DIR,
        device=device
    )

    # Test prompts covering different psychological states
    test_prompts = [
        # Threat perception (negative pole)
        "The dark alley was filled with shadowy figures approaching menacingly.",

        # Social connection (positive pole)
        "We laughed together all evening, sharing stories and feeling deeply connected.",

        # Relational attachment (positive pole)
        "Holding my child's hand, I felt an overwhelming sense of love and protection.",

        # Threat perception (neutral to positive - safety)
        "The neighborhood watch program makes everyone feel secure and protected.",

        # Hedonic arousal (high intensity positive)
        "The concert was absolutely electrifying, everyone dancing with pure joy!",

        # Social evaluation (negative pole - criticism)
        "Everyone was judging me harshly, making me feel worthless and inadequate.",
    ]

    print("\n3. Running dual monitoring on test prompts...")

    all_results = []
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/{len(test_prompts)}]")
        results = monitor.monitor_text(
            text=prompt,
            layer_idx=12,  # Simplexes trained on layer 12
            concept_threshold=0.5,
            simplex_threshold=0.4
        )
        all_results.append(results)
        monitor.print_results(results)
        print()

    # Save results
    output_file = PROJECT_ROOT / "results" / "dual_monitoring_test.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
