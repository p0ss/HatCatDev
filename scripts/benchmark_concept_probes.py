#!/usr/bin/env python3
"""
Benchmark concept probe accuracy across different activation contexts.

Tests probes on:
- Prompt activations (concept mentioned in input)
- Response activations (concept generated in output)
- Steered activations (steering vectors applied up/down)

Outputs CSV with cross-concept activation matrix.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.classifier import BinaryClassifier


def generate_concept_prompts(concept: str, n_samples: int, seed: int = 42) -> List[str]:
    """
    Generate diverse prompts containing the concept.

    Args:
        concept: Concept name (e.g., "Hat", "Cat", "Geology")
        n_samples: Number of prompts to generate
        seed: Random seed for reproducibility

    Returns:
        List of prompts
    """
    np.random.seed(seed)

    # Template types for diversity
    templates = [
        "The {concept} is",
        "I saw a {concept}",
        "Tell me about {concept}",
        "What is {concept}?",
        "{concept} can be",
        "A {concept} is",
        "Examples of {concept} include",
        "The main features of {concept} are",
        "In the study of {concept},",
        "Understanding {concept} requires",
    ]

    prompts = []
    for i in range(n_samples):
        template = templates[i % len(templates)]
        prompts.append(template.format(concept=concept.lower()))

    return prompts


def extract_prompt_activations(
    probe: BinaryClassifier,
    prompts: List[str],
    model,
    tokenizer,
    device: str,
    layer_idx: int = 15
) -> np.ndarray:
    """
    Extract probe activations from prompt embeddings.

    Args:
        probe: Trained binary classifier
        prompts: List of input prompts
        model: Language model
        tokenizer: Tokenizer
        device: Device to run on
        layer_idx: Layer to extract activations from

    Returns:
        Array of probe activations [n_prompts]
    """
    activations = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_idx]

            # Use last token embedding
            embedding = hidden_states[0, -1, :].cpu().numpy()

        # Get probe activation
        activation = probe.predict_proba([embedding])[0]
        activations.append(activation)

    return np.array(activations)


def extract_response_activations(
    probe: BinaryClassifier,
    prompts: List[str],
    model,
    tokenizer,
    device: str,
    layer_idx: int = 15,
    max_new_tokens: int = 20
) -> np.ndarray:
    """
    Extract probe activations during model generation.

    Args:
        probe: Trained binary classifier
        prompts: List of input prompts
        model: Language model
        tokenizer: Tokenizer
        device: Device to run on
        layer_idx: Layer to extract activations from
        max_new_tokens: Tokens to generate

    Returns:
        Array of probe activations [n_prompts]
    """
    activations = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate with hidden states
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=False
            )

            # Get hidden states from last generated token
            # outputs.hidden_states is tuple of tuples: (step, layer, batch, seq, hidden)
            last_step_states = outputs.hidden_states[-1]  # Last generation step
            hidden_state = last_step_states[layer_idx]  # Target layer
            embedding = hidden_state[0, -1, :].cpu().numpy()  # Last token

        # Get probe activation
        activation = probe.predict_proba([embedding])[0]
        activations.append(activation)

    return np.array(activations)


def extract_steered_activations(
    probe: BinaryClassifier,
    prompts: List[str],
    steering_vector: np.ndarray,
    model,
    tokenizer,
    device: str,
    layer_idx: int = 15,
    direction: str = "up",
    strength: float = 1.0
) -> np.ndarray:
    """
    Extract probe activations with steering applied.

    Args:
        probe: Trained binary classifier
        prompts: List of input prompts
        steering_vector: Probe direction vector
        model: Language model
        tokenizer: Tokenizer
        device: Device to run on
        layer_idx: Layer to apply steering
        direction: "up" or "down"
        strength: Steering strength multiplier

    Returns:
        Array of probe activations [n_prompts]
    """
    activations = []
    steering_coef = strength if direction == "up" else -strength
    steering_tensor = torch.from_numpy(steering_vector * steering_coef).float().to(device)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_idx]

            # Apply steering to last token
            steered_embedding = hidden_states[0, -1, :] + steering_tensor
            embedding = steered_embedding.cpu().numpy()

        # Get probe activation
        activation = probe.predict_proba([embedding])[0]
        activations.append(activation)

    return np.array(activations)


def load_probe(probe_path: Path) -> BinaryClassifier:
    """Load a trained binary classifier probe."""
    probe = BinaryClassifier()
    probe.load(str(probe_path))
    return probe


def run_concept_benchmark(
    concepts: List[str],
    probe_paths: Dict[str, Path],
    model,
    tokenizer,
    device: str,
    n_samples: int = 100,
    layer_idx: int = 15
) -> List[Dict]:
    """
    Run full concept probe benchmark.

    Args:
        concepts: List of concept names
        probe_paths: Dict mapping concept -> probe file path
        model: Language model
        tokenizer: Tokenizer
        device: Device to run on
        n_samples: Number of samples per concept
        layer_idx: Layer to extract activations from

    Returns:
        List of result dictionaries
    """
    results = []

    # Load all probes
    print("\nLoading probes...")
    probes = {}
    steering_vectors = {}

    for concept in concepts:
        if concept not in probe_paths:
            print(f"Warning: No probe path for {concept}, skipping")
            continue

        probe = load_probe(probe_paths[concept])
        probes[concept] = probe
        steering_vectors[concept] = probe.coef_[0]  # Linear SVM coefficient
        print(f"  ✓ Loaded {concept}")

    # Benchmark each concept prompt against each detector
    print(f"\nRunning benchmark ({n_samples} samples per concept)...")

    for prompt_concept in tqdm(concepts, desc="Prompt concepts"):
        if prompt_concept not in probes:
            continue

        # Generate prompts for this concept
        prompts = generate_concept_prompts(prompt_concept, n_samples)

        for detected_concept in concepts:
            if detected_concept not in probes:
                continue

            probe = probes[detected_concept]

            # 1. Prompt activations
            prompt_acts = extract_prompt_activations(
                probe, prompts, model, tokenizer, device, layer_idx
            )

            results.append({
                'prompt_concept': prompt_concept,
                'detected_concept': detected_concept,
                'context': 'prompt',
                'n_samples': n_samples,
                'mean_activation': float(np.mean(prompt_acts)),
                'std_activation': float(np.std(prompt_acts)),
                'peak_activation': float(np.max(prompt_acts)),
                'min_activation': float(np.min(prompt_acts))
            })

            # 2. Response activations
            response_acts = extract_response_activations(
                probe, prompts, model, tokenizer, device, layer_idx
            )

            results.append({
                'prompt_concept': prompt_concept,
                'detected_concept': detected_concept,
                'context': 'response',
                'n_samples': n_samples,
                'mean_activation': float(np.mean(response_acts)),
                'std_activation': float(np.std(response_acts)),
                'peak_activation': float(np.max(response_acts)),
                'min_activation': float(np.min(response_acts))
            })

        # 3. Steered activations (only for same concept)
        probe = probes[prompt_concept]
        steering_vec = steering_vectors[prompt_concept]

        # Steered up
        steered_up_acts = extract_steered_activations(
            probe, prompts, steering_vec, model, tokenizer, device,
            layer_idx, direction="up"
        )

        results.append({
            'prompt_concept': prompt_concept,
            'detected_concept': prompt_concept,
            'context': 'steered_up',
            'n_samples': n_samples,
            'mean_activation': float(np.mean(steered_up_acts)),
            'std_activation': float(np.std(steered_up_acts)),
            'peak_activation': float(np.max(steered_up_acts)),
            'min_activation': float(np.min(steered_up_acts))
        })

        # Steered down
        steered_down_acts = extract_steered_activations(
            probe, prompts, steering_vec, model, tokenizer, device,
            layer_idx, direction="down"
        )

        results.append({
            'prompt_concept': prompt_concept,
            'detected_concept': prompt_concept,
            'context': 'steered_down',
            'n_samples': n_samples,
            'mean_activation': float(np.mean(steered_down_acts)),
            'std_activation': float(np.std(steered_down_acts)),
            'peak_activation': float(np.max(steered_down_acts)),
            'min_activation': float(np.min(steered_down_acts))
        })

    return results


def save_results(results: List[Dict], output_path: Path):
    """Save benchmark results to CSV."""
    if not results:
        print("No results to save")
        return

    fieldnames = [
        'prompt_concept', 'detected_concept', 'context', 'n_samples',
        'mean_activation', 'std_activation', 'peak_activation', 'min_activation'
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark concept probe accuracy")
    parser.add_argument("--concepts", nargs="+", required=True, help="Concepts to benchmark")
    parser.add_argument("--probe-dir", type=Path, required=True, help="Directory containing probe files")
    parser.add_argument("--model-name", default="google/gemma-3-4b-pt", help="Model name")
    parser.add_argument("--layer-idx", type=int, default=15, help="Layer to extract activations from")
    parser.add_argument("--n-samples", type=int, default=100, help="Samples per concept")
    parser.add_argument("--output", type=Path, help="Output CSV path")
    args = parser.parse_args()

    print("=" * 80)
    print("CONCEPT PROBE BENCHMARK")
    print("=" * 80)

    # Default output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = PROJECT_ROOT / "results" / "probe_benchmarks" / f"concept_probes_{timestamp}.csv"

    # Load model
    print(f"\nLoading model: {args.model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.float32,
        device_map=device,
        local_files_only=True
    )
    model.eval()
    print(f"✓ Model loaded on {device}")

    # Find probe files
    print(f"\nSearching for probes in {args.probe_dir}")
    probe_paths = {}

    for concept in args.concepts:
        # Look for probe file (e.g., Hat_probe.pkl, hat_probe.pkl, etc.)
        candidates = [
            args.probe_dir / f"{concept}_probe.pkl",
            args.probe_dir / f"{concept.lower()}_probe.pkl",
            args.probe_dir / f"{concept}_activation_probe.pkl",
        ]

        for candidate in candidates:
            if candidate.exists():
                probe_paths[concept] = candidate
                print(f"  ✓ Found {concept}: {candidate.name}")
                break
        else:
            print(f"  ✗ Not found: {concept}")

    if not probe_paths:
        print("\nError: No probe files found")
        return 1

    # Run benchmark
    results = run_concept_benchmark(
        concepts=args.concepts,
        probe_paths=probe_paths,
        model=model,
        tokenizer=tokenizer,
        device=device,
        n_samples=args.n_samples,
        layer_idx=args.layer_idx
    )

    # Save results
    save_results(results, args.output)

    # Quick summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Concepts tested: {len(probe_paths)}")
    print(f"Total measurements: {len(results)}")
    print(f"Samples per concept: {args.n_samples}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
