#!/usr/bin/env python3
"""
Probe Pack Calibration Test

Systematically evaluates all probes in a probe pack to identify:
- Over-firing probes (high false positive rate)
- Under-firing probes (low true positive rate)
- Well-calibrated probes (balanced performance)

Test conditions:
a) Positive samples: Definitions/examples containing the concept
b) Negative samples: Definitions/examples of semantically distant concepts
c) Irrelevant samples: Random text, unrelated to the concept
d) Single-term recognition: Just the concept name itself

Outputs detailed calibration metrics per probe for diagnosis.
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from nltk.corpus import wordnet as wn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.dynamic_probe_manager import DynamicProbeManager


def generate_positive_samples(concept_name: str, synsets: List[str], n_samples: int = 5) -> List[str]:
    """
    Generate positive samples for a concept.
    Uses definitions and example sentences from WordNet synsets.
    """
    samples = []

    # Add concept name itself (single-term test)
    samples.append(concept_name)

    # Add variations with context
    samples.append(f"This is about {concept_name}.")
    samples.append(f"An example of {concept_name} is when")
    samples.append(f"The concept of {concept_name} refers to")

    # If we have synsets, add definitions from WordNet
    for synset_id in synsets[:n_samples-4]:  # Use remaining slots for synset definitions
        try:
            synset = wn.synset(synset_id)
            definition = synset.definition()
            if definition:
                samples.append(definition)
        except:
            continue

    return samples[:n_samples]


def generate_negative_samples(
    concept_name: str,
    all_concepts: List[Tuple[str, int, List[str]]],  # (name, layer, synsets)
    n_samples: int = 5
) -> List[str]:
    """
    Generate negative samples: definitions of semantically distant concepts.
    """
    samples = []

    # Filter to concepts from same or nearby layers (semantic distance)
    # but exclude the target concept
    candidates = [
        (name, synsets) for name, layer, synsets in all_concepts
        if name != concept_name
    ]

    # Sample random distant concepts
    np.random.shuffle(candidates)

    for neg_concept, synsets in candidates[:n_samples * 2]:  # Try more candidates
        if len(samples) >= n_samples:
            break

        if synsets and len(synsets) > 0:
            # Use first synset definition from WordNet
            try:
                synset = wn.synset(synsets[0])
                definition = synset.definition()
                if definition:
                    samples.append(definition)
            except:
                continue

    return samples[:n_samples]


def generate_irrelevant_samples(
    concept_name: str,
    concept_layer: int,
    all_concepts: List[Tuple[str, int, List[str]]],
    n_samples: int = 5
) -> List[str]:
    """
    Generate irrelevant samples: concepts from very different semantic domains.

    Strategy: Sample from distant layers and unrelated domains.
    E.g., if testing "Animal" (concrete), use abstract concepts like "Philosophy", "Mathematics"
    """
    samples = []

    # Define semantic distance heuristic: concepts from very different layers
    # or with very different names are likely semantically distant
    candidates = []

    for name, layer, synsets in all_concepts:
        if name == concept_name:
            continue

        # Prefer concepts from different layers (different abstraction levels)
        layer_distance = abs(layer - concept_layer)

        # Add to candidates with distance score
        candidates.append((name, layer, synsets, layer_distance))

    # Sort by layer distance and take from the most distant
    candidates.sort(key=lambda x: -x[3])

    for name, layer, synsets, _ in candidates[:n_samples * 2]:  # Try more candidates
        if len(samples) >= n_samples:
            break

        if synsets and len(synsets) > 0:
            # Use first synset definition from WordNet
            try:
                synset = wn.synset(synsets[0])
                definition = synset.definition()
                if definition:
                    samples.append(definition)
            except:
                continue

    return samples


def capture_activation(
    text: str,
    model,
    tokenizer,
    target_layer: int = -1
) -> torch.Tensor:
    """
    Capture activation for a text sample at the target layer.
    Returns: [1, hidden_dim] tensor
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
        # Get last token's hidden state from target layer
        hidden_states = outputs.hidden_states[target_layer]  # [1, seq_len, hidden_dim]
        activation = hidden_states[:, -1, :]  # [1, hidden_dim]

    return activation.float()  # Convert to float32 for probes


def evaluate_probe(
    probe_key: Tuple[str, int],
    probe,
    positive_activations: List[torch.Tensor],
    negative_activations: List[torch.Tensor],
    irrelevant_activations: List[torch.Tensor],
    single_term_activation: torch.Tensor,
) -> Dict:
    """
    Evaluate a single probe across all test conditions.

    Returns:
        Dict with metrics: tp_rate, fp_rate, fn_rate, precision, recall, f1,
        avg_positive_score, avg_negative_score, avg_irrelevant_score, single_term_score
    """
    with torch.inference_mode():
        # Positive samples (should fire)
        positive_scores = [probe(act).item() for act in positive_activations]

        # Negative samples (should NOT fire)
        negative_scores = [probe(act).item() for act in negative_activations]

        # Irrelevant samples (should NOT fire)
        irrelevant_scores = [probe(act).item() for act in irrelevant_activations]

        # Single term (should fire)
        single_term_score = probe(single_term_activation).item()

    # Calculate metrics at threshold 0.5
    threshold = 0.5

    tp = sum(1 for s in positive_scores if s >= threshold)
    fn = sum(1 for s in positive_scores if s < threshold)
    fp_neg = sum(1 for s in negative_scores if s >= threshold)
    fp_irr = sum(1 for s in irrelevant_scores if s >= threshold)
    tn_neg = sum(1 for s in negative_scores if s < threshold)
    tn_irr = sum(1 for s in irrelevant_scores if s < threshold)

    total_positive = len(positive_scores)
    total_negative = len(negative_scores) + len(irrelevant_scores)

    tp_rate = tp / total_positive if total_positive > 0 else 0
    fn_rate = fn / total_positive if total_positive > 0 else 0
    fp_rate = (fp_neg + fp_irr) / total_negative if total_negative > 0 else 0

    precision = tp / (tp + fp_neg + fp_irr) if (tp + fp_neg + fp_irr) > 0 else 0
    recall = tp_rate
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp_rate': float(tp_rate),
        'fp_rate': float(fp_rate),
        'fn_rate': float(fn_rate),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'avg_positive_score': float(np.mean(positive_scores)),
        'avg_negative_score': float(np.mean(negative_scores)),
        'avg_irrelevant_score': float(np.mean(irrelevant_scores)),
        'single_term_score': float(single_term_score),
        'positive_scores': [float(s) for s in positive_scores],
        'negative_scores': [float(s) for s in negative_scores],
        'irrelevant_scores': [float(s) for s in irrelevant_scores],
    }


def categorize_probe(metrics: Dict) -> str:
    """
    Categorize probe performance:
    - well_calibrated: High precision, high recall
    - over_firing: High FP rate
    - under_firing: Low TP rate
    - broken: Very low F1
    """
    if metrics['f1'] < 0.3:
        return 'broken'
    elif metrics['fp_rate'] > 0.5:
        return 'over_firing'
    elif metrics['tp_rate'] < 0.5:
        return 'under_firing'
    elif metrics['f1'] > 0.7:
        return 'well_calibrated'
    else:
        return 'marginal'


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate probe pack by testing on OOD samples"
    )
    parser.add_argument('--probe-pack', default='gemma-3-4b-pt_sumo-wordnet-v2',
                       help='Probe pack ID to calibrate')
    parser.add_argument('--model', default='google/gemma-3-4b-pt',
                       help='Model to use for activation capture')
    parser.add_argument('--n-positive', type=int, default=5,
                       help='Number of positive samples per concept')
    parser.add_argument('--n-negative', type=int, default=5,
                       help='Number of negative samples per concept')
    parser.add_argument('--n-irrelevant', type=int, default=5,
                       help='Number of irrelevant samples')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--output-dir', default='results/probe_calibration',
                       help='Output directory')
    parser.add_argument('--max-concepts', type=int, default=None,
                       help='Limit number of concepts to test (for quick testing)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("PROBE PACK CALIBRATION TEST")
    print("="*80)
    print(f"Probe pack: {args.probe_pack}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print("="*80)

    # Load model and tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device
    )
    model.eval()
    print("✓ Model loaded")

    # Load probe manager (don't load base layers yet - load on demand)
    print("\nLoading probe manager...")
    manager = DynamicProbeManager(
        probe_pack_id=args.probe_pack,
        base_layers=[],  # Don't load any probes upfront - load on demand
        device=args.device
    )
    print(f"✓ Total concepts in metadata: {len(manager.concept_metadata)}")
    print(f"✓ Parent-child relationships: {len(manager.parent_to_children)}")

    # Debug: Show layer distribution
    layer_counts = {}
    for (concept_name, layer) in manager.concept_metadata.keys():
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    print(f"✓ Concepts by layer: {dict(sorted(layer_counts.items()))}")

    # Load synset data from layer files
    print(f"\nLoading synset data from {manager.layers_data_dir}...")
    concept_synsets = {}  # (sumo_term, layer) -> [synsets]
    layer_files = sorted(manager.layers_data_dir.glob("layer*.json"))

    for layer_file in layer_files:
        with open(layer_file) as f:
            layer_data = json.load(f)
            layer = layer_data['metadata']['layer']

            for concept in layer_data.get('concepts', []):
                sumo_term = concept['sumo_term']
                key = (sumo_term, layer)

                # For layer 6: collect individual synsets from each entry
                if layer == 6:
                    if key not in concept_synsets:
                        concept_synsets[key] = []
                    synset = concept.get('synset')  # Note: singular 'synset' field in layer 6
                    if synset:
                        concept_synsets[key].append(synset)
                else:
                    # For layers 0-5: use synsets list directly
                    synsets = concept.get('synsets', [])
                    if key not in concept_synsets:
                        concept_synsets[key] = synsets

    print(f"✓ Loaded synset data for {len(concept_synsets)} (concept, layer) pairs")

    # Get concept metadata
    concepts = []
    for (concept_name, layer), metadata in manager.concept_metadata.items():
        # Get synsets from concept pack using (concept, layer) tuple
        synsets = concept_synsets.get((concept_name, layer), [])
        concepts.append((concept_name, layer, synsets))

    if args.max_concepts:
        concepts = concepts[:args.max_concepts]

    print(f"\nTesting {len(concepts)} concepts...")
    print(f"Generating {args.n_positive} positive, {args.n_negative} negative, {args.n_irrelevant} irrelevant samples per concept")

    # Test each concept's probe
    results = {}
    categories = defaultdict(int)

    for concept_name, layer, synsets in tqdm(concepts, desc="Testing probes"):
        probe_key = (concept_name, layer)

        # Load probe on-demand
        if probe_key not in manager.loaded_activation_probes:
            # Check if probe exists (hierarchy structure: all probes in one dir)
            probe_file = manager.probes_dir / "hierarchy" / f"{concept_name}_classifier.pt"
            if not probe_file.exists():
                continue
            # Load this concept
            manager._load_concepts([probe_key], reason="calibration")

        if probe_key not in manager.loaded_activation_probes:
            continue

        probe = manager.loaded_activation_probes[probe_key]

        # Generate test samples (concept-specific)
        positive_samples = generate_positive_samples(concept_name, synsets, args.n_positive)
        negative_samples = generate_negative_samples(concept_name, concepts, args.n_negative)
        irrelevant_samples = generate_irrelevant_samples(concept_name, layer, concepts, args.n_irrelevant)

        # Capture activations
        positive_activations = [
            capture_activation(sample, model, tokenizer)
            for sample in positive_samples[1:]  # Skip first (single term)
        ]
        single_term_activation = capture_activation(positive_samples[0], model, tokenizer)

        negative_activations = [
            capture_activation(sample, model, tokenizer)
            for sample in negative_samples
        ]

        irrelevant_activations = [
            capture_activation(sample, model, tokenizer)
            for sample in irrelevant_samples
        ]

        # Evaluate probe
        metrics = evaluate_probe(
            probe_key,
            probe,
            positive_activations,
            negative_activations,
            irrelevant_activations,
            single_term_activation
        )

        # Categorize
        category = categorize_probe(metrics)
        categories[category] += 1

        results[f"{concept_name}_L{layer}"] = {
            'concept': concept_name,
            'layer': layer,
            'category': category,
            'metrics': metrics
        }

        # Unload probe to free memory
        if probe_key in manager.loaded_activation_probes:
            del manager.loaded_activation_probes[probe_key]
            del probe
            torch.cuda.empty_cache()

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'calibration_{timestamp}.json'

    summary = {
        'probe_pack': args.probe_pack,
        'model': args.model,
        'timestamp': timestamp,
        'total_probes': len(results),
        'categories': dict(categories),
        'results': results
    }

    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print("CALIBRATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total probes tested: {len(results)}")
    print(f"\nCategories:")
    for category, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(results)
        print(f"  {category:20s}: {count:4d} ({pct:5.1f}%)")

    # Identify worst offenders
    print(f"\n{'='*80}")
    print("TOP 10 OVER-FIRING PROBES (High FP Rate)")
    print(f"{'='*80}")
    over_firing = sorted(
        [(name, r['metrics']['fp_rate']) for name, r in results.items()],
        key=lambda x: -x[1]
    )[:10]
    for name, fp_rate in over_firing:
        print(f"  {name:40s}: FP Rate = {fp_rate:.3f}")

    print(f"\n{'='*80}")
    print("TOP 10 UNDER-FIRING PROBES (Low TP Rate)")
    print(f"{'='*80}")
    under_firing = sorted(
        [(name, r['metrics']['tp_rate']) for name, r in results.items()],
        key=lambda x: x[1]
    )[:10]
    for name, tp_rate in under_firing:
        print(f"  {name:40s}: TP Rate = {tp_rate:.3f}")

    print(f"\n{'='*80}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
