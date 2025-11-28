#!/usr/bin/env python3
"""
Hierarchical Probe Pack Calibration Test

Tests probes against their actual hierarchical relationships:
1. **Positive samples**: All synsets for the target concept
2. **Sibling samples**: Synsets from sibling concepts (should NOT fire)
3. **Parent samples**: Synsets from parent concepts (may partially fire - general concepts)
4. **Child samples**: Synsets from child concepts (may partially fire - specific instances)
5. **Distant samples**: Synsets from unrelated branches (should NOT fire)

This tests whether probes correctly distinguish:
- Concept from siblings (specificity)
- Concept from parents (abstraction level)
- Concept from children (generalization)
- Concept from unrelated concepts (semantic distance)
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from nltk.corpus import wordnet as wn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.dynamic_probe_manager import DynamicProbeManager


def get_wordnet_definitions(synsets: List[str], max_samples: int = 10) -> List[str]:
    """Get WordNet definitions for synset IDs."""
    definitions = []
    for synset_id in synsets[:max_samples * 2]:  # Try more to get enough valid ones
        if len(definitions) >= max_samples:
            break
        try:
            synset = wn.synset(synset_id)
            definition = synset.definition()
            if definition and len(definition) > 10:  # Skip very short definitions
                definitions.append(definition)
        except:
            continue
    return definitions


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


def evaluate_probe_hierarchical(
    probe_key: Tuple[str, int],
    probe,
    concept_name: str,
    concept_synsets: List[str],
    siblings_synsets: List[List[str]],  # List of sibling concept synset lists
    parent_synsets: List[str],
    children_synsets: List[List[str]],  # List of child concept synset lists
    distant_synsets: List[List[str]],  # List of distant concept synset lists
    model,
    tokenizer,
    n_samples_per_category: int = 10
) -> Dict:
    """
    Evaluate probe against hierarchical relationships.

    Returns metrics for each category:
    - positive (target concept): should fire (high scores)
    - siblings: should NOT fire (low scores)
    - parent: may partially fire (medium scores acceptable)
    - children: may partially fire (medium scores acceptable)
    - distant: should NOT fire (low scores)
    """
    # Get definitions for each category
    positive_defs = get_wordnet_definitions(concept_synsets, n_samples_per_category)

    sibling_defs = []
    for sibling_syns in siblings_synsets[:5]:  # Sample up to 5 siblings
        sibling_defs.extend(get_wordnet_definitions(sibling_syns, max_samples=2))
    sibling_defs = sibling_defs[:n_samples_per_category]

    parent_defs = get_wordnet_definitions(parent_synsets, n_samples_per_category)

    child_defs = []
    for child_syns in children_synsets[:5]:  # Sample up to 5 children
        child_defs.extend(get_wordnet_definitions(child_syns, max_samples=2))
    child_defs = child_defs[:n_samples_per_category]

    distant_defs = []
    for distant_syns in distant_synsets[:5]:  # Sample up to 5 distant concepts
        distant_defs.extend(get_wordnet_definitions(distant_syns, max_samples=2))
    distant_defs = distant_defs[:n_samples_per_category]

    # Capture activations
    def get_scores(definitions: List[str]) -> List[float]:
        if not definitions:
            return []
        scores = []
        for definition in definitions:
            activation = capture_activation(definition, model, tokenizer)
            score = probe(activation).item()
            scores.append(score)
        return scores

    with torch.inference_mode():
        positive_scores = get_scores(positive_defs)
        sibling_scores = get_scores(sibling_defs)
        parent_scores = get_scores(parent_defs)
        child_scores = get_scores(child_defs)
        distant_scores = get_scores(distant_defs)

        # Single term test
        single_term_activation = capture_activation(concept_name, model, tokenizer)
        single_term_score = probe(single_term_activation).item()

    # Calculate metrics
    threshold = 0.5

    # Positive metrics (should fire)
    tp = sum(1 for s in positive_scores if s >= threshold)
    fn = sum(1 for s in positive_scores if s < threshold)
    tp_rate = tp / len(positive_scores) if positive_scores else 0

    # Negative metrics (siblings + distant should NOT fire)
    all_negative_scores = sibling_scores + distant_scores
    fp = sum(1 for s in all_negative_scores if s >= threshold)
    tn = sum(1 for s in all_negative_scores if s < threshold)
    fp_rate = fp / len(all_negative_scores) if all_negative_scores else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp_rate
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp_rate': float(tp_rate),
        'fp_rate': float(fp_rate),
        'fn_rate': float(fn / len(positive_scores) if positive_scores else 0),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'single_term_score': float(single_term_score),
        'positive_scores': [float(s) for s in positive_scores],
        'sibling_scores': [float(s) for s in sibling_scores],
        'parent_scores': [float(s) for s in parent_scores],
        'child_scores': [float(s) for s in child_scores],
        'distant_scores': [float(s) for s in distant_scores],
        'avg_positive': float(np.mean(positive_scores)) if positive_scores else None,
        'avg_sibling': float(np.mean(sibling_scores)) if sibling_scores else None,
        'avg_parent': float(np.mean(parent_scores)) if parent_scores else None,
        'avg_child': float(np.mean(child_scores)) if child_scores else None,
        'avg_distant': float(np.mean(distant_scores)) if distant_scores else None,
        'sample_counts': {
            'positive': len(positive_scores),
            'sibling': len(sibling_scores),
            'parent': len(parent_scores),
            'child': len(child_scores),
            'distant': len(distant_scores)
        }
    }


def categorize_probe(metrics: Dict) -> str:
    """
    Categorize probe performance based on hierarchical metrics.
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
        description="Hierarchical calibration test for probe pack"
    )
    parser.add_argument('--probe-pack', default='gemma-3-4b-pt_sumo-wordnet-v2',
                       help='Probe pack ID to calibrate')
    parser.add_argument('--model', default='google/gemma-3-4b-pt',
                       help='Model to use for activation capture')
    parser.add_argument('--n-samples', type=int, default=10,
                       help='Number of samples per category')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--output-dir', default='results/probe_calibration',
                       help='Output directory')
    parser.add_argument('--max-concepts', type=int, default=None,
                       help='Limit number of concepts to test (for quick testing)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("HIERARCHICAL PROBE PACK CALIBRATION TEST")
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

    # Load synset data from layer files
    print(f"\nLoading synset data from {manager.layers_data_dir}...")
    concept_synsets = {}  # (sumo_term, layer) -> [synsets]
    concept_parents = {}  # (concept, layer) -> list of parent names
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

                # Get parent concepts (category_children in reverse)
                category_children = concept.get('category_children', [])
                # Note: We'd need to build reverse mapping for parents
                # For now, we'll use the parent_to_children mapping from manager

    print(f"✓ Loaded synset data for {len(concept_synsets)} (concept, layer) pairs")

    # Build concept lookup by layer
    concepts_by_layer = defaultdict(list)
    for (concept_name, layer), metadata in manager.concept_metadata.items():
        synsets = concept_synsets.get((concept_name, layer), [])
        concepts_by_layer[layer].append((concept_name, layer, synsets, metadata))

    # Test concepts
    all_concepts = []
    for layer in sorted(concepts_by_layer.keys()):
        all_concepts.extend(concepts_by_layer[layer])

    if args.max_concepts:
        all_concepts = all_concepts[:args.max_concepts]

    print(f"\nTesting {len(all_concepts)} concepts with hierarchical relationships...")
    print(f"Generating {args.n_samples} samples per category (positive, siblings, parent, children, distant)")

    results = {}
    categories = defaultdict(int)

    for concept_name, layer, synsets, metadata in tqdm(all_concepts, desc="Testing probes"):
        probe_key = (concept_name, layer)

        # Load probe on-demand
        if probe_key not in manager.loaded_activation_probes:
            probe_file = manager.probes_dir / "hierarchy" / f"{concept_name}_classifier.pt"
            if not probe_file.exists():
                continue
            manager._load_concepts([probe_key], reason="calibration")

        if probe_key not in manager.loaded_activation_probes:
            continue

        probe = manager.loaded_activation_probes[probe_key]

        # Get hierarchical relationships
        # Siblings: concepts with same parent (or same layer if no parent info)
        siblings_synsets = []
        for sib_name, sib_layer, sib_synsets, _ in concepts_by_layer[layer]:
            if sib_name != concept_name:
                siblings_synsets.append(sib_synsets)

        # Parent: concepts from layer-1
        parent_synsets = []
        if layer > 0:
            parent_concepts = concepts_by_layer[layer - 1]
            for parent_name, _, parent_syns, _ in parent_concepts[:3]:  # Sample a few parents
                parent_synsets.extend(parent_syns)

        # Children: concepts from layer+1 that are children of this concept
        children_synsets = []
        children_keys = manager.parent_to_children.get(probe_key, [])  # Returns list of (name, layer) tuples
        for child_key in children_keys[:5]:  # Sample up to 5 children
            child_synsets = concept_synsets.get(child_key, [])
            if child_synsets:
                children_synsets.append(child_synsets)

        # Distant: concepts from very different layers
        distant_synsets = []
        distant_layer = (layer + 3) % 6  # Pick a distant layer
        if distant_layer in concepts_by_layer:
            for dist_name, _, dist_synsets, _ in concepts_by_layer[distant_layer][:5]:
                distant_synsets.append(dist_synsets)

        # Evaluate probe
        metrics = evaluate_probe_hierarchical(
            probe_key,
            probe,
            concept_name,
            synsets,
            siblings_synsets,
            parent_synsets,
            children_synsets,
            distant_synsets,
            model,
            tokenizer,
            args.n_samples
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
    results_file = output_dir / f'hierarchical_calibration_{timestamp}.json'

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
        pct = 100 * count / len(results) if results else 0
        print(f"  {category:20s}: {count:4d} ({pct:5.1f}%)")

    print(f"\n{'='*80}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
