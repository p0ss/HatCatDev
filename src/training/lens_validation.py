"""
Lens validation to detect poorly-calibrated classifiers.

Validates that lenses fire specifically on their target domain,
not universally across all inputs (which indicates overfitting to
activation distribution rather than learning semantic concept).
"""

import torch
import numpy as np
from typing import Dict, Optional, List, Tuple
from pathlib import Path


# Load layer 0 concepts for validation (top-level SUMO categories)
def load_layer0_validation_prompts() -> Dict[str, str]:
    """
    Load layer 0 concepts and generate validation prompts.

    Returns dict of {concept_name: prompt} for validation testing.
    """
    import json
    from pathlib import Path

    layer0_path = Path("data/concept_graph/abstraction_layers/layer0.json")
    with open(layer0_path) as f:
        layer0_data = json.load(f)

    prompts = {}
    for concept in layer0_data['concepts']:
        concept_name = concept['sumo_term']
        definition = concept.get('definition', '')

        # Create a natural prompt using the definition
        if definition and definition != f"SUMO category: {concept_name}":
            prompts[concept_name] = definition
        else:
            # Fallback for concepts without good definitions
            prompts[concept_name] = f"The concept of {concept_name}"

    return prompts


# Cache the prompts at module load
DEFAULT_TEST_PROMPTS = load_layer0_validation_prompts()


def infer_concept_domain(concept: Dict) -> str:
    """
    Infer the best matching layer 0 domain for a concept.

    Uses the SUMO category hierarchy to find which layer 0 concept
    this concept is most likely to belong to.

    Args:
        concept: Concept dict with 'sumo_term' and optional 'definition'

    Returns:
        Layer 0 concept name (e.g., 'Process', 'Object', 'Abstract')
    """
    import json

    concept_name = concept['sumo_term']

    # Load full concept graph to find parent relationships
    try:
        # Check if concept is directly in layer 0
        layer0_path = Path("data/concept_graph/abstraction_layers/layer0.json")
        with open(layer0_path) as f:
            layer0_data = json.load(f)

        layer0_names = [c['sumo_term'] for c in layer0_data['concepts']]
        if concept_name in layer0_names:
            return concept_name

        # For layer 1+ concepts, look at category relationships
        # Try to find parent in layer 0 by checking common patterns
        name_lower = concept_name.lower()

        # Common mappings based on SUMO hierarchy
        if 'process' in name_lower or 'action' in name_lower or 'event' in name_lower:
            return 'Process'
        elif 'object' in name_lower or 'device' in name_lower or 'artifact' in name_lower:
            return 'Object'
        elif 'attribute' in name_lower or 'quality' in name_lower:
            return 'Attribute'
        elif 'relation' in name_lower:
            return 'Relation'
        elif 'quantity' in name_lower or 'number' in name_lower:
            return 'Quantity'
        elif 'collection' in name_lower or 'group' in name_lower:
            return 'Collection'
        elif 'list' in name_lower or 'sequence' in name_lower:
            return 'List'
        elif 'proposition' in name_lower or 'statement' in name_lower:
            return 'Proposition'
        elif 'physical' in name_lower:
            return 'Physical'

        # Default to Abstract for unknown
        return 'Abstract'

    except Exception:
        # Fallback if file not found
        return 'Abstract'


def validate_lens_calibration(
    lens_path: Path,
    concept: Dict,
    model,
    tokenizer,
    device: str = "cuda",
    layer_idx: int = 15,
    test_prompts: Optional[Dict[str, str]] = None,
    expected_domain: Optional[str] = None,
    top_k: int = 20,
) -> Dict[str, float]:
    """
    Validate that a lens fires specifically on its target domain.

    Tests the lens on diverse prompts and checks:
    1. Does it fire strongly on expected domain?
    2. Does it fire weakly on other domains?

    A well-calibrated lens should rank high on target domain and
    low on other domains. Universal firing (high rank everywhere)
    indicates poor calibration.

    Args:
        lens_path: Path to saved lens .pt file
        concept: Concept dict with metadata
        model: Language model
        tokenizer: Tokenizer
        device: Device for inference
        layer_idx: Model layer to extract activations from
        test_prompts: Optional custom test prompts by domain
        expected_domain: Optional explicit target domain
        top_k: Number of top lenses to check

    Returns:
        Dict with validation metrics:
        - target_rank: Rank on expected domain (lower is better)
        - target_logit: Logit on expected domain
        - avg_other_rank: Average rank on other domains (higher is better)
        - avg_other_logit: Average logit on other domains
        - calibration_score: 0-1 score (higher is better)
        - passed: Whether lens passed validation
    """
    from .classifier import BinaryClassifier

    # Load lens
    hidden_dim = 2560  # Gemma-3-4b
    lens = BinaryClassifier(hidden_dim).to(device)
    lens.load_state_dict(torch.load(lens_path))
    lens.eval()

    # Use default test prompts if not provided
    if test_prompts is None:
        test_prompts = DEFAULT_TEST_PROMPTS.copy()

    # Infer expected domain if not provided
    if expected_domain is None:
        expected_domain = infer_concept_domain(concept)

    # Ensure expected domain is in test prompts
    if expected_domain not in test_prompts:
        expected_domain = 'abstract'  # Fallback

    # Get target layer
    target_layer = model.model.language_model.layers[layer_idx]

    # Test on each domain
    results = {}

    with torch.no_grad():
        for domain, prompt in test_prompts.items():
            inputs = tokenizer(prompt, return_tensors='pt').to(device)

            # Capture activations
            captured_hidden = []

            def hook(module, input, output):
                captured_hidden.append(output[0])

            handle = target_layer.register_forward_hook(hook)
            outputs = model(inputs['input_ids'])
            handle.remove()

            if captured_hidden:
                # Get last token activation
                h = captured_hidden[0][:, -1, :].cpu().numpy()
                h_tensor = torch.tensor(h, dtype=torch.float32).to(device)

                # Get lens prediction
                logit = lens(h_tensor).item()

                results[domain] = {
                    'logit': logit,
                }

    # Calculate relative rankings
    sorted_domains = sorted(results.items(), key=lambda x: x[1]['logit'], reverse=True)
    for rank, (domain, data) in enumerate(sorted_domains, 1):
        data['rank'] = rank

    # Extract metrics
    target_rank = results[expected_domain]['rank']
    target_logit = results[expected_domain]['logit']

    other_domains = [d for d in results if d != expected_domain]
    other_ranks = [results[d]['rank'] for d in other_domains]
    other_logits = [results[d]['logit'] for d in other_domains]

    avg_other_rank = np.mean(other_ranks) if other_ranks else 0
    avg_other_logit = np.mean(other_logits) if other_logits else 0

    # Calibration score:
    # - Good if fires strongly on target (rank 1-3) and weakly on others (rank > 5)
    # - Bad if fires uniformly (rank 1 everywhere)

    # Normalize ranks to 0-1 (1 = best rank, 0 = worst rank)
    num_domains = len(results)
    target_score = 1.0 - (target_rank - 1) / (num_domains - 1)
    specificity_score = (avg_other_rank - 1) / (num_domains - 1)

    # Combined score: balance sensitivity and specificity
    calibration_score = (target_score + specificity_score) / 2

    # Pass criteria: target in top 3 AND avg_other_rank > 4
    passed = (target_rank <= 3) and (avg_other_rank > 4)

    return {
        'concept': concept['sumo_term'],
        'expected_domain': expected_domain,
        'target_rank': target_rank,
        'target_logit': target_logit,
        'avg_other_rank': avg_other_rank,
        'avg_other_logit': avg_other_logit,
        'calibration_score': calibration_score,
        'passed': passed,
        'all_results': results,
    }


def validate_lens_set(
    lens_dir: Path,
    concepts: List[Dict],
    model,
    tokenizer,
    device: str = "cuda",
    layer_idx: int = 15,
    save_results: bool = True,
) -> Dict[str, any]:
    """
    Validate all lenses in a directory.

    Args:
        lens_dir: Directory containing lens .pt files
        concepts: List of concept dicts
        model: Language model
        tokenizer: Tokenizer
        device: Device for inference
        layer_idx: Model layer for activations
        save_results: Whether to save results to JSON

    Returns:
        Dict with summary statistics and per-lens results
    """
    concept_map = {c['sumo_term']: c for c in concepts}

    lens_files = list(lens_dir.glob("*_classifier.pt"))

    results = []
    passed = 0
    failed = 0

    print(f"Validating {len(lens_files)} lenses...")

    for i, lens_path in enumerate(lens_files):
        concept_name = lens_path.stem.replace('_classifier', '')

        if concept_name not in concept_map:
            print(f"  [{i+1}/{len(lens_files)}] Skipping {concept_name} (no metadata)")
            continue

        concept = concept_map[concept_name]

        try:
            result = validate_lens_calibration(
                lens_path, concept, model, tokenizer, device, layer_idx
            )
            results.append(result)

            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"  [{i+1}/{len(lens_files)}] {concept_name}: {status} "
                  f"(target=#{result['target_rank']}, others={result['avg_other_rank']:.1f}, "
                  f"score={result['calibration_score']:.2f})")

            if result['passed']:
                passed += 1
            else:
                failed += 1

        except Exception as e:
            print(f"  [{i+1}/{len(lens_files)}] ERROR validating {concept_name}: {e}")
            failed += 1

    # Summary statistics
    total = passed + failed
    pass_rate = passed / total if total > 0 else 0

    summary = {
        'total_lenses': total,
        'passed': passed,
        'failed': failed,
        'pass_rate': pass_rate,
        'results': results,
    }

    # Save results
    if save_results:
        output_file = lens_dir / 'validation_results.json'
        import json
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Saved validation results to {output_file}")

    print(f"\n{'='*80}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total lenses: {total}")
    print(f"Passed: {passed} ({pass_rate:.1%})")
    print(f"Failed: {failed} ({(1-pass_rate):.1%})")

    # Show worst offenders
    failed_lenses = [r for r in results if not r['passed']]
    if failed_lenses:
        print(f"\nWorst calibrated lenses (universal firing):")
        failed_lenses.sort(key=lambda r: r['avg_other_rank'])
        for r in failed_lenses[:10]:
            print(f"  {r['concept']:30s} target=#{r['target_rank']} "
                  f"others={r['avg_other_rank']:.1f} score={r['calibration_score']:.2f}")

    return summary


__all__ = [
    'validate_lens_calibration',
    'validate_lens_set',
    'infer_concept_domain',
]
