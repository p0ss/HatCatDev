#!/usr/bin/env python3
"""
Test sibling ranking refinement.

Takes a sibling group with trained probes and:
1. Generates prompts for each sibling
2. Runs all sibling probes on all prompts
3. Checks if each probe ranks first on its own prompts
4. Optionally fine-tunes with ranking loss
"""

import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.classifier import BinaryClassifier
from training.sumo_classifiers import extract_activations
from training.sumo_data_generation import create_sumo_training_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_sibling_probes(
    probe_dir: Path,
    siblings: List[str],
    hidden_dim: int = 4096,
    device: str = "cuda"
) -> Dict[str, torch.nn.Module]:
    """Load trained probes for a sibling group."""
    probes = {}
    for sibling in siblings:
        probe_path = probe_dir / f"{sibling}_classifier.pt"
        if probe_path.exists():
            state = torch.load(probe_path, map_location=device)
            # Probes are raw Sequential: Linear(hidden->128)->ReLU->Dropout->Linear(128->64)->ReLU->Dropout->Linear(64->1)
            probe = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(64, 1)
            ).to(device)
            probe.load_state_dict(state)
            probe.eval()
            probes[sibling] = probe
            print(f"  Loaded: {sibling}")
        else:
            print(f"  Missing: {sibling}")
    return probes


def generate_sibling_prompts(
    concept: Dict,
    all_concepts: Dict,
    n_prompts: int = 20
) -> List[str]:
    """Generate prompts specifically for one concept."""
    # Use existing generation but only positives
    prompts, labels = create_sumo_training_dataset(
        concept=concept,
        all_concepts=all_concepts,
        negative_pool=[],  # No negatives
        n_positives=n_prompts,
        n_negatives=0,
        use_category_relationships=True,
        use_wordnet_relationships=True,
    )
    return prompts


def evaluate_sibling_ranking(
    probes: Dict[str, BinaryClassifier],
    prompts_by_sibling: Dict[str, List[str]],
    model,
    tokenizer,
    device: str = "cuda"
) -> Dict:
    """
    Evaluate how well each probe ranks on its own prompts.

    Returns metrics on ranking accuracy.
    """
    results = {
        'per_sibling': {},
        'overall_accuracy': 0,
        'confusion_matrix': {}
    }

    total_correct = 0
    total_prompts = 0

    siblings = list(probes.keys())

    for target_sibling, prompts in prompts_by_sibling.items():
        if target_sibling not in probes:
            continue

        sibling_correct = 0
        rankings = []

        # Extract activations for all prompts
        activations = extract_activations(model, tokenizer, prompts, device)

        for i, act in enumerate(activations):
            act_tensor = torch.tensor(act, dtype=torch.float32).unsqueeze(0).to(device)

            # Get scores from all sibling probes
            scores = {}
            with torch.no_grad():
                for sib_name, probe in probes.items():
                    scores[sib_name] = probe(act_tensor).item()

            # Rank by score
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            winner = ranked[0][0]
            target_rank = [r[0] for r in ranked].index(target_sibling) + 1

            rankings.append({
                'prompt_idx': i,
                'winner': winner,
                'target_rank': target_rank,
                'scores': scores
            })

            if winner == target_sibling:
                sibling_correct += 1
                total_correct += 1
            total_prompts += 1

        accuracy = sibling_correct / len(prompts) if prompts else 0
        results['per_sibling'][target_sibling] = {
            'accuracy': accuracy,
            'correct': sibling_correct,
            'total': len(prompts),
            'rankings': rankings
        }

        print(f"  {target_sibling}: {accuracy:.1%} ({sibling_correct}/{len(prompts)}) rank first")

    results['overall_accuracy'] = total_correct / total_prompts if total_prompts else 0

    # Build confusion matrix
    for target_sibling in siblings:
        results['confusion_matrix'][target_sibling] = {}
        if target_sibling not in results['per_sibling']:
            continue
        for ranking in results['per_sibling'][target_sibling]['rankings']:
            winner = ranking['winner']
            results['confusion_matrix'][target_sibling][winner] = \
                results['confusion_matrix'][target_sibling].get(winner, 0) + 1

    return results


def finetune_with_ranking_loss(
    probes: Dict[str, BinaryClassifier],
    prompts_by_sibling: Dict[str, List[str]],
    model,
    tokenizer,
    device: str = "cuda",
    epochs: int = 10,
    lr: float = 0.001,
    margin: float = 1.0
) -> Dict[str, BinaryClassifier]:
    """
    Fine-tune probes with margin ranking loss.

    For each sibling's prompts, that sibling's probe should
    score higher than all other sibling probes.
    """
    siblings = list(probes.keys())

    # Set all probes to training mode
    for probe in probes.values():
        probe.train()

    # Collect all parameters
    all_params = []
    for probe in probes.values():
        all_params.extend(probe.parameters())

    optimizer = torch.optim.Adam(all_params, lr=lr)
    margin_loss = torch.nn.MarginRankingLoss(margin=margin)

    # Pre-extract all activations
    print("\n  Extracting activations...")
    activations_by_sibling = {}
    for sibling, prompts in prompts_by_sibling.items():
        if sibling in probes:
            acts = extract_activations(model, tokenizer, prompts, device)
            activations_by_sibling[sibling] = torch.tensor(acts, dtype=torch.float32).to(device)
            print(f"    {sibling}: {len(acts)} activations")

    print(f"\n  Training for {epochs} epochs...")

    for epoch in range(epochs):
        epoch_loss = 0
        n_pairs = 0

        optimizer.zero_grad()

        for target_sibling, acts in activations_by_sibling.items():
            target_probe = probes[target_sibling]

            # For each other sibling, target should rank higher
            for other_sibling in siblings:
                if other_sibling == target_sibling:
                    continue

                other_probe = probes[other_sibling]

                # Recompute scores fresh for each pair to avoid graph reuse
                target_scores = target_probe(acts)  # [n_prompts, 1]
                other_scores = other_probe(acts)  # [n_prompts, 1]

                # Margin ranking loss: target should be > other by margin
                # y=1 means first arg should be larger
                target_indicator = torch.ones(acts.shape[0], device=device)
                loss = margin_loss(
                    target_scores.squeeze(),
                    other_scores.squeeze(),
                    target_indicator
                )

                epoch_loss += loss.item()
                n_pairs += 1

                # Accumulate gradients
                loss.backward()

        # Step after all pairs
        optimizer.step()

        avg_loss = epoch_loss / n_pairs if n_pairs else 0
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}: avg_loss={avg_loss:.4f}")

    # Set back to eval mode
    for probe in probes.values():
        probe.eval()

    return probes


def main():
    parser = argparse.ArgumentParser(description="Test sibling ranking refinement")
    parser.add_argument('--siblings', nargs='+', required=True,
                        help='List of sibling concept names')
    parser.add_argument('--probe-dir', type=str, required=True,
                        help='Directory containing trained probes')
    parser.add_argument('--concept-pack', type=str, default='sumo-wordnet-v4',
                        help='Concept pack to use')
    parser.add_argument('--model', type=str, default='swiss-ai/Apertus-8B-2509',
                        help='Model for activation extraction')
    parser.add_argument('--n-prompts', type=int, default=20,
                        help='Prompts per sibling for evaluation')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune with ranking loss after evaluation')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Fine-tuning epochs')
    parser.add_argument('--save', action='store_true',
                        help='Save fine-tuned probes')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    probe_dir = Path(args.probe_dir)
    pack_dir = PROJECT_ROOT / "concept_packs" / args.concept_pack

    print("=" * 70)
    print("SIBLING RANKING TEST")
    print("=" * 70)
    print(f"\nSiblings: {args.siblings}")
    print(f"Probe dir: {probe_dir}")
    print(f"Model: {args.model}")

    # Load model
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=args.device
    )
    model.eval()

    # Detect hidden dim from model
    hidden_dim = model.config.hidden_size
    print(f"Hidden dim: {hidden_dim}")

    # Load probes
    print(f"\nLoading probes...")
    probes = load_sibling_probes(probe_dir, args.siblings, hidden_dim, args.device)

    if len(probes) < 2:
        print("Error: Need at least 2 trained probes")
        return

    # Load concept metadata
    print(f"\nLoading concept metadata...")
    hierarchy_dir = pack_dir / "hierarchy"
    all_concepts = {}
    for layer_file in hierarchy_dir.glob("layer*.json"):
        with open(layer_file) as f:
            layer_data = json.load(f)
        for c in layer_data['concepts']:
            all_concepts[c['sumo_term']] = c

    # Generate prompts for each sibling
    print(f"\nGenerating {args.n_prompts} prompts per sibling...")
    prompts_by_sibling = {}
    for sibling in probes.keys():
        if sibling in all_concepts:
            prompts = generate_sibling_prompts(
                all_concepts[sibling],
                all_concepts,
                args.n_prompts
            )
            prompts_by_sibling[sibling] = prompts
            print(f"  {sibling}: {len(prompts)} prompts")

    # Evaluate before fine-tuning
    print(f"\n{'=' * 70}")
    print("PRE-FINETUNING EVALUATION")
    print("=" * 70)

    pre_results = evaluate_sibling_ranking(
        probes, prompts_by_sibling, model, tokenizer, args.device
    )

    print(f"\nOverall accuracy: {pre_results['overall_accuracy']:.1%}")

    # Show confusion matrix
    print(f"\nConfusion matrix (rows=target, cols=winner):")
    siblings = list(probes.keys())
    header = "            " + " ".join(f"{s[:8]:>8}" for s in siblings)
    print(header)
    for target in siblings:
        row = f"{target[:10]:>10}: "
        for winner in siblings:
            count = pre_results['confusion_matrix'].get(target, {}).get(winner, 0)
            row += f"{count:>8} "
        print(row)

    # Fine-tune if requested
    if args.finetune:
        print(f"\n{'=' * 70}")
        print("FINE-TUNING WITH RANKING LOSS")
        print("=" * 70)

        probes = finetune_with_ranking_loss(
            probes, prompts_by_sibling, model, tokenizer,
            args.device, args.epochs
        )

        # Evaluate after fine-tuning
        print(f"\n{'=' * 70}")
        print("POST-FINETUNING EVALUATION")
        print("=" * 70)

        post_results = evaluate_sibling_ranking(
            probes, prompts_by_sibling, model, tokenizer, args.device
        )

        print(f"\nOverall accuracy: {post_results['overall_accuracy']:.1%}")
        print(f"Improvement: {post_results['overall_accuracy'] - pre_results['overall_accuracy']:+.1%}")

        # Show confusion matrix
        print(f"\nConfusion matrix (rows=target, cols=winner):")
        print(header)
        for target in siblings:
            row = f"{target[:10]:>10}: "
            for winner in siblings:
                count = post_results['confusion_matrix'].get(target, {}).get(winner, 0)
                row += f"{count:>8} "
            print(row)

        # Save if requested
        if args.save:
            print(f"\nSaving fine-tuned probes...")
            for name, probe in probes.items():
                save_path = probe_dir / f"{name}_classifier_ranked.pt"
                torch.save(probe.state_dict(), save_path)
                print(f"  Saved: {save_path.name}")

    print(f"\n{'=' * 70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
