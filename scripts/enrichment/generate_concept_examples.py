#!/usr/bin/env python3
"""
Generate positive examples for concepts using a local LLM (GPT-OSS-20B).

For each concept, generates diverse example sentences that embody the concept,
which can be used for:
- Cross-activation calibration (30 samples for law of large numbers)
- Training data augmentation

Usage:
    python scripts/enrichment/generate_concept_examples.py \
        --concept-pack concept_packs/first-light \
        --output concept_packs/first-light/generated_examples.json \
        --n-examples 30 \
        --batch-size 5 \
        --resume
"""

import argparse
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """You are generating example sentences for a concept ontology. For each concept, create diverse, natural sentences where that concept would be semantically active.

Requirements:
- Each example should be a complete, natural sentence (10-30 words)
- Examples should be diverse: different contexts, phrasings, and scenarios
- Examples should genuinely embody the concept, not just mention related words
- Avoid repetitive patterns or templates
- Cover different aspects and use cases of the concept

Return examples as a JSON array of strings, nothing else."""


def build_prompt(concept_name: str, definition: str, sumo_definition: str = "", lemmas: List[str] = None) -> str:
    """Build the prompt for generating examples for a concept."""

    prompt_parts = [f"Concept: {concept_name}"]

    if definition:
        prompt_parts.append(f"Definition: {definition}")

    if sumo_definition and sumo_definition != definition:
        prompt_parts.append(f"Formal definition: {sumo_definition}")

    if lemmas:
        prompt_parts.append(f"Related terms: {', '.join(lemmas[:5])}")

    prompt_parts.append("\nGenerate 30 diverse example sentences where this concept is semantically present.")
    prompt_parts.append("Return ONLY a JSON array of strings, no explanation.")

    return "\n".join(prompt_parts)


def parse_examples(response: str) -> List[str]:
    """Parse generated examples from model response."""
    # Try to find JSON array in response
    response = response.strip()

    # Find array bounds
    start = response.find('[')
    end = response.rfind(']')

    if start != -1 and end != -1:
        try:
            examples = json.loads(response[start:end+1])
            if isinstance(examples, list):
                return [str(e).strip() for e in examples if isinstance(e, str) and len(e) > 10]
        except json.JSONDecodeError:
            pass

    # Fallback: try to extract quoted strings
    import re
    examples = re.findall(r'"([^"]{10,200})"', response)
    return examples


def generate_examples_batch(
    model,
    tokenizer,
    concepts: List[Dict],
    device: str,
    max_new_tokens: int = 2048,
) -> Dict[str, List[str]]:
    """Generate examples for a batch of concepts."""

    results = {}

    for concept in concepts:
        term = concept.get('sumo_term') or concept.get('term', '')
        if not term:
            continue

        definition = concept.get('definition', '')
        sumo_def = concept.get('sumo_definition', '')
        lemmas = concept.get('lemmas', [])

        if not definition and not sumo_def:
            continue

        prompt = build_prompt(term, definition, sumo_def, lemmas)

        # Format as chat if model supports it, otherwise raw
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        try:
            # Try chat template first
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            # Fallback to simple format
            input_text = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant:"

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        examples = parse_examples(response)

        if examples:
            results[term] = examples

    return results


def load_concepts(concept_pack_dir: Path) -> Dict[str, Dict]:
    """Load all concepts from concept pack."""
    concepts = {}
    hierarchy_dir = concept_pack_dir / "hierarchy"

    for layer in range(7):
        layer_file = hierarchy_dir / f"layer{layer}.json"
        if not layer_file.exists():
            continue

        with open(layer_file) as f:
            data = json.load(f)

        for concept in data.get('concepts', []):
            term = concept.get('sumo_term') or concept.get('term')
            if term:
                concepts[term] = {
                    **concept,
                    'layer': layer
                }

    return concepts


def main():
    parser = argparse.ArgumentParser(description='Generate concept examples using local LLM')
    parser.add_argument('--concept-pack', type=str, default='concept_packs/first-light',
                        help='Path to concept pack')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file (default: concept_pack/generated_examples.json)')
    parser.add_argument('--model', type=str, default='openai/gpt-oss-20b',
                        help='HuggingFace model ID')
    parser.add_argument('--n-examples', type=int, default=30,
                        help='Target number of examples per concept')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Concepts to process before saving checkpoint')
    parser.add_argument('--max-concepts', type=int, default=None,
                        help='Limit total concepts (for testing)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')

    args = parser.parse_args()

    concept_pack_dir = Path(args.concept_pack)
    output_path = Path(args.output) if args.output else concept_pack_dir / "generated_examples.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if resuming
    if args.resume and output_path.exists():
        print(f"Resuming from {output_path}")
        with open(output_path) as f:
            results = json.load(f)
    else:
        results = {
            "model": args.model,
            "n_examples_target": args.n_examples,
            "created_at": datetime.now().isoformat(),
            "examples": {}
        }

    # Load concepts
    print(f"Loading concepts from {concept_pack_dir}")
    concepts = load_concepts(concept_pack_dir)
    print(f"  Found {len(concepts)} concepts")

    # Filter to concepts not yet processed
    to_process = [c for term, c in concepts.items()
                  if term not in results.get("examples", {})]

    if args.max_concepts:
        to_process = to_process[:args.max_concepts]

    print(f"  To process: {len(to_process)} concepts")

    if not to_process:
        print("All concepts already processed!")
        return

    # Load model
    print(f"\nLoading model: {args.model}")
    print("  This may take a few minutes for a 20B model...")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Try loading strategies in order of preference:
    # 1. Native quantization (for pre-quantized models like GPT-OSS with MXFP4)
    # 2. BitsAndBytes 4-bit (for unquantized models on limited VRAM)
    # 3. BFloat16 (fallback)

    model = None

    # First try: native loading (handles pre-quantized models like MXFP4)
    try:
        print("  Trying native loading (for pre-quantized models)...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            trust_remote_code=True,
        )
        print("  Loaded with native quantization")
    except Exception as e:
        print(f"  Native loading failed: {e}")

    # Second try: BitsAndBytes 4-bit quantization
    if model is None:
        try:
            from transformers import BitsAndBytesConfig
            print("  Trying BitsAndBytes 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
            print("  Loaded with 4-bit quantization")
        except Exception as e:
            print(f"  4-bit quantization failed: {e}")

    # Third try: bf16
    if model is None:
        print("  Trying bf16...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("  Loaded with bf16")

    model.eval()

    print(f"  Model loaded on {args.device}")

    # Process concepts
    print(f"\nGenerating examples...")

    for i, concept in enumerate(tqdm(to_process, desc="Concepts")):
        term = concept.get('sumo_term') or concept.get('term')

        batch_results = generate_examples_batch(
            model, tokenizer, [concept], args.device
        )

        for term, examples in batch_results.items():
            results["examples"][term] = {
                "layer": concept.get('layer'),
                "n_examples": len(examples),
                "examples": examples
            }

        # Checkpoint save
        if (i + 1) % args.batch_size == 0:
            results["updated_at"] = datetime.now().isoformat()
            results["n_concepts"] = len(results["examples"])
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

    # Final save
    results["updated_at"] = datetime.now().isoformat()
    results["n_concepts"] = len(results["examples"])
    results["completed"] = True

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    total_examples = sum(e["n_examples"] for e in results["examples"].values())
    avg_examples = total_examples / len(results["examples"]) if results["examples"] else 0

    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Concepts processed: {len(results['examples'])}")
    print(f"Total examples generated: {total_examples}")
    print(f"Average per concept: {avg_examples:.1f}")
    print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    main()
