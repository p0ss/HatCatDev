#!/usr/bin/env python3
"""
Test the old legacy probes from results/sumo_classifiers to see if they
produce better geology detection than the new probe pack.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.dynamic_probe_manager import DynamicProbeManager


def test_legacy_probes():
    """Test legacy probes on geology prompt"""

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-4b-pt', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        'google/gemma-3-4b-pt',
        torch_dtype=torch.float32,
        device_map='cuda',
        local_files_only=True
    )
    model.eval()

    # Initialize with LEGACY probes (results/sumo_classifiers)
    print("\n=== TESTING LEGACY PROBES (results/sumo_classifiers) ===")
    legacy_manager = DynamicProbeManager(
        probes_dir=Path("results/sumo_classifiers"),
        base_layers=[1],
        max_loaded_probes=500,
        load_threshold=0.5,
        keep_top_k=20,
        aggressive_pruning=False,  # Don't prune yet
        use_activation_probes=True,
        use_text_probes=False
    )

    # Test prompt
    prompt = "Mountains are formed when"
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

    # Get target layer
    target_layer = model.model.language_model.layers[15]

    with torch.no_grad():
        captured_hidden = []

        def hook(module, input, output):
            captured_hidden.append(output[0])

        handle = target_layer.register_forward_hook(hook)
        outputs = model(inputs['input_ids'])

        if captured_hidden:
            # Get last token hidden state
            h = captured_hidden[0][:, -1, :].cpu().numpy()

            # Detect with legacy probes
            detected, _ = legacy_manager.detect_and_expand(
                torch.tensor(h, dtype=torch.float32).cuda(),
                top_k=20,
                return_timing=True,
                return_logits=True
            )

            print(f"\nLegacy probes detected {len(detected)} concepts:")
            for concept_name, prob, logit, layer in detected[:15]:
                print(f"  {concept_name:30s} prob={prob:.4f} logit={logit:+.2f} layer={layer}")

        handle.remove()

    # Now test with NEW probe pack
    print("\n=== TESTING NEW PROBE PACK (probe_packs/gemma-3-4b-pt_sumo-wordnet-v1) ===")
    new_manager = DynamicProbeManager(
        probe_pack_id='gemma-3-4b-pt_sumo-wordnet-v1',
        base_layers=[1],
        max_loaded_probes=500,
        load_threshold=0.5,
        keep_top_k=20,
        aggressive_pruning=False,
        use_activation_probes=True,
        use_text_probes=False
    )

    with torch.no_grad():
        captured_hidden = []

        handle = target_layer.register_forward_hook(hook)
        outputs = model(inputs['input_ids'])

        if captured_hidden:
            h = captured_hidden[0][:, -1, :].cpu().numpy()

            detected, _ = new_manager.detect_and_expand(
                torch.tensor(h, dtype=torch.float32).cuda(),
                top_k=20,
                return_timing=True,
                return_logits=True
            )

            print(f"\nNew probe pack detected {len(detected)} concepts:")
            for concept_name, prob, logit, layer in detected[:15]:
                print(f"  {concept_name:30s} prob={prob:.4f} logit={logit:+.2f} layer={layer}")

        handle.remove()


if __name__ == '__main__':
    test_legacy_probes()
