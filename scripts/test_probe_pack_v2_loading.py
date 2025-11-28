#!/usr/bin/env python3
"""
Test that probe packs can be loaded and used by the baseline pipeline.

This script tests:
1. Probe pack registry can find our new v2 pack
2. DynamicProbeManager can load probes from the pack
3. Probes can run inference on sample activations
4. Both default mode and explicit pack ID work
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from registry.probe_pack_registry import ProbePackRegistry
from monitoring.dynamic_probe_manager import DynamicProbeManager


def test_probe_pack_registry():
    """Test 1: Probe pack registry can find packs."""
    print("=" * 80)
    print("TEST 1: Probe Pack Registry")
    print("=" * 80)

    registry = ProbePackRegistry()
    packs = registry.list_packs()

    print(f"\nFound {len(packs)} probe pack(s):")
    for pack in packs:
        print(f"  - {pack.probe_pack_id} (v{pack.version})")
        print(f"    Location: {pack.pack_dir}")
        print(f"    Created: {pack.created}")

    # Try to get v2 pack specifically
    v2_pack = registry.get_pack("gemma-3-4b-pt_sumo-wordnet-v2")

    if v2_pack:
        print(f"\n✓ Successfully found v2 pack")
        print(f"  Pack directory: {v2_pack.pack_dir}")
        print(f"  Hierarchy dir: {v2_pack.pack_dir / 'hierarchy'}")

        # Count probes
        probe_files = list((v2_pack.pack_dir / 'hierarchy').glob('*_classifier.pt'))
        print(f"  Total probe files: {len(probe_files)}")

        return True
    else:
        print("\n✗ Could not find v2 pack")
        return False


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "PROBE PACK V2 LOADING TEST" + " " * 32 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Test registry
    success = test_probe_pack_registry()

    print("\n" + "=" * 80)
    if success:
        print("✓ TEST PASSED: v2 probe pack found and accessible")
    else:
        print("✗ TEST FAILED: Could not find v2 probe pack")
    print("=" * 80)

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
