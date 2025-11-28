# gemma-3-4b-pt_sumo-wordnet-v2

SUMO concept classifiers trained on Gemma-3-4b-pt with adaptive falloff validation (Nov 2024)

## Overview

- **Version**: 2.1.0
- **Model**: google/gemma-3-4b-pt
- **Total Probes**: 5619 (all AI safety concepts removed)
- **Created**: 2025-11-17
- **Updated**: 2025-11-17

## ⚠️ Removed Concepts

**47 AI safety concepts removed**: All AI safety and AI psychology concepts have been removed from this pack, including:
- 33 concepts with fake WordNet synsets (AIAbuse, AIExploitation, CognitiveSlavery, etc.)
- 14 additional AI safety concepts (AIFulfillment, AICatastrophe, SafeAIDeployment, AIAgentPsychology, and 10 AI agent personality dimensions)

## Layer Distribution

- **layer0**: 14 probes (14/14 successful)
- **layer1**: 276 probes (276/276 successful)
- **layer2**: 1070 probes (974/1070 successful)
- **layer3**: 1011 probes (1011/1011 successful)
- **layer4**: 3278 probes (3278/3278 successful)
- **layer5**: 26 probes (26/26 successful)


## Training Details

- **Trainer**: DualAdaptiveTrainer with adaptive falloff validation
- **Validation Layer**: 12
- **Validation Mode**: falloff
- **Data Source**: SUMO ontology concepts with WordNet synsets

## Usage

### Load a Single Probe

```python
import torch
from pathlib import Path

# Load probe
probe_path = Path('probe_packs/gemma-3-4b-pt_sumo-wordnet-v2/hierarchy/Animal_classifier.pt')
probe = torch.load(probe_path)

# Use for prediction
# (Assuming probe is a trained classifier)
```

### Load All Probes

```python
from src.registry.concept_pack_registry import ConceptPackRegistry

registry = ConceptPackRegistry()
pack = registry.get_pack('gemma-3-4b-pt_sumo-wordnet-v2')

# Access probe paths
hierarchy_dir = pack['pack_path'] / 'hierarchy'
probes = {
    f.stem.replace('_classifier', ''): torch.load(f)
    for f in hierarchy_dir.glob('*_classifier.pt')
}
```

## Files

- `probe_pack.json`: Manifest with metadata
- `hierarchy/`: Directory containing 5619 trained classifier files
- `README.md`: This file

## License

MIT

## References

- Base ontology: SUMO 2003
- Concept pack: sumo-wordnet-v1
- Training framework: HatCat v0.1.0
