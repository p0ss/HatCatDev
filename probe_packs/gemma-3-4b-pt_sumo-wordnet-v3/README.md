# gemma-3-4b-pt_sumo-wordnet-v3

SUMO concept classifiers trained on Gemma-3-4b-pt with adaptive falloff validation (V3)

## Overview

- **Version**: 2.20251123.0
- **Model**: google/gemma-3-4b-pt
- **Total Probes**: 5668
- **Created**: 2025-11-23

## Layer Distribution

- **layer0**: 10 probes (10/10 successful)
- **layer1**: 257 probes (257/257 successful)
- **layer2**: 1093 probes (1093/1093 successful)
- **layer3**: 1027 probes (1027/1027 successful)
- **layer4**: 3271 probes (665/3271 successful)
- **layer5**: 10 probes (10/10 successful)


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
probe_path = Path('concept_packs/gemma-3-4b-pt_sumo-wordnet-v3/hierarchy/Animal_classifier.pt')
probe = torch.load(probe_path)

# Use for prediction
# (Assuming probe is a trained classifier)
```

### Load All Probes

```python
from src.registry.concept_pack_registry import ConceptPackRegistry

registry = ConceptPackRegistry()
pack = registry.get_pack('gemma-3-4b-pt_sumo-wordnet-v3')

# Access probe paths
hierarchy_dir = pack['pack_path'] / 'hierarchy'
probes = {
    f.stem.replace('_classifier', ''): torch.load(f)
    for f in hierarchy_dir.glob('*_classifier.pt')
}
```

## Files

- `probe_pack.json`: Manifest with metadata
- `hierarchy/`: Directory containing 5668 trained classifier files
- `README.md`: This file

## License

MIT

## References

- Base ontology: SUMO 2003
- Concept pack: sumo-wordnet-v1
- Training framework: HatCat v0.1.0
