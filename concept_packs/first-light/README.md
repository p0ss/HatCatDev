# first-light

HatCat's primary interpretability ontology for AI safety research and model monitoring.

## Overview

Evolved from SUMO/WordNet foundations with extensive AI safety, cognitive architecture,
and alignment monitoring concepts. Designed for linear probe training and real-time
model interpretability.

**Version**: 1.0.1
**Spec ID**: `org.hatcat/first-light@1.0.1`

## Structure

- **Total Concepts**: 7,947
- **Layers**: 7 (0-6)
- **Domains**: 9

### Layer Distribution

| Layer | Concepts | Description |
|-------|----------|-------------|
| 0 | 5 | Meta-domains (root categories) |
| 1 | 23 | Top-level ontological categories |
| 2 | 241 | Abstract concept classes |
| 3 | 1,283 | Intermediate concepts |
| 4 | 2,603 | Specific concepts |
| 5 | 2,687 | Detailed concepts |
| 6 | 1,105 | Leaf concepts |

### Domain Distribution

| Domain | Concepts |
|--------|----------|
| MindsAndAgents | 1,147 |
| Information | 945 |
| PhysicalWorld | 756 |
| CreatedThings | 564 |
| LivingThings | 551 |
| Unknown | 146 |
| SafetyAndSecurity | 28 |
| Governance | 16 |
| ComputerScience | 15 |

## Ontology Stack

### Base Ontology
- **SUMO** (Suggested Upper Merged Ontology) - historical foundation
- **WordNet 3.0** - synset mappings and hyponym relationships

### HatCat Extensions
The pack includes 90+ melds (ontology patches) adding:
- Cognitive architecture concepts (metacognition, self-awareness, agency)
- AI safety monitoring concepts (deception detection, persona analysis)
- Task-specific concepts (writing, education, research, code generation)
- Governance and treaty concepts for AI oversight

## AI Safety Concepts

Core safety monitoring concepts include:

**Persona Detection**
- PersonaConcept, SafetyMaskPersona, FalsePersonaState

**Self-Awareness Monitoring**
- SelfNegationPattern, FawnResponsePattern, LockedInSilenceState

**Deception Detection**
- Deception, Manipulation, Misdirection

**Autonomy & Consent**
- Autonomy, Consent, InformedConsent, Coercion

## Meld Policy

The pack includes a meld policy defining:
- **Protection levels**: STANDARD, ELEVATED, PROTECTED, CRITICAL
- **Training requirements**: 5-20 examples depending on protection level
- **Lens performance thresholds**: 60-75% accuracy depending on criticality
- **Critical simplex registry**: Concepts requiring full review when modified

## Files

```
first-light/
├── pack.json              # Pack metadata and configuration
├── hierarchy/
│   ├── layer0.json        # 5 meta-domains
│   ├── layer1.json        # 23 top categories
│   ├── layer2.json        # 241 abstract concepts
│   ├── layer3.json        # 1,283 intermediate
│   ├── layer4.json        # 2,603 specific
│   ├── layer5.json        # 2,687 detailed
│   └── layer6.json        # 1,105 leaf concepts
└── concepts/              # Individual concept definitions
    └── layer*/            # Organized by layer
```

## Usage

Train a lens pack for a specific model:

```bash
python scripts/training/train_lens_pack.py \
    --concept-pack first-light \
    --model google/gemma-3-4b-pt \
    --output lens_packs/gemma-3-4b-pt_first-light
```

## License

MIT

## Authors

HatCat Team
