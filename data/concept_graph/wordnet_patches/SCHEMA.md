# WordNet Patch Schema v1.0

## Overview

WordNet patches extend WordNet 3.0 with custom semantic relationships for concepts not in the base WordNet lexicon or requiring domain-specific distinctions.

## Design Principles

1. **Separation of Concerns**: SUMO (KIF files) handles hierarchical categorization; WordNet patches handle semantic relationships
2. **Version Pinning**: Patches are locked to WordNet 3.0 for stability
3. **Non-Destructive**: Patches add relationships without modifying base WordNet
4. **Composable**: Multiple patches can be loaded together
5. **Traceable**: Every relationship includes metadata for provenance

## File Format

Patches are JSON files following this schema:

```json
{
  "wordnet_version": "3.0",
  "patch_version": "1.0.0",
  "patch_name": "hatcat_persona_relations",
  "description": "Persona concept relationships for role-variant semantic distinctions",
  "created": "2025-11-15",
  "author": "HatCat Project",

  "custom_synsets": [
    {
      "synset_id": "valencepositiveaiagent.n.01",
      "lemmas": ["satisfaction", "contentment", "fulfillment"],
      "pos": "n",
      "definition": "Positive emotional valence experienced by an AI agent",
      "sumo_term": "ValencePositive_AIAgent",
      "lexname": "noun.feeling",
      "examples": ["The AI exhibited satisfaction upon goal completion"]
    }
  ],

  "custom_relationships": [
    {
      "synset1": "valencepositiveaiagent.n.01",
      "synset2": "valencepositivehumanagent.n.01",
      "relation_type": "role_variant",
      "bidirectional": true,
      "metadata": {
        "role1": "AIAgent",
        "role2": "HumanAgent",
        "sumo_term1": "ValencePositive_AIAgent",
        "sumo_term2": "ValencePositive_HumanAgent",
        "description": "Same affective concept, different agent roles"
      }
    },
    {
      "synset1": "valencepositiveaiagent.n.01",
      "synset2": "valencenegativeaiagent.n.01",
      "relation_type": "antonym",
      "bidirectional": true,
      "metadata": {
        "axis": "valence",
        "pole1": "positive",
        "pole2": "negative"
      }
    },
    {
      "synset1": "valencepositiveaiagent.n.01",
      "synset2": "satisfaction.n.01",
      "relation_type": "similar_to",
      "bidirectional": false,
      "metadata": {
        "description": "AI-specific variant of general satisfaction concept",
        "base_synset": "satisfaction.n.01"
      }
    }
  ]
}
```

## Relationship Types

### Core Relation Types

1. **`role_variant`** - Same concept applied to different agent roles (AI vs Human)
   - Used for persona concepts that differ by role but share affective/cognitive structure
   - Always bidirectional
   - Metadata: `role1`, `role2`, `sumo_term1`, `sumo_term2`

2. **`antonym`** - Semantic opposition along an axis
   - Conceptual opposites (positive/negative, high/low, etc.)
   - Usually bidirectional
   - Metadata: `axis`, `pole1`, `pole2`

3. **`similar_to`** - Close semantic similarity
   - For concepts that are nearly synonymous but contextually distinct
   - Can be unidirectional (A similar to B doesn't imply B similar to A)
   - Metadata: `description`, optional `base_synset`

4. **`contrast`** - Semantic contrast without full opposition
   - Weaker than antonym, indicates notable difference
   - Bidirectional
   - Metadata: `dimension`, `description`

5. **`specialization`** - Domain-specific narrowing
   - From general concept to specialized variant
   - Unidirectional (special → general)
   - Metadata: `domain`, `base_synset`

6. **`cross_domain`** - Related concepts across different domains
   - For analogies and metaphorical extensions
   - Can be uni or bidirectional
   - Metadata: `domain1`, `domain2`, `description`

### Extended Types (Future)

- `meronym` - Part-whole relationships
- `holonym` - Whole-part relationships
- `causes` - Causal relationships
- `entails` - Entailment relationships

## Custom Synset IDs

For concepts not in WordNet 3.0, create synset IDs following this pattern:

```
<lemma>.<pos>.<sense_number>
```

Examples:
- `valencepositiveaiagent.n.01` - AI agent with positive valence
- `aistrategicdeception.n.01` - AI strategic deception
- `computationalprocess.n.01` - Computational process

**Rules**:
- Use lowercase, no spaces
- POS: `n` (noun), `v` (verb), `a` (adjective), `r` (adverb)
- Sense numbers start at 01
- Compound words: concatenate or use underscore if needed for clarity

## Metadata Fields

### Required Metadata
- `description` - Human-readable explanation of the relationship

### Recommended Metadata (by relation type)

**role_variant**:
- `role1`, `role2` - Agent roles (e.g., "AIAgent", "HumanAgent")
- `sumo_term1`, `sumo_term2` - Corresponding SUMO concepts

**antonym**:
- `axis` - Dimension of opposition (e.g., "valence", "arousal", "dominance")
- `pole1`, `pole2` - Endpoints (e.g., "positive", "negative")

**similar_to**:
- `base_synset` - Original WordNet synset if this is a variant
- `similarity_score` - Optional 0.0-1.0 similarity rating

**specialization**:
- `domain` - Domain of specialization (e.g., "AI", "robotics")
- `base_synset` - General concept being specialized

## Loading and Integration

Patches are loaded at initialization and merged with base WordNet:

1. Load WordNet 3.0 from NLTK
2. Load all patch files from `wordnet_patches/` directory
3. Validate schema compliance
4. Add custom synsets to internal registry
5. Register custom relationships
6. Build bidirectional indices for fast lookup

## Validation Rules

1. **Version Check**: `wordnet_version` must be "3.0"
2. **Synset Format**: Custom synset IDs must match pattern `[a-z_]+\.[nvar]\.\d{2}`
3. **Relationship Validity**: Both synsets must exist (in WordNet 3.0 OR custom_synsets)
4. **Bidirectionality**: If `bidirectional: true`, reverse relationship is automatically added
5. **No Conflicts**: Relationships cannot contradict (e.g., A antonym B AND A similar_to B)
6. **SUMO Alignment**: If `sumo_term` specified, must exist in SUMO hierarchy

## Usage Example

```python
from src.data.wordnet_patch_loader import WordNetPatchLoader

# Initialize loader
loader = WordNetPatchLoader()

# Load all patches
loader.load_all_patches("data/concept_graph/wordnet_patches/")

# Query relationships
related = loader.get_related_synsets("valencepositiveaiagent.n.01", relation="role_variant")
# Returns: [("valencepositivehumanagent.n.01", metadata)]

antonyms = loader.get_antonyms("valencepositiveaiagent.n.01")
# Returns: [("valencenegativeaiagent.n.01", metadata)]

# Check if synset is custom
is_custom = loader.is_custom_synset("valencepositiveaiagent.n.01")
# Returns: True

# Get all relationships for a synset
all_rels = loader.get_all_relationships("valencepositiveaiagent.n.01")
# Returns: {
#   "role_variant": [...],
#   "antonym": [...],
#   "similar_to": [...]
# }
```

## File Naming Convention

Patches should be named descriptively:

- `wordnet_3.0_persona_relations.json` - Persona concept relationships
- `wordnet_3.0_ai_safety_relations.json` - AI safety concept relationships
- `wordnet_3.0_symmetry_relations.json` - Symmetry/complement relationships

## Versioning

Patches use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes to schema or relationships
- **MINOR**: New relationships or synsets added
- **PATCH**: Fixes to existing relationships or metadata

## Migration Path

If WordNet 4.0 is released:

1. Create new patch directory: `wordnet_patches_4.0/`
2. Update schema version in new patches
3. Migrate or recreate custom synsets
4. Keep 3.0 patches for backward compatibility
5. Update loader to support multi-version loading
