# Concept Pack Creation Workflow

**Quick Start Guide**

---

## Creating a New Concept Pack

### 1. Initialize Pack Structure

```bash
python scripts/create_concept_pack.py my-domain-v1 \
  --description "My domain-specific concepts" \
  --author "Your Name" \
  --license MIT
```

This creates:
```
concept_packs/my-domain-v1/
  pack.json              # Metadata
  concepts.kif           # Empty, ready for editing
  wordnet_patches/       # Empty directory
  layer_entries/         # Empty directory
  README.md              # Template documentation
  LICENSE                # MIT license
```

### 2. Add Concepts

Edit `concept_packs/my-domain-v1/concepts.kif`:

```lisp
;; Layer 2 - Intermediate categories
(subclass MyProcess IntentionalProcess)
(documentation MyProcess EnglishLanguage "Description of MyProcess")

;; Layer 3 - Domain-specific
(subclass MySpecificProcess MyProcess)
(documentation MySpecificProcess EnglishLanguage "More specific process")
```

### 3. (Optional) Add WordNet Patches

Create `concept_packs/my-domain-v1/wordnet_patches/my_relations.json`:

```json
{
  "patch_id": "my-domain-relations-v1",
  "wordnet_version": "3.0",
  "created": "2024-11-15",
  "relationships": [
    {
      "type": "role_variant",
      "source_concept": "Deception",
      "target_concept": "MyDeception",
      "description": "My domain-specific deception",
      "bidirectional": true
    }
  ]
}
```

### 4. Update pack.json Metadata

Edit `concept_packs/my-domain-v1/pack.json`:

```json
{
  "ontology_stack": {
    "domain_extensions": [
      {
        "name": "My Domain Concepts",
        "concepts_file": "concepts.kif",
        "wordnet_patches": ["wordnet_patches/my_relations.json"],
        "new_concepts": 2
      }
    ]
  },
  "concept_metadata": {
    "total_concepts": 2,
    "layers": [2, 3],
    "layer_distribution": {
      "2": 1,
      "3": 1
    }
  }
}
```

### 5. Install Pack

```bash
python scripts/install_concept_pack.py concept_packs/my-domain-v1/
```

This will:
1. ✅ Validate pack structure
2. ✅ Create backup
3. ✅ Append concepts to AI.kif
4. ✅ Copy WordNet patches
5. ✅ Recalculate layers
6. ✅ Record installation

---

## Example: AI Safety Pack

See `concept_packs/ai-safety-v1/` for a complete example.

### Creation Process

```bash
# 1. Create pack structure
python scripts/create_concept_pack.py ai-safety-v1 \
  --description "AI safety concepts including alignment, failure modes, and governance"

# 2. Extract concepts from existing AI.kif
python scripts/extract_ai_safety_concepts.py

# 3. Update metadata (manually edit pack.json)

# 4. Test discovery
python -c "from src.registry.concept_pack_registry import ConceptPackRegistry; \
  r = ConceptPackRegistry(); print(r.get_pack('ai-safety-v1'))"
```

### Installation

```bash
# Install (creates backup automatically)
python scripts/install_concept_pack.py concept_packs/ai-safety-v1/

# Verify
python -c "import json; print(json.load(open('data/installed_packs.json')))"
```

---

## Advanced Workflows

### Extracting from Existing Ontology

If concepts already exist in AI.kif:

```python
# scripts/extract_my_concepts.py
concepts_to_extract = ['MyConcept1', 'MyConcept2']

for concept in concepts_to_extract:
    # Extract definition, write to pack concepts.kif
    pass
```

### Version Updates

```bash
# Update to v1.1.0
cp -r concept_packs/my-domain-v1 concept_packs/my-domain-v1.1

# Edit pack.json version field
# Add new concepts to concepts.kif
# Update metadata

# Install new version
python scripts/install_concept_pack.py concept_packs/my-domain-v1.1/
```

### Packaging for Distribution

```bash
# Create tarball
cd concept_packs
tar -czf my-domain-v1.tar.gz my-domain-v1/

# Install from tarball
python scripts/install_concept_pack.py my-domain-v1.tar.gz
```

---

## Best Practices

### Concept Organization

- **Start high in hierarchy**: Add intermediate categories at layers 2-3
- **Follow ontology depth**: More specific = deeper layers
- **Use existing parents**: Don't create orphan concepts

### Naming

- **Pack ID**: `{domain}-v{major}` (e.g., `medical-v1`, `finance-v2`)
- **Concepts**: `CamelCase` (e.g., `MyDomainProcess`)
- **Files**: `snake_case.ext` (e.g., `my_relations.json`)

### Documentation

- Fill in README.md with:
  - Concept list by layer
  - Usage examples
  - References
- Add English documentation to all KIF concepts
- Include changelog

### Testing

Before distribution:
1. Validate KIF syntax
2. Test installation on clean instance
3. Verify layer recalculation
4. Check concept integration

---

## Troubleshooting

### "Concept not found" during installation

- Ensure parent concepts exist in base SUMO
- Check concept names match exactly (case-sensitive)

### Layer recalculation fails

- Validate KIF syntax: `(subclass Child Parent)` format
- Ensure all parent concepts are defined
- Check for circular dependencies

### Pack not discoverable

- Ensure `pack.json` exists and is valid JSON
- Check `pack_id` field matches directory name
- Verify `concept_packs/` directory location

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `create_concept_pack.py` | Initialize new pack structure |
| `install_concept_pack.py` | Install pack into ontology |
| `extract_ai_safety_concepts.py` | Extract specific concepts from AI.kif |
| `validate_concept_pack.py` | Validate pack before installation (TODO) |
| `package_concept_pack.py` | Create .tar.gz distribution (TODO) |

---

## See Also

- `docs/CONCEPT_PACK_FORMAT.md` - Full format specification
- `docs/WORDNET_PATCH_SYSTEM.md` - WordNet patch schema
- `concept_packs/ai-safety-v1/README.md` - Complete example
