#!/usr/bin/env python3
"""Search for writing/content-related concepts in the v4 pack."""
import json

# Load all layers
all_concepts = []
for layer in range(5):
    path = f'concept_packs/sumo-wordnet-v4/hierarchy/layer{layer}.json'
    with open(path) as f:
        layer_data = json.load(f)
        for concept in layer_data.get('concepts', []):
            all_concepts.append(concept)

print(f'Total concepts across all layers: {len(all_concepts)}')

# Search for writing, narrative, content, composition related concepts
keywords = ['writ', 'narrat', 'text', 'author', 'compos', 'story', 'discourse', 'rhetoric',
            'express', 'communic', 'lingu', 'semant', 'speech', 'document', 'content',
            'edit', 'publish', 'prose', 'style', 'genre', 'fiction', 'poem', 'verse',
            'sentence', 'paragraph', 'word', 'phrase', 'grammar', 'syntax', 'symbol',
            'novel', 'essay', 'letter', 'drama', 'literat', 'reading', 'language']

matches = []
for concept in all_concepts:
    term = concept.get('sumo_term', '').lower()
    definition = (concept.get('sumo_definition', '') or concept.get('definition', '') or '').lower()
    lemmas = ' '.join(concept.get('lemmas', [])).lower()

    for kw in keywords:
        if kw in term or kw in definition or kw in lemmas:
            matches.append({
                'term': concept['sumo_term'],
                'definition': (concept.get('sumo_definition', '') or concept.get('definition', '') or '')[:90],
                'layer': concept['layer'],
                'domain': concept.get('domain', 'Unknown')
            })
            break

# Sort by layer then term
matches.sort(key=lambda x: (x['layer'], x['term']))
print(f'Found {len(matches)} writing/content-related concepts:\n')

current_layer = None
for m in matches:
    if m['layer'] != current_layer:
        print(f"\n=== Layer {m['layer']} ===")
        current_layer = m['layer']
    defn = m['definition'].replace('\n', ' ')[:70]
    print(f"  [{m['domain'][:15]:15}] {m['term']}: {defn}...")
