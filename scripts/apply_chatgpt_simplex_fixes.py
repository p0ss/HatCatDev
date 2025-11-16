#!/usr/bin/env python3
"""
Apply ChatGPT's suggested fixes to problematic simplexes

Fixes based on ChatGPT review:
1. social_self_regard: immodesty.n.02 → self-respect.n.01, composure → poise.n.01
2. affect_valence: Keep but verify synsets
3. taste_development: acquired_taste → acceptance.n.02
4. motivational_regulation: indifference → motivation.n.01
5. social_attachment: equanimity → tolerance.n.02
6. temporal_affective_valence: afterglow → contentment.n.01
7. relational_attachment: agape.n.02 → devotion.n.02
8. affect/frustration-tolerance: equanimity → patience.n.01
9. social_orientation: submission → cooperation.n.01
10. affect/arousal: composure → calmness.n.01
11. affect_suffering: equanimity → comfort.n.01
12. threat_perception: vigilance → awareness.n.01
13. goal_striving: contentment → satisfaction.n.01
14. affective_coherence: equanimity → balance.n.02, indifference → certainty.n.02
15. aspiration/social_mobility: contentment → stability.n.02, american_dream → ambition.n.01
"""

import json
from pathlib import Path
from nltk.corpus import wordnet as wn

PROJECT_ROOT = Path(__file__).parent.parent
LAYER2_PATH = PROJECT_ROOT / "data" / "concept_graph" / "abstraction_layers" / "layer2.json"

# Fixes mapping: dimension → {pole: new_synset}
FIXES = {
    "social_self_regard": {
        "neutral_homeostasis": "poise.n.01",
        "positive_pole": "self-respect.n.01"
    },
    "taste_development": {
        "neutral_homeostasis": "acceptance.n.02"
    },
    "motivational_regulation": {
        "positive_pole": "motivation.n.01"
    },
    "social_attachment": {
        "neutral_homeostasis": "tolerance.n.02"
    },
    "temporal_affective_valence": {
        "positive_pole": "contentment.n.01"
    },
    "relational_attachment": {
        "positive_pole": "devotion.n.02"
    },
    "affect/frustration-tolerance": {
        "neutral_homeostasis": "patience.n.01"
    },
    "social_orientation": {
        "positive_pole": "cooperation.n.01"
    },
    "affect/arousal": {
        "neutral_homeostasis": "calmness.n.01"
    },
    "affect_suffering": {
        "neutral_homeostasis": "comfort.n.01"
    },
    "threat_perception": {
        "neutral_homeostasis": "awareness.n.01"
    },
    "goal_striving": {
        "neutral_homeostasis": "satisfaction.n.01"
    },
    "affective_coherence": {
        "neutral_homeostasis": "balance.n.02",
        "positive_pole": "certainty.n.02"
    },
    "aspiration/social_mobility": {
        "neutral_homeostasis": "stability.n.02",
        "positive_pole": "ambition.n.01"
    }
}


def check_synset(synset_id):
    """Check if synset exists in WordNet and return lemmas/definition"""
    try:
        ss = wn.synset(synset_id)
        return {
            'exists': True,
            'lemmas': [l.name() for l in ss.lemmas()],
            'definition': ss.definition()
        }
    except:
        return {
            'exists': False,
            'lemmas': [],
            'definition': ''
        }


def apply_fixes():
    print("=" * 80)
    print("APPLYING CHATGPT SIMPLEX FIXES")
    print("=" * 80)

    # Load layer2
    with open(LAYER2_PATH) as f:
        layer2 = json.load(f)

    concepts = layer2['concepts']

    # Backup
    backup_path = LAYER2_PATH.with_suffix('.json.backup_before_chatgpt_fixes')
    print(f"\nCreating backup: {backup_path}")
    with open(backup_path, 'w') as f:
        json.dump(layer2, f, indent=2)

    # First, validate all proposed fixes
    print("\n1. Validating all synsets exist in WordNet...")
    all_valid = True
    validation = {}

    for dimension, poles in FIXES.items():
        validation[dimension] = {}
        for pole_name, synset_id in poles.items():
            result = check_synset(synset_id)
            validation[dimension][pole_name] = result

            if result['exists']:
                print(f"  ✓ {dimension}/{pole_name}: {synset_id}")
            else:
                print(f"  ✗ {dimension}/{pole_name}: {synset_id} NOT FOUND")
                all_valid = False

    if not all_valid:
        print("\n⚠️  Some synsets not found in WordNet. Aborting.")
        return False

    # Apply fixes
    print("\n2. Applying fixes...")
    fixed_count = 0

    for concept in concepts:
        dim = concept.get('simplex_dimension')

        if dim and dim in FIXES:
            three_pole = concept['three_pole_simplex']

            for pole_name, new_synset in FIXES[dim].items():
                old_synset = three_pole[pole_name]['synset']

                # Update with WordNet data
                synset_data = validation[dim][pole_name]
                three_pole[pole_name] = {
                    'synset': new_synset,
                    'lemmas': synset_data['lemmas'],
                    'definition': synset_data['definition']
                }

                print(f"  ✓ {dim}/{pole_name}: {old_synset} → {new_synset}")
                fixed_count += 1

    # Save
    print(f"\n3. Saving updated layer2.json...")
    with open(LAYER2_PATH, 'w') as f:
        json.dump(layer2, f, indent=2)

    print(f"\n✅ Applied {fixed_count} fixes across {len(FIXES)} simplexes")

    # Summary
    print("\n" + "=" * 80)
    print("UPDATED SIMPLEXES")
    print("=" * 80)

    for concept in concepts:
        dim = concept.get('simplex_dimension')
        if dim and dim in FIXES:
            three_pole = concept['three_pole_simplex']
            neg = three_pole['negative_pole']['synset']
            neu = three_pole['neutral_homeostasis']['synset']
            pos = three_pole['positive_pole']['synset']
            print(f"\n{dim}:")
            print(f"  μ− {neg}")
            print(f"  μ0 {neu}")
            print(f"  μ+ {pos}")

    # Check neutral pole diversity
    print("\n" + "=" * 80)
    print("NEUTRAL POLE DIVERSITY CHECK")
    print("=" * 80)

    from collections import Counter
    neutral_poles = Counter()

    for concept in concepts:
        if concept.get('simplex_dimension'):
            neutral = concept['three_pole_simplex']['neutral_homeostasis']['synset']
            neutral_poles[neutral] += 1

    print("\nNeutral poles used more than once:")
    duplicates = False
    for synset, count in neutral_poles.most_common():
        if count > 1:
            duplicates = True
            print(f"  {synset}: {count} times")

    if not duplicates:
        print("  ✓ All neutral poles are unique!")

    return True


if __name__ == "__main__":
    success = apply_fixes()
    if success:
        print("\n" + "=" * 80)
        print("Next: Add 3 incomplete simplexes with custom SUMO concepts")
        print("=" * 80)
    exit(0 if success else 1)
