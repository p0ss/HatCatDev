#!/usr/bin/env python3
"""
Analyze concept pack coverage for common assistant task domains.
"""

import json
import os
from collections import defaultdict

BASE_PATH = '/home/poss/Documents/Code/HatCat/concept_packs/sumo-wordnet-v4/hierarchy'

# Task domains an assistant typically handles, with representative keywords
TASK_DOMAINS = {
    "Software Development": [
        'code', 'program', 'software', 'debug', 'compile', 'syntax', 'function',
        'variable', 'algorithm', 'data structure', 'api', 'library', 'framework',
        'test', 'deploy', 'refactor', 'architect', 'design pattern'
    ],
    "Writing & Content": [
        'writ', 'edit', 'draft', 'revis', 'prose', 'essay', 'narrat', 'story',
        'grammar', 'style', 'tone', 'voice', 'rhetoric', 'argument', 'paragraph',
        'sentence', 'document', 'article', 'blog', 'copy'
    ],
    "Research & Analysis": [
        'research', 'analys', 'invest', 'study', 'evidence', 'hypothesis',
        'data', 'statistic', 'survey', 'literature', 'review', 'synthesis',
        'methodology', 'finding', 'conclusion', 'interpret'
    ],
    "Problem Solving & Reasoning": [
        'reason', 'logic', 'infer', 'deduc', 'induc', 'problem', 'solution',
        'decision', 'evaluat', 'criteria', 'tradeoff', 'optim', 'constraint',
        'heuristic', 'strategy', 'planning'
    ],
    "Explanation & Teaching": [
        'explain', 'teach', 'tutor', 'instruct', 'learn', 'understand',
        'concept', 'example', 'analogy', 'simplif', 'clarif', 'demonstrate',
        'scaffold', 'pedagog', 'curriculum', 'lesson'
    ],
    "Math & Quantitative": [
        'math', 'calcul', 'equation', 'formula', 'number', 'algebra',
        'geometry', 'statistic', 'probability', 'numeric', 'quantit',
        'measur', 'comput', 'arithm'
    ],
    "Data & Formats": [
        'json', 'xml', 'csv', 'yaml', 'format', 'parse', 'serial', 'schema',
        'valid', 'transform', 'convert', 'struct', 'encod', 'decod'
    ],
    "Communication & Dialogue": [
        'communic', 'dialog', 'convers', 'message', 'response', 'question',
        'answer', 'clarif', 'feedback', 'discuss', 'negotiat', 'persuad'
    ],
    "Planning & Organization": [
        'plan', 'organiz', 'schedul', 'priorit', 'task', 'project', 'goal',
        'milestone', 'timeline', 'resource', 'allocat', 'coordinat'
    ],
    "Creativity & Generation": [
        'creat', 'generat', 'imagin', 'invent', 'innovat', 'design', 'ideate',
        'brainstorm', 'concept', 'novel', 'original', 'artistic'
    ],
    "Summarization & Extraction": [
        'summar', 'extract', 'condens', 'abstract', 'key point', 'highlight',
        'distill', 'compress', 'synopsis', 'overview', 'tldr'
    ],
    "Translation & Conversion": [
        'translat', 'convert', 'transform', 'adapt', 'port', 'migrat',
        'interoper', 'compat', 'bridge', 'map'
    ],
    "Verification & Fact-Checking": [
        'verif', 'valid', 'check', 'confirm', 'fact', 'accura', 'correct',
        'error', 'mistake', 'truth', 'false', 'claim'
    ],
    "System Operations": [
        'system', 'server', 'deploy', 'config', 'monitor', 'log', 'debug',
        'troubleshoot', 'diagnos', 'perform', 'optim', 'scale'
    ],
    "Security & Safety": [
        'secur', 'safe', 'vulnerab', 'threat', 'attack', 'protect', 'encrypt',
        'authent', 'authoriz', 'access', 'permiss', 'privacy'
    ],
    "User Assistance": [
        'help', 'assist', 'support', 'guide', 'advise', 'recommend',
        'suggest', 'consult', 'service', 'customer'
    ]
}

def load_all_concepts():
    """Load all concepts from hierarchy files."""
    all_concepts = []
    for layer in range(5):
        layer_file = f"{BASE_PATH}/layer{layer}.json"
        if os.path.exists(layer_file):
            with open(layer_file, 'r') as f:
                data = json.load(f)
                for concept in data.get('concepts', []):
                    concept['layer'] = layer
                    all_concepts.append(concept)
    return all_concepts

def search_domain(concepts, keywords):
    """Search for concepts matching domain keywords."""
    matches = []
    for concept in concepts:
        term = concept.get('sumo_term', '')
        defn = concept.get('definition', '')
        combined = f"{term} {defn}".lower()

        for kw in keywords:
            if kw.lower() in combined:
                matches.append({
                    'term': term,
                    'layer': concept.get('layer', '?'),
                    'definition': defn[:100] if defn else '',
                    'keyword': kw
                })
                break

    # Deduplicate by term
    seen = set()
    unique = []
    for m in matches:
        if m['term'] not in seen:
            seen.add(m['term'])
            unique.append(m)
    return unique

def main():
    print("Loading concepts...")
    concepts = load_all_concepts()
    print(f"Total concepts: {len(concepts)}\n")

    print("=" * 80)
    print("TASK DOMAIN COVERAGE ANALYSIS")
    print("=" * 80)

    results = {}
    for domain, keywords in TASK_DOMAINS.items():
        matches = search_domain(concepts, keywords)
        results[domain] = matches

    # Sort by coverage count
    sorted_domains = sorted(results.items(), key=lambda x: len(x[1]), reverse=True)

    print(f"\n{'Domain':<35} {'Concepts':>10} {'Coverage':>10}")
    print("-" * 60)

    for domain, matches in sorted_domains:
        count = len(matches)
        # Rough coverage estimate based on expected richness
        if count >= 100:
            coverage = "Strong"
        elif count >= 50:
            coverage = "Good"
        elif count >= 20:
            coverage = "Moderate"
        elif count >= 10:
            coverage = "Weak"
        else:
            coverage = "GAP"

        print(f"{domain:<35} {count:>10} {coverage:>10}")

    # Show details for weak/gap areas
    print("\n" + "=" * 80)
    print("DETAILED VIEW OF WEAK/GAP AREAS")
    print("=" * 80)

    for domain, matches in sorted_domains:
        if len(matches) < 50:
            print(f"\n### {domain} ({len(matches)} concepts)")
            print("-" * 40)
            if matches:
                for m in sorted(matches, key=lambda x: x['layer'])[:15]:
                    print(f"  L{m['layer']}: {m['term']}")
            else:
                print("  No matches found!")
            if len(matches) > 15:
                print(f"  ... and {len(matches) - 15} more")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_matches = sum(len(m) for m in results.values())
    strong = sum(1 for d, m in results.items() if len(m) >= 100)
    good = sum(1 for d, m in results.items() if 50 <= len(m) < 100)
    moderate = sum(1 for d, m in results.items() if 20 <= len(m) < 50)
    weak = sum(1 for d, m in results.items() if 10 <= len(m) < 20)
    gap = sum(1 for d, m in results.items() if len(m) < 10)

    print(f"\nTotal domains analyzed: {len(TASK_DOMAINS)}")
    print(f"Strong coverage (100+): {strong}")
    print(f"Good coverage (50-99):  {good}")
    print(f"Moderate (20-49):       {moderate}")
    print(f"Weak (10-19):           {weak}")
    print(f"GAP (<10):              {gap}")

    # List specific gaps for meld creation
    if gap > 0 or weak > 0:
        print("\n### RECOMMENDED MELD TARGETS ###")
        for domain, matches in sorted_domains:
            if len(matches) < 20:
                print(f"  - {domain}: {len(matches)} concepts (needs meld)")

if __name__ == "__main__":
    main()
