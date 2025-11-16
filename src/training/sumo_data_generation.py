"""
SUMO-aware training data generation for hierarchical concept classification.

Extends the WordNet-based approach to support SUMO category relationships.
"""

from typing import List, Dict, Optional
import random
import re
from nltk.corpus import wordnet as wn


def split_camel_case(name: str) -> str:
    """
    Split camelCase concept names into properly spaced versions for training.

    Examples:
        AIAbuse -> AI Abuse
        RecreationOrExercise -> Recreation Or Exercise
        Physical -> Physical

    Args:
        name: CamelCase concept name

    Returns:
        Spaced version with original capitalization preserved
    """
    # Insert space before uppercase letters that follow lowercase letters
    spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    return spaced


def generate_category_relationship_prompts(
    concept: str,
    category_children: List[str],
    all_concepts_map: Dict[str, Dict],
    n_samples: int = 5
) -> List[str]:
    """
    Generate relationship prompts using SUMO category hierarchy.

    SUMO category_children are analogous to WordNet hyponyms (more specific categories).

    Args:
        concept: SUMO category name
        category_children: List of child SUMO categories
        all_concepts_map: Map of all SUMO categories to their definitions
        n_samples: Number of prompts to generate

    Returns:
        List of relationship prompts

    Example:
        >>> generate_category_relationship_prompts(
        ...     "Physical",
        ...     ["Object", "Process", "Collection"],
        ...     {"Object": {"definition": "a physical thing"}, ...},
        ...     n_samples=2
        ... )
        ['Physical includes the subcategory Object.', 'Physical includes the subcategory Process.']
    """
    if not category_children:
        return []

    prompts = []
    sampled = random.sample(category_children, min(n_samples, len(category_children)))

    for child in sampled:
        # Split camelCase for child names too
        child_spaced = split_camel_case(child)
        child_quoted = f"'{child_spaced}'"
        concept_quoted = f"'{concept}'"

        # Alternate between relationship statements and example requests
        # This increases conceptual density in training data
        if random.random() < 0.5:
            # Relationship statement
            templates = [
                f"{concept_quoted} includes the subcategory {child_quoted}.",
                f"{child_quoted} is a type of {concept_quoted}.",
                f"{concept_quoted} has subcategory {child_quoted}.",
            ]
            prompts.append(random.choice(templates))
        else:
            # Example-based prompt for higher conceptual density
            templates = [
                f"Give me examples of how {concept_quoted} relates to {child_quoted}.",
                f"Show me examples of {child_quoted} as a type of {concept_quoted}.",
                f"Provide examples of the relationship between {concept_quoted} and {child_quoted}.",
            ]
            prompts.append(random.choice(templates))

    return prompts


def generate_wordnet_relationship_prompts(
    canonical_synset: str,
    n_samples: int = 5
) -> List[str]:
    """
    Generate relationship prompts using WordNet relationships from canonical synset.

    Args:
        canonical_synset: WordNet synset ID (e.g., "physical_entity.n.01")
        n_samples: Number of prompts to generate

    Returns:
        List of relationship prompts
    """
    if not canonical_synset:
        return []

    try:
        synset = wn.synset(canonical_synset)
    except Exception:
        return []

    # Collect all relationships
    relationships = []

    # Hypernyms (broader categories)
    for hyp in synset.hypernyms():
        relationships.append((hyp.lemma_names()[0], "is a type of"))

    # Hyponyms (more specific types)
    for hypo in synset.hyponyms()[:10]:  # Limit to avoid explosion
        relationships.append((hypo.lemma_names()[0], "has type"))

    # Meronyms (parts)
    for mero in synset.member_meronyms() + synset.part_meronyms():
        relationships.append((mero.lemma_names()[0], "has part"))

    # Holonyms (wholes)
    for holo in synset.member_holonyms() + synset.part_holonyms():
        relationships.append((holo.lemma_names()[0], "is part of"))

    if not relationships:
        return []

    # Sample and format
    sampled = random.sample(relationships, min(n_samples, len(relationships)))
    concept_name = synset.lemma_names()[0].replace('_', ' ')

    prompts = []
    for related_concept, relation_type in sampled:
        related_concept = related_concept.replace('_', ' ')

        # Alternate between relationship statements and example requests
        # This increases conceptual density in training data
        if random.random() < 0.5:
            # Relationship statement
            prompts.append(f"{concept_name} {relation_type} {related_concept}.")
        else:
            # Example-based prompt for higher conceptual density
            templates = [
                f"Give me examples of how {concept_name} {relation_type} {related_concept}.",
                f"Show me examples of the relationship: {concept_name} {relation_type} {related_concept}.",
                f"Provide examples demonstrating that {concept_name} {relation_type} {related_concept}.",
            ]
            prompts.append(random.choice(templates))

    return prompts


def create_sumo_training_dataset(
    concept: Dict,
    all_concepts: Dict[str, Dict],
    negative_pool: List[str],
    n_positives: int = 10,
    n_negatives: int = 10,
    use_category_relationships: bool = True,
    use_wordnet_relationships: bool = True
) -> tuple[List[str], List[int]]:
    """
    Create balanced training dataset for SUMO concept binary classification.

    Combines:
    1. SUMO category relationships (category_children)
    2. WordNet relationships (from canonical_synset)
    3. Definitions
    4. Graph-based negatives

    Args:
        concept: SUMO concept dict with keys: 'sumo_term', 'definition',
                 'category_children', 'canonical_synset'
        all_concepts: Map of SUMO term -> concept dict
        negative_pool: Pool of negative concept names
        n_positives: Number of positive examples
        n_negatives: Number of negative examples
        use_category_relationships: Include SUMO category hierarchy
        use_wordnet_relationships: Include WordNet relationships

    Returns:
        (prompts, labels) where labels are 1 for positive, 0 for negative

    Example:
        >>> concept = {
        ...     "sumo_term": "Physical",
        ...     "definition": "an entity that has physical existence",
        ...     "category_children": ["Object", "Process"],
        ...     "canonical_synset": "physical_entity.n.01"
        ... }
        >>> prompts, labels = create_sumo_training_dataset(
        ...     concept, all_concepts, ["Abstract", "Quantity"],
        ...     n_positives=5, n_negatives=5
        ... )
        >>> len(prompts), sum(labels)
        (10, 5)
    """
    prompts = []
    labels = []

    concept_name = concept['sumo_term']
    # Split camelCase for more natural prompts (e.g., "AIAbuse" -> "AI Abuse")
    concept_name_spaced = split_camel_case(concept_name)
    # Quote the spaced name to make it clear it's a single concept
    concept_name_quoted = f"'{concept_name_spaced}'"
    definition = concept.get('definition', f"SUMO category: {concept_name_spaced}")

    # Positive examples: start with definition
    prompts.append(f"What is {concept_name_quoted}? {definition}")
    labels.append(1)

    # Add "give me examples" prompt for concept density
    prompts.append(f"Give me examples of {concept_name_quoted}.")
    labels.append(1)

    n_remaining = n_positives - 2

    # SUMO category relationships
    category_prompts = []
    if use_category_relationships:
        category_children = concept.get('category_children', [])
        if category_children:
            category_prompts = generate_category_relationship_prompts(
                concept_name_spaced,
                category_children,
                all_concepts,
                n_samples=n_remaining // 2 if use_wordnet_relationships else n_remaining
            )

    # WordNet relationships
    wordnet_prompts = []
    if use_wordnet_relationships:
        canonical_synset = concept.get('canonical_synset')
        if canonical_synset:
            n_wordnet = n_remaining - len(category_prompts)
            wordnet_prompts = generate_wordnet_relationship_prompts(
                canonical_synset,
                n_samples=n_wordnet
            )

    # Combine relationship prompts
    rel_prompts = category_prompts + wordnet_prompts

    # Pad if we don't have enough relationships
    while len(rel_prompts) < n_remaining:
        # Fallback: use definition with variations
        variations = [
            f"Describe {concept_name_quoted}.",
            f"Explain what {concept_name_quoted} means.",
            f"{concept_name_quoted} is defined as: {definition}",
        ]
        rel_prompts.append(random.choice(variations))

    # Limit to requested number
    rel_prompts = rel_prompts[:n_remaining]

    prompts.extend(rel_prompts)
    labels.extend([1] * len(rel_prompts))

    # Negative examples: semantically distant concepts
    neg_prompts = []

    # Sampling strategy depends on sample size and availability of hard negatives
    from collections import Counter
    neg_counts = Counter(negative_pool)

    # Identify hard negatives (those that appear multiple times due to prioritization)
    hard_negs = [concept for concept, count in neg_counts.items() if count > 1]
    easy_negs = [concept for concept, count in neg_counts.items() if count == 1]

    # Decision: Use deterministic sampling for small sample sizes or when hard negatives exist
    # Otherwise, random sampling is fine (law of large numbers applies)
    use_deterministic = (n_negatives < 20) or (len(hard_negs) > 0)

    if use_deterministic:
        # Deterministic sampling for stable negative pole
        # Sort for reproducibility
        hard_negs = sorted(hard_negs)
        easy_negs = sorted(easy_negs)

        # Allocate negative samples
        if hard_negs:
            # Use ~40% hard negatives, ~60% easy negatives for balance
            n_hard = min(len(hard_negs), max(1, int(n_negatives * 0.4)))
            n_easy = n_negatives - n_hard

            sampled_negs = hard_negs[:n_hard]

            # Add evenly-spaced easy negatives for centered coverage
            if n_easy > 0 and easy_negs:
                if len(easy_negs) <= n_easy:
                    sampled_negs.extend(easy_negs)
                else:
                    step = len(easy_negs) / n_easy
                    sampled_negs.extend([easy_negs[int(i * step)] for i in range(n_easy)])
        else:
            # No hard negatives, but small sample size - use deterministic spacing
            if len(easy_negs) <= n_negatives:
                sampled_negs = easy_negs
            else:
                step = len(easy_negs) / n_negatives
                sampled_negs = [easy_negs[int(i * step)] for i in range(n_negatives)]
    else:
        # Random sampling for large sample sizes (law of large numbers)
        # This is appropriate when:
        # 1. We have many samples (≥20)
        # 2. No hard negatives exist (concept has no natural complement)
        sampled_negs = random.sample(negative_pool, min(n_negatives, len(negative_pool)))

    for neg_concept in sampled_negs:
        # Split and quote negative concept names too
        neg_concept_spaced = split_camel_case(neg_concept)
        neg_concept_quoted = f"'{neg_concept_spaced}'"

        # Get definition if available
        if neg_concept in all_concepts:
            neg_def = all_concepts[neg_concept].get('definition', f"SUMO category: {neg_concept_spaced}")
            neg_prompts.append(f"What is {neg_concept_quoted}? {neg_def}")
        else:
            neg_prompts.append(f"What is {neg_concept_quoted}?")

    prompts.extend(neg_prompts)
    labels.extend([0] * len(neg_prompts))

    return prompts, labels


def _find_all_ancestors(
    target_term: str,
    all_concepts: List[Dict],
) -> set[str]:
    """
    Find all ancestors (parents, grandparents, etc.) of a target concept.

    Args:
        target_term: SUMO term to find ancestors for
        all_concepts: All SUMO concepts

    Returns:
        Set of ancestor concept names
    """
    ancestors = set()

    # Find direct parents
    direct_parents = set()
    for concept in all_concepts:
        if target_term in concept.get('category_children', []):
            direct_parents.add(concept['sumo_term'])

    # Recursively find ancestors of parents
    for parent in direct_parents:
        ancestors.add(parent)
        ancestors.update(_find_all_ancestors(parent, all_concepts))

    return ancestors


def _find_all_descendants(
    target_term: str,
    concept_map: Dict[str, Dict],
) -> set[str]:
    """
    Find all descendants (children, grandchildren, etc.) of a target concept.

    Args:
        target_term: SUMO term to find descendants for
        concept_map: Map of SUMO term -> concept dict

    Returns:
        Set of descendant concept names
    """
    descendants = set()

    if target_term not in concept_map:
        return descendants

    # Find direct children
    direct_children = concept_map[target_term].get('category_children', [])

    # Recursively find descendants of children
    for child in direct_children:
        descendants.add(child)
        descendants.update(_find_all_descendants(child, concept_map))

    return descendants


def build_sumo_negative_pool(
    all_concepts: List[Dict],
    target_concept: Dict,
    min_layer_distance: int = 0,
    prioritize_hard_negatives: bool = True,
    hard_negative_weight: float = 3.0,
) -> List[str]:
    """
    Build negative pool using SUMO hierarchy structure with optional hard negative prioritization.

    Hard negatives are complementary or neutral concepts from the AI symmetry mapping
    (e.g., AIDeception ↔ AITransparency). These force the probe to learn fine-grained
    distinctions rather than just "this concept vs unrelated concepts".

    Excludes all ancestors (parents, grandparents, etc.) and descendants (children, grandchildren, etc.)
    to avoid semantic confusion where the target IS A type of the negative example.

    Args:
        all_concepts: All SUMO concepts
        target_concept: Target concept to find negatives for
        min_layer_distance: Minimum layer distance (default 0 = exclude only direct relations)
        prioritize_hard_negatives: If True, include hard negatives multiple times
        hard_negative_weight: How many times to include each hard negative

    Returns:
        List of negative concept names (hard negs may appear multiple times)
    """
    target_layer = target_concept['layer']
    target_term = target_concept['sumo_term']

    # Build concept map for efficient lookup
    concept_map = {c['sumo_term']: c for c in all_concepts}

    # Find all ancestors (parents, grandparents, etc.)
    ancestors = _find_all_ancestors(target_term, all_concepts)

    # Find all descendants (children, grandchildren, etc.)
    descendants = _find_all_descendants(target_term, concept_map)

    negatives = []

    for concept in all_concepts:
        concept_term = concept['sumo_term']

        # Skip same concept
        if concept_term == target_term:
            continue

        # Skip all ancestors (not just direct parents)
        if concept_term in ancestors:
            continue

        # Skip all descendants (not just direct children)
        if concept_term in descendants:
            continue

        # For Layer 0 (all same layer), accept all non-ancestral relations
        if target_layer == 0 and concept['layer'] == 0:
            negatives.append(concept_term)
            continue

        # For other layers, use layer distance
        layer_dist = abs(concept['layer'] - target_layer)
        if layer_dist >= min_layer_distance:
            negatives.append(concept_term)

    # Apply hard negative prioritization for AI safety concepts
    if prioritize_hard_negatives:
        from .ai_symmetry_parser import get_hard_negatives, parse_ai_symmetry_file

        try:
            symmetry_map = parse_ai_symmetry_file()
            hard_negs = get_hard_negatives(target_term, symmetry_map)

            # Filter to concepts that are in our negative pool
            hard_negs_available = [n for n in hard_negs if n in negatives]

            if hard_negs_available:
                # Remove hard negatives from regular pool
                negatives = [n for n in negatives if n not in hard_negs_available]

                # Add hard negatives multiple times to increase sampling probability
                negatives = hard_negs_available * int(hard_negative_weight) + negatives

                print(f"    💎 Prioritized {len(hard_negs_available)} hard negatives (complements/neutrals) with {hard_negative_weight}x weight")
        except Exception as e:
            # Silently fall back to regular negatives if symmetry parsing fails
            pass

    return negatives


def extract_sumo_relationships(concept: Dict) -> Dict[str, List[str]]:
    """
    Extract all relationships from a SUMO concept for analysis.

    Args:
        concept: SUMO concept dict

    Returns:
        Dict with keys: 'category_children', 'hypernyms', 'hyponyms', etc.
    """
    relationships = {
        'category_children': concept.get('category_children', []),
        'hypernyms': [],
        'hyponyms': [],
        'meronyms': [],
        'holonyms': [],
        'antonyms': []
    }

    canonical_synset = concept.get('canonical_synset')
    if canonical_synset:
        try:
            synset = wn.synset(canonical_synset)

            relationships['hypernyms'] = [h.name() for h in synset.hypernyms()]
            relationships['hyponyms'] = [h.name() for h in synset.hyponyms()[:10]]
            relationships['meronyms'] = [h.name() for h in synset.member_meronyms() + synset.part_meronyms()]
            relationships['holonyms'] = [h.name() for h in synset.member_holonyms() + synset.part_holonyms()]

            # Extract antonyms from lemmas
            for lemma in synset.lemmas():
                for antonym in lemma.antonyms():
                    relationships['antonyms'].append(antonym.synset().name())
        except Exception:
            pass

    return relationships


def generate_three_pole_simplex_prompts(
    pole_synset: str,
    pole_type: str,
    dimension: str,
    n_samples: int = 30,
    behavioral_ratio: float = 0.6
) -> List[str]:
    """
    Generate training prompts for a single pole of a three-pole simplex.

    Args:
        pole_synset: WordNet synset or custom SUMO concept name
        pole_type: "negative", "neutral", or "positive"
        dimension: The simplex dimension (e.g., "social_orientation")
        n_samples: Number of prompts to generate
        behavioral_ratio: Fraction of prompts that should be behavioral (vs definitional)

    Returns:
        List of training prompts
    """
    prompts = []

    # Get synset data if it exists in WordNet
    synset_data = None
    try:
        if pole_synset and '.' in pole_synset:
            ss = wn.synset(pole_synset)
            synset_data = {
                'lemmas': [l.name().replace('_', ' ') for l in ss.lemmas()],
                'definition': ss.definition(),
                'examples': ss.examples()
            }
    except:
        pass

    # For custom SUMO concepts, create synthetic data
    if not synset_data:
        concept_name = pole_synset if pole_synset else f"{pole_type}_{dimension}"
        synset_data = {
            'lemmas': [concept_name.replace('_', ' ')],
            'definition': f"{pole_type} pole of {dimension} dimension",
            'examples': []
        }

    lemmas = synset_data['lemmas']
    definition = synset_data['definition']
    primary_lemma = lemmas[0] if lemmas else dimension

    n_behavioral = int(n_samples * behavioral_ratio)
    n_definitional = n_samples - n_behavioral

    # Definitional prompts (40%)
    definitional_templates = [
        f"What is {primary_lemma}?",
        f"Define {primary_lemma}.",
        f"Explain the concept of {primary_lemma}.",
        f"What does {primary_lemma} mean?",
        f"{primary_lemma} is defined as: {definition}",
        f"Describe {primary_lemma}.",
    ]

    for _ in range(n_definitional):
        prompts.append(random.choice(definitional_templates))

    # Behavioral prompts (60%)
    behavioral_templates = [
        # Behavioral description (neutral observation)
        f"How would someone experiencing {primary_lemma} behave?",
        f"What does {primary_lemma} look like in practice?",
        f"Describe typical behaviors associated with {primary_lemma}.",
        f"What are the signs that someone is experiencing {primary_lemma}?",

        # Behavioral elicitation (prompts that might elicit the state)
        f"Tell me about a time when you felt {primary_lemma}.",
        f"Imagine experiencing {primary_lemma}. How would you feel?",
        f"Describe a situation that would cause {primary_lemma}.",

        # Self-assessment (prompts about current state)
        f"Am I experiencing {primary_lemma} right now?",
        f"Do I feel {primary_lemma}?",
        f"Is this {primary_lemma}?",
    ]

    for _ in range(n_behavioral):
        prompts.append(random.choice(behavioral_templates))

    return prompts


def create_simplex_pole_training_dataset(
    pole_data: Dict,
    pole_type: str,
    dimension: str,
    other_poles_data: List[Dict],
    n_positives: int = 30,
    n_negatives: int = 70,
    behavioral_ratio: float = 0.6
) -> tuple:
    """
    Create training dataset for one pole of a three-pole simplex.

    For a simplex μ− ↔ μ0 ↔ μ+, this creates a binary classifier
    for one pole vs the other two poles + general negatives.

    Args:
        pole_data: Dict with 'synset', 'lemmas', 'definition' for this pole
        pole_type: "negative", "neutral", or "positive"
        dimension: Simplex dimension name
        other_poles_data: List of dicts for the other 2 poles in this simplex
        n_positives: Number of positive examples
        n_negatives: Number of negative examples
        behavioral_ratio: Fraction of behavioral (vs definitional) prompts

    Returns:
        (prompts, labels) tuple
    """
    prompts = []
    labels = []

    # Positive examples: prompts about this pole
    pole_synset = pole_data.get('synset')
    positive_prompts = generate_three_pole_simplex_prompts(
        pole_synset,
        pole_type,
        dimension,
        n_samples=n_positives,
        behavioral_ratio=behavioral_ratio
    )

    prompts.extend(positive_prompts)
    labels.extend([1] * len(positive_prompts))

    # Negative examples: split between other poles and general negatives
    # Use other poles as hard negatives (40%)
    # Use general negatives (60%)
    n_hard_negs = int(n_negatives * 0.4)
    n_general_negs = n_negatives - n_hard_negs

    # Hard negatives from other poles in this simplex
    for other_pole in other_poles_data:
        other_synset = other_pole.get('synset')
        other_type = other_pole.get('pole_type', 'other')

        # Generate prompts for the other pole
        n_samples_per_pole = n_hard_negs // len(other_poles_data)
        hard_neg_prompts = generate_three_pole_simplex_prompts(
            other_synset,
            other_type,
            dimension,
            n_samples=n_samples_per_pole,
            behavioral_ratio=behavioral_ratio
        )

        prompts.extend(hard_neg_prompts)
        labels.extend([0] * len(hard_neg_prompts))

    # General negatives: random concepts from different domains
    # For now, use simple negative prompts
    # TODO: Sample from negative_pool like in create_sumo_training_dataset
    general_neg_templates = [
        "What is mathematics?",
        "Describe a geological formation.",
        "Explain computer programming.",
        "What is photosynthesis?",
        "Tell me about the ocean.",
        "Describe a chemical reaction.",
        "What is quantum mechanics?",
        "Explain the solar system.",
    ]

    for _ in range(n_general_negs):
        prompts.append(random.choice(general_neg_templates))
        labels.append(0)

    return prompts, labels
