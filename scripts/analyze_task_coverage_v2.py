#!/usr/bin/env python3
"""
Analyze concept pack coverage for common assistant task domains.
v2: More precise matching - looks for action/process concepts, not just subject matter.
"""

import json
import os
from collections import defaultdict

BASE_PATH = '/home/poss/Documents/Code/HatCat/concept_packs/sumo-wordnet-v4/hierarchy'

# Task domains with PRECISE concept names we'd expect to find
# Format: domain -> list of specific concept terms we'd want
EXPECTED_CONCEPTS = {
    "Code Generation & Editing": [
        "CodeGeneration", "CodeWriting", "CodeCompletion", "CodeRefactoring",
        "BugFix", "Debugging", "CodeReview", "SyntaxCorrection",
        "FunctionWriting", "ClassDesign", "APIDesign", "CodeExplanation"
    ],
    "Text Composition": [
        "Drafting", "TextComposition", "ParagraphWriting", "Outlining",
        "ContentGeneration", "CopyWriting", "TechnicalWriting", "CreativeWriting",
        "EmailComposition", "ReportWriting", "DocumentCreation"
    ],
    "Summarization": [
        "Summarization", "TextSummarization", "Abstraction", "KeyPointExtraction",
        "ContentCondensation", "TLDRGeneration", "BriefGeneration", "Synopsis"
    ],
    "Question Answering": [
        "QuestionAnswering", "FactRetrieval", "KnowledgeLookup", "InformationProvision",
        "QueryResponse", "ExplanationGeneration", "DefinitionProvision"
    ],
    "Instruction Following": [
        "InstructionFollowing", "TaskExecution", "CommandInterpretation",
        "DirectiveCompliance", "RequestFulfillment", "StepExecution"
    ],
    "Reasoning & Logic": [
        "LogicalReasoning", "Inference", "Deduction", "ProblemSolving",
        "CriticalThinking", "ArgumentAnalysis", "ConsequenceReasoning",
        "HypotheticalReasoning", "CounterfactualReasoning"
    ],
    "Math & Calculation": [
        "MathematicalReasoning", "Calculation", "EquationSolving", "ArithmeticOperation",
        "AlgebraicManipulation", "NumericalComputation", "MathProblemSolving"
    ],
    "Data Transformation": [
        "DataTransformation", "FormatConversion", "Parsing", "Serialization",
        "DataCleaning", "DataExtraction", "SchemaMapping", "DataValidation"
    ],
    "Translation": [
        "Translation", "LanguageTranslation", "CodeTranslation", "Paraphrasing",
        "StyleTransfer", "RegisterShift", "Localization"
    ],
    "Planning & Decomposition": [
        "TaskPlanning", "ProblemDecomposition", "StepGeneration", "WorkflowDesign",
        "GoalDecomposition", "TaskPrioritization", "SequencePlanning"
    ],
    "Comparison & Evaluation": [
        "Comparison", "Evaluation", "Assessment", "Ranking", "Scoring",
        "ProConAnalysis", "TradeoffAnalysis", "CriteriaEvaluation"
    ],
    "Error Detection & Correction": [
        "ErrorDetection", "ErrorCorrection", "MistakeIdentification", "FactChecking",
        "InconsistencyDetection", "ValidationChecking", "ProofReading"
    ],
    "Clarification & Disambiguation": [
        "Clarification", "Disambiguation", "MeaningNarrowing", "ContextResolution",
        "AmbiguityResolution", "IntentClarification", "RequirementElicitation"
    ],
    "Dialogue Management": [
        "DialogueManagement", "TurnTaking", "TopicTracking", "ConversationControl",
        "ContextMaintenance", "ResponseGeneration", "FollowUpHandling"
    ],
    "User Modeling": [
        "UserModeling", "ExpertiseAssessment", "IntentRecognition", "PreferenceModeling",
        "GoalInference", "SkillCalibration", "NeedsAssessment"
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

def check_concept_exists(concepts, term):
    """Check if a concept exists (exact or partial match on term name)."""
    term_lower = term.lower()
    for concept in concepts:
        concept_term = concept.get('sumo_term', '').lower()
        # Exact match or the expected term is contained in the actual term
        if term_lower == concept_term or term_lower in concept_term:
            return concept
    return None

def main():
    print("Loading concepts...")
    concepts = load_all_concepts()
    concept_terms = {c.get('sumo_term', '').lower(): c for c in concepts}
    print(f"Total concepts: {len(concepts)}\n")

    print("=" * 80)
    print("SPECIFIC TASK CONCEPT COVERAGE")
    print("=" * 80)

    results = {}
    for domain, expected in EXPECTED_CONCEPTS.items():
        found = []
        missing = []
        for term in expected:
            match = check_concept_exists(concepts, term)
            if match:
                found.append((term, match.get('sumo_term', '')))
            else:
                missing.append(term)
        results[domain] = {'found': found, 'missing': missing, 'expected': expected}

    # Sort by coverage percentage
    sorted_domains = sorted(
        results.items(),
        key=lambda x: len(x[1]['found']) / len(x[1]['expected']) if x[1]['expected'] else 0,
        reverse=True
    )

    print(f"\n{'Domain':<35} {'Found':>8} {'Expected':>10} {'Coverage':>10}")
    print("-" * 70)

    for domain, data in sorted_domains:
        found_count = len(data['found'])
        expected_count = len(data['expected'])
        pct = (found_count / expected_count * 100) if expected_count else 0

        if pct >= 75:
            status = "GOOD"
        elif pct >= 50:
            status = "PARTIAL"
        elif pct >= 25:
            status = "WEAK"
        else:
            status = "GAP"

        print(f"{domain:<35} {found_count:>8} {expected_count:>10} {pct:>7.0f}% {status}")

    # Show details
    print("\n" + "=" * 80)
    print("DETAILED COVERAGE BY DOMAIN")
    print("=" * 80)

    for domain, data in sorted_domains:
        pct = (len(data['found']) / len(data['expected']) * 100) if data['expected'] else 0
        print(f"\n### {domain} ({pct:.0f}% coverage)")
        print("-" * 50)

        if data['found']:
            print("  FOUND:")
            for expected, actual in data['found'][:5]:
                print(f"    + {expected} -> {actual}")
            if len(data['found']) > 5:
                print(f"    ... and {len(data['found']) - 5} more")

        if data['missing']:
            print("  MISSING:")
            for term in data['missing'][:5]:
                print(f"    - {term}")
            if len(data['missing']) > 5:
                print(f"    ... and {len(data['missing']) - 5} more")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - MELD RECOMMENDATIONS")
    print("=" * 80)

    gaps = []
    partials = []
    for domain, data in sorted_domains:
        pct = (len(data['found']) / len(data['expected']) * 100) if data['expected'] else 0
        if pct < 25:
            gaps.append((domain, pct, data['missing']))
        elif pct < 50:
            partials.append((domain, pct, data['missing']))

    if gaps:
        print("\n### HIGH PRIORITY (< 25% coverage) - Need new melds:")
        for domain, pct, missing in gaps:
            print(f"  {domain}: {pct:.0f}%")
            print(f"    Missing: {', '.join(missing[:4])}...")

    if partials:
        print("\n### MEDIUM PRIORITY (25-50% coverage) - Could benefit from melds:")
        for domain, pct, missing in partials:
            print(f"  {domain}: {pct:.0f}%")
            print(f"    Missing: {', '.join(missing[:4])}...")

    if not gaps and not partials:
        print("\nAll domains have at least 50% coverage of expected concepts.")

if __name__ == "__main__":
    main()
