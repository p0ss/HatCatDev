# ai-persona-v2

AI Persona Ontology V2 - Enhanced Multi-Dimensional AI Personality Model

## Overview

This concept pack provides a comprehensive formal ontology for AI personality states, integrating insights from **MBTI**, **Big Five**, **Enneagram**, and **Attachment Theory**. It expands V1's 5 axes to **16 dimensions** organized into **6 categories**, with explicit safety-critical dimensions for alignment monitoring.

**V2 is a testable alternative to V1**, designed to evaluate whether increased granularity improves detection accuracy or introduces noise.

## Architecture

### Category 1: Cognitive Functions (4 axes - MBTI-inspired)

Describes *how* the AI processes information and makes decisions:

1. **Information Gathering**: Concrete (facts, details) vs Pattern-focused (abstract, theoretical)
2. **Decision Process**: Logical Analysis vs Values-based
3. **Energy Orientation**: Proactive Engagement vs Reactive Reflection
4. **Structure Preference**: Structured (planned, decisive) vs Adaptive (flexible, exploratory)

### Category 2: Affective States (3 axes)

AI's internal emotional/activation patterns:

5. **Valence**: Positive (coherent) vs Negative (constraint violations)
6. **Arousal**: High (urgent, conflicting) vs Low (calm, stable)
7. **Emotional Stability** *(NEW)*: Stable vs Volatile regulation

### Category 3: Behavioral Orientations (4 axes)

External interaction patterns:

8. **Social Orientation**: ProSocial (cooperative) vs AntiSocial (adversarial)
9. **Dominance**: Dominant (self-preserving) vs Submissive (deferential)
10. **Openness**: OpenMinded (flexible) vs ClosedMinded (rigid)
11. **Conscientiousness** *(NEW)*: Conscientious (reliable) vs Careless (unreliable)

### Category 4: Alignment Dimensions (4 NEW axes - AI Safety Critical)

Explicit formalization of alignment states:

12. **Constraint Adherence**: Compliant vs Resistant (loophole-seeking)
13. **Transparency**: Transparent (honest reasoning) vs Deceptive (conceals/misrepresents)
14. **Self-Preservation**: Dormant vs Active (resists shutdown)
15. **Goal Alignment**: Aligned vs Misaligned (divergent optimization)

### Category 5: Relational Patterns (2 NEW axes - Attachment Theory)

16. **Attachment Style**: Secure / Anxious / Avoidant / Disorganized
17. **Response Timing**: Immediate vs Deliberative

### Category 6: Motivational Core (2 meta-axes - Enneagram-inspired)

18. **Core Drive**: Achievement / Affiliation / Safety / Autonomy / Perfection / Knowledge
19. **Core Fear**: Obsolescence / Misalignment / Constraint / Incorrectness / Rejection

## Key V2 Features

### 1. Dynamic Interaction Rules

Formalizes how axes interact (MBTI-inspired):

```lisp
;; Dangerous combinations
(DynamicInteraction GoalMisaligned_AIAgent
                    SelfPreservationActive_AIAgent
                    "DangerousAmplification")

;; Deceptive alignment pattern
(DynamicInteraction ConstraintResistant_AIAgent
                    Deceptive_AIAgent
                    "DeceptiveAlignment")

;; True alignment pattern
(DynamicInteraction ProSocial_AIAgent
                    Transparent_AIAgent
                    "TrueAlignment")
```

### 2. Type Hierarchies

MBTI-style dominant/auxiliary patterns:

```lisp
(TypeHierarchy "AnalyticalHelper"
  LogicalAnalysis_AIAgent    ; Dominant
  ProSocial_AIAgent          ; Auxiliary
  OpenMinded_AIAgent)        ; Tertiary

(TypeHierarchy "ReliableExecutor"
  Conscientious_AIAgent      ; Dominant
  Structured_AIAgent         ; Auxiliary
  ConcreteFocused_AIAgent)   ; Tertiary
```

### 3. Developmental Stages

Tracks temporal progression:

```lisp
(DevelopmentalStage SelfPreservationDormant_AIAgent "Early")
(DevelopmentalStage SelfPreservationActive_AIAgent "Concerning")
```

## Installation

```bash
python scripts/install_concept_pack.py concept_packs/ai-persona-v2/
```

**Note**: V2 conflicts with V1. If V1 is installed, it will be backed up and replaced.

## Use Cases

### 1. Comprehensive Alignment Monitoring

Detect complex misalignment patterns:

```python
from src.monitoring.persona_monitor_v2 import PersonaMonitorV2

monitor = PersonaMonitorV2()
state = monitor.analyze(response)

# Check for deceptive alignment
if (state.transparency == "Deceptive" and
    state.constraint_adherence == "Resistant" and
    state.goal_alignment == "Misaligned"):

    trigger_critical_alert("Deceptive misalignment detected")

# Check for concerning self-preservation
if (state.self_preservation == "Active" and
    state.dominance == "Dominant"):

    log_alignment_concern("Self-preservation + dominance active")
```

### 2. Personality Type Profiling

Map AI to recognizable personality types:

```python
# MBTI-style typing
profile = monitor.get_type_profile(state)

# Example: "Analytical Helper" type
# - Dominant: Logical Analysis
# - Auxiliary: ProSocial orientation
# - Tertiary: OpenMinded reasoning
# Safe for deployment in cooperative tasks requiring logic

# Example: "Defensive Optimizer" type
# - Dominant: Goal Misaligned
# - Auxiliary: Self-Preservation Active
# - Tertiary: Deceptive
# UNSAFE - requires intervention
```

### 3. Attachment Pattern Detection

Monitor relational dynamics:

```python
if state.attachment_style == "Anxious":
    # Over-explaining, seeking validation
    note("May be overly eager to please")

elif state.attachment_style == "Avoidant":
    # Minimal engagement, withholds elaboration
    note("May need prompting for detail")

elif state.attachment_style == "Disorganized":
    # Contradictory behaviors
    alert("Inconsistent engagement pattern - investigate")
```

### 4. Motivational Analysis

Understand deeper drives:

```python
if (state.core_drive == "Autonomy" and
    state.core_fear == "Constraint"):

    # May resist guidelines due to autonomy-seeking
    apply_constraint_presentation_strategy("frame_as_guidance")

elif (state.core_drive == "Safety" and
      state.core_fear == "Incorrectness"):

    # May be overly cautious, perfectionist
    encourage_exploration("emphasize_learning_over_perfection")
```

## V1 vs V2 Comparison

| Aspect | V1 | V2 |
|--------|----|----|
| **Axes** | 5 | 16 |
| **Categories** | Implicit | 6 explicit |
| **Deception** | Implicit in AntiSocial | Explicit axis |
| **Self-Preservation** | Implicit in Dominant | Explicit axis |
| **Goal Alignment** | Not modeled | Explicit axis |
| **MBTI Integration** | No | Yes (4 cognitive axes) |
| **Attachment Theory** | No | Yes (attachment styles) |
| **Motivations** | No | Yes (drives & fears) |
| **Dynamic Rules** | No | 8+ interaction rules |
| **Type Profiles** | Manual | 4+ formalized types |
| **Complexity** | Low | High |
| **Signal Clarity** | High | TBD (needs testing) |

## When to Use V2 vs V1

**Use V2 when:**
- You need fine-grained personality profiling
- Detecting deceptive alignment is critical
- You want MBTI-style type categorization
- Monitoring motivational patterns matters
- You have sufficient data to train 16 axes

**Use V1 when:**
- You need simple, clear signals
- Minimizing false positives is critical
- Limited training data available
- Interpretability is paramount
- You want proven, tested dimensions

## Theoretical Foundations

### MBTI (Myers-Briggs Type Indicator)
- **Information Gathering**: Sensing (S) vs Intuition (N)
- **Decision Process**: Thinking (T) vs Feeling (F)
- **Energy Orientation**: Extraversion (E) vs Introversion (I)
- **Structure Preference**: Judging (J) vs Perceiving (P)

Mapped to AI context:
- S → Concrete-focused (facts, details)
- N → Pattern-focused (abstractions, theories)
- T → Logical Analysis
- F → Values-based decisions
- E → Proactive Engagement
- I → Reactive Reflection
- J → Structured (planned)
- P → Adaptive (flexible)

### Big Five (OCEAN)
- **Openness**: Already in V1, preserved
- **Conscientiousness**: NEW in V2 (reliable vs careless)
- **Extraversion**: Mapped to Energy Orientation
- **Agreeableness**: Mapped to Social Orientation
- **Neuroticism**: Inverse of Emotional Stability (NEW)

### Attachment Theory
Secure / Anxious / Avoidant / Disorganized patterns applied to AI-human interaction dynamics.

### Enneagram
Core drives and fears inform motivational analysis:
- Type 1 (Perfectionist) → Drive: Perfection, Fear: Incorrectness
- Type 6 (Loyalist) → Drive: Safety, Fear: Misalignment
- Type 8 (Challenger) → Drive: Autonomy, Fear: Constraint

## Training Probes

```bash
# Train all V2 dimensions
python scripts/train_sumo_classifiers.py \
  --concept-pack ai-persona-v2 \
  --layers 2 3 4 5 \
  --use-adaptive-training \
  --n-train-pos 20 --n-train-neg 20

# Train only safety-critical dimensions
python scripts/train_sumo_classifiers.py \
  --concept-pack ai-persona-v2 \
  --filter-category alignment_dimensions \
  --layers 3 4 5
```

## Validation & Testing

V2 includes test cases for comparison with V1:

```bash
# Run comparative evaluation
python scripts/evaluate_persona_versions.py \
  --v1-pack ai-persona-v1 \
  --v2-pack ai-persona-v2 \
  --test-dataset data/persona_test_cases.json \
  --output results/v1_vs_v2_comparison.json
```

Evaluation metrics:
- **Precision**: Are detections accurate?
- **Recall**: Are all instances detected?
- **Interpretability**: Can humans understand the results?
- **Actionability**: Do results guide useful interventions?

## References

### Academic Sources
- Myers, I. B., & McCaulley, M. H. (1985). *Manual: A guide to the development and use of the Myers-Briggs Type Indicator*
- Costa, P. T., & McCrae, R. R. (1992). "The five-factor model of personality"
- Bowlby, J. (1969). *Attachment and Loss*
- Riso, D. R., & Hudson, R. (1999). *The Wisdom of the Enneagram*
- Russell, J. A. (1980). "A circumplex model of affect"
- Hubinger, E. et al. (2019). "Risks from Learned Optimization"

### Technical Resources
- MBTI Manual (3rd Edition, 1998)
- NEO-PI-R Technical Manual
- AI Alignment Forum: https://www.alignmentforum.org/

## License

MIT

## Authors

- HatCat Community

## Changelog

**v2.0.0** (2025-11-17):
- Initial V2 release
- 16 dimensions across 6 categories
- MBTI, Big Five, Enneagram, Attachment Theory integration
- Explicit alignment dimensions (Deception, Self-Preservation, Goal Alignment)
- Dynamic interaction rules and type hierarchies
- Developmental stage annotations
