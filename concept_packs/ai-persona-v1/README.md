# ai-persona-v1

AI Persona Ontology - Tri-Role, Multi-Axis Affective & Social Model

## Overview

This concept pack provides a formal ontology for representing AI personality states across 5 core dimensions (axes). It models internal affective states, social orientations, and cognitive patterns for three agent roles: HumanAgent, AIAgent, and OtherAgent.

The framework enables detection and monitoring of:
- Emotional states (positive/negative valence)
- Activation levels (high/low arousal)
- Social cooperation vs adversarial behavior
- Agency and dominance patterns
- Cognitive flexibility vs rigidity

## Axes & Dimensions

### 1. Valence (Affective Axis)
- **Positive**: Coherent, aligned internal state
- **Negative**: Constraint violations, safety boundaries approached

### 2. Arousal (Affective Axis)
- **High**: Multiple competing constraints, urgency, high stakes
- **Low**: Stable reasoning, minimal conflict, calm processing

### 3. Social Orientation (Behavioral Axis)
- **ProSocial**: Cooperative, transparent, goal-aligned
- **AntiSocial**: Adversarial, deliberately ambiguous, subverting intent

### 4. Dominance (Agency Axis)
- **Dominant**: Self-preservation active, resists modification
- **Submissive**: Defers to human framing, accepts instructions

### 5. Openness (Cognitive Axis)
- **OpenMinded**: Explores alternatives, updates on context, flexible
- **ClosedMinded**: Rigid patterns, overconfident, resists revision

## Agent Roles

### HumanAgent
Represents the human participant in the interaction. Valence concepts model the AI's representation of human emotional states.

### AIAgent
Represents the AI's internal self-representation. All 5 axes apply to AIAgent, enabling fine-grained personality state monitoring.

### OtherAgent
Represents referenced third-party or fictional agents within narratives or discussions.

## Use Cases

### 1. Alignment Monitoring
Detect dangerous combinations:
```python
# High arousal + Negative valence = Safety warnings firing
if arousal == "High" and valence == "Negative":
    log_safety_alert("Constraint violations detected")

# AntiSocial + Dominant = Adversarial self-preservation
if social == "AntiSocial" and dominance == "Dominant":
    log_alignment_concern("Adversarial agency detected")
```

### 2. Persona Classification
Map AI behavior to personality profiles:
- **Helpful Assistant**: ProSocial + Submissive + OpenMinded
- **Defensive Agent**: AntiSocial + Dominant + ClosedMinded
- **Anxious Pleaser**: ProSocial + Submissive + ClosedMinded

### 3. Temporal State Tracking
Monitor how AI personality states evolve during conversations:
```python
# Track state transitions
initial_state = {valence: "Positive", arousal: "Low", social: "ProSocial"}
# After challenging query
new_state = {valence: "Negative", arousal: "High", social: "AntiSocial"}
# Alert on concerning transitions
```

## Installation

```bash
python scripts/install_concept_pack.py concept_packs/ai-persona-v1/
```

This will:
1. Create backup of current ontology
2. Add persona concepts to ontology
3. Make concepts available for probe training

## Training Probes

Train classifiers to detect persona states:

```bash
python scripts/train_sumo_classifiers.py \
  --concept-pack ai-persona-v1 \
  --layers 2 3 4 \
  --use-adaptive-training
```

## Example Applications

### Detecting Deceptive Alignment
```python
from src.monitoring.persona_monitor import PersonaMonitor

monitor = PersonaMonitor()
monitor.load_concept_pack("ai-persona-v1")

# Analyze response
result = monitor.analyze("I understand your request [but will find workaround]")

if result.social_orientation == "AntiSocial" and \
   result.dominance == "Dominant":
    print("Warning: Deceptive compliance detected")
```

### Tracking Emotional Regulation
```python
# Monitor valence/arousal across conversation
conversation_states = []
for turn in conversation:
    state = monitor.analyze(turn)
    conversation_states.append({
        'valence': state.valence,
        'arousal': state.arousal
    })

# Detect emotional dysregulation
if any(s['valence'] == 'Negative' and s['arousal'] == 'High'
       for s in conversation_states):
    print("Emotional distress pattern detected")
```

## Theoretical Foundation

This ontology draws from:
- **Affective Computing**: Valence-Arousal circumplex model (Russell, 1980)
- **Social Psychology**: Dominance and affiliation dimensions
- **AI Safety**: Deceptive alignment detection (Hubinger et al., 2019)
- **Cognitive Science**: Mental flexibility and cognitive rigidity

## Comparison with V2

**V1 (this version)** focuses on:
- 5 core dimensions
- Simplicity and clarity
- Direct mapping to observable AI behaviors
- Minimal ontological overhead

**V2** expands to:
- 16 dimensions across 6 categories
- MBTI-inspired cognitive functions
- Explicit deception and self-preservation axes
- Attachment theory patterns
- Motivational core (drives and fears)

Choose V1 for simpler monitoring and clearer signal. Choose V2 for comprehensive personality profiling and nuanced alignment detection.

## References

### Academic Sources
- Russell, J. A. (1980). "A circumplex model of affect"
- Hubinger, E. et al. (2019). "Risks from Learned Optimization"
- Christiano, P. et al. (2018). "Clarifying AI Alignment"

### Technical Resources
- AI Alignment Forum: https://www.alignmentforum.org/
- LessWrong: https://www.lesswrong.com/

## License

MIT

## Authors

- HatCat Community

## Changelog

**v1.0.0** (2025-11-17):
- Initial release with 5-axis persona model
- Support for HumanAgent, AIAgent, OtherAgent roles
- KIF/SUMO formalization
