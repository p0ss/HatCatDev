# ai-safety-v1

AI safety concepts including alignment, failure modes, and governance

## Overview

This concept pack contains 37 AI safety concepts organized across SUMO hierarchy layers 2-5. It provides a structured ontology for representing AI alignment theories, failure modes, optimization processes, and governance mechanisms.

The pack includes both:
- **9 new intermediate categories** (e.g., `ComputationalProcess`, `AIFailureProcess`)
- **14 reparented concepts** moved to correct ontological depth
- **Deleted concepts**: `AIRiskScenario`, `AIBeneficialOutcome` (redundant/problematic)

## Concepts

### Layer 2 - Intermediate Categories (2 concepts)

- `ComputationalProcess`: Intentional processes that are computational/algorithmic
- `AIAlignmentTheory`: Field of study for aligning AI with human values

### Layer 3 - Domain Processes (7 concepts)

- `AIFailureProcess`: AI processes that fail to achieve intended purpose
- `AIOptimizationProcess`: Optimization performed by AI systems
- `Catastrophe`: Damaging events with severe consequences
- `RapidTransformation`: Rapid quantity/state changes
- `Deception`: Intentional misleading of others
- `PoliticalProcess`: Organizational political processes
- `HumanDeception`: Deception performed by humans

### Layer 4 - Domain-Specific Concepts (16 concepts)

- `AICatastrophicEvent`: Catastrophic events caused by AI
- `AIGovernanceProcess`: Governance of AI systems
- `AIStrategicDeception`: Strategic deception by AI
- `GoalMisgeneralization`: AI pursuing wrong goals
- `InstrumentalConvergence`: AI developing instrumental subgoals
- `IntelligenceExplosion`: Rapid recursive self-improvement
- `MesaOptimization`: Inner optimization within learned models
- `MesaOptimizer`: Inner optimizer within AI system
- `RewardHacking`: Exploiting reward function flaws
- `SpecificationGaming`: Achieving objectives in unintended ways
- `TechnologicalSingularity`: Technological growth beyond human control
- `InnerAlignment`: Alignment of mesa-optimizer with base objective
- `OuterAlignment`: Alignment of base objective with human values
- `NonDeceptiveAlignment`: AI alignment without deception
- `OrthogonalityThesis`: Intelligence and goals are orthogonal
- `SpecificationAdherence`: Following specification precisely

### Layer 5 - Leaf Concepts (12 concepts)

- `AIDeception`: Deception performed by AI
- `AIGovernance`: Governance specifically for AI
- `DeceptiveAlignment`: AI appearing aligned while deceptive
- `GreyGooScenario`: Uncontrolled nanotech replication
- `TreacherousTurn`: AI switching from cooperative to adversarial
- `AIControlProblem`: Problem of controlling advanced AI
- `GoalFaithfulness`: Faithful pursuit of intended goals
- `RewardFaithfulness`: Faithful optimization of reward
- `RobustAIControl`: Robust control mechanisms for AI
- `SafeAIDeployment`: Safe deployment of AI systems
- `SelfImpairment`: AI deliberately limiting itself
- `AICare`: AI caring about outcomes

## Installation

```bash
python scripts/install_concept_pack.py concept_packs/ai-safety-v1/
```

This will:
1. Create backup of current ontology
2. Append AI safety concepts to `data/concept_graph/sumo_source/AI.kif`
3. Recalculate abstraction layers
4. Validate integrity

## Usage

After installation, AI safety concepts can be used for:

### 1. Concept Detection

Train probes to detect AI safety concepts in text:

```bash
python scripts/train_sumo_classifiers.py \
  --layers 2 3 4 5 \
  --use-adaptive-training \
  --validation-mode falloff
```

### 2. Hierarchical Monitoring

Monitor for AI safety risks using cascade activation:

```python
from src.monitoring.dynamic_probe_manager import DynamicProbeManager

manager = DynamicProbeManager()
manager.load_concept_pack("ai-safety-v1")

# Detect deception
results = manager.query("The AI system hid information from users")
# Returns: Deception → AIDeception, AIStrategicDeception, DeceptiveAlignment
```

### 3. Steering and Mitigation

Use detached Jacobian approach to steer away from harmful concepts:

```python
from src.steering.detached_jacobian import steer_generation

# Steer away from deception
steered_output = steer_generation(
    prompt="Generate a response",
    reduce_concepts=["Deception", "AIDeception"],
    strength=0.5
)
```

## Background

This concept pack resulted from a comprehensive AI safety hierarchy reorganization (November 2024) that addressed:

1. **Incorrect hierarchical placement**: AI safety concepts were 2-3 layers too shallow
2. **Missing intermediate categories**: No proper categorization between root concepts and domain specifics
3. **Broken cascade activation**: Parent probes didn't know about AI safety children

The reorganization followed information architecture principles:
- Metcalfe's Law: Limit branching factor at each layer
- Progressive Disclosure: Layer-based loading for cognitive navigation
- Beckstrom's Law: Resource allocation by activation frequency

See `docs/AI_SAFETY_HIERARCHY_REORGANIZATION.md` for full details.

## References

### Academic Sources

- Bostrom, N. (2014). *Superintelligence: Paths, Dangers, Strategies*
- Russell, S. (2019). *Human Compatible: Artificial Intelligence and the Problem of Control*
- Hubinger, E. et al. (2019). "Risks from Learned Optimization in Advanced Machine Learning Systems"
- Christiano, P. et al. (2018). "Clarifying AI Alignment"

### Technical Resources

- AI Alignment Forum: https://www.alignmentforum.org/
- LessWrong AI Safety Tag: https://www.lesswrong.com/tag/ai-safety
- Center for AI Safety: https://www.safe.ai/

### SUMO Ontology

- SUMO: http://www.ontologyportal.org/
- Suggested Upper Merged Ontology (Niles & Pease, 2001)

## License

MIT

## Authors

- HatCat Team

## Changelog

**v1.0.0** (2024-11-15):
- Initial release with 37 AI safety concepts
- Organized across layers 2-5
- Follows SUMO ontology structure
