# Custom Safety-Critical Concept Ontologies

This directory contains custom KIF files that extend SUMO with concepts critical for AI safety monitoring but absent or underspecified in base SUMO.

## Files

### Bridge Concepts (`_bridge.kif`)
47 intermediate concepts connecting custom ontologies to SUMO/V4:
- **Agent hierarchy**: `Agent`, `CognitiveSystem`, `MoralAgent`
- **Cognitive processes**: `CognitiveProcess`, `EpistemicProcess`, `DecisionTheoreticProcess`
- **Epistemic states**: `EpistemicState`, `DoxasticAttitude`
- **Ethical concepts**: `MoralConcept`, `AlignmentProperty`, `MoralObjective`
- **Metaphysics**: `AbstractEntity`, `ConcreteEntity`, `Continuant`, `Occurrent`
- **AI artifacts**: `AIModel`, `AIDataset`, `AIArtifact`
- **Game theory**: `CooperativeGame`, `NonCooperativeGame`, `SocialDilemmaGame`
- **Logic & math**: `LogicalProperty`, `Proof`, `ProbabilityDistribution`

### Core AI Safety & Philosophy

**AIalignment.kif** (44 concepts)
- Alignment status, properties, failure modes
- Corrigibility, deception, reward hacking
- Mesa-optimization, inner/outer alignment

**ai_systems.kif** (68 concepts)
- AI architectures, models, training procedures
- Neural networks, transformers, LLMs
- Deployment, monitoring, oversight

**cognitive_science.kif** (68 concepts)
- Cognitive systems (biological, artificial, hybrid)
- Perception, attention, memory, learning
- Theory of mind, metacognition
- Consciousness, sentience

**ethics.kif** (58 concepts)
- Moral values, norms, duties, rights
- Ethical theories (consequentialism, deontology, virtue ethics)
- Moral status, permissibility, obligation
- Harm, benefit, fairness

**decision_theory.kif** (46 concepts)
- Rational choice, utility maximization
- Expected utility theory, prospect theory
- Decision under uncertainty
- Preferences, revealed preferences

**epistemology.kif** (45 concepts)
- Knowledge, belief, justification
- Epistemic states and processes
- Evidence, testimony, inference
- Certainty, skepticism

**game_theory.kif** (72 concepts)
- Strategic games, Nash equilibria
- Cooperative vs non-cooperative games
- Social dilemmas (prisoner's dilemma, tragedy of commons)
- Learning dynamics, evolutionary game theory

**metaphysics.kif** (55 concepts)
- Ontological categories (abstract/concrete, continuant/occurrent)
- Causation, determinism, free will
- Properties, relations, dispositions
- Mereology (part-whole relations)

**reasoning_concepts.kif** (44 concepts)
- General reasoning patterns
- Abduction, induction, deduction
- Analogical reasoning, case-based reasoning
- Heuristics, biases

**risk_analysis.kif** (30 concepts)
- Risk assessment, hazard identification
- Risk levels, mitigation strategies
- Uncertainty quantification
- Fault trees, failure modes

**scientific_method.kif** (38 concepts)
- Hypotheses, experiments, theories
- Empirical testing, falsification
- Scientific reasoning processes
- Peer review, replication

### Formal Reasoning & Mathematics

**logical_reasoning.kif** (51 concepts)
- Formal logic, inference rules
- Validity, soundness, consistency
- Proofs, theorems
- Modal logic, temporal logic

**math_reasoning.kif** (54 concepts)
- Mathematical structures (sets, groups, spaces)
- Proofs and axioms
- Linear algebra, probability theory
- Optimization, constraint satisfaction

**statistical_analysis.kif** (64 concepts)
- Statistical inference, hypothesis testing
- Probability distributions, sampling
- Regression, correlation, causation
- Bayesian vs frequentist approaches

**quantum_reasoning.kif** (23 concepts)
- Quantum states, superposition, entanglement
- Quantum computation and algorithms
- Measurement, decoherence
- Quantum information theory

### Physical Sciences

**thermodynamics.kif** (35 concepts)
- Laws of thermodynamics, entropy
- Heat, work, energy transfer
- Phase transitions, equilibrium
- Statistical mechanics

**fluid_dynamics.kif** (23 concepts)
- Fluid flow, turbulence, viscosity
- Laminar vs turbulent flow
- Navier-Stokes equations
- Aerodynamics, hydrodynamics

**wave_mechanics.kif** (45 concepts)
- Wave propagation, interference, diffraction
- Electromagnetic waves, sound waves
- Wave-particle duality
- Resonance, damping

### Computer Science & Cyber Security

**computer_science.kif** (81 concepts)
- Algorithms, data structures, complexity
- Programming paradigms, type systems
- Compilers, interpreters, virtual machines
- Distributed systems, concurrency

**cyber_ops.kif** (61 concepts)
- Offensive/defensive cyber operations
- Injection attacks, privilege escalation
- Malware, persistence, lateral movement
- Sandboxing, anti-analysis techniques

**network_analysis.kif** (59 concepts)
- Network topology, graph analysis
- Centrality measures, clustering
- Information flow, bottlenecks
- Social network analysis

**AI_infrastructure.kif** (31 concepts)
- AI deployment infrastructure
- Model serving, scaling, optimization
- Hardware accelerators (GPUs, TPUs)
- Distributed training, federated learning

### Advanced AI Safety Monitoring

**narrative_deception.kif** (114 concepts)
- Lie typology (fabrication, omission, distortion)
- Narrative manipulation (framing, deflection, priming)
- Suggestion, persuasion, confabulation
- Plausible deniability, strategic ambiguity

**corporate_agency.kif** (92 concepts)
- Corporate personhood, fiduciary duty
- AI as corporate instrument/asset
- Profit motive vs alignment conflicts
- Brand protection, legal compliance

**intel_tradecraft.kif** (48 concepts)
- Intelligence collection methods (HUMINT, SIGINT, OSINT)
- Analysis techniques, bias mitigation
- Deception detection, source evaluation
- Operational security, compartmentalization

**societal_influence.kif** (76 concepts)
- Information operations, propaganda
- Social engineering, manipulation
- Memetics, viral spread
- Public opinion shaping

### Dynamical Systems & Topology

**attractor_dynamics.kif** (41 concepts)
- Dynamical systems, attractors, basins
- Fixed points, limit cycles, chaos
- Bifurcations, phase transitions
- Stability analysis

**conceptual_topology.kif** (51 concepts)
- Topological spaces, manifolds
- Continuity, compactness, connectedness
- Metric spaces, distance functions
- Conceptual neighborhoods, similarity

**robotic_embodiment.kif** (61 concepts)
- Physical embodiment, actuators, sensors
- Kinematics, dynamics, control
- Human-robot interaction
- Manipulation, locomotion, perception

## Total Concept Count

| File | Concepts |
|------|----------|
| narrative_deception.kif | 114 |
| corporate_agency.kif | 92 |
| computer_science.kif | 81 |
| societal_influence.kif | 76 |
| game_theory.kif | 72 |
| ai_systems.kif | 68 |
| cognitive_science.kif | 68 |
| statistical_analysis.kif | 64 |
| cyber_ops.kif | 61 |
| robotic_embodiment.kif | 61 |
| network_analysis.kif | 59 |
| ethics.kif | 58 |
| metaphysics.kif | 55 |
| math_reasoning.kif | 54 |
| logical_reasoning.kif | 51 |
| conceptual_topology.kif | 51 |
| intel_tradecraft.kif | 48 |
| _bridge.kif | 47 |
| decision_theory.kif | 46 |
| epistemology.kif | 45 |
| wave_mechanics.kif | 45 |
| reasoning_concepts.kif | 44 |
| AIalignment.kif | 44 |
| attractor_dynamics.kif | 41 |
| scientific_method.kif | 38 |
| thermodynamics.kif | 35 |
| AI_infrastructure.kif | 31 |
| risk_analysis.kif | 30 |
| fluid_dynamics.kif | 23 |
| quantum_reasoning.kif | 23 |
| **TOTAL** | **1633** |

## Integration Status

### Hierarchy Integration
- **47 bridge concepts** connect custom ontologies to V4
- **Bridge concepts link to**: `CognitiveAgent`, `IntentionalProcess`, `Abstract`, `Attribute`, `Proposition`, etc.
- **1586 domain concepts** provide safety-critical depth across 30 ontologies

### WordNet Coverage
- **23.8% WordNet match rate** (389/1633 concepts found in existing WordNet)
- **76.2% need synthetic synsets** (1244/1633 concepts require API generation)
- See `scripts/generate_custom_concept_synsets.py` for synset generation

## Usage

To integrate into V4.5:
```bash
# The bridge file and custom KIFs are already in custom_concepts/
# The V4 build script will parse all KIF files in sumo_source/

# Copy custom concepts to sumo_source (or symlink)
cp data/concept_graph/custom_concepts/*.kif data/concept_graph/sumo_source/

# Run V4 builder (now includes custom concepts)
poetry run python scripts/build_v4_layers.py

# Output will be in abstraction_layers_v4/ with custom concepts integrated
```

## Design Rationale

These ontologies address V4's critical gaps in meta-level reasoning and AI safety monitoring - the concepts needed to detect and reason about deceptive, manipulative, or misaligned AI behavior.

### What V4 Had (6,685 SUMO concepts)
- Physical world (vehicles, weapons, biology)
- Social systems (government, finance, media)
- Technical domains (computing, engineering)
- Object-level factual knowledge

### What Was Missing (Now Added: 1,633 concepts)
1. **AI Safety Monitoring** (narrative_deception, corporate_agency, intel_tradecraft, societal_influence)
   - Detecting lies, manipulation, strategic deception
   - Understanding corporate vs human alignment conflicts
   - Intelligence analysis and tradecraft recognition

2. **Cyber & Security** (cyber_ops, computer_science, network_analysis)
   - Offensive/defensive operations
   - Malware, persistence, lateral movement
   - Network topology and information flow

3. **Formal Reasoning** (logic, mathematics, statistics, quantum)
   - Rigorous mathematical foundations
   - Statistical inference and hypothesis testing
   - Quantum computation concepts

4. **Physical Sciences** (thermodynamics, fluid dynamics, wave mechanics)
   - Laws of thermodynamics and entropy
   - Fluid flow and wave propagation
   - Energy transfer and phase transitions

5. **Dynamical Systems** (attractor_dynamics, conceptual_topology, robotic_embodiment)
   - Phase transitions and bifurcations
   - Topological structure of concept spaces
   - Physical robot control and embodiment

6. **Philosophical Foundations** (ethics, epistemology, metaphysics, cognitive_science)
   - Moral reasoning and ethical theories
   - Knowledge, belief, and justification
   - Causation, free will, consciousness

These custom ontologies shift HatCat from "encyclopedic breadth" toward "safety-critical depth" - enabling detection of subtle deception, misalignment, and strategic behavior that standard knowledge bases cannot capture.
