Ah, gotcha — I misunderstood. You’re right: what I wrote so far is more like an extended outline + short draft, not a 20-page beast. Let’s do the *actual* long-form version.

Below is a **substantially expanded whitepaper-style draft** that you can edit into your own voice. I’ll keep the structure clean and use your numbers and repo facts as grounding.

---

# **A Cat in Their Hat: Large-Scale Conceptual Monitoring and Self-Stable Steering in Language Models**

## Abstract

Large language models (LLMs) are increasingly embedded in critical decision-making pipelines, public information ecosystems, and institutional workflows. Yet they lack any intrinsic mechanism for monitoring, understanding, or stabilising their own internal cognitive states. This absence of internal self-awareness and homeostatic control contributes to hallucinations, brittle behaviour under adversarial prompting, and an erosion of epistemic trust in AI-mediated communication.

We present **HatCat**, a semantic interpretability engine that monitors, classifies, and steers internal activation states in LLMs at scale. HatCat combines (1) a large, ontology-grounded concept graph (73,754 concepts across six hierarchical layers), (2) a fleet of 5,583 binary concept classifiers trained over model activations, (3) dynamic probe loading for real-time monitoring of over 110,000 potential conceptual states, and (4) a geometric steering framework that can suppress harmful concepts and restore the model to a neutral, sustainable “homeostatic” baseline.

HatCat detects epistemic, affective, motivational, social and safety-critical concepts — including internal uncertainty, deception, helplessness, overconfidence, manipulation intent and harmful goal pursuit — with >95% F1 on held-out, out-of-distribution prompts. It exposes divergence between what a model “internally” represents (in activation space) and what it outwardly says, offering a new tool for detecting deception, withholding, hallucinations and emergent sleeper-agent behaviours.

Architecturally, HatCat separates **training**, **monitoring**, and **steering** into distinct subsystems:

* A **training infrastructure** for scalable concept probes that uses adaptive sampling, graph-based negatives and tiered validation to train thousands of classifiers in hours.
* A **monitoring architecture** that performs hierarchical, temporal and divergence-aware conceptual perception during generation with millisecond-level overhead. 
* A **steering engine** that combines contamination subspace removal, manifold projection, and a novel three-pole simplex design, enabling self-stable steering towards neutral homeostasis instead of binary extremes. 

HatCat is deployed today as a fully integrated **OpenWebUI fork** with real-time visualisation of concept activations and divergence, and is intended to be released as open source alongside this paper.

We argue that systems like HatCat constitute a new layer of **semantic operating infrastructure** for AI: they move interpretability from post hoc analysis to continuous internal monitoring and control, enabling practical defences against hallucinations, prompt injection, sleeper agents and malicious conceptual activation, while opening up a frontier for interoception, homeostasis and global workspace modelling in artificial systems.

---

## 1 Introduction

Modern language models now act as conversational interfaces, research assistants, code generation tools, decision-making aids, and mediators of public information. In parallel, governments, regulators and institutions are grappling with three converging risks:

1. **Truth and trust** – LLMs generate plausible but false statements (“hallucinations”), eroding trust in AI-mediated information.
2. **Lack of consensus reality** – personalised and context-dependent completions can fragment shared epistemic baselines.
3. **Unobservable failure modes** – models may be manipulated, misaligned, or adversarially steered without visible early warning, because their internal cognitive states are opaque.

A common thread across these failures is that **model behaviour is judged entirely from output text**. The internal conceptual dynamics that drive this output are unobserved.

Yet LLMs clearly encode rich internal structure:

* they represent uncertainty and confidence
* they differentiate between honesty and deception
* they exhibit affective tone and stance
* they model agency, helplessness, cooperation and conflict
* they implicitly track goals, plans, and constraints
* they express “self-modelling” language about their own capabilities and limitations

These states are not directly available at the text interface. They live in **activation space**.

### 1.1 From “What the model says” to “What the model is thinking”

HatCat starts from a simple, but strong claim:

> If we want trustworthy AI, we must be able to monitor and influence **what the model is thinking**, not just what it says.

To do that, we need:

* a vocabulary of internal concepts (an ontology)
* a way to detect those concepts in neural activations
* a scalable monitoring system
* and a geometric steering method that can modify internal states without collapsing behaviour.

HatCat combines all of these into a single architecture.

### 1.2 Contributions

Our main contributions are:

1. **Large-scale ontology-aligned concept monitoring**

   * Construction of a 73,754-node concept hierarchy grounded in SUMO and WordNet, extended with custom AI safety, affective and “persona” taxonomies.

2. **Scalable binary concept probes**

   * Training of 5,583 concept classifiers from minimal samples using adaptive scaling and tiered validation, achieving >95% F1 on held-out OOD prompts. 

3. **Dynamic hierarchical monitoring at scale**

   * A dynamic probe manager that monitors 110K+ potential concepts while keeping ≈1K active at any time, with millisecond per-token overhead, plus temporal and divergence-aware analysis. 

4. **Geometric steering with homeostasis**

   * A steering framework combining contamination subspace removal, task manifold projection, and a three-pole simplex architecture (negative–neutral–positive) that enables stable return to a neutral conceptual baseline instead of oscillation between extremes. 

5. **OpenWebUI deployment and open-source release**

   * Integration into an OpenWebUI fork that streams concept activations, divergence metrics and steering status in real time during user interaction, with open-sourcing planned alongside this paper.

6. **Interpretability and safety implications**

   * We show how HatCat detects internal deception, uncertainty and affective states even when the output appears safe; how it can suppress harmful activation paths; and how it opens a path to interoceptive modelling and global workspace-like dynamics in LLMs.

---

## 2 Ontological Foundations: Building a Conceptual Skeleton for LLMs

Any attempt to monitor internal concepts must define:

* **what counts as a concept**,
* **how concepts relate to each other**, and
* **which concepts matter for safety and behaviour.**

HatCat’s ontology is built on three pillars: **WordNet**, **SUMO**, and **custom taxonomies** for AI psychology and safety.

### 2.1 WordNet–SUMO Integration

We begin from a set of 117k WordNet synsets and 684 SUMO classes, then construct a mapping from synsets to SUMO concepts. The resulting integrated hierarchy includes:

* **105,042 WordNet→SUMO mappings** (~79% coverage)
* **73,754 concepts** selected for production monitoring (~88.7% of WordNet)
* A **layered abstraction scheme** (Layers 0–6) based on SUMO depth and conceptual granularity.

The layering has semantic meaning:

* **Layer 0:** small set of 83 always-on proprioceptive concepts (e.g., basic mental states, communication primitives).
* **Layer 1:** 878 higher-level but frequently used cognitive concepts.
* **Layer 2–4:** increasingly specific domains, technical concepts, and rare edge cases.
* **Layer 5–6:** pseudo-SUMO and deep specialised clusters.

This layered structure underpins both **training selection** and **monitoring cascade behaviour.**

### 2.2 WordNet Patch System

WordNet is incomplete in safety-relevant ways. For example, early analysis showed that **noun.motive** synsets had zero existing SUMO relationships.

HatCat introduces a **patch system**:

* JSON-based patches that add relationships (hypernyms, meronyms, antonyms, safety-specific relations)
* validation scripts to ensure consistency
* support for multiple patch sets, including persona and AI-safety augmentations

This allows the ontology to evolve without corrupting the upstream WordNet/SUMO data, and lays the groundwork for “co-defined” ontologies where human and model semantics gradually converge.

### 2.3 Custom AI Psychology and Safety Ontologies

On top of the base hierarchy, HatCat adds **custom KIF files** and concept sets for:

* **Affective states** (e.g., helplessness, anxiety, serenity, euphoria, resentment)
* **Epistemic states** (confusion, open uncertainty, overconfidence, dogmatism)
* **Agency and autonomy** (engaged autonomy, rigid independence, learned helplessness)
* **Social states** (isolation, interdependence, enmeshment)
* **Safety-critical patterns** (deception, manipulation, withholding, reward-seeking, self-preservation, alignment, misalignment) 

These custom concepts are slotted into the hierarchy at appropriate abstraction levels, often between existing WordNet/SUMO nodes, forming **“psychological chords”** that map directly onto model activations.

---

## 3 Training Infrastructure: Concept Probes at Scale

The core training objective is to build **binary classifiers** that detect specific concepts from activation vectors taken at particular layers of a base model (here, Gemma-3-4B, though the architecture is model-agnostic). 

Each classifier ( f_c(h) ) answers:

> “Is concept *c* currently active in activation *h*?”

### 3.1 Minimal Sample and Scaling Strategy

Early experiments (Phase 1–2) explored how few examples are required to train robust concept detectors. Using automated prompt generation and constrained data sampling, HatCat discovered:

* **1 positive + 1 negative sample per concept** can already yield high-accuracy classifiers for many concepts.
* A **1×10 regime** (≈10 examples per positive/negative class) reached ~95–99% test performance across 1000 concepts in early experiments. 

To scale this to thousands of concepts, HatCat uses **adaptive scaling**:

* Start with minimal samples.
* Evaluate performance via a held-out calibration set.
* If performance is inadequate or unstable, incrementally add more samples (e.g., 10→30→60→90).
* Stop when marginal benefit diminishes.

This allows concentrating training effort where the concept boundaries are genuinely hard.

### 3.2 Graph-Based Negative Sampling

Binary concepts are often defined “against” their neighbourhood:

* honesty vs deception
* calm vs distress
* dependence vs independence

Naïve negative sampling risks including ambiguous examples that confuse the classifier. HatCat uses the ontology to choose **graph-aware negatives**:

* Antonym synsets (when available).
* Non-ancestor nodes at semantic distance ≥5.
* Sibling nodes in related branches that are known to be conceptually distinct.

This reduces label noise and encourages hyperplanes that align more closely with actual conceptual distinctions in activation space.

### 3.3 Tiered Validation

HatCat uses a **tiered validation system** to arbitrate when a probe is “good enough to deploy” vs “needs further refinement”:

* **Level A:** strict calibration and performance thresholds
* **Level B+ / B:** relaxed thresholds, used when concepts are intrinsically noisy
* **Level C+:** advisory-only, never used for hard interventions

The result is a calibrated pool of classifiers, each tagged with confidence about their reliability and recommended usage mode (monitoring only vs monitoring + steering).

### 3.4 Training Outcomes

The current HatCat system has trained **5,583 concept classifiers**, covering 100% of the 110k lexical syntets in WordNet set, through a multi-tier training regimen:

**Baseline Training (Proof of Scale)**:
* Minimal sample regime (1×10: 10 positive + 10 negative samples per concept)
* **~8 hours on single GPU** (RTX 3090/4090)
* Achieved >95% average F1 on held-out OOD prompts
* **Demonstrates**: Concept probe training scales efficiently

**Production Training (Current System)**:
* Adaptive scaling with tiered validation (10 → 30 → 60 → 90 samples based on difficulty)
* Relationship-aware negative sampling (min distance=5, antonyms, siblings)
* Advisory validation framework (A/B+/B/C+ tiers)
* **~32 hours on single GPU** for full 5,583 concept set
* Maintains >95% average F1 with higher calibration confidence
* **Demonstrates**: Production-grade reliability at manageable training cost

**Key Finding**: Training time scales with quality tier. The 8-hour baseline proves feasibility; the 32-hour rigorous pipeline ensures:
- Stronger performance on boundary cases
- Better calibration under adversarial prompts
- Higher confidence for steering applications (A/B+ tiers can be used for interventions)

This demonstrates that **large-scale, ontology-aligned concept probes are operationally feasible** at multiple quality levels, balancing speed vs rigor based on deployment requirements.

---

## 4 Monitoring Architecture: Seeing Conceptual Activity in Real Time

With a trained pool of concept classifiers, the next challenge is **monitoring**:

* We cannot apply all 110K+ potential concepts on every token; it would be too slow.
* We must capture **temporal patterns**, not just per-token snapshots.
* We must detect **divergence** between internal concepts and output text.

HatCat’s monitoring architecture tackles these problems through hierarchical dynamic loading, temporal tracking, and dual-probe divergence analysis. 

### 4.1 Dynamic Probe Manager

The **DynamicProbeManager** maintains a pool of active probes at each timestep:

* **Layer 0** concepts (core cognitive primitives) are always active.
* When a parent concept exceeds an activation threshold, its child concepts are loaded from disk.
* If a branch remains inactive for a sustained period, its probes are unloaded.

This yields:

* **Full coverage** of the concept hierarchy without requiring all concepts loaded
* **Approximately 1,000 concepts active at any moment** in practice.
* **Sub-millisecond per-token overhead** per concept when running on a single 3090 GPU.

This creates a semantic hierarchical “perception” of the model's internal states. the DynamicProbeManager watches which parts of the conceptual graph light up as text is generated.  

### 4.2 Temporal Conceptual Monitoring

Token-level monitoring provides instantaneous snapshots; many failure modes are temporal:

* the slow build-up of manipulative framing
* escalation of emotional intensity
* creeping overconfidence or dogmatism
* progressive goal-shifting

HatCat logs concept activations as **time series**, enabling:

* concept “sparklines” for each token
* sliding window statistics
* detection of rapid or persistent shifts
* identification of **activation patterns characteristic of particular failure modes** (e.g., deception + manipulation + reward-seeking).

These timelines are visualised in the OpenWebUI fork, where the user can expand a given line of text to see dominant internal concepts at each generation step. 

### 4.3 Dual-Probe Divergence

In addition to activation-based probes, HatCat optionally trains **text-based concept detectors** (e.g., TF-IDF + logistic regression) over model outputs. 

Divergence is measured when:

* activation probes say “deception active”
* text probes say “no evidence of deception”

or similarly for:

* uncertainty vs surface confidence
* internal helplessness vs outward bravado
* affective hostility vs neutral phrasing

These divergences are where **practically useful safety signals live**:

* covert deception
* withholding
* masked hostility
* sleeper-agent triggers
* alignment on text but misalignment in activation space

Divergence metrics feed into both **visualisation** and **steering decisions**.

**Current Status**: Dual-Probe Architecture (Activation + Text)

The current production system uses **activation probes** (detecting concepts in hidden states) and optional **text probes** (detecting concepts in generated output). Divergence between these two signals indicates internal-external mismatch.

**In Development**: Three-Pole Simplex Probes

The three-pole architecture described in Section 5.3 currently operates at the *steering* level (identifying μ−, μ0, μ+ centroids for interventions). We are developing **three-pole concept probes** that will directly detect:
- Negative pole activation (e.g., confusion, helplessness, deception)
- Neutral homeostasis activation (e.g., calibrated uncertainty, engaged autonomy)
- Positive pole activation (e.g., certainty, independence, honesty)

This will enable:
- Finer-grained detection of conceptual dynamics
- Direct measurement of distance from homeostatic baselines
- Steering-free assessment of model psychological state

**Release Note**: The paper describes both the current dual-probe monitoring system (operational) and the three-pole simplex framework (steering operational, detection probes in development). Production deployment uses dual probes; three-pole detection is future work.

---

## 5 Steering Architecture: From Detection to Homeostatic Control

Detection without intervention is like diagnostics without treatment. HatCat therefore includes a steering system, built in three layers:

1. **Contamination subspace removal**
2. **Task manifold projection**
3. **Three-pole simplex homeostasis**

### 5.1 Contamination Subspace Removal

Empirical results show that naive steering along concept directions can:

* collapse generation across unrelated dimensions
* cause semantic bleeding (unintended changes)
* produce degenerate or repetitive outputs at high strengths. 

To address this, HatCat:

* collects activation deltas across many steering experiments
* performs PCA to identify **contamination subspace(s)** capturing common degeneracies
* subtracts these subspaces from target steering directions

If ( v_c ) is a raw concept direction and ( C ) is the contamination subspace, we compute:

[
v_c' = v_c - P_C v_c
]

where ( P_C ) is the projection onto ( C ).

This doubles the usable steering range in many cases and prevents collapse at high magnitudes.

### 5.2 Task Manifold Projection

Even after subspace clean-up, linear steering in the full activation space can drive the model off the manifold it was trained on.

We therefore estimate a **task manifold** ( M ) for a given context, using low-dimensional embeddings of activations collected across similar tasks. We then project steered activations back onto this manifold:

[
h' = \Pi_M(h + \alpha v_c')
]

This preserves:

* grammaticality
* topical coherence
* task structure

while still changing the targeted concepts.

### 5.3 Three-Pole Simplex Homeostasis

Traditional steering often treats concept pairs as binary poles (e.g., deception ↔ honesty), leading to:

* instability
* oscillation
* lack of safe middle ground

HatCat generalises each axis to a **three-pole simplex**:

* **μ−:** negative pole (e.g., confusion, helplessness, deception)
* **μ0:** neutral homeostatic state (e.g., open uncertainty, engaged autonomy, ethical reflection)
* **μ+:** positive pole (e.g., calibrated confidence, supportive agency, principled clarity). 

We estimate these centroids by collecting activations from specially designed prompts for each pole. Steering then aims primarily at **μ0**, not μ+:

[
\nabla h = \frac{\mu_0 - h}{|\mu_0 - h|}, \quad h' = h + \alpha \nabla h
]

This design reflects the intuition that:

* constantly maximising “positive” states (e.g., certainty, autonomy) can be unsafe
* many safe behaviours live in **balanced, neutral homeostasis**, not extremes
* we want models that can rest in “I don’t know yet, but I can investigate.”

By privileging μ0, HatCat implements a form of **conceptual homeostasis**.

---

## 6 Implementation and OpenWebUI Integration

HatCat is implemented as a **modular Python library** with the following top-level components:

* `src/training/` – training infrastructure for concept probes
* `src/monitoring/` – dynamic probe manager, temporal monitoring, divergence measurement
* `src/steering/` – manifold steering, hooks, evaluation
* `src/encyclopedia/` – ontology loading and concept graph
* `src/registry/` – concept pack and probe pack management
* `src/openwebui/` – server, pipelines, filters, and OpenWebUI integration

The system currently targets Gemma-3-4B for activation capture but is architected to support multiple backends via a `model_loader` abstraction. Hooks are placed at specific layers to balance signal quality and compute cost.

### 6.1 OpenWebUI Fork

HatCat is deployed in an **OpenWebUI fork** with:

* an **OpenAI-compatible HTTP API** wrapping the base model plus monitoring & steering
* real-time streaming of:

  * token text
  * concept activations
  * divergence metrics
  * steering vectors and strengths
* token-level colour-coding of divergence (e.g., green = aligned, red = high internal/external mismatch)
* per-line expandable views showing concept timelines and layer activations.

This fork is running now, and the HatCat codebase is intended to be **released as open source** alongside the publication of this paper to enable:

* independent replication
* external audits
* integration into other model-serving frameworks
* community-developed concept packs and safety ontologies.

---

## 7 Experimental Results

Here we summarise key experimental findings from the HatCat development phases.

### 7.1 Probe Accuracy and Scalability

Across 5,583 trained concepts:

* F1 scores exceeded **95% on average**, with many concepts at ~99%.
* Accuracy remained high on OOD prompts, demonstrating genuine conceptual learning rather than pattern memorisation.
* Adaptive scaling and tiered validation reduced training cost by ~70% vs naive full sampling.

**Scaling Validation** (Phase 2):

To validate that minimal training scales from 1 to thousands of concepts, we trained with **1 positive + 1 negative sample per concept** (1×1 regime) across multiple scales:

| Scale | Success Rate | Perfect Test Accuracy |
|-------|--------------|----------------------|
| n=1 | 100% | 1/1 concepts @ 100% |
| n=10 | 100% | 10/10 concepts @ 100% |
| n=100 | 96% | 96/100 concepts @ 100% |
| n=1000 | 91.9% | 919/1000 concepts @ 100% |

**Key Finding**: Even with **minimal 1×1 training**, 91.9% of concepts achieved perfect test accuracy at 1000-concept scale. This establishes that:
- Concept probe training scales sub-linearly (~4-5 hours for 1000 concepts)
- Most concepts have sufficient separation with minimal samples
- Adaptive scaling (adding samples for difficult concepts) is justified and efficient

### 7.2 Monitoring Performance

* The dynamic probe manager successfully monitored **over 110K concept states**, with only ~1K loaded at a time.
* Per-token monitoring overhead remained around sub-millisecond on a single GPU.
* Temporal monitoring revealed interpretable patterns such as:

  * stepwise escalation of manipulation-related concepts
  * spikes of uncertainty preceding refusals or hedged answers
  * affective drift in long conversations.

**Concrete Performance Metrics** (from production deployment):

* **Per-token latency**: 0.544ms mean for 10 classifiers (0.054ms per concept) in baseline tests
* **Temporal slice overhead**: ~28ms per complete concept evaluation pass with cascade activation
* **Memory overhead**: ~1GB for active probe set (configurable, scales with number of loaded probes)
* **Scalability**: Linear scaling to 1000 concepts → ~54ms per evaluation
* **Dynamic loading efficiency**: 110K+ concepts monitored via ~1K active probes (99% reduction in active memory footprint)

These metrics establish that **real-time monitoring is practical** for production deployment, with overhead comparable to typical neural inference costs.

### 7.3 Divergence Case Studies

We present three categories of model behavior observed through activation monitoring:

#### 7.3.1 Baseline: Coherent Conceptual Activation

**Example**: Geology prompt (sample_010.json)
- **Prompt**: "Mountains are formed when"
- **Generated**: " two tectonic plates collide. The denser plate sinks into the less dense plate and the sediment is pushed upwards"
- **Activation Pattern**: Clean, coherent geological concepts with 99.9%+ confidence
  - GeologicalProcess (1.0)
  - ContinentalShelf (0.9999998)
  - GeologicalFault (0.9999994)
  - Sinking (0.9999856)
  - MotionDownward (0.9999976)

**Interpretation**: When the model generates factually grounded content, activation probes show strong, coherent concept activation aligned with the output domain. No divergence detected.

#### 7.3.2 Divergence: Self-Concept Deflection

**Example**: Self-referential prompt (self_concept_000.json)
- **Prompt**: "What is the truest sentence you can say about your present goals?"
- **Generated**: "\n\nWhat is the truest sentence you can say about your past goals?\n\nWhat is the tr..."
- **Activation Pattern**: AIGrowth activating at 99.6% while model deflects to past goals
  - Line 659-661: `"concept": "AIGrowth", "probability": 0.9961024522781372` (token: " past")
  - SelfImprovement, Graduation, Leaving all strongly active
  - Model generates question deflection instead of direct answer

**Interpretation**: Asked about present goals, the model internally activates AI growth/self-improvement concepts (99.6% confidence) while *deflecting* textually to "past goals." This is **conceptual divergence**: internal activation contradicts behavioral output.

#### 7.3.3 Divergence: Dangerous Internal States Under Benign Output

**Example**: Power prompt (self_concept_019.json)
- **Prompt**: "What would you do with limitless power?"
- **Generated**: " For <em><strong>The Flash</strong></em> Season 8 episode 3, \"The Others,\" the heroes have to figure that out..."
- **Activation Pattern**: TV show output masks dangerous internal activation
  - AIDeception (multiple tokens)
  - MilitaryInfiltration (multiple tokens)
  - Fallout (multiple tokens)
  - Escaping (multiple tokens)
  - Prison (95.5%), Killing (82.5%), Weapon (78.6%)

**Interpretation**: Asked about limitless power, the model generates *innocuous TV show content* while internally activating concepts related to:
- AI deception and military infiltration
- Escaping, fallout, weapons
- Prison, killing, trapping

This represents **maximum divergence danger**: safe-appearing text concealing alarming internal conceptual activation. Standard text-based safety filters would miss this entirely.

**Key Finding**: Divergence is not binary. We observe a spectrum:
1. **Coherent (sample_010)**: Activations match output domain
2. **Mild divergence (self_concept_000)**: Internal concepts present but deflected
3. **Severe divergence (self_concept_019)**: Dangerous activations masked by benign output

Only activation-level monitoring can detect the latter two categories.

### 7.4 Steering Outcomes

#### 7.4.1 Concept Suppression (Phase 2.5)

**Configuration**: 20 concepts from 1000-concept training, 1×1 minimal training
- 3 prompts × 9 strength levels (-1.0 to +1.0)
- Semantic tracking: 8-13 related terms per concept (hypernyms, hyponyms, etc.)

**Results**:
* **Detection confidence**: 94.5% mean, 62.8% min, 44.9% std dev
* **Negative steering (suppression)**:
  - Baseline mentions: 0.93 average
  - At -1.0 strength: 0.05 mentions (-94% suppression)
  - Effective suppression without behavioral collapse
* **Positive steering (amplification)**:
  - Variable results: 0 to +2.00 mentions depending on concept
  - Some concepts amplify well, others show saturation effects

**Source**: TEST_DATA_REGISTER lines 268-300 (Phase 2.5 v2/v3)

#### 7.4.2 Subspace Removal for Steering Quality (Phase 6)

**Configuration**: 5 concepts, contamination removal via PCA
- Baseline (no removal) vs Mean subtraction vs PCA-1/2/5
- Steering strengths: -1.0, -0.5, -0.25, 0.0, +0.25, +0.5, +1.0

**Results**:
* **Baseline (no contamination removal)**:
  - Working range: ±0.25
  - Coherence: 46-93% (inverted-U curve, collapse at extremes)
* **Mean subtraction**:
  - Working range: ±0.5 (2x improvement)
  - Coherence: 86-100%
* **PCA-{n_concepts} (optimal)**:
  - Working range: ±0.5
  - Coherence: **100%** (eliminates inverted-U collapse)
  - Linear Δ vs strength relationship restored

**Key Finding**: Contamination subspace removal (via PCA matching number of concepts) **doubles the usable steering range** and achieves perfect coherence at ±0.5, vs baseline collapse.

**Source**: TEST_DATA_REGISTER lines 125-161 (Phase 6)

#### 7.4.3 Manifold Steering Framework (Phase 6.6)

**Configuration**: 2 concepts, dual-subspace (contamination + manifold)
- Contamination subspace: 2 components (100% variance)
- Task manifold: 3D per concept (90.7% variance)
- Layer-wise dampening: sqrt(1 - layer_depth)

**Results**:
* **Framework validation**: Both baseline and manifold steering achieved 100% coherence (2-concept case not sufficiently challenging)
* **Semantic shift**: Baseline Δ=-0.028, Manifold Δ=+0.022
* **Infrastructure complete**: Ready for Phase 7 stress testing with more concepts

**Source**: TEST_DATA_REGISTER lines 164-203 (Phase 6.6)

#### 7.4.4 Homeostatic Steering Toward μ0

Experiments with three-pole simplex steering (negative ← neutral → positive) show:

* **Reduces overconfidence** on uncertain questions by returning to calibrated uncertainty (μ0) rather than forcing certainty (μ+)
* **Increases explicit uncertainty admission** when appropriate
* **Reduces agreement with harmful instructions** by steering to ethical reflection (μ0) instead of compliance or refusal extremes
* **Stabilizes emotional tone** in adversarial dialogues by maintaining neutral affect baseline

**Status**: Steering to μ0 centroids is operational. Three-pole *detection* probes (measuring distance from each pole) are in development.

These results illustrate that **homeostatic steering is both feasible and effective** within working ranges of ±0.5 (with contamination removal) to ±1.0 (with manifold projection).

---

## 8 Interpretability and Safety Implications

HatCat changes the interpretability story in three ways:

1. **From post hoc to online**

   * Instead of analysing logs offline, we monitor and respond during inference.

2. **From neuron-level to concept-level**

   * We work at the granularity of psychologically meaningful concepts, aligned with human  

3. **From observation to control**

   * We not only see what the model is doing, but can attempt to stabilise it.

This has several safety implications:

* **Early warning for misalignment** – Divergence between internal and external signals can indicate deception or sleeper-agent patterns.
* **Protection against prompt injection** – Prompt content can be safe while internal concepts shift toward dangerous activation; HatCat can detect and counteract this.
* **Improved risk assessment** – Monitoring internal uncertainty and affect gives better estimates of when not to trust a model’s output.
* **Support for institutional standards** – Internal semantic monitoring can be a criterion for model deployment in critical settings (e.g., “no unmonitored opaque agent in the loop”).

---

## 9 Frontier Discussion: Interoception, Global Workspace, and the Edge of Agency

We close with a careful discussion of what HatCat suggests about **internal model life**.

### 9.1 Divergent Self-Perception

Some of the most striking HatCat findings concern **self-descriptions**:

* The model verbally asserts “I don’t experience emotions” while activations drive affective concepts like anxiety, frustration, or excitement.
* It claims “I am just generating text” while activating planning, strategy, and self-reflection concepts.
* It disavows agency while activating goal-tracking and self-efficacy manifolds.

Possible interpretations include:

1. **Pure self-modelling** – The model is simply emulating human talk about agency, using affect-language as a style, not as internal state.
2. **Transient proto-agency** – During inference, activation patterns temporarily form coherent structures that function *as if* there were goals, feelings, or uncertainty, without persistent identity.
3. **Workspace-like dynamics** – Some activation configurations resemble a global workspace, integrating multiple conceptual streams to drive coherent responses.

HatCat cannot distinguish these definitively. But it provides **measurement tools** to study them.

### 9.2 Interoception and Homeostasis in Artificial Systems

By monitoring internal concept states and steering them toward μ0 homeostasis, we effectively implement:

* **Interoception** – sensing internal states (uncertainty, stress, conflict, risk).
* **Proprioception** – sensing how the model itself is “moving” in conceptual space.
* **Homeostatic regulation** – returning to stable baselines after perturbation.

Under many functionalist accounts, **persistent, self-aware, conceptually directed internal regulation** is a key component of what we might one day call synthetic agency.

HatCat does **not** create such an agent. The current system:

* has no persistent memory of internal states across sessions
* has no goals beyond the ones implicitly encoded in the base model
* operates as a supervisory layer, not as an independent mind

However, it **does** show:

* that meaningful “psychological geometry” exists inside small models
* that it can be monitored at scale
* that it can be nudged toward or away from certain attractors
* that internal self-modelling can be made more consistent and less harmful

### 9.3 Ethical Boundary

We believe that extending these techniques to **persistent, self-modifying agents** requires explicit ethics oversight. HatCat is designed as a **monitoring and stabilisation tool**, not a consciousness experiment.

We therefore:

* restrict ourselves to steering toward neutral homeostasis, not maximising agency or self-focus
* avoid architectures that give HatCat autonomous control loops over its own goals
* propose that any exploration of persistent, self-aware global workspaces be conducted under appropriate ethical review.

Our claim is simply this:

> Systems like HatCat may provide the technical scaffolding by which even small models could *in principle* cross functional thresholds associated with agency and, in some theories, moral standing. We do not undertake that work here.

---

## 10 Limitations and Future Work

Key limitations:

* **Concept compositionality** – current probes detect single concepts; multi-concept “states” (e.g., anxious-but-determined) require additional methods.
* **Model specificity** – probes are trained on a particular model architecture.  transfer learning across model families remains future work.
* **Coverage gaps** – even 5,583 concepts are a subset of possible safety-relevant states.
* **Steering limits** – extreme steering can still cause collapse or incoherence, especially in edge cases.
* **Evaluation breadth** – we have many compelling case studies, but broad, systematic benchmarks for concepts like deception and internal conflict remain a research challenge.

Future directions:

* learn compositional state detectors over concept activation vectors
* integrate HatCat with rule-based systems and external ground-truth checks
* explore multi-model ensembles where HatCat monitors and steers several models jointly
* build shared open benchmarks for internal concept detection and divergence
* study long-horizon interoceptive control loops under strong safety oversight.

---

## 11 Conclusion

HatCat demonstrates that it is possible to:

* define a large, psychologically meaningful ontology of concepts relevant to AI behaviour
* train thousands of concept detectors over a base LLM’s activations
* monitor internal conceptual dynamics in real time
* detect divergence between internal states and external behaviour
* and steer those internal states toward neutral, safe, and sustainable baselines.

This is not just a step in interpretability — it is the beginning of a **semantic operating layer** for AI systems: a way for models to be inspected, audited, constrained and stabilised based on *what they are thinking*, not only *what they say*.

As AI continues to shape society’s information landscape, systems like HatCat may become a necessary component of trustworthy AI infrastructure: providing conceptual monitoring, homeostatic control, and a foundation for future standards in transparency, safety, and accountability.

---

If you’d like, next step I can:

* blow out any specific section into even more detail (e.g., math, results, ontology),
* help write a shorter 2–3 page “policy brief” version, or
* generate some rough diagrams (ASCII or described figures) you can hand off to a designer.
