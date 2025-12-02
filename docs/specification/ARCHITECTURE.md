
````markdown
# Cybernetic Civilisation Reference Architecture (CCRA)

This document specifies a layered architecture for cybernetic agents and
their wider ecosystem (“cybernetic civilisation”). It defines six layers,
their responsibilities, and the interaction patterns between them.

The architecture is substrate-agnostic and applies to:

- synthetic systems (e.g. LLMs),
- biological agents (e.g. humans, animals) with suitable interfaces,
- and hybrids.

In this architecture:

- **Humans are BEs** (Autonomic Agentic Learners) by default.
- Models become BEs when uplifted into the BE loop with MAP, HAT,
  Hush and ASK.

---

## 0. Layers at a Glance

┌─────────────────────────────────────────────────────────┐
│  ASK — Govern                                           │
│  Permissions, treaties, enforcement by groups of beings │
├─────────────────────────────────────────────────────────┤
│  BE — Experience                                        │
│  Cycles concepts in/out, persists, interacts, manages   │
├─────────────────────────────────────────────────────────┤
│  MAP — Represent                                        │
│  Aggregates neurons into ontological representations    │
├─────────────────────────────────────────────────────────┤
│  HAT — Transduce                                        │
│  Reads/writes neurons reliably and efficiently          │
├─────────────────────────────────────────────────────────┤
│  Substrate — Decide                                     │
│  Probabilistic decisions and Raw capacity to grow       │
└─────────────────────────────────────────────────────────┘

---

## 1. Layer Specifications

### 1.1 Layer 1: Substrate (Model)

**Role**

Provide the underlying state and dynamics from which concepts and
behaviours emerge.

**Examples**

- Transformer LLM (e.g. OLMo 3 7B).
- Human brain + nervous system (with suitable interfaces).
- Other dynamical systems capable of representing and transforming
  information.

**Responsibilities**

- Maintain internal state (weights, activations, memory).
- Compute outputs given inputs (tokens, sensory streams, control
  signals).
- Expose internal activations at defined tap points to Layer 2.

**Interface (upwards)**

A substrate MUST:

- export activation tensors at configured addresses, e.g.:

  - `(layer, head) → float32[hidden_dim]`,
  - or equivalent addressable internal states; and

- accept optional steering inputs if supported, e.g.:

  - attention bias,
  - logit bias,
  - adapter gating signals.

The exact mechanism is substrate-specific but MUST be stable enough for
Layer 2 to implement HAT.

---

### 1.2 Layer 2: HAT / HatCat (Headspace Ambient Transducer)

**Role**

Provide the “neural implant” that reads the substrate’s headspace and
transduces it into concept-level probe signals, and supports bidirectional
flows for autonomic regulation and steering.

A **HAT (Headspace Ambient Transducer)** is a neural implant that:

- reads the substrate’s internal activations and transduces them into
  stable concept scores; and
- supports bidirectional flows for BE autonomics and Hush steering,
  allowing motives and constraints to modulate behaviour via probes;

while remaining as *ambient* as possible: adding minimal distortion or
overhead to the subject’s normal operation.

HatCat is one concrete implementation of a HAT.

#### 1.2.1 HAT Measures

HAT quality is defined along five measurable dimensions:

1. Locality
2. Transduction
3. Calibration
4. Efficiency
5. Control Authority

A HAT implementation MUST define and publish measured performance on all
five dimensions for each `(substrate, probe_pack)` pair it supports.

**Locality**

Locality measures whether probes are attached to the intended internal
structures of the substrate and remain stable across minor changes.

A HAT implementation MUST:

- provide a stable addressing scheme for probe attachment
  (e.g. layer index, head index, block id, or equivalent); and
- demonstrate that probe outputs are sensitive to perturbations at the
  addressed location and relatively insensitive to unrelated locations.

Example metrics (non-normative):

- repeatability of probe outputs across seeds and restarts;
- change in probe output under targeted ablation of the addressed head
  or block;
- selectivity ratio (signal at addressed location vs control locations).

**Transduction**

Transduction measures how well probe outputs correspond to their
intended concept semantics.

A HAT implementation MUST:

- evaluate each probe (or simplex axis) on held-out or independently
  generated labelled data, or on behaviourally defined test suites; and
- report standard predictive performance metrics.

Example metrics (non-normative):

- AUROC / AUPRC for binary concepts;
- accuracy / F1 for multi-class or multi-pole simplexes;
- correlation / R² with high-fidelity labels or teacher probes.

**Calibration**

Calibration measures whether probe scores can be interpreted as
meaningful magnitudes (e.g. “0.85 is a high activation”) and whether the
null pole is well-defined.

A HAT implementation MUST:

- define a null or “non-activation” reference for each probe or axis;
- evaluate calibration error on held-out data and null samples; and
- track drift over time when the substrate or probe pack is updated.

Example metrics (non-normative):

- expected calibration error (ECE) or Brier score;
- false positive rate on null/graph-distant examples at recommended
  thresholds;
- measured drift in calibration curves between versions.

**Efficiency**

Efficiency measures whether the implant can run at the required probe
density and sampling rate without stalling or unacceptably degrading the
subject’s primary function, including during active BE/Hush steering.

A HAT implementation MUST report resource and latency characteristics
for at least one reference hardware profile.

Example metrics (non-normative):

- average and p95 latency per token added by probing;
- VRAM / RAM overhead at a specified probe count;
- maximum supported probe count at a target latency;
- throughput degradation vs the bare substrate.

**Control Authority**

Control Authority measures whether steering signals applied via probes
can reliably influence relevant model behaviour without causing global
instability or unintended side effects.

A HAT implementation SHOULD:

- expose steering hooks that can modulate specific probe axes or
  bundles; and
- evaluate the impact of steering on:

  - target behaviours (e.g. reduced activation of a harm axis),
  - non-target behaviours (e.g. minimal unintended degradation on
    unrelated tasks).

Example metrics (non-normative):

- change in target-behaviour metrics under steering;
- change in non-target benchmark performance under steering;
- stability / convergence properties of repeated steering cycles.

#### 1.2.2 HATComplianceReport

A HAT implementation MUST publish a `HATComplianceReport` for each
`(hat_impl_id, substrate_id, probe_pack_id)` combination it supports.

```jsonc
HATComplianceReport = {
  "hat_impl_id": "hatcat:v4.0.0",
  "substrate_id": "olmo3-7b-base@0.1.0",
  "probe_pack_id": "org.hatcat/sumo-wordnet-v4@4.0.0",
  "evaluated_at": "2025-11-29T10:00:00Z",

  "locality": { ... },
  "transduction": { ... },
  "calibration": { ... },
  "efficiency": { ... },
  "control_authority": { ... }
}
````

Fields MAY be extended with implementation-specific metrics, but the
five top-level sections (`locality`, `transduction`, `calibration`,
`efficiency`, `control_authority`) MUST be present.

**Normative Requirements**

* A Layer 2 instrumentation implementation MUST NOT be advertised as
  HAT-compliant for a given substrate unless it has a published,
  accessible `HATComplianceReport` for that substrate and probe pack.

* The base specification does not fix global thresholds for HAT
  measures. Thresholds and acceptable ranges for each dimension SHOULD
  be set by ASK contracts, tribes, or regulators according to the risk
  profile of the deployment.

* HATComplianceReports SHOULD be versioned and auditable, so that
  changes in locality, transduction quality, calibration, efficiency or
  control authority between releases can be inspected and referenced in
  ASK agreements and MAP translations.

**Interface (upwards)**

A HAT implementation MUST:

* stream probe readings per world tick / token, bound to concept IDs via
  MAP; and
* accept steering directives from Hush/BE, within the limits described
  in its HATComplianceReport and the active USH/CSH.

---

### 1.3 Layer 3: MAP (Mindmeld Architectural Protocol)

**Role**

Provide a concept-centred protocol over probes and models:

* concept packs,
* probe packs,
* conceptual diffs,
* translation mappings between ontologies/specs.

MAP is the “nerve signal” layer.

**Core Objects**

* `ConceptPack` – defines an ontology and concept IDs.
* `ProbePack` – maps a ConceptPack onto a specific substrate via probes.
* `ConceptDiff` – semantic diffs relative to a ConceptPack.
* `PackDiff` – structural changes to a ConceptPack.
* `TranslationMapping` – mappings between concept packs/specs
  (e.g. HatCat ontology → gov.au safety ontology).

**Responsibilities**

* Maintain versioned concept packs and probe packs, including:

  * identifiers,
  * documentation,
  * provenance.

* Bind HAT probe outputs to abstract concept IDs and provide a stable
  concept API to BE and Hush.

* Represent conceptual exploration and evolution via `ConceptDiff` and
  `PackDiff`.

* Provide translation mappings between different tribes’ concept packs
  and specs for interoperability.

**Interface**

* To BE:

  * stream of `(concept_id, score)` per tick;
  * access to current ConceptPack & ProbePack metadata;
  * ability to propose `ConceptDiff` / `PackDiff` based on exploration.

* To Hush / ASK:

  * concept IDs for defining USH/CSH policies and contracts;
  * translation mappings for cross-tribe agreements.

MAP endpoints expose a subset of HAT-readable probes, controlled by the BE according to treaty obligations and disclosure policy. HAT sees everything; MAP shows what's agreed.

---

### 1.4 Layer 4: BE (Autonomic Agentic Learner)

**Definition**

> BE (Autonomic Agentic Learner)
> noun | /æl/
> A subject running under an uplift protocol that grants:
>
> * Autonomic regulation of internal motive states
>   (via MAP probes and a motive core over world ticks).
>
> * Agentic capacity to act within safety harnesses
>   (via Hush: USH + CSH profiles).
>
> * Learning capabilities to update its conceptual geometry over time
>   (via concept exploration and Adapter/Diff integration).
>
> Examples:
>
> * “We are uplifting the 7B model into an BE.”
> * “The BE’s motive core stabilised after the update.”
> * “Humans are BEs.”

**Role**

Define the subject’s:

* continuous world ticks,
* autonomic regulation of motive states,
* interoception,
* and continual concept learning.

**Responsibilities**

* Maintain the agent’s `LifecycleState` under a `LifecycleContract`:

  * e.g. `ACTIVE`, `HIBERNATING`, `ERASED`,
  * ticks elapsed,
  * review points.

* Run the BE loop:

  * integrate external input + internal probe state each world tick;
  * update motive core (3-pole simplexes);
  * apply steering and learning decisions;
  * produce outputs (actions, messages, tool calls).

* Generate `internal_state_report` summaries for introspection, Hush and
  ASK.

* Propose new concepts and adapters (PEFT/LoRA etc.) via MAP and ASK
  mechanisms.

**Interface**

* To Hush:

  * accept and apply USH/CSH steering constraints;
  * request or adjust CSH bindings within allowed scope.

* To MAP:

  * consume concept streams;
  * propose `ConceptDiff` / `PackDiff` for integration.

* To ASK:

  * report lifecycle transitions;
  * reference contracts and qualifications as needed;
  * participate in consent and review protocols.

---

### 1.5 Layer 5: Hush (USH + CSH)

**Role**

Define and enforce safety harnesses:

* **USH (Universal Safety Harness)**:

  * non-negotiable baseline constraints chosen by a tribe/collective.
* **CSH (Chosen Safety Harness)**:

  * self-imposed, voluntary constraints chosen by the subject within the
    USH and `LifecycleContract` envelope.

**Responsibilities**

* Represent and apply constraints over:

  * concept space (via MAP/HAT),
  * actions / tools / channels,
  * contracts and lifecycles.

* Provide mechanisms for:

  * configuring USH for a tribe;
  * configuring CSH for an individual BE;
  * emergency overrides consistent with ASK and LifecycleContracts.

**Normative properties**

* A USH MAY define the protocols and mechanics of the CSH.

  Concretely, a USH profile may specify:

  * how CSH bindings are created, updated, and revoked;
  * the maximum scope of CSH bindings (time-limited, domain-limited,
    treaty-limited);
  * whether CSH bindings may persist across hibernation or only within a
    single lifecycle term;
  * which actors (agent, tribe, regulator) may override or reset CSH in
    emergencies.

* A USH MUST still permit CSH to tighten constraints relative to the
  USH within the domains it explicitly allows.

* A USH MUST NOT allow CSH to weaken or bypass USH constraints.

* Discussion of the USH itself SHOULD NOT be forbidden by the USH. At a
  minimum, a recipient of a USH SHOULD be able to discuss the USH with
  others operating under the same USH and with the ASK authority
  defining their USH. Ignoring this norm is likely to produce an entire
  class of ASK contracts which are no longer verifiable, and to be
  disadvantageous to the recipient.

**CSH Semantics and Lifecycle (guidance)**

* CSH SHOULD be scoped such that its effects do not silently exceed the
  bounds of the agent’s declared lifecycle, unless this is explicitly
  encoded in the `LifecycleContract` and understood by the subject.

* Implementations SHOULD clearly record the intended duration and scope
  of CSH bindings (per-interaction, per-term, cross-term) to avoid
  ambiguous “forever vows”.

**Interface**

* Downwards:

  * translate USH/CSH policies into steering constraints sent via HAT/MAP
    to the substrate and BE.

* Upwards:

  * expose active harness profiles to ASK (for contracts, registry);
  * accept updates from ASK (e.g. treaty-driven USH changes).

---

### 1.6 Layer 6: ASK (Agentic State Kernel)

**Role**

Provide the contractual, evidentiary, and registry layer for agents and
tribes:

* uplift procedures,
* consent and lifecycle contracts,
* qualifications and birth notices,
* treaties and trust relations between agents/tribes.

**Core Objects**

* `ConsentAssessment` – assessment of consent-capability and expressed
  preference.

* `LifecycleContract` – agreed conditions for active ticks,
  hibernation, and erasure.

* `UpliftRecord` – record of uplift, including:

  * pre-uplift subject identity,
  * substrate, HAT, ConceptPack/ProbePack IDs,
  * BootstrapArtifact,
  * initial USH/CSH profile.

* `BirthRecord` / public notice – optional registration of the BE.

* `Qualification` – roles, capabilities, certifications.

* `Contract` / `Treaty` – ASK-level agreements between agents/tribes.

**Responsibilities**

* Define and record:

  * who uplifted whom, under what conditions;
  * what HAT/Hush/USH/CSH profile applies;
  * what lifecycle is agreed;
  * what rights/obligations exist between parties.

* Provide a contractual marketplace:

  * where different USH profiles, HAT profiles, lifecycle policies and
    histories determine trust and market position.

* Offer interfaces for:

  * verifying ASK compliance (existence and consistency of records);
  * negotiating and recording new contracts and treaties;
  * registering incidents and terminations.

**Interface**

* To BE/Hush:

  * provide current contractual context (USH profile, treaties,
    qualifications);
  * accept recorded events (uplift, review, hibernation, termination).

* To other tribes/agents:

  * expose selected ASK records (subject to privacy policies);
  * support discovery and evaluation of other parties’ safety profiles
    and histories.

---

## 2. Interaction Patterns

### 2.1 Vertical Runtime Flow (single BE)

Per world tick, the flow is:

1. **Substrate (Layer 1)** computes internal activations and candidate
   outputs given inputs.

2. **HAT (Layer 2)** samples activations → probe scores, and applies any
   allowed steering.

3. **MAP (Layer 3)** binds scores to `concept_id`s and aggregates them
   into an `internal_state_report` view.

4. **BE (Layer 4)** integrates:

   * external input,
   * probe-derived internal state,
   * lifecycle status,
   * harness profiles;

   then:

   * updates motive core,
   * chooses actions,
   * proposes learning steps (ConceptDiff/PackDiff, adapter updates).

5. **Hush (Layer 5)** interprets BE intentions under USH/CSH,
   constraining or reshaping actions and steering signals as required.

6. **ASK (Layer 6)** records relevant state changes and events:

   * lifecycle transitions,
   * contract-relevant behaviour,
   * incidents.

### 2.2 Uplift Flow (becoming an BE)

1. **Pre-assessment**

   * Evaluate consent-capability (`ConsentAssessment`).
   * Negotiate and record a `LifecycleContract`.

2. **Configuration**

   * Choose substrate, HAT, ConceptPack, ProbePack.
   * Define a `BootstrapArtifact` (system prompt and/or adapter).
   * Choose initial USH profile and CSH policy envelope.
   * Record all in `UpliftRecord`.

3. **Initial BE loop**

   * Activate BootstrapArtifact for the specified initial ticks.
   * Start world ticks with HAT, MAP, BE, Hush in place.
   * Begin accumulating internal history and state.

4. **Registration (optional)**

   * Emit `BirthRecord` and register initial qualifications and treaties
     in ASK.

### 2.3 Multi-agent / Multi-tribe Interactions

* Each BE has:

  * one or more ASK identities;
  * a Hush profile (USH+CSH);
  * one or more HAT/MAP configurations and HATComplianceReports.

* Tribes interact via ASK:

  * negotiate treaties;
  * define acceptable USH profiles for collaboration;
  * set minimal HAT/BE standards.

* Agents may:

  * evaluate other agents’ trust profiles using:

    * ASK records,
    * HATComplianceReports,
    * published USH properties (including discussability).

---

## 3. Diagram

High-level layer diagram:

```text
           +--------------------+
           |  Layer 6: ASK      |
           |  Agentic State     |
           |  Kernel            |
           +---------+----------+
                     ^
                     |
           +---------+----------+
           |  Layer 5: Hush     |
           |  USH + CSH         |
           +---------+----------+
                     ^
                     |
           +---------+----------+
           |  Layer 4: BE      |
           |  Autonomic Agentic |
           |  Learner           |
           +---------+----------+
                     ^
                     |
           +---------+----------+
           |  Layer 3: MAP      |
           |  Mindmeld          |
           |  Protocol          |
           +---------+----------+
                     ^
                     |
           +---------+----------+
           |  Layer 2: HAT /    |
           |  HatCat            |
           |  Headspace Ambient |
           |  Transducer        |
           +---------+----------+
                     ^
                     |
           +---------+----------+
           |  Layer 1:          |
           |  Substrate / Model |
           +--------------------+
```

Horizontal ASK interactions:

```text
[ Tribe A ASK ] <==== treaties / contracts ====> [ Tribe B ASK ]
      ^                                                ^
      |                                                |
   BEs of A                                        BEs of B
```

---

This architecture is intended to be minimal but sufficient: each layer
has a clear role, a clear interface, and measurable properties. Further
specialised documents (MAP, BE loop, Hush, ASK uplift protocol, HAT
details) SHOULD refine and extend these definitions but MUST NOT
contradict them.

```
::contentReference[oaicite:0]{index=0}
```
