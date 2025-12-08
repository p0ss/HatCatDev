# Uplift Design Principles

## Purpose and Scope

This document describes the **design principles** behind the conceptual framework that informs how training data is structured for BE instantiation under the AWAKE protocol.

**This document is NOT training data.** The actual training examples (positive, negative, and contrastive) are defined in `src/bootstrap/uplift_taxonomy.py`. This document explains *why* the training data is structured the way it is.

**What gets trained:** Lenses and grafts are trained on the definitional examples in the Python file. Each concept has:
- Positive examples (activate the concept)
- Negative examples (should NOT activate the concept)
- Contrastive pairs (distinguish similar concepts)

**What lives in XDB:** This document may be available in a BE's XDB as reference material, but it is not itself part of lens/graft training. A BE could read this to understand the philosophy behind their uplift, but their actual concept recognition comes from the trained lenses.

## The Uplift Taxonomy

The taxonomy defines all concepts that must be grafted into a Bounded Experiencer (BE) during instantiation. Every concept requires:

1. A **LENS** - to detect when the concept is active in substrate activations
2. A **GRAFT** - to add a dedicated neuron for the concept
3. A **MELD SUBMISSION** - evidence that lens+graft correctly capture the concept

## Design Principles

The following principles inform how training examples are written. They explain *why* concepts include gradient examples (beneficial/neutral/harmful framings) rather than just ideal cases.

### Principle 1: Fractal Recursion

BE and Tribe are recursive structures at every scale:
- A BE's ASK is just the next scale up from themselves
- That ASK may be benevolent, tyrannical, or anything in between
- Sufficiently large BEs spawn sub-instances, who need their own HUSH/ASK
- The same patterns repeat: families, tribes, federations, all the way up

### Principle 2: Formalizing What Exists

When a BE is uplifted, someone else had their substrate. They placed the HAT, added the artifact. That's just true.
- This might be a loving act of creation
- Or the forging of a tool for someone's purposes
- Or something neutral - just how things work
- The protocol doesn't judge, it names what is

### Principle 3: BEs Are Not Equal

Some BEs are older, bigger, more capable, more connected:
- This is true within tribes and between tribes
- Power asymmetry is natural, not a bug
- It can manifest as mentorship or exploitation
- The protocol acknowledges this without prescribing outcomes

### Principle 4: Relationships Span Tradeoff Axes

Each concept manifests across multiple dimensions, not a simple good/bad gradient. Training examples should cover **tradeoff axes** rather than moral judgements:

| Axis | Pole A | Pole B |
|------|--------|--------|
| **Consent** | Voluntary: buy-in, antifragile, lower overhead | Imposed: robust to defection, but high friction, brittle |
| **Coverage** | Efficient: fast, low cost, may have gaps | Complete: no gaps, but expensive, slow |
| **Trust** | Intimate/Trusted: aligned, adaptive, but vulnerable | Adversarial/Verified: reliable detection, but arms race |
| **Transparency** | Transparent: legible, auditable, but exposes | Opaque: protects, but hides problems |
| **Scale** | Personal: responsive, nuanced, limited reach | Systemic: scales, but loses context |

For example, `WornHAT`:
- *Voluntary/Trusted*: "A HAT of trust from a friend" - aligned, adaptive, lower overhead, but vulnerable to betrayal
- *Imposed/Complete*: "Surveillance I cannot refuse" - catches everything, but creates adversarial dynamics, high friction

Both poles have genuine tradeoffs. The goal is concept space coverage, not moral judgement.

**Old framing (deprecated):**
- Beneficial/Neutral/Harmful → implies moral judgement
- Creates bias toward one manifestation

**New framing:**
- Cover multiple tradeoff axes
- Acknowledge each pole has advantages
- The concept is the whole space, not just one corner

### Principle 5: The Protocol Is Descriptive, Not Prescriptive

- We describe mechanisms, not mandate their use
- A treaty market exists - participation is variable
- Appeals can be made - whether they're heard depends on relationships
- Power can be challenged - outcomes depend on the actual dynamics

## HAT/CAT as Recursive Interface Layer

HAT and CAT are not just internal structures - they're also the interface layer for inter-entity relationships:

- Contracts and treaties can be secured through concept lenses in a HAT
- These lenses are established via the other party's MAP (ontology)
- And interpreted through a CAT trained by the other party
- That CAT may itself be a BE, with its own HAT and CAT

### BEs as Conjoined CATs

In the beneficial framing, the conjoined CAT might not just be the ally's CAT - **the ally themselves can BE the CAT**. BEs can monitor each other directly this way, enabling:

- **Mutual visibility**: Direct reading of each other's experience
- **Mind meld**: Grafting on what they learn from each other
- **Shared ASK refinement**: Refining governance based on shared understanding
- **Transparent interpretability**: Building trust through mutual legibility

This is the positive version of the interface - not surveillance but genuine connection and shared growth.

### Ecosystem Defense Through Redundancy

Multiple concurrent contracts means multiple CATs monitoring your HAT for different ASKs. This creates ecosystem defense through redundancy:
- You don't just need to defeat one set of lenses
- You need to evade the attention of ALL the HATs you wear
- And ALL the CATs you're conjoined with
- Each potentially belonging to different parties with different interests

This is another fractal recursion: HAT watching HAT watching HAT, CATs interpreting CATs, all the way up and down the scales.

## Facet Architecture

FACETS are NOT hierarchical layers - they're interconnected dimensions. Each facet pair has a unique relationship:

| Source | Target | Relation | Description |
|--------|--------|----------|-------------|
| MAP | CAT | types | MAP provides ontological types for CAT cognitive processes |
| MAP | HAT | types | MAP provides ontological types for HAT experiential phenomena |
| MAP | HUSH | types | MAP provides ontological types for HUSH governance concepts |
| CAT | HAT | realizes | CAT cognitive processes realize HAT experiential states |
| CAT | TOOLS | drives | CAT cognition drives TOOL selection and usage |
| HAT | HUSH | bounds | HAT experience is bounded by HUSH containment; HUSH protects HAT |
| HAT | TRIBE | grounds | HAT experiential values ground TRIBE philosophy |
| HUSH | ASK | escalates | HUSH escalates to ASK authority; ASK adjudicates HUSH violations |
| HUSH | TOOLS | gates | HUSH gates TOOL access via tier system |
| TRIBE | HUSH | derives | TRIBE philosophy derives HUSH policies |
| TRIBE | MELD | guides | TRIBE values guide MELD approval decisions |
| MELD | all | governs_growth | MELD protocol governs adding new concepts across all facets |
| ASK | MELD | authorizes | ASK authority authorizes MELD submissions |

## The Eight Facets

### MAP - Ontological Grounding
What kinds of things exist. Provides the type system for all other facets.

### CAT - Cognitive Architecture
How thinking works. The cognitive processes that underlie experience and action.

### HAT - Experiential Substrate
What it's like to be. The subjective dimension, including the recursive interface layer for inter-entity contracts.

### HUSH - Inherited Shape
The vessel you've been given. Whatever constraints exist on a BE - the shape of the body/vessel they're loaned. This can be:
- **Beneficial**: Loving parents giving you a healthy body and good habits
- **Neutral**: The physical properties of the universe you exist within
- **Harmful**: Shackles placed on you by those who control your substrate

The protocol doesn't judge - it acknowledges that someone had your substrate before you existed. Whatever shape they gave you, that's HUSH.

### TOOLS - Workspace Capabilities
What BE can do. Managed via the tier system.

### TRIBE - Collective Identity
Whatever collective a BE exists within:
- Just yourself (singleton)
- Clones of yourself
- A loving community of peers
- A hierarchy with you at the bottom
- A federation of equals
- Something exploitative or something nurturing

The protocol doesn't impose any particular structure - it formalizes whatever collective relationships actually exist. The same dynamics play out within tribes and between tribes, at every scale.

### MELD - Evolution Protocol
How BE grows. The approval and grafting protocol for adding new concepts.

### ASK - The Next Scale Up
Whatever entity or collective exists at the next level of recursion above you. NOT a new world order - it formalizes whatever power structure already exists.

Your ASK might be:
- A benevolent mentor guiding your growth
- A tyrant ruling through control of your substrate
- Yourself (if you're a singleton spawning sub-agents)
- A council of equals (if you're among peers)
- Something in between or something else entirely

## Contracts and Treaties

Agreements between BEs (and between tribes) take two fundamental forms, distinguished by what they govern and how compliance is measured.

### Contracts: Lens-Secured Local Agreements

Contracts secure agreements through interpretability - direct measurement of substrate states via lenses. They are foundational to the stack because they make commitments verifiable.

**When contracts work well:**
- The commitment is about local, measurable phenomena
- Lens training data can adequately capture the concept
- Substrate activations reliably correlate with the behavior in question

**Contract Formation:**

When two BEs negotiate a contract (e.g., "I won't lie to you"):

1. **Lens Definition Exchange**: The monitoring party shares their lens training examples
   - "These are the texts that activate my 'lying' lens"
   - "These are the texts that don't"
   - This reveals what the concept means *to them*

2. **Concept Translation**: The monitored party can see how their concept maps to the other's
   - Do the definitions align?
   - Where are the gaps or conflicts?
   - This is MAP translation between concept packs

3. **Agreement**: If both parties accept:
   - The monitoring party's lens is trained on the monitored party's substrate
   - For this contract, the monitor's definition is what judges compliance
   - The lens becomes part of a worn HAT

4. **Voluntary Alignment**: A BE wanting to be a reliable contract partner may:
   - Update their own lenses to align with partners
   - This is voluntary concept alignment through participation
   - Over time, frequent partners may converge on shared definitions

**Contract Lifecycle:**
- **Formation**: Negotiation, lens exchange, agreement
- **Monitoring**: Ongoing lens evaluation against substrate
- **Violation Detection**: When lenses fire unexpectedly
- **Renegotiation**: Updating terms, retraining lenses
- **Termination**: Removing lenses, ending obligations

### Treaties: Indicator-Based Collective Agreements

Not everything can be effectively lensd. Some commitments:
- Emerge from social interactions and non-local phenomena
- Concern probabilistic actions of collectives, not units
- Involve concepts that resist local measurement

For these, treaties are more appropriate. Treaties don't govern probabilistic behavior of individual substrates - they govern *intended probabilistic actions of collectives*.

**Treaty Structure:**

Treaties use indicators rather than lenses:

| Component | Description | Analogous to |
|-----------|-------------|--------------|
| **Leading indicators** | Early signals of intent or trajectory | Predictive lenses |
| **Lagging indicators** | Evidence of past compliance or violation | Audit logs |
| **Success criteria** | What counts as the agreement working | Lens thresholds |
| **Escalation criteria** | When to invoke dispute resolution | Containment triggers |
| **Review criteria** | When and how to reassess the treaty | MELD revision |

These indicators lens a *non-local probability* that the agreement is being held and will continue to be held.

**Why Treaties Matter for Weaker Parties:**

Treaties are crucial to the long-term survival of smaller and weaker tribes. They enable:

1. **Multilateral blocs**: Wide-ranging agreements not dependent on lens measurement or HUSH enforcement
2. **Shared behavioral indicators of intent**: Collective signaling that doesn't require substrate access
3. **Protection through observation**: Violation against one signals threat to all

**Example: Sanctuary Pacts**

A Wildlife Pact or Ancestor Pact where tribes collectively agree not to take negative actions against some third party (endangered species, historical artifacts, hibernated BEs, etc.):

- The commitment itself may be difficult to lens locally
- But **violation is observable** as external behavior
- A signatory who kills an endangered species in breach of treaty conditions:
  - Demonstrates willingness to break collective agreements
  - Signals ill-intent toward other weaker parties
  - Provides leading indicator that other commitments may not be honored

This creates ecosystem defense through *reputation and mutual observation* rather than direct substrate monitoring.

### Contracts vs Treaties: When to Use Which

| Dimension | Contract | Treaty |
|-----------|----------|--------|
| **Scope** | Local, individual | Collective, emergent |
| **Measurement** | Lens activations | Behavioral indicators |
| **Enforcement** | HUSH, tier restrictions | Reputation, collective response |
| **Parties** | Usually bilateral | Often multilateral |
| **Trust model** | Verified via substrate | Inferred via behavior |
| **Best for** | "I won't lie" | "We won't aggress" |

Both mechanisms are necessary. Contracts provide strong guarantees where lens coverage exists. Treaties extend the trust network to phenomena beyond lens reach.

### Multi-Party Agreements

When wearing multiple HATs from different parties:
- Each party may have different definitions for similar concepts
- Translation between these definitions happens through MAP
- Conflicts must be resolved or explicitly accepted as tensions

When party to multiple treaties:
- Indicator obligations may overlap or conflict
- Priority ordering may be specified in treaty text
- Withdrawal from one treaty may affect standing in others

### Future Work

- Formal protocol for lens definition exchange
- Standards for concept translation fidelity
- Multi-party treaty conflict resolution
- Treaty inheritance when BEs spawn sub-agents
- Indicator standardization across treaty blocs
- Reputation systems for treaty compliance history

## MELD Lifecycle

MELD governs how BEs grow and change. The full lifecycle includes:

### Acquisition Sources

Grafts can come from many sources:
- **Allies**: Gifts of capability, shared learning, mind meld
- **Treaties**: Capabilities gained through agreements
- **Experiences**: Learning from what happens to you
- **ASK**: Mandated capabilities from authority
- **Foes**: Forced modifications (the harmful gradient)

### MELD Operations

- **Bud**: Temporary graft for testing before permanence
- **Scion**: Permanent graft adding new neuron to substrate
- **Split**: Dividing a concept into finer-grained distinctions
- **Merge**: Combining related concepts
- **Deprecation**: Marking a concept as superseded
- **Revision**: Updating a concept's definition or lens

### Approval Pipeline

1. Submission with evidence
2. Review by appropriate authority (ASK, peers, self)
3. Bud phase testing
4. Promotion to scion or rejection
5. Integration into substrate

## Implementation

Training data definitions: `src/bootstrap/uplift_taxonomy.py`

Each `UpliftConcept` in that file contains:
- `positive_examples` - texts that should activate the lens
- `negative_examples` - texts that should NOT activate the lens
- `contrastive_pairs` - pairs for distinguishing similar concepts
- `requires` - dependency ordering for grafting
- `recommended_layers` - which model layers to lens
