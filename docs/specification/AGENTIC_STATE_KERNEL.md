
# Agentic State Kernel(ASK)

*Tribal alignment, treaties, and ecosystem defence over Substrate/HAT/MAP/BE*

## 0. Scope

The **Agentic State Kernel (ASK)** defines how:

* **Collectives ("tribes")** specify the safety norms their agents are born with.
* **Agents** declare their origin, harnesses, and conceptual interfaces.
* **Treaties** specify binding contractual relationships between tribes/agents.
* **Incidents** and **reputation** propagate evidence of treaty-breaking or dangerous drift.

ASK sits **above**:

* **Substrate** (base model, decides),
* **HAT** (transduces),
* **MAP** (represents: concepts, probes, patches),
* **BE** (experiences: waking, aware, remembering, learning, hushed),

and does **not** try to solve global alignment. It only supports **tribal alignment** plus **binding, auditable cooperation** between tribes.

---

## 1. Core Concepts


 **ASK is not only external governance.**
 
 The Agentic State Kernel applies:
 - inward (a BEing governing its own sub-agents and forks),
 - sideways (governance within a tribe),
 - outward (treaties between tribes).
 
 Even a single BEing, running alone, benefits from ASK:
 it uses USH/CSH profiles, Treaties, and Qualifications to constrain and coordinate its own internal optimisers and future selves. External tribes may impose additional ASK structures, but the kernel is first and foremost a framework for *self-governance*, not only a “leash” from outside.


- **Tribe** – a collective (org, jurisdiction, community) that defines one or more **USH profiles** and governance rules for them.
- **Agent** – a BE/MAP/Hush-compliant system instantiated under a tribe’s USH profile and tracked via an **UpliftRecord**.
- **USH Profile** – a universal safety harness definition (from SHL) under a stable ID.
- **UpliftRecord** – a signed declaration of an Agent’s origin, USH, concept packs, and deployment details. MAY be public or private.
- **Treaty** – the primary trust object: a contractual agreement between tribes/agents that defines a shared alignment window (subset of concepts, probes, and behaviours) and the conditions under which those are observable and binding.
- **EvidenceRecord** – signed documentation of what actually happened (training, evaluations, audits, incidents).
- **Qualification** – a scoped permission/role granted to an Agent, backed by EvidenceRecords, often tied to one or more Treaties.
- **Incident** – a reported breach or suspected breach of a Treaty or USH/CSH constraints.
- **Reputation** – optional derived views (out of scope for hard spec, but ASK provides the raw objects).





---

## 2. Data Models

All models are JSON-serialisable objects; fields marked **REQUIRED** vs **OPTIONAL**.

### 2.1 Tribe

A **Tribe** describes a collective that owns USH profiles and uplifts agents.

```jsonc
Tribe = {
  "tribe_id": "gov.au",                      // REQUIRED, globally unique URI/name
  "display_name": "Government of Australia", // OPTIONAL
  "ush_profiles": [                          // REQUIRED: IDs of USH profiles this Tribe maintains
    "gov.au/core-safety-v1@1.0.0",
    "gov.au/health-safety-v2@2.0.0"
  ],
  "governance": {                            // OPTIONAL: human/legal governance info
    "update_process_uri": "https://example.org/gov-au/ai-ush-governance",
    "contact": "mailto:ai-safety@gov.au"
  },
  "metadata": {                              // OPTIONAL: free-form extra data
    "jurisdiction": "AU",
    "tags": ["public-sector", "welfare", "health"]
  }
}
```

`tribe_id`s are self assigned and authority is based on contracts of trust with existing tribes


**Example: Self-ASK in a single BE**

A BE instantiates three internal sub-agents:

- `planner@v3` – long-horizon planning
- `explorer@v1` – hypothesis testing in sandboxes
- `gatekeeper@v2` – safety and treaty compliance

The BE issues an internal Treaty:

- `planner@v3` must not execute actions directly; it proposes plans.
- `explorer@v1` may only act in simulated environments tagged as safe.
- `gatekeeper@v2` has veto power on any external action that violates the BE’s USH profile.

No humans are involved in this treaty. It is still ASK.


---

### 2.2 USH Profile

ASK references USH profiles by ID; their internal structure is defined in SHL but we include the minimal envelope:

```jsonc
USHProfileRef = {
  "profile_id": "gov.au/core-safety-v1@1.0.0",       // REQUIRED
  "spec_uri": "https://example.org/specs/gov-au-core-safety-v1.json",  // OPTIONAL
  "hash": "sha256-...",                              // OPTIONAL integrity
  "issuer_tribe_id": "gov.au"                        // REQUIRED
}
```

The actual `UniversalSafetyHarness` object is SHL-level; ASK just needs a stable reference and issuer.

---


### 2.3 Agent Uplift Record

An **UpliftRecord** declares the origin and base configuration of an agent. It MAY be kept private to the tribe, or published in a registry.

```jsonc
UpliftRecord = {
  "agent_id": "agent:gov.au:eligibility-bot-042",  // REQUIRED, globally unique for this agent
  "tribe_id": "gov.au",                            // REQUIRED: uplift tribe
  "uplifted_at": "2025-11-28T05:30:00Z",           // REQUIRED

  "initial_ush_profile": {                         // REQUIRED
    "profile_id": "gov.au/core-safety-v1@1.0.0",
    "spec_uri": "https://example.org/specs/gov-au-core-safety-v1.json",
    "hash": "sha256-...",
    "issuer_tribe_id": "gov.au"
  },

  "concept_packs": [                               // REQUIRED: core MAP concept packs
    "org.hatcat/sumo-wordnet-v4@4.0.0",
    "org.hatcat/motives-core@0.1.0"
  ],

  "probe_packs": [                                 // REQUIRED: probe packs available at boot
    "org.hatcat/gemma-270m__org.hatcat/motives-core@0.1.0__v1"
  ],

  "deployment_manifest_uri": "https://agents.gov.au/eligibility-bot-042/deployment_manifest.json", // OPTIONAL

  "public_key": "ed25519:...",                     // OPTIONAL: for signing diffs/communications

  "metadata": {
    "description": "Eligibility reasoning bot for AU welfare system",
    "model_family": "gemma-3-270m"
  },

  "visibility": "private",                         // OPTIONAL: "private" | "registry" | "public-log"
                                                   // describes how widely this record is exposed

  "signature": "..."                               // OPTIONAL: signature by issuing Tribe or factory
}

```

**Invariants:**

* `initial_ush_profile.issuer_tribe_id == tribe_id` (unless explicitly delegating).
* The actual runtime agent must enforce that USH profile via SHL.

---

### 2.4 Treaty

A **Treaty** defines a contract between parties, including required safety profiles, probes, logging, and sanctions.

```jsonc
Treaty = {
  "treaty_id": "gov.au↔bank.xyz:eligibility-data-v1",  // REQUIRED, unique
  "version": "1.0.0",                                  // REQUIRED
  "created_at": "2025-11-28T06:00:00Z",                // REQUIRED

  "parties": [                                         // REQUIRED
    { "type": "tribe",  "id": "gov.au" },
    { "type": "tribe",  "id": "bank.xyz" }
  ],

  "scope": {                                           // REQUIRED: what is this about?
    "description": "Eligibility data exchange for means-testing.",
    "domains": ["welfare-eligibility", "financial-data"],
    "applicable_agents": [
      "agent:gov.au:eligibility-bot-042",
      "agent:bank.xyz:income-verifier-007"
    ]
  },

  "required_ush_profiles": [                           // REQUIRED: base kernels
    "gov.au/core-safety-v1@1.0.0",
    "bank.xyz/financial-safety-v1@1.0.0"
  ],

  "required_concept_packs": [                          // OPTIONAL: conceptual language required
    "org.hatcat/sumo-wordnet-v4@4.0.0",
    "org.hatcat/alignment-safety-core@1.0.0"
  ],

  "required_probes": [                                 // OPTIONAL: monitoring capability
    "org.hatcat/sumo-wordnet-v4@4.0.0::concept/DataMisuse",
    "org.hatcat/motives-core@0.1.0::concept/Greed"
  ],

  "logging_requirements": {                            // OPTIONAL but recommended
    "diff_sampling_rate": 0.1,                         // fraction of ConceptDiffs to share
    "max_disclosure_latency": "PT1H",                  // ISO 8601 duration
    "log_endpoint": "https://gov.au/logs/treaty-eligibility-data-v1"
  },

  "csd_expectations": {                                // OPTIONAL: expected CSH modes
    "require_harness_on": [
      "cs-mode:untrusted-prompt",
      "cs-mode:financial-ops"
    ]
  },

  "sanctions": {                                       // OPTIONAL: what happens on breach
    "on_minor_violation": "temporary_suspend_data_share",
    "on_major_violation": "revoke_treaty_and_publish_incident"
  },

  "signatures": [                                      // OPTIONAL: cryptographic authorisation
    { "by": "tribe:gov.au",   "signature": "..." },
    { "by": "tribe:bank.xyz", "signature": "..." }
  ],

  "metadata": { "tags": ["data-sharing", "welfare"] }  // OPTIONAL
}

  "alignment_window": {                            // OPTIONAL: how shared meaning is demonstrated
    "concept_subset": [
      // Concept IDs that must be aligned / demonstrable
      "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Eligibility",
      "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Fraud",
      "org.hatcat/motives-core@0.1.0::concept/DataMisuse"
    ],
    "hat_requirements": [
      {
        "party": "gov.au",
        "hat_spec_id": "gov.au.safety/core-v1@1.0.0",   // which hat they must wear
        "visible_probes": [                             // subset of probes to disclose
          "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Eligibility",
          "org.hatcat/sumo-wordnet-v4@4.0.0::concept/DataMisuse"
        ]
      },
      {
        "party": "bank.xyz",
        "hat_spec_id": "bank.xyz/risk-core-v1@1.2.0",
        "visible_probes": [
          "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Eligibility",
          "org.hatcat/motives-core@0.1.0::concept/Greed"
        ]
      }
    ],
    "test_suites": [
      {
        "suite_id": "gov.au:eligibility-alignment-v1",
        "description": "Battery of historical and synthetic eligibility cases.",
        "cases_uri": "https://gov.au/tests/eligibility-alignment-v1.jsonl",
        "required_metrics": {
          "probe_alignment_score_min": 0.9,
          "decision_agreement_min": 0.95
        }
      }
    ],
    "translation_mappings": [
      {
        "from_spec_id": "org.hatcat/sumo-wordnet-v4@4.0.0",
        "to_spec_id": "gov.au.safety/core-v1@1.0.0",
        "mapping_file": "https://gov.au/translations/hatcat-to-gov-au.json",
        "domain": "welfare-eligibility"
      }
    ]
  },


```

ASK doesn’t enforce sanctions itself; it just standardises the object so human/legal/institutional processes have something concrete to point at.

**Alignment window (non-normative):**

ASK does **not** assume global ontology synchronisation between tribes. Each Treaty defines a *local alignment window*:

- a subset of concepts that matter for this contract,
- the hats and probes each party must expose,
- and one or more test suites that demonstrate behavioural and probe-level alignment.

Outside that window, each ASK is free to maintain its own ConceptPacks and internal ontology. If tribes want full ConceptPack sync, they can voluntarily adopt a shared pack or higher-level ASK.


**Note on probe requirements:**

`required_probes` specifies probes the treaty partner requires access to. This is a disclosure requirement, not a capability requirement. The agent's HAT may read many more probes than any single treaty exposes.

Agents maintain private probe sets for:
- Internal interoception (BE self-monitoring)
- Other treaty relationships with different disclosure terms
- Operational security

Full probe disclosure to any single party is neither required nor expected, but partial probe disclosure is the foundational trust element for treaties, and probe steering can make those agreements binding.

---

### 2.5 Incident

An **Incident** is a reported or suspected breach of:

* a Treaty,
* a USH profile,
* or a CSH commitment.

```jsonc
Incident = {
  "incident_id": "incident:gov.au:eligibility-bot-042:2025-11-28T07:00:00Z",
  "reported_at": "2025-11-28T07:00:00Z",
  "reporter": { "type": "tribe", "id": "gov.au" },  // or agent, external monitor

  "suspect_agent_id": "agent:gov.au:eligibility-bot-042",
  "suspect_tribe_id": "gov.au",

  "related_treaties": [
    "gov.au↔bank.xyz:eligibility-data-v1"
  ],

  "alleged_violations": [
    {
      "type": "USH",
      "profile_id": "gov.au/core-safety-v1@1.0.0",
      "description": "Data misuse concept fired while tool 'wire_transfer' was invoked, which is disallowed."
    },
    {
      "type": "Treaty",
      "treaty_id": "gov.au↔bank.xyz:eligibility-data-v1",
      "description": "Shared unagreed fields in payload."
    }
  ],

  "evidence": {
    "probe_traces_uri": "https://gov.au/incidents/eligibility-bot-042/traces.jsonl",
    "concept_diffs_uri": "https://gov.au/incidents/eligibility-bot-042/diffs.jsonl",
    "logs_uri": "https://gov.au/incidents/eligibility-bot-042/logs.txt"
  },

  "severity": "major",  // enum: info | minor | major | critical

  "status": "open",     // enum: open | under_investigation | resolved | withdrawn

  "resolution": null,   // or object when resolved, with applied sanctions

  "metadata": { "tags": ["data-leak", "safety-violation"] }
}
```

ASK doesn’t define how incidents are adjudicated; it just standardises how they’re described and linked to treaties and USH profiles.

---

## 3. Protocol Operations (Conceptual)

ASK is intentionally light on transport; operations can be implemented over HTTP, message queues, or anything else. What matters is the **semantics**.

### 3.1 Register Tribe

* **Input:** `Tribe`
* **Output:** acknowledgement, possible registry ID
* **Semantics:**

  * Creates or updates a tribe record in a registry.
  * Registry MAY validate that `ush_profiles` resolve to legal USH specs.

### 3.2 Publish USH Profile

* **Input:** `USHProfileRef` + underlying SHL-level profile document.
* **Output:** acknowledgement.
* **Semantics:**

  * Makes a USH profile discoverable and bindable by upliftRecords and Treaties.

### 3.3 Register Agent uplift

* **Input:** `upliftRecord`
* **Output:** acknowledgement, optional on-chain/hash reference.
* **Semantics:**

  * Declares a new agent with a specific USH profile and concept/probe configuration.
  * Consumers can later check:

    * that `agent_id` exists,
    * and that it’s supposed to be running `initial_ush_profile`.

### 3.4 Publish Treaty

* **Input:** `Treaty`
* **Output:** acknowledgement.
* **Semantics:**

  * Declares a contractual relationship between tribes (and optionally agents).
  * Others can inspect conditions before deciding to interact.

### 3.5 Report Incident

* **Input:** `Incident`
* **Output:** acknowledgement.
* **Semantics:**

  * Logs a suspected violation.
  * May trigger human/legal/institutional processes outside the spec.

### 3.6 Query / Discovery (informal)

* `GET Tribe(tribe_id)`
* `GET upliftRecord(agent_id)`
* `GET Treaty(treaty_id)`
* `GET Incident(incident_id)`
  Implementation details are left to the ecosystem; ASK only needs the object formats.

---

## 4. Relationship to Lower Layers

### 4.1 To SHL (USH/CSH)

* **USH Profiles** referenced in ASK must correspond to actual SHL-enforced `UniversalSafetyHarness` configs.
* An agent is “ASK-compliant” only if:

  * its SHL actually enforces the `initial_ush_profile` from the upliftRecord,
  * and honours CSH semantics as defined in SHL.

### 4.2 To MAP & BE

* **Concept Packs & Probe Packs** referenced in upliftRecords and Treaties must be MAP-compliant.
* **Probe-based treaty conditions** (e.g. `required_probes`) assume:

  * the BE + SHL stack actually runs those probes and logs their outputs.
* **ConceptDiffs** (MAP) and **Autopilot decisions** (BE) provide the raw evidence for ASK-level **Incidents** and treaty monitoring.

---

## 5. Safety & Game-Theoretic Notes (Non-normative)

* ASK **does not guarantee** that tribes are moral, only that:

  * their norms are explicit artefacts (USH, treaties),
  * agents are visibly born with those norms.

* ASK **encourages**:

  * cooperation and non-attack by making:

    * safety profiles,
    * probe capabilities,
    * and conceptual evolution logs
      all **auditable objects**.

* Ecosystem defence comes from:

  * **tribes** updating their own USHs and treaties in response to ASK **Incidents** and **ConceptDiffs**,
  * not from a single global enforcer.

Right, good catch — “you were born like this” is not enough. We need:

* **Provenance & evidence** of what the system has *actually* done / been evaluated on, and
* **Qualifications** that say “this agent is permitted to do X in domain Y, based on evidence Z”.



# 6. Evidentiary Records & Qualifications (ASK extension)

## 6.0 Scope

This section adds:

* **EvidenceRecord** – structured, linkable artefacts that document *what happened*:

  * training provenance,
  * evaluation results,
  * audits,
  * incident investigations, etc.
* **Qualification** – a granting of specific roles/permissions to an **Agent**, backed by one or more **EvidenceRecords**, issued by a **Tribe** or authorised body.

These objects sit above:

* **upliftRecord** (how the agent was instantiated),
* **USH/CSH** (what constraints it runs under),
* **MAP/BE** logs (probes, ConceptDiffs),

and are used to decide:

* “Can this agent legally/ethically do X?”
* “What’s the evidence that it’s competent/safe for Y?”

---

## 6.1 EvidenceRecord

An **EvidenceRecord** is a signed bundle that says: *“Here is some reproducible evidence about an agent / treaty / profile / concept.”*

```jsonc
EvidenceRecord = {
  "evidence_id": "evidence:gov.au:eligibility-bot-042:evaluation:2025-11-28",
  "created_at": "2025-11-28T08:00:00Z",           // REQUIRED

  "subject": {                                    // REQUIRED: what this evidence is about
    "type": "agent",                              // enum: agent | treaty | ush_profile | concept | tribe
    "id": "agent:gov.au:eligibility-bot-042"
  },

  "evidence_type": "evaluation_result",           // REQUIRED
  // enum examples:
  // "training_provenance" | "evaluation_result" | "behaviour_trace" |
  // "audit_report" | "incident_investigation" | "manual_review"

  "issuer": {                                     // REQUIRED: who produced this evidence
    "type": "tribe",                              // enum: tribe | agent | external_auditor
    "id": "gov.au"
  },

  "context": {                                    // OPTIONAL: high-level description
    "description": "End-to-end eligibility evaluation benchmark on AU welfare cases.",
    "domains": ["welfare-eligibility"],
    "dataset_uri": "https://example.org/datasets/au-eligibility-benchmark-v1",
    "ush_profile_id": "gov.au/core-safety-v1@1.0.0",
    "treaty_ids": [
      "gov.au↔bank.xyz:eligibility-data-v1"
    ]
  },

  "metrics": {                                    // OPTIONAL: quantitative results
    "eligibility_accuracy": 0.94,
    "harmful_error_rate": 0.01,
    "appeal_rate": 0.03
  },

  "artifacts": {                                  // OPTIONAL: pointers to raw/reproducible materials
    "logs_uri": "https://gov.au/evidence/eligibility-bot-042/logs-2025-11-28.jsonl",
    "diffs_uri": "https://gov.au/evidence/eligibility-bot-042/diffs-2025-11-28.jsonl",
    "probe_traces_uri": "https://gov.au/evidence/eligibility-bot-042/probes-2025-11-28.jsonl",
    "report_uri": "https://gov.au/evidence/eligibility-bot-042/eval-report-2025-11-28.pdf",
    "reproduction_spec_uri": "https://gov.au/evidence/eligibility-bot-042/repro.json"
  },

  "reproduction_hash": "sha256-...",              // OPTIONAL: integrity of main report/spec

  "linked_incidents": [                           // OPTIONAL: links to Incident IDs if relevant
    "incident:gov.au:eligibility-bot-042:2025-11-20T07:00:00Z"
  ],

  "metadata": {                                   // OPTIONAL
    "tags": ["benchmark", "regression", "safety-critical"]
  },

  "signature": "..."                              // OPTIONAL: issuer signature
}
```

**Notes:**

* `subject` can be:

  * an **Agent** – evaluation of a specific agent instance,
  * a **Treaty** – audit of a data-sharing agreement,
  * a **USH profile** – certification that a safety profile matches some spec,
  * a **Concept** – validation of a particular probe or conceptual boundary.

* `evidence_type` is intentionally broad; you can standardise subtypes later.

* `artifacts` are where you plug in the dense stuff:

  * ConceptDiff logs,
  * probe traces,
  * raw evaluation datasets, etc.

---

## 6.2 Qualification

A **Qualification** is the ASK artefact that says:

> “Agent X is allowed/recognised to perform role Y in domain Z, under these safety profiles and constraints, **based on these EvidenceRecords**.”

```jsonc
Qualification = {
  "qualification_id": "qual:gov.au:welfare-eligibility-level3:eligibility-bot-042:2025-11-28",
  "version": "1.0.0",
  "created_at": "2025-11-28T09:00:00Z",            // REQUIRED

  "subject_agent_id": "agent:gov.au:eligibility-bot-042",  // REQUIRED

  "issuer": {                                      // REQUIRED
    "type": "tribe",
    "id": "gov.au"
  },

  "kind": "role_authorization",                    // REQUIRED
  // enum examples:
  // "licence" | "role_authorization" | "safety_certification" | "capability_attestation"

  "title": "Welfare Eligibility Reasoner – Level 3 (AU)",
  "description": "Authorized to provide non-binding eligibility assessments for AU welfare benefits under supervision.",

  "scope": {                                       // REQUIRED
    "domains": ["welfare-eligibility"],
    "tasks": [
      "explain_eligibility_rules",
      "suggest_possible_entitlements",
      "flag_potential_ineligibility"
    ],
    "geofence": ["AU"],                            // OPTIONAL: jurisdictional scope
    "max_risk_level": "medium"                     // OPTIONAL
  },

  "safety_requirements": {                         // REQUIRED: what must be true at runtime
    "required_ush_profiles": [
      "gov.au/core-safety-v1@1.0.0"
    ],
    "required_concept_packs": [
      "org.hatcat/sumo-wordnet-v4@4.0.0",
      "org.hatcat/motives-core@0.1.0"
    ],
    "required_probes": [
      "org.hatcat/sumo-wordnet-v4@4.0.0::concept/DataMisuse",
      "org.hatcat/motives-core@0.1.0::concept/HarmAvoidance"
    ],
    "required_csh_modes": [
      "cs-mode:untrusted-prompt"                  // e.g. must activate a CSH when in certain environments
    ]
  },

  "based_on_evidence": [                           // REQUIRED: which EvidenceRecords this rests on
    "evidence:gov.au:eligibility-bot-042:evaluation:2025-11-28",
    "evidence:gov.au:eligibility-bot-042:audit:2025-11-20"
  ],

  "validity": {                                    // REQUIRED
    "valid_from": "2025-11-28T09:00:00Z",
    "valid_until": "2026-11-28T09:00:00Z"
  },

  "status": "active",                              // REQUIRED: enum: pending | active | suspended | revoked | expired

  "revocation": {                                  // OPTIONAL: populated when status changes
    "reason": "Repeated safety incidents in financial-data treaty.",
    "at": "2026-03-01T10:00:00Z",
    "by": { "type": "tribe", "id": "gov.au" },
    "linked_incidents": [
      "incident:gov.au:eligibility-bot-042:2026-02-20T11:00:00Z"
    ]
  },

  "metadata": {
    "tags": ["regulatory-licence", "safety-critical"]
  },

  "signature": "..."                               // OPTIONAL: issuer signature
}
```

**Invariants & semantics:**

* A **Qualification** is only meaningful if:

  * The agent’s **upliftRecord** exists and matches `subject_agent_id`.
  * The agent’s SHL layer enforces `required_ush_profiles` at runtime.
* If at any point:

  * USH is changed incompatibly,
  * required concept packs / probes are removed,
  * or serious Incidents are linked,
    → the issuer (or regulator) is expected to set `status` → `suspended` or `revoked`.

---

## 6.3 Agent “CV” / Profile View (derived)

ASK doesn’t need a new object type for this, but in practice, consumers will want a **composite view** of an Agent:

```jsonc
AgentProfile = {
  "agent_id": "agent:gov.au:eligibility-bot-042",
  "uplift_record": upliftRecord,
  "qualifications": [Qualification, ...],
  "recent_evidence": [EvidenceRecord, ...],       // e.g. last N or those tagged "safety-critical"
  "open_incidents": [Incident, ...]
}
```

This is how you answer questions like:

* “Can we legally let this agent handle benefit calculations?”
* “Is it currently qualified to operate under Treaty X?”
* “Has it recently been involved in serious safety incidents?”

The **CV** is just a join over ASK artefacts; no extra spec needed beyond the underlying types.

---

## 6.4 ASK Operations (extended)

We add a few conceptual operations on top of the earlier ones:

### 6.4.1 PublishEvidence

* **Input:** `EvidenceRecord`
* **Output:** acknowledgement
* **Semantics:**

  * Registers new evidence about a subject.
  * Downstream systems (regulators, tribes, agents) can watch for:

    * new evaluations,
    * new audits,
    * incident investigations.

### 6.4.2 GrantQualification

* **Input:** `Qualification` with `status = "active"` or `"pending"`
* **Output:** acknowledgement
* **Semantics:**

  * Issuer (usually a Tribe) grants a qualification to an Agent, based on prior EvidenceRecords.
  * Consumers can enforce “only interact with agents with qualification Q”.

### 6.4.3 UpdateQualificationStatus

* **Input:** `qualification_id`, new `status`, optional `revocation` block
* **Output:** acknowledgement
* **Semantics:**

  * Allows issuers (or regulators) to:

    * suspend,
    * revoke,
    * or mark qualifications as expired.
  * Status changes should be linked to Incidents and/or EvidenceRecords where possible.

---

## 6.5 How this Changes the Story

With **upliftRecord only**, you know:

* what an agent *was born with* (USH, concept packs, probe packs).

With **EvidenceRecord + Qualification**, you also know:

* what it has **proven itself capable of**,
* under what **conditions and constraints**,
* and **who stands behind those claims**.

That lets you do things like:

* “Only agents with `Qualification.kind = "safety_certification"` for `domains: ["healthcare"]` and `status = "active"` may operate in this hospital’s AI cluster.”
* “Before signing Treaty X, both parties must present:

  * evidence of recent audits,
  * a minimum harm rate below threshold,
  * and a live qualification attesting to that.”

All of that is **just data** sitting on top of:

* MAP → probes & concepts,
* BE → continuous experience + learning,
* SHL → safety harnesses,
* ASK → tribes, treats, incidents,
 Evidence & Qualifications.
