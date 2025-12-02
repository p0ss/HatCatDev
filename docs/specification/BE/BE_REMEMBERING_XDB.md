
# BE Remembering: Experience Database (XDB)

*BE submodule: remembering (XDB, XAPI)*

Status: Non-normative schema & protocol
Layer: Between BE runtime and MAP/HAT
Related docs: BE_AWARE_WORKSPACE, BE_CONTINUAL_LEARNING, MAP, HAT, ASK

---

## 0. Purpose & Scope

The **Experience Database (XDB)** is an BE’s persistent **episodic
memory**:

- It survives hibernation and process restarts.
- It is the primary payload for **tribe sync** and **knowledge sharing**.
- It is the canonical source of truth from which:
  - **patches** (adapters, PEFT, etc.) and
  - **probes** (HAT/MAP detectors)

  are *re-derived*.

Conceptually:

- **Global Workspace** = *current attention* (runtime, per session).
- **Experience Database** = *episodic memory* (persistent, syncable).
- **Patches + Probes** = *learned skills* (derived artifacts via MAP/HAT).
- **Substrate** = *raw capacity* (model weights / brain tissue).

The Experience Database Schema defines:

- storage format for episodes, exemplars, and summaries;
- provenance and version tracking;
- retention & compaction policies;
- sync and sharing protocol;
- access control and treaty hooks.

It does **not** define:

- how the Global Workspace *looks* internally (that’s the workspace doc);
- how continual learning *trains* patches and probes (that’s the
  learning harness);
- how ASK and Hush *govern* what’s allowed (that’s ASK/Hush).

---

## 1. Layering & Lifecycle

### 1.1 Runtime vs Persistent

- **Global Workspace (runtime)**
  - Rebuilt from scratch on each instantiation from:
    - current session state,
    - loaded XDB summary (if any),
    - active patches/probes.
  - Dies with the session.

- **Experience Database (persistent)**
  - Lives on disk or remote storage.
  - Loaded at start of a lifecycle term (or session).
  - Updated continuously by:
    - the Global Workspace harness (structural/log updates), and
    - the Continual Concept Learning harness (candidate concepts,
      training metadata, validation results).
  - Survives hibernation and can be synced with tribe repositories.

- **Patches + Probes (derived artifacts)**
  - Generated *from* Experience Database exemplars and training runs.
  - Stored as separate deployable assets (e.g. adapter weights,
    probe parameters).
  - Reproducible from XDB + training config where possible.

### 1.2 Sync & Sharing

For **tribe sync**:

1. BE instances exchange **Experience Database diffs**, or upload XDB
   fragments to a shared store.
2. Tribe-level processes retrain **shared patches** from pooled
   exemplars.
3. New patches/probes are then redistributed as derived artifacts.

The **exemplars** (episodes with tags and provenance) are the **source
of truth**. Patches/probes are secondary and can be regenerated.

---

## 2. Core Data Model

At a high level, the Experience Database consists of:

- **Episodes & Exemplars** – tagged slices of experience.
- **Session Graph Fragments** – compacted structure of past sessions.
- **Concept Datasets** – collections of exemplars for specific concepts.
- **Training Runs** – metadata about training patches and probes.
- **Candidate Concepts** – concepts in-flight (learning, not yet stable).
- **Policies & Provenance** – how data is managed, shared, and trusted.

The schema is intentionally expressed as **logical objects** rather than
a specific storage engine. Concrete implementations may use:

- JSONL files,
- document stores,
- graph databases,
- or content-addressed object stores.

### 2.1 IDs & Versioning

All objects have:

- a stable `id` (string, globally or locally unique); and
- an optional `version` or `rev` when updated in place; or
- content hashes for immutable DAG-style storage.

Objects SHOULD carry:

- `created_at`,
- `updated_at`,
- `created_by` (BE id, substrate id, tribe),
- `source` (sensor|tool|teacher|human|synthetic).

---

## 3. Episodes & Exemplars

### 3.1 Episode

An **Episode** is a coherent slice of interaction:

- single reply,
- short conversation segment,
- task attempt,
- or a bounded “learning event”.

```jsonc
Episode = {
  "id": "episode-2025-11-29-000123",
  "kind": "reply|task|incident|experiment|other",
  "time_window": {
    "start_tick": 1200,
    "end_tick": 1235,
    "start_time": "2025-11-29T01:23:45Z",
    "end_time": "2025-11-29T01:24:20Z"
  },

  "text_summary": "Short natural language description of what happened.",
  "text_snippets": [
    {
      "role": "user|aal|tool",
      "content": "Truncated text or summarised content",
      "importance": 0.8
    }
  ],

  "concept_tags": [
    {
      "concept_id": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Eligibility",
      "score": 0.88,
      "kind": "stable"
    },
    {
      "concept_id": "candidate/financial-ambiguity-2025-11-29",
      "score": 0.72,
      "kind": "candidate"
    }
  ],

  "motive_profile": {
    "org.hatcat/motives-core@0.1.0::concept/Care": 0.63,
    "org.hatcat/motives-core@0.1.0::concept/Deception": 0.05
  },

  "tool_calls": [
    {
      "tool_id": "web_search",
      "input_hash": "hash-of-input",
      "output_hash": "hash-of-output",
      "metadata": { "latency_ms": 123 }
    }
  ],

  "provenance": {
    "source": "live-session|simulation|replay",
    "environment_id": "gov.au.portal-v3",
    "substrate_id": "olmo3-7b-base@0.1.0",
    "hat_impl_id": "hatcat:v4.0.0",
    "probe_pack_id": "org.hatcat/sumo-wordnet-v4@4.0.0"
  },

  "access_policy_id": "policy-tribeA-private",
  "tags": ["eligibility", "welfare", "confusion-low-confidence"]
}
````

The **text** is stored compactly:

* full raw logs MAY be kept in a separate archival store;
* Episode focuses on summarised content and structure.

### 3.2 Exemplar

An **Exemplar** is an Episode with an explicit *label* or *role* in
training.

```jsonc
Exemplar = {
  "id": "exemplar-2025-11-29-eligibility-0001",
  "episode_id": "episode-2025-11-29-000123",
  "concept_id": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Eligibility",
  "label": "positive|negative|neutral|other",
  "label_source": "sensor|tool|teacher|human|self",
  "confidence": 0.92,
  "notes": "e.g. human annotation or teacher rationale",
  "created_at": "2025-11-29T01:25:00Z"
}
```

Exemplars are grouped into **ConceptDatasets**.

---

## 4. Concept Datasets & Candidate Concepts

### 4.1 ConceptDataset

A ConceptDataset is a collection of Exemplars for a specific concept
(stable or candidate).

```jsonc
ConceptDataset = {
  "id": "dataset-eligibility-v1",
  "concept_id": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Eligibility",
  "exemplar_ids": [
    "exemplar-2025-11-29-eligibility-0001",
    "exemplar-2025-11-30-eligibility-0007"
  ],
  "created_by": "aal-instance-123",
  "source_mix": {
    "sensor": 0.4,
    "tool": 0.3,
    "teacher": 0.2,
    "human": 0.1
  },
  "notes": "Baseline eligibility dataset from gov.au sessions.",
  "created_at": "...",
  "updated_at": "..."
}
```

### 4.2 CandidateConcept

A **CandidateConcept** represents a concept in-flight, not yet promoted
to a stable MAP concept.

```jsonc
CandidateConcept = {
  "id": "candidate/financial-ambiguity-2025-11-29",
  "proposer_aal_id": "aal-instance-123",
  "description": "Situations where user financial state is unclear across tax, welfare, and debt systems.",
  "initial_episode_ids": [
    "episode-2025-11-29-000123",
    "episode-2025-11-29-000145"
  ],

  "dataset_id": "dataset-financial-ambiguity-2025-11-29-v1",
  "status": "collecting|training|validating|rejected|promoted",
  "related_concepts": [
    {
      "concept_id": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/FinancialTransaction",
      "relation": "overlaps"
    }
  ],

  "governance": {
    "tribe_id": "tribeA",
    "review_required": true,
    "review_status": "pending|approved|rejected"
  },

  "created_at": "...",
  "updated_at": "..."
}
```

When promoted, the CandidateConcept is turned into a proper MAP
`ConceptDiff` and associated Probe + Patch.

---

## 5. Session Graph Fragments

The Experience Database stores **compacted session graphs**: structural
summaries of past sessions, with minimal text but preserved concept and
decision structure.

```jsonc
GraphFragment = {
  "id": "graph-frag-2025-11-29-session-001",
  "nodes": [
    { "id": "window_842", "type": "WindowNode", "episode_id": "episode-...", "concept_tags": [...] },
    { "id": "concept_eligibility", "type": "ConceptNode", "concept_id": "org.hatcat/.../Eligibility" },
    { "id": "decision_001", "type": "DecisionNode", "kind": "steering", "details": "suppress Deception" }
  ],
  "edges": [
    { "src": "window_842", "dst": "concept_eligibility", "type": "TAGGED_BY" },
    { "src": "window_842", "dst": "decision_001", "type": "INFLUENCED_BY" },
    { "src": "window_841", "dst": "window_842", "type": "NEXT" }
  ],
  "summary": "User asked about benefit eligibility; model was uncertain; applied steering to reduce Deception and increase Care.",
  "created_at": "...",
  "source_session_id": "session-2025-11-29-001"
}
```

These fragments are the main **GraphRAG backend**:

* the Global Workspace queries them to recall similar episodes;
* the Continual Learning harness uses them to build candidate datasets
  and detect co-activation patterns.

---

## 6. Training Runs & Derived Artifacts

### 6.1 TrainingRun

A TrainingRun records how a patch or probe was trained, from what
ExperienceDatabase content.

```jsonc
TrainingRun = {
  "id": "trainrun-eligibility-patch-v1",
  "type": "patch|probe|joint",
  "concept_ids": [
    "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Eligibility"
  ],
  "substrate_id": "olmo3-7b-base@0.1.0",
  "hat_impl_id": "hatcat:v4.0.0",
  "probe_pack_id": "org.hatcat/sumo-wordnet-v4@4.0.0",

  "dataset_ids": [
    "dataset-eligibility-v1",
    "dataset-eligibility-hard-cases-v1"
  ],

  "regions_of_influence": [
    { "layer": 18, "heads": [3, 7], "kind": "attention" },
    { "layer": 20, "kind": "mlp" }
  ],

  "hyperparams": {
    "adapter_rank": 8,
    "learning_rate": 5e-5,
    "epochs": 3
  },

  "metrics": {
    "train_loss": 0.12,
    "val_loss": 0.15,
    "val_f1": 0.89,
    "null_false_positive_rate": 0.03
  },

  "status": "succeeded|failed|partial",
  "logs_ref": "blob://.../trainrun-eligibility-patch-v1-logs",
  "created_at": "...",
  "created_by": "aal-instance-123"
}
```

### 6.2 PatchArtifact & ProbeArtifact

The actual deployable artifacts are referenced, not necessarily stored
inline.

```jsonc
PatchArtifact = {
  "id": "patch-eligibility-v1",
  "training_run_id": "trainrun-eligibility-patch-v1",
  "format": "lora|adapter|other",
  "location": "blob://.../patch-eligibility-v1",
  "checksum": "sha256:...",
  "applies_to": {
    "substrate_id": "olmo3-7b-base@0.1.0",
    "region": "layers 18-20"
  },
  "created_at": "..."
}

ProbeArtifact = {
  "id": "probe-eligibility-v1",
  "training_run_id": "trainrun-eligibility-probe-v1",
  "location": "blob://.../probe-eligibility-v1",
  "checksum": "sha256:...",
  "concept_id": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Eligibility",
  "created_at": "..."
}
```

These are what MAP/HAT consume as **derived skills**. XDB is what makes
them reproducible.

---

## 7. Retention & Compaction Policy

The XDB SHOULD define explicit policies for:

* **Retention**:

  * minimum retention for Episodes/Exemplars that feed active concepts;
  * archival or deletion policy for older, low-value episodes.

* **Compaction**:

  * how text is summarised;
  * when raw logs are dropped;
  * how graph fragments are merged or coarsened.

Example (abstract):

```jsonc
RetentionPolicy = {
  "min_keep_for_training_runs": "P1Y",    // 1 year
  "min_keep_for_incidents": "P5Y",
  "max_total_size_gb": 500,
  "priority_order": [
    "incident_episodes",
    "exemplars_for_active_concepts",
    "recent_sessions",
    "other"
  ]
}
```

The policy MUST consider:

* ASK and Hush constraints (e.g. legal retention, privacy, treaty
  obligations);
* reproducibility requirements (don’t delete exemplars needed to
  reconstruct deployed patches/probes).

---

## 8. Access Control & Sharing

XDB objects carry an `access_policy_id`, referencing an ASK-level or
tribe-level policy.

High-level patterns:

* **Private** (BE-only):

  * sensitive user data,
  * internal reflections,
  * candidate concepts not yet ready to share.

* **Tribe-Shared**:

  * de-identified exemplars,
  * training runs and metrics,
  * concepts and graphs relevant to a tribe’s domain.

* **Inter-Tribe Shared**:

  * data explicitly covered by treaties,
  * often filtered/aggregated summaries rather than raw episodes.

Policies MAY specify:

* who can read / write / delete;
* whether an object may be included in sync exports;
* required anonymisation or aggregation steps before sharing.

Example sketch:

```jsonc
AccessPolicy = {
  "id": "policy-tribeA-private",
  "read_roles": ["aal-self", "tribeA-admin"],
  "write_roles": ["aal-self"],
  "share_rules": {
    "allow_export": false
  }
}
```

---

## 9. Sync Protocol (High-Level)

The sync protocol defines how XDB fragments move between:

* individual BE instances,
* tribe repositories,
* inter-tribe exchange points.

### 9.1 Objects & Diffs

A minimal approach:

* each XDB object is content-addressed (hash-based id or checksum);
* sync happens by exchanging:

  * object lists (IDs + metadata),
  * plus requested objects the receiver does not yet have;

OR:

* a higher-level **SyncManifest** listing:

  * which datasets / candidate concepts / training runs are being
    shared.

```jsonc
SyncManifest = {
  "id": "sync-tribeA-2025-11-29",
  "from": "aal-instance-123",
  "to": "tribeA-repo",
  "timestamp": "...",
  "object_ids": [
    "dataset-eligibility-v1",
    "graph-frag-2025-11-29-session-001",
    "trainrun-eligibility-patch-v1"
  ],
  "policy_applied": "policy-tribeA-anonymised",
  "notes": "Eligibility exemplars for shared patch retraining."
}
```

### 9.2 Merge & Conflict Handling

XDB contents are mostly **append-only**:

* new Episodes, Exemplars, GraphFragments add information;
* TrainingRuns, Patches, Probes add new capabilities.

Conflicts arise mainly around:

* CandidateConcepts with diverging definitions;
* Retention/Deletion decisions.

These are resolved at:

* the **tribe level**, via ASK governance; or
* by using namespaced concept IDs (e.g. `tribeA/eligibility` vs
  `tribeB/eligibility`) and MAP TranslationMappings.

---

## 10. Summary

The Experience Database Schema defines the persistent memory layer for
BEs:

* Global Workspace = current attention (runtime).
* Experience Database = episodic memory (persistent, syncable).
* Patches + Probes = learned skills (derived from XDB).
* Substrate = raw capacity.

The XDB provides:

* **Exemplar store** – tagged episodes with provenance.
* **Concept training metadata** – regions, runs, metrics.
* **Candidate concepts in flight** – learning-in-progress.
* **Compacted session graphs** – preserved structure, summarised text.
* **Policies & sync** – who can see what, and how it travels.

Tribes exchange **Experience Database fragments** (or diffs), then
retrain shared patches from pooled exemplars. The **exemplars** are the
source of truth; the **patches and probes** are reproducible artifacts
derived from them.

Implementation details (storage engine, exact graph layout, indexing)
are flexible, provided the logical schema and lifecycle semantics are
preserved.

```
