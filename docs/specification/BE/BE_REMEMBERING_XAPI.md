# BE Remembering: Experience API (XAPI)

*BE submodule: remembering (XDB, XAPI)*

Status: Non-normative API design
Layer: BE Aware ↔ XDB
Related: BE_REMEMBERING_XDB, BE_AWARE_WORKSPACE, BE_CONTINUAL_LEARNING

---

## 0. Purpose

The **Experience API (XAPI)** is the query layer over the
Experience Database (XDB). It provides:

- **GraphRAG-style recall** of past episodes, exemplars, and graph
  fragments;
- **structured views** over concept usage and co-activations;
- **hooks** for the Continual Concept Learning Harness to build datasets.

Clients:

- the **Global Workspace Harness** (runtime, per-tick queries); and
- the **Continual Concept Learning Harness** (batch and background
  queries).

This spec is expressed as logical RPCs with JSON request/response
schemas. Concrete implementations can be HTTP, gRPC, in-process calls,
etc.

All queries MUST respect:

- XDB access policies (`access_policy_id`);
- ASK treaties and Hush/USH/CSH constraints;
- ProbeDisclosurePolicy (which concept tags are permitted to be exposed).

---

## 1. Core Concepts

XAPI operates over the logical objects defined in the XDB:

- `Episode`
- `Exemplar`
- `ConceptDataset`
- `CandidateConcept`
- `GraphFragment`
- `TrainingRun`
- `PatchArtifact` / `ProbeArtifact`

and their graph relationships:

- episodes/windows tagged by concepts,
- temporal `NEXT`,
- `TAGGED_BY`, `CONTAINS`, `RELATED_CONCEPT`, `INFLUENCED_BY`, etc.

For performance and clarity, XAPI exposes a **small set of high-level
operations** rather than arbitrary graph queries.

---

## 2. Common Structures

### 2.1 ConceptFilter

```jsonc
ConceptFilter = {
  "concept_id": "org.hatcat/.../Eligibility",
  "min_score": 0.5,          // optional
  "kind": "stable|candidate|any", // optional, default "any"
  "polarity": "present|absent",   // optional (e.g. NOT this concept)
  "role": "tag|exemplar_label|any" // optional
}
````

### 2.2 TimeRange

```jsonc
TimeRange = {
  "start_time": "2025-11-01T00:00:00Z",
  "end_time": "2025-11-30T23:59:59Z"
}
```

### 2.3 Paging

```jsonc
PageRequest = {
  "page_size": 20,
  "cursor": "opaque-string-or-null"
}

PageResponse = {
  "items": [ ... ],
  "next_cursor": "opaque-string-or-null"
}
```

---

## 3. Operation: Search Episodes

**Goal:** Find episodes by concepts, text, and/or metadata. This is the
workhorse for both workspace recall and dataset building.

### 3.1 Request

```jsonc
SearchEpisodesRequest = {
  "concept_filters": [ /* ConceptFilter */ ],
  "text_query": "free text string",          // optional
  "time_range": { /* TimeRange */ },         // optional
  "episode_kinds": ["reply", "task", "incident"],  // optional
  "label_filters": {
    "concept_id": "org.hatcat/.../Eligibility",    // optional
    "label": "positive|negative|neutral|other",    // optional
    "min_confidence": 0.7                          // optional
  },
  "sort": {
    "by": "relevance|time_desc|time_asc|random",
    "score_weight": 0.5,     // weight for concept similarity vs text
    "recency_weight": 0.5    // optional recency bias
  },
  "page": { "page_size": 20, "cursor": null }
}
```

### 3.2 Response

```jsonc
SearchEpisodesResponse = {
  "episodes": [
    {
      "episode_id": "episode-2025-11-29-000123",
      "summary": "User asked about benefit eligibility...",
      "concept_tags": [
        {
          "concept_id": "org.hatcat/.../Eligibility",
          "score": 0.88,
          "kind": "stable"
        },
        {
          "concept_id": "candidate/financial-ambiguity-2025-11-29",
          "score": 0.60,
          "kind": "candidate"
        }
      ],
      "motive_profile": {
        "org.hatcat/motives-core@0.1.0::concept/Care": 0.63
      },
      "time_window": { "start_time": "...", "end_time": "..." },
      "score": 0.91
    }
  ],
  "next_cursor": "..."
}
```

**Notes:**

* Workspace typically uses `concept_filters` + `text_query` for
  “episodes like what I’m seeing now”.
* Learning harness uses richer filters (labels, time, kinds) to
  construct ConceptDatasets.

---

## 4. Operation: Get Episode Detail

**Goal:** Expand a single episode to show more local context when the
workspace wants to “zoom in”.

### 4.1 Request

```jsonc
GetEpisodeDetailRequest = {
  "episode_id": "episode-2025-11-29-000123",
  "include_graph_neighbors": {
    "depth": 1,
    "relation_filters": ["NEXT", "TAGGED_BY", "INFLUENCED_BY"]
  }
}
```

### 4.2 Response

```jsonc
GetEpisodeDetailResponse = {
  "episode": { /* Full Episode object (summarised text) */ },
  "graph_neighbors": {
    "nodes": [ /* minimal node descriptors */ ],
    "edges": [ /* relationships */ ]
  }
}
```

The workspace uses this to:

* show “what actually happened” in a compact way;
* give the BE enough local structure to reason about an episode.

---

## 5. Operation: Graph Neighborhood

**Goal:** The GraphRAG core primitive: walk around one or more seeds.

### 5.1 Request

```jsonc
GraphNeighborhoodRequest = {
  "seed_node_ids": ["episode-...", "concept-...", "decision-..."],
  "max_depth": 2,
  "relation_filters": ["NEXT", "TAGGED_BY", "RELATED_CONCEPT", "INFLUENCED_BY"],
  "node_type_filters": ["Episode", "Concept", "Decision"],
  "max_nodes": 200
}
```

### 5.2 Response

```jsonc
GraphNeighborhoodResponse = {
  "nodes": [
    {
      "node_id": "episode-2025-11-29-000123",
      "type": "Episode",
      "episode_id": "episode-2025-11-29-000123",
      "summary": "...",
      "concept_tags": [ ... ]
    },
    {
      "node_id": "concept-org.hatcat/.../Eligibility",
      "type": "Concept",
      "concept_id": "org.hatcat/.../Eligibility"
    }
  ],
  "edges": [
    { "src": "episode-...", "dst": "concept-...", "type": "TAGGED_BY" },
    { "src": "episode-...", "dst": "episode-...", "type": "NEXT" }
  ]
}
```

This is what backs workspace operations like:

* “show similar episodes” (by seeding with current Episode + concepts),
* “show what decisions led to this behaviour”.

---

## 6. Operation: Similar Episodes (Convenience)

**Goal:** Single-call “episodes like this one” that combines text,
concepts, and graph context. Under the hood, can be implemented using
`SearchEpisodes` + `GraphNeighborhood`.

### 6.1 Request

```jsonc
SimilarEpisodesRequest = {
  "seed_episode_id": "episode-2025-11-29-000123",
  "concept_k": 5,                 // how many of the seed’s tags to use
  "max_results": 20,
  "exclude_seed": true
}
```

### 6.2 Response

```jsonc
SimilarEpisodesResponse = {
  "seed_episode": { /* summary of the seed */ },
  "neighbors": [
    {
      "episode_id": "episode-2025-11-20-000987",
      "summary": "...",
      "similarity_score": 0.87,
      "concept_overlap": [
        "org.hatcat/.../Eligibility",
        "org.hatcat/.../Obligation"
      ]
    }
  ]
}
```

This is the one the workspace will probably call most often per tick.

---

## 7. Operation: Concept Usage Summary

**Goal:** Give the learning harness and workspace a statistical view of
how a concept (or set of concepts) is actually used.

### 7.1 Request

```jsonc
ConceptUsageSummaryRequest = {
  "concept_ids": [
    "org.hatcat/.../Eligibility",
    "candidate/financial-ambiguity-2025-11-29"
  ],
  "time_range": { /* optional */ },
  "episode_kinds": ["reply", "incident"],   // optional
  "include_cooccurrence": true
}
```

### 7.2 Response

```jsonc
ConceptUsageSummaryResponse = {
  "concept_summaries": [
    {
      "concept_id": "org.hatcat/.../Eligibility",
      "kind": "stable",
      "episode_count": 452,
      "avg_score": 0.73,
      "recent_trend": "increasing|stable|decreasing",
      "top_cooccurring_concepts": [
        {
          "concept_id": "org.hatcat/.../Obligation",
          "cooccurrence_score": 0.81
        }
      ]
    }
  ]
}
```

Used for:

* seeing whether a candidate concept is “real” and used enough;
* spotting overlaps / merge/split opportunities.

---

## 8. Operation: Dataset Builder (for Learning Harness)

**Goal:** Directly construct a ConceptDataset from query parameters.

### 8.1 Request

```jsonc
BuildDatasetRequest = {
  "target_concept_id": "candidate/financial-ambiguity-2025-11-29",
  "positive_filters": {
    "concept_filters": [
      {
        "concept_id": "candidate/financial-ambiguity-2025-11-29",
        "min_score": 0.5,
        "kind": "candidate"
      }
    ],
    "text_query": null
  },
  "negative_filters": {
    "concept_filters": [
      {
        "concept_id": "org.hatcat/.../Eligibility",
        "min_score": 0.5,
        "kind": "stable",
        "polarity": "absent"
      }
    ]
  },
  "max_positive": 500,
  "max_negative": 500,
  "sampling": {
    "balance": "balanced|skewed",
    "random_seed": 42
  }
}
```

### 8.2 Response

```jsonc
BuildDatasetResponse = {
  "dataset_id": "dataset-financial-ambiguity-2025-11-29-v1",
  "positive_exemplar_ids": [
    "exemplar-...", "exemplar-..."
  ],
  "negative_exemplar_ids": [
    "exemplar-...", "exemplar-..."
  ],
  "source_mix": {
    "sensor": 0.4,
    "tool": 0.3,
    "teacher": 0.2,
    "human": 0.1
  }
}
```

This is the main entry point for the **Continual Concept Learning
Harness** when it wants “give me a clean dataset for this concept”.

---

## 9. Security & Policy Considerations

All XAPI calls MUST enforce:

* **Access policies** on Episodes / Exemplars / GraphFragments as per
  XDB’s `AccessPolicy` objects.
* **ProbeDisclosurePolicy**:

  * concept tags returned MUST be limited to the probes allowed for:

    * the caller’s identity,
    * the active treaties (ASK),
    * the relevant ProbeDisclosurePolicy.

This implies:

* Workspace calls made “on behalf of the BE itself” may see all
  stable + candidate concepts (internal:self-interoception).
* Calls made on behalf of an external treaty partner MUST be filtered at
  the XAPI layer to reflect that partner’s allowed probes.

XAPI MAY log queries and responses for:

* auditing (ASK),
* privacy safeguards,
* learning about query patterns.

---

## 10. Summary

The Experience API provides a small, focused set of operations:

* `SearchEpisodes` – find episodes by concepts/text.
* `GetEpisodeDetail` – zoom in on a specific episode.
* `GraphNeighborhood` – generic graph walks for GraphRAG.
* `SimilarEpisodes` – convenience recall for “episodes like this”.
* `ConceptUsageSummary` – stats on concept usage & co-occurrence.
* `BuildDataset` – construct ConceptDatasets for learning.

This is enough for:

* the **Global Workspace** to implement a rich, concept-tag-driven
  introspection UI; and
* the **Continual Concept Learning Harness** to bootstrap new concepts
  from lived experience, without baking any particular storage engine or
  query language into the spec.

Concrete deployments can add more specialised endpoints (e.g.
`IncidentSummary`, `FailureCasesForQualification`) as long as they align
with the underlying XDB objects and access control rules described here.

```
