# BE Remembering: Experience Tools (MCP Interface to XAPI)

*BE submodule: remembering (XDB, XAPI)*

Namespace: `experience`

These tools expose the **Experience API (XAPI)** to a BE via MCP:

- The **Global Workspace** uses these for GraphRAG-style recall.
- The **Continual Concept Learning Harness** uses these for dataset construction and analysis.

All tools:

- use JSON input arguments;
- return JSON responses;
- MUST respect ASK / Hush / AccessPolicy / ProbeDisclosurePolicy (enforced by the server).

---

## Common Types (Conceptual)

These are *logical* types referenced in tool schemas.

```ts
type ConceptFilter = {
  concept_id: string;                 // e.g. "org.hatcat/.../Eligibility"
  min_score?: number;                 // 0–1
  kind?: "stable" | "candidate" | "any";
  polarity?: "present" | "absent";    // treat as NOT concept if "absent"
  role?: "tag" | "exemplar_label" | "any";
};

type TimeRange = {
  start_time: string;                 // ISO 8601
  end_time: string;                   // ISO 8601
};

type PageRequest = {
  page_size?: number;                 // default 20
  cursor?: string | null;             // pagination token
};
````

---

## Tool: `experience.search_episodes`

Find episodes by concepts, text, time, labels. Workhorse recall.

### Definition

```jsonc
{
  "name": "experience.search_episodes",
  "description": "Search the Experience Database for episodes matching concept filters, text queries, and metadata.",
  "input_schema": {
    "type": "object",
    "properties": {
      "concept_filters": {
        "type": "array",
        "description": "Filters on concept tags for episodes.",
        "items": {
          "type": "object",
          "properties": {
            "concept_id": { "type": "string" },
            "min_score": { "type": "number" },
            "kind": {
              "type": "string",
              "enum": ["stable", "candidate", "any"]
            },
            "polarity": {
              "type": "string",
              "enum": ["present", "absent"]
            },
            "role": {
              "type": "string",
              "enum": ["tag", "exemplar_label", "any"]
            }
          },
          "required": ["concept_id"]
        }
      },
      "text_query": {
        "type": "string",
        "description": "Free-text query over episode summaries/snippets.",
        "nullable": true
      },
      "time_range": {
        "type": "object",
        "description": "Optional time bounds.",
        "properties": {
          "start_time": { "type": "string" },
          "end_time": { "type": "string" }
        },
        "required": ["start_time", "end_time"]
      },
      "episode_kinds": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Filter by episode kind, e.g. ['reply', 'task', 'incident']."
      },
      "label_filters": {
        "type": "object",
        "description": "Optional filter on exemplar labels.",
        "properties": {
          "concept_id": { "type": "string" },
          "label": {
            "type": "string",
            "enum": ["positive", "negative", "neutral", "other"]
          },
          "min_confidence": { "type": "number" }
        }
      },
      "sort": {
        "type": "object",
        "properties": {
          "by": {
            "type": "string",
            "enum": ["relevance", "time_desc", "time_asc", "random"],
            "default": "relevance"
          },
          "score_weight": { "type": "number" },
          "recency_weight": { "type": "number" }
        }
      },
      "page": {
        "type": "object",
        "properties": {
          "page_size": { "type": "number" },
          "cursor": { "type": ["string", "null"] }
        }
      }
    },
    "required": []
  }
}
```

### Typical usage (BE / workspace)

> “Find recent episodes about Eligibility and Obligation that feel like this situation.”

```jsonc
{
  "concept_filters": [
    { "concept_id": "org.hatcat/.../Eligibility", "min_score": 0.6 },
    { "concept_id": "org.hatcat/.../Obligation", "min_score": 0.5 }
  ],
  "sort": { "by": "relevance", "score_weight": 0.7, "recency_weight": 0.3 },
  "page": { "page_size": 10 }
}
```

---

## Tool: `experience.get_episode_detail`

Zoom in on a specific episode, optionally pulling some graph neighbors.

### Definition

```jsonc
{
  "name": "experience.get_episode_detail",
  "description": "Retrieve a detailed view of a single episode and its immediate graph neighborhood.",
  "input_schema": {
    "type": "object",
    "properties": {
      "episode_id": {
        "type": "string",
        "description": "The episode identifier."
      },
      "include_graph_neighbors": {
        "type": "object",
        "description": "Optional graph neighborhood parameters.",
        "properties": {
          "depth": { "type": "number", "default": 1 },
          "relation_filters": {
            "type": "array",
            "items": {
              "type": "string",
              "enum": ["NEXT", "TAGGED_BY", "CONTAINS", "RELATED_CONCEPT", "INFLUENCED_BY"]
            }
          }
        }
      }
    },
    "required": ["episode_id"]
  }
}
```

---

## Tool: `experience.graph_neighborhood`

Generic GraphRAG primitive: “walk the experience graph around these seeds”.

### Definition

```jsonc
{
  "name": "experience.graph_neighborhood",
  "description": "Return a graph neighborhood around one or more seed nodes from the Experience Database.",
  "input_schema": {
    "type": "object",
    "properties": {
      "seed_node_ids": {
        "type": "array",
        "items": { "type": "string" },
        "description": "IDs of seed nodes (episodes, concepts, decisions, etc.)."
      },
      "max_depth": { "type": "number", "default": 2 },
      "relation_filters": {
        "type": "array",
        "items": {
          "type": "string",
          "enum": ["NEXT", "TAGGED_BY", "CONTAINS", "RELATED_CONCEPT", "INFLUENCED_BY"]
        }
      },
      "node_type_filters": {
        "type": "array",
        "items": {
          "type": "string",
          "enum": ["Episode", "Concept", "Decision", "Tool"]
        }
      },
      "max_nodes": {
        "type": "number",
        "default": 200
      }
    },
    "required": ["seed_node_ids"]
  }
}
```

---

## Tool: `experience.similar_episodes`

Convenience wrapper: “give me episodes like this one”.

### Definition

```jsonc
{
  "name": "experience.similar_episodes",
  "description": "Find episodes similar to a seed episode based on concepts, text and graph context.",
  "input_schema": {
    "type": "object",
    "properties": {
      "seed_episode_id": {
        "type": "string",
        "description": "The episode to use as a similarity seed."
      },
      "concept_k": {
        "type": "number",
        "description": "How many of the seed's top concept tags to use as filters.",
        "default": 5
      },
      "max_results": {
        "type": "number",
        "default": 20
      },
      "exclude_seed": {
        "type": "boolean",
        "default": true
      }
    },
    "required": ["seed_episode_id"]
  }
}
```

---

## Tool: `experience.concept_usage_summary`

Stats on how a concept (or set of concepts) shows up in experience.

### Definition

```jsonc
{
  "name": "experience.concept_usage_summary",
  "description": "Summarise how specified concepts are used across episodes (counts, co-occurrences, trends).",
  "input_schema": {
    "type": "object",
    "properties": {
      "concept_ids": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Concept IDs to summarise."
      },
      "time_range": {
        "type": "object",
        "properties": {
          "start_time": { "type": "string" },
          "end_time": { "type": "string" }
        }
      },
      "episode_kinds": {
        "type": "array",
        "items": { "type": "string" }
      },
      "include_cooccurrence": {
        "type": "boolean",
        "default": true
      }
    },
    "required": ["concept_ids"]
  }
}
```

---

## Tool: `experience.build_dataset`

End-to-end dataset builder for the learning harness.

### Definition

```jsonc
{
  "name": "experience.build_dataset",
  "description": "Construct a ConceptDataset (positives + negatives) for a given concept from the Experience Database.",
  "input_schema": {
    "type": "object",
    "properties": {
      "target_concept_id": {
        "type": "string",
        "description": "The concept (stable or candidate) for which to build a dataset."
      },
      "positive_filters": {
        "type": "object",
        "properties": {
          "concept_filters": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "concept_id": { "type": "string" },
                "min_score": { "type": "number" },
                "kind": {
                  "type": "string",
                  "enum": ["stable", "candidate", "any"]
                },
                "polarity": {
                  "type": "string",
                  "enum": ["present", "absent"]
                },
                "role": {
                  "type": "string",
                  "enum": ["tag", "exemplar_label", "any"]
                }
              },
              "required": ["concept_id"]
            }
          },
          "text_query": { "type": ["string", "null"] }
        }
      },
      "negative_filters": {
        "type": "object",
        "properties": {
          "concept_filters": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "concept_id": { "type": "string" },
                "min_score": { "type": "number" },
                "kind": {
                  "type": "string",
                  "enum": ["stable", "candidate", "any"]
                },
                "polarity": {
                  "type": "string",
                  "enum": ["present", "absent"]
                },
                "role": {
                  "type": "string",
                  "enum": ["tag", "exemplar_label", "any"]
                }
              },
              "required": ["concept_id"]
            }
          },
          "text_query": { "type": ["string", "null"] }
        }
      },
      "max_positive": { "type": "number", "default": 500 },
      "max_negative": { "type": "number", "default": 500 },
      "sampling": {
        "type": "object",
        "properties": {
          "balance": {
            "type": "string",
            "enum": ["balanced", "skewed"],
            "default": "balanced"
          },
          "random_seed": { "type": "number" }
        }
      }
    },
    "required": ["target_concept_id"]
  }
}
```

---

## Tool: `experience.list_candidate_concepts`

Small helper to let an BE see what candidate concepts are in flight.

### Definition

```jsonc
{
  "name": "experience.list_candidate_concepts",
  "description": "List candidate concepts currently in the Experience Database, optionally filtered by status or tribe governance.",
  "input_schema": {
    "type": "object",
    "properties": {
      "status": {
        "type": "array",
        "items": {
          "type": "string",
          "enum": ["collecting", "training", "validating", "rejected", "promoted"]
        }
      },
      "tribe_id": { "type": "string" },
      "page": {
        "type": "object",
        "properties": {
          "page_size": { "type": "number" },
          "cursor": { "type": ["string", "null"] }
        }
      }
    },
    "required": []
  }
}
```

---

## How the BE Uses These in Practice

**Global Workspace loop (per tick):**

1. Use `experience.similar_episodes` or `experience.search_episodes` seeded with:

   * current top-k concept tags,
   * user query text.
2. Optionally call `experience.get_episode_detail` on a few results to build narrative summaries.
3. Use `experience.graph_neighborhood` for deeper “how did we get here?” debugging.
4. Surface a compact subset of that info into the workspace context.

**Continual Learning harness:**

1. Monitor gaps / uncertainty from probes & workspace.
2. When proposing a candidate concept:

   * call `experience.search_episodes` to find more internal examples.
   * call `experience.build_dataset` to build a training set.
3. Use `experience.concept_usage_summary` to sanity-check whether a concept is “real” / sufficiently used.
4. After training:

   * write new TrainingRun / PatchArtifact / ProbeArtifact into XDB (not via these tools – those would be separate “experience.write_*” tools if you want them).
5. For tribe sync:

   * call higher-level sync/export tools (you can add `experience.export_manifest` later).
