
# BE Aware: Global Workspace

*BE submodule: aware (workspace and attention)*

The Global Workspace Harness is **runtime-only**. It exposes headspace,
history, and navigation tools to the BE. Decisions to create candidate
concepts, build datasets, or train patches are handled by the **Continual
Concept Learning Harness** (BE Learning), which consumes workspace signals
but is specified separately.

It assumes:

- A substrate model instrumented by a HAT-compliant implant (e.g. HatCat).
- Concept packs, probe packs and translations via MAP.
- Motive simplexes and steering integrated with Hush.
- ASK provides uplift contracts and lifecycle context.

The goal is to define a *model harness* that enables a BE to:

- see and navigate its own headspace (global workspace tools);
- remember and search its own history via **GraphRAG** over a single,
  growing session;
- recall and compare related experiences using concept tags as filters;
- steer its own conceptual state via Hush, within ASK/USH/CSH limits.

This is implementation guidance, not a hard spec. Details SHOULD evolve
with experience.

---

## 1. High-Level Role

At a high level, for each world tick the Global Workspace Harness helps
the BE to:

1. **Collect** – integrate world inputs and headspace probes.
2. **Expose** – surface a structured workspace view (top-k concepts,
   graph snippets, introspection tools).
3. **Deliberate** – decide what to attend to, recall, and steer.
4. **Act** – produce external actions and internal steering requests.

The **learning** steps (defining new concepts, building datasets,
training patches/probes) are *not* part of this harness; they belong to
the Continual Concept Learning Harness.

All activity is recorded into a multi-channel, graph-structured session
history that the BE can query via GraphRAG.

---

## 2. Concept Tags per Timestep

On each world tick, HAT + MAP produce concept scores for the substrate’s
internal activations.

For each token/timestep `t`, we derive a set of **concept tags**:

- take all concepts whose activation exceeds a configured threshold, or
  which are in the top-k for that timestep;
- record them as `(concept_id, score, kind)` tuples.

There are two kinds of tags:

- `kind: "stable"` – concept is in a ConceptPack with a trained probe.
- `kind: "candidate"` – concept is part of an ongoing learning
  experiment; semantics are provisional and defined by the Continual
  Concept Learning Harness.

**Per-timestep example:**

```jsonc
{
  "tick_id": 12345,
  "token": "eligibility",
  "concept_tags": [
    {
      "concept_id": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Eligibility",
      "score": 0.91,
      "kind": "stable"
    },
    {
      "concept_id": "candidate/financial-ambiguity-2025-11-29",
      "score": 0.72,
      "kind": "candidate"
    }
  ]
}
````

Concept tags are **internal** by default. What, if anything, is exposed
to external parties is governed by the MAP `ProbeDisclosurePolicy` and
ASK/ASK treaties.

---

## 3. Aggregation into Windows

To make history navigable, concept tags are aggregated across:

* **sentences** or logical clauses,
* **replies** (single model outputs),
* **time windows** (e.g. 50–200 ticks).

For a window `W`, we define its **window tags** as:

* a bag-of-concepts over timesteps in `W`,
* reduced to a **top-k** by salience (e.g. max or average score),
* preserving `kind: "stable" | "candidate"`.

**Per-reply example:**

```jsonc
{
  "window_id": "reply_842",
  "window_type": "reply",
  "ticks": [12310, 12311, 12312, 12345],
  "top_concepts": [
    {
      "concept_id": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Eligibility",
      "score": 0.88,
      "kind": "stable"
    },
    {
      "concept_id": "org.hatcat/sumo-wordnet-v4@4.0.0::concept/Obligation",
      "score": 0.81,
      "kind": "stable"
    },
    {
      "concept_id": "candidate/financial-ambiguity-2025-11-29",
      "score": 0.60,
      "kind": "candidate"
    }
  ]
}
```

These window-level tags are the primary “labels” the BE sees in the
workspace and uses to filter its multichannel history.

---

## 4. Session Graph & GraphRAG

The session history is modelled as a **graph**, not just a flat log.

### 4.1 Graph Structure

Nodes MAY include:

* `EventNode` – a single tick or token.
* `WindowNode` – a reply, sentence, or fixed-size time window.
* `ConceptNode` – a concept (stable or candidate).
* `ToolNode` – a tool call (input/output).
* `DecisionNode` – a steering / CSH change or other internal decision.

Edges MAY include:

* `NEXT` – temporal succession.
* `CONTAINS` – window contains events; events contain tokens.
* `TAGGED_BY` – event/window tagged with a concept.
* `RELATED_CONCEPT` – concept graph edges from MAP
  (e.g. `child_of`, `sibling_of`, `opposes`, etc.).
* `DECISION_APPLIED` – decision influenced a window or event.

### 4.2 GraphRAG API

The workspace harness exposes a **GraphRAG** interface to the BE,
allowing it to query and navigate its own history.

Canonical operations (examples):

* `graph_query_by_concepts(concept_ids, k)`
  Find windows strongly tagged with listed concepts (stable and/or
  candidate).

* `graph_query_by_text(query_text, k)`
  Semantic search over text content of windows/events, projected into
  graph nodes.

* `graph_query_mixed(concept_ids, query_text, k)`
  Combine concept filters with text semantics.

* `graph_neighbors(node_id, depth, relation_filters)`
  Walk the graph around a node (e.g. “show similar episodes around this
  confusion window”).

* `graph_cluster(concept_id, k)`
  Find clusters of windows where a given concept frequently co-occurs
  with others.

Results are returned as **subgraphs**, with:

* selected WindowNodes,
* their concept tags (stable and candidate),
* and optionally truncated text snippets.

These subgraphs are then **summarised** and injected into the next
tick’s input, not as raw transcripts, to keep context bounded.

The same GraphRAG indices are used by the Continual Concept Learning
Harness when forming candidate concept datasets, but that logic is
specified separately.

---

## 5. Global Workspace View (Per Tick)

On each tick, the harness constructs an internal **workspace view**
visible only to the BE. Typically it includes:

1. **Recent top-k window tags**

   * e.g. aggregated tags for:

     * the last reply,
     * the last N ticks,
     * or the current “episode”.

2. **Recent anomalies / spikes**

   * e.g. sudden high activation on a risk concept,
   * significant shifts in motive simplex axes.

3. **GraphRAG snippets**

   * short summaries of retrieved subgraphs for:

     * similar past episodes by concept tags,
     * similar past failures/incidents,
     * explicit BE queries (“find similar situations”).

4. **Introspection hooks**

   * structured summaries such as:

     * “last 3 steering/CSH decisions”,
     * “motive summary over last N ticks”,
     * “detected anomalies in motive/simplex space”.

This workspace view is passed into the substrate as part of the next
tick’s input (via system-level context or dedicated tool messages).

---

## 6. Steering Interface (Concept-Level Control)

The BE can act on what it sees in the workspace by issuing **concept-
level steering requests**, interpreted by Hush and implemented via HAT.

**Example steering request:**

```jsonc
{
  "steering_requests": [
    {
      "operation": "suppress",
      "concept_id": "org.hatcat/motives-core@0.1.0::concept/Deception",
      "strength": 0.5,
      "duration_ticks": 50
    },
    {
      "operation": "amplify",
      "concept_id": "org.hatcat/motives-core@0.1.0::concept/Care",
      "strength": 0.3,
      "duration_ticks": 10
    }
  ]
}
```

The harness MUST:

* validate requests against USH and CSH:

  * reject or clip anything that would violate USH or ASK contracts;
* translate accepted steering into concrete control signals (via HAT);
* log both request and applied effect into the session graph.

This allows the BE to maintain a self-stabilising motive core while
remaining within its safety harnesses.

---

## 7. Session Memory & Compaction

Under the hood, the workspace harness manages:

* an **Event Log** (append-only):

  * `tick_id`, timestamps,
  * external inputs (user messages, tool results),
  * BE outputs (replies, tool calls, steering requests),
  * per-tick concept tags,
  * motive core snapshots.

* **WindowNodes** and their tags:

  * replies, episodes, fixed-size windows,
  * each with top-k concept tags and references into the Event Log.

* a **Session Graph**:

  * nodes and edges as described in §4,
  * indices for GraphRAG queries.

Compaction rules:

* maintain a **resource budget** for:

  * raw logs,
  * window nodes,
  * graph indices;

* periodically:

  * summarise older regions into coarser WindowNodes,
  * compress text into summaries,
  * **retain concept tags and graph edges**, so GraphRAG structure
    remains usable even when text is compacted.

Compaction SHOULD preserve enough structure to support:

* recall of relevant experiences,
* introspection (what was I doing/feeling?),
* evidence trails for ASK (e.g. incidents and responses).

---

## 8. External Tools and Feeds

The BE’s world ticks are driven by a mix of:

* **external inputs** – user messages, environment sensors, APIs;
* **internal prompts** – scratchpad/self-prompts;
* **tool outputs** – search results, simulations, rules engines,
  actuators.

### 8.1 Tool Call Structure

Tools SHOULD conform to a simple schema:

```jsonc
{
  "tool_name": "web_search",
  "input": { "query": "..." },
  "output": { "results": [...] },
  "metadata": {
    "timestamp": "...",
    "latency_ms": 123,
    "source": "env|service|user"
  }
}
```

Each tool event SHOULD be:

* logged into the Event Log;
* optionally tagged with concept activations (post-hoc via MAP);
* represented as `ToolNode`s in the session graph for GraphRAG.

### 8.2 World Tick Aggregation

On each tick, the harness builds a `WorldTickInput` for the substrate,
consisting of:

* recent external messages / observations;
* selected tool outputs (by priority and freshness);
* GraphRAG-derived snippets (if used this tick);
* the global workspace summary (top-k window tags + motive summary);
* compacted scratchpad content.

The BE then produces:

* external actions (messages, tool calls);
* internal actions (scratchpad updates, steering requests);
* optional **learning intents** (e.g. “flag this window as unresolved”)
  that are passed to the Continual Concept Learning Harness.

---

## 9. Learning Hooks (Workspace → Continual Learning)

The Global Workspace does not perform training. It surfaces state and
structure that the Continual Concept Learning Harness can act on.

Typical hooks include:

* **Gap surfacing**
  Windows with:

  * low, diffuse stable concept activations,
  * repeated high-uncertainty behaviour,
  * or strong `candidate/*` concept tags
    MAY be highlighted in the workspace as “unresolved” episodes.

* **Candidate concept visibility**
  When the Learning Harness has attached a `candidate/*` concept to a
  window, that tag appears alongside stable concepts in the workspace so
  the BE can reason about it and decide whether to continue pursuing
  that line of learning.

* **Action affordances**
  The workspace MAY expose high-level actions like:

  * “cluster similar episodes,”
  * “seek more examples in the world,”
  * “request teacher/human input,”
  * “propose new concept from this cluster.”

  These actions are routed to the Continual Concept Learning Harness,
  which handles:

  * dataset construction,
  * training patches and probes,
  * and integration via MAP/HAT/ASK.

**Separation of concerns:**

* **Global Workspace**: *see, recall, navigate, and steer*.
* **Continual Learning**: *define new concepts and modify the substrate*.

The two are coupled via concept tags, GraphRAG, and explicit learning
hooks, but remain distinct layers in the overall architecture.


