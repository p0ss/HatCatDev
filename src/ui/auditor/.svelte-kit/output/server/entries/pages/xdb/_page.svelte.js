import { s as store_get, b as attr, e as ensure_array_like, a as attr_class, u as unsubscribe_stores } from "../../../chunks/index2.js";
import { b as bedStatus } from "../../../chunks/bed.js";
import { W as escape_html } from "../../../chunks/context.js";
const API_BASE = "/api";
async function fetchJson(path, options) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...options?.headers
    },
    ...options
  });
  if (!response.ok) {
    const error = await response.text();
    throw new Error(`API error ${response.status}: ${error}`);
  }
  return response.json();
}
const xdb = {
  async getStatus(xdbId) {
    return fetchJson(`/xdb/status/${xdbId}`);
  },
  async getRecent(xdbId, n = 100) {
    return fetchJson(`/xdb/recent/${xdbId}?n=${n}`);
  },
  async query(xdbId, params) {
    return fetchJson("/xdb/query", {
      method: "POST",
      body: JSON.stringify({ xdb_id: xdbId, ...params })
    });
  },
  async getTags(xdbId, tagType, pattern) {
    const params = new URLSearchParams();
    if (tagType) params.set("tag_type", tagType);
    if (pattern) params.set("pattern", pattern);
    return fetchJson(`/xdb/tags/${xdbId}?${params}`);
  },
  async getBuds(xdbId, status) {
    const params = status ? `?status=${status}` : "";
    return fetchJson(`/xdb/buds/${xdbId}${params}`);
  },
  async getConcepts(xdbId, parent) {
    const params = parent ? `?parent=${parent}` : "";
    return fetchJson(`/xdb/concepts/${xdbId}${params}`);
  },
  async findConcept(xdbId, query) {
    return fetchJson(`/xdb/find-concept/${xdbId}?query=${encodeURIComponent(query)}`);
  },
  async getGraphNeighborhood(xdbId, seedIds, maxDepth = 2, direction = "both") {
    return fetchJson("/xdb/graph-neighborhood", {
      method: "POST",
      body: JSON.stringify({
        xdb_id: xdbId,
        seed_ids: seedIds,
        max_depth: maxDepth,
        direction
      })
    });
  },
  async getQuota(xdbId) {
    return fetchJson(`/xdb/quota/${xdbId}`);
  },
  async getContext(xdbId) {
    return fetchJson(`/xdb/context/${xdbId}`);
  }
};
function _page($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    var $$store_subs;
    let xdbId = store_get($$store_subs ??= {}, "$bedStatus", bedStatus)?.xdb_id ?? "xdb-default";
    let timesteps = [];
    let tags = [];
    let searchText = "";
    let selectedTagType = "";
    let queryLimit = 100;
    async function loadTags() {
      try {
        const result = await xdb.getTags(xdbId, selectedTagType || void 0);
        tags = result.tags ?? [];
      } catch (e) {
        console.error("Failed to load tags:", e);
      }
    }
    function formatTime(ts) {
      return new Date(ts).toLocaleString();
    }
    $$renderer2.push(`<div class="xdb-page svelte-4p4iat"><div class="header svelte-4p4iat"><h2 class="svelte-4p4iat">XDB Browser</h2> <span class="xdb-id svelte-4p4iat">${escape_html(
      // Initial load
      xdbId
    )}</span></div> <div class="search-bar svelte-4p4iat"><input type="text" placeholder="Search content..."${attr("value", searchText)} class="svelte-4p4iat"/> <button class="svelte-4p4iat">Search</button> <button class="svelte-4p4iat">Recent</button> `);
    $$renderer2.select(
      { value: queryLimit, class: "" },
      ($$renderer3) => {
        $$renderer3.option({ value: 50 }, ($$renderer4) => {
          $$renderer4.push(`50`);
        });
        $$renderer3.option({ value: 100 }, ($$renderer4) => {
          $$renderer4.push(`100`);
        });
        $$renderer3.option({ value: 500 }, ($$renderer4) => {
          $$renderer4.push(`500`);
        });
      },
      "svelte-4p4iat"
    );
    $$renderer2.push(`</div> `);
    {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> <div class="content svelte-4p4iat"><div class="timesteps-panel svelte-4p4iat"><h3 class="svelte-4p4iat">Timesteps (${escape_html(timesteps.length)})</h3> `);
    {
      $$renderer2.push("<!--[!-->");
      if (timesteps.length === 0) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<p class="empty svelte-4p4iat">No timesteps found</p>`);
      } else {
        $$renderer2.push("<!--[!-->");
        $$renderer2.push(`<div class="timestep-list svelte-4p4iat"><!--[-->`);
        const each_array = ensure_array_like(timesteps);
        for (let $$index_1 = 0, $$length = each_array.length; $$index_1 < $$length; $$index_1++) {
          let ts = each_array[$$index_1];
          $$renderer2.push(`<div class="timestep-item svelte-4p4iat"><div class="ts-header svelte-4p4iat"><span class="ts-tick svelte-4p4iat">#${escape_html(ts.tick)}</span> <span class="ts-type svelte-4p4iat">${escape_html(ts.event_type)}</span> <span class="ts-time svelte-4p4iat">${escape_html(formatTime(ts.timestamp))}</span> <span class="ts-fidelity svelte-4p4iat">${escape_html(ts.fidelity)}</span></div> <div class="ts-content svelte-4p4iat">${escape_html(ts.content || "(empty)")}</div> `);
          if (Object.keys(ts.concept_activations || {}).length > 0) {
            $$renderer2.push("<!--[-->");
            $$renderer2.push(`<div class="ts-concepts svelte-4p4iat"><!--[-->`);
            const each_array_1 = ensure_array_like(Object.entries(ts.concept_activations).slice(0, 5));
            for (let $$index = 0, $$length2 = each_array_1.length; $$index < $$length2; $$index++) {
              let [concept, score] = each_array_1[$$index];
              $$renderer2.push(`<span class="concept-tag svelte-4p4iat">${escape_html(concept)}: ${escape_html(score.toFixed(2))}</span>`);
            }
            $$renderer2.push(`<!--]--></div>`);
          } else {
            $$renderer2.push("<!--[!-->");
          }
          $$renderer2.push(`<!--]--></div>`);
        }
        $$renderer2.push(`<!--]--></div>`);
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!--]--></div> <div class="tags-panel svelte-4p4iat"><h3 class="svelte-4p4iat">Tags (${escape_html(tags.length)})</h3> <div class="tag-filters svelte-4p4iat">`);
    $$renderer2.select(
      { value: selectedTagType, onchange: loadTags, class: "" },
      ($$renderer3) => {
        $$renderer3.option({ value: "" }, ($$renderer4) => {
          $$renderer4.push(`All types`);
        });
        $$renderer3.option({ value: "concept" }, ($$renderer4) => {
          $$renderer4.push(`Concept`);
        });
        $$renderer3.option({ value: "entity" }, ($$renderer4) => {
          $$renderer4.push(`Entity`);
        });
        $$renderer3.option({ value: "bud" }, ($$renderer4) => {
          $$renderer4.push(`Bud`);
        });
        $$renderer3.option({ value: "custom" }, ($$renderer4) => {
          $$renderer4.push(`Custom`);
        });
      },
      "svelte-4p4iat"
    );
    $$renderer2.push(`</div> `);
    if (tags.length === 0) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<p class="empty svelte-4p4iat">No tags found</p>`);
    } else {
      $$renderer2.push("<!--[!-->");
      $$renderer2.push(`<div class="tag-list svelte-4p4iat"><!--[-->`);
      const each_array_2 = ensure_array_like(tags);
      for (let $$index_2 = 0, $$length = each_array_2.length; $$index_2 < $$length; $$index_2++) {
        let tag = each_array_2[$$index_2];
        $$renderer2.push(`<div${attr_class("tag-item svelte-4p4iat", void 0, { "bud": tag.tag_type === "bud" })}><span class="tag-name svelte-4p4iat">${escape_html(tag.name)}</span> <span class="tag-type svelte-4p4iat">${escape_html(tag.tag_type)}</span> <span class="tag-count svelte-4p4iat">${escape_html(tag.use_count)} uses</span> `);
        if (tag.bud_status) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<span class="bud-status svelte-4p4iat">${escape_html(tag.bud_status)}</span>`);
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]--></div>`);
      }
      $$renderer2.push(`<!--]--></div>`);
    }
    $$renderer2.push(`<!--]--></div></div></div>`);
    if ($$store_subs) unsubscribe_stores($$store_subs);
  });
}
export {
  _page as default
};
