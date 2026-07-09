// Universal search store for HatStand admin search.
// Backed by MiniSearch (BM25, prefix matching, fuzzy) for full-text scoring,
// with type/facet filtering and facet aggregation done over the candidate
// set. The class is the abstraction — swap the implementation later (SQLite
// FTS5, Tantivy) without touching consumers.

import MiniSearch, { type SearchResult } from "minisearch";
import type {
  FacetCounts,
  FacetKey,
  FacetValueCount,
  ResourceType,
  SearchDocument,
  SearchResponse,
} from "@/types";

export type SearchInternalQuery = {
  q?: string;
  type?: ResourceType;
  filter?: Partial<Record<FacetKey, string | string[]>>;
  limit?: number;
  offset?: number;
};

const STORE_FIELDS = [
  "id",
  "resource_type",
  "title",
  "body_excerpt",
  "url",
  "facets",
  "parent_ids",
  "updated_at",
] as const;

const SEARCH_FIELDS = ["title", "body_excerpt", "id"] as const;

export class IndexStore {
  private docs = new Map<string, SearchDocument>();
  private ms = new MiniSearch<SearchDocument>({
    idField: "id",
    fields: [...SEARCH_FIELDS],
    storeFields: [...STORE_FIELDS],
    searchOptions: {
      boost: { title: 3, id: 5, body_excerpt: 1 },
      prefix: true,
      fuzzy: 0.15,
      combineWith: "AND",
    },
  });

  add(documents: SearchDocument[]): void {
    // Last-write-wins semantics. Duplicates across or within a batch are
    // tolerated — there are real cases (e.g. a meld with the same
    // meld_request_id appearing in both pending/ and applied/) we don't want
    // to crash on. MiniSearch's add throws on duplicates, so we must use
    // `replace` for already-present ids.
    for (const doc of documents) {
      if (this.ms.has(doc.id)) {
        this.ms.replace(doc);
      } else {
        this.ms.add(doc);
      }
      this.docs.set(doc.id, doc);
    }
  }

  size(): number {
    return this.docs.size;
  }

  search(query: SearchInternalQuery): SearchResponse {
    const limit = Math.min(Math.max(query.limit ?? 25, 1), 200);
    const offset = Math.max(query.offset ?? 0, 0);

    const passesFilters = (doc: SearchDocument): boolean => {
      if (query.type && doc.resource_type !== query.type) return false;
      if (query.filter) {
        for (const [key, raw] of Object.entries(query.filter) as Array<
          [FacetKey, string | string[] | undefined]
        >) {
          if (raw == null) continue;
          const wanted = Array.isArray(raw) ? raw : [raw];
          if (wanted.length === 0) continue;
          const docFacet = doc.facets[key];
          if (docFacet == null) return false;
          const docValues = Array.isArray(docFacet) ? docFacet : [docFacet];
          if (!docValues.some((v) => wanted.includes(v))) return false;
        }
      }
      return true;
    };

    const candidates: SearchDocument[] = [];
    for (const doc of this.docs.values()) {
      if (passesFilters(doc)) candidates.push(doc);
    }

    let ordered: SearchDocument[];
    if (query.q && query.q.trim().length > 0) {
      // MiniSearch handles BM25, prefix, and fuzzy. We pass a filter callback
      // so only documents that already pass type+facet filtering can match.
      const candidateIds = new Set(candidates.map((c) => c.id));
      const results: SearchResult[] = this.ms.search(query.q, {
        filter: (r) => candidateIds.has(r.id as string),
      });
      ordered = results.map((r) => ({
        ...(r as unknown as SearchDocument),
        score: r.score,
      }));
    } else {
      // No query: return everything in the candidate set, newest first.
      ordered = [...candidates].sort((a, b) =>
        b.updated_at.localeCompare(a.updated_at),
      );
    }

    const facets = aggregateFacets(candidates);
    const total = ordered.length;
    const items = ordered.slice(offset, offset + limit);
    const next = offset + items.length;

    return {
      items,
      facets,
      total,
      next_cursor: next < total ? String(next) : undefined,
    };
  }
}

function aggregateFacets(docs: SearchDocument[]): FacetCounts {
  const counts = new Map<FacetKey, Map<string, number>>();
  for (const doc of docs) {
    for (const [key, raw] of Object.entries(doc.facets) as Array<
      [FacetKey, string | string[] | undefined]
    >) {
      if (raw == null) continue;
      const values = Array.isArray(raw) ? raw : [raw];
      const map = counts.get(key) ?? new Map<string, number>();
      for (const v of values) map.set(v, (map.get(v) ?? 0) + 1);
      counts.set(key, map);
    }
  }
  const out: FacetCounts = {};
  for (const [key, map] of counts) {
    const list: FacetValueCount[] = [];
    for (const [value, count] of map) list.push({ value, count });
    list.sort((a, b) => b.count - a.count || a.value.localeCompare(b.value));
    out[key] = list;
  }
  return out;
}
