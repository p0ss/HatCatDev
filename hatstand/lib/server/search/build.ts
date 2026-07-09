// Build the universal search index by calling each slice's indexer.
// The index is cached in module scope and rebuilt lazily after a TTL.

import { IndexStore } from "@/lib/server/search/index-store";
import { indexModels } from "@/lib/server/hf-cache";
import { indexLensPacks } from "@/lib/server/lens-packs";
import {
  indexConceptPacks,
  indexConcepts,
} from "@/lib/server/concept-packs";
import { indexMelds } from "@/lib/server/melds";
import { indexDocs } from "@/lib/server/docs-index";

const REBUILD_TTL_MS = 30_000; // rebuild at most every 30s under load

let cached: { store: IndexStore; builtAt: number; building?: Promise<IndexStore> } | null = null;

async function buildIndex(): Promise<IndexStore> {
  const store = new IndexStore();
  // Build concurrently — each slice indexer is independent.
  const [models, lensPacks, conceptPacks, melds, docs, concepts] = await Promise.all([
    indexModels(),
    indexLensPacks(),
    indexConceptPacks(),
    indexMelds(),
    indexDocs(),
    indexConcepts(),
  ]);
  store.add(models);
  store.add(lensPacks);
  store.add(conceptPacks);
  store.add(melds);
  store.add(docs);
  store.add(concepts);
  return store;
}

export async function getIndex(): Promise<IndexStore> {
  const now = Date.now();
  if (cached && now - cached.builtAt < REBUILD_TTL_MS) return cached.store;
  if (cached?.building) return cached.building;

  const building = buildIndex();
  cached = {
    store: cached?.store ?? new IndexStore(),
    builtAt: now,
    building,
  };
  const store = await building;
  cached = { store, builtAt: now };
  return store;
}

// For an explicit /reindex endpoint or test usage.
export function invalidateIndex(): void {
  cached = null;
}
