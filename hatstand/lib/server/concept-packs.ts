// Shared parsing for HatCatDev/concept_packs/. Used by:
//   - app/api/admin/concept-packs/route.ts (list)
//   - app/api/admin/concept-packs/[name]/route.ts (pack detail)
//   - app/api/admin/concept-packs/[name]/concepts/route.ts (concept list)
//   - app/api/admin/concept-packs/[name]/concepts/[term]/route.ts (concept detail)
//   - app/api/admin/registry/route.ts (composite registry)

import path from "node:path";
import fs from "node:fs/promises";
import {
  hatcatdevPath,
  pathExists,
  statMtimeIso,
  tryReadJson,
} from "@/lib/server/hatcatdev";
import {
  findLensPacksForConcept,
  loadLensPacksByConcept,
} from "@/lib/server/lens-packs";
import type {
  Concept,
  ConceptPackSummary,
  SearchDocument,
  SimplexBinding,
} from "@/types";

export const CONCEPT_PACKS_DIR = "concept_packs";

// ---------- On-disk shapes ----------

export type PackJson = {
  pack_id?: string;
  version?: string;
  created?: string;
  forked_from?: { pack_id?: string; version?: string };
  description?: string;
  concept_metadata?: {
    total_concepts?: number;
    layers?: number[];
    layer_distribution?: Record<string, number>;
  };
};

export type RawConcept = {
  // Different packs use slightly different keys for the term.
  sumo_term?: string;
  term?: string;
  original_term?: string;
  id?: string;
  label?: string;
  layer?: number;
  domain?: string;
  definition?: string;
  parent_concepts?: string[];
  child_concepts?: string[];
  category_children?: string[];
  children?: string[];
  simplex_dimension?: string;
  pole?: string;
  wordnet?: { synsets?: string[]; lemmas?: string[] };
  synsets?: string[];
  lemmas?: string[];
  safety_tags?:
    | { risk_level?: string; treaty_relevant?: boolean; harness_relevant?: boolean }
    | string[];
};

export type LayerFile = {
  layer?: number;
  concepts?: RawConcept[];
};

// ---------- Pack summary ----------

export function packDir(name: string): string {
  return hatcatdevPath(CONCEPT_PACKS_DIR, name);
}

function sourcePackOf(pack: PackJson): string | undefined {
  if (!pack.forked_from?.pack_id) return undefined;
  return pack.forked_from.version
    ? `${pack.forked_from.pack_id}@${pack.forked_from.version}`
    : pack.forked_from.pack_id;
}

export async function readPackSummary(
  name: string,
): Promise<ConceptPackSummary | null> {
  const dir = packDir(name);
  const pack = await tryReadJson<PackJson>(path.join(dir, "pack.json"));
  if (!pack) return null;

  const layers = pack.concept_metadata?.layers ?? [];
  const created = pack.created ?? "";
  const updated =
    (await statMtimeIso(path.join(dir, "pack.json"))) ??
    (await statMtimeIso(dir)) ??
    created;

  let simplexCount = 0;
  const simplexes = await tryReadJson<{ simplexes?: unknown[] }>(
    path.join(dir, "simplexes.json"),
  );
  if (Array.isArray(simplexes?.simplexes)) {
    simplexCount = simplexes.simplexes.length;
  }

  return {
    name: pack.pack_id ?? name,
    version: pack.version ?? "0.0.0",
    source_pack: sourcePackOf(pack),
    concept_count: pack.concept_metadata?.total_concepts ?? 0,
    simplex_count: simplexCount,
    layer_count: layers.length,
    created_at: created,
    updated_at: updated ?? "",
  };
}

// ---------- Concept projection ----------

export function termOf(raw: RawConcept): string | null {
  return (
    raw.sumo_term ??
    raw.term ??
    raw.original_term ??
    raw.id ??
    raw.label ??
    null
  );
}

export function safetyTagsOf(raw: RawConcept): string[] {
  if (!raw.safety_tags) return [];
  if (Array.isArray(raw.safety_tags)) return raw.safety_tags;
  const tags: string[] = [];
  const t = raw.safety_tags;
  if (t.risk_level && t.risk_level.toLowerCase() !== "low") {
    tags.push(`risk:${t.risk_level}`);
  }
  if (t.treaty_relevant) tags.push("treaty");
  if (t.harness_relevant) tags.push("harness");
  return tags;
}

export function projectConcept(
  raw: RawConcept,
  fallbackLayer: number,
  lensPackIds: string[] = [],
): Concept | null {
  const term = termOf(raw);
  if (!term) return null;
  const synsets = raw.wordnet?.synsets ?? raw.synsets ?? [];
  const lemmas = raw.wordnet?.lemmas ?? raw.lemmas ?? [];
  const childIds = raw.child_concepts ?? raw.children ?? raw.category_children ?? [];
  const parentIds = raw.parent_concepts ?? [];
  const simplex_bindings: SimplexBinding[] = [];
  if (
    raw.simplex_dimension &&
    (raw.pole === "positive" || raw.pole === "negative" || raw.pole === "neutral")
  ) {
    simplex_bindings.push({ simplex_id: raw.simplex_dimension, pole: raw.pole });
  }
  return {
    term,
    sumo_term: raw.sumo_term ?? raw.term,
    definition: raw.definition,
    layer: typeof raw.layer === "number" ? raw.layer : fallbackLayer,
    synsets,
    lemmas,
    parent_ids: parentIds,
    sibling_ids: [],
    children_ids: childIds,
    simplex_bindings,
    safety_tags: safetyTagsOf(raw),
    domain: raw.domain,
    lens_pack_ids: lensPackIds,
  };
}

// ---------- Hierarchy file walking ----------

export async function listLayerFiles(packDir: string): Promise<string[]> {
  const hierDir = path.join(packDir, "hierarchy");
  if (!(await pathExists(hierDir))) return [];
  try {
    const entries = await fs.readdir(hierDir, { withFileTypes: true });
    return entries
      .filter((e) => e.isFile() && /^layer\d+\.json$/.test(e.name))
      .map((e) => path.join(hierDir, e.name))
      .sort();
  } catch {
    return [];
  }
}

function layerFromFilename(file: string): number {
  const m = file.match(/layer(\d+)\.json$/);
  return m ? Number(m[1]) : 0;
}

// Load every concept across hierarchy/layer{N}.json files in a pack.
// Cross-references with lens_pack_ids by term.
export async function loadAllConcepts(
  packDir: string,
  conceptPackId: string,
): Promise<Concept[]> {
  const layerFiles = await listLayerFiles(packDir);
  if (layerFiles.length === 0) return [];

  const lensPacksByConcept = await loadLensPacksByConcept(conceptPackId);

  const all: Concept[] = [];
  for (const file of layerFiles) {
    const fallbackLayer = layerFromFilename(file);
    const data = await tryReadJson<LayerFile>(file);
    if (!data?.concepts) continue;
    for (const raw of data.concepts) {
      const term = termOf(raw);
      const concept = projectConcept(
        raw,
        fallbackLayer,
        term ? lensPacksByConcept.get(term) ?? [] : [],
      );
      if (concept) all.push(concept);
    }
  }
  return all;
}

// Find a single concept by term in the hierarchy files.
export async function findConceptInHierarchy(
  packDir: string,
  term: string,
): Promise<Concept | null> {
  const layerFiles = await listLayerFiles(packDir);
  for (const file of layerFiles) {
    const fallbackLayer = layerFromFilename(file);
    const data = await tryReadJson<LayerFile>(file);
    if (!data?.concepts) continue;
    for (const raw of data.concepts) {
      if (termOf(raw) === term) {
        return projectConcept(raw, fallbackLayer);
      }
    }
  }
  return null;
}

// Search indexer — list every concept pack as a SearchDocument.
export async function indexConceptPacks(): Promise<SearchDocument[]> {
  const root = hatcatdevPath(CONCEPT_PACKS_DIR);
  const dirNames: string[] = await (async () => {
    try {
      const entries = await fs.readdir(root, { withFileTypes: true });
      return entries
        .filter((e) => e.isDirectory() && !e.name.startsWith("."))
        .map((e) => e.name);
    } catch {
      return [];
    }
  })();
  const summaries = (
    await Promise.all(dirNames.map((n) => readPackSummary(n)))
  ).filter((s): s is ConceptPackSummary => s !== null);

  return summaries.map((p) => ({
    id: `concept_pack:${p.name}`,
    resource_type: "concept_pack" as const,
    title: `${p.name}@${p.version}`,
    body_excerpt: [
      p.source_pack ? `forked from ${p.source_pack}` : "",
      `${p.concept_count} concepts`,
      `${p.layer_count} layers`,
      p.simplex_count > 0 ? `${p.simplex_count} simplexes` : "",
    ]
      .filter(Boolean)
      .join(" · "),
    url: `/concept-packs/${encodeURIComponent(p.name)}`,
    facets: {},
    parent_ids: [],
    updated_at: p.updated_at,
  }));
}

// Search indexer — every concept across every populated pack.
// This is the heaviest indexer (~9k docs for first-light alone). Called once
// at index-build time; subsequent searches are in-memory.
export async function indexConcepts(): Promise<SearchDocument[]> {
  const root = hatcatdevPath(CONCEPT_PACKS_DIR);
  const dirNames: string[] = await (async () => {
    try {
      const entries = await fs.readdir(root, { withFileTypes: true });
      return entries
        .filter((e) => e.isDirectory() && !e.name.startsWith("."))
        .map((e) => e.name);
    } catch {
      return [];
    }
  })();

  const all: SearchDocument[] = [];
  for (const dirName of dirNames) {
    const summary = await readPackSummary(dirName);
    if (!summary) continue;
    const concepts = await loadAllConcepts(packDir(dirName), summary.name);
    const updatedAt = summary.updated_at;
    for (const c of concepts) {
      const facets: Record<string, string | string[]> = {
        layer: String(c.layer),
      };
      if (c.domain) facets.domain = c.domain;
      if (c.safety_tags.length > 0) facets.safety_tag = c.safety_tags;
      all.push({
        id: `concept:${summary.name}:${c.term}`,
        resource_type: "concept" as const,
        title: c.term,
        body_excerpt: [
          c.definition ?? "",
          c.lemmas.length > 0 ? `lemmas: ${c.lemmas.join(", ")}` : "",
        ]
          .filter(Boolean)
          .join(" · "),
        url: `/concept-packs/${encodeURIComponent(summary.name)}/concepts/${encodeURIComponent(c.term)}`,
        facets,
        parent_ids: [`concept_pack:${summary.name}`],
        updated_at: updatedAt,
      });
    }
  }
  return all;
}

// Best-effort: read concepts/layer{N}/<term-lower>.json if present (first-light
// has these with richer fields than the hierarchy).
export async function tryReadPerConceptFile(
  packDir: string,
  term: string,
  layer: number,
): Promise<RawConcept | null> {
  const lower = term.toLowerCase();
  const candidate = path.join(
    packDir,
    "concepts",
    `layer${layer}`,
    `${lower}.json`,
  );
  return tryReadJson<RawConcept>(candidate);
}

// Combined detail loader: hierarchy projection enriched with per-concept file
// when available, with lens_pack_ids cross-link.
export async function loadConceptDetail(
  packDir: string,
  conceptPackId: string,
  term: string,
): Promise<Concept | null> {
  const fromHierarchy = await findConceptInHierarchy(packDir, term);
  if (!fromHierarchy) return null;

  const richer = await tryReadPerConceptFile(packDir, term, fromHierarchy.layer);
  let concept: Concept = fromHierarchy;
  if (richer) {
    const projected = projectConcept(richer, fromHierarchy.layer);
    if (projected) {
      concept = {
        ...fromHierarchy,
        ...projected,
        synsets:
          projected.synsets.length > 0 ? projected.synsets : fromHierarchy.synsets,
        lemmas:
          projected.lemmas.length > 0 ? projected.lemmas : fromHierarchy.lemmas,
        children_ids:
          projected.children_ids.length > 0
            ? projected.children_ids
            : fromHierarchy.children_ids,
        parent_ids:
          projected.parent_ids.length > 0
            ? projected.parent_ids
            : fromHierarchy.parent_ids,
        definition: projected.definition ?? fromHierarchy.definition,
      };
    }
  }

  concept.lens_pack_ids = await findLensPacksForConcept(conceptPackId, term);
  return concept;
}
