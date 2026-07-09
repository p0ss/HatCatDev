// Shared parsing for HatCatDev/src/lens_packs/. Used by:
//   - app/api/admin/lens-packs/route.ts (list)
//   - app/api/admin/lens-packs/[id]/route.ts (detail)
//   - app/api/admin/registry/route.ts (composite registry)
//   - app/api/admin/models/* (lens_packs targeting a substrate)
//   - app/api/admin/concept-packs/* (lens packs containing a concept)

import path from "node:path";
import {
  hatcatdevPath,
  pathExists,
  statMtimeIso,
  tryReadJson,
} from "@/lib/server/hatcatdev";
import type {
  CalibrationCycle,
  CalibrationOverFirerSummary,
  CalibrationStatus,
  ConceptCalibration,
  Lens,
  LensPackStatus,
  SearchDocument,
  Simplex,
  SimplexPole,
} from "@/types";

export const LENS_PACKS_DIR = "src/lens_packs";

// ---------- On-disk shapes ----------

export type LensRegistryEntry = {
  source?: string;
  version?: string;
  revision?: string | null;
  synced_at?: string | null;
  created_at?: string;
  modified?: boolean;
  based_on?: string | null;
  size_bytes?: number;
};

export type LensRegistryFile = {
  schema_version?: string;
  packs?: Record<string, LensRegistryEntry>;
};

export type LensPackInfo = {
  source_pack?: string;
  pack_version?: string;
  based_on?: string;
  model?: string;
  trained_at?: string;
  calibration_status?: string;
  simplexes?: PackInfoSimplexes;
};

export type RawManifestLens = {
  concept?: string;
  trained_at_version?: string;
  trained_timestamp?: string;
  default_layer?: number;
  layer?: number;
  lens_file?: string;
  training_samples?: number;
  metrics?: { f1?: number; precision?: number; recall?: number };
  classifiers?: Record<
    string,
    {
      layer?: number;
      category?: string;
      technique?: string;
      file?: string;
      trained_at?: string;
      metrics?: { f1?: number; precision?: number; recall?: number };
    }
  >;
};

export type LensVersionManifest = {
  current_version?: string;
  created?: string;
  updated?: string;
  lenses?: Record<string, RawManifestLens>;
};

// Pole entry shape varies between calibration regimes:
//   gemma-3 v2 dict form: { label, definition }
//   gemma-4 v1 list form: { definition, lemmas, synset }
// We accept both and project to a unified shape downstream.
type RawSimplexPole = {
  label?: string;
  definition?: string;
  lemmas?: string[];
  synset?: string;
};

// Two on-disk shapes for _definitions.json — keep both representable.
type DefsDictEntry = {
  dimension?: string;
  description?: string;
  positive_pole?: RawSimplexPole;
  neutral_homeostasis?: RawSimplexPole;
  negative_pole?: RawSimplexPole;
};

type DefsListEntry = {
  simplex_dimension?: string;
  description?: string;
  definition?: string;
  three_pole_simplex?: {
    positive_pole?: RawSimplexPole;
    neutral_homeostasis?: RawSimplexPole;
    negative_pole?: RawSimplexPole;
  };
};

export type RawSimplexDefinitions = {
  metadata?: {
    version?: string;
    created_date?: string;
    description?: string;
    note?: string;
  };
  simplexes?: Record<string, DefsDictEntry> | DefsListEntry[];
};

// Per-pole training results from simplex/results.json (produced by the
// canonical train_s_tier_simplexes pipeline). Older packs don't have this.
export type RawSimplexResults = {
  timestamp?: string;
  total_simplexes?: number;
  completed?: number;
  failed_lenses?: unknown;
  simplexes?: Array<{
    dimension?: string;
    poles?: Record<
      string,
      {
        success?: boolean;
        test_f1?: number;
        iterations?: number;
        samples_used?: number;
      }
    >;
  }>;
};

export type PackInfoSimplexes = {
  count?: number;
  format?: string;
  names?: string[];
  source?: string;
  definitions_file?: string;
  trained_with?: string;
};

// Raw calibration entry shape on disk. Field set differs between modes;
// gen_* and noise_* are mutually-exclusive in practice.
export type RawCalibrationEntry = {
  concept?: string;
  layer?: number;
  self_mean?: number;
  self_std?: number;
  cross_mean?: number;
  cross_std?: number;
  cross_fire_count?: number;
  cross_fire_rate?: number;
  times_loaded?: number;
  n_self_samples?: number;
  n_cross_samples?: number;
  gen_mean?: number;
  gen_fire_count?: number;
  gen_fire_rate?: number;
  noise_mean?: number;
  noise_std?: number;
  noise_max?: number;
  noise_fire_count?: number;
  noise_fire_rate?: number;
};

export type LensCalibrationFile = {
  timestamp?: string;
  lens_pack?: string;
  mode?: string;
  cross_calibration_source?: string;
  gen_calibration_source?: string;
  total_concepts_calibrated?: number;
  noise_calibration_samples?: number;
  noise_calibration_timestamp?: string;
  firing_threshold?: number;
  // <concept>_L<layer> -> entry
  calibration?: Record<string, RawCalibrationEntry>;
};

// Cycle file shapes (see calibration_analysis_cycle{N}.json /
// calibration_finetune_cycle{N}.json).
export type RawCycleAnalysis = {
  timestamp?: string;
  total_concepts?: number;
  top_k?: number;
  avg_in_top_k_rate?: number;
  under_firing?: string[];
  over_firing?: string[];
  well_calibrated?: string[];
};

export type RawCycleFinetune = {
  finetune_timestamp?: string;
  total_lenses_processed?: number;
  lenses_boosted?: number;
  lenses_suppressed?: number;
  avg_improvement?: number;
};

// ---------- Loaders ----------

export async function loadRegistry(): Promise<LensRegistryFile> {
  return (
    (await tryReadJson<LensRegistryFile>(
      hatcatdevPath(LENS_PACKS_DIR, ".registry.json"),
    )) ?? { packs: {} }
  );
}

export function packDir(packId: string): string {
  return hatcatdevPath(LENS_PACKS_DIR, packId);
}

export function loadPackInfo(packId: string): Promise<LensPackInfo | null> {
  return tryReadJson<LensPackInfo>(path.join(packDir(packId), "pack_info.json"));
}

export function loadVersionManifest(
  packId: string,
): Promise<LensVersionManifest | null> {
  return tryReadJson<LensVersionManifest>(
    path.join(packDir(packId), "version_manifest.json"),
  );
}

export function loadCalibration(
  packId: string,
): Promise<LensCalibrationFile | null> {
  return tryReadJson<LensCalibrationFile>(
    path.join(packDir(packId), "calibration.json"),
  );
}

// ---------- Calibration analysis ----------

// Threshold for "over-firer" classification by cross_fire_rate.
// Pragmatic: a lens that fires on ≥30% of out-of-domain samples is leaking
// signal. Same threshold for noise_fire_rate.
export const OVER_FIRER_THRESHOLD = 0.3;

export function projectCalibrationEntry(
  raw: RawCalibrationEntry,
): ConceptCalibration | null {
  if (!raw.concept || typeof raw.layer !== "number") return null;
  return {
    concept: raw.concept,
    layer: raw.layer,
    self_mean: raw.self_mean ?? 0,
    self_std: raw.self_std ?? 0,
    cross_mean: raw.cross_mean ?? 0,
    cross_std: raw.cross_std ?? 0,
    cross_fire_count: raw.cross_fire_count ?? 0,
    cross_fire_rate: raw.cross_fire_rate ?? 0,
    times_loaded: raw.times_loaded ?? 0,
    n_self_samples: raw.n_self_samples ?? 0,
    n_cross_samples: raw.n_cross_samples ?? 0,
    gen_mean: raw.gen_mean,
    gen_fire_count: raw.gen_fire_count,
    gen_fire_rate: raw.gen_fire_rate,
    noise_mean: raw.noise_mean,
    noise_std: raw.noise_std,
    noise_max: raw.noise_max,
    noise_fire_count: raw.noise_fire_count,
    noise_fire_rate: raw.noise_fire_rate,
  };
}

export function listCalibrationEntries(
  cal: LensCalibrationFile | null,
): ConceptCalibration[] {
  if (!cal?.calibration) return [];
  const out: ConceptCalibration[] = [];
  for (const raw of Object.values(cal.calibration)) {
    const projected = projectCalibrationEntry(raw);
    if (projected) out.push(projected);
  }
  return out;
}

export function summariseOverFirers(
  entries: ConceptCalibration[],
  threshold: number = OVER_FIRER_THRESHOLD,
): CalibrationOverFirerSummary {
  const perLayer = new Map<
    number,
    { total: number; over: number; noiseOver: number }
  >();
  let totalOver = 0;
  let totalNoiseOver = 0;
  let hasNoise = false;
  for (const e of entries) {
    const slot = perLayer.get(e.layer) ?? { total: 0, over: 0, noiseOver: 0 };
    slot.total += 1;
    if (e.cross_fire_rate > threshold) {
      slot.over += 1;
      totalOver += 1;
    }
    if (typeof e.noise_fire_rate === "number") {
      hasNoise = true;
      if (e.noise_fire_rate > threshold) {
        slot.noiseOver += 1;
        totalNoiseOver += 1;
      }
    }
    perLayer.set(e.layer, slot);
  }
  const per_layer = Array.from(perLayer.entries())
    .map(([layer, s]) => ({
      layer,
      total: s.total,
      over_firers: s.over,
      ...(hasNoise ? { noise_over_firers: s.noiseOver } : {}),
    }))
    .sort((a, b) => a.layer - b.layer);
  return {
    threshold,
    total_entries: entries.length,
    over_firers: totalOver,
    ...(hasNoise ? { noise_over_firers: totalNoiseOver } : {}),
    per_layer,
  };
}

// Load all calibration_{analysis,finetune}_cycleN.json files for a pack and
// stitch them into an ordered cycle progression.
export async function loadCalibrationCycles(
  packId: string,
): Promise<CalibrationCycle[]> {
  const out: CalibrationCycle[] = [];
  for (let cycle = 1; cycle <= 20; cycle++) {
    const analysis = await tryReadJson<RawCycleAnalysis>(
      path.join(packDir(packId), `calibration_analysis_cycle${cycle}.json`),
    );
    if (!analysis) break;
    const finetune = await tryReadJson<RawCycleFinetune>(
      path.join(packDir(packId), `calibration_finetune_cycle${cycle}.json`),
    );
    out.push({
      cycle,
      analysis_timestamp: analysis.timestamp,
      finetune_timestamp: finetune?.finetune_timestamp,
      total_concepts: analysis.total_concepts ?? 0,
      top_k: analysis.top_k,
      avg_in_top_k_rate: analysis.avg_in_top_k_rate,
      well_calibrated: Array.isArray(analysis.well_calibrated)
        ? analysis.well_calibrated.length
        : 0,
      under_firing: Array.isArray(analysis.under_firing)
        ? analysis.under_firing.length
        : 0,
      over_firing: Array.isArray(analysis.over_firing)
        ? analysis.over_firing.length
        : 0,
      finetune: finetune
        ? {
            total_lenses_processed: finetune.total_lenses_processed ?? 0,
            lenses_boosted: finetune.lenses_boosted ?? 0,
            lenses_suppressed: finetune.lenses_suppressed ?? 0,
            avg_improvement: finetune.avg_improvement ?? 0,
          }
        : undefined,
    });
  }
  return out;
}

// Distinct concept count from the calibration entries (since
// `total_concepts_calibrated` actually counts concept × layer combinations).
export function distinctConceptCount(entries: ConceptCalibration[]): number {
  const seen = new Set<string>();
  for (const e of entries) seen.add(e.concept);
  return seen.size;
}

// ---------- Lens inventory ----------

// Project a manifest lens entry into the public `Lens` type. The manifest only
// gives us the model layer (default_layer); for "ontological layer" we'd have
// to cross-reference the source concept pack's hierarchy. Until that's wired,
// we set both `layer` and `selected_layer` to the manifest's default_layer.
export function projectLens(
  packId: string,
  raw: RawManifestLens,
): Lens | null {
  if (!raw.concept) return null;
  const selectedLayer =
    raw.default_layer ??
    raw.layer ??
    (raw.classifiers ? Number(Object.keys(raw.classifiers)[0] ?? 0) : 0);
  return {
    pack_id: packId,
    term: raw.concept,
    layer: selectedLayer,
    selected_layer: selectedLayer,
    file_path: raw.lens_file ?? "",
    training_metrics: {
      test_f1: raw.metrics?.f1 ?? 0,
      test_precision: raw.metrics?.precision,
      test_recall: raw.metrics?.recall,
      selected_layer: selectedLayer,
      trained_at: raw.trained_timestamp ?? "",
    },
  };
}

// Extra projection that surfaces category + training_samples — fields that
// aren't part of the canonical Lens type but are useful in the inventory UI.
// Keep this in lib/server so the route can opt into the richer shape.
export type LensWithMeta = Lens & {
  category?: string;
  training_samples?: number;
};

export function projectLensWithMeta(
  packId: string,
  raw: RawManifestLens,
): LensWithMeta | null {
  const lens = projectLens(packId, raw);
  if (!lens) return null;
  const defaultLayerKey = String(lens.selected_layer);
  const classifier = raw.classifiers?.[defaultLayerKey];
  return {
    ...lens,
    category: classifier?.category,
    training_samples: raw.training_samples,
  };
}

export async function listManifestLenses(
  packId: string,
): Promise<LensWithMeta[]> {
  const manifest = await loadVersionManifest(packId);
  if (!manifest?.lenses) return [];
  const out: LensWithMeta[] = [];
  for (const raw of Object.values(manifest.lenses)) {
    const lens = projectLensWithMeta(packId, raw);
    if (lens) out.push(lens);
  }
  return out;
}

// ---------- Simplex inventory ----------

export async function loadSimplexDefinitions(
  packId: string,
): Promise<RawSimplexDefinitions | null> {
  return tryReadJson<RawSimplexDefinitions>(
    path.join(packDir(packId), "simplex", "_definitions.json"),
  );
}

export async function loadSimplexResults(
  packId: string,
): Promise<RawSimplexResults | null> {
  return tryReadJson<RawSimplexResults>(
    path.join(packDir(packId), "simplex", "results.json"),
  );
}

function poleFromRaw(
  name: "positive" | "neutral" | "negative",
  raw: RawSimplexPole | undefined,
  testF1: number | undefined,
): SimplexPole {
  return {
    name,
    label: raw?.label ?? raw?.lemmas?.[0],
    definition: raw?.definition,
    synsets: raw?.synset ? [raw.synset] : [],
    test_f1: testF1,
  };
}

// Build a dim → DefsDictEntry map regardless of whether the on-disk shape is
// dict-keyed or list-with-dimension-field.
function indexDefinitions(
  defs: RawSimplexDefinitions | null,
): Map<string, DefsDictEntry> {
  const out = new Map<string, DefsDictEntry>();
  const sx = defs?.simplexes;
  if (!sx) return out;
  if (Array.isArray(sx)) {
    for (const entry of sx) {
      const dim = entry.simplex_dimension;
      if (!dim) continue;
      const triple = entry.three_pole_simplex;
      out.set(dim, {
        dimension: dim,
        description: entry.description ?? entry.definition,
        positive_pole: triple?.positive_pole,
        neutral_homeostasis: triple?.neutral_homeostasis,
        negative_pole: triple?.negative_pole,
      });
    }
  } else {
    for (const [dim, entry] of Object.entries(sx)) {
      out.set(dim, entry);
    }
  }
  return out;
}

type PoleResults = {
  positive?: number;
  neutral?: number;
  negative?: number;
};

function indexResults(
  results: RawSimplexResults | null,
): Map<string, PoleResults> {
  const out = new Map<string, PoleResults>();
  if (!results?.simplexes) return out;
  for (const entry of results.simplexes) {
    if (!entry.dimension) continue;
    const poles = entry.poles ?? {};
    out.set(entry.dimension, {
      positive: poles.positive?.test_f1,
      neutral: poles.neutral?.test_f1,
      negative: poles.negative?.test_f1,
    });
  }
  return out;
}

export async function listPackSimplexes(packId: string): Promise<Simplex[]> {
  const [packInfo, defs, results] = await Promise.all([
    loadPackInfo(packId),
    loadSimplexDefinitions(packId),
    loadSimplexResults(packId),
  ]);
  // pack_info.simplexes.names is the source of truth for what's IN this pack.
  // simplex/_definitions.json may be a catalogue with unrelated entries.
  const indexedDefs = indexDefinitions(defs);
  const indexedResults = indexResults(results);
  const allNames =
    packInfo?.simplexes?.names && packInfo.simplexes.names.length > 0
      ? [...packInfo.simplexes.names].sort()
      : Array.from(indexedDefs.keys()).sort();

  const trainedAt = packInfo?.trained_at;
  return allNames.map((name): Simplex => {
    const def = indexedDefs.get(name);
    const res = indexedResults.get(name);
    return {
      id: `${packId}:${name}`,
      pack_id: packId,
      dimension: def?.dimension ?? name,
      poles: [
        poleFromRaw("positive", def?.positive_pole, res?.positive),
        poleFromRaw("neutral", def?.neutral_homeostasis, res?.neutral),
        poleFromRaw("negative", def?.negative_pole, res?.negative),
      ],
      trained_at: trainedAt,
    };
  });
}

// Extra context the simplex section UI shows above the grid.
export type SimplexInventoryInfo = {
  format?: string;
  source?: string;
  trained_with?: string;
  definitions_version?: string;
  definitions_note?: string;
};

export async function getSimplexInventoryInfo(
  packId: string,
): Promise<SimplexInventoryInfo> {
  const packInfo = await loadPackInfo(packId);
  const defs = await loadSimplexDefinitions(packId);
  return {
    format: packInfo?.simplexes?.format,
    source: packInfo?.simplexes?.source,
    trained_with: packInfo?.simplexes?.trained_with,
    definitions_version: defs?.metadata?.version,
    definitions_note: defs?.metadata?.note,
  };
}

// ---------- Status derivation ----------
//
// Filesystem reality wins over pack_info.json metadata. The `calibration_status`
// note in pack_info is human-authored and easily goes stale (e.g. it says
// "MISSING — calibration cycle has not been run" even after calibration.json
// has been produced). We only fall back to the note when we have no
// filesystem evidence either way.

export function deriveStatus(
  packInfo: LensPackInfo | null,
  hasCalibration: boolean,
): LensPackStatus {
  if (hasCalibration) return "validated";
  const note = packInfo?.calibration_status?.toUpperCase() ?? "";
  if (note.startsWith("MISSING")) return "uncalibrated";
  return "trained";
}

export function deriveCalibrationStatus(
  packInfo: LensPackInfo | null,
  calibration: LensCalibrationFile | null,
  conceptCount = 0,
): CalibrationStatus {
  if (calibration) {
    const calibrated = calibration.total_concepts_calibrated ?? 0;
    if (calibrated === 0) return "missing";
    if (conceptCount > 0 && calibrated < conceptCount * 0.95) return "partial";
    return "complete";
  }
  const note = packInfo?.calibration_status?.toUpperCase() ?? "";
  if (note.startsWith("MISSING")) return "missing";
  return "missing";
}

// ---------- Cross-resource lookups ----------

// Map of model_id -> [lens_pack_id, ...] (for Models slice).
export async function loadLensPacksByModel(): Promise<Map<string, string[]>> {
  const out = new Map<string, string[]>();
  const registry = await loadRegistry();
  if (!registry.packs) return out;

  await Promise.all(
    Object.keys(registry.packs).map(async (packId) => {
      const info = await loadPackInfo(packId);
      if (!info?.model) return;
      const list = out.get(info.model) ?? [];
      list.push(packId);
      out.set(info.model, list);
    }),
  );
  return out;
}

// Map of concept_term -> [lens_pack_id, ...] for lens packs whose source_pack
// matches the given concept pack id (for Concept Packs concept list).
export async function loadLensPacksByConcept(
  conceptPackId: string,
): Promise<Map<string, string[]>> {
  const out = new Map<string, string[]>();
  const registry = await loadRegistry();
  if (!registry.packs) return out;

  await Promise.all(
    Object.keys(registry.packs).map(async (lensPackId) => {
      const info = await loadPackInfo(lensPackId);
      if (info?.source_pack !== conceptPackId) return;
      const manifest = await loadVersionManifest(lensPackId);
      const lensTerms = Object.keys(manifest?.lenses ?? {});
      for (const term of lensTerms) {
        const list = out.get(term) ?? [];
        list.push(lensPackId);
        out.set(term, list);
      }
    }),
  );
  return out;
}

// Search indexer — projects every registered lens pack into a SearchDocument.
export async function indexLensPacks(): Promise<SearchDocument[]> {
  const registry = await loadRegistry();
  const ids = Object.keys(registry.packs ?? {});
  return Promise.all(
    ids.map(async (id) => {
      const dir = packDir(id);
      const info = await loadPackInfo(id);
      const hasCalibration = await pathExists(
        path.join(dir, "calibration.json"),
      );
      const status = deriveStatus(info, hasCalibration);
      const calibStatus = deriveCalibrationStatus(
        info,
        hasCalibration ? { total_concepts_calibrated: 1 } : null,
      );
      const updatedAt =
        (await statMtimeIso(path.join(dir, "pack_info.json"))) ??
        (await statMtimeIso(dir)) ??
        info?.trained_at ??
        "";
      return {
        id: `lens_pack:${id}`,
        resource_type: "lens_pack" as const,
        title: id,
        body_excerpt: [
          info?.model ? `substrate: ${info.model}` : "",
          info?.source_pack
            ? `pack: ${info.source_pack}@${info.pack_version ?? "?"}`
            : "",
          `status: ${status}`,
          `calibration: ${calibStatus}`,
        ]
          .filter(Boolean)
          .join(" · "),
        url: `/lens-packs/${encodeURIComponent(id)}`,
        facets: {
          substrate: info?.model ?? "unknown",
          status,
          calibration_status: calibStatus,
        },
        parent_ids: [],
        updated_at: updatedAt,
      };
    }),
  );
}

// Single-concept variant for the concept detail route.
export async function findLensPacksForConcept(
  conceptPackId: string,
  term: string,
): Promise<string[]> {
  const out: string[] = [];
  const registry = await loadRegistry();
  if (!registry.packs) return out;

  for (const lensPackId of Object.keys(registry.packs)) {
    const info = await loadPackInfo(lensPackId);
    if (info?.source_pack !== conceptPackId) continue;
    const manifest = await loadVersionManifest(lensPackId);
    if (
      manifest?.lenses &&
      Object.prototype.hasOwnProperty.call(manifest.lenses, term)
    ) {
      out.push(lensPackId);
    }
  }
  return out;
}
