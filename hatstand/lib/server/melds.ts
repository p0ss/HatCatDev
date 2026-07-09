// Shared parsing for HatCatDev/melds/. Used by:
//   - app/api/admin/melds/route.ts (list)
//   - app/api/admin/melds/[id]/route.ts (detail)

import path from "node:path";
import fs from "node:fs/promises";
import {
  getHatCatDevRoot,
  pathExists,
  statTimes,
  tryReadJson,
} from "@/lib/server/hatcatdev";
import { parseFlatFrontmatter } from "@/lib/server/frontmatter";
import type {
  Meld,
  MeldCandidate,
  MeldEvidence,
  MeldImpact,
  MeldReview,
  MeldSource,
  MeldState,
  MeldStructuralOp,
  ProtectionLevel,
  SearchDocument,
} from "@/types";

export const MELDS_DIR = "melds";

export const MELD_FOLDERS: Array<{ folder: string; defaultState: MeldState }> = [
  { folder: "pending", defaultState: "review" },
  { folder: "reference", defaultState: "commit" },
  { folder: "applied", defaultState: "evaluate" },
];

export type RawMeld = {
  meld_request_id?: string;
  target_pack_spec_id?: string;
  state?: string;
  metadata?: {
    name?: string;
    description?: string;
    source?: string;
    author?: string;
    created?: string;
    updated?: string;
    version?: string;
    changelog?: string;
  };
  protection_assessment?: {
    protection_level?: string;
    triggers?: Array<{ type?: string; concept?: string; reason?: string }>;
  };
  attachment_points?: Array<{
    target_concept_id?: string;
    relationship?: string;
    candidate_concept?: string;
  }>;
  candidates?: Array<{
    term?: string;
    role?: string;
    parent_concepts?: string[];
    children?: string[];
    definition?: string;
    rationale?: string;
    domain?: string;
    layer_hint?: number;
  }>;
  structural_ops?: unknown;
  impact?: unknown;
  evidence?: unknown;
  reviews?: unknown;
};

const ALLOWED_STATES = new Set<MeldState>([
  "tender",
  "review",
  "authorise",
  "commit",
  "evaluate",
  "rejected",
]);

const ALLOWED_SOURCES = new Set<MeldSource>([
  "manual",
  "be_discovery",
  "cat",
  "cross_be",
  "external",
]);

const ALLOWED_PROTECTION = new Set<ProtectionLevel>([
  "open",
  "guarded",
  "sealed",
]);

export function normaliseSource(raw?: string): MeldSource {
  if (!raw) return "manual";
  const lower = raw.toLowerCase();
  if (ALLOWED_SOURCES.has(lower as MeldSource)) return lower as MeldSource;
  if (lower.includes("be") && lower.includes("discov")) return "be_discovery";
  if (lower.includes("cat")) return "cat";
  if (lower.includes("cross")) return "cross_be";
  if (lower.includes("external") || lower.includes("import")) return "external";
  if (lower.includes("checker") || lower.includes("generator")) return "manual";
  return "manual";
}

export function normaliseProtection(raw?: string): ProtectionLevel {
  if (!raw) return "open";
  const lower = raw.toLowerCase();
  if (ALLOWED_PROTECTION.has(lower as ProtectionLevel))
    return lower as ProtectionLevel;
  // On-disk vocabulary uses standard / elevated / protected / critical.
  if (lower === "standard") return "open";
  if (lower === "elevated") return "guarded";
  if (lower === "protected" || lower === "critical") return "sealed";
  return "open";
}

export function normaliseStateOverride(raw?: string): MeldState | undefined {
  if (!raw) return undefined;
  const lower = raw.toLowerCase();
  return ALLOWED_STATES.has(lower as MeldState)
    ? (lower as MeldState)
    : undefined;
}

export function deriveCandidates(raw: RawMeld): MeldCandidate[] {
  const out: MeldCandidate[] = [];
  if (Array.isArray(raw.candidates)) {
    for (const c of raw.candidates) {
      out.push({
        kind: c.role === "relationship" ? "relationship" : "concept",
        term: c.term,
        parent: c.parent_concepts?.[0],
        children: c.children,
        rationale: c.rationale ?? c.definition,
      });
    }
  }
  if (out.length === 0 && Array.isArray(raw.attachment_points)) {
    for (const ap of raw.attachment_points) {
      out.push({
        kind: "relationship",
        term: ap.candidate_concept,
        parent: ap.target_concept_id,
        rationale: ap.relationship,
      });
    }
  }
  return out;
}

export function targetPackFromSpec(specId?: string): string {
  if (!specId) return "unknown";
  const slash = specId.indexOf("/");
  return slash >= 0 ? specId.slice(slash + 1) : specId;
}

export function basenameId(file: string): string {
  return file.replace(/\.json$/i, "").replace(/\.md$/i, "");
}

export function rawIdToShort(meldRequestId?: string): string | null {
  if (!meldRequestId) return null;
  return meldRequestId.replace(/^.*?\//, "");
}

export async function loadJsonMeld(
  filePath: string,
  fileName: string,
  defaultState: MeldState,
): Promise<Meld | null> {
  const raw = await tryReadJson<RawMeld>(filePath);
  if (!raw) return null;
  const id = rawIdToShort(raw.meld_request_id) ?? basenameId(fileName);
  const state = normaliseStateOverride(raw.state) ?? defaultState;
  const times = await statTimes(filePath);
  return {
    id,
    state,
    source: normaliseSource(raw.metadata?.source),
    target_pack: targetPackFromSpec(raw.target_pack_spec_id),
    protection_level: normaliseProtection(
      raw.protection_assessment?.protection_level,
    ),
    candidates: deriveCandidates(raw),
    structural_ops: Array.isArray(raw.structural_ops)
      ? (raw.structural_ops as MeldStructuralOp[])
      : [],
    impact: (raw.impact as MeldImpact | undefined) ?? undefined,
    evidence: (raw.evidence as MeldEvidence | undefined) ?? undefined,
    reviews: Array.isArray(raw.reviews) ? (raw.reviews as MeldReview[]) : [],
    created_at: raw.metadata?.created ?? times.created ?? "",
    updated_at:
      raw.metadata?.updated ?? times.updated ?? raw.metadata?.created ?? "",
  };
}

export async function loadMdMeld(
  filePath: string,
  fileName: string,
  defaultState: MeldState,
): Promise<Meld | null> {
  let raw: string;
  try {
    raw = await fs.readFile(filePath, "utf-8");
  } catch {
    return null;
  }
  const { fm } = parseFlatFrontmatter(raw);
  const id = (fm.id as string | undefined) ?? basenameId(fileName);
  const state =
    normaliseStateOverride(fm.state as string | undefined) ?? defaultState;
  const times = await statTimes(filePath);
  return {
    id,
    state,
    source: normaliseSource(fm.source as string | undefined),
    target_pack: targetPackFromSpec(
      (fm.target_pack_spec_id as string | undefined) ??
        (fm.target_pack as string | undefined),
    ),
    protection_level: normaliseProtection(
      fm.protection_level as string | undefined,
    ),
    candidates: [],
    structural_ops: [],
    reviews: [],
    created_at:
      (fm.created_at as string | undefined) ??
      (fm.created as string | undefined) ??
      times.created ??
      "",
    updated_at:
      (fm.updated_at as string | undefined) ??
      (fm.updated as string | undefined) ??
      times.updated ??
      "",
  };
}

async function listMeldFiles(absDir: string): Promise<string[]> {
  try {
    const entries = await fs.readdir(absDir, { withFileTypes: true });
    return entries
      .filter(
        (e) =>
          e.isFile() && (e.name.endsWith(".json") || e.name.endsWith(".md")),
      )
      .map((e) => e.name);
  } catch {
    return [];
  }
}

// Walk all three meld folders and load every meld file. The order of the
// returned array matches MELD_FOLDERS for predictable downstream sorting.
export async function loadAllMelds(): Promise<Meld[]> {
  const root = getHatCatDevRoot();
  const items: Meld[] = [];
  for (const { folder, defaultState } of MELD_FOLDERS) {
    const dir = path.join(root, MELDS_DIR, folder);
    if (!(await pathExists(dir))) continue;
    const files = await listMeldFiles(dir);
    for (const fileName of files) {
      const filePath = path.join(dir, fileName);
      const meld = fileName.endsWith(".md")
        ? await loadMdMeld(filePath, fileName, defaultState)
        : await loadJsonMeld(filePath, fileName, defaultState);
      if (meld) items.push(meld);
    }
  }
  return items;
}

// Search indexer — every meld in every folder as a SearchDocument.
export async function indexMelds(): Promise<SearchDocument[]> {
  const melds = await loadAllMelds();
  return melds.map((m) => ({
    id: `meld:${m.id}`,
    resource_type: "meld" as const,
    title: m.id,
    body_excerpt: [
      `target: ${m.target_pack}`,
      `state: ${m.state}`,
      `source: ${m.source}`,
      m.candidates.length > 0
        ? `candidates: ${m.candidates
            .slice(0, 8)
            .map((c) => c.term ?? "")
            .filter(Boolean)
            .join(", ")}`
        : "",
    ]
      .filter(Boolean)
      .join(" · "),
    url: `/melds/${encodeURIComponent(m.id)}`,
    facets: {
      state: m.state,
      source: m.source,
      target_pack: m.target_pack,
      protection_level: m.protection_level,
    },
    parent_ids: [],
    updated_at: m.updated_at,
  }));
}

// Locate one meld by id. Tries direct filename match first (fast path); falls
// back to scanning every file for matching meld_request_id short form.
export async function findMeldById(
  id: string,
): Promise<{ meld: Meld; foundIn: string } | null> {
  const root = getHatCatDevRoot();

  // Fast path: direct filename match
  for (const { folder, defaultState } of MELD_FOLDERS) {
    const dir = path.join(root, MELDS_DIR, folder);
    if (!(await pathExists(dir))) continue;

    const directJson = path.join(dir, `${id}.json`);
    if (await pathExists(directJson)) {
      const meld = await loadJsonMeld(directJson, `${id}.json`, defaultState);
      if (meld) return { meld, foundIn: folder };
    }
    const directMd = path.join(dir, `${id}.md`);
    if (await pathExists(directMd)) {
      const meld = await loadMdMeld(directMd, `${id}.md`, defaultState);
      if (meld) return { meld, foundIn: folder };
    }
  }

  // Slow path: scan files for matching meld_request_id short form
  for (const { folder, defaultState } of MELD_FOLDERS) {
    const dir = path.join(root, MELDS_DIR, folder);
    if (!(await pathExists(dir))) continue;
    const files = await listMeldFiles(dir);
    for (const fileName of files) {
      const filePath = path.join(dir, fileName);
      const meld = fileName.endsWith(".md")
        ? await loadMdMeld(filePath, fileName, defaultState)
        : await loadJsonMeld(filePath, fileName, defaultState);
      if (meld && meld.id === id) return { meld, foundIn: folder };
    }
  }
  return null;
}
