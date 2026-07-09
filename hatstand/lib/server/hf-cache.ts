// HuggingFace hub cache parsing. Walks ${HF_HOME ?? ~/.cache/huggingface}/hub/
// for `models--<org>--<name>` directories and projects each into a Model.

import os from "node:os";
import path from "node:path";
import fs from "node:fs/promises";
import { dirSize } from "@/lib/server/hatcatdev";
import { loadLensPacksByModel } from "@/lib/server/lens-packs";
import type {
  Model,
  ModelSnapshot,
  ModelStatus,
  SearchDocument,
} from "@/types";

const MODEL_FILE_PATTERNS = [
  /^model\.safetensors$/,
  /^pytorch_model\.bin$/,
  /^model-\d+-of-\d+\.safetensors$/,
];

export function hubCacheDir(): string {
  const hfHome = process.env.HF_HOME;
  if (hfHome) return path.join(hfHome, "hub");
  return path.join(os.homedir(), ".cache", "huggingface", "hub");
}

export function decodeModelId(dirName: string): string | null {
  // models--google--gemma-4-E4B -> google/gemma-4-E4B
  if (!dirName.startsWith("models--")) return null;
  return dirName.slice("models--".length).replaceAll("--", "/");
}

export function encodeModelDirName(id: string): string {
  return `models--${id.replaceAll("/", "--")}`;
}

export function deriveFamily(modelId: string): string {
  const name = modelId.includes("/") ? modelId.split("/")[1] : modelId;
  const parts = name.split("-");
  if (parts.length <= 2) return name.toLowerCase();
  return parts.slice(0, 2).join("-").toLowerCase();
}

async function listFiles(dir: string): Promise<string[]> {
  try {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    return entries
      .filter((e) => e.isFile() || e.isSymbolicLink())
      .map((e) => e.name);
  } catch {
    return [];
  }
}

// Snapshot dirs are full of symlinks into ../../blobs. Resolve targets and
// stat real files so size reflects actual disk usage.
async function snapshotSizeBytes(dir: string): Promise<number> {
  let total = 0;
  try {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      try {
        const stat = await fs.stat(full); // follows symlinks
        if (stat.isFile()) total += stat.size;
      } catch {
        // dangling symlink — skip
      }
    }
  } catch {
    // ignore
  }
  return total;
}

async function readDefaultRevision(modelDir: string): Promise<string | undefined> {
  try {
    const content = await fs.readFile(path.join(modelDir, "refs", "main"), "utf-8");
    return content.trim() || undefined;
  } catch {
    return undefined;
  }
}

async function listSnapshotDirs(snapshotsDir: string): Promise<string[]> {
  try {
    const entries = await fs.readdir(snapshotsDir, { withFileTypes: true });
    return entries
      .filter((e) => e.isDirectory() && !e.name.startsWith("."))
      .map((e) => e.name);
  } catch {
    return [];
  }
}

async function statMtimeIso(p: string): Promise<string> {
  try {
    const stat = await fs.stat(p);
    return stat.mtime.toISOString();
  } catch {
    return new Date(0).toISOString();
  }
}

// Read one model dir into a Model record, or null if the dir doesn't look
// like a model (no snapshots). When `requireFullSnapshot` is true, also returns
// null if no snapshot has both config.json and a model file (skips fragments).
export async function readModel(
  hubDir: string,
  dirName: string,
  packsByModel: Map<string, string[]>,
  options: { requireSnapshots?: boolean } = { requireSnapshots: true },
): Promise<Model | null> {
  const modelId = decodeModelId(dirName);
  if (!modelId) return null;

  const modelDir = path.join(hubDir, dirName);
  const snapshotsDir = path.join(modelDir, "snapshots");

  const revisions = await listSnapshotDirs(snapshotsDir);
  if (options.requireSnapshots && revisions.length === 0) return null;

  const defaultRev = await readDefaultRevision(modelDir);

  const snapshots: ModelSnapshot[] = [];
  let hasFullSnapshot = false;
  for (const rev of revisions) {
    const revDir = path.join(snapshotsDir, rev);
    const files = await listFiles(revDir);
    const hasConfig = files.includes("config.json");
    const hasModelFile = files.some((f) =>
      MODEL_FILE_PATTERNS.some((rx) => rx.test(f)),
    );
    if (hasConfig && hasModelFile) hasFullSnapshot = true;
    snapshots.push({
      revision: rev,
      size_bytes: await snapshotSizeBytes(revDir),
      files: files.length,
      is_default: defaultRev === rev,
    });
  }

  const status: ModelStatus =
    revisions.length === 0 ? "partial" : hasFullSnapshot ? "cached" : "partial";

  return {
    id: modelId,
    family: deriveFamily(modelId),
    status,
    disk_path: modelDir,
    size_bytes: await dirSize(modelDir),
    snapshots,
    default_snapshot: defaultRev,
    lens_packs_targeting: packsByModel.get(modelId) ?? [],
    updated_at: await statMtimeIso(modelDir),
  };
}

// List models from the cache. Returns models sorted by id.
export async function listCachedModels(): Promise<Model[]> {
  const hubDir = hubCacheDir();
  const hubExists = await fs
    .access(hubDir)
    .then(() => true)
    .catch(() => false);
  if (!hubExists) return [];

  let dirNames: string[] = [];
  try {
    const entries = await fs.readdir(hubDir, { withFileTypes: true });
    dirNames = entries
      .filter((e) => e.isDirectory() && e.name.startsWith("models--"))
      .map((e) => e.name);
  } catch {
    return [];
  }

  const packsByModel = await loadLensPacksByModel();
  const settled = await Promise.all(
    dirNames.map((d) => readModel(hubDir, d, packsByModel)),
  );
  return settled
    .filter((m): m is Model => m !== null)
    .sort((a, b) => a.id.localeCompare(b.id));
}

// Search indexer — projects each cached model into a SearchDocument.
export async function indexModels(): Promise<SearchDocument[]> {
  const models = await listCachedModels();
  return models.map((m) => ({
    id: `model:${m.id}`,
    resource_type: "model" as const,
    title: m.id,
    body_excerpt: [
      m.family,
      m.status,
      `${m.snapshots.length} snapshot${m.snapshots.length === 1 ? "" : "s"}`,
      m.lens_packs_targeting.length > 0
        ? `targets: ${m.lens_packs_targeting.join(", ")}`
        : "",
    ]
      .filter(Boolean)
      .join(" · "),
    url: `/models/${encodeURIComponent(m.id)}`,
    facets: {
      family: m.family,
      cached: m.status === "cached" ? "yes" : "no",
    },
    parent_ids: [],
    updated_at: m.updated_at,
  }));
}

// Single-model lookup by id — for the detail route. Includes partial-snapshot
// dirs (sets requireSnapshots=false) so the user can see an "in-progress"
// download as a partial entry.
export async function readModelById(modelId: string): Promise<Model | null> {
  const hubDir = hubCacheDir();
  const dirName = encodeModelDirName(modelId);
  const modelDir = path.join(hubDir, dirName);
  const exists = await fs
    .access(modelDir)
    .then(() => true)
    .catch(() => false);
  if (!exists) return null;

  const packsByModel = await loadLensPacksByModel();
  return readModel(hubDir, dirName, packsByModel, { requireSnapshots: false });
}
