// Docs search indexer. Walks the same indexed sources as
// app/api/admin/docs/tree/route.ts and produces a SearchDocument per .md file.
// Reads each file's body (best-effort, capped) for substring-search snippets.

import path from "node:path";
import fs from "node:fs/promises";
import { getHatCatDevRoot } from "@/lib/server/hatcatdev";
import { parseFlatFrontmatter } from "@/lib/server/frontmatter";
import type { SearchDocument } from "@/types";

const ROOT_MD_FILES = [
  "README.md",
  "PROJECT_PLAN_PHASE_A.md",
  "PROJECT_PLAN_PHASE_B.md",
  "PROJECT_OVERVIEW.md",
  "QUICKSTART.md",
  "DEPLOYMENT.md",
  "TRAINING_QUICK_START.md",
];

// Walk a directory subtree under HatCatDev and yield (relPath, absPath) for
// every *.md file (skipping dotfiles).
async function* walkMarkdown(
  absDir: string,
  relDir: string,
): AsyncGenerator<{ relPath: string; absPath: string }> {
  let entries: import("node:fs").Dirent[];
  try {
    entries = await fs.readdir(absDir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const entry of entries) {
    if (entry.name.startsWith(".")) continue;
    const absChild = path.join(absDir, entry.name);
    const relChild = relDir ? `${relDir}/${entry.name}` : entry.name;
    if (entry.isDirectory()) {
      yield* walkMarkdown(absChild, relChild);
    } else if (entry.isFile() && entry.name.endsWith(".md")) {
      yield { relPath: relChild, absPath: absChild };
    }
  }
}

async function readMaybe(absPath: string): Promise<string | null> {
  try {
    return await fs.readFile(absPath, "utf-8");
  } catch {
    return null;
  }
}

function pickTitle(fm: Record<string, unknown>, body: string, relPath: string): string {
  const fmTitle = fm["title"];
  if (typeof fmTitle === "string" && fmTitle.trim()) return fmTitle.trim();
  // first H1
  const h1 = body.match(/^#\s+(.+)$/m);
  if (h1) return h1[1].trim();
  return path.basename(relPath).replace(/\.md$/, "");
}

function bodySnippet(body: string, maxLen = 400): string {
  // strip code fences and html tags for a more useful snippet
  const stripped = body
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return stripped.length > maxLen ? `${stripped.slice(0, maxLen)}…` : stripped;
}

export async function indexDocs(): Promise<SearchDocument[]> {
  const root = getHatCatDevRoot();
  const out: SearchDocument[] = [];

  // Top-level .md files
  for (const name of ROOT_MD_FILES) {
    const absPath = path.join(root, name);
    const raw = await readMaybe(absPath);
    if (raw === null) continue;
    const stat = await fs.stat(absPath);
    const { fm, rest } = parseFlatFrontmatter(raw);
    out.push({
      id: `doc:${name}`,
      resource_type: "doc" as const,
      title: pickTitle(fm, rest, name),
      body_excerpt: bodySnippet(rest),
      url: `/docs/${name}`,
      facets: { folder: "" },
      parent_ids: [],
      updated_at: stat.mtime.toISOString(),
    });
  }

  // Walk the indexed subtrees
  const subtrees: Array<{ rel: string; absRoot: string }> = [
    { rel: "docs", absRoot: path.join(root, "docs") },
    { rel: "melds/reference", absRoot: path.join(root, "melds", "reference") },
    { rel: "melds/applied", absRoot: path.join(root, "melds", "applied") },
  ];

  for (const { rel, absRoot } of subtrees) {
    for await (const { relPath, absPath } of walkMarkdown(absRoot, rel)) {
      const raw = await readMaybe(absPath);
      if (raw === null) continue;
      let mtime: Date;
      try {
        mtime = (await fs.stat(absPath)).mtime;
      } catch {
        mtime = new Date(0);
      }
      const { fm, rest } = parseFlatFrontmatter(raw);
      const folder = path.posix.dirname(relPath);
      out.push({
        id: `doc:${relPath}`,
        resource_type: "doc" as const,
        title: pickTitle(fm, rest, relPath),
        body_excerpt: bodySnippet(rest),
        url: `/docs/${relPath}`,
        facets: { folder: folder === "." ? "" : folder },
        parent_ids: [],
        updated_at: mtime.toISOString(),
      });
    }
  }

  // Per-{concept-pack,lens-pack}/README.md
  for (const parent of ["concept_packs", path.join("src", "lens_packs")]) {
    const parentAbs = path.join(root, parent);
    let dirs: import("node:fs").Dirent[];
    try {
      dirs = await fs.readdir(parentAbs, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const dir of dirs) {
      if (!dir.isDirectory() || dir.name.startsWith(".")) continue;
      const readmeAbs = path.join(parentAbs, dir.name, "README.md");
      const raw = await readMaybe(readmeAbs);
      if (raw === null) continue;
      const stat = await fs.stat(readmeAbs);
      const { fm, rest } = parseFlatFrontmatter(raw);
      const relPath = `${parent.replaceAll(path.sep, "/")}/${dir.name}/README.md`;
      out.push({
        id: `doc:${relPath}`,
        resource_type: "doc" as const,
        title: pickTitle(fm, rest, relPath),
        body_excerpt: bodySnippet(rest),
        url: `/docs/${relPath}`,
        facets: { folder: path.posix.dirname(relPath) },
        parent_ids: [],
        updated_at: stat.mtime.toISOString(),
      });
    }
  }

  return out;
}
