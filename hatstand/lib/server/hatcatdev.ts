// Server-only helpers for resolving and reading HatCatDev paths.
// HatStand lives at HatCatDev/hatstand, so the parent dir is the HatCatDev root.
// Override with HATCATDEV_ROOT env var if HatStand is moved or run elsewhere.

import path from "node:path";
import fs from "node:fs/promises";

let cachedRoot: string | null = null;

export function getHatCatDevRoot(): string {
  if (cachedRoot) return cachedRoot;
  const fromEnv = process.env.HATCATDEV_ROOT;
  if (fromEnv) {
    cachedRoot = path.resolve(fromEnv);
    return cachedRoot;
  }
  // hatstand/ -> HatCatDev/
  cachedRoot = path.resolve(process.cwd(), "..");
  return cachedRoot;
}

export function hatcatdevPath(...segments: string[]): string {
  return path.join(getHatCatDevRoot(), ...segments);
}

export async function readJson<T>(filePath: string): Promise<T> {
  const text = await fs.readFile(filePath, "utf-8");
  return JSON.parse(text) as T;
}

export async function tryReadJson<T>(filePath: string): Promise<T | null> {
  try {
    return await readJson<T>(filePath);
  } catch (err) {
    const e = err as NodeJS.ErrnoException;
    if (e.code === "ENOENT") return null;
    throw err;
  }
}

export async function pathExists(p: string): Promise<boolean> {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}

export async function listDirs(p: string): Promise<string[]> {
  try {
    const entries = await fs.readdir(p, { withFileTypes: true });
    return entries
      .filter((e) => e.isDirectory() && !e.name.startsWith("."))
      .map((e) => e.name);
  } catch (err) {
    const e = err as NodeJS.ErrnoException;
    if (e.code === "ENOENT") return [];
    throw err;
  }
}

export async function statMtimeIso(p: string): Promise<string | undefined> {
  try {
    const stat = await fs.stat(p);
    return stat.mtime.toISOString();
  } catch {
    return undefined;
  }
}

export async function statTimes(
  p: string,
): Promise<{ created?: string; updated?: string }> {
  try {
    const s = await fs.stat(p);
    return { created: s.birthtime.toISOString(), updated: s.mtime.toISOString() };
  } catch {
    return {};
  }
}

export async function dirSize(p: string): Promise<number> {
  let total = 0;
  try {
    const entries = await fs.readdir(p, { withFileTypes: true });
    for (const entry of entries) {
      const full = path.join(p, entry.name);
      if (entry.isDirectory()) {
        total += await dirSize(full);
      } else if (entry.isFile()) {
        const stat = await fs.stat(full);
        total += stat.size;
      }
    }
  } catch {
    // ignore inaccessible dirs
  }
  return total;
}
