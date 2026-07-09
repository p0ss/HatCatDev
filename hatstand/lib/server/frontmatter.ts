// Best-effort flat YAML frontmatter parser. Handles `key: value` pairs where
// value is a scalar (string, number, boolean). No arrays, no nested objects.
// Anything more complex stays as a raw string. We intentionally do NOT pull
// in a YAML library — frontmatter in this codebase is uniformly flat.

export type Frontmatter = Record<string, string | number | boolean | undefined>;

export function parseFlatFrontmatter(raw: string): {
  fm: Frontmatter;
  rest: string;
} {
  if (!raw.startsWith("---\n") && !raw.startsWith("---\r\n")) {
    return { fm: {}, rest: raw };
  }
  const afterOpen = raw.indexOf("\n") + 1;
  const closeRe = /\n---\s*(\r?\n|$)/;
  const closeMatch = closeRe.exec(raw.slice(afterOpen));
  if (!closeMatch) return { fm: {}, rest: raw };
  const block = raw.slice(afterOpen, afterOpen + closeMatch.index);
  const restStart = afterOpen + closeMatch.index + closeMatch[0].length;
  const rest = raw.slice(restStart);

  const fm: Frontmatter = {};
  for (const line of block.split(/\r?\n/)) {
    if (!line.trim() || line.trimStart().startsWith("#")) continue;
    const m = /^([A-Za-z0-9_\-]+)\s*:\s*(.*)$/.exec(line);
    if (!m) continue;
    const key = m[1];
    let value: string = m[2].trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    if (value === "") {
      fm[key] = "";
      continue;
    }
    if (value === "true" || value === "false") {
      fm[key] = value === "true";
      continue;
    }
    if (/^-?\d+(\.\d+)?$/.test(value)) {
      fm[key] = Number(value);
      continue;
    }
    fm[key] = value;
  }
  return { fm, rest };
}
