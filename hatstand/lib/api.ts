import { ApiError, ApiErrorBody, ApiResponse, Page } from "@/types";

// Base URL for the admin API. In current shape (Phase 0), HatStand serves its
// own /api/admin/* via Next.js route handlers that read HatCatDev's filesystem
// directly. Later phases may proxy to a separate HatCatDev process; consumers
// shouldn't care — they always call adminFetch with a path under /api/admin.
const API_BASE = "/api/admin";

export async function adminFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const url = path.startsWith("http") ? path : `${API_BASE}${path}`;
  const res = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (!res.ok) {
    let body: ApiErrorBody | undefined;
    try {
      body = (await res.json()) as ApiErrorBody;
    } catch {
      // non-JSON error body
    }
    throw new ApiError(
      body?.error?.code ?? `http_${res.status}`,
      body?.error?.message ?? res.statusText,
      res.status,
      body?.error?.details,
    );
  }

  return (await res.json()) as T;
}

export async function adminGet<T>(path: string): Promise<T> {
  return adminFetch<T>(path);
}

export async function adminGetData<T>(path: string): Promise<T> {
  const wrapped = await adminFetch<ApiResponse<T>>(path);
  return wrapped.data;
}

export async function adminGetPage<T>(path: string): Promise<Page<T>> {
  // List endpoints follow the same envelope contract as detail endpoints —
  // ApiResponse<Page<T>> on the wire, unwrapped here so consumers see Page<T>.
  const wrapped = await adminFetch<ApiResponse<Page<T>>>(path);
  return wrapped.data;
}

export async function adminPost<T, B = unknown>(
  path: string,
  body: B,
): Promise<T> {
  return adminFetch<T>(path, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function adminDelete<T>(path: string): Promise<T> {
  return adminFetch<T>(path, { method: "DELETE" });
}

export function buildSearchQuery(params: {
  q?: string;
  type?: string;
  filter?: Record<string, string | string[]>;
  sort?: { field: string; direction: "asc" | "desc" };
  cursor?: string;
  limit?: number;
}): string {
  const parts: string[] = [];
  if (params.q) parts.push(`q=${encodeURIComponent(params.q)}`);
  if (params.type) parts.push(`type=${encodeURIComponent(params.type)}`);
  if (params.filter) {
    for (const [k, v] of Object.entries(params.filter)) {
      const values = Array.isArray(v) ? v : [v];
      for (const val of values) {
        parts.push(`filter[${encodeURIComponent(k)}]=${encodeURIComponent(val)}`);
      }
    }
  }
  if (params.sort)
    parts.push(`sort=${encodeURIComponent(`${params.sort.field}:${params.sort.direction}`)}`);
  if (params.cursor) parts.push(`cursor=${encodeURIComponent(params.cursor)}`);
  if (params.limit) parts.push(`limit=${params.limit}`);
  return parts.length > 0 ? `?${parts.join("&")}` : "";
}
