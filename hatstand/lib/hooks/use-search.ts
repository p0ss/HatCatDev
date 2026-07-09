"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { adminGetData } from "@/lib/api";
import type { ResourceType, SearchResponse } from "@/types";

export type UseSearchOptions = {
  q?: string;
  type?: ResourceType;
  filter?: Record<string, string[]>;
  cursor?: string;
  limit?: number;
};

function buildSearchUrl(opts: UseSearchOptions): string {
  const parts: string[] = [];
  if (opts.q) parts.push(`q=${encodeURIComponent(opts.q)}`);
  if (opts.type) parts.push(`type=${encodeURIComponent(opts.type)}`);
  if (opts.filter) {
    for (const [k, values] of Object.entries(opts.filter)) {
      for (const v of values) {
        parts.push(`filter[${encodeURIComponent(k)}]=${encodeURIComponent(v)}`);
      }
    }
  }
  if (opts.cursor) parts.push(`cursor=${encodeURIComponent(opts.cursor)}`);
  if (opts.limit) parts.push(`limit=${opts.limit}`);
  return parts.length > 0 ? `/search?${parts.join("&")}` : "/search";
}

export function useSearch(opts: UseSearchOptions) {
  const url = buildSearchUrl(opts);
  return useQuery<SearchResponse, Error>({
    queryKey: ["search", opts],
    queryFn: () => adminGetData<SearchResponse>(url),
    placeholderData: keepPreviousData,
    staleTime: 15_000,
  });
}
