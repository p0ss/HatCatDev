"use client";

import { useQuery } from "@tanstack/react-query";
import { adminGetData, adminGetPage } from "@/lib/api";
import type { Meld, MeldSource, MeldState, Page } from "@/types";

export type MeldFilters = {
  state?: MeldState;
  source?: MeldSource;
  target_pack?: string;
  q?: string;
};

function buildQuery(filters?: MeldFilters): string {
  if (!filters) return "";
  const parts: string[] = [];
  if (filters.state) parts.push(`state=${encodeURIComponent(filters.state)}`);
  if (filters.source) parts.push(`source=${encodeURIComponent(filters.source)}`);
  if (filters.target_pack)
    parts.push(`target_pack=${encodeURIComponent(filters.target_pack)}`);
  if (filters.q) parts.push(`q=${encodeURIComponent(filters.q)}`);
  return parts.length ? `?${parts.join("&")}` : "";
}

export function useMelds(filters?: MeldFilters) {
  const qs = buildQuery(filters);
  return useQuery<Page<Meld>, Error>({
    queryKey: ["melds", filters ?? {}],
    queryFn: () => adminGetPage<Meld>(`/melds${qs}`),
  });
}

export function useMeld(id: string | undefined) {
  return useQuery<Meld, Error>({
    queryKey: ["meld", id],
    queryFn: () => adminGetData<Meld>(`/melds/${encodeURIComponent(id!)}`),
    enabled: !!id,
  });
}
