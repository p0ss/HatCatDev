"use client";

import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { adminGetData, adminGetPage } from "@/lib/api";
import type { Concept, ConceptPackSummary, Page } from "@/types";

export function useConceptPacks() {
  return useQuery<Page<ConceptPackSummary>, Error>({
    queryKey: ["concept-packs"],
    queryFn: () => adminGetPage<ConceptPackSummary>("/concept-packs"),
  });
}

export function useConceptPack(name: string | undefined) {
  return useQuery<ConceptPackSummary, Error>({
    queryKey: ["concept-pack", name],
    queryFn: () =>
      adminGetData<ConceptPackSummary>(
        `/concept-packs/${encodeURIComponent(name!)}`,
      ),
    enabled: !!name,
  });
}

type ConceptsQuery = {
  layer?: number;
  q?: string;
  cursor?: string;
  limit?: number;
};

function buildConceptsQS(opts: ConceptsQuery): string {
  const parts: string[] = [];
  if (typeof opts.layer === "number") parts.push(`layer=${opts.layer}`);
  if (opts.q) parts.push(`q=${encodeURIComponent(opts.q)}`);
  if (opts.cursor) parts.push(`cursor=${encodeURIComponent(opts.cursor)}`);
  if (typeof opts.limit === "number") parts.push(`limit=${opts.limit}`);
  return parts.length > 0 ? `?${parts.join("&")}` : "";
}

export function useConceptPackConcepts(
  name: string | undefined,
  opts: ConceptsQuery = {},
) {
  const qs = buildConceptsQS(opts);
  return useQuery<Page<Concept>, Error>({
    queryKey: ["concept-pack-concepts", name, opts],
    queryFn: () =>
      adminGetPage<Concept>(
        `/concept-packs/${encodeURIComponent(name!)}/concepts${qs}`,
      ),
    enabled: !!name,
    placeholderData: keepPreviousData,
  });
}

export function useConcept(name: string | undefined, term: string | undefined) {
  return useQuery<Concept, Error>({
    queryKey: ["concept", name, term],
    queryFn: () =>
      adminGetData<Concept>(
        `/concept-packs/${encodeURIComponent(name!)}/concepts/${encodeURIComponent(term!)}`,
      ),
    enabled: !!name && !!term,
  });
}
