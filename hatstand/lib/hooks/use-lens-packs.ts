"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { adminGetData, adminGetPage } from "@/lib/api";
import type {
  CalibrationCycle,
  CalibrationOverFirerSummary,
  ConceptCalibration,
  Lens,
  LensPack,
  Page,
  Simplex,
} from "@/types";

export function useLensPacks() {
  return useQuery<Page<LensPack>, Error>({
    queryKey: ["lens-packs"],
    queryFn: () => adminGetPage<LensPack>("/lens-packs"),
  });
}

export function useLensPack(id: string | undefined) {
  return useQuery<LensPack, Error>({
    queryKey: ["lens-pack", id],
    queryFn: () => adminGetData<LensPack>(`/lens-packs/${encodeURIComponent(id!)}`),
    enabled: !!id,
  });
}

export type CalibrationQuery = {
  overFirersOnly?: boolean;
  layer?: number;
  sort?: "cross_fire_rate" | "noise_fire_rate" | "self_mean" | "concept";
  dir?: "asc" | "desc";
  cursor?: string;
  limit?: number;
};

export type LensPackCalibrationResponse = {
  pack_id: string;
  mode?: string;
  has_noise_track: boolean;
  summary: CalibrationOverFirerSummary;
  cycles: CalibrationCycle[];
  entries: Page<ConceptCalibration>;
};

function buildCalibrationQS(opts: CalibrationQuery): string {
  const parts: string[] = [];
  if (opts.overFirersOnly) parts.push("over_firers_only=1");
  if (typeof opts.layer === "number") parts.push(`layer=${opts.layer}`);
  if (opts.sort) parts.push(`sort=${encodeURIComponent(opts.sort)}`);
  if (opts.dir) parts.push(`dir=${encodeURIComponent(opts.dir)}`);
  if (opts.cursor) parts.push(`cursor=${encodeURIComponent(opts.cursor)}`);
  if (typeof opts.limit === "number") parts.push(`limit=${opts.limit}`);
  return parts.length > 0 ? `?${parts.join("&")}` : "";
}

export function useLensPackCalibration(
  id: string | undefined,
  opts: CalibrationQuery = {},
) {
  const qs = buildCalibrationQS(opts);
  return useQuery<LensPackCalibrationResponse, Error>({
    queryKey: ["lens-pack-calibration", id, opts],
    queryFn: () =>
      adminGetData<LensPackCalibrationResponse>(
        `/lens-packs/${encodeURIComponent(id!)}/calibration${qs}`,
      ),
    enabled: !!id,
    placeholderData: keepPreviousData,
  });
}

// ---------- Lens inventory ----------

export type LensesQuery = {
  q?: string;
  layer?: number;
  sort?: "concept" | "layer" | "f1" | "training_samples";
  dir?: "asc" | "desc";
  cursor?: string;
  limit?: number;
};

export type LensWithMeta = Lens & {
  category?: string;
  training_samples?: number;
};

export type LensesResponse = {
  pack_id: string;
  entries: Page<LensWithMeta>;
};

function buildLensesQS(opts: LensesQuery): string {
  const parts: string[] = [];
  if (opts.q) parts.push(`q=${encodeURIComponent(opts.q)}`);
  if (typeof opts.layer === "number") parts.push(`layer=${opts.layer}`);
  if (opts.sort) parts.push(`sort=${encodeURIComponent(opts.sort)}`);
  if (opts.dir) parts.push(`dir=${encodeURIComponent(opts.dir)}`);
  if (opts.cursor) parts.push(`cursor=${encodeURIComponent(opts.cursor)}`);
  if (typeof opts.limit === "number") parts.push(`limit=${opts.limit}`);
  return parts.length > 0 ? `?${parts.join("&")}` : "";
}

export function useLensPackLenses(
  id: string | undefined,
  opts: LensesQuery = {},
) {
  const qs = buildLensesQS(opts);
  return useQuery<LensesResponse, Error>({
    queryKey: ["lens-pack-lenses", id, opts],
    queryFn: () =>
      adminGetData<LensesResponse>(
        `/lens-packs/${encodeURIComponent(id!)}/lenses${qs}`,
      ),
    enabled: !!id,
    placeholderData: keepPreviousData,
  });
}

// ---------- Simplex inventory ----------

export type SimplexInventoryResponse = {
  pack_id: string;
  info: {
    format?: string;
    source?: string;
    trained_with?: string;
    definitions_version?: string;
    definitions_note?: string;
  };
  simplexes: Simplex[];
};

export function useLensPackSimplexes(id: string | undefined) {
  return useQuery<SimplexInventoryResponse, Error>({
    queryKey: ["lens-pack-simplexes", id],
    queryFn: () =>
      adminGetData<SimplexInventoryResponse>(
        `/lens-packs/${encodeURIComponent(id!)}/simplexes`,
      ),
    enabled: !!id,
  });
}
