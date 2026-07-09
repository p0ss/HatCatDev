"use client";

import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import { adminGetData } from "@/lib/api";
import type { ApiResponse } from "@/types";

// Reference hook: how slice agents should consume resource detail endpoints.
// Slice authors should write their own typed wrappers, e.g. useModel(id), but
// follow this pattern: useQuery + adminGetData (which unwraps ApiResponse<T>).

export function useResource<T>(
  key: readonly unknown[],
  path: string,
  options?: Omit<
    UseQueryOptions<T, Error, T, readonly unknown[]>,
    "queryKey" | "queryFn"
  >,
) {
  return useQuery<T, Error, T, readonly unknown[]>({
    queryKey: key,
    queryFn: () => adminGetData<T>(path),
    ...(options ?? {}),
  });
}

// Re-export for convenience inside slices
export type { ApiResponse };
