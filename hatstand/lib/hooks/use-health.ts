"use client";

import { useQuery } from "@tanstack/react-query";
import { adminGetData } from "@/lib/api";
import type { HealthStatus } from "@/types";

export function useHealth() {
  return useQuery<HealthStatus, Error>({
    queryKey: ["admin", "health"],
    queryFn: () => adminGetData<HealthStatus>("/health"),
    // Health is cheap; keep it relatively fresh so the connection badge reacts.
    staleTime: 5_000,
    refetchInterval: 30_000,
  });
}
