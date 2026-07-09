"use client";

import { useQuery } from "@tanstack/react-query";
import { adminGetData } from "@/lib/api";
import type { EnvironmentReport } from "@/types";

export function useEnv() {
  return useQuery<EnvironmentReport, Error>({
    queryKey: ["admin", "env"],
    queryFn: () => adminGetData<EnvironmentReport>("/env"),
    staleTime: 30_000,
  });
}
