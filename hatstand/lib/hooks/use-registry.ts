"use client";

import { useQuery } from "@tanstack/react-query";
import { adminGetData } from "@/lib/api";
import type { Registry } from "@/types";

export function useRegistry() {
  return useQuery<Registry, Error>({
    queryKey: ["admin", "registry"],
    queryFn: () => adminGetData<Registry>("/registry"),
    staleTime: 30_000,
  });
}
