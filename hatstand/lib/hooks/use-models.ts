"use client";

import { useQuery } from "@tanstack/react-query";
import { adminGetData, adminGetPage } from "@/lib/api";
import type { Model, Page } from "@/types";

export function useModels() {
  return useQuery<Page<Model>, Error>({
    queryKey: ["models"],
    queryFn: () => adminGetPage<Model>("/models"),
  });
}

export function useModel(id: string) {
  return useQuery<Model, Error>({
    queryKey: ["models", id],
    queryFn: () => adminGetData<Model>(`/models/${encodeURIComponent(id)}`),
    enabled: id.length > 0,
  });
}
