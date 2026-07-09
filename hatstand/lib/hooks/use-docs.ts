"use client";

import { useQuery } from "@tanstack/react-query";
import { adminGetData } from "@/lib/api";
import type { Doc, DocTreeNode } from "@/types";

// Hook for the docs tree. Static-ish data — cached for a few minutes.
export function useDocTree() {
  return useQuery<DocTreeNode, Error>({
    queryKey: ["docs", "tree"],
    queryFn: () => adminGetData<DocTreeNode>("/docs/tree"),
    staleTime: 5 * 60 * 1000,
  });
}

// Hook for a single doc by relative path. Pass null/undefined to disable.
export function useDoc(path: string | null | undefined) {
  return useQuery<Doc, Error>({
    queryKey: ["docs", "file", path ?? ""],
    enabled: !!path,
    queryFn: () => {
      // Encode each segment but leave slashes intact so the catch-all
      // route resolves correctly.
      const encoded = (path as string)
        .split("/")
        .map((s) => encodeURIComponent(s))
        .join("/");
      return adminGetData<Doc>(`/docs/file/${encoded}`);
    },
  });
}
