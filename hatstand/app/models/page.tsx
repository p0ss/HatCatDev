"use client";

import Link from "next/link";
import { PageHeader } from "@/components/page-header";
import {
  Badge,
  Card,
  CardBody,
  CardHeader,
  CardTitle,
  EmptyState,
  ErrorState,
} from "@/components/ui";
import { useModels } from "@/lib/hooks/use-models";
import type { Model, ModelStatus } from "@/types";

const STATUS_VARIANT: Record<ModelStatus, "muted" | "info" | "success" | "warning" | "error"> = {
  cached: "success",
  partial: "warning",
  hub_only: "info",
  error: "error",
};

function formatBytes(bytes?: number): string {
  if (!bytes || bytes <= 0) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let value = bytes;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i++;
  }
  return `${value.toFixed(value < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}

function ModelCard({ model }: { model: Model }) {
  return (
    <Link href={`/models/${encodeURIComponent(model.id)}`} className="group block">
      <Card className="transition group-hover:border-zinc-400 dark:group-hover:border-zinc-600">
        <CardHeader className="flex items-center justify-between gap-3">
          <CardTitle className="truncate font-mono text-xs">{model.id}</CardTitle>
          <Badge variant={STATUS_VARIANT[model.status]}>{model.status}</Badge>
        </CardHeader>
        <CardBody className="space-y-2 text-xs text-zinc-600 dark:text-zinc-400">
          <div className="flex justify-between">
            <span>Family</span>
            <span className="font-mono text-zinc-900 dark:text-zinc-100">{model.family}</span>
          </div>
          <div className="flex justify-between">
            <span>Snapshots</span>
            <span className="text-zinc-900 dark:text-zinc-100">{model.snapshots.length}</span>
          </div>
          <div className="flex justify-between">
            <span>Size</span>
            <span className="text-zinc-900 dark:text-zinc-100">{formatBytes(model.size_bytes)}</span>
          </div>
          <div className="flex justify-between">
            <span>Lens packs</span>
            <span className="text-zinc-900 dark:text-zinc-100">
              {model.lens_packs_targeting.length}
            </span>
          </div>
        </CardBody>
      </Card>
    </Link>
  );
}

export default function ModelsPage() {
  const { data, isLoading, isError, error, refetch } = useModels();

  return (
    <div className="px-8 py-6">
      <PageHeader
        title="Models"
        description="Substrate inventory — local cache."
      />
      <div className="mt-6">
        {isLoading ? (
          <div className="text-sm text-zinc-500 dark:text-zinc-400">Loading…</div>
        ) : isError ? (
          <ErrorState message={error?.message} onRetry={() => refetch()} />
        ) : !data || data.items.length === 0 ? (
          <EmptyState
            title="No models cached yet"
            description="HuggingFace cache is empty. Pull a model with `huggingface-cli download` or run a training step that downloads one. Cache path: $HF_HOME/hub or ~/.cache/huggingface/hub."
          />
        ) : (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            {data.items.map((m) => (
              <ModelCard key={m.id} model={m} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
