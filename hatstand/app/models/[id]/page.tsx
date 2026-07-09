"use client";

import Link from "next/link";
import { use } from "react";
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
import { useModel } from "@/lib/hooks/use-models";
import type { ModelStatus } from "@/types";
import { ApiError } from "@/types";

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

function shortRev(rev: string): string {
  return rev.length > 12 ? `${rev.slice(0, 12)}…` : rev;
}

export default function ModelDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id: encodedId } = use(params);
  const modelId = decodeURIComponent(encodedId);
  const { data: model, isLoading, isError, error, refetch } = useModel(modelId);

  if (isLoading) {
    return (
      <div className="px-8 py-6 text-sm text-zinc-500 dark:text-zinc-400">Loading…</div>
    );
  }

  if (isError) {
    if (error instanceof ApiError && error.status === 404) {
      return (
        <div className="px-8 py-6">
          <PageHeader title={modelId} description="Model detail" />
          <div className="mt-6">
            <EmptyState
              title="Model not found in local cache"
              description={`No HuggingFace cache entry for ${modelId}. Run \`huggingface-cli download ${modelId}\` to populate it.`}
            />
          </div>
        </div>
      );
    }
    return (
      <div className="px-8 py-6">
        <PageHeader title={modelId} description="Model detail" />
        <div className="mt-6">
          <ErrorState message={error?.message} onRetry={() => refetch()} />
        </div>
      </div>
    );
  }

  if (!model) {
    return (
      <div className="px-8 py-6">
        <PageHeader title={modelId} />
        <EmptyState title="Model not found in local cache" />
      </div>
    );
  }

  return (
    <div className="px-8 py-6">
      <PageHeader
        title={model.id}
        description={`Family: ${model.family}`}
        actions={<Badge variant={STATUS_VARIANT[model.status]}>{model.status}</Badge>}
      />

      <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Identity</CardTitle>
          </CardHeader>
          <CardBody className="space-y-2 text-xs">
            <Row label="Model id" value={<span className="font-mono">{model.id}</span>} />
            <Row label="Family" value={model.family} />
            <Row
              label="Default snapshot"
              value={
                model.default_snapshot ? (
                  <span className="font-mono">{shortRev(model.default_snapshot)}</span>
                ) : (
                  "—"
                )
              }
            />
            <Row label="Total size" value={formatBytes(model.size_bytes)} />
            <Row label="Updated" value={new Date(model.updated_at).toLocaleString()} />
          </CardBody>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Lens packs targeting</CardTitle>
          </CardHeader>
          <CardBody>
            {model.lens_packs_targeting.length === 0 ? (
              <p className="text-xs text-zinc-500 dark:text-zinc-400">
                No lens packs target this substrate.
              </p>
            ) : (
              <ul className="space-y-1 text-xs">
                {model.lens_packs_targeting.map((packId) => (
                  <li key={packId}>
                    <Link
                      href={`/lens-packs/${encodeURIComponent(packId)}`}
                      className="font-mono text-sky-700 hover:underline dark:text-sky-300"
                    >
                      {packId}
                    </Link>
                  </li>
                ))}
              </ul>
            )}
          </CardBody>
        </Card>
      </div>

      <div className="mt-4">
        <Card>
          <CardHeader>
            <CardTitle>Snapshots</CardTitle>
          </CardHeader>
          <CardBody>
            {model.snapshots.length === 0 ? (
              <p className="text-xs text-zinc-500 dark:text-zinc-400">No snapshots on disk.</p>
            ) : (
              <table className="w-full text-xs">
                <thead className="text-left text-zinc-500 dark:text-zinc-400">
                  <tr>
                    <th className="pb-2 font-medium">Revision</th>
                    <th className="pb-2 font-medium">Size</th>
                    <th className="pb-2 font-medium">Files</th>
                    <th className="pb-2 font-medium">Default</th>
                  </tr>
                </thead>
                <tbody>
                  {model.snapshots.map((s) => (
                    <tr key={s.revision} className="border-t border-zinc-100 dark:border-zinc-800">
                      <td className="py-2 font-mono">{shortRev(s.revision)}</td>
                      <td className="py-2">{formatBytes(s.size_bytes)}</td>
                      <td className="py-2">{s.files}</td>
                      <td className="py-2">
                        {s.is_default ? <Badge variant="info">default</Badge> : null}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </CardBody>
        </Card>
      </div>

      {model.disk_path ? (
        <div className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Disk path</CardTitle>
            </CardHeader>
            <CardBody>
              <code className="block break-all text-xs text-zinc-700 dark:text-zinc-300">
                {model.disk_path}
              </code>
            </CardBody>
          </Card>
        </div>
      ) : null}
    </div>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-4">
      <span className="text-zinc-500 dark:text-zinc-400">{label}</span>
      <span className="text-right text-zinc-900 dark:text-zinc-100">{value}</span>
    </div>
  );
}
