// Helpers for Next.js route handlers under app/api/admin/*.

import { NextResponse } from "next/server";
import type { ApiResponse, ApiErrorBody } from "@/types";

export function ok<T>(data: T, source?: string): NextResponse {
  const body: ApiResponse<T> = {
    data,
    _meta: { fetched_at: new Date().toISOString(), source },
  };
  return NextResponse.json(body);
}

export function fail(
  code: string,
  message: string,
  status = 500,
  details?: Record<string, unknown>,
): NextResponse {
  const body: ApiErrorBody = { error: { code, message, details } };
  return NextResponse.json(body, { status });
}

export function notFound(message = "Not found"): NextResponse {
  return fail("not_found", message, 404);
}

export function badRequest(message: string, details?: Record<string, unknown>): NextResponse {
  return fail("bad_request", message, 400, details);
}
