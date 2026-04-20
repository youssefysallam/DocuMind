import type {
  ChatResponse,
  EvalData,
  FeaturedResponse,
  SuggestResponse,
  SystemInfo,
} from "./types";

const BASE = "";

export async function sendMessage(
  message: string,
  sessionId: string
): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
  });
  if (!res.ok) throw new Error(`Chat request failed: ${res.status}`);
  return res.json();
}

// Hit /api/suggest with a debounced keyword — accept an AbortSignal so stale keystrokes cancel cleanly.
export async function fetchSuggestions(
  keyword: string,
  topK: number = 5,
  sourceFilter: "pdf" | "website" | null = null,
  signal?: AbortSignal
): Promise<SuggestResponse> {
  const body: Record<string, unknown> = { keyword, top_k: topK };
  // Only send the field when a real filter is active — keeps the wire payload minimal.
  if (sourceFilter) body.source_filter = sourceFilter;
  const res = await fetch(`${BASE}/api/suggest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw new Error(`Suggest request failed: ${res.status}`);
  return res.json();
}

// Pull the curated featured set once on mount — backend caches the payload, so repeat calls are cheap.
export async function fetchFeatured(): Promise<FeaturedResponse> {
  const res = await fetch(`${BASE}/api/featured`);
  if (!res.ok) throw new Error(`Featured fetch failed: ${res.status}`);
  return res.json();
}

export async function clearSession(sessionId: string): Promise<void> {
  await fetch(`${BASE}/api/session/clear`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
}

export async function fetchEval(): Promise<EvalData> {
  const res = await fetch(`${BASE}/api/eval`);
  if (!res.ok) throw new Error(`Eval fetch failed: ${res.status}`);
  return res.json();
}

export async function fetchSystemInfo(): Promise<SystemInfo> {
  const res = await fetch(`${BASE}/api/system`);
  if (!res.ok) throw new Error(`System fetch failed: ${res.status}`);
  return res.json();
}
