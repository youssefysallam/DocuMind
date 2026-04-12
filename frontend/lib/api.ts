import type { ChatResponse, EvalData, SystemInfo } from "./types";

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
