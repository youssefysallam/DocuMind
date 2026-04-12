"use client";

import { useState, useCallback } from "react";
import Link from "next/link";
import { ArrowLeft, Plus } from "@phosphor-icons/react";
import { sendMessage, clearSession } from "@/lib/api";
import type {
  ChatMessage,
  RetrievedSource,
  RetrievalLog,
  ConsistencyResult,
} from "@/lib/types";
import MessageList from "./MessageList";
import InputBar from "./InputBar";
import EmptyState from "./EmptyState";
import SourcePanel from "./SourcePanel";
import RetrievalDetails from "./RetrievalDetails";
import ConsistencyCheck from "./ConsistencyCheck";

function generateSessionId() {
  return crypto.randomUUID();
}

export default function ChatInterface() {
  const [sessionId] = useState(generateSessionId);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sources, setSources] = useState<RetrievedSource[]>([]);
  const [retrievalLog, setRetrievalLog] = useState<RetrievalLog | null>(null);
  const [consistency, setConsistency] = useState<ConsistencyResult | null>(
    null
  );
  const [loading, setLoading] = useState(false);

  const handleSend = useCallback(
    async (message: string) => {
      setMessages((prev) => [...prev, { role: "user", content: message }]);
      setLoading(true);

      try {
        const res = await sendMessage(message, sessionId);
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: res.answer },
        ]);
        setSources(res.retrieved);
        setRetrievalLog(res.retrieval_log);
        setConsistency(res.consistency);
      } catch {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content:
              "Connection to the backend failed. Make sure the FastAPI server is running on port 8000.",
          },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [sessionId]
  );

  const handleClear = useCallback(async () => {
    await clearSession(sessionId);
    setMessages([]);
    setSources([]);
    setRetrievalLog(null);
    setConsistency(null);
  }, [sessionId]);

  const isEmpty = messages.length === 0;

  return (
    <div className="flex h-[100dvh] flex-col bg-[#f9fafb]">
      {/* Top bar */}
      <div className="flex items-center justify-between border-b border-slate-200/50 bg-white/80 px-4 py-3 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="flex items-center gap-1.5 text-sm text-slate-500 transition-colors hover:text-zinc-900"
          >
            <ArrowLeft size={14} />
            <span>Back</span>
          </Link>
          <span className="text-sm font-semibold text-zinc-900">
            InfoWeave
          </span>
        </div>
        <button
          onClick={handleClear}
          className="flex items-center gap-1.5 rounded-lg border border-slate-200/50 px-3 py-1.5 text-xs text-slate-500 transition-all hover:border-slate-300 hover:text-zinc-900 active:scale-[0.98]"
        >
          <Plus size={12} />
          New conversation
        </button>
      </div>

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Conversation area */}
        <div className="flex flex-1 flex-col">
          {isEmpty ? (
            <EmptyState onSelectExample={handleSend} />
          ) : (
            <MessageList messages={messages} loading={loading} />
          )}
          <InputBar
            onSend={handleSend}
            disabled={loading}
            showTypewriter={isEmpty}
          />
        </div>

        {/* Context panel -- desktop only */}
        <div className="hidden w-[380px] flex-shrink-0 overflow-y-auto border-l border-slate-200/50 bg-white md:block">
          <SourcePanel sources={sources} />
          <RetrievalDetails log={retrievalLog} />
          <ConsistencyCheck consistency={consistency} />
        </div>
      </div>
    </div>
  );
}
