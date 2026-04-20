"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { fetchSuggestions } from "@/lib/api";
import type { Suggestion } from "@/lib/types";

interface FollowUpBarProps {
  lastUserMessage: string;
  onSelect: (question: string) => void;
  disabled?: boolean;
  // Lowercased+trimmed user questions from the current conversation — drop any chip that repeats one.
  priorQuestions?: string[];
}

// Cap chips so the row stays a single readable line, not a wall of links.
const FOLLOW_UP_LIMIT = 3;

export default function FollowUpBar({
  lastUserMessage,
  onSelect,
  disabled = false,
  priorQuestions = [],
}: FollowUpBarProps) {
  const [items, setItems] = useState<Suggestion[]>([]);

  useEffect(() => {
    const trimmed = lastUserMessage.trim();
    if (!trimmed) {
      setItems([]);
      return;
    }

    const controller = new AbortController();
    // Defer fetch by a tick so the assistant's enter-animation doesn't fight the chip mount.
    const timer = setTimeout(async () => {
      try {
        // Overfetch top_k so the dedup pass still has enough candidates to fill FOLLOW_UP_LIMIT.
        const res = await fetchSuggestions(trimmed, 8, null, controller.signal);
        // Build a single skip-set covering both the latest question and everything asked earlier.
        const skipSet = new Set<string>([trimmed.toLowerCase()]);
        for (const q of priorQuestions) skipSet.add(q.toLowerCase().trim());
        const filtered = res.suggestions
          .filter((s) => !skipSet.has(s.question.toLowerCase().trim()))
          .slice(0, FOLLOW_UP_LIMIT);
        setItems(filtered);
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          console.warn("[follow-up] fetch failed:", err);
        }
      }
    }, 120);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
    // priorQuestions changes per turn — refetch so previously-asked items drop out of the chip set.
  }, [lastUserMessage, priorQuestions]);

  if (items.length === 0) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.15, ease: "easeOut" }}
      className="flex flex-wrap items-center gap-2"
    >
      <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">
        Try also
      </span>
      {items.map((item, i) => (
        <motion.button
          key={item.qa_id}
          type="button"
          disabled={disabled}
          onClick={() => onSelect(item.question)}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.25, delay: 0.2 + i * 0.05, ease: "easeOut" }}
          className="cursor-pointer rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs text-slate-700 shadow-sm transition-all hover:border-emerald-300 hover:bg-emerald-50/60 hover:text-emerald-700 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-50"
        >
          {item.question}
        </motion.button>
      ))}
    </motion.div>
  );
}
