"use client";

import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { ChatMessage } from "@/lib/types";

interface MessageListProps {
  messages: ChatMessage[];
  loading?: boolean;
}

const messageVariants = {
  hidden: { opacity: 0, y: 10 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { type: "spring", stiffness: 100, damping: 20 },
  },
};

function SkeletonMessage() {
  return (
    <div className="flex justify-start">
      <div className="max-w-[75%] rounded-2xl rounded-bl-md bg-white px-4 py-3 shadow-sm">
        <div className="space-y-2">
          <div className="h-3 w-64 animate-pulse rounded bg-slate-200" />
          <div className="h-3 w-48 animate-pulse rounded bg-slate-200" />
          <div className="h-3 w-56 animate-pulse rounded bg-slate-200" />
        </div>
      </div>
    </div>
  );
}

export default function MessageList({ messages, loading }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, loading]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6">
      <div className="mx-auto max-w-3xl space-y-4">
        <AnimatePresence initial={false}>
          {messages.map((msg, i) => (
            <motion.div
              key={i}
              className={`flex ${
                msg.role === "user" ? "justify-end" : "justify-start"
              }`}
              variants={messageVariants}
              initial="hidden"
              animate="visible"
              layout
            >
              <div
                className={`max-w-[75%] whitespace-pre-wrap text-sm leading-relaxed ${
                  msg.role === "user"
                    ? "rounded-2xl rounded-br-md bg-emerald-accent px-4 py-3 text-white"
                    : "rounded-2xl rounded-bl-md bg-white px-4 py-3 text-slate-700 shadow-sm"
                }`}
              >
                {msg.content}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.2 }}
          >
            <SkeletonMessage />
          </motion.div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
