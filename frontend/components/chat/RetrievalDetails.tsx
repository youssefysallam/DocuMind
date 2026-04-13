"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CaretDown } from "@phosphor-icons/react";
import type { RetrievalLog } from "@/lib/types";

interface RetrievalDetailsProps {
  log: RetrievalLog | null;
}

export default function RetrievalDetails({ log }: RetrievalDetailsProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="border-t border-slate-200/50">
      <button
        className="flex w-full items-center justify-between px-4 py-3 text-sm font-medium text-zinc-900 transition-colors hover:bg-slate-50"
        onClick={() => setOpen(!open)}
      >
        <span>Retrieval Details</span>
        <CaretDown
          size={14}
          className={`text-slate-400 transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4">
              {!log ? (
                <p className="py-4 text-center text-sm text-slate-400">
                  Retrieval details will appear here
                </p>
              ) : (
                <div className="space-y-3 text-xs">
                  {log.intent && (
                    <div>
                      <span className="text-slate-400">Intent</span>
                      <span className="ml-2 rounded bg-slate-100 px-2 py-0.5 font-[family-name:var(--font-geist-mono)] text-zinc-700">
                        {log.intent}
                      </span>
                    </div>
                  )}
                  {log.strategy && (
                    <div>
                      <span className="text-slate-400">Strategy</span>
                      <span className="ml-2 rounded bg-slate-100 px-2 py-0.5 font-[family-name:var(--font-geist-mono)] text-zinc-700">
                        {log.strategy}
                      </span>
                    </div>
                  )}
                  {log.resolved_query &&
                    log.original_query &&
                    log.resolved_query !== log.original_query && (
                      <div>
                        <span className="text-slate-400">Resolved query</span>
                        <p className="mt-1 text-slate-600 italic">
                          {log.resolved_query}
                        </p>
                      </div>
                    )}
                  {log.sub_queries && log.sub_queries.length > 1 && (
                    <div>
                      <span className="text-slate-400">Sub-queries</span>
                      <ul className="mt-1 space-y-0.5">
                        {log.sub_queries.map((sq, i) => (
                          <li key={i} className="text-slate-600">
                            {sq}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  <div className="flex flex-wrap gap-2">
                    {log.hyde_used && (
                      <span className="rounded-full bg-emerald-accent-light px-2 py-0.5 text-[11px] font-medium text-emerald-accent">
                        HyDE
                      </span>
                    )}
                    {log.multihop?.triggered && (
                      <span className="rounded-full bg-emerald-accent-light px-2 py-0.5 text-[11px] font-medium text-emerald-accent">
                        Multi-hop +{log.multihop.supplements_added}
                      </span>
                    )}
                    {log.qa_neg_used && (
                      <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[11px] font-medium text-amber-700">
                        No-evidence signal
                      </span>
                    )}
                  </div>
                  <p className="text-slate-400">
                    Retrieved: {log.final_raw ?? "?"} corpus, {log.final_qa ?? "?"} QA
                  </p>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
