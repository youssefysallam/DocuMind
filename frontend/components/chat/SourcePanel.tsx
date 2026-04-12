"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BookOpen, Globe, File, CaretDown } from "@phosphor-icons/react";
import type { RetrievedSource } from "@/lib/types";

const SOURCE_ICONS: Record<string, typeof BookOpen> = {
  curated_qa: BookOpen,
  website: Globe,
  pdf: File,
};

interface SourcePanelProps {
  sources: RetrievedSource[];
}

export default function SourcePanel({ sources }: SourcePanelProps) {
  const [open, setOpen] = useState(true);

  return (
    <div className="border-t border-slate-200/50">
      <button
        className="flex w-full items-center justify-between px-4 py-3 text-sm font-medium text-zinc-900 transition-colors hover:bg-slate-50"
        onClick={() => setOpen(!open)}
      >
        <span>Retrieved Sources</span>
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
              {sources.length === 0 ? (
                <p className="py-4 text-center text-sm text-slate-400">
                  Ask a question to see sources
                </p>
              ) : (
                <div className="space-y-2">
                  {sources.map((src, i) => {
                    const Icon = SOURCE_ICONS[src.source_type] ?? File;
                    const isQa = src.layer === "qa_memory";
                    return (
                      <div
                        key={i}
                        className={`rounded-lg border-l-2 bg-white px-3 py-2 ${
                          isQa ? "border-l-emerald-accent" : "border-l-slate-300"
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Icon size={14} className="text-slate-500" />
                            <span className="text-xs font-medium text-zinc-900">
                              {src.source}
                            </span>
                          </div>
                          <span className="text-[11px] text-slate-400 font-[family-name:var(--font-geist-mono)]">
                            {src.score.toFixed(3)}
                          </span>
                        </div>
                        {src.section_title && (
                          <p className="mt-0.5 text-[11px] text-slate-400">
                            {src.section_title}
                          </p>
                        )}
                        {src.chunk_text && (
                          <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-slate-500">
                            {src.chunk_text}
                          </p>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
