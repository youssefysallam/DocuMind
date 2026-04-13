"use client";

import { useRef } from "react";
import { motion, useInView } from "framer-motion";

interface PipelineNode {
  label: string;
  detail: string;
  accent?: boolean;
}

const PIPELINE: PipelineNode[] = [
  { label: "Query", detail: "User question" },
  { label: "Intent Classification", detail: "LLM-enhanced", accent: true },
  { label: "Query Transform", detail: "HyDE / Sub-query", accent: true },
  { label: "Dense Retrieval", detail: "all-MiniLM-L6-v2" },
  { label: "Sparse Retrieval", detail: "BM25" },
  { label: "Reranking", detail: "ms-marco-MiniLM-L-6-v2" },
  { label: "Diversity Selection", detail: "Top-5 final", accent: true },
  { label: "Generation", detail: "LLM" },
  { label: "Consistency Check", detail: "Answer-evidence", accent: true },
];

const nodeVariants = {
  hidden: { scale: 0.8, opacity: 0 },
  visible: {
    scale: 1,
    opacity: 1,
    transition: { type: "spring", stiffness: 100, damping: 20 },
  },
};

const lineVariants = {
  hidden: { pathLength: 0, opacity: 0 },
  visible: {
    pathLength: 1,
    opacity: 1,
    transition: { duration: 0.4, ease: "easeOut" },
  },
};

export default function PipelineShowcase() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section id="architecture" className="py-24 md:py-32">
      <div className="mx-auto max-w-[1400px] px-6">
        <h2 className="text-3xl font-bold tracking-tight text-zinc-900 md:text-4xl">
          Architecture
        </h2>
        <p className="mt-3 max-w-[65ch] text-base leading-relaxed text-slate-500">
          A multi-stage retrieval-augmented generation pipeline with intent
          classification, query transformation, hybrid retrieval, and
          answer-evidence consistency verification.
        </p>

        {/* Desktop: horizontal flow */}
        <div ref={ref} className="mt-16 hidden md:block">
          <motion.div
            className="flex items-start gap-0"
            initial="hidden"
            animate={isInView ? "visible" : "hidden"}
            transition={{ staggerChildren: 0.1 }}
          >
            {PIPELINE.map((node, i) => (
              <div key={node.label} className="flex items-center">
                <motion.div
                  className="flex flex-col items-center"
                  variants={nodeVariants}
                >
                  <div
                    className={`flex h-14 w-14 items-center justify-center rounded-xl border ${
                      node.accent
                        ? "border-emerald-accent/30 bg-emerald-accent-light"
                        : "border-slate-200 bg-white"
                    } shadow-[0_4px_12px_-4px_rgba(0,0,0,0.08)]`}
                  >
                    <span className="text-xs font-semibold text-zinc-700">
                      {i + 1}
                    </span>
                  </div>
                  <p className="mt-3 max-w-[100px] text-center text-xs font-medium text-zinc-900">
                    {node.label}
                  </p>
                  <p className="mt-0.5 max-w-[100px] text-center text-[11px] text-slate-400 font-[family-name:var(--font-geist-mono)]">
                    {node.detail}
                  </p>
                </motion.div>

                {i < PIPELINE.length - 1 && (
                  <motion.svg
                    width="40"
                    height="2"
                    className="mx-1 mt-7 flex-shrink-0"
                    variants={lineVariants}
                  >
                    <motion.line
                      x1="0"
                      y1="1"
                      x2="40"
                      y2="1"
                      stroke="#cbd5e1"
                      strokeWidth="2"
                      strokeLinecap="round"
                    />
                  </motion.svg>
                )}
              </div>
            ))}
          </motion.div>

          {/* Feedback loop arrow label */}
          <p className="mt-8 text-xs text-slate-400 italic">
            Session Memory and Coreference Resolution feed back from Generation
            to Query Transform for multi-turn dialogue.
          </p>
        </div>

        {/* Mobile: vertical flow */}
        <div className="mt-12 md:hidden">
          <motion.div
            className="flex flex-col gap-4"
            initial="hidden"
            animate={isInView ? "visible" : "hidden"}
            transition={{ staggerChildren: 0.08 }}
          >
            {PIPELINE.map((node, i) => (
              <motion.div
                key={node.label}
                className="flex items-center gap-4"
                variants={nodeVariants}
              >
                <div
                  className={`flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg border ${
                    node.accent
                      ? "border-emerald-accent/30 bg-emerald-accent-light"
                      : "border-slate-200 bg-white"
                  } shadow-sm`}
                >
                  <span className="text-xs font-semibold text-zinc-700">
                    {i + 1}
                  </span>
                </div>
                <div>
                  <p className="text-sm font-medium text-zinc-900">
                    {node.label}
                  </p>
                  <p className="text-xs text-slate-400 font-[family-name:var(--font-geist-mono)]">
                    {node.detail}
                  </p>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>
    </section>
  );
}
