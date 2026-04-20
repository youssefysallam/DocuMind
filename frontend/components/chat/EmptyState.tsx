"use client";

import { motion } from "framer-motion";

const EXAMPLES = [
  "What is the Sustainable Solutions Lab?",
  "What is C3I and who funds it?",
  "Who leads SSL?",
  "Compare SSL's East Boston work with the harbor barrier study.",
  "Has SSL published research on nuclear energy?",
];

interface EmptyStateProps {
  onSelectExample: (question: string) => void;
}

export default function EmptyState({ onSelectExample }: EmptyStateProps) {
  return (
    <div className="flex h-full flex-col items-center justify-center px-6">
      <motion.div
        className="text-center"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ type: "spring", stiffness: 100, damping: 20 }}
      >
        <h2 className="text-2xl font-bold tracking-tight text-zinc-900">
          InfoWeave
        </h2>
        <p className="mt-2 text-sm text-slate-500">
          Ask about SSL's research, projects, or publications
        </p>
      </motion.div>

      <motion.div
        className="mt-8 flex max-w-xl flex-wrap justify-center gap-2"
        initial="hidden"
        animate="visible"
        variants={{ visible: { transition: { staggerChildren: 0.06 } } }}
      >
        {EXAMPLES.map((example) => (
          <motion.button
            key={example}
            className="cursor-pointer rounded-lg border border-slate-200/50 bg-white px-4 py-2 text-sm text-slate-600 shadow-sm transition-all hover:border-slate-300 hover:text-zinc-900 hover:shadow-md active:scale-[0.98]"
            onClick={() => onSelectExample(example)}
            variants={{
              hidden: { opacity: 0, y: 10 },
              visible: {
                opacity: 1,
                y: 0,
                transition: { type: "spring", stiffness: 100, damping: 20 },
              },
            }}
          >
            {example}
          </motion.button>
        ))}
      </motion.div>
    </div>
  );
}
