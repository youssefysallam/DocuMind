"use client";

import { useRef } from "react";
import { motion, useInView } from "framer-motion";
import AnimatedCounter from "./AnimatedCounter";

interface BentoTile {
  label: string;
  description: string;
  value: number;
  suffix: string;
  prefix?: string;
  decimals?: number;
  span?: string;
}

const TILES: BentoTile[] = [
  {
    label: "Hallucination reduction",
    description: "Cut in half from V1 to V2",
    value: 50,
    suffix: "%",
    prefix: "-",
    decimals: 0,
    span: "md:col-span-2",
  },
  {
    label: "Coverage",
    description: "Near-complete corpus coverage",
    value: 96.2,
    suffix: "%",
    decimals: 1,
  },
  {
    label: "Answer accuracy",
    description: "On answerable questions",
    value: 75.0,
    suffix: "%",
    decimals: 1,
  },
  {
    label: "Correct refusal",
    description: "Knows when it doesn't know",
    value: 90.9,
    suffix: "%",
    decimals: 1,
  },
  {
    label: "Hallucination rate",
    description: "Down from 24% in V1",
    value: 12.0,
    suffix: "%",
    decimals: 1,
  },
];

const containerVariants = {
  hidden: {},
  visible: {
    transition: { staggerChildren: 0.12 },
  },
};

const tileVariants = {
  hidden: { y: 30, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: { type: "spring", stiffness: 100, damping: 20 },
  },
};

export default function FeaturesBento() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section id="evaluation" className="py-24 md:py-32">
      <div className="mx-auto max-w-[1400px] px-6">
        <h2 className="text-3xl font-bold tracking-tight text-zinc-900 md:text-4xl">
          Evaluation
        </h2>
        <p className="mt-3 max-w-[65ch] text-base leading-relaxed text-slate-500">
          Benchmarked against 104 questions across 6 categories. V2 halves
          hallucination, improves accuracy, and nearly doubles correct refusal
          rate compared to V1.
        </p>

        <motion.div
          ref={ref}
          className="mt-12 grid grid-cols-1 gap-5 md:grid-cols-3"
          variants={containerVariants}
          initial="hidden"
          animate={isInView ? "visible" : "hidden"}
        >
          {TILES.map((tile) => (
            <motion.div
              key={tile.label}
              className={`rounded-2xl border border-slate-200/50 bg-white p-8 shadow-[0_20px_40px_-15px_rgba(0,0,0,0.05)] ${
                tile.span ?? ""
              }`}
              variants={tileVariants}
            >
              <div className="text-4xl font-bold tracking-tight text-zinc-900 md:text-5xl">
                <AnimatedCounter
                  target={tile.value}
                  suffix={tile.suffix}
                  prefix={tile.prefix}
                  decimals={tile.decimals}
                />
              </div>
              <p className="mt-4 text-sm font-medium text-zinc-900">
                {tile.label}
              </p>
              <p className="mt-1 text-sm text-slate-500">{tile.description}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
