"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CaretDown, CheckCircle, Warning } from "@phosphor-icons/react";
import type { ConsistencyResult } from "@/lib/types";

interface ConsistencyCheckProps {
  consistency: ConsistencyResult | null;
}

export default function ConsistencyCheck({
  consistency,
}: ConsistencyCheckProps) {
  const [open, setOpen] = useState(false);

  const confidencePercent = consistency
    ? Math.round(consistency.confidence * 100)
    : null;

  const confidenceColor =
    confidencePercent === null
      ? "text-slate-400"
      : confidencePercent >= 90
        ? "text-emerald-700"
        : confidencePercent >= 80
          ? "text-emerald-500"
          : confidencePercent >= 70
            ? "text-yellow-500"
            : confidencePercent >= 60
              ? "text-orange-500"
              : "text-red-600";

  return (
    <div className="border-t border-slate-200/50">
      <button
        className="flex w-full cursor-pointer items-center justify-between px-4 py-3 text-sm font-medium text-zinc-900 transition-colors hover:bg-slate-50"
        onClick={() => setOpen(!open)}
      >
        <span>Consistency Check</span>
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
              {!consistency ? (
                <p className="py-4 text-center text-sm text-slate-400">
                  Consistency check results will appear here
                </p>
              ) : (
                <div className="space-y-2 text-xs">
                  <p className="text-sm font-semibold text-zinc-900">
                    Confidence in result{" "}
                    <span
                      className={`text-sm font-[family-name:var(--font-geist-mono)] ${confidenceColor}`}
                    >
                      {confidencePercent}%
                    </span>
                  </p>

                  <div className="flex items-center gap-2">
                    {consistency.is_consistent ? (
                      <CheckCircle
                        size={16}
                        weight="fill"
                        className="text-emerald-accent"
                      />
                    ) : (
                      <Warning
                        size={16}
                        weight="fill"
                        className="text-amber-500"
                      />
                    )}
                    <span className="font-medium text-zinc-900">
                      {consistency.is_consistent
                        ? "Evidence aligned"
                        : "Some claims need review"}
                    </span>
                  </div>

                  {consistency.unsupported_claims &&
                    consistency.unsupported_claims.length > 0 && (
                      <div>
                        <span className="text-slate-400">
                          Claims that may go beyond the retrieved evidence
                        </span>
                        <ul className="mt-1 space-y-1">
                          {consistency.unsupported_claims.map((claim, i) => (
                            <li key={i} className="text-slate-600">
                              {claim}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                  {consistency.explanation && (
                    <p className="italic text-slate-500">
                      {consistency.explanation}
                    </p>
                  )}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
