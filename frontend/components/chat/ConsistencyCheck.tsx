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

  return (
    <div className="border-t border-slate-200/50">
      <button
        className="flex w-full items-center justify-between px-4 py-3 text-sm font-medium text-zinc-900 transition-colors hover:bg-slate-50"
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
                        ? "Consistent"
                        : "Issues detected"}
                    </span>
                    <span className="font-[family-name:var(--font-geist-mono)] text-slate-400">
                      {(consistency.confidence * 100).toFixed(0)}%
                    </span>
                  </div>

                  {consistency.unsupported_claims &&
                    consistency.unsupported_claims.length > 0 && (
                      <div>
                        <span className="text-slate-400">
                          Unsupported claims
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
