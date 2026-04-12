"use client";

import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import Link from "next/link";

const NAV_LINKS = [
  { label: "Evaluation", href: "#evaluation" },
  { label: "Architecture", href: "#architecture" },
];

const IDLE_TIMEOUT = 1500;

export default function Navbar() {
  const [expanded, setExpanded] = useState(true);
  const expandedRef = useRef(true);
  const idleTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const handleScroll = () => {
      if (expandedRef.current) {
        expandedRef.current = false;
        setExpanded(false);
      }

      if (idleTimer.current) clearTimeout(idleTimer.current);
      idleTimer.current = setTimeout(() => {
        expandedRef.current = true;
        setExpanded(true);
      }, IDLE_TIMEOUT);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", handleScroll);
      if (idleTimer.current) clearTimeout(idleTimer.current);
    };
  }, []);

  return (
    <motion.nav
      className={`fixed left-0 right-0 z-50 flex justify-center pointer-events-none ${
        expanded ? "top-0" : "top-4"
      }`}
      layout
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
    >
      <motion.div
        className={`pointer-events-auto flex items-center backdrop-blur-2xl backdrop-saturate-[1.8] ${
          expanded
            ? "w-full max-w-[900px] justify-between rounded-b-2xl border border-t-0 border-slate-200/40 bg-slate-100/50 px-8 py-4 shadow-[0_8px_40px_rgba(0,0,0,0.07),0_1.5px_0_rgba(255,255,255,0.9)_inset,0_-0.5px_0_rgba(0,0,0,0.04)_inset]"
            : "gap-6 rounded-2xl border border-slate-200/40 bg-slate-100/50 px-5 py-2.5 shadow-[0_8px_40px_rgba(0,0,0,0.07),0_1.5px_0_rgba(255,255,255,0.9)_inset,0_-0.5px_0_rgba(0,0,0,0.04)_inset]"
        }`}
        layout
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
      >
        <Link
          href="/"
          className="text-lg font-semibold tracking-tight text-zinc-900"
        >
          InfoWeave
        </Link>

        <div className={`hidden items-center md:flex ${expanded ? "gap-8" : "gap-5"}`}>
          {NAV_LINKS.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className={`text-slate-500 transition-colors hover:text-zinc-900 ${
                expanded ? "text-sm" : "text-xs"
              }`}
            >
              {link.label}
            </a>
          ))}
          <Link
            href="/chat"
            className={`rounded-lg bg-emerald-accent font-medium text-white transition-transform hover:bg-emerald-accent-hover active:scale-[0.98] ${
              expanded ? "px-4 py-2 text-sm" : "px-3 py-1.5 text-xs"
            }`}
          >
            Launch Chat
          </Link>
        </div>
      </motion.div>
    </motion.nav>
  );
}
