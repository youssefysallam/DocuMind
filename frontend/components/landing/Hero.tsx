"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { CaretDown } from "@phosphor-icons/react";

const spring = { type: "spring" as const, stiffness: 100, damping: 20 };

export default function Hero() {
  return (
    <section className="relative min-h-[100dvh] flex items-center overflow-hidden">
      {/* Video background */}
      <video
        autoPlay
        muted
        loop
        playsInline
        className="absolute inset-0 h-full w-full object-cover"
        poster=""
      >
        <source src="/video/hero.mp4" type="video/mp4" />
      </video>

      {/* Gradient fade from video white to page off-white */}
      <div className="absolute inset-x-0 bottom-0 h-40 bg-gradient-to-b from-transparent to-[#f9fafb]" />

      {/* Text overlay -- left aligned */}
      <div className="relative z-10 mx-auto w-full max-w-[1400px] px-6">
        <motion.h1
          className="text-4xl font-bold tracking-tighter leading-none text-zinc-900 md:text-6xl"
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ ...spring, delay: 0.1 }}
        >
          InfoWeave
        </motion.h1>

        <motion.p
          className="mt-4 max-w-[65ch] text-base leading-relaxed text-slate-500"
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ ...spring, delay: 0.25 }}
        >
          Sustainable Solutions Lab Research Assistant
        </motion.p>

        <motion.div
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ ...spring, delay: 0.4 }}
        >
          <Link
            href="/chat"
            className="mt-8 inline-block rounded-lg bg-emerald-accent px-6 py-3 text-sm font-medium text-white transition-transform hover:bg-emerald-accent-hover active:scale-[0.98]"
          >
            Start a conversation
          </Link>
        </motion.div>
      </div>

      {/* Scroll indicator */}
      <motion.div
        className="absolute bottom-8 left-1/2 -translate-x-1/2"
        animate={{ y: [0, 8, 0] }}
        transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
      >
        <CaretDown size={24} weight="bold" className="text-slate-400" />
      </motion.div>
    </section>
  );
}
