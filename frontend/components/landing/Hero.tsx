"use client";

import { useRef } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import Link from "next/link";
import { CaretDown } from "@phosphor-icons/react";

export default function Hero() {
  const sectionRef = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ["start start", "end start"],
  });

  // Scroll-driven animations
  const videoScale = useTransform(scrollYProgress, [0, 1], [1, 1.15]);
  const videoOpacity = useTransform(scrollYProgress, [0, 0.8], [1, 0]);
  const textY = useTransform(scrollYProgress, [0, 0.5], [0, -60]);
  const textOpacity = useTransform(scrollYProgress, [0, 0.4], [1, 0]);

  return (
    <section
      ref={sectionRef}
      className="relative min-h-[100dvh] flex items-end overflow-hidden"
    >
      {/* Video -- contained, centered, not cropped */}
      <motion.div
        className="absolute inset-0 flex items-center justify-center"
        style={{ scale: videoScale, opacity: videoOpacity }}
      >
        <video
          autoPlay
          muted
          loop
          playsInline
          className="h-full w-full object-contain"
        >
          <source src="/video/hero.mp4" type="video/mp4" />
        </video>
      </motion.div>

      {/* Gradient fade from video white to page off-white */}
      <div className="absolute inset-x-0 bottom-0 h-60 bg-gradient-to-b from-transparent to-[#f9fafb]" />

      {/* Text overlay -- bottom-left, with frosted backdrop for legibility */}
      <motion.div
        className="relative z-10 mx-auto w-full max-w-[1400px] px-6 pb-24"
        style={{ y: textY, opacity: textOpacity }}
      >
        <div className="inline-block rounded-2xl bg-white/70 px-8 py-6 backdrop-blur-md shadow-[0_8px_32px_-8px_rgba(0,0,0,0.08)]">
          <motion.h1
            className="text-4xl font-bold tracking-tighter leading-none text-zinc-900 md:text-6xl"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ type: "spring", stiffness: 100, damping: 20, delay: 0.1 }}
          >
            InfoWeave
          </motion.h1>

          <motion.p
            className="mt-3 max-w-[65ch] text-base leading-relaxed text-slate-600"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ type: "spring", stiffness: 100, damping: 20, delay: 0.25 }}
          >
            Sustainable Solutions Lab Research Assistant
          </motion.p>

          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ type: "spring", stiffness: 100, damping: 20, delay: 0.4 }}
          >
            <Link
              href="/chat"
              className="mt-5 inline-block rounded-lg bg-emerald-accent px-6 py-3 text-sm font-medium text-white transition-transform hover:bg-emerald-accent-hover active:scale-[0.98]"
            >
              Start a conversation
            </Link>
          </motion.div>
        </div>
      </motion.div>

      {/* Scroll indicator */}
      <motion.div
        className="absolute bottom-8 left-1/2 -translate-x-1/2 z-10"
        animate={{ y: [0, 8, 0] }}
        transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
        style={{ opacity: textOpacity }}
      >
        <CaretDown size={24} weight="bold" className="text-slate-400" />
      </motion.div>
    </section>
  );
}
