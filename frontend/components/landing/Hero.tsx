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
  const videoY = useTransform(scrollYProgress, [0, 1], [0, 80]);
  const videoOpacity = useTransform(scrollYProgress, [0, 0.8], [1, 0]);
  const textY = useTransform(scrollYProgress, [0, 0.5], [0, -60]);
  const textOpacity = useTransform(scrollYProgress, [0, 0.4], [1, 0]);

  return (
    <section
      ref={sectionRef}
      className="relative min-h-[100dvh] overflow-hidden bg-[#f9fafb]"
    >
      <div className="mx-auto flex min-h-[100dvh] max-w-[1400px] items-center px-6">
        {/* Left -- text content */}
        <motion.div
          className="relative z-10 w-full py-20 md:w-1/2 md:pr-12"
          style={{ y: textY, opacity: textOpacity }}
        >
          <motion.h1
            className="text-4xl font-bold tracking-tighter leading-none text-zinc-900 md:text-6xl"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ type: "spring", stiffness: 100, damping: 20, delay: 0.1 }}
          >
            InfoWeave
          </motion.h1>

          <motion.p
            className="mt-4 max-w-[50ch] text-lg leading-relaxed text-slate-500"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ type: "spring", stiffness: 100, damping: 20, delay: 0.25 }}
          >
            Sustainable Solutions Lab Research Assistant
          </motion.p>

          <motion.p
            className="mt-3 max-w-[50ch] text-sm leading-relaxed text-slate-400"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ type: "spring", stiffness: 100, damping: 20, delay: 0.35 }}
          >
            Multi-turn dialogue with source attribution, powered by a hybrid
            retrieval-augmented generation pipeline.
          </motion.p>

          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ type: "spring", stiffness: 100, damping: 20, delay: 0.45 }}
          >
            <Link
              href="/chat"
              className="mt-8 inline-block rounded-lg bg-emerald-accent px-6 py-3 text-sm font-medium text-white transition-transform hover:bg-emerald-accent-hover active:scale-[0.98]"
            >
              Start a conversation
            </Link>
          </motion.div>
        </motion.div>

        {/* Right -- video panel */}
        <motion.div
          className="hidden md:block md:w-1/2"
          style={{ y: videoY, opacity: videoOpacity }}
        >
          <div className="relative">
            {/* Edge fades to blend video white into page background */}
            <div className="absolute inset-y-0 left-0 z-10 w-20 bg-gradient-to-r from-[#f9fafb] to-transparent" />
            <div className="absolute inset-y-0 right-0 z-10 w-16 bg-gradient-to-l from-[#f9fafb] to-transparent" />
            <div className="absolute inset-x-0 top-0 z-10 h-16 bg-gradient-to-b from-[#f9fafb] to-transparent" />
            <div className="absolute inset-x-0 bottom-0 z-10 h-16 bg-gradient-to-t from-[#f9fafb] to-transparent" />

            <video
              autoPlay
              muted
              loop
              playsInline
              className="w-full rounded-lg"
            >
              <source src="/video/hero.mp4" type="video/mp4" />
            </video>
          </div>
        </motion.div>
      </div>

      {/* Mobile -- video below text */}
      <motion.div
        className="relative mx-auto max-w-[1400px] px-6 pb-8 md:hidden"
        style={{ opacity: videoOpacity }}
      >
        <div className="relative">
          <div className="absolute inset-y-0 left-0 z-10 w-12 bg-gradient-to-r from-[#f9fafb] to-transparent" />
          <div className="absolute inset-y-0 right-0 z-10 w-12 bg-gradient-to-l from-[#f9fafb] to-transparent" />
          <div className="absolute inset-x-0 top-0 z-10 h-12 bg-gradient-to-b from-[#f9fafb] to-transparent" />
          <div className="absolute inset-x-0 bottom-0 z-10 h-12 bg-gradient-to-t from-[#f9fafb] to-transparent" />

          <video
            autoPlay
            muted
            loop
            playsInline
            className="w-full rounded-lg"
          >
            <source src="/video/hero.mp4" type="video/mp4" />
          </video>
        </div>
      </motion.div>

      {/* Scroll indicator */}
      <motion.div
        className="absolute bottom-8 left-1/2 z-10 -translate-x-1/2"
        animate={{ y: [0, 8, 0] }}
        transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
        style={{ opacity: textOpacity }}
      >
        <CaretDown size={24} weight="bold" className="text-slate-400" />
      </motion.div>
    </section>
  );
}
