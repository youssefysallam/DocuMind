"use client";

import { useState, useRef, useEffect } from "react";
import { PaperPlaneRight } from "@phosphor-icons/react";

const PLACEHOLDERS = [
  "What is the Sustainable Solutions Lab?",
  "What is C3I and who funds it?",
  "Who leads SSL?",
  "Compare SSL's East Boston work with the harbor barrier study.",
  "Has SSL published research on nuclear energy?",
];

interface InputBarProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  showTypewriter?: boolean;
}

export default function InputBar({
  onSend,
  disabled = false,
  showTypewriter = false,
}: InputBarProps) {
  const [value, setValue] = useState("");
  const [placeholder, setPlaceholder] = useState(PLACEHOLDERS[0]);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Typewriter effect for placeholder cycling
  useEffect(() => {
    if (!showTypewriter) return;

    let phraseIndex = 0;
    let charIndex = 0;
    let deleting = false;
    let timeout: ReturnType<typeof setTimeout>;

    const tick = () => {
      const phrase = PLACEHOLDERS[phraseIndex];
      if (!deleting) {
        charIndex++;
        setPlaceholder(phrase.slice(0, charIndex));
        if (charIndex === phrase.length) {
          timeout = setTimeout(() => {
            deleting = true;
            tick();
          }, 2000);
          return;
        }
        timeout = setTimeout(tick, 50);
      } else {
        charIndex--;
        setPlaceholder(phrase.slice(0, charIndex));
        if (charIndex === 0) {
          deleting = false;
          phraseIndex = (phraseIndex + 1) % PLACEHOLDERS.length;
          timeout = setTimeout(tick, 300);
          return;
        }
        timeout = setTimeout(tick, 30);
      }
    };

    tick();
    return () => clearTimeout(timeout);
  }, [showTypewriter]);

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="border-t border-slate-200/50 bg-[#f9fafb] px-4 py-3">
      <div className="mx-auto flex max-w-3xl items-end gap-3">
        <textarea
          ref={inputRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          rows={1}
          className="flex-1 resize-none rounded-xl border border-slate-200/50 bg-white px-4 py-3 text-sm text-zinc-900 shadow-[inset_0_1px_2px_rgba(0,0,0,0.05)] outline-none transition-shadow placeholder:text-slate-400 focus:border-slate-300 focus:shadow-[inset_0_1px_2px_rgba(0,0,0,0.05),0_0_0_2px_rgba(5,150,105,0.15)] disabled:opacity-50"
        />
        <button
          onClick={handleSubmit}
          disabled={disabled || !value.trim()}
          className="flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-xl bg-emerald-accent text-white transition-all hover:bg-emerald-accent-hover active:scale-[0.98] disabled:opacity-40 disabled:active:scale-100"
        >
          <PaperPlaneRight size={18} weight="bold" />
        </button>
      </div>
    </div>
  );
}
