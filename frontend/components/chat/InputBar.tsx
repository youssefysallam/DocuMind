"use client";

import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { PaperPlaneRight } from "@phosphor-icons/react";
import { AnimatePresence } from "framer-motion";
import { fetchFeatured, fetchSuggestions } from "@/lib/api";
import type { FeaturedItem, FeaturedResponse, SourceFilter, Suggestion } from "@/lib/types";
import SuggestBox from "./SuggestBox";

const PLACEHOLDERS = [
  "What is the Sustainable Solutions Lab?",
  "What is C3I and who funds it?",
  "Who leads SSL?",
  "Compare SSL's East Boston work with the harbor barrier study.",
  "Has SSL published research on nuclear energy?",
];

// Wait this long after the last keystroke before hitting /api/suggest.
const SUGGEST_DEBOUNCE_MS = 250;

// Require at least this many non-whitespace chars before fetching suggestions.
const SUGGEST_MIN_CHARS = 2;

interface InputBarProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  showTypewriter?: boolean;
  // Lowercased+trimmed user messages from this conversation — drop them from featured + search results.
  priorQuestions?: string[];
}

export default function InputBar({
  onSend,
  disabled = false,
  showTypewriter = false,
  priorQuestions = [],
}: InputBarProps) {
  const [value, setValue] = useState("");
  const [placeholder, setPlaceholder] = useState(PLACEHOLDERS[0]);

  // Search-mode state — driven by the debounced /api/suggest fetch.
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  // Track the keyword tied to the last successful fetch — keeps the highlight in sync when typing outraces the debounce.
  const [matchedKeyword, setMatchedKeyword] = useState("");
  // Active source-type filter — null means "all sources allowed".
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>(null);
  // True when the strict pass returned nothing and the dropdown is showing did-you-mean fallbacks.
  const [isFallback, setIsFallback] = useState(false);
  // True between debounce-fire and fetch-resolve — drives the dropdown skeleton state.
  const [isLoading, setIsLoading] = useState(false);

  // Featured-mode state — fetched once on mount, then filtered locally by category pill.
  const [featured, setFeatured] = useState<FeaturedResponse | null>(null);
  const [activeCategory, setActiveCategory] = useState<string | null>(null);

  // Shared dropdown state.
  const [mode, setMode] = useState<"search" | "featured">("featured");
  const [activeIndex, setActiveIndex] = useState(-1);
  const [isOpen, setIsOpen] = useState(false);
  const [isFocused, setIsFocused] = useState(false);
  // Block the featured dropdown from auto-opening on mount-time autofocus — wait for a real gesture.
  const [userTouched, setUserTouched] = useState(false);

  const inputRef = useRef<HTMLTextAreaElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  // Suppress one debounce-effect cycle after submit — prevents featured from blinking back on cleared input.
  const suppressFeaturedAfterSubmitRef = useRef(false);

  // Pull curated featured payload once — backend caches the result, so this is cheap on subsequent mounts.
  useEffect(() => {
    let cancelled = false;
    fetchFeatured()
      .then((res) => {
        if (!cancelled) setFeatured(res);
      })
      .catch((err) => {
        // Featured is a nice-to-have — failures should not break the chat input.
        console.warn("[featured] fetch failed:", err);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Focus the textarea on mount — chat UIs expect an immediately-typable input.
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Cycle placeholder text — runs on the empty state only.
  useEffect(() => {
    // Swap to a static follow-up prompt once the conversation has started.
    if (!showTypewriter) {
      setPlaceholder("Ask a follow-up…");
      return;
    }

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

  // Lowercased+trimmed asked-set — recompute once per turn, reuse across both featured and search dedup paths.
  const askedSet = useMemo(() => {
    const s = new Set<string>();
    for (const q of priorQuestions) s.add(q.toLowerCase().trim());
    return s;
  }, [priorQuestions]);

  // Available groups — drop any group whose items are all already asked, so empty pills never render.
  const availableGroups = useMemo(() => {
    if (!featured) return [];
    const isUnasked = (item: FeaturedItem) => !askedSet.has(item.question.toLowerCase().trim());
    return featured.groups.filter((g) => g.items.some(isUnasked));
  }, [featured, askedSet]);

  // Derive featured items from the cached payload — either spotlight set (one per group) or full filtered group.
  // Drop anything the user already asked so the dropdown surfaces only fresh suggestions.
  const filteredFeaturedItems = useMemo<FeaturedItem[]>(() => {
    const isUnasked = (item: FeaturedItem) => !askedSet.has(item.question.toLowerCase().trim());

    if (activeCategory === null) {
      // Spotlight set — first unasked item from each available group in display order.
      return availableGroups
        .map((g) => g.items.find(isUnasked))
        .filter((x): x is FeaturedItem => Boolean(x));
    }
    const group = availableGroups.find((g) => g.intent === activeCategory);
    return group ? group.items.filter(isUnasked) : [];
  }, [availableGroups, activeCategory, askedSet]);

  // Reset stale category when the previously selected group becomes empty (every item asked).
  useEffect(() => {
    if (activeCategory && !availableGroups.some((g) => g.intent === activeCategory)) {
      setActiveCategory(null);
    }
  }, [availableGroups, activeCategory]);

  // Search results echoing previously-asked questions add noise — strip them before they hit the dropdown.
  const visibleSuggestions = useMemo<Suggestion[]>(
    () => suggestions.filter((s) => !askedSet.has(s.question.toLowerCase().trim())),
    [suggestions, askedSet]
  );

  // Debounce keyword → /api/suggest. Abort stale requests as new keystrokes arrive.
  useEffect(() => {
    const trimmed = value.trim();
    if (trimmed.length < SUGGEST_MIN_CHARS) {
      setSuggestions([]);
      // Consume the post-submit suppression — keep the dropdown closed for one cycle.
      if (suppressFeaturedAfterSubmitRef.current) {
        suppressFeaturedAfterSubmitRef.current = false;
        setIsOpen(false);
        setActiveIndex(-1);
        return;
      }
      // Below the search threshold — fall back to featured only after the user has actually touched the input.
      if (userTouched && isFocused && filteredFeaturedItems.length > 0) {
        setMode("featured");
        setIsOpen(true);
      } else {
        setIsOpen(false);
      }
      setActiveIndex(-1);
      return;
    }

    // Open the dropdown immediately into a loading-skeleton state — perceived responsiveness.
    setMode("search");
    setIsLoading(true);
    setIsOpen(true);
    // Clear stale state from prior queries — keeps the loading skeleton from inheriting old labels and rows.
    setSuggestions([]);
    setIsFallback(false);

    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const res = await fetchSuggestions(trimmed, 5, sourceFilter, controller.signal);
        setSuggestions(res.suggestions);
        setIsFallback(Boolean(res.is_fallback));
        setMatchedKeyword(trimmed);
        setIsLoading(false);
        // Count what will actually render after asked-question dedup — avoids leaving isOpen=true with nothing visible.
        const visibleCount = res.suggestions.filter(
          (s) => !askedSet.has(s.question.toLowerCase().trim())
        ).length;
        setIsOpen(visibleCount > 0);
        setActiveIndex(-1);
      } catch (err) {
        // Swallow AbortError — a newer keystroke has already scheduled the next fetch.
        if ((err as Error).name !== "AbortError") {
          console.error("[suggest] fetch failed:", err);
          setIsLoading(false);
        }
      }
    }, SUGGEST_DEBOUNCE_MS);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [value, isFocused, filteredFeaturedItems.length, userTouched, sourceFilter]);

  // Track focus only — opening the dropdown is deferred to a real user gesture.
  const handleFocus = useCallback(() => setIsFocused(true), []);
  const handleBlur = useCallback(() => setIsFocused(false), []);

  // Pointer click on the textarea is the explicit gesture that unlocks the featured dropdown.
  const handleTextareaMouseDown = useCallback(() => {
    if (!userTouched) setUserTouched(true);
    if (value.trim().length < SUGGEST_MIN_CHARS && filteredFeaturedItems.length > 0) {
      setMode("featured");
      setIsOpen(true);
      setActiveIndex(-1);
    }
  }, [userTouched, value, filteredFeaturedItems.length]);

  // Close the dropdown on outside clicks — scope the listener to while the box is open.
  useEffect(() => {
    if (!isOpen) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setIsOpen(false);
        setIsFocused(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isOpen]);

  // Reset highlight when the rendered list flips so arrow-keys never index a stale row.
  // sourceFilter narrows search results too — without it, activeIndex can point past the new shorter list.
  useEffect(() => {
    setActiveIndex(-1);
  }, [mode, activeCategory, sourceFilter]);

  // Central submit path — clear input, close dropdown, forward to the parent handler.
  const submitValue = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || disabled) return;
      // Arm the suppression so the value-cleared debounce cycle doesn't reopen featured.
      suppressFeaturedAfterSubmitRef.current = true;
      onSend(trimmed);
      setValue("");
      setSuggestions([]);
      setIsOpen(false);
      setActiveIndex(-1);
    },
    [disabled, onSend]
  );

  const handleSubmit = () => submitValue(value);

  const handleSelectSuggestion = useCallback(
    (suggestion: Suggestion) => submitValue(suggestion.question),
    [submitValue]
  );

  const handleSelectFeatured = useCallback(
    (item: FeaturedItem) => submitValue(item.question),
    [submitValue]
  );

  // Pick the active list based on mode — keyboard nav and Enter use this.
  const activeListLength =
    mode === "search" ? visibleSuggestions.length : filteredFeaturedItems.length;

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Any keystroke counts as a real interaction — unlocks the featured fallback for later empties.
    if (!userTouched) setUserTouched(true);

    // Escape always dismisses the dropdown when it is open — even if it is only showing loading skeletons.
    if (isOpen && e.key === "Escape") {
      e.preventDefault();
      setIsOpen(false);
      return;
    }

    // Route arrow/enter through the dropdown first when items are present.
    if (isOpen && activeListLength > 0) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setActiveIndex((i) => (i + 1) % activeListLength);
        return;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setActiveIndex((i) => (i <= 0 ? activeListLength - 1 : i - 1));
        return;
      }
      // Guard against a stale activeIndex pointing past the active list length — prevents an undefined crash.
      if (e.key === "Enter" && !e.shiftKey && activeIndex >= 0 && activeIndex < activeListLength) {
        e.preventDefault();
        if (mode === "search") {
          handleSelectSuggestion(visibleSuggestions[activeIndex]);
        } else {
          handleSelectFeatured(filteredFeaturedItems[activeIndex]);
        }
        return;
      }
    }

    // Fall through to raw submit when dropdown is closed or no item is active.
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="border-t border-slate-200/50 bg-[#f9fafb] px-4 py-3">
      <div
        ref={containerRef}
        className="mx-auto flex max-w-3xl items-center gap-3"
      >
        {/* Wrap textarea in a relative box so SuggestBox anchors to its width, not the full row. */}
        <div className="relative flex-1">
          <AnimatePresence mode="wait">
            {isOpen && mode === "search" && (visibleSuggestions.length > 0 || isLoading) && (
              <SuggestBox
                key="search-box"
                mode="search"
                suggestions={visibleSuggestions}
                activeIndex={activeIndex}
                keyword={matchedKeyword}
                sourceFilter={sourceFilter}
                isFallback={isFallback}
                isLoading={isLoading}
                onSelectSuggestion={handleSelectSuggestion}
                onSourceChange={setSourceFilter}
                onHover={setActiveIndex}
              />
            )}
            {isOpen && mode === "featured" && filteredFeaturedItems.length > 0 && featured && (
              <SuggestBox
                key="featured-box"
                mode="featured"
                groups={availableGroups}
                filteredItems={filteredFeaturedItems}
                activeCategory={activeCategory}
                activeIndex={activeIndex}
                onCategoryChange={setActiveCategory}
                onSelectFeatured={handleSelectFeatured}
                onHover={setActiveIndex}
              />
            )}
          </AnimatePresence>

          <textarea
            ref={inputRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={handleFocus}
            onBlur={handleBlur}
            onMouseDown={handleTextareaMouseDown}
            placeholder={placeholder}
            disabled={disabled}
            rows={1}
            className="block h-11 w-full resize-none rounded-xl border border-slate-200/50 bg-white px-4 py-2.5 text-sm leading-5 text-zinc-900 shadow-[inset_0_1px_2px_rgba(0,0,0,0.05)] outline-none transition-shadow placeholder:text-slate-400 focus:border-slate-300 focus:shadow-[inset_0_1px_2px_rgba(0,0,0,0.05),0_0_0_2px_rgba(5,150,105,0.15)] disabled:opacity-50"
          />
        </div>
        <button
          onClick={handleSubmit}
          disabled={disabled || !value.trim()}
          className="flex h-11 w-11 cursor-pointer items-center justify-center rounded-xl bg-emerald-accent text-white transition-all hover:bg-emerald-accent-hover active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-40 disabled:active:scale-100"
        >
          <PaperPlaneRight size={18} weight="bold" />
        </button>
      </div>
    </div>
  );
}
