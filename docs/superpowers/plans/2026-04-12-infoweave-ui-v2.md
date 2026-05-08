# InfoWeave UI V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a high-end Next.js website for the InfoWeave RAG V2 system with a cinematic landing page and dedicated chat experience, without modifying the existing Gradio UI.

**Architecture:** A standalone Next.js 14 App Router frontend in `frontend/` communicates with a new FastAPI wrapper (`src/rag_v2/api.py`) that exposes the existing RAG pipeline over HTTP. The Gradio app remains untouched. Two independent entry points coexist on different ports.

**Tech Stack:** Next.js 14 (App Router), Tailwind CSS v4, Framer Motion, Geist + Geist Mono fonts, Phosphor Icons, FastAPI, Python uvicorn

**Spec:** `docs/superpowers/specs/2026-04-12-infoweave-ui-v2-design.md`

**CRITICAL CONSTRAINTS:**
- Do NOT modify `src/rag_v2/app.py` or any existing files
- Do NOT push to remote -- all commits are local on `ui-v2-improved` branch
- Do NOT delete any existing files

---

## File Map

### New Python files
| File | Responsibility |
|------|---------------|
| `src/rag_v2/api.py` | FastAPI server wrapping existing RAG pipeline |

### New frontend files
| File | Responsibility |
|------|---------------|
| `frontend/package.json` | Dependencies and scripts |
| `frontend/next.config.ts` | Next.js configuration (API proxy) |
| `frontend/postcss.config.mjs` | PostCSS config for Tailwind v4 |
| `frontend/tsconfig.json` | TypeScript config |
| `frontend/app/globals.css` | Tailwind imports and global styles |
| `frontend/app/layout.tsx` | Root layout with Geist fonts |
| `frontend/app/page.tsx` | Landing page (Server Component shell) |
| `frontend/app/chat/page.tsx` | Chat page (Client Component) |
| `frontend/lib/api.ts` | FastAPI client functions |
| `frontend/lib/types.ts` | Shared TypeScript types |
| `frontend/components/landing/Navbar.tsx` | Fixed nav with glassmorphism |
| `frontend/components/landing/Hero.tsx` | Full-bleed video hero |
| `frontend/components/landing/FeaturesBento.tsx` | Bento grid metrics |
| `frontend/components/landing/AnimatedCounter.tsx` | Counter animation (isolated client) |
| `frontend/components/landing/PipelineShowcase.tsx` | Animated architecture diagram |
| `frontend/components/landing/Footer.tsx` | Minimal footer |
| `frontend/components/chat/ChatInterface.tsx` | Main chat layout (split view) |
| `frontend/components/chat/MessageList.tsx` | Conversation thread |
| `frontend/components/chat/InputBar.tsx` | Message input with typewriter |
| `frontend/components/chat/SourcePanel.tsx` | Retrieved sources sidebar |
| `frontend/components/chat/RetrievalDetails.tsx` | Intent/strategy display |
| `frontend/components/chat/ConsistencyCheck.tsx` | Consistency results |
| `frontend/components/chat/EmptyState.tsx` | Initial state with example prompts |
| `frontend/public/video/.gitkeep` | Placeholder for Kling video |

---

## Task 1: FastAPI Backend

**Files:**
- Create: `src/rag_v2/api.py`

This is the bridge between the React frontend and the existing Python RAG pipeline. It reuses the same `get_system()`, `ask()`, and session management from `app.py`.

- [ ] **Step 1: Create `src/rag_v2/api.py`**

```python
"""
FastAPI backend for InfoWeave UI V2.

Exposes the RAG V2 pipeline over HTTP for the Next.js frontend.
The existing Gradio UI (app.py) is NOT modified — both can run independently.

Usage (from repo root, with PYTHONPATH=src):
    set PYTHONPATH=src
    uvicorn rag_v2.api:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

sys.stdout.reconfigure(encoding="utf-8")

_bu = os.environ.get("OPENAI_BASE_URL")
if _bu is not None and not str(_bu).strip():
    os.environ.pop("OPENAI_BASE_URL", None)

# ── App ────────────────────────────────────────────────────────────

app = FastAPI(title="InfoWeave API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy system loading (same pattern as app.py) ──────────────────

_SYSTEM: dict | None = None
_SESSIONS: dict[str, "SessionMemory"] = {}


def _load_system():
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from rag_v1.pipeline import load_all, openai_client, EMBED_MODEL_NAME, RERANK_MODEL_NAME

    print("[API] Loading models and corpus ...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    reranker = CrossEncoder(RERANK_MODEL_NAME)
    client = openai_client()
    meta, ctx_idx, qa_items, qa_idx, bm25, dataset = load_all(embed_model)
    print("[API] System ready.")
    return dict(
        embed_model=embed_model, reranker=reranker, client=client,
        meta=meta, ctx_idx=ctx_idx, qa_items=qa_items, qa_idx=qa_idx,
        bm25=bm25, dataset=dataset,
    )


def get_system():
    global _SYSTEM
    if _SYSTEM is None:
        _SYSTEM = _load_system()
    return _SYSTEM


def _get_session(session_id: str):
    from rag_v2.session import SessionMemory
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = SessionMemory(max_turns=10)
    return _SESSIONS[session_id]


# ── Request/response models ───────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str


class ClearRequest(BaseModel):
    session_id: str


# ── Endpoints ─────────────────────────────────────────────────────

@app.post("/api/chat")
def chat(req: ChatRequest):
    from rag_v2.pipeline import ask

    if not req.message.strip():
        return {"answer": "", "retrieved": [], "retrieval_log": {}, "consistency": None}

    sys_data = get_system()
    session = _get_session(req.session_id)

    result = ask(
        req.message,
        session=session,
        client=sys_data["client"],
        embed_model=sys_data["embed_model"],
        corpus_idx=sys_data["ctx_idx"],
        corpus_meta=sys_data["meta"],
        bm25=sys_data["bm25"],
        qa_items=sys_data["qa_items"],
        qa_idx=sys_data["qa_idx"],
        reranker=sys_data["reranker"],
    )

    return {
        "answer": result["answer"],
        "retrieved": result.get("retrieved", []),
        "retrieval_log": result.get("retrieval_log", {}),
        "consistency": result.get("consistency"),
    }


@app.post("/api/session/clear")
def clear_session(req: ClearRequest):
    if req.session_id in _SESSIONS:
        _SESSIONS[req.session_id].clear()
    return {"status": "ok"}


@app.get("/api/eval")
def get_eval():
    comparison_path = PROJECT_ROOT / "results" / "v2" / "v1_vs_v2_metrics.json"
    mt_path = PROJECT_ROOT / "results" / "v2" / "multiturn_eval_metrics.json"

    data = {}
    if comparison_path.exists():
        with open(comparison_path, "r", encoding="utf-8") as f:
            data["comparison"] = json.load(f)
    if mt_path.exists():
        with open(mt_path, "r", encoding="utf-8") as f:
            data["multiturn"] = json.load(f)
    return data


@app.get("/api/system")
def get_system_info():
    from rag_v1.pipeline import EMBED_MODEL_NAME, RERANK_MODEL_NAME

    llm_model = os.getenv("LLM_MODEL_V2") or os.getenv("LLM_MODEL", "openai/gpt-5.4")
    return {
        "version": "RAG V2 — Phase 2",
        "llm_model": llm_model,
        "embed_model": EMBED_MODEL_NAME,
        "sparse_model": "BM25",
        "reranker": RERANK_MODEL_NAME,
        "top_k": {"dense": 20, "sparse": 20, "final": 5},
        "features": [
            "HyDE",
            "Intent Classification",
            "Sub-query Decomposition",
            "Multi-hop Retrieval",
            "Diversity Selection",
            "Consistency Check",
            "Session Memory",
            "Coreference Resolution",
            "Multi-turn Dialogue",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  InfoWeave API — FastAPI Backend")
    print("  http://localhost:8000")
    print("  Docs: http://localhost:8000/docs")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- [ ] **Step 2: Verify it doesn't break existing Gradio app**

Run: `cd c:/Users/Youssef/DocuMind-1 && python -c "import ast; ast.parse(open('src/rag_v2/api.py').read()); print('SYNTAX OK')"`
Expected: `SYNTAX OK`

Run: `cd c:/Users/Youssef/DocuMind-1 && python -c "import ast; ast.parse(open('src/rag_v2/app.py').read()); print('GRADIO UNTOUCHED OK')"`
Expected: `GRADIO UNTOUCHED OK`

- [ ] **Step 3: Commit**

```bash
git add src/rag_v2/api.py
git commit -m "feat: add FastAPI backend for InfoWeave UI V2

Thin HTTP wrapper over existing RAG pipeline. Exposes /api/chat,
/api/eval, /api/system, and /api/session/clear. Gradio app untouched."
```

---

## Task 2: Next.js Project Scaffold

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/next.config.ts`
- Create: `frontend/postcss.config.mjs`
- Create: `frontend/tsconfig.json`
- Create: `frontend/app/globals.css`
- Create: `frontend/app/layout.tsx`
- Create: `frontend/public/video/.gitkeep`

- [ ] **Step 1: Create `frontend/package.json`**

```json
{
  "name": "infoweave-ui",
  "version": "2.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start"
  },
  "dependencies": {
    "next": "^14.2.0",
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "framer-motion": "^11.0.0",
    "@phosphor-icons/react": "^2.1.0"
  },
  "devDependencies": {
    "@tailwindcss/postcss": "^4.0.0",
    "tailwindcss": "^4.0.0",
    "typescript": "^5.4.0",
    "@types/node": "^20.0.0",
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0"
  }
}
```

- [ ] **Step 2: Create `frontend/next.config.ts`**

```typescript
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
    ];
  },
};

export default nextConfig;
```

- [ ] **Step 3: Create `frontend/postcss.config.mjs`**

```javascript
/** @type {import('postcss-load-config').Config} */
const config = {
  plugins: {
    "@tailwindcss/postcss": {},
  },
};

export default config;
```

- [ ] **Step 4: Create `frontend/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2017",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": {
      "@/*": ["./*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

- [ ] **Step 5: Create `frontend/app/globals.css`**

```css
@import "tailwindcss";

@theme {
  --color-emerald-accent: #059669;
  --color-emerald-accent-hover: #047857;
  --color-emerald-accent-light: #d1fae5;
  --color-surface: #f9fafb;
  --color-surface-card: #ffffff;
  --color-border-subtle: rgba(148, 163, 184, 0.3);
}

html {
  scroll-behavior: smooth;
}

body {
  background-color: var(--color-surface);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* Scrollbar styling */
::-webkit-scrollbar {
  width: 6px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: #94a3b8;
}
```

- [ ] **Step 6: Create `frontend/app/layout.tsx`**

```tsx
import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

const geist = localFont({
  src: [
    {
      path: "../node_modules/geist/dist/fonts/geist-sans/Geist-Regular.woff2",
      weight: "400",
      style: "normal",
    },
    {
      path: "../node_modules/geist/dist/fonts/geist-sans/Geist-Medium.woff2",
      weight: "500",
      style: "normal",
    },
    {
      path: "../node_modules/geist/dist/fonts/geist-sans/Geist-SemiBold.woff2",
      weight: "600",
      style: "normal",
    },
    {
      path: "../node_modules/geist/dist/fonts/geist-sans/Geist-Bold.woff2",
      weight: "700",
      style: "normal",
    },
  ],
  variable: "--font-geist",
  display: "swap",
});

const geistMono = localFont({
  src: "../node_modules/geist/dist/fonts/geist-mono/GeistMono-Regular.woff2",
  variable: "--font-geist-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "InfoWeave — SSL Research Assistant",
  description:
    "Sustainable Solutions Lab Research Assistant. Multi-turn dialogue with source attribution, powered by RAG V2.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${geist.variable} ${geistMono.variable}`}>
      <body className="font-[family-name:var(--font-geist)] text-zinc-900 antialiased">
        {children}
      </body>
    </html>
  );
}
```

- [ ] **Step 7: Create `frontend/public/video/.gitkeep`**

Empty file -- placeholder directory for the Kling video.

- [ ] **Step 8: Install dependencies**

Run: `cd c:/Users/Youssef/DocuMind-1/frontend && npm install`

Note: Also install `geist` font package:
Run: `cd c:/Users/Youssef/DocuMind-1/frontend && npm install geist`

- [ ] **Step 9: Verify Next.js builds**

Run: `cd c:/Users/Youssef/DocuMind-1/frontend && npx next build`
Expected: Build succeeds (may warn about missing page.tsx -- that's fine, we create it in the next task)

- [ ] **Step 10: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/next.config.ts frontend/postcss.config.mjs frontend/tsconfig.json frontend/app/globals.css frontend/app/layout.tsx frontend/public/video/.gitkeep
git commit -m "feat: scaffold Next.js project with Tailwind v4, Geist fonts

Standalone frontend in frontend/ directory. API proxy configured
to forward /api/* to FastAPI on port 8000."
```

---

## Task 3: Shared Types and API Client

**Files:**
- Create: `frontend/lib/types.ts`
- Create: `frontend/lib/api.ts`

- [ ] **Step 1: Create `frontend/lib/types.ts`**

```typescript
export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface RetrievedSource {
  source: string;
  section_title?: string;
  source_type: string;
  score: number;
  layer: string;
  chunk_text?: string;
}

export interface RetrievalLog {
  intent?: string;
  strategy?: string;
  resolved_query?: string;
  original_query?: string;
  sub_queries?: string[];
  hyde_used?: boolean;
  final_raw?: number;
  final_qa?: number;
  qa_neg_used?: boolean;
  multihop?: {
    triggered: boolean;
    supplements_added: number;
    missing_entities: string[];
  };
}

export interface ConsistencyResult {
  is_consistent: boolean;
  confidence: number;
  unsupported_claims?: string[];
  explanation?: string;
}

export interface ChatResponse {
  answer: string;
  retrieved: RetrievedSource[];
  retrieval_log: RetrievalLog;
  consistency: ConsistencyResult | null;
}

export interface EvalData {
  comparison?: {
    rag_v1_same_dataset_95?: Record<string, unknown>;
    rag_v2?: Record<string, unknown>;
  };
  multiturn?: {
    overall?: Record<string, unknown>;
    coreference?: Record<string, unknown>;
    by_turn_position?: Record<string, unknown>;
  };
}

export interface SystemInfo {
  version: string;
  llm_model: string;
  embed_model: string;
  sparse_model: string;
  reranker: string;
  top_k: { dense: number; sparse: number; final: number };
  features: string[];
}
```

- [ ] **Step 2: Create `frontend/lib/api.ts`**

```typescript
import type { ChatResponse, EvalData, SystemInfo } from "./types";

const BASE = "";

export async function sendMessage(
  message: string,
  sessionId: string
): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
  });
  if (!res.ok) throw new Error(`Chat request failed: ${res.status}`);
  return res.json();
}

export async function clearSession(sessionId: string): Promise<void> {
  await fetch(`${BASE}/api/session/clear`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
}

export async function fetchEval(): Promise<EvalData> {
  const res = await fetch(`${BASE}/api/eval`);
  if (!res.ok) throw new Error(`Eval fetch failed: ${res.status}`);
  return res.json();
}

export async function fetchSystemInfo(): Promise<SystemInfo> {
  const res = await fetch(`${BASE}/api/system`);
  if (!res.ok) throw new Error(`System fetch failed: ${res.status}`);
  return res.json();
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/lib/types.ts frontend/lib/api.ts
git commit -m "feat: add TypeScript types and API client for FastAPI backend"
```

---

## Task 4: Landing Page -- Navbar

**Files:**
- Create: `frontend/components/landing/Navbar.tsx`

- [ ] **Step 1: Create `frontend/components/landing/Navbar.tsx`**

```tsx
"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import Link from "next/link";

const NAV_LINKS = [
  { label: "Evaluation", href: "#evaluation" },
  { label: "Architecture", href: "#architecture" },
];

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <motion.nav
      className={`fixed top-0 left-0 right-0 z-40 transition-all duration-300 ${
        scrolled
          ? "bg-white/70 backdrop-blur-xl border-b border-white/20 shadow-[inset_0_1px_0_rgba(255,255,255,0.1)]"
          : "bg-transparent"
      }`}
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ type: "spring", stiffness: 100, damping: 20 }}
    >
      <div className="mx-auto max-w-[1400px] flex items-center justify-between px-6 py-4">
        <Link
          href="/"
          className="text-lg font-semibold tracking-tight text-zinc-900"
        >
          InfoWeave
        </Link>

        <div className="flex items-center gap-8">
          {NAV_LINKS.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className="text-sm text-slate-500 transition-colors hover:text-zinc-900"
            >
              {link.label}
            </a>
          ))}
          <Link
            href="/chat"
            className="rounded-lg bg-emerald-accent px-4 py-2 text-sm font-medium text-white transition-transform hover:bg-emerald-accent-hover active:scale-[0.98]"
          >
            Launch Chat
          </Link>
        </div>
      </div>
    </motion.nav>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/components/landing/Navbar.tsx
git commit -m "feat: add Navbar with glassmorphism scroll transition"
```

---

## Task 5: Landing Page -- Hero

**Files:**
- Create: `frontend/components/landing/Hero.tsx`

- [ ] **Step 1: Create `frontend/components/landing/Hero.tsx`**

```tsx
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
```

- [ ] **Step 2: Commit**

```bash
git add frontend/components/landing/Hero.tsx
git commit -m "feat: add Hero section with full-bleed video and staggered text reveal"
```

---

## Task 6: Landing Page -- Animated Counter (Isolated Client Component)

**Files:**
- Create: `frontend/components/landing/AnimatedCounter.tsx`

This is a memoized, isolated client component for the counter animation. Prevents parent re-renders.

- [ ] **Step 1: Create `frontend/components/landing/AnimatedCounter.tsx`**

```tsx
"use client";

import { memo, useEffect, useRef, useState } from "react";
import { useInView } from "framer-motion";

interface AnimatedCounterProps {
  target: number;
  suffix?: string;
  prefix?: string;
  decimals?: number;
  duration?: number;
}

function AnimatedCounterInner({
  target,
  suffix = "",
  prefix = "",
  decimals = 1,
  duration = 1500,
}: AnimatedCounterProps) {
  const ref = useRef<HTMLSpanElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });
  const [value, setValue] = useState(0);

  useEffect(() => {
    if (!isInView) return;

    const start = performance.now();
    const step = (now: number) => {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(eased * target);
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [isInView, target, duration]);

  return (
    <span ref={ref} className="font-[family-name:var(--font-geist-mono)] tabular-nums">
      {prefix}
      {value.toFixed(decimals)}
      {suffix}
    </span>
  );
}

const AnimatedCounter = memo(AnimatedCounterInner);
export default AnimatedCounter;
```

- [ ] **Step 2: Commit**

```bash
git add frontend/components/landing/AnimatedCounter.tsx
git commit -m "feat: add memoized AnimatedCounter for scroll-triggered metric reveals"
```

---

## Task 7: Landing Page -- Evaluation Bento Grid

**Files:**
- Create: `frontend/components/landing/FeaturesBento.tsx`

- [ ] **Step 1: Create `frontend/components/landing/FeaturesBento.tsx`**

```tsx
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
```

- [ ] **Step 2: Commit**

```bash
git add frontend/components/landing/FeaturesBento.tsx
git commit -m "feat: add evaluation bento grid with animated counters and staggered reveal"
```

---

## Task 8: Landing Page -- Pipeline Architecture

**Files:**
- Create: `frontend/components/landing/PipelineShowcase.tsx`

- [ ] **Step 1: Create `frontend/components/landing/PipelineShowcase.tsx`**

```tsx
"use client";

import { useRef } from "react";
import { motion, useInView } from "framer-motion";

interface PipelineNode {
  label: string;
  detail: string;
  accent?: boolean;
}

const PIPELINE: PipelineNode[] = [
  { label: "Query", detail: "User question" },
  { label: "Intent Classification", detail: "LLM-enhanced", accent: true },
  { label: "Query Transform", detail: "HyDE / Sub-query", accent: true },
  { label: "Dense Retrieval", detail: "all-MiniLM-L6-v2" },
  { label: "Sparse Retrieval", detail: "BM25" },
  { label: "Reranking", detail: "ms-marco-MiniLM-L-6-v2" },
  { label: "Diversity Selection", detail: "Top-5 final", accent: true },
  { label: "Generation", detail: "LLM" },
  { label: "Consistency Check", detail: "Answer-evidence", accent: true },
];

const nodeVariants = {
  hidden: { scale: 0.8, opacity: 0 },
  visible: {
    scale: 1,
    opacity: 1,
    transition: { type: "spring", stiffness: 100, damping: 20 },
  },
};

const lineVariants = {
  hidden: { pathLength: 0, opacity: 0 },
  visible: {
    pathLength: 1,
    opacity: 1,
    transition: { duration: 0.4, ease: "easeOut" },
  },
};

export default function PipelineShowcase() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section id="architecture" className="py-24 md:py-32">
      <div className="mx-auto max-w-[1400px] px-6">
        <h2 className="text-3xl font-bold tracking-tight text-zinc-900 md:text-4xl">
          Architecture
        </h2>
        <p className="mt-3 max-w-[65ch] text-base leading-relaxed text-slate-500">
          A multi-stage retrieval-augmented generation pipeline with intent
          classification, query transformation, hybrid retrieval, and
          answer-evidence consistency verification.
        </p>

        {/* Desktop: horizontal flow */}
        <div ref={ref} className="mt-16 hidden md:block">
          <motion.div
            className="flex items-start gap-0"
            initial="hidden"
            animate={isInView ? "visible" : "hidden"}
            transition={{ staggerChildren: 0.1 }}
          >
            {PIPELINE.map((node, i) => (
              <div key={node.label} className="flex items-center">
                <motion.div
                  className="flex flex-col items-center"
                  variants={nodeVariants}
                >
                  <div
                    className={`flex h-14 w-14 items-center justify-center rounded-xl border ${
                      node.accent
                        ? "border-emerald-accent/30 bg-emerald-accent-light"
                        : "border-slate-200 bg-white"
                    } shadow-[0_4px_12px_-4px_rgba(0,0,0,0.08)]`}
                  >
                    <span className="text-xs font-semibold text-zinc-700">
                      {i + 1}
                    </span>
                  </div>
                  <p className="mt-3 max-w-[100px] text-center text-xs font-medium text-zinc-900">
                    {node.label}
                  </p>
                  <p className="mt-0.5 max-w-[100px] text-center text-[11px] text-slate-400 font-[family-name:var(--font-geist-mono)]">
                    {node.detail}
                  </p>
                </motion.div>

                {i < PIPELINE.length - 1 && (
                  <motion.svg
                    width="40"
                    height="2"
                    className="mx-1 mt-7 flex-shrink-0"
                    variants={lineVariants}
                  >
                    <motion.line
                      x1="0"
                      y1="1"
                      x2="40"
                      y2="1"
                      stroke="#cbd5e1"
                      strokeWidth="2"
                      strokeLinecap="round"
                    />
                  </motion.svg>
                )}
              </div>
            ))}
          </motion.div>

          {/* Feedback loop arrow label */}
          <p className="mt-8 text-xs text-slate-400 italic">
            Session Memory and Coreference Resolution feed back from Generation
            to Query Transform for multi-turn dialogue.
          </p>
        </div>

        {/* Mobile: vertical flow */}
        <div ref={ref} className="mt-12 md:hidden">
          <motion.div
            className="flex flex-col gap-4"
            initial="hidden"
            animate={isInView ? "visible" : "hidden"}
            transition={{ staggerChildren: 0.08 }}
          >
            {PIPELINE.map((node, i) => (
              <motion.div
                key={node.label}
                className="flex items-center gap-4"
                variants={nodeVariants}
              >
                <div
                  className={`flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg border ${
                    node.accent
                      ? "border-emerald-accent/30 bg-emerald-accent-light"
                      : "border-slate-200 bg-white"
                  } shadow-sm`}
                >
                  <span className="text-xs font-semibold text-zinc-700">
                    {i + 1}
                  </span>
                </div>
                <div>
                  <p className="text-sm font-medium text-zinc-900">
                    {node.label}
                  </p>
                  <p className="text-xs text-slate-400 font-[family-name:var(--font-geist-mono)]">
                    {node.detail}
                  </p>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/components/landing/PipelineShowcase.tsx
git commit -m "feat: add animated pipeline architecture showcase with scroll-triggered reveals"
```

---

## Task 9: Landing Page -- Footer

**Files:**
- Create: `frontend/components/landing/Footer.tsx`

- [ ] **Step 1: Create `frontend/components/landing/Footer.tsx`**

```tsx
export default function Footer() {
  return (
    <footer className="border-t border-slate-200/50 py-8">
      <div className="mx-auto flex max-w-[1400px] items-center justify-between px-6">
        <p className="text-sm text-slate-400">
          Sustainable Solutions Lab &middot; {new Date().getFullYear()}
        </p>
        <a
          href="#"
          className="text-sm text-slate-400 transition-colors hover:text-zinc-900"
          onClick={(e) => {
            e.preventDefault();
            window.scrollTo({ top: 0, behavior: "smooth" });
          }}
        >
          Back to top
        </a>
      </div>
    </footer>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/components/landing/Footer.tsx
git commit -m "feat: add minimal footer with back-to-top"
```

---

## Task 10: Landing Page -- Assemble

**Files:**
- Create: `frontend/app/page.tsx`

- [ ] **Step 1: Create `frontend/app/page.tsx`**

```tsx
import Navbar from "@/components/landing/Navbar";
import Hero from "@/components/landing/Hero";
import FeaturesBento from "@/components/landing/FeaturesBento";
import PipelineShowcase from "@/components/landing/PipelineShowcase";
import Footer from "@/components/landing/Footer";

export default function LandingPage() {
  return (
    <>
      <Navbar />
      <main>
        <Hero />
        <FeaturesBento />
        <PipelineShowcase />
      </main>
      <Footer />
    </>
  );
}
```

- [ ] **Step 2: Run the dev server and verify landing page renders**

Run: `cd c:/Users/Youssef/DocuMind-1/frontend && npm run dev`

Open `http://localhost:3000` in a browser. Verify:
- Navbar is visible and transparent
- Hero section renders with text (video placeholder is fine)
- Scrolling reveals the evaluation bento grid with counter animations
- Pipeline section shows the architecture flow
- Footer renders at the bottom
- Navbar transitions to frosted glass on scroll

- [ ] **Step 3: Commit**

```bash
git add frontend/app/page.tsx
git commit -m "feat: assemble landing page with Navbar, Hero, Bento, Pipeline, Footer"
```

---

## Task 11: Chat Page -- Empty State and Input Bar

**Files:**
- Create: `frontend/components/chat/EmptyState.tsx`
- Create: `frontend/components/chat/InputBar.tsx`

- [ ] **Step 1: Create `frontend/components/chat/EmptyState.tsx`**

```tsx
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
            className="rounded-lg border border-slate-200/50 bg-white px-4 py-2 text-sm text-slate-600 shadow-sm transition-all hover:border-slate-300 hover:text-zinc-900 hover:shadow-md active:scale-[0.98]"
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
```

- [ ] **Step 2: Create `frontend/components/chat/InputBar.tsx`**

```tsx
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
```

- [ ] **Step 3: Commit**

```bash
git add frontend/components/chat/EmptyState.tsx frontend/components/chat/InputBar.tsx
git commit -m "feat: add chat EmptyState with example chips and InputBar with typewriter placeholder"
```

---

## Task 12: Chat Page -- Message List

**Files:**
- Create: `frontend/components/chat/MessageList.tsx`

- [ ] **Step 1: Create `frontend/components/chat/MessageList.tsx`**

```tsx
"use client";

import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { ChatMessage } from "@/lib/types";

interface MessageListProps {
  messages: ChatMessage[];
  loading?: boolean;
}

const messageVariants = {
  hidden: { opacity: 0, y: 10 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { type: "spring", stiffness: 100, damping: 20 },
  },
};

function SkeletonMessage() {
  return (
    <div className="flex justify-start">
      <div className="max-w-[75%] rounded-2xl rounded-bl-md bg-white px-4 py-3 shadow-sm">
        <div className="space-y-2">
          <div className="h-3 w-64 animate-pulse rounded bg-slate-200" />
          <div className="h-3 w-48 animate-pulse rounded bg-slate-200" />
          <div className="h-3 w-56 animate-pulse rounded bg-slate-200" />
        </div>
      </div>
    </div>
  );
}

export default function MessageList({ messages, loading }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, loading]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6">
      <div className="mx-auto max-w-3xl space-y-4">
        <AnimatePresence initial={false}>
          {messages.map((msg, i) => (
            <motion.div
              key={i}
              className={`flex ${
                msg.role === "user" ? "justify-end" : "justify-start"
              }`}
              variants={messageVariants}
              initial="hidden"
              animate="visible"
              layout
            >
              <div
                className={`max-w-[75%] whitespace-pre-wrap text-sm leading-relaxed ${
                  msg.role === "user"
                    ? "rounded-2xl rounded-br-md bg-emerald-accent px-4 py-3 text-white"
                    : "rounded-2xl rounded-bl-md bg-white px-4 py-3 text-slate-700 shadow-sm"
                }`}
              >
                {msg.content}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.2 }}
          >
            <SkeletonMessage />
          </motion.div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/components/chat/MessageList.tsx
git commit -m "feat: add MessageList with animated messages and skeleton loader"
```

---

## Task 13: Chat Page -- Context Panel (Sources, Retrieval, Consistency)

**Files:**
- Create: `frontend/components/chat/SourcePanel.tsx`
- Create: `frontend/components/chat/RetrievalDetails.tsx`
- Create: `frontend/components/chat/ConsistencyCheck.tsx`

- [ ] **Step 1: Create `frontend/components/chat/SourcePanel.tsx`**

```tsx
"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BookOpen, Globe, File, CaretDown } from "@phosphor-icons/react";
import type { RetrievedSource } from "@/lib/types";

const SOURCE_ICONS: Record<string, typeof BookOpen> = {
  curated_qa: BookOpen,
  website: Globe,
  pdf: File,
};

interface SourcePanelProps {
  sources: RetrievedSource[];
}

export default function SourcePanel({ sources }: SourcePanelProps) {
  const [open, setOpen] = useState(true);

  return (
    <div className="border-t border-slate-200/50">
      <button
        className="flex w-full items-center justify-between px-4 py-3 text-sm font-medium text-zinc-900 transition-colors hover:bg-slate-50"
        onClick={() => setOpen(!open)}
      >
        <span>Retrieved Sources</span>
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
              {sources.length === 0 ? (
                <p className="py-4 text-center text-sm text-slate-400">
                  Ask a question to see sources
                </p>
              ) : (
                <div className="space-y-2">
                  {sources.map((src, i) => {
                    const Icon = SOURCE_ICONS[src.source_type] ?? File;
                    const isQa = src.layer === "qa_memory";
                    return (
                      <div
                        key={i}
                        className={`rounded-lg border-l-2 bg-white px-3 py-2 ${
                          isQa ? "border-l-emerald-accent" : "border-l-slate-300"
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Icon size={14} className="text-slate-500" />
                            <span className="text-xs font-medium text-zinc-900">
                              {src.source}
                            </span>
                          </div>
                          <span className="text-[11px] text-slate-400 font-[family-name:var(--font-geist-mono)]">
                            {src.score.toFixed(3)}
                          </span>
                        </div>
                        {src.section_title && (
                          <p className="mt-0.5 text-[11px] text-slate-400">
                            {src.section_title}
                          </p>
                        )}
                        {src.chunk_text && (
                          <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-slate-500">
                            {src.chunk_text}
                          </p>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
```

- [ ] **Step 2: Create `frontend/components/chat/RetrievalDetails.tsx`**

```tsx
"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CaretDown } from "@phosphor-icons/react";
import type { RetrievalLog } from "@/lib/types";

interface RetrievalDetailsProps {
  log: RetrievalLog | null;
}

export default function RetrievalDetails({ log }: RetrievalDetailsProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="border-t border-slate-200/50">
      <button
        className="flex w-full items-center justify-between px-4 py-3 text-sm font-medium text-zinc-900 transition-colors hover:bg-slate-50"
        onClick={() => setOpen(!open)}
      >
        <span>Retrieval Details</span>
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
              {!log ? (
                <p className="py-4 text-center text-sm text-slate-400">
                  Retrieval details will appear here
                </p>
              ) : (
                <div className="space-y-3 text-xs">
                  {log.intent && (
                    <div>
                      <span className="text-slate-400">Intent</span>
                      <span className="ml-2 rounded bg-slate-100 px-2 py-0.5 font-[family-name:var(--font-geist-mono)] text-zinc-700">
                        {log.intent}
                      </span>
                    </div>
                  )}
                  {log.strategy && (
                    <div>
                      <span className="text-slate-400">Strategy</span>
                      <span className="ml-2 rounded bg-slate-100 px-2 py-0.5 font-[family-name:var(--font-geist-mono)] text-zinc-700">
                        {log.strategy}
                      </span>
                    </div>
                  )}
                  {log.resolved_query &&
                    log.original_query &&
                    log.resolved_query !== log.original_query && (
                      <div>
                        <span className="text-slate-400">Resolved query</span>
                        <p className="mt-1 text-slate-600 italic">
                          {log.resolved_query}
                        </p>
                      </div>
                    )}
                  {log.sub_queries && log.sub_queries.length > 1 && (
                    <div>
                      <span className="text-slate-400">Sub-queries</span>
                      <ul className="mt-1 space-y-0.5">
                        {log.sub_queries.map((sq, i) => (
                          <li key={i} className="text-slate-600">
                            {sq}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  <div className="flex flex-wrap gap-2">
                    {log.hyde_used && (
                      <span className="rounded-full bg-emerald-accent-light px-2 py-0.5 text-[11px] font-medium text-emerald-accent">
                        HyDE
                      </span>
                    )}
                    {log.multihop?.triggered && (
                      <span className="rounded-full bg-emerald-accent-light px-2 py-0.5 text-[11px] font-medium text-emerald-accent">
                        Multi-hop +{log.multihop.supplements_added}
                      </span>
                    )}
                    {log.qa_neg_used && (
                      <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[11px] font-medium text-amber-700">
                        No-evidence signal
                      </span>
                    )}
                  </div>
                  <p className="text-slate-400">
                    Retrieved: {log.final_raw ?? "?"} corpus, {log.final_qa ?? "?"} QA
                  </p>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
```

- [ ] **Step 3: Create `frontend/components/chat/ConsistencyCheck.tsx`**

```tsx
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
```

- [ ] **Step 4: Commit**

```bash
git add frontend/components/chat/SourcePanel.tsx frontend/components/chat/RetrievalDetails.tsx frontend/components/chat/ConsistencyCheck.tsx
git commit -m "feat: add context panel components (SourcePanel, RetrievalDetails, ConsistencyCheck)"
```

---

## Task 14: Chat Page -- ChatInterface and Page Assembly

**Files:**
- Create: `frontend/components/chat/ChatInterface.tsx`
- Create: `frontend/app/chat/page.tsx`

- [ ] **Step 1: Create `frontend/components/chat/ChatInterface.tsx`**

```tsx
"use client";

import { useState, useCallback } from "react";
import Link from "next/link";
import { ArrowLeft, Plus } from "@phosphor-icons/react";
import { sendMessage, clearSession } from "@/lib/api";
import type {
  ChatMessage,
  RetrievedSource,
  RetrievalLog,
  ConsistencyResult,
} from "@/lib/types";
import MessageList from "./MessageList";
import InputBar from "./InputBar";
import EmptyState from "./EmptyState";
import SourcePanel from "./SourcePanel";
import RetrievalDetails from "./RetrievalDetails";
import ConsistencyCheck from "./ConsistencyCheck";

function generateSessionId() {
  return crypto.randomUUID();
}

export default function ChatInterface() {
  const [sessionId] = useState(generateSessionId);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sources, setSources] = useState<RetrievedSource[]>([]);
  const [retrievalLog, setRetrievalLog] = useState<RetrievalLog | null>(null);
  const [consistency, setConsistency] = useState<ConsistencyResult | null>(
    null
  );
  const [loading, setLoading] = useState(false);

  const handleSend = useCallback(
    async (message: string) => {
      setMessages((prev) => [...prev, { role: "user", content: message }]);
      setLoading(true);

      try {
        const res = await sendMessage(message, sessionId);
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: res.answer },
        ]);
        setSources(res.retrieved);
        setRetrievalLog(res.retrieval_log);
        setConsistency(res.consistency);
      } catch {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content:
              "Connection to the backend failed. Make sure the FastAPI server is running on port 8000.",
          },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [sessionId]
  );

  const handleClear = useCallback(async () => {
    await clearSession(sessionId);
    setMessages([]);
    setSources([]);
    setRetrievalLog(null);
    setConsistency(null);
  }, [sessionId]);

  const isEmpty = messages.length === 0;

  return (
    <div className="flex h-[100dvh] flex-col bg-[#f9fafb]">
      {/* Top bar */}
      <div className="flex items-center justify-between border-b border-slate-200/50 bg-white/80 px-4 py-3 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="flex items-center gap-1.5 text-sm text-slate-500 transition-colors hover:text-zinc-900"
          >
            <ArrowLeft size={14} />
            <span>Back</span>
          </Link>
          <span className="text-sm font-semibold text-zinc-900">
            InfoWeave
          </span>
        </div>
        <button
          onClick={handleClear}
          className="flex items-center gap-1.5 rounded-lg border border-slate-200/50 px-3 py-1.5 text-xs text-slate-500 transition-all hover:border-slate-300 hover:text-zinc-900 active:scale-[0.98]"
        >
          <Plus size={12} />
          New conversation
        </button>
      </div>

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Conversation area */}
        <div className="flex flex-1 flex-col">
          {isEmpty ? (
            <EmptyState onSelectExample={handleSend} />
          ) : (
            <MessageList messages={messages} loading={loading} />
          )}
          <InputBar
            onSend={handleSend}
            disabled={loading}
            showTypewriter={isEmpty}
          />
        </div>

        {/* Context panel -- desktop only */}
        <div className="hidden w-[380px] flex-shrink-0 overflow-y-auto border-l border-slate-200/50 bg-white md:block">
          <SourcePanel sources={sources} />
          <RetrievalDetails log={retrievalLog} />
          <ConsistencyCheck consistency={consistency} />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create `frontend/app/chat/page.tsx`**

```tsx
import ChatInterface from "@/components/chat/ChatInterface";

export default function ChatPage() {
  return <ChatInterface />;
}
```

- [ ] **Step 3: Run dev server and verify chat page**

Run: `cd c:/Users/Youssef/DocuMind-1/frontend && npm run dev`

Open `http://localhost:3000/chat` in a browser. Verify:
- Empty state renders with InfoWeave heading and example chips
- Clicking an example chip populates the input (or sends directly)
- Top bar shows "Back" link and "New conversation" button
- Context panel is visible on the right side on desktop
- If FastAPI is not running, the error message appears gracefully

- [ ] **Step 4: Commit**

```bash
git add frontend/components/chat/ChatInterface.tsx frontend/app/chat/page.tsx
git commit -m "feat: assemble chat page with split layout, message thread, and context panel"
```

---

## Task 15: Integration Test -- Full Stack

This is a manual verification task to confirm the full stack works end-to-end.

- [ ] **Step 1: Start the FastAPI backend**

Run in terminal 1:
```bash
cd c:/Users/Youssef/DocuMind-1
set PYTHONPATH=src
python -m rag_v2.api
```

Wait for `[API] System ready.` to appear.

- [ ] **Step 2: Start the Next.js dev server**

Run in terminal 2:
```bash
cd c:/Users/Youssef/DocuMind-1/frontend
npm run dev
```

- [ ] **Step 3: Verify landing page**

Open `http://localhost:3000`. Check:
- Navbar renders, glassmorphism on scroll
- Hero section with text overlay and scroll indicator
- Evaluation bento grid with counter animations on scroll
- Pipeline architecture section with staggered node reveals
- Footer with "Back to top"
- "Launch Chat" and "Start a conversation" buttons navigate to `/chat`

- [ ] **Step 4: Verify chat page end-to-end**

Navigate to `http://localhost:3000/chat`. Check:
- Empty state shows with example chips
- Click an example, message sends to backend
- Assistant response appears with spring animation
- Sources populate in the right panel
- Retrieval details show intent, strategy, sub-queries
- Consistency check renders if available
- "New conversation" clears everything
- "Back" link returns to landing page

- [ ] **Step 5: Verify Gradio still works independently**

Run in terminal 3:
```bash
cd c:/Users/Youssef/DocuMind-1
set PYTHONPATH=src
python -m rag_v2.app
```

Open `http://localhost:7860`. Confirm the original Gradio UI is completely intact and functional.

- [ ] **Step 6: Final commit -- add .gitignore for frontend**

Create `frontend/.gitignore`:
```
node_modules/
.next/
out/
```

```bash
git add frontend/.gitignore
git commit -m "chore: add frontend .gitignore for node_modules and .next"
```

---

## Summary

| Task | Description | Key Files |
|------|-------------|-----------|
| 1 | FastAPI backend | `src/rag_v2/api.py` |
| 2 | Next.js scaffold | `frontend/package.json`, `layout.tsx`, `globals.css` |
| 3 | Types and API client | `frontend/lib/types.ts`, `frontend/lib/api.ts` |
| 4 | Navbar | `frontend/components/landing/Navbar.tsx` |
| 5 | Hero | `frontend/components/landing/Hero.tsx` |
| 6 | Animated counter | `frontend/components/landing/AnimatedCounter.tsx` |
| 7 | Evaluation bento | `frontend/components/landing/FeaturesBento.tsx` |
| 8 | Pipeline showcase | `frontend/components/landing/PipelineShowcase.tsx` |
| 9 | Footer | `frontend/components/landing/Footer.tsx` |
| 10 | Assemble landing page | `frontend/app/page.tsx` |
| 11 | Empty state + input bar | `frontend/components/chat/EmptyState.tsx`, `InputBar.tsx` |
| 12 | Message list | `frontend/components/chat/MessageList.tsx` |
| 13 | Context panel | `SourcePanel.tsx`, `RetrievalDetails.tsx`, `ConsistencyCheck.tsx` |
| 14 | Assemble chat page | `ChatInterface.tsx`, `frontend/app/chat/page.tsx` |
| 15 | Integration test | Manual verification of full stack |
