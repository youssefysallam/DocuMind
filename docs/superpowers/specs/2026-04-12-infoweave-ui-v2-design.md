# InfoWeave UI V2 -- High-End Website Redesign

**Date:** 2026-04-12
**Branch:** `ui-v2-improved`
**Status:** Pending tech core approval

## Context

The current InfoWeave web interface is a Gradio-based app (`src/rag_v2/app.py`) with three tabs: Chat, Evaluation, and System. It works but looks like a research prototype. The goal is to build a production-grade, high-end website that showcases the RAG V2 system to the tech core for approval.

**Constraints:**
- The existing Gradio app MUST remain untouched and runnable
- Nothing gets pushed to remote without explicit user approval
- All work lives on the `ui-v2-improved` branch
- The new frontend lives in `frontend/`, completely separate from the Gradio app

## Architecture

### Project Structure

```
frontend/                       # New -- standalone Next.js app
  app/
    layout.tsx                  # Root layout, Geist font loading
    page.tsx                    # Landing page (Server Component)
    chat/
      page.tsx                  # Chat experience (Client Component)
  components/
    landing/
      Hero.tsx                  # Full-bleed video hero
      Navbar.tsx                # Fixed nav with glassmorphism on scroll
      FeaturesBento.tsx         # Bento grid evaluation metrics
      PipelineShowcase.tsx      # Animated architecture flow diagram
      Footer.tsx                # Minimal footer
    chat/
      ChatInterface.tsx         # Main chat layout (split view)
      MessageList.tsx           # Conversation thread
      SourcePanel.tsx           # Retrieved sources sidebar
      RetrievalDetails.tsx      # Intent/strategy/sub-queries
      ConsistencyCheck.tsx      # Consistency results
      EmptyState.tsx            # Initial state with example prompts
      InputBar.tsx              # Message input with typewriter placeholder
    ui/                         # Shared primitives
  lib/
    api.ts                      # FastAPI client functions
  public/
    video/                      # Kling 3D solar panel video (user provides)

src/rag_v2/
  app.py                        # UNTOUCHED -- original Gradio UI
  api.py                        # NEW -- FastAPI wrapper for the React frontend
```

### Two Independent Entry Points

1. **Original Gradio UI** (unchanged): `python -m rag_v2.app` -> `http://localhost:7860`
2. **New UI**: `python -m rag_v2.api` (FastAPI on port 8000) + `npm run dev` in `frontend/` (Next.js on port 3000)

### FastAPI Backend (`src/rag_v2/api.py`)

Thin HTTP layer over the existing RAG pipeline. No logic duplication.

**Endpoints:**
- `POST /api/chat` -- accepts `{ message: string, session_id: string }`, returns `{ answer, retrieved, retrieval_log, consistency }`
- `GET /api/eval` -- returns evaluation metrics (V1 vs V2, multi-turn, by question type)
- `POST /api/session/clear` -- clears a session by ID
- `GET /api/system` -- returns system config (model names, Top-K values, feature list)

**Implementation:** Reuses the exact same `get_system()` lazy loader, `ask()` function, `_load_eval_dashboard()` data, and session management from the existing codebase. CORS configured for `localhost:3000`.

## Tech Stack

- **Framework:** Next.js 14+ App Router
- **Styling:** Tailwind CSS v4
- **Animation:** Framer Motion (scroll reveals, stagger, spring physics, layout transitions)
- **Fonts:** Geist + Geist Mono via `next/font`
- **Icons:** Phosphor Icons (`@phosphor-icons/react`)
- **No shadcn/ui** -- all components are custom to avoid generic look

## Visual Identity

- **Product name:** InfoWeave
- **Subtitle:** Sustainable Solutions Lab Research Assistant
- **Base palette:** Light, airy -- off-white `#f9fafb` base to match the Kling video's white background
- **Accent:** Emerald (single accent color, saturation < 80%)
- **Typography neutrals:** Slate/Zinc family (no warm/cool mixing)
- **No pure black** -- Off-black (`zinc-950`) for darkest text
- **No AI purple/blue glow aesthetic**
- **No emojis** -- Phosphor icons replace all emoji usage from the Gradio UI

## Landing Page (`/`)

### Navbar

- Fixed top position, transparent over the hero
- Transitions to frosted glass on scroll: `backdrop-blur` + 1px inner border (`border-white/10`) + inner shadow for edge refraction
- Left: InfoWeave wordmark in Geist
- Right: anchor links ("Features", "Evaluation", "Architecture") + "Launch Chat" CTA in emerald
- Smooth-scroll to sections on click

### Hero Section

- `min-h-[100dvh]` (no `h-screen` -- mobile Safari safe)
- Full-bleed `<video autoPlay muted loop playsInline>` as background
- Kling 3D solar panel engineering exploded view on white background
- Video's white background dissolves seamlessly into the page's `#f9fafb` base -- no visible frame or border
- Text overlay **left-aligned** (anti-center per design rules):
  - "InfoWeave" -- `text-4xl md:text-6xl tracking-tighter leading-none` in Geist, zinc-900
  - "Sustainable Solutions Lab Research Assistant" -- `text-base text-slate-500 leading-relaxed max-w-[65ch]`
  - Single CTA button: "Start a conversation" in emerald with tactile `:active` feedback (`scale-[0.98]`)
- Text enters with staggered fade-up, spring physics (`type: "spring", stiffness: 100, damping: 20`)
- Subtle scroll indicator at bottom (animated chevron)
- **Placeholder until video is provided:** Clean gradient or static image in the same white-to-off-white tone

### Evaluation Bento Grid

- Section heading: "Evaluation" or similar, left-aligned
- Asymmetric grid: `grid-template-columns: 2fr 1fr 1fr` top row, `1fr 1fr` bottom row
- Staggered scroll-reveal entry using Framer Motion `useInView` + `staggerChildren`
- Each tile:
  - Off-white background, 1px `border-slate-200/50`, `rounded-2xl`, diffusion shadow
  - Metric number animates up on scroll entry (counter animation)
  - Numbers in `font-mono` (Geist Mono) for tabular alignment
  - Subtle looping micro-animation per tile (pulse or shimmer), isolated in memoized client components
  - Label sits outside/below the tile, gallery-style

**Metrics to feature (from `results/v2/v1_vs_v2_metrics.json`):**

| Tile | Metric | Value | Story |
|------|--------|-------|-------|
| Large (2fr) | Hallucination reduction | -50% | 0.24 -> 0.12 hallucination rate |
| Standard | Coverage | 96.2% | Near-complete corpus coverage |
| Standard | Answer accuracy | 75% | Answerable questions |
| Standard | Correct refusal | 90.9% | Knows when it doesn't know |
| Wide | V1 vs V2 delta | Comparison | Before/after improvement bar |

- Mobile: collapses to single column, maintains staggered reveal

### Pipeline Architecture Section

- Scroll-triggered animated flow diagram
- Horizontal pipeline (left to right on desktop, vertical on mobile):

```
Query -> Intent Classification -> Query Transform (HyDE / Sub-query) ->
Dense + Sparse Retrieval -> Cross-encoder Reranking ->
Diversity Selection -> Generation -> Consistency Check
```

- Each stage is a node that reveals as you scroll, with connecting SVG paths drawing themselves between nodes (animated stroke-dashoffset)
- Each node displays the model/config underneath:
  - Dense Retrieval: `all-MiniLM-L6-v2`
  - Sparse Retrieval: `BM25`
  - Reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - Generation: LLM model from config
  - Top-K values: Dense=20, Sparse=20, Final=5
- Phase 2 features highlighted with emerald accent dots
- Session Memory and Coreference Resolution shown as a feedback loop arrow from Generation back to Query Transform

### Footer

- Minimal: "Sustainable Solutions Lab" credit, year, "Back to top" link
- Single row, generous padding, `border-t border-slate-200/50`

## Chat Page (`/chat`)

### Layout

- Full viewport, no landing page content visible
- Split: **60% conversation** (left) | **40% context panel** (right)
- Slim top bar: InfoWeave wordmark (links to `/`), session indicator, "New conversation" button
- Off-white background with 1px bottom border on top bar

### Conversation Area (Left 60%)

**Messages:**
- Clean vertical thread with generous padding
- User messages: right-aligned, emerald background, white text, `rounded-2xl` with tighter bottom-right corner
- Assistant messages: left-aligned, white background, slate text, `rounded-2xl` with tighter bottom-left corner
- Staggered fade-in on new messages with spring physics
- Auto-scroll to latest

**Loading state:**
- Skeleton loader matching assistant message shape
- Shimmer animation across the skeleton

**Empty state (no conversation yet):**
- InfoWeave mark centered in the conversation area
- "Ask about SSL's research, projects, or publications" subtitle
- 5 example questions as clickable chips with hover scale effect:
  1. "What is the Sustainable Solutions Lab?"
  2. "What is C3I and who funds it?"
  3. "Who leads SSL?"
  4. "Compare SSL's East Boston work with the harbor barrier study."
  5. "Has SSL published research on nuclear energy?"

### Input Bar (Bottom-fixed)

- Fixed to bottom of conversation area
- Large text input, `rounded-xl`, subtle inner shadow
- Send button: emerald, `scale-[0.98]` on `:active`
- Placeholder text cycles through example questions with typewriter effect (when empty state is showing)

### Context Panel (Right 40%)

Three collapsible sections using `border-t` dividers and spacing (no card containers per taste-skill anti-card rule for dense info):

**Retrieved Sources:**
- Each source as a compact row
- Phosphor icon by type: BookOpen (curated_qa), Globe (website), File (pdf)
- Source name, relevance score in `font-mono`, layer tag
- Chunk preview: 2-line truncation
- Emerald left-border for `qa_memory` layer, slate for corpus
- Placeholder when no query yet: "Ask a question to see sources"

**Retrieval Details:**
- Intent and strategy as inline code-style labels
- Sub-queries as a compact list
- HyDE and multi-hop indicators as small status pills
- Retrieved count summary

**Consistency Check:**
- Status: emerald checkmark (consistent) or amber warning (issues)
- Confidence as a mono number
- Unsupported claims as a compact list if present
- Explanation in italic

### Mobile Behavior

- Context panel collapses into a bottom sheet (slides up on tap)
- Conversation takes full width
- Input bar stays fixed at bottom

## Performance

Per taste-skill guardrails:
- All animations use `transform` and `opacity` only -- no animating `top`, `left`, `width`, `height`
- Perpetual micro-animations (metric pulse, shimmer) isolated in memoized client components (`React.memo`)
- Landing page sections are Server Components; animation wrappers are thin client component islands
- Scroll-triggered reveals use Framer Motion `useInView`, never `window.addEventListener('scroll')`
- `will-change: transform` used sparingly on animated elements
- Grain/noise overlays (if used) on `fixed inset-0 pointer-events-none` pseudo-elements only
- Z-index used strictly for systemic layers (navbar, modals)
- Video: native `<video>` element, no JS player library

## Data Flow

```
User types message
  -> InputBar.tsx sends POST /api/chat { message, session_id }
  -> FastAPI api.py calls ask() from rag_v2.pipeline (same function Gradio uses)
  -> Returns { answer, retrieved, retrieval_log, consistency }
  -> MessageList.tsx appends new messages
  -> SourcePanel.tsx updates with retrieved sources
  -> RetrievalDetails.tsx updates with log
  -> ConsistencyCheck.tsx updates with consistency result
```

Evaluation data:
```
Landing page mounts
  -> FeaturesBento.tsx calls GET /api/eval
  -> Returns metrics JSON from results/v2/*.json files
  -> Numbers animate on scroll-into-view
```

System/architecture data:
```
Landing page mounts
  -> PipelineShowcase.tsx calls GET /api/system
  -> Returns { version, llm_model, embed_model, sparse_model, reranker, top_k, features[] }
  -> Pipeline nodes render with real model names and config values
```

## What This Design Does NOT Change

- `src/rag_v2/app.py` -- Gradio UI stays untouched and runnable
- `src/rag_v2/pipeline.py` -- RAG pipeline logic unchanged
- `src/rag_v2/session.py` -- Session management unchanged
- `src/rag_v1/` -- V1 pipeline unchanged
- `data/` -- All data files unchanged
- `results/` -- All evaluation results unchanged
- No files are deleted, renamed, or moved
