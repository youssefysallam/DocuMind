"""
RAG V2 Gradio Web UI — multi-turn chatbot with source attribution.

Usage (from repo root, with PYTHONPATH=src):
    set PYTHONPATH=src
    python -m rag_v2.app
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

sys.stdout.reconfigure(encoding="utf-8")

_bu = os.environ.get("OPENAI_BASE_URL")
if _bu is not None and not str(_bu).strip():
    os.environ.pop("OPENAI_BASE_URL", None)

TOP_K_DENSE = 20
TOP_K_SPARSE = 20
TOP_K_FINAL = 5

LLM_DISPLAY = os.getenv("LLM_MODEL_V2") or os.getenv("LLM_MODEL", "openai/gpt-5.4")

# ── System loading ──────────────────────────────────────────────────

_SYSTEM: dict | None = None
_SESSIONS: dict[str, "SessionMemory"] = {}


def _load_system():
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from rag_v1.pipeline import load_all, openai_client, EMBED_MODEL_NAME, RERANK_MODEL_NAME

    print("[UI] Loading models and corpus …")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    reranker = CrossEncoder(RERANK_MODEL_NAME)
    client = openai_client()
    meta, ctx_idx, qa_items, qa_idx, bm25, dataset = load_all(embed_model)
    print("[UI] System ready.")
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


# ── Formatters ──────────────────────────────────────────────────────

def _format_sources(retrieved: list[dict]) -> str:
    if not retrieved:
        return "<p style='color:#999;text-align:center;padding:24px 0'>Ask a question to see sources</p>"
    icons = {"curated_qa": "\U0001f4da", "website": "\U0001f310", "pdf": "\U0001f4c4"}
    rows = []
    for i, c in enumerate(retrieved, 1):
        src = c.get("source", "?")
        sec = c.get("section_title", "")
        st = c.get("source_type", "")
        score = c.get("score", 0)
        layer = c.get("layer", "")
        icon = icons.get(st, "\U0001f4ce")
        preview = (c.get("chunk_text") or "")[:200].replace("\n", " ").strip()
        if len(preview) >= 200:
            preview += " ..."

        border_color = "#3b82f6" if layer == "qa_memory" else "#64748b"
        bg_color = "#eff6ff" if layer == "qa_memory" else "#f8fafc"
        sec_html = (
            f"<div style='color:#64748b;font-size:0.85em;margin-top:2px'>{sec}</div>"
            if sec else ""
        )
        rows.append(
            f"<div style='padding:8px 12px;margin-bottom:6px;"
            f"border-left:3px solid {border_color};"
            f"background:{bg_color};"
            f"border-radius:0 6px 6px 0;font-size:0.88em'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<span><b>{icon} {src}</b></span>"
            f"<span style='color:#94a3b8;font-size:0.82em'>{score:.3f} | {layer}</span>"
            f"</div>"
            f"{sec_html}"
            f"<div style='color:#475569;margin-top:4px;line-height:1.4'>{preview}</div>"
            f"</div>"
        )
    return "\n".join(rows)


def _format_retrieval_log(log: dict) -> str:
    intent = log.get("intent", "?")
    strategy = log.get("strategy", "")
    resolved = log.get("resolved_query", "")
    original = log.get("original_query", "")
    multihop = log.get("multihop", {})

    parts = [f"**Intent:** `{intent}`"]
    if strategy:
        parts.append(f"**Strategy:** `{strategy}`")
    if resolved and original and resolved != original:
        parts.append(f"**Coreference resolved:**\n> {resolved}")
    if log.get("sub_queries") and len(log["sub_queries"]) > 1:
        sqs = "  \n".join(f"• {sq}" for sq in log["sub_queries"])
        parts.append(f"**Sub-queries:**\n{sqs}")
    if log.get("hyde_used"):
        parts.append("**HyDE:** ✓ enabled")
    parts.append(
        f"**Retrieved:** {log.get('final_raw', '?')} corpus chunks, "
        f"{log.get('final_qa', '?')} QA entries"
    )
    if log.get("qa_neg_used"):
        parts.append("**No-evidence signal:** injected")
    if multihop.get("triggered"):
        parts.append(
            f"**Multi-hop补检索:** +{multihop['supplements_added']} chunks "
            f"(missing: {', '.join(multihop['missing_entities'])})"
        )
    return "\n\n".join(parts)


def _format_consistency(consistency: dict | None) -> str:
    if not consistency:
        return ""
    ok = consistency["is_consistent"]
    conf = consistency.get("confidence", 0)
    icon = "✅" if ok else "⚠️"
    text = f"{icon} **{'Consistent' if ok else 'Issues detected'}** — confidence {conf:.0%}"
    if consistency.get("unsupported_claims"):
        text += "\n\n**Unsupported claims:**\n"
        for claim in consistency["unsupported_claims"]:
            text += f"- {claim}\n"
    if consistency.get("explanation"):
        text += f"\n_{consistency['explanation']}_"
    return text


# ── Chat handler ────────────────────────────────────────────────────

def chat_fn(message: str, history: list, session_id: str):
    from rag_v2.pipeline import ask

    if not message.strip():
        return "", history, "", "", ""

    sys_data = get_system()
    session = _get_session(session_id)

    result = ask(
        message,
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

    answer = result["answer"]
    sources_md = _format_sources(result.get("retrieved", []))
    log_md = _format_retrieval_log(result.get("retrieval_log", {}))
    consistency_md = _format_consistency(result.get("consistency"))

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return "", history, sources_md, log_md, consistency_md


def clear_session(session_id: str):
    if session_id in _SESSIONS:
        _SESSIONS[session_id].clear()
    return [], "", "", ""


# ── Evaluation dashboard ────────────────────────────────────────────

def _load_eval_dashboard() -> str:
    comparison = PROJECT_ROOT / "results" / "v2" / "v1_vs_v2_metrics.json"
    mt_metrics = PROJECT_ROOT / "results" / "v2" / "multiturn_eval_metrics.json"
    v2_metrics = PROJECT_ROOT / "results" / "v2" / "rag_v2_metrics.json"

    lines: list[str] = []

    # V1 vs V2
    if comparison.exists():
        with open(comparison, "r", encoding="utf-8") as f:
            data = json.load(f)
        v1 = data.get("rag_v1_same_dataset_95") or data.get("rag_v1", {})
        v2 = data.get("rag_v2", {})
        v1_70 = data.get("rag_v1_baseline_70") or {}

        lines += [
            "## Single-turn: same 95-question set (V1 vs V2)\n",
            "| Metric | V1 | V2 | Δ |",
            "|--------|:--:|:--:|:-:|",
        ]
        for key, label in [
            ("raw_corpus_hit_at_3", "Hit@3"),
            ("raw_corpus_hit_at_5", "Hit@5"),
            ("answer_accuracy_answerable", "Accuracy (ans.)"),
            ("answer_accuracy_all", "Accuracy (all)"),
            ("answer_completeness", "Completeness"),
            ("hallucination_rate", "Hallucination ↓"),
            ("coverage", "Coverage"),
            ("correct_refusal_rate", "Correct refusal"),
            ("false_refusal_rate", "False refusal ↓"),
        ]:
            v1v = v1.get(key, 0)
            v2v = v2.get(key, 0)
            d = v2v - v1v
            sign = "+" if d > 0 else ""
            lines.append(f"| {label} | {v1v:.1%} | {v2v:.1%} | {sign}{d:.1%} |")

        if v2.get("by_question_type"):
            lines += ["\n### By Question Type\n",
                      "| Type | N | Behavior | Accuracy | Perfect |",
                      "|------|:-:|:--------:|:--------:|:-------:|"]
            for qt, v in v2["by_question_type"].items():
                lines.append(
                    f"| {qt} | {v['total']} | {v['behavior_rate']:.0%} "
                    f"| {v['avg_accuracy']:.0%} | {v['perfect']}/{v['total']} |"
                )

    # Multi-turn
    if mt_metrics.exists():
        with open(mt_metrics, "r", encoding="utf-8") as f:
            mt = json.load(f)
        o = mt.get("overall", {})
        cs = mt.get("coreference", {})
        by_turn = mt.get("by_turn_position", {})

        lines += [
            "\n---\n",
            "## Multi-turn Dialogue (15 questions, 5 chains)\n",
            "| Metric | Score / 2.0 |",
            "|--------|:-----------:|",
            f"| Factual Accuracy | **{o.get('factual_accuracy_avg', 0):.3f}** |",
            f"| Completeness | {o.get('completeness_avg', 0):.3f} |",
            f"| Appropriate Behavior | {o.get('appropriate_behavior_avg', 0):.3f} |",
            f"| Hallucination (2=none) | {o.get('hallucination_avg', 0):.3f} |",
        ]

        if by_turn:
            lines += [
                "\n### By Turn Position\n",
                "| Turn | Accuracy | Completeness | Hallucination |",
                "|------|:--------:|:------------:|:-------------:|",
            ]
            for tk in sorted(by_turn):
                t = by_turn[tk]
                lines.append(
                    f"| {tk} | {t['factual_accuracy_avg']:.3f} "
                    f"| {t['completeness_avg']:.3f} | {t['hallucination_avg']:.3f} |"
                )

        lines += [
            f"\n**Coreference resolution:** {cs.get('coreference_triggered', 0)}"
            f"/{cs.get('follow_up_questions', 0)} triggered "
            f"({cs.get('coreference_rate', 0):.0%})",
        ]

    if not lines:
        lines.append("_No evaluation results yet. Run the pipeline first._")

    return "\n".join(lines)


# ── Build UI ────────────────────────────────────────────────────────

CSS = """
/* Global */
.gradio-container { max-width: 1400px !important; }

/* Chat area */
#chat-col .chatbot { border-radius: 12px !important; }
#chat-col .chatbot .message { font-size: 0.95em !important; line-height: 1.6 !important; }

/* Input bar */
#input-row { margin-top: 4px !important; }
#input-row .textbox textarea { border-radius: 10px !important; font-size: 0.95em !important; }

/* Side panel */
#side-panel { padding-left: 8px; }
#side-panel .accordion { border-radius: 10px !important; margin-bottom: 6px !important; }

/* Source cards — handled inline in _format_sources */

/* Dashboard */
#dashboard-tab table { font-size: 0.9em; }
#dashboard-tab th { background: #f1f5f9 !important; }

/* System info pills */
.sys-pill {
    display: inline-block; padding: 3px 10px; margin: 2px 4px;
    border-radius: 20px; font-size: 0.82em; background: #f1f5f9; color: #334155;
}
"""


def build_ui():
    import uuid

    with gr.Blocks(
        title="SSL RAG V2 — InfoWeave",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css=CSS,
    ) as demo:
        session_id = gr.State(lambda: str(uuid.uuid4()))

        # Header
        gr.HTML(
            "<div style='text-align:center;padding:16px 0 8px'>"
            "<h1 style='margin:0;font-size:1.6em;font-weight:700'>"
            "🔬 SSL RAG V2 — InfoWeave"
            "</h1>"
            "<p style='margin:4px 0 0;color:#64748b;font-size:0.95em'>"
            "Sustainable Solutions Lab Research Assistant · Multi-turn Dialogue · Source Attribution"
            "</p>"
            "</div>"
        )

        with gr.Tabs():
            # ─── Chat Tab ───
            with gr.Tab("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=3, elem_id="chat-col"):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=520,
                            show_copy_button=True,
                            type="messages",
                        )
                        with gr.Row(elem_id="input-row"):
                            msg = gr.Textbox(
                                placeholder="Ask about SSL's research, projects, or publications …",
                                show_label=False,
                                scale=6,
                                container=False,
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)
                        with gr.Row():
                            clear_btn = gr.Button("🗑  New Chat", size="sm")

                    with gr.Column(scale=2, elem_id="side-panel"):
                        with gr.Accordion("📎 Retrieved Sources", open=True):
                            sources_display = gr.HTML(
                                value="<p style='color:#999;text-align:center;padding:24px 0'>"
                                      "Ask a question to see sources</p>",
                            )
                        with gr.Accordion("🔍 Retrieval Details", open=False):
                            retrieval_display = gr.Markdown(
                                value="*Retrieval details will appear here.*",
                            )
                        with gr.Accordion("✅ Consistency Check", open=False):
                            consistency_display = gr.Markdown(
                                value="*Consistency check results will appear here.*",
                            )

                gr.Markdown("#### 💡 Try these", elem_id="examples-header")
                examples = gr.Examples(
                    examples=[
                        ["What is the Sustainable Solutions Lab?"],
                        ["What is C3I and who funds it?"],
                        ["Who leads SSL?"],
                        ["Compare SSL's East Boston work with the harbor barrier study."],
                        ["Has SSL published research on nuclear energy?"],
                    ],
                    inputs=[msg],
                    label="",
                )

                send_btn.click(
                    fn=chat_fn,
                    inputs=[msg, chatbot, session_id],
                    outputs=[msg, chatbot, sources_display, retrieval_display, consistency_display],
                )
                msg.submit(
                    fn=chat_fn,
                    inputs=[msg, chatbot, session_id],
                    outputs=[msg, chatbot, sources_display, retrieval_display, consistency_display],
                )
                clear_btn.click(
                    fn=lambda sid: ("",) + clear_session(sid),
                    inputs=[session_id],
                    outputs=[msg, chatbot, sources_display, retrieval_display, consistency_display],
                )

            # ─── Dashboard Tab ───
            with gr.Tab("📊 Evaluation", elem_id="dashboard-tab"):
                dashboard_md = gr.Markdown(value=_load_eval_dashboard)
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
                refresh_btn.click(fn=_load_eval_dashboard, outputs=dashboard_md)

            # ─── System Info Tab ───
            with gr.Tab("⚙ System"):
                gr.HTML(
                    "<div style='max-width:680px;margin:0 auto;padding:16px 0'>"
                    "<h3 style='margin-bottom:12px'>System Configuration</h3>"
                    "<table style='width:100%;border-collapse:collapse;font-size:0.92em'>"
                    "<tr><td style='padding:6px 12px;color:#64748b'>Version</td>"
                    "<td style='padding:6px 12px'><b>RAG V2</b> — Phase 2</td></tr>"
                    f"<tr><td style='padding:6px 12px;color:#64748b'>LLM</td>"
                    f"<td style='padding:6px 12px'><code>{LLM_DISPLAY}</code></td></tr>"
                    "<tr><td style='padding:6px 12px;color:#64748b'>Dense Encoder</td>"
                    "<td style='padding:6px 12px'><code>all-MiniLM-L6-v2</code></td></tr>"
                    "<tr><td style='padding:6px 12px;color:#64748b'>Sparse Model</td>"
                    "<td style='padding:6px 12px'><code>BM25</code></td></tr>"
                    "<tr><td style='padding:6px 12px;color:#64748b'>Reranker</td>"
                    "<td style='padding:6px 12px'><code>cross-encoder/ms-marco-MiniLM-L-6-v2</code></td></tr>"
                    f"<tr><td style='padding:6px 12px;color:#64748b'>Top-K</td>"
                    f"<td style='padding:6px 12px'>Dense={TOP_K_DENSE} · Sparse={TOP_K_SPARSE} · Final={TOP_K_FINAL}</td></tr>"
                    "</table>"
                    "<h3 style='margin:20px 0 10px'>Phase 2 Features</h3>"
                    "<div>"
                    "<span class='sys-pill'>✓ HyDE</span>"
                    "<span class='sys-pill'>✓ Intent Classification</span>"
                    "<span class='sys-pill'>✓ Sub-query Decomposition</span>"
                    "<span class='sys-pill'>✓ Multi-hop Retrieval</span>"
                    "<span class='sys-pill'>✓ Diversity Selection</span>"
                    "<span class='sys-pill'>✓ Consistency Check</span>"
                    "<span class='sys-pill'>✓ Session Memory</span>"
                    "<span class='sys-pill'>✓ Coreference Resolution</span>"
                    "<span class='sys-pill'>✓ Multi-turn Dialogue</span>"
                    "</div>"
                    "</div>"
                )

    return demo


def main():
    demo = build_ui()
    print("\n" + "=" * 60)
    print("  SSL RAG V2 — Web UI")
    print("  http://localhost:7860")
    print("=" * 60 + "\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
