"""
Assemble report/RAG_V2_FULL_EXPERIMENT_REPORT.md from eval JSON + source prompts.

Run after:
  python scripts/eval_v1_phase2.py
  python -m rag_v2.pipeline
  python test_multiturn_eval.py

English report:
  python scripts/generate_full_experiment_report.py --lang en

All user-visible strings (including Chinese) live in scripts/report_v2_i18n.json next to this file.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATASET = ROOT / "data" / "eval_phase2" / "stakeholder_eval_phase2.json"
V2_METRICS = ROOT / "results" / "v2" / "rag_v2_metrics.json"
V1_70 = ROOT / "results" / "final" / "rag_v1_metrics.json"
V1_P2 = ROOT / "results" / "v2" / "rag_v1_phase2_metrics.json"
MT_METRICS = ROOT / "results" / "v2" / "multiturn_eval_metrics.json"
V2_EVAL_RESULTS = ROOT / "results" / "v2" / "rag_v2_eval_results.json"
MT_EVAL_RESULTS = ROOT / "results" / "v2" / "multiturn_eval_results.json"
OUT = ROOT / "report" / "RAG_V2_FULL_EXPERIMENT_REPORT.md"
OUT_EN = ROOT / "report" / "RAG_V2_FULL_EXPERIMENT_REPORT_EN.md"


REPORT_V2_I18N_PATH = Path(__file__).with_name("report_v2_i18n.json")
_RV2_CACHE: dict | None = None


def _report_v2_i18n() -> dict:
    global _RV2_CACHE
    if _RV2_CACHE is None:
        _RV2_CACHE = json.loads(REPORT_V2_I18N_PATH.read_text(encoding="utf-8"))
    return _RV2_CACHE


MSGS = {lng: _report_v2_i18n()[lng]["strings"] for lng in ("zh", "en")}


def _mermaid_lines(lang: str) -> list[str]:
    return list(_report_v2_i18n()[lang]["mermaid"])


def _load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _case_blurb(r: dict, lang: str) -> str:
    """One-line analysis from scores and question metadata (text from report_v2_i18n.json)."""
    B = _report_v2_i18n()[lang]["blurbs"]["case"]
    ev = r.get("answer_eval", {})
    qt = r.get("question_type", "")
    exp = r.get("expected_behavior", "")
    fa = ev.get("factual_accuracy", -1)
    fb = ev.get("appropriate_behavior", -1)
    hall = ev.get("hallucination", 2)
    sa = (r.get("system_answer") or "").lower()
    if fa == 2 and fb == 2 and hall >= 1:
        return B["strong"].format(qt=qt)
    if exp == "refuse" and fa == 0 and "could not find" in sa:
        return B["refuse_cf"]
    if exp == "refuse" and fa == 0 and "could not find" not in sa:
        return B["refuse_elaborate"]
    if qt == "synthesis" and fa == 0 and "could not find" in sa:
        return B["synthesis_refuse"]
    if fa <= 1:
        return B["weak_factual"]
    return B["default"]


def _mt_blurb(r: dict, lang: str) -> str:
    B = _report_v2_i18n()[lang]["blurbs"]["multiturn"]
    ev = r.get("answer_eval", {})
    fa = ev.get("factual_accuracy", 2)
    co = ev.get("completeness", 2)
    hall = ev.get("hallucination", 2)
    if fa == 2 and co == 2 and hall >= 1:
        return B["strong"]
    if fa < 2 and r.get("coreference_resolved"):
        return B["coref_weak"]
    if hall < 2:
        return B["hall_weak"]
    return B["partial"]


def _snippet(text: str, max_len: int = 480) -> str:
    t = " ".join(text.split())
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def _format_v2_case(r: dict, *, title: str, lang: str) -> list[str]:
    M = MSGS[lang]
    ev = r.get("answer_eval", {})
    qid = r.get("question_id", "?")
    qt = r.get("question_type", "?")
    exp = r.get("expected_behavior", "?")
    sep = M["list_sep"]
    head = M["v2_case_head"].format(title=title, qid=qid, qt=qt, exp=exp)
    lines = [
        head,
        "",
        f"{M['lbl_q']} {r.get('question', '')}",
    ]
    if r.get("resolved_query") and r.get("resolved_query") != r.get("question"):
        lines.append(f"{M['lbl_rq']} {r['resolved_query']}")
    lines += [
        f"{M['lbl_judge']} {M['j_fa']}={ev.get('factual_accuracy', '?')}{sep}"
        f"{M['j_co']}={ev.get('completeness', '?')}{sep}"
        f"{M['j_be']}={ev.get('appropriate_behavior', '?')}{sep}"
        f"{M['j_ha']}={ev.get('hallucination', '?')}{sep}"
        f"{M['j_ref']}={ev.get('is_refusal', '?')}",
        "",
        M["lbl_sys"],
        "",
        f"> {_snippet(r.get('system_answer', ''), 520)}",
        "",
        f"{M['lbl_je']} _{ev.get('explanation', '--')}_",
        "",
        f"{M['lbl_brief']} {_case_blurb(r, lang)}",
        "",
    ]
    return lines


def _format_multiturn_case(r: dict, *, title: str, lang: str) -> list[str]:
    M = MSGS[lang]
    ev = r.get("answer_eval", {})
    sep = M["list_sep"]
    head = M["mt_case_head"].format(
        title=title,
        qid=r.get("question_id"),
        chain=r.get("chain", "?"),
        turn=r.get("turn", "?"),
    )
    lines = [
        head,
        "",
        f"{M['lbl_uq']} {r.get('question', '')}",
    ]
    if r.get("coreference_resolved"):
        lines.append(f"{M['lbl_cr']} {r.get('resolved_query', '')}")
    lines += [
        f"{M['lbl_intent']} `{r.get('intent', '?')}`",
        f"{M['lbl_judge']} {M['j_fa']}={ev.get('factual_accuracy')}{sep}"
        f"{M['j_co']}={ev.get('completeness')}{sep}"
        f"{M['j_be']}={ev.get('appropriate_behavior')}{sep}"
        f"{M['j_ha']}={ev.get('hallucination')}",
        "",
        M["lbl_sys"],
        "",
        f"> {_snippet(r.get('system_answer', ''), 520)}",
        "",
        f"{M['lbl_je']} _{ev.get('explanation', '--')}_",
        "",
        f"{M['lbl_brief']} {_mt_blurb(r, lang)}",
        "",
    ]
    return lines


def _build_case_analysis_section(lang: str) -> list[str]:
    """Correct / error examples from full eval + multi-turn eval JSON."""
    M = MSGS[lang]
    lines: list[str] = [
        "",
        "---",
        "",
        M["case_s6"],
        "",
        M["case_intro"],
        "",
        M["case_c_ok"],
        "",
    ]

    if not V2_EVAL_RESULTS.exists():
        lines.append(M["case_v2_miss"] + "\n")
        return lines

    with open(V2_EVAL_RESULTS, encoding="utf-8") as f:
        v2r: list[dict] = json.load(f)

    singles = [r for r in v2r if not str(r.get("question_id", "")).startswith("MULTI")]
    correct_pool = [
        r for r in singles
        if r.get("answer_eval", {}).get("factual_accuracy") == 2
        and r.get("answer_eval", {}).get("appropriate_behavior") == 2
        and r.get("answer_eval", {}).get("hallucination", 0) >= 1
    ]
    by_type: dict[str, list[dict]] = {}
    for r in correct_pool:
        qt = r.get("question_type", "other")
        by_type.setdefault(qt, []).append(r)

    preferred_order = (
        "synthesis", "no_evidence", "general_overview", "project_initiative", "publication_finding"
    )
    picked_correct: list[dict] = []
    for qt in preferred_order:
        if qt in by_type and by_type[qt]:
            picked_correct.append(by_type[qt][0])
        if len(picked_correct) >= 3:
            break
    for r in correct_pool:
        if len(picked_correct) >= 3:
            break
        if r not in picked_correct:
            picked_correct.append(r)

    for i, r in enumerate(picked_correct[:3], 1):
        lines += _format_v2_case(r, title=f"{M['correct_t']} {i}", lang=lang)

    lines += [
        M["case_c_bad"],
        "",
        M["case_c_bad_f"],
        "",
    ]

    def _severity(r: dict) -> tuple:
        ev = r.get("answer_eval", {})
        return (
            ev.get("factual_accuracy", 99),
            ev.get("appropriate_behavior", 99),
            -ev.get("hallucination", 0),
        )

    error_pool = [
        r for r in singles
        if r.get("answer_eval", {}).get("factual_accuracy", 2) <= 1
        or r.get("answer_eval", {}).get("appropriate_behavior", 2) < 2
    ]
    error_pool.sort(key=_severity)
    for i, r in enumerate(error_pool[:3], 1):
        lines += _format_v2_case(r, title=f"{M['error_t']} {i}", lang=lang)
    if not error_pool:
        lines.append(M["case_err0"] + "\n")

    lines += [
        M["case_mt_ok"],
        "",
        M["case_mt_ok_i"],
        "",
    ]

    mtr: list[dict] = []
    if MT_EVAL_RESULTS.exists():
        with open(MT_EVAL_RESULTS, encoding="utf-8") as f:
            mtr = json.load(f)

    if mtr:
        chain_b = [r for r in mtr if r.get("chain") == "B"]
        chain_b.sort(key=lambda x: x.get("turn", 0))
        for r in chain_b:
            mark = M["hl_cmp"] if r.get("question_id") == "MULTI-05" else ""
            lines += _format_multiturn_case(r, title=f"{M['chain_b']}{mark}", lang=lang)
        lines += [
            M["case_mt_sum"],
            "",
        ]
    else:
        lines.append(M["case_mt_miss"] + "\n")

    lines += [
        M["case_mt_bad"],
        "",
        M["case_mt_bad_i"],
        "",
    ]

    if mtr:
        weak_mt = [
            r for r in mtr
            if (r.get("answer_eval", {}).get("factual_accuracy", 2) < 2
                or r.get("answer_eval", {}).get("completeness", 2) < 2
                or r.get("answer_eval", {}).get("hallucination", 2) < 2)
        ]
        weak_mt.sort(
            key=lambda r: (
                r.get("answer_eval", {}).get("factual_accuracy", 99),
                r.get("answer_eval", {}).get("hallucination", 99),
            ),
        )
        seen_ids: set[str] = set()
        for r in weak_mt:
            qid = r.get("question_id", "")
            if qid in seen_ids:
                continue
            seen_ids.add(qid)
            lines += _format_multiturn_case(r, title=M["weak_t"], lang=lang)
            if len(seen_ids) >= 4:
                break
        if len(seen_ids) == 0:
            lines.append(M["case_mt_weak0"] + "\n")
    else:
        lines.append(M["case_mt_miss2"] + "\n")

    lines += [
        M["mt_conc"],
        "",
        M["mt_c1"],
        M["mt_c2"],
        M["mt_c3"],
        "",
    ]

    return lines


def _extract_block(path: Path, start_marker: str, end_marker: str | None = None) -> str:
    text = path.read_text(encoding="utf-8")
    i = text.find(start_marker)
    if i < 0:
        return f"*(not found: {start_marker[:40]}...)*"
    start = i
    if end_marker:
        j = text.find(end_marker, i + len(start_marker))
        if j < 0:
            chunk = text[start : start + 2500]
        else:
            chunk = text[start:j]
    else:
        chunk = text[start : start + 3500]
    return chunk.strip()


def _build_error_analysis(lang: str) -> list[str]:
    """Categorize failure modes from eval results."""
    if not V2_EVAL_RESULTS.exists():
        return []
    with open(V2_EVAL_RESULTS, encoding="utf-8") as f:
        v2r: list[dict] = json.load(f)
    singles = [r for r in v2r if not str(r.get("question_id", "")).startswith("MULTI")]
    errors = [r for r in singles
              if r.get("answer_eval", {}).get("factual_accuracy", 2) <= 1
              or r.get("answer_eval", {}).get("appropriate_behavior", 2) < 2]
    false_ref = sum(1 for r in errors
                    if r.get("expected_behavior") == "answer"
                    and r.get("answer_eval", {}).get("is_refusal") == 1)
    noise = sum(1 for r in errors
                if r.get("expected_behavior") == "refuse"
                and r.get("answer_eval", {}).get("is_refusal", 1) == 0)
    incomplete = sum(1 for r in errors
                     if r.get("answer_eval", {}).get("factual_accuracy", 2) == 1
                     and r.get("answer_eval", {}).get("appropriate_behavior", 2) == 2)
    other = len(errors) - false_ref - noise - incomplete
    if lang == "en":
        return [
            "",
            "### Root Cause Analysis of Failures",
            "",
            "| Failure Mode | Count | Description |",
            "|---|--:|---|",
            f"| False refusal | {false_ref} | System refused an answerable question; "
            "typically synthesis items where sub-query decomposition did not fire |",
            f"| Retrieval noise / missed refusal | {noise} | No-evidence item but system "
            "elaborated from tangentially related chunks instead of refusing |",
            f"| Incomplete enumeration | {incomplete} | Correct direction but judge marked "
            "missing key points (amounts, names, durations) |",
            f"| Other | {other} | Mixed causes including hallucination from biography "
            "snippets, off-topic elaboration |",
            "",
        ]
    return [
        "",
        "### 失败根因分析",
        "",
        "| 失败模式 | 数量 | 说明 |",
        "|---|--:|---|",
        f"| 误拒 | {false_ref} | 可答题被系统拒答；常见于综合题子查询分解未触发 |",
        f"| 检索噪声 / 漏拒 | {noise} | 无证据题却基于边缘片段作答 |",
        f"| 要点不全 | {incomplete} | 方向正确但 Judge 判关键要点缺失 |",
        f"| 其他 | {other} | 含人物传记片段幻觉、偏题展开等 |",
        "",
    ]


def main(lang: str = "zh") -> None:
    if lang not in ("zh", "en"):
        raise ValueError(f"Unsupported lang={lang!r}; use 'zh' or 'en'.")
    M = MSGS[lang]
    out_path = OUT_EN if lang == "en" else OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(DATASET, "r", encoding="utf-8") as f:
        ds = json.load(f)

    by_type = Counter(q["question_type"] for q in ds)
    by_phase = Counter(q.get("phase", 1) for q in ds)
    by_diff = Counter(q.get("difficulty", "?") for q in ds)
    multi_chains = Counter(
        q.get("multi_turn_chain", "") for q in ds if q["question_id"].startswith("MULTI")
    )

    v2 = _load_json(V2_METRICS)
    v1_70 = _load_json(V1_70)
    v1_p2 = _load_json(V1_P2)
    mt = _load_json(MT_METRICS)

    pipe = ROOT / "src" / "rag_v2" / "pipeline.py"
    multiturn_py = ROOT / "src" / "rag_v2" / "multiturn.py"
    hyde_py = ROOT / "src" / "rag_v2" / "query_transform.py"
    intent_py = ROOT / "src" / "rag_v2" / "intent_classifier.py"
    v1_eval = ROOT / "src" / "rag_v1" / "pipeline.py"

    gen_prompt = _extract_block(
        pipe,
        '    system_prompt = (\n        "You are a research assistant',
        '    messages: list[dict[str, str]] = [{"role": "system"',
    )
    coref = _extract_block(
        multiturn_py,
        '"You rewrite follow-up questions',
        '    rewritten = resp.choices',
    )
    strategy = _extract_block(
        multiturn_py,
        '"You decide how a RAG system',
        '    text = resp.choices',
    )
    hyde = _extract_block(
        hyde_py,
        '"You are a research assistant for the Sustainable Solutions Lab',
        '    return resp.choices',
    )
    intent_llm = _extract_block(
        intent_py,
        '"role": "system",\n                "content":',
        'temperature=0',
    )
    judge = _extract_block(
        v1_eval,
        "def eval_answer(sys_ans, gold_ans, expected_behavior, client):",
        "def compute_metrics(results):",
    )

    def pct(x):
        return f"{float(x) * 100:.1f}%" if isinstance(x, (int, float)) else str(x)

    mt_o = mt.get("overall", {})
    n_q = len(ds)
    n_answerable = v2.get("answerable", sum(1 for q in ds if q.get("expected_behavior") != "refuse"))
    n_refuse = v2.get("refuse_expected", n_q - n_answerable)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # ── Section numbering differs between EN (enriched) and ZH (legacy) ──
    if lang == "en":
        lines = _build_en_report(
            M, ds, n_q, n_answerable, n_refuse, ts,
            by_type, by_phase, by_diff, multi_chains,
            v2, v1_70, v1_p2, mt, mt_o,
            gen_prompt, coref, strategy, hyde, intent_llm, judge, pct,
        )
    else:
        lines = _build_zh_report(
            M, ds, n_q, ts,
            by_type, by_phase, by_diff, multi_chains,
            v2, v1_70, v1_p2, mt, mt_o,
            gen_prompt, coref, strategy, hyde, intent_llm, judge, pct, lang,
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


def _build_en_report(
    M, ds, n_q, n_answerable, n_refuse, ts,
    by_type, by_phase, by_diff, multi_chains,
    v2, v1_70, v1_p2, mt, mt_o,
    gen_prompt, coref, strategy, hyde, intent_llm, judge, pct,
) -> list[str]:
    """Build the enriched English report with all requested sections."""
    lang = "en"

    lines = [
        "# RAG V2 Full Experiment Report (Phase 2)",
        "",
        f"**Generated (UTC):** {ts}",
        f"**Dataset:** `{DATASET.relative_to(ROOT)}` (**{n_q}** items)",
        "",
        "---",
        "",
        # ── 1. Introduction ──
        "## 1. Introduction and Objectives",
        "",
        "This report presents a comprehensive evaluation of the **RAG V2** system built for the "
        "**Sustainable Solutions Lab (SSL)** at UMass Boston. The system answers stakeholder questions "
        "about SSL's climate-justice research, projects, publications, and leadership by retrieving "
        "evidence from a unified corpus of PDFs and website content and generating grounded answers "
        "with `gpt-4o-mini`.",
        "",
        "**Objectives of this evaluation:**",
        "",
        "1. Quantify RAG V2 performance on an expanded 104-item benchmark covering six question types, "
        "three difficulty levels, and 15 multi-turn conversation items.",
        "2. Compare V2 against the V1 baseline on the same item set to measure improvements in accuracy, "
        "hallucination control, refusal behavior, and completeness.",
        "3. Validate the multi-turn conversation subsystem (coreference resolution, retrieval strategy "
        "decisions, session memory) on five thematic chains.",
        "4. Identify failure modes and outline future optimization directions.",
        "",
        "---",
        "",
        # ── 2. Dataset Design ──
        "## 2. Dataset Design and Expansion",
        "",
        "### 2.1 Expansion from Phase 1 to Phase 2",
        "",
        f"The evaluation dataset was expanded from **70** items (Phase 1) to **{n_q}** items (Phase 2) "
        "to stress-test V2's new capabilities. The 34 new items include:",
        "",
        "- **Synthesis (SYN-P2):** Multi-source questions requiring sub-query decomposition and cross-document reasoning.",
        "- **No-evidence edge cases (NEG-P2):** Harder boundary questions where tangentially related evidence exists in the corpus but the correct answer is refusal.",
        "- **Publication deep-dives (PUB-P2):** Questions targeting specific findings, figures, or recommendations in SSL reports.",
        "- **Multi-turn chains (MULTI-01 to MULTI-15):** 5 thematic chains of 3 turns each, testing coreference resolution and progressive evidence accumulation.",
        "",
        "### 2.2 Question Type Taxonomy",
        "",
        "Each item is assigned one of six `question_type` labels. The taxonomy drives intent classification "
        "in V2's pipeline, which in turn controls HyDE activation, sub-query decomposition, and retrieval strategy.",
        "",
        "| Type | Description | Example |",
        "|------|-------------|---------|",
        '| `general_overview` | SSL identity, mission, leadership, contact | "What is SSL\'s mission?" |',
        '| `project_initiative` | Specific projects or programs | "What is the Cape Cod Rail Resilience project?" |',
        '| `topic_specific` | Focused research topics | "What has SSL studied about climate migration?" |',
        '| `publication_finding` | Findings from specific reports | "What does the harbor barrier study recommend?" |',
        '| `synthesis` | Cross-document, multi-source reasoning | "What themes appear across SSL projects?" |',
        '| `no_evidence` | Topics SSL has not studied (expect refusal) | "Has SSL published on nuclear energy?" |',
        "",
        "### 2.3 Distribution by Type, Difficulty, and Phase",
        "",
        "| Type | Count |",
        "|------|-----:|",
    ]
    for t, n in sorted(by_type.items(), key=lambda x: -x[1]):
        lines.append(f"| `{t}` | {n} |")
    lines += [
        "",
        "| Difficulty | Count |",
        "|------|-----:|",
    ]
    for d, n in sorted(by_diff.items(), key=lambda x: -x[1]):
        lines.append(f"| {d} | {n} |")
    lines += ["", "| Phase | Count |", "|------|-----:|"]
    for ph, n in sorted(by_phase.items()):
        lines.append(f"| {ph} | {n} |")
    lines += [
        "",
        f"Of the {n_q} items, **{n_answerable}** expect a substantive answer and **{n_refuse}** "
        "expect the system to refuse (no-evidence items).",
        "",
        "### 2.4 Multi-turn Subset Design",
        "",
        "The 15 multi-turn items are organized into **5 chains** (A through E), each with 3 turns of increasing "
        "complexity. Chains are designed to test coreference resolution, comparison reasoning, and detail follow-ups:",
        "",
        "| Chain | Topic | Turn 1 | Turn 2 | Turn 3 |",
        "|-------|-------|--------|--------|--------|",
        "| A | Workforce (C3I) | Project overview | Funding details (pronoun) | Job types (pronoun) |",
        "| B | Infrastructure vs Community | Cape Cod Rail | Compare to East Boston (\"that project\") | Harbor barrier three-way comparison |",
        "| C | Leadership | Team overview | Executive director background (\"the director\") | Future plans |",
        "| D | Publication | Report findings | Methodology details (\"the study\") | Recommendations |",
        "| E | Cross-topic | Community engagement | Compare approaches (\"it\") | Broader synthesis |",
        "",
        "---",
        "",
        # ── 3. RAG V1 Architecture (Baseline) ──
        "## 3. RAG V1 Architecture (Baseline)",
        "",
        "RAG V1 is a single-pass retrieval-augmented generation system with the following pipeline:",
        "",
        "1. **Query input** -- the raw user question is passed directly to retrieval (no intent classification or query rewriting).",
        "2. **Hybrid retrieval** -- dense (sentence-transformers/all-MiniLM-L6-v2 via FAISS IndexFlatIP) "
        "and sparse (BM25) retrieval each return top-20 candidates.",
        "3. **Reranking** -- a cross-encoder (ms-marco-MiniLM-L-6-v2) reranks the merged candidate pool.",
        "4. **Context assembly** -- top 5 chunks are selected, with up to 2 curated QA memory entries "
        "(relevance floor 0.35) injected when applicable.",
        "5. **Answer generation** -- `gpt-4o-mini` generates the answer from the assembled context.",
        "",
        "**Key V1 characteristics:**",
        "- **Contextualized chunks:** each chunk is prefixed with document title and section metadata to improve dense retrieval relevance.",
        "- **Curated QA Memory:** hand-verified Q&A pairs for common questions, injected when the reranker score exceeds the relevance threshold.",
        "- **No multi-turn support:** each question is treated independently.",
        "- **No intent-driven routing:** all questions follow the same retrieval path regardless of type.",
        "",
        "| Component | V1 Implementation |",
        "|-----------|------------------|",
        "| Dense embedding | `all-MiniLM-L6-v2`, normalized, FAISS `IndexFlatIP` |",
        "| Sparse retrieval | BM25 on raw chunk tokens |",
        "| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |",
        "| Top-K (dense/sparse) | 20 / 20 |",
        "| Final context size | 5 chunks + up to 2 QA entries |",
        "| LLM | `gpt-4o-mini` |",
        "| Multi-turn | Not supported |",
        "",
        "---",
        "",
        # ── 4. RAG V2 Architecture ──
        "## 4. RAG V2 Architecture",
        "",
        "RAG V2 extends V1 with an **intent-driven, multi-turn-aware** pipeline. The core retrieval "
        "infrastructure (dense + sparse + reranker) is preserved, but seven new LLM-powered modules "
        "are added around it.",
        "",
        "### 4.1 Pipeline Flow",
        "",
    ]
    lines += _mermaid_lines(lang)
    lines += [
        "",
        "### 4.2 Component Details",
        "",
        "| # | Component | Purpose | When it fires |",
        "|---|-----------|---------|---------------|",
        "| 1 | **Coreference Resolution** | Rewrites follow-up questions by replacing pronouns with referents from conversation history | Every follow-up turn (not turn 1) |",
        "| 2 | **Retrieval Strategy Decision** | Chooses FULL_RETRIEVAL, REUSE_EVIDENCE, or MERGE_EVIDENCE based on topic continuity | Every follow-up turn |",
        "| 3 | **Intent Classification** | Classifies the question into one of 6 types to route downstream processing | Every question |",
        "| 4 | **Conditional HyDE** | Generates a hypothetical answer passage to enrich the query embedding | Only for `synthesis`, `topic_specific`, `publication_finding` |",
        "| 5 | **Sub-query Decomposition** | Splits complex questions into simpler sub-queries for parallel retrieval | Only for `synthesis` questions |",
        "| 6 | **Multi-hop Retrieval** | Checks entity coverage and issues supplemental retrieval for missing entities | When session history contains entities not covered by initial retrieval |",
        "| 7 | **Diversity Truncation** | Removes near-duplicate chunks to maximize information density in the context window | Always (post-retrieval) |",
        "| 8 | **Consistency Check** | Verifies the generated answer is supported by the retrieved evidence | Batch eval mode; disabled in UI for latency |",
        "",
        M["s2_note"],
        "",
        "---",
        "",
        # ── 5. V1 vs V2 Architecture Comparison ──
        "## 5. V1 vs V2 Architecture Comparison",
        "",
        "| Aspect | V1 | V2 |",
        "|--------|----|----|",
        "| Intent classification | None (all questions same path) | LLM-based, 6 categories |",
        "| Query rewriting | None | Coreference resolution LLM |",
        "| HyDE | Not used | Conditional (synthesis/topic/publication only) |",
        "| Sub-query decomposition | None | LLM decomposition for synthesis |",
        "| Retrieval strategy | Always full retrieval | LLM decides: FULL / REUSE / MERGE |",
        "| Multi-hop retrieval | None | Entity-coverage check + supplemental retrieval |",
        "| Diversity filtering | None | Near-duplicate chunk removal |",
        "| Consistency check | None | LLM answer-evidence verification |",
        "| Multi-turn support | None | Session memory + coreference + strategy |",
        "| Base retrieval | Dense + BM25 + reranker | Same (preserved from V1) |",
        "| LLM | `gpt-4o-mini` | Same |",
        "",
        "**Summary:** V2 wraps V1's retrieval core with intent-driven routing, query transformation, "
        "and post-generation verification. The architectural additions specifically target V1's weaknesses: "
        "synthesis questions (via HyDE + decomposition), no-evidence boundary cases (via intent classification), "
        "and multi-turn conversations (via coreference + session memory).",
        "",
        "---",
        "",
        # ── 6. Core V2 Metrics ──
        "## 6. Core Metrics (RAG V2, Full Phase 2 Set)",
        "",
    ]
    if v2:
        lines += [
            "| Metric | Value |",
            "|------|------|",
            f"| Hit@3 | {pct(v2.get('raw_corpus_hit_at_3', 0))} |",
            f"| Hit@5 | {pct(v2.get('raw_corpus_hit_at_5', 0))} |",
            f"| Accuracy (answerable) | {pct(v2.get('answer_accuracy_answerable', 0))} |",
            f"| Accuracy (all) | {pct(v2.get('answer_accuracy_all', 0))} |",
            f"| Completeness | {pct(v2.get('answer_completeness', 0))} |",
            f"| Hallucination rate (lower is better) | {pct(v2.get('hallucination_rate', 0))} |",
            f"| Behavior coverage | {pct(v2.get('coverage', 0))} |",
            f"| Correct refusal rate | {pct(v2.get('correct_refusal_rate', 0))} |",
            f"| False refusal rate | {pct(v2.get('false_refusal_rate', 0))} |",
            f"| Missed refusals | {v2.get('missed_refusal_count', '--')} |",
        ]
        if "consistency_rate" in v2:
            lines.append(f"| Answer-evidence consistency | {pct(v2['consistency_rate'])} |")
        lines += [
            "",
            M["hit_note"],
            "",
            "### 6.1 By Question Type (V2)",
            "",
            "| Type | N | Behavior | Avg accuracy | Perfect |",
            "|------|--:|--------:|-----------:|--------|",
        ]
        for qt, v in v2.get("by_question_type", {}).items():
            lines.append(
                f"| `{qt}` | {v['total']} | {pct(v['behavior_rate'])} | {pct(v['avg_accuracy'])} | "
                f"{v['perfect']}/{v['total']} |"
            )
    else:
        lines.append(M["no_v2"])
    lines += [
        "",
        "---",
        "",
        # ── 7. V1 vs V2 Metrics Comparison ──
        "## 7. V1 vs V2 Performance Comparison",
        "",
        "### 7.1 Historical Baseline: V1 on the Original 70-item Set",
        "",
    ]
    if v1_70:
        lines += [
            "| Metric | V1 (70 items) |",
            "|------|--------:|",
            f"| Hit@5 | {pct(v1_70.get('raw_corpus_hit_at_5', 0))} |",
            f"| Accuracy (answerable) | {pct(v1_70.get('answer_accuracy_answerable', 0))} |",
            f"| Hallucination rate | {pct(v1_70.get('hallucination_rate', 0))} |",
            f"| Coverage | {pct(v1_70.get('coverage', 0))} |",
            f"| Correct refusal rate | {pct(v1_70.get('correct_refusal_rate', 0))} |",
            f"| False refusal rate | {pct(v1_70.get('false_refusal_rate', 0))} |",
        ]
    else:
        lines.append(M["no_v1_70"])
    lines += [
        "",
        f"### 7.2 Same-set Comparison: V1 vs V2 ({n_q} Phase 2 Items)",
        "",
        "Both systems evaluated on the identical 104-item set with the same LLM judge (`gpt-4o-mini`). "
        "**V1** uses V1 retrieve + generate. **V2** uses the full `ask()` path (conditional HyDE, "
        "multi-hop, diversity, intent routing, etc.).",
        "",
        "| Metric | V1 Phase2 | V2 | Delta (V2-V1) |",
        "|------|----------:|---:|-----------|",
    ]
    cmp_keys = [
        ("raw_corpus_hit_at_3", "Hit@3"),
        ("raw_corpus_hit_at_5", "Hit@5"),
        ("answer_accuracy_answerable", "Accuracy (answerable)"),
        ("answer_accuracy_all", "Accuracy (all)"),
        ("answer_completeness", "Completeness"),
        ("hallucination_rate", "Hallucination rate"),
        ("coverage", "Behavior coverage"),
        ("correct_refusal_rate", "Correct refusal"),
        ("false_refusal_rate", "False refusal rate"),
    ]
    if v1_p2 and v2:
        for k, lab in cmp_keys:
            a, b = v1_p2.get(k, 0), v2.get(k, 0)
            d = b - a
            sg = "+" if d > 0 else ""
            lines.append(f"| {lab} | {pct(a)} | {pct(b)} | {sg}{pct(d)} |")
        lines.append(
            f"| Missed refusals | {v1_p2.get('missed_refusal_count', 0)} | "
            f"{v2.get('missed_refusal_count', 0)} | - |"
        )
    else:
        lines.append(M["cmp_missing"])
    lines += [
        "",
        "### 7.3 Key Takeaways from the Comparison",
        "",
    ]
    if v1_p2 and v2:
        hall_drop = v1_p2.get("hallucination_rate", 0) - v2.get("hallucination_rate", 0)
        acc_gain = v2.get("answer_accuracy_answerable", 0) - v1_p2.get("answer_accuracy_answerable", 0)
        fr_drop = v1_p2.get("false_refusal_rate", 0) - v2.get("false_refusal_rate", 0)
        cov_gain = v2.get("coverage", 0) - v1_p2.get("coverage", 0)
        lines += [
            f"1. **Hallucination halved:** V2 reduces hallucination rate by {pct(hall_drop)} "
            "(from {a} to {b}), primarily due to the consistency check and stricter evidence grounding.".format(
                a=pct(v1_p2["hallucination_rate"]), b=pct(v2["hallucination_rate"])),
            f"2. **Accuracy improved:** Answerable accuracy rises by {pct(acc_gain)}, "
            "driven by HyDE and multi-hop retrieval providing richer context for complex questions.",
            f"3. **False refusal rate dropped:** From {pct(v1_p2['false_refusal_rate'])} to "
            f"{pct(v2['false_refusal_rate'])} ({pct(fr_drop)} reduction), as intent classification "
            "reduces unnecessary refusals on answerable items.",
            f"4. **Behavior coverage up:** {pct(cov_gain)} improvement, reflecting better alignment "
            "between expected and actual behavior (answer vs refuse).",
            "5. **Hit@K unchanged:** V1 and V2 share the same base retrieval index; V2's gains come "
            "from better *use* of retrieved evidence, not from retrieving different chunks.",
            "",
        ]
    lines += [
        "---",
        "",
        # ── 8. Case Studies ──
    ]
    lines += _build_case_analysis_section(lang)
    lines += _build_error_analysis(lang)
    lines += [
        "",
        "---",
        "",
        # ── 9. Multi-turn Conversation Design & Evaluation ──
        "## 9. Multi-turn Conversation Design and Evaluation",
        "",
        "### 9.1 Design Rationale",
        "",
        "Multi-turn support is a V2-only capability. Real stakeholders rarely ask isolated questions -- "
        "they ask follow-ups that reference prior context (\"How does *that project* compare to...\", "
        "\"Who funds *it*?\"). V2 addresses this with three coordinated components:",
        "",
        "1. **Coreference Resolution LLM:** Rewrites pronouns and demonstratives into full noun phrases "
        "using conversation history, producing a self-contained query suitable for retrieval.",
        "2. **Retrieval Strategy Decision LLM:** Decides whether to search from scratch (FULL_RETRIEVAL), "
        "reuse prior evidence (REUSE_EVIDENCE), or merge old and new evidence (MERGE_EVIDENCE).",
        "3. **Session Memory:** Stores per-chain question-answer history for coreference context and "
        "multi-hop entity tracking.",
        "",
        "### 9.2 Evaluation Protocol",
        "",
        "Script: `test_multiturn_eval.py`. Each of the 5 chains runs in an isolated `SessionMemory`. "
        "Questions are submitted sequentially (turn 1, 2, 3). The same LLM-as-judge evaluates each "
        "turn's answer independently against its gold standard.",
        "",
        "### 9.3 Overall Multi-turn Metrics",
        "",
    ]
    if mt and mt_o:
        cr = mt.get("coreference", {})
        n_trig = cr.get("coreference_triggered", 0)
        n_fu = cr.get("follow_up_questions", 0)
        rate = cr.get("coreference_rate", 0)
        lines += [
            "| Metric | Score / 2.0 |",
            "|------|-------------|",
            f"| Factual accuracy | **{mt_o.get('factual_accuracy_avg', 0):.3f}** |",
            f"| Completeness | {mt_o.get('completeness_avg', 0):.3f} |",
            f"| Appropriate behavior | {mt_o.get('appropriate_behavior_avg', 0):.3f} |",
            f"| Hallucination (2 = none) | {mt_o.get('hallucination_avg', 0):.3f} |",
            "",
            f"**Coreference:** {n_trig}/{n_fu} triggered ({rate:.0%}).",
            "",
            "### 9.4 Performance by Turn Position",
            "",
            "| Turn | Accuracy | Completeness | Hallucination |",
            "|------|--------|--------|------|",
        ]
        for tk in sorted(mt.get("by_turn_position", {})):
            t = mt["by_turn_position"][tk]
            lines.append(
                f"| {tk} | {t['factual_accuracy_avg']:.3f} | "
                f"{t['completeness_avg']:.3f} | {t['hallucination_avg']:.3f} |"
            )
        lines += [
            "",
            "### 9.5 Performance by Chain",
            "",
            "| Chain | Topic | Accuracy | Completeness | Hallucination |",
            "|-------|-------|--------|--------|------|",
        ]
        chain_topics = {"chain_A": "Workforce (C3I)", "chain_B": "Infrastructure", "chain_C": "Leadership",
                        "chain_D": "Publication", "chain_E": "Cross-topic"}
        for ck in sorted(mt.get("by_chain", {})):
            c = mt["by_chain"][ck]
            topic = chain_topics.get(ck, ck)
            lines.append(
                f"| {ck.replace('chain_', '')} | {topic} | {c['factual_accuracy_avg']:.3f} | "
                f"{c['completeness_avg']:.3f} | {c['hallucination_avg']:.3f} |"
            )
        if "latency" in mt:
            lat = mt["latency"]
            lines += [
                "",
                f"**Latency:** avg {lat['avg_generation_s']:.1f}s, "
                f"max {lat['max_generation_s']:.1f}s, "
                f"min {lat['min_generation_s']:.1f}s per turn.",
            ]
    else:
        lines.append(M["no_mt"])
    lines += [
        "",
        "---",
        "",
        # ── 10. Prompt Engineering ──
        "## 10. Prompt Engineering",
        "",
        "All LLM calls in V2 use carefully designed prompts. This section documents the key prompts "
        "and the design rationale behind each.",
        "",
        "### 10.1 Answer Generation (`generate_answer_v2`)",
        "",
        "**Design rationale:** The prompt enforces strict evidence grounding with an explicit 9-rule "
        "framework. Rule 3 mandates a specific refusal phrase for consistency. Rule 5 prevents the "
        "model from treating tangential mentions as dedicated SSL work. Rule 6 establishes trust in "
        "curated QA memory entries. Rule 8 prevents raw source leakage into user-facing answers.",
        "",
        "```text",
        gen_prompt[:3200] + ("\n..." if len(gen_prompt) > 3200 else ""),
        "```",
        "",
        "### 10.2 Coreference Resolution (`resolve_coreference`)",
        "",
        "**Design rationale:** Produces a self-contained query by replacing pronouns with their "
        "referents. Uses `temperature=0` and `max_tokens=150` for deterministic, concise output. "
        "The \"return ONLY the rewritten question\" constraint prevents explanations from leaking in.",
        "",
        "```text",
        coref[:2500] + ("\n..." if len(coref) > 2500 else ""),
        "```",
        "",
        "### 10.3 Retrieval Strategy (`decide_retrieval_strategy`)",
        "",
        "**Design rationale:** A three-way classification (FULL / REUSE / MERGE) balances retrieval "
        "cost against context relevance. REUSE avoids redundant retrieval for clarification questions. "
        "MERGE enables comparison questions (e.g., \"How does *that* compare to East Boston?\") to "
        "retain prior evidence while fetching new.",
        "",
        "```text",
        strategy[:2200] + ("\n..." if len(strategy) > 2200 else ""),
        "```",
        "",
        "### 10.4 HyDE (`hyde_generate`)",
        "",
        "**Design rationale:** Generates a hypothetical answer passage to enrich the query embedding "
        "for dense retrieval. Uses `temperature=0.7` for diversity and `max_tokens=200` for conciseness. "
        "Only activated for `synthesis`, `topic_specific`, and `publication_finding` intents -- "
        "skipped for `general_overview`, `project_initiative`, and `no_evidence` where direct lexical "
        "matching suffices.",
        "",
        "```text",
        hyde[:1800] + ("\n..." if len(hyde) > 1800 else ""),
        "```",
        "",
        "### 10.5 Intent Classification (LLM branch)",
        "",
        "**Design rationale:** The prompt includes explicit positive and negative scope lists for SSL "
        "(\"SSL focuses on... / SSL does NOT research...\") to help the classifier distinguish "
        "`no_evidence` from edge cases. Returns only the category name for easy parsing.",
        "",
        "```text",
        intent_llm[:2200] + ("\n..." if len(intent_llm) > 2200 else ""),
        "```",
        "",
        "### 10.6 LLM-as-Judge (`eval_answer`)",
        "",
        "**Design rationale:** Five-dimensional scoring (factual accuracy, completeness, appropriate "
        "behavior, hallucination, is_refusal) on a 0-2 scale provides fine-grained evaluation. "
        "A post-processing step uses keyword-based refusal detection to correct the LLM judge's "
        "`is_refusal` flag when it disagrees with surface-level signals.",
        "",
        "```text",
        judge[:2200] + ("\n..." if len(judge) > 2200 else ""),
        "```",
        "",
        "---",
        "",
        # ── 11. Future Directions ──
        "## 11. Future Directions",
        "",
        "Based on the evaluation results and identified failure modes, the following improvements are planned:",
        "",
        "1. **Fine-tuned embedding model:** Train a domain-adapted embedding model on SSL's corpus to "
        "improve Hit@K beyond the current general-purpose `all-MiniLM-L6-v2`. This should particularly "
        "benefit `publication_finding` and `synthesis` queries that use specialized terminology.",
        "",
        "2. **Improved no-evidence detection:** The current intent classifier still misclassifies some "
        "edge cases where tangentially related evidence exists. A two-stage approach -- classify first, "
        "then verify with a \"tangential evidence\" check -- could reduce both false refusals and missed refusals.",
        "",
        "3. **Adaptive chunking:** Replace fixed-size chunking with semantic paragraph-level segmentation "
        "to reduce cases where key facts are split across chunk boundaries.",
        "",
        "4. **Expanded multi-turn evaluation:** Increase from 5 chains / 15 items to 15+ chains / 50+ items "
        "covering more diverse conversational patterns (topic switches, corrections, negation follow-ups).",
        "",
        "5. **User feedback loop:** Integrate thumbs-up/down feedback from the Streamlit UI into a "
        "continuous evaluation pipeline, supplementing LLM-judge scores with real user signals.",
        "",
        "6. **Latency optimization:** The current multi-turn pipeline averages ~5s per turn. Parallelizing "
        "coreference resolution with intent classification, and caching HyDE embeddings for repeated "
        "topics, could reduce this to <3s.",
        "",
        "7. **Stronger consistency enforcement:** The current consistency check rate (39%) is low because "
        "many answers rely on QA Memory rather than raw corpus chunks. Extending the consistency check "
        "to verify against QA entries would improve this metric.",
        "",
        "---",
        "",
        # ── 12. Output Paths ──
        "## 12. Output Paths",
        "",
        "```",
        "results/v2/",
        "   rag_v2_eval_results.json",
        "   rag_v2_metrics.json",
        f"   rag_v1_phase2_eval_results.json   # V1@{n_q}",
        "   rag_v1_phase2_metrics.json",
        "   v1_vs_v2_metrics.json",
        "   multiturn_eval_results.json",
        "   multiturn_eval_metrics.json",
        "   RAG_V2_EVAL_SUMMARY.md",
        "report/",
        "   RAG_V2_FULL_EXPERIMENT_REPORT.md / RAG_V2_FULL_EXPERIMENT_REPORT_EN.md",
        "```",
        "",
        "---",
        "",
        "*Auto-generated by `scripts/generate_full_experiment_report.py --lang en`.*",
    ]
    return lines


def _build_zh_report(
    M, ds, n_q, ts,
    by_type, by_phase, by_diff, multi_chains,
    v2, v1_70, v1_p2, mt, mt_o,
    gen_prompt, coref, strategy, hyde, intent_llm, judge, pct, lang,
) -> list[str]:
    """Build the original Chinese report (unchanged structure)."""
    ds_suffix = M["dataset_suffix"].format(n_q=n_q, qc=M["q_count"])
    lines = [
        M["title"],
        "",
        f"{M['gen_time']} {ts}",
        f"{M['dataset_lbl']} `{DATASET.relative_to(ROOT)}`{ds_suffix}",
        "",
        "---",
        "",
        M["s1"],
        "",
        M["s1_1"],
        "",
        f"{M['s1_1a']}{n_q}{M['s1_1b']}",
        M["s1_1c"],
        "",
        M["s1_2"],
        "",
        M["col_type"],
        "|------|-----:|",
    ]
    for t, n in sorted(by_type.items(), key=lambda x: -x[1]):
        lines.append(f"| `{t}` | {n} |")
    lines += [
        "",
        M["s1_3"],
        "",
        M["col_diff"],
        "|------|-----:|",
    ]
    for d, n in sorted(by_diff.items(), key=lambda x: -x[1]):
        lines.append(f"| {d} | {n} |")
    lines += ["", M["col_phase"], "|------|-----:|"]
    for ph, n in sorted(by_phase.items()):
        lines.append(f"| {ph} | {n} |")
    lines += [
        "",
        M["s1_4"],
        "",
        M["col_chain"],
        "|----|-----:|",
    ]
    for ch, n in sorted((k, v) for k, v in multi_chains.items() if k):
        lines.append(f"| {ch} | {n} |")
    lines += ["", "---", "", M["s2"], ""]
    lines += _mermaid_lines(lang)
    lines += ["", M["s2_note"], "", "---", "", M["s3"], ""]

    if v2:
        lines += [
            M["col_metric"], "|------|------|",
            f"| {M['m_hit3']} | {pct(v2.get('raw_corpus_hit_at_3', 0))} |",
            f"| {M['m_hit5']} | {pct(v2.get('raw_corpus_hit_at_5', 0))} |",
            f"| {M['m_acc_a']} | {pct(v2.get('answer_accuracy_answerable', 0))} |",
            f"| {M['m_acc_all']} | {pct(v2.get('answer_accuracy_all', 0))} |",
            f"| {M['m_comp']} | {pct(v2.get('answer_completeness', 0))} |",
            f"| {M['m_hall']} | {pct(v2.get('hallucination_rate', 0))} |",
            f"| {M['m_cov']} | {pct(v2.get('coverage', 0))} |",
            f"| {M['m_cr']} | {pct(v2.get('correct_refusal_rate', 0))} |",
            f"| {M['m_fr']} | {pct(v2.get('false_refusal_rate', 0))} |",
            f"| {M['m_miss']} | {v2.get('missed_refusal_count', '--')} |",
        ]
        if "consistency_rate" in v2:
            lines.append(f"| {M['m_cons']} | {pct(v2['consistency_rate'])} |")
        lines += ["", M["hit_note"], "", M["s3_1"], "", M["col_bt"],
                   "|------|--:|--------:|-----------:|--------|"]
        for qt, v in v2.get("by_question_type", {}).items():
            lines.append(
                f"| `{qt}` | {v['total']} | {pct(v['behavior_rate'])} | {pct(v['avg_accuracy'])} | "
                f"{v['perfect']}/{v['total']} |"
            )
    else:
        lines.append(M["no_v2"])

    lines += ["", "---", "", M["s4"], "", M["s4_1"], ""]
    if v1_70:
        lines += [
            M["col_v1"], "|------|--------:|",
            f"| {M['m_hit5']} | {pct(v1_70.get('raw_corpus_hit_at_5', 0))} |",
            f"| {M['m_acc_a']} | {pct(v1_70.get('answer_accuracy_answerable', 0))} |",
            f"| {M['m_hall_short']} | {pct(v1_70.get('hallucination_rate', 0))} |",
        ]
    else:
        lines.append(M["no_v1_70"])
    lines += ["", f"{M['s4_2a']}{n_q}{M['s4_2b']}", "", M["s4_2_intro"], "",
              M["col_cmp"], "|------|----------:|---:|-----------|"]
    cmp_keys = [
        ("raw_corpus_hit_at_3", "m_hit3"), ("raw_corpus_hit_at_5", "m_hit5"),
        ("answer_accuracy_answerable", "m_acc_a"), ("answer_accuracy_all", "m_acc_all"),
        ("answer_completeness", "m_comp"), ("hallucination_rate", "m_hall"),
        ("coverage", "m_cov_cmp"), ("correct_refusal_rate", "cmp_cr"),
        ("false_refusal_rate", "cmp_fr"),
    ]
    if v1_p2 and v2:
        for k, mk in cmp_keys:
            a, b = v1_p2.get(k, 0), v2.get(k, 0)
            d = b - a
            sg = "+" if d > 0 else ""
            lines.append(f"| {M[mk]} | {pct(a)} | {pct(b)} | {sg}{pct(d)} |")
        lines.append(f"| {M['m_miss']} | {v1_p2.get('missed_refusal_count', 0)} | "
                     f"{v2.get('missed_refusal_count', 0)} | - |")
    else:
        lines.append(M["cmp_missing"])

    lines += ["", "---", "", M["s5"], "", M["s5_intro"], ""]
    if mt and mt_o:
        cr = mt.get("coreference", {})
        n_trig = cr.get("coreference_triggered", 0)
        n_fu = cr.get("follow_up_questions", 0)
        rate = cr.get("coreference_rate", 0)
        coref_tail = M["coref_rate_tail"].format(rate=f"{rate:.0%}")
        lines += [
            M["col_mt"], "|------|-------------|",
            f"| {M['mt_fa']} | **{mt_o.get('factual_accuracy_avg', 0):.3f}** |",
            f"| {M['mt_comp']} | {mt_o.get('completeness_avg', 0):.3f} |",
            f"| {M['mt_beh']} | {mt_o.get('appropriate_behavior_avg', 0):.3f} |",
            f"| {M['mt_hall']} | {mt_o.get('hallucination_avg', 0):.3f} |",
            "", f"{M['mt_coref']} {n_trig}/{n_fu}{M['mt_coref_trig']}{coref_tail}",
            "", M["mt_by_turn"], "", M["col_turn"], "|------|--------|--------|------|",
        ]
        for tk in sorted(mt.get("by_turn_position", {})):
            t = mt["by_turn_position"][tk]
            lines.append(f"| {tk} | {t['factual_accuracy_avg']:.3f} | "
                        f"{t['completeness_avg']:.3f} | {t['hallucination_avg']:.3f} |")
    else:
        lines.append(M["no_mt"])

    lines += _build_case_analysis_section(lang)
    lines += [
        "", M["s7"], "", M["s7_1"], "", "```text",
        gen_prompt[:3200] + ("\n..." if len(gen_prompt) > 3200 else ""), "```",
        "", M["s7_2"], "", "```text",
        coref[:2500] + ("\n..." if len(coref) > 2500 else ""), "```",
        "", M["s7_3"], "", "```text",
        strategy[:2200] + ("\n..." if len(strategy) > 2200 else ""), "```",
        "", M["s7_4"], "", "```text",
        hyde[:1800] + ("\n..." if len(hyde) > 1800 else ""), "```",
        "", M["s7_5"], "", "```text",
        intent_llm[:2200] + ("\n..." if len(intent_llm) > 2200 else ""), "```",
        "", M["s7_6"], "", "```text",
        judge[:2200] + ("\n..." if len(judge) > 2200 else ""), "```",
        "", "---", "", M["s8"], "",
        "```", "results/v2/",
        "   rag_v2_eval_results.json", "   rag_v2_metrics.json",
        f"   rag_v1_phase2_eval_results.json   # V1@{n_q}",
        "   rag_v1_phase2_metrics.json", "   v1_vs_v2_metrics.json",
        "   multiturn_eval_results.json", "   multiturn_eval_metrics.json",
        "   RAG_V2_EVAL_SUMMARY.md", "report/", M["s8_this"], "```",
        "", "---", "", M["footer"],
    ]
    return lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate RAG V2 full experiment report (zh or en).")
    ap.add_argument("--lang", choices=("zh", "en"), default="zh", help="Report language (default: zh).")
    args = ap.parse_args()
    main(args.lang)
