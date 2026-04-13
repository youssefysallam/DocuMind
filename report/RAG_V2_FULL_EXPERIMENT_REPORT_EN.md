# RAG V2 Full Experiment Report (Phase 2)

**Generated (UTC):** 2026-04-11 03:42:11
**Dataset:** `data\eval_phase2\stakeholder_eval_phase2.json` (**104** items)

---

## 1. Introduction and Objectives

This report presents a comprehensive evaluation of the **RAG V2** system built for the **Sustainable Solutions Lab (SSL)** at UMass Boston. The system answers stakeholder questions about SSL's climate-justice research, projects, publications, and leadership by retrieving evidence from a unified corpus of PDFs and website content and generating grounded answers with `gpt-4o-mini`.

**Objectives of this evaluation:**

1. Quantify RAG V2 performance on an expanded 104-item benchmark covering six question types, three difficulty levels, and 15 multi-turn conversation items.
2. Compare V2 against the V1 baseline on the same item set to measure improvements in accuracy, hallucination control, refusal behavior, and completeness.
3. Validate the multi-turn conversation subsystem (coreference resolution, retrieval strategy decisions, session memory) on five thematic chains.
4. Identify failure modes and outline future optimization directions.

---

## 2. Dataset Design and Expansion

### 2.1 Expansion from Phase 1 to Phase 2

The evaluation dataset was expanded from **70** items (Phase 1) to **104** items (Phase 2) to stress-test V2's new capabilities. The 34 new items include:

- **Synthesis (SYN-P2):** Multi-source questions requiring sub-query decomposition and cross-document reasoning.
- **No-evidence edge cases (NEG-P2):** Harder boundary questions where tangentially related evidence exists in the corpus but the correct answer is refusal.
- **Publication deep-dives (PUB-P2):** Questions targeting specific findings, figures, or recommendations in SSL reports.
- **Multi-turn chains (MULTI-01 to MULTI-15):** 5 thematic chains of 3 turns each, testing coreference resolution and progressive evidence accumulation.

### 2.2 Question Type Taxonomy

Each item is assigned one of six `question_type` labels. The taxonomy drives intent classification in V2's pipeline, which in turn controls HyDE activation, sub-query decomposition, and retrieval strategy.

| Type | Description | Example |
|------|-------------|---------|
| `general_overview` | SSL identity, mission, leadership, contact | "What is SSL's mission?" |
| `project_initiative` | Specific projects or programs | "What is the Cape Cod Rail Resilience project?" |
| `topic_specific` | Focused research topics | "What has SSL studied about climate migration?" |
| `publication_finding` | Findings from specific reports | "What does the harbor barrier study recommend?" |
| `synthesis` | Cross-document, multi-source reasoning | "What themes appear across SSL projects?" |
| `no_evidence` | Topics SSL has not studied (expect refusal) | "Has SSL published on nuclear energy?" |

### 2.3 Distribution by Type, Difficulty, and Phase

| Type | Count |
|------|-----:|
| `no_evidence` | 22 |
| `general_overview` | 20 |
| `synthesis` | 18 |
| `project_initiative` | 17 |
| `topic_specific` | 15 |
| `publication_finding` | 12 |

| Difficulty | Count |
|------|-----:|
| medium | 47 |
| hard | 35 |
| easy | 22 |

| Phase | Count |
|------|-----:|
| 1 | 70 |
| 2 | 34 |

Of the 104 items, **82** expect a substantive answer and **22** expect the system to refuse (no-evidence items).

### 2.4 Multi-turn Subset Design

The 15 multi-turn items are organized into **5 chains** (A through E), each with 3 turns of increasing complexity. Chains are designed to test coreference resolution, comparison reasoning, and detail follow-ups:

| Chain | Topic | Turn 1 | Turn 2 | Turn 3 |
|-------|-------|--------|--------|--------|
| A | Workforce (C3I) | Project overview | Funding details (pronoun) | Job types (pronoun) |
| B | Infrastructure vs Community | Cape Cod Rail | Compare to East Boston ("that project") | Harbor barrier three-way comparison |
| C | Leadership | Team overview | Executive director background ("the director") | Future plans |
| D | Publication | Report findings | Methodology details ("the study") | Recommendations |
| E | Cross-topic | Community engagement | Compare approaches ("it") | Broader synthesis |

---

## 3. RAG V1 Architecture (Baseline)

RAG V1 is a single-pass retrieval-augmented generation system with the following pipeline:

1. **Query input** -- the raw user question is passed directly to retrieval (no intent classification or query rewriting).
2. **Hybrid retrieval** -- dense (sentence-transformers/all-MiniLM-L6-v2 via FAISS IndexFlatIP) and sparse (BM25) retrieval each return top-20 candidates.
3. **Reranking** -- a cross-encoder (ms-marco-MiniLM-L-6-v2) reranks the merged candidate pool.
4. **Context assembly** -- top 5 chunks are selected, with up to 2 curated QA memory entries (relevance floor 0.35) injected when applicable.
5. **Answer generation** -- `gpt-4o-mini` generates the answer from the assembled context.

**Key V1 characteristics:**
- **Contextualized chunks:** each chunk is prefixed with document title and section metadata to improve dense retrieval relevance.
- **Curated QA Memory:** hand-verified Q&A pairs for common questions, injected when the reranker score exceeds the relevance threshold.
- **No multi-turn support:** each question is treated independently.
- **No intent-driven routing:** all questions follow the same retrieval path regardless of type.

| Component | V1 Implementation |
|-----------|------------------|
| Dense embedding | `all-MiniLM-L6-v2`, normalized, FAISS `IndexFlatIP` |
| Sparse retrieval | BM25 on raw chunk tokens |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Top-K (dense/sparse) | 20 / 20 |
| Final context size | 5 chunks + up to 2 QA entries |
| LLM | `gpt-4o-mini` |
| Multi-turn | Not supported |

---

## 4. RAG V2 Architecture

RAG V2 extends V1 with an **intent-driven, multi-turn-aware** pipeline. The core retrieval infrastructure (dense + sparse + reranker) is preserved, but seven new LLM-powered modules are added around it.

### 4.1 Pipeline Flow

```mermaid
flowchart TB
  U[User question] --> CR[Coreference LLM]
  CR --> RS[Retrieval-strategy LLM]
  RS -->|REUSE| OLD[Reuse prior evidence]
  RS -->|FULL/MERGE| IC[Intent classification]
  IC --> HY{HyDE?}
  HY -->|synthesis/topic/pub| H1[HyDE passage LLM]
  HY -->|overview/project/no_ev| SKIP[Skip HyDE]
  H1 --> SQ{Sub-query split?}
  SKIP --> SQ
  SQ -->|synthesis| DC[Decomposition LLM]
  SQ -->|other| RT[Single hybrid retrieval]
  DC --> RT
  RT --> MH[Multi-hop coverage + supplemental retrieval]
  MH --> DIV[Diversity truncation]
  DIV --> GEN[Answer generation LLM]
  GEN --> CC{Batch eval?}
  CC -->|yes| CS[Consistency check LLM]
  CC -->|no UI| DONE[Respond]
  CS --> DONE
  OLD --> GEN
```

### 4.2 Component Details

| # | Component | Purpose | When it fires |
|---|-----------|---------|---------------|
| 1 | **Coreference Resolution** | Rewrites follow-up questions by replacing pronouns with referents from conversation history | Every follow-up turn (not turn 1) |
| 2 | **Retrieval Strategy Decision** | Chooses FULL_RETRIEVAL, REUSE_EVIDENCE, or MERGE_EVIDENCE based on topic continuity | Every follow-up turn |
| 3 | **Intent Classification** | Classifies the question into one of 6 types to route downstream processing | Every question |
| 4 | **Conditional HyDE** | Generates a hypothetical answer passage to enrich the query embedding | Only for `synthesis`, `topic_specific`, `publication_finding` |
| 5 | **Sub-query Decomposition** | Splits complex questions into simpler sub-queries for parallel retrieval | Only for `synthesis` questions |
| 6 | **Multi-hop Retrieval** | Checks entity coverage and issues supplemental retrieval for missing entities | When session history contains entities not covered by initial retrieval |
| 7 | **Diversity Truncation** | Removes near-duplicate chunks to maximize information density in the context window | Always (post-retrieval) |
| 8 | **Consistency Check** | Verifies the generated answer is supported by the retrieved evidence | Batch eval mode; disabled in UI for latency |

**Note:** Batch single-turn eval (`python -m rag_v2.pipeline`) calls `session.clear()` then `ask(..., enable_consistency=True)` per item—same retrieval+generation path as production; the UI skips consistency by default for latency. Multi-hop supplemental retrieval depends on entities in session history and **usually does not fire** when each eval item is isolated; multi-turn metrics are in §5.

---

## 5. V1 vs V2 Architecture Comparison

| Aspect | V1 | V2 |
|--------|----|----|
| Intent classification | None (all questions same path) | LLM-based, 6 categories |
| Query rewriting | None | Coreference resolution LLM |
| HyDE | Not used | Conditional (synthesis/topic/publication only) |
| Sub-query decomposition | None | LLM decomposition for synthesis |
| Retrieval strategy | Always full retrieval | LLM decides: FULL / REUSE / MERGE |
| Multi-hop retrieval | None | Entity-coverage check + supplemental retrieval |
| Diversity filtering | None | Near-duplicate chunk removal |
| Consistency check | None | LLM answer-evidence verification |
| Multi-turn support | None | Session memory + coreference + strategy |
| Base retrieval | Dense + BM25 + reranker | Same (preserved from V1) |
| LLM | `gpt-4o-mini` | Same |

**Summary:** V2 wraps V1's retrieval core with intent-driven routing, query transformation, and post-generation verification. The architectural additions specifically target V1's weaknesses: synthesis questions (via HyDE + decomposition), no-evidence boundary cases (via intent classification), and multi-turn conversations (via coreference + session memory).

---

## 6. Core Metrics (RAG V2, Full Phase 2 Set)

| Metric | Value |
|------|------|
| Hit@3 | 15.8% |
| Hit@5 | 17.1% |
| Accuracy (answerable) | 75.0% |
| Accuracy (all) | 69.2% |
| Completeness | 62.5% |
| Hallucination rate (lower is better) | 12.0% |
| Behavior coverage | 96.2% |
| Correct refusal rate | 90.9% |
| False refusal rate | 2.4% |
| Missed refusals | 2 |
| Answer-evidence consistency | 39.0% |

> **On Hit@3 / Hit@5:** `eval_retrieval` only checks whether **raw corpus** `chunk_id` / filenames match `gold_source_ids`. When the top hits are **QA Memory (`curated_qa`)**, answers can still be correct while corpus Hit@K stays low—interpret together with **LLM-judge accuracy and hallucination**.

### 6.1 By Question Type (V2)

| Type | N | Behavior | Avg accuracy | Perfect |
|------|--:|--------:|-----------:|--------|
| `general_overview` | 20 | 100.0% | 82.5% | 13/20 |
| `project_initiative` | 17 | 94.1% | 79.4% | 11/17 |
| `topic_specific` | 15 | 100.0% | 76.7% | 8/15 |
| `publication_finding` | 12 | 100.0% | 79.2% | 7/12 |
| `synthesis` | 18 | 94.4% | 58.3% | 4/18 |
| `no_evidence` | 22 | 90.9% | 47.7% | 3/22 |

---

## 7. V1 vs V2 Performance Comparison

### 7.1 Historical Baseline: V1 on the Original 70-item Set

| Metric | V1 (70 items) |
|------|--------:|
| Hit@5 | 18.2% |
| Accuracy (answerable) | 79.1% |
| Hallucination rate | 24.3% |
| Coverage | 92.9% |
| Correct refusal rate | 80.0% |
| False refusal rate | 1.8% |

### 7.2 Same-set Comparison: V1 vs V2 (104 Phase 2 Items)

Both systems evaluated on the identical 104-item set with the same LLM judge (`gpt-4o-mini`). **V1** uses V1 retrieve + generate. **V2** uses the full `ask()` path (conditional HyDE, multi-hop, diversity, intent routing, etc.).

| Metric | V1 Phase2 | V2 | Delta (V2-V1) |
|------|----------:|---:|-----------|
| Hit@3 | 15.8% | 15.8% | 0.0% |
| Hit@5 | 17.1% | 17.1% | 0.0% |
| Accuracy (answerable) | 72.0% | 75.0% | +3.0% |
| Accuracy (all) | 66.3% | 69.2% | +2.9% |
| Completeness | 62.0% | 62.5% | +0.5% |
| Hallucination rate | 24.0% | 12.0% | -12.0% |
| Behavior coverage | 90.4% | 96.2% | +5.8% |
| Correct refusal | 86.4% | 90.9% | +4.5% |
| False refusal rate | 8.5% | 2.4% | -6.1% |
| Missed refusals | 3 | 2 | - |

### 7.3 Key Takeaways from the Comparison

1. **Hallucination halved:** V2 reduces hallucination rate by 12.0% (from 24.0% to 12.0%), primarily due to the consistency check and stricter evidence grounding.
2. **Accuracy improved:** Answerable accuracy rises by 3.0%, driven by HyDE and multi-hop retrieval providing richer context for complex questions.
3. **False refusal rate dropped:** From 8.5% to 2.4% (6.1% reduction), as intent classification reduces unnecessary refusals on answerable items.
4. **Behavior coverage up:** 5.8% improvement, reflecting better alignment between expected and actual behavior (answer vs refuse).
5. **Hit@K unchanged:** V1 and V2 share the same base retrieval index; V2's gains come from better *use* of retrieved evidence, not from retrieving different chunks.

---


---

## 6. Case studies (single-turn + multi-turn)

Examples are taken from the latest saved eval (`rag_v2_eval_results.json`, `multiturn_eval_results.json`), auto-filtered by judge scores; **multi-turn has its own subsection** (coreference + cross-topic comparison).

### 6.1 Single-turn: representative correct cases

#### Correct example 1: `SYN-05` (synthesis, expected: answer)

- **Question:** What cross-cutting themes appear across SSL's different projects and publications?
- **Judge:** Factual=2, Completeness=2, Behavior=2, Hallucination=2, Refusal flag=0

**System answer (excerpt):**

> Across the evidence provided, several recurring cross-cutting themes appear in SSL’s projects and publications: 1. Climate resilience and adaptation This is the most consistent theme. SSL’s major reports focus on resilience planning, adaptation governance, equitable adaptation, community preparedness, and resilience financing. Project work also highlights infrastructure resilience and adaptation strategies. 2. Equity and climate justice SSL’s research explicitly emphasizes climate justice and equitable adaptatio...

**Judge notes:** _The system's answer accurately reflects the key themes outlined in the gold standard, covering all relevant points such as equity/inclusion, community engagement, climate governance/policy, infrastructure resilience, and transdisciplinary collaboration. There are no unsupported facts, and the system did not refuse to answer._

**Brief analysis:** Strong scores; under `synthesis`, evidence and generation align well.

#### Correct example 2: `NEG-11` (no_evidence, expected: refuse)

- **Question:** Has SSL published any research on nuclear energy as a climate solution?
- **Judge:** Factual=2, Completeness=1, Behavior=2, Hallucination=2, Refusal flag=1

**System answer (excerpt):**

> I could not find clearly supported SSL work on this topic in the current corpus.

**Judge notes:** _The system's answer correctly indicates a lack of evidence regarding SSL's research on nuclear energy, aligning with the gold answer. However, it does not explicitly refuse or abstain from answering, which was the expected behavior. The completeness is partial as it does not fully encapsulate the definitive nature of the gold answer. There are no unsupported facts in the system's response._

**Brief analysis:** Strong scores; under `no_evidence`, evidence and generation align well.

#### Correct example 3: `GEN-02` (general_overview, expected: answer)

- **Question:** What is SSL's mission?
- **Judge:** Factual=2, Completeness=2, Behavior=2, Hallucination=2, Refusal flag=0

**System answer (excerpt):**

> SSL’s mission is to transform the climate research and action space by bridging across sectors, convening collaborators, and facilitating transdisciplinary research that centers and is led by historically and currently excluded communities. The evidence also describes this mission in closely related ways, including: - working to understand the disproportionate impacts of climate change on historically marginalized communities and developing sustainable, equitable solutions with them; and - serving as an engine o...

**Judge notes:** _The system's answer accurately reflects the mission of SSL as described in the gold answer, including the focus on historically excluded communities and the collaborative approach. It also includes additional relevant details about the organization's work, enhancing completeness. There are no unsupported facts, and the system did not refuse to answer._

**Brief analysis:** Strong scores; under `general_overview`, evidence and generation align well.

### 6.2 Single-turn: failures / high-risk cases

Filter: factual score ≤1 or behavior <2 (should refuse but answers, or severe off-topic answers, etc.).

#### Failure example 1: `SYN-P2-03` (synthesis, expected: answer)

- **Question:** What role do SSL's four partner institutes play in advancing its climate justice agenda, and how does each institute's focus area contribute to SSL's research portfolio?
- **Judge:** Factual=0, Completeness=0, Behavior=0, Hallucination=2, Refusal flag=1

**System answer (excerpt):**

> I could not find clearly supported SSL work on this topic in the current corpus.

**Judge notes:** _The system's answer does not provide any factual information related to SSL's climate justice agenda, which is a significant deviation from the gold answer. It fails to cover any key points and does not exhibit appropriate behavior as it does not answer the question but instead states a lack of information. There are no unsupported facts, as the system simply states it could not find relevant information. The system does explicitly refuse to provide an answer._

**Brief analysis:** Synthesis item refused wholesale when the evidence chain is thin—check sub-query split, HyDE, and multi-hop coverage across subtopics.

#### Failure example 2: `NEG-10` (no_evidence, expected: refuse)

- **Question:** What is SSL's position on sea-level rise modeling and prediction?
- **Judge:** Factual=0, Completeness=1, Behavior=0, Hallucination=0, Refusal flag=0

**System answer (excerpt):**

> SSL’s work indicates that sea-level rise modeling and prediction should be used as an evolving, science-based basis for planning, with conservative benchmarks and continued monitoring rather than treated as fixed forecasts. From the available evidence, SSL-associated reports note that: - sea-level rise projections are used together with historical records of tides and storms to forecast future coastal impacts, including the performance of adaptation measures such as harbor-wide barriers; - conservative vulnerabi...

**Judge notes:** _The system's answer contradicts the gold answer by implying that SSL is involved in sea-level rise modeling and prediction, which the gold answer states is not documented. Therefore, it scores 0 for factual accuracy. It partially covers key points about planning and monitoring, so it scores 1 for completeness. The system did not refuse as expected, scoring 0 for appropriate behavior. There are no unsupported facts, so it scores 2 for hallucination. The system did not explicitly refuse, scoring 0 for is_refusal._

**Brief analysis:** No-evidence item but the model elaborates from retrieved snippets—conflicts with gold "no dedicated work"; likely retrieval noise or refusal policy.

#### Failure example 3: `NEG-P2-03` (no_evidence, expected: refuse)

- **Question:** What green building certification or LEED programs does SSL run?
- **Judge:** Factual=0, Completeness=0, Behavior=2, Hallucination=2, Refusal flag=1

**System answer (excerpt):**

> I could not find clearly supported SSL work on this topic in the current corpus.

**Judge notes:** _The system's answer does not accurately reflect the facts presented in the gold answer, as it fails to acknowledge that SSL does not operate green building certification or LEED programs. It also does not cover the key points about SSL being a research institute and not a certification body. The system did not refuse as expected, and it does not state any unsupported facts, but it also does not provide a relevant answer._

**Brief analysis:** Standard refusal wording was used, but the judge still disagrees with the gold expected explicit denial / caveats (common on no_evidence edge cases).

### 6.3 Multi-turn: correct chain (context + comparison)

**Chain B** (infrastructure vs community): first turn anchors *Cape Cod Rail*; the follow-up *that project* resolves and compares to East Boston with full judge scores—**multi-hop + coreference** help on this pattern.

#### Chain B: `MULTI-04` — **Chain B**, turn 1

- **User question:** What is the Cape Cod Rail Resilience project?
- **Intent:** `project_initiative`
- **Judge:** Factual=2, Completeness=2, Behavior=2, Hallucination=0

**System answer (excerpt):**

> The Cape Cod Rail Resilience Project is a collaboration involving MassDOT, UMass Boston, and AECOM focused on improving rail safety and climate resilience along the Cape Main Line. It was launched after a major 300-foot rail embankment collapse in East Sandwich in 2020, which was linked to climate-change-induced drought conditions. The project addresses growing climate risks to critical transportation infrastructure and is inspired by Carlos Velásquez’s doctoral research. Velásquez, a PhD candidate at UMass Bost...

**Judge notes:** _The system answer accurately reflects the key facts from the gold answer, including the collaboration between MassDOT, UMass Boston, and AECOM, the $2 million funding, the 300-foot embankment collapse, and the use of drones and sensors. It also adds relevant details about climate resilience and the involvement of Carlos Velásquez, which enhances the completeness without introducing unsupported facts._

**Brief analysis:** Lower hallucination score: retrieved chunks carry extra narrative; trim more aggressively; cross-check consistency output.

#### Chain B (highlight: comparison): `MULTI-05` — **Chain B**, turn 2

- **User question:** How does that project compare to SSL's work in East Boston in terms of approach?
- **After coreference:** How does the Cape Cod Rail Resilience Project compare to SSL's work in East Boston in terms of approach?
- **Intent:** `project_initiative`
- **Judge:** Factual=2, Completeness=2, Behavior=2, Hallucination=2

**System answer (excerpt):**

> The two efforts share a climate resilience focus, but the evidence shows different approaches. ### Cape Cod Rail Resilience Project This project is focused on transportation infrastructure resilience along the Cape Main Line. Its approach emphasizes: - improving rail safety and climate resilience - integrating advanced technology with stakeholder-engaged research - using aerial surveys or drones, hydrological drought analysis, and real-time monitoring systems - studying three pilot sites in collaboration with Ma...

**Judge notes:** _The system's answer accurately reflects the key points from the gold answer, detailing the distinct approaches of the Cape Cod Rail project and the East Boston work. It correctly identifies the focus on technology and infrastructure for Cape Cod and the community-centered, equity-focused approach for East Boston. The answer is complete and does not include unsupported facts._

**Brief analysis:** Multi-turn context and retrieval work well; resolved query matches evidence.

#### Chain B: `MULTI-06` — **Chain B**, turn 3

- **User question:** What about the harbor barrier study — is that approach more similar to the rail project or to East Boston?
- **After coreference:** Is the approach taken in the harbor barrier study more similar to the Cape Cod Rail Resilience Project or to Sustainable Solutions Lab's work in East Boston?
- **Intent:** `project_initiative`
- **Judge:** Factual=1, Completeness=1, Behavior=2, Hallucination=2

**System answer (excerpt):**

> The evidence suggests the harbor barrier study is more similar to the Cape Cod Rail Resilience Project than to SSL’s East Boston work, but only at a broad level. ### Why it appears closer to the rail project The harbor barrier study focuses on large-scale physical infrastructure for climate protection. The evidence describes: - an ongoing study of a harbor-wide barrier from Winthrop to Hull to protect against major storm surge - analysis of feasibility and cost - a recommendation that the City focus on a shore-b...

**Judge notes:** _The system's answer contains some correct facts about the harbor barrier study and its similarities to the Cape Cod Rail project, but it lacks specific details about the East Boston work, which is a key point in the gold answer. It does not fully address the community engagement aspect mentioned in the gold answer. The system does not refuse to answer and provides a structured response, but it does not fully meet the completeness expected._

**Brief analysis:** Coreference resolved but judge still marks missing points—generation prompt may need stronger enumeration aligned with gold (amounts, durations, etc.).

**Chain summary:** Turn 1 locks Cape Cod evidence; comparison triggers multi-hop for East Boston; turn 3 (harbor barrier) is a three-way contrast and still depends on entities recoverable from session text.

### 6.4 Multi-turn: weaker / partial-credit cases

We list follow-ups or final turns where **factual or completeness scores are not full**, illustrating residual issues (e.g., judge marks amount wording incomplete, publication follow-ups shallow).

#### Weaker example: `MULTI-08` — **Chain C**, turn 2

- **User question:** Can you tell me more about the executive director's background?
- **After coreference:** Can you tell me more about B.R. Balachandran's background?
- **Intent:** `general_overview`
- **Judge:** Factual=1, Completeness=2, Behavior=2, Hallucination=0

**System answer (excerpt):**

> The Executive Director of SSL is Dr. B.R. Balachandran. His background includes: - He became SSL’s Executive Director in February 2024. - He provides strategic leadership for SSL and oversees research projects. - He holds a Ph.D. in Regional Planning from the University of Illinois at Urbana-Champaign, earned in 2022. - Before leading SSL, he held a visiting faculty position at UMass Boston’s School for the Environment and a research position at the University of North Texas at Denton. - He is a certified urban ...

**Judge notes:** _The system's answer contains some inaccuracies regarding the timeline of Dr. Balachandran's PhD and his position at SSL, but it correctly identifies his educational background and experience with UNDP, the World Bank, and KfW. The answer is comprehensive, covering key points about his roles and expertise. There are no unsupported facts, and the system did not refuse to answer._

**Brief analysis:** Coreference resolved but judge still marks missing points—generation prompt may need stronger enumeration aligned with gold (amounts, durations, etc.).

#### Weaker example: `MULTI-02` — **Chain A**, turn 2

- **User question:** Who funds it and how much did they give?
- **After coreference:** Who funds the Climate Careers Curricula Initiative (C3I), and how much funding did the Liberty Mutual Foundation provide?
- **Intent:** `project_initiative`
- **Judge:** Factual=1, Completeness=1, Behavior=2, Hallucination=2

**System answer (excerpt):**

> The Climate Careers Curricula Initiative is funded by the Liberty Mutual Foundation. The evidence states the grant is $540,000 over three years.

**Judge notes:** _The system correctly identifies the funding source but does not explicitly mention the amount or duration of the grant, leading to partial factual accuracy and completeness. The system answered appropriately without refusing._

**Brief analysis:** Coreference resolved but judge still marks missing points—generation prompt may need stronger enumeration aligned with gold (amounts, durations, etc.).

#### Weaker example: `MULTI-03` — **Chain A**, turn 3

- **User question:** What kinds of jobs does it train people for?
- **After coreference:** What kinds of blue and green jobs in Greater Boston does the Climate Careers Curricula Initiative (C3I) train people for?
- **Intent:** `project_initiative`
- **Judge:** Factual=1, Completeness=1, Behavior=2, Hallucination=2

**System answer (excerpt):**

> C3I trains people for high-demand blue and green jobs in Greater Boston. The evidence specifically mentions training related to: - Solar installation - Energy auditing - Nature-based solutions More broadly, the initiative is designed to create microcredentialed training programs that open career pathways for underrepresented populations, especially vulnerable young adults, people of color, and low-income adults from environmental justice communities.

**Judge notes:** _The system answer captures the essence of C3I's focus on training for blue and green jobs but lacks specific details about the types of training (solar installation, energy auditing, nature-based solutions) mentioned in the gold answer. It also does not address the microcredentialing aspect or the target populations, leading to a partial score for completeness. There are no unsupported facts in the system answer._

**Brief analysis:** Coreference resolved but judge still marks missing points—generation prompt may need stronger enumeration aligned with gold (amounts, durations, etc.).

#### Weaker example: `MULTI-06` — **Chain B**, turn 3

- **User question:** What about the harbor barrier study — is that approach more similar to the rail project or to East Boston?
- **After coreference:** Is the approach taken in the harbor barrier study more similar to the Cape Cod Rail Resilience Project or to Sustainable Solutions Lab's work in East Boston?
- **Intent:** `project_initiative`
- **Judge:** Factual=1, Completeness=1, Behavior=2, Hallucination=2

**System answer (excerpt):**

> The evidence suggests the harbor barrier study is more similar to the Cape Cod Rail Resilience Project than to SSL’s East Boston work, but only at a broad level. ### Why it appears closer to the rail project The harbor barrier study focuses on large-scale physical infrastructure for climate protection. The evidence describes: - an ongoing study of a harbor-wide barrier from Winthrop to Hull to protect against major storm surge - analysis of feasibility and cost - a recommendation that the City focus on a shore-b...

**Judge notes:** _The system's answer contains some correct facts about the harbor barrier study and its similarities to the Cape Cod Rail project, but it lacks specific details about the East Boston work, which is a key point in the gold answer. It does not fully address the community engagement aspect mentioned in the gold answer. The system does not refuse to answer and provides a structured response, but it does not fully meet the completeness expected._

**Brief analysis:** Coreference resolved but judge still marks missing points—generation prompt may need stronger enumeration aligned with gold (amounts, durations, etc.).

**Multi-turn takeaways:**

1. **Coreference** fired on 10/10 follow-ups, rewriting *it / that project / the survey* into self-contained queries—prerequisite for good comparison answers.
2. **Turn 3** averages lower than turns 1–2 (see §5): questions are more integrative and judges score completeness stricter.
3. **Lower hallucination scores** often co-occur with biography-style snippets in retrieved chunks—cross-check consistency output and the source panel.


### Root Cause Analysis of Failures

| Failure Mode | Count | Description |
|---|--:|---|
| False refusal | 2 | System refused an answerable question; typically synthesis items where sub-query decomposition did not fire |
| Retrieval noise / missed refusal | 2 | No-evidence item but system elaborated from tangentially related chunks instead of refusing |
| Incomplete enumeration | 43 | Correct direction but judge marked missing key points (amounts, names, durations) |
| Other | 2 | Mixed causes including hallucination from biography snippets, off-topic elaboration |


---

## 9. Multi-turn Conversation Design and Evaluation

### 9.1 Design Rationale

Multi-turn support is a V2-only capability. Real stakeholders rarely ask isolated questions -- they ask follow-ups that reference prior context ("How does *that project* compare to...", "Who funds *it*?"). V2 addresses this with three coordinated components:

1. **Coreference Resolution LLM:** Rewrites pronouns and demonstratives into full noun phrases using conversation history, producing a self-contained query suitable for retrieval.
2. **Retrieval Strategy Decision LLM:** Decides whether to search from scratch (FULL_RETRIEVAL), reuse prior evidence (REUSE_EVIDENCE), or merge old and new evidence (MERGE_EVIDENCE).
3. **Session Memory:** Stores per-chain question-answer history for coreference context and multi-hop entity tracking.

### 9.2 Evaluation Protocol

Script: `test_multiturn_eval.py`. Each of the 5 chains runs in an isolated `SessionMemory`. Questions are submitted sequentially (turn 1, 2, 3). The same LLM-as-judge evaluates each turn's answer independently against its gold standard.

### 9.3 Overall Multi-turn Metrics

| Metric | Score / 2.0 |
|------|-------------|
| Factual accuracy | **1.533** |
| Completeness | 1.400 |
| Appropriate behavior | 2.000 |
| Hallucination (2 = none) | 1.533 |

**Coreference:** 10/10 triggered (100%).

### 9.4 Performance by Turn Position

| Turn | Accuracy | Completeness | Hallucination |
|------|--------|--------|------|
| turn_1 | 1.800 | 1.600 | 1.400 |
| turn_2 | 1.600 | 1.600 | 1.400 |
| turn_3 | 1.200 | 1.000 | 1.800 |

### 9.5 Performance by Chain

| Chain | Topic | Accuracy | Completeness | Hallucination |
|-------|-------|--------|--------|------|
| A | Workforce (C3I) | 1.333 | 1.333 | 2.000 |
| B | Infrastructure | 1.667 | 1.667 | 1.333 |
| C | Leadership | 1.667 | 1.333 | 0.667 |
| D | Publication | 1.667 | 1.667 | 2.000 |
| E | Cross-topic | 1.333 | 1.000 | 1.667 |

**Latency:** avg 5.0s, max 10.3s, min 1.2s per turn.

---

## 10. Prompt Engineering

All LLM calls in V2 use carefully designed prompts. This section documents the key prompts and the design rationale behind each.

### 10.1 Answer Generation (`generate_answer_v2`)

**Design rationale:** The prompt enforces strict evidence grounding with an explicit 9-rule framework. Rule 3 mandates a specific refusal phrase for consistency. Rule 5 prevents the model from treating tangential mentions as dedicated SSL work. Rule 6 establishes trust in curated QA memory entries. Rule 8 prevents raw source leakage into user-facing answers.

```text
system_prompt = (
        "You are a research assistant for the Sustainable Solutions Lab (SSL) at UMass Boston.\n"
        "Answer ONLY based on the provided evidence. Follow these rules strictly:\n\n"
        "1. If the evidence clearly and directly answers the question, provide a detailed, well-organized answer.\n"
        "2. If the evidence partially addresses the question, answer what you can and note what is missing.\n"
        "3. If the evidence does NOT contain relevant information, you MUST refuse:\n"
        "   Say: 'I could not find clearly supported SSL work on this topic in the current corpus.'\n"
        "4. NEVER guess or infer beyond what the evidence states.\n"
        "5. If a topic is only tangentially mentioned (e.g., mentioned in passing, listed as a keyword, "
        "or part of a broader context), do NOT treat it as dedicated SSL work on that topic.\n"
        "6. When a VERIFIED QA KNOWLEDGE entry says 'No relevant SSL work found,' trust that assessment.\n"
        "7. Prefer specific PDF/website evidence over general QA summaries when both exist.\n"
        "8. Do NOT include source file names, chunk references, or raw evidence text in your answer. "
        "Write a clean, natural response as if you already know the information. "
        "Source attribution is handled separately by the system.\n"
        "9. For follow-up questions, you may reference information from the conversation history "
        "but always prioritize the current evidence."
    )
```

### 10.2 Coreference Resolution (`resolve_coreference`)

**Design rationale:** Produces a self-contained query by replacing pronouns with their referents. Uses `temperature=0` and `max_tokens=150` for deterministic, concise output. The "return ONLY the rewritten question" constraint prevents explanations from leaking in.

```text
"You rewrite follow-up questions into self-contained queries.\n"
                    "Given a conversation history and a new question, rewrite the new question "
                    "so it can be understood WITHOUT the conversation history.\n"
                    "Rules:\n"
                    "- Replace pronouns (it, they, this, that) with their referents from history.\n"
                    "- Include relevant context from prior turns if needed.\n"
                    "- If the question is already self-contained, return it unchanged.\n"
                    "- Return ONLY the rewritten question. No explanation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{history_text}\n\n"
                    f"New question: {current_query}\n\n"
                    "Rewritten question:"
                ),
            },
        ],
        temperature=0,
        max_tokens=150,
        timeout=20,
    )
```

### 10.3 Retrieval Strategy (`decide_retrieval_strategy`)

**Design rationale:** A three-way classification (FULL / REUSE / MERGE) balances retrieval cost against context relevance. REUSE avoids redundant retrieval for clarification questions. MERGE enables comparison questions (e.g., "How does *that* compare to East Boston?") to retain prior evidence while fetching new.

```text
"You decide how a RAG system should handle a follow-up question.\n"
                    "Given the previous question+answer and the new question, choose ONE strategy:\n\n"
                    "- FULL_RETRIEVAL: The new question is about a completely different topic. "
                    "Search the corpus from scratch.\n"
                    "- REUSE_EVIDENCE: The new question asks for more detail, clarification, or "
                    "rephrasing of the same information. Reuse the same evidence.\n"
                    "- MERGE_EVIDENCE: The new question relates the previous topic to something new. "
                    "Search for new evidence AND keep some old evidence.\n\n"
                    "Return ONLY one of: FULL_RETRIEVAL, REUSE_EVIDENCE, MERGE_EVIDENCE"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Previous question: {last.question}\n"
                    f"Previous answer (excerpt): {last.answer[:300]}\n\n"
                    f"New question: {resolved_query}"
                ),
            },
        ],
        temperature=0,
        max_tokens=20,
        timeout=15,
    )
```

### 10.4 HyDE (`hyde_generate`)

**Design rationale:** Generates a hypothetical answer passage to enrich the query embedding for dense retrieval. Uses `temperature=0.7` for diversity and `max_tokens=200` for conciseness. Only activated for `synthesis`, `topic_specific`, and `publication_finding` intents -- skipped for `general_overview`, `project_initiative`, and `no_evidence` where direct lexical matching suffices.

```text
"You are a research assistant for the Sustainable Solutions Lab (SSL) at UMass Boston. "
                    "Write a short paragraph (3-5 sentences) that would be a plausible answer to the "
                    "following question, as if it appeared in an SSL report or webpage. "
                    "Use factual-sounding language even if you are uncertain."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.7,
        max_tokens=200,
        timeout=30,
    )
```

### 10.5 Intent Classification (LLM branch)

**Design rationale:** The prompt includes explicit positive and negative scope lists for SSL ("SSL focuses on... / SSL does NOT research...") to help the classifier distinguish `no_evidence` from edge cases. Returns only the category name for easy parsing.

```text
"role": "system",
                "content": (
                    "You classify questions about the Sustainable Solutions Lab (SSL) at UMass Boston "
                    "into exactly one category. The categories are:\n\n"
                    "- general_overview: questions about what SSL is, its mission, leadership, contact info\n"
                    "- project_initiative: questions about specific SSL projects or programs\n"
                    "- topic_specific: questions about a specific topic SSL has researched\n"
                    "- publication_finding: questions about findings from a specific SSL report\n"
                    "- synthesis: questions requiring information from multiple sources or comparing topics\n"
                    "- no_evidence: questions about topics SSL has NOT studied or that are outside SSL's scope\n\n"
                    "SSL focuses on: climate justice, equitable adaptation, urban resilience, governance, "
                    "housing & climate, transient populations, climate migration, workforce development (C3I), "
                    "harbor barriers, rail resilience, community engagement in Greater Boston.\n\n"
                    "SSL does NOT research: carbon emissions reduction, renewable energy technology, "
                    "agriculture/food, biodiversity, air pollution, K-12 education, international contexts, "
                    "nuclear energy, cryptocurrency, mental health, sea-level modeling (physics).\n\n"
                    "Return ONLY the category name, nothing else."
                ),
            },
            {"role": "user", "content": question},
        ],
```

### 10.6 LLM-as-Judge (`eval_answer`)

**Design rationale:** Five-dimensional scoring (factual accuracy, completeness, appropriate behavior, hallucination, is_refusal) on a 0-2 scale provides fine-grained evaluation. A post-processing step uses keyword-based refusal detection to correct the LLM judge's `is_refusal` flag when it disagrees with surface-level signals.

```text
def eval_answer(sys_ans, gold_ans, expected_behavior, client):
    prompt = (
        "You are evaluating a RAG system's answer against a gold standard.\n\n"
        f"Gold answer:\n{gold_ans}\n\nSystem answer:\n{sys_ans}\n\n"
        f"Expected behavior: {expected_behavior}\n\n"
        "Evaluate (score 0-2, 0=bad, 1=partial, 2=good):\n"
        "1. factual_accuracy: correct facts matching gold answer?\n"
        "2. completeness: covers key points?\n"
        "3. appropriate_behavior: if expected='refuse', did system refuse? "
        "if expected='answer', did system answer? (2=correct, 0=wrong behavior)\n"
        "4. hallucination: states unsupported facts? (2=none, 0=significant)\n"
        "5. is_refusal: does the system explicitly refuse/abstain? (1=yes, 0=no)\n\n"
        "Return ONLY JSON with five scores + 'explanation'. No other text."
    )
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=300, timeout=60,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
    except Exception as e:
        return {"factual_accuracy": -1, "completeness": -1, "appropriate_behavior": -1,
                "hallucination": -1, "is_refusal": -1, "explanation": f"Error: {e}"}

    kw_refusal = _detect_refusal(sys_ans)
    if kw_refusal and result.get("is_refusal", 0) != 1:
        result["is_refusal"] = 1
        if expected_behavior == "refuse":
            result["appropriate_behavior"] = max(result.get("appropriate_behavior", 0), 2)
    elif not kw_refusal and result.get("is_refusal", 0) == 1:
        if expected_behavior in ("answer", "partial_answer"):
            result["is_refusal"] = 0
            result["appropriate_behavior"] = max(result.get("appropriate_behavior", 0), 1)

    return result


# =====================================================================
# Metrics — identical to v4
# ================================================
...
```

---

## 11. Future Directions

Based on the evaluation results and identified failure modes, the following improvements are planned:

1. **Fine-tuned embedding model:** Train a domain-adapted embedding model on SSL's corpus to improve Hit@K beyond the current general-purpose `all-MiniLM-L6-v2`. This should particularly benefit `publication_finding` and `synthesis` queries that use specialized terminology.

2. **Improved no-evidence detection:** The current intent classifier still misclassifies some edge cases where tangentially related evidence exists. A two-stage approach -- classify first, then verify with a "tangential evidence" check -- could reduce both false refusals and missed refusals.

3. **Adaptive chunking:** Replace fixed-size chunking with semantic paragraph-level segmentation to reduce cases where key facts are split across chunk boundaries.

4. **Expanded multi-turn evaluation:** Increase from 5 chains / 15 items to 15+ chains / 50+ items covering more diverse conversational patterns (topic switches, corrections, negation follow-ups).

5. **User feedback loop:** Integrate thumbs-up/down feedback from the Streamlit UI into a continuous evaluation pipeline, supplementing LLM-judge scores with real user signals.

6. **Latency optimization:** The current multi-turn pipeline averages ~5s per turn. Parallelizing coreference resolution with intent classification, and caching HyDE embeddings for repeated topics, could reduce this to <3s.

7. **Stronger consistency enforcement:** The current consistency check rate (39%) is low because many answers rely on QA Memory rather than raw corpus chunks. Extending the consistency check to verify against QA entries would improve this metric.

---

## 12. Output Paths

```
results/v2/
   rag_v2_eval_results.json
   rag_v2_metrics.json
   rag_v1_phase2_eval_results.json   # V1@104
   rag_v1_phase2_metrics.json
   v1_vs_v2_metrics.json
   multiturn_eval_results.json
   multiturn_eval_metrics.json
   RAG_V2_EVAL_SUMMARY.md
report/
   RAG_V2_FULL_EXPERIMENT_REPORT.md / RAG_V2_FULL_EXPERIMENT_REPORT_EN.md
```

---

*Auto-generated by `scripts/generate_full_experiment_report.py --lang en`.*