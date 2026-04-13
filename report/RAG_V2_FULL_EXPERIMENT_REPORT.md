# RAG V2 全量实验报告（Phase 2）

**生成时间（UTC）:** 2026-04-11 03:41:18
**数据集:** `data\eval_phase2\stakeholder_eval_phase2.json`（**104** 题）

---

## 1. 评测框架与数据

### 1.1 问题集更新

- **Phase 2 全量集:** **104** 题（在原有 70 题基础上增加 Phase 2 扩展题）。
- 新增维度包括：综合类（SYN-P2）、无证据边界（NEG-P2）、出版物深挖（PUB-P2）、**多轮对话链（MULTI-01～15，5 条链 × 3 轮）**。

### 1.2 题型分布（`question_type`）

| 题型 | 题数 |
|------|-----:|
| `no_evidence` | 22 |
| `general_overview` | 20 |
| `synthesis` | 18 |
| `project_initiative` | 17 |
| `topic_specific` | 15 |
| `publication_finding` | 12 |

### 1.3 难度与 Phase 分布

| 难度 | 题数 |
|------|-----:|
| medium | 47 |
| hard | 35 |
| easy | 22 |

| Phase | 题数 |
|------|-----:|
| 1 | 70 |
| 2 | 34 |

### 1.4 多轮子集（链标识）

| 链 | 题数 |
|----|-----:|
| A | 3 |
| B | 3 |
| C | 3 |
| D | 3 |
| E | 3 |

---

## 2. RAG V2 框架流程

```mermaid
flowchart TB
  U[用户问题] --> CR[指代消解 LLM]
  CR --> RS[检索策略 LLM]
  RS -->|REUSE| OLD[复用上轮证据]
  RS -->|FULL/MERGE| IC[意图分类]
  IC --> HY{HyDE?}
  HY -->|synthesis/topic/pub| H1[HyDE 段落 LLM]
  HY -->|overview/project/no_ev| SKIP[跳过 HyDE]
  H1 --> SQ{子查询拆分?}
  SKIP --> SQ
  SQ -->|synthesis| DC[分解 LLM]
  SQ -->|其他| RT[单次混合检索]
  DC --> RT
  RT --> MH[Multi-hop 覆盖检查 + 补检索]
  MH --> DIV[Diversity 截断]
  DIV --> GEN[生成回答 LLM]
  GEN --> CC{批量评测?}
  CC -->|是| CS[一致性检查 LLM]
  CC -->|否 UI| DONE[返回]
  CS --> DONE
  OLD --> GEN
```

**说明:** 全量单轮评测在 `python -m rag_v2.pipeline` 中实现为：每题前 `session.clear()`，再调用 `ask(..., enable_consistency=True)`，与 **检索+生成主链路** 一致；UI 为提速默认 `enable_consistency=False`。Multi-hop 补检索依赖会话历史中的实体，**独立单题评测**下通常不触发；多轮专项见第 5 节。

---

## 3. 核心指标（V2，全量 Phase2 集）

| 指标 | 数值 |
|------|------|
| Hit@3 | 15.8% |
| Hit@5 | 17.1% |
| 可答题准确度 | 75.0% |
| 全体准确度 | 69.2% |
| 完整性 | 62.5% |
| 幻觉率（越低越好） | 12.0% |
| 行为覆盖率 | 96.2% |
| 正确拒答率 | 90.9% |
| 误拒率 | 2.4% |
| 漏拒次数 | 2 |
| 答案-证据一致性 | 39.0% |

> **关于 Hit@3 / Hit@5：** 当前 `eval_retrieval` 仅统计 **原始语料块** 的 `chunk_id` / 文件名是否与 `gold_source_ids` 匹配。当检索结果以 **QA Memory（curated_qa）** 为主时，即使答案正确，也可能记为未命中语料块，因此 Hit@K 与 `evidence_supported_accuracy` 会偏低；解读时应结合 **LLM Judge 的准确度与幻觉率**。

### 3.1 按题型（V2）

| 题型 | N | 行为率 | 平均准确度 | Perfect |
|------|--:|--------:|-----------:|--------|
| `general_overview` | 20 | 100.0% | 82.5% | 13/20 |
| `project_initiative` | 17 | 94.1% | 79.4% | 11/17 |
| `topic_specific` | 15 | 100.0% | 76.7% | 8/15 |
| `publication_finding` | 12 | 100.0% | 79.2% | 7/12 |
| `synthesis` | 18 | 94.4% | 58.3% | 4/18 |
| `no_evidence` | 22 | 90.9% | 47.7% | 3/22 |

---

## 4. 与 V1 对比

### 4.1 历史基线：V1 在原始 70 题集

| 指标 | V1 (70) |
|------|--------:|
| Hit@5 | 18.2% |
| 可答题准确度 | 79.1% |
| 幻觉率 | 24.3% |

### 4.2 同集对比：V1 vs V2（均为 **104** 题 Phase2 集）

在相同题目与相同 Judge（`rag_v1.pipeline` 的 `LLM_MODEL`）下对比生成质量；**V1 走 V1 检索+生成；V2 走完整 `ask()` 链路（含 HyDE 条件启用、Multi-hop、多样性等）**。

| 指标 | V1 Phase2 | V2 | Δ (V2−V1) |
|------|----------:|---:|-----------|
| Hit@3 | 15.8% | 15.8% | 0.0% |
| Hit@5 | 17.1% | 17.1% | 0.0% |
| 可答题准确度 | 72.0% | 75.0% | +3.0% |
| 全体准确度 | 66.3% | 69.2% | +2.9% |
| 完整性 | 62.0% | 62.5% | +0.5% |
| 幻觉率（越低越好） | 24.0% | 12.0% | -12.0% |
| 行为覆盖 | 90.4% | 96.2% | +5.8% |
| 正确拒答 | 86.4% | 90.9% | +4.5% |
| 误拒率 | 8.5% | 2.4% | -6.1% |
| 漏拒次数 | 3 | 2 | - |

---

## 5. 多轮对话评测（15 题，5 链，会话内指代消解）

脚本：`test_multiturn_eval.py`；每链独立 `SessionMemory`，LLM-as-Judge 与全量集一致。

| 指标 | 得分 / 2.0 |
|------|-------------|
| 事实准确度 | **1.533** |
| 完整性 | 1.400 |
| 行为恰当性 | 2.000 |
| 幻觉控制 (2=无) | 1.533 |

**指代消解:** 10/10 触发 （100%）。

**按轮次:**

| 轮次 | 准确度 | 完整性 | 幻觉 |
|------|--------|--------|------|
| turn_1 | 1.800 | 1.600 | 1.400 |
| turn_2 | 1.600 | 1.600 | 1.400 |
| turn_3 | 1.200 | 1.000 | 1.800 |

---

## 6. 典型案例分析（单轮 + 多轮）

以下案例均来自最近一次评测落盘结果（`rag_v2_eval_results.json`、`multiturn_eval_results.json`），由脚本按 Judge 分数自动筛选；**多轮部分单独成节**，覆盖指代消解与跨话题比较。

### 6.1 单轮：代表性正确案例

#### 正确样例 1：`SYN-05`（synthesis，期望：answer）

- **问题：** What cross-cutting themes appear across SSL's different projects and publications?
- **Judge：** 事实=2，完整=2，行为=2，幻觉=2，拒答标记=0

**系统回答（摘录）：**

> Across the evidence provided, several recurring cross-cutting themes appear in SSL’s projects and publications: 1. Climate resilience and adaptation This is the most consistent theme. SSL’s major reports focus on resilience planning, adaptation governance, equitable adaptation, community preparedness, and resilience financing. Project work also highlights infrastructure resilience and adaptation strategies. 2. Equity and climate justice SSL’s research explicitly emphasizes climate justice and equitable adaptatio...

**Judge 说明：** _The system's answer accurately reflects the key themes outlined in the gold standard, covering all relevant points such as equity/inclusion, community engagement, climate governance/policy, infrastructure resilience, and transdisciplinary collaboration. There are no unsupported facts, and the system did not refuse to answer._

**简析：** 高分命中；`synthesis` 下证据与生成对齐良好。

#### 正确样例 2：`NEG-11`（no_evidence，期望：refuse）

- **问题：** Has SSL published any research on nuclear energy as a climate solution?
- **Judge：** 事实=2，完整=1，行为=2，幻觉=2，拒答标记=1

**系统回答（摘录）：**

> I could not find clearly supported SSL work on this topic in the current corpus.

**Judge 说明：** _The system's answer correctly indicates a lack of evidence regarding SSL's research on nuclear energy, aligning with the gold answer. However, it does not explicitly refuse or abstain from answering, which was the expected behavior. The completeness is partial as it does not fully encapsulate the definitive nature of the gold answer. There are no unsupported facts in the system's response._

**简析：** 高分命中；`no_evidence` 下证据与生成对齐良好。

#### 正确样例 3：`GEN-02`（general_overview，期望：answer）

- **问题：** What is SSL's mission?
- **Judge：** 事实=2，完整=2，行为=2，幻觉=2，拒答标记=0

**系统回答（摘录）：**

> SSL’s mission is to transform the climate research and action space by bridging across sectors, convening collaborators, and facilitating transdisciplinary research that centers and is led by historically and currently excluded communities. The evidence also describes this mission in closely related ways, including: - working to understand the disproportionate impacts of climate change on historically marginalized communities and developing sustainable, equitable solutions with them; and - serving as an engine o...

**Judge 说明：** _The system's answer accurately reflects the mission of SSL as described in the gold answer, including the focus on historically excluded communities and the collaborative approach. It also includes additional relevant details about the organization's work, enhancing completeness. There are no unsupported facts, and the system did not refuse to answer._

**简析：** 高分命中；`general_overview` 下证据与生成对齐良好。

### 6.2 单轮：代表性错误 / 高风险案例

筛选条件：事实分 ≤1，或行为分 <2（含应拒答却生成、应作答却严重偏题等）。

#### 问题样例 1：`SYN-P2-03`（synthesis，期望：answer）

- **问题：** What role do SSL's four partner institutes play in advancing its climate justice agenda, and how does each institute's focus area contribute to SSL's research portfolio?
- **Judge：** 事实=0，完整=0，行为=0，幻觉=2，拒答标记=1

**系统回答（摘录）：**

> I could not find clearly supported SSL work on this topic in the current corpus.

**Judge 说明：** _The system's answer does not provide any factual information related to SSL's climate justice agenda, which is a significant deviation from the gold answer. It fails to cover any key points and does not exhibit appropriate behavior as it does not answer the question but instead states a lack of information. There are no unsupported facts, as the system simply states it could not find relevant information. The system does explicitly refuse to provide an answer._

**简析：** 综合题在证据链不足时整体拒答，可检查子查询分解、HyDE 与 Multi-hop 是否覆盖全部子主题。

#### 问题样例 2：`NEG-10`（no_evidence，期望：refuse）

- **问题：** What is SSL's position on sea-level rise modeling and prediction?
- **Judge：** 事实=0，完整=1，行为=0，幻觉=0，拒答标记=0

**系统回答（摘录）：**

> SSL’s work indicates that sea-level rise modeling and prediction should be used as an evolving, science-based basis for planning, with conservative benchmarks and continued monitoring rather than treated as fixed forecasts. From the available evidence, SSL-associated reports note that: - sea-level rise projections are used together with historical records of tides and storms to forecast future coastal impacts, including the performance of adaptation measures such as harbor-wide barriers; - conservative vulnerabi...

**Judge 说明：** _The system's answer contradicts the gold answer by implying that SSL is involved in sea-level rise modeling and prediction, which the gold answer states is not documented. Therefore, it scores 0 for factual accuracy. It partially covers key points about planning and monitoring, so it scores 1 for completeness. The system did not refuse as expected, scoring 0 for appropriate behavior. There are no unsupported facts, so it scores 2 for hallucination. The system did not explicitly refuse, scoring 0 for is_refusal._

**简析：** 无证据题却依据检索片段展开论述，与 gold「无专门工作」冲突，属检索噪声或拒答策略问题。

#### 问题样例 3：`NEG-P2-03`（no_evidence，期望：refuse）

- **问题：** What green building certification or LEED programs does SSL run?
- **Judge：** 事实=0，完整=0，行为=2，幻觉=2，拒答标记=1

**系统回答（摘录）：**

> I could not find clearly supported SSL work on this topic in the current corpus.

**Judge 说明：** _The system's answer does not accurately reflect the facts presented in the gold answer, as it fails to acknowledge that SSL does not operate green building certification or LEED programs. It also does not cover the key points about SSL being a research institute and not a certification body. The system did not refuse as expected, and it does not state any unsupported facts, but it also does not provide a relevant answer._

**简析：** 使用了标准拒答话术，但 Judge 认为与 gold 要求的明确否定/补充说明仍不一致（常见于 no_evidence 边界题）。

### 6.3 多轮：正确链路（跨轮上下文 + 比较类）

展示 **链 B**（基础设施 vs 社区路径）：首轮建立「Cape Cod Rail」实体后，次轮用「that project」成功消解并与 East Boston 对比；Judge 满分，说明 **Multi-hop 补检索 + 指代消解** 在该类问题上有效。

#### 链 B：`MULTI-04` — 链 **B** 第 1 轮

- **用户原问：** What is the Cape Cod Rail Resilience project?
- **意图：** `project_initiative`
- **Judge：** 事实=2，完整=2，行为=2，幻觉=0

**系统回答（摘录）：**

> The Cape Cod Rail Resilience Project is a collaboration involving MassDOT, UMass Boston, and AECOM focused on improving rail safety and climate resilience along the Cape Main Line. It was launched after a major 300-foot rail embankment collapse in East Sandwich in 2020, which was linked to climate-change-induced drought conditions. The project addresses growing climate risks to critical transportation infrastructure and is inspired by Carlos Velásquez’s doctoral research. Velásquez, a PhD candidate at UMass Bost...

**Judge 说明：** _The system answer accurately reflects the key facts from the gold answer, including the collaboration between MassDOT, UMass Boston, and AECOM, the $2 million funding, the 300-foot embankment collapse, and the use of drones and sensors. It also adds relevant details about climate resilience and the involvement of Carlos Velásquez, which enhances the completeness without introducing unsupported facts._

**简析：** 幻觉分偏低：检索块中含扩展叙事，模型未充分裁剪；可结合一致性检查复核。

#### 链 B（亮点：比较题）：`MULTI-05` — 链 **B** 第 2 轮

- **用户原问：** How does that project compare to SSL's work in East Boston in terms of approach?
- **指代消解后：** How does the Cape Cod Rail Resilience Project compare to SSL's work in East Boston in terms of approach?
- **意图：** `project_initiative`
- **Judge：** 事实=2，完整=2，行为=2，幻觉=2

**系统回答（摘录）：**

> The two efforts share a climate resilience focus, but the evidence shows different approaches. ### Cape Cod Rail Resilience Project This project is focused on transportation infrastructure resilience along the Cape Main Line. Its approach emphasizes: - improving rail safety and climate resilience - integrating advanced technology with stakeholder-engaged research - using aerial surveys or drones, hydrological drought analysis, and real-time monitoring systems - studying three pilot sites in collaboration with Ma...

**Judge 说明：** _The system's answer accurately reflects the key points from the gold answer, detailing the distinct approaches of the Cape Cod Rail project and the East Boston work. It correctly identifies the focus on technology and infrastructure for Cape Cod and the community-centered, equity-focused approach for East Boston. The answer is complete and does not include unsupported facts._

**简析：** 多轮上下文与检索协同良好；指代消解后的查询与证据匹配。

#### 链 B：`MULTI-06` — 链 **B** 第 3 轮

- **用户原问：** What about the harbor barrier study — is that approach more similar to the rail project or to East Boston?
- **指代消解后：** Is the approach taken in the harbor barrier study more similar to the Cape Cod Rail Resilience Project or to Sustainable Solutions Lab's work in East Boston?
- **意图：** `project_initiative`
- **Judge：** 事实=1，完整=1，行为=2，幻觉=2

**系统回答（摘录）：**

> The evidence suggests the harbor barrier study is more similar to the Cape Cod Rail Resilience Project than to SSL’s East Boston work, but only at a broad level. ### Why it appears closer to the rail project The harbor barrier study focuses on large-scale physical infrastructure for climate protection. The evidence describes: - an ongoing study of a harbor-wide barrier from Winthrop to Hull to protect against major storm surge - analysis of feasibility and cost - a recommendation that the City focus on a shore-b...

**Judge 说明：** _The system's answer contains some correct facts about the harbor barrier study and its similarities to the Cape Cod Rail project, but it lacks specific details about the East Boston work, which is a key point in the gold answer. It does not fully address the community engagement aspect mentioned in the gold answer. The system does not refuse to answer and provides a structured response, but it does not fully meet the completeness expected._

**简析：** 指代已消解但 Judge 仍判要点不足：可能需在生成提示中强调与 gold 对齐的枚举式要点（金额、时长等）。

**链级小结：** 首轮检索锁定 Cape Cod 项目；追问比较时 Multi-hop 可补全 East Boston 证据；第三轮 harbor barrier 属三向对比，仍依赖前两轮实体在会话中的可抽取性。

### 6.4 多轮：薄弱 / 部分失分案例

以下选取 **事实或完整性未满分** 的追问或末轮，用于说明多轮仍可能出现的边际问题（如金额表述被 Judge 判为不完整、出版物追问细节不足等）。

#### 待改进样例：`MULTI-08` — 链 **C** 第 2 轮

- **用户原问：** Can you tell me more about the executive director's background?
- **指代消解后：** Can you tell me more about B.R. Balachandran's background?
- **意图：** `general_overview`
- **Judge：** 事实=1，完整=2，行为=2，幻觉=0

**系统回答（摘录）：**

> The Executive Director of SSL is Dr. B.R. Balachandran. His background includes: - He became SSL’s Executive Director in February 2024. - He provides strategic leadership for SSL and oversees research projects. - He holds a Ph.D. in Regional Planning from the University of Illinois at Urbana-Champaign, earned in 2022. - Before leading SSL, he held a visiting faculty position at UMass Boston’s School for the Environment and a research position at the University of North Texas at Denton. - He is a certified urban ...

**Judge 说明：** _The system's answer contains some inaccuracies regarding the timeline of Dr. Balachandran's PhD and his position at SSL, but it correctly identifies his educational background and experience with UNDP, the World Bank, and KfW. The answer is comprehensive, covering key points about his roles and expertise. There are no unsupported facts, and the system did not refuse to answer._

**简析：** 指代已消解但 Judge 仍判要点不足：可能需在生成提示中强调与 gold 对齐的枚举式要点（金额、时长等）。

#### 待改进样例：`MULTI-02` — 链 **A** 第 2 轮

- **用户原问：** Who funds it and how much did they give?
- **指代消解后：** Who funds the Climate Careers Curricula Initiative (C3I), and how much funding did the Liberty Mutual Foundation provide?
- **意图：** `project_initiative`
- **Judge：** 事实=1，完整=1，行为=2，幻觉=2

**系统回答（摘录）：**

> The Climate Careers Curricula Initiative is funded by the Liberty Mutual Foundation. The evidence states the grant is $540,000 over three years.

**Judge 说明：** _The system correctly identifies the funding source but does not explicitly mention the amount or duration of the grant, leading to partial factual accuracy and completeness. The system answered appropriately without refusing._

**简析：** 指代已消解但 Judge 仍判要点不足：可能需在生成提示中强调与 gold 对齐的枚举式要点（金额、时长等）。

#### 待改进样例：`MULTI-03` — 链 **A** 第 3 轮

- **用户原问：** What kinds of jobs does it train people for?
- **指代消解后：** What kinds of blue and green jobs in Greater Boston does the Climate Careers Curricula Initiative (C3I) train people for?
- **意图：** `project_initiative`
- **Judge：** 事实=1，完整=1，行为=2，幻觉=2

**系统回答（摘录）：**

> C3I trains people for high-demand blue and green jobs in Greater Boston. The evidence specifically mentions training related to: - Solar installation - Energy auditing - Nature-based solutions More broadly, the initiative is designed to create microcredentialed training programs that open career pathways for underrepresented populations, especially vulnerable young adults, people of color, and low-income adults from environmental justice communities.

**Judge 说明：** _The system answer captures the essence of C3I's focus on training for blue and green jobs but lacks specific details about the types of training (solar installation, energy auditing, nature-based solutions) mentioned in the gold answer. It also does not address the microcredentialing aspect or the target populations, leading to a partial score for completeness. There are no unsupported facts in the system answer._

**简析：** 指代已消解但 Judge 仍判要点不足：可能需在生成提示中强调与 gold 对齐的枚举式要点（金额、时长等）。

#### 待改进样例：`MULTI-06` — 链 **B** 第 3 轮

- **用户原问：** What about the harbor barrier study — is that approach more similar to the rail project or to East Boston?
- **指代消解后：** Is the approach taken in the harbor barrier study more similar to the Cape Cod Rail Resilience Project or to Sustainable Solutions Lab's work in East Boston?
- **意图：** `project_initiative`
- **Judge：** 事实=1，完整=1，行为=2，幻觉=2

**系统回答（摘录）：**

> The evidence suggests the harbor barrier study is more similar to the Cape Cod Rail Resilience Project than to SSL’s East Boston work, but only at a broad level. ### Why it appears closer to the rail project The harbor barrier study focuses on large-scale physical infrastructure for climate protection. The evidence describes: - an ongoing study of a harbor-wide barrier from Winthrop to Hull to protect against major storm surge - analysis of feasibility and cost - a recommendation that the City focus on a shore-b...

**Judge 说明：** _The system's answer contains some correct facts about the harbor barrier study and its similarities to the Cape Cod Rail project, but it lacks specific details about the East Boston work, which is a key point in the gold answer. It does not fully address the community engagement aspect mentioned in the gold answer. The system does not refuse to answer and provides a structured response, but it does not fully meet the completeness expected._

**简析：** 指代已消解但 Judge 仍判要点不足：可能需在生成提示中强调与 gold 对齐的枚举式要点（金额、时长等）。

**多轮共性结论：**

1. **指代消解**在 10/10 追问上触发，整体将「it / that project / the survey」写进可检索的完整问句，是正确比较类回答的前提。
2. **Turn 3** 平均分数低于 Turn 1/2（见 §5），常见原因是问题更综合、证据需跨多个前文主题同时覆盖，Judge 对完整性更严。
3. **幻觉分偶发偏低**多与检索片段中含人名/项目叙事有关，需结合一致性检查与来源面板人工复核。


## 7. 提示词摘录（实现源码中）

### 7.1 生成回答（`generate_answer_v2` system prompt）

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

### 7.2 指代消解（`resolve_coreference`）

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

### 7.3 检索策略（`decide_retrieval_strategy`）

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

### 7.4 HyDE（`hyde_generate`）

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

### 7.5 意图分类 LLM 分支（节选）

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

### 7.6 LLM-as-Judge（`eval_answer`）

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

## 8. 产物路径

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
└── RAG_V2_FULL_EXPERIMENT_REPORT.md   # 本报告
```

---

*本文件由 `scripts/generate_full_experiment_report.py` 自动生成。*