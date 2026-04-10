# Phase 2 Evaluation Dataset Design

## Overview

For Phase 2, I extended the original evaluation dataset from Phase 1 to better test the weaknesses observed in our RAG system. The goal was not to introduce new knowledge, but to **stress-test the system’s ability to retrieve, reason, and respond under more realistic and challenging conditions**.

The final Phase 2 dataset contains **24 additional questions**, designed to expand evaluation coverage and difficulty.

---

## Motivation

From Phase 1 results, several key weaknesses were identified:

- Difficulty with **multi-document synthesis**
- Retrieval failures due to **wording mismatch**
- Sensitivity to **clean, structured queries only**
- Occasional **hallucinations on borderline questions**

To address these issues, the Phase 2 dataset shifts focus toward:

- reasoning over simple retrieval  
- robustness to real-world phrasing  
- correct refusal of unsupported queries  

---

## Dataset Summary

The Phase 1 evaluation dataset consisted of 70 questions distributed across multiple categories, including general overview, topic-specific, synthesis, and no-evidence (refusal) cases.

Key characteristics of the Phase 1 dataset:

- 53 answerable questions and 15 refusal cases  
- 8 synthesis questions (~28.6%)  
- Balanced mix of website, PDF, and multi-source evidence  

For Phase 2, 24 additional questions were introduced with a focus on:

- increasing synthesis complexity  
- improving coverage of retrieval mismatch scenarios  
- introducing realistic user phrasing  
- strengthening refusal edge cases  

This results in a more challenging and representative evaluation set for measuring system improvements.

---

## Dataset Structure

The 24 new questions are divided into four categories:

---

### 1. Synthesis Questions (10)

These questions require the system to combine information from multiple sources rather than retrieving a single fact.

Examples include:
- connecting housing and climate resilience  
- linking community engagement with research  
- combining policy, infrastructure, and community work  

These were added because Phase 1 had limited synthesis coverage, and the system struggled with multi-step reasoning.

---

### 2. Retrieval Mismatch Questions (4)

These questions use **indirect or paraphrased wording** that does not directly match the original documents.

Examples include:
- “Spanish-speaking communities” instead of “multilingual report”
- “coastal communities” instead of “East Boston”
- “training people for jobs” instead of “C3I program”

These test whether the retriever can handle **semantic similarity instead of keyword matching**, which was a known weakness in Phase 1.

---

### 3. Realistic User Phrasing (4)

These questions are written in a more **casual, conversational style** to reflect real user behavior.

Examples:
- “What kind of stuff does SSL actually work on?”
- “Do they actually do anything about climate change?”

These differ from Phase 1’s structured questions and are designed to test:
- intent understanding  
- robustness to vague or informal queries  

---

### 4. Refusal Edge Cases (6)

These are **plausible but unsupported questions** that the system should refuse.

Unlike Phase 1 (which included mostly obvious out-of-scope questions), these are more subtle:

Examples:
- emissions reduction in cities  
- climate prediction tools  
- building infrastructure projects  

These questions test whether the system can:
- avoid hallucinating  
- correctly identify unsupported claims  
- distinguish between related vs actual scope  

---

## Key Improvements Over Phase 1

Compared to the original dataset:

- Increased emphasis on **reasoning over retrieval**
- Introduced **real-world query variability**
- Added **harder refusal scenarios**
- Expanded evaluation beyond keyword-based matching

The dataset now better reflects real user interactions and provides a stronger benchmark for evaluating improvements in retrieval, reranking, and generation.

---

## Design Approach

All questions were grounded in the **existing Phase 1 knowledge base** and reuse valid source documents.

However, rather than repeating or lightly rewording existing questions, the focus was on:

- re-framing existing topics in more challenging ways  
- introducing indirect and realistic phrasing  
- increasing reasoning complexity (especially for synthesis)  
- targeting known failure modes observed in Phase 1  

This ensures that improvements measured in Phase 2 reflect **true system performance gains** (retrieval, reasoning, and refusal), rather than differences caused by expanding the dataset with new knowledge.

---

## Summary

Overall, the Phase 2 evaluation dataset shifts the focus from:

> Can the system answer known questions?

to:

> Can the system handle realistic, complex, and imperfect queries?

This makes it a more effective benchmark for evaluating a production-level RAG system.