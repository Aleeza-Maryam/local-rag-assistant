# local-rag-assistant
Contextual Q&amp;A system with PDF citations using RAG
<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!--                        HEADER WITH BADGES                              -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->

<div align="center">

# Local Knowledge Base RAG Assistant

### Contextual Q&A over your PDF documents — with hybrid retrieval, streaming answers, and exact citations.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B6B?style=for-the-badge&logo=databricks&logoColor=white)](https://www.trychroma.com/)
[![Gemini](https://img.shields.io/badge/Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=lightning&logoColor=white)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[![GitHub Stars](https://img.shields.io/github/stars/Aleeza-Maryam/local-rag-assistant?style=social)](https://github.com/Aleeza-Maryam/local-rag-assistant/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Aleeza-Maryam/local-rag-assistant?style=social)](https://github.com/Aleeza-Maryam/local-rag-assistant/network)
[![GitHub Issues](https://img.shields.io/github/issues/Aleeza-Maryam/local-rag-assistant)](https://github.com/Aleeza-Maryam/local-rag-assistant/issues)

</div>

---

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!--                          TABLE OF CONTENTS                             -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->

## Table of Contents

<details open>
<summary><b>Click to expand / collapse</b></summary>

- [Overview](#-overview)
- [Key Features](#-key-features)
- [What's New](#-whats-new)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
  - [Streamlit UI](#streamlit-ui)
  - [REST API](#rest-api)
  - [Python Module](#python-module)
- [API Reference](#-api-reference)
- [How It Works](#-how-it-works)
  - [Hybrid Retrieval](#hybrid-retrieval)
  - [Streaming Answers](#streaming-answers)
  - [Conversation Memory](#conversation-memory)
- [Design Decisions](#-design-decisions)
- [Testing](#-testing)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

</details>

---

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!--                             OVERVIEW                                   -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->

## Overview

**Local Knowledge Base RAG Assistant** is a production-ready Retrieval-Augmented Generation (RAG) system that ingests PDF documents, stores embeddings locally, and answers user questions with **exact citations** — file name and page number included.

Unlike generic chatbots, this system **will not hallucinate**. If the answer is not present in the ingested documents, the model explicitly states so. This makes it suitable for:

- **Academic research** — Quickly find passages in papers
- **Legal document analysis** — Verify claims with page-level citations
- **Technical documentation review** — Onboarding and reference
- **Compliance workflows** — Auditable Q&A

The system combines **three advanced techniques** typically found only in production RAG pipelines:

1. **Hybrid Search** — BM25 keyword matching + semantic vector search, fused with Reciprocal Rank Fusion
2. **Streaming Answers** — Token-by-token response generation for real-time UX
3. **Multi-Turn Memory** — Conversational context preserved across follow-up questions

> **Try it:** Clone the repo, drop in your PDFs, and ask questions in natural language.

---

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!--                          KEY FEATURES                                  -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->

## Key Features

<table>
<tr>
<td width="50%" valign="top">

### Retrieval

- **Hybrid Search** — BM25 + semantic vector search
- **Reciprocal Rank Fusion** — Combines keyword and semantic rankings
- **Tunable Alpha** — Control search balance (0 = keyword, 1 = semantic)
- **Page-Level Citations** — Every answer cites `file.pdf, Page X`
- **Deterministic Chunk IDs** — No duplicates on re-ingestion

</td>
<td width="50%" valign="top">

### Generation

- **Streaming Responses** — Token-by-token output
- **Multi-Turn Memory** — Follow-up questions work naturally
- **Dual LLM Providers** — Gemini or Groq, switchable at runtime
- **Strict Grounding** — Model refuses to answer outside context
- **Conversation History** — Last 4 turns passed to the LLM

</td>
</tr>
<tr>
<td width="50%" valign="top">

### Interfaces

- **Streamlit Web UI** — Interactive chat with source cards
- **FastAPI REST API** — Programmatic access with Swagger docs
- **Python Module** — Import `src` into your own scripts
- **Dark Theme** — Professional look out of the box

</td>
<td width="50%" valign="top">

### Developer Experience

- **Modular Design** — Ingestion, retrieval, generation fully decoupled
- **Environment Config** — All secrets and paths via `.env`
- **Type Hints** — Clear interfaces between components
- **Local Vector Storage** — No cloud dependency for retrieval
- **Secrets Protected** — API keys never leave your machine

</td>
</tr>
</table>

---

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!--                          WHAT'S NEW                                    -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->

## What's New

### v1.1 — Advanced RAG Features

| Feature | Description | Impact |
|:---:|:---|:---|
| **Hybrid Search** | BM25 + semantic search with RRF fusion | Exact keyword matches now work (e.g., "Section 4.2") |
| **Streaming Answers** | `stream_answer()` yields tokens as they are generated | ChatGPT-like UX; perceived latency cut by ~80% |
| **Conversation Memory** | Last 4 turns of dialogue passed to the LLM | Natural follow-ups like "explain that further" |
| **Alpha Slider** | Tune search mode from keyword to semantic | Users can adapt to their document type |
| **`rank-bm25`** | Industry-standard keyword ranking library | Zero-configuration BM25 |

### v1.0 — Initial Release

- PDF ingestion with page-level metadata
- ChromaDB persistent local vector store
- Dual LLM support (Gemini + Groq)
- Streamlit UI and FastAPI endpoints
- Comprehensive README and setup instructions

---

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!--                           ARCHITECTURE                                 -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->

## Architecture

### Indexing Phase (offline)

```text
   ┌─────────────┐
   │  PDF File   │
   └──────┬──────┘
          │
          ▼
   ┌─────────────────┐
   │  PDF Parser     │   pypdf extracts text page by page
   │  (ingest.py)    │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  Text Chunker   │   500-char chunks with 50-char overlap
   │  (utils.py)     │   Preserves source + page metadata
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐     ┌─────────────────┐
   │  Embedder       │     │  BM25 Indexer   │
   │  (retriever.py) │     │  (retriever.py) │
   └────────┬────────┘     └────────┬────────┘
            │                       │
            ▼                       ▼
   ┌─────────────────┐     ┌─────────────────┐
   │  ChromaDB       │     │  BM25Okapi      │
   │  (384-dim vecs) │     │  (in-memory)    │
   └─────────────────┘     └─────────────────┘
