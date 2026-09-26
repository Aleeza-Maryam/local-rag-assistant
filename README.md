<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!--                        HEADER WITH BADGES                              -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->

<div align="center">

# Local Knowledge Base RAG Assistant

### Contextual Q&A over your PDF documents — with exact citations, zero hallucination.

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

> **Try it:** Clone the repo, drop in your PDFs, and ask questions in natural language.

---

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!--                          KEY FEATURES                                  -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->

## Key Features

<table>
<tr>
<td width="50%" valign="top">

### Core Capabilities

- **Multi-PDF Ingestion** — Upload many PDFs at once
- **Page-Level Citations** — Every answer cites `file.pdf, Page X`
- **Local Vector Storage** — ChromaDB runs on your machine
- **Semantic Search** — Sentence-transformer embeddings
- **Dual LLM Providers** — Gemini or Groq, switchable at runtime
- **No Hallucination** — Strict prompt enforces grounding

</td>
<td width="50%" valign="top">

### Interfaces & Tooling

- **Streamlit Web UI** — Interactive chat interface
- **FastAPI REST API** — Programmatic access
- **Python Module** — Import `src` into your scripts
- **Environment Config** — Secrets via `.env`
- **Modular Design** — Ingestion, retrieval, generation decoupled
- **Type Hints Throughout** — Clear component interfaces

</td>
</tr>
</table>

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
   ┌─────────────────┐
   │  Embedder       │   all-MiniLM-L6-v2 → 384-dim vectors
   │  (retriever.py) │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  ChromaDB       │   Persistent local vector store
   │  (data/chroma)  │   Cosine similarity
   └─────────────────┘
