# LiveDB ETL — Continuous Literature Ingestion, Triage, and Vectorized Indexing

A production‑ready async pipeline that **discovers the latest biomedical papers**, **classifies abstracts for PICOS‑style eligibility**, **acquires legal full‑text (PDF or BioC fallback)**, and **ingests chunked content into a pgvector‑backed knowledge base** for retrieval‑augmented applications.

> **Status:** Agents for orchestration/QA are **in development** (see “Agents (Planned)” below). The current system focuses on acquisition, triage, and indexing.

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Data Flow](#data-flow)
- [Repository Layout](#repository-layout)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Environment Variables](#environment-variables)
  - [Install](#install)
  - [Run the Flow](#run-the-flow)
- [Configuration](#configuration)
- [ETL Details](#etl-details)
  - [OpenAlex Fetch](#openalex-fetch)
  - [PubMed/PMC Fetch](#pubmedpmc-fetch)
  - [Abstract Classification](#abstract-classification)
  - [Full‑text Acquisition](#full-text-acquisition)
  - [Chunking & Ingestion](#chunking--ingestion)
- [Database Expectations](#database-expectations)
- [Operational Guidance](#operational-guidance)
  - [Logging](#logging)
  - [Performance Tuning](#performance-tuning)
  - [Troubleshooting](#troubleshooting)
- [Security & Compliance Notes](#security--compliance-notes)
- [Agents (Planned)](#agents-planned)
- [Roadmap](#roadmap)
- [FAQ](#faq)
- [License](#license)
- [Citations & Attribution](#citations--attribution)

---

## Overview

**LiveDB ETL** continuously surfaces recent literature for a given query (e.g., *“dementia”*), performs **eligibility triage** using a fine‑tuned multi‑head classifier over a transformer encoder, **downloads or reconstructs full‑text**, and **indexes** results into a Postgres/pgvector‑backed knowledge base wrapped by **Agno** abstractions (vector DB + contents DB).

It is designed to be:
- **Asynchronous & resilient** (httpx, asyncio, tenacity retries; bounded concurrency for downloads)
- **Legally compliant** (Open Access first; BioC fallback; PMC OA FTP for licensed content)
- **RAG‑ready** (semantic chunking; reference removal; pgvector hybrid search)
- **Operable** (Prefect orchestration, structured logging, configurable concurrency)

---

## Key Features

- **Multi‑source discovery**
  - **OpenAlex**: newest articles by `publication_date` (filters for OA & language).
  - **PubMed + PMC**: PMID discovery + legal full‑text via PMC OA utilities.
- **Abstract triage**
  - Multi‑task classifier (`P_AB`, `I_AB`, `C_AB`, `O_AB`, `S_AB`) → `yes/maybe/no`.
  - Simple rule for `final_pred`: if `S_AB_pred == "no"` then `final_pred = "no"` else `"yes"` (customizable).
- **Full‑text acquisition**
  - Direct OA PDF via `oa_pdf` (OpenAlex) or **PMC OA FTP**.
  - **BioC** fallback → reconstructs text and renders to PDF to preserve the ingestion contract.
- **Chunking**
  - **Semantic chunking** with `CustomChunking` that **stops at “References”** to avoid noisy embeddings.
- **Indexing**
  - **Agno** Knowledge layer → Postgres/pgvector for hybrid retrieval + separate contents store.
- **Orchestration & Observability**
  - **Prefect** flow with caching, retries, bounded concurrency, and rotating log files.

---

## Architecture

```
                ┌──────────────────────────────────────────────────────────┐
                │                           User                           │
                └───────────────┬──────────────────────────────────────────┘
                                │ query="dementia", window, max_records
                                ▼
                     ┌──────────────────────┐
                     │  Prefect Flow        │
                     │  (main.livedb_flow)  │
                     └─────────┬────────────┘
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
┌─────────────────┐   ┌──────────────────┐   ┌───────────────────────┐
│ OpenAlex Fetch  │   │ PubMed ESearch   │   │ Classifier (multihead)│
│ (OA results)    │   │ + EFetch (meta)  │   │  P/I/C/O/S -> yes/no  │
└───────┬─────────┘   └─────────┬────────┘   └───────────┬───────────┘
        │                       │                        │ filter(final_pred=="yes")
        │                       │                        ▼
        │                       │             ┌────────────────────────┐
        │                       │             │ Full-text Acquisition  │
        │                       │             │ OA PDF | PMC OA | BioC │
        │                       │             └───────────┬────────────┘
        │                       │                         ▼
        │                       │              ┌─────────────────────┐
        │                       │              │ Chunk & Embed       │
        │                       │              │ (CustomChunking)    │
        │                       │              └──────────┬──────────┘
        │                       │                         ▼
        │                       │           ┌────────────────────────────┐
        │                       └──────────▶  Agno Knowledge (pgvector)  │
        │                                   │  + Contents Postgres       │
        │                                   └────────────────────────────┘
        ▼
   Logs/Prefect UI
```

---

## Data Flow

1. **Discovery**
   - OpenAlex `search` + `filter` window: `(start_day, days_back)` defines a **moving window** (e.g., “from 31 to 30 days ago”).  
   - PubMed ESearch → PMIDs for the same window; EFetch returns structured metadata.

2. **Triage**
   - `livedb/CheckAbsModel.py` loads a **Saved Multi‑Task Model** (5 heads) from `Config.MODEL_DIR` and predicts task‑wise labels + confidences.
   - `final_pred` policy is currently simple and can be replaced with your inclusion logic.

3. **Acquisition**
   - **OpenAlex**: try `oa_pdf` with headers, fall back to **Playwright** with stealth when needed (handles CF/JS‑gated flows).
   - **PMC**: OA FTP packages (`.tar.gz`) are downloaded and **extracted to PDFs**; if not available, fetch **BioC** XML and render to PDF text.

4. **Ingestion**
   - **CustomChunking** removes everything after a “References” sentinel.
   - Metadata (year, authors, journal, P/I/C/O/S flags, etc.) is stored alongside embedded chunks in **pgvector** and **contents** tables.

---

## Repository Layout

```
agents/
  agents.py				 # (in development) — autonomous AI agents that will operate on top of the pgvector KB (retrieval, maintenance, QA, and smart acquisition logic)
dbs/
  IngestToDB.py          # Agno Knowledge setup + async ingestion
  utils.py               # CustomChunking (SemanticChunking subclass)
livedb/
  CheckAbsModel.py       # Multi-head classifier load + async inference
  GetLatestPapers.py     # PubMed/PMC helpers, FTP, BioC -> PDF, utilities
  OpenAlexDownload.py    # OpenAlex client + robust PDF downloader
  utils.py               # save_text_as_pdf_async (ReportLab)
.gitignore
.python-version
Config.py                # Pydantic config model + env var binding
main.py                  # Prefect flow: end-to-end ETL
```

---

## Quick Start

### Prerequisites

- **Python**: 3.13 (as per `.python-version`)
- **Postgres** with **pgvector** extension enabled
- **Node‑less Playwright Python** runtime requirements (Chromium is auto‑managed by Playwright)
- **System packages** commonly needed for playwright & reportlab (varies by OS)

```bash
# Ubuntu/Debian (example)
sudo apt-get update
sudo apt-get install -y libglib2.0-0 libnss3 libatk1.0-0 libatk-bridge2.0-0 \
    libx11-xcb1 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
    libasound2 fonts-liberation libxshmfence1
```

### Environment Variables

Create a `.env` in repo root:

```env
# NCBI
NCBI_EMAIL=you@example.com
NCBI_API_KEY=

# OpenAlex
OPENALEX_MAILTO=you@example.com

# PMC FTP (optional if anonymous)
FTP_USER=anonymous
FTP_PASSWORD=anonymous@

# OpenAI / Embeddings
OPENAI_API_KEY=sk-...
# Choose models in Config.py
# MODEL_NAME=gpt-4.1-mini
# EMBEDDING_MODEL=text-embedding-3-small

# Postgres / pgvector
PGVECTOR_URL=postgresql+psycopg://user:pass@host:5432/dbname
PGVECTOR_TABLE=knowledge_vectors
PGVECTOR_CONTENTS_URL=postgresql+psycopg://user:pass@host:5432/dbname
PGVECTOR_CONTENTS_TABLE=knowledge_contents
```

> The `Config.py` sets sensible defaults (e.g., API endpoints, headers, paths). Adjust the table names to match your schema.

### Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

# Core deps (pin versions as needed)
pip install httpx[http2] tenacity arrow lxml aiofiles aioftp pandas loguru pydantic python-dotenv \
            reportlab tqdm prefect playwright playwright-stealth torch transformers \
            agno pgvector psycopg[binary] sqlalchemy

# Install browser binaries for Playwright
python -m playwright install chromium
```

### Run the Flow

```bash
# Optional: run a temporary Prefect server (local)
prefect server start  # in a separate terminal

# Execute the ETL flow
python -m main  # or: python main.py

# Pass custom args by editing defaults in livedb_flow(...)
```

Default behavior pulls a **small window** of recent “dementia” papers, triages, downloads, and ingests.

---

## Configuration

`Config.py` (Pydantic) reads environment variables and exposes:
- **External APIs**: E‑Utils, BioC, OpenAlex
- **Storage**: `PDF_DIR`, `MODEL_DIR`
- **Models**: `MODEL_NAME`, `EMBEDDING_MODEL`
- **DB**: `PGVECTOR_URL`, `PGVECTOR_TABLE`, `PGVECTOR_CONTENTS_URL`, `PGVECTOR_CONTENTS_TABLE`
- **HTTP Headers**: `COMMON_HEADERS` for PDF downloads
- **HEADLESS** mode for Playwright

You can override via `.env` or environment variables.

---

## ETL Details

### OpenAlex Fetch
- `livedb/OpenAlexDownload.py::fetch_openalex_latest(...)`
- Filters on `publication_date` window; `only_articles=True`; `only_oa=True`; `language=en`
- Returns a de‑duplicated `DataFrame` with fields:
  - `id`, `pmid`, `title`, `publication_date`, `pub_year`, `journal`, `doi`, `is_oa`, `oa_pdf`, `authors`, `concepts`, `url`, `abstract`

### PubMed/PMC Fetch
- `pubmed_esearch` → PMIDs for the window (*edat* filter); `retmax` configurable
- `pubmed_efetch` → Medline XML parsed into: `pmid`, `pmcid`, `doi`, `title`, `journal`, `pub_year`, `authors`, `url`, `abstract`
- `try_fetch_pmc_fulltext_pdf`
  - Calls PMC OA service for **licensed** links (FTP: `oa_package`/`oa_pdf`)
  - Extracts **PDFs** from `*.tar.gz` or downloads direct `.pdf`
  - If no PDF, tries **BioC** → reconstructs text and persists as PDF

### Abstract Classification
- `livedb/CheckAbsModel.py`
  - Loads **tokenizer + encoder** from `MODEL_DIR`
  - Multi‑head linear classifiers (one per task) → logits → softmax
  - Returns task predictions & confidences
- Current decision rule:
  - `final_pred = "no"` **iff** `S_AB_pred == "no"`, else `"yes"`

### Full‑text Acquisition
- `download_pdf_async` first tries HTTP (HEAD/GET) with realistic headers & referer
- On `401/403` or non‑PDF responses: **Playwright (Chromium) + Stealth** fallback to handle bot mitigations
- PMC path tries legal OA routes and licenses; BioC produces a text PDF via `ReportLab`

### Chunking & Ingestion
- `dbs/utils.py::CustomChunking` derives from `SemanticChunking` and truncates at “References”
- `dbs/IngestToDB.py` wires:
  - **Agno** `OpenAIEmbedder`
  - `PDFReader` (split pages, read images, chunking strategy)
  - `PgVector` (hybrid search) + `PostgresDb` for contents
- Per‑record metadata stored with chunks:
  - `publication_year`, `date_added`, `author`, `title`, `journal`, `abstract`,
  - `population_flag`, `intervention_flag`, `comparator_flag`, `outcome_flag`, `study_design_flag`, `qualification_flag`

---

## Database Expectations

You should provision:
- **pgvector extension**: `CREATE EXTENSION IF NOT EXISTS vector;`
- Tables (example; Agno can manage its own schema—adjust if you manage DDL yourself):
  - `knowledge_vectors` (id, doc_id, chunk, embedding, metadata, ...)
  - `knowledge_contents` (doc_id, path, metadata, ...)

> Ensure the `PGVECTOR_*` URLs and table names in `.env` match your deployment.

---

## Operational Guidance

### Logging
- Rotating file: `logs/livedb.log` (10 MB rotation, 10-day retention)
- Prefect‑compatible sink forwards Loguru records to the task/flow run logger

### Performance Tuning
- Concurrency:
  - **Downloads**: `DOWNLOAD_SEM = 8` (increase cautiously—remote servers may throttle)
  - **Playwright**: `BROWSER_SEM = 3` (each Chromium is heavy)
- Retries & timeouts are defined via **tenacity** decorators and task‑level Prefect options

### Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| 403/401 on PDF | Bot/CF protection | Ensure Playwright installed; keep `HEADLESS=True`; fallback path triggers |
| Non‑PDF bytes saved | Misleading `content-type` or redirect to HTML | `_looks_like_pdf` guard logs errors; fallback to Playwright |
| Empty OA results | Filter window too narrow | Increase `max_records` or widen `start_day/days_back` |
| CUDA OOM / CPU slow | Model too large or env mismatch | Set `DEVICE=cpu` in classifier call or reduce batch size |
| No pgvector table | Missing DDL | Create tables or let Agno manage; verify `PGVECTOR_*` configs |
| BioC fallback garbled | Unicode font missing | Pass `ttf_font_path` to `save_text_as_pdf_async` |

---

## Security & Compliance Notes

- **Only ingest legally accessible content** (Open Access or licensed through PMC OA services).
- `.gitignore` avoids committing `.env`, `pdfs/`, `models/`, and logs.
- The packed repository (“Repomix” output) may include **sensitive references**. Treat it as **read‑only** and avoid re‑distributing raw dumps.

---

## Agents (Planned)

Work in progress (not yet included in this repository build):
- **Quality‑Assurance Agent** to validate metadata completeness per document (P/I/C/O/S rationale, trial design cues)
- **Acquisition Agent** to retry failed sources with site‑specific policies (e.g., pre‑prints)
- **Index Maintenance Agent** to deduplicate, re‑embed with upgraded models, and archive superseded versions
- **Query Assistant** for retrieval‑augmented answers with structured provenance

---

## Roadmap

- [ ] Configurable inclusion logic (thresholds using confidences, composite rules with P/I/C/O/S)
- [ ] Batch scheduling & backfill by month/quarter
- [ ] Multi‑tenant schema (namespaced knowledge bases)
- [ ] Inline OCR for scanned PDFs (e.g., Tesseract, PaddleOCR)
- [ ] Evaluation harness (precision/recall of inclusion vs human labels)
- [ ] Metrics → Prometheus/Grafana

---

## FAQ

**Q: Can I run without a GPU?**  
Yes. The classifier selects `cuda` only if available; otherwise runs on CPU.

**Q: How do I widen the freshness window?**  
Adjust `start_day` and `days_back` in `livedb_flow(...)`. Example: `start_day=7, days_back=7` (from 14 to 7 days ago).

**Q: How do I change the embedding model?**  
Edit `Config.py` (`EMBEDDING_MODEL`) and ensure your embedder supports it.

**Q: Where do PDFs go?**  
`Config.PDF_DIR` (defaults to `./pdfs`).

**Q: Can I skip Playwright?**  
Set `HEADLESS=True` (default) and rely on httpx first; if you remove Playwright you may lose some gated PDFs.

---

## License

Choose a license that fits your needs (e.g., MIT, Apache‑2.0). Place it as `LICENSE` in the repo root.

---

## Citations & Attribution

- **OpenAlex** — community‑maintained index of scholarly works  
- **NCBI E‑utils / PubMed / PMC OA** — programmatic biomedical literature access  
- **Agno** — abstraction layer for knowledge bases with pgvector  
- **pgvector** — high‑dimensional vector similarity for Postgres  
- **Playwright** and **playwright‑stealth** — browser automation for bot‑gated flows  
- **ReportLab** — PDF generation for BioC text fallback

If you publish using this pipeline, consider acknowledging these services.
