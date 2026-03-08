# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LiveDB is an async ETL pipeline for biomedical literature discovery, triage, and indexing. It fetches papers from OpenAlex and PubMed, classifies abstracts using a multi-head transformer (PICOS criteria), downloads PDFs, chunks content semantically, and ingests into a pgvector-backed Postgres knowledge base. A multi-agent research assistant layer (Agno framework) enables interactive querying.

## Commands

### Run
```bash
python main.py                    # Interactive menu: [1] ETL pipeline, [2] AI Agent server
```

ETL runs a Prefect flow; Agent mode starts FastAPI + AgentOS on `http://localhost:7777`.

### Lint & Format
```bash
ruff check --fix livedb agents dbs main.py Config.py
ruff format livedb agents dbs main.py Config.py
pre-commit run --all-files        # Runs ruff via pre-commit hooks
```

### Install
```bash
pip install -e .
python -m playwright install chromium
```

### Prefect Observability (optional)
```bash
prefect server start              # Flow visualization dashboard
```

## Architecture

### ETL Pipeline (`main.py` → `livedb/`)

```
OpenAlex API → PubMed ESearch/EFetch → PICOS Classifier → PDF Download → Semantic Chunking → pgvector Ingestion
```

1. **OpenAlexDownload.py**: Async OpenAlex client with cursor pagination; resilient PDF downloader (direct HTTP → Playwright+stealth fallback for JS-gated/Cloudflare sites)
2. **GetLatestPapers.py**: PubMed/PMC helpers — ESearch, EFetch (Medline XML), FTP tar.gz extraction, BioC XML→PDF conversion
3. **CheckAbsModel.py**: Multi-head DistilBERT classifier with 5 PICOS task heads (P_AB, I_AB, C_AB, O_AB, S_AB). Decision: `final_pred="no"` iff `S_AB_pred=="no"`
4. **utils.py**: ReportLab PDF generation from text

### Database Layer (`dbs/`)

- **IngestToDB.py**: Agno Knowledge setup + async ingestion; per-chunk metadata includes PICOS flags, publication info
- **utils.py**: `CustomChunking` extends `SemanticChunking` — stops at "References" section, 1000-token chunks

### Agent Layer (`agents/`)

4 specialist agents coordinated by `ResearchAssistantTeam`:
- **KnowledgeAgent**: RAG via pgvector hybrid search (BM25 + semantic)
- **SQLAgent**: Database schema inspection + safe SQL execution
- **ReasoningAgent**: Structured step-by-step analysis
- **GeneralAgent**: Synthesis fallback

All agents share OpenAI model, Postgres memory, and markdown output format.

### Configuration (`Config.py`)

Pydantic settings loaded from `.env`. Key groups: NCBI credentials, OpenAlex mailto, OpenAI keys, multiple Postgres connection URLs (vectors, contents, memory, SQL), PDF/model directories, HTTP headers, Playwright headless flag.

## Key Patterns

- **Async-first**: All I/O uses `asyncio.gather()` with bounded concurrency via semaphores (8 HTTP downloads, 3 Playwright instances)
- **Retry strategy**: Tenacity decorators with exponential backoff (3-5 attempts)
- **Graceful degradation**: Errors are logged (Loguru) but pipeline continues processing remaining records
- **PDF validation**: Magic byte checking + content-type sniffing before saving
- **Prefect caching**: Input hashing gives 1-hour cache on API queries
- **Logging**: Loguru rotating file at `logs/livedb.log` (10 MB rotation, 10-day retention)

## Tech Stack

- Python 3.12+ (3.13 in `.python-version`)
- Async HTTP: httpx (http2), aiohttp
- Browser automation: Playwright + playwright-stealth
- ML: PyTorch + Transformers (DistilBERT)
- Database: PostgreSQL + pgvector, psycopg3, SQLModel
- Agent framework: Agno (Knowledge, Teams, AgentOS)
- Orchestration: Prefect
- Linting: Ruff (via pre-commit)
