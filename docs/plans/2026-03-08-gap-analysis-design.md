# Research Gap Analysis Feature — Design Document

**Date:** 2026-03-08
**Status:** Approved

## Overview

Add a research gap analysis capability to LiveDB that fetches recent papers on a user-specified topic, identifies research gaps using a hybrid embedding-clustering + LLM approach, and produces a detailed report as an interactive dashboard with PDF export.

## Requirements

- **Input:** Natural language queries, keywords, and/or structured filters (flexible)
- **Gap types:** Contradictions, under-explored areas, methodological gaps, population gaps, missing comparisons, future directions
- **Output:** Interactive web dashboard + downloadable PDF report
- **Fetching:** On-demand per query (not scheduled)
- **Scope:** User-configurable (10-500 papers)
- **LLM:** OpenAI (gpt-4.1-mini, targeting gpt-5.4 when available)
- **Integration:** Option [3] in main.py menu + GapAnalysisAgent in existing ResearchAssistantTeam

## Architecture: Hybrid (Embedding Clustering + Targeted LLM)

Embeddings organize papers into themes; LLM performs targeted reasoning on clusters to identify gaps. This balances cost, speed, and quality.

### Pipeline Flow

```
User query + scope params
        │
        ▼
Phase 1: FETCH
  OpenAlex + PubMed → deduplicate by DOI/PMID → PICOS filter
        │
        ▼
Phase 2: EXTRACT
  Batched async LLM calls (5 abstracts/batch)
  → PaperExtraction JSON per paper
        │
        ▼
Phase 3: CLUSTER
  Embed extractions → HDBSCAN clustering → LLM labels themes
        │
        ▼
Phase 4: ANALYZE
  Pass 1: per-cluster gap identification
  Pass 2: cross-cluster synthesis (contradictions, missing areas, priorities)
        │
        ▼
Phase 5: REPORT
  Generate: interactive HTML dashboard + PDF via ReportLab
  Store: GapReport as JSONB in gap_reports table
```

Orchestrated as a Prefect flow with caching and bounded concurrency.

## File Structure

```
gap_analysis/
├── __init__.py
├── pipeline.py       # Prefect flow orchestrating phases 1-5
├── fetch.py          # Wraps existing OpenAlex + PubMed fetchers + deduplication
├── extract.py        # LLM structured extraction (per-paper → JSON)
├── cluster.py        # Embedding + HDBSCAN clustering + LLM theme labeling
├── analyze.py        # Gap identification (within-cluster + cross-cluster)
├── report.py         # Report generation (markdown, PDF, HTML dashboard)
├── models.py         # Pydantic models for extractions, clusters, gaps, reports
└── prompts.py        # All LLM prompt templates
```

Also modified:
- `main.py` — add option [3] Gap Analysis
- `agents/Agents.py` — add GapAnalysisAgent
- `agents/Teams.py` — register agent in team
- `dbs/IngestToDB.py` — add gap_reports table ingestion
- `Config.py` — add gap analysis settings

## Data Models

### PaperMetadata
- doi, pmid, title, authors, journal, publication_date, abstract

### PaperExtraction (LLM-extracted from each paper)
- paper: PaperMetadata
- claims: list[str]
- methodology: str
- population, intervention, comparator: str | None
- outcomes, limitations, future_directions: list[str]
- picos_flags: dict[str, str]

### ThemeCluster
- cluster_id: int
- label, description: str (LLM-generated)
- papers: list[PaperExtraction]
- centroid_embedding: list[float]

### ResearchGap
- gap_type: "contradiction" | "under_explored" | "methodological" | "population" | "missing_comparison" | "future_direction"
- title, description: str
- severity: "high" | "medium" | "low"
- evidence: list[str] (paper DOIs/titles)
- related_themes: list[str]
- suggested_research: str

### GapReport
- id: UUID
- query, created_at, scope, date_range
- themes: list[ThemeCluster]
- gaps: list[ResearchGap]
- executive_summary, methodology_overview, population_overview, conclusion: str

## LLM Interaction Strategy

### Phase 2: Extraction
- Batch 5 abstracts per LLM call with structured JSON output
- Fallback: retry individual papers on batch failure
- ~40 calls for 200 papers

### Phase 3: Cluster Labeling
- One call per cluster to generate theme label + description
- ~8 calls for typical clustering

### Phase 4: Gap Analysis
- Pass 1 (within-cluster): one call per cluster identifying internal gaps
- Pass 2 (cross-cluster): single call receiving all cluster summaries, identifies cross-cutting gaps and generates executive summary, methodology overview, population overview, conclusion
- ~9 calls total

### Cost Estimate (200 papers, gpt-4.1-mini)
- ~57 LLM calls, ~515K tokens, ~$0.20

## Report Output

### PDF Report Sections
1. Executive Summary (landscape overview + top 3-5 priority gaps)
2. Research Landscape (theme map, methodology distribution, population coverage)
3. Identified Gaps by severity (type badge, description, evidence, suggested research)
4. Theme Deep-Dives (per cluster: findings, within-theme gaps)
5. Conclusion & Priorities (ranked opportunities, suggested study designs)
6. Appendix: Full paper bibliography with PICOS flags

### Interactive Dashboard (FastAPI + Jinja2)
- Query summary bar
- Cluster scatter plot (2D UMAP projection, clickable)
- Filterable/sortable gap cards (by type, severity)
- Expandable theme accordions with papers and findings
- Searchable paper table with PICOS flags
- PDF export button

**Tech:** HTML + vanilla JS + Chart.js. No heavy frameworks.

### Database Storage
`gap_reports` table:
| Column | Type |
|--------|------|
| id | UUID PK |
| query | TEXT |
| created_at | TIMESTAMP |
| scope | INT |
| report_json | JSONB |
| executive_summary | TEXT |
| status | TEXT ("running"/"completed"/"failed") |

## GapAnalysisAgent

Joins the existing ResearchAssistantTeam. Capabilities:
- Query past gap reports from gap_reports table
- Answer follow-up questions about specific gaps, themes, papers
- Compare gaps across different topics/time periods
- Quick mini-analysis on <20 papers conversationally

Does NOT run the full pipeline — that stays in option [3]. Team delegation routes gap-related questions to this agent.

## Design Principles (Consistent with Existing Codebase)
- Async-first: all I/O uses asyncio
- Graceful degradation: extraction failures don't block pipeline
- Bounded concurrency: semaphores on LLM calls
- Prefect orchestration: caching, observability, task tracking
- Loguru logging: consistent with existing log setup
- Pydantic models: typed throughout
- Retry with exponential backoff via tenacity
