# Research Gap Analysis — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a research gap analysis pipeline that fetches papers on a topic, clusters them thematically, identifies research gaps via LLM, and produces an interactive dashboard + PDF report.

**Architecture:** Hybrid embedding-clustering + LLM approach. Papers are fetched from OpenAlex/PubMed, abstracts are extracted into structured findings via batched LLM calls, embeddings are clustered with HDBSCAN, and targeted LLM analysis identifies gaps within and across clusters. Results are stored in Postgres and served via FastAPI.

**Tech Stack:** OpenAI (gpt-4.1-mini), HDBSCAN, UMAP, OpenAI embeddings (text-embedding-3-small), Prefect, FastAPI, Jinja2, Chart.js, ReportLab, Pydantic

**Design doc:** `docs/plans/2026-03-08-gap-analysis-design.md`

---

## Task 1: Add New Dependencies

**Files:**
- Modify: `pyproject.toml:7-34`

**Step 1: Add dependencies for clustering, dimensionality reduction, and templating**

Add these to the `dependencies` list in `pyproject.toml`:

```toml
"hdbscan>=0.8.40",
"umap-learn>=0.5.7",
"scikit-learn>=1.6.1",
"jinja2>=3.1.6",
"openai>=1.82.0",
```

**Step 2: Install**

Run: `pip install -e .`
Expected: All new packages install successfully

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add gap analysis dependencies (hdbscan, umap, jinja2, openai)"
```

---

## Task 2: Create Pydantic Data Models

**Files:**
- Create: `gap_analysis/__init__.py`
- Create: `gap_analysis/models.py`

**Step 1: Create package and models**

`gap_analysis/__init__.py` — empty file

`gap_analysis/models.py`:

```python
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    doi: str | None = None
    pmid: str | None = None
    title: str
    authors: list[str] = Field(default_factory=list)
    journal: str | None = None
    publication_date: str | None = None
    abstract: str


class PaperExtraction(BaseModel):
    """LLM-extracted structured findings from a single paper."""

    paper: PaperMetadata
    claims: list[str] = Field(default_factory=list)
    methodology: str = ""
    population: str | None = None
    intervention: str | None = None
    comparator: str | None = None
    outcomes: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    future_directions: list[str] = Field(default_factory=list)
    picos_flags: dict[str, str] = Field(default_factory=dict)


class ThemeCluster(BaseModel):
    """A group of papers clustered by thematic similarity."""

    cluster_id: int
    label: str
    description: str
    papers: list[PaperExtraction] = Field(default_factory=list)
    centroid_embedding: list[float] = Field(default_factory=list)


class ResearchGap(BaseModel):
    """A single identified research gap."""

    gap_type: str  # contradiction | under_explored | methodological | population | missing_comparison | future_direction
    title: str
    description: str
    severity: str  # high | medium | low
    evidence: list[str] = Field(default_factory=list)
    related_themes: list[str] = Field(default_factory=list)
    suggested_research: str = ""


class GapReport(BaseModel):
    """Full output of a gap analysis run."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    created_at: datetime = Field(default_factory=datetime.now)
    scope: int = 0
    date_range: str = ""
    themes: list[ThemeCluster] = Field(default_factory=list)
    gaps: list[ResearchGap] = Field(default_factory=list)
    executive_summary: str = ""
    methodology_overview: str = ""
    population_overview: str = ""
    conclusion: str = ""
```

**Step 2: Verify models parse correctly**

Run: `python -c "from gap_analysis.models import GapReport; print(GapReport(query='test').model_dump_json(indent=2))"`
Expected: Valid JSON output with defaults

**Step 3: Commit**

```bash
git add gap_analysis/
git commit -m "feat(gap): add Pydantic data models for gap analysis pipeline"
```

---

## Task 3: Create LLM Prompt Templates

**Files:**
- Create: `gap_analysis/prompts.py`

**Step 1: Write prompt templates**

```python
"""Centralized LLM prompt templates for gap analysis."""

EXTRACTION_SYSTEM = """\
You are a biomedical research analyst. Extract structured findings from each \
paper abstract provided. Return a JSON array where each element matches the \
schema exactly. Be precise and factual — only extract what is stated.

Output schema per paper:
{
  "title": "exact paper title",
  "claims": ["key finding 1", "key finding 2"],
  "methodology": "study type and methods used",
  "population": "who was studied or null",
  "intervention": "what was tested or null",
  "comparator": "what it was compared against or null",
  "outcomes": ["measured outcome 1", "measured outcome 2"],
  "limitations": ["limitation 1"],
  "future_directions": ["what authors say needs more work"]
}
"""

EXTRACTION_USER = """\
Extract structured findings from these {count} paper abstracts:

{papers}

Return a JSON array with {count} objects matching the schema.
"""

CLUSTER_LABEL_SYSTEM = """\
You are a research taxonomy expert. Given paper titles and key claims from a \
thematic cluster, provide a concise theme label (3-8 words) and a 1-2 sentence \
description of what this research cluster covers.

Return JSON: {{"label": "...", "description": "..."}}
"""

CLUSTER_LABEL_USER = """\
Papers in this cluster:

{papers}

Provide a theme label and description.
"""

WITHIN_CLUSTER_SYSTEM = """\
You are a systematic review expert. Analyze these papers within the same \
research theme. Identify research gaps in these categories:

- contradiction: findings that disagree with each other
- under_explored: subtopics with insufficient investigation
- methodological: limitations in study designs used
- population: demographics or patient groups not covered
- missing_comparison: interventions not compared head-to-head
- future_direction: what authors explicitly say needs more research

For each gap, provide:
{
  "gap_type": "one of the types above",
  "title": "short description (5-10 words)",
  "description": "detailed explanation (2-4 sentences)",
  "severity": "high | medium | low",
  "evidence": ["paper title 1", "paper title 2"],
  "suggested_research": "what study could fill this gap"
}

Return JSON: {{"gaps": [...]}}
"""

WITHIN_CLUSTER_USER = """\
Theme: {theme_label}
Description: {theme_description}

Papers and their extracted findings:

{papers}

Identify all research gaps within this theme.
"""

CROSS_CLUSTER_SYSTEM = """\
You are a research strategist. Given multiple research themes and their \
individual gaps, identify cross-cutting gaps and synthesize an overall analysis.

Identify:
1. Under-explored areas that fall between themes
2. Missing head-to-head comparisons across themes
3. Populations not covered across any theme
4. The most impactful research opportunities

Also provide:
- executive_summary: 2-3 paragraph overview of the research landscape and top priorities
- methodology_overview: distribution of study types and methodological strengths/weaknesses
- population_overview: what demographics are well/poorly covered
- conclusion: synthesis of top 3-5 research priorities with suggested study designs

Return JSON:
{
  "cross_cluster_gaps": [same gap schema as within-cluster],
  "executive_summary": "...",
  "methodology_overview": "...",
  "population_overview": "...",
  "conclusion": "..."
}
"""

CROSS_CLUSTER_USER = """\
Research topic: {query}
Total papers analyzed: {total_papers}
Date range: {date_range}

Themes and their gaps:

{themes}

Provide cross-cluster gap analysis and overall synthesis.
"""
```

**Step 2: Verify import**

Run: `python -c "from gap_analysis.prompts import EXTRACTION_SYSTEM; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add gap_analysis/prompts.py
git commit -m "feat(gap): add LLM prompt templates for extraction and analysis"
```

---

## Task 4: Implement Fetch Module (Phase 1)

**Files:**
- Create: `gap_analysis/fetch.py`
- Reference: `livedb/OpenAlexDownload.py` (fetch_openalex_latest)
- Reference: `livedb/GetLatestPapers.py` (pubmed_esearch, pubmed_efetch)
- Reference: `livedb/CheckAbsModel.py` (check_abs_model_async)

**Step 1: Write fetch module**

This wraps the existing ETL fetchers, deduplicates, and optionally runs PICOS filtering.

```python
"""Phase 1: Fetch papers from OpenAlex + PubMed, deduplicate, PICOS filter."""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger as log

from gap_analysis.models import PaperMetadata
from livedb.CheckAbsModel import check_abs_model_async
from livedb.GetLatestPapers import pubmed_efetch, pubmed_esearch
from livedb.OpenAlexDownload import fetch_openalex_latest


async def fetch_papers(
    query: str,
    max_records: int = 100,
    start_day: int = 180,
    days_back: int = 1,
    run_picos: bool = True,
) -> list[PaperMetadata]:
    """Fetch from both sources, deduplicate by DOI, optionally PICOS filter."""

    oa_task = fetch_openalex_latest(
        query=query,
        start_day=start_day,
        days_back=days_back,
        max_records=max_records,
        only_articles=True,
        only_oa=False,
    )
    pm_task = pubmed_esearch(
        query, days_back=days_back, start_day=start_day, retmax=max_records
    )

    oa_df, pmids = await asyncio.gather(oa_task, pm_task)

    # Convert OpenAlex DataFrame to PaperMetadata
    papers: dict[str, PaperMetadata] = {}
    for _, row in oa_df.iterrows():
        if not row.get("abstract") or not row.get("title"):
            continue
        doi = row.get("doi")
        key = doi or row.get("pmid") or row["id"]
        papers[key] = PaperMetadata(
            doi=doi,
            pmid=row.get("pmid"),
            title=row["title"],
            authors=row.get("authors", []) if isinstance(row.get("authors"), list) else [],
            journal=row.get("journal"),
            publication_date=str(row.get("publication_date", "")),
            abstract=row["abstract"],
        )

    # PubMed EFetch
    if pmids:
        pm_records = await pubmed_efetch(pmids)
        for rec in pm_records:
            if not rec.get("abstract") or not rec.get("title"):
                continue
            doi = rec.get("doi")
            key = doi or rec.get("pmid")
            if key not in papers:
                papers[key] = PaperMetadata(
                    doi=doi,
                    pmid=rec.get("pmid"),
                    title=rec["title"],
                    authors=rec.get("authors", []) if isinstance(rec.get("authors"), list) else [],
                    journal=rec.get("journal"),
                    publication_date=str(rec.get("pub_year", "")),
                    abstract=rec["abstract"],
                )

    log.info(f"Fetched {len(papers)} unique papers for query: {query}")

    result = list(papers.values())

    # Optional PICOS filter
    if run_picos and result:
        filtered = []
        for paper in result:
            try:
                text = paper.title + "\n" + paper.abstract
                preds, _ = await check_abs_model_async(text)
                if preds.get("S_AB_pred") != "no":
                    filtered.append(paper)
            except Exception as e:
                log.warning(f"PICOS check failed for '{paper.title[:50]}': {e}")
                filtered.append(paper)  # include on failure
        log.info(f"PICOS filter: {len(result)} → {len(filtered)} papers")
        result = filtered

    return result
```

**Step 2: Verify import**

Run: `python -c "from gap_analysis.fetch import fetch_papers; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add gap_analysis/fetch.py
git commit -m "feat(gap): add fetch module wrapping OpenAlex + PubMed with dedup"
```

---

## Task 5: Implement Extract Module (Phase 2)

**Files:**
- Create: `gap_analysis/extract.py`
- Reference: `gap_analysis/models.py`
- Reference: `gap_analysis/prompts.py`
- Reference: `Config.py` (OPENAI_API_KEY, MODEL_NAME)

**Step 1: Write extraction module**

Batched async LLM calls to extract structured findings from abstracts.

```python
"""Phase 2: LLM-based structured extraction from paper abstracts."""

from __future__ import annotations

import asyncio
import json

from loguru import logger as log
from openai import AsyncOpenAI

from Config import config
from gap_analysis.models import PaperExtraction, PaperMetadata
from gap_analysis.prompts import EXTRACTION_SYSTEM, EXTRACTION_USER

_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

LLM_SEM = asyncio.Semaphore(5)  # max concurrent LLM calls


async def _extract_batch(
    papers: list[PaperMetadata], model: str
) -> list[PaperExtraction]:
    """Extract structured findings from a batch of papers via LLM."""
    papers_text = "\n\n".join(
        f"--- Paper {i+1} ---\nTitle: {p.title}\nAbstract: {p.abstract}"
        for i, p in enumerate(papers)
    )

    async with LLM_SEM:
        try:
            resp = await _client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM},
                    {
                        "role": "user",
                        "content": EXTRACTION_USER.format(
                            count=len(papers), papers=papers_text
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            raw = json.loads(resp.choices[0].message.content)
            items = raw if isinstance(raw, list) else raw.get("papers", raw.get("results", [raw]))
        except Exception as e:
            log.warning(f"Batch extraction failed: {e}")
            return []

    extractions = []
    for paper, item in zip(papers, items):
        try:
            extractions.append(
                PaperExtraction(
                    paper=paper,
                    claims=item.get("claims", []),
                    methodology=item.get("methodology", ""),
                    population=item.get("population"),
                    intervention=item.get("intervention"),
                    comparator=item.get("comparator"),
                    outcomes=item.get("outcomes", []),
                    limitations=item.get("limitations", []),
                    future_directions=item.get("future_directions", []),
                )
            )
        except Exception as e:
            log.warning(f"Failed to parse extraction for '{paper.title[:50]}': {e}")
    return extractions


async def extract_papers(
    papers: list[PaperMetadata],
    batch_size: int = 5,
    model: str | None = None,
) -> list[PaperExtraction]:
    """Extract structured findings from all papers in batches."""
    model = model or config.MODEL_NAME
    batches = [papers[i : i + batch_size] for i in range(0, len(papers), batch_size)]
    log.info(f"Extracting findings from {len(papers)} papers in {len(batches)} batches")

    tasks = [_extract_batch(batch, model) for batch in batches]
    results = await asyncio.gather(*tasks)

    extractions = [ext for batch_result in results for ext in batch_result]
    log.info(f"Extracted findings from {len(extractions)}/{len(papers)} papers")
    return extractions
```

**Step 2: Verify import**

Run: `python -c "from gap_analysis.extract import extract_papers; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add gap_analysis/extract.py
git commit -m "feat(gap): add LLM extraction module with batched async calls"
```

---

## Task 6: Implement Cluster Module (Phase 3)

**Files:**
- Create: `gap_analysis/cluster.py`
- Reference: `gap_analysis/models.py`
- Reference: `gap_analysis/prompts.py`

**Step 1: Write clustering module**

Embeds extractions, clusters with HDBSCAN, labels themes via LLM.

```python
"""Phase 3: Embedding-based clustering + LLM theme labeling."""

from __future__ import annotations

import asyncio
import json

import numpy as np
from hdbscan import HDBSCAN
from loguru import logger as log
from openai import AsyncOpenAI

from Config import config
from gap_analysis.models import PaperExtraction, ThemeCluster
from gap_analysis.prompts import CLUSTER_LABEL_SYSTEM, CLUSTER_LABEL_USER

_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)


async def _embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts using OpenAI embeddings API."""
    resp = await _client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=texts,
    )
    return np.array([d.embedding for d in resp.data])


async def _label_cluster(
    papers: list[PaperExtraction], model: str
) -> dict[str, str]:
    """Ask LLM to name a cluster theme."""
    papers_text = "\n".join(
        f"- {p.paper.title}: {', '.join(p.claims[:3])}" for p in papers[:15]
    )
    try:
        resp = await _client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CLUSTER_LABEL_SYSTEM},
                {"role": "user", "content": CLUSTER_LABEL_USER.format(papers=papers_text)},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        log.warning(f"Cluster labeling failed: {e}")
        return {"label": "Unlabeled cluster", "description": "Could not generate label"}


async def cluster_papers(
    extractions: list[PaperExtraction],
    min_cluster_size: int = 3,
    model: str | None = None,
) -> list[ThemeCluster]:
    """Cluster papers by embedding similarity and label each cluster."""
    model = model or config.MODEL_NAME

    if len(extractions) < min_cluster_size:
        # Too few papers — put everything in one cluster
        label_info = await _label_cluster(extractions, model)
        return [
            ThemeCluster(
                cluster_id=0,
                label=label_info.get("label", "All papers"),
                description=label_info.get("description", ""),
                papers=extractions,
            )
        ]

    # Embed: title + top claims for each paper
    texts = [
        f"{e.paper.title}. {' '.join(e.claims[:3])}" for e in extractions
    ]
    embeddings = await _embed_texts(texts)

    # Cluster
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=2,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embeddings)
    unique_labels = set(labels)

    log.info(f"HDBSCAN found {len(unique_labels - {-1})} clusters, {(labels == -1).sum()} noise points")

    # Group papers by cluster
    clusters_dict: dict[int, list[PaperExtraction]] = {}
    noise_papers: list[PaperExtraction] = []
    for ext, label in zip(extractions, labels):
        if label == -1:
            noise_papers.append(ext)
        else:
            clusters_dict.setdefault(label, []).append(ext)

    # Assign noise papers to nearest cluster
    if noise_papers and clusters_dict:
        centroids = {}
        for cid, papers in clusters_dict.items():
            indices = [i for i, l in enumerate(labels) if l == cid]
            centroids[cid] = embeddings[indices].mean(axis=0)

        for paper in noise_papers:
            idx = extractions.index(paper)
            emb = embeddings[idx]
            nearest = min(centroids, key=lambda c: np.linalg.norm(emb - centroids[c]))
            clusters_dict[nearest].append(paper)

    # Label each cluster via LLM
    label_tasks = [_label_cluster(papers, model) for papers in clusters_dict.values()]
    label_results = await asyncio.gather(*label_tasks)

    theme_clusters = []
    for (cid, papers), label_info in zip(clusters_dict.items(), label_results):
        indices = [i for i, l in enumerate(labels) if l == cid]
        centroid = embeddings[indices].mean(axis=0).tolist() if indices else []
        theme_clusters.append(
            ThemeCluster(
                cluster_id=cid,
                label=label_info.get("label", f"Cluster {cid}"),
                description=label_info.get("description", ""),
                papers=papers,
                centroid_embedding=centroid,
            )
        )

    log.info(f"Created {len(theme_clusters)} theme clusters")
    return theme_clusters
```

**Step 2: Verify import**

Run: `python -c "from gap_analysis.cluster import cluster_papers; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add gap_analysis/cluster.py
git commit -m "feat(gap): add HDBSCAN clustering with LLM theme labeling"
```

---

## Task 7: Implement Analyze Module (Phase 4)

**Files:**
- Create: `gap_analysis/analyze.py`
- Reference: `gap_analysis/models.py`
- Reference: `gap_analysis/prompts.py`

**Step 1: Write analysis module**

Two-pass LLM gap identification: within-cluster then cross-cluster.

```python
"""Phase 4: LLM-based gap identification (within-cluster + cross-cluster)."""

from __future__ import annotations

import asyncio
import json

from loguru import logger as log
from openai import AsyncOpenAI

from Config import config
from gap_analysis.models import GapReport, ResearchGap, ThemeCluster
from gap_analysis.prompts import (
    CROSS_CLUSTER_SYSTEM,
    CROSS_CLUSTER_USER,
    WITHIN_CLUSTER_SYSTEM,
    WITHIN_CLUSTER_USER,
)

_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)


def _format_papers_for_prompt(cluster: ThemeCluster) -> str:
    parts = []
    for p in cluster.papers:
        parts.append(
            f"Title: {p.paper.title}\n"
            f"Claims: {'; '.join(p.claims)}\n"
            f"Methodology: {p.methodology}\n"
            f"Population: {p.population or 'N/A'}\n"
            f"Outcomes: {'; '.join(p.outcomes)}\n"
            f"Limitations: {'; '.join(p.limitations)}\n"
            f"Future directions: {'; '.join(p.future_directions)}"
        )
    return "\n\n".join(parts)


async def _analyze_within_cluster(
    cluster: ThemeCluster, model: str
) -> list[ResearchGap]:
    """Identify gaps within a single theme cluster."""
    papers_text = _format_papers_for_prompt(cluster)
    try:
        resp = await _client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": WITHIN_CLUSTER_SYSTEM},
                {
                    "role": "user",
                    "content": WITHIN_CLUSTER_USER.format(
                        theme_label=cluster.label,
                        theme_description=cluster.description,
                        papers=papers_text,
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        raw = json.loads(resp.choices[0].message.content)
        gaps_data = raw.get("gaps", [])
    except Exception as e:
        log.warning(f"Within-cluster analysis failed for '{cluster.label}': {e}")
        return []

    gaps = []
    for g in gaps_data:
        try:
            gaps.append(
                ResearchGap(
                    gap_type=g.get("gap_type", "under_explored"),
                    title=g.get("title", ""),
                    description=g.get("description", ""),
                    severity=g.get("severity", "medium"),
                    evidence=g.get("evidence", []),
                    related_themes=[cluster.label],
                    suggested_research=g.get("suggested_research", ""),
                )
            )
        except Exception as e:
            log.warning(f"Failed to parse gap: {e}")
    return gaps


async def _analyze_cross_cluster(
    clusters: list[ThemeCluster],
    within_gaps: dict[int, list[ResearchGap]],
    query: str,
    total_papers: int,
    date_range: str,
    model: str,
) -> dict:
    """Identify cross-cutting gaps and generate synthesis."""
    themes_text = ""
    for cluster in clusters:
        cluster_gaps = within_gaps.get(cluster.cluster_id, [])
        gaps_text = "\n".join(
            f"  - [{g.severity}] {g.gap_type}: {g.title}" for g in cluster_gaps
        )
        themes_text += (
            f"\n### {cluster.label}\n"
            f"Description: {cluster.description}\n"
            f"Papers: {len(cluster.papers)}\n"
            f"Gaps found:\n{gaps_text}\n"
        )

    try:
        resp = await _client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CROSS_CLUSTER_SYSTEM},
                {
                    "role": "user",
                    "content": CROSS_CLUSTER_USER.format(
                        query=query,
                        total_papers=total_papers,
                        date_range=date_range,
                        themes=themes_text,
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        log.error(f"Cross-cluster analysis failed: {e}")
        return {}


async def analyze_gaps(
    clusters: list[ThemeCluster],
    query: str,
    date_range: str,
    model: str | None = None,
) -> GapReport:
    """Run full gap analysis: within-cluster + cross-cluster."""
    model = model or config.MODEL_NAME
    total_papers = sum(len(c.papers) for c in clusters)

    # Pass 1: within-cluster analysis (parallel)
    log.info(f"Running within-cluster gap analysis on {len(clusters)} clusters")
    within_tasks = [_analyze_within_cluster(c, model) for c in clusters]
    within_results = await asyncio.gather(*within_tasks)

    within_gaps: dict[int, list[ResearchGap]] = {}
    all_gaps: list[ResearchGap] = []
    for cluster, gaps in zip(clusters, within_results):
        within_gaps[cluster.cluster_id] = gaps
        all_gaps.extend(gaps)

    log.info(f"Found {len(all_gaps)} within-cluster gaps")

    # Pass 2: cross-cluster synthesis
    log.info("Running cross-cluster synthesis")
    cross_result = await _analyze_cross_cluster(
        clusters, within_gaps, query, total_papers, date_range, model
    )

    # Parse cross-cluster gaps
    for g in cross_result.get("cross_cluster_gaps", []):
        try:
            all_gaps.append(
                ResearchGap(
                    gap_type=g.get("gap_type", "under_explored"),
                    title=g.get("title", ""),
                    description=g.get("description", ""),
                    severity=g.get("severity", "medium"),
                    evidence=g.get("evidence", []),
                    related_themes=g.get("related_themes", []),
                    suggested_research=g.get("suggested_research", ""),
                )
            )
        except Exception as e:
            log.warning(f"Failed to parse cross-cluster gap: {e}")

    # Sort gaps by severity
    severity_order = {"high": 0, "medium": 1, "low": 2}
    all_gaps.sort(key=lambda g: severity_order.get(g.severity, 1))

    return GapReport(
        query=query,
        scope=total_papers,
        date_range=date_range,
        themes=clusters,
        gaps=all_gaps,
        executive_summary=cross_result.get("executive_summary", ""),
        methodology_overview=cross_result.get("methodology_overview", ""),
        population_overview=cross_result.get("population_overview", ""),
        conclusion=cross_result.get("conclusion", ""),
    )
```

**Step 2: Verify import**

Run: `python -c "from gap_analysis.analyze import analyze_gaps; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add gap_analysis/analyze.py
git commit -m "feat(gap): add two-pass LLM gap analysis (within + cross cluster)"
```

---

## Task 8: Implement Report Module (Phase 5)

**Files:**
- Create: `gap_analysis/report.py`
- Create: `gap_analysis/templates/dashboard.html`
- Reference: `livedb/utils.py` (ReportLab pattern)

**Step 1: Write report generation module**

```python
"""Phase 5: Report generation — PDF + HTML dashboard."""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger as log
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from gap_analysis.models import GapReport

TEMPLATE_DIR = Path(__file__).parent / "templates"

GAP_TYPE_COLORS = {
    "contradiction": "#e74c3c",
    "under_explored": "#f39c12",
    "methodological": "#3498db",
    "population": "#9b59b6",
    "missing_comparison": "#1abc9c",
    "future_direction": "#2ecc71",
}

SEVERITY_COLORS = {
    "high": "#e74c3c",
    "medium": "#f39c12",
    "low": "#27ae60",
}


def generate_pdf(report: GapReport, output_path: str) -> str:
    """Generate a PDF report from a GapReport."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"], fontSize=18, spaceAfter=20
    )
    heading_style = ParagraphStyle(
        "CustomHeading", parent=styles["Heading1"], fontSize=14, spaceAfter=10
    )
    body_style = ParagraphStyle(
        "CustomBody", parent=styles["Normal"], fontSize=10, spaceAfter=6, leading=14
    )

    elements = []

    # Title
    elements.append(Paragraph("Research Gap Analysis Report", title_style))
    elements.append(
        Paragraph(f"Topic: {report.query}", body_style)
    )
    elements.append(
        Paragraph(
            f"Date: {report.created_at.strftime('%Y-%m-%d')} | "
            f"Papers analyzed: {report.scope} | "
            f"Period: {report.date_range}",
            body_style,
        )
    )
    elements.append(Spacer(1, 0.3 * inch))

    # Executive Summary
    elements.append(Paragraph("1. Executive Summary", heading_style))
    elements.append(Paragraph(report.executive_summary or "N/A", body_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Research Landscape
    elements.append(Paragraph("2. Research Landscape", heading_style))
    elements.append(Paragraph("<b>Methodology Overview</b>", body_style))
    elements.append(Paragraph(report.methodology_overview or "N/A", body_style))
    elements.append(Paragraph("<b>Population Coverage</b>", body_style))
    elements.append(Paragraph(report.population_overview or "N/A", body_style))

    # Theme table
    if report.themes:
        theme_data = [["Theme", "Papers", "Description"]]
        for t in report.themes:
            theme_data.append([t.label, str(len(t.papers)), t.description[:80]])
        theme_table = Table(theme_data, colWidths=[2 * inch, 0.8 * inch, 4 * inch])
        theme_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(theme_table)
    elements.append(Spacer(1, 0.2 * inch))

    # Gaps
    elements.append(Paragraph("3. Identified Research Gaps", heading_style))
    for gap in report.gaps:
        color = SEVERITY_COLORS.get(gap.severity, "#999")
        elements.append(
            Paragraph(
                f'<font color="{color}">[{gap.severity.upper()}]</font> '
                f"<b>{gap.gap_type}</b>: {gap.title}",
                body_style,
            )
        )
        elements.append(Paragraph(gap.description, body_style))
        if gap.evidence:
            evidence_text = ", ".join(gap.evidence[:5])
            elements.append(
                Paragraph(f"<i>Evidence: {evidence_text}</i>", body_style)
            )
        if gap.suggested_research:
            elements.append(
                Paragraph(
                    f"<i>Suggested research: {gap.suggested_research}</i>", body_style
                )
            )
        elements.append(Spacer(1, 0.1 * inch))

    # Conclusion
    elements.append(Paragraph("4. Conclusion & Priorities", heading_style))
    elements.append(Paragraph(report.conclusion or "N/A", body_style))

    # Appendix — paper list
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph("Appendix: Analyzed Papers", heading_style))
    for theme in report.themes:
        for p in theme.papers:
            elements.append(
                Paragraph(
                    f"<b>{p.paper.title}</b> — {p.paper.journal or 'N/A'} "
                    f"({p.paper.publication_date or 'N/A'})",
                    body_style,
                )
            )

    doc.build(elements)
    log.info(f"PDF report saved to {output_path}")
    return output_path


def generate_dashboard_html(report: GapReport) -> str:
    """Generate an HTML dashboard string from a GapReport."""
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template("dashboard.html")
    return template.render(
        report=report,
        gap_type_colors=GAP_TYPE_COLORS,
        severity_colors=SEVERITY_COLORS,
    )
```

**Step 2: Create the HTML dashboard template**

Create `gap_analysis/templates/dashboard.html` — this is a self-contained Jinja2 template with Chart.js for visualizations. *(Full template content will be written during implementation — it includes: query summary bar, gap cards with filters, theme accordion sections, paper table, scatter plot placeholder, PDF export button.)*

**Step 3: Verify import**

Run: `python -c "from gap_analysis.report import generate_pdf; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add gap_analysis/report.py gap_analysis/templates/
git commit -m "feat(gap): add PDF + HTML dashboard report generation"
```

---

## Task 9: Implement Pipeline Orchestrator

**Files:**
- Create: `gap_analysis/pipeline.py`
- Reference: `main.py` (Prefect flow pattern)

**Step 1: Write the Prefect flow**

```python
"""Gap analysis pipeline — Prefect flow orchestrating phases 1-5."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from loguru import logger as log
from prefect import flow

from gap_analysis.analyze import analyze_gaps
from gap_analysis.cluster import cluster_papers
from gap_analysis.extract import extract_papers
from gap_analysis.fetch import fetch_papers
from gap_analysis.models import GapReport
from gap_analysis.report import generate_dashboard_html, generate_pdf


@flow(name="Gap Analysis")
async def gap_analysis_flow(
    query: str,
    max_records: int = 100,
    start_day: int = 180,
    days_back: int = 1,
    run_picos: bool = True,
    model: str | None = None,
    output_dir: str = "reports",
) -> GapReport:
    """Run the full gap analysis pipeline."""
    log.info(f"Starting gap analysis for: {query} (max {max_records} papers)")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=start_day)
    date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

    # Phase 1: Fetch
    log.info("Phase 1: Fetching papers...")
    papers = await fetch_papers(
        query=query,
        max_records=max_records,
        start_day=start_day,
        days_back=days_back,
        run_picos=run_picos,
    )
    if not papers:
        log.warning("No papers found — aborting")
        return GapReport(query=query, date_range=date_range, executive_summary="No papers found for this query.")

    # Phase 2: Extract
    log.info(f"Phase 2: Extracting findings from {len(papers)} papers...")
    extractions = await extract_papers(papers, model=model)
    if not extractions:
        log.warning("No extractions produced — aborting")
        return GapReport(query=query, scope=len(papers), date_range=date_range, executive_summary="Extraction failed.")

    # Phase 3: Cluster
    log.info(f"Phase 3: Clustering {len(extractions)} papers...")
    clusters = await cluster_papers(extractions, model=model)

    # Phase 4: Analyze
    log.info(f"Phase 4: Analyzing gaps across {len(clusters)} clusters...")
    report = await analyze_gaps(clusters, query=query, date_range=date_range, model=model)

    # Phase 5: Generate reports
    log.info("Phase 5: Generating reports...")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = query.replace(" ", "_")[:50]

    pdf_path = os.path.join(output_dir, f"gap_report_{safe_query}_{timestamp}.pdf")
    generate_pdf(report, pdf_path)

    html_path = os.path.join(output_dir, f"gap_report_{safe_query}_{timestamp}.html")
    html_content = generate_dashboard_html(report)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Save JSON for agent queries
    json_path = os.path.join(output_dir, f"gap_report_{safe_query}_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    log.info(f"Gap analysis complete. Reports saved to {output_dir}/")
    log.info(f"  PDF: {pdf_path}")
    log.info(f"  Dashboard: {html_path}")
    log.info(f"  JSON: {json_path}")

    return report
```

**Step 2: Verify import**

Run: `python -c "from gap_analysis.pipeline import gap_analysis_flow; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add gap_analysis/pipeline.py
git commit -m "feat(gap): add Prefect pipeline orchestrating all 5 phases"
```

---

## Task 10: Add Config Settings + DB Storage

**Files:**
- Modify: `Config.py:46` (add after PGVECTOR_MEMORY_TABLE)
- Modify: `dbs/IngestToDB.py` (add gap report storage function)

**Step 1: Add config settings**

In `Config.py`, after line 46 (`PGVECTOR_MEMORY_TABLE`), add:

```python
    # Gap Analysis
    GAP_REPORTS_TABLE: str = os.getenv("GAP_REPORTS_TABLE", "gap_reports")
    GAP_REPORTS_DB_URL: str = os.getenv("GAP_REPORTS_DB_URL", os.getenv("SQL_DATABASE_URL"))
    GAP_LLM_BATCH_SIZE: int = 5
    GAP_LLM_CONCURRENCY: int = 5
    GAP_DEFAULT_SCOPE: int = 100
    GAP_DEFAULT_DAYS: int = 180
```

**Step 2: Add gap report storage to dbs/IngestToDB.py**

Append a new function at the end of the file:

```python
async def store_gap_report(report) -> None:
    """Store a GapReport in the gap_reports table."""
    import psycopg

    async with await psycopg.AsyncConnection.connect(config.GAP_REPORTS_DB_URL) as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gap_reports (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                scope INTEGER,
                report_json JSONB,
                executive_summary TEXT,
                status TEXT DEFAULT 'completed'
            )
            """,
        )
        await conn.execute(
            """
            INSERT INTO gap_reports (id, query, created_at, scope, report_json, executive_summary, status)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
            """,
            (
                report.id,
                report.query,
                report.created_at,
                report.scope,
                report.model_dump_json(),
                report.executive_summary,
                "completed",
            ),
        )
        await conn.commit()
    log.info(f"Gap report {report.id} stored in database")
```

**Step 3: Commit**

```bash
git add Config.py dbs/IngestToDB.py
git commit -m "feat(gap): add config settings and database storage for gap reports"
```

---

## Task 11: Add main.py Option [3]

**Files:**
- Modify: `main.py:217-229`

**Step 1: Update the menu and add gap analysis handler**

Replace the menu section (lines 217-229) with:

```python
    console.print(Panel("[1] ETL\n[2] AI Agent\n[3] Gap Analysis", title="Run Mode"))

    c = Prompt.ask("Choice", choices=["1", "2", "3"], show_choices=False)

    if c == "1":
        asyncio.run(livedb_flow(query="dementia", max_records=10))
    elif c == "2":
        from agents.RunTeam import run_team

        session_state = {}
        agent_os, app = run_team(session_state)
        agent_os.serve(app=app, port=7777)
        console.print(Panel("Agent server stopped", title="Run Mode"))
    elif c == "3":
        from rich.prompt import IntPrompt

        from gap_analysis.pipeline import gap_analysis_flow

        query = Prompt.ask("Research topic")
        max_papers = IntPrompt.ask("Max papers to analyze", default=100)
        days = IntPrompt.ask("Days to look back", default=180)

        console.print(f"\nAnalyzing gaps in: [bold]{query}[/bold]")
        console.print(f"Scope: {max_papers} papers, last {days} days\n")

        report = asyncio.run(
            gap_analysis_flow(
                query=query,
                max_records=max_papers,
                start_day=days,
            )
        )
        console.print(Panel(
            f"Gaps found: {len(report.gaps)}\n"
            f"Themes: {len(report.themes)}\n"
            f"Papers analyzed: {report.scope}\n\n"
            f"Reports saved to reports/ directory",
            title="Gap Analysis Complete",
        ))
```

**Step 2: Verify menu displays**

Run: `python -c "from rich.panel import Panel; from rich.console import Console; Console().print(Panel('[1] ETL\n[2] AI Agent\n[3] Gap Analysis', title='Run Mode'))"`
Expected: Panel with 3 options renders

**Step 3: Commit**

```bash
git add main.py
git commit -m "feat(gap): add option [3] gap analysis to main menu"
```

---

## Task 12: Add GapAnalysisAgent to Team

**Files:**
- Modify: `agents/Agents.py` (add new agent after existing agents)
- Modify: `agents/Teams.py` (add to team members + instructions)

**Step 1: Add GapAnalysisAgent in Agents.py**

After the existing agent definitions, add:

```python
gap_analysis_agent = Agent(
    name="GapAnalysisAgent",
    model=model,
    db=memory_db,
    description=(
        "You are a research gap analysis specialist. You query past gap analysis "
        "reports stored in the gap_reports database table and help users understand "
        "research gaps, themes, and priorities."
    ),
    instructions=[
        "Query the gap_reports table for past analyses using SQL",
        "Answer follow-up questions about specific gaps, themes, or papers",
        "Compare gaps across different topics or time periods",
        "For new full-scope analysis, suggest the user run option [3] in main.py",
        "Present findings with citations to specific papers",
    ],
    tools=[PostgresTools(db_url=config.SQL_DATABASE_URL, db_engine=db_engine)],
    markdown=True,
    exponential_backoff=True,
    read_chat_history=True,
    add_history_to_context=True,
)
```

**Step 2: Update Teams.py**

Import the new agent and add to team members list and instructions.

**Step 3: Commit**

```bash
git add agents/Agents.py agents/Teams.py
git commit -m "feat(gap): add GapAnalysisAgent to research assistant team"
```

---

## Task 13: Create Dashboard HTML Template

**Files:**
- Create: `gap_analysis/templates/dashboard.html`

**Step 1: Write the Jinja2 dashboard template**

Full self-contained HTML with:
- Chart.js CDN for visualizations
- Query summary header bar
- Gap severity distribution chart (doughnut)
- Gap type distribution chart (bar)
- Filterable gap cards with type badges and severity colors
- Expandable theme accordion sections
- Searchable paper table
- PDF export link
- Responsive CSS (no framework needed)

**Step 2: Verify template renders**

Run: `python -c "from jinja2 import Environment, FileSystemLoader; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add gap_analysis/templates/dashboard.html
git commit -m "feat(gap): add interactive HTML dashboard template"
```

---

## Task 14: Add Dashboard FastAPI Route

**Files:**
- Modify: `agents/RunTeam.py` (add `/gap-analysis` route)

**Step 1: Add routes for serving dashboard and triggering analysis**

Add FastAPI routes to serve past reports and the dashboard HTML:
- `GET /gap-analysis` — list past reports
- `GET /gap-analysis/{report_id}` — serve dashboard HTML for a specific report
- `POST /gap-analysis` — trigger a new analysis (async, returns report ID)

**Step 2: Commit**

```bash
git add agents/RunTeam.py
git commit -m "feat(gap): add FastAPI routes for gap analysis dashboard"
```

---

## Task 15: Integration Testing & Polish

**Step 1: Run end-to-end test with small scope**

```bash
python main.py
# Select [3], enter a topic, set max papers to 10, days to 30
```

Expected: Pipeline completes, PDF + HTML + JSON saved to `reports/`

**Step 2: Verify agent integration**

```bash
python main.py
# Select [2], ask: "What gaps have been identified in recent analyses?"
```

Expected: GapAnalysisAgent queries gap_reports table and responds

**Step 3: Verify dashboard**

Open the generated HTML file in a browser. Verify:
- Summary bar shows correct counts
- Gap cards are filterable
- Theme sections expand
- Charts render

**Step 4: Run linter**

```bash
ruff check --fix gap_analysis/ agents/ dbs/ main.py Config.py
ruff format gap_analysis/ agents/ dbs/ main.py Config.py
```

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat(gap): complete gap analysis pipeline with dashboard + agent integration"
```

---

Plan complete and saved to `docs/plans/2026-03-08-gap-analysis-implementation.md`. Two execution options:

**1. Subagent-Driven (this session)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** — Open new session with executing-plans, batch execution with checkpoints

Which approach?