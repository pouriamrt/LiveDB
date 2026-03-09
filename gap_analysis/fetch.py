"""Phase 1: Fetch papers from OpenAlex + PubMed, deduplicate, PICOS filter."""

from __future__ import annotations

import asyncio
import json

import pandas as pd
from loguru import logger as log

from gap_analysis.models import PaperMetadata
from gap_analysis.prompts import QUERY_TRANSLATE_SYSTEM, QUERY_TRANSLATE_USER
from livedb.CheckAbsModel import check_abs_model_async
from livedb.GetLatestPapers import pubmed_efetch, pubmed_esearch
from livedb.OpenAlexDownload import fetch_openalex_latest


async def translate_query(user_input: str, model: str | None = None) -> list[str]:
    """Translate a natural language query into optimized API search queries."""
    from Config import config
    from gap_analysis import openai_client

    model = model or config.MODEL_NAME
    try:
        resp = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": QUERY_TRANSLATE_SYSTEM},
                {
                    "role": "user",
                    "content": QUERY_TRANSLATE_USER.format(user_input=user_input),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        raw = json.loads(resp.choices[0].message.content)
        queries = raw.get("queries", [user_input])
        log.info(f"Translated query into {len(queries)} search queries: {queries}")
        return queries
    except Exception as e:
        log.warning(f"Query translation failed: {e}. Using original input.")
        return [user_input]


def _parse_authors(raw: object) -> list[str]:
    """Convert author data from OpenAlex/PubMed into a list of strings.

    Both sources return authors as a comma-separated string (e.g. "Smith J, Doe A").
    """
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw.strip():
        return [a.strip() for a in raw.split(",") if a.strip()]
    return []


async def fetch_papers(
    queries: list[str],
    max_records: int = 100,
    start_day: int = 0,
    days_back: int = 180,
    run_picos: bool = True,
) -> list[PaperMetadata]:
    """Fetch from both sources for each query, deduplicate, optionally PICOS filter."""

    # Run all queries in parallel
    per_query_max = max(max_records // len(queries), 10)
    all_tasks = []
    for q in queries:
        all_tasks.append(
            fetch_openalex_latest(
                query=q,
                start_day=start_day,
                days_back=days_back,
                max_records=per_query_max,
                only_articles=True,
                only_oa=True,
            )
        )
        all_tasks.append(
            pubmed_esearch(
                q, days_back=days_back, start_day=start_day, retmax=per_query_max
            )
        )

    results = await asyncio.gather(*all_tasks, return_exceptions=True)

    # Separate OpenAlex DataFrames (even indices) and PubMed PMID lists (odd indices)
    oa_dfs = []
    all_pmids = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            log.warning(f"Fetch failed for query index {i}: {result}")
            continue
        if i % 2 == 0:
            oa_dfs.append(result)
        else:
            all_pmids.extend(result)

    # Merge OpenAlex DataFrames
    if oa_dfs:
        oa_df = pd.concat(oa_dfs, ignore_index=True).drop_duplicates(subset=["id"])
    else:
        oa_df = pd.DataFrame()

    # Deduplicate PMIDs
    pmids = list(dict.fromkeys(all_pmids))

    # Convert OpenAlex DataFrame to PaperMetadata
    # NaN values from pandas must be converted to None for Pydantic
    def _clean(val: object) -> str | None:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return str(val)

    papers: dict[str, PaperMetadata] = {}
    if not oa_df.empty:
        for _, row in oa_df.iterrows():
            abstract = _clean(row.get("abstract"))
            title = _clean(row.get("title"))
            if not abstract or not title:
                continue
            doi = _clean(row.get("doi"))
            pmid = _clean(row.get("pmid"))
            key = doi or pmid or row["id"]
            papers[key] = PaperMetadata(
                doi=doi,
                pmid=pmid,
                title=title,
                authors=_parse_authors(row.get("authors")),
                journal=_clean(row.get("journal")),
                publication_date=str(row.get("publication_date", "")),
                abstract=abstract,
            )

    # PubMed EFetch
    if pmids:
        pm_records = await pubmed_efetch(pmids)
        for rec in pm_records:
            if not rec.get("abstract") or not rec.get("title"):
                continue
            doi = rec.get("doi")
            key = doi or rec.get("pmid")
            if key and key not in papers:
                papers[key] = PaperMetadata(
                    doi=doi,
                    pmid=rec.get("pmid"),
                    title=rec["title"],
                    authors=_parse_authors(rec.get("authors")),
                    journal=rec.get("journal"),
                    publication_date=str(rec.get("pub_year", "")),
                    abstract=rec["abstract"],
                )

    log.info(f"Fetched {len(papers)} unique papers across {len(queries)} queries")

    result = list(papers.values())

    # Optional PICOS filter
    if run_picos and result:
        filtered = []
        for paper in result:
            try:
                text = paper.title + "\n" + paper.abstract
                preds, _ = await check_abs_model_async(text)
                # preds values are lists, e.g. {"S_AB_pred": ["no"]}
                s_ab = preds.get("S_AB_pred", [])
                if s_ab and s_ab[0] == "no":
                    continue
                filtered.append(paper)
            except Exception as e:
                log.warning(f"PICOS check failed for '{paper.title[:50]}': {e}")
                filtered.append(paper)  # include on failure
        log.info(f"PICOS filter: {len(result)} → {len(filtered)} papers")
        result = filtered

    return result
