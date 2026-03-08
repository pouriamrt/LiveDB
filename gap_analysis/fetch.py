"""Phase 1: Fetch papers from OpenAlex + PubMed, deduplicate, PICOS filter."""

from __future__ import annotations

import asyncio

from loguru import logger as log

from gap_analysis.models import PaperMetadata
from livedb.CheckAbsModel import check_abs_model_async
from livedb.GetLatestPapers import pubmed_efetch, pubmed_esearch
from livedb.OpenAlexDownload import fetch_openalex_latest


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
    if not oa_df.empty:
        for _, row in oa_df.iterrows():
            if not row.get("abstract") or not row.get("title"):
                continue
            doi = row.get("doi")
            key = doi or row.get("pmid") or row["id"]
            papers[key] = PaperMetadata(
                doi=doi,
                pmid=row.get("pmid"),
                title=row["title"],
                authors=_parse_authors(row.get("authors")),
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

    log.info(f"Fetched {len(papers)} unique papers for query: {query}")

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
