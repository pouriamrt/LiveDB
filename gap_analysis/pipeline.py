"""Gap analysis pipeline — Prefect flow orchestrating phases 1-5."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from loguru import logger as log
from prefect import flow

from gap_analysis.analyze import analyze_gaps
from gap_analysis.cluster import cluster_papers
from gap_analysis.extract import extract_papers
from gap_analysis.fetch import FilterMode, fetch_papers, translate_query
from gap_analysis.models import GapReport
from gap_analysis.report import generate_dashboard_html, generate_pdf


@flow(name="Gap Analysis")
async def gap_analysis_flow(
    query: str,
    max_records: int = 100,
    days_back: int = 180,
    filter_mode: FilterMode = FilterMode.PICOS,
    filter_description: str = "",
    model: str | None = None,
    output_dir: str = "reports",
) -> GapReport:
    """Run the full gap analysis pipeline.

    Args:
        days_back: How many days back from today to search.
        filter_mode: How to filter papers — "picos" (model), "llm" (description), or "none".
        filter_description: Required when filter_mode is "llm". Describes which papers to include.
    """
    log.info(f"Starting gap analysis for: {query} (max {max_records} papers)")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

    # Phase 0: Translate natural language query into search keywords
    log.info("Phase 0: Translating query...")
    search_queries = await translate_query(query, model=model)

    # Phase 1: Fetch
    # OpenAlex/PubMed API convention: start_day=0 (end at today), days_back=N (window width)
    log.info("Phase 1: Fetching papers...")
    papers = await fetch_papers(
        queries=search_queries,
        max_records=max_records,
        start_day=0,
        days_back=days_back,
        filter_mode=filter_mode,
        filter_description=filter_description,
        model=model,
    )
    if not papers:
        log.warning("No papers found — aborting")
        return GapReport(
            query=query,
            date_range=date_range,
            executive_summary="No papers found for this query.",
        )

    # Phase 2: Extract
    log.info(f"Phase 2: Extracting findings from {len(papers)} papers...")
    extractions = await extract_papers(papers, model=model)
    if not extractions:
        log.warning("No extractions produced — aborting")
        return GapReport(
            query=query,
            scope=len(papers),
            date_range=date_range,
            executive_summary="Extraction failed.",
        )

    # Phase 3: Cluster
    log.info(f"Phase 3: Clustering {len(extractions)} papers...")
    clusters = await cluster_papers(extractions, model=model)

    # Phase 4: Analyze
    log.info(f"Phase 4: Analyzing gaps across {len(clusters)} clusters...")
    report = await analyze_gaps(
        clusters, query=query, date_range=date_range, model=model
    )

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
