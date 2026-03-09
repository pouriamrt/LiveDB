"""Phase 4: LLM-based gap identification (within-cluster + cross-cluster)."""

from __future__ import annotations

import asyncio
import json

from loguru import logger as log
from Config import config
from gap_analysis import openai_client as _client
from gap_analysis.models import GapReport, ResearchGap, ThemeCluster
from gap_analysis.prompts import (
    CROSS_CLUSTER_SYSTEM,
    CROSS_CLUSTER_USER,
    WITHIN_CLUSTER_SYSTEM,
    WITHIN_CLUSTER_USER,
)


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
