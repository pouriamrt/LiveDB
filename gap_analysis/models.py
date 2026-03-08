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
