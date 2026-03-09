"""Phase 2: LLM-based structured extraction from paper abstracts."""

from __future__ import annotations

import asyncio
import json

from loguru import logger as log

from Config import config
from gap_analysis import openai_client as _client
from gap_analysis.models import PaperExtraction, PaperMetadata
from gap_analysis.prompts import EXTRACTION_SYSTEM, EXTRACTION_USER


async def _extract_batch(
    papers: list[PaperMetadata], model: str, sem: asyncio.Semaphore
) -> list[PaperExtraction]:
    """Extract structured findings from a batch of papers via LLM."""
    papers_text = "\n\n".join(
        f"--- Paper {i + 1} ---\nTitle: {p.title}\nAbstract: {p.abstract}"
        for i, p in enumerate(papers)
    )

    async with sem:
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
            items = (
                raw
                if isinstance(raw, list)
                else raw.get("papers", raw.get("results", [raw]))
            )
        except Exception as e:
            log.warning(f"Batch extraction failed: {e}")
            return []

    # I2: Warn when LLM returns different number of items than the batch size
    if len(items) != len(papers):
        log.warning(
            f"Batch count mismatch: sent {len(papers)} papers but LLM returned "
            f"{len(items)} items — some papers may be dropped"
        )

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
    batch_size: int = config.GAP_LLM_BATCH_SIZE,
    model: str | None = None,
) -> list[PaperExtraction]:
    """Extract structured findings from all papers in batches."""
    model = model or config.MODEL_NAME
    # C3: Create semaphore inside the function so it's bound to the current event loop
    sem = asyncio.Semaphore(config.GAP_LLM_CONCURRENCY)

    batches = [papers[i : i + batch_size] for i in range(0, len(papers), batch_size)]
    log.info(f"Extracting findings from {len(papers)} papers in {len(batches)} batches")

    tasks = [_extract_batch(batch, model, sem) for batch in batches]
    results = await asyncio.gather(*tasks)

    extractions = [ext for batch_result in results for ext in batch_result]
    log.info(f"Extracted findings from {len(extractions)}/{len(papers)} papers")
    return extractions
