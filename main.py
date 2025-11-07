import os, sys, asyncio
from datetime import timedelta
from typing import List, Dict, Any

from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner
from prefect.tasks import task_input_hash

from loguru import logger as log

from livedb.GetLatestPapers import pubmed_esearch, pubmed_efetch, try_fetch_pmc_fulltext_pdf
from livedb.CheckAbsModel import check_abs_model_async
from livedb.OpenAlexDownload import fetch_openalex_latest, download_pdf_async
from dbs.IngestToDB import ingest_to_db_async
from Config import config

# --- optional: Windows async policy
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
DOWNLOAD_SEM = asyncio.Semaphore(8) # limit concurrent downloads to 8

# --- Prefect loguru sink ---
os.makedirs("logs", exist_ok=True)
log.remove()
log.add("logs/livedb.log", rotation="10 MB", retention="10 days", enqueue=True, backtrace=True, diagnose=False)

def _prefect_loguru_sink(message):
    try:
        prlog = get_run_logger()
        r = message.record
        lvl_name = r["level"].name
        prlog.log(lvl_name, r["message"])
    except Exception:
        sys.stderr.write(message + "\n")

log.add(_prefect_loguru_sink, level="DEBUG")

# -------------------- TASKS --------------------

@task(
    name="Fetch OpenAlex",
    retries=2, retry_delay_seconds=15,
    timeout_seconds=1200,
    cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1),
)
async def fetch_openalex_task(query: str, start_day: int, days_back: int, max_records: int) -> List[Dict[str, Any]]:
    df = await fetch_openalex_latest(
        query=query, start_day=start_day, days_back=days_back,
        max_records=max_records, only_articles=True, only_oa=True,
        # extra_filters={"primary_location.source.type": "journal"},
    )
    df = df.dropna(subset=["oa_pdf"])
    return [row.to_dict() for _, row in df.iterrows()]

@task(name="Classify Abstract", retries=0)
async def classify_record_task(rec: Dict[str, Any]) -> Dict[str, Any]:
    text = (rec.get("title") or "") + "\n" + (rec.get("abstract") or "")
    preds, confs = await check_abs_model_async(text)
    final_pred = "no" if preds.get("S_AB_pred") == "no" else "yes"
    return {**rec, "final_pred": final_pred, **preds}

@task(name="Download OpenAlex PDF", retries=2, retry_delay_seconds=30, timeout_seconds=900)
async def download_openalex_pdf_task(rec: Dict[str, Any]) -> Dict[str, Any]:
    async with DOWNLOAD_SEM:
        target = os.path.join(config.PDF_DIR, f"{rec.get('pmid') or rec['id'].split('/')[-1]}.pdf")
        path = await download_pdf_async(rec["oa_pdf"], target, rec["id"], headless=config.HEADLESS)
        rec["fulltext_path"] = path
    return rec

@task(name="PubMed Search", retries=2, retry_delay_seconds=15, cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
async def pubmed_search_task(query: str, start_day: int, days_back: int, retmax: int) -> List[str]:
    return await pubmed_esearch(query, days_back=days_back, start_day=start_day, retmax=retmax)

@task(name="PubMed EFetch", retries=2, retry_delay_seconds=15, timeout_seconds=1200)
async def pubmed_fetch_task(pmids: List[str]) -> List[Dict[str, Any]]:
    return await pubmed_efetch(pmids)

@task(name="Fetch PMC Fulltext", retries=2, retry_delay_seconds=20, timeout_seconds=1200)
async def fetch_pmc_fulltext_task(rec: Dict[str, Any]) -> Dict[str, Any]:
    res = await try_fetch_pmc_fulltext_pdf(rec.get("pmid"), rec.get("pmcid"))
    if res and res.get("path"):
        # your function returns list-of-tuples; take first path
        rec["fulltext_path"] = res["path"][0][0]
    return rec

@task(name="Ingest to DB", retries=2, retry_delay_seconds=30, timeout_seconds=3600)
async def ingest_task(recs: List[Dict[str, Any]]) -> int:
    await ingest_to_db_async(recs)
    return len(recs)

# -------------------- FLOW --------------------

@flow(name="LiveDB ETL", task_runner=ConcurrentTaskRunner(max_workers=16))
async def livedb_flow(
    query: str = "dementia",
    start_day: int = 30,
    days_back: int = 1,
    max_records: int = 10,
):
    log.info("Starting LiveDB ETL")

    # Kick off OA + PubMed concurrently
    oa_fut    = fetch_openalex_task(query, start_day, days_back, max_records)
    pmids_fut = pubmed_search_task(query, start_day, days_back, max_records)

    openalex_records, pmids = await asyncio.gather(oa_fut, pmids_fut)
    log.info(f"OpenAlex records: {len(openalex_records)} | PMIDs: {len(pmids)}")

    # Classify OA concurrently
    cls_tasks_oa = [classify_record_task(r)
                    for r in openalex_records if r.get("abstract") and r.get("title")]
    classified_openalex = await asyncio.gather(*cls_tasks_oa)
    included_openalex = [r for r in classified_openalex if r.get("final_pred") == "yes"]

    # Download OA PDFs with bounded concurrency
    dl_tasks = [download_openalex_pdf_task(r) for r in included_openalex]
    downloaded_openalex = await asyncio.gather(*dl_tasks)
    not_included_from_oa = [r for r in downloaded_openalex if not r.get("fulltext_path")]

    # PubMed EFetch + classify
    pmc_records = await pubmed_fetch_task(pmids)
    cls_tasks_pmc = [classify_record_task(r)
                     for r in pmc_records if r.get("abstract") and r.get("title")]
    classified_pmc = await asyncio.gather(*cls_tasks_pmc)

    # Combine; keep only "yes"
    combined = classified_pmc + not_included_from_oa
    included = [r for r in combined if r.get("final_pred") == "yes"]

    # Fetch PMC fulltexts
    ft_tasks = [fetch_pmc_fulltext_task(r) for r in included]
    included_with_ft = await asyncio.gather(*ft_tasks)

    # Merge with OA successes
    all_included = included_with_ft + [r for r in downloaded_openalex if r.get("fulltext_path")]
    total_fulltexts = sum(1 for r in all_included if r.get("fulltext_path"))
    log.info(f"Total fulltexts: {total_fulltexts} / {len(all_included)}")

    # Ingest
    await ingest_task(all_included)
    log.info("Finished LiveDB ETL")
    return {"total_fulltexts": total_fulltexts, "total_records": len(all_included)}

# local run
if __name__ == "__main__":
    from rich.prompt import Prompt
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    
    console.print(Panel("[1] ETL\n[2] AI Agent", title="Run Mode"))

    c = Prompt.ask("Choice", choices=["1","2"], show_choices=False)

    if c == "1":
        asyncio.run(livedb_flow(query="dementia", max_records=10))
    else:
        from agents.RunTeam import run_team
        
        session_state = {}
        agent_os, app = run_team(session_state)
        agent_os.serve(app=app, port=7777)
        console.print(Panel("Agent server stopped", title="Run Mode"))
    