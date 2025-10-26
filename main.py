from livedb.GetLatestPapers import pubmed_esearch, pubmed_efetch, try_fetch_pmc_fulltext_pdf
import asyncio
from tqdm.asyncio import tqdm_asyncio
from loguru import logger
import sys
from itertools import chain
from livedb.CheckAbsModel import check_abs_model_async
from dbs.IngestToDB import ingest_to_db_async

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logger.add("logs/livedb.log", rotation="10 MB", retention="10 days")

async def main():
    logger.info("Starting livedb")
    
    pmids = await pubmed_esearch("dementia", days_back=1, start_day=30, retmax=10)
    logger.info(f"Found {len(pmids)} PMIDs")
    
    records = await pubmed_efetch(pmids)
    logger.info(f"Sample record: {records[0]}")
    logger.info(f"Found {len(records)} records")
    
    predictions = await tqdm_asyncio.gather(*[
        check_abs_model_async(rec.get("title") + "\n" + rec.get("abstract"))
        for rec in records if rec.get("abstract") is not None and 
        rec.get("title") is not None
    ], desc="Checking abstracts")
    
    for i, pred in enumerate(predictions):
        # final_pred = "no" if "no" in list(chain(*pred[0].values())) else "yes"
        final_pred = "no" if pred[0]['S_AB_pred'] == "no" else "yes"
        logger.info(f"PMID {records[i]['pmid']} - PMCID {records[i]['pmcid']} - Final prediction: {final_pred}")
        records[i]["final_pred"] = final_pred
        for key, value in pred[0].items():
            records[i][key] = value
    
    included_records = [rec for rec in records if rec.get("final_pred") == "yes"]
    logger.info(f"Included {len(included_records)} records out of {len(records)} total records")
    
    fulltexts = await tqdm_asyncio.gather(*[
        try_fetch_pmc_fulltext_pdf(rec.get("pmid"), rec.get("pmcid"))
        for rec in included_records
    ], desc="Fetching full texts")
    
    for i, rec in enumerate(included_records):
        included_records[i]["fulltext_path"] = fulltexts[i].get("path")[0][0] if fulltexts[i].get("path") else None
    
    logger.info(f"Found {len(fulltexts)} fulltexts")
    
    await ingest_to_db_async(included_records)
    
    logger.info("Finished livedb")

if __name__ == "__main__":
    asyncio.run(main())
