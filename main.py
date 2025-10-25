from livedb.GetLatestPapers import pubmed_esearch, pubmed_efetch, try_fetch_pmc_fulltext, try_fetch_pmc_fulltext_pdf
import asyncio
from tqdm.asyncio import tqdm_asyncio
from loguru import logger
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logger.add("logs/livedb.log", rotation="10 MB", retention="10 days")

async def main():
    logger.info("Starting livedb")
    
    pmids = await pubmed_esearch("covid-19", days_back=1, start_day=30, retmax=2)
    logger.info(f"Found {len(pmids)} PMIDs")
    
    records = await pubmed_efetch(pmids)
    logger.info(f"Found {len(records)} records")
    
    fulltexts = await tqdm_asyncio.gather(*[
        try_fetch_pmc_fulltext_pdf(rec.get("pmid"), rec.get("pmcid"))
        for rec in records
    ], desc="Fetching full texts")
    
    logger.info(f"Found {len(fulltexts)} fulltexts")
    logger.info("Finished livedb")

if __name__ == "__main__":
    asyncio.run(main())
