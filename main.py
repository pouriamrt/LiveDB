from livedb.GetLatestPapers import pubmed_esearch, pubmed_efetch, try_fetch_pmc_fulltext_pdf
import asyncio
from tqdm.asyncio import tqdm_asyncio
from loguru import logger
import sys
from itertools import chain
from livedb.CheckAbsModel import check_abs_model_async
from dbs.IngestToDB import ingest_to_db_async
from livedb.OpenAlexDownload import fetch_openalex_latest, download_pdf_async
import os
from Config import config

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logger.add("logs/livedb.log", rotation="10 MB", retention="10 days")

async def main():
    logger.info("Starting livedb")
    
    query = "dementia"
    
    ##########################################################
    ########## Fetch records from OpenAlex ###################
    ##########################################################
    df = await fetch_openalex_latest(
        query=query,
        start_day=30,
        days_back=1,
        max_records=100,
        only_articles=True,
        only_oa=True,
        extra_filters={"primary_location.source.type": "journal"},
    )
    df = df.dropna(subset=["oa_pdf"])
    
    # Convert the DataFrame to a list of dictionaries
    records_from_openalex = [row.to_dict() for _, row in df.iterrows()]
    logger.info(f"Found {len(records_from_openalex)} records from OpenAlex")
    
    # Fetch predictions from the model for the abstracts from OpenAlex
    predictions_from_openalex = await tqdm_asyncio.gather(*[
        check_abs_model_async(row.get("title") + "\n" + row.get("abstract"))
        for row in records_from_openalex if row.get("abstract") is not None and 
        row.get("title") is not None
    ], desc="Checking abstracts from OpenAlex")
    
    for i, pred in enumerate(predictions_from_openalex):
        # final_pred = "no" if "no" in list(chain(*pred[0].values())) else "yes"
        final_pred = "no" if pred[0]['S_AB_pred'] == "no" else "yes"
        logger.info(f"PMID (from OpenAlex) {records_from_openalex[i]['pmid']} - Final prediction: {final_pred}")
        records_from_openalex[i]["final_pred"] = final_pred
        for key, value in pred[0].items():
            records_from_openalex[i][key] = value
            
    included_openalex_records = [rec for rec in records_from_openalex if rec.get("final_pred") == "yes"]
    
    # Download PDFs from OpenAlex
    openalex_pdf_paths = await tqdm_asyncio.gather(*[
            download_pdf_async(row["oa_pdf"], os.path.join(config.PDF_DIR, f"{row['pmid'] or row['id'].split('/')[-1]}.pdf"), row["id"], headless=config.HEADLESS)
            for row in included_openalex_records
        ], desc="Downloading PDFs from OpenAlex")

    records_not_included_from_openalex = []
    for i, rec in enumerate(included_openalex_records):
        if openalex_pdf_paths[i] is None:
            records_not_included_from_openalex.append(rec)
            included_openalex_records[i]["fulltext_path"] = None
            continue
        included_openalex_records[i]["fulltext_path"] = openalex_pdf_paths[i]
        
    logger.info(f"Found {len(included_openalex_records)} included records from OpenAlex from {len(records_from_openalex)} total records")
    
    ##########################################################
    ########## Fetch records from PMC ########################
    ##########################################################
    pmids = await pubmed_esearch(query, days_back=1, start_day=30, retmax=10)
    logger.info(f"Found {len(pmids)} PMIDs")
    
    records_from_pmc = await pubmed_efetch(pmids)
    logger.info(f"Found {len(records_from_pmc)} records from PMC")
    
    # Fetch predictions from the model for the abstracts from PMC
    predictions_from_pmc = await tqdm_asyncio.gather(*[
        check_abs_model_async(rec.get("title") + "\n" + rec.get("abstract"))
        for rec in records_from_pmc if rec.get("abstract") is not None and 
        rec.get("title") is not None
    ], desc="Checking abstracts from PMC")
    
    for i, pred in enumerate(predictions_from_pmc):
        # final_pred = "no" if "no" in list(chain(*pred[0].values())) else "yes"
        final_pred = "no" if pred[0]['S_AB_pred'] == "no" else "yes"
        logger.info(f"PMID {records_from_pmc[i]['pmid']} - PMCID {records_from_pmc[i]['pmcid']} - Final prediction: {final_pred}")
        records_from_pmc[i]["final_pred"] = final_pred
        for key, value in pred[0].items():
            records_from_pmc[i][key] = value
            
    ##########################################################
    ########## Combine records from OpenAlex and PMC #########
    ##########################################################
    records = records_from_pmc + records_not_included_from_openalex
    
    included_records = [rec for rec in records if rec.get("final_pred") == "yes"]
    logger.info(f"Included {len(included_records)} records out of {len(records)} total records")
    
    # Fetch full texts from PMC
    fulltexts_from_pmc = await tqdm_asyncio.gather(*[
        try_fetch_pmc_fulltext_pdf(rec.get("pmid"), rec.get("pmcid"))
        for rec in included_records
    ], desc="Fetching full texts")
    
    for i, rec in enumerate(included_records):
        if fulltexts_from_pmc[i].get("path") is None:
            continue
        included_records[i]["fulltext_path"] = fulltexts_from_pmc[i].get("path")[0][0]
    
    included_records += included_openalex_records
    
    total_fulltexts = 0
    for rec in included_records:
        if rec.get("fulltext_path"):
            total_fulltexts += 1
    logger.info(f"Found {total_fulltexts} total fulltexts")
    
    ##########################################################
    ########## Ingest records to database ####################
    ##########################################################
    await ingest_to_db_async(included_records)
    
    logger.info("Finished livedb")

if __name__ == "__main__":
    asyncio.run(main())
