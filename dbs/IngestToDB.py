from dbs.utils import CustomChunking
from Config import config
from agno.models.openai import OpenAIChat
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector, SearchType
from agno.db.postgres import PostgresDb
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.reader.pdf_reader import PDFReader
import arrow
from loguru import logger
from tqdm.asyncio import tqdm_asyncio

model = OpenAIChat(id=config.MODEL_NAME, api_key=config.OPENAI_API_KEY)

embedder = OpenAIEmbedder(id=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)
reader = PDFReader(
    split_on_pages=True, 
    read_images=True, 
    chunk=True, 
    chunking_strategy=CustomChunking(similarity_threshold=0.5, chunk_size=1000, embedder=embedder)
)

contents_db = PostgresDb(
    db_url=config.PGVECTOR_CONTENTS_URL,
    knowledge_table=config.PGVECTOR_CONTENTS_TABLE
)

vector_db=PgVector(table_name=config.PGVECTOR_TABLE, db_url=config.PGVECTOR_URL, search_type=SearchType.hybrid, 
                       embedder=embedder)

knowledge_base = Knowledge(
    vector_db=vector_db,
    contents_db=contents_db
)

logger.info("IngestToDB initialized")

async def process_one(record):
    try:
        file_path = str(record.get("fulltext_path"))
        if not file_path:
            logger.error(f"No fulltext path found for record {record.get('pmid')}")
            return False
        
        pub_year = int(record.get("pub_year"))
        added_year = int(arrow.now().year)
        author = str(record.get("authors"))
        title = str(record.get("title"))
        journal = str(record.get("journal"))
        abstract = str(record.get("abstract"))
        population_flag = str(record.get("P_AB_pred"))
        intervention_flag = str(record.get("I_AB_pred"))
        comparator_flag = str(record.get("C_AB_pred"))
        outcome_flag = str(record.get("O_AB_pred"))
        study_design_flag = str(record.get("S_AB_pred"))
        qualification_flag = str(record.get("final_pred"))
        
        metadata = {
            "publication_year": pub_year,
            "date_added": added_year,
            "author": author,
            "title": title,
            "journal": journal,
            "abstract": abstract,
            "population_flag": population_flag,
            "intervention_flag": intervention_flag,
            "comparator_flag": comparator_flag,
            "outcome_flag": outcome_flag,
            "study_design_flag": study_design_flag,
            "qualification_flag": qualification_flag
        }
        
        await knowledge_base.add_content_async(
            path=file_path,
            reader=reader,
            metadata=metadata
        )

        return True
    
    except Exception as e:
        logger.error(f"Error processing record {record.get('pmid')}: {e}")
        return False
    
async def ingest_to_db_async(records):
    results = await tqdm_asyncio.gather(*[process_one(record) for record in records], desc="Ingesting to database")
    success_count = results.count(True)
    failed_count = results.count(False)
    logger.info(f"Successfully processed {success_count} records, failed to process {failed_count} records")
    return True

