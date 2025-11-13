from dotenv import load_dotenv
import os
from pydantic import BaseModel

load_dotenv(override=True)


class Config(BaseModel):
    NCBI_EMAIL: str = os.getenv("NCBI_EMAIL")
    NCBI_API_KEY: str = os.getenv("NCBI_API_KEY")
    BASE_EUTILS: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    BASE_BIOC: str = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi"
    FTP_HOST: str = "ftp.ncbi.nlm.nih.gov"
    FTP_USER: str = os.getenv("FTP_USER")
    FTP_PASSWORD: str = os.getenv("FTP_PASSWORD")
    PMC_OA_SERVICE: str = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
    PDF_DIR: str = os.path.join(os.path.dirname(__file__), "pdfs")
    CHUNK_SIZE: int = 256 * 1024
    MODEL_DIR: str = os.path.join(
        os.path.dirname(__file__), "models/saved_multitask_model"
    )
    MODEL_NAME: str = "gpt-4.1-mini"
    PGVECTOR_URL: str = os.getenv("PGVECTOR_URL")
    PGVECTOR_TABLE: str = os.getenv("PGVECTOR_TABLE")
    PGVECTOR_CONTENTS_URL: str = os.getenv("PGVECTOR_CONTENTS_URL")
    PGVECTOR_CONTENTS_TABLE: str = os.getenv("PGVECTOR_CONTENTS_TABLE")
    SQL_DATABASE_URL: str = os.getenv("SQL_DATABASE_URL")
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    OPENALEX_BASE: str = "https://api.openalex.org/works"
    OPENALEX_MAILTO: str = os.getenv("OPENALEX_MAILTO")
    COMMON_HEADERS: dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/129.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "DNT": "1",
    }
    HEADLESS: bool = True
    PGVECTOR_MEMORY_URL: str = os.getenv("PGVECTOR_MEMORY_URL")
    PGVECTOR_MEMORY_TABLE: str = os.getenv("PGVECTOR_MEMORY_TABLE")


config = Config()
