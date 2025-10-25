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
    
config = Config()