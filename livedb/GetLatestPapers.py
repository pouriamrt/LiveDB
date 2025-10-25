from typing import List, Optional, Dict, Any
from urllib.parse import urlencode
import httpx
import aioftp
import aiofiles
from tenacity import retry, stop_after_attempt, wait_exponential
from lxml import etree
from urllib.parse import urlparse
from Config import config
import arrow
import os
import asyncio
import tarfile
from livedb.utils import save_text_as_pdf_async


def eutils_params(extra: Dict[str, Any]) -> Dict[str, Any]:
    params = {"email": config.NCBI_EMAIL}
    if config.NCBI_API_KEY:
        params["api_key"] = config.NCBI_API_KEY
    params.update(extra)
    return params


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
async def pubmed_esearch(query: str, days_back: int = 1, start_day: int = 30, retmax: int = 200) -> List[str]:
    """Use ESearch to find recent PMIDs (start_day to start_day + days_back), sorted by most recent."""
    mindate = arrow.now().shift(days=-(start_day+days_back)).format("YYYY/MM/DD")
    maxdate = arrow.now().shift(days=-start_day).format("YYYY/MM/DD")
    params = eutils_params({
        "db": "pubmed",
        "term": query,
        "datetype": "edat",
        "mindate": mindate,
        "maxdate": maxdate,
        "retmax": retmax,
        "retmode": "json",
        "sort": "most+recent",
    })
    url = f"{config.BASE_EUTILS}/esearch.fcgi?{urlencode(params)}"

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url)

    r.raise_for_status()
    data = r.json()
    return data.get("esearchresult", {}).get("idlist", [])


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
async def pubmed_efetch(pmids: List[str]) -> List[Dict[str, Any]]:
    """Fetch article metadata in Medline XML; parse essentials."""
    if not pmids:
        return []

    params = eutils_params({
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    })
    url = f"{config.BASE_EUTILS}/efetch.fcgi"

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, data=params)

    r.raise_for_status()
    root = etree.fromstring(r.content)
    out: List[Dict[str, Any]] = []

    for article in root.xpath("//PubmedArticle"):
        def x(path):
            el = article.xpath(path)
            return el[0].text if el else None

        pmid = x(".//PMID")
        doi = x(".//ArticleIdList/ArticleId[@IdType='doi']")
        title = x(".//ArticleTitle")
        journal = x(".//Journal/Title")
        year = x(".//JournalIssue/PubDate/Year")
        abstract = " ".join([t.text for t in article.xpath(".//Abstract/AbstractText") if t.text]) or None
        url_article = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
        pmcid = x(".//ArticleIdList/ArticleId[@IdType='pmc']")

        names = []
        for au in article.xpath(".//AuthorList/Author"):
            last_el = (au.xpath("./LastName") or [None])[0]
            ini_el = (au.xpath("./Initials") or [None])[0]
            if last_el is not None and ini_el is not None:
                names.append(f"{last_el.text} {ini_el.text}")
        authors = ", ".join(names) if names else None

        out.append({
            "pmid": pmid,
            "pmcid": pmcid,
            "doi": doi,
            "title": title,
            "journal": journal,
            "pub_year": int(year) if year and year.isdigit() else None,
            "authors": authors,
            "url": url_article,
            "abstract": abstract,
        })
    return out


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
async def try_fetch_pmc_fulltext(pmid: Optional[str], pmcid: Optional[str]) -> Dict[str, Any]:
    """
    Use BioC PMC OA endpoint for legal full text when available.
    Returns dict: {full_text, license, is_open_access}
    """
    if not (pmid or pmcid):
        return {"full_text": None, "license": None, "is_open_access": False}

    ident = pmcid if pmcid else pmid
    url = f"{config.BASE_BIOC}/BioC_xml/{ident}/unicode"

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url)

    if r.status_code != 200:
        return {"full_text": None, "license": None, "is_open_access": False}

    try:
        root = etree.fromstring(r.content)
        sections = root.xpath("//passage/infon[@key='section_type']")
        passages = root.xpath("//passage/text")
        text_content = "\n\n".join([s.text + "\n" + p.text for s, p in zip(sections, passages) if p is not None and p.text])

        lic = None
        inf = root.xpath("//document/infon[@key='license']")
        if inf and inf[0].text:
            lic = inf[0].text

        return {"full_text": text_content or None, "license": lic, "is_open_access": True}
    except Exception:
        return {"full_text": None, "license": None, "is_open_access": False}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
async def extract_pdfs_from_tar_async(tar_path: str, output_dir: str) -> List[str]:
    def _extract_sync() -> List[str]:
        os.makedirs(output_dir, exist_ok=True)
        extracted: List[str] = []
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.lower().endswith(".pdf") and "s00" not in member.name:
                    basename = os.path.dirname(member.name) + '.pdf'
                    src = tar.extractfile(member)
                    if src is None:
                        continue
                    out_path = os.path.join(output_dir, basename)
                    with open(out_path, "wb") as out:
                        out.write(src.read())
                    extracted.append(out_path)
        return extracted
    return await asyncio.to_thread(_extract_sync)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
async def download_pdf_ftp(url: str, *, dest_dir: Optional[str] = None, chunk_size: int = 256*1024) -> List[str]:
    """
    Download a PMC OA FTP URL.
    - If URL points to a .tar.gz (oa_package), download then extract PDFs and return their paths.
    - If URL points to a .pdf (oa_pdf), download the PDF and return [pdf_path].
    """
    dest_dir_pdfs = dest_dir or config.PDF_DIR
    os.makedirs(dest_dir_pdfs, exist_ok=True)
    dest_dir = dest_dir_pdfs + '/zips'
    os.makedirs(dest_dir, exist_ok=True)

    parsed = urlparse(url)
    if parsed.scheme != "ftp":
        raise ValueError(f"Expected ftp:// URL, got: {url}")

    host = parsed.hostname or config.FTP_HOST
    remote_path = parsed.path                   # e.g. /pub/pmc/oa_package/00/00/PMC1790863.tar.gz
    remote_dir, remote_name = os.path.split(remote_path)
    local_path = os.path.join(dest_dir, remote_name)
    
    async with aioftp.Client.context(
        host,
        user=getattr(config, "FTP_USER", "anonymous"),
        password=getattr(config, "FTP_PASSWORD", "anonymous@"),
        socket_timeout=120,
    ) as client:
        await client.change_directory(remote_dir)
        async with client.download_stream(remote_name) as stream:
            async with aiofiles.open(local_path, "wb") as out:
                async for block in stream.iter_by_block(chunk_size):
                    await out.write(block)

    if remote_name.lower().endswith(".pdf"):
        return [local_path]

    if remote_name.lower().endswith((".tar.gz", ".tgz")):
        pdfs = await extract_pdfs_from_tar_async(local_path, dest_dir_pdfs)
        return pdfs

    return [local_path]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
async def try_fetch_pmc_fulltext_pdf(pmid: Optional[str], pmcid: Optional[str]) -> Dict[str, Any]:
    """
    Use FTP to fetch PDF full text when available.
    Save PDF to local directory.
    Returns dict: {path, license, is_open_access}
    """
    if not (pmid or pmcid):
        return {"path": None, "license": None, "is_open_access": False}

    ident = pmcid if pmcid else pmid
    url = f"{config.PMC_OA_SERVICE}?id={ident}"

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url)

    if r.status_code != 200:
        return {"path": None, "license": None, "is_open_access": False}

    try:
        root = etree.fromstring(r.content)
        records = root.xpath("//records/record")
        lic = records[0].xpath("./@license")[0]
        urls = [record.xpath("./link/@href")[0] for record in records if record.xpath("./link/@href")]
        if not urls:
            return {"path": None, "license": None, "is_open_access": False}
        
    except Exception as e:
        return {"path": None, "license": None, "is_open_access": False, "error": str(e)}
    
    try:
        paths = await asyncio.gather(*[download_pdf_ftp(url) for url in urls])
        
        if len(paths[0]) < 1:
            data = await try_fetch_pmc_fulltext(pmid, pmcid)
            if data.get("full_text") is None:
                return {"path": None, "license": None, "is_open_access": False}
            
            save_dir = f"{config.PDF_DIR}/{pmcid}.pdf"
            await save_text_as_pdf_async(data.get("full_text"), save_dir)
            return {"path": save_dir, "license": data["license"], "is_open_access": data["is_open_access"]}

        return {"path": paths, "license": lic, "is_open_access": True}
    except Exception as e:
        return {"path": None, "license": None, "is_open_access": False, "error": str(e)}
    