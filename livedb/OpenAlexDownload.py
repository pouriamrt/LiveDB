import asyncio
import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import arrow
import aiofiles
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
import re
from Config import config
from loguru import logger
from typing import Optional, Dict, List


BROWSER_SEM = asyncio.Semaphore(3) # limit concurrent browsers

def abstract_from_inverted(inv: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """
    Reconstruct plaintext abstract from OpenAlex-style abstract_inverted_index.
    Returns None if the index is missing/empty.
    """
    if not inv:
        return None

    max_pos = max(p for pos in inv.values() for p in pos)
    tokens = [""] * (max_pos + 1)
    for token, positions in inv.items():
        for p in positions:
            tokens[p] = token

    text = " ".join(tokens).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)     # no space before punctuation
    text = re.sub(r"\(\s+", "(", text)               # no space after opening parenthesis
    text = re.sub(r"\s+\)", ")", text)               # no space before closing parenthesis
    text = re.sub(r"\s{2,}", " ", text)              # collapse doubles
    return text

def _looks_like_pdf(content_type: str | None, content: bytes, content_disp: str | None = None, url: str | None = None) -> bool:
    ct = (content_type or "").lower()
    cd = (content_disp or "").lower()
    u  = (url or "").lower()

    if "application/pdf" in ct:
        return True
    if ct.startswith("application/octet-stream"):
        return True
    if "filename=" in cd and ".pdf" in cd:
        return True
    # if u.split("?", 1)[0].endswith(".pdf"):
    #     return True

    head = content[:4096]
    if b"%PDF" in head:
        return True

    return False

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=6),
    retry=retry_if_exception_type(httpx.RequestError),
    reraise=True,
)
async def _get_with_retry(client: httpx.AsyncClient, params: dict) -> dict:
    """GET request with exponential backoff retries."""
    resp = await client.get(config.OPENALEX_BASE, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _row_from_work(w: dict) -> dict:
    authors = [a["author"]["display_name"] for a in w.get("authorships", []) if "author" in a]
    primary_loc = w.get("primary_location") or {}
    venue = (primary_loc.get("source") or {}).get("display_name")
    url = primary_loc.get("landing_page_url") or w.get("id")
    oa = bool(primary_loc.get("is_oa"))
    oa_pdf = primary_loc.get("pdf_url")
    concepts = w.get("concepts") or []
    concept_names = ", ".join(
        [c["display_name"] for c in sorted(concepts, key=lambda c: c.get("score", 0), reverse=True)[:5]]
    )
    pmid = w.get("ids", {}).get("pmid")
    if pmid:
        pmid = pmid.split("/")[-1]
        
    abstract = abstract_from_inverted(w.get("abstract_inverted_index"))

    return {
        "id": w.get("id"),
        "pmid": pmid,
        "title": w.get("title"),
        "publication_date": w.get("publication_date"),
        "pub_year": w.get("publication_year"),
        "type": w.get("type"),
        "journal": venue,
        "doi": w.get("doi"),
        "is_oa": oa,
        "oa_pdf": oa_pdf,
        "cited_by_count": w.get("cited_by_count"),
        "authors": ", ".join(authors),
        "concepts": concept_names,
        "url": url,
        "abstract": abstract,
    }


async def fetch_openalex_latest(
    query: str,
    start_day: int = 30,
    days_back: int = 1,
    max_records: int = 100,
    per_page: int = 25,
    only_articles: bool = True,
    only_oa: bool = False,
    language: str | None = "en",
    extra_filters: dict | None = None,
) -> pd.DataFrame:
    """Fetch latest papers from OpenAlex."""
    mindate = arrow.now().shift(days=-(start_day+days_back)).format("YYYY-MM-DD")
    maxdate = arrow.now().shift(days=-start_day).format("YYYY-MM-DD")
    
    params = {
        "search": query,
        "sort": "publication_date:desc",
        "per-page": per_page,
        "mailto": config.OPENALEX_MAILTO,
        "cursor": "*",
    }

    filters = []
    if mindate:
        filters.append(f"from_publication_date:{mindate}")
    if maxdate:
        filters.append(f"to_publication_date:{maxdate}")
    if only_articles:
        filters.append("type:article")
    if only_oa:
        filters.append("open_access.is_oa:true")
    if language:
        filters.append(f"language:{language}")
    if extra_filters:
        for k, v in extra_filters.items():
            if isinstance(v, (list, tuple, set)):
                filters.append(f"{k}:{'|'.join(map(str, v))}")
            else:
                filters.append(f"{k}:{v}")
    if filters:
        params["filter"] = ",".join(filters)

    results = []
    seen = 0

    async with httpx.AsyncClient(headers={"User-Agent": "openalex-client/2.0"}) as client:
        cursor = "*"
        while seen < max_records:
            params["cursor"] = cursor
            data = await _get_with_retry(client, params)

            works = data.get("results", [])
            for w in works:
                results.append(_row_from_work(w))
                seen += 1
                if seen >= max_records:
                    break

            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor or not works:
                break
            
            await asyncio.sleep(0.2)

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    return df


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=10),
    reraise=True,
)
async def download_pdf_async(url: str, save_path: str, landing_url: str, timeout: float = 20.0, headless: bool = True) -> str:
    """Asynchronously downloads a PDF file from the given URL and saves it to save_path."""
    async with httpx.AsyncClient(http2=True, follow_redirects=True, timeout=timeout) as client:
        if landing_url:
            await asyncio.wait_for(
                client.get(landing_url, headers={**config.COMMON_HEADERS, "Referer": landing_url}),
                timeout=timeout,
            )
        try:
            try:
                await asyncio.wait_for(
                    client.head(url, headers={**config.COMMON_HEADERS, "Referer": (landing_url or "https://www.google.com/")}),
                    timeout=timeout,
                )
            except httpx.HTTPError:
                pass

            response = await asyncio.wait_for(
                client.get(url, headers={**config.COMMON_HEADERS, "Referer": (landing_url or "https://www.google.com/")}),
                timeout=timeout,
            )
            
            if response.status_code in (401, 403):
                if landing_url:
                    await asyncio.wait_for(
                        client.get(landing_url, headers={**config.COMMON_HEADERS, "Referer": landing_url}),
                        timeout=timeout,
                    )
                response = await asyncio.wait_for(
                    client.get(
                        url,
                        headers={
                            **config.COMMON_HEADERS,
                            "Referer": (landing_url or "https://www.google.com/"),
                            "Range": "bytes=0-",
                        },
                    ),
                    timeout=timeout,
                )

            response.raise_for_status()

            if not _looks_like_pdf(response.headers.get("content-type"), response.content, 
                                   response.headers.get("content-disposition"), str(response.url)):
                logger.error("Not a PDF content-type or magic bytes")

            async with aiofiles.open(save_path, "wb") as f:
                await f.write(response.content)
            return save_path

        except Exception as e:
            async with BROWSER_SEM:
                async with Stealth().use_async(async_playwright()) as p:
                    browser = await p.chromium.launch(headless=headless)
                    context = await browser.new_context(
                        user_agent=config.COMMON_HEADERS["User-Agent"],
                        java_script_enabled=True,
                        ignore_https_errors=True,
                    )
                    try:
                        page = await context.new_page()
                        if landing_url:
                            await asyncio.wait_for(
                                page.goto(landing_url, wait_until="networkidle"), timeout=timeout
                            )

                        api_resp = await asyncio.wait_for(
                            context.request.get(
                                url,
                                headers={
                                    "Referer": landing_url or "https://www.google.com/",
                                    "Accept": config.COMMON_HEADERS["Accept"],
                                    "Accept-Language": config.COMMON_HEADERS["Accept-Language"],
                                    "Accept-Encoding": config.COMMON_HEADERS["Accept-Encoding"],
                                },
                                max_redirects=3,
                            ),
                            timeout=timeout,
                        )

                        if not api_resp.ok:
                            api_resp = await asyncio.wait_for(
                                context.request.get(
                                    url,
                                    headers={
                                        "Referer": landing_url or "https://www.google.com/",
                                        "Accept": config.COMMON_HEADERS["Accept"],
                                        "Accept-Language": config.COMMON_HEADERS["Accept-Language"],
                                        "Accept-Encoding": config.COMMON_HEADERS["Accept-Encoding"],
                                        "Range": "bytes=0-",
                                    },
                                    max_redirects=3,
                                ),
                                timeout=timeout,
                            )
                            
                        body = await api_resp.body()
                        if not _looks_like_pdf(api_resp.headers.get("content-type"), body,
                                            api_resp.headers.get("content-disposition"), str(api_resp.url)):
                            def _is_pdf_response(r):
                                ct = (r.headers.get("content-type") or "").lower()
                                url_l = (str(r.url) or "").lower()
                                return ("application/pdf" in ct) or url_l.endswith(".pdf")

                            await page.bring_to_front()
                            
                            nav_target = (landing_url or (re.sub(r"/pdf(/|$)", r"/", url)))
                            try:
                                await page.goto(nav_target, wait_until="domcontentloaded")
                            except Exception:
                                pass

                            async with page.expect_response(_is_pdf_response, timeout=timeout * 1000) as waiter:
                                await page.goto(url, wait_until="domcontentloaded")
                            pdf_resp = await waiter.value
                            body = await pdf_resp.body()

                            if not _looks_like_pdf(
                                pdf_resp.headers.get("content-type"),
                                body,
                                pdf_resp.headers.get("content-disposition"),
                                str(pdf_resp.url),
                            ):
                                logger.error("Playwright fetched non-PDF content after CF fallback")

                        async with aiofiles.open(save_path, "wb") as f:
                            await f.write(body)
                        return save_path
                    except Exception as e:
                        logger.error(f"Error downloading PDF from {url}: {e}")
                    finally:
                        await context.close()
                        await browser.close()

