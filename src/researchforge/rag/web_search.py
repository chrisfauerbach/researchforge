"""Web search fallback — fetch and chunk web pages for evidence."""

from __future__ import annotations

import asyncio
import hashlib

import httpx
import structlog
from bs4 import BeautifulSoup

from researchforge.config import WebSearchConfig
from researchforge.rag.chunker import chunk_document

logger = structlog.get_logger()


async def _ddg_search(query: str, max_results: int) -> list[dict]:
    """Run a DuckDuckGo text search (sync library, wrapped in executor)."""
    from duckduckgo_search import DDGS

    def _search() -> list[dict]:
        with DDGS() as ddgs:
            return [
                {"url": r["href"], "title": r["title"]}
                for r in ddgs.text(query, max_results=max_results)
            ]

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _search)


async def _fetch_page(url: str, timeout: int) -> str | None:
    """Fetch a URL and return raw HTML, or None on failure."""
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": "ResearchForge/0.1"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type:
                logger.debug("web_search_skip_non_html", url=url, content_type=content_type)
                return None
            return resp.text
    except Exception as exc:
        logger.warning("web_search_fetch_failed", url=url, error=str(exc))
        return None


def _extract_text(html: str, max_chars: int) -> str:
    """Strip non-content tags from HTML and return plain text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return text[:max_chars]


def _to_chunks(url: str, title: str, text: str) -> list[dict]:
    """Chunk extracted web text and return chunk dicts ready for the gatherer."""
    chunks = chunk_document(text, source_type="txt", extra_metadata={
        "source_path": url,
        "section_h1": title,
        "source_type": "web",
        "content_type": "web_search",
    })
    result = []
    for chunk in chunks:
        raw_id = f"{url}:{chunk.chunk_index}"
        chunk_id = "web:" + hashlib.md5(raw_id.encode()).hexdigest()[:12]
        result.append({
            "chunk_id": chunk_id,
            "text": chunk.text,
            "source_path": url,
            "section_h1": title,
            "source_type": "web",
            "content_type": "web_search",
        })
    return result


async def web_search_for_question(query: str, cfg: WebSearchConfig) -> list[dict]:
    """Search the web for a query, fetch pages, extract text, and return chunks.

    Never raises — returns [] on any failure.
    """
    try:
        hits = await _ddg_search(query, cfg.max_results)
        logger.info("web_search_hits", query=query, count=len(hits))
    except Exception as exc:
        logger.warning("web_search_ddg_failed", query=query, error=str(exc))
        return []

    if not hits:
        return []

    # Fetch all pages concurrently
    fetch_tasks = [_fetch_page(h["url"], cfg.fetch_timeout_seconds) for h in hits]
    pages = await asyncio.gather(*fetch_tasks)

    all_chunks: list[dict] = []
    for hit, html in zip(hits, pages):
        if html is None:
            continue
        text = _extract_text(html, cfg.max_page_chars)
        if not text.strip():
            continue
        chunks = _to_chunks(hit["url"], hit["title"], text)
        all_chunks.extend(chunks)

    logger.info("web_search_chunks", query=query, total_chunks=len(all_chunks))
    return all_chunks
