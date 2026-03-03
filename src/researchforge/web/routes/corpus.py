"""Corpus routes: search, upload/ingest, statistics."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import structlog
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from researchforge.web.app import get_repo, get_templates

logger = structlog.get_logger()

router = APIRouter()


@router.get("/corpus", response_class=HTMLResponse)
async def corpus_page(request: Request):
    """Corpus browser page."""
    templates = get_templates()
    return templates.TemplateResponse(
        "corpus.html",
        {"request": request, "active_page": "corpus"},
    )


@router.get("/api/corpus/search", response_class=HTMLResponse)
async def search_corpus(request: Request, q: str = ""):
    """Hybrid search the corpus, returns HTML partial."""
    if not q or len(q.strip()) < 2:
        return HTMLResponse("")

    from researchforge.rag.retriever import retrieve
    from researchforge.rag.store import VectorStore

    store = VectorStore()
    try:
        results = await retrieve(q.strip(), store, top_k=10)
    except Exception as exc:
        logger.warning("corpus_search_error", error=str(exc))
        results = []

    templates = get_templates()
    return templates.TemplateResponse(
        "partials/search_results.html",
        {"request": request, "results": results, "query": q},
    )


@router.get("/api/corpus/stats", response_class=HTMLResponse)
async def corpus_stats(request: Request):
    """Corpus statistics, returns HTML partial."""
    from researchforge.rag.store import VectorStore

    store = VectorStore()
    try:
        chunk_count = store.count()
    except Exception:
        chunk_count = 0

    # Count documents from metadata DB
    repo = await get_repo()
    try:
        cursor = await repo.db.execute("SELECT COUNT(*) FROM documents")
        row = await cursor.fetchone()
        doc_count = row[0] if row else 0
    except Exception:
        doc_count = 0

    templates = get_templates()
    return templates.TemplateResponse(
        "partials/corpus_stats.html",
        {"request": request, "documents": doc_count, "chunks": chunk_count},
    )


@router.post("/api/corpus/ingest", response_class=HTMLResponse)
async def ingest_upload(request: Request, file: UploadFile = File(...)):
    """Upload and ingest a file into the corpus."""
    from researchforge.rag.ingest import ingest_file
    from researchforge.rag.store import VectorStore

    supported = {".pdf", ".md", ".markdown", ".txt", ".html", ".htm", ".docx"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in supported:
        return HTMLResponse(
            f'<div class="flash flash-error">'
            f"Unsupported file type: {suffix}. "
            f"Supported: {', '.join(sorted(supported))}"
            f"</div>"
        )

    # Save upload to temp file
    tmp_dir = tempfile.mkdtemp()
    tmp_path = Path(tmp_dir) / (file.filename or "upload" + suffix)
    try:
        with open(tmp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        store = VectorStore()
        repo = await get_repo()
        result = await ingest_file(tmp_path, store, repo)

        if result.skipped:
            return HTMLResponse(
                f'<div class="flash flash-warning">'
                f"Skipped (already ingested): {file.filename}"
                f"</div>"
            )
        return HTMLResponse(
            f'<div class="flash flash-success">'
            f"Ingested {file.filename}: {result.chunk_count} chunks"
            f"</div>"
        )
    except Exception as exc:
        logger.error("ingest_upload_error", error=str(exc))
        return HTMLResponse(
            f'<div class="flash flash-error">'
            f"Error ingesting {file.filename}: {exc}"
            f"</div>"
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.get("/api/corpus/stats/json")
async def corpus_stats_json():
    """Corpus statistics as JSON."""
    from researchforge.rag.store import VectorStore

    store = VectorStore()
    try:
        chunk_count = store.count()
    except Exception:
        chunk_count = 0

    repo = await get_repo()
    try:
        cursor = await repo.db.execute("SELECT COUNT(*) FROM documents")
        row = await cursor.fetchone()
        doc_count = row[0] if row else 0
    except Exception:
        doc_count = 0

    return JSONResponse({"documents": doc_count, "chunks": chunk_count})
