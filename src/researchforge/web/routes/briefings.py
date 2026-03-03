"""Briefings routes: list and view research briefings."""

from __future__ import annotations

import json

import markdown
import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from researchforge.web.app import get_repo, get_templates

logger = structlog.get_logger()

router = APIRouter()


@router.get("/briefings", response_class=HTMLResponse)
async def briefings_page(request: Request):
    """Briefings list page."""
    repo = await get_repo()
    briefings = await repo.list_briefings(limit=50)
    templates = get_templates()
    return templates.TemplateResponse(
        "briefings.html",
        {
            "request": request,
            "active_page": "briefings",
            "briefings": briefings,
        },
    )


@router.get("/briefings/{briefing_id}", response_class=HTMLResponse)
async def briefing_detail_page(request: Request, briefing_id: str):
    """Single briefing viewer page with rendered Markdown."""
    repo = await get_repo()
    briefing = await repo.get_briefing(briefing_id)

    if not briefing:
        return HTMLResponse("<h2>Briefing not found</h2>", status_code=404)

    # Render Markdown to HTML
    briefing_html = ""
    if briefing.get("briefing_markdown"):
        briefing_html = markdown.markdown(
            briefing["briefing_markdown"],
            extensions=["fenced_code", "tables", "toc"],
        )

    # Parse pipeline trace
    trace = []
    if briefing.get("pipeline_trace"):
        try:
            trace = json.loads(briefing["pipeline_trace"])
        except (json.JSONDecodeError, TypeError):
            pass

    templates = get_templates()
    return templates.TemplateResponse(
        "briefing_detail.html",
        {
            "request": request,
            "active_page": "briefings",
            "briefing": briefing,
            "briefing_html": briefing_html,
            "trace": trace,
        },
    )


@router.get("/api/briefings")
async def list_briefings_api(limit: int = 10, status: str = "all"):
    """List briefings as JSON."""
    repo = await get_repo()
    briefings = await repo.list_briefings(limit=limit, status=status)
    return JSONResponse(briefings)


@router.get("/api/briefings/{briefing_id}")
async def get_briefing_api(briefing_id: str):
    """Get a single briefing as JSON."""
    repo = await get_repo()
    briefing = await repo.get_briefing(briefing_id)
    if not briefing:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse(briefing)
