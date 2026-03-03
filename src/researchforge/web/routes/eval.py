"""Eval routes: dashboard, scores, human scoring API."""

from __future__ import annotations

import json

import structlog
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse

from researchforge.web.app import get_repo, get_templates

logger = structlog.get_logger()

router = APIRouter()


@router.get("/eval", response_class=HTMLResponse)
async def eval_dashboard(request: Request):
    """Eval dashboard page."""
    repo = await get_repo()

    # Load latest eval results from JSONL files
    retrieval_scores = _load_jsonl_scores("eval/results/retrieval_scores.jsonl")
    agent_scores = _load_jsonl_scores("eval/results/agent_scores.jsonl")
    e2e_scores = _load_jsonl_scores("eval/results/e2e_scores.jsonl")

    # Load recent briefings with human scores
    briefings = await repo.list_briefings(limit=20, status="completed")

    templates = get_templates()
    return templates.TemplateResponse(
        "eval.html",
        {
            "request": request,
            "active_page": "eval",
            "retrieval_scores": retrieval_scores[-10:],
            "agent_scores": agent_scores[-10:],
            "e2e_scores": e2e_scores[-10:],
            "briefings": briefings,
        },
    )


@router.post("/api/briefings/{briefing_id}/score", response_class=HTMLResponse)
async def score_briefing(
    briefing_id: str,
    score: int = Form(...),
    feedback: str = Form(""),
):
    """Record a human score for a briefing (thumbs up/down)."""
    repo = await get_repo()

    if score not in (1, -1):
        return HTMLResponse(
            '<span class="badge badge-error">Invalid score</span>',
            status_code=400,
        )

    await repo.set_human_score(briefing_id, score, feedback)

    badge_class = "badge-success" if score == 1 else "badge-error"
    label = "Thumbs up" if score == 1 else "Thumbs down"
    return HTMLResponse(
        f'<span class="badge {badge_class}">{label}</span>'
    )


@router.get("/api/eval/scores")
async def eval_scores_json():
    """Return all eval scores as JSON."""
    retrieval_scores = _load_jsonl_scores("eval/results/retrieval_scores.jsonl")
    agent_scores = _load_jsonl_scores("eval/results/agent_scores.jsonl")
    e2e_scores = _load_jsonl_scores("eval/results/e2e_scores.jsonl")
    return JSONResponse({
        "retrieval": retrieval_scores[-10:],
        "agent": agent_scores[-10:],
        "e2e": e2e_scores[-10:],
    })


def _load_jsonl_scores(path: str) -> list[dict]:
    """Load scores from a JSONL results file."""
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return []
    results = []
    try:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    except Exception:
        pass
    return results
