"""Research routes: start pipeline, SSE streaming, list pipelines."""

from __future__ import annotations

import asyncio
import json
import uuid

import structlog
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from researchforge.web.app import get_repo, get_templates
from researchforge.web.events import get_event_bus

logger = structlog.get_logger()

router = APIRouter()

# Track active pipelines: {job_id: {question, status, depth, task}}
_active_pipelines: dict[str, dict] = {}


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    templates = get_templates()
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "active_page": "research"},
    )


@router.post("/api/research", response_class=HTMLResponse)
async def start_research(
    request: Request,
    question: str = Form(...),
    depth: str = Form("standard"),
):
    """Start a research pipeline. Returns a pipeline card partial."""
    job_id = str(uuid.uuid4())

    _active_pipelines[job_id] = {
        "question": question,
        "status": "running",
        "depth": depth,
        "completed_stages": [],
        "current_stage": "planner",
        "error_stages": [],
    }

    # Launch pipeline in background
    task = asyncio.create_task(_run_pipeline_with_events(job_id, question, depth))
    _active_pipelines[job_id]["task"] = task

    templates = get_templates()
    return templates.TemplateResponse(
        "partials/pipeline_card.html",
        {
            "request": request,
            "job_id": job_id,
            "question": question,
            "status": "running",
            "completed_stages": [],
            "current_stage": "planner",
            "error_stages": [],
            "briefing_id": None,
        },
    )


@router.get("/api/pipelines", response_class=HTMLResponse)
async def list_pipelines(request: Request):
    """List active and recent pipelines as HTML cards."""
    repo = await get_repo()
    briefings = await repo.list_briefings(limit=10)
    templates = get_templates()

    cards_html = ""
    # Show active pipelines first
    for job_id, info in reversed(list(_active_pipelines.items())):
        card = templates.TemplateResponse(
            "partials/pipeline_card.html",
            {
                "request": request,
                "job_id": job_id,
                "question": info["question"],
                "status": info["status"],
                "completed_stages": info.get("completed_stages", []),
                "current_stage": info.get("current_stage"),
                "error_stages": info.get("error_stages", []),
                "briefing_id": job_id if info["status"] == "completed" else None,
            },
        )
        cards_html += card.body.decode()

    # Show completed briefings from DB (not already shown)
    active_ids = set(_active_pipelines.keys())
    for b in briefings:
        if b["briefing_id"] in active_ids:
            continue
        card = templates.TemplateResponse(
            "partials/pipeline_card.html",
            {
                "request": request,
                "job_id": b["briefing_id"],
                "question": b["research_question"],
                "status": b["status"],
                "completed_stages": (
                    ["planner", "gatherer", "analyst", "critic", "writer"]
                    if b["status"] == "completed"
                    else []
                ),
                "current_stage": None,
                "error_stages": [],
                "briefing_id": (
                    b["briefing_id"] if b["status"] == "completed" else None
                ),
            },
        )
        cards_html += card.body.decode()

    if not cards_html:
        cards_html = '<p class="text-muted text-sm">No pipelines yet.</p>'

    return HTMLResponse(cards_html)


@router.get("/api/pipelines/{job_id}/card", response_class=HTMLResponse)
async def pipeline_card(request: Request, job_id: str):
    """Return the current pipeline card HTML for a given job."""
    templates = get_templates()
    info = _active_pipelines.get(job_id)
    if info:
        return templates.TemplateResponse(
            "partials/pipeline_card.html",
            {
                "request": request,
                "job_id": job_id,
                "question": info["question"],
                "status": info["status"],
                "completed_stages": info.get("completed_stages", []),
                "current_stage": info.get("current_stage"),
                "error_stages": info.get("error_stages", []),
                "briefing_id": job_id if info["status"] == "completed" else None,
            },
        )
    # Fall back to DB
    repo = await get_repo()
    briefing = await repo.get_briefing(job_id)
    if briefing:
        return templates.TemplateResponse(
            "partials/pipeline_card.html",
            {
                "request": request,
                "job_id": job_id,
                "question": briefing["research_question"],
                "status": briefing["status"],
                "completed_stages": (
                    ["planner", "gatherer", "analyst", "critic", "writer"]
                    if briefing["status"] == "completed"
                    else []
                ),
                "current_stage": None,
                "error_stages": [],
                "briefing_id": (
                    job_id if briefing["status"] == "completed" else None
                ),
            },
        )
    return HTMLResponse('<div class="card">Pipeline not found.</div>')


@router.get("/api/pipelines/{job_id}/stream")
async def stream_pipeline(job_id: str):
    """SSE stream for pipeline progress events."""
    event_bus = get_event_bus()
    queue = event_bus.subscribe(job_id)

    async def event_generator():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=60.0)
                except TimeoutError:
                    # Send keepalive
                    yield ": keepalive\n\n"
                    continue

                yield event_bus.format_sse(event)

                # Also send a stage_update event with the card HTML for HTMX
                if event.get("type") in (
                    "stage_start", "stage_complete", "error", "pipeline_complete"
                ):
                    update_data = json.dumps({
                        "type": event.get("type"),
                        "agent": event.get("agent", ""),
                        "status": event.get("status", ""),
                    })
                    yield f"event: stage_update\ndata: {update_data}\n\n"

                if event.get("type") == "pipeline_complete":
                    break
        finally:
            event_bus.unsubscribe(job_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _run_pipeline_with_events(
    job_id: str, question: str, depth: str
) -> None:
    """Run the pipeline and publish events to the event bus."""
    from researchforge.agents.graph import astream_pipeline

    event_bus = get_event_bus()
    repo = await get_repo()

    try:
        await repo.insert_briefing(job_id, question, status="running")

        async for event in astream_pipeline(question, depth=depth, pipeline_id=job_id):
            agent = event.get("agent", "")
            status = event.get("status", "")
            state_update = event.get("state_update", {})

            await event_bus.publish_stage_complete(
                job_id, agent, 0, status=status
            )

            # Update active pipeline tracking
            if job_id in _active_pipelines:
                info = _active_pipelines[job_id]
                if agent not in info["completed_stages"]:
                    info["completed_stages"].append(agent)
                info["current_stage"] = agent

        # Pipeline finished — get final state
        info = _active_pipelines.get(job_id, {})
        info["status"] = "completed"

        # Update DB with final briefing
        # The pipeline already wrote to state, we need to fetch it
        # Since astream yields per-node updates, the last writer update has the briefing
        # We'll update the briefing from the last state_update
        briefing = state_update.get("briefing", "")
        trace = []

        await repo.update_briefing(
            job_id,
            briefing_markdown=briefing,
            status="completed",
            pipeline_trace=trace,
        )

        # Corpus feedback loop
        try:
            from researchforge.rag.feedback import maybe_ingest_briefing
            from researchforge.rag.store import VectorStore

            critic_verdict = state_update.get("critic_verdict")
            feedback_store = VectorStore()
            await maybe_ingest_briefing(
                job_id, repo, feedback_store, critic_verdict=critic_verdict
            )
        except Exception as fb_exc:
            logger.warning("feedback_loop_error", job_id=job_id, error=str(fb_exc))

        await event_bus.publish_complete(job_id, briefing_id=job_id)

    except Exception as exc:
        logger.error("pipeline_failed", job_id=job_id, error=str(exc))
        if job_id in _active_pipelines:
            _active_pipelines[job_id]["status"] = "failed"

        await repo.update_briefing(job_id, status="failed")
        await event_bus.publish_error(job_id, "pipeline", str(exc))
        await event_bus.publish_complete(job_id, briefing_id=job_id)
