"""MCP server exposing ResearchForge tools to AI clients."""

from __future__ import annotations

import asyncio
import json
import uuid

import structlog
from mcp.server.fastmcp import FastMCP

from researchforge.config import get_settings

logger = structlog.get_logger()

mcp = FastMCP("researchforge")

# Track in-flight research jobs: {job_id: {status, question, result}}
_mcp_jobs: dict[str, dict] = {}


async def _get_repo():
    """Get or create a Repository instance."""
    from researchforge.db.repository import Repository

    settings = get_settings()
    repo = Repository(settings.storage.metadata_db_path)
    await repo.initialize()
    return repo


async def _run_research_job(job_id: str, topic: str, depth: str) -> None:
    """Background task that runs the research pipeline and feedback loop."""
    from researchforge.agents.graph import run_pipeline
    from researchforge.rag.feedback import maybe_ingest_briefing
    from researchforge.rag.store import VectorStore

    repo = await _get_repo()
    try:
        _mcp_jobs[job_id]["status"] = "running"

        await repo.insert_briefing(job_id, topic, status="running")

        final_state = await run_pipeline(topic, depth=depth, pipeline_id=job_id)

        briefing = final_state.get("briefing", "")
        status = final_state.get("status", "unknown")
        trace = final_state.get("trace", [])

        await repo.update_briefing(
            job_id,
            briefing_markdown=briefing,
            status="completed" if briefing else status,
            pipeline_trace=trace,
        )

        # Extract critic verdict from trace
        critic_verdict = None
        for entry in trace:
            if entry.get("agent") == "critic":
                critic_verdict = entry.get("verdict")
                break

        # Corpus feedback loop
        store = VectorStore()
        feedback = await maybe_ingest_briefing(
            job_id, repo, store, critic_verdict=critic_verdict
        )

        _mcp_jobs[job_id].update({
            "status": "completed",
            "result": {
                "briefing_id": job_id,
                "status": "completed",
                "feedback": feedback,
            },
        })

    except Exception as exc:
        logger.error("mcp_research_failed", job_id=job_id, error=str(exc))
        _mcp_jobs[job_id].update({
            "status": "failed",
            "result": {"error": str(exc)},
        })
        await repo.update_briefing(job_id, status="failed")
    finally:
        await repo.close()


@mcp.tool()
async def research(topic: str, depth: str = "standard") -> str:
    """Start a research pipeline on a topic. Returns a job_id for tracking progress.

    Args:
        topic: The research question or topic to investigate.
        depth: Pipeline depth — "standard" (with critic review) or "quick".
    """
    job_id = str(uuid.uuid4())
    _mcp_jobs[job_id] = {
        "status": "queued",
        "question": topic,
        "result": None,
    }
    asyncio.create_task(_run_research_job(job_id, topic, depth))
    return json.dumps({"job_id": job_id, "status": "queued"})


@mcp.tool()
async def query_corpus(query: str, limit: int = 5, source_only: bool = False) -> str:
    """Search the RAG corpus using hybrid retrieval (vector + BM25).

    Args:
        query: Natural language search query.
        limit: Maximum number of results to return.
        source_only: If true, exclude agent-generated content.
    """
    from researchforge.rag.retriever import retrieve
    from researchforge.rag.store import VectorStore

    store = VectorStore()
    results = await retrieve(query, store, top_k=limit, source_only=source_only)

    formatted = []
    for r in results:
        formatted.append({
            "text": r.get("text", ""),
            "source_path": r.get("source_path", ""),
            "source_type": r.get("source_type", ""),
            "content_type": r.get("content_type", ""),
            "section_h1": r.get("section_h1", ""),
        })
    return json.dumps(formatted, indent=2)


@mcp.tool()
async def ingest_document(file_path: str) -> str:
    """Ingest a document into the RAG corpus.

    Args:
        file_path: Absolute path to the file to ingest.
    """
    from researchforge.rag.ingest import ingest_file
    from researchforge.rag.store import VectorStore

    store = VectorStore()
    repo = await _get_repo()
    try:
        result = await ingest_file(file_path, store, repo)
        return json.dumps({
            "document_id": result.document_id,
            "chunk_count": result.chunk_count,
            "source_type": result.source_type,
            "skipped": result.skipped,
            "reason": result.reason,
        })
    finally:
        await repo.close()


@mcp.tool()
async def list_briefings(limit: int = 10, status: str = "all") -> str:
    """List recent research briefings.

    Args:
        limit: Maximum number of briefings to return.
        status: Filter by status ("all", "completed", "running", "failed").
    """
    repo = await _get_repo()
    try:
        briefings = await repo.list_briefings(limit=limit, status=status)
        result = []
        for b in briefings:
            result.append({
                "briefing_id": b["briefing_id"],
                "research_question": b["research_question"],
                "status": b["status"],
                "started_at": b.get("started_at"),
                "quality_score": b.get("quality_score"),
            })
        return json.dumps(result, indent=2)
    finally:
        await repo.close()


@mcp.tool()
async def get_briefing(briefing_id: str) -> str:
    """Get the full content of a research briefing.

    Args:
        briefing_id: The ID of the briefing to retrieve.
    """
    repo = await _get_repo()
    try:
        briefing = await repo.get_briefing(briefing_id)
        if briefing is None:
            return json.dumps({"error": "Briefing not found"})
        return json.dumps({
            "briefing_id": briefing["briefing_id"],
            "research_question": briefing["research_question"],
            "status": briefing["status"],
            "briefing_markdown": briefing.get("briefing_markdown", ""),
            "quality_score": briefing.get("quality_score"),
            "started_at": briefing.get("started_at"),
            "completed_at": briefing.get("completed_at"),
        }, indent=2)
    finally:
        await repo.close()


@mcp.tool()
async def get_status(job_id: str) -> str:
    """Check the status of a running research job.

    Args:
        job_id: The job ID returned by the research tool.
    """
    job = _mcp_jobs.get(job_id)
    if job is None:
        return json.dumps({"error": "Job not found", "job_id": job_id})
    return json.dumps({
        "job_id": job_id,
        "status": job["status"],
        "question": job.get("question", ""),
        "result": job.get("result"),
    }, indent=2)
