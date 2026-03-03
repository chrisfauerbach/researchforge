"""Gatherer agent — retrieves evidence for each sub-question in the research plan."""

from __future__ import annotations

import asyncio
import time

import structlog

from researchforge.agents.ollama_client import ollama_chat
from researchforge.agents.prompts import load_prompt
from researchforge.agents.state import PipelineState, add_trace_entry
from researchforge.config import WebSearchConfig, get_settings
from researchforge.rag.retriever import retrieve
from researchforge.rag.store import VectorStore
from researchforge.rag.web_search import web_search_for_question

logger = structlog.get_logger()


def _format_chunks_for_prompt(chunks: list[dict]) -> str:
    """Format retrieved chunks into numbered text for the gatherer prompt."""
    parts = []
    for i, chunk in enumerate(chunks):
        source = chunk.get("source_path", "unknown")
        section = chunk.get("section_h1", "")
        header = f"[Chunk {i}: {source}"
        if section:
            header += f" > {section}"
        header += "]"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


async def _gather_for_sub_question(
    sub_q: dict,
    store: VectorStore,
    system_prompt: str,
    model: str,
    web_search_cfg: WebSearchConfig | None = None,
) -> dict:
    """Retrieve evidence and assess it for a single sub-question."""
    question = sub_q["question"]

    # Retrieve relevant chunks from local corpus
    chunks = await retrieve(question, store)

    # Optionally supplement with web search
    web_chunks: list[dict] = []
    if web_search_cfg is not None:
        should_search = (
            web_search_cfg.mode == "always"
            or (web_search_cfg.mode == "auto" and not chunks)
        )
        if should_search:
            web_chunks = await web_search_for_question(question, cfg=web_search_cfg)

    all_chunks = chunks + web_chunks

    if not all_chunks:
        return {
            "sub_question_id": sub_q["id"],
            "sub_question": question,
            "chunks": [],
            "assessment": {
                "relevant_evidence": [],
                "gaps": [f"No documents found for: {question}"],
                "sufficiency": "insufficient",
            },
        }

    # Ask the gatherer model to assess relevance
    chunks_text = _format_chunks_for_prompt(all_chunks)
    user_message = (
        f"Sub-question: {question}\n\n"
        f"Retrieved chunks:\n\n{chunks_text}"
    )

    result = await ollama_chat(
        model=model,
        system_prompt=system_prompt,
        user_message=user_message,
        expect_json=True,
        agent_name="gatherer",
    )

    return {
        "sub_question_id": sub_q["id"],
        "sub_question": question,
        "chunks": all_chunks,
        "assessment": result["parsed"],
        "model_result": result,
    }


async def run_gatherer(state: PipelineState) -> dict:
    """Retrieve and assess evidence for all sub-questions in the research plan.

    Returns a dict of state updates to merge into PipelineState.
    """
    settings = get_settings()
    model = settings.models.gatherer
    plan = state["research_plan"]
    sub_questions = plan.get("sub_questions", [])

    system_prompt = load_prompt("gatherer")
    store = VectorStore()

    web_search_cfg = (
        None if settings.web_search.mode == "disabled" else settings.web_search
    )

    logger.info("gatherer_start", model=model, sub_questions=len(sub_questions))
    start = time.monotonic()

    try:
        # Retrieve evidence for all sub-questions in parallel
        tasks = [
            _gather_for_sub_question(sq, store, system_prompt, model, web_search_cfg)
            for sq in sub_questions
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect evidence and gaps
        all_evidence = []
        all_gaps = []
        all_errors = []
        total_input_tokens = 0
        total_output_tokens = 0

        for r in results:
            if isinstance(r, Exception):
                all_gaps.append(f"Retrieval failed: {r}")
                all_errors.append(f"Gatherer error: {r}")
                continue
            all_evidence.append({
                "sub_question_id": r["sub_question_id"],
                "sub_question": r["sub_question"],
                "chunks": r["chunks"],
                "assessment": r["assessment"],
            })
            if "model_result" in r:
                total_input_tokens += r["model_result"]["input_tokens"]
                total_output_tokens += r["model_result"]["output_tokens"]
            assessment_gaps = r.get("assessment", {}).get("gaps", [])
            all_gaps.extend(assessment_gaps)

        duration_ms = int((time.monotonic() - start) * 1000)

        add_trace_entry(
            state,
            agent="gatherer",
            model=model,
            duration_ms=duration_ms,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

        logger.info(
            "gatherer_complete",
            model=model,
            evidence_groups=len(all_evidence),
            gaps=len(all_gaps),
            duration_ms=duration_ms,
        )

        updates: dict = {
            "evidence": all_evidence,
            "gaps": all_gaps if all_gaps else None,
            "status": "analyzing",
        }
        if all_errors:
            updates["errors"] = [*state.get("errors", []), *all_errors]
        return updates

    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        add_trace_entry(
            state,
            agent="gatherer",
            model=model,
            duration_ms=duration_ms,
            status="error",
            error=str(exc),
        )
        logger.error("gatherer_failed", error=str(exc))

        return {
            "evidence": [],
            "gaps": [f"Gatherer error: {exc}"],
            "status": "analyzing",
            "errors": [*state.get("errors", []), f"Gatherer error: {exc}"],
        }
