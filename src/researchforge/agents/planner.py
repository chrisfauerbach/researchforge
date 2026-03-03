"""Planner agent — decomposes a research question into a structured plan."""

from __future__ import annotations

import time

import structlog

from researchforge.agents.ollama_client import ollama_chat
from researchforge.agents.prompts import load_prompt
from researchforge.agents.state import PipelineState, add_trace_entry
from researchforge.config import get_settings

logger = structlog.get_logger()


async def run_planner(state: PipelineState) -> dict:
    """Decompose the research question into sub-questions and a research plan.

    Returns a dict of state updates to merge into PipelineState.
    """
    settings = get_settings()
    model = settings.models.planner
    question = state["research_question"]

    system_prompt = load_prompt("planner")
    user_message = f"Research question: {question}"

    logger.info("planner_start", model=model, question=question)
    start = time.monotonic()

    try:
        result = await ollama_chat(
            model=model,
            system_prompt=system_prompt,
            user_message=user_message,
            expect_json=True,
            agent_name="planner",
        )

        plan = result["parsed"]
        duration_ms = int((time.monotonic() - start) * 1000)

        add_trace_entry(
            state,
            agent="planner",
            model=result["model"],
            duration_ms=duration_ms,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            fallback_used=result["fallback_used"],
        )

        logger.info(
            "planner_complete",
            model=result["model"],
            sub_questions=len(plan.get("sub_questions", [])),
            duration_ms=duration_ms,
        )

        return {
            "research_plan": plan,
            "status": "gathering",
        }

    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        add_trace_entry(
            state,
            agent="planner",
            model=model,
            duration_ms=duration_ms,
            status="error",
            error=str(exc),
        )
        logger.error("planner_failed", error=str(exc))

        # Graceful degradation: create a minimal plan with just the original question
        return {
            "research_plan": {
                "sub_questions": [
                    {
                        "id": 1,
                        "question": question,
                        "info_needs": ["General information"],
                        "priority": "high",
                    }
                ],
                "overall_approach": "Direct search (planner failed)",
                "expected_source_types": ["pdf", "markdown", "txt"],
            },
            "status": "gathering",
            "errors": [*state.get("errors", []), f"Planner error: {exc}"],
        }
