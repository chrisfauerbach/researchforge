"""Writer agent — produces the final Markdown research briefing."""

from __future__ import annotations

import json
import time

import structlog

from researchforge.agents.ollama_client import ollama_chat
from researchforge.agents.prompts import load_prompt
from researchforge.agents.state import PipelineState, add_trace_entry
from researchforge.config import get_settings

logger = structlog.get_logger()


def _format_sources_list(evidence: list[dict]) -> str:
    """Build a numbered source list for the writer."""
    sources = []
    idx = 1
    for group in evidence:
        for chunk in group.get("chunks", []):
            source = chunk.get("source_path", "unknown")
            section = chunk.get("section_h1", "")
            entry = f"Source {idx}: {source}"
            if section:
                entry += f" > {section}"
            sources.append(entry)
            idx += 1
    return "\n".join(sources)


async def run_writer(state: PipelineState) -> dict:
    """Produce the final Markdown briefing from the reviewed analysis.

    Returns a dict of state updates including the final briefing.
    """
    settings = get_settings()
    model = settings.models.writer
    question = state["research_question"]
    analysis = state.get("analysis", {})
    evidence = state.get("evidence", [])
    critic_issues = state.get("critic_issues")
    errors = state.get("errors", [])

    system_prompt = load_prompt("writer")

    sources_list = _format_sources_list(evidence)
    user_message = (
        f"Research question: {question}\n\n"
        f"Analysis:\n{json.dumps(analysis, indent=2)}\n\n"
        f"Available sources:\n{sources_list}"
    )

    # If there are unresolved critic issues, tell the writer to include them as caveats
    if critic_issues:
        user_message += "\n\nUnresolved reviewer concerns (include in Caveats section):\n"
        for issue in critic_issues:
            if isinstance(issue, dict):
                user_message += f"- {issue.get('description', str(issue))}\n"
            else:
                user_message += f"- {issue}\n"

    # If there were pipeline errors, note them
    if errors:
        user_message += "\n\nPipeline notes (include in Caveats if relevant):\n"
        for err in errors:
            user_message += f"- {err}\n"

    logger.info("writer_start", model=model)
    start = time.monotonic()

    try:
        result = await ollama_chat(
            model=model,
            system_prompt=system_prompt,
            user_message=user_message,
            expect_json=False,
            agent_name="writer",
        )

        briefing = result["content"]
        duration_ms = int((time.monotonic() - start) * 1000)

        add_trace_entry(
            state,
            agent="writer",
            model=result["model"],
            duration_ms=duration_ms,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            fallback_used=result["fallback_used"],
        )

        logger.info(
            "writer_complete",
            model=result["model"],
            briefing_len=len(briefing),
            duration_ms=duration_ms,
        )

        return {
            "briefing": briefing,
            "status": "completed",
        }

    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        add_trace_entry(
            state,
            agent="writer",
            model=model,
            duration_ms=duration_ms,
            status="error",
            error=str(exc),
        )
        logger.error("writer_failed", error=str(exc))

        # Best-effort briefing from raw analysis
        findings = analysis.get("findings", [])
        fallback_briefing = f"# Research Briefing: {question}\n\n"
        fallback_briefing += "## Executive Summary\n\n"
        fallback_briefing += (
            "This briefing was generated from raw analysis data "
            "due to a writer agent failure.\n\n"
        )
        fallback_briefing += "## Findings\n\n"
        for f in findings:
            fallback_briefing += f"- {f.get('finding', 'Unknown')}\n"
        fallback_briefing += (
            f"\n## Caveats\n\n- Writer agent failed: {exc}\n"
        )

        return {
            "briefing": fallback_briefing,
            "status": "completed",
            "errors": [*errors, f"Writer error: {exc}"],
        }
