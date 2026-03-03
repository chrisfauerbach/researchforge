"""Critic agent — reviews analysis for quality, accuracy, and completeness."""

from __future__ import annotations

import json
import time

import structlog

from researchforge.agents.ollama_client import ollama_chat
from researchforge.agents.prompts import load_prompt
from researchforge.agents.state import PipelineState, add_trace_entry
from researchforge.config import get_settings

logger = structlog.get_logger()


def _format_evidence_summary(evidence: list[dict]) -> str:
    """Create a concise evidence summary for the critic."""
    parts = []
    source_idx = 1
    for group in evidence:
        sub_q = group.get("sub_question", "Unknown")
        chunks = group.get("chunks", [])
        parts.append(f"Sub-question: {sub_q} ({len(chunks)} chunks)")
        for chunk in chunks:
            source = chunk.get("source_path", "unknown")
            text_preview = chunk.get("text", "")[:200]
            parts.append(f"  [Source {source_idx}: {source}] {text_preview}...")
            source_idx += 1
    return "\n".join(parts)


async def run_critic(state: PipelineState) -> dict:
    """Review the analyst's analysis for quality issues.

    Returns a dict of state updates including critic_verdict and critic_issues.
    """
    settings = get_settings()
    model = settings.models.critic
    question = state["research_question"]
    analysis = state.get("analysis", {})
    evidence = state.get("evidence", [])

    system_prompt = load_prompt("critic")

    evidence_summary = _format_evidence_summary(evidence)
    user_message = (
        f"Research question: {question}\n\n"
        f"Evidence summary:\n{evidence_summary}\n\n"
        f"Analysis to review:\n{json.dumps(analysis, indent=2)}"
    )

    logger.info(
        "critic_start",
        model=model,
        revision_count=state.get("revision_count", 0),
    )
    start = time.monotonic()

    try:
        result = await ollama_chat(
            model=model,
            system_prompt=system_prompt,
            user_message=user_message,
            expect_json=True,
            agent_name="critic",
        )

        review = result["parsed"]
        verdict = review.get("verdict", "pass")
        issues = review.get("issues", [])
        duration_ms = int((time.monotonic() - start) * 1000)

        add_trace_entry(
            state,
            agent="critic",
            model=result["model"],
            duration_ms=duration_ms,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            fallback_used=result["fallback_used"],
        )

        logger.info(
            "critic_complete",
            model=result["model"],
            verdict=verdict,
            issues=len(issues),
            duration_ms=duration_ms,
        )

        updates: dict = {
            "critic_verdict": verdict,
            "critic_issues": issues if issues else None,
        }

        if verdict == "revise":
            updates["revision_count"] = state.get("revision_count", 0) + 1
            updates["status"] = "analyzing"  # Will be overridden by conditional edge
        else:
            updates["status"] = "writing"

        return updates

    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        add_trace_entry(
            state,
            agent="critic",
            model=model,
            duration_ms=duration_ms,
            status="error",
            error=str(exc),
        )
        logger.error("critic_failed", error=str(exc))

        # On critic failure, pass through to writer
        return {
            "critic_verdict": "pass",
            "critic_issues": None,
            "status": "writing",
            "errors": [*state.get("errors", []), f"Critic error: {exc}"],
        }
