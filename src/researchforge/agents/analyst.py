"""Analyst agent — synthesizes evidence into a structured analysis."""

from __future__ import annotations

import json
import time

import structlog

from researchforge.agents.ollama_client import ollama_chat
from researchforge.agents.prompts import load_prompt
from researchforge.agents.state import PipelineState, add_trace_entry
from researchforge.config import get_settings

logger = structlog.get_logger()


def _format_evidence_for_prompt(evidence: list[dict]) -> str:
    """Format gathered evidence into a numbered text block for the analyst."""
    parts = []
    source_idx = 1
    for group in evidence:
        sub_q = group.get("sub_question", "Unknown")
        chunks = group.get("chunks", [])
        assessment = group.get("assessment", {})

        parts.append(f"### Sub-question: {sub_q}")

        for chunk in chunks:
            source = chunk.get("source_path", "unknown")
            section = chunk.get("section_h1", "")
            header = f"[Source {source_idx}: {source}"
            if section:
                header += f" > {section}"
            header += "]"
            parts.append(f"{header}\n{chunk['text']}")
            source_idx += 1

        # Include assessment summary if available
        relevant = assessment.get("relevant_evidence", [])
        if relevant:
            key_points = []
            for ev in relevant:
                key_points.extend(ev.get("key_points", []))
            if key_points:
                parts.append("Key points from gatherer: " + "; ".join(key_points))

    return "\n\n---\n\n".join(parts)


async def run_analyst(state: PipelineState) -> dict:
    """Synthesize evidence into a structured analysis.

    If this is a revision (critic sent back issues), the analyst receives
    the previous analysis and critic feedback to improve upon.

    Returns a dict of state updates to merge into PipelineState.
    """
    settings = get_settings()
    model = settings.models.analyst
    question = state["research_question"]
    evidence = state.get("evidence", [])
    gaps = state.get("gaps", [])

    system_prompt = load_prompt("analyst")

    # Build user message
    evidence_text = _format_evidence_for_prompt(evidence)
    user_message = f"Research question: {question}\n\n"
    user_message += f"Evidence:\n\n{evidence_text}"

    if gaps:
        user_message += "\n\nIdentified gaps:\n" + "\n".join(f"- {g}" for g in gaps)

    # If this is a revision, include critic feedback
    critic_issues = state.get("critic_issues")
    previous_analysis = state.get("analysis")
    if critic_issues and previous_analysis:
        user_message += "\n\n--- REVISION REQUEST ---\n"
        user_message += "The previous analysis was reviewed and needs improvement.\n"
        user_message += "Issues to address:\n"
        for issue in critic_issues:
            if isinstance(issue, dict):
                user_message += f"- [{issue.get('type', 'issue')}] {issue.get('description', '')}"
                suggestion = issue.get("suggestion", "")
                if suggestion:
                    user_message += f" (Suggestion: {suggestion})"
                user_message += "\n"
            else:
                user_message += f"- {issue}\n"
        user_message += "\nPrevious analysis for reference:\n"
        user_message += json.dumps(previous_analysis, indent=2)

    logger.info(
        "analyst_start",
        model=model,
        evidence_groups=len(evidence),
        is_revision=bool(critic_issues),
        revision_count=state.get("revision_count", 0),
    )
    start = time.monotonic()

    try:
        result = await ollama_chat(
            model=model,
            system_prompt=system_prompt,
            user_message=user_message,
            expect_json=True,
            agent_name="analyst",
        )

        analysis = result["parsed"]
        duration_ms = int((time.monotonic() - start) * 1000)

        add_trace_entry(
            state,
            agent="analyst",
            model=result["model"],
            duration_ms=duration_ms,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            fallback_used=result["fallback_used"],
        )

        logger.info(
            "analyst_complete",
            model=result["model"],
            findings=len(analysis.get("findings", [])),
            duration_ms=duration_ms,
        )

        return {
            "analysis": analysis,
            "status": "critiquing",
        }

    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        add_trace_entry(
            state,
            agent="analyst",
            model=model,
            duration_ms=duration_ms,
            status="error",
            error=str(exc),
        )
        logger.error("analyst_failed", error=str(exc))

        # Graceful degradation: produce a minimal analysis
        return {
            "analysis": {
                "findings": [
                    {
                        "finding": "Analysis could not be completed due to an error.",
                        "evidence_sources": [],
                        "confidence": "low",
                        "reasoning": f"Error: {exc}",
                    }
                ],
                "cross_references": [],
                "contradictions": [],
                "gaps": [f"Analysis failed: {exc}"],
                "overall_confidence": "low",
            },
            "status": "critiquing",
            "errors": [*state.get("errors", []), f"Analyst error: {exc}"],
        }
