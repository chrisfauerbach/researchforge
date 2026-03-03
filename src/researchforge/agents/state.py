"""Pipeline state definition for the multi-agent research pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TypedDict


class TraceEntry(TypedDict, total=False):
    """A single entry in the pipeline execution trace."""

    timestamp: str
    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    duration_ms: int
    status: str  # "success", "retry", "fallback", "error"
    fallback_used: bool
    error: str


class PipelineState(TypedDict, total=False):
    """Typed state dict flowing through the LangGraph pipeline.

    Each agent node reads from state, does its work, and writes results back.
    There is no direct agent-to-agent messaging.
    """

    # Input
    research_question: str
    depth: str  # "standard" or "quick"

    # Planner output
    research_plan: dict | None  # Sub-questions, info needs, priorities

    # Gatherer output
    evidence: list[dict] | None  # Retrieved chunks with attributions
    gaps: list[str] | None  # Info gaps identified

    # Analyst output
    analysis: dict | None  # Findings, cross-refs, contradictions

    # Critic output
    critic_verdict: str | None  # "pass" or "revise"
    critic_issues: list[str] | None  # Issues found
    revision_count: int  # Track retry attempts

    # Writer output
    briefing: str | None  # Final markdown briefing

    # Pipeline metadata
    pipeline_id: str
    # "planning", "gathering", "analyzing", "critiquing", "writing", "completed", "failed"
    status: str
    errors: list[str]
    trace: list[TraceEntry]


def make_initial_state(
    research_question: str,
    pipeline_id: str,
    depth: str = "standard",
) -> PipelineState:
    """Create a fresh pipeline state for a new research run."""
    return PipelineState(
        research_question=research_question,
        depth=depth,
        research_plan=None,
        evidence=None,
        gaps=None,
        analysis=None,
        critic_verdict=None,
        critic_issues=None,
        revision_count=0,
        briefing=None,
        pipeline_id=pipeline_id,
        status="planning",
        errors=[],
        trace=[],
    )


def add_trace_entry(
    state: PipelineState,
    *,
    agent: str,
    model: str,
    duration_ms: int,
    status: str = "success",
    input_tokens: int = 0,
    output_tokens: int = 0,
    fallback_used: bool = False,
    error: str = "",
) -> TraceEntry:
    """Create and append a trace entry to the pipeline state.

    Returns the created trace entry.
    """
    entry = TraceEntry(
        timestamp=datetime.now(UTC).isoformat(),
        agent=agent,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_ms=duration_ms,
        status=status,
        fallback_used=fallback_used,
        error=error,
    )
    state["trace"].append(entry)
    return entry
