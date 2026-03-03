"""LangGraph pipeline: Planner → Gatherer → Analyst → Critic → Writer."""

from __future__ import annotations

import uuid
from typing import Literal

import structlog
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from researchforge.agents.analyst import run_analyst
from researchforge.agents.critic import run_critic
from researchforge.agents.gatherer import run_gatherer
from researchforge.agents.planner import run_planner
from researchforge.agents.state import PipelineState, make_initial_state
from researchforge.agents.writer import run_writer
from researchforge.config import get_settings

logger = structlog.get_logger()


def should_revise(state: PipelineState) -> Literal["analyst", "writer"]:
    """Conditional edge: route from critic to analyst (revise) or writer (pass)."""
    max_retries = get_settings().pipeline.max_critic_retries

    if state.get("critic_verdict") == "pass":
        return "writer"
    if state.get("revision_count", 0) >= max_retries:
        logger.warning(
            "max_retries_reached",
            revision_count=state.get("revision_count", 0),
            max_retries=max_retries,
        )
        return "writer"
    return "analyst"


def should_skip_critic(state: PipelineState) -> Literal["critic", "writer"]:
    """Conditional edge: skip critic in 'quick' depth mode."""
    if state.get("depth") == "quick":
        return "writer"
    return "critic"


def build_graph() -> StateGraph:
    """Build the research pipeline StateGraph (uncompiled)."""
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("planner", run_planner)
    graph.add_node("gatherer", run_gatherer)
    graph.add_node("analyst", run_analyst)
    graph.add_node("critic", run_critic)
    graph.add_node("writer", run_writer)

    # Add edges
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "gatherer")
    graph.add_edge("gatherer", "analyst")

    # After analyst: check depth mode to decide if we go to critic or writer
    graph.add_conditional_edges("analyst", should_skip_critic)

    # After critic: check verdict to decide if we go back to analyst or to writer
    graph.add_conditional_edges("critic", should_revise)

    graph.add_edge("writer", END)

    return graph


async def run_pipeline(
    question: str,
    depth: str = "standard",
    pipeline_id: str | None = None,
) -> PipelineState:
    """Run the full research pipeline synchronously.

    Args:
        question: The research question.
        depth: "standard" (full pipeline with critic) or "quick" (skip critic).
        pipeline_id: Optional pipeline ID (generated if not provided).

    Returns:
        The final PipelineState with all results.
    """
    if pipeline_id is None:
        pipeline_id = str(uuid.uuid4())

    settings = get_settings()
    checkpoints_path = settings.storage.checkpoints_db_path

    initial_state = make_initial_state(question, pipeline_id, depth=depth)

    logger.info(
        "pipeline_start",
        pipeline_id=pipeline_id,
        question=question,
        depth=depth,
    )

    graph = build_graph()

    async with AsyncSqliteSaver.from_conn_string(checkpoints_path) as checkpointer:
        compiled = graph.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": pipeline_id}}
        final_state = await compiled.ainvoke(initial_state, config=config)

    logger.info(
        "pipeline_complete",
        pipeline_id=pipeline_id,
        status=final_state.get("status", "unknown"),
        trace_len=len(final_state.get("trace", [])),
    )

    return final_state


async def astream_pipeline(
    question: str,
    depth: str = "standard",
    pipeline_id: str | None = None,
):
    """Stream pipeline events as each agent completes.

    Yields dicts with agent name, status, and output summary.
    """
    if pipeline_id is None:
        pipeline_id = str(uuid.uuid4())

    settings = get_settings()
    checkpoints_path = settings.storage.checkpoints_db_path

    initial_state = make_initial_state(question, pipeline_id, depth=depth)

    graph = build_graph()

    async with AsyncSqliteSaver.from_conn_string(checkpoints_path) as checkpointer:
        compiled = graph.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": pipeline_id}}

        async for event in compiled.astream(initial_state, config=config):
            # Each event is a dict {node_name: state_update}
            for node_name, state_update in event.items():
                yield {
                    "pipeline_id": pipeline_id,
                    "agent": node_name,
                    "status": state_update.get("status", "unknown"),
                    "state_update": state_update,
                }
