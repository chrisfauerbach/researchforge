"""Tests for the LangGraph pipeline."""

from langgraph.graph import END, START, StateGraph

from researchforge.agents.graph import (
    build_graph,
    should_revise,
    should_skip_critic,
)
from researchforge.agents.state import PipelineState, make_initial_state

# --- Unit tests for conditional edges ---


class TestShouldRevise:
    def test_pass_routes_to_writer(self):
        state = make_initial_state("test", "pid-1")
        state["critic_verdict"] = "pass"
        assert should_revise(state) == "writer"

    def test_revise_routes_to_analyst(self):
        state = make_initial_state("test", "pid-1")
        state["critic_verdict"] = "revise"
        state["revision_count"] = 0
        assert should_revise(state) == "analyst"

    def test_max_retries_routes_to_writer(self):
        state = make_initial_state("test", "pid-1")
        state["critic_verdict"] = "revise"
        state["revision_count"] = 2  # Max retries reached
        assert should_revise(state) == "writer"

    def test_revision_count_1_routes_to_analyst(self):
        state = make_initial_state("test", "pid-1")
        state["critic_verdict"] = "revise"
        state["revision_count"] = 1
        assert should_revise(state) == "analyst"


class TestShouldSkipCritic:
    def test_standard_routes_to_critic(self):
        state = make_initial_state("test", "pid-1", depth="standard")
        assert should_skip_critic(state) == "critic"

    def test_quick_routes_to_writer(self):
        state = make_initial_state("test", "pid-1", depth="quick")
        assert should_skip_critic(state) == "writer"


# --- Graph topology tests ---


class TestBuildGraph:
    def test_graph_has_all_nodes(self):
        graph = build_graph()
        node_names = set(graph.nodes.keys())
        assert "planner" in node_names
        assert "gatherer" in node_names
        assert "analyst" in node_names
        assert "critic" in node_names
        assert "writer" in node_names

    def test_graph_compiles(self):
        graph = build_graph()
        compiled = graph.compile()
        assert compiled is not None


# --- Integration tests with mock agent functions injected directly ---

MOCK_PLAN = {
    "sub_questions": [
        {"id": 1, "question": "What is X?", "info_needs": ["definition"], "priority": "high"}
    ],
    "overall_approach": "Survey",
    "expected_source_types": ["pdf"],
}

MOCK_EVIDENCE = [
    {
        "sub_question_id": 1,
        "sub_question": "What is X?",
        "chunks": [{"text": "X is a thing.", "source_path": "doc.pdf", "section_h1": "Intro"}],
        "assessment": {"relevant_evidence": [], "gaps": [], "sufficiency": "sufficient"},
    }
]

MOCK_ANALYSIS = {
    "findings": [
        {
            "finding": "X is well-defined.",
            "evidence_sources": [1],
            "confidence": "high",
            "reasoning": "Source says so.",
        }
    ],
    "cross_references": [],
    "contradictions": [],
    "gaps": [],
    "overall_confidence": "high",
}

MOCK_BRIEFING = "# Research Briefing\n\nX is well-defined."


def _build_test_graph(
    planner_fn=None,
    gatherer_fn=None,
    analyst_fn=None,
    critic_fn=None,
    writer_fn=None,
):
    """Build a test graph with injectable mock functions."""

    def default_planner(state):
        return {"research_plan": MOCK_PLAN, "status": "gathering"}

    def default_gatherer(state):
        return {"evidence": MOCK_EVIDENCE, "gaps": None, "status": "analyzing"}

    def default_analyst(state):
        return {"analysis": MOCK_ANALYSIS, "status": "critiquing"}

    def default_critic(state):
        return {"critic_verdict": "pass", "critic_issues": None, "status": "writing"}

    def default_writer(state):
        return {"briefing": MOCK_BRIEFING, "status": "completed"}

    graph = StateGraph(PipelineState)
    graph.add_node("planner", planner_fn or default_planner)
    graph.add_node("gatherer", gatherer_fn or default_gatherer)
    graph.add_node("analyst", analyst_fn or default_analyst)
    graph.add_node("critic", critic_fn or default_critic)
    graph.add_node("writer", writer_fn or default_writer)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "gatherer")
    graph.add_edge("gatherer", "analyst")
    graph.add_conditional_edges("analyst", should_skip_critic)
    graph.add_conditional_edges("critic", should_revise)
    graph.add_edge("writer", END)

    return graph


class TestRunPipelineStandard:
    async def test_full_pipeline_pass(self):
        """Standard pipeline: planner→gatherer→analyst→critic(pass)→writer."""
        graph = _build_test_graph()
        compiled = graph.compile()

        initial = make_initial_state("What is X?", "test-pid", depth="standard")
        result = await compiled.ainvoke(initial)

        assert result["research_plan"] == MOCK_PLAN
        assert result["evidence"] == MOCK_EVIDENCE
        assert result["analysis"] == MOCK_ANALYSIS
        assert result["critic_verdict"] == "pass"
        assert result["briefing"] == MOCK_BRIEFING
        assert result["status"] == "completed"


class TestRunPipelineQuick:
    async def test_quick_pipeline_skips_critic(self):
        """Quick pipeline: planner→gatherer→analyst→writer (no critic)."""
        critic_called = {"value": False}

        def tracking_critic(state):
            critic_called["value"] = True
            return {"critic_verdict": "pass", "critic_issues": None, "status": "writing"}

        graph = _build_test_graph(critic_fn=tracking_critic)
        compiled = graph.compile()

        initial = make_initial_state("What is X?", "test-pid", depth="quick")
        result = await compiled.ainvoke(initial)

        assert result["briefing"] == MOCK_BRIEFING
        assert result["status"] == "completed"
        assert critic_called["value"] is False


class TestCriticRetryLoop:
    async def test_critic_revise_triggers_analyst_rerun(self):
        """Critic says 'revise' → analyst re-runs → critic passes on second try."""
        call_count = {"analyst": 0, "critic": 0}

        def counting_analyst(state):
            call_count["analyst"] += 1
            return {"analysis": MOCK_ANALYSIS, "status": "critiquing"}

        def counting_critic(state):
            call_count["critic"] += 1
            if call_count["critic"] == 1:
                return {
                    "critic_verdict": "revise",
                    "critic_issues": [{"type": "issue", "description": "Fix it"}],
                    "revision_count": state.get("revision_count", 0) + 1,
                    "status": "analyzing",
                }
            return {"critic_verdict": "pass", "critic_issues": None, "status": "writing"}

        graph = _build_test_graph(
            analyst_fn=counting_analyst,
            critic_fn=counting_critic,
        )
        compiled = graph.compile()

        initial = make_initial_state("What is X?", "test-pid", depth="standard")
        result = await compiled.ainvoke(initial)

        assert call_count["analyst"] == 2
        assert call_count["critic"] == 2
        assert result["briefing"] == MOCK_BRIEFING
        assert result["status"] == "completed"

    async def test_max_retries_stops_loop(self):
        """Critic always says 'revise' → after max retries, proceeds to writer."""
        call_count = {"analyst": 0, "critic": 0}

        def counting_analyst(state):
            call_count["analyst"] += 1
            return {"analysis": MOCK_ANALYSIS, "status": "critiquing"}

        def always_revise_critic(state):
            call_count["critic"] += 1
            return {
                "critic_verdict": "revise",
                "critic_issues": [{"type": "issue", "description": "Still bad"}],
                "revision_count": state.get("revision_count", 0) + 1,
                "status": "analyzing",
            }

        graph = _build_test_graph(
            analyst_fn=counting_analyst,
            critic_fn=always_revise_critic,
        )
        compiled = graph.compile()

        initial = make_initial_state("What is X?", "test-pid", depth="standard")
        result = await compiled.ainvoke(initial)

        # Flow: analyst(1)→critic(1,revise,rev=1)→analyst(2)→critic(2,revise,rev=2)→writer
        assert call_count["analyst"] == 2
        assert call_count["critic"] == 2
        assert result["briefing"] == MOCK_BRIEFING
        assert result["status"] == "completed"
        assert result["revision_count"] == 2

    async def test_pipeline_preserves_errors(self):
        """Errors from upstream agents are preserved through the pipeline."""

        def error_planner(state):
            return {
                "research_plan": MOCK_PLAN,
                "status": "gathering",
                "errors": ["Planner fallback used"],
            }

        graph = _build_test_graph(planner_fn=error_planner)
        compiled = graph.compile()

        initial = make_initial_state("What is X?", "test-pid", depth="standard")
        result = await compiled.ainvoke(initial)

        assert "Planner fallback used" in result["errors"]
        assert result["briefing"] == MOCK_BRIEFING
