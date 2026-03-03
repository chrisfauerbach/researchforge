"""Tests for pipeline state definition and helpers."""

from researchforge.agents.state import (
    add_trace_entry,
    make_initial_state,
)


class TestMakeInitialState:
    def test_creates_state_with_defaults(self):
        state = make_initial_state("What is RAG?", "pid-123")
        assert state["research_question"] == "What is RAG?"
        assert state["pipeline_id"] == "pid-123"
        assert state["depth"] == "standard"
        assert state["status"] == "planning"
        assert state["revision_count"] == 0
        assert state["errors"] == []
        assert state["trace"] == []

    def test_none_fields_start_empty(self):
        state = make_initial_state("test", "pid-1")
        assert state["research_plan"] is None
        assert state["evidence"] is None
        assert state["gaps"] is None
        assert state["analysis"] is None
        assert state["critic_verdict"] is None
        assert state["critic_issues"] is None
        assert state["briefing"] is None

    def test_custom_depth(self):
        state = make_initial_state("test", "pid-1", depth="quick")
        assert state["depth"] == "quick"


class TestAddTraceEntry:
    def test_appends_entry(self):
        state = make_initial_state("test", "pid-1")
        entry = add_trace_entry(
            state, agent="planner", model="deepseek-r1:14b", duration_ms=5000
        )
        assert len(state["trace"]) == 1
        assert state["trace"][0] is entry

    def test_entry_fields(self):
        state = make_initial_state("test", "pid-1")
        entry = add_trace_entry(
            state,
            agent="analyst",
            model="qwen2.5:14b",
            duration_ms=12000,
            input_tokens=500,
            output_tokens=300,
            status="success",
        )
        assert entry["agent"] == "analyst"
        assert entry["model"] == "qwen2.5:14b"
        assert entry["duration_ms"] == 12000
        assert entry["input_tokens"] == 500
        assert entry["output_tokens"] == 300
        assert entry["status"] == "success"
        assert entry["fallback_used"] is False
        assert "timestamp" in entry

    def test_multiple_entries_accumulate(self):
        state = make_initial_state("test", "pid-1")
        add_trace_entry(state, agent="planner", model="m1", duration_ms=100)
        add_trace_entry(state, agent="gatherer", model="m2", duration_ms=200)
        add_trace_entry(state, agent="analyst", model="m3", duration_ms=300)
        assert len(state["trace"]) == 3
        assert [e["agent"] for e in state["trace"]] == [
            "planner",
            "gatherer",
            "analyst",
        ]

    def test_fallback_entry(self):
        state = make_initial_state("test", "pid-1")
        entry = add_trace_entry(
            state,
            agent="analyst",
            model="qwen2.5:7b",
            duration_ms=8000,
            status="fallback",
            fallback_used=True,
        )
        assert entry["fallback_used"] is True
        assert entry["status"] == "fallback"

    def test_error_entry(self):
        state = make_initial_state("test", "pid-1")
        entry = add_trace_entry(
            state,
            agent="planner",
            model="deepseek-r1:14b",
            duration_ms=0,
            status="error",
            error="Connection refused",
        )
        assert entry["status"] == "error"
        assert entry["error"] == "Connection refused"
