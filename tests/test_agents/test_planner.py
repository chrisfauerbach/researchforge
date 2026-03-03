"""Tests for the planner agent."""

import json
from unittest.mock import AsyncMock, patch

from researchforge.agents.planner import run_planner
from researchforge.agents.state import make_initial_state

VALID_PLAN = {
    "sub_questions": [
        {
            "id": 1,
            "question": "What are the main types of RAG architectures?",
            "info_needs": ["Architecture types", "Comparison"],
            "priority": "high",
        },
        {
            "id": 2,
            "question": "How does chunking strategy affect retrieval quality?",
            "info_needs": ["Chunking methods", "Quality metrics"],
            "priority": "medium",
        },
    ],
    "overall_approach": "Survey existing literature on RAG systems",
    "expected_source_types": ["pdf", "markdown"],
}


class TestRunPlanner:
    async def test_produces_valid_plan(self):
        state = make_initial_state("What is RAG?", "pid-1")

        with patch("researchforge.agents.planner.ollama_chat", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "content": json.dumps(VALID_PLAN),
                "parsed": VALID_PLAN,
                "model": "deepseek-r1:14b",
                "input_tokens": 200,
                "output_tokens": 300,
                "duration_ms": 5000,
                "fallback_used": False,
            }

            updates = await run_planner(state)

        assert updates["research_plan"] == VALID_PLAN
        assert updates["status"] == "gathering"
        assert len(state["trace"]) == 1
        assert state["trace"][0]["agent"] == "planner"

    async def test_passes_question_to_ollama(self):
        state = make_initial_state("How does RAG work?", "pid-1")

        with patch("researchforge.agents.planner.ollama_chat", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "content": json.dumps(VALID_PLAN),
                "parsed": VALID_PLAN,
                "model": "deepseek-r1:14b",
                "input_tokens": 200,
                "output_tokens": 300,
                "duration_ms": 5000,
                "fallback_used": False,
            }

            await run_planner(state)

        call_kwargs = mock.call_args.kwargs
        assert "How does RAG work?" in call_kwargs["user_message"]
        assert call_kwargs["expect_json"] is True
        assert call_kwargs["agent_name"] == "planner"

    async def test_graceful_degradation_on_error(self):
        state = make_initial_state("Test question", "pid-1")

        with patch("researchforge.agents.planner.ollama_chat", new_callable=AsyncMock) as mock:
            mock.side_effect = Exception("Connection refused")

            updates = await run_planner(state)

        # Should produce a fallback plan with the original question
        plan = updates["research_plan"]
        assert len(plan["sub_questions"]) == 1
        assert plan["sub_questions"][0]["question"] == "Test question"
        assert "Planner error" in updates["errors"][0]
        assert updates["status"] == "gathering"
        # Trace should record the error
        assert state["trace"][0]["status"] == "error"

    async def test_records_fallback_in_trace(self):
        state = make_initial_state("Test", "pid-1")

        with patch("researchforge.agents.planner.ollama_chat", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "content": json.dumps(VALID_PLAN),
                "parsed": VALID_PLAN,
                "model": "qwen2.5:7b",
                "input_tokens": 100,
                "output_tokens": 150,
                "duration_ms": 3000,
                "fallback_used": True,
            }

            await run_planner(state)

        assert state["trace"][0]["fallback_used"] is True
