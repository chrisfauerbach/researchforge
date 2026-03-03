"""Tests for the gatherer agent."""

import json
from unittest.mock import AsyncMock, patch

from researchforge.agents.gatherer import run_gatherer
from researchforge.agents.state import make_initial_state

SAMPLE_PLAN = {
    "sub_questions": [
        {
            "id": 1, "question": "What is RAG?",
            "info_needs": ["definition"], "priority": "high",
        },
        {
            "id": 2, "question": "How does chunking work?",
            "info_needs": ["methods"], "priority": "medium",
        },
    ],
    "overall_approach": "Survey literature",
    "expected_source_types": ["pdf"],
}

SAMPLE_CHUNKS = [
    {
        "chunk_id": "c1", "text": "RAG is retrieval augmented generation...",
        "source_path": "doc.pdf", "section_h1": "Intro",
    },
    {
        "chunk_id": "c2", "text": "Chunking divides documents into pieces...",
        "source_path": "doc.pdf", "section_h1": "Methods",
    },
]

GATHERER_ASSESSMENT = {
    "sub_question_id": 1,
    "sub_question": "What is RAG?",
    "relevant_evidence": [
        {
            "chunk_index": 0, "relevance": "high",
            "key_points": ["RAG combines retrieval with generation"],
        }
    ],
    "gaps": [],
    "sufficiency": "sufficient",
}


_P = "researchforge.agents.gatherer"


class TestRunGatherer:
    async def test_retrieves_evidence_for_sub_questions(self):
        state = make_initial_state("What is RAG?", "pid-1")
        state["research_plan"] = SAMPLE_PLAN
        state["status"] = "gathering"

        with (
            patch(f"{_P}.retrieve", new_callable=AsyncMock) as mock_retrieve,
            patch(f"{_P}.ollama_chat", new_callable=AsyncMock) as mock_chat,
            patch(f"{_P}.VectorStore"),
        ):
            mock_retrieve.return_value = SAMPLE_CHUNKS
            mock_chat.return_value = {
                "content": json.dumps(GATHERER_ASSESSMENT),
                "parsed": GATHERER_ASSESSMENT,
                "model": "qwen2.5:7b",
                "input_tokens": 100,
                "output_tokens": 80,
                "duration_ms": 2000,
                "fallback_used": False,
            }

            updates = await run_gatherer(state)

        assert updates["status"] == "analyzing"
        assert len(updates["evidence"]) == 2  # One per sub-question
        # Retriever called once per sub-question
        assert mock_retrieve.call_count == 2
        assert len(state["trace"]) == 1
        assert state["trace"][0]["agent"] == "gatherer"

    async def test_handles_empty_retrieval(self):
        state = make_initial_state("Obscure topic", "pid-1")
        state["research_plan"] = {
            "sub_questions": [
                {
                    "id": 1, "question": "What is xyzzy?",
                    "info_needs": ["anything"], "priority": "high",
                },
            ],
            "overall_approach": "Search",
            "expected_source_types": [],
        }

        with (
            patch(f"{_P}.retrieve", new_callable=AsyncMock) as mock_retrieve,
            patch(f"{_P}.VectorStore"),
        ):
            mock_retrieve.return_value = []

            updates = await run_gatherer(state)

        evidence = updates["evidence"]
        assert len(evidence) == 1
        assert evidence[0]["assessment"]["sufficiency"] == "insufficient"
        assert len(evidence[0]["assessment"]["gaps"]) > 0

    async def test_collects_gaps(self):
        state = make_initial_state("Test", "pid-1")
        state["research_plan"] = SAMPLE_PLAN

        assessment_with_gaps = {
            **GATHERER_ASSESSMENT,
            "gaps": ["Missing historical context"],
        }

        with (
            patch(f"{_P}.retrieve", new_callable=AsyncMock) as mock_retrieve,
            patch(f"{_P}.ollama_chat", new_callable=AsyncMock) as mock_chat,
            patch(f"{_P}.VectorStore"),
        ):
            mock_retrieve.return_value = SAMPLE_CHUNKS
            mock_chat.return_value = {
                "content": json.dumps(assessment_with_gaps),
                "parsed": assessment_with_gaps,
                "model": "qwen2.5:7b",
                "input_tokens": 100,
                "output_tokens": 80,
                "duration_ms": 2000,
                "fallback_used": False,
            }

            updates = await run_gatherer(state)

        # Gaps from both sub-questions (same assessment returned for both)
        assert updates["gaps"] is not None
        assert "Missing historical context" in updates["gaps"]

    async def test_graceful_degradation_on_total_failure(self):
        state = make_initial_state("Test", "pid-1")
        state["research_plan"] = SAMPLE_PLAN

        with (
            patch(f"{_P}.retrieve", new_callable=AsyncMock) as mock_retrieve,
            patch(f"{_P}.VectorStore"),
        ):
            mock_retrieve.side_effect = Exception("DB connection failed")

            updates = await run_gatherer(state)

        assert updates["status"] == "analyzing"
        assert "Gatherer error" in updates["errors"][0]
