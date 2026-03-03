"""Tests for the gatherer agent."""

import json
from unittest.mock import AsyncMock, patch

from researchforge.agents.gatherer import run_gatherer
from researchforge.agents.state import make_initial_state
from researchforge.config import WebSearchConfig

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
            patch(f"{_P}.web_search_for_question", new_callable=AsyncMock) as mock_ws,
            patch(f"{_P}.VectorStore"),
        ):
            mock_retrieve.return_value = []
            mock_ws.return_value = []

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


SINGLE_Q_PLAN = {
    "sub_questions": [
        {
            "id": 1, "question": "What is xyzzy?",
            "info_needs": ["anything"], "priority": "high",
        },
    ],
    "overall_approach": "Search",
    "expected_source_types": [],
}

WEB_CHUNKS = [
    {
        "chunk_id": "web:abc123",
        "text": "Xyzzy is a magic word from Colossal Cave Adventure.",
        "source_path": "https://example.com/xyzzy",
        "section_h1": "Xyzzy",
        "source_type": "web",
        "content_type": "web_search",
    },
]

WEB_ASSESSMENT = {
    "sub_question_id": 1,
    "sub_question": "What is xyzzy?",
    "relevant_evidence": [
        {"chunk_index": 0, "relevance": "high", "key_points": ["magic word"]},
    ],
    "gaps": [],
    "sufficiency": "sufficient",
}


class TestGathererWebSearch:
    async def test_web_search_triggered_auto_mode_empty_rag(self):
        """In auto mode, web search fires when RAG returns no chunks."""
        state = make_initial_state("What is xyzzy?", "pid-1")
        state["research_plan"] = SINGLE_Q_PLAN

        with (
            patch(f"{_P}.retrieve", new_callable=AsyncMock) as mock_retrieve,
            patch(f"{_P}.web_search_for_question", new_callable=AsyncMock) as mock_ws,
            patch(f"{_P}.ollama_chat", new_callable=AsyncMock) as mock_chat,
            patch(f"{_P}.VectorStore"),
            patch(f"{_P}.get_settings") as mock_settings,
        ):
            mock_settings.return_value.models.gatherer = "qwen2.5:7b"
            mock_settings.return_value.web_search = WebSearchConfig(mode="auto")
            mock_retrieve.return_value = []
            mock_ws.return_value = WEB_CHUNKS
            mock_chat.return_value = {
                "content": json.dumps(WEB_ASSESSMENT),
                "parsed": WEB_ASSESSMENT,
                "model": "qwen2.5:7b",
                "input_tokens": 50,
                "output_tokens": 40,
                "duration_ms": 1000,
                "fallback_used": False,
            }

            updates = await run_gatherer(state)

        mock_ws.assert_called_once()
        # Web chunks should flow through to the LLM assessment
        assert len(updates["evidence"]) == 1
        assert updates["evidence"][0]["chunks"] == WEB_CHUNKS

    async def test_web_search_not_triggered_when_rag_has_results(self):
        """In auto mode, web search does NOT fire when RAG returns chunks."""
        state = make_initial_state("What is RAG?", "pid-1")
        state["research_plan"] = SINGLE_Q_PLAN

        with (
            patch(f"{_P}.retrieve", new_callable=AsyncMock) as mock_retrieve,
            patch(f"{_P}.web_search_for_question", new_callable=AsyncMock) as mock_ws,
            patch(f"{_P}.ollama_chat", new_callable=AsyncMock) as mock_chat,
            patch(f"{_P}.VectorStore"),
            patch(f"{_P}.get_settings") as mock_settings,
        ):
            mock_settings.return_value.models.gatherer = "qwen2.5:7b"
            mock_settings.return_value.web_search = WebSearchConfig(mode="auto")
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

        mock_ws.assert_not_called()
        assert len(updates["evidence"]) == 1

    async def test_web_search_not_triggered_when_disabled(self):
        """When mode is disabled, web search never fires even with empty RAG."""
        state = make_initial_state("What is xyzzy?", "pid-1")
        state["research_plan"] = SINGLE_Q_PLAN

        with (
            patch(f"{_P}.retrieve", new_callable=AsyncMock) as mock_retrieve,
            patch(f"{_P}.web_search_for_question", new_callable=AsyncMock) as mock_ws,
            patch(f"{_P}.VectorStore"),
            patch(f"{_P}.get_settings") as mock_settings,
        ):
            mock_settings.return_value.models.gatherer = "qwen2.5:7b"
            mock_settings.return_value.web_search = WebSearchConfig(mode="disabled")
            mock_retrieve.return_value = []

            updates = await run_gatherer(state)

        mock_ws.assert_not_called()
        # Falls through to insufficient assessment
        assert updates["evidence"][0]["assessment"]["sufficiency"] == "insufficient"
