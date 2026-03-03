"""Tests for the writer agent."""

from unittest.mock import AsyncMock, patch

from researchforge.agents.state import make_initial_state
from researchforge.agents.writer import run_writer

SAMPLE_EVIDENCE = [
    {
        "sub_question_id": 1,
        "sub_question": "What is RAG?",
        "chunks": [
            {
                "text": "RAG is retrieval augmented generation.",
                "source_path": "doc.pdf",
                "section_h1": "Intro",
            },
        ],
    },
]

SAMPLE_ANALYSIS = {
    "findings": [
        {
            "finding": "RAG combines retrieval with generation.",
            "evidence_sources": [1],
            "confidence": "high",
            "reasoning": "Directly stated in source.",
        }
    ],
    "cross_references": [],
    "contradictions": [],
    "gaps": [],
    "overall_confidence": "high",
}

SAMPLE_BRIEFING = """\
# Research Briefing: RAG Overview

## Executive Summary
RAG combines retrieval systems with language models to produce grounded responses.

## Detailed Findings

### RAG Architecture
RAG retrieves relevant documents and provides them as context [Source 1].

## Caveats and Limitations
- Limited to available corpus

## Confidence Assessment
High confidence based on available evidence.

## Further Research
1. How does chunk size affect RAG quality?
2. What are the latency implications of RAG?
"""


class TestRunWriter:
    async def test_produces_briefing(self):
        state = make_initial_state("What is RAG?", "pid-1")
        state["evidence"] = SAMPLE_EVIDENCE
        state["analysis"] = SAMPLE_ANALYSIS

        with patch("researchforge.agents.writer.ollama_chat", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "content": SAMPLE_BRIEFING,
                "parsed": None,
                "model": "mistral-nemo:12b",
                "input_tokens": 600,
                "output_tokens": 800,
                "duration_ms": 15000,
                "fallback_used": False,
            }

            updates = await run_writer(state)

        assert updates["briefing"] == SAMPLE_BRIEFING
        assert updates["status"] == "completed"
        assert len(state["trace"]) == 1
        assert state["trace"][0]["agent"] == "writer"

    async def test_does_not_request_json(self):
        state = make_initial_state("Test", "pid-1")
        state["evidence"] = SAMPLE_EVIDENCE
        state["analysis"] = SAMPLE_ANALYSIS

        with patch("researchforge.agents.writer.ollama_chat", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "content": "# Briefing\nSome text",
                "parsed": None,
                "model": "mistral-nemo:12b",
                "input_tokens": 400,
                "output_tokens": 500,
                "duration_ms": 8000,
                "fallback_used": False,
            }

            await run_writer(state)

        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["expect_json"] is False

    async def test_includes_critic_issues_as_caveats(self):
        state = make_initial_state("Test", "pid-1")
        state["evidence"] = SAMPLE_EVIDENCE
        state["analysis"] = SAMPLE_ANALYSIS
        state["critic_issues"] = [
            {"type": "missing_perspective", "description": "No discussion of limitations"},
        ]

        with patch("researchforge.agents.writer.ollama_chat", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "content": "# Briefing with caveats",
                "parsed": None,
                "model": "mistral-nemo:12b",
                "input_tokens": 600,
                "output_tokens": 500,
                "duration_ms": 10000,
                "fallback_used": False,
            }

            await run_writer(state)

        call_kwargs = mock.call_args.kwargs
        assert "No discussion of limitations" in call_kwargs["user_message"]
        assert "Unresolved reviewer concerns" in call_kwargs["user_message"]

    async def test_graceful_degradation_on_error(self):
        state = make_initial_state("What is RAG?", "pid-1")
        state["evidence"] = SAMPLE_EVIDENCE
        state["analysis"] = SAMPLE_ANALYSIS

        with patch("researchforge.agents.writer.ollama_chat", new_callable=AsyncMock) as mock:
            mock.side_effect = Exception("Model crashed")

            updates = await run_writer(state)

        # Should produce a fallback briefing from raw analysis
        assert "Research Briefing" in updates["briefing"]
        assert "RAG combines retrieval" in updates["briefing"]
        assert "Writer agent failed" in updates["briefing"]
        assert updates["status"] == "completed"

    async def test_includes_pipeline_errors_in_prompt(self):
        state = make_initial_state("Test", "pid-1")
        state["evidence"] = SAMPLE_EVIDENCE
        state["analysis"] = SAMPLE_ANALYSIS
        state["errors"] = ["Planner error: timeout"]

        with patch("researchforge.agents.writer.ollama_chat", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "content": "# Briefing",
                "parsed": None,
                "model": "mistral-nemo:12b",
                "input_tokens": 400,
                "output_tokens": 300,
                "duration_ms": 7000,
                "fallback_used": False,
            }

            await run_writer(state)

        call_kwargs = mock.call_args.kwargs
        assert "Planner error: timeout" in call_kwargs["user_message"]
