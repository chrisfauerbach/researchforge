"""Tests for the analyst agent."""

import json
from unittest.mock import AsyncMock, patch

from researchforge.agents.analyst import run_analyst
from researchforge.agents.state import make_initial_state

SAMPLE_EVIDENCE = [
    {
        "sub_question_id": 1,
        "sub_question": "What is RAG?",
        "chunks": [
            {
                "text": "RAG combines retrieval with generation.",
                "source_path": "doc.pdf",
                "section_h1": "Intro",
            },
        ],
        "assessment": {
            "relevant_evidence": [
                {"chunk_index": 0, "relevance": "high", "key_points": ["RAG definition"]}
            ]
        },
    },
]

VALID_ANALYSIS = {
    "findings": [
        {
            "finding": "RAG systems combine retrieval with generation to ground responses.",
            "evidence_sources": [1],
            "confidence": "high",
            "reasoning": "Directly stated in source material.",
        }
    ],
    "cross_references": [],
    "contradictions": [],
    "gaps": [],
    "overall_confidence": "medium",
}


class TestRunAnalyst:
    async def test_produces_analysis(self):
        state = make_initial_state("What is RAG?", "pid-1")
        state["evidence"] = SAMPLE_EVIDENCE
        state["status"] = "analyzing"

        with patch("researchforge.agents.analyst.ollama_chat", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "content": json.dumps(VALID_ANALYSIS),
                "parsed": VALID_ANALYSIS,
                "model": "qwen2.5:14b",
                "input_tokens": 500,
                "output_tokens": 400,
                "duration_ms": 10000,
                "fallback_used": False,
            }

            updates = await run_analyst(state)

        assert updates["analysis"] == VALID_ANALYSIS
        assert updates["status"] == "critiquing"
        assert len(state["trace"]) == 1
        assert state["trace"][0]["agent"] == "analyst"

    async def test_includes_gaps_in_prompt(self):
        state = make_initial_state("Test", "pid-1")
        state["evidence"] = SAMPLE_EVIDENCE
        state["gaps"] = ["Missing historical context"]

        with patch("researchforge.agents.analyst.ollama_chat", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "content": json.dumps(VALID_ANALYSIS),
                "parsed": VALID_ANALYSIS,
                "model": "qwen2.5:14b",
                "input_tokens": 500,
                "output_tokens": 400,
                "duration_ms": 10000,
                "fallback_used": False,
            }

            await run_analyst(state)

        call_kwargs = mock.call_args.kwargs
        assert "Missing historical context" in call_kwargs["user_message"]

    async def test_revision_includes_critic_feedback(self):
        state = make_initial_state("Test", "pid-1")
        state["evidence"] = SAMPLE_EVIDENCE
        state["analysis"] = VALID_ANALYSIS
        state["critic_issues"] = [
            {
                "type": "unsupported_claim",
                "description": "Finding 1 lacks sufficient evidence",
                "suggestion": "Add more sources",
            }
        ]
        state["revision_count"] = 1

        with patch("researchforge.agents.analyst.ollama_chat", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "content": json.dumps(VALID_ANALYSIS),
                "parsed": VALID_ANALYSIS,
                "model": "qwen2.5:14b",
                "input_tokens": 800,
                "output_tokens": 500,
                "duration_ms": 12000,
                "fallback_used": False,
            }

            await run_analyst(state)

        call_kwargs = mock.call_args.kwargs
        assert "REVISION REQUEST" in call_kwargs["user_message"]
        assert "unsupported_claim" in call_kwargs["user_message"]
        assert "Add more sources" in call_kwargs["user_message"]

    async def test_graceful_degradation_on_error(self):
        state = make_initial_state("Test", "pid-1")
        state["evidence"] = SAMPLE_EVIDENCE

        with patch("researchforge.agents.analyst.ollama_chat", new_callable=AsyncMock) as mock:
            mock.side_effect = Exception("Model OOM")

            updates = await run_analyst(state)

        analysis = updates["analysis"]
        assert analysis["overall_confidence"] == "low"
        assert "Analyst error" in updates["errors"][0]
        assert state["trace"][0]["status"] == "error"
