"""Tests for the critic agent."""

import json
from unittest.mock import AsyncMock, patch

from researchforge.agents.critic import run_critic
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
    },
]

SAMPLE_ANALYSIS = {
    "findings": [
        {
            "finding": "RAG grounds LLM responses in factual data.",
            "evidence_sources": [1],
            "confidence": "high",
            "reasoning": "Stated in source.",
        }
    ],
    "cross_references": [],
    "contradictions": [],
    "gaps": [],
    "overall_confidence": "high",
}

PASS_REVIEW = {
    "verdict": "pass",
    "issues": [],
    "strengths": ["Well-supported findings", "Clear reasoning"],
    "overall_assessment": "Analysis is solid and well-grounded in evidence.",
}

REVISE_REVIEW = {
    "verdict": "revise",
    "issues": [
        {
            "type": "unsupported_claim",
            "description": "Claim about 'always improving quality' lacks evidence",
            "suggestion": "Either cite a source or soften the language",
        },
        {
            "type": "missing_perspective",
            "description": "No discussion of RAG limitations",
            "suggestion": "Add a section on known challenges",
        },
    ],
    "strengths": ["Good structure"],
    "overall_assessment": "Needs improvement on evidence support.",
}


class TestRunCritic:
    async def test_pass_verdict(self):
        state = make_initial_state("What is RAG?", "pid-1")
        state["evidence"] = SAMPLE_EVIDENCE
        state["analysis"] = SAMPLE_ANALYSIS

        with patch("researchforge.agents.critic.ollama_chat", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "content": json.dumps(PASS_REVIEW),
                "parsed": PASS_REVIEW,
                "model": "deepseek-r1:7b",
                "input_tokens": 300,
                "output_tokens": 150,
                "duration_ms": 4000,
                "fallback_used": False,
            }

            updates = await run_critic(state)

        assert updates["critic_verdict"] == "pass"
        assert updates["critic_issues"] is None
        assert updates["status"] == "writing"

    async def test_revise_verdict(self):
        state = make_initial_state("What is RAG?", "pid-1")
        state["evidence"] = SAMPLE_EVIDENCE
        state["analysis"] = SAMPLE_ANALYSIS
        state["revision_count"] = 0

        with patch("researchforge.agents.critic.ollama_chat", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "content": json.dumps(REVISE_REVIEW),
                "parsed": REVISE_REVIEW,
                "model": "deepseek-r1:7b",
                "input_tokens": 300,
                "output_tokens": 200,
                "duration_ms": 5000,
                "fallback_used": False,
            }

            updates = await run_critic(state)

        assert updates["critic_verdict"] == "revise"
        assert len(updates["critic_issues"]) == 2
        assert updates["revision_count"] == 1

    async def test_increments_revision_count(self):
        state = make_initial_state("Test", "pid-1")
        state["evidence"] = SAMPLE_EVIDENCE
        state["analysis"] = SAMPLE_ANALYSIS
        state["revision_count"] = 1  # Already revised once

        with patch("researchforge.agents.critic.ollama_chat", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "content": json.dumps(REVISE_REVIEW),
                "parsed": REVISE_REVIEW,
                "model": "deepseek-r1:7b",
                "input_tokens": 300,
                "output_tokens": 200,
                "duration_ms": 5000,
                "fallback_used": False,
            }

            updates = await run_critic(state)

        assert updates["revision_count"] == 2

    async def test_graceful_degradation_on_error(self):
        state = make_initial_state("Test", "pid-1")
        state["evidence"] = SAMPLE_EVIDENCE
        state["analysis"] = SAMPLE_ANALYSIS

        with patch("researchforge.agents.critic.ollama_chat", new_callable=AsyncMock) as mock:
            mock.side_effect = Exception("Connection reset")

            updates = await run_critic(state)

        # On failure, pass through to writer
        assert updates["critic_verdict"] == "pass"
        assert updates["status"] == "writing"
        assert "Critic error" in updates["errors"][0]

    async def test_records_trace(self):
        state = make_initial_state("Test", "pid-1")
        state["evidence"] = SAMPLE_EVIDENCE
        state["analysis"] = SAMPLE_ANALYSIS

        with patch("researchforge.agents.critic.ollama_chat", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "content": json.dumps(PASS_REVIEW),
                "parsed": PASS_REVIEW,
                "model": "deepseek-r1:7b",
                "input_tokens": 300,
                "output_tokens": 150,
                "duration_ms": 4000,
                "fallback_used": False,
            }

            await run_critic(state)

        assert len(state["trace"]) == 1
        assert state["trace"][0]["agent"] == "critic"
