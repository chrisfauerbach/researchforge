"""Tests for LLM judge module."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from researchforge.eval.judge import (
    LLMJudge,
    RubricScore,
    compute_heuristic_scores,
    heuristic_citation_count,
    heuristic_readability,
    heuristic_section_count,
    heuristic_word_count,
)


class TestHeuristicWordCount:
    def test_above_200(self):
        assert heuristic_word_count("word " * 201) == 1.0

    def test_below_200(self):
        score = heuristic_word_count("word " * 100)
        assert score == 0.5

    def test_empty(self):
        assert heuristic_word_count("") == 0.0


class TestHeuristicCitationCount:
    def test_no_citations(self):
        assert heuristic_citation_count("No citations here.") == 0.0

    def test_numbered_citations(self):
        assert heuristic_citation_count("See [1] and [2].") == 0.6

    def test_three_or_more(self):
        assert heuristic_citation_count("[1] [2] [3] cited.") == 1.0

    def test_source_pattern(self):
        assert heuristic_citation_count("[Source: Smith] data.") == 0.6


class TestHeuristicSectionCount:
    def test_no_headers(self):
        assert heuristic_section_count("Just plain text.") == 0.0

    def test_one_header(self):
        assert heuristic_section_count("# Title\nContent") == 0.4

    def test_two_headers(self):
        score = heuristic_section_count("# Title\n## Sub\nContent")
        assert score == 0.7

    def test_four_plus_headers(self):
        text = "# A\n## B\n### C\n## D\nContent"
        assert heuristic_section_count(text) == 1.0


class TestHeuristicReadability:
    def test_ideal_length(self):
        # Sentences with ~20 words each
        text = ("This is a sentence with approximately twenty words in it. " * 3)
        score = heuristic_readability(text)
        assert score >= 0.6

    def test_empty(self):
        assert heuristic_readability("") == 0.0


class TestComputeHeuristicScores:
    def test_returns_all_keys(self):
        scores = compute_heuristic_scores("Some text with [1] citation.\n# Header\n")
        assert "word_count" in scores
        assert "citations" in scores
        assert "sections" in scores
        assert "readability" in scores


class TestRubricScore:
    def test_weighted_total(self):
        score = RubricScore(
            structural_validity=1.0,
            relevance=1.0,
            completeness=1.0,
            coherence=1.0,
            conciseness=1.0,
        )
        total = score.compute_weighted_total()
        assert total == 1.0

    def test_weighted_total_partial(self):
        score = RubricScore(
            structural_validity=0.5,
            relevance=0.5,
            completeness=0.5,
            coherence=0.5,
            conciseness=0.5,
        )
        total = score.compute_weighted_total()
        assert total == 0.5

    def test_weighted_total_zero(self):
        score = RubricScore()
        total = score.compute_weighted_total()
        assert total == 0.0


class TestLLMJudge:
    @patch("researchforge.agents.ollama_client.ollama_chat", new_callable=AsyncMock)
    async def test_score_aggregates_runs(self, mock_chat):
        mock_chat.return_value = {
            "parsed": {
                "structural_validity": 0.8,
                "relevance": 0.9,
                "completeness": 0.7,
                "coherence": 0.85,
                "conciseness": 0.6,
            }
        }
        judge = LLMJudge(model="test-model", num_runs=3)
        score = await judge.score("Test text", task_description="Test task")
        assert score.structural_validity == 0.8
        assert score.relevance == 0.9
        assert score.weighted_total > 0
        assert mock_chat.call_count == 3

    @patch("researchforge.agents.ollama_client.ollama_chat", new_callable=AsyncMock)
    async def test_score_clamps_values(self, mock_chat):
        mock_chat.return_value = {
            "parsed": {
                "structural_validity": 1.5,
                "relevance": -0.2,
                "completeness": 0.7,
                "coherence": 0.85,
                "conciseness": 0.6,
            }
        }
        judge = LLMJudge(model="test-model", num_runs=1)
        score = await judge.score("Test text")
        assert score.structural_validity == 1.0
        assert score.relevance == 0.0

    @patch("researchforge.agents.ollama_client.ollama_chat", new_callable=AsyncMock)
    async def test_all_runs_fail_returns_empty_score(self, mock_chat):
        mock_chat.side_effect = Exception("Ollama down")
        judge = LLMJudge(model="test-model", num_runs=3)
        score = await judge.score("Test text")
        assert score.weighted_total == 0.0
        assert "word_count" in score.heuristic_scores

    async def test_heuristic_only_score(self):
        judge = LLMJudge()
        text = "word " * 201 + "\n# Header\n## Sub\n[1] cite."
        score = await judge.score_heuristic_only(text)
        assert score.weighted_total > 0
        assert score.completeness > 0
