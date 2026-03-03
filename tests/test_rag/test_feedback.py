"""Tests for corpus feedback loop."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from researchforge.rag.feedback import compute_quality_score, maybe_ingest_briefing


class TestComputeQualityScore:
    def test_empty_briefing_scores_low(self):
        # Empty text still gets 0.2 for "no errors in trace" (empty trace = no errors)
        score = compute_quality_score("", [], None)
        assert score == 0.2

    def test_word_count_above_200(self):
        text = "word " * 201
        score = compute_quality_score(text, [], None)
        assert score >= 0.2

    def test_word_count_below_200(self):
        text = "short text"
        score = compute_quality_score(text, [], None)
        # Only no-error bonus applies (empty trace = no errors)
        assert score == 0.2

    def test_citations_numbered(self):
        text = "word " * 201 + "According to [1] the data shows..."
        score = compute_quality_score(text, [], None)
        # word count + citations + no errors = 0.6
        assert score >= 0.6

    def test_citations_source_pattern(self):
        text = "word " * 201 + "[Source: Smith 2024] shows..."
        score = compute_quality_score(text, [], None)
        assert score >= 0.6

    def test_section_headers(self):
        text = "word " * 201 + "\n## Section One\n\n## Section Two\n"
        score = compute_quality_score(text, [], None)
        # word count + headers + no errors = 0.6
        assert score >= 0.6

    def test_no_errors_in_trace(self):
        trace = [{"agent": "planner", "status": "ok"}]
        score = compute_quality_score("short", trace, None)
        # no-error bonus only
        assert score == 0.2

    def test_errors_in_trace(self):
        trace = [{"agent": "gatherer", "status": "error"}]
        score = compute_quality_score("short", trace, None)
        assert score == 0.0

    def test_critic_pass(self):
        text = "word " * 201
        score = compute_quality_score(text, [], "pass")
        # word count + no errors + critic pass = 0.6
        assert score == 0.6

    def test_critic_fail_no_bonus(self):
        text = "word " * 201
        score = compute_quality_score(text, [], "fail")
        # word count + no errors = 0.4
        assert score == 0.4

    def test_max_score(self):
        text = (
            "word " * 201
            + "\n## Section One\n\n## Section Two\n"
            + "Citing [1] and [2]."
        )
        trace = [{"agent": "planner", "status": "ok"}]
        score = compute_quality_score(text, trace, "pass")
        assert score == 1.0

    def test_score_caps_at_one(self):
        text = (
            "word " * 201
            + "\n## A\n\n## B\n\n## C\n"
            + "[1] [2] [Source: foo]"
        )
        score = compute_quality_score(text, [], "pass")
        assert score <= 1.0


class TestMaybeIngestBriefing:
    @pytest.fixture
    def mock_repo(self):
        repo = AsyncMock()
        repo.db = AsyncMock()
        repo.db.execute = AsyncMock()
        repo.db.commit = AsyncMock()
        return repo

    @pytest.fixture
    def mock_store(self):
        store = MagicMock()
        store.add_chunks = MagicMock(return_value=3)
        store.create_fts_index = MagicMock()
        return store

    async def test_briefing_not_found(self, mock_repo, mock_store):
        mock_repo.get_briefing.return_value = None
        result = await maybe_ingest_briefing("missing-id", mock_repo, mock_store)
        assert result["ingested"] is False
        assert result["reason"] == "briefing_not_found"

    async def test_empty_briefing(self, mock_repo, mock_store):
        mock_repo.get_briefing.return_value = {
            "briefing_id": "b1",
            "briefing_markdown": "",
            "pipeline_trace": "[]",
        }
        result = await maybe_ingest_briefing("b1", mock_repo, mock_store)
        assert result["ingested"] is False
        assert result["reason"] == "empty_briefing"

    async def test_below_threshold(self, mock_repo, mock_store):
        mock_repo.get_briefing.return_value = {
            "briefing_id": "b2",
            "briefing_markdown": "Short text.",
            "pipeline_trace": "[]",
        }
        result = await maybe_ingest_briefing("b2", mock_repo, mock_store)
        assert result["ingested"] is False
        assert "threshold" in result["reason"]

    @patch("researchforge.rag.feedback.embed_texts")
    async def test_above_threshold_ingests(self, mock_embed, mock_repo, mock_store):
        # Build a high-quality briefing
        md = (
            "word " * 201
            + "\n## Section One\n\n## Section Two\n"
            + "Citing [1] source."
        )
        mock_repo.get_briefing.return_value = {
            "briefing_id": "b3",
            "briefing_markdown": md,
            "pipeline_trace": "[]",
        }
        mock_embed.return_value = [[0.1] * 768]
        mock_repo.insert_chunk_lineage_batch = AsyncMock()

        result = await maybe_ingest_briefing(
            "b3", mock_repo, mock_store, critic_verdict="pass"
        )
        assert result["ingested"] is True
        assert result["chunk_count"] > 0
        assert result["quality_score"] >= 0.6
        mock_store.add_chunks.assert_called_once()

    @patch("researchforge.rag.feedback.embed_texts")
    async def test_lineage_recorded(self, mock_embed, mock_repo, mock_store):
        md = (
            "word " * 201
            + "\n## A\n\n## B\n"
            + "[1] ref."
        )
        mock_repo.get_briefing.return_value = {
            "briefing_id": "b4",
            "briefing_markdown": md,
            "pipeline_trace": "[]",
        }
        mock_embed.return_value = [[0.1] * 768]
        mock_repo.insert_chunk_lineage_batch = AsyncMock()

        await maybe_ingest_briefing("b4", mock_repo, mock_store, critic_verdict="pass")
        mock_repo.insert_chunk_lineage_batch.assert_called_once()
        rows = mock_repo.insert_chunk_lineage_batch.call_args[0][0]
        assert all(r["content_type"] == "agent_generated" for r in rows)
        assert all(r["briefing_id"] == "b4" for r in rows)
