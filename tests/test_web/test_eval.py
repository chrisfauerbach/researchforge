"""Tests for eval dashboard and human scoring routes."""

from __future__ import annotations

import researchforge.web.app as app_module


class TestEvalDashboard:
    async def test_eval_page_renders(self, web_client):
        resp = await web_client.get("/eval")
        assert resp.status_code == 200
        assert "Evaluation Dashboard" in resp.text

    async def test_eval_page_shows_no_results_message(self, web_client):
        resp = await web_client.get("/eval")
        assert resp.status_code == 200
        assert "researchforge eval run" in resp.text

    async def test_eval_scores_json(self, web_client):
        resp = await web_client.get("/api/eval/scores")
        assert resp.status_code == 200
        data = resp.json()
        assert "retrieval" in data
        assert "agent" in data
        assert "e2e" in data


class TestHumanScoring:
    async def test_score_thumbs_up(self, web_client):
        # Insert a briefing first
        repo = app_module._repo
        await repo.insert_briefing("test-b1", "Test question", status="completed")

        resp = await web_client.post(
            "/api/briefings/test-b1/score",
            data={"score": "1", "feedback": ""},
        )
        assert resp.status_code == 200
        assert "Thumbs up" in resp.text

        # Verify it was saved
        briefing = await repo.get_briefing("test-b1")
        assert briefing["human_score"] == 1

    async def test_score_thumbs_down(self, web_client):
        repo = app_module._repo
        await repo.insert_briefing("test-b2", "Test question 2", status="completed")

        resp = await web_client.post(
            "/api/briefings/test-b2/score",
            data={"score": "-1", "feedback": "Needs improvement"},
        )
        assert resp.status_code == 200
        assert "Thumbs down" in resp.text

    async def test_score_invalid(self, web_client):
        repo = app_module._repo
        await repo.insert_briefing("test-b3", "Test question 3", status="completed")

        resp = await web_client.post(
            "/api/briefings/test-b3/score",
            data={"score": "5"},
        )
        assert resp.status_code == 400
