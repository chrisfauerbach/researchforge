"""Tests for eval runner and regression detection."""

from __future__ import annotations

import json

from researchforge.eval.runner import (
    _load_recent_scores,
    detect_regressions,
)


class TestDetectRegressions:
    def test_no_history(self):
        regressions = detect_regressions({"precision": 0.8}, [])
        assert regressions == []

    def test_no_regression(self):
        history = [
            {"precision": 0.80},
            {"precision": 0.82},
            {"precision": 0.79},
        ]
        regressions = detect_regressions({"precision": 0.78}, history)
        assert regressions == []

    def test_detects_regression(self):
        history = [
            {"precision": 0.80},
            {"precision": 0.82},
            {"precision": 0.81},
        ]
        # Current is 0.60 — much lower than avg ~0.81 (~26% drop)
        regressions = detect_regressions({"precision": 0.60}, history)
        assert len(regressions) == 1
        assert regressions[0]["metric"] == "precision"
        assert regressions[0]["drop_pct"] > 10

    def test_multiple_metrics(self):
        history = [{"precision": 0.80, "recall": 0.70}]
        # Precision drops, recall stays
        regressions = detect_regressions(
            {"precision": 0.50, "recall": 0.68}, history
        )
        assert len(regressions) == 1
        assert regressions[0]["metric"] == "precision"

    def test_zero_avg_no_crash(self):
        history = [{"precision": 0.0}]
        regressions = detect_regressions({"precision": 0.0}, history)
        assert regressions == []

    def test_threshold_boundary(self):
        history = [{"precision": 1.0}]
        # Exactly 10% drop — should NOT trigger (> threshold required)
        regressions = detect_regressions(
            {"precision": 0.90}, history, threshold=0.10
        )
        assert regressions == []

        # Just over 10% drop
        regressions = detect_regressions(
            {"precision": 0.89}, history, threshold=0.10
        )
        assert len(regressions) == 1


class TestLoadRecentScores:
    def test_loads_jsonl(self, tmp_path):
        path = tmp_path / "scores.jsonl"
        data = [{"score": 0.5}, {"score": 0.6}, {"score": 0.7}]
        with open(path, "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")

        result = _load_recent_scores(tmp_path, "scores.jsonl", limit=2)
        assert len(result) == 2
        assert result[0]["score"] == 0.6
        assert result[1]["score"] == 0.7

    def test_missing_file(self, tmp_path):
        result = _load_recent_scores(tmp_path, "nonexistent.jsonl")
        assert result == []

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        result = _load_recent_scores(tmp_path, "empty.jsonl")
        assert result == []
