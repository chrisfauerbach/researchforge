"""Tests for model benchmarking module."""

from __future__ import annotations

from researchforge.eval.benchmark import (
    BENCHMARK_PROMPTS,
    BenchmarkResult,
    ModelBenchmarkEntry,
)
from researchforge.eval.judge import RubricScore


class TestBenchmarkPrompts:
    def test_has_standard_roles(self):
        assert "planner" in BENCHMARK_PROMPTS
        assert "analyst" in BENCHMARK_PROMPTS
        assert "writer" in BENCHMARK_PROMPTS
        assert "critic" in BENCHMARK_PROMPTS

    def test_prompts_have_system_and_user(self):
        for role, prompts in BENCHMARK_PROMPTS.items():
            assert "system" in prompts, f"{role} missing system prompt"
            assert "user" in prompts, f"{role} missing user prompt"


class TestBenchmarkResult:
    def test_summary_table(self):
        entry = ModelBenchmarkEntry(
            model="test-model",
            role="writer",
            score=RubricScore(
                structural_validity=0.8,
                relevance=0.9,
                completeness=0.7,
                coherence=0.85,
                conciseness=0.6,
                weighted_total=0.8,
            ),
            latency_ms=1000,
            output_tokens=500,
        )
        result = BenchmarkResult(
            role="writer",
            models=["test-model"],
            entries=[entry],
        )
        table = result.summary_table()
        assert len(table) == 1
        assert table[0]["model"] == "test-model"
        assert table[0]["weighted_score"] == 0.8
        assert table[0]["latency_ms"] == 1000

    def test_summary_table_with_error(self):
        entry = ModelBenchmarkEntry(
            model="broken-model",
            role="writer",
            error="Model not found",
        )
        result = BenchmarkResult(
            role="writer",
            models=["broken-model"],
            entries=[entry],
        )
        table = result.summary_table()
        assert table[0]["weighted_score"] == 0.0
        assert table[0]["error"] == "Model not found"
