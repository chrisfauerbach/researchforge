"""Model benchmarking: compare models per agent role on quality and latency."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

from researchforge.eval.judge import LLMJudge, RubricScore

logger = structlog.get_logger()

BENCHMARK_PROMPTS = {
    "planner": {
        "system": (
            "You are a research planner. Given a topic, produce a research plan "
            "with 3-5 sub-questions and search strategies. Respond in Markdown."
        ),
        "user": (
            "Create a research plan for: The impact of retrieval "
            "augmented generation on reducing LLM hallucinations"
        ),
    },
    "analyst": {
        "system": (
            "You are a research analyst. Synthesize the following evidence into "
            "a structured analysis with key findings, supporting evidence, and gaps. "
            "Respond in Markdown."
        ),
        "user": (
            "Evidence:\n"
            "- RAG systems reduce hallucination by 30-50% compared to base LLMs (Lewis 2020)\n"
            "- Hybrid retrieval (vector + BM25) outperforms vector-only by 15% on precision\n"
            "- Chunk size of 500-1500 tokens optimal for most use cases\n"
            "- Small embedding models (384-dim) underperform larger ones (768+dim)\n"
            "- Query reformulation improves retrieval recall by 20%\n\n"
            "Synthesize these findings into a structured analysis."
        ),
    },
    "writer": {
        "system": (
            "You are a research writer. Produce a well-structured briefing document "
            "from the analysis provided. Include an executive summary, key findings, "
            "and recommendations. Respond in Markdown."
        ),
        "user": (
            "Analysis summary:\n"
            "Topic: Best practices for RAG system design\n"
            "Key findings: 1) Hybrid search outperforms vector-only, "
            "2) Chunk overlap prevents information loss, "
            "3) Query reformulation boosts recall significantly.\n"
            "Gaps: Limited data on very large corpora (>1M documents).\n\n"
            "Write a complete briefing from this analysis."
        ),
    },
    "critic": {
        "system": (
            "You are a quality reviewer. Review the following analysis for errors, "
            "logical issues, missing evidence, and areas for improvement. "
            "Provide specific, actionable feedback."
        ),
        "user": (
            "Analysis: RAG systems always eliminate hallucinations completely. "
            "All embedding models produce identical results. Chunk size does not matter. "
            "Vector search is always better than keyword search.\n\n"
            "Review this analysis for quality issues."
        ),
    },
}


@dataclass
class ModelBenchmarkEntry:
    """Benchmark result for a single model on a single role."""

    model: str
    role: str
    score: RubricScore | None = None
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""


@dataclass
class BenchmarkResult:
    """Complete benchmark comparison."""

    role: str
    models: list[str]
    entries: list[ModelBenchmarkEntry] = field(default_factory=list)

    def summary_table(self) -> list[dict]:
        """Return a summary as a list of dicts for display."""
        return [
            {
                "model": e.model,
                "role": e.role,
                "weighted_score": e.score.weighted_total if e.score else 0.0,
                "latency_ms": e.latency_ms,
                "output_tokens": e.output_tokens,
                "error": e.error,
            }
            for e in self.entries
        ]


async def benchmark_models(
    role: str,
    models: list[str],
    judge: LLMJudge | None = None,
) -> BenchmarkResult:
    """Benchmark multiple models on the same agent role.

    Runs each model with the same prompt for the given role,
    then scores the output using the LLM judge.
    """
    from researchforge.agents.ollama_client import ollama_chat

    if judge is None:
        judge = LLMJudge(num_runs=1)  # Single run for benchmarking speed

    prompts = BENCHMARK_PROMPTS.get(role)
    if not prompts:
        return BenchmarkResult(role=role, models=models)

    result = BenchmarkResult(role=role, models=models)

    for model in models:
        entry = ModelBenchmarkEntry(model=model, role=role)

        try:
            start = time.monotonic()
            response = await ollama_chat(
                model=model,
                system_prompt=prompts["system"],
                user_message=prompts["user"],
                expect_json=False,
                agent_name=f"benchmark_{role}",
            )
            entry.latency_ms = int((time.monotonic() - start) * 1000)
            entry.input_tokens = response.get("input_tokens", 0)
            entry.output_tokens = response.get("output_tokens", 0)

            # Score the output
            content = response.get("content", "")
            if content.strip():
                score = await judge.score(
                    content,
                    task_description=prompts["system"],
                )
                entry.score = score
            else:
                entry.error = "Empty response"

        except Exception as exc:
            logger.warning("benchmark_error", model=model, role=role, error=str(exc))
            entry.error = str(exc)

        result.entries.append(entry)
        logger.info(
            "benchmark_entry",
            model=model,
            role=role,
            score=entry.score.weighted_total if entry.score else 0,
            latency_ms=entry.latency_ms,
        )

    return result
