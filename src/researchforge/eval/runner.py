"""Eval suite runner: orchestrates all evaluation dimensions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import structlog

from researchforge.config import get_settings

logger = structlog.get_logger()

DEFAULT_DATASETS_DIR = Path(__file__).parent.parent.parent.parent / "eval" / "datasets"
DEFAULT_RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "eval" / "results"


@dataclass
class FullEvalResult:
    """Combined results from all evaluation dimensions."""

    timestamp: str = ""
    retrieval: dict = field(default_factory=dict)
    agent_scores: dict = field(default_factory=dict)
    e2e: dict = field(default_factory=dict)
    regressions: list[dict] = field(default_factory=list)


def _load_recent_scores(results_dir: Path, filename: str, limit: int = 5) -> list[dict]:
    """Load the last N eval results from a JSONL file."""
    path = results_dir / filename
    if not path.exists():
        return []
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results[-limit:]


def detect_regressions(
    current: dict[str, float],
    history: list[dict],
    threshold: float = 0.10,
) -> list[dict]:
    """Detect metrics that dropped >threshold from rolling average.

    Args:
        current: Current metric values {metric_name: value}.
        history: List of previous result dicts, each with the same keys.
        threshold: Fraction drop that triggers a regression flag.
    """
    if not history:
        return []

    regressions = []
    for metric, current_value in current.items():
        past_values = [h.get(metric, 0.0) for h in history if metric in h]
        if not past_values:
            continue
        avg = sum(past_values) / len(past_values)
        if avg > 0 and (avg - current_value) / avg > threshold:
            regressions.append({
                "metric": metric,
                "current": round(current_value, 4),
                "rolling_avg": round(avg, 4),
                "drop_pct": round((avg - current_value) / avg * 100, 1),
            })
    return regressions


def _append_result(results_dir: Path, filename: str, data: dict) -> None:
    """Append a result to a JSONL file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / filename
    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")


async def run_retrieval_eval(
    datasets_dir: Path | None = None,
    results_dir: Path | None = None,
) -> dict:
    """Run retrieval evaluation and record results."""
    from researchforge.eval.retrieval_eval import evaluate_retrieval

    datasets_dir = datasets_dir or DEFAULT_DATASETS_DIR
    results_dir = results_dir or DEFAULT_RESULTS_DIR

    test_set = datasets_dir / "retrieval_test.jsonl"
    if not test_set.exists():
        logger.warning("retrieval_test_set_not_found", path=str(test_set))
        return {}

    result = await evaluate_retrieval(test_set, top_k=5)
    data = {
        "timestamp": datetime.now(UTC).isoformat(),
        "mean_precision_at_k": result.mean_precision_at_k,
        "mean_recall_at_k": result.mean_recall_at_k,
        "mean_mrr": result.mean_mrr,
        "case_count": result.case_count,
    }

    _append_result(results_dir, "retrieval_scores.jsonl", data)

    # Check for regressions
    history = _load_recent_scores(results_dir, "retrieval_scores.jsonl")
    current_metrics = {
        "precision": result.mean_precision_at_k,
        "recall": result.mean_recall_at_k,
        "mrr": result.mean_mrr,
    }
    regressions = detect_regressions(current_metrics, history[:-1])

    return {**data, "regressions": regressions}


async def run_agent_eval(
    results_dir: Path | None = None,
) -> dict:
    """Run agent evaluation on completed briefings."""
    from researchforge.db.repository import Repository
    from researchforge.eval.agent_eval import evaluate_agent
    from researchforge.eval.judge import LLMJudge

    results_dir = results_dir or DEFAULT_RESULTS_DIR
    settings = get_settings()
    repo = Repository(settings.storage.metadata_db_path)
    await repo.initialize()

    try:
        briefings = await repo.list_briefings(limit=10, status="completed")
        if not briefings:
            logger.warning("no_completed_briefings_for_eval")
            return {}

        judge = LLMJudge()
        agent_scores = {}

        # Evaluate writer output (the briefing itself)
        writer_result = await evaluate_agent("writer", briefings, judge=judge)
        agent_scores["writer"] = {
            "mean_score": writer_result.mean_score,
            "case_count": writer_result.case_count,
        }

        data = {
            "timestamp": datetime.now(UTC).isoformat(),
            **agent_scores,
        }
        _append_result(results_dir, "agent_scores.jsonl", data)
        return data
    finally:
        await repo.close()


async def run_e2e_eval(
    datasets_dir: Path | None = None,
    results_dir: Path | None = None,
) -> dict:
    """Run end-to-end evaluation against reference briefings."""
    from researchforge.eval.e2e_eval import evaluate_e2e

    datasets_dir = datasets_dir or DEFAULT_DATASETS_DIR
    results_dir = results_dir or DEFAULT_RESULTS_DIR

    ref_dir = datasets_dir / "reference_briefings"
    if not ref_dir.exists() or not list(ref_dir.glob("*.md")):
        logger.warning("no_reference_briefings_found", path=str(ref_dir))
        return {}

    result = await evaluate_e2e(ref_dir, depth="quick")
    data = {
        "timestamp": datetime.now(UTC).isoformat(),
        "mean_score": result.mean_score,
        "case_count": result.case_count,
        "cases": [
            {
                "topic": c.topic,
                "score": c.score.weighted_total if c.score else 0.0,
                "status": c.pipeline_status,
                "error": c.error,
            }
            for c in result.cases
        ],
    }
    _append_result(results_dir, "e2e_scores.jsonl", data)
    return data


async def run_full_eval(
    datasets_dir: Path | None = None,
    results_dir: Path | None = None,
    skip_e2e: bool = False,
) -> FullEvalResult:
    """Run the complete evaluation suite.

    Args:
        datasets_dir: Path to eval datasets directory.
        results_dir: Path to eval results directory.
        skip_e2e: Skip end-to-end eval (which requires running full pipelines).
    """
    datasets_dir = datasets_dir or DEFAULT_DATASETS_DIR
    results_dir = results_dir or DEFAULT_RESULTS_DIR

    result = FullEvalResult(timestamp=datetime.now(UTC).isoformat())

    # Retrieval eval
    logger.info("eval_starting", dimension="retrieval")
    result.retrieval = await run_retrieval_eval(datasets_dir, results_dir)

    # Agent eval
    logger.info("eval_starting", dimension="agent")
    result.agent_scores = await run_agent_eval(results_dir)

    # E2E eval (optional — runs full pipeline which is slow)
    if not skip_e2e:
        logger.info("eval_starting", dimension="e2e")
        result.e2e = await run_e2e_eval(datasets_dir, results_dir)

    # Collect all regressions
    result.regressions = result.retrieval.get("regressions", [])

    logger.info(
        "eval_complete",
        retrieval=bool(result.retrieval),
        agent=bool(result.agent_scores),
        e2e=bool(result.e2e),
    )
    return result
