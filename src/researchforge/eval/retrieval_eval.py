"""Retrieval quality evaluation: Precision@K, Recall@K, MRR."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import structlog

logger = structlog.get_logger()


@dataclass
class RetrievalTestCase:
    """A single retrieval evaluation test case."""

    question: str
    expected_keywords: list[str]
    topic: str = ""


@dataclass
class RetrievalCaseResult:
    """Result of evaluating a single test case."""

    question: str
    topic: str
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    hits: int
    total_expected: int
    retrieved_count: int


@dataclass
class RetrievalEvalResult:
    """Aggregated retrieval evaluation results."""

    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_mrr: float
    case_count: int
    case_results: list[RetrievalCaseResult] = field(default_factory=list)


def load_retrieval_test_set(path: str | Path) -> list[RetrievalTestCase]:
    """Load test cases from a JSONL file."""
    path = Path(path)
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(RetrievalTestCase(
                question=data["question"],
                expected_keywords=data.get("expected_keywords", []),
                topic=data.get("topic", ""),
            ))
    return cases


def keyword_hits(retrieved_texts: list[str], expected_keywords: list[str]) -> list[bool]:
    """Check which expected keywords appear in retrieved texts.

    Returns a list of bools, one per retrieved chunk, indicating whether
    that chunk contains at least one expected keyword (case-insensitive).
    """
    results = []
    lower_keywords = [kw.lower() for kw in expected_keywords]
    for text in retrieved_texts:
        text_lower = text.lower()
        hit = any(kw in text_lower for kw in lower_keywords)
        results.append(hit)
    return results


def precision_at_k(hits: list[bool], k: int | None = None) -> float:
    """Fraction of top-K retrieved results that are relevant."""
    if not hits:
        return 0.0
    if k is not None:
        hits = hits[:k]
    relevant = sum(hits)
    return relevant / len(hits)


def recall_at_k(hits: list[bool], total_expected: int, k: int | None = None) -> float:
    """Fraction of expected relevant items found in top-K results."""
    if total_expected == 0:
        return 0.0
    if k is not None:
        hits = hits[:k]
    found = sum(hits)
    return min(found / total_expected, 1.0)


def reciprocal_rank(hits: list[bool]) -> float:
    """Reciprocal rank of the first relevant result (1/rank)."""
    for i, hit in enumerate(hits):
        if hit:
            return 1.0 / (i + 1)
    return 0.0


async def evaluate_retrieval(
    test_set_path: str | Path,
    top_k: int = 5,
) -> RetrievalEvalResult:
    """Run retrieval evaluation on a test set.

    For each test case, runs the retriever and computes keyword-based
    relevance metrics. This requires the corpus to be populated.
    """
    from researchforge.rag.retriever import retrieve
    from researchforge.rag.store import VectorStore

    cases = load_retrieval_test_set(test_set_path)
    store = VectorStore()
    case_results: list[RetrievalCaseResult] = []

    for case in cases:
        try:
            results = await retrieve(case.question, store, top_k=top_k)
            retrieved_texts = [r.get("text", "") for r in results]
        except Exception as exc:
            logger.warning("retrieval_eval_error", question=case.question, error=str(exc))
            retrieved_texts = []

        hits = keyword_hits(retrieved_texts, case.expected_keywords)
        total_expected = len(case.expected_keywords)

        case_results.append(RetrievalCaseResult(
            question=case.question,
            topic=case.topic,
            precision_at_k=precision_at_k(hits, top_k),
            recall_at_k=recall_at_k(hits, total_expected, top_k),
            reciprocal_rank=reciprocal_rank(hits),
            hits=sum(hits),
            total_expected=total_expected,
            retrieved_count=len(retrieved_texts),
        ))

    # Aggregate
    n = len(case_results)
    if n == 0:
        return RetrievalEvalResult(
            mean_precision_at_k=0.0,
            mean_recall_at_k=0.0,
            mean_mrr=0.0,
            case_count=0,
        )

    return RetrievalEvalResult(
        mean_precision_at_k=round(sum(c.precision_at_k for c in case_results) / n, 4),
        mean_recall_at_k=round(sum(c.recall_at_k for c in case_results) / n, 4),
        mean_mrr=round(sum(c.reciprocal_rank for c in case_results) / n, 4),
        case_count=n,
        case_results=case_results,
    )
