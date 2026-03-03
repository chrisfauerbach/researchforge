"""Hybrid retrieval: vector + BM25 with Reciprocal Rank Fusion."""

from __future__ import annotations

import structlog

from researchforge.config import get_settings
from researchforge.rag.embeddings import embed_query
from researchforge.rag.store import VectorStore

logger = structlog.get_logger()


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[str]:
    """Fuse multiple ranked lists using RRF.

    Args:
        rankings: List of ranked lists, each containing chunk_ids.
        k: RRF constant (default 60, standard value).

    Returns:
        Fused ranking as a list of chunk_ids, highest score first.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


async def retrieve(
    query: str,
    store: VectorStore,
    top_k: int | None = None,
    source_only: bool = False,
) -> list[dict]:
    """Hybrid search: vector + BM25, fused with RRF.

    Args:
        query: Natural language query.
        store: VectorStore instance to search.
        top_k: Number of results to return (default from config).
        source_only: If True, exclude agent-generated content.

    Returns:
        List of chunk dicts, ranked by RRF score.
    """
    settings = get_settings()
    cfg = settings.retrieval
    if top_k is None:
        top_k = cfg.final_top_k

    where_clause = "content_type = 'source'" if source_only else None

    # Embed the query
    query_vector = await embed_query(query)

    # Vector search
    vector_results = store.vector_search(
        query_vector,
        limit=cfg.vector_candidates,
        where=where_clause,
    )
    vector_ids = [r["chunk_id"] for r in vector_results]

    # BM25/FTS search
    try:
        fts_results = store.fts_search(
            query,
            limit=cfg.bm25_candidates,
            where=where_clause,
        )
        fts_ids = [r["chunk_id"] for r in fts_results]
    except Exception:
        # FTS index may not exist yet — fall back to vector-only
        logger.warning("fts_search_failed_fallback_to_vector_only")
        fts_ids = []

    # Fuse rankings
    fused_ids = reciprocal_rank_fusion([vector_ids, fts_ids])[:top_k]

    # Build result set from combined results, preserving fused order
    all_results = {r["chunk_id"]: r for r in vector_results + fts_results}
    results = []
    for chunk_id in fused_ids:
        if chunk_id in all_results:
            result = all_results[chunk_id]
            # Remove the raw vector from results to keep output clean
            result.pop("vector", None)
            results.append(result)

    logger.info(
        "retrieval_complete",
        query_len=len(query),
        vector_hits=len(vector_ids),
        fts_hits=len(fts_ids),
        fused_results=len(results),
    )
    return results
