"""Corpus feedback loop: score briefings and auto-ingest high-quality output."""

from __future__ import annotations

import re
import uuid

import structlog

from researchforge.config import get_settings
from researchforge.db.repository import Repository
from researchforge.rag.chunker import chunk_document
from researchforge.rag.embeddings import embed_texts
from researchforge.rag.store import VectorStore

logger = structlog.get_logger()


def compute_quality_score(
    briefing_markdown: str,
    pipeline_trace: list[dict],
    critic_verdict: str | None = None,
) -> float:
    """Heuristic quality score for a briefing (0.0 – 1.0).

    Each criterion adds 0.2 (max 1.0):
      - Word count > 200
      - Has citations ([1], [Source...] patterns)
      - Has 2+ ## section headers
      - No "error" status entries in pipeline trace
      - Critic verdict == "pass"
    """
    score = 0.0

    # Word count
    if len(briefing_markdown.split()) > 200:
        score += 0.2

    # Citations
    if re.search(r"\[\d+\]|\[Source", briefing_markdown):
        score += 0.2

    # Section headers (2+)
    header_count = len(re.findall(r"^##\s", briefing_markdown, re.MULTILINE))
    if header_count >= 2:
        score += 0.2

    # No errors in trace
    has_errors = any(
        entry.get("status") == "error" for entry in pipeline_trace
    )
    if not has_errors:
        score += 0.2

    # Critic verdict
    if critic_verdict and critic_verdict.lower() == "pass":
        score += 0.2

    return round(min(score, 1.0), 2)


async def maybe_ingest_briefing(
    briefing_id: str,
    repo: Repository,
    store: VectorStore,
    *,
    critic_verdict: str | None = None,
) -> dict:
    """Score a completed briefing and ingest into the corpus if quality is high enough.

    Returns:
        dict with keys: quality_score, ingested, chunk_count, reason
    """
    settings = get_settings()
    threshold = settings.pipeline.quality_threshold_for_corpus

    briefing = await repo.get_briefing(briefing_id)
    if briefing is None:
        return {
            "quality_score": 0.0,
            "ingested": False,
            "chunk_count": 0,
            "reason": "briefing_not_found",
        }

    markdown = briefing.get("briefing_markdown") or ""
    if not markdown.strip():
        return {
            "quality_score": 0.0,
            "ingested": False,
            "chunk_count": 0,
            "reason": "empty_briefing",
        }

    # Parse pipeline trace from JSON string if needed
    trace_raw = briefing.get("pipeline_trace")
    if isinstance(trace_raw, str):
        import json

        try:
            trace = json.loads(trace_raw)
        except (json.JSONDecodeError, TypeError):
            trace = []
    elif isinstance(trace_raw, list):
        trace = trace_raw
    else:
        trace = []

    score = compute_quality_score(markdown, trace, critic_verdict)

    # Update quality_score in DB
    await repo.update_briefing(briefing_id, quality_score=score)

    if score < threshold:
        logger.info(
            "feedback_below_threshold",
            briefing_id=briefing_id,
            score=score,
            threshold=threshold,
        )
        return {
            "quality_score": score,
            "ingested": False,
            "chunk_count": 0,
            "reason": f"score {score} < threshold {threshold}",
        }

    # Chunk the briefing
    chunks = chunk_document(
        markdown,
        source_type="markdown",
        extra_metadata={"source_path": f"briefing:{briefing_id}"},
    )

    if not chunks:
        return {
            "quality_score": score,
            "ingested": False,
            "chunk_count": 0,
            "reason": "no_chunks",
        }

    # Embed
    chunk_texts = [c.text for c in chunks]
    vectors = await embed_texts(chunk_texts)

    # Generate chunk IDs
    chunk_ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [c.metadata for c in chunks]

    # Store in LanceDB with agent_generated content type
    store.add_chunks(
        chunk_ids=chunk_ids,
        texts=chunk_texts,
        vectors=vectors,
        document_id=briefing_id,
        source_path=f"briefing:{briefing_id}",
        source_type="briefing",
        metadatas=metadatas,
        content_type="agent_generated",
        briefing_id=briefing_id,
        quality_score=score,
    )

    # Rebuild FTS index
    store.create_fts_index()

    # Record chunk lineage
    lineage_rows = [
        {
            "chunk_id": cid,
            "document_id": None,
            "briefing_id": briefing_id,
            "content_type": "agent_generated",
            "chunk_index": c.chunk_index,
            "char_start": None,
            "char_end": None,
        }
        for cid, c in zip(chunk_ids, chunks)
    ]
    await repo.insert_chunk_lineage_batch(lineage_rows)

    # Mark briefing as ingested
    await repo.db.execute(
        "UPDATE briefings SET ingested_into_corpus = 1 WHERE briefing_id = ?",
        (briefing_id,),
    )
    await repo.db.commit()

    logger.info(
        "feedback_ingested",
        briefing_id=briefing_id,
        score=score,
        chunk_count=len(chunks),
    )

    return {
        "quality_score": score,
        "ingested": True,
        "chunk_count": len(chunks),
        "reason": "quality_above_threshold",
    }
