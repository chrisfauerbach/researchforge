"""Document ingestion orchestrator: parse → chunk → embed → store."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path

import structlog

from researchforge.db.repository import Repository
from researchforge.rag.chunker import chunk_document
from researchforge.rag.embeddings import embed_texts
from researchforge.rag.parsers import detect_source_type, parse_document
from researchforge.rag.store import VectorStore

logger = structlog.get_logger()


@dataclass
class IngestResult:
    document_id: str
    source_path: str
    source_type: str
    chunk_count: int
    skipped: bool = False
    reason: str = ""


def _file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file for deduplication."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


async def ingest_file(
    path: str | Path,
    store: VectorStore,
    repo: Repository,
) -> IngestResult:
    """Ingest a single file into the RAG corpus.

    Orchestrates: parse → chunk → embed → store + metadata.
    Deduplicates by file hash (skips if already ingested).
    """
    path = Path(path).resolve()
    source_type = detect_source_type(path)

    logger.info("ingest_start", path=str(path), source_type=source_type)

    # Dedup check
    fhash = _file_hash(path)
    existing = await repo.get_document_by_hash(fhash)
    if existing is not None:
        logger.info("ingest_skipped_duplicate", path=str(path), existing_id=existing["document_id"])
        return IngestResult(
            document_id=existing["document_id"],
            source_path=str(path),
            source_type=source_type,
            chunk_count=existing["chunk_count"],
            skipped=True,
            reason="duplicate",
        )

    document_id = str(uuid.uuid4())

    # Parse
    text, source_type = parse_document(path)

    # Chunk
    chunks = chunk_document(
        text,
        source_type=source_type,
        extra_metadata={"source_path": str(path)},
    )

    if not chunks:
        logger.warning("ingest_no_chunks", path=str(path))
        return IngestResult(
            document_id=document_id,
            source_path=str(path),
            source_type=source_type,
            chunk_count=0,
            skipped=True,
            reason="no_chunks",
        )

    # Embed
    chunk_texts = [c.text for c in chunks]
    vectors = await embed_texts(chunk_texts)

    # Generate chunk IDs
    chunk_ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [c.metadata for c in chunks]

    # Store in LanceDB
    store.add_chunks(
        chunk_ids=chunk_ids,
        texts=chunk_texts,
        vectors=vectors,
        document_id=document_id,
        source_path=str(path),
        source_type=source_type,
        metadatas=metadatas,
    )

    # Rebuild FTS index
    store.create_fts_index()

    # Record metadata in SQLite
    await repo.insert_document(
        document_id=document_id,
        source_path=str(path),
        source_type=source_type,
        file_hash=fhash,
        file_size_bytes=path.stat().st_size,
        chunk_count=len(chunks),
        title=path.stem,
    )

    # Record chunk lineage
    lineage_rows = [
        {
            "chunk_id": cid,
            "document_id": document_id,
            "briefing_id": None,
            "content_type": "source",
            "chunk_index": c.chunk_index,
            "char_start": None,
            "char_end": None,
        }
        for cid, c in zip(chunk_ids, chunks)
    ]
    await repo.insert_chunk_lineage_batch(lineage_rows)

    logger.info(
        "ingest_complete",
        document_id=document_id,
        path=str(path),
        chunk_count=len(chunks),
    )

    return IngestResult(
        document_id=document_id,
        source_path=str(path),
        source_type=source_type,
        chunk_count=len(chunks),
    )


async def ingest_directory(
    dir_path: str | Path,
    store: VectorStore,
    repo: Repository,
) -> list[IngestResult]:
    """Ingest all supported files in a directory (non-recursive)."""
    dir_path = Path(dir_path)
    supported_extensions = {".pdf", ".md", ".markdown", ".txt"}
    results = []

    for file_path in sorted(dir_path.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            result = await ingest_file(file_path, store, repo)
            results.append(result)

    return results
