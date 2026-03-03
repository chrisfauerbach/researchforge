"""LanceDB vector store wrapper."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import lancedb
import pyarrow as pa
import structlog

from researchforge.config import get_settings

logger = structlog.get_logger()

EMBEDDING_DIM = 768

CHUNKS_SCHEMA = pa.schema([
    pa.field("chunk_id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
    pa.field("document_id", pa.string()),
    pa.field("source_path", pa.string()),
    pa.field("source_type", pa.string()),
    pa.field("content_type", pa.string()),
    pa.field("section_h1", pa.string()),
    pa.field("section_h2", pa.string()),
    pa.field("chunk_index", pa.int32()),
    pa.field("ingested_at", pa.string()),
    pa.field("briefing_id", pa.string()),
    pa.field("quality_score", pa.float32()),
])

TABLE_NAME = "chunks"


class VectorStore:
    """Wrapper around LanceDB for chunk storage and retrieval."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = get_settings().storage.vector_db_path
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.db_path))
        self._table = None

    @property
    def table(self):
        if self._table is None:
            try:
                self._table = self._db.open_table(TABLE_NAME)
            except Exception:
                # Table doesn't exist yet — create empty
                self._table = self._db.create_table(
                    TABLE_NAME,
                    schema=CHUNKS_SCHEMA,
                    mode="overwrite",
                )
        return self._table

    def add_chunks(
        self,
        chunk_ids: list[str],
        texts: list[str],
        vectors: list[list[float]],
        document_id: str,
        source_path: str,
        source_type: str,
        metadatas: list[dict] | None = None,
        content_type: str = "source",
        briefing_id: str = "",
        quality_score: float | None = None,
    ) -> int:
        """Add chunks with their embeddings to the store.

        Returns the number of chunks added.
        """
        now = datetime.now(UTC).isoformat()
        metadatas = metadatas or [{}] * len(chunk_ids)

        rows = []
        for i, (cid, text, vec, meta) in enumerate(
            zip(chunk_ids, texts, vectors, metadatas)
        ):
            rows.append({
                "chunk_id": cid,
                "text": text,
                "vector": vec,
                "document_id": document_id,
                "source_path": source_path,
                "source_type": source_type,
                "content_type": content_type,
                "section_h1": meta.get("section_h1", ""),
                "section_h2": meta.get("section_h2", ""),
                "chunk_index": i,
                "ingested_at": now,
                "briefing_id": briefing_id,
                "quality_score": quality_score,
            })

        self.table.add(rows)
        logger.info("chunks_added", count=len(rows), document_id=document_id)
        return len(rows)

    def create_fts_index(self) -> None:
        """Build or rebuild the Tantivy full-text search index on the text column."""
        self.table.create_fts_index("text", replace=True)
        logger.info("fts_index_created")

    def count(self) -> int:
        """Return the total number of chunks in the store."""
        return self.table.count_rows()

    def vector_search(
        self,
        query_vector: list[float],
        limit: int = 20,
        where: str | None = None,
    ) -> list[dict]:
        """Pure vector similarity search."""
        q = self.table.search(query_vector).limit(limit)
        if where:
            q = q.where(where)
        return q.to_list()

    def fts_search(
        self,
        query_text: str,
        limit: int = 20,
        where: str | None = None,
    ) -> list[dict]:
        """Pure full-text (BM25) search."""
        q = self.table.search(query_text, query_type="fts").limit(limit)
        if where:
            q = q.where(where)
        return q.to_list()
