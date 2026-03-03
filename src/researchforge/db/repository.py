"""Async SQLite data access layer."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from researchforge.db.models import SCHEMA_SQL


class Repository:
    """Async wrapper around the SQLite metadata database."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create the database and tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Repository not initialized. Call initialize() first.")
        return self._db

    # --- Documents ---

    async def get_document_by_hash(self, file_hash: str) -> dict | None:
        cursor = await self.db.execute(
            "SELECT * FROM documents WHERE file_hash = ?", (file_hash,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def insert_document(
        self,
        document_id: str,
        source_path: str,
        source_type: str,
        file_hash: str,
        file_size_bytes: int,
        chunk_count: int,
        title: str | None = None,
    ) -> None:
        await self.db.execute(
            """INSERT INTO documents
               (document_id, source_path, source_type, title, file_hash,
                file_size_bytes, ingested_at, chunk_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                document_id,
                source_path,
                source_type,
                title,
                file_hash,
                file_size_bytes,
                datetime.now(UTC).isoformat(),
                chunk_count,
            ),
        )
        await self.db.commit()

    # --- Briefings ---

    async def insert_briefing(
        self,
        briefing_id: str,
        research_question: str,
        status: str = "running",
    ) -> None:
        await self.db.execute(
            """INSERT INTO briefings
               (briefing_id, research_question, status, started_at)
               VALUES (?, ?, ?, ?)""",
            (
                briefing_id,
                research_question,
                status,
                datetime.now(UTC).isoformat(),
            ),
        )
        await self.db.commit()

    async def update_briefing(
        self,
        briefing_id: str,
        *,
        status: str | None = None,
        briefing_markdown: str | None = None,
        quality_score: float | None = None,
        pipeline_trace: list[dict] | None = None,
    ) -> None:
        updates: list[str] = []
        params: list = []
        if status is not None:
            updates.append("status = ?")
            params.append(status)
            if status == "completed":
                updates.append("completed_at = ?")
                params.append(datetime.now(UTC).isoformat())
        if briefing_markdown is not None:
            updates.append("briefing_markdown = ?")
            params.append(briefing_markdown)
        if quality_score is not None:
            updates.append("quality_score = ?")
            params.append(quality_score)
        if pipeline_trace is not None:
            updates.append("pipeline_trace = ?")
            params.append(json.dumps(pipeline_trace))
        if not updates:
            return
        params.append(briefing_id)
        sql = f"UPDATE briefings SET {', '.join(updates)} WHERE briefing_id = ?"
        await self.db.execute(sql, params)
        await self.db.commit()

    async def get_briefing(self, briefing_id: str) -> dict | None:
        cursor = await self.db.execute(
            "SELECT * FROM briefings WHERE briefing_id = ?", (briefing_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_briefings(
        self, limit: int = 10, status: str | None = None
    ) -> list[dict]:
        if status and status != "all":
            cursor = await self.db.execute(
                "SELECT * FROM briefings WHERE status = ? ORDER BY started_at DESC LIMIT ?",
                (status, limit),
            )
        else:
            cursor = await self.db.execute(
                "SELECT * FROM briefings ORDER BY started_at DESC LIMIT ?", (limit,)
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # --- Chunk lineage ---

    async def insert_chunk_lineage(
        self,
        chunk_id: str,
        content_type: str,
        chunk_index: int,
        document_id: str | None = None,
        briefing_id: str | None = None,
        char_start: int | None = None,
        char_end: int | None = None,
    ) -> None:
        await self.db.execute(
            """INSERT INTO chunk_lineage
               (chunk_id, document_id, briefing_id, content_type,
                chunk_index, char_start, char_end)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (chunk_id, document_id, briefing_id, content_type, chunk_index, char_start, char_end),
        )
        await self.db.commit()

    async def insert_chunk_lineage_batch(self, rows: list[dict]) -> None:
        await self.db.executemany(
            """INSERT INTO chunk_lineage
               (chunk_id, document_id, briefing_id, content_type,
                chunk_index, char_start, char_end)
               VALUES (:chunk_id, :document_id, :briefing_id, :content_type,
                       :chunk_index, :char_start, :char_end)""",
            rows,
        )
        await self.db.commit()
