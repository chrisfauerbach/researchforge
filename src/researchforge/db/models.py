"""SQLite schema definitions and initialization."""

from __future__ import annotations

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    source_type TEXT NOT NULL,
    title TEXT,
    file_hash TEXT NOT NULL,
    file_size_bytes INTEGER,
    ingested_at TEXT NOT NULL,
    chunk_count INTEGER
);

CREATE TABLE IF NOT EXISTS briefings (
    briefing_id TEXT PRIMARY KEY,
    research_question TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    quality_score REAL,
    ingested_into_corpus INTEGER DEFAULT 0,
    briefing_markdown TEXT,
    pipeline_trace TEXT
);

CREATE TABLE IF NOT EXISTS chunk_lineage (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT,
    briefing_id TEXT,
    content_type TEXT NOT NULL,
    chunk_index INTEGER,
    char_start INTEGER,
    char_end INTEGER
);

CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_briefings_status ON briefings(status);
CREATE INDEX IF NOT EXISTS idx_chunk_lineage_document_id ON chunk_lineage(document_id);
"""
