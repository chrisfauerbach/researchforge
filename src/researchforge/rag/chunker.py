"""Document chunking: two-pass strategy for structure-aware splitting."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from researchforge.config import get_settings


@dataclass
class Chunk:
    """A single text chunk with metadata."""

    text: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def chunk_document(
    text: str,
    source_type: str = "txt",
    extra_metadata: dict | None = None,
) -> list[Chunk]:
    """Split document text into chunks using a two-pass strategy.

    Pass 1 (Markdown/PDF): MarkdownHeaderTextSplitter to preserve section structure.
    Pass 2: RecursiveCharacterTextSplitter for size-constrained splitting.

    For plain text, only Pass 2 is applied.
    """
    settings = get_settings()
    cfg = settings.chunking
    base_metadata = extra_metadata or {}

    # Pass 1: Structure-aware split for Markdown-like content
    is_structured = source_type in ("pdf", "markdown")

    if is_structured and _has_headers(text):
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "section_h1"),
                ("##", "section_h2"),
                ("###", "section_h3"),
            ],
            strip_headers=False,
        )
        md_docs = md_splitter.split_text(text)

        # Pass 2: Size-constrained split on each structural section
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=cfg.separators + [""],
            length_function=len,
        )
        final_docs = text_splitter.split_documents(md_docs)

        chunks = []
        for i, doc in enumerate(final_docs):
            meta = {**base_metadata}
            meta["section_h1"] = doc.metadata.get("section_h1", "")
            meta["section_h2"] = doc.metadata.get("section_h2", "")
            chunks.append(Chunk(text=doc.page_content, chunk_index=i, metadata=meta))
        return chunks

    # Plain text / no headers: direct recursive split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=cfg.separators + [""],
        length_function=len,
    )
    split_texts = text_splitter.split_text(text)

    return [
        Chunk(text=t, chunk_index=i, metadata={**base_metadata})
        for i, t in enumerate(split_texts)
    ]


def _has_headers(text: str) -> bool:
    """Quick check: does the text contain Markdown headers?"""
    for line in text.split("\n")[:200]:
        stripped = line.strip()
        if stripped.startswith("#") and " " in stripped:
            return True
    return False
