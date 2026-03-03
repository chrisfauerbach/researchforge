"""Document parsers: extract text from PDF, Markdown, and plain text."""

from __future__ import annotations

from pathlib import Path


def detect_source_type(path: str | Path) -> str:
    """Detect document type from file extension."""
    suffix = Path(path).suffix.lower()
    mapping = {
        ".pdf": "pdf",
        ".md": "markdown",
        ".markdown": "markdown",
        ".txt": "txt",
        ".html": "html",
        ".htm": "html",
        ".docx": "docx",
    }
    return mapping.get(suffix, "txt")


def parse_pdf(path: str | Path) -> str:
    """Extract Markdown text from a PDF using pymupdf4llm."""
    import pymupdf4llm

    md_text = pymupdf4llm.to_markdown(str(path))
    return md_text


def parse_markdown(path: str | Path) -> str:
    """Read a Markdown file as-is."""
    return Path(path).read_text(encoding="utf-8")


def parse_txt(path: str | Path) -> str:
    """Read a plain text file."""
    return Path(path).read_text(encoding="utf-8")


def parse_document(path: str | Path) -> tuple[str, str]:
    """Parse a document and return (text, source_type).

    Returns Markdown-formatted text for structured formats (PDF),
    or raw text for plain text files.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    source_type = detect_source_type(path)

    parsers = {
        "pdf": parse_pdf,
        "markdown": parse_markdown,
        "txt": parse_txt,
    }

    parser = parsers.get(source_type)
    if parser is None:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            f"Supported: {', '.join(parsers.keys())}"
        )

    text = parser(path)
    return text, source_type
