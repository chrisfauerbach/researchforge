"""Single researcher agent — Phase 1 vertical slice.

Takes a research question + retrieved context and produces a Markdown briefing
via a single Ollama chat call. Replaced by the full 5-agent pipeline in Phase 2.
"""

from __future__ import annotations

import httpx
import structlog

from researchforge.config import get_settings

logger = structlog.get_logger()

SYSTEM_PROMPT = """\
You are a research analyst. Given a research question and relevant source material, \
produce a clear, well-structured research briefing in Markdown format.

Your briefing MUST include:
1. **Executive Summary** — 2-3 sentence overview of key findings
2. **Detailed Findings** — Organized by theme or sub-topic, with specific evidence
3. **Source Citations** — Reference which source each finding comes from
4. **Confidence Assessment** — Note where evidence is strong vs. weak or missing
5. **Further Research** — Suggest 2-3 follow-up questions

Rules:
- Base your findings ONLY on the provided source material
- If the sources don't cover an aspect of the question, say so explicitly
- Be concise but thorough
- Use bullet points and headers for readability
"""


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source_path", "unknown")
        section = chunk.get("section_h1", "")
        if section:
            section = f" > {section}"
        parts.append(f"[Source {i}: {source}{section}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


async def research(question: str, chunks: list[dict]) -> str:
    """Run the researcher agent: question + context → briefing.

    Args:
        question: The research question.
        chunks: Retrieved context chunks from the RAG system.

    Returns:
        Markdown-formatted research briefing.
    """
    settings = get_settings()
    model = settings.models.researcher
    base_url = settings.ollama.base_url
    timeout = settings.ollama.request_timeout_seconds

    context = _format_context(chunks)

    user_message = f"""Research Question: {question}

Source Material:

{context}

Produce a research briefing based on the above source material."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    logger.info("researcher_start", model=model, question_len=len(question), chunks=len(chunks))

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"num_ctx": 8192},
            },
        )
        resp.raise_for_status()
        data = resp.json()

    briefing = data["message"]["content"]
    logger.info(
        "researcher_complete",
        model=model,
        output_len=len(briefing),
    )
    return briefing
