"""Shared Ollama chat client with retry, fallback, and JSON enforcement."""

from __future__ import annotations

import json
import time

import httpx
import structlog

from researchforge.config import get_settings

logger = structlog.get_logger()

MAX_STRUCTURAL_RETRIES = 2


class OllamaError(Exception):
    """Base error for Ollama client failures."""


class OllamaTimeoutError(OllamaError):
    """Ollama request timed out."""


class OllamaResponseError(OllamaError):
    """Ollama returned an invalid or unparseable response."""


async def ollama_chat(
    *,
    model: str,
    system_prompt: str,
    user_message: str,
    expect_json: bool = False,
    json_schema: dict | None = None,
    agent_name: str = "unknown",
) -> dict:
    """Call Ollama /api/chat with retry and fallback logic.

    Args:
        model: The Ollama model tag (e.g. "qwen2.5:14b").
        system_prompt: System prompt for the model.
        user_message: User message content.
        expect_json: If True, parse the response as JSON and retry on failure.
        json_schema: Optional JSON schema to pass to Ollama's format parameter.
        agent_name: Name of the calling agent (for logging and fallback lookup).

    Returns:
        Dict with keys:
            - "content": str (raw text response)
            - "parsed": dict | None (parsed JSON if expect_json, else None)
            - "model": str (model actually used, may differ if fallback triggered)
            - "input_tokens": int
            - "output_tokens": int
            - "duration_ms": int
            - "fallback_used": bool
    """
    settings = get_settings()
    base_url = settings.ollama.base_url
    timeout = settings.ollama.request_timeout_seconds

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    request_body: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": 8192},
    }

    if json_schema is not None:
        request_body["format"] = json_schema
    elif expect_json:
        request_body["format"] = "json"

    current_model = model
    fallback_used = False

    # Try primary model, then fallback on timeout
    for attempt_model in _model_sequence(model, agent_name):
        if attempt_model != model:
            fallback_used = True
            current_model = attempt_model
            request_body["model"] = current_model
            logger.warning(
                "ollama_fallback",
                agent=agent_name,
                primary=model,
                fallback=current_model,
            )

        try:
            result = await _call_with_retries(
                base_url=base_url,
                timeout=timeout,
                request_body=request_body,
                expect_json=expect_json,
                agent_name=agent_name,
                model=current_model,
            )
            result["fallback_used"] = fallback_used
            result["model"] = current_model
            return result
        except OllamaTimeoutError:
            logger.warning(
                "ollama_timeout",
                agent=agent_name,
                model=current_model,
                timeout=timeout,
            )
            continue

    raise OllamaTimeoutError(
        f"All models timed out for agent {agent_name} "
        f"(primary: {model}, timeout: {timeout}s)"
    )


async def _call_with_retries(
    *,
    base_url: str,
    timeout: int,
    request_body: dict,
    expect_json: bool,
    agent_name: str,
    model: str,
) -> dict:
    """Execute the Ollama call with structural retries on invalid JSON."""
    last_error = None

    for attempt in range(1 + MAX_STRUCTURAL_RETRIES):
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{base_url}/api/chat",
                    json=request_body,
                )
                resp.raise_for_status()
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(str(exc)) from exc

        elapsed_ms = int((time.monotonic() - start) * 1000)
        data = resp.json()

        content = data.get("message", {}).get("content", "")
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        logger.info(
            "ollama_call",
            agent=agent_name,
            model=model,
            attempt=attempt + 1,
            duration_ms=elapsed_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        if not expect_json:
            return {
                "content": content,
                "parsed": None,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "duration_ms": elapsed_ms,
            }

        # Try to parse JSON
        try:
            parsed = json.loads(content)
            return {
                "content": content,
                "parsed": parsed,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "duration_ms": elapsed_ms,
            }
        except json.JSONDecodeError as exc:
            last_error = exc
            logger.warning(
                "ollama_json_parse_failed",
                agent=agent_name,
                model=model,
                attempt=attempt + 1,
                error=str(exc),
            )
            # Append retry instruction to messages for next attempt
            if attempt < MAX_STRUCTURAL_RETRIES:
                retry_msg = (
                    "Your previous response was invalid JSON. "
                    "You MUST respond with valid JSON. "
                    f"Parse error: {exc}"
                )
                request_body = {
                    **request_body,
                    "messages": [
                        *request_body["messages"],
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": retry_msg},
                    ],
                }

    raise OllamaResponseError(
        f"Failed to get valid JSON from {model} after "
        f"{1 + MAX_STRUCTURAL_RETRIES} attempts: {last_error}"
    )


def _model_sequence(primary: str, agent_name: str) -> list[str]:
    """Return [primary_model, fallback_model] if a fallback is configured."""
    settings = get_settings()
    fallback = settings.models.fallbacks.get(agent_name)
    if fallback and fallback != primary:
        return [primary, fallback]
    return [primary]
