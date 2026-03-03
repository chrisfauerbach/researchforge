"""Asyncio-based pub/sub event bus for pipeline SSE streaming."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import structlog

logger = structlog.get_logger()


class PipelineEventBus:
    """Publish/subscribe event bus for pipeline progress events.

    Each pipeline (job_id) can have multiple subscribers. Events are
    delivered via asyncio.Queue to each subscriber independently.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    def subscribe(self, job_id: str) -> asyncio.Queue:
        """Subscribe to events for a given pipeline job.

        Returns an asyncio.Queue that will receive event dicts.
        """
        if job_id not in self._subscribers:
            self._subscribers[job_id] = []
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers[job_id].append(queue)
        logger.debug("event_bus_subscribe", job_id=job_id)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        """Remove a subscriber's queue."""
        if job_id in self._subscribers:
            try:
                self._subscribers[job_id].remove(queue)
            except ValueError:
                pass
            if not self._subscribers[job_id]:
                del self._subscribers[job_id]

    async def publish(self, job_id: str, event: dict) -> None:
        """Publish an event to all subscribers of a pipeline job."""
        event.setdefault("timestamp", datetime.now(UTC).isoformat())
        event.setdefault("job_id", job_id)

        queues = self._subscribers.get(job_id, [])
        for queue in queues:
            await queue.put(event)

        logger.debug(
            "event_bus_publish",
            job_id=job_id,
            event_type=event.get("type"),
            subscribers=len(queues),
        )

    async def publish_stage_start(
        self, job_id: str, agent: str, model: str
    ) -> None:
        """Convenience: publish a stage_start event."""
        await self.publish(job_id, {
            "type": "stage_start",
            "agent": agent,
            "model": model,
        })

    async def publish_stage_complete(
        self, job_id: str, agent: str, duration_ms: int, status: str = "success"
    ) -> None:
        """Convenience: publish a stage_complete event."""
        await self.publish(job_id, {
            "type": "stage_complete",
            "agent": agent,
            "duration_ms": duration_ms,
            "status": status,
        })

    async def publish_error(
        self, job_id: str, agent: str, error: str
    ) -> None:
        """Convenience: publish an error event."""
        await self.publish(job_id, {
            "type": "error",
            "agent": agent,
            "error": error,
        })

    async def publish_complete(
        self, job_id: str, briefing_id: str | None = None
    ) -> None:
        """Convenience: publish pipeline_complete and signal end of stream."""
        await self.publish(job_id, {
            "type": "pipeline_complete",
            "briefing_id": briefing_id or job_id,
        })

    def format_sse(self, event: dict) -> str:
        """Format an event dict as an SSE message string."""
        event_type = event.get("type", "message")
        data = json.dumps(event)
        return f"event: {event_type}\ndata: {data}\n\n"


# Module-level singleton
_event_bus: PipelineEventBus | None = None


def get_event_bus() -> PipelineEventBus:
    """Get or create the global event bus singleton."""
    global _event_bus
    if _event_bus is None:
        _event_bus = PipelineEventBus()
    return _event_bus
