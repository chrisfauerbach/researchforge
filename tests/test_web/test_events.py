"""Tests for the pipeline event bus."""

import asyncio
import json

from researchforge.web.events import PipelineEventBus


class TestPipelineEventBus:
    async def test_subscribe_and_publish(self):
        bus = PipelineEventBus()
        queue = bus.subscribe("job-1")

        await bus.publish("job-1", {"type": "stage_start", "agent": "planner"})

        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event["type"] == "stage_start"
        assert event["agent"] == "planner"
        assert "timestamp" in event
        assert event["job_id"] == "job-1"

    async def test_multiple_subscribers(self):
        bus = PipelineEventBus()
        q1 = bus.subscribe("job-1")
        q2 = bus.subscribe("job-1")

        await bus.publish("job-1", {"type": "test"})

        e1 = await asyncio.wait_for(q1.get(), timeout=1.0)
        e2 = await asyncio.wait_for(q2.get(), timeout=1.0)
        assert e1["type"] == "test"
        assert e2["type"] == "test"

    async def test_different_jobs_isolated(self):
        bus = PipelineEventBus()
        q1 = bus.subscribe("job-1")
        q2 = bus.subscribe("job-2")

        await bus.publish("job-1", {"type": "for_job_1"})

        event = await asyncio.wait_for(q1.get(), timeout=1.0)
        assert event["type"] == "for_job_1"
        assert q2.empty()

    async def test_unsubscribe(self):
        bus = PipelineEventBus()
        queue = bus.subscribe("job-1")
        bus.unsubscribe("job-1", queue)

        await bus.publish("job-1", {"type": "test"})
        assert queue.empty()

    async def test_unsubscribe_nonexistent(self):
        bus = PipelineEventBus()
        queue = asyncio.Queue()
        # Should not raise
        bus.unsubscribe("job-1", queue)

    async def test_publish_stage_start(self):
        bus = PipelineEventBus()
        queue = bus.subscribe("job-1")

        await bus.publish_stage_start("job-1", "analyst", "qwen2.5:14b")

        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event["type"] == "stage_start"
        assert event["agent"] == "analyst"
        assert event["model"] == "qwen2.5:14b"

    async def test_publish_stage_complete(self):
        bus = PipelineEventBus()
        queue = bus.subscribe("job-1")

        await bus.publish_stage_complete("job-1", "planner", 5000)

        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event["type"] == "stage_complete"
        assert event["duration_ms"] == 5000

    async def test_publish_error(self):
        bus = PipelineEventBus()
        queue = bus.subscribe("job-1")

        await bus.publish_error("job-1", "gatherer", "Connection refused")

        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event["type"] == "error"
        assert event["error"] == "Connection refused"

    async def test_publish_complete(self):
        bus = PipelineEventBus()
        queue = bus.subscribe("job-1")

        await bus.publish_complete("job-1", briefing_id="b-123")

        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event["type"] == "pipeline_complete"
        assert event["briefing_id"] == "b-123"

    async def test_format_sse(self):
        bus = PipelineEventBus()
        event = {"type": "stage_start", "agent": "planner"}
        sse = bus.format_sse(event)

        assert sse.startswith("event: stage_start\n")
        assert "data: " in sse
        assert sse.endswith("\n\n")
        data_line = sse.split("data: ")[1].strip()
        parsed = json.loads(data_line)
        assert parsed["agent"] == "planner"

    async def test_multiple_events_in_order(self):
        bus = PipelineEventBus()
        queue = bus.subscribe("job-1")

        await bus.publish("job-1", {"type": "a", "order": 1})
        await bus.publish("job-1", {"type": "b", "order": 2})
        await bus.publish("job-1", {"type": "c", "order": 3})

        events = []
        for _ in range(3):
            events.append(await asyncio.wait_for(queue.get(), timeout=1.0))

        assert [e["order"] for e in events] == [1, 2, 3]
