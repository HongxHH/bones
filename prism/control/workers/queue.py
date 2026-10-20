from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional, Protocol

from ..schemas import WorkerTask

log = logging.getLogger("bones.server.queue")


class QueueBackend(Protocol):
    async def enqueue(self, task: WorkerTask) -> None: ...
    async def claim(self, worker_labels: list[str]) -> Optional[WorkerTask]: ...
    async def ack(self, task_id: str, result: dict) -> None: ...
    async def fail(self, task_id: str, error: str) -> None: ...
    def pending_count(self) -> int: ...


class InProcessQueue:
    """Default queue. Swap for RedisQueue without changing orchestrator."""

    def __init__(self) -> None:
        self._pending: list[WorkerTask] = []
        self._inflight: dict[str, WorkerTask] = {}
        self._results: dict[str, dict] = {}
        self._cv = asyncio.Condition()

    async def enqueue(self, task: WorkerTask) -> None:
        async with self._cv:
            self._pending.append(task)
            self._cv.notify_all()

    async def claim(self, worker_labels: list[str]) -> Optional[WorkerTask]:
        labels = set(worker_labels)
        async with self._cv:
            for i, task in enumerate(self._pending):
                required = set(task.required_labels or [])
                if required and not required.issubset(labels):
                    continue
                self._pending.pop(i)
                self._inflight[task.id] = task
                return task
            return None

    async def ack(self, task_id: str, result: dict) -> None:
        async with self._cv:
            self._inflight.pop(task_id, None)
            self._results[task_id] = {"ok": True, "result": result}

    async def fail(self, task_id: str, error: str) -> None:
        async with self._cv:
            self._inflight.pop(task_id, None)
            self._results[task_id] = {"ok": False, "error": error}

    def pending_count(self) -> int:
        return len(self._pending)


class RedisQueue:
    """Redis-backed queue for multi-machine workers.

    Keys:
      prism:tasks:pending  — LIST of task JSON
      prism:tasks:inflight — HASH task_id -> task JSON
    """

    def __init__(self, url: str):
        self.url = url
        self._redis = None
        self._pending_key = "prism:tasks:pending"
        self._inflight_key = "prism:tasks:inflight"
        self._local_inflight: dict[str, WorkerTask] = {}

    async def _client(self):
        if self._redis is None:
            try:
                import redis.asyncio as redis  # type: ignore
            except ImportError as exc:
                raise RuntimeError("redis package required for RedisQueue; pip install redis") from exc
            self._redis = redis.from_url(self.url, decode_responses=True)
        return self._redis

    async def enqueue(self, task: WorkerTask) -> None:
        client = await self._client()
        await client.rpush(self._pending_key, task.model_dump_json())

    async def claim(self, worker_labels: list[str]) -> Optional[WorkerTask]:
        """Pop until match; restore non-matching in original relative order (no busy reshuffle)."""
        client = await self._client()
        labels = set(worker_labels)
        length = int(await client.llen(self._pending_key) or 0)
        skipped: list[str] = []
        chosen: Optional[WorkerTask] = None
        chosen_raw: Optional[str] = None
        for _ in range(length):
            raw = await client.lpop(self._pending_key)
            if raw is None:
                break
            task = WorkerTask.model_validate_json(raw)
            required = set(task.required_labels or [])
            if required and not required.issubset(labels):
                skipped.append(raw)
                continue
            chosen = task
            chosen_raw = raw
            break
        if skipped:
            # Preserve order of skipped tasks at the head of the queue.
            await client.lpush(self._pending_key, *reversed(skipped))
        if chosen is None or chosen_raw is None:
            return None
        await client.hset(self._inflight_key, chosen.id, chosen_raw)
        self._local_inflight[chosen.id] = chosen
        return chosen

    async def ack(self, task_id: str, result: dict) -> None:
        client = await self._client()
        await client.hdel(self._inflight_key, task_id)
        self._local_inflight.pop(task_id, None)

    async def fail(self, task_id: str, error: str) -> None:
        client = await self._client()
        await client.hdel(self._inflight_key, task_id)
        self._local_inflight.pop(task_id, None)
        log.warning("Redis task %s failed: %s", task_id, error)

    def pending_count(self) -> int:
        # Best-effort sync peek; supervisor mainly uses async claim.
        try:
            if self._redis is None:
                return 0
            # redis asyncio client: llen is awaitable — cannot call sync here.
            return -1
        except Exception:
            return -1


def build_queue(redis_url: str | None) -> QueueBackend:
    if redis_url:
        log.info("Using RedisQueue at %s", redis_url)
        return RedisQueue(redis_url)
    return InProcessQueue()
