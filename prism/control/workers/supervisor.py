from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from ..orchestrator.state import PlatformState
from ..schemas import WorkerInfo, WorkerStatus
from .handlers import TaskHandlers, finish_job

log = logging.getLogger("bones.server.workers")


class WorkerSupervisor:
    """Local worker pool. Remote workers claim via /v1/workers/{id}/claim."""

    def __init__(self, state: PlatformState, local_slots: list[dict] | None = None):
        self.state = state
        self.handlers = TaskHandlers(state)
        self.local_slots = local_slots or [
            {"id": "local-cpu", "labels": ["cpu", "package"]},
            {"id": "local-transcode", "labels": ["cpu", "transcode"]},
            {"id": "local-measure", "labels": ["cpu", "measure"]},
        ]
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()

    async def start(self) -> None:
        for slot in self.local_slots:
            worker = WorkerInfo(
                id=slot["id"],
                labels=list(slot["labels"]),
                capacity=1,
                status=WorkerStatus.ONLINE,
                extra={"kind": "local"},
            )
            await self.state.register_worker(worker)
            self._tasks.append(asyncio.create_task(self._loop(worker), name=f"worker-{worker.id}"))
        self._tasks.append(asyncio.create_task(self._reaper(), name="worker-reaper"))

    async def stop(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _loop(self, worker: WorkerInfo) -> None:
        while not self._stop.is_set():
            task = await self.state.queue.claim(worker.labels)
            if task is None:
                await asyncio.sleep(self.state.config.worker_poll_s)
                continue
            worker.inflight += 1
            worker.last_heartbeat = datetime.now(timezone.utc)
            try:
                result = await self.handlers.dispatch(task)
                await self.state.queue.ack(task.id, result)
                await finish_job(self.state, task, result, None)
            except Exception as exc:
                log.exception("Worker %s failed task %s", worker.id, task.id)
                await self.state.queue.fail(task.id, str(exc))
                await finish_job(self.state, task, {}, str(exc))
            finally:
                worker.inflight = max(0, worker.inflight - 1)

    async def _reaper(self) -> None:
        while not self._stop.is_set():
            self.state.mark_stale_workers()
            await asyncio.sleep(5)


async def claim_and_run_remote(state: PlatformState, worker_id: str) -> dict | None:
    worker = state.workers.get(worker_id)
    if worker is None or worker.status != WorkerStatus.ONLINE:
        return None
    task = await state.queue.claim(worker.labels)
    if task is None:
        return None
    return task.model_dump(mode="json")
