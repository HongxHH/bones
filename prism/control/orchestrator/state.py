from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel

from ..config import ServerConfig
from ..registry import PluginRegistry
from ..schemas import (
    Asset,
    Experiment,
    ExperimentRun,
    Job,
    JobStatus,
    OriginAccessLog,
    Publication,
    Session,
    SessionStatus,
    WorkerInfo,
    WorkerStatus,
    WorkerTask,
    utcnow,
)
from ..store.local import LocalAssetStore
from ..workers.queue import InProcessQueue, QueueBackend


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class PlatformState:
    def __init__(self, config: ServerConfig, registry: PluginRegistry, queue: QueueBackend | None = None):
        self.config = config
        self.registry = registry
        self.store = LocalAssetStore(config)
        self.queue: QueueBackend = queue or InProcessQueue()
        self._lock = asyncio.Lock()
        self.assets: dict[str, Asset] = {}
        self.jobs: dict[str, Job] = {}
        self.publications: dict[str, Publication] = {}
        self.sessions: dict[str, Session] = {}
        self.experiments: dict[str, Experiment] = {}
        self.runs: dict[str, ExperimentRun] = {}
        self.workers: dict[str, WorkerInfo] = {}
        self.origin_logs: list[OriginAccessLog] = []
        self.pending_prepares: dict[str, dict] = {}
        self.frame_shm = (os.environ.get("PRISM_FRAME_SHM") or "").strip()
        self.picture_focus: dict | None = None
        self.picture_seq = 0
        self._load()

    def _load(self) -> None:
        for item in self.store.load_collection("assets", Asset):
            self.assets[item.id] = item
        for item in self.store.load_collection("jobs", Job):
            self.jobs[item.id] = item
        for item in self.store.load_collection("publications", Publication):
            self.publications[item.id] = item
        for item in self.store.load_collection("sessions", Session):
            self.sessions[item.id] = item
        for item in self.store.load_collection("experiments", Experiment):
            self.experiments[item.id] = item
        for item in self.store.load_collection("runs", ExperimentRun):
            self.runs[item.id] = item

    def _touch(self, collection: str, item: BaseModel) -> None:
        self.store.save_entity(collection, item)

    def persist(self) -> None:
        """Full rewrite (migration / rare). Prefer _touch for hot paths."""
        self.store.save_list("assets", list(self.assets.values()))
        self.store.save_list("jobs", list(self.jobs.values()))
        self.store.save_list("publications", list(self.publications.values()))
        self.store.save_list("sessions", list(self.sessions.values()))
        self.store.save_list("experiments", list(self.experiments.values()))
        self.store.save_list("runs", list(self.runs.values()))

    async def add_asset(self, asset: Asset) -> Asset:
        async with self._lock:
            self.assets[asset.id] = asset
            self._touch("assets", asset)
            return asset

    async def add_job(self, job: Job) -> Job:
        async with self._lock:
            self.jobs[job.id] = job
            self._touch("jobs", job)
            if job.experiment_id and job.experiment_id in self.experiments:
                exp = self.experiments[job.experiment_id]
                if job.id not in exp.job_ids:
                    exp.job_ids.append(job.id)
                    self._touch("experiments", exp)
            return job

    async def update_job(self, job: Job) -> Job:
        async with self._lock:
            job.updated_at = utcnow()
            self.jobs[job.id] = job
            self._touch("jobs", job)
            return job

    async def add_publication(self, pub: Publication) -> Publication:
        async with self._lock:
            self.publications[pub.id] = pub
            self._touch("publications", pub)
            return pub

    async def add_session(self, session: Session) -> Session:
        async with self._lock:
            if "policy_version" not in session.extra:
                session.extra["policy_version"] = 0
            self.sessions[session.id] = session
            self._touch("sessions", session)
            return session

    async def update_session(self, session: Session) -> Session:
        async with self._lock:
            session.updated_at = utcnow()
            self.sessions[session.id] = session
            self._touch("sessions", session)
            return session

    async def add_experiment(self, exp: Experiment) -> Experiment:
        async with self._lock:
            self.experiments[exp.id] = exp
            self._touch("experiments", exp)
            return exp

    async def add_run(self, run: ExperimentRun) -> ExperimentRun:
        async with self._lock:
            self.runs[run.id] = run
            self._touch("runs", run)
            return run

    async def update_run(self, run: ExperimentRun) -> ExperimentRun:
        async with self._lock:
            run.updated_at = utcnow()
            self.runs[run.id] = run
            self._touch("runs", run)
            return run

    async def register_worker(self, worker: WorkerInfo) -> WorkerInfo:
        async with self._lock:
            self.workers[worker.id] = worker
            return worker

    async def heartbeat_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        async with self._lock:
            worker = self.workers.get(worker_id)
            if worker is None:
                return None
            worker.last_heartbeat = utcnow()
            worker.status = WorkerStatus.ONLINE
            return worker

    def mark_stale_workers(self, timeout_s: float = 30.0) -> None:
        now = datetime.now(timezone.utc)
        for worker in self.workers.values():
            age = (now - worker.last_heartbeat).total_seconds()
            if age > timeout_s:
                worker.status = WorkerStatus.OFFLINE

    def log_origin(self, entry: OriginAccessLog) -> None:
        self.origin_logs.append(entry)
        if len(self.origin_logs) > 2000:
            self.origin_logs = self.origin_logs[-1000:]

    def publication_mpd_path(self, publication_id: str) -> Optional[str]:
        pub = self.publications.get(publication_id)
        if pub is None:
            return None
        return str(self.store.publication_dir(publication_id) / pub.mpd_name)

    def origin_publication_url(self, publication_id: str) -> str:
        pub = self.publications[publication_id]
        return f"{self.config.public_base_url}{self.config.origin_prefix}/publications/{publication_id}/{pub.mpd_name}"

    def origin_session_url(self, session_id: str) -> str:
        return f"{self.config.public_base_url}{self.config.origin_prefix}/sessions/{session_id}/manifest.mpd"

    async def enqueue_job(self, job: Job) -> WorkerTask:
        step = job.steps[job.step_index]
        plugin = job.plugin
        if step == "package":
            plugin = self.registry.packager_name_for(job.plugin, job.params)
        labels = self.registry.labels_for_step(job.plugin, step, job.params)
        task = WorkerTask(
            id=new_id("task"),
            job_id=job.id,
            step=step,
            plugin=plugin,
            params=job.params,
            required_labels=labels,
            payload={
                "asset_id": job.asset_id,
                "publication_id": job.publication_id,
                "session_id": job.session_id,
                "kind": job.kind.value,
            },
        )
        await self.queue.enqueue(task)
        return task
