from __future__ import annotations

from typing import Any

from ..orchestrator.state import PlatformState, new_id
from ..schemas import JobStatus, PackageResult, Publication, RecipeOutput, WorkerTask


class TaskHandlers:
    def __init__(self, state: PlatformState):
        self.state = state

    async def dispatch(self, task: WorkerTask) -> dict[str, Any]:
        job = self.state.jobs[task.job_id]
        job.status = JobStatus.RUNNING
        await self.state.update_job(job)
        if task.step == "package":
            result = await self.step_package(task)
        elif task.step == "measure":
            result = await self.step_measure(task)
        elif task.step == "apply_policy":
            result = await self.step_apply_policy(task)
        else:
            result = await self.step_produce(task)
        return result

    async def step_produce(self, task: WorkerTask) -> dict[str, Any]:
        job = self.state.jobs[task.job_id]
        asset = self.state.assets[job.asset_id]
        staging = self.state.store.staging_dir(job.id)
        recipe = self.state.registry.recipe(job.plugin)
        artifact = await recipe.run(asset_path=asset.source_path, staging_dir=str(staging), params=job.params)
        dumped = artifact.model_dump(mode="json")
        job.artifacts.update(dumped)
        await self.state.update_job(job)
        return dumped

    async def step_package(self, task: WorkerTask) -> dict[str, Any]:
        job = self.state.jobs[task.job_id]
        packager = self.state.registry.packager(task.plugin)
        pub_id = job.publication_id or new_id("pub")
        output_dir = self.state.store.publication_dir(pub_id)
        artifact = RecipeOutput.model_validate(job.artifacts)
        result: PackageResult = await packager.publish(
            artifact=artifact,
            output_dir=str(output_dir),
            params={**job.params, "title": job.params.get("title", pub_id)},
        )
        dumped = result.model_dump(mode="json")
        pub = Publication(
            id=pub_id,
            asset_id=job.asset_id or "",
            title=job.params.get("title") or pub_id,
            mpd_name=result.mpd_name,
            ladder=result.ladder,
            policy_tags=list(job.params.get("policy_tags") or []),
            segment_duration=result.segment_duration,
            extra={"duration_s": result.duration_s},
        )
        await self.state.add_publication(pub)
        job.publication_id = pub.id
        job.artifacts.update(dumped)
        job.artifacts["publication_id"] = pub.id
        await self.state.update_job(job)
        if job.experiment_id and job.experiment_id in self.state.experiments:
            exp = self.state.experiments[job.experiment_id]
            if pub.id not in exp.publication_ids:
                exp.publication_ids.append(pub.id)
                self.state.persist()
        return dumped

    async def step_measure(self, task: WorkerTask) -> dict[str, Any]:
        job = self.state.jobs[task.job_id]
        measurer = self.state.registry.measurer(job.plugin)
        pub_id = job.publication_id
        pub_dir = str(self.state.store.publication_dir(pub_id)) if pub_id else str(self.state.store.staging_dir(job.id))
        result = await measurer.measure(publication_dir=pub_dir, params=job.params)
        job.artifacts.update(result)
        await self.state.update_job(job)
        return result

    async def step_apply_policy(self, task: WorkerTask) -> dict[str, Any]:
        job = self.state.jobs[task.job_id]
        session = self.state.sessions.get(job.session_id or "")
        if session is None:
            raise KeyError("Session job missing session_id")
        delivery = session.extra.get("delivery") or job.params.get("delivery") or "session_filter"
        adapter = self.state.registry.delivery_adapter(delivery)
        pub = self.state.publications[session.publication_id]
        mpd_path = self.state.store.publication_dir(pub.id) / pub.mpd_name
        xml = mpd_path.read_text(encoding="utf-8")
        resolved = adapter.resolve_mpd(
            mpd_xml=xml,
            policy=session.policy.model_dump(),
            public_base=self.state.config.public_base_url,
            session_id=session.id,
            publication_id=pub.id,
        )
        job.artifacts["session_mpd"] = resolved
        await self.state.update_job(job)
        return job.artifacts


async def finish_job(state: PlatformState, task: WorkerTask, result: dict, error: str | None) -> None:
    job = state.jobs[task.job_id]
    if error:
        job.status = JobStatus.FAILED
        job.error = error
        await state.update_job(job)
        await _run_hooks(state, job)
        return
    job.artifacts.update(result or {})
    if job.step_index + 1 < len(job.steps):
        job.step_index += 1
        job.status = JobStatus.QUEUED
        await state.update_job(job)
        await state.enqueue_job(job)
        return
    if job.artifacts.get("status") == "skipped":
        job.status = JobStatus.SKIPPED
    else:
        job.status = JobStatus.SUCCEEDED
    await state.update_job(job)
    await _run_hooks(state, job)
    from ..metrics.bundle import ingest_measure_job

    ingest_measure_job(state, job)


async def _run_hooks(state: PlatformState, job) -> None:
    if not job.experiment_id:
        return
    exp = state.experiments.get(job.experiment_id)
    if exp is None:
        return
    exp_dir = exp.artifact_dir or str(state.store.experiment_dir(exp.id))
    for hook in state.registry.hooks.values():
        await hook.on_job_finished(job=job.model_dump(mode="json"), artifacts=job.artifacts, experiment_dir=exp_dir)
