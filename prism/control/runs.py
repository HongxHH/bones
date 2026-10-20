from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from prism.paths import repo_root

from .attach import STATUS_PORT_BASE, kill_pid, player_cmd, spawn_player
from .experiment_spec import (
    ArmSpec,
    ExperimentSpec,
    parse_experiment_spec,
    resolve_content_source,
)
from .frames import claim_current_frames
from .orchestrator.state import PlatformState, new_id
from .schemas import (
    ArmRuntime,
    Asset,
    DeliveryPolicy,
    Experiment,
    ExperimentRun,
    Job,
    JobKind,
    JobStatus,
    RunStatus,
    Session,
    SessionStatus,
    utcnow,
)

log = logging.getLogger("prism.runs")

REPO_ROOT = repo_root()

# identity aliases import_dash for DASH trees / existing publications
IDENTITY_PLUGINS = frozenset({"identity", "", "import_dash"})


async def _wait_job(state: PlatformState, job_id: str, timeout_s: float = 600.0) -> Job:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = state.jobs.get(job_id)
        if job is None:
            raise RuntimeError(f"job {job_id} missing")
        if job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.SKIPPED):
            return job
        await asyncio.sleep(0.05)
    raise TimeoutError(f"job {job_id} timed out")


def _validate_publish_plugins(state: PlatformState, spec: ExperimentSpec) -> None:
    for arm_spec in spec.spec.arms:
        plugin = arm_spec.publish.plugin or "identity"
        if plugin in IDENTITY_PLUGINS:
            continue
        if plugin not in state.registry.recipes:
            raise ValueError(
                f"arm {arm_spec.id}: unknown publish.plugin={plugin!r}. "
                f"Known recipes: {sorted(state.registry.recipes)}"
            )


def _abs_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _publish_cache_key(plugin: str, asset_path: str, params: dict[str, Any]) -> str:
    blob = json.dumps({"plugin": plugin, "asset": asset_path, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


async def _run_content_job(
    state: PlatformState,
    *,
    plugin: str,
    asset: Asset,
    params: dict[str, Any],
    experiment_id: str,
) -> str:
    steps, labels = state.registry.pipeline_for(plugin, JobKind.CONTENT)
    job = Job(
        id=new_id("job"),
        kind=JobKind.CONTENT,
        steps=steps,
        plugin=plugin,
        params=params,
        asset_id=asset.id,
        experiment_id=experiment_id,
        required_labels=labels,
    )
    await state.add_job(job)
    await state.enqueue_job(job)
    done = await _wait_job(state, job.id)
    if done.status != JobStatus.SUCCEEDED or not done.publication_id:
        raise RuntimeError(f"publish job {job.id} failed: {done.error or done.status}")
    return done.publication_id


async def resolve_publication_id(state: PlatformState, source: str, *, experiment_id: str) -> str:
    """Resolve identity/import_dash publication from publication: id or DASH path."""
    kind, value = resolve_content_source(source)
    if kind == "publication":
        if value not in state.publications:
            raise ValueError(f"publication not found: {value}")
        return value
    path = _abs_path(value)
    if not path.exists():
        raise ValueError(f"content path not found: {path}")
    asset = Asset(id=new_id("asset"), name=path.name, source_path=str(path), kind="dash_tree")
    await state.add_asset(asset)
    return await _run_content_job(
        state,
        plugin="import_dash",
        asset=asset,
        params={"title": path.name},
        experiment_id=experiment_id,
    )


async def resolve_arm_publication(
    state: PlatformState,
    *,
    content_source: str,
    arm: ArmSpec,
    experiment_id: str,
    cache: dict[str, str],
) -> str:
    """Per-arm publication. identity imports an existing DASH tree."""
    plugin = arm.publish.plugin or "identity"
    params = dict(arm.publish.params or {})

    if plugin in IDENTITY_PLUGINS:
        source = params.get("source") or content_source
        key = f"identity:{source}"
        if key in cache:
            return cache[key]
        pub_id = await resolve_publication_id(state, source, experiment_id=experiment_id)
        cache[key] = pub_id
        return pub_id

    # Recipe arm: need a media file (mp4/mkv/…)
    source = params.pop("source", None) or content_source
    kind, value = resolve_content_source(source)
    if kind == "publication":
        pub = state.publications.get(value)
        if pub is None:
            raise ValueError(f"arm {arm.id}: publication {value} not found for recipe source")
        asset = state.assets.get(pub.asset_id)
        if asset is None:
            raise ValueError(f"arm {arm.id}: publication {value} has no source asset for re-encode")
        asset_path = asset.source_path
    else:
        path = _abs_path(value)
        if not path.exists():
            raise ValueError(f"arm {arm.id}: source not found: {path}")
        if path.is_dir() or path.suffix.lower() == ".mpd":
            raise ValueError(
                f"arm {arm.id}: publish.plugin={plugin!r} needs a media file (mp4/…), "
                f"not a DASH tree. Set content.source or publish.params.source to a video file "
                f"(e.g. path:player/data/video_mp4box/output_720p.mp4)."
            )
        asset_path = str(path)

    cache_key = _publish_cache_key(plugin, asset_path, params)
    if cache_key in cache:
        return cache[cache_key]

    asset = Asset(
        id=new_id("asset"),
        name=Path(asset_path).name,
        source_path=asset_path,
        kind="source",
    )
    await state.add_asset(asset)
    job_params = {**params, "title": params.get("title") or f"{arm.id}-{Path(asset_path).stem}"}
    pub_id = await _run_content_job(
        state,
        plugin=plugin,
        asset=asset,
        params=job_params,
        experiment_id=experiment_id,
    )
    cache[cache_key] = pub_id
    return pub_id


async def ensure_experiment_for_spec(state: PlatformState, spec: ExperimentSpec) -> Experiment:
    exp_id = new_id("exp")
    exp = Experiment(
        id=exp_id,
        name=spec.name(),
        notes=spec.notes(),
        params={"spec": spec.model_dump(mode="json")},
        artifact_dir=str(state.store.experiment_dir(exp_id)),
    )
    return await state.add_experiment(exp)


async def _materialize_arms(
    state: PlatformState,
    *,
    run: ExperimentRun,
    exp: Experiment,
    spec: ExperimentSpec,
    renderer_override: Optional[str],
) -> list[ArmRuntime]:
    arms: list[ArmRuntime] = []
    pub_cache: dict[str, str] = {}
    for i, arm_spec in enumerate(spec.spec.arms):
        pub_id = await resolve_arm_publication(
            state,
            content_source=spec.spec.content.source,
            arm=arm_spec,
            experiment_id=exp.id,
            cache=pub_cache,
        )
        if pub_id not in exp.publication_ids:
            exp.publication_ids.append(pub_id)
        policy = arm_spec.session.policy
        session = Session(
            id=new_id("sess"),
            publication_id=pub_id,
            policy=policy if isinstance(policy, DeliveryPolicy) else DeliveryPolicy.model_validate(policy),
            extra={
                "experiment_id": exp.id,
                "run_id": run.id,
                "arm_id": arm_spec.id,
                "delivery": "session_filter",
                "policy_version": 0,
                "publish_plugin": arm_spec.publish.plugin or "identity",
            },
        )
        await state.add_session(session)
        if session.id not in exp.session_ids:
            exp.session_ids.append(session.id)
        mpd_url = state.origin_session_url(session.id)
        player = arm_spec.player.model_dump()
        if renderer_override:
            player["mod_renderer"] = renderer_override
        # Wire ExperimentSpec.collect → player CLI (interval + host sampling).
        collect = spec.spec.collect
        player["telemetry_interval_ms"] = int(collect.interval_ms)
        player["collect_system_metrics"] = bool(collect.system_metrics)
        port = STATUS_PORT_BASE + i
        cmd = player_cmd(
            mpd_url,
            player,
            port,
            run_id=run.id,
            arm_id=arm_spec.id,
            session_id=session.id,
            experiment_id=exp.id,
            server_url=state.config.public_base_url,
            frame_shm=getattr(state, "frame_shm", "") or None,
            frame_archive=str(claim_current_frames(state, run.id, arm_spec.id)),
        )
        arms.append(
            ArmRuntime(
                id=arm_spec.id,
                session_id=session.id,
                publication_id=pub_id,
                mpd_url=mpd_url,
                status="ready",
                status_port=port,
                player_cmd=" ".join(cmd),
                player=player,
            )
        )
    return arms


async def _prepare_run(
    state: PlatformState,
    run_id: str,
    *,
    attach_players: bool,
    renderer_override: Optional[str],
) -> None:
    run = state.runs.get(run_id)
    if run is None:
        return
    try:
        spec = parse_experiment_spec(run.spec)
        exp = state.experiments[run.experiment_id]
        arms = await _materialize_arms(
            state,
            run=run,
            exp=exp,
            spec=spec,
            renderer_override=renderer_override,
        )
        run = state.runs[run_id]
        run.arms = arms
        run.status = RunStatus.RUNNING
        run.started_at = utcnow()
        if arms:
            run.active_arm_id = arms[0].id
        run.error = None
        await state.update_run(run)
        await state.add_experiment(exp)
        if attach_players:
            if run.arm_mode == "parallel":
                for arm in run.arms:
                    await attach_arm(state, run.id, arm.id, enforce_security=False)
            else:
                await attach_arm(state, run.id, run.active_arm_id or run.arms[0].id, enforce_security=False)
    except Exception as exc:
        log.exception("prepare run %s failed", run_id)
        run = state.runs.get(run_id)
        if run is None:
            return
        run.status = RunStatus.FAILED
        run.error = str(exc)
        run.updated_at = utcnow()
        await state.update_run(run)


async def create_run_from_spec(
    state: PlatformState,
    raw_spec: dict[str, Any] | ExperimentSpec,
    *,
    attach_players: bool = False,
    renderer_override: Optional[str] = None,
    experiment_id: Optional[str] = None,
    wait: bool = True,
) -> ExperimentRun:
    spec = parse_experiment_spec(raw_spec)
    _validate_publish_plugins(state, spec)
    if experiment_id:
        exp = state.experiments.get(experiment_id)
        if exp is None:
            raise ValueError("experiment not found")
        exp.params["spec"] = spec.model_dump(mode="json")
        await state.add_experiment(exp)
    else:
        exp = await ensure_experiment_for_spec(state, spec)

    run = ExperimentRun(
        id=new_id("run"),
        experiment_id=exp.id,
        name=spec.name(),
        status=RunStatus.PREPARING,
        arm_mode=spec.spec.execution.arm_mode,
        spec=spec.model_dump(mode="json"),
        arms=[],
    )
    await state.add_run(run)
    if run.id not in exp.run_ids:
        exp.run_ids.append(run.id)
        await state.add_experiment(exp)

    if wait:
        await _prepare_run(
            state, run.id, attach_players=attach_players, renderer_override=renderer_override
        )
        return state.runs[run.id]
    # Non-blocking: API schedules prepare_run via FastAPI BackgroundTasks.
    state.pending_prepares[run.id] = {
        "attach_players": attach_players,
        "renderer_override": renderer_override,
    }
    return run


async def prepare_run(
    state: PlatformState,
    run_id: str,
    *,
    attach_players: bool = False,
    renderer_override: Optional[str] = None,
) -> None:
    await _prepare_run(
        state, run_id, attach_players=attach_players, renderer_override=renderer_override
    )


async def attach_arm(
    state: PlatformState,
    run_id: str,
    arm_id: str,
    *,
    enforce_security: bool = True,
    request=None,
) -> ExperimentRun:
    if enforce_security:
        from .security import assert_attach_allowed

        if request is None:
            raise PermissionError("attach requires Request for security check")
        assert_attach_allowed(request)

    run = state.runs.get(run_id)
    if run is None:
        raise ValueError("run not found")
    if run.status == RunStatus.PREPARING:
        raise ValueError("run is still preparing")
    arm = next((a for a in run.arms if a.id == arm_id), None)
    if arm is None:
        raise ValueError("arm not found")
    if not arm.mpd_url:
        raise ValueError("arm has no mpd_url")
    if arm.pid:
        kill_pid(arm.pid)
        arm.pid = None
    port = arm.status_port or STATUS_PORT_BASE
    cmd = player_cmd(
        arm.mpd_url,
        arm.player,
        port,
        run_id=run.id,
        arm_id=arm.id,
        session_id=arm.session_id,
        experiment_id=run.experiment_id,
        server_url=state.config.public_base_url,
        frame_shm=getattr(state, "frame_shm", "") or None,
        frame_archive=str(claim_current_frames(state, run.id, arm.id)),
    )
    from .attach import PLAYER_LOG_DIR, pid_alive, tail_log

    log_path = PLAYER_LOG_DIR / f"{run.id}_{arm.id}.log"
    try:
        pid = spawn_player(cmd, log_path=log_path)
        arm.pid = pid
        arm.player_cmd = " ".join(cmd)
        # Trace/HTTP misconfig and quality-table crashes often exit within <1s.
        await asyncio.sleep(0.9)
        if not pid_alive(pid):
            arm.status = "failed"
            detail = tail_log(log_path) or "player exited immediately (no log)"
            arm.error = detail
            arm.pid = None
        else:
            arm.status = "attached"
            arm.error = None
    except Exception as exc:
        arm.status = "failed"
        arm.error = str(exc)
    run.active_arm_id = arm_id
    run.updated_at = utcnow()
    await state.update_run(run)
    return run


async def detach_arm(state: PlatformState, run_id: str, arm_id: str, *, request=None) -> ExperimentRun:
    if request is not None:
        from .security import assert_attach_allowed

        assert_attach_allowed(request)
    run = state.runs.get(run_id)
    if run is None:
        raise ValueError("run not found")
    arm = next((a for a in run.arms if a.id == arm_id), None)
    if arm is None:
        raise ValueError("arm not found")
    if arm.pid:
        kill_pid(arm.pid)
        arm.pid = None
    if arm.status == "attached":
        arm.status = "ready"
    run.updated_at = utcnow()
    await state.update_run(run)
    return run


async def advance_serial(state: PlatformState, run_id: str, *, request=None) -> ExperimentRun:
    run = state.runs.get(run_id)
    if run is None:
        raise ValueError("run not found")
    if run.arm_mode != "serial":
        raise ValueError("advance only valid for serial arm_mode")
    ids = [a.id for a in run.arms]
    if not ids:
        raise ValueError("no arms")
    cur = run.active_arm_id or ids[0]
    idx = ids.index(cur) if cur in ids else 0
    if idx + 1 >= len(ids):
        raise ValueError("already on last arm")
    await detach_arm(state, run_id, cur, request=request)
    return await attach_arm(state, run_id, ids[idx + 1], request=request)


async def _enqueue_measure_jobs(state: PlatformState, run: ExperimentRun) -> list[str]:
    spec_root = run.spec.get("spec") or {}
    measures = spec_root.get("measure") or []
    if not measures:
        return []
    pub_id = next((a.publication_id for a in run.arms if a.publication_id), None)
    if not pub_id:
        return []
    job_ids: list[str] = []
    for item in measures:
        plugin = (item or {}).get("plugin") or "noop"
        params = dict((item or {}).get("params") or {})
        params.setdefault("run_id", run.id)
        try:
            steps, labels = state.registry.pipeline_for(plugin, JobKind.CONTENT)
        except Exception:
            steps, labels = ["measure"], ["cpu", "measure"]
        if "measure" not in steps:
            steps = ["measure"]
        job = Job(
            id=new_id("job"),
            kind=JobKind.CONTENT,
            steps=steps,
            plugin=plugin,
            params=params,
            publication_id=pub_id,
            experiment_id=run.experiment_id,
            required_labels=labels,
        )
        await state.add_job(job)
        await state.enqueue_job(job)
        job_ids.append(job.id)
        log.info("enqueued measure job %s plugin=%s for run %s", job.id, plugin, run.id)
    return job_ids


async def stop_run(state: PlatformState, run_id: str, *, request=None) -> ExperimentRun:
    from .telemetry import persist_run_metrics

    if request is not None:
        # stopping may kill local pids
        from .security import assert_attach_allowed

        try:
            assert_attach_allowed(request)
        except PermissionError:
            # Still allow stop metrics / session close without kill if remote — kill best-effort
            pass

    run = state.runs.get(run_id)
    if run is None:
        raise ValueError("run not found")
    for arm in run.arms:
        if arm.pid:
            try:
                kill_pid(arm.pid)
            except Exception as exc:
                log.warning("kill pid %s: %s", arm.pid, exc)
            arm.pid = None
        if arm.status == "attached":
            arm.status = "stopped"
        if arm.session_id and arm.session_id in state.sessions:
            sess = state.sessions[arm.session_id]
            sess.status = SessionStatus.CLOSED
            await state.update_session(sess)
    metrics = persist_run_metrics(state, run)
    measure_jobs = await _enqueue_measure_jobs(state, run)
    if measure_jobs:
        metrics["measure_job_ids"] = measure_jobs
    run.metrics = metrics
    for arm in run.arms:
        arm.metrics = (metrics.get("arms") or {}).get(arm.id) or {}
    run.status = RunStatus.STOPPED
    run.stopped_at = utcnow()
    run.updated_at = utcnow()
    await state.update_run(run)
    return run
