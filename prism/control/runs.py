from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
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
from .plugins.mpdutil import parse_ladder_from_mpd
from .schemas import (
    ArmRuntime,
    Asset,
    DeliveryPolicy,
    Experiment,
    ExperimentRun,
    Job,
    JobKind,
    JobStatus,
    Publication,
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


def _stable_pub_id_for_path(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"pub_{digest}"


def _dash_root_and_mpd(path: Path) -> tuple[Path, str]:
    if path.is_file() and path.suffix.lower() == ".mpd":
        return path.parent.resolve(), path.name
    if path.is_dir():
        mpds = sorted(path.glob("*.mpd"), key=lambda p: (p.name.lower() != "bbb.mpd", p.name))
        if not mpds:
            raise FileNotFoundError(f"No MPD found under {path}")
        return path.resolve(), mpds[0].name
    raise FileNotFoundError(f"DASH source not found: {path}")


def _find_publication_for_path(state: PlatformState, path: Path) -> Optional[Publication]:
    resolved = path.resolve()
    stable_id = _stable_pub_id_for_path(resolved)
    existing = state.publications.get(stable_id)
    if existing is not None:
        root = state.publication_root(existing)
        if (root / existing.mpd_name).is_file():
            return existing
    for pub in state.publications.values():
        extra = pub.extra or {}
        for key in ("source_path", "serve_root"):
            raw = extra.get(key)
            if not raw:
                continue
            try:
                if Path(str(raw)).resolve() == resolved:
                    if (state.publication_root(pub) / pub.mpd_name).is_file():
                        return pub
            except OSError:
                continue
        asset = state.assets.get(pub.asset_id)
        if asset and asset.source_path:
            try:
                if Path(asset.source_path).resolve() == resolved:
                    if (state.publication_root(pub) / pub.mpd_name).is_file():
                        return pub
            except OSError:
                continue
    return None


async def _register_inplace_dash_publication(
    state: PlatformState,
    *,
    path: Path,
    experiment_id: str,
) -> str:
    """Bind a local DASH tree as a publication without copying into var/publications."""
    root, mpd_name = _dash_root_and_mpd(path)
    mpd_path = root / mpd_name
    ladder, duration = parse_ladder_from_mpd(mpd_path.read_text(encoding="utf-8"))
    pub_id = _stable_pub_id_for_path(path.resolve())
    existing = state.publications.get(pub_id)
    if existing is not None and (state.publication_root(existing) / existing.mpd_name).is_file():
        return pub_id

    asset = Asset(
        id=new_id("asset"),
        name=path.name,
        source_path=str(path.resolve()),
        kind="dash_tree",
        extra={"inplace": True},
    )
    await state.add_asset(asset)
    pub = Publication(
        id=pub_id,
        asset_id=asset.id,
        title=path.name,
        mpd_name=mpd_name,
        ladder=ladder,
        segment_duration=4.0,
        extra={
            "source_path": str(path.resolve()),
            "serve_root": str(root),
            "inplace": True,
            "duration_s": duration,
            "bound_by_experiment": experiment_id,
        },
    )
    await state.add_publication(pub)
    if experiment_id and experiment_id in state.experiments:
        exp = state.experiments[experiment_id]
        if pub.id not in exp.publication_ids:
            exp.publication_ids.append(pub.id)
            state.persist()
    return pub_id


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
    """Resolve identity/import_dash publication from publication: id or DASH path.

    Path sources are bound in-place (serve_root → original tree) and reused across
    experiments so Lab no longer accumulates copied pub_* trees for BBB.
    """
    kind, value = resolve_content_source(source)
    if kind == "publication":
        if value not in state.publications:
            raise ValueError(f"publication not found: {value}")
        return value
    path = _abs_path(value)
    if not path.exists():
        raise ValueError(f"content path not found: {path}")
    found = _find_publication_for_path(state, path)
    if found is not None:
        return found.id
    return await _register_inplace_dash_publication(state, path=path, experiment_id=experiment_id)


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
        from .telemetry import telemetry_dir

        player["metric_output"] = str(telemetry_dir(state, run).parent / "analyzer")
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

    dumped = spec.model_dump(mode="json")
    arm_player = spec.spec.arms[0].player if spec.spec.arms else None
    from .plugins.hooks import _git_rev

    dumped["provenance"] = {
        "git_sha": _git_rev() or "",
        "renderer": arm_player.mod_renderer if arm_player else "",
        "latency_adapt": bool(arm_player.latency_adapt) if arm_player else False,
        "use_fast_complete": bool(arm_player.use_fast_complete) if arm_player else False,
        "collect_system_metrics": bool(spec.spec.collect.system_metrics),
    }
    run = ExperimentRun(
        id=new_id("run"),
        experiment_id=exp.id,
        name=spec.name(),
        status=RunStatus.PREPARING,
        arm_mode=spec.spec.execution.arm_mode,
        spec=dumped,
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
    player = dict(arm.player or {})
    from .telemetry import telemetry_dir

    player["metric_output"] = str(telemetry_dir(state, run).parent / "analyzer")
    cmd = player_cmd(
        arm.mpd_url,
        player,
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
        quality = ((spec_root.get("metrics") or {}).get("quality") or {}).get("compute") or {}
        if plugin == "vmaf":
            if quality.get("reference") and not params.get("reference"):
                params["reference"] = quality["reference"]
            distorted = quality.get("distorted") or ""
            if distorted and distorted != "playback_frames" and not params.get("distorted"):
                params["distorted"] = distorted
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


# How long to let the player finish writing analyzer JSON before a forced kill.
FINISH_WAIT_S = 8.0
_finalizers: dict[str, asyncio.Task] = {}


async def _ask_player_stop(port: int) -> None:
    """Ask the player to stop itself so analysis can flush before we kill it."""
    def _post() -> None:
        import urllib.request

        req = urllib.request.Request(
            f"http://127.0.0.1:{int(port)}/api/stop",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=1.5):
                return
        except Exception:
            return

    await asyncio.to_thread(_post)


def schedule_finalize(state: PlatformState, run_id: str, *, ask_stop: bool) -> asyncio.Task:
    """One finisher per run. END telemetry must not await it (the player is inside that POST)."""
    current = _finalizers.get(run_id)
    if current is not None and not current.done():
        return current
    task = asyncio.create_task(finalize_run(state, run_id, ask_stop=ask_stop), name=f"finish-{run_id}")
    _finalizers[run_id] = task
    return task


def _run_has_arm_metrics(run: ExperimentRun) -> bool:
    arms = (run.metrics or {}).get("arms") if isinstance(run.metrics, dict) else None
    return isinstance(arms, dict) and bool(arms)


async def finalize_run(state: PlatformState, run_id: str, *, ask_stop: bool) -> None:
    """Wait for the player to exit and write analyzer JSON, then mark the run stopped."""
    from .attach import pid_alive
    from .telemetry import persist_run_metrics

    run = state.runs.get(run_id)
    if run is None:
        return
    # reconcile_dead_runs used to mark STOPPED without writing metrics; still finish those.
    if run.status == RunStatus.STOPPED and _run_has_arm_metrics(run):
        return
    if ask_stop and run.status != RunStatus.STOPPED:
        for arm in run.arms:
            if arm.pid and arm.status_port:
                await _ask_player_stop(int(arm.status_port))
    if run.status != RunStatus.STOPPED:
        deadline = time.monotonic() + FINISH_WAIT_S
        while time.monotonic() < deadline:
            if not any(arm.pid and pid_alive(arm.pid) for arm in run.arms):
                break
            await asyncio.sleep(0.25)
        else:
            for arm in run.arms:
                if arm.pid and pid_alive(arm.pid):
                    try:
                        kill_pid(arm.pid)
                    except Exception as exc:
                        log.warning("kill pid %s: %s", arm.pid, exc)
    run = state.runs.get(run_id)
    if run is None:
        return
    if run.status == RunStatus.STOPPED and _run_has_arm_metrics(run):
        return
    for arm in run.arms:
        arm.pid = None
        if arm.status == "attached":
            arm.status = "stopped"
        if arm.session_id and arm.session_id in state.sessions:
            sess = state.sessions[arm.session_id]
            sess.status = SessionStatus.CLOSED
            await state.update_session(sess)
    metrics = await asyncio.to_thread(persist_run_metrics, state, run)
    measure_jobs = await _enqueue_measure_jobs(state, run)
    if measure_jobs:
        metrics["measure_job_ids"] = measure_jobs
    run.metrics = metrics
    for arm in run.arms:
        arm.metrics = (metrics.get("arms") or {}).get(arm.id) or {}
    run.status = RunStatus.STOPPED
    if run.stopped_at is None:
        run.stopped_at = utcnow()
    run.updated_at = utcnow()
    await state.update_run(run)


def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _player_alive(run: ExperimentRun) -> bool:
    from .attach import pid_is_player

    return any(arm.pid and pid_is_player(int(arm.pid), run.id) for arm in run.arms)


def reconcile_dead_runs(state: PlatformState) -> None:
    """Runs left at 'running' after the player is gone should not stay locked.

    Route them through finalize so history gets rebuffer / bandwidth / QoE.
    Also backfill stopped runs that still have telemetry but empty metrics.
    """
    now = time.time()
    for run in list(state.runs.values()):
        if run.status in (RunStatus.RUNNING, RunStatus.PREPARING):
            if _player_alive(run):
                continue
            started = run.started_at or run.created_at
            age_s = now - started.timestamp() if started is not None else 0.0
            had_pid = any(arm.pid for arm in run.arms)
            if run.status == RunStatus.PREPARING or (not had_pid and age_s < 120):
                continue
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                _mark_stopped_without_finalize(state, run)
                _persist_metrics_inplace(state, run)
                continue
            schedule_finalize(state, run.id, ask_stop=False)
            continue
        if run.status == RunStatus.STOPPED and not _run_has_arm_metrics(run):
            _persist_metrics_inplace(state, run)


def _mark_stopped_without_finalize(state: PlatformState, run: ExperimentRun) -> None:
    for arm in run.arms:
        arm.pid = None
        if arm.status == "attached":
            arm.status = "stopped"
    run.status = RunStatus.STOPPED
    if run.stopped_at is None:
        run.stopped_at = utcnow()
    run.updated_at = utcnow()
    state.store.save_entity("runs", run)


def _persist_metrics_inplace(state: PlatformState, run: ExperimentRun) -> None:
    """Write summary for stopped/orphaned runs that still have telemetry on disk."""
    from .telemetry import persist_run_metrics, telemetry_path

    if _run_has_arm_metrics(run):
        return
    if not run.arms:
        return
    tel = telemetry_path(state, run, run.arms[0].id)
    if not tel.is_file() or tel.stat().st_size <= 0:
        return
    try:
        metrics = persist_run_metrics(state, run)
    except Exception as exc:
        log.warning("persist metrics for %s: %s", run.id, exc)
        return
    run.metrics = metrics
    for arm in run.arms:
        arm.metrics = (metrics.get("arms") or {}).get(arm.id) or {}
    run.updated_at = utcnow()
    state.store.save_entity("runs", run)
    state.runs[run.id] = run


async def delete_runs(state: PlatformState, run_ids: list[str]) -> dict[str, Any]:
    """Drop stopped runs and the files that belong to them."""
    from .frames import current_frames_dir, read_owner
    from .telemetry import run_archive_dir

    deleted: list[str] = []
    skipped: list[dict[str, str]] = []
    seen: set[str] = set()
    for run_id in run_ids:
        if run_id in seen:
            continue
        seen.add(run_id)
        run = state.runs.get(run_id)
        if run is None:
            skipped.append({"id": run_id, "reason": "not found"})
            continue
        if _player_alive(run):
            skipped.append({"id": run_id, "reason": "running"})
            continue
        paths = {
            run_archive_dir(state, run),
            state.store.experiment_dir(run.experiment_id) / "runs" / run.id,
        }
        exp = state.experiments.get(run.experiment_id)
        if exp is not None and exp.artifact_dir:
            paths.add(Path(exp.artifact_dir) / "runs" / run.id)
        for path in paths:
            _remove_tree(path)
        saved = (run.saved_frame_dir or "").strip()
        if saved:
            _remove_tree(Path(saved))
        shared = current_frames_dir(state)
        owner = read_owner(shared)
        if owner is not None and owner[0] == run.id:
            _remove_tree(shared)
        if exp is not None and run.id in exp.run_ids:
            exp.run_ids = [item for item in exp.run_ids if item != run.id]
            await state.add_experiment(exp)
        task = _finalizers.pop(run_id, None)
        if task is not None and not task.done():
            task.cancel()
        async with state._lock:
            state.runs.pop(run_id, None)
        state.store.delete_entity("runs", run_id)
        deleted.append(run_id)
    return {"deleted": deleted, "skipped": skipped}


async def stop_run(state: PlatformState, run_id: str, *, request=None) -> ExperimentRun:
    if request is not None:
        from .security import assert_attach_allowed

        try:
            assert_attach_allowed(request)
        except PermissionError:
            pass

    run = state.runs.get(run_id)
    if run is None:
        raise ValueError("run not found")
    if run.status == RunStatus.STOPPED:
        return run
    await schedule_finalize(state, run_id, ask_stop=True)
    current = state.runs.get(run_id)
    if current is None:
        raise ValueError("run not found")
    return current


def samples_show_playback_end(samples: list[dict]) -> bool:
    """True when the player reported END (or a transition into END)."""
    for sample in samples:
        state = str(sample.get("state") or "").upper()
        if state == "END":
            return True
        event = str(sample.get("event") or "")
        if event.endswith("->END") or event.endswith(":END"):
            return True
    return False


def run_player_exited(run: ExperimentRun) -> bool:
    """True when every arm that had a player pid is no longer alive."""
    from .attach import pid_alive

    watched = False
    for arm in run.arms:
        pid = arm.pid
        if not pid:
            continue
        watched = True
        if pid_alive(int(pid)):
            return False
    return watched


async def reap_finished_runs(state: PlatformState) -> list[str]:
    """Stop runs whose player already exited. Returns stopped run ids."""
    stopped: list[str] = []
    for run in list(state.runs.values()):
        if run.status != RunStatus.RUNNING:
            continue
        if not run_player_exited(run):
            continue
        try:
            await stop_run(state, run.id)
            stopped.append(run.id)
            log.info("reaped finished run %s (player exited)", run.id)
        except Exception as exc:
            log.warning("reap run %s failed: %s", run.id, exc)
    return stopped
