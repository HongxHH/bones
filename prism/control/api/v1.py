from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from ..experiment_spec import parse_experiment_spec
from ..orchestrator.state import PlatformState, new_id
from ..runs import (
    advance_serial,
    attach_arm,
    create_run_from_spec,
    detach_arm,
    delete_runs,
    reconcile_dead_runs,
    prepare_run,
    samples_show_playback_end,
    schedule_finalize,
    stop_run,
)
from ..frames import (
    locate_frames,
    read_frame,
    read_frame_base_image,
    read_frame_image,
    read_frames,
    save_current_frames,
)
from ..preview import get_preview, put_preview
from ..telemetry import (
    append_samples,
    analyzer_report,
    build_live_view,
    build_run_metrics,
    export_run_csv,
    iter_samples,
    persist_run_metrics,
    telemetry_path,
)
from ..schemas import (
    Asset,
    AssetCreate,
    DeliveryPolicy,
    Experiment,
    ExperimentCreate,
    ExperimentRun,
    Job,
    JobCreate,
    JobKind,
    JobStatus,
    Publication,
    RunStatus,
    Session,
    SessionCreate,
    SessionPolicyUpdate,
    SessionStatus,
    WorkerInfo,
    WorkerRegister,
    WorkerStatus,
)
from ..workers.handlers import finish_job
from ..workers.supervisor import claim_and_run_remote

router = APIRouter()


def _state(request: Request) -> PlatformState:
    return request.app.state.platform


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    state = _state(request)
    return {
        "ok": True,
        "product": "Prism",
        "plugins": state.registry.catalog(),
        "queue_pending": state.queue.pending_count(),
        "workers": len(state.workers),
    }


class FrameSinkBody(BaseModel):
    name: str


@router.put("/system/frame-sink")
async def set_frame_sink(body: FrameSinkBody, request: Request) -> dict:
    from ..security import assert_attach_allowed

    try:
        assert_attach_allowed(request)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    name = (body.name or "").strip()
    if not name or len(name) > 64 or any(ch in name for ch in "\\/\0"):
        raise HTTPException(400, "invalid frame sink name")
    state = _state(request)
    state.frame_shm = name
    return {"ok": True, "name": name}


class PictureBody(BaseModel):
    run_id: str
    arm_id: str
    frame: int


def _picture_view(state) -> dict:
    if not state.picture_focus:
        return {"active": False, "seq": state.picture_seq}
    return {"active": True, **state.picture_focus}


@router.get("/system/picture")
async def get_picture(request: Request) -> dict:
    return _picture_view(_state(request))


@router.put("/system/picture")
async def set_picture(body: PictureBody, request: Request) -> dict:
    from ..security import assert_attach_allowed

    try:
        assert_attach_allowed(request)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    if body.frame < 0:
        raise HTTPException(400, "frame must be >= 0")
    state = _state(request)
    run = state.runs.get(body.run_id)
    if run is None or not any(arm.id == body.arm_id for arm in run.arms):
        raise HTTPException(404, "run or arm not found")
    state.picture_seq += 1
    state.picture_focus = {
        "seq": state.picture_seq,
        "run_id": body.run_id,
        "arm_id": body.arm_id,
        "frame": body.frame,
    }
    return _picture_view(state)


@router.delete("/system/picture")
async def clear_picture(request: Request) -> dict:
    from ..security import assert_attach_allowed

    try:
        assert_attach_allowed(request)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    state = _state(request)
    state.picture_focus = None
    state.picture_seq += 1
    return _picture_view(state)


@router.get("/system")
async def system_info(request: Request) -> dict[str, Any]:
    """Lab System page: paths, presets, toolchain probes."""
    state = _state(request)
    cfg = state.config
    bbb = cfg.default_bbb_dash
    ffmpeg = shutil.which(cfg.ffmpeg_bin) or cfg.ffmpeg_bin
    return {
        "product": "Prism",
        "public_base_url": cfg.public_base_url,
        "data_dir": str(cfg.data_dir),
        "repo_root": str(cfg.repo_root),
        "origin_prefix": cfg.origin_prefix,
        "redis_url": cfg.redis_url,
        "ffmpeg": ffmpeg,
        "ffmpeg_found": bool(shutil.which(cfg.ffmpeg_bin) or Path(cfg.ffmpeg_bin).exists()),
        "presets": {
            "bbb_dash": {
                "name": "bbb",
                "path": str(bbb),
                "exists": bbb.is_dir() and (bbb / "bbb.mpd").exists(),
                "mpd": "bbb.mpd",
            }
        },
        "attach": {
            "allow_env": os.environ.get("PRISM_ALLOW_ATTACH"),
            "agent_url": os.environ.get("PRISM_ATTACH_URL"),
            "loopback_only_default": True,
        },
        "counts": {
            "assets": len(state.assets),
            "publications": len(state.publications),
            "sessions": len(state.sessions),
            "jobs": len(state.jobs),
            "experiments": len(state.experiments),
            "runs": len(state.runs),
            "workers": len(state.workers),
        },
    }


@router.get("/plugins")
async def plugins(request: Request) -> dict[str, list[str]]:
    return _state(request).registry.catalog()


@router.post("/assets", response_model=Asset)
async def create_asset(body: AssetCreate, request: Request) -> Asset:
    state = _state(request)
    asset = Asset(id=new_id("asset"), name=body.name, source_path=body.source_path, kind=body.kind, extra=body.extra)
    return await state.add_asset(asset)


@router.get("/assets", response_model=list[Asset])
async def list_assets(request: Request) -> list[Asset]:
    return list(_state(request).assets.values())


@router.get("/assets/{asset_id}", response_model=Asset)
async def get_asset(asset_id: str, request: Request) -> Asset:
    asset = _state(request).assets.get(asset_id)
    if asset is None:
        raise HTTPException(404, "asset not found")
    return asset


@router.post("/jobs", response_model=Job)
async def create_job(body: JobCreate, request: Request) -> Job:
    state = _state(request)
    if body.kind == JobKind.CONTENT and body.plugin in state.registry.recipes and not body.asset_id:
        raise HTTPException(400, "content job requires asset_id")
    if body.asset_id and body.asset_id not in state.assets:
        raise HTTPException(404, "asset not found")
    if body.session_id and body.session_id not in state.sessions:
        raise HTTPException(404, "session not found")
    try:
        steps, labels = state.registry.pipeline_for(body.plugin, body.kind)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if body.steps:
        steps = body.steps
    if body.required_labels:
        labels = body.required_labels
    job = Job(
        id=new_id("job"),
        kind=body.kind,
        steps=steps,
        plugin=body.plugin,
        params=body.params,
        asset_id=body.asset_id,
        publication_id=body.publication_id,
        session_id=body.session_id,
        experiment_id=body.experiment_id,
        required_labels=labels,
        status=JobStatus.QUEUED,
    )
    await state.add_job(job)
    await state.enqueue_job(job)
    return job


@router.get("/jobs", response_model=list[Job])
async def list_jobs(request: Request) -> list[Job]:
    return list(_state(request).jobs.values())


@router.get("/jobs/{job_id}", response_model=Job)
async def get_job(job_id: str, request: Request) -> Job:
    job = _state(request).jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@router.get("/publications", response_model=list[Publication])
async def list_publications(request: Request) -> list[Publication]:
    return list(_state(request).publications.values())


@router.get("/publications/{publication_id}", response_model=Publication)
async def get_publication(publication_id: str, request: Request) -> Publication:
    pub = _state(request).publications.get(publication_id)
    if pub is None:
        raise HTTPException(404, "publication not found")
    return pub


@router.get("/publications/{publication_id}/play")
async def publication_play_hint(publication_id: str, request: Request) -> dict[str, str]:
    state = _state(request)
    if publication_id not in state.publications:
        raise HTTPException(404, "publication not found")
    url = state.origin_publication_url(publication_id)
    return {
        "mpd_url": url,
        "player_cmd": f"python play.py -i {url} --mod_downloader tcp --mod_status status",
    }


@router.post("/sessions", response_model=Session)
async def create_session(body: SessionCreate, request: Request) -> Session:
    state = _state(request)
    if body.publication_id not in state.publications:
        raise HTTPException(404, "publication not found")
    session = Session(
        id=new_id("sess"),
        publication_id=body.publication_id,
        policy=body.policy or DeliveryPolicy(),
        token=body.token,
        trace=body.trace,
        player_id=body.player_id,
        extra=body.extra,
    )
    await state.add_session(session)
    if session.extra.get("experiment_id") and session.extra["experiment_id"] in state.experiments:
        exp = state.experiments[session.extra["experiment_id"]]
        if session.id not in exp.session_ids:
            exp.session_ids.append(session.id)
            state.persist()
    return session


@router.get("/sessions", response_model=list[Session])
async def list_sessions(request: Request) -> list[Session]:
    return list(_state(request).sessions.values())


@router.get("/sessions/{session_id}", response_model=Session)
async def get_session(session_id: str, request: Request) -> Session:
    session = _state(request).sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    return session


@router.get("/sessions/{session_id}/play")
async def session_play_hint(session_id: str, request: Request) -> dict[str, str]:
    state = _state(request)
    if session_id not in state.sessions:
        raise HTTPException(404, "session not found")
    url = state.origin_session_url(session_id)
    return {
        "mpd_url": url,
        "player_cmd": f"python play.py -i {url} --mod_downloader tcp --mod_status status",
    }


@router.patch("/sessions/{session_id}/policy", response_model=Session)
async def update_session_policy(session_id: str, body: SessionPolicyUpdate, request: Request) -> Session:
    state = _state(request)
    session = state.sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    session.policy = body.policy
    session.extra["policy_version"] = int(session.extra.get("policy_version", 0)) + 1
    await state.update_session(session)
    job = Job(
        id=new_id("job"),
        kind=JobKind.SESSION,
        steps=["apply_policy"],
        plugin="session_filter",
        params={"delivery": "session_filter"},
        session_id=session.id,
        publication_id=session.publication_id,
        required_labels=["cpu"],
        status=JobStatus.QUEUED,
    )
    await state.add_job(job)
    await state.enqueue_job(job)
    return session


@router.post("/sessions/{session_id}/close", response_model=Session)
async def close_session(session_id: str, request: Request) -> Session:
    state = _state(request)
    session = state.sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    session.status = SessionStatus.CLOSED
    return await state.update_session(session)


@router.post("/experiments", response_model=Experiment)
async def create_experiment(body: ExperimentCreate, request: Request) -> Experiment:
    state = _state(request)
    exp_id = new_id("exp")
    artifact_dir = str(state.store.experiment_dir(exp_id))
    exp = Experiment(
        id=exp_id,
        name=body.name,
        notes=body.notes,
        params=body.params,
        artifact_dir=artifact_dir,
    )
    return await state.add_experiment(exp)


@router.get("/experiments", response_model=list[Experiment])
async def list_experiments(request: Request) -> list[Experiment]:
    return list(_state(request).experiments.values())


@router.get("/experiments/{experiment_id}", response_model=Experiment)
async def get_experiment(experiment_id: str, request: Request) -> Experiment:
    exp = _state(request).experiments.get(experiment_id)
    if exp is None:
        raise HTTPException(404, "experiment not found")
    return exp


class RunCreateBody(BaseModel):
    """Start a run from an ExperimentSpec (inline) or a saved experiment's params.spec."""

    spec: Optional[dict] = None
    experiment_id: Optional[str] = None
    attach_players: bool = True
    renderer_override: Optional[str] = None  # e.g. headless for batch
    wait: bool = True  # False → return preparing; client polls GET /runs/{id}


@router.post("/runs", response_model=ExperimentRun)
async def create_run(body: RunCreateBody, request: Request, background_tasks: BackgroundTasks) -> ExperimentRun:
    from ..security import assert_attach_allowed

    state = _state(request)
    if body.attach_players:
        try:
            assert_attach_allowed(request)
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
    raw_spec = body.spec
    if raw_spec is None and body.experiment_id:
        exp = state.experiments.get(body.experiment_id)
        if exp is None:
            raise HTTPException(404, "experiment not found")
        raw_spec = exp.params.get("spec")
        if not raw_spec:
            raise HTTPException(400, "experiment has no params.spec; pass body.spec")
    if raw_spec is None:
        raise HTTPException(400, "spec or experiment_id with saved spec required")
    try:
        run = await create_run_from_spec(
            state,
            raw_spec,
            attach_players=body.attach_players,
            renderer_override=body.renderer_override,
            experiment_id=body.experiment_id,
            wait=body.wait,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    if not body.wait and run.status == RunStatus.PREPARING:
        opts = state.pending_prepares.pop(run.id, {})
        background_tasks.add_task(
            prepare_run,
            state,
            run.id,
            attach_players=bool(opts.get("attach_players", body.attach_players)),
            renderer_override=opts.get("renderer_override", body.renderer_override),
        )
    if body.wait and run.status.value == "failed":
        raise HTTPException(500, run.error or "run preparation failed")
    return run


class RunDeleteBody(BaseModel):
    ids: list[str]


@router.post("/runs/delete")
async def delete_runs_route(body: RunDeleteBody, request: Request) -> dict:
    if not body.ids:
        raise HTTPException(400, "ids required")
    return await delete_runs(_state(request), body.ids)


@router.get("/runs", response_model=list[ExperimentRun])
async def list_runs(request: Request) -> list[ExperimentRun]:
    state = _state(request)
    reconcile_dead_runs(state)
    return list(state.runs.values())


@router.get("/runs/{run_id}", response_model=ExperimentRun)
async def get_run(run_id: str, request: Request) -> ExperimentRun:
    run = _state(request).runs.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return run


@router.post("/runs/{run_id}/stop", response_model=ExperimentRun)
async def run_stop(run_id: str, request: Request) -> ExperimentRun:
    try:
        return await stop_run(_state(request), run_id, request=request)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/runs/{run_id}/arms/{arm_id}/attach", response_model=ExperimentRun)
async def run_attach_arm(run_id: str, arm_id: str, request: Request) -> ExperimentRun:
    try:
        return await attach_arm(_state(request), run_id, arm_id, request=request)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/runs/{run_id}/arms/{arm_id}/detach", response_model=ExperimentRun)
async def run_detach_arm(run_id: str, arm_id: str, request: Request) -> ExperimentRun:
    try:
        return await detach_arm(_state(request), run_id, arm_id, request=request)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post(
    "/runs/{run_id}/advance",
    response_model=ExperimentRun,
    deprecated=True,
    summary="Unsupported: specs are single-arm; advance/parallel is not a Lab path",
)
async def run_advance(run_id: str, request: Request) -> ExperimentRun:
    try:
        return await advance_serial(_state(request), run_id, request=request)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class TelemetryBody(BaseModel):
    run_id: str
    arm_id: str
    session_id: Optional[str] = None
    samples: list[dict] = []
    # allow single-sample shorthand
    sample: Optional[dict] = None


@router.post("/telemetry")
async def post_telemetry(body: TelemetryBody, request: Request) -> dict:
    state = _state(request)
    samples = list(body.samples or [])
    if body.sample:
        samples.append(body.sample)
    if not samples:
        raise HTTPException(400, "samples or sample required")
    try:
        result = append_samples(
            state,
            run_id=body.run_id,
            arm_id=body.arm_id,
            samples=samples,
            session_id=body.session_id,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    # Player finished the video. Do not wait or kill here: this POST is the
    # player's own request, and it still has to leave cleanup and write the file.
    if samples_show_playback_end(samples):
        run = state.runs.get(body.run_id)
        if run is not None and run.status == RunStatus.RUNNING:
            schedule_finalize(state, body.run_id, ask_stop=False)
            result = dict(result)
            result["run_status"] = "finishing"
    return result


@router.post("/preview")
async def post_preview(request: Request) -> dict:
    run_id = (request.headers.get("x-prism-run") or "").strip()
    arm_id = (request.headers.get("x-prism-arm") or "").strip()
    state = _state(request)
    if state.runs.get(run_id) is None:
        raise HTTPException(404, "run not found")
    jpeg = await request.body()
    try:
        put_preview(run_id, arm_id, jpeg)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "bytes": len(jpeg)}


@router.get("/runs/{run_id}/preview.jpg")
async def get_run_preview(run_id: str, request: Request, arm_id: str) -> Response:
    state = _state(request)
    if state.runs.get(run_id) is None:
        raise HTTPException(404, "run not found")
    jpeg = get_preview(run_id, arm_id)
    if jpeg is None:
        raise HTTPException(404, "no preview")
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@router.post("/runs/{run_id}/frames/save")
async def save_frames(run_id: str, request: Request, arm_id: str) -> dict:
    from ..security import assert_attach_allowed

    try:
        assert_attach_allowed(request)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    state = _state(request)
    run = state.runs.get(run_id)
    if run is None or not any(arm.id == arm_id for arm in run.arms):
        raise HTTPException(404, "run or arm not found")
    try:
        dest = save_current_frames(state, run, arm_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    await state.update_run(run)
    return {"ok": True, "run_id": run_id, "arm_id": arm_id, "path": str(dest)}


@router.get("/runs/{run_id}/frames")
async def list_frames(run_id: str, request: Request, arm_id: str, offset: int = 0, limit: int = 50) -> dict:
    state = _state(request)
    run = state.runs.get(run_id)
    if run is None or not any(arm.id == arm_id for arm in run.arms):
        raise HTTPException(404, "run or arm not found")
    storage, root = locate_frames(state, run, arm_id)
    page = read_frames(root, offset=offset, limit=limit) if storage != "none" else {
        "total": 0, "offset": offset, "frames": [],
    }
    page["run_id"] = run_id
    page["arm_id"] = arm_id
    page["storage"] = storage
    return page


@router.get("/runs/{run_id}/frames/{frame}")
async def get_frame(run_id: str, frame: int, request: Request, arm_id: str) -> dict:
    state = _state(request)
    run = state.runs.get(run_id)
    if run is None or not any(arm.id == arm_id for arm in run.arms):
        raise HTTPException(404, "run or arm not found")
    storage, root = locate_frames(state, run, arm_id)
    row = None if storage == "none" else read_frame(root, frame)
    if row is None:
        raise HTTPException(404, "frame not found")
    from ..metrics.bundle import annotate_frame_quality

    return annotate_frame_quality(state, run, row)


@router.get("/runs/{run_id}/frames/{frame}/image")
async def get_frame_image(run_id: str, frame: int, request: Request, arm_id: str) -> Response:
    state = _state(request)
    run = state.runs.get(run_id)
    if run is None or not any(arm.id == arm_id for arm in run.arms):
        raise HTTPException(404, "run or arm not found")
    storage, root = locate_frames(state, run, arm_id)
    image = None if storage == "none" else read_frame_image(root, frame)
    if image is None:
        raise HTTPException(404, "frame image not found")
    return Response(content=image, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/runs/{run_id}/frames/{frame}/base")
async def get_frame_base_image(run_id: str, frame: int, request: Request, arm_id: str) -> Response:
    state = _state(request)
    run = state.runs.get(run_id)
    if run is None or not any(arm.id == arm_id for arm in run.arms):
        raise HTTPException(404, "run or arm not found")
    storage, root = locate_frames(state, run, arm_id)
    image = None if storage == "none" else read_frame_base_image(root, frame)
    if image is None:
        raise HTTPException(404, "frame baseline not found")
    return Response(content=image, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/runs/{run_id}/telemetry")
async def get_run_telemetry(run_id: str, request: Request, arm_id: Optional[str] = None, limit: int = 300) -> dict:
    state = _state(request)
    run = state.runs.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    arms = [arm_id] if arm_id else [a.id for a in run.arms]
    series: dict[str, list] = {}
    for aid in arms:
        series[aid] = iter_samples(telemetry_path(state, run, aid), limit=limit)
    return {"run_id": run_id, "series": series}


@router.get("/runs/{run_id}/live")
async def get_run_live(
    run_id: str,
    request: Request,
    arm_id: Optional[str] = None,
    limit: int = 120,
) -> dict:
    """Tail samples and scalar summary. Does not assemble a metrics bundle or write the run."""
    state = _state(request)
    run = state.runs.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    if limit < 1:
        limit = 1
    if limit > 500:
        limit = 500
    try:
        return build_live_view(state, run, arm_id=arm_id, limit=limit)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/runs/{run_id}/analyzer")
async def get_run_analyzer(run_id: str, request: Request) -> dict:
    """Player end-of-run report, without the raw segment trace."""
    state = _state(request)
    run = state.runs.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return analyzer_report(state, run)


@router.get("/runs/{run_id}/metrics")
async def get_run_metrics(run_id: str, request: Request, persist: bool = False) -> dict:
    state = _state(request)
    run = state.runs.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    # Full bundle (catalog / optional compute) stays off the event loop.
    builder = persist_run_metrics if persist else build_run_metrics
    metrics = await asyncio.to_thread(builder, state, run)
    if persist:
        run.metrics = metrics
        for arm in run.arms:
            arm.metrics = (metrics.get("arms") or {}).get(arm.id) or {}
        await state.update_run(run)
    return metrics


@router.get("/runs/{run_id}/export")
async def export_run(run_id: str, request: Request, format: str = "csv") -> Response:
    state = _state(request)
    run = state.runs.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    fmt = (format or "csv").lower()
    if fmt != "csv":
        raise HTTPException(400, "only format=csv supported")
    body = export_run_csv(state, run)
    return Response(
        content=body,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{run_id}.csv"'},
    )


@router.post("/experiments/{experiment_id}/runs", response_model=ExperimentRun)
async def create_run_for_experiment(experiment_id: str, body: RunCreateBody, request: Request) -> ExperimentRun:
    body.experiment_id = experiment_id
    return await create_run(body, request)


@router.get("/workspaces")
async def get_workspaces() -> dict:
    from ..workspaces import list_workspaces

    return {"workspaces": list_workspaces()}


@router.get("/specs/options")
async def get_spec_options(request: Request) -> dict:
    """Module / dataset catalogs for the Lab recipe form."""
    from ..spec_options import recipe_options_payload

    return recipe_options_payload(_state(request).config.repo_root)


@router.get("/specs/traces")
async def get_spec_traces(
    request: Request,
    dataset: str,
    q: str = "",
    limit: int = 40,
    offset: int = 0,
    random: bool = False,
) -> dict:
    from ..spec_options import list_traces

    try:
        return list_traces(
            _state(request).config.repo_root,
            dataset,
            q=q,
            limit=limit,
            offset=offset,
            pick_random=bool(random),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/specs/validate")
async def validate_spec(body: dict, request: Request) -> dict:
    try:
        from ..runs import IDENTITY_PLUGINS, _validate_publish_plugins

        spec = parse_experiment_spec(body)
        state = _state(request)
        _validate_publish_plugins(state, spec)
        recipe_arms = [
            a.id
            for a in spec.spec.arms
            if (a.publish.plugin or "identity") not in IDENTITY_PLUGINS
        ]
        warnings = []
        if recipe_arms:
            warnings.append(
                f"recipe arms {recipe_arms} need a media file source "
                "(content.source or publish.params.source → mp4/…, not DASH tree)"
            )
        return {
            "ok": True,
            "name": spec.name(),
            "arms": [a.id for a in spec.spec.arms],
            "measure": [m.plugin for m in spec.spec.measure],
            "warnings": warnings,
        }
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


class SpecTextBody(BaseModel):
    text: str
    format: str = "auto"  # auto|json|yaml


@router.post("/specs/parse")
async def parse_spec_text(body: SpecTextBody, request: Request) -> dict:
    """Parse YAML or JSON ExperimentSpec text (Lab dual-track editor)."""
    text = body.text or ""
    fmt = (body.format or "auto").lower()
    data: Any
    try:
        if fmt == "yaml" or (fmt == "auto" and not text.lstrip().startswith("{")):
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise HTTPException(500, "PyYAML required for YAML specs") from exc
            data = yaml.safe_load(text)
        else:
            import json as _json

            data = _json.loads(text)
        spec = parse_experiment_spec(data)
        return {"ok": True, "spec": spec.model_dump(mode="json"), "name": spec.name()}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/workers/register", response_model=WorkerInfo)
async def register_worker(body: WorkerRegister, request: Request) -> WorkerInfo:
    state = _state(request)
    worker = WorkerInfo(
        id=body.id or new_id("worker"),
        labels=body.labels,
        capacity=body.capacity,
        extra={**body.extra, "kind": "remote"},
        status=WorkerStatus.ONLINE,
    )
    return await state.register_worker(worker)


@router.post("/workers/{worker_id}/heartbeat", response_model=WorkerInfo)
async def worker_heartbeat(worker_id: str, request: Request) -> WorkerInfo:
    worker = await _state(request).heartbeat_worker(worker_id)
    if worker is None:
        raise HTTPException(404, "worker not found")
    return worker


@router.post("/workers/{worker_id}/claim")
async def worker_claim(worker_id: str, request: Request) -> dict[str, Any]:
    task = await claim_and_run_remote(_state(request), worker_id)
    if task is None:
        return {"task": None}
    return {"task": task}


class TaskResultBody(BaseModel):
    result: dict[str, Any] = {}
    error: Optional[str] = None


@router.post("/workers/{worker_id}/tasks/{task_id}/complete")
async def worker_complete(worker_id: str, task_id: str, body: TaskResultBody, request: Request) -> dict[str, str]:
    state = _state(request)
    if worker_id not in state.workers:
        raise HTTPException(404, "worker not found")
    pending = getattr(state.queue, "_inflight", None) or getattr(state.queue, "_local_inflight", {})
    pending_task = pending.get(task_id) if isinstance(pending, dict) else None
    if pending_task is None:
        raise HTTPException(404, "task not in-flight")
    if body.error:
        await state.queue.fail(task_id, body.error)
        await finish_job(state, pending_task, {}, body.error)
    else:
        await state.queue.ack(task_id, body.result)
        await finish_job(state, pending_task, body.result, None)
    return {"status": "ok"}


@router.get("/workers", response_model=list[WorkerInfo])
async def list_workers(request: Request) -> list[WorkerInfo]:
    state = _state(request)
    state.mark_stale_workers()
    return list(state.workers.values())


@router.get("/origin/logs")
async def origin_logs(request: Request, session_id: Optional[str] = None, limit: int = 100) -> list[dict[str, Any]]:
    logs = _state(request).origin_logs
    if session_id:
        logs = [x for x in logs if x.session_id == session_id]
    return [x.model_dump(mode="json") for x in logs[-limit:]]


class BootstrapLocal(BaseModel):
    name: str
    path: str
    experiment_name: Optional[str] = None


@router.post("/bootstrap/import-dash")
async def bootstrap_import_dash(body: BootstrapLocal, request: Request) -> dict[str, Any]:
    """Phase 0 helper: import an existing DASH tree and return a playable MPD URL."""
    state = _state(request)
    asset = Asset(id=new_id("asset"), name=body.name, source_path=body.path, kind="dash_tree")
    await state.add_asset(asset)
    experiment_id = None
    if body.experiment_name:
        exp = Experiment(
            id=new_id("exp"),
            name=body.experiment_name,
            artifact_dir="",
        )
        exp.artifact_dir = str(state.store.experiment_dir(exp.id))
        await state.add_experiment(exp)
        experiment_id = exp.id
    steps, labels = state.registry.pipeline_for("import_dash", JobKind.CONTENT)
    job = Job(
        id=new_id("job"),
        kind=JobKind.CONTENT,
        steps=steps,
        plugin="import_dash",
        params={"title": body.name},
        asset_id=asset.id,
        experiment_id=experiment_id,
        required_labels=labels,
    )
    await state.add_job(job)
    await state.enqueue_job(job)
    return {"asset": asset, "job": job, "experiment_id": experiment_id}
