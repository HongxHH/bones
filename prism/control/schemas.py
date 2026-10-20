from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobKind(str, Enum):
    CONTENT = "content"
    SESSION = "session"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"


class WorkerStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"


class LadderRung(BaseModel):
    id: int
    width: int
    height: int
    bitrate_kbps: int


DEFAULT_LADDER: list[LadderRung] = [
    LadderRung(id=0, width=426, height=240, bitrate_kbps=400),
    LadderRung(id=1, width=640, height=360, bitrate_kbps=800),
    LadderRung(id=2, width=854, height=480, bitrate_kbps=1200),
    LadderRung(id=3, width=1280, height=720, bitrate_kbps=2400),
    LadderRung(id=4, width=1920, height=1080, bitrate_kbps=4700),
]


class EncodedRung(LadderRung):
    path: str


class RecipeOutput(BaseModel):
    """Typed handoff from a transcode/import recipe to a packager."""

    model_config = ConfigDict(extra="ignore")
    kind: Literal["dash_tree", "rungs"]
    dash_dir: Optional[str] = None
    mpd_name: Optional[str] = None
    representations: list[EncodedRung] = Field(default_factory=list)
    seg_duration: float = 4.0


class PackageResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    mpd_name: str
    ladder: list[LadderRung] = Field(default_factory=list)
    duration_s: float = 0.0
    output_dir: str
    segment_duration: float = 4.0


class Asset(BaseModel):
    id: str
    name: str
    source_path: str
    kind: str = "source"  # source | dash_tree
    created_at: datetime = Field(default_factory=utcnow)
    extra: dict[str, Any] = Field(default_factory=dict)


class Job(BaseModel):
    id: str
    kind: JobKind
    steps: list[str]
    step_index: int = 0
    status: JobStatus = JobStatus.QUEUED
    plugin: str
    params: dict[str, Any] = Field(default_factory=dict)
    asset_id: Optional[str] = None
    publication_id: Optional[str] = None
    session_id: Optional[str] = None
    experiment_id: Optional[str] = None
    required_labels: list[str] = Field(default_factory=lambda: ["cpu"])
    error: Optional[str] = None
    artifacts: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Publication(BaseModel):
    id: str
    asset_id: str
    title: str
    mpd_name: str = "manifest.mpd"
    ladder: list[LadderRung] = Field(default_factory=list)
    policy_tags: list[str] = Field(default_factory=list)
    segment_duration: float = 4.0
    extra: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class DeliveryPolicy(BaseModel):
    allowed_representation_ids: Optional[list[int]] = None
    mpd_type: str = "static"  # static | dynamic
    minimum_update_period_s: float = 2.0
    extra: dict[str, Any] = Field(default_factory=dict)


class Session(BaseModel):
    id: str
    publication_id: str
    status: SessionStatus = SessionStatus.ACTIVE
    policy: DeliveryPolicy = Field(default_factory=DeliveryPolicy)
    token: Optional[str] = None
    trace: Optional[str] = None
    player_id: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Experiment(BaseModel):
    id: str
    name: str
    notes: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    job_ids: list[str] = Field(default_factory=list)
    publication_ids: list[str] = Field(default_factory=list)
    session_ids: list[str] = Field(default_factory=list)
    run_ids: list[str] = Field(default_factory=list)
    artifact_dir: str = ""
    created_at: datetime = Field(default_factory=utcnow)


class RunStatus(str, Enum):
    QUEUED = "queued"
    PREPARING = "preparing"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"


class ArmRuntime(BaseModel):
    id: str
    session_id: Optional[str] = None
    publication_id: Optional[str] = None
    mpd_url: Optional[str] = None
    status: str = "pending"  # pending|ready|attached|stopped|failed
    status_port: Optional[int] = None
    pid: Optional[int] = None
    player_cmd: Optional[str] = None
    error: Optional[str] = None
    player: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)


class ExperimentRun(BaseModel):
    id: str
    experiment_id: str
    name: str
    status: RunStatus = RunStatus.QUEUED
    arm_mode: str = "serial"
    active_arm_id: Optional[str] = None
    spec: dict[str, Any] = Field(default_factory=dict)
    arms: list[ArmRuntime] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    saved_frame_dir: Optional[str] = None


class WorkerInfo(BaseModel):
    id: str
    labels: list[str] = Field(default_factory=list)
    capacity: int = 1
    inflight: int = 0
    status: WorkerStatus = WorkerStatus.ONLINE
    last_heartbeat: datetime = Field(default_factory=utcnow)
    extra: dict[str, Any] = Field(default_factory=dict)


class WorkerTask(BaseModel):
    id: str
    job_id: str
    step: str
    plugin: str
    params: dict[str, Any] = Field(default_factory=dict)
    required_labels: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


class AssetCreate(BaseModel):
    name: str
    source_path: str
    kind: str = "source"
    extra: dict[str, Any] = Field(default_factory=dict)


class JobCreate(BaseModel):
    kind: JobKind = JobKind.CONTENT
    plugin: str = "import_dash"
    steps: Optional[list[str]] = None
    params: dict[str, Any] = Field(default_factory=dict)
    asset_id: Optional[str] = None
    publication_id: Optional[str] = None
    session_id: Optional[str] = None
    experiment_id: Optional[str] = None
    required_labels: Optional[list[str]] = None


class SessionCreate(BaseModel):
    publication_id: str
    policy: Optional[DeliveryPolicy] = None
    token: Optional[str] = None
    trace: Optional[str] = None
    player_id: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class SessionPolicyUpdate(BaseModel):
    policy: DeliveryPolicy


class ExperimentCreate(BaseModel):
    name: str
    notes: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


class WorkerRegister(BaseModel):
    id: Optional[str] = None
    labels: list[str] = Field(default_factory=lambda: ["cpu"])
    capacity: int = 1
    extra: dict[str, Any] = Field(default_factory=dict)


class OriginAccessLog(BaseModel):
    session_id: Optional[str] = None
    publication_id: Optional[str] = None
    path: str
    status: int
    at: datetime = Field(default_factory=utcnow)
