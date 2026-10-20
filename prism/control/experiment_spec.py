from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from .schemas import DeliveryPolicy


class ArmPlayerSpec(BaseModel):
    mod_nes: str = "always_greedy"
    mod_enhancer: str = "imdn_always"
    mod_renderer: str = "opencv"
    # tcp = live / unlimited; trace = replay network file in `trace`.
    mod_downloader: str = "tcp"
    mod_status: str = "status"
    # Relative to player/ cwd when set, e.g. data/3Glogs/report....json. Empty = no sim.
    trace: str = ""
    telemetry_interval_ms: int = 500
    collect_system_metrics: bool = True


class ArmPublishSpec(BaseModel):
    plugin: str = "identity"
    params: dict[str, Any] = Field(default_factory=dict)


class ArmSessionSpec(BaseModel):
    policy: DeliveryPolicy = Field(default_factory=DeliveryPolicy)


class ArmSpec(BaseModel):
    id: str
    publish: ArmPublishSpec = Field(default_factory=ArmPublishSpec)
    session: ArmSessionSpec = Field(default_factory=ArmSessionSpec)
    player: ArmPlayerSpec = Field(default_factory=ArmPlayerSpec)


class ContentSpec(BaseModel):
    """source: publication:<id> | path:<filesystem path> | bare publication id."""

    source: str


class MeasureSpec(BaseModel):
    plugin: str = "noop"
    params: dict[str, Any] = Field(default_factory=dict)


class CollectSpec(BaseModel):
    client_metrics: bool = True
    # Host CPU/GPU/RSS via RuntimeMetrics. False = zero host-sampling overhead.
    system_metrics: bool = True
    interval_ms: int = 500
    export: list[str] = Field(default_factory=lambda: ["json"])


class ExecutionSpec(BaseModel):
    arm_mode: Literal["serial", "parallel"] = "serial"


class ExperimentSpecInner(BaseModel):
    content: ContentSpec
    arms: list[ArmSpec] = Field(default_factory=list)
    measure: list[MeasureSpec] = Field(default_factory=list)
    collect: CollectSpec = Field(default_factory=CollectSpec)
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)


class ExperimentSpec(BaseModel):
    apiVersion: str = "prism.lab/v1"
    kind: str = "ExperimentSpec"
    metadata: dict[str, Any] = Field(default_factory=dict)
    spec: ExperimentSpecInner

    def name(self) -> str:
        return str(self.metadata.get("name") or "unnamed")

    def notes(self) -> str:
        return str(self.metadata.get("notes") or "")


def parse_experiment_spec(data: dict[str, Any] | ExperimentSpec) -> ExperimentSpec:
    if isinstance(data, ExperimentSpec):
        spec = data
    else:
        spec = ExperimentSpec.model_validate(data)
    if spec.kind != "ExperimentSpec":
        raise ValueError(f"unsupported kind: {spec.kind}")
    if len(spec.spec.arms) != 1:
        raise ValueError("an experiment has one condition")
    return spec


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PyYAML required for YAML specs; pip install pyyaml") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return parse_experiment_spec(data)


def resolve_content_source(source: str) -> tuple[str, Optional[str]]:
    """Return (kind, value) where kind is publication|path."""
    raw = source.strip()
    if raw.startswith("publication:"):
        return "publication", raw.split(":", 1)[1].strip()
    if raw.startswith("path:"):
        return "path", raw.split(":", 1)[1].strip()
    # bare id or filesystem path
    p = Path(raw)
    if p.exists():
        return "path", str(p)
    return "publication", raw
