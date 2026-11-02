"""Named experiment kits. A workspace sets defaults for new runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prism.paths import repo_root

PUBLIC_CARDS: list[dict[str, Any]] = [
    {"key": "qoe", "label": "QoE", "digits": 2, "group": "播放质量", "hint": "平均画质减去振荡，再减去卡顿"},
    {"key": "avg_vmaf", "label": "VMAF", "digits": 2, "group": "播放质量", "hint": "已评分段的平均画质"},
    {"key": "oscillation_vmaf", "label": "振荡", "digits": 2, "group": "播放质量", "hint": "相邻段画质差的平均"},
    {"key": "avg_freeze", "label": "均卡顿", "digits": 2, "suffix": " s", "group": "播放质量", "hint": "总卡顿秒数除以播放段数"},
]


def workspace_dir() -> Path:
    return repo_root() / "prism" / "workspaces"


def load_workspaces() -> dict[str, dict[str, Any]]:
    directory = workspace_dir()
    found: dict[str, dict[str, Any]] = {}
    if not directory.is_dir():
        return found
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML required to load workspaces") from exc
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            continue
        ws_id = str(data.get("id") or path.stem)
        data["id"] = ws_id
        data.setdefault("label", ws_id)
        data.setdefault("player", {})
        data.setdefault("metrics", {})
        data.setdefault("cards", list(PUBLIC_CARDS))
        found[ws_id] = data
    return found


def list_workspaces() -> list[dict[str, Any]]:
    return list(load_workspaces().values())


def get_workspace(ws_id: str) -> dict[str, Any] | None:
    return load_workspaces().get(ws_id)


def cards_for_run(spec: dict[str, Any] | None) -> tuple[str, list[dict[str, Any]]]:
    inner = ((spec or {}).get("spec") or {}) if isinstance(spec, dict) else {}
    ws_id = str(inner.get("workspace") or "")
    stored = inner.get("cards")
    if isinstance(stored, list) and stored:
        return ws_id, [dict(card) for card in stored if isinstance(card, dict)]
    workspace = get_workspace(ws_id) if ws_id else None
    if workspace is None:
        return ws_id, list(PUBLIC_CARDS)
    cards = workspace.get("cards") or list(PUBLIC_CARDS)
    return workspace["id"], [dict(card) for card in cards if isinstance(card, dict)]
