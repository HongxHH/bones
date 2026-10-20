"""Catalog of Lab recipe options (modules + network traces) for form UI."""

from __future__ import annotations

import random as _random
from pathlib import Path
from typing import Any


# Labels are research-facing; ids match player ModuleOption / CLI values.
NES_OPTIONS: list[dict[str, str]] = [
    {"id": "always_greedy", "label": "Always greedy（跟 MPD）"},
    {"id": "bones", "label": "BONES"},
    {"id": "dynamic_greedy", "label": "Dynamic greedy"},
    {"id": "bola_greedy", "label": "BOLA greedy"},
    {"id": "buffer_greedy", "label": "Buffer greedy"},
    {"id": "tput_greedy", "label": "Throughput greedy"},
    {"id": "nas", "label": "NAS"},
    {"id": "content_aware_bones", "label": "Content-aware BONES"},
    {"id": "conservative_ca_bones", "label": "Conservative CA-BONES"},
    {"id": "ca_bones", "label": "CA-BONES"},
    {"id": "mt_bones", "label": "Multi-task BONES"},
    {"id": "drl_nes", "label": "DRL NES"},
]

ENHANCER_OPTIONS: list[dict[str, str]] = [
    {"id": "imdn_always", "label": "IMDN always"},
    {"id": "imdn", "label": "IMDN"},
    {"id": "multitask", "label": "Multi-task"},
    {"id": "pacnet", "label": "PACNet"},
]

RENDERER_OPTIONS: list[dict[str, str]] = [
    {"id": "opencv", "label": "OpenCV（窗口/壳画面）"},
    {"id": "opengl", "label": "OpenGL"},
    {"id": "headless", "label": "Headless（无画面）"},
]

DOWNLOADER_OPTIONS: list[dict[str, str]] = [
    {"id": "tcp", "label": "TCP（不模拟网络）"},
    {"id": "trace", "label": "Trace（按轨迹限速）"},
    {"id": "local", "label": "Local"},
    {"id": "quic", "label": "QUIC"},
]

# Subdirs under player/data that hold network JSON traces.
TRACE_DATASET_DIRS: tuple[str, ...] = ("3Glogs", "4Glogs", "hd_fs", "sd_fs")


def player_data_dir(repo_root: Path) -> Path:
    return Path(repo_root) / "player" / "data"


def list_trace_datasets(repo_root: Path) -> list[dict[str, Any]]:
    root = player_data_dir(repo_root)
    out: list[dict[str, Any]] = []
    for name in TRACE_DATASET_DIRS:
        path = root / name
        if not path.is_dir():
            continue
        count = sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".json")
        if count:
            out.append({"id": name, "label": name, "count": count})
    return out


def list_traces(
    repo_root: Path,
    dataset: str,
    *,
    q: str = "",
    limit: int = 40,
    offset: int = 0,
    pick_random: bool = False,
) -> dict[str, Any]:
    root = player_data_dir(repo_root)
    if dataset not in TRACE_DATASET_DIRS:
        raise ValueError(f"unknown trace dataset: {dataset}")
    folder = root / dataset
    if not folder.is_dir():
        return {
            "dataset": dataset,
            "total": 0,
            "items": [],
            "offset": offset,
            "limit": limit,
            "random": pick_random,
        }

    needle = (q or "").strip().lower()
    files = sorted(
        p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".json"
    )
    if needle:
        files = [n for n in files if needle in n.lower()]
    total = len(files)

    def _item(name: str) -> dict[str, str]:
        return {
            "id": name,
            "label": name,
            # Paths are relative to player/ cwd (attach spawn cwd).
            "path": f"data/{dataset}/{name}",
        }

    if pick_random:
        if not files:
            return {
                "dataset": dataset,
                "total": 0,
                "items": [],
                "offset": 0,
                "limit": 1,
                "random": True,
            }
        name = _random.choice(files)
        return {
            "dataset": dataset,
            "total": total,
            "items": [_item(name)],
            "offset": files.index(name),
            "limit": 1,
            "random": True,
        }

    limit = max(1, min(int(limit or 40), 200))
    offset = max(0, int(offset or 0))
    page = files[offset : offset + limit]
    return {
        "dataset": dataset,
        "total": total,
        "items": [_item(name) for name in page],
        "offset": offset,
        "limit": limit,
        "random": False,
    }


def recipe_options_payload(repo_root: Path) -> dict[str, Any]:
    return {
        "nes": NES_OPTIONS,
        "enhancer": ENHANCER_OPTIONS,
        "renderer": RENDERER_OPTIONS,
        "downloader": DOWNLOADER_OPTIONS,
        "trace_datasets": list_trace_datasets(repo_root),
    }
