"""Control-plane access to the shared telemetry field schema.

The canonical schema lives in the player package (publisher side). This module
loads it by path so server tests/runtime do not need istream_player installed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from prism.paths import repo_root

_SCHEMA_PATH = (
    repo_root()
    / "player"
    / "istream_player"
    / "modules"
    / "status"
    / "telemetry_schema.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "istream_player_telemetry_schema",
        _SCHEMA_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"telemetry schema not found: {_SCHEMA_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load()

METRICS_VERSION: int = _mod.METRICS_VERSION
SAMPLE_VALUE_KEYS: tuple[str, ...] = _mod.SAMPLE_VALUE_KEYS
project_sample = _mod.project_sample
project_row = _mod.project_row
sample_key_set = _mod.sample_key_set


def schema_path() -> Path:
    return _SCHEMA_PATH
