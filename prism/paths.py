"""Repo-root helpers shared by Prism CLI, control plane, and desk."""

from __future__ import annotations

from pathlib import Path

# prism/paths.py → prism/ → repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    return _REPO_ROOT
