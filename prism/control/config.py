from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from prism.paths import repo_root as _repo_root


def _env_first(*keys: str, default: str = "") -> str:
    for key in keys:
        val = os.environ.get(key)
        if val:
            return val
    return default


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    public_base_url: str = "http://127.0.0.1:8080"
    data_dir: Path = field(default_factory=lambda: _repo_root() / "prism" / "var")
    # Prefer PRISM_*; fall back to legacy env names during migration.
    ffmpeg_bin: str = field(
        default_factory=lambda: _env_first("PRISM_FFMPEG", "BONES_FFMPEG", default="ffmpeg")
    )
    ffprobe_bin: str = field(
        default_factory=lambda: _env_first("PRISM_FFPROBE", "BONES_FFPROBE", default="ffprobe")
    )
    worker_poll_s: float = 0.2
    default_segment_duration: float = 4.0
    origin_prefix: str = "/origin"
    redis_url: str | None = field(
        default_factory=lambda: _env_first("PRISM_REDIS_URL", "BONES_REDIS_URL") or None
    )

    @property
    def repo_root(self) -> Path:
        return _repo_root()

    @property
    def default_bbb_dash(self) -> Path:
        return self.repo_root / "player" / "data" / "video_mp4box"

    @property
    def assets_dir(self) -> Path:
        return self.data_dir / "assets"

    @property
    def publications_dir(self) -> Path:
        return self.data_dir / "publications"

    @property
    def experiments_dir(self) -> Path:
        return self.data_dir / "experiments"

    @property
    def staging_dir(self) -> Path:
        return self.data_dir / "staging"

    @property
    def meta_dir(self) -> Path:
        return self.data_dir / "meta"

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.assets_dir,
            self.publications_dir,
            self.experiments_dir,
            self.staging_dir,
            self.meta_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
