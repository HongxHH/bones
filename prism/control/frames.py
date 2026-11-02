"""Read the per-frame index and JPEG archive written by the player."""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from typing import Any, Optional

from .orchestrator.state import PlatformState
from .schemas import ExperimentRun
from .telemetry import telemetry_dir


def safe_arm_id(arm_id: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (arm_id or "").strip())
    if not cleaned:
        raise ValueError("arm id required")
    return cleaned


def archive_dir(state: PlatformState, run: ExperimentRun, arm_id: str, *, ensure: bool = True) -> Path:
    path = telemetry_dir(state, run).parent / "frames" / safe_arm_id(arm_id)
    if ensure:
        path.mkdir(parents=True, exist_ok=True)
    return path


def current_frames_dir(state: PlatformState) -> Path:
    """One shared archive. The next playback wipes it."""
    return state.config.data_dir / "frame_current"


def _owner_path(root: Path) -> Path:
    return root / "owner.json"


def read_owner(root: Path) -> tuple[str, str] | None:
    path = _owner_path(root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    run_id = str(data.get("run_id") or "")
    arm_id = str(data.get("arm_id") or "")
    if not run_id or not arm_id:
        return None
    return run_id, arm_id


def claim_current_frames(state: PlatformState, run_id: str, arm_id: str) -> Path:
    """Give this run the shared archive, wiping whatever the previous run left."""
    root = current_frames_dir(state)
    root.mkdir(parents=True, exist_ok=True)
    owner = read_owner(root)
    if owner != (run_id, arm_id):
        for child in list(root.iterdir()):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        _owner_path(root).write_text(
            json.dumps({"run_id": run_id, "arm_id": arm_id}),
            encoding="utf-8",
        )
    return root


def frame_count_hint(state: PlatformState, run: ExperimentRun, arm_id: str) -> int:
    """Cheap frame total from count.txt. Does not parse frames.jsonl."""
    _storage, root = locate_frames(state, run, arm_id)
    count_path = root / "count.txt"
    if not count_path.is_file():
        return 0
    try:
        return max(0, int(count_path.read_text(encoding="ascii").strip() or "0"))
    except (OSError, ValueError):
        return 0


def locate_frames(state: PlatformState, run: ExperimentRun, arm_id: str) -> tuple[str, Path]:
    """Return (storage, directory). storage is current, saved, archive, or none."""
    shared = current_frames_dir(state)
    owner = read_owner(shared)
    if owner == (run.id, arm_id):
        return "current", shared
    saved = (run.saved_frame_dir or "").strip()
    if saved:
        path = Path(saved)
        if path.is_dir():
            return "saved", path
    legacy = archive_dir(state, run, arm_id, ensure=False)
    if (legacy / "frames.jsonl").is_file() or (legacy / "count.txt").is_file():
        return "archive", legacy
    return "none", legacy


def save_current_frames(state: PlatformState, run: ExperimentRun, arm_id: str) -> Path:
    shared = current_frames_dir(state)
    if read_owner(shared) != (run.id, arm_id):
        raise ValueError("current frames belong to another run")
    dest = state.config.data_dir / "frame_saved" / run.id
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(shared, dest)
    # The copy must not become a second live owner.
    owner = dest / "owner.json"
    if owner.exists():
        owner.unlink()
    run.saved_frame_dir = str(dest)
    return dest


def stored_count(root: Path) -> int | None:
    path = root / "count.txt"
    if not path.is_file():
        return None
    try:
        return max(0, int(path.read_text(encoding="ascii").strip() or "0"))
    except ValueError:
        return None


def _index_path(root: Path) -> Path:
    return root / "frames.jsonl"


def _image_path(root: Path, frame: int) -> Path:
    return root / "frames" / f"{frame:06d}.jpg"


def _base_image_path(root: Path, frame: int) -> Path:
    return root / "frames" / f"{frame:06d}.base.jpg"


# Live Lab polls metrics about once a second. Re-reading a multi-megabyte
# index from byte 0 each time fights the player's JPEG writer.
_ROW_CACHE: dict[str, dict[str, Any]] = {}
_ROW_LOCK = threading.Lock()


def iter_frame_rows(root: Path) -> list[dict[str, Any]]:
    """Every index row. Metrics use this; the paged reader stays capped.

    A growing index is parsed from the previous byte offset. A shorter or
    replaced file is read again from the start.
    """
    path = _index_path(root)
    if not path.is_file():
        return []
    try:
        stat = path.stat()
    except OSError:
        return []
    key = str(path.resolve())
    size = int(stat.st_size)
    mtime = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)))
    ident = (getattr(stat, "st_dev", 0), getattr(stat, "st_ino", 0))
    with _ROW_LOCK:
        slot = _ROW_CACHE.get(key)
        same_file = slot is not None and slot.get("ident") == ident
        if same_file and slot["size"] == size and slot["mtime"] == mtime:
            return list(slot["rows"])
        offset = 0
        rows: list[dict[str, Any]] = []
        if same_file and size > int(slot["size"]):
            offset = int(slot["size"])
            rows = list(slot["rows"])
        try:
            with path.open("rb") as handle:
                handle.seek(offset)
                blob = handle.read()
        except OSError:
            return rows
        if blob and not blob.endswith(b"\n"):
            cut = blob.rfind(b"\n")
            if cut < 0:
                return rows
            blob = blob[: cut + 1]
        new_rows: list[dict[str, Any]] = []
        for line in blob.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                new_rows.append(row)
        stored = rows + new_rows
        _ROW_CACHE[key] = {
            "size": offset + len(blob),
            "mtime": mtime,
            "ident": ident,
            "rows": stored,
        }
        return list(stored)


def read_frames(root: Path, *, offset: int = 0, limit: int = 50) -> dict[str, Any]:
    path = _index_path(root)
    if offset < 0:
        offset = 0
    if limit < 1:
        limit = 1
    if limit > 200:
        limit = 200
    known = stored_count(root)
    if not path.exists():
        return {"total": known or 0, "offset": offset, "frames": []}
    rows: list[dict[str, Any]] = []
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if seen >= offset and len(rows) < limit:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            seen += 1
            if known is not None and seen >= offset + limit:
                break
    return {"total": known if known is not None else seen, "offset": offset, "frames": rows}


def read_frame(root: Path, frame: int) -> Optional[dict[str, Any]]:
    path = _index_path(root)
    if frame < 0 or not path.exists():
        return None
    target = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(row.get("frame", -1)) == frame:
                target = row
                break
    return target


def read_frame_image(root: Path, frame: int) -> Optional[bytes]:
    if frame < 0:
        return None
    path = _image_path(root, frame)
    if not path.is_file():
        return None
    data = path.read_bytes()
    if not data.startswith(b"\xff\xd8"):
        return None
    return data


def read_frame_base_image(root: Path, frame: int) -> Optional[bytes]:
    if frame < 0:
        return None
    path = _base_image_path(root, frame)
    if not path.is_file():
        return None
    data = path.read_bytes()
    if not data.startswith(b"\xff\xd8"):
        return None
    return data
