from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_rev() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parents[3]),
            timeout=2,
        )
        return out.decode().strip()
    except Exception:
        return None


def fingerprint_for_job(job: dict[str, Any], artifacts: dict[str, Any]) -> dict[str, Any]:
    params = job.get("params") or {}
    blob = json.dumps(
        {
            "plugin": job.get("plugin"),
            "kind": job.get("kind"),
            "steps": job.get("steps"),
            "params": params,
            "asset_id": job.get("asset_id"),
            "publication_id": job.get("publication_id") or artifacts.get("publication_id"),
        },
        sort_keys=True,
        default=str,
    )
    return {
        "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
        "plugin": job.get("plugin"),
        "kind": job.get("kind"),
        "params": params,
        "asset_id": job.get("asset_id"),
        "publication_id": job.get("publication_id") or artifacts.get("publication_id"),
        "git": _git_rev(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


class ArchiveExperimentHook:
    name = "archive"

    async def on_job_finished(self, *, job: dict[str, Any], artifacts: dict[str, Any], experiment_dir: str) -> None:
        dest = Path(experiment_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "metrics").mkdir(exist_ok=True)
        fp = fingerprint_for_job(job, artifacts)
        record = {
            "job": job,
            "artifacts": artifacts,
            "fingerprint": fp,
        }
        job_id = job.get("id", "unknown")
        (dest / f"job_{job_id}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        # Rolling experiment manifest for reproducibility
        manifest_path = dest / "manifest.json"
        manifest: dict[str, Any]
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = {"jobs": []}
        else:
            manifest = {"experiment_dir": str(dest), "jobs": []}
        jobs = manifest.setdefault("jobs", [])
        jobs = [j for j in jobs if j.get("job_id") != job_id]
        jobs.append(
            {
                "job_id": job_id,
                "status": job.get("status"),
                "fingerprint": fp,
                "publication_id": artifacts.get("publication_id") or job.get("publication_id"),
            }
        )
        manifest["jobs"] = jobs
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        # Measurement metrics sink
        if artifacts.get("vmaf_mean") is not None or artifacts.get("status") in ("ok", "skipped"):
            metrics_path = dest / "metrics" / f"measure_{job_id}.json"
            metrics_path.write_text(
                json.dumps({"job_id": job_id, "artifacts": artifacts, "fingerprint": fp}, indent=2, default=str),
                encoding="utf-8",
            )
