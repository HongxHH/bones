"""Remote quality catalog. Same entry schema as a local write-back document."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from .catalog import QualityCatalog


def pull_remote_catalog(catalog: QualityCatalog, base_url: str, content_id: str, *, timeout: float = 5.0) -> dict:
    """Fetch ``{base}/{content_id}.json`` once per process and import entries."""
    url = (base_url or "").strip()
    if not url:
        return {"status": "skipped", "reason": "remote catalog not configured", "written": 0}
    key = (url.rstrip("/"), content_id)
    if key in catalog._remote_pulled:
        return {"status": "cached", "reason": "", "written": 0}
    catalog._remote_pulled.add(key)
    target = url if url.endswith(".json") else f"{url.rstrip('/')}/{content_id}.json"
    try:
        if target.startswith("file:"):
            path = _file_url_path(target)
            document = json.loads(path.read_text(encoding="utf-8"))
        else:
            with urlopen(target, timeout=timeout) as response:  # noqa: S310 — operator-configured catalog URL
                document = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError, ValueError) as exc:
        return {"status": "skipped", "reason": f"remote catalog unavailable: {exc}", "written": 0}
    if not isinstance(document, dict):
        return {"status": "skipped", "reason": "remote catalog document is not an object", "written": 0}
    written = catalog.import_entries(
        content_id,
        document,
        provenance={"source": "remote", "url": target},
    )
    return {"status": "ok" if written else "skipped", "reason": "" if written else "remote catalog had no entries", "written": written}


def _file_url_path(url: str) -> Path:
    from urllib.parse import unquote, urlparse

    parsed = urlparse(url)
    raw = unquote(parsed.path)
    if raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
        raw = raw[1:]
    return Path(raw)
