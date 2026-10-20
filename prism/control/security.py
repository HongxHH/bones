from __future__ import annotations

import os
from typing import Optional

from fastapi import Request


def _truthy(val: str | None) -> bool:
    return (val or "").strip().lower() in ("1", "true", "yes", "on")


def attach_explicitly_allowed() -> bool:
    return _truthy(os.environ.get("PRISM_ALLOW_ATTACH"))


def is_loopback_host(host: Optional[str]) -> bool:
    if not host:
        return False
    h = host.strip().lower()
    return h in ("127.0.0.1", "::1", "localhost", "testclient") or h.startswith("127.")


def client_host(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return ""


def assert_attach_allowed(request: Request) -> None:
    """Raise PermissionError if this request may not spawn/kill local players."""
    if attach_explicitly_allowed():
        return
    host = client_host(request)
    if is_loopback_host(host):
        return
    raise PermissionError(
        "local player attach denied: client is not loopback. "
        "Set PRISM_ALLOW_ATTACH=1 to override, or use prism agent (PRISM_ATTACH_URL)."
    )
