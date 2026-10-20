"""Shared FastAPI dependencies for Prism control-plane routes."""

from __future__ import annotations

from fastapi import Request

from ..orchestrator.state import PlatformState


def platform_state(request: Request) -> PlatformState:
    return request.app.state.platform
