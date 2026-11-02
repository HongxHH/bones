from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response

from .orchestrator.state import PlatformState
from .schemas import OriginAccessLog

router = APIRouter()


def _state(request: Request) -> PlatformState:
    return request.app.state.platform


def _safe_join(root: Path, rel: str) -> Path:
    target = (root / rel).resolve()
    if not str(target).startswith(str(root.resolve())):
        raise HTTPException(status_code=400, detail="invalid path")
    return target


@router.get("/publications/{publication_id}/{path:path}")
async def serve_publication(publication_id: str, path: str, request: Request):
    state = _state(request)
    pub = state.publications.get(publication_id)
    if pub is None:
        raise HTTPException(status_code=404, detail="publication not found")
    root = state.publication_root(pub)
    target = _safe_join(root, path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    state.log_origin(OriginAccessLog(publication_id=publication_id, path=path, status=200))
    return FileResponse(target)


@router.get("/sessions/{session_id}/manifest.mpd")
async def serve_session_mpd(session_id: str, request: Request):
    state = _state(request)
    session = state.sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    pub = state.publications.get(session.publication_id)
    if pub is None:
        raise HTTPException(status_code=404, detail="publication not found")
    mpd_path = state.publication_root(pub) / pub.mpd_name
    if not mpd_path.exists():
        raise HTTPException(status_code=404, detail="mpd not found")
    delivery = session.extra.get("delivery") or "session_filter"
    adapter = state.registry.delivery_adapter(delivery)
    xml = adapter.resolve_mpd(
        mpd_xml=mpd_path.read_text(encoding="utf-8"),
        policy=session.policy.model_dump(),
        public_base=state.config.public_base_url,
        session_id=session.id,
        publication_id=pub.id,
    )
    state.log_origin(OriginAccessLog(session_id=session_id, publication_id=pub.id, path="manifest.mpd", status=200))
    # Session MPD is regenerated from live policy — never cache.
    return Response(
        content=xml,
        media_type="application/dash+xml",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "X-Bones-Policy-Version": str(session.extra.get("policy_version", 0)),
        },
    )


@router.get("/sessions/{session_id}/{path:path}")
async def serve_session_media(session_id: str, path: str, request: Request):
    state = _state(request)
    session = state.sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    pub = state.publications.get(session.publication_id)
    if pub is None:
        raise HTTPException(status_code=404, detail="publication not found")
    root = state.publication_root(pub)
    target = _safe_join(root, path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    state.log_origin(OriginAccessLog(session_id=session_id, publication_id=pub.id, path=path, status=200))
    return FileResponse(target)
