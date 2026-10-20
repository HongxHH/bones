from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .api.v1 import router as v1_router
from .config import ServerConfig
from .orchestrator.state import PlatformState
from .origin import router as origin_router
from .plugins import build_registry
from .workers.queue import build_queue
from .workers.supervisor import WorkerSupervisor

LAB_DIR = Path(__file__).resolve().parent / "lab"


def create_app(config: ServerConfig | None = None) -> FastAPI:
    config = config or ServerConfig()
    config.ensure_dirs()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        registry = build_registry(config)
        queue = build_queue(config.redis_url)
        state = PlatformState(config, registry, queue=queue)
        supervisor = WorkerSupervisor(state)
        app.state.platform = state
        app.state.supervisor = supervisor
        await supervisor.start()
        yield
        await supervisor.stop()

    app = FastAPI(
        title="Prism",
        version="0.1.0",
        description="Prism control plane: DASH origin, experiment jobs, and session delivery.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(v1_router, prefix="/v1")
    app.include_router(origin_router, prefix=config.origin_prefix)

    @app.get("/")
    async def root():
        return RedirectResponse(url="/lab/")

    @app.get("/lab")
    async def lab_redirect():
        return RedirectResponse(url="/lab/")

    if LAB_DIR.is_dir():
        index = LAB_DIR / "index.html"

        @app.get("/lab/")
        async def lab_index():
            return FileResponse(index)

        app.mount("/lab/static", StaticFiles(directory=LAB_DIR), name="lab-static")

    return app


app = create_app()
