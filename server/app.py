from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config.settings import AGENT_NAME, AppConfig
from server.schemas import HealthResponse, SessionSummary, TaskCreateRequest
from server.task_manager import TaskAlreadyRunningError, TaskManager


def create_app(base_config: AppConfig | None = None) -> FastAPI:
    config = base_config or AppConfig.from_sources()
    config.ensure_state_dirs()
    task_manager = TaskManager(config)
    static_dir = Path(__file__).resolve().parent.parent / "webui"

    app = FastAPI(title=f"{AGENT_NAME} Web Console", version="1.0.0")
    app.state.task_manager = task_manager
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(ok=True, active_sessions=task_manager.active_sessions())

    @app.get("/api/config")
    async def get_config() -> dict:
        return config.to_public_dict()

    @app.get("/api/workspace/inspect")
    async def inspect_workspace(focus: str | None = Query(default=None)) -> dict:
        response = task_manager.inspect_workspace(focus)
        return response.model_dump()

    @app.get("/api/sessions", response_model=list[SessionSummary])
    async def list_sessions(limit: int = Query(default=100, ge=1, le=500)) -> list[SessionSummary]:
        return task_manager.list_sessions(limit=limit)

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> dict:
        session = task_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        return session.model_dump()

    @app.get("/api/sessions/{session_id}/logs")
    async def get_session_logs(session_id: str) -> list[dict]:
        if task_manager.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        return [record.model_dump() for record in task_manager.get_logs(session_id)]

    @app.post("/api/tasks", response_model=SessionSummary, status_code=202)
    async def create_task(request: TaskCreateRequest) -> SessionSummary:
        try:
            return task_manager.start_task(
                request.prompt,
                session_id=request.session_id,
                overrides=request.model_dump(
                    include={
                        "access_mode",
                        "dry_run",
                        "read_only",
                        "approval_mode",
                        "verbose",
                    }
                ),
            )
        except TaskAlreadyRunningError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}/events")
    async def stream_session_events(session_id: str) -> StreamingResponse:
        if task_manager.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found.")

        async def event_stream():
            last_updated = None
            sent_logs = 0
            while True:
                session = task_manager.get_session(session_id)
                if session is not None and session.updated_at != last_updated:
                    last_updated = session.updated_at
                    yield _sse("session", session.model_dump())

                logs = task_manager.get_logs(session_id)
                for record in logs[sent_logs:]:
                    yield _sse("log", record.model_dump())
                sent_logs = len(logs)

                if session is not None and session.status in {"completed", "failed", "partial"}:
                    yield _sse("done", {"status": session.status, "phase": session.current_phase})
                    break

                yield ": keep-alive\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
