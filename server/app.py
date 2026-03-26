from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config.settings import AGENT_NAME, AppConfig
from llm.ollama_client import OllamaClient
from server.model_manager import ModelManager
from server.schemas import (
    HealthResponse,
    ModelCatalogResponse,
    SessionSummary,
    SessionUpdateRequest,
    TaskCreateRequest,
    WorkspaceCreateRequest,
    WorkspaceRecord,
    WorkspaceUpdateRequest,
)
from server.task_manager import (
    SessionBusyError,
    TaskAlreadyRunningError,
    TaskManager,
    WorkspaceBusyError,
    WorkspaceNotFoundError,
    WorkspaceRequiredError,
)


def create_app(base_config: AppConfig | None = None) -> FastAPI:
    config = base_config or AppConfig.from_sources()
    config = _with_available_model(config)
    config.ensure_state_dirs()
    task_manager = TaskManager(config)
    model_manager = ModelManager(config)
    static_dir = Path(__file__).resolve().parent.parent / "webui"

    app = FastAPI(title=f"{AGENT_NAME} Web Console", version="1.0.0")
    app.state.task_manager = task_manager
    app.state.model_manager = model_manager
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.on_event("startup")
    async def ensure_models_on_startup() -> None:
        await asyncio.to_thread(model_manager.ensure_recommended)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(ok=True, active_sessions=task_manager.active_sessions())

    @app.get("/api/config")
    async def get_config() -> dict:
        public_config = config.to_public_dict()
        model_catalog = model_manager.catalog()
        installed_models = model_catalog["installed_models"]
        preferred_model = config.model_name
        model_candidates = _merge_model_candidates(preferred_model, installed_models)
        public_config["preferred_model_name"] = preferred_model
        public_config["installed_ollama_models"] = installed_models
        public_config["model_candidates"] = model_candidates
        public_config["recommended_models"] = model_catalog["recommended_models"]
        return public_config

    @app.get("/api/models", response_model=ModelCatalogResponse)
    async def get_models() -> ModelCatalogResponse:
        return ModelCatalogResponse.model_validate(model_manager.catalog())

    @app.post("/api/models/ensure-recommended", response_model=ModelCatalogResponse, status_code=202)
    async def ensure_recommended_models() -> ModelCatalogResponse:
        return ModelCatalogResponse.model_validate(model_manager.ensure_recommended())

    @app.get("/api/workspace/inspect")
    async def inspect_workspace(
        focus: str | None = Query(default=None),
        workspace_id: str | None = Query(default=None),
    ) -> dict:
        try:
            response = task_manager.inspect_workspace_for(
                workspace_id=workspace_id,
                focus=focus,
            )
        except WorkspaceRequiredError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except WorkspaceNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return response.model_dump()

    @app.get("/api/workspaces", response_model=list[WorkspaceRecord])
    async def list_workspaces() -> list[WorkspaceRecord]:
        return task_manager.list_workspaces()

    @app.post("/api/workspaces", response_model=WorkspaceRecord, status_code=201)
    async def create_workspace(request: WorkspaceCreateRequest) -> WorkspaceRecord:
        return task_manager.create_workspace(request.name, request.path)

    @app.patch("/api/workspaces/{workspace_id}", response_model=WorkspaceRecord)
    async def update_workspace(
        workspace_id: str,
        request: WorkspaceUpdateRequest,
    ) -> WorkspaceRecord:
        workspace = task_manager.update_workspace(
            workspace_id,
            name=request.name,
            path=request.path,
        )
        if workspace is None:
            raise HTTPException(status_code=404, detail="Workspace not found.")
        return workspace

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
                workspace_id=request.workspace_id,
                overrides=request.model_dump(
                    include={
                        "access_mode",
                        "agent_profile",
                        "dry_run",
                        "execution_profile",
                        "model_name",
                        "read_only",
                        "approval_mode",
                        "verbose",
                    }
                ),
            )
        except TaskAlreadyRunningError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except WorkspaceRequiredError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except WorkspaceNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.patch("/api/sessions/{session_id}")
    async def update_session(session_id: str, request: SessionUpdateRequest) -> dict:
        session = task_manager.update_session(
            session_id,
            archived=request.archived,
            stop_requested=request.stop_requested,
        )
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        return session.model_dump()

    @app.delete("/api/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str) -> Response:
        try:
            deleted = task_manager.delete_session(session_id)
        except SessionBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found.")
        return Response(status_code=204)

    @app.delete("/api/workspaces/{workspace_id}", status_code=204)
    async def delete_workspace(workspace_id: str) -> Response:
        try:
            deleted = task_manager.delete_workspace(workspace_id)
        except WorkspaceBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Workspace not found.")
        return Response(status_code=204)

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


def _fetch_installed_ollama_models(config: AppConfig) -> list[dict]:
    client = OllamaClient(config)
    return client.list_models_safe()


def _with_available_model(config: AppConfig) -> AppConfig:
    installed_models = _fetch_installed_ollama_models(config)
    installed_names = [str(item.get("name")) for item in installed_models if item.get("name")]
    if installed_names and config.model_name not in installed_names:
        ranked = sorted(installed_names, key=_model_preference_rank)
        return replace(config, model_name=ranked[0])
    return config


def _merge_model_candidates(preferred_model: str, installed_models: list[dict]) -> list[str]:
    candidates = [preferred_model]
    for item in installed_models:
        name = item.get("name")
        if name and name not in candidates:
            candidates.append(name)
    return candidates


def _model_preference_rank(name: str) -> tuple[int, str]:
    lowered = name.lower()
    if "coder" in lowered and "qwen3" in lowered:
        return (0, lowered)
    if "coder" in lowered:
        return (1, lowered)
    if "qwen3" in lowered:
        return (2, lowered)
    return (3, lowered)
