from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config.settings import AGENT_NAME, AppConfig
from llm.ollama_client import OllamaClient
from server.auth_schemas import AuthStatusResponse, LoginRequest, PasswordPolicyResponse
from server.auth_service import AuthBootstrapError, AuthService
from server.model_manager import ModelManager
from server.setup_schemas import SetupCompleteRequest, SetupCompleteResponse, SetupStatusResponse
from server.setup_service import SetupService
from server.security import configure_security
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


@dataclass(slots=True)
class RuntimeBundle:
    config: AppConfig
    task_manager: TaskManager
    model_manager: ModelManager
    auth_service: AuthService | None
    setup_service: SetupService
    setup_required: bool = False
    setup_reason: str | None = None
    env_file_present_at_startup: bool = False


def create_app(base_config: AppConfig | None = None) -> FastAPI:
    runtime = {"bundle": _build_runtime(base_config or AppConfig.from_sources())}
    static_dir = Path(__file__).resolve().parent.parent / "webui"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        bundle = current_runtime()
        if bundle.auth_service is not None:
            await asyncio.to_thread(
                bundle.auth_service.store.purge_expired_sessions,
                datetime.now(timezone.utc),
            )
        setup_required, _ = _current_setup_state(bundle)
        if not setup_required:
            await asyncio.to_thread(bundle.model_manager.ensure_recommended)
            if bundle.config.warmup_models_on_startup:
                asyncio.create_task(asyncio.to_thread(bundle.model_manager.warmup_preferred_models))
        yield

    app = FastAPI(title=f"{AGENT_NAME} Web Console", version="1.0.0", lifespan=lifespan)

    def current_runtime() -> RuntimeBundle:
        return runtime["bundle"]

    def refresh_runtime(config: AppConfig) -> RuntimeBundle:
        bundle = _build_runtime(config)
        runtime["bundle"] = bundle
        app.state.task_manager = bundle.task_manager
        app.state.model_manager = bundle.model_manager
        app.state.auth_service = bundle.auth_service
        app.state.setup_service = bundle.setup_service
        app.state.runtime_bundle = bundle
        return bundle

    refresh_runtime(current_runtime().config)
    configure_security(app, current_runtime().config)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    async def require_auth(request: Request, response: Response):
        bundle = current_runtime()
        setup_required, _ = _current_setup_state(bundle)
        if setup_required:
            raise HTTPException(
                status_code=503,
                detail="Der Setup-Assistent muss zuerst abgeschlossen werden.",
            )
        if bundle.auth_service is None:
            return None
        return bundle.auth_service.current_user(request, response)

    async def require_csrf(request: Request, response: Response):
        bundle = current_runtime()
        setup_required, _ = _current_setup_state(bundle)
        if setup_required:
            raise HTTPException(
                status_code=503,
                detail="Der Setup-Assistent muss zuerst abgeschlossen werden.",
            )
        if bundle.auth_service is None:
            return None
        bundle.auth_service.require_csrf(request, response)
        return None

    async def require_setup_csrf(request: Request, response: Response):
        bundle = current_runtime()
        setup_required, _ = _current_setup_state(bundle)
        if not setup_required:
            raise HTTPException(status_code=409, detail="Der Setup-Assistent wurde bereits abgeschlossen.")
        if bundle.auth_service is None:
            raise HTTPException(status_code=503, detail="Setup-Schutz konnte nicht initialisiert werden.")
        bundle.auth_service.require_csrf(request, response)
        return None

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(
            static_dir / "index.html",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/auth/session", response_model=AuthStatusResponse)
    async def auth_session(request: Request, response: Response) -> AuthStatusResponse:
        bundle = current_runtime()
        if bundle.auth_service is None:
            raise HTTPException(status_code=404, detail="Authentication is disabled.")
        context = bundle.auth_service.resolve_session(request, response)
        bundle.auth_service.ensure_csrf_cookie(request, response, context)
        return bundle.auth_service.build_auth_status(context)

    @app.post(
        "/api/auth/login",
        response_model=AuthStatusResponse,
        dependencies=[Depends(require_csrf)],
    )
    async def auth_login(
        request: Request,
        response: Response,
        payload: LoginRequest,
    ) -> AuthStatusResponse:
        bundle = current_runtime()
        setup_required, _ = _current_setup_state(bundle)
        if setup_required:
            raise HTTPException(
                status_code=503,
                detail="Der Setup-Assistent muss zuerst abgeschlossen werden.",
            )
        if bundle.auth_service is None:
            raise HTTPException(status_code=404, detail="Authentication is disabled.")
        return bundle.auth_service.login(request, response, payload)

    @app.post(
        "/api/auth/logout",
        status_code=204,
        dependencies=[Depends(require_auth), Depends(require_csrf)],
    )
    async def auth_logout(request: Request, response: Response) -> Response:
        bundle = current_runtime()
        if bundle.auth_service is None:
            return Response(status_code=204)
        bundle.auth_service.logout(request, response)
        response.status_code = 204
        return response

    @app.get("/api/setup/status", response_model=SetupStatusResponse)
    async def setup_status(request: Request, response: Response) -> SetupStatusResponse:
        bundle = current_runtime()
        if bundle.auth_service is not None:
            context = bundle.auth_service.resolve_session(request, response)
            bundle.auth_service.ensure_csrf_cookie(request, response, context)
        catalog = bundle.model_manager.catalog()
        setup_required, setup_reason = _current_setup_state(bundle)
        return bundle.setup_service.build_status(
            config=bundle.config,
            password_policy=(
                bundle.auth_service.password_policy()
                if bundle.auth_service is not None
                else _default_password_policy()
            ),
            setup_required=setup_required,
            setup_reason=setup_reason,
            installed_models=catalog["installed_models"],
            recommended_models=catalog["recommended_models"],
        )

    @app.post(
        "/api/setup/complete",
        response_model=SetupCompleteResponse,
        dependencies=[Depends(require_setup_csrf)],
    )
    async def complete_setup(
        request: Request,
        response: Response,
        payload: SetupCompleteRequest,
    ) -> SetupCompleteResponse:
        bundle = current_runtime()
        setup_required, _ = _current_setup_state(bundle)
        if not setup_required:
            raise HTTPException(status_code=409, detail="Der Setup-Assistent wurde bereits abgeschlossen.")
        if bundle.auth_service is None:
            raise HTTPException(status_code=503, detail="Der Setup-Modus ist nicht verfuegbar.")

        secure_cookie = bool(payload.auth_cookie_secure)
        if request.url.scheme != "https" and request.url.hostname not in {"localhost", "127.0.0.1"}:
            secure_cookie = False

        secret_key = bundle.setup_service.generate_secret_key()
        next_config = replace(
            bundle.config,
            ollama_host=payload.ollama_host,
            model_name=payload.model_name,
            router_model_name=payload.router_model_name or payload.model_name,
            access_mode=payload.access_mode,
            auth_cookie_secure=secure_cookie,
            auth_secret_key=secret_key,
            public_base_url=payload.public_base_url,
        ).normalized()
        bundle.setup_service.write_runtime_env(config=next_config, secret_key=secret_key)
        next_config.ensure_state_dirs()
        provisional_auth_service = AuthService(next_config)

        try:
            if provisional_auth_service.store.count_users() > 0:
                provisional_auth_service.recover_admin_access(
                    email=payload.admin_email,
                    password=payload.admin_password,
                    display_name=payload.admin_display_name,
                )
            else:
                provisional_auth_service.create_initial_admin(
                    email=payload.admin_email,
                    password=payload.admin_password,
                    display_name=payload.admin_display_name,
                )
        except AuthBootstrapError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        bundle = refresh_runtime(next_config)
        if bundle.auth_service is None:
            raise HTTPException(status_code=500, detail="Authentication konnte nicht initialisiert werden.")

        workspace = bundle.task_manager.create_workspace(
            payload.initial_workspace_name,
            payload.initial_workspace_path,
        )
        auth_status = bundle.auth_service.login(
            request,
            response,
            LoginRequest(
                email=payload.admin_email,
                password=payload.admin_password,
                totp_code=None,
            ),
        )
        return SetupCompleteResponse(
            auth=auth_status,
            workspace=workspace,
            env_path=str(bundle.setup_service.env_path),
        )

    @app.get("/api/health", response_model=HealthResponse, dependencies=[Depends(require_auth)])
    async def health() -> HealthResponse:
        return HealthResponse(ok=True, active_sessions=current_runtime().task_manager.active_sessions())

    @app.get("/api/config", dependencies=[Depends(require_auth)])
    async def get_config() -> dict:
        bundle = current_runtime()
        config = bundle.config
        model_manager = bundle.model_manager
        public_config = config.to_public_dict()
        model_catalog = model_manager.catalog()
        installed_models = model_catalog["installed_models"]
        preferred_model = config.model_name
        model_candidates = _merge_model_candidates(preferred_model, installed_models)
        public_config["preferred_model_name"] = preferred_model
        public_config["router_preferred_model_name"] = config.router_model_name
        public_config["installed_ollama_models"] = installed_models
        public_config["model_candidates"] = model_candidates
        public_config["recommended_models"] = model_catalog["recommended_models"]
        return public_config

    @app.get("/api/models", response_model=ModelCatalogResponse, dependencies=[Depends(require_auth)])
    async def get_models() -> ModelCatalogResponse:
        return ModelCatalogResponse.model_validate(current_runtime().model_manager.catalog())

    @app.post(
        "/api/models/ensure-recommended",
        response_model=ModelCatalogResponse,
        status_code=202,
        dependencies=[Depends(require_auth), Depends(require_csrf)],
    )
    async def ensure_recommended_models() -> ModelCatalogResponse:
        return ModelCatalogResponse.model_validate(current_runtime().model_manager.ensure_recommended())

    @app.get("/api/workspace/inspect", dependencies=[Depends(require_auth)])
    async def inspect_workspace(
        focus: str | None = Query(default=None),
        workspace_id: str | None = Query(default=None),
    ) -> dict:
        task_manager = current_runtime().task_manager
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

    @app.get("/api/workspaces", response_model=list[WorkspaceRecord], dependencies=[Depends(require_auth)])
    async def list_workspaces() -> list[WorkspaceRecord]:
        return current_runtime().task_manager.list_workspaces()

    @app.post(
        "/api/workspaces",
        response_model=WorkspaceRecord,
        status_code=201,
        dependencies=[Depends(require_auth), Depends(require_csrf)],
    )
    async def create_workspace(request: WorkspaceCreateRequest) -> WorkspaceRecord:
        return current_runtime().task_manager.create_workspace(request.name, request.path)

    @app.patch(
        "/api/workspaces/{workspace_id}",
        response_model=WorkspaceRecord,
        dependencies=[Depends(require_auth), Depends(require_csrf)],
    )
    async def update_workspace(
        workspace_id: str,
        request: WorkspaceUpdateRequest,
    ) -> WorkspaceRecord:
        workspace = current_runtime().task_manager.update_workspace(
            workspace_id,
            name=request.name,
            path=request.path,
        )
        if workspace is None:
            raise HTTPException(status_code=404, detail="Workspace not found.")
        return workspace

    @app.get("/api/sessions", response_model=list[SessionSummary], dependencies=[Depends(require_auth)])
    async def list_sessions(limit: int = Query(default=100, ge=1, le=500)) -> list[SessionSummary]:
        return current_runtime().task_manager.list_sessions(limit=limit)

    @app.get("/api/sessions/{session_id}", dependencies=[Depends(require_auth)])
    async def get_session(session_id: str) -> dict:
        session = current_runtime().task_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        return session.model_dump()

    @app.get("/api/sessions/{session_id}/logs", dependencies=[Depends(require_auth)])
    async def get_session_logs(session_id: str) -> list[dict]:
        task_manager = current_runtime().task_manager
        if task_manager.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        return [record.model_dump() for record in task_manager.get_logs(session_id)]

    @app.post(
        "/api/tasks",
        response_model=SessionSummary,
        status_code=202,
        dependencies=[Depends(require_auth), Depends(require_csrf)],
    )
    async def create_task(request: TaskCreateRequest) -> SessionSummary:
        task_manager = current_runtime().task_manager
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

    @app.patch(
        "/api/sessions/{session_id}",
        dependencies=[Depends(require_auth), Depends(require_csrf)],
    )
    async def update_session(session_id: str, request: SessionUpdateRequest) -> dict:
        session = current_runtime().task_manager.update_session(
            session_id,
            archived=request.archived,
            stop_requested=request.stop_requested,
        )
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        return session.model_dump()

    @app.delete(
        "/api/sessions/{session_id}",
        status_code=204,
        dependencies=[Depends(require_auth), Depends(require_csrf)],
    )
    async def delete_session(session_id: str) -> Response:
        task_manager = current_runtime().task_manager
        try:
            deleted = task_manager.delete_session(session_id)
        except SessionBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found.")
        return Response(status_code=204)

    @app.delete(
        "/api/workspaces/{workspace_id}",
        status_code=204,
        dependencies=[Depends(require_auth), Depends(require_csrf)],
    )
    async def delete_workspace(workspace_id: str) -> Response:
        task_manager = current_runtime().task_manager
        try:
            deleted = task_manager.delete_workspace(workspace_id)
        except WorkspaceBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Workspace not found.")
        return Response(status_code=204)

    @app.get("/api/sessions/{session_id}/events", dependencies=[Depends(require_auth)])
    async def stream_session_events(session_id: str) -> StreamingResponse:
        task_manager = current_runtime().task_manager
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


def _build_runtime(base_config: AppConfig) -> RuntimeBundle:
    config = _with_available_models(base_config)
    config.ensure_state_dirs()
    task_manager = TaskManager(config)
    model_manager = ModelManager(config)
    setup_service = SetupService(config.workspace_path / ".env")
    env_file_present_at_startup = setup_service.has_env_file()
    auth_service: AuthService | None = None
    setup_required = False
    setup_reason: str | None = None

    if config.auth_enabled:
        auth_config = config
        missing_secret = not config.auth_secret_key
        if missing_secret:
            auth_config = replace(
                config,
                auth_secret_key=setup_service.generate_secret_key(),
                auth_cookie_secure=bool(config.auth_cookie_secure and config.public_base_url),
            )
        auth_service = AuthService(auth_config)
        has_users = auth_service.store.count_users() > 0
        if missing_secret and not has_users:
            setup_required = True
            setup_reason = "missing_auth_secret_key"
        if not setup_required:
            try:
                auth_service.bootstrap_initial_admin()
            except AuthBootstrapError as exc:
                if auth_service.store.count_users() == 0:
                    setup_required = True
                    setup_reason = _setup_reason_from_exception(exc)
                else:
                    raise

    return RuntimeBundle(
        config=config,
        task_manager=task_manager,
        model_manager=model_manager,
        auth_service=auth_service,
        setup_service=setup_service,
        setup_required=setup_required,
        setup_reason=setup_reason,
        env_file_present_at_startup=env_file_present_at_startup,
    )


def _current_setup_state(bundle: RuntimeBundle) -> tuple[bool, str | None]:
    if not bundle.config.auth_enabled:
        return False, None

    has_env_file = bundle.setup_service.has_env_file()
    has_users = bool(bundle.auth_service and bundle.auth_service.store.count_users() > 0)
    missing_bootstrap_credentials = (
        not bundle.config.auth_initial_admin_email or not bundle.config.auth_initial_admin_password
    )

    if (
        bundle.env_file_present_at_startup
        and not has_env_file
        and has_users
        and missing_bootstrap_credentials
    ):
        return True, "missing_runtime_env"

    if not bundle.config.auth_secret_key:
        if not has_env_file and has_users:
            return True, "missing_runtime_env"
        return True, bundle.setup_reason or "missing_auth_secret_key"

    if bundle.setup_required:
        return True, bundle.setup_reason

    return False, None


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _fetch_installed_ollama_models(config: AppConfig) -> list[dict]:
    client = OllamaClient(config)
    return client.list_models_safe()


def _default_password_policy() -> PasswordPolicyResponse:
    return PasswordPolicyResponse(
        min_length=14,
        blocked_password_examples=["password", "12345678", "changeme"],
    )


def _setup_reason_from_exception(exc: Exception) -> str:
    text = str(exc).lower()
    if "secret_key" in text:
        return "missing_auth_secret_key"
    if "initial admin credentials" in text:
        return "missing_initial_admin"
    return "setup_required"


def _with_available_models(config: AppConfig) -> AppConfig:
    installed_models = _fetch_installed_ollama_models(config)
    installed_names = [str(item.get("name")) for item in installed_models if item.get("name")]
    if not installed_names:
        router_model_name = config.router_model_name or config.model_name
        return replace(config, router_model_name=router_model_name)

    primary_model = config.model_name
    if primary_model not in installed_names:
        ranked = sorted(installed_names, key=_model_preference_rank)
        primary_model = ranked[0]

    router_model = _select_router_model(
        preferred_router=config.router_model_name,
        installed_names=installed_names,
        primary_model=primary_model,
    )
    return replace(
        config,
        model_name=primary_model,
        router_model_name=router_model,
    )


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


def _select_router_model(
    *,
    preferred_router: str | None,
    installed_names: list[str],
    primary_model: str,
) -> str:
    if preferred_router and preferred_router in installed_names:
        return preferred_router
    if "qwen2.5-coder:14b" in installed_names:
        return "qwen2.5-coder:14b"
    coder_ranked = sorted(
        installed_names,
        key=_router_model_preference_rank,
    )
    if coder_ranked:
        return coder_ranked[0]
    return primary_model


def _router_model_preference_rank(name: str) -> tuple[int, int, str]:
    lowered = name.lower()
    is_coder = 0 if "coder" in lowered else 1
    is_smallish = 0 if "14b" in lowered or "7b" in lowered or "8b" in lowered else 1
    return (is_coder, is_smallish, lowered)
