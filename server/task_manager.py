from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2
from threading import Event, Lock, Thread
from typing import Any

from agent.core import AgentCore
from agent.models import FollowUpContext, SessionState, utc_now
from agent.session import SessionStore
from config.settings import AccessMode, AppConfig
from runtime.logger import AgentLogger
from server.schemas import LogRecord, SessionSummary, WorkspaceInspectResponse
from server.workspace_store import WorkspaceStore


class TaskAlreadyRunningError(RuntimeError):
    pass


class WorkspaceRequiredError(RuntimeError):
    pass


class WorkspaceNotFoundError(RuntimeError):
    pass


class SessionBusyError(RuntimeError):
    pass


class WorkspaceBusyError(RuntimeError):
    pass


@dataclass(slots=True)
class ResolvedWorkspace:
    id: str | None
    name: str
    path: str


class TaskManager:
    def __init__(self, base_config: AppConfig):
        self.base_config = base_config
        self.base_config.ensure_state_dirs()
        self.session_store = SessionStore(self.base_config.session_dir_path)
        self.workspace_store = WorkspaceStore(self.base_config.state_root / "workspaces.json")
        self._sync_workspace_state_to_base()
        self._lock = Lock()
        self._threads: dict[str, Thread] = {}
        self._stop_events: dict[str, Event] = {}

    def start_task(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        workspace_id: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> SessionSummary:
        existing = self.session_store.load(session_id) if session_id else None
        target_session_id = existing.id if existing else session_id
        workspace = self._resolve_workspace(existing, workspace_id)
        if workspace is None:
            raise WorkspaceRequiredError(
                "Select a workspace before starting a new chat."
            )
        config = self._config_with_overrides(
            overrides or {},
            workspace_root=workspace.path,
        )

        with self._lock:
            if target_session_id and self._is_active(target_session_id):
                raise TaskAlreadyRunningError(
                    f"Session {target_session_id} is already running."
                )

            session = existing or SessionState(
                task=prompt,
                title=self._derive_title(prompt),
                status="queued",
                workspace_root=str(config.workspace_path),
                workspace_id=workspace.id,
                workspace_label=workspace.name,
                access_mode=config.access_mode,
            )
            if existing is not None:
                self._ensure_message_history(session)
            follow_up_context: FollowUpContext | None = None
            follow_up_candidates: list[str] = []
            if existing is not None:
                follow_up_context = self._build_follow_up_context(session)
                diagnostic_paths = [
                    path
                    for item in follow_up_context.diagnostics[-6:]
                    for path in item.file_hints
                ]
                follow_up_candidates = self._unique_strings(
                    [
                        *follow_up_context.target_paths,
                        *follow_up_context.changed_files,
                        *follow_up_context.read_files,
                        *diagnostic_paths,
                    ]
                )
            preserved_title = session.title or self._derive_title(session.task)
            session.task = prompt
            session.title = preserved_title or self._derive_title(prompt)
            session.status = "queued"
            session.plan = []
            session.plan_summary = None
            session.task_analysis = None
            session.task_state = None
            session.task_understanding = None
            session.router_result = None
            session.follow_up_context = follow_up_context
            session.candidate_files = follow_up_candidates
            session.verification_commands = []
            session.completion_criteria = []
            session.helper_artifacts = []
            session.final_response = None
            session.blockers = []
            session.stop_reason = None
            session.last_error = None
            session.validation_status = "not_run"
            session.repair_attempts = 0
            session.current_phase = "planning"
            session.workflow_stage = "plan"
            session.edit_generation = 0
            session.validation_plan = []
            session.validation_runs = []
            session.diagnostics = []
            session.tool_calls = []
            session.changed_files = []
            session.executed_commands = []
            session.notes = []
            session.report = None
            session.stop_requested = False
            session.access_mode = config.access_mode
            session.workspace_root = str(config.workspace_path)
            session.workspace_id = workspace.id
            session.workspace_label = workspace.name
            session.archived = False
            session.archived_at = None
            session.runtime_options = self._runtime_options(config, overrides)
            session.append_message("user", prompt)
            session.touch()
            self.session_store.save(session)

            stop_event = Event()
            self._stop_events[session.id] = stop_event
            thread = Thread(
                target=self._run_task_thread,
                args=(config, prompt, session, stop_event),
                daemon=True,
            )
            self._threads[session.id] = thread
            thread.start()

        return self._summary_for_session(session)

    def list_sessions(self, limit: int = 100) -> list[SessionSummary]:
        sessions = self.session_store.list_sessions(limit=limit)
        return [self._summary_for_session(session) for session in sessions]

    def get_session(self, session_id: str) -> SessionState | None:
        session = self.session_store.load(session_id)
        if session is None:
            return None
        session = self._normalize_session_workspace(session)
        if self._is_active(session_id):
            return session.model_copy(
                update={
                    "status": "running",
                    "current_phase": session.current_phase,
                    "workflow_stage": session.workflow_stage,
                }
            )
        session = self._normalize_inactive_session(session)
        return session

    def get_logs(self, session_id: str) -> list[LogRecord]:
        log_path = self.base_config.log_dir_path / f"{session_id}.jsonl"
        if not log_path.exists():
            return []
        records: list[LogRecord] = []
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(LogRecord.model_validate(json.loads(line)))
        return records

    def inspect_workspace(self, focus: str | None = None) -> WorkspaceInspectResponse:
        agent = AgentCore(self.base_config)
        snapshot = agent.memory.build_snapshot(focus)
        return WorkspaceInspectResponse(
            text=agent.memory.render_snapshot(snapshot),
            snapshot=snapshot.model_dump(),
        )

    def inspect_workspace_for(
        self,
        workspace_id: str | None = None,
        focus: str | None = None,
    ) -> WorkspaceInspectResponse:
        workspace = self._resolve_workspace(None, workspace_id)
        if workspace is None:
            raise WorkspaceRequiredError("Select a workspace before inspecting it.")
        config = self._config_with_overrides({}, workspace_root=workspace.path)
        agent = AgentCore(config)
        snapshot = agent.memory.build_snapshot(focus)
        return WorkspaceInspectResponse(
            text=agent.memory.render_snapshot(snapshot),
            snapshot=snapshot.model_dump(),
        )

    def list_workspaces(self):
        return self.workspace_store.list_workspaces()

    def create_workspace(self, name: str, path: str):
        workspace = self.workspace_store.create(name, path)
        Path(workspace.path).mkdir(parents=True, exist_ok=True)
        return workspace

    def update_workspace(
        self,
        workspace_id: str,
        *,
        name: str | None = None,
        path: str | None = None,
    ):
        workspace = self.workspace_store.update(workspace_id, name=name, path=path)
        if workspace is None:
            return None
        Path(workspace.path).mkdir(parents=True, exist_ok=True)
        self._refresh_session_workspace_labels(workspace.id, workspace.name, workspace.path)
        return workspace

    def delete_session(self, session_id: str) -> bool:
        session = self.session_store.load(session_id)
        if session is None:
            return False

        with self._lock:
            if self._is_active(session_id):
                raise SessionBusyError(
                    "The chat is still running and cannot be deleted yet."
                )
            deleted = self.session_store.delete(session_id)
            self._delete_session_artifacts(session_id)
            return deleted

    def delete_workspace(self, workspace_id: str) -> bool:
        workspace = self.workspace_store.get(workspace_id)
        if workspace is None:
            return False

        sessions = [
            session
            for session in self.session_store.list_sessions(limit=10_000)
            if session.workspace_id == workspace_id
        ]

        with self._lock:
            if any(self._is_active(session.id) for session in sessions):
                raise WorkspaceBusyError(
                    "The workspace still has a running chat and cannot be deleted yet."
                )

            for session in sessions:
                self.session_store.delete(session.id)
                self._delete_session_artifacts(session.id)

            deleted_workspace = self.workspace_store.delete(workspace_id)
            return deleted_workspace is not None

    def update_session(
        self,
        session_id: str,
        *,
        archived: bool | None = None,
        stop_requested: bool | None = None,
    ) -> SessionState | None:
        session = self.session_store.load(session_id)
        if session is None:
            return None

        if archived is not None:
            session.archived = archived
            session.archived_at = utc_now() if archived else None
            session.touch()
            self.session_store.save(session)

        if stop_requested:
            stop_event = self._stop_events.get(session_id)
            if stop_event is not None and self._is_active(session_id):
                stop_event.set()
                session.stop_requested = True
                session.touch()
                self.session_store.save(session)
                AgentLogger(self.base_config.log_dir_path, session_id).log_event(
                    "task_stop_requested",
                    session_id=session_id,
                )

        return self.get_session(session_id) or session

    def active_sessions(self) -> list[str]:
        with self._lock:
            inactive = [
                session_id
                for session_id, thread in self._threads.items()
                if not thread.is_alive()
            ]
            for session_id in inactive:
                self._threads.pop(session_id, None)
            return sorted(self._threads.keys())

    def _run_task_thread(
        self,
        config: AppConfig,
        prompt: str,
        session: SessionState,
        stop_event: Event,
    ) -> None:
        agent = AgentCore(config)
        try:
            agent.run_task(prompt, session=session, should_stop=stop_event.is_set)
        except Exception as exc:
            session.status = "failed"
            session.current_phase = "blocked"
            session.workflow_stage = "blocked"
            session.stop_reason = "runtime_exception"
            session.final_response = f"Task crashed: {exc}"
            session.append_message("assistant", session.final_response)
            session.touch()
            AgentLogger(self.base_config.log_dir_path, session.id).log_event(
                "task_crashed",
                session_id=session.id,
                error=str(exc),
            )
            agent.session_store.save(session)
        finally:
            with self._lock:
                self._threads.pop(session.id, None)
                self._stop_events.pop(session.id, None)

    def _config_with_overrides(
        self,
        overrides: dict[str, Any],
        *,
        workspace_root: str | None = None,
    ) -> AppConfig:
        config = replace(
            self.base_config,
            state_root_override=str(self.base_config.state_root),
        )
        if workspace_root:
            config = replace(config, workspace_root=workspace_root)
        access_mode = overrides.get("access_mode")
        if access_mode is None:
            if overrides.get("read_only"):
                access_mode = AccessMode.SAFE.value
            elif overrides.get("approval_mode"):
                access_mode = AccessMode.APPROVAL.value
        execution_profile = overrides.get("execution_profile")
        if execution_profile:
            config = self._apply_execution_profile(config, execution_profile)

        for key in ("dry_run", "verbose", "model_name"):
            if overrides.get(key) is not None:
                config = replace(config, **{key: overrides[key]})
        if access_mode is not None:
            config = replace(config, access_mode=access_mode).normalized()
        else:
            config = config.normalized()
        config.ensure_state_dirs()
        return config

    def _runtime_options(
        self,
        config: AppConfig,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        runtime_options = {
            "access_mode": config.access_mode,
            "dry_run": config.dry_run,
            "read_only": config.read_only,
            "approval_mode": config.approval_mode,
            "verbose": config.verbose,
            "max_repair_attempts": config.max_repair_attempts,
            "report_dir": str(config.report_dir_path),
            "model_name": config.model_name,
            "router_model_name": config.router_model_name,
            "llm_timeout": config.llm_timeout,
            "router_timeout": config.router_timeout,
            "llm_request_retries": config.llm_request_retries,
            "path_scope": "system" if config.full_access else "workspace",
        }
        extra = overrides or {}
        for key in ("agent_profile", "execution_profile"):
            if extra.get(key):
                runtime_options[key] = extra[key]
        return runtime_options

    def _apply_execution_profile(self, config: AppConfig, profile: str) -> AppConfig:
        normalized = profile.strip().lower()
        if normalized == "fast":
            return replace(
                config,
                max_iterations=min(config.max_iterations, 10),
                max_tool_calls=min(config.max_tool_calls, 18),
                llm_timeout=min(config.llm_timeout, 18),
                ollama_num_ctx=min(config.ollama_num_ctx, 8192),
                ollama_temperature=0.05,
            )
        if normalized == "balanced":
            return replace(
                config,
                llm_timeout=min(max(config.llm_timeout, 20), 25),
                ollama_num_ctx=min(config.ollama_num_ctx, 8192),
                ollama_temperature=min(config.ollama_temperature, 0.1),
            )
        if normalized == "deep":
            return replace(
                config,
                max_iterations=max(config.max_iterations, 24),
                max_tool_calls=max(config.max_tool_calls, 44),
                llm_timeout=max(config.llm_timeout, 45),
                ollama_num_ctx=max(config.ollama_num_ctx, 16384),
                ollama_temperature=min(config.ollama_temperature, 0.12),
            )
        return config

    def _summary_for_session(self, session: SessionState) -> SessionSummary:
        session = self._normalize_session_workspace(session)
        if not self._is_active(session.id):
            session = self._normalize_inactive_session(session)
        status = "running" if self._is_active(session.id) else session.status
        return SessionSummary(
            id=session.id,
            task=session.task,
            title=session.title,
            status=status,
            current_phase=session.current_phase,
            workflow_stage=session.workflow_stage,
            validation_status=session.validation_status,
            access_mode=session.access_mode,
            workspace_id=session.workspace_id,
            workspace_label=session.workspace_label,
            created_at=session.created_at,
            updated_at=session.updated_at,
            iterations=session.iterations,
            changed_file_count=len(session.changed_files),
            tool_call_count=len(session.tool_calls),
            message_count=len(session.messages),
            last_message_preview=self._last_message_preview(session),
            archived=session.archived,
            archived_at=session.archived_at,
            stop_requested=session.stop_requested,
            stop_reason=session.stop_reason,
            runtime_options=session.runtime_options,
        )

    def _is_active(self, session_id: str) -> bool:
        thread = self._threads.get(session_id)
        return thread is not None and thread.is_alive()

    def _resolve_workspace(
        self,
        session: SessionState | None,
        workspace_id: str | None,
    ) -> ResolvedWorkspace | None:
        if session is not None and session.workspace_id:
            workspace = self.workspace_store.get(session.workspace_id)
            if workspace is not None:
                return ResolvedWorkspace(
                    id=workspace.id,
                    name=workspace.name,
                    path=workspace.path,
                )
        if session is not None and session.workspace_root:
            return self._transient_workspace(
                session.workspace_root,
                name=session.workspace_label,
                workspace_id=session.workspace_id,
            )
        if workspace_id:
            workspace = self.workspace_store.get(workspace_id)
            if workspace is not None:
                return ResolvedWorkspace(
                    id=workspace.id,
                    name=workspace.name,
                    path=workspace.path,
                )
            raise WorkspaceNotFoundError("Workspace not found.")
        return None

    def _derive_title(self, prompt: str) -> str:
        cleaned = " ".join(str(prompt or "").split()).strip()
        if not cleaned:
            return "Neuer Chat"
        return cleaned[:88]

    def _last_message_preview(self, session: SessionState) -> str | None:
        if session.messages:
            return session.messages[-1].content[:140]
        if session.final_response:
            return session.final_response[:140]
        return session.task[:140] if session.task else None

    def _ensure_message_history(self, session: SessionState) -> None:
        if session.messages:
            return
        if session.task:
            session.append_message("user", session.task)
        if session.final_response:
            session.append_message("assistant", session.final_response)

    def _build_follow_up_context(self, session: SessionState) -> FollowUpContext:
        router_result = session.router_result
        task_state = session.task_state
        understanding = session.task_understanding
        target_paths = self._unique_strings(
            [
                *[
                    str(item.path).strip()
                    for item in (task_state.target_artifacts if task_state is not None else [])
                    if getattr(item, "path", None)
                ],
                *[
                    str(item.path).strip()
                    for item in (understanding.target_artifacts if understanding is not None else [])
                    if getattr(item, "path", None)
                ],
                *(router_result.entities.target_paths if router_result else []),
                *[
                    str(item.path).strip()
                    for item in session.changed_files[-8:]
                    if getattr(item, "path", None)
                ],
                *[
                    str(item.tool_args.get("path") or "").strip()
                    for item in session.tool_calls[-10:]
                    if item.tool_name == "read_file" and str(item.tool_args.get("path") or "").strip()
                ],
                *session.candidate_files[-10:],
            ]
        )
        return FollowUpContext(
            previous_task=session.task,
            previous_root_goal=task_state.root_goal if task_state else None,
            previous_active_goal=task_state.active_goal if task_state else None,
            previous_next_action=task_state.next_action if task_state else None,
            previous_intent=router_result.intent.value if router_result else None,
            previous_requested_outcome=router_result.requested_outcome if router_result else None,
            previous_final_response=session.final_response,
            previous_interpreted_goal=understanding.interpreted_goal if understanding else None,
            previous_recommended_mode=understanding.recommended_mode if understanding else None,
            previous_confidence=understanding.confidence if understanding else None,
            previous_assumptions=list(understanding.assumptions[:8]) if understanding else [],
            previous_constraints=list(understanding.constraints[:8]) if understanding else [],
            target_paths=target_paths[:12],
            changed_files=[
                str(item.path).strip()
                for item in session.changed_files[-8:]
                if getattr(item, "path", None)
            ],
            read_files=[
                str(item.tool_args.get("path") or "").strip()
                for item in session.tool_calls[-10:]
                if item.tool_name == "read_file" and str(item.tool_args.get("path") or "").strip()
            ],
            recent_commands=[
                str(item).strip()
                for item in session.executed_commands[-8:]
                if str(item or "").strip()
            ],
            notes=[str(item).strip() for item in session.notes[-12:] if str(item or "").strip()],
            diagnostics=session.diagnostics[-8:],
            validation_runs=session.validation_runs[-8:],
            last_error=session.last_error,
        )

    def _normalize_session_workspace(self, session: SessionState) -> SessionState:
        if session.workspace_label and session.workspace_root:
            return session
        workspace_label = session.workspace_label
        if not workspace_label and session.workspace_root:
            workspace_label = Path(session.workspace_root).name or "Workspace"
        return session.model_copy(
            update={
                "workspace_label": workspace_label,
                "workspace_root": session.workspace_root,
            }
        )

    def _refresh_session_workspace_labels(
        self,
        workspace_id: str,
        workspace_label: str,
        workspace_root: str,
    ) -> None:
        for session in self.session_store.list_sessions(limit=500):
            if session.workspace_id != workspace_id:
                continue
            session.workspace_label = workspace_label
            session.workspace_root = workspace_root
            session.touch()
            self.session_store.save(session)

    def _sync_workspace_state_to_base(self) -> None:
        base_root = self.base_config.state_root
        for workspace in self.workspace_store.list_workspaces():
            workspace_root = Path(workspace.path).expanduser().resolve()
            if workspace_root == self.base_config.workspace_path:
                continue
            workspace_state_root = workspace_root / self.base_config.state_dir_name
            if not workspace_state_root.exists() or workspace_state_root == base_root:
                continue
            self._copy_newer_files(
                workspace_state_root / "sessions",
                self.base_config.session_dir_path,
            )
            self._copy_newer_files(
                workspace_state_root / "logs",
                self.base_config.log_dir_path,
            )
            self._copy_newer_files(
                workspace_state_root / "reports",
                self.base_config.report_dir_path,
            )

    def _copy_newer_files(self, source_dir: Path, target_dir: Path) -> None:
        if not source_dir.exists():
            return
        target_dir.mkdir(parents=True, exist_ok=True)
        for source in source_dir.glob("*"):
            if not source.is_file():
                continue
            target = target_dir / source.name
            if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
                continue
            copy2(source, target)

    def _normalize_inactive_session(self, session: SessionState) -> SessionState:
        updates: dict[str, Any] = {}

        if session.status == "failed" and session.final_response and not session.blockers:
            if session.validation_status not in {"failed", "blocked"} and not session.stop_requested:
                updates["status"] = "completed"

        if session.status in {"queued", "running"}:
            if session.final_response:
                updates["status"] = "completed"
            elif self._is_stale_session(session):
                updates["status"] = "failed"
                updates["stop_reason"] = session.stop_reason or "stale_session"
                updates["last_error"] = session.last_error or "Session thread is no longer active."
                message = (
                    "Der vorherige Lauf wurde nicht sauber abgeschlossen. "
                    "Bitte starte den Chat erneut."
                )
                updates["final_response"] = message

        if not updates:
            return session

        normalized = session.model_copy(update=updates)
        if (
            updates.get("final_response")
            and not any(
                item.role == "assistant" and item.content == updates["final_response"]
                for item in normalized.messages
            )
        ):
            normalized.append_message("assistant", updates["final_response"])
        normalized.touch()
        self.session_store.save(normalized)
        return normalized

    def _is_stale_session(self, session: SessionState) -> bool:
        try:
            updated_at = datetime.fromisoformat(session.updated_at)
        except ValueError:
            return False
        age_seconds = (datetime.now(timezone.utc) - updated_at).total_seconds()
        return age_seconds > max(self.base_config.llm_timeout * 3, 90)

    def _transient_workspace(
        self,
        path: str,
        *,
        name: str | None = None,
        workspace_id: str | None = None,
    ) -> ResolvedWorkspace:
        resolved_path = str(Path(path).expanduser().resolve())
        return ResolvedWorkspace(
            id=workspace_id,
            name=name or Path(resolved_path).name or "Workspace",
            path=resolved_path,
        )

    def _delete_session_artifacts(self, session_id: str) -> None:
        for target in (
            self.base_config.log_dir_path / f"{session_id}.jsonl",
            self.base_config.report_dir_path / f"{session_id}.json",
        ):
            target.unlink(missing_ok=True)

    def _unique_strings(self, values: list[str]) -> list[str]:
        unique: list[str] = []
        for raw in values:
            text = str(raw or "").strip()
            if not text or text in unique:
                continue
            unique.append(text)
        return unique
