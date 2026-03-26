from __future__ import annotations

import json
from dataclasses import replace
from threading import Lock, Thread
from typing import Any

from agent.core import AgentCore
from agent.models import SessionState
from agent.session import SessionStore
from config.settings import AccessMode, AppConfig
from server.schemas import LogRecord, SessionSummary, WorkspaceInspectResponse


class TaskAlreadyRunningError(RuntimeError):
    pass


class TaskManager:
    def __init__(self, base_config: AppConfig):
        self.base_config = base_config
        self.base_config.ensure_state_dirs()
        self.session_store = SessionStore(self.base_config.session_dir_path)
        self._lock = Lock()
        self._threads: dict[str, Thread] = {}

    def start_task(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> SessionSummary:
        config = self._config_with_overrides(overrides or {})
        existing = self.session_store.load(session_id) if session_id else None
        target_session_id = existing.id if existing else session_id

        with self._lock:
            if target_session_id and self._is_active(target_session_id):
                raise TaskAlreadyRunningError(
                    f"Session {target_session_id} is already running."
                )

            session = existing or SessionState(
                task=prompt,
                status="queued",
                workspace_root=str(config.workspace_path),
                access_mode=config.access_mode,
            )
            session.task = prompt
            session.status = "queued"
            session.plan = []
            session.plan_summary = None
            session.candidate_files = []
            session.verification_commands = []
            session.completion_criteria = []
            session.helper_artifacts = []
            session.final_response = None
            session.blockers = []
            session.validation_status = "not_run"
            session.repair_attempts = 0
            session.current_phase = "planning"
            session.workflow_stage = "plan"
            session.edit_generation = 0
            session.validation_plan = []
            session.validation_runs = []
            session.diagnostics = []
            session.report = None
            session.access_mode = config.access_mode
            session.runtime_options = self._runtime_options(config)
            session.touch()
            self.session_store.save(session)

            thread = Thread(
                target=self._run_task_thread,
                args=(config, prompt, session),
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
        if self._is_active(session_id):
            return session.model_copy(
                update={
                    "status": "running",
                    "current_phase": session.current_phase,
                    "workflow_stage": session.workflow_stage,
                }
            )
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

    def _run_task_thread(self, config: AppConfig, prompt: str, session: SessionState) -> None:
        agent = AgentCore(config)
        try:
            agent.run_task(prompt, session=session)
        except Exception as exc:
            session.status = "failed"
            session.current_phase = "blocked"
            session.workflow_stage = "blocked"
            session.stop_reason = "runtime_exception"
            session.final_response = f"Task crashed: {exc}"
            session.touch()
            agent.session_store.save(session)
        finally:
            with self._lock:
                self._threads.pop(session.id, None)

    def _config_with_overrides(self, overrides: dict[str, Any]) -> AppConfig:
        config = self.base_config
        access_mode = overrides.get("access_mode")
        if access_mode is None:
            if overrides.get("read_only"):
                access_mode = AccessMode.SAFE.value
            elif overrides.get("approval_mode"):
                access_mode = AccessMode.APPROVAL.value
        for key in ("dry_run", "verbose"):
            if overrides.get(key) is not None:
                config = replace(config, **{key: overrides[key]})
        if access_mode is not None:
            config = replace(config, access_mode=access_mode).normalized()
        else:
            config = config.normalized()
        config.ensure_state_dirs()
        return config

    def _runtime_options(self, config: AppConfig) -> dict[str, Any]:
        return {
            "access_mode": config.access_mode,
            "dry_run": config.dry_run,
            "read_only": config.read_only,
            "approval_mode": config.approval_mode,
            "verbose": config.verbose,
            "max_repair_attempts": config.max_repair_attempts,
            "report_dir": str(config.report_dir_path),
        }

    def _summary_for_session(self, session: SessionState) -> SessionSummary:
        status = "running" if self._is_active(session.id) else session.status
        return SessionSummary(
            id=session.id,
            task=session.task,
            status=status,
            current_phase=session.current_phase,
            workflow_stage=session.workflow_stage,
            validation_status=session.validation_status,
            access_mode=session.access_mode,
            updated_at=session.updated_at,
            iterations=session.iterations,
            changed_file_count=len(session.changed_files),
            tool_call_count=len(session.tool_calls),
            runtime_options=session.runtime_options,
        )

    def _is_active(self, session_id: str) -> bool:
        thread = self._threads.get(session_id)
        return thread is not None and thread.is_alive()
