from __future__ import annotations

import json
from typing import Iterable

from agent.executor import Executor
from agent.memory import RepoMemoryStore
from agent.models import (
    FileChangeRecord,
    PlanItem,
    SessionState,
    ToolCallRecord,
    WorkspaceSnapshot,
)
from agent.planner import Planner
from agent.session import SessionStore
from config.settings import AppConfig
from llm.ollama_client import OllamaClient
from llm.schemas import AgentActionType, AgentDecision
from runtime.logger import AgentLogger
from runtime.tool_dispatcher import ToolDispatcher
from runtime.workspace import WorkspaceManager
from tools.filesystem import FileSystemTools
from tools.gittools import GitTools
from tools.registry import build_default_registry
from tools.safety import SafetyManager
from tools.search import SearchTools
from tools.shell import ShellTools


WRITE_TOOLS = {
    "write_file",
    "append_file",
    "create_file",
    "delete_file",
    "replace_in_file",
    "patch_file",
    "git_create_branch",
}
VERIFY_TOOLS = {"run_tests", "run_shell"}


class AgentCore:
    def __init__(self, config: AppConfig):
        self.config = config
        self.config.ensure_state_dirs()
        self.workspace = WorkspaceManager(
            config.workspace_root,
            allow_outside_root=config.full_access,
        )
        self.safety = SafetyManager(config, self.workspace)
        self.memory = RepoMemoryStore(config, self.workspace)
        self.session_store = SessionStore(config.session_dir_path)

    def inspect_workspace(self, focus: str | None = None) -> str:
        snapshot = self.memory.build_snapshot(focus)
        return self.memory.render_snapshot(snapshot)

    def run_task(self, task: str, session: SessionState | None = None) -> SessionState:
        session = session or SessionState(
            task=task,
            workspace_root=str(self.workspace.root),
            access_mode=self.config.access_mode,
        )
        if session.task != task:
            session.task = task
        session.status = "running"
        session.access_mode = self.config.access_mode
        session.runtime_options = self._runtime_options()
        session.touch()

        logger = AgentLogger(self.config.log_dir_path, session.id, verbose=self.config.verbose)
        filesystem = FileSystemTools(self.config, self.workspace, self.safety)
        search = SearchTools(self.config, self.workspace, self.memory)
        shell = ShellTools(self.config, self.workspace, self.safety)
        gittools = GitTools(self.config, self.workspace)
        registry = build_default_registry(filesystem, search, shell, gittools)
        dispatcher = ToolDispatcher(registry, logger, self.safety)
        executor = Executor(dispatcher)
        llm = OllamaClient(self.config)
        planner = Planner(llm, registry.render_for_prompt())

        if session.workspace_snapshot is None:
            session.workspace_snapshot = self.memory.build_snapshot(task)
        if not session.plan:
            session.current_phase = "planning"
            plan = planner.create_plan(task, session.workspace_snapshot)
            session.plan_summary = plan.summary
            session.plan = [PlanItem(step=step) for step in plan.steps]
            if session.plan:
                session.plan[0].status = "in_progress"
            session.candidate_files = self._unique(
                plan.files_to_inspect or session.workspace_snapshot.important_files[:12]
            )
            session.verification_commands = self._unique(
                plan.tests_to_run or session.workspace_snapshot.likely_commands[:6]
            )
            session.completion_criteria = self._unique(plan.completion_criteria)
            session.current_phase = "exploring"
        self.session_store.save(session)

        logger.log_event(
            "task_started",
            session_id=session.id,
            task=task,
            workspace=str(self.workspace.root),
            access_mode=self.config.access_mode,
            dry_run=self.config.dry_run,
        )

        final_response: str | None = None
        while (
            session.iterations < self.config.max_iterations
            and len(session.tool_calls) < self.config.max_tool_calls
        ):
            session.current_phase = self._determine_phase(session)
            decision = planner.decide_next_action(task, session)
            decision = self._enforce_iteration_rules(session, decision)
            logger.log_event(
                "decision",
                iteration=session.iterations + 1,
                phase=session.current_phase,
                action_type=decision.action_type,
                tool_name=decision.tool_name,
                expected_outcome=decision.expected_outcome,
                thought_summary=decision.thought_summary,
            )

            if decision.action_type == AgentActionType.FINAL:
                final_response = decision.final_response or planner.summarize_session(session)
                session.stop_reason = session.stop_reason or self._derive_stop_reason(session)
                session.status = self._resolve_final_status(session)
                break

            session.iterations += 1
            result = executor.execute(decision, session.iterations)
            session.tool_calls.append(
                ToolCallRecord(
                    iteration=session.iterations,
                    tool_name=decision.tool_name or "",
                    tool_args=decision.tool_args,
                    success=result.success,
                    summary=result.message,
                    phase=session.current_phase,
                    thought_summary=decision.thought_summary,
                    expected_outcome=decision.expected_outcome,
                    output_excerpt=self._build_output_excerpt(result.data),
                    risk_level=result.risk_level,
                )
            )
            self._merge_changed_files(session, result.changed_files)
            command = result.data.get("command")
            if command and command not in session.executed_commands:
                session.executed_commands.append(command)
            self._append_note(session, result)
            self._update_session_after_result(session, decision, result)
            self._advance_plan(session, result)
            session.touch()
            self.session_store.save(session)

        if final_response is None:
            session.stop_reason = session.stop_reason or self._derive_limit_reason(session)
            session.status = "partial" if session.tool_calls else "failed"
            final_response = planner.summarize_session(session)

        session.final_response = final_response
        if session.status == "running":
            session.status = self._resolve_final_status(session)
        if session.status == "completed":
            session.current_phase = "completed"
        elif session.blockers:
            session.current_phase = "blocked"
        session.touch()
        self.session_store.save(session)
        logger.log_event(
            "task_finished",
            session_id=session.id,
            status=session.status,
            phase=session.current_phase,
            iterations=session.iterations,
            validation_status=session.validation_status,
            stop_reason=session.stop_reason,
        )
        return session

    def _append_note(self, session: SessionState, result) -> None:
        if result.data.get("snapshot"):
            session.workspace_snapshot = WorkspaceSnapshot.model_validate(result.data["snapshot"])
        note = f"{result.tool_name}: {result.message}"
        if result.data.get("exit_code") is not None:
            note += f" (exit={result.data['exit_code']})"
        session.notes.append(note)
        session.notes = session.notes[-30:]

    def _advance_plan(self, session: SessionState, result) -> None:
        if not session.plan:
            return
        current_index = next(
            (idx for idx, item in enumerate(session.plan) if item.status == "in_progress"),
            None,
        )
        if current_index is None:
            return

        current = session.plan[current_index]
        if result.success:
            current.status = "completed"
            if current_index + 1 < len(session.plan):
                next_item = session.plan[current_index + 1]
                if next_item.status == "pending":
                    next_item.status = "in_progress"
        elif result.data.get("blocked") or "blocked" in result.message.lower():
            current.status = "blocked"

    def _build_output_excerpt(self, data: dict) -> str | None:
        if not data:
            return None
        excerpt: str | None = None
        if data.get("content"):
            excerpt = str(data["content"])
        elif data.get("matches"):
            excerpt = json.dumps(data["matches"][:10], ensure_ascii=False, indent=2)
        elif data.get("files"):
            excerpt = json.dumps(data["files"][:20], ensure_ascii=False, indent=2)
        elif data.get("snapshot"):
            snapshot = data["snapshot"]
            excerpt = snapshot.get("repo_summary") or json.dumps(snapshot, ensure_ascii=False, indent=2)
        elif data.get("stdout") or data.get("stderr"):
            excerpt = "\n".join(
                part for part in [data.get("stdout", ""), data.get("stderr", "")] if part
            )
        elif data.get("diff"):
            excerpt = str(data["diff"])
        elif data.get("message"):
            excerpt = str(data["message"])
        if excerpt is None:
            excerpt = json.dumps(data, ensure_ascii=False, indent=2)
        return excerpt[: self.config.max_read_chars]

    def _determine_phase(self, session: SessionState) -> str:
        if session.blockers:
            return "blocked"
        if session.validation_status == "failed":
            return "repairing"
        if session.changed_files and session.validation_status != "passed":
            return "verifying"
        if session.tool_calls and session.tool_calls[-1].tool_name in WRITE_TOOLS:
            return "editing"
        if session.plan and any(item.status == "in_progress" for item in session.plan):
            return "exploring"
        return "exploring"

    def _enforce_iteration_rules(
        self,
        session: SessionState,
        decision: AgentDecision,
    ) -> AgentDecision:
        if (
            session.repair_attempts >= self.config.max_repair_attempts
            and session.validation_status == "failed"
        ):
            session.stop_reason = "max_repair_attempts_reached"
            return AgentDecision(
                thought_summary="Repair limit reached after repeated failed validation.",
                action_type=AgentActionType.FINAL,
                tool_name=None,
                tool_args={},
                expected_outcome="Return the current state and blocker.",
                final_response=None,
            )

        if (
            decision.action_type == AgentActionType.FINAL
            and session.changed_files
            and session.validation_status != "passed"
            and not session.blockers
        ):
            command = self._pick_validation_command(session)
            if command:
                return AgentDecision(
                    thought_summary="Code changed but is not yet validated.",
                    action_type=AgentActionType.CALL_TOOL,
                    tool_name="run_tests",
                    tool_args={"command": command, "cwd": ".", "timeout": self.config.shell_timeout},
                    expected_outcome="Verify the current implementation before finalizing.",
                    final_response=None,
                )
            session.stop_reason = "validation_command_missing"
        return decision

    def _update_session_after_result(self, session: SessionState, decision, result) -> None:
        if result.data.get("snapshot"):
            session.workspace_snapshot = WorkspaceSnapshot.model_validate(result.data["snapshot"])
            if not session.candidate_files:
                session.candidate_files = self._unique(
                    session.workspace_snapshot.important_files[:12]
                )

        if decision.tool_name == "search_in_files" and result.success:
            matched_paths = [item["path"] for item in result.data.get("matches", [])]
            session.candidate_files = self._unique(
                [*matched_paths[:10], *session.candidate_files]
            )

        if decision.tool_name == "list_files" and result.success:
            session.candidate_files = self._unique(
                [*result.data.get("files", [])[:20], *session.candidate_files]
            )

        if decision.tool_name in WRITE_TOOLS and result.success:
            session.validation_status = "not_run"
            session.last_error = None
            session.blockers = []
            self._track_helper_artifacts(session, result.changed_files)

        if decision.tool_name in VERIFY_TOOLS:
            if result.success:
                session.validation_status = "passed"
                session.repair_attempts = 0
                session.last_error = None
            elif result.data.get("blocked"):
                session.validation_status = "blocked"
                self._add_blocker(session, result.message)
            else:
                session.validation_status = "failed"
                session.repair_attempts += 1
                session.last_error = self._build_output_excerpt(result.data) or result.message

        if not result.success and result.data.get("blocked"):
            self._add_blocker(session, result.message)

    def _add_blocker(self, session: SessionState, message: str) -> None:
        if message not in session.blockers:
            session.blockers.append(message)
            session.blockers = session.blockers[-10:]

    def _merge_changed_files(
        self,
        session: SessionState,
        changes: Iterable[FileChangeRecord],
    ) -> None:
        for change in changes:
            existing_index = next(
                (idx for idx, item in enumerate(session.changed_files) if item.path == change.path),
                None,
            )
            if existing_index is None:
                session.changed_files.append(change)
            else:
                session.changed_files[existing_index] = change

    def _track_helper_artifacts(
        self,
        session: SessionState,
        changes: Iterable[FileChangeRecord],
    ) -> None:
        for change in changes:
            lowered = change.path.lower()
            if any(token in lowered for token in ("helper", "harness", "parser", "converter")):
                if change.path not in session.helper_artifacts:
                    session.helper_artifacts.append(change.path)

    def _pick_validation_command(self, session: SessionState) -> str | None:
        for command in session.verification_commands:
            if command:
                return command
        if session.workspace_snapshot:
            for command in session.workspace_snapshot.likely_commands:
                if command:
                    return command
        return None

    def _resolve_final_status(self, session: SessionState) -> str:
        if session.blockers or session.validation_status in {"failed", "blocked"}:
            return "partial"
        if session.changed_files and session.validation_status != "passed":
            return "partial"
        return "completed" if session.tool_calls else "failed"

    def _derive_stop_reason(self, session: SessionState) -> str:
        if session.blockers:
            return "blocked"
        if session.changed_files and session.validation_status == "passed":
            return "validated"
        if session.changed_files and session.validation_status != "passed":
            return "unverified_changes"
        return "analysis_complete"

    def _derive_limit_reason(self, session: SessionState) -> str:
        if session.iterations >= self.config.max_iterations:
            return "max_iterations_reached"
        if len(session.tool_calls) >= self.config.max_tool_calls:
            return "max_tool_calls_reached"
        return "loop_stopped"

    def _runtime_options(self) -> dict[str, object]:
        return {
            "access_mode": self.config.access_mode,
            "dry_run": self.config.dry_run,
            "read_only": self.config.read_only,
            "approval_mode": self.config.approval_mode,
            "verbose": self.config.verbose,
            "max_repair_attempts": self.config.max_repair_attempts,
            "helper_dir": str(self.config.helper_dir_path),
        }

    def _unique(self, values: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        items: list[str] = []
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            items.append(value)
        return items
