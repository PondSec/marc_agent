from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

from agent.diagnostics import FailureAnalyzer
from agent.executor import Executor
from agent.layered_memory import AgentMemoryStore
from agent.memory import RepoMemoryStore
from agent.models import (
    DiagnosticRecord,
    FileChangeRecord,
    RepairAttemptRecord,
    PlanItem,
    SessionState,
    ToolCallRecord,
    ValidationCommand,
    ValidationRunRecord,
    WorkspaceSnapshot,
)
from agent.planner import Planner
from agent.reporting import SessionReporter
from agent.session import SessionStore
from agent.verification import ValidationPlanner
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
SEMANTIC_RUNTIME_OPERATIONS = {"router_generation", "task_state_generation", "task_understanding"}
READ_LIKE_TOOLS = {"inspect_workspace", "list_files", "search_in_files", "read_file", "show_diff", "git_status", "git_diff", "git_log"}


class AgentCore:
    def __init__(self, config: AppConfig):
        self.config = config
        self.config.ensure_state_dirs()
        self.workspace = WorkspaceManager(
            config.workspace_root,
            allow_outside_root=config.full_access,
        )
        self.safety = SafetyManager(config, self.workspace)
        self.memory = self._build_memory_store()
        self.session_store = SessionStore(config.session_dir_path)
        self.failure_analyzer = FailureAnalyzer(
            self.workspace,
            max_excerpt=self.config.max_read_chars,
        )
        self.validation_planner = ValidationPlanner()
        self.reporter = SessionReporter(config)

    def inspect_workspace(self, focus: str | None = None) -> str:
        snapshot = self.memory.build_snapshot(focus)
        return self.memory.render_snapshot(snapshot)

    def run_task(
        self,
        task: str,
        session: SessionState | None = None,
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> SessionState:
        session = session or SessionState(
            task=task,
            workspace_root=str(self.workspace.root),
            access_mode=self.config.access_mode,
        )
        if session.task != task:
            session.task = task
        session.status = "running"
        session.access_mode = self.config.access_mode
        self.memory = self._build_memory_store(session)
        session.project_id = session.project_id or getattr(self.memory, "project_id", None)
        session.runtime_options = self._runtime_options(session)
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
        planner = Planner(llm, registry.render_for_prompt(), logger=logger)

        if session.workspace_snapshot is None:
            session.workspace_snapshot = self.memory.build_snapshot(task)
        self._refresh_memory_session(task, session)
        if not session.plan:
            self._initialize_session(task, session, planner)
            self._refresh_memory_session(task, session)
        else:
            self._refresh_session_context(task, session)
            self._refresh_memory_session(task, session)
        self.session_store.save(session)

        logger.log_event(
            "task_started",
            session_id=session.id,
            task=task,
            workspace=str(self.workspace.root),
            access_mode=self.config.access_mode,
            dry_run=self.config.dry_run,
            workflow_stage=session.workflow_stage,
        )

        final_response: str | None = None
        while (
            session.iterations < self.config.max_iterations
            and len(session.tool_calls) < self.config.max_tool_calls
        ):
            if self._should_stop(session, should_stop):
                session.stop_requested = True
                session.stop_reason = "user_cancelled"
                session.status = "partial"
                logger.log_event(
                    "task_stopped",
                    session_id=session.id,
                    phase=session.current_phase,
                    workflow_stage=session.workflow_stage,
                )
                final_response = (
                    "Ich habe den laufenden Agenten auf deinen Wunsch gestoppt."
                    "\n\n"
                    "Du kannst direkt weiterschreiben oder einen neuen Chat starten."
                )
                break

            session.current_phase = self._determine_phase(session)
            session.workflow_stage = self._determine_workflow_stage(session)
            decision = planner.decide_next_action(task, session)
            decision = self._enforce_iteration_rules(session, decision)
            logger.log_event(
                "decision",
                iteration=session.iterations + 1,
                phase=session.current_phase,
                workflow_stage=session.workflow_stage,
                action_type=decision.action_type,
                tool_name=decision.tool_name,
                expected_outcome=decision.expected_outcome,
                thought_summary=decision.thought_summary,
            )

            if decision.action_type == AgentActionType.FINAL:
                final_response = decision.final_response or planner.summarize_session(session)
                session.stop_reason = session.stop_reason or self._derive_stop_reason(session)
                session.status = self._resolve_final_status(session, final_action=True)
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
                    tool_meta=result.tool_meta,
                )
            )
            self._merge_changed_files(session, result.changed_files)
            command = result.data.get("command")
            if command and command not in session.executed_commands:
                session.executed_commands.append(command)
            self._append_note(session, result)
            self._add_diagnostics(session, result)
            self._update_session_after_result(task, session, decision, result)
            self._advance_plan(session, result)
            self._refresh_memory_session(task, session)
            session.touch()
            self.session_store.save(session)

        if final_response is None:
            session.stop_reason = session.stop_reason or self._derive_limit_reason(session)
            session.status = "partial" if session.tool_calls else "failed"
            final_response = planner.summarize_session(session)

        session.status = self._resolve_final_status(session, final_action=final_response is not None)
        session.current_phase = "reporting"
        session.workflow_stage = "report"
        session.report = self.reporter.build_report(session)
        session.final_response = self.reporter.render_final_response(
            session,
            draft_response=final_response,
        )
        session.append_message("assistant", session.final_response)
        self._refresh_memory_session(task, session)
        self._persist_memory_session(session)

        if session.status == "completed":
            session.current_phase = "completed"
            session.workflow_stage = "completed"
        elif session.stop_reason == "user_cancelled":
            session.current_phase = "blocked"
            session.workflow_stage = "blocked"
        elif session.blockers:
            session.current_phase = "blocked"
            session.workflow_stage = "blocked"
        session.touch()
        self.session_store.save(session)
        logger.log_event(
            "task_finished",
            session_id=session.id,
            status=session.status,
            phase=session.current_phase,
            workflow_stage=session.workflow_stage,
            iterations=session.iterations,
            validation_status=session.validation_status,
            stop_reason=session.stop_reason,
            report_path=session.report.report_path if session.report else None,
        )
        return session

    def _should_stop(
        self,
        session: SessionState,
        should_stop: Callable[[], bool] | None,
    ) -> bool:
        if session.stop_requested:
            return True
        if should_stop is None:
            return False
        try:
            return bool(should_stop())
        except Exception:
            return False

    def _initialize_session(
        self,
        task: str,
        session: SessionState,
        planner: Planner,
    ) -> None:
        session.current_phase = "planning"
        session.workflow_stage = "plan"
        session.task_state = planner.update_task_state(
            task,
            session.workspace_snapshot,
            session=session,
        )
        session.task_understanding = session.task_state.to_task_understanding()
        session.router_result = planner.route_task_state(
            session.task_state,
            session.workspace_snapshot,
            session=session,
        )
        session.task_analysis = {
            "task_state": session.task_state.model_dump(),
            "task_understanding": session.task_understanding.model_dump(),
            "router_result": session.router_result.model_dump(),
        }
        if session.router_result.direct_response and not session.router_result.repo_context_needed:
            session.plan_summary = "Respond directly without repository work."
            session.plan = []
            session.completion_criteria = ["Return a concise, user-facing reply."]
            session.current_phase = "exploring"
            session.workflow_stage = "discover"
            return
        plan = planner.create_plan(task, session, session.router_result)
        session.plan_summary = plan.summary
        session.plan = [PlanItem(step=step) for step in plan.steps]
        if session.plan:
            session.plan[0].status = "in_progress"
        self._refresh_session_context(task, session, planned_files=plan.files_to_inspect)
        session.completion_criteria = self._unique(plan.completion_criteria)
        session.current_phase = "exploring"
        session.workflow_stage = "discover"

    def _refresh_session_context(
        self,
        task: str,
        session: SessionState,
        *,
        planned_files: list[str] | None = None,
    ) -> None:
        snapshot = session.workspace_snapshot
        if snapshot is None:
            return
        route = session.router_result
        follow_up_context = session.follow_up_context
        extension_candidates: list[str] = []
        if route and route.relevant_extensions:
            extension_candidates = [
                path
                for path in snapshot.important_files[:24]
                if any(path.endswith(extension) for extension in route.relevant_extensions)
            ]
        follow_up_files: list[str] = []
        if follow_up_context is not None:
            follow_up_files.extend(follow_up_context.target_paths)
            follow_up_files.extend(follow_up_context.changed_files)
            follow_up_files.extend(follow_up_context.read_files)
            follow_up_files.extend(
                path
                for item in follow_up_context.diagnostics[-6:]
                for path in item.file_hints
            )
        understanding_files: list[str] = []
        if session.task_understanding is not None:
            understanding_files.extend(
                artifact.path
                for artifact in session.task_understanding.target_artifacts
                if artifact.path
            )
        task_state_files: list[str] = []
        if session.task_state is not None:
            task_state_files.extend(
                artifact.path
                for artifact in session.task_state.target_artifacts
                if artifact.path
            )
        candidate_files = [
            *session.candidate_files,
            *self._memory_guided_candidate_files(session),
            *task_state_files,
            *understanding_files,
            *follow_up_files,
            *((route.entities.target_paths if route else [])),
            *(planned_files or []),
            *extension_candidates,
            *snapshot.focus_files,
            *snapshot.important_files[:18],
        ]
        diagnostic_files = [
            path
            for item in session.diagnostics[-6:]
            for path in item.file_hints
        ]
        session.candidate_files = self._unique([*diagnostic_files, *candidate_files])[:24]
        session.validation_plan = self.validation_planner.build_plan(
            task,
            snapshot,
            changed_files=[item.path for item in session.changed_files],
            session=session,
        )
        session.verification_commands = [item.command for item in session.validation_plan]
        if not session.completion_criteria:
            session.completion_criteria = [
                "Relevant files were inspected before editing.",
                "Changes were validated with the project-aware command plan or blocked with a clear reason.",
                "Final output includes changed files, commands, diagnostics, and stop reason.",
            ]

    def _build_memory_store(self, session: SessionState | None = None):
        profile = self._agent_profile(session)
        if profile == "a1":
            return RepoMemoryStore(self.config, self.workspace)
        return AgentMemoryStore(self.config, self.workspace)

    def _agent_profile(self, session: SessionState | None = None) -> str:
        runtime_options = getattr(session, "runtime_options", {}) or {}
        raw = str(runtime_options.get("agent_profile") or "").strip().lower()
        if raw == "a1":
            return "a1"
        return "a2"

    def _memory_guided_candidate_files(self, session: SessionState) -> list[str]:
        memory_context = getattr(session, "memory_context", None)
        if memory_context is None:
            return []
        candidates: list[str] = []
        candidates.extend(memory_context.suggested_files[:8])
        for item in memory_context.selected:
            candidates.extend(item.file_paths[:6])
            entry = item.entry
            if entry is None:
                continue
            candidates.extend(list(getattr(entry, "chosen_targets", []) or [])[:6])
            candidates.extend(list(getattr(entry, "changed_files", []) or [])[:4])
            candidates.extend(list(getattr(entry, "known_hotspots", []) or [])[:4])
        return self._unique(candidates)

    def _refresh_memory_session(self, task: str, session: SessionState) -> None:
        if hasattr(self.memory, "refresh_session_memory"):
            self.memory.refresh_session_memory(task, session)
            return
        session.working_memory = None
        session.memory_context = None

    def _persist_memory_session(self, session: SessionState) -> None:
        if hasattr(self.memory, "persist_session_memory"):
            self.memory.persist_session_memory(session)

    def _append_note(self, session: SessionState, result) -> None:
        if result.data.get("snapshot"):
            session.workspace_snapshot = WorkspaceSnapshot.model_validate(result.data["snapshot"])
        note = self._tool_note_summary(result)
        if result.data.get("exit_code") is not None:
            note += f" (exit={result.data['exit_code']})"
        if session.notes and session.notes[-1] == note:
            return
        if note in session.notes:
            session.notes.remove(note)
        session.notes.append(note)
        session.notes = session.notes[-40:]

    def _tool_note_summary(self, result) -> str:
        tool_name = str(result.tool_name or "").strip()
        data = getattr(result, "data", {}) or {}
        changed_files = list(getattr(result, "changed_files", []) or [])
        path = self._tool_note_path(data, changed_files)
        command = self._tool_note_command(data)
        label = self._tool_note_label(tool_name, path=path, command=command, data=data)
        if result.success:
            return label
        message = str(getattr(result, "message", "") or "").strip()
        if not message:
            return f"{label} failed"
        return f"{label} failed: {self._trim_note_text(message, 140)}"

    def _tool_note_label(
        self,
        tool_name: str,
        *,
        path: str | None,
        command: str | None,
        data: dict[str, object],
    ) -> str:
        target = self._trim_note_text(path or "", 72) if path else None
        command_label = self._trim_note_text(command or "", 72) if command else None
        branch = str(
            data.get("branch")
            or data.get("branch_name")
            or data.get("name")
            or ""
        ).strip()
        labels = {
            "inspect_workspace": "Inspected workspace",
            "list_files": f"Listed files in {target}" if target else "Listed files",
            "search_in_files": f"Searched {target}" if target else "Searched files",
            "read_file": f"Read {target}" if target else "Read file",
            "show_diff": f"Read diff for {target}" if target else "Read diff",
            "write_file": f"Updated {target}" if target else "Updated file",
            "append_file": f"Appended to {target}" if target else "Appended file",
            "create_file": f"Created {target}" if target else "Created file",
            "delete_file": f"Deleted {target}" if target else "Deleted file",
            "replace_in_file": f"Updated {target}" if target else "Updated file",
            "patch_file": f"Patched {target}" if target else "Patched file",
            "run_tests": f"Ran {command_label}" if command_label else "Ran validation",
            "run_shell": f"Executed {command_label}" if command_label else "Executed shell command",
            "git_status": "Read git status",
            "git_diff": "Read git diff",
            "git_log": "Read git history",
            "git_create_branch": (
                f"Created branch {self._trim_note_text(branch, 60)}"
                if branch
                else "Created branch"
            ),
        }
        if tool_name in labels:
            return labels[tool_name]
        if tool_name in READ_LIKE_TOOLS:
            return f"Read via {tool_name}"
        return self._trim_note_text(f"{tool_name}: {str(data.get('message') or '')}".strip(" :"), 100) or tool_name

    def _tool_note_path(self, data: dict[str, object], changed_files: list[FileChangeRecord]) -> str | None:
        if changed_files:
            candidate = str(getattr(changed_files[0], "path", "") or "").strip()
            if candidate:
                return candidate
        for key in ("path", "file_path", "target", "root", "directory"):
            candidate = str(data.get(key) or "").strip()
            if candidate:
                return candidate
        return None

    def _tool_note_command(self, data: dict[str, object]) -> str | None:
        for key in ("command", "cmd"):
            candidate = str(data.get(key) or "").strip()
            if candidate:
                return candidate
        return None

    def _trim_note_text(self, text: str, limit: int) -> str:
        normalized = str(text or "").strip()
        if not normalized:
            return ""
        if len(normalized) <= limit:
            return normalized
        target = Path(normalized).name.strip()
        if target and len(target) <= limit:
            return target
        return normalized[: limit - 1].rstrip() + "…"

    def _add_diagnostics(self, session: SessionState, result) -> None:
        diagnostics = self.failure_analyzer.analyze(result, iteration=session.iterations)
        if not diagnostics:
            return
        for diagnostic in diagnostics:
            if self._diagnostic_exists(session.diagnostics, diagnostic):
                continue
            session.diagnostics.append(diagnostic)
            session.diagnostics = session.diagnostics[-20:]
            if diagnostic.file_hints:
                session.candidate_files = self._unique(
                    [*diagnostic.file_hints, *session.candidate_files]
                )[:24]

    def _diagnostic_exists(
        self,
        diagnostics: list[DiagnosticRecord],
        candidate: DiagnosticRecord,
    ) -> bool:
        return any(
            item.category == candidate.category
            and item.summary == candidate.summary
            and item.command == candidate.command
            for item in diagnostics
        )

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
        if session.validation_status == "bootstrap_reset_required":
            return "blocked"
        if session.blockers:
            return "blocked"
        if session.validation_status in {"failed", "bootstrap_failed"}:
            return "repairing"
        if session.changed_files and self.validation_planner.pending_commands(session):
            return "verifying"
        if session.tool_calls and session.tool_calls[-1].tool_name in WRITE_TOOLS:
            return "editing"
        if session.plan and any(item.status == "in_progress" for item in session.plan):
            return "exploring"
        return "exploring"

    def _determine_workflow_stage(self, session: SessionState) -> str:
        if session.blockers:
            return "blocked"
        if session.current_phase == "planning":
            return "plan"
        if session.current_phase == "exploring":
            return "discover"
        if session.current_phase == "editing":
            return "act"
        if session.current_phase == "verifying":
            return "verify"
        if session.current_phase == "repairing":
            return "repair"
        if session.current_phase == "reporting":
            return "report"
        if session.current_phase == "completed":
            return "completed"
        return "discover"

    def _enforce_iteration_rules(
        self,
        session: SessionState,
        decision: AgentDecision,
    ) -> AgentDecision:
        if (
            session.repair_attempts >= self.config.max_repair_attempts
            and session.validation_status in {"failed", "bootstrap_failed"}
        ):
            if self._should_allow_progressive_repair_write(session, decision):
                return decision
            if session.validation_status == "bootstrap_failed":
                session.validation_status = "bootstrap_reset_required"
                session.stop_reason = "needs_human_or_bootstrap_reset"
            else:
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
                return self._validation_decision(
                    "Code changed but the validation plan is not complete yet.",
                    command,
                )
            session.stop_reason = "validation_command_missing"
        return decision

    def _should_allow_progressive_repair_write(
        self,
        session: SessionState,
        decision: AgentDecision,
    ) -> bool:
        if decision.action_type != AgentActionType.CALL_TOOL:
            return False
        if str(decision.tool_name or "").strip() not in WRITE_TOOLS:
            return False
        repair_context = session.active_repair_context
        if repair_context is None:
            return False
        latest_attempt = self._latest_verified_failed_repair_attempt(session)
        if latest_attempt is None or latest_attempt.behavior_changed is not True:
            return False
        target = str(decision.tool_args.get("path") or "").strip()
        if not target:
            return False
        brief = repair_context.repair_brief
        if brief is None:
            return target == str(latest_attempt.artifact_path or "").strip()
        allowed_files = {
            str(item or "").strip()
            for item in brief.allowed_files
            if str(item or "").strip()
        }
        locked_target = str(brief.locked_target or brief.primary_target or "").strip()
        if allowed_files and target not in allowed_files:
            return False
        if locked_target and target != locked_target and allowed_files and target not in allowed_files:
            return False
        latest_target = str(latest_attempt.artifact_path or "").strip()
        return not latest_target or target == latest_target or target == locked_target or target in allowed_files

    def _latest_verified_failed_repair_attempt(
        self,
        session: SessionState,
    ) -> RepairAttemptRecord | None:
        return next(
            (
                attempt
                for attempt in reversed(session.repair_history)
                if attempt.result == "mutation_planned" and attempt.independent_verification is False
            ),
            None,
        )

    def _validation_decision(
        self,
        thought_summary: str,
        command: ValidationCommand,
    ) -> AgentDecision:
        return AgentDecision(
            thought_summary=thought_summary,
            action_type=AgentActionType.CALL_TOOL,
            tool_name="run_tests",
            tool_args={
                "command": command.command,
                "cwd": command.cwd,
                "timeout": self.config.shell_timeout,
            },
            expected_outcome=f"Run the next {command.kind} validation step.",
            final_response=None,
        )

    def _update_session_after_result(self, task: str, session: SessionState, decision, result) -> None:
        if result.data.get("snapshot"):
            session.workspace_snapshot = WorkspaceSnapshot.model_validate(result.data["snapshot"])
            self._refresh_session_context(task, session)

        if decision.tool_name == "search_in_files" and result.success:
            matched_paths = [item["path"] for item in result.data.get("matches", [])]
            session.candidate_files = self._unique(
                [*matched_paths[:10], *session.candidate_files]
            )[:24]

        if decision.tool_name == "list_files" and result.success:
            session.candidate_files = self._unique(
                [*result.data.get("files", [])[:20], *session.candidate_files]
            )[:24]

        if decision.tool_name in WRITE_TOOLS and result.success:
            if session.validation_status in {"failed", "bootstrap_failed"}:
                session.repair_attempts += 1
            session.edit_generation += 1
            session.validation_status = "not_run"
            session.active_repair_context = None
            session.last_error = None
            session.blockers = []
            self._track_helper_artifacts(session, result.changed_files)
            # Rebuild the snapshot after successful mutations so newly created tests,
            # entrypoints, and validation commands become visible immediately.
            session.workspace_snapshot = self.memory.build_snapshot(task)
            self._refresh_session_context(task, session)

        if decision.tool_name in VERIFY_TOOLS:
            run = self._record_validation_run(session, decision, result)
            failure_evidence = None
            if run.status in {"failed", "timeout"}:
                failure_evidence = self.validation_planner.build_failure_evidence(session, run)
                self._decorate_failed_validation_run(run, failure_evidence)
                self._mark_repair_attempt_verification_outcome(
                    session,
                    run,
                    failure_evidence=failure_evidence,
                )
                if self.validation_planner.should_require_bootstrap_reset(session, failure_evidence.repair_brief):
                    run.bootstrap_status = "bootstrap_reset_required"
                    failure_evidence.bootstrap_status = "bootstrap_reset_required"
                    if failure_evidence.repair_brief is not None:
                        failure_evidence.repair_brief.bootstrap_status = "bootstrap_reset_required"
            elif run.status == "passed":
                self._mark_repair_attempt_verification_outcome(session, run)
            session.validation_status = self.validation_planner.rollup_status(session)
            if session.validation_status in {"failed", "bootstrap_failed", "bootstrap_reset_required"}:
                failed_run = self.validation_planner.latest_failed_run(
                    session,
                    current_generation_only=False,
                )
                if failed_run is not None:
                    session.active_repair_context = failure_evidence or self.validation_planner.build_failure_evidence(
                        session,
                        failed_run,
                    )
                    session.last_error = (
                        getattr(session.active_repair_context, "root_cause_summary", None)
                        or failed_run.excerpt
                        or failed_run.summary
                        or self._build_output_excerpt(result.data)
                        or result.message
                    )
                    self._append_validation_note(session, failed_run)
                    if session.validation_status == "bootstrap_reset_required" and session.last_error:
                        self._add_blocker(session, session.last_error)
                        session.stop_reason = "needs_human_or_bootstrap_reset"
                else:
                    session.active_repair_context = None
                    session.last_error = self._build_output_excerpt(result.data) or result.message
            elif session.validation_status == "passed":
                session.repair_attempts = 0
                session.active_repair_context = None
                session.last_error = None
            elif session.validation_status == "blocked":
                self._add_blocker(session, result.message)

        if not result.success and result.data.get("blocked"):
            self._add_blocker(session, result.message)

    def _record_validation_run(self, session: SessionState, decision, result) -> ValidationRunRecord:
        command_text = result.data.get("command") or decision.tool_args.get("command", "")
        command_identity = self.validation_planner.command_identity(command_text)
        plan_item = next(
            (
                item
                for item in session.validation_plan
                if self.validation_planner.command_identity(item.command) == command_identity
            ),
            None,
        )
        status = "passed"
        if result.data.get("blocked"):
            status = "blocked"
        elif result.data.get("timeout"):
            status = "timeout"
        elif not result.success:
            status = "failed"
        run = ValidationRunRecord(
            command=command_text,
            cwd=str(decision.tool_args.get("cwd", ".")),
            kind=plan_item.kind if plan_item else "check",
            verification_scope=(
                self.validation_planner.command_scope(plan_item)
                if plan_item is not None
                else self.validation_planner.command_scope(command_text)
            ),
            status=status,
            exit_code=result.data.get("exit_code"),
            risk_level=result.risk_level,
            iteration=session.iterations,
            edit_generation=session.edit_generation,
            summary=result.message,
            excerpt=self._build_output_excerpt(result.data),
        )
        session.validation_runs.append(run)
        session.validation_runs = session.validation_runs[-20:]
        return run

    def _decorate_failed_validation_run(
        self,
        run: ValidationRunRecord,
        failure_evidence,
    ) -> None:
        brief = getattr(failure_evidence, "repair_brief", None)
        run.failure_signature = str(getattr(brief, "failure_signature", "") or "").strip() or None
        run.root_cause_summary = str(getattr(failure_evidence, "root_cause_summary", "") or "").strip() or None
        bootstrap_status = str(getattr(failure_evidence, "bootstrap_status", "") or "").strip()
        if bootstrap_status in {"bootstrap_failed", "bootstrap_reset_required"}:
            run.bootstrap_status = bootstrap_status

    def _latest_pending_repair_attempt(self, session: SessionState):
        return next(
            (
                attempt
                for attempt in reversed(session.repair_history)
                if attempt.result == "mutation_planned" and attempt.independent_verification is None
            ),
            None,
        )

    def _mark_repair_attempt_verification_outcome(
        self,
        session: SessionState,
        run: ValidationRunRecord,
        *,
        failure_evidence=None,
    ) -> None:
        attempt = self._latest_pending_repair_attempt(session)
        if attempt is None:
            return
        if run.status == "passed":
            attempt.independent_verification = True
            attempt.behavior_changed = True
            attempt.post_validation_failure_signature = None
            return
        attempt.independent_verification = False
        if failure_evidence is None or getattr(failure_evidence, "repair_brief", None) is None:
            return
        post_signature = str(getattr(failure_evidence.repair_brief, "failure_signature", "") or "").strip() or None
        attempt.post_validation_failure_signature = post_signature
        attempt.behavior_changed = not bool(
            post_signature
            and str(attempt.failure_signature or "").strip()
            and post_signature == str(attempt.failure_signature or "").strip()
        )

    def _append_validation_note(self, session: SessionState, run: ValidationRunRecord) -> None:
        root_cause = str(getattr(run, "root_cause_summary", "") or "").strip()
        if not root_cause:
            return
        status = str(getattr(run, "bootstrap_status", "") or "").strip() or "failed"
        if status == "none":
            status = "failed"
        note = f"Validation classified as {status}: {root_cause}"
        if note not in session.notes:
            session.notes.append(note)
            session.notes = session.notes[-20:]

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

    def _pick_validation_command(self, session: SessionState) -> ValidationCommand | None:
        for item in self.validation_planner.pending_commands(session):
            if self.validation_planner.can_repeat_command(session, item.command):
                return item

        if session.workspace_snapshot:
            fallback_plan = self.validation_planner.build_plan(
                session.task,
                session.workspace_snapshot,
                changed_files=[item.path for item in session.changed_files],
                session=session,
            )
            if fallback_plan:
                session.validation_plan = fallback_plan
                session.verification_commands = [item.command for item in fallback_plan]
                for item in self.validation_planner.pending_commands(session):
                    if self.validation_planner.can_repeat_command(session, item.command):
                        return item
        return None

    def _resolve_final_status(self, session: SessionState, *, final_action: bool = False) -> str:
        if session.stop_reason == "user_cancelled" or session.stop_requested:
            return "partial"
        if self._semantic_fallback_uncertain(session):
            return "partial"
        if self._debug_repair_incomplete(session):
            return "partial"
        if self._web_functional_validation_missing(session):
            return "partial"
        if session.blockers or session.validation_status in {"failed", "blocked", "bootstrap_failed", "bootstrap_reset_required"}:
            return "partial"
        if session.changed_files and session.validation_status == "not_run":
            return "partial"
        if session.changed_files and self.validation_planner.pending_commands(session):
            return "partial"
        if self._requirements_review_missing(session):
            return "partial"
        if final_action or session.tool_calls:
            return "completed"
        return "failed"

    def _derive_stop_reason(self, session: SessionState) -> str:
        if session.validation_status == "bootstrap_reset_required":
            return "needs_human_or_bootstrap_reset"
        if session.validation_status == "bootstrap_failed":
            return "bootstrap_failed"
        if session.blockers:
            return "blocked"
        if self._semantic_fallback_uncertain(session):
            if (
                (session.router_result is not None and session.router_result.needs_clarification)
                or (session.task_state is not None and session.task_state.needs_clarification)
            ):
                return "clarification_required"
            return "analysis_incomplete"
        if self._runtime_verification_missing(session):
            return "functional_validation_missing"
        if self._web_functional_validation_missing(session):
            return "functional_validation_missing"
        if self._reproduction_missing(session):
            return "reproduction_missing"
        if self._analysis_incomplete(session):
            return "analysis_incomplete"
        if self._repair_incomplete(session):
            return "repair_incomplete"
        if session.changed_files and session.validation_status == "passed":
            if self._requirements_review_missing(session):
                return "requirements_review_missing"
            return "validated"
        if session.changed_files and session.validation_status == "not_run":
            return "validation_missing"
        if session.changed_files and self.validation_planner.pending_commands(session):
            return "validation_incomplete"
        if self._requirements_review_missing(session):
            return "requirements_review_missing"
        if self._safe_degraded_semantic_completion(session):
            resolution = self._semantic_resolution(session)
            if resolution in {"reserve_model", "reduced_model"}:
                return "reduced_semantic_complete"
            if resolution == "minimal_inference":
                return "minimal_semantic_complete"
        return "analysis_complete"

    def _derive_limit_reason(self, session: SessionState) -> str:
        if session.iterations >= self.config.max_iterations:
            return "max_iterations_reached"
        if len(session.tool_calls) >= self.config.max_tool_calls:
            return "max_tool_calls_reached"
        return "loop_stopped"

    def _runtime_options(self, session: SessionState | None = None) -> dict[str, object]:
        options = {
            "access_mode": self.config.access_mode,
            "dry_run": self.config.dry_run,
            "read_only": self.config.read_only,
            "approval_mode": self.config.approval_mode,
            "verbose": self.config.verbose,
            "max_repair_attempts": self.config.max_repair_attempts,
            "helper_dir": str(self.config.helper_dir_path),
            "report_dir": str(self.config.report_dir_path),
        }
        existing = dict(getattr(session, "runtime_options", {}) or {})
        for key in ("agent_profile", "execution_profile"):
            value = existing.get(key)
            if value:
                options[key] = value
        return options

    def _debug_repair_incomplete(self, session: SessionState) -> bool:
        if session.task_state is None or session.task_state.execution_strategy != "debug_repair":
            return False
        if session.changed_files:
            return self._runtime_verification_missing(session)
        return self._reproduction_missing(session) or self._analysis_incomplete(session) or self._repair_incomplete(session)

    def _runtime_verification_missing(self, session: SessionState) -> bool:
        if not session.changed_files:
            return False
        if not self.validation_planner.runtime_verification_required(session):
            return False
        return not self.validation_planner.has_runtime_success(session)

    def _web_functional_validation_missing(self, session: SessionState) -> bool:
        if not session.changed_files:
            return False
        if not self.validation_planner.web_functional_verification_required(session):
            return False
        if not self.validation_planner.has_structural_web_success(session):
            return False
        if (
            self.validation_planner.has_semantic_review_success(session)
            and self.validation_planner.web_structural_proxy_sufficient(session)
        ):
            return False
        return not self.validation_planner.has_runtime_success(session)

    def _requirements_review_missing(self, session: SessionState) -> bool:
        if not session.changed_files:
            return False
        return not self.validation_planner.has_semantic_review(session)

    def _reproduction_missing(self, session: SessionState) -> bool:
        if session.changed_files:
            return False
        if not self.validation_planner.runtime_verification_required(session):
            return False
        return not self.validation_planner.has_runtime_attempt(session, current_generation_only=False)

    def _analysis_incomplete(self, session: SessionState) -> bool:
        if session.changed_files:
            return False
        if session.task_state is None or session.task_state.execution_strategy != "debug_repair":
            return False
        return self.validation_planner.has_runtime_attempt(session, current_generation_only=False) and not session.diagnostics

    def _repair_incomplete(self, session: SessionState) -> bool:
        if session.changed_files:
            return False
        if session.task_state is None or session.task_state.execution_strategy != "debug_repair":
            return False
        return bool(session.diagnostics)

    def _semantic_fallback_uncertain(self, session: SessionState) -> bool:
        resolution = self._semantic_resolution(session)
        if resolution == "full_model":
            return False
        if resolution == "blocked":
            return True
        route = session.router_result
        if route is not None:
            if route.needs_clarification or route.intent.value == "unknown" or route.confidence < 0.5:
                return True
        task_state = session.task_state
        if task_state is None:
            return True
        if task_state.needs_clarification or task_state.next_action == "clarify":
            return True
        if task_state.goal_relation in {"unknown", "clarify"}:
            return True
        return task_state.confidence < 0.5

    def _safe_degraded_semantic_completion(self, session: SessionState) -> bool:
        if self._semantic_resolution(session) == "full_model":
            return False
        if self._semantic_fallback_uncertain(session):
            return False
        return not session.changed_files and not session.tool_calls

    def _semantic_resolution(self, session: SessionState) -> str:
        for item in reversed(session.runtime_executions):
            if str(item.get("operation_name") or "").strip() not in SEMANTIC_RUNTIME_OPERATIONS:
                continue
            resolution = str(item.get("semantic_resolution") or "").strip()
            if resolution:
                return resolution
            if str(item.get("final_state") or "").strip() == "degraded_success":
                return "minimal_inference"
        if session.task_state is not None:
            resolution = str(getattr(session.task_state, "semantic_resolution", "") or "").strip()
            if resolution:
                return resolution
        if session.task_understanding is not None:
            resolution = str(getattr(session.task_understanding, "semantic_resolution", "") or "").strip()
            if resolution:
                return resolution
        return "full_model"

    def _has_degraded_semantic_execution(self, session: SessionState) -> bool:
        return self._semantic_resolution(session) != "full_model"

    def _unique(self, values: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        items: list[str] = []
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            items.append(value)
        return items
