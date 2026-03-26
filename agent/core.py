from __future__ import annotations

import json

from agent.executor import Executor
from agent.memory import RepoMemoryStore
from agent.models import SessionState, ToolCallRecord, WorkspaceSnapshot
from agent.planner import Planner
from agent.session import SessionStore
from config.settings import AppConfig
from llm.ollama_client import OllamaClient
from llm.schemas import AgentActionType
from runtime.logger import AgentLogger
from runtime.tool_dispatcher import ToolDispatcher
from runtime.workspace import WorkspaceManager
from tools.filesystem import FileSystemTools
from tools.gittools import GitTools
from tools.registry import build_default_registry
from tools.safety import SafetyManager
from tools.search import SearchTools
from tools.shell import ShellTools


class AgentCore:
    def __init__(self, config: AppConfig):
        self.config = config
        self.config.ensure_state_dirs()
        self.workspace = WorkspaceManager(config.workspace_root)
        self.safety = SafetyManager(config, self.workspace)
        self.memory = RepoMemoryStore(config, self.workspace)
        self.session_store = SessionStore(config.session_dir_path)

    def inspect_workspace(self, focus: str | None = None) -> str:
        snapshot = self.memory.build_snapshot(focus)
        return self.memory.render_snapshot(snapshot)

    def run_task(self, task: str, session: SessionState | None = None) -> SessionState:
        session = session or SessionState(task=task, workspace_root=str(self.workspace.root))
        if session.task != task:
            session.task = task
        session.status = "running"
        session.touch()

        logger = AgentLogger(self.config.log_dir_path, session.id, verbose=self.config.verbose)
        filesystem = FileSystemTools(self.config, self.workspace, self.safety)
        search = SearchTools(self.config, self.workspace, self.memory)
        shell = ShellTools(self.config, self.workspace, self.safety)
        gittools = GitTools(self.config, self.workspace)
        registry = build_default_registry(filesystem, search, shell, gittools)
        dispatcher = ToolDispatcher(registry, logger)
        executor = Executor(dispatcher)
        llm = OllamaClient(self.config)
        planner = Planner(llm, registry.render_for_prompt())

        if session.workspace_snapshot is None:
            session.workspace_snapshot = self.memory.build_snapshot(task)
        if not session.plan:
            session.plan = planner.create_plan(task, session.workspace_snapshot)
            if session.plan:
                session.plan[0].status = "in_progress"

        logger.log_event(
            "task_started",
            session_id=session.id,
            task=task,
            workspace=str(self.workspace.root),
        )

        final_response: str | None = None
        while session.iterations < self.config.max_iterations and len(session.tool_calls) < self.config.max_tool_calls:
            decision = planner.decide_next_action(task, session)
            logger.log_event(
                "decision",
                iteration=session.iterations + 1,
                action_type=decision.action_type,
                tool_name=decision.tool_name,
                expected_outcome=decision.expected_outcome,
            )

            if decision.action_type == AgentActionType.FINAL:
                final_response = decision.final_response or planner.summarize_session(session)
                session.status = "completed" if session.changed_files or session.tool_calls else "partial"
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
                    output_excerpt=self._build_output_excerpt(result.data),
                    risk_level=result.risk_level,
                )
            )
            session.changed_files.extend(result.changed_files)
            command = result.data.get("command")
            if command and command not in session.executed_commands:
                session.executed_commands.append(command)
            self._append_note(session, result)
            self._advance_plan(session, result)
            session.touch()
            self.session_store.save(session)

        if final_response is None:
            session.status = "partial" if session.tool_calls else "failed"
            final_response = planner.summarize_session(session)

        session.final_response = final_response
        if session.status == "running":
            session.status = "completed"
        session.touch()
        self.session_store.save(session)
        logger.log_event(
            "task_finished",
            session_id=session.id,
            status=session.status,
            iterations=session.iterations,
        )
        return session

    def _append_note(self, session: SessionState, result) -> None:
        if result.data.get("snapshot"):
            session.workspace_snapshot = WorkspaceSnapshot.model_validate(result.data["snapshot"])
        note = f"{result.tool_name}: {result.message}"
        if result.data.get("exit_code") is not None:
            note += f" (exit={result.data['exit_code']})"
        session.notes.append(note)
        session.notes = session.notes[-20:]

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
        elif "blocked" in result.message.lower():
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
