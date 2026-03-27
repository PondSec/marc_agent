from __future__ import annotations

import json
import shlex
import shutil
from pathlib import Path

from agent.models import SessionState, ValidationCommand, WorkspaceSnapshot


class ValidationPlanner:
    DEFAULT_KIND_ORDER = {
        "test": 10,
        "lint": 20,
        "typecheck": 30,
        "build": 40,
        "check": 50,
        "dev": 60,
        "release": 90,
        "deploy": 100,
    }

    def build_plan(
        self,
        task: str,
        snapshot: WorkspaceSnapshot,
        *,
        changed_files: list[str] | None = None,
    ) -> list[ValidationCommand]:
        commands = snapshot.validation_commands or [
            ValidationCommand(command=item, kind="check", source="fallback")
            for item in snapshot.likely_commands
        ]
        if not commands:
            commands = self._default_commands(changed_files or [])
        if not commands:
            return []

        relevant_kinds = self._relevant_kinds(task, changed_files or [], snapshot)
        ranked: list[tuple[int, ValidationCommand]] = []
        seen: set[str] = set()

        for command in commands:
            if command.command in seen:
                continue
            seen.add(command.command)
            penalty = 0
            if relevant_kinds and command.kind not in relevant_kinds and command.kind in {
                "build",
                "release",
                "deploy",
            }:
                penalty += 50
            ranked.append((command.priority + penalty, command))

        ranked.sort(key=lambda item: (item[0], item[1].command))
        selected = [item[1] for item in ranked[:6]]

        if relevant_kinds:
            required = {kind for kind in relevant_kinds if kind in {"test", "lint", "typecheck", "build"}}
            selected = [
                command.model_copy(update={"required": command.kind in required or command.required})
                for command in selected
            ]
        return selected

    def pending_commands(self, session: SessionState) -> list[ValidationCommand]:
        required_commands = [
            command for command in session.validation_plan if command.required
        ] or session.validation_plan
        if not required_commands:
            return []

        passed = {
            run.command
            for run in session.validation_runs
            if run.edit_generation == session.edit_generation and run.status == "passed"
        }
        return [command for command in required_commands if command.command not in passed]

    def next_command(self, session: SessionState) -> ValidationCommand | None:
        pending = self.pending_commands(session)
        return pending[0] if pending else None

    def rollup_status(self, session: SessionState) -> str:
        if not session.validation_plan:
            if not session.changed_files:
                return "not_run"
            current_runs = [
                run for run in session.validation_runs if run.edit_generation == session.edit_generation
            ]
            if not current_runs:
                return "not_run"
            if any(run.status == "blocked" for run in current_runs):
                return "blocked"
            if any(run.status in {"failed", "timeout"} for run in current_runs):
                return "failed"
            if any(run.status == "passed" for run in current_runs):
                return "passed"
            return "not_run"

        current_runs = [
            run for run in session.validation_runs if run.edit_generation == session.edit_generation
        ]
        if not current_runs:
            return "not_run"
        if any(run.status == "blocked" for run in current_runs):
            return "blocked"
        if any(run.status in {"failed", "timeout"} for run in current_runs):
            return "failed"
        if not self.pending_commands(session):
            return "passed"
        return "not_run"

    def _relevant_kinds(
        self,
        task: str,
        changed_files: list[str],
        snapshot: WorkspaceSnapshot,
    ) -> set[str]:
        lowered_task = task.lower()
        kinds: set[str] = set()

        if any(token in lowered_task for token in ("test", "bug", "fix", "beheb", "repair")):
            kinds.add("test")
        if any(token in lowered_task for token in ("lint", "format", "style")):
            kinds.add("lint")
        if any(token in lowered_task for token in ("type", "typing", "ts", "mypy", "pyright")):
            kinds.add("typecheck")
        if any(token in lowered_task for token in ("build", "bundle")):
            kinds.add("build")
        if any(token in lowered_task for token in ("release", "deploy")):
            kinds.update({"build", "release"})

        for path in changed_files:
            suffix = Path(path).suffix.lower()
            if suffix in {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt"}:
                kinds.add("test")
            if suffix in {".ts", ".tsx"}:
                kinds.add("typecheck")
            if suffix in {".js", ".jsx", ".ts", ".tsx", ".css", ".html"}:
                kinds.add("build")
            if "/tests/" in f"/{path}" or Path(path).name.startswith("test_"):
                kinds.add("test")

        available_kinds = {command.kind for command in snapshot.validation_commands}
        if not kinds:
            for fallback_kind in ("test", "lint", "typecheck", "build"):
                if fallback_kind in available_kinds:
                    kinds.add(fallback_kind)
        return kinds

    def _default_commands(self, changed_files: list[str]) -> list[ValidationCommand]:
        unique_paths = self._unique_paths(changed_files)
        if not unique_paths:
            return []

        commands: list[ValidationCommand] = []
        python_files = [path for path in unique_paths if Path(path).suffix.lower() == ".py"]
        html_files = [path for path in unique_paths if Path(path).suffix.lower() == ".html"]
        js_files = [
            path for path in unique_paths if Path(path).suffix.lower() in {".js", ".jsx", ".mjs", ".cjs"}
        ]

        if python_files:
            commands.append(
                ValidationCommand(
                    command=f"internal:python_syntax:{json.dumps(python_files)}",
                    kind="check",
                    source="default",
                    priority=15,
                    reason="Syntax-check the changed Python starter artifacts.",
                )
            )

        if html_files:
            commands.append(
                ValidationCommand(
                    command=f"internal:html_refs:{json.dumps(html_files)}",
                    kind="check",
                    source="default",
                    priority=20,
                    reason="Check that local HTML asset references resolve.",
                )
            )

        if js_files and shutil.which("node"):
            commands.append(
                ValidationCommand(
                    command=f"node --check {' '.join(shlex.quote(path) for path in js_files)}",
                    kind="check",
                    source="default",
                    priority=25,
                    reason="Syntax-check the changed JavaScript starter artifacts with Node.",
                )
            )

        return commands

    def _unique_paths(self, values: list[str]) -> list[str]:
        unique: list[str] = []
        for raw in values:
            text = str(raw or "").strip()
            if not text or text in unique:
                continue
            unique.append(text)
        return unique
