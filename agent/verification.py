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
        session: SessionState | None = None,
    ) -> list[ValidationCommand]:
        changed_paths = self._unique_paths(changed_files or [])
        commands = [
            command.model_copy(
                update={
                    "verification_scope": self.command_scope(command),
                }
            )
            for command in snapshot.validation_commands
        ] or [
            ValidationCommand(
                command=item,
                kind="check",
                verification_scope=self.command_scope(item, kind="check"),
                source="fallback",
            )
            for item in snapshot.likely_commands
        ]
        if changed_paths and not any(self.command_scope(command) == "runtime" for command in commands):
            commands.extend(self._default_commands(snapshot, changed_paths))
        synthesized = self._synthesized_runtime_commands(
            snapshot,
            changed_paths,
            session=session,
        )
        if synthesized:
            commands.extend(synthesized)
        if not commands:
            return []

        relevant_kinds = self._relevant_kinds(task, changed_paths, snapshot)
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
        return self._prefer_runtime_over_static_fallbacks(selected)

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

    def build_diagnostic_plan(self, session: SessionState) -> list[ValidationCommand]:
        snapshot = session.workspace_snapshot
        if snapshot is None:
            return []
        candidates = self._diagnostic_candidate_paths(session)
        return self._python_smoke_commands(
            self._runnable_python_targets(candidates, snapshot=snapshot),
            reason="Run the active Python entry artifact with bounded sample input to reproduce the reported behavior.",
        )

    def command_scope(self, command: ValidationCommand | str, *, kind: str | None = None) -> str:
        if isinstance(command, ValidationCommand):
            if command.verification_scope in {"syntax", "runtime"}:
                return command.verification_scope
            kind = command.kind
            text = command.command
        else:
            text = command
        lowered = str(text or "").strip().lower()
        normalized_kind = str(kind or "").strip().lower()

        if lowered.startswith("internal:python_cli_smoke:"):
            return "runtime"
        if lowered.startswith("internal:python_syntax:"):
            return "syntax"
        if lowered.startswith("internal:html_refs:"):
            return "static"
        if normalized_kind == "test":
            return "runtime"
        if any(token in lowered for token in ("pytest", "python -m unittest", "go test", "cargo test")):
            return "runtime"
        if any(
            token in lowered
            for token in ("ruff", "eslint", "flake8", "mypy", "pyright", "node --check", "html_refs")
        ):
            return "static"
        return "static"

    def runtime_verification_required(self, session: SessionState) -> bool:
        task_state = session.task_state
        if task_state is None or task_state.execution_strategy != "debug_repair":
            return False
        if any(command.required and command.verification_scope == "runtime" for command in session.validation_plan):
            return True
        if any(command.verification_scope == "runtime" for command in session.validation_plan):
            return True
        return True

    def has_runtime_attempt(
        self,
        session: SessionState,
        *,
        current_generation_only: bool = True,
    ) -> bool:
        return any(
            run.verification_scope == "runtime"
            for run in self._runtime_runs(session, current_generation_only=current_generation_only)
        )

    def has_runtime_success(
        self,
        session: SessionState,
        *,
        current_generation_only: bool = True,
    ) -> bool:
        return any(
            run.status == "passed" and run.verification_scope == "runtime"
            for run in self._runtime_runs(session, current_generation_only=current_generation_only)
        )

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

    def _default_commands(
        self,
        snapshot: WorkspaceSnapshot,
        changed_files: list[str],
    ) -> list[ValidationCommand]:
        unique_paths = self._unique_paths(changed_files)
        if not unique_paths:
            return []

        commands: list[ValidationCommand] = []
        python_files = [path for path in unique_paths if Path(path).suffix.lower() == ".py"]
        html_files = [path for path in unique_paths if Path(path).suffix.lower() == ".html"]
        js_files = [
            path for path in unique_paths if Path(path).suffix.lower() in {".js", ".jsx", ".mjs", ".cjs"}
        ]

        commands.extend(
            self._python_smoke_commands(
                self._runnable_python_targets(unique_paths, snapshot=snapshot),
                reason="Run the changed Python entry script with bounded sample input to smoke-test runtime behavior.",
            )
        )

        if python_files:
            commands.append(
                ValidationCommand(
                    command=f"internal:python_syntax:{json.dumps(python_files)}",
                    kind="check",
                    verification_scope="syntax",
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
                    verification_scope="static",
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
                    verification_scope="static",
                    source="default",
                    priority=25,
                    reason="Syntax-check the changed JavaScript starter artifacts with Node.",
                )
            )

        return self._prefer_runtime_over_static_fallbacks(commands)

    def _synthesized_runtime_commands(
        self,
        snapshot: WorkspaceSnapshot,
        changed_files: list[str],
        *,
        session: SessionState | None,
    ) -> list[ValidationCommand]:
        if any(self.command_scope(command) == "runtime" for command in snapshot.validation_commands):
            return []
        if session is not None and self.runtime_verification_required(session):
            return self.build_diagnostic_plan(session)
        return []

    def _prefer_runtime_over_static_fallbacks(
        self,
        commands: list[ValidationCommand],
    ) -> list[ValidationCommand]:
        if not any(command.verification_scope == "runtime" for command in commands):
            return commands
        return [
            command.model_copy(update={"required": False})
            if command.source == "default" and command.verification_scope in {"syntax", "static"}
            else command
            for command in commands
        ]

    def _diagnostic_candidate_paths(self, session: SessionState) -> list[str]:
        task_state = session.task_state
        snapshot = session.workspace_snapshot
        follow_up = session.follow_up_context
        paths: list[str] = []
        if task_state is not None:
            paths.extend(
                artifact.path
                for artifact in task_state.target_artifacts
                if artifact.path
            )
        paths.extend(session.candidate_files[:8])
        if follow_up is not None:
            paths.extend(follow_up.target_paths[:8])
            paths.extend(follow_up.changed_files[:8])
            paths.extend(follow_up.read_files[:8])
        if snapshot is not None:
            paths.extend(snapshot.focus_files[:4])
            paths.extend(snapshot.entrypoints[:4])
        return self._unique_paths(paths)

    def _python_smoke_commands(
        self,
        targets: list[str],
        *,
        reason: str,
    ) -> list[ValidationCommand]:
        return [
            ValidationCommand(
                command=f"internal:python_cli_smoke:{json.dumps([path])}",
                kind="test",
                verification_scope="runtime",
                source="default",
                priority=12,
                reason=reason,
            )
            for path in targets[:2]
        ]

    def _runnable_python_targets(
        self,
        candidate_paths: list[str],
        *,
        snapshot: WorkspaceSnapshot | None,
    ) -> list[str]:
        unique = self._unique_paths(candidate_paths)
        if not unique:
            return []
        entrypoints = set(snapshot.entrypoints if snapshot is not None else [])
        single_candidate = len(unique) == 1
        small_python_workspace = bool(
            snapshot is not None
            and snapshot.file_count <= 8
            and snapshot.language_counts.get("python", 0) >= 1
            and not snapshot.test_files
        )
        targets: list[str] = []
        for path in unique:
            candidate = str(path or "").strip()
            if not candidate:
                continue
            file_path = Path(candidate)
            name = file_path.name.lower()
            if file_path.suffix.lower() != ".py":
                continue
            if self._is_test_path(candidate) or name == "__init__.py":
                continue
            if (
                candidate in entrypoints
                or name in {"main.py", "cli.py", "app.py", "server.py", "manage.py"}
                or "/" not in candidate
                or single_candidate
                or small_python_workspace
            ):
                targets.append(candidate)
        return self._unique_paths(targets)

    def _is_test_path(self, path: str) -> bool:
        lowered = str(path or "").lower()
        name = Path(lowered).name
        return "/tests/" in f"/{lowered}" or name.startswith("test_") or name.endswith("_test.py")

    def _runtime_runs(
        self,
        session: SessionState,
        *,
        current_generation_only: bool,
    ):
        runs = session.validation_runs
        if current_generation_only:
            runs = [
                run
                for run in runs
                if run.edit_generation == session.edit_generation
            ]
        return runs

    def _unique_paths(self, values: list[str]) -> list[str]:
        unique: list[str] = []
        for raw in values:
            text = str(raw or "").strip()
            if not text or text in unique:
                continue
            unique.append(text)
        return unique
