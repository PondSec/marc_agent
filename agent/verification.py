from __future__ import annotations

import json
import re
import shlex
import shutil
from pathlib import Path

from agent.models import (
    DiagnosticRecord,
    SessionState,
    ValidationCommand,
    ValidationFailureEvidence,
    ValidationRunRecord,
    WorkspaceSnapshot,
)


MUTATION_TOOL_NAMES = {
    "write_file",
    "append_file",
    "create_file",
    "delete_file",
    "replace_in_file",
    "patch_file",
}


class ValidationPlanner:
    TRACEBACK_FRAME_PATTERN = re.compile(
        r'File "(?P<path>[^"]+)", line (?P<line>\d+)(?:, in (?P<symbol>[^\n]+))?'
    )
    WORKSPACE_REFERENCE_PATTERN = re.compile(
        r"(?P<quote>['\"])(?P<path>(?:[\w.-]+/)+[\w.-]+(?:\.[A-Za-z0-9_-]+)?)(?P=quote)"
    )
    EXPLICIT_VALIDATION_COMMAND_SPECS = (
        {
            "kind": "test",
            "verification_scope": "runtime",
            "priority": 5,
            "pattern": re.compile(r"(?P<command>python(?:3)?\s+-m\s+pytest\b[^`\n]*)", re.IGNORECASE),
        },
        {
            "kind": "test",
            "verification_scope": "runtime",
            "priority": 5,
            "pattern": re.compile(r"(?P<command>python(?:3)?\s+-m\s+unittest\b[^`\n]*)", re.IGNORECASE),
        },
        {
            "kind": "test",
            "verification_scope": "runtime",
            "priority": 6,
            "pattern": re.compile(r"(?P<command>pytest\b[^`\n]*)", re.IGNORECASE),
        },
        {
            "kind": "lint",
            "verification_scope": "static",
            "priority": 8,
            "pattern": re.compile(r"(?P<command>ruff(?:\s+check)?\b[^`\n]*)", re.IGNORECASE),
        },
        {
            "kind": "typecheck",
            "verification_scope": "static",
            "priority": 10,
            "pattern": re.compile(r"(?P<command>mypy\b[^`\n]*)", re.IGNORECASE),
        },
        {
            "kind": "typecheck",
            "verification_scope": "static",
            "priority": 10,
            "pattern": re.compile(r"(?P<command>pyright\b[^`\n]*)", re.IGNORECASE),
        },
        {
            "kind": "check",
            "verification_scope": "static",
            "priority": 12,
            "pattern": re.compile(r"(?P<command>node\s+--check\b[^`\n]*)", re.IGNORECASE),
        },
        {
            "kind": "test",
            "verification_scope": "runtime",
            "priority": 7,
            "pattern": re.compile(
                r"(?P<command>(?:npm|pnpm|yarn)\s+(?:test|run\s+(?:test|lint|build|typecheck))\b[^`\n]*)",
                re.IGNORECASE,
            ),
        },
        {
            "kind": "test",
            "verification_scope": "runtime",
            "priority": 7,
            "pattern": re.compile(r"(?P<command>go\s+test\b[^`\n]*)", re.IGNORECASE),
        },
        {
            "kind": "test",
            "verification_scope": "runtime",
            "priority": 7,
            "pattern": re.compile(r"(?P<command>cargo\s+(?:test|check)\b[^`\n]*)", re.IGNORECASE),
        },
        {
            "kind": "test",
            "verification_scope": "runtime",
            "priority": 9,
            "pattern": re.compile(r"(?P<command>make\s+(?:test|check|lint|build)\b[^`\n]*)", re.IGNORECASE),
        },
    )
    EXPLICIT_COMMAND_TRAILING_STOPWORDS = re.compile(
        r"(?i)\s+(?:"
        r"then|after|afterward|until|finish|complete|continue|summarize|report|"
        r"and|danach|danachmals|danachhin|"
        r"und|bis|aus|ausfuehren|ausführen|abschliessen|abschließen"
        r")\b.*$"
    )
    EXPLICIT_COMMAND_SENTENCE_BOUNDARY = re.compile(r"(?<=[\w)\]'\"])[.!?]\s+(?=[A-ZÄÖÜ])")
    WEB_FEATURE_KEYWORDS = {
        "menu": ("menu", "menue", "menü", "navigation", "nav"),
        "highscore": ("highscore", "high score", "scoreboard", "leaderboard", "best score", "bestenliste"),
        "score": ("score", "punkte", "punktestand", "scoreboard"),
        "start_controls": ("start", "play", "pause", "resume", "restart", "reset"),
        "dialog": ("dialog", "modal", "popup", "overlay"),
        "canvas": ("canvas", "spielfeld", "game board"),
        "settings": ("settings", "options", "config", "einstellungen"),
    }
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
    MISSING_WEB_FEATURES_PATTERN = re.compile(
        r"missing expected web features \((?P<features>[^)]+)\)",
        re.IGNORECASE,
    )
    MISSING_REF_PATTERN = re.compile(r"(?P<path>[\w./\\-]+\.(?:html|htm))\s*->\s*(?P<ref>[^\s]+)")
    NO_TESTS_RAN_PATTERNS = (
        re.compile(r"\bran\s+0\s+tests?\b", re.IGNORECASE),
        re.compile(r"\bno\s+tests\s+ran\b", re.IGNORECASE),
        re.compile(r"\bcollected\s+0\s+items\b", re.IGNORECASE),
        re.compile(r"start directory is not importable", re.IGNORECASE),
    )

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
        commands = self._merge_explicit_validation_commands(commands, session=session)
        commands = self._replace_generic_unittest_commands(
            commands,
            snapshot=snapshot,
            changed_files=changed_paths,
        )
        if changed_paths and not any(self.command_scope(command) == "runtime" for command in commands):
            commands.extend(self._default_commands(snapshot, changed_paths, task=task))
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
            self.command_identity(run.command)
            for run in session.validation_runs
            if run.edit_generation == session.edit_generation and run.status == "passed"
        }
        return [
            command
            for command in required_commands
            if self.command_identity(command.command) not in passed
            and not self._generic_unittest_satisfied_by_targeted_success(
                session,
                command.command,
                current_generation_only=True,
            )
        ]

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
            if command.verification_scope in {"syntax", "structural", "semantic", "runtime"}:
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
        if lowered.startswith("internal:web_artifact:"):
            return "structural"
        if lowered.startswith("internal:semantic_review:"):
            return "semantic"
        if lowered.startswith("internal:html_refs:"):
            return "static"
        if normalized_kind == "test":
            return "runtime"
        if any(
            token in lowered
            for token in (
                "pytest",
                "python -m unittest",
                "python3 -m unittest",
                "python -m pytest",
                "python3 -m pytest",
                "go test",
                "cargo test",
            )
        ):
            return "runtime"
        if any(
            token in lowered
            for token in ("ruff", "eslint", "flake8", "mypy", "pyright", "node --check", "html_refs")
        ):
            return "static"
        return "static"

    def _merge_explicit_validation_commands(
        self,
        commands: list[ValidationCommand],
        *,
        session: SessionState | None,
    ) -> list[ValidationCommand]:
        explicit = self._explicit_validation_commands(session)
        if not explicit:
            return commands

        merged: list[ValidationCommand] = []
        seen: set[str] = set()
        for item in [*explicit, *commands]:
            identity = self.command_identity(item.command)
            if not identity or identity in seen:
                continue
            seen.add(identity)
            merged.append(item)
        return merged

    def _explicit_validation_commands(
        self,
        session: SessionState | None,
    ) -> list[ValidationCommand]:
        if session is None:
            return []

        texts: list[tuple[str, str]] = []
        task_state = session.task_state
        if task_state is not None:
            if task_state.verification_target:
                texts.append(("task_state", task_state.verification_target))
            if task_state.latest_user_turn:
                texts.append(("user_request", task_state.latest_user_turn))
        elif session.task:
            texts.append(("user_request", session.task))

        extracted: list[ValidationCommand] = []
        for source, text in texts:
            raw_text = str(text or "")
            if not raw_text.strip():
                continue
            for spec in self.EXPLICIT_VALIDATION_COMMAND_SPECS:
                for match in spec["pattern"].finditer(raw_text):
                    command = self._normalize_explicit_validation_command(match.group("command"))
                    if not command:
                        continue
                    extracted.append(
                        ValidationCommand(
                            command=command,
                            kind=str(spec["kind"]),
                            verification_scope=str(spec["verification_scope"]),
                            source=source,
                            priority=int(spec["priority"]),
                            reason="Explicit validation command requested in the active task.",
                            required=True,
                        )
                    )
        unique: list[ValidationCommand] = []
        seen: set[str] = set()
        for item in extracted:
            identity = self.command_identity(item.command)
            if not identity or identity in seen:
                continue
            seen.add(identity)
            unique.append(item)
        return unique

    def _normalize_explicit_validation_command(self, raw: str) -> str | None:
        command = " ".join(str(raw or "").strip().split())
        if not command:
            return None
        command = command.strip("`'\"")
        sentence_split = self.EXPLICIT_COMMAND_SENTENCE_BOUNDARY.split(command, maxsplit=1)
        if sentence_split:
            command = sentence_split[0].strip()
        command = self.EXPLICIT_COMMAND_TRAILING_STOPWORDS.sub("", command).strip()
        command = self._trim_python_test_command_tokens(command)
        command = command.rstrip(".,;:!?")
        if not command:
            return None
        lowered = command.lower()
        for spec in self.EXPLICIT_VALIDATION_COMMAND_SPECS:
            if spec["pattern"].fullmatch(command):
                return command
        if any(
            lowered.startswith(prefix)
            for prefix in (
                "python -m pytest",
                "python3 -m pytest",
                "python -m unittest",
                "python3 -m unittest",
                "pytest",
                "ruff",
                "mypy",
                "pyright",
                "node --check",
                "npm test",
                "npm run test",
                "npm run lint",
                "npm run build",
                "npm run typecheck",
                "pnpm test",
                "pnpm lint",
                "pnpm build",
                "pnpm typecheck",
                "yarn test",
                "yarn lint",
                "yarn build",
                "yarn typecheck",
                "go test",
                "cargo test",
                "cargo check",
                "make test",
                "make check",
                "make lint",
                "make build",
            )
        ):
            return command
        return None

    def _trim_python_test_command_tokens(self, command: str) -> str:
        try:
            tokens = shlex.split(str(command or "").strip())
        except ValueError:
            return command
        if not tokens:
            return command

        lowered = [token.lower() for token in tokens]
        if lowered[:3] in (["python", "-m", "unittest"], ["python3", "-m", "unittest"]):
            trimmed = self._trim_unittest_command_tokens(tokens)
            return " ".join(trimmed) if trimmed else command
        if lowered[:3] in (["python", "-m", "pytest"], ["python3", "-m", "pytest"]):
            trimmed = self._trim_pytest_command_tokens(tokens, prefix_len=3)
            return " ".join(trimmed) if trimmed else command
        if lowered[:1] == ["pytest"]:
            trimmed = self._trim_pytest_command_tokens(tokens, prefix_len=1)
            return " ".join(trimmed) if trimmed else command
        return command

    def _trim_unittest_command_tokens(self, tokens: list[str]) -> list[str]:
        if len(tokens) <= 3:
            return tokens

        option_value_flags = {"-k", "-s", "-p", "-t", "--start-directory", "--pattern", "--top-level-directory"}
        trimmed = tokens[:3]
        saw_target = False
        expect_option_value = False
        for token in tokens[3:]:
            lowered = token.lower()
            if expect_option_value:
                trimmed.append(token)
                expect_option_value = False
                continue
            if token.startswith("-"):
                trimmed.append(token)
                if lowered in option_value_flags:
                    expect_option_value = True
                continue
            if lowered == "discover" and not saw_target:
                trimmed.append(token)
                saw_target = True
                continue
            if self._looks_like_unittest_cli_target(token):
                trimmed.append(token)
                saw_target = True
                continue
            if saw_target:
                break
            break
        return trimmed

    def _trim_pytest_command_tokens(self, tokens: list[str], *, prefix_len: int) -> list[str]:
        if len(tokens) <= prefix_len:
            return tokens

        option_value_flags = {"-k", "-m", "-c", "--maxfail", "--rootdir"}
        trimmed = tokens[:prefix_len]
        saw_target = False
        expect_option_value = False
        for token in tokens[prefix_len:]:
            lowered = token.lower()
            if expect_option_value:
                trimmed.append(token)
                expect_option_value = False
                continue
            if token.startswith("-"):
                trimmed.append(token)
                if lowered in option_value_flags:
                    expect_option_value = True
                continue
            if self._looks_like_pytest_cli_target(token):
                trimmed.append(token)
                saw_target = True
                continue
            if saw_target:
                break
            break
        return trimmed

    def _looks_like_unittest_cli_target(self, token: str) -> bool:
        cleaned = str(token or "").strip()
        normalized = cleaned.rstrip(".,;:!?")
        if not normalized or normalized.startswith("-"):
            return False
        lowered = normalized.lower()
        if normalized.endswith(".py") or "/" in normalized or "\\" in normalized or "::" in normalized:
            return True
        if "." in normalized:
            parts = [part for part in normalized.split(".") if part]
            return bool(parts) and all(part.isidentifier() for part in parts)
        return lowered in {"discover", "tests", "test"} or lowered.startswith("test_") or lowered.endswith("_test")

    def _looks_like_pytest_cli_target(self, token: str) -> bool:
        cleaned = str(token or "").strip()
        normalized = cleaned.rstrip(".,;:!?")
        if not normalized or normalized.startswith("-"):
            return False
        lowered = normalized.lower()
        return (
            normalized.endswith(".py")
            or "/" in normalized
            or "\\" in normalized
            or "::" in normalized
            or lowered in {".", "tests", "test"}
        )

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

    def strongest_passed_scope(
        self,
        session: SessionState,
        *,
        current_generation_only: bool = True,
    ) -> str | None:
        ranking = {"syntax": 1, "static": 2, "structural": 3, "semantic": 4, "runtime": 5}
        strongest: str | None = None
        for run in self._runtime_runs(session, current_generation_only=current_generation_only):
            if run.status != "passed":
                continue
            if strongest is None or ranking.get(run.verification_scope, 0) > ranking.get(strongest, 0):
                strongest = run.verification_scope
        return strongest

    def web_functional_verification_required(self, session: SessionState) -> bool:
        changed_paths = [item.path for item in session.changed_files]
        html_files = [path for path in changed_paths if Path(path).suffix.lower() == ".html"]
        if not html_files:
            return False
        if len(changed_paths) > 4:
            return False
        if session.workspace_snapshot is not None and session.workspace_snapshot.file_count > 40:
            runtime_commands = [
                command
                for command in session.validation_plan
                if command.verification_scope == "runtime"
            ]
            if runtime_commands:
                return True
            return False
        return True

    def has_structural_web_success(
        self,
        session: SessionState,
        *,
        current_generation_only: bool = True,
    ) -> bool:
        return any(
            run.status == "passed" and run.verification_scope == "structural"
            for run in self._runtime_runs(session, current_generation_only=current_generation_only)
        )

    def has_semantic_review(
        self,
        session: SessionState,
        *,
        current_generation_only: bool = True,
    ) -> bool:
        return any(
            run.verification_scope == "semantic"
            for run in self._runtime_runs(session, current_generation_only=current_generation_only)
        )

    def has_semantic_review_success(
        self,
        session: SessionState,
        *,
        current_generation_only: bool = True,
    ) -> bool:
        return any(
            run.status == "passed" and run.verification_scope == "semantic"
            for run in self._runtime_runs(session, current_generation_only=current_generation_only)
        )

    def semantic_review_command(self, changed_files: list[str]) -> str:
        payload = [{"path": path} for path in self._unique_paths(changed_files)[:8]]
        return f"internal:semantic_review:{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"

    def latest_failed_run(
        self,
        session: SessionState,
        *,
        current_generation_only: bool = True,
        command: str | None = None,
    ) -> ValidationRunRecord | None:
        normalized_command = str(command or "").strip()
        for run in reversed(self._runtime_runs(session, current_generation_only=current_generation_only)):
            if run.status not in {"failed", "timeout", "blocked"}:
                continue
            if normalized_command and self.command_identity(run.command) != self.command_identity(normalized_command):
                continue
            return run
        return None

    def build_failure_evidence(
        self,
        session: SessionState,
        failed_run: ValidationRunRecord,
    ) -> ValidationFailureEvidence:
        traceback_file_hints, traceback_line_hints = self._traceback_workspace_hints(
            session,
            failed_run,
        )
        referenced_workspace_paths = self._referenced_workspace_paths(
            session,
            failed_run,
        )
        artifact_paths = self._unique_paths(
            [
                *traceback_file_hints,
                *self._artifact_paths_for_failed_run(session, failed_run),
                *referenced_workspace_paths,
            ]
        )
        artifact_paths = self._prioritize_runtime_artifact_paths(
            artifact_paths,
            failed_run,
        )
        diagnostics = self._related_diagnostics(session, failed_run, artifact_paths)
        missing_unittest_package_inits = self._missing_unittest_package_inits(
            session,
            failed_run,
            artifact_paths,
        )
        expected_features = self._expected_features_from_command(failed_run.command)
        missing_features = self._missing_features_from_failure(failed_run, diagnostics)
        file_hints = self._unique_paths(
            [
                *artifact_paths,
                *traceback_file_hints,
                *referenced_workspace_paths,
                *missing_unittest_package_inits,
                *(path for diagnostic in diagnostics for path in diagnostic.file_hints),
            ]
        )
        line_hints = self._unique_line_hints(
            [
                *traceback_line_hints,
                *(line for diagnostic in diagnostics for line in diagnostic.line_hints),
            ]
        )
        action_hints = self._unique_strings(
            [hint for diagnostic in diagnostics for hint in diagnostic.action_hints]
        )
        summary = str(failed_run.summary or "").strip() or "Validation failed."
        excerpt = self._failure_excerpt(failed_run, diagnostics)
        failure_summary = self._failure_summary(
            failed_run,
            diagnostics,
            artifact_paths=artifact_paths,
            missing_features=missing_features,
        )
        repair_requirements = self._repair_requirements(
            failed_run,
            diagnostics,
            artifact_paths=artifact_paths,
            expected_features=expected_features,
            missing_features=missing_features,
            missing_unittest_package_inits=missing_unittest_package_inits,
        )
        evidence_signature = json.dumps(
            {
                "command": self.command_identity(failed_run.command),
                "verification_scope": failed_run.verification_scope,
                "status": failed_run.status,
                "artifact_paths": artifact_paths,
                "summary": summary,
                "excerpt": excerpt,
                "expected_features": expected_features,
                "missing_features": missing_features,
                "file_hints": file_hints,
                "line_hints": line_hints,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return ValidationFailureEvidence(
            command=failed_run.command,
            verification_scope=failed_run.verification_scope,
            status=failed_run.status,
            artifact_paths=artifact_paths,
            summary=summary,
            excerpt=excerpt,
            failure_summary=failure_summary,
            expected_features=expected_features,
            missing_features=missing_features,
            file_hints=file_hints,
            line_hints=line_hints,
            action_hints=action_hints,
            repair_requirements=repair_requirements,
            evidence_signature=evidence_signature,
        )

    def _referenced_workspace_paths(
        self,
        session: SessionState,
        failed_run: ValidationRunRecord,
    ) -> list[str]:
        workspace_root = Path(session.workspace_root).resolve()
        referenced_paths: list[str] = []
        texts = [
            str(failed_run.excerpt or "").strip(),
            str(failed_run.summary or "").strip(),
        ]
        for text in texts:
            if not text:
                continue
            for match in self.WORKSPACE_REFERENCE_PATTERN.finditer(text):
                raw_path = str(match.group("path") or "").strip().replace("\\", "/")
                if not raw_path:
                    continue
                normalized = raw_path.removeprefix("./")
                if normalized.startswith("/") or normalized.startswith("../") or "/../" in f"/{normalized}":
                    continue
                candidate = (workspace_root / normalized).resolve()
                try:
                    relative = candidate.relative_to(workspace_root).as_posix()
                except ValueError:
                    continue
                if relative and relative not in referenced_paths:
                    referenced_paths.append(relative)
        return referenced_paths

    def _traceback_workspace_hints(
        self,
        session: SessionState,
        failed_run: ValidationRunRecord,
    ) -> tuple[list[str], list[int]]:
        workspace_root = Path(session.workspace_root).resolve()
        paths: list[str] = []
        lines: list[int] = []
        texts = [
            str(failed_run.excerpt or "").strip(),
            str(failed_run.summary or "").strip(),
        ]
        for text in texts:
            if not text:
                continue
            for match in self.TRACEBACK_FRAME_PATTERN.finditer(text):
                raw_path = str(match.group("path") or "").strip()
                if not raw_path:
                    continue
                candidate = Path(raw_path).resolve()
                try:
                    relative = candidate.relative_to(workspace_root).as_posix()
                except ValueError:
                    continue
                if relative and relative not in paths:
                    paths.append(relative)
                try:
                    line = int(match.group("line") or 0)
                except ValueError:
                    line = 0
                if line > 0 and line not in lines:
                    lines.append(line)

        implementation_paths = [path for path in paths if not self._is_test_path(path)]
        test_paths = [path for path in paths if self._is_test_path(path)]
        return [*implementation_paths, *test_paths], lines

    def can_repeat_command(
        self,
        session: SessionState,
        command: str,
        *,
        current_generation_only: bool = True,
    ) -> bool:
        normalized = self.command_identity(command)
        if not normalized:
            return False
        if self._generic_unittest_satisfied_by_targeted_success(
            session,
            normalized,
            current_generation_only=current_generation_only,
        ):
            return False
        failed_run = self.latest_failed_run(
            session,
            current_generation_only=current_generation_only,
            command=normalized,
        )
        if failed_run is None:
            return True
        if failed_run.edit_generation != session.edit_generation:
            return True

        failed_iteration = int(failed_run.iteration or 0)
        for item in session.tool_calls:
            iteration = int(getattr(item, "iteration", 0) or 0)
            if iteration <= failed_iteration:
                continue
            if item.tool_name in MUTATION_TOOL_NAMES:
                return True
        return False

    def _generic_unittest_satisfied_by_targeted_success(
        self,
        session: SessionState,
        command: str,
        *,
        current_generation_only: bool,
    ) -> bool:
        normalized = self.command_identity(command).lower()
        if normalized not in {"python -m unittest", "python3 -m unittest"}:
            return False

        for run in session.validation_runs:
            if current_generation_only and run.edit_generation != session.edit_generation:
                continue
            if str(run.status or "").strip() != "passed":
                continue
            run_command = self.command_identity(run.command).lower()
            if not (
                run_command.startswith("python -m unittest ")
                or run_command.startswith("python3 -m unittest ")
            ):
                continue
            target_tokens = run_command.split()[3:]
            if not target_tokens:
                continue
            if target_tokens[0] == "discover":
                continue
            return True
        return False

    def command_identity(self, command: ValidationCommand | str) -> str:
        text = command.command if isinstance(command, ValidationCommand) else command
        normalized = " ".join(str(text or "").strip().split())
        if not normalized.startswith("internal:"):
            return normalized

        prefix, _, payload = normalized.partition(":")
        kind, _, raw_json = payload.partition(":")
        if not kind:
            return normalized
        try:
            decoded = json.loads(raw_json or "[]")
        except json.JSONDecodeError:
            return normalized
        compact = json.dumps(decoded, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return f"{prefix}:{kind}:{compact}"

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
            if suffix in {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt", ".html"}:
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
        *,
        task: str,
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
            html_targets = [
                {"path": path, "expected_features": self._expected_web_features(task)}
                for path in html_files
            ]
            commands.append(
                ValidationCommand(
                    command=f"internal:web_artifact:{json.dumps(html_targets)}",
                    kind="check",
                    verification_scope="structural",
                    source="default",
                    priority=18,
                    reason="Run structural HTML/JavaScript smoke checks for the changed web artifact.",
                )
            )
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
            if command.source == "default" and command.verification_scope in {"syntax", "static", "structural"}
            else command
            for command in commands
        ]

    def _replace_generic_unittest_commands(
        self,
        commands: list[ValidationCommand],
        *,
        snapshot: WorkspaceSnapshot,
        changed_files: list[str],
    ) -> list[ValidationCommand]:
        generic_indexes: list[int] = []
        command_prefix = "python"
        for index, command in enumerate(commands):
            normalized = self.command_identity(command.command).lower()
            if normalized not in {"python -m unittest", "python3 -m unittest"}:
                continue
            if command.source in {"task_state", "user_request"}:
                continue
            generic_indexes.append(index)
            if normalized.startswith("python3"):
                command_prefix = "python3"

        if not generic_indexes:
            return commands

        modules = self._targeted_unittest_modules(
            changed_files=changed_files,
            snapshot=snapshot,
        )
        if not modules:
            return commands

        template = commands[generic_indexes[0]]
        replacement = template.model_copy(
            update={
                "command": f"{command_prefix} -m unittest {' '.join(modules)}",
                "source": "unittest-targeted",
                "priority": min(int(template.priority or 10), 9),
                "reason": "Run the changed unittest modules directly to avoid zero-test discovery runs.",
            }
        )
        return [replacement, *(item for index, item in enumerate(commands) if index not in generic_indexes)]

    def _targeted_unittest_modules(
        self,
        *,
        changed_files: list[str],
        snapshot: WorkspaceSnapshot,
    ) -> list[str]:
        candidate_paths = [
            path
            for path in changed_files
            if self._is_test_path(path) and Path(path).suffix.lower() == ".py"
        ]
        if not candidate_paths:
            candidate_paths = [
                path
                for path in snapshot.test_files
                if Path(path).suffix.lower() == ".py"
            ]

        modules: list[str] = []
        for path in self._unique_paths(candidate_paths):
            module = self._python_module_from_path(path)
            if module:
                modules.append(module)
        return modules[:6]

    def _python_module_from_path(self, path: str) -> str | None:
        candidate = str(path or "").strip()
        if not candidate.endswith(".py"):
            return None
        stem = candidate[:-3]
        parts = [part for part in stem.split("/") if part]
        if not parts or not all(part.isidentifier() for part in parts):
            return None
        return ".".join(parts)

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

    def _expected_web_features(self, task: str) -> list[str]:
        token_space = self._normalized_keyword_space(task)
        expected: list[str] = []
        for feature, markers in self.WEB_FEATURE_KEYWORDS.items():
            if any(self._normalized_keyword_space(marker) in token_space for marker in markers):
                expected.append(feature)
        return expected

    def _normalized_keyword_space(self, value: str) -> str:
        lowered = str(value or "").lower()
        collapsed = re.sub(r"[^0-9a-zäöüß]+", " ", lowered)
        return f" {collapsed.strip()} "

    def _related_diagnostics(
        self,
        session: SessionState,
        failed_run: ValidationRunRecord,
        artifact_paths: list[str],
    ) -> list[DiagnosticRecord]:
        related: list[DiagnosticRecord] = []
        normalized_command = self.command_identity(failed_run.command)
        artifact_set = set(artifact_paths)
        for diagnostic in reversed(session.diagnostics):
            diagnostic_command = self.command_identity(diagnostic.command or "")
            if diagnostic_command and diagnostic_command == normalized_command:
                related.append(diagnostic)
                continue
            if artifact_set and artifact_set.intersection(diagnostic.file_hints):
                related.append(diagnostic)
        return list(reversed(related[-4:]))

    def _artifact_paths_for_failed_run(
        self,
        session: SessionState,
        failed_run: ValidationRunRecord,
    ) -> list[str]:
        paths: list[str] = []
        task_state = session.task_state
        no_tests_executed = self._no_tests_executed(failed_run)
        if failed_run.verification_scope == "runtime" and task_state is not None and not no_tests_executed:
            paths.extend(
                artifact.path
                for artifact in task_state.target_artifacts
                if artifact.path and artifact.role != "supporting_context"
            )
        if no_tests_executed:
            if task_state is not None:
                paths.extend(
                    artifact.path
                    for artifact in task_state.target_artifacts
                    if artifact.path
                    and (
                        artifact.role == "validation_target"
                        or artifact.kind == "test"
                    )
                )
            paths.extend(
                item.path
                for item in session.changed_files
                if self._is_test_path(item.path)
            )
            paths.extend(self._missing_unittest_package_inits(session, failed_run, paths))
        paths.extend(self._paths_from_validation_command(failed_run.command))
        if session.active_repair_context is not None:
            paths.extend(session.active_repair_context.artifact_paths)
        paths.extend(item.path for item in session.changed_files)
        return self._unique_paths(paths)

    def _prioritize_runtime_artifact_paths(
        self,
        artifact_paths: list[str],
        failed_run: ValidationRunRecord,
    ) -> list[str]:
        if failed_run.verification_scope != "runtime" or self._no_tests_executed(failed_run):
            return artifact_paths
        implementation_paths = [
            path
            for path in artifact_paths
            if path and not self._is_test_path(path)
        ]
        validation_paths = [
            path
            for path in artifact_paths
            if path and self._is_test_path(path)
        ]
        ordered = [*implementation_paths, *validation_paths]
        return ordered or artifact_paths

    def _paths_from_validation_command(self, command: str) -> list[str]:
        kind, payload = self._decoded_internal_validation_payload(command)
        if kind is None:
            return self._paths_from_explicit_test_command(command)
        values = payload if isinstance(payload, list) else [payload]
        paths: list[str] = []
        for item in values:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    paths.append(cleaned)
            elif isinstance(item, dict):
                cleaned = str(item.get("path") or "").strip()
                if cleaned:
                    paths.append(cleaned)
        return self._unique_paths(paths)

    def _paths_from_explicit_test_command(self, command: str) -> list[str]:
        try:
            tokens = shlex.split(str(command or "").strip())
        except ValueError:
            return []
        if not tokens:
            return []
        lowered = [token.lower() for token in tokens]
        if len(tokens) >= 3 and lowered[:3] in (["python", "-m", "unittest"], ["python3", "-m", "unittest"]):
            return self._unittest_targets_to_paths(tokens[3:])
        if len(tokens) >= 3 and lowered[:3] in (["python", "-m", "pytest"], ["python3", "-m", "pytest"]):
            return self._pytest_targets_to_paths(tokens[3:])
        if lowered[0] == "pytest":
            return self._pytest_targets_to_paths(tokens[1:])
        return []

    def _unittest_targets_to_paths(self, targets: list[str]) -> list[str]:
        paths: list[str] = []
        for target in targets:
            cleaned = str(target or "").strip()
            if not cleaned or cleaned.startswith("-"):
                continue
            path = self._path_from_unittest_target(cleaned)
            if path:
                paths.append(path)
        return self._unique_paths(paths)

    def _pytest_targets_to_paths(self, targets: list[str]) -> list[str]:
        paths: list[str] = []
        for target in targets:
            cleaned = str(target or "").strip()
            if not cleaned or cleaned.startswith("-"):
                continue
            path = cleaned.split("::", 1)[0].strip()
            if not path or path.endswith(".py") is False:
                continue
            paths.append(path)
        return self._unique_paths(paths)

    def _path_from_unittest_target(self, target: str) -> str | None:
        cleaned = str(target or "").strip()
        if not cleaned:
            return None
        cleaned = cleaned.split("::", 1)[0].strip()
        if cleaned.endswith(".py"):
            return cleaned
        if "/" in cleaned:
            return f"{cleaned}.py" if "." not in Path(cleaned).suffix else cleaned

        parts = [part for part in cleaned.split(".") if part]
        if not parts:
            return None
        if len(parts) == 1:
            lowered = cleaned.lower()
            if lowered not in {"tests", "test"} and not lowered.startswith("test_") and not lowered.endswith("_test"):
                return None

        module_parts: list[str] = []
        for part in parts:
            if part[:1].isupper():
                break
            module_parts.append(part)
            if len(module_parts) >= 2 and module_parts[-1].startswith("test_"):
                break
        if not module_parts:
            return None
        return "/".join(module_parts) + ".py"

    def _expected_features_from_command(self, command: str) -> list[str]:
        kind, payload = self._decoded_internal_validation_payload(command)
        if kind != "web_artifact":
            return []
        values = payload if isinstance(payload, list) else [payload]
        features: list[str] = []
        for item in values:
            if not isinstance(item, dict):
                continue
            features.extend(
                str(feature or "").strip()
                for feature in item.get("expected_features", [])
                if str(feature or "").strip()
            )
        return self._unique_strings(features)

    def _missing_features_from_failure(
        self,
        failed_run: ValidationRunRecord,
        diagnostics: list[DiagnosticRecord],
    ) -> list[str]:
        texts = [
            str(failed_run.summary or "").strip(),
            str(failed_run.excerpt or "").strip(),
        ]
        texts.extend(str(item.summary or "").strip() for item in diagnostics)
        texts.extend(str(item.excerpt or "").strip() for item in diagnostics)
        features: list[str] = []
        for text in texts:
            if not text:
                continue
            for match in self.MISSING_WEB_FEATURES_PATTERN.finditer(text):
                raw_features = match.group("features") or ""
                features.extend(
                    part.strip()
                    for part in raw_features.split(",")
                    if part.strip()
                )
        return self._unique_strings(features)

    def _failure_excerpt(
        self,
        failed_run: ValidationRunRecord,
        diagnostics: list[DiagnosticRecord],
    ) -> str | None:
        for text in [
            str(failed_run.excerpt or "").strip(),
            *(str(item.excerpt or "").strip() for item in diagnostics),
            *(str(item.summary or "").strip() for item in diagnostics),
        ]:
            if text:
                return text
        return None

    def _failure_summary(
        self,
        failed_run: ValidationRunRecord,
        diagnostics: list[DiagnosticRecord],
        *,
        artifact_paths: list[str],
        missing_features: list[str],
    ) -> str:
        if missing_features:
            feature_text = ", ".join(missing_features)
            if artifact_paths:
                return f"{artifact_paths[0]} is missing validation-required features: {feature_text}."
            return f"Validation-required features are missing: {feature_text}."
        excerpt = self._failure_excerpt(failed_run, diagnostics)
        if excerpt:
            return excerpt[:240]
        summary = str(failed_run.summary or "").strip()
        if summary:
            return summary[:240]
        return "Validation failed without a readable summary."

    def _repair_requirements(
        self,
        failed_run: ValidationRunRecord,
        diagnostics: list[DiagnosticRecord],
        *,
        artifact_paths: list[str],
        expected_features: list[str],
        missing_features: list[str],
        missing_unittest_package_inits: list[str] | None = None,
    ) -> list[str]:
        requirements: list[str] = []
        scope = failed_run.verification_scope
        primary_target = artifact_paths[0] if artifact_paths else "the implicated artifact"
        missing_unittest_package_inits = missing_unittest_package_inits or []
        if scope == "structural":
            if missing_features:
                requirements.append(
                    f"Add or restore the structural features explicitly reported as missing in {primary_target}: {', '.join(missing_features)}."
                )
            elif expected_features:
                requirements.append(
                    f"Ensure {primary_target} now includes the validation-targeted features: {', '.join(expected_features)}."
                )
            if self._failure_mentions_missing_refs(failed_run, diagnostics):
                requirements.append(
                    f"Fix broken local references or missing assets reported by the structural validation for {primary_target}."
                )
            requirements.append(
                f"Change {primary_target} in a way that directly addresses the failed structural check, not just formatting."
            )
        elif scope == "syntax":
            requirements.append(
                f"Fix the syntax error or parse failure reported for {primary_target} before rerunning validation."
            )
        elif scope == "runtime":
            if self._no_tests_executed(failed_run, diagnostics):
                requirements.append(
                    "Ensure the requested test command actually discovers and executes the intended tests; repair test discovery, package layout, or validation wiring before changing unrelated production code."
                )
                for init_path in missing_unittest_package_inits[:2]:
                    requirements.append(
                        f"Consider adding or restoring {init_path} so the unittest discovery path can import the affected test package."
                    )
                requirements.append(
                    f"Update {primary_target} or adjacent test-discovery files only if that is what will make the requested test command execute real tests."
                )
            else:
                requirements.append(
                    f"Change {primary_target} so the failing runtime or test path can complete successfully."
                )
        elif scope == "semantic":
            requirements.append(
                f"Close the remaining task-to-code gaps reported by the semantic review for {primary_target}."
            )
            requirements.append(
                f"Repair any dangling or inconsistent references implicated by the semantic review before claiming the task is complete."
            )
        else:
            requirements.append(
                f"Address the static validation failure in {primary_target} using the reported file, line, or reference hints."
            )
        requirements.extend(
            hint
            for hint in self._unique_strings(
                [hint for diagnostic in diagnostics for hint in diagnostic.action_hints]
            )[:2]
        )
        requirements.append("Do not stop at an equivalent or formatting-only rewrite.")
        return self._unique_strings(requirements)

    def _failure_mentions_missing_refs(
        self,
        failed_run: ValidationRunRecord,
        diagnostics: list[DiagnosticRecord],
    ) -> bool:
        texts = [
            str(failed_run.excerpt or "").strip(),
            str(failed_run.summary or "").strip(),
        ]
        texts.extend(str(item.excerpt or "").strip() for item in diagnostics)
        texts.extend(str(item.summary or "").strip() for item in diagnostics)
        return any(self.MISSING_REF_PATTERN.search(text) for text in texts if text)

    def _no_tests_executed(
        self,
        failed_run: ValidationRunRecord,
        diagnostics: list[DiagnosticRecord] | None = None,
    ) -> bool:
        texts = [
            str(failed_run.summary or "").strip(),
            str(failed_run.excerpt or "").strip(),
        ]
        if diagnostics:
            texts.extend(str(item.summary or "").strip() for item in diagnostics)
            texts.extend(str(item.excerpt or "").strip() for item in diagnostics)
        return any(
            pattern.search(text)
            for text in texts
            if text
            for pattern in self.NO_TESTS_RAN_PATTERNS
        )

    def _missing_unittest_package_inits(
        self,
        session: SessionState,
        failed_run: ValidationRunRecord,
        artifact_paths: list[str],
    ) -> list[str]:
        lowered_command = str(failed_run.command or "").lower()
        if "python -m unittest" not in lowered_command and "python3 -m unittest" not in lowered_command:
            return []

        workspace_root = Path(session.workspace_root)
        candidates: list[str] = []
        for artifact_path in artifact_paths:
            if not self._is_test_path(artifact_path):
                continue
            parent = Path(artifact_path).parent
            if str(parent) in {"", "."}:
                continue
            init_relative = (parent / "__init__.py").as_posix()
            if not (workspace_root / init_relative).exists():
                candidates.append(init_relative)
        return self._unique_paths(candidates)

    def _decoded_internal_validation_payload(self, command: str) -> tuple[str | None, object]:
        normalized = str(command or "").strip()
        if not normalized.startswith("internal:"):
            return None, []
        prefix, _, payload = normalized.partition(":")
        kind, _, raw_json = payload.partition(":")
        if prefix != "internal" or not kind:
            return None, []
        try:
            return kind, json.loads(raw_json or "[]")
        except json.JSONDecodeError:
            return kind, []

    def _unique_strings(self, values: list[str]) -> list[str]:
        unique: list[str] = []
        for raw in values:
            text = str(raw or "").strip()
            if not text or text in unique:
                continue
            unique.append(text)
        return unique

    def _unique_line_hints(self, values: list[int]) -> list[int]:
        unique: list[int] = []
        for raw in values:
            try:
                line = int(raw)
            except (TypeError, ValueError):
                continue
            if line <= 0 or line in unique:
                continue
            unique.append(line)
        return unique
