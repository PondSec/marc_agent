from __future__ import annotations

import ast
import hashlib
import json
import re
import shlex
import shutil
from pathlib import Path

from agent.models import (
    DiagnosticRecord,
    RepairAttemptSummary,
    RepairBrief,
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
    BARE_WORKSPACE_REFERENCE_PATTERN = re.compile(
        r"(?<![\w./-])(?P<path>(?:(?:[\w.-]+/)+[\w.-]+|[\w.-]+)\.[A-Za-z0-9_-]+)(?![\w./-])"
    )
    IMPORT_ERROR_SYMBOL_PATTERN = re.compile(
        r"ImportError:\s+cannot import name ['\"](?P<symbol>[A-Za-z_][A-Za-z0-9_]*)['\"]\s+from\s+['\"](?P<module>[A-Za-z_][A-Za-z0-9_.]*)['\"](?:\s+\((?P<path>[^)]+)\))?",
        re.IGNORECASE,
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
            "priority": 5,
            "pattern": re.compile(
                r"(?P<command>python(?:3)?\s+(?!-m\b)[^`\n]*?\.py\b[^`\n]*)",
                re.IGNORECASE,
            ),
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
    PYTHON_SCRIPT_COMMAND_BOUNDARY_WORDS = {
        "ausgibt",
        "exactly",
        "genau",
        "liefert",
        "output",
        "outputs",
        "print",
        "prints",
        "produce",
        "produces",
        "return",
        "returns",
        "show",
        "shows",
        "zeigt",
    }
    WEB_FEATURE_KEYWORDS = {
        "menu": ("menu", "menue", "menü", "navigation", "nav"),
        "highscore": ("highscore", "high score", "scoreboard", "leaderboard", "best score", "bestenliste"),
        "score": ("score", "punkte", "punktestand", "scoreboard"),
        "start_controls": ("start", "play", "pause", "resume", "restart", "reset", "reload", "neustart", "fortsetzen"),
        "keyboard_controls": (
            "keyboard",
            "keyboard control",
            "keyboard controls",
            "tastatur",
            "tastatursteuerung",
            "keydown",
            "keyup",
            "arrow key",
            "arrow keys",
            "arrow",
            "pfeiltaste",
            "pfeiltasten",
        ),
        "game_over": ("game over", "game-over", "gameover", "neustart", "restart", "reset"),
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
    FAILURE_EVIDENCE_LINE_MARKERS = (
        "assertionerror",
        "calledprocesserror",
        "traceback",
        "missing expected",
        "missing dom ids",
        "missing html file",
        "missing html",
        "not found",
        "filenotfounderror",
        "syntaxerror",
        "indentationerror",
        "taberror",
        "nameerror",
        "importerror",
        "modulenotfounderror",
        "unboundlocalerror",
    )
    NON_FAILURE_SUMMARY_LINE_MARKERS = (
        "refs ok",
        "js parsed:",
        "js parse skipped:",
        "markers:",
        "expected features:",
        "structural web checks only",
    )
    NO_TESTS_RAN_PATTERNS = (
        re.compile(r"\bran\s+0\s+tests?\b", re.IGNORECASE),
        re.compile(r"\bno\s+tests\s+ran\b", re.IGNORECASE),
        re.compile(r"\bcollected\s+0\s+items\b", re.IGNORECASE),
        re.compile(r"start directory is not importable", re.IGNORECASE),
    )
    ASSERTION_MISMATCH_PATTERN = re.compile(
        r"AssertionError:\s*(?P<observed>.+?)\s*!=\s*(?P<expected>.+)"
    )
    ASSERTION_CONTAINMENT_PATTERN = re.compile(
        r"AssertionError:\s*(?P<expected>.+?)\s+not found in\s+(?P<observed>.+)",
        re.IGNORECASE,
    )
    UNDEFINED_RUNTIME_SYMBOL_PATTERNS = (
        re.compile(r"NameError:\s+name ['\"](?P<name>[A-Za-z_][A-Za-z0-9_]*)['\"] is not defined"),
        re.compile(r"UnboundLocalError:\s+cannot access local variable ['\"](?P<name>[A-Za-z_][A-Za-z0-9_]*)['\"]"),
    )
    TEST_HARNESS_RUNTIME_ERROR_MARKERS = (
        "nameerror",
        "importerror",
        "modulenotfounderror",
        "syntaxerror",
        "indentationerror",
        "taberror",
        "unboundlocalerror",
    )
    NON_ASSERT_RUNTIME_ERROR_MARKERS = (
        "attributeerror",
        "calledprocesserror",
        "filenotfounderror",
        "importerror",
        "indentationerror",
        "keyerror",
        "modulenotfounderror",
        "nameerror",
        "runtimeerror",
        "syntaxerror",
        "typeerror",
        "unboundlocalerror",
        "valueerror",
    )
    EXCEPTION_EVIDENCE_PATTERN = re.compile(
        r"\b(?:[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)|SystemExit|CalledProcessError)\b"
    )
    OUTPUT_VERB_PATTERN = (
        r"(?:ausgibt|ausgeben|ausgegeben(?:\s+werden)?|outputs?|prints?|returns?|shows?|zeigt|liefert|produces?)"
    )
    OUTPUT_EXPECTATION_QUOTED_BEFORE_VERB_PATTERN = re.compile(
        rf"(?P<quote>[`'\"])(?P<expected>.+?)(?P=quote)\s+{OUTPUT_VERB_PATTERN}\b",
        re.IGNORECASE,
    )
    OUTPUT_EXPECTATION_QUOTED_AFTER_VERB_PATTERN = re.compile(
        rf"{OUTPUT_VERB_PATTERN}\s+(?:exactly|genau|exakt|weiterhin|still\s+)?(?P<quote>[`'\"])(?P<expected>.+?)(?P=quote)",
        re.IGNORECASE,
    )
    OUTPUT_EXPECTATION_INLINE_PATTERN = re.compile(
        rf"(?:exactly|genau|exakt)\s+(?P<expected>[^`'\n.,;:!?]+?)\s+{OUTPUT_VERB_PATTERN}\b",
        re.IGNORECASE,
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
        commands = self._prefer_explicit_python_runtime_over_cli_smoke(commands)
        commands = self._prefer_python_module_pytest_over_binary_pytest(commands)
        commands = self._replace_generic_unittest_commands(
            commands,
            snapshot=snapshot,
            changed_files=changed_paths,
            session=session,
        )
        if changed_paths and not any(self.command_scope(command) == "runtime" for command in commands):
            commands.extend(self._default_commands(snapshot, changed_paths, task=task))
        synthesized = self._synthesized_runtime_commands(
            snapshot,
            changed_paths,
            commands=commands,
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
                return self._failed_validation_status(current_runs)
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
            return self._failed_validation_status(current_runs)
        if not self.pending_commands(session):
            return "passed"
        return "not_run"

    def _failed_validation_status(self, runs: list[ValidationRunRecord]) -> str:
        latest_failed = next(
            (run for run in reversed(runs) if run.status in {"failed", "timeout"}),
            None,
        )
        if latest_failed is None:
            return "failed"
        bootstrap_status = str(getattr(latest_failed, "bootstrap_status", "") or "").strip()
        if bootstrap_status in {"bootstrap_failed", "bootstrap_reset_required"}:
            return bootstrap_status
        return "failed"

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
        if lowered.startswith("internal:web_runtime_smoke:"):
            return "runtime"
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

    def _prefer_explicit_python_runtime_over_cli_smoke(
        self,
        commands: list[ValidationCommand],
    ) -> list[ValidationCommand]:
        explicit_python_targets = {
            path
            for command in commands
            if command.source in {"task_state", "user_request"}
            and command.verification_scope == "runtime"
            and self._is_explicit_python_script_command(command.command)
            for path in self._paths_from_validation_command(command.command)
        }
        if not explicit_python_targets:
            return commands

        filtered: list[ValidationCommand] = []
        for command in commands:
            if not command.command.startswith("internal:python_cli_smoke:"):
                filtered.append(command)
                continue
            command_paths = set(self._paths_from_validation_command(command.command))
            if command_paths & explicit_python_targets:
                continue
            filtered.append(command)
        return filtered

    def _prefer_python_module_pytest_over_binary_pytest(
        self,
        commands: list[ValidationCommand],
    ) -> list[ValidationCommand]:
        explicit_module_signatures = {
            signature[1]
            for command in commands
            if command.source in {"task_state", "user_request"}
            and command.verification_scope == "runtime"
            and (signature := self._pytest_command_signature(command.command)) is not None
            and signature[0] == "python_module"
        }
        if not explicit_module_signatures:
            return commands

        filtered: list[ValidationCommand] = []
        for command in commands:
            signature = self._pytest_command_signature(command.command)
            if signature is not None and signature[0] == "pytest_binary" and signature[1] in explicit_module_signatures:
                continue
            filtered.append(command)
        return filtered

    def _pytest_command_signature(self, command: str) -> tuple[str, tuple[str, ...]] | None:
        try:
            tokens = shlex.split(str(command or "").strip())
        except ValueError:
            return None
        if not tokens:
            return None
        lowered = [token.lower() for token in tokens]
        if lowered[:3] in (["python", "-m", "pytest"], ["python3", "-m", "pytest"]):
            return ("python_module", tuple(tokens[3:]))
        if lowered[:1] == ["pytest"]:
            return ("pytest_binary", tuple(tokens[1:]))
        return None

    def _is_explicit_python_script_command(self, command: str) -> bool:
        try:
            tokens = shlex.split(str(command or "").strip())
        except ValueError:
            return False
        if len(tokens) < 2:
            return False
        lowered = [token.lower() for token in tokens]
        return lowered[0] in {"python", "python3"} and lowered[1] != "-m" and tokens[1].endswith(".py")

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
            covered_pytest_signatures: set[tuple[str, ...]] = set()
            for spec in self.EXPLICIT_VALIDATION_COMMAND_SPECS:
                for match in spec["pattern"].finditer(raw_text):
                    command = self._normalize_explicit_validation_command(match.group("command"))
                    if not command:
                        continue
                    pytest_signature = self._pytest_command_signature(command)
                    if pytest_signature is not None:
                        flavor, suffix = pytest_signature
                        if flavor == "python_module":
                            covered_pytest_signatures.add(suffix)
                        elif flavor == "pytest_binary" and suffix in covered_pytest_signatures:
                            continue
                    expected_stdout = self._expected_stdout_for_explicit_validation_command(
                        raw_text,
                        match=match,
                        command=command,
                    )
                    extracted.append(
                        ValidationCommand(
                            command=command,
                            kind=str(spec["kind"]),
                            verification_scope=str(spec["verification_scope"]),
                            source=source,
                            priority=int(spec["priority"]),
                            reason="Explicit validation command requested in the active task.",
                            required=True,
                            expected_stdout=expected_stdout,
                        )
                    )
        unique_by_identity: dict[str, ValidationCommand] = {}
        order: list[str] = []
        for item in extracted:
            identity = self.command_identity(item.command)
            if not identity:
                continue
            existing = unique_by_identity.get(identity)
            if existing is None:
                unique_by_identity[identity] = item
                order.append(identity)
                continue
            if existing.expected_stdout is None and item.expected_stdout is not None:
                unique_by_identity[identity] = existing.model_copy(
                    update={"expected_stdout": item.expected_stdout}
                )
        return [unique_by_identity[identity] for identity in order]

    def _expected_stdout_for_explicit_validation_command(
        self,
        raw_text: str,
        *,
        match: re.Match[str],
        command: str,
    ) -> str | None:
        if not self._is_explicit_python_script_command(command):
            return None

        raw_command = str(match.group("command") or "")
        normalized_raw_command = " ".join(raw_command.strip().split())
        inline_context = ""
        if normalized_raw_command.startswith(command):
            inline_context = normalized_raw_command[len(command) :]
        tail_context = str(raw_text or "")[match.end() : match.end() + 240].lstrip(" \t`'\"")
        context = " ".join(part.strip() for part in (inline_context, tail_context) if part.strip())
        if not context:
            return None

        for pattern in (
            self.OUTPUT_EXPECTATION_QUOTED_BEFORE_VERB_PATTERN,
            self.OUTPUT_EXPECTATION_QUOTED_AFTER_VERB_PATTERN,
            self.OUTPUT_EXPECTATION_INLINE_PATTERN,
        ):
            output_match = pattern.search(context)
            if output_match is None:
                continue
            expected = self._normalized_expected_stdout(output_match.group("expected"))
            if expected is not None:
                return expected
        return None

    def _normalized_expected_stdout(self, raw: str | None) -> str | None:
        text = str(raw or "").strip()
        if not text:
            return None
        return text or None

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
        command = self._trim_python_script_command(command)
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

    def _trim_python_script_command(self, command: str) -> str:
        try:
            tokens = shlex.split(str(command or "").strip())
        except ValueError:
            return command
        if len(tokens) < 2:
            return command

        lowered = [token.lower() for token in tokens]
        if lowered[0] not in {"python", "python3"} or lowered[1] == "-m":
            return command

        trimmed = tokens[:2]
        for token in tokens[2:]:
            cleaned = str(token or "").strip().rstrip(".,;:!?").lower()
            if cleaned in self.PYTHON_SCRIPT_COMMAND_BOUNDARY_WORDS:
                break
            trimmed.append(token)
        return " ".join(trimmed) if trimmed else command

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

    def web_structural_proxy_sufficient(
        self,
        session: SessionState,
        *,
        current_generation_only: bool = True,
    ) -> bool:
        if not self.web_functional_verification_required(session):
            return False
        if not self.has_structural_web_success(session, current_generation_only=current_generation_only):
            return False
        if self.has_runtime_success(session, current_generation_only=current_generation_only):
            return True

        changed_paths = [
            item.path
            for item in session.changed_files
            if item.path
        ]
        if not changed_paths or len(changed_paths) > 4:
            return False

        allowed_suffixes = {".html", ".htm", ".css", ".js", ".mjs", ".cjs"}
        if any(Path(path).suffix.lower() not in allowed_suffixes for path in changed_paths):
            return False

        html_files = [path for path in changed_paths if Path(path).suffix.lower() in {".html", ".htm"}]
        if len(html_files) != 1:
            return False

        snapshot = session.workspace_snapshot
        if snapshot is not None and snapshot.file_count > 20:
            return False

        if any(
            command.required and command.verification_scope == "runtime"
            for command in session.validation_plan
        ):
            return False

        task_text = (
            session.task_state.latest_user_turn
            if session.task_state is not None and session.task_state.latest_user_turn
            else session.task
        )
        expected_features = self._expected_web_features(task_text)
        if self._interactive_web_request(task_text) and len(expected_features) < 2:
            return False
        return True

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
        traceback_frames = self._traceback_workspace_frames(
            session,
            failed_run,
        )
        prefer_test_runtime_target = self._test_harness_runtime_failure(
            session,
            failed_run,
            traceback_frames=traceback_frames,
        )
        traceback_file_hints, traceback_line_hints = self._traceback_workspace_hints(
            session,
            failed_run,
            traceback_frames=traceback_frames,
            prefer_test_runtime_target=prefer_test_runtime_target,
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
        diagnostics = self._related_diagnostics(session, failed_run, artifact_paths)
        runtime_failure_text = "\n".join(
            part
            for part in [
                str(failed_run.excerpt or "").strip(),
                str(failed_run.summary or "").strip(),
                *(str(item.excerpt or "").strip() for item in diagnostics),
                *(str(item.summary or "").strip() for item in diagnostics),
            ]
            if part
        )
        content_assertion_target = self._single_changed_content_assertion_target(
            session,
            failed_run,
            runtime_failure_text,
        )
        entrypoint_assertion_target = self._entrypoint_assertion_runtime_target(
            session,
            failed_run,
            traceback_frames=traceback_frames,
            candidate_paths=artifact_paths,
        )
        implicated_symbols = (
            self._implicated_symbols_from_failure_text(runtime_failure_text, runtime_failure_text)
            if failed_run.verification_scope == "runtime"
            and content_assertion_target is None
            else []
        )
        symbol_resolved_paths = self._symbol_resolved_workspace_paths(
            session,
            implicated_symbols,
        )
        artifact_paths = self._unique_paths(
            [content_assertion_target, entrypoint_assertion_target, *artifact_paths, *symbol_resolved_paths]
        )
        failure_scoped_paths = self._failure_scoped_workspace_paths(
            session,
            failed_run,
            diagnostics,
        )
        artifact_paths = self._prioritize_failure_scoped_artifact_paths(
            artifact_paths,
            failure_scoped_paths,
        )
        artifact_paths = self._prioritize_runtime_artifact_paths(
            artifact_paths,
            failed_run,
            prefer_test_runtime_target=prefer_test_runtime_target,
            prefer_content_runtime_target=content_assertion_target is not None,
        )
        artifact_paths = self._unique_paths([content_assertion_target, entrypoint_assertion_target, *artifact_paths])
        missing_unittest_package_inits = self._missing_unittest_package_inits(
            session,
            failed_run,
            artifact_paths,
        )
        expected_features = self._expected_features_from_command(failed_run.command)
        missing_features = self._missing_features_from_failure(failed_run, diagnostics)
        runtime_hint_order = (
            [
                *artifact_paths,
                *failure_scoped_paths,
                *(symbol_resolved_paths if content_assertion_target is None else []),
                *traceback_file_hints,
                *referenced_workspace_paths,
                *missing_unittest_package_inits,
                *(path for diagnostic in diagnostics for path in diagnostic.file_hints),
            ]
            if failed_run.verification_scope == "runtime"
            else [
                *failure_scoped_paths,
                *artifact_paths,
                *symbol_resolved_paths,
                *traceback_file_hints,
                *referenced_workspace_paths,
                *missing_unittest_package_inits,
                *(path for diagnostic in diagnostics for path in diagnostic.file_hints),
            ]
        )
        file_hints = self._unique_paths(runtime_hint_order)
        line_hints = self._unique_line_hints(
            [
                *traceback_line_hints,
                *(line for diagnostic in diagnostics for line in diagnostic.line_hints),
            ]
        )
        import_wrapper_target = self._unittest_import_wrapper_non_test_target(
            failed_run,
            candidate_paths=[*referenced_workspace_paths, *artifact_paths, *file_hints],
        )
        if import_wrapper_target:
            artifact_paths = self._unique_paths([import_wrapper_target, *artifact_paths])
            file_hints = self._unique_paths([import_wrapper_target, *file_hints])
        action_hints = self._unique_strings(
            [hint for diagnostic in diagnostics for hint in diagnostic.action_hints]
        )
        summary = str(failed_run.summary or "").strip() or "Validation failed."
        excerpt = self._failure_excerpt(failed_run, diagnostics)
        raw_failure_text = runtime_failure_text
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
        repair_brief = self._build_repair_brief(
            session,
            failed_run,
            artifact_paths=artifact_paths,
            symbol_resolved_paths=symbol_resolved_paths,
            file_hints=file_hints,
            line_hints=line_hints,
            summary=summary,
            excerpt=excerpt,
            raw_failure_text=raw_failure_text,
            failure_summary=failure_summary,
            repair_requirements=repair_requirements,
            missing_features=missing_features,
            prefer_test_runtime_target=prefer_test_runtime_target,
            prefer_content_runtime_target=content_assertion_target is not None,
            implicated_symbols=implicated_symbols,
        )
        brief_primary_target = str(getattr(repair_brief, "primary_target", "") or "").strip()
        if brief_primary_target and brief_primary_target != (artifact_paths[0] if artifact_paths else ""):
            repair_requirements = self._repair_requirements(
                failed_run,
                diagnostics,
                artifact_paths=artifact_paths,
                expected_features=expected_features,
                missing_features=missing_features,
                missing_unittest_package_inits=missing_unittest_package_inits,
                primary_target_override=brief_primary_target,
            )
            repair_brief = repair_brief.model_copy(
                update={
                    "repair_constraints": [
                        str(item or "").strip()
                        for item in repair_requirements[:4]
                        if str(item or "").strip()
                    ]
                }
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
            root_cause_summary=repair_brief.root_cause_summary if repair_brief is not None else None,
            bootstrap_status=repair_brief.bootstrap_status if repair_brief is not None else "none",
            repair_brief=repair_brief,
        )

    def _build_repair_brief(
        self,
        session: SessionState,
        failed_run: ValidationRunRecord,
        *,
        artifact_paths: list[str],
        symbol_resolved_paths: list[str],
        file_hints: list[str],
        line_hints: list[int],
        summary: str,
        excerpt: str,
        raw_failure_text: str,
        failure_summary: str,
        repair_requirements: list[str],
        missing_features: list[str],
        prefer_test_runtime_target: bool = False,
        prefer_content_runtime_target: bool = False,
        implicated_symbols: list[str] | None = None,
    ) -> RepairBrief:
        primary_target = self._primary_repair_target(
            artifact_paths=artifact_paths,
            file_hints=file_hints,
            verification_scope=failed_run.verification_scope,
            prefer_test_runtime_target=prefer_test_runtime_target,
            prefer_content_runtime_target=prefer_content_runtime_target,
        )
        expected_semantics, observed_semantics = self._failure_semantics(
            failed_run,
            excerpt=excerpt,
            failure_summary=failure_summary,
            missing_features=missing_features,
        )
        raw_failure = raw_failure_text or "\n".join(part for part in [excerpt, summary] if part)
        import_symbol, _import_module, import_provider_path = self._missing_import_symbol_details(raw_failure)
        implicated_symbols = list(implicated_symbols or []) or self._implicated_symbols_from_failure_text(
            excerpt,
            failure_summary,
        )
        if import_symbol and import_symbol not in implicated_symbols:
            implicated_symbols.insert(0, import_symbol)
        failure_type = self._failure_type(
            failed_run,
            excerpt=excerpt,
            failure_summary=failure_summary,
            missing_features=missing_features,
        )
        if failure_type == "test_discovery_gap":
            primary_target = self._test_discovery_primary_target(
                artifact_paths=artifact_paths,
                file_hints=file_hints,
                current_primary=primary_target,
            )
        script_target = self._python_script_target_from_failure(failed_run, raw_failure_text)
        normalized_script = self._workspace_relative_path(session, script_target)
        normalized_import_provider = self._workspace_relative_path(session, import_provider_path)
        primary_target = self._prefer_behavioral_runtime_primary_target(
            artifact_paths=artifact_paths,
            file_hints=file_hints,
            verification_scope=failed_run.verification_scope,
            failure_type=failure_type,
            primary_target=primary_target,
            normalized_script_target=normalized_script,
        )
        primary_target = self._prefer_multi_assertion_symbol_primary_target(
            session,
            verification_scope=failed_run.verification_scope,
            failure_type=failure_type,
            primary_target=primary_target,
            expected_semantics=expected_semantics,
            observed_semantics=observed_semantics,
            symbol_resolved_paths=symbol_resolved_paths,
        )
        if failure_type == "import_failure":
            if normalized_script:
                primary_target = normalized_script
            provider_target = self._prefer_import_failure_provider_target(
                artifact_paths=artifact_paths,
                file_hints=file_hints,
                current_primary=primary_target,
                normalized_provider_target=normalized_import_provider,
            )
            if provider_target:
                primary_target = provider_target
        target_line_hint = self._target_traceback_line_hint(
            primary_target,
            text=raw_failure,
        )
        implicated_region_hint = self._implicated_region_hint(
            primary_target,
            target_line_hint=target_line_hint,
            implicated_symbols=implicated_symbols,
            failure_type=failure_type,
        )
        failure_signature = self._failure_signature(
            failed_run,
            failure_type=failure_type,
            primary_target=primary_target,
            expected_semantics=expected_semantics,
            observed_semantics=observed_semantics,
            implicated_symbols=implicated_symbols,
        )
        bootstrap_status = self._bootstrap_status(
            failed_run,
            failure_type=failure_type,
            raw_failure_text=raw_failure_text,
            primary_target=primary_target,
            target_line_hint=target_line_hint,
        )
        root_cause_summary = self._root_cause_summary(
            failed_run,
            failure_type=failure_type,
            raw_failure_text=raw_failure_text,
            primary_target=primary_target,
            expected_semantics=expected_semantics,
            observed_semantics=observed_semantics,
            missing_features=missing_features,
            bootstrap_status=bootstrap_status,
        )
        locked_target = self._locked_repair_target(
            session,
            primary_target=primary_target,
            candidate_paths=[*artifact_paths, *file_hints],
            failure_signature=failure_signature,
            verification_scope=failed_run.verification_scope,
        )
        allowed_files = self._repair_allowed_files(
            primary_target=primary_target,
            locked_target=locked_target,
            candidate_paths=[*artifact_paths, *file_hints],
            failure_type=failure_type,
            bootstrap_status=bootstrap_status,
        )
        forbidden_files = self._repair_forbidden_files(
            allowed_files=allowed_files,
            candidate_paths=[*artifact_paths, *file_hints],
            failure_type=failure_type,
        )
        return RepairBrief(
            failure_type=failure_type,
            failure_signature=failure_signature,
            primary_target=primary_target,
            locked_target=locked_target,
            root_cause_summary=root_cause_summary,
            bootstrap_status=bootstrap_status,
            bootstrap_reason=root_cause_summary if bootstrap_status != "none" else None,
            expected_semantics=expected_semantics[:3],
            observed_semantics=observed_semantics[:3],
            implicated_symbols=implicated_symbols[:6],
            implicated_region_hint=implicated_region_hint,
            repair_constraints=[str(item or "").strip() for item in repair_requirements[:4] if str(item or "").strip()],
            recent_failed_attempts=self._recent_failed_attempt_summaries(session, failure_signature=failure_signature),
            allowed_files=allowed_files[:4],
            forbidden_files=forbidden_files[:4],
        )

    def _primary_repair_target(
        self,
        *,
        artifact_paths: list[str],
        file_hints: list[str],
        verification_scope: str,
        prefer_test_runtime_target: bool = False,
        prefer_content_runtime_target: bool = False,
    ) -> str | None:
        candidates = self._unique_paths([*artifact_paths, *file_hints])
        if verification_scope == "runtime":
            if prefer_test_runtime_target:
                validation = [path for path in candidates if path and self._is_test_path(path)]
                if validation:
                    return validation[0]
            if prefer_content_runtime_target:
                content_candidates = [path for path in candidates if path and not self._is_test_path(path)]
                if content_candidates:
                    return content_candidates[0]
            implementation = [
                path
                for path in candidates
                if path and not self._is_test_path(path) and not self._is_documentation_path(path)
            ]
            if implementation:
                return implementation[0]
        return next((path for path in candidates if path), None)

    def _test_discovery_primary_target(
        self,
        *,
        artifact_paths: list[str],
        file_hints: list[str],
        current_primary: str | None,
    ) -> str | None:
        candidates = self._unique_paths([*artifact_paths, *file_hints])
        existing_test_modules = [
            path
            for path in candidates
            if path
            and self._is_test_path(path)
            and not Path(path).name.lower() == "__init__.py"
        ]
        if existing_test_modules:
            return existing_test_modules[0]
        test_support = [
            path
            for path in candidates
            if path and self._is_test_path(path)
        ]
        if test_support:
            return test_support[0]
        return str(current_primary or "").strip() or None

    def _prefer_behavioral_runtime_primary_target(
        self,
        *,
        artifact_paths: list[str],
        file_hints: list[str],
        verification_scope: str,
        failure_type: str,
        primary_target: str | None,
        normalized_script_target: str | None,
    ) -> str | None:
        if verification_scope != "runtime" or failure_type != "assertion_mismatch":
            return primary_target
        script_target = str(normalized_script_target or "").strip()
        if not script_target:
            return primary_target
        current_target = str(primary_target or "").strip()
        if current_target and current_target != script_target:
            return primary_target
        candidates = self._unique_paths([*artifact_paths, *file_hints])
        implementation = [
            path
            for path in candidates
            if (
                path
                and path != script_target
                and not self._is_test_path(path)
                and not self._is_documentation_path(path)
            )
        ]
        if implementation:
            return implementation[0]
        return primary_target

    def _prefer_multi_assertion_symbol_primary_target(
        self,
        session: SessionState,
        *,
        verification_scope: str,
        failure_type: str,
        primary_target: str | None,
        expected_semantics: list[str],
        observed_semantics: list[str],
        symbol_resolved_paths: list[str],
    ) -> str | None:
        if verification_scope != "runtime" or failure_type != "assertion_mismatch":
            return primary_target
        if len(expected_semantics) < 2 or len(observed_semantics) < 2:
            return primary_target

        snapshot = session.workspace_snapshot
        if snapshot is None:
            return primary_target

        entrypoints = {
            str(path or "").strip()
            for path in getattr(snapshot, "entrypoints", [])
            if str(path or "").strip()
        }
        current_target = str(primary_target or "").strip()
        preferred_candidates = [
            path
            for path in self._unique_paths(symbol_resolved_paths)
            if path
            and not self._is_test_path(path)
            and not self._is_documentation_path(path)
            and path not in entrypoints
            and Path(path).name.lower() != "__init__.py"
        ]
        if not preferred_candidates:
            return primary_target
        if not current_target:
            return preferred_candidates[0]
        if current_target in entrypoints or Path(current_target).name.lower() == "__init__.py":
            return preferred_candidates[0]
        return primary_target

    def _failure_type(
        self,
        failed_run: ValidationRunRecord,
        *,
        excerpt: str,
        failure_summary: str,
        missing_features: list[str],
    ) -> str:
        text = "\n".join(part for part in [failure_summary, excerpt, str(failed_run.summary or "").strip()] if part).lower()
        if missing_features and failed_run.verification_scope == "structural":
            return "structural_missing_feature"
        if "assertionerror" in text and ("!=" in text or "not found in" in text):
            return "assertion_mismatch"
        if "unrecognized arguments" in text:
            return "runtime_argument_parsing"
        if "filenotfounderror" in text or "no such file or directory" in text:
            return "missing_artifact"
        if any(pattern.search(text) for pattern in self.NO_TESTS_RAN_PATTERNS):
            return "test_discovery_gap"
        if "importerror" in text or "modulenotfounderror" in text:
            return "import_failure"
        if failed_run.verification_scope == "runtime":
            return "runtime_failure"
        return f"{failed_run.verification_scope}_failure"

    def _failure_semantics(
        self,
        failed_run: ValidationRunRecord,
        *,
        excerpt: str,
        failure_summary: str,
        missing_features: list[str],
    ) -> tuple[list[str], list[str]]:
        text = "\n".join(part for part in [excerpt, failure_summary, str(failed_run.summary or "").strip()] if part)
        expected: list[str] = []
        observed: list[str] = []
        for assertion_observed, assertion_expected in self._assertion_semantic_pairs(text)[:3]:
            if assertion_expected is not None:
                expected.append(f"Validation should produce: {self._semantic_display_value(assertion_expected)}")
            if assertion_observed is not None:
                observed.append(f"Validation currently produces: {self._semantic_display_value(assertion_observed)}")
        lowered = text.lower()
        if not expected and missing_features:
            expected.append("Required validation features must be present: " + ", ".join(missing_features[:3]))
        if "unrecognized arguments" in lowered:
            observed.append("The current runtime path rejects the exercised invocation as extra CLI arguments.")
        if "filenotfounderror" in lowered or "no such file or directory" in lowered:
            missing_path = self._missing_path_from_failure_text(text)
            expected.append("Referenced runtime inputs should exist and load successfully.")
            if missing_path:
                observed.append(f"The current runtime path cannot open: {missing_path}")
        called_process_command, exit_code = self._called_process_error_details(text)
        if called_process_command:
            expected.append("The exercised runtime command should exit successfully.")
            observed_text = f"The current runtime command exits non-zero when invoking: {called_process_command}"
            if exit_code is not None:
                observed_text += f" (exit status {exit_code})"
            observed.append(observed_text)
        if any(pattern.search(lowered) for pattern in self.NO_TESTS_RAN_PATTERNS):
            expected.append("The targeted tests should be discoverable and execute.")
            observed.append("The current validation run executes zero tests.")
        undefined_symbol, usage_line = self._undefined_runtime_symbol_details(text)
        if undefined_symbol:
            expected.append(f"The symbol '{undefined_symbol}' should be bound or imported before it is used.")
            observed_text = f"The current runtime path uses '{undefined_symbol}' before it is bound or imported."
            if usage_line:
                observed_text += f" Current use: {usage_line}"
            observed.append(observed_text)
        return self._unique_strings(expected), self._unique_strings(observed)

    def _assertion_semantic_pairs(self, text: str) -> list[tuple[str | None, str | None]]:
        lines = [str(raw or "").rstrip() for raw in str(text or "").splitlines()]
        if not lines:
            return []

        pairs: list[tuple[str | None, str | None]] = []
        for index, raw in enumerate(lines):
            stripped = raw.strip()
            if not stripped.startswith("AssertionError:"):
                continue
            diff_lines = self._adjacent_assertion_diff_lines(lines, start_index=index)
            observed = self._first_semantic_diff_value(diff_lines, prefix="-")
            expected = self._first_semantic_diff_value(diff_lines, prefix="+")
            line_observed, line_expected = self._assertion_line_pair(stripped)
            if observed is None:
                observed = line_observed
            if expected is None:
                expected = line_expected
            if observed is None and expected is None:
                continue
            pairs.append((observed, expected))

        if pairs:
            return self._unique_semantic_pairs(pairs)

        fallback_pairs: list[tuple[str | None, str | None]] = []
        parsed_observed, parsed_expected = self._assertion_line_pair(str(text or ""))
        if parsed_observed is not None or parsed_expected is not None:
            fallback_pairs.append((parsed_observed, parsed_expected))
        diff_observed = self._first_semantic_diff_value(lines, prefix="-")
        diff_expected = self._first_semantic_diff_value(lines, prefix="+")
        if diff_observed is not None or diff_expected is not None:
            fallback_pairs.append((diff_observed, diff_expected))
        return self._unique_semantic_pairs(fallback_pairs)

    def _assertion_diff_values(self, text: str) -> tuple[str | None, str | None]:
        pairs = self._assertion_semantic_pairs(text)
        if pairs:
            return pairs[0]
        return None, None

    def _adjacent_assertion_diff_lines(
        self,
        lines: list[str],
        *,
        start_index: int,
        limit: int = 6,
    ) -> list[str]:
        collected: list[str] = []
        for raw in lines[start_index + 1 :]:
            stripped = str(raw or "").strip()
            if not stripped:
                if collected:
                    break
                continue
            if stripped.startswith(("Traceback", "File ")):
                break
            if re.match(r"[=\-]{5,}", stripped):
                if collected:
                    break
                continue
            if re.search(r"\b(?:FAIL|FAILED|ERROR)\b", stripped):
                break
            if stripped.startswith(("-", "+", "?")):
                if self._semantic_diff_value(stripped) is None:
                    if collected:
                        break
                    continue
                collected.append(stripped)
                if len(collected) >= limit:
                    break
                continue
            if collected:
                break
        return collected

    def _first_semantic_diff_value(
        self,
        lines: list[str],
        *,
        prefix: str,
    ) -> str | None:
        for raw in lines:
            stripped = str(raw or "").strip()
            if not stripped.startswith(prefix):
                continue
            value = self._semantic_diff_value(stripped)
            if value is not None:
                return value
        return None

    def _semantic_diff_value(self, line: str) -> str | None:
        stripped = str(line or "").strip()
        if not stripped or stripped[0] not in {"-", "+"}:
            return None
        value = stripped[1:].strip()
        if not value:
            return None
        if re.fullmatch(r"[-=+?_~*.]{4,}", value):
            return None
        lowered = value.lower()
        if lowered in {"traceback", "assertionerror", "e assertionerror"}:
            return None
        return value

    def _assertion_line_values(self, text: str) -> tuple[str | None, str | None]:
        for raw in str(text or "").splitlines():
            observed, expected = self._assertion_line_pair(raw)
            if observed is not None or expected is not None:
                return observed, expected
        return None, None

    def _assertion_line_pair(self, text: str) -> tuple[str | None, str | None]:
        stripped = str(text or "").strip()
        if not stripped:
            return None, None
        match = self.ASSERTION_MISMATCH_PATTERN.search(stripped)
        if match is None:
            match = self.ASSERTION_CONTAINMENT_PATTERN.search(stripped)
        if match is None:
            return None, None
        observed = self._literal_or_text(match.group("observed"))
        expected = self._literal_or_text(match.group("expected"))
        return observed, expected

    def _semantic_display_value(self, value: str | None) -> str | None:
        if value is None:
            return None
        return "''" if value == "" else value

    def _single_changed_content_assertion_target(
        self,
        session: SessionState,
        failed_run: ValidationRunRecord,
        raw_failure_text: str,
    ) -> str | None:
        if failed_run.verification_scope != "runtime":
            return None
        if not self.ASSERTION_CONTAINMENT_PATTERN.search(str(raw_failure_text or "")):
            return None
        if self._contains_non_assert_runtime_error(raw_failure_text):
            return None
        changed_targets = [
            item.path
            for item in session.changed_files
            if item.path and not self._is_test_path(item.path)
        ]
        changed_targets = self._unique_paths(changed_targets)
        if len(changed_targets) == 1:
            return changed_targets[0]
        return None

    def _contains_non_assert_runtime_error(self, text: str) -> bool:
        lowered = str(text or "").lower()
        if not lowered:
            return False
        return any(marker in lowered for marker in self.NON_ASSERT_RUNTIME_ERROR_MARKERS)

    def _entrypoint_assertion_runtime_target(
        self,
        session: SessionState,
        failed_run: ValidationRunRecord,
        *,
        traceback_frames: list[tuple[str, int]] | None = None,
        candidate_paths: list[str],
    ) -> str | None:
        if failed_run.verification_scope != "runtime" or self._no_tests_executed(failed_run):
            return None
        failure_text = "\n".join(
            part
            for part in [
                str(failed_run.excerpt or "").strip(),
                str(failed_run.summary or "").strip(),
            ]
            if part
        )
        lowered = failure_text.lower()
        if "assertionerror" not in lowered or ("!=" not in failure_text and "not found in" not in lowered):
            return None
        frames = traceback_frames if traceback_frames is not None else self._traceback_workspace_frames(session, failed_run)
        if any(path and not self._is_test_path(path) for path, _line in frames):
            return None

        snapshot = session.workspace_snapshot
        entrypoints = {
            str(path or "").strip()
            for path in getattr(snapshot, "entrypoints", [])
            if str(path or "").strip()
        }
        services = {
            str(path or "").strip()
            for path in getattr(snapshot, "service_files", [])
            if str(path or "").strip()
        }
        if not entrypoints and not services:
            return None

        task_primary_targets = {
            str(getattr(artifact, "path", "") or "").strip()
            for artifact in getattr(session.task_state, "target_artifacts", []) or []
            if str(getattr(artifact, "path", "") or "").strip()
            and str(getattr(artifact, "role", "") or "").strip() == "primary_target"
        }
        changed_targets = {
            str(item.path or "").strip()
            for item in session.changed_files
            if str(item.path or "").strip()
        }
        implementation_candidates = [
            path
            for path in self._unique_paths(candidate_paths)
            if path
            and not self._is_test_path(path)
            and not self._is_documentation_path(path)
            and Path(path).name.lower() != "__init__.py"
        ]
        if len(implementation_candidates) < 2:
            return None

        def score(path: str) -> tuple[int, int, int, int]:
            return (
                1 if path in entrypoints else 0,
                1 if path in services else 0,
                1 if path in task_primary_targets else 0,
                1 if path in changed_targets else 0,
            )

        current = implementation_candidates[0]
        scored_candidates = sorted(
            implementation_candidates,
            key=lambda path: (
                -score(path)[0],
                -score(path)[1],
                -score(path)[2],
                -score(path)[3],
                path,
            ),
        )
        best = scored_candidates[0]
        current_score = score(current)
        best_score = score(best)
        if best == current or best_score <= current_score:
            return None
        if len(scored_candidates) > 1 and score(scored_candidates[1]) == best_score:
            return None
        return best

    def _literal_or_text(self, raw: str | None) -> str | None:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text[:160]
        return str(value)

    def _missing_path_from_failure_text(self, text: str) -> str | None:
        for raw in str(text or "").splitlines():
            stripped = raw.strip()
            if "no such file or directory" not in stripped.lower():
                continue
            parts = re.findall(r"['\"]([^'\"]+)['\"]", stripped)
            if parts:
                return parts[-1]
        return None

    def _called_process_error_details(self, text: str) -> tuple[str | None, int | None]:
        pattern = re.compile(
            r"CalledProcessError:\s+Command\s+'?(?P<command>\[[^\]]+\])'?\s+returned non-zero exit status\s+(?P<code>\d+)",
            re.IGNORECASE,
        )
        match = pattern.search(str(text or ""))
        if match is None:
            return None, None
        command = str(match.group("command") or "").strip()
        if not command:
            return None, None
        try:
            exit_code = int(match.group("code") or 0)
        except ValueError:
            exit_code = None
        return command, exit_code

    def _implicated_symbols_from_failure_text(self, excerpt: str, failure_summary: str) -> list[str]:
        symbols: list[str] = []
        text = "\n".join(part for part in [excerpt, failure_summary] if part)
        import_symbol, import_module, _import_path = self._missing_import_symbol_details(text)
        if import_symbol:
            symbols.append(import_symbol)
        if import_module and import_module not in symbols:
            symbols.append(import_module)
        undefined_symbol, _usage_line = self._undefined_runtime_symbol_details(text)
        if undefined_symbol:
            symbols.append(undefined_symbol)
        missing_module = self._missing_module_name(text)
        if missing_module and missing_module not in symbols:
            symbols.append(missing_module)
        for match in self.TRACEBACK_FRAME_PATTERN.finditer(text):
            symbol = str(match.group("symbol") or "").strip()
            if (
                symbol
                and not symbol.startswith("test_")
                and not self._is_generic_runtime_failure_symbol(symbol)
                and symbol not in symbols
            ):
                symbols.append(symbol)
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\(", text):
            name = str(match.group(1) or "").strip()
            prefix = text[: match.start(1)].rstrip()
            if (
                not name
                or prefix.endswith(".")
                or name in {"self", "print", "open"}
                or name.startswith("test_")
                or self._is_generic_runtime_failure_symbol(name)
            ):
                continue
            if name not in symbols:
                symbols.append(name)
        return symbols

    def _missing_import_symbol_details(self, text: str) -> tuple[str | None, str | None, str | None]:
        match = self.IMPORT_ERROR_SYMBOL_PATTERN.search(str(text or ""))
        if match is None:
            return None, None, None
        symbol = str(match.group("symbol") or "").strip() or None
        module = str(match.group("module") or "").strip() or None
        path = str(match.group("path") or "").strip() or None
        return symbol, module, path

    def _missing_module_name(self, text: str) -> str | None:
        for pattern in (
            re.compile(r"ModuleNotFoundError:\s+No module named ['\"](?P<name>[A-Za-z_][A-Za-z0-9_.]*)['\"]"),
            re.compile(r"ImportError:\s+No module named ['\"](?P<name>[A-Za-z_][A-Za-z0-9_.]*)['\"]"),
        ):
            match = pattern.search(str(text or ""))
            if match is None:
                continue
            name = str(match.group("name") or "").strip()
            if name:
                return name
        return None

    def _is_generic_runtime_failure_symbol(self, symbol: str) -> bool:
        lowered = str(symbol or "").strip().lower()
        if not lowered:
            return True
        if lowered in {
            "assertionerror",
            "calledprocesserror",
            "command",
            "error",
            "errors",
            "fail",
            "failed",
            "file",
            "output",
            "process",
            "retcode",
            "run",
            "stderr",
            "stdout",
            "traceback",
        }:
            return True
        return False

    def _undefined_runtime_symbol_details(self, text: str) -> tuple[str | None, str | None]:
        symbol_name = self._undefined_runtime_symbol_name(text)
        if not symbol_name:
            return None, None
        usage_line = self._undefined_runtime_symbol_usage_line(text, symbol_name)
        return symbol_name, usage_line

    def _undefined_runtime_symbol_name(self, text: str) -> str | None:
        for pattern in self.UNDEFINED_RUNTIME_SYMBOL_PATTERNS:
            match = pattern.search(str(text or ""))
            if match is None:
                continue
            name = str(match.group("name") or "").strip()
            if name:
                return name
        return None

    def _undefined_runtime_symbol_usage_line(self, text: str, symbol_name: str) -> str | None:
        target = str(symbol_name or "").strip()
        if not target:
            return None
        symbol_pattern = re.compile(rf"\b{re.escape(target)}\b")
        for raw in str(text or "").splitlines():
            stripped = str(raw or "").strip()
            if not stripped:
                continue
            if stripped.startswith(("Traceback", "File ")):
                continue
            if "NameError:" in stripped or "UnboundLocalError:" in stripped:
                continue
            if stripped.startswith("^"):
                continue
            if symbol_pattern.search(stripped):
                return stripped
        return None

    def _implicated_region_hint(
        self,
        primary_target: str | None,
        *,
        target_line_hint: int | None,
        implicated_symbols: list[str],
        failure_type: str,
    ) -> str | None:
        target = str(primary_target or "").strip()
        if not target:
            return None
        if target_line_hint is not None and target_line_hint > 0:
            return f"{target}:line {target_line_hint}"
        if failure_type == "import_failure" and implicated_symbols:
            return f"{target}:symbol {implicated_symbols[0]}"
        if self._is_test_path(target) and implicated_symbols:
            return f"{target}:symbol {implicated_symbols[0]}"
        return target

    def _prefer_import_failure_provider_target(
        self,
        *,
        artifact_paths: list[str],
        file_hints: list[str],
        current_primary: str | None,
        normalized_provider_target: str | None,
    ) -> str | None:
        provider_target = str(normalized_provider_target or "").strip()
        if not provider_target or self._is_test_path(provider_target) or self._is_documentation_path(provider_target):
            return str(current_primary or "").strip() or None
        candidates = self._unique_paths([*artifact_paths, *file_hints])
        if provider_target in candidates:
            return provider_target
        if any(provider_target.endswith(f"/{candidate}") for candidate in candidates if candidate):
            matched = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate and provider_target.endswith(f"/{candidate}")
                ),
                None,
            )
            if matched:
                return matched
        return provider_target or str(current_primary or "").strip() or None

    def _target_traceback_line_hint(
        self,
        primary_target: str | None,
        *,
        text: str,
    ) -> int | None:
        target = str(primary_target or "").strip().replace("\\", "/")
        if not target:
            return None
        for match in self.TRACEBACK_FRAME_PATTERN.finditer(str(text or "")):
            raw_path = str(match.group("path") or "").strip().replace("\\", "/")
            if not raw_path:
                continue
            normalized = raw_path.removeprefix("./")
            if not (normalized == target or normalized.endswith(f"/{target}")):
                continue
            try:
                line = int(match.group("line") or 0)
            except ValueError:
                line = 0
            if line > 0:
                return line
        return None

    def _failure_signature(
        self,
        failed_run: ValidationRunRecord,
        *,
        failure_type: str,
        primary_target: str | None,
        expected_semantics: list[str],
        observed_semantics: list[str],
        implicated_symbols: list[str],
    ) -> str:
        normalized_command = self.command_identity(failed_run.command)
        command_parts = normalized_command.split()
        validator_token = command_parts[-1] if len(command_parts) > 2 else normalized_command
        signature_payload = {
            "scope": failed_run.verification_scope,
            "failure_type": failure_type,
            "validator": validator_token,
            "primary_target": str(primary_target or "").strip(),
            "symbols": implicated_symbols[:3],
            "expected": expected_semantics[:2],
            "observed": observed_semantics[:2],
        }
        digest = hashlib.sha1(
            json.dumps(signature_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        return f"{failed_run.verification_scope}:{failure_type}:{digest}"

    def _bootstrap_status(
        self,
        failed_run: ValidationRunRecord,
        *,
        failure_type: str,
        raw_failure_text: str,
        primary_target: str | None,
        target_line_hint: int | None,
    ) -> str:
        if failed_run.verification_scope != "runtime":
            return "none"
        if self._no_tests_executed(failed_run):
            return "bootstrap_failed"
        if failure_type != "import_failure":
            return "none"
        lowered_failure = str(raw_failure_text or "").strip().lower()
        if "failed to import test module" in lowered_failure or "loader._failed_test" in lowered_failure:
            return "bootstrap_failed"

        script_target = self._python_script_target_from_failure(failed_run, raw_failure_text)
        if script_target:
            return "bootstrap_failed"
        if target_line_hint is not None and 0 < target_line_hint <= 12:
            return "bootstrap_failed"
        return "none"

    def _root_cause_summary(
        self,
        failed_run: ValidationRunRecord,
        *,
        failure_type: str,
        raw_failure_text: str,
        primary_target: str | None,
        expected_semantics: list[str],
        observed_semantics: list[str],
        missing_features: list[str],
        bootstrap_status: str,
    ) -> str:
        target = str(primary_target or "").strip() or "the implicated artifact"
        missing_module = self._missing_module_name(raw_failure_text)
        script_target = self._python_script_target_from_failure(failed_run, raw_failure_text)

        if bootstrap_status == "bootstrap_failed":
            if self._no_tests_executed(failed_run):
                return (
                    "The requested validation is failing in its bootstrap or discovery path before the intended "
                    f"tests execute, so {target} or the adjacent test/package wiring must be repaired first."
                )
            if failure_type == "import_failure" and script_target:
                subject = script_target if "/" in script_target or script_target.endswith(".py") else target
                if missing_module:
                    return (
                        f"{subject} is failing during direct script bootstrap because the startup import path cannot "
                        f"resolve '{missing_module}' before runtime logic begins."
                    )
                return (
                    f"{subject} is failing during direct script bootstrap before the intended runtime path can start."
                )
            return f"The active runtime path is failing during import/bootstrap before {target} can reach its intended behavior."

        if failure_type == "structural_missing_feature" and missing_features:
            return f"{target} is missing validation-required behavior or structure: {', '.join(missing_features[:3])}."
        if failure_type == "assertion_mismatch" and expected_semantics and observed_semantics:
            if len(expected_semantics) > 1 and len(observed_semantics) > 1:
                return (
                    f"{target} still produces the wrong behavior across multiple validation assertions: "
                    f"{expected_semantics[0]} but {observed_semantics[0]}; "
                    f"{expected_semantics[1]} but {observed_semantics[1]}."
                )
            return f"{target} still produces the wrong behavior: expected {expected_semantics[0]} but observed {observed_semantics[0]}."
        if failure_type == "runtime_argument_parsing":
            return f"{target} rejects the exercised runtime invocation instead of accepting the intended arguments."
        if failure_type == "missing_artifact":
            return f"{target} cannot complete because a required runtime artifact is missing or unresolved."
        if failure_type == "import_failure":
            if missing_module:
                return f"{target} fails because the active runtime path cannot import '{missing_module}'."
            return f"{target} fails during import before the requested runtime behavior can execute."
        if failure_type == "test_discovery_gap":
            return "The requested validation command is not reaching the intended tests, so the discovery or package layout must be repaired first."
        if failure_type == "runtime_failure" and observed_semantics:
            return f"{target} still fails on the exercised runtime path: {observed_semantics[0]}"
        summary = str(failed_run.summary or "").strip()
        if summary:
            return summary[:220]
        return f"The current {failed_run.verification_scope} validation still fails for {target}."

    def _python_script_target_from_failure(
        self,
        failed_run: ValidationRunRecord,
        raw_failure_text: str,
    ) -> str | None:
        from_called_process = self._called_process_python_script_target(raw_failure_text)
        if from_called_process:
            return from_called_process
        return self._python_script_target_from_command(failed_run.command)

    def _workspace_relative_path(
        self,
        session: SessionState,
        candidate: str | None,
    ) -> str | None:
        text = str(candidate or "").strip()
        if not text:
            return None
        normalized = text.replace("\\", "/")
        workspace_root = str(session.workspace_root or "").strip()
        if not workspace_root:
            return normalized
        try:
            resolved_candidate = Path(text).resolve()
            resolved_workspace = Path(workspace_root).resolve()
            relative = resolved_candidate.relative_to(resolved_workspace)
        except (ValueError, OSError):
            return normalized
        return relative.as_posix()

    def _existing_workspace_candidate_paths(
        self,
        session: SessionState,
        values: list[str],
    ) -> list[str]:
        workspace_root = Path(session.workspace_root).resolve()
        existing: list[str] = []
        for raw in self._unique_paths(values):
            normalized = str(raw or "").strip().replace("\\", "/").removeprefix("./")
            if not normalized:
                continue
            candidate = (workspace_root / normalized).resolve()
            try:
                candidate.relative_to(workspace_root)
            except ValueError:
                continue
            if candidate.exists() and candidate.is_file():
                existing.append(normalized)
        return existing

    def _declared_workspace_candidate_paths(
        self,
        session: SessionState,
        values: list[str],
    ) -> list[str]:
        workspace_root = Path(session.workspace_root).resolve()
        declared: list[str] = []
        for raw in self._unique_paths(values):
            normalized = str(raw or "").strip().replace("\\", "/").removeprefix("./")
            if not normalized:
                continue
            candidate = (workspace_root / normalized).resolve()
            try:
                candidate.relative_to(workspace_root)
            except ValueError:
                continue
            declared.append(normalized)
        return declared

    def _python_script_target_from_command(self, command: str) -> str | None:
        try:
            tokens = shlex.split(str(command or "").strip())
        except ValueError:
            return None
        if len(tokens) < 2:
            return None
        lowered = [token.lower() for token in tokens]
        if lowered[0] not in {"python", "python3"}:
            return None
        for candidate in tokens[1:]:
            cleaned = str(candidate or "").strip()
            if not cleaned:
                continue
            if cleaned in {"-m", "-c"}:
                return None
            if cleaned.startswith("-"):
                continue
            if cleaned.lower().endswith(".py"):
                return cleaned
            break
        return None

    def _called_process_python_script_target(self, text: str) -> str | None:
        command_text, _exit_code = self._called_process_error_details(text)
        if not command_text:
            return None
        try:
            parsed = ast.literal_eval(command_text)
        except (SyntaxError, ValueError):
            return None
        if not isinstance(parsed, (list, tuple)):
            return None
        tokens = [str(item or "").strip() for item in parsed if str(item or "").strip()]
        if len(tokens) < 2:
            return None
        for candidate in tokens[1:]:
            if candidate in {"-m", "-c"}:
                return None
            if candidate.startswith("-"):
                continue
            if candidate.lower().endswith(".py"):
                return candidate
            break
        return None

    def same_failure_without_behavior_change_count(
        self,
        session: SessionState,
        repair_brief: RepairBrief | None,
    ) -> int:
        if repair_brief is None:
            return 0
        failure_signature = str(repair_brief.failure_signature or "").strip()
        if not failure_signature:
            return 0
        count = 0
        for attempt in reversed(session.repair_history):
            if str(attempt.failure_signature or "").strip() != failure_signature:
                continue
            if attempt.result == "no_effective_change":
                count += 1
            elif (
                attempt.result == "mutation_planned"
                and attempt.independent_verification is False
                and attempt.behavior_changed is False
            ):
                count += 1
        return count

    def should_require_bootstrap_reset(
        self,
        session: SessionState,
        repair_brief: RepairBrief | None,
    ) -> bool:
        if repair_brief is None:
            return False
        if str(repair_brief.bootstrap_status or "").strip() != "bootstrap_failed":
            return False
        return self.same_failure_without_behavior_change_count(session, repair_brief) >= 2

    def _locked_repair_target(
        self,
        session: SessionState,
        *,
        primary_target: str | None,
        candidate_paths: list[str],
        failure_signature: str,
        verification_scope: str,
    ) -> str | None:
        allowed_candidates = {
            path
            for path in self._unique_paths(candidate_paths)
            if path and not self._is_test_path(path) and not self._is_documentation_path(path)
        }
        if verification_scope == "runtime":
            for attempt in reversed(session.repair_history):
                candidate = str(attempt.artifact_path or "").strip()
                if not candidate or candidate not in allowed_candidates:
                    continue
                if str(attempt.failure_signature or "").strip() != failure_signature:
                    continue
                # Only preserve a prior lock when this signature already produced a
                # substantive mutation. No-op retries should not overwrite the
                # stronger current primary target inferred from fresh evidence.
                if attempt.result != "mutation_planned":
                    continue
                return candidate
        return str(primary_target or "").strip() or None

    def _repair_allowed_files(
        self,
        *,
        primary_target: str | None,
        locked_target: str | None,
        candidate_paths: list[str],
        failure_type: str,
        bootstrap_status: str,
    ) -> list[str]:
        candidates = self._unique_paths(candidate_paths)
        preferred = [str(locked_target or "").strip(), str(primary_target or "").strip()]
        if failure_type == "test_discovery_gap" or bootstrap_status == "bootstrap_failed":
            support = [
                path
                for path in candidates
                if path and not self._is_documentation_path(path)
            ]
            return self._unique_paths([*preferred, *support])[:4]
        implementation = [
            path
            for path in candidates
            if path and not self._is_test_path(path) and not self._is_documentation_path(path)
        ]
        if implementation:
            return self._unique_paths([*preferred, *implementation])[:4]
        support = [
            path
            for path in candidates
            if path and path not in implementation and not self._is_documentation_path(path)
        ]
        return self._unique_paths([*preferred, *support])[:4]

    def _repair_forbidden_files(
        self,
        *,
        allowed_files: list[str],
        candidate_paths: list[str],
        failure_type: str,
    ) -> list[str]:
        if failure_type in {"missing_artifact", "test_discovery_gap"}:
            return []
        allowed = set(allowed_files)
        if not any(path and not self._is_test_path(path) and not self._is_documentation_path(path) for path in allowed):
            return []
        forbidden: list[str] = []
        for path in self._unique_paths(candidate_paths):
            if not path or path in allowed:
                continue
            if self._is_test_path(path) or self._is_documentation_path(path):
                forbidden.append(path)
        return forbidden[:4]

    def _recent_failed_attempt_summaries(
        self,
        session: SessionState,
        *,
        failure_signature: str,
    ) -> list[RepairAttemptSummary]:
        summaries: list[RepairAttemptSummary] = []
        for attempt in reversed(session.repair_history):
            if str(attempt.failure_signature or "").strip() != failure_signature:
                continue
            if attempt.result == "mutation_planned":
                continue
            summaries.append(
                RepairAttemptSummary(
                    target=str(attempt.artifact_path or "").strip() or None,
                    strategy=attempt.strategy,
                    result=attempt.result,
                    reason=str(attempt.reason or "").strip() or None,
                )
            )
            if len(summaries) >= 3:
                break
        return summaries

    def _is_documentation_path(self, path: str) -> bool:
        suffix = Path(str(path or "").strip()).suffix.lower()
        return suffix in {".md", ".markdown", ".rst", ".txt"}

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
            for path in self._workspace_paths_from_text(
                workspace_root,
                text,
                require_existing_bare_refs=True,
            ):
                if path not in referenced_paths:
                    referenced_paths.append(path)
        return referenced_paths

    def _failure_scoped_workspace_paths(
        self,
        session: SessionState,
        failed_run: ValidationRunRecord,
        diagnostics: list[DiagnosticRecord],
    ) -> list[str]:
        workspace_root = Path(session.workspace_root).resolve()
        failure_paths: list[str] = []
        fallback_paths: list[str] = []
        texts = [
            str(failed_run.excerpt or "").strip(),
            str(failed_run.summary or "").strip(),
        ]
        for diagnostic in diagnostics:
            texts.append(str(diagnostic.excerpt or "").strip())
            texts.append(str(diagnostic.summary or "").strip())
        for text in texts:
            if not text:
                continue
            for line in self._evidence_lines_from_text(text):
                line_paths = self._workspace_paths_from_text(
                    workspace_root,
                    line,
                    require_existing_bare_refs=True,
                )
                if not line_paths:
                    continue
                target = (
                    failure_paths
                    if self._line_contains_failure_evidence(line)
                    else fallback_paths
                )
                for path in line_paths:
                    if path not in target:
                        target.append(path)
        return failure_paths or fallback_paths

    def _prioritize_failure_scoped_artifact_paths(
        self,
        artifact_paths: list[str],
        failure_scoped_paths: list[str],
    ) -> list[str]:
        if not failure_scoped_paths:
            return artifact_paths
        ordered: list[str] = []
        for path in [*failure_scoped_paths, *artifact_paths]:
            normalized = str(path or "").strip()
            if normalized and normalized not in ordered:
                ordered.append(normalized)
        return ordered or artifact_paths

    def _workspace_paths_from_text(
        self,
        workspace_root: Path,
        text: str,
        *,
        require_existing_bare_refs: bool,
    ) -> list[str]:
        referenced_paths: list[str] = []

        def add_candidate(raw_path: str, *, require_exists: bool) -> None:
            normalized = str(raw_path or "").strip().replace("\\", "/").removeprefix("./")
            if not normalized:
                return
            if normalized.startswith("/"):
                candidate = Path(normalized).expanduser().resolve()
            else:
                if normalized.startswith("../") or "/../" in f"/{normalized}":
                    return
                candidate = (workspace_root / normalized).resolve()
            if require_exists and not candidate.exists():
                return
            try:
                relative = candidate.relative_to(workspace_root).as_posix()
            except ValueError:
                return
            if relative and relative not in referenced_paths:
                referenced_paths.append(relative)

        for match in self.WORKSPACE_REFERENCE_PATTERN.finditer(text):
            add_candidate(str(match.group("path") or ""), require_exists=False)
        for match in self.BARE_WORKSPACE_REFERENCE_PATTERN.finditer(text):
            add_candidate(
                str(match.group("path") or ""),
                require_exists=require_existing_bare_refs,
            )
        for raw_token in str(text or "").split():
            token = str(raw_token or "").strip().strip("\"'`,;:()[]{}")
            if not token.startswith("/") or not Path(token).suffix:
                continue
            add_candidate(token, require_exists=False)
        return referenced_paths

    def _evidence_lines_from_text(self, text: str) -> list[str]:
        lines = [str(line or "").strip() for line in str(text or "").splitlines()]
        return [line for line in lines if line]

    def _text_contains_failure_evidence(self, text: str) -> bool:
        for line in self._evidence_lines_from_text(text):
            if self._line_contains_failure_evidence(line):
                return True
        return False

    def _line_contains_failure_evidence(self, line: str) -> bool:
        lowered = str(line or "").strip().lower()
        if not lowered:
            return False
        if self.MISSING_REF_PATTERN.search(line):
            return True
        if any(marker in lowered for marker in self.FAILURE_EVIDENCE_LINE_MARKERS):
            return True
        if self.EXCEPTION_EVIDENCE_PATTERN.search(line):
            return True
        return not any(marker in lowered for marker in self.NON_FAILURE_SUMMARY_LINE_MARKERS) and " fail" in f" {lowered}"

    def _adjacent_assertion_diff_lines(
        self,
        lines: list[str],
        *,
        start_index: int,
        limit: int = 4,
    ) -> list[str]:
        collected: list[str] = []
        for raw in lines[start_index + 1 :]:
            stripped = str(raw or "").strip()
            if not stripped:
                break
            if stripped.startswith(("Traceback", "File ")):
                break
            if re.fullmatch(r"[=\-]{5,}", stripped):
                break
            if re.search(r"\b(?:FAIL|FAILED|ERROR)\b", stripped):
                break
            if stripped.startswith(("-", "+", "?")):
                collected.append(stripped)
                if len(collected) >= limit:
                    break
                continue
            if collected:
                break
            break
        return collected

    def _traceback_workspace_frames(
        self,
        session: SessionState,
        failed_run: ValidationRunRecord,
    ) -> list[tuple[str, int]]:
        workspace_root = Path(session.workspace_root).resolve()
        frames: list[tuple[str, int]] = []
        seen_frames: set[tuple[str, int]] = set()
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
                try:
                    line = int(match.group("line") or 0)
                except ValueError:
                    line = 0
                if not relative:
                    continue
                frame = (relative, line if line > 0 else 0)
                if frame in seen_frames:
                    continue
                seen_frames.add(frame)
                frames.append(frame)
        return frames

    def _traceback_workspace_hints(
        self,
        session: SessionState,
        failed_run: ValidationRunRecord,
        *,
        traceback_frames: list[tuple[str, int]] | None = None,
        prefer_test_runtime_target: bool = False,
    ) -> tuple[list[str], list[int]]:
        frames = traceback_frames if traceback_frames is not None else self._traceback_workspace_frames(session, failed_run)
        paths: list[str] = []
        lines: list[int] = []
        for relative, line in frames:
            if relative and relative not in paths:
                paths.append(relative)
            if line > 0 and line not in lines:
                lines.append(line)
        if prefer_test_runtime_target:
            test_paths = [path for path in paths if self._is_test_path(path)]
            implementation_paths = [path for path in paths if not self._is_test_path(path)]
            return [*test_paths, *implementation_paths], lines
        implementation_paths = [path for path in paths if not self._is_test_path(path)]
        test_paths = [path for path in paths if self._is_test_path(path)]
        return [*implementation_paths, *test_paths], lines

    def _test_harness_runtime_failure(
        self,
        session: SessionState,
        failed_run: ValidationRunRecord,
        *,
        traceback_frames: list[tuple[str, int]] | None = None,
    ) -> bool:
        if failed_run.verification_scope != "runtime" or self._no_tests_executed(failed_run):
            return False
        frames = traceback_frames if traceback_frames is not None else self._traceback_workspace_frames(session, failed_run)
        if not frames:
            return False
        innermost_workspace_frame = next((path for path, _line in reversed(frames) if path), "")
        if not innermost_workspace_frame or not self._is_test_path(innermost_workspace_frame):
            return False
        failure_text = "\n".join(
            part
            for part in [
                str(failed_run.excerpt or "").strip(),
                str(failed_run.summary or "").strip(),
            ]
            if part
        ).lower()
        if self._unittest_import_wrapper_non_test_target(
            failed_run,
            candidate_paths=self._referenced_workspace_paths(session, failed_run),
        ):
            return False
        return any(marker in failure_text for marker in self.TEST_HARNESS_RUNTIME_ERROR_MARKERS)

    def _unittest_import_wrapper_non_test_target(
        self,
        failed_run: ValidationRunRecord,
        *,
        candidate_paths: list[str],
    ) -> str | None:
        if failed_run.verification_scope != "runtime":
            return None
        failure_text = "\n".join(
            part
            for part in [
                str(failed_run.excerpt or "").strip(),
                str(failed_run.summary or "").strip(),
            ]
            if part
        ).lower()
        if not failure_text:
            return None
        if "failed to import test module" not in failure_text and "loader._failed_test" not in failure_text:
            return None
        for candidate in self._unique_paths(candidate_paths):
            if not candidate:
                continue
            if self._is_test_path(candidate) or self._is_documentation_path(candidate):
                continue
            return candidate
        return None

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
            if snapshot.file_count <= 40 and shutil.which("node"):
                commands.append(
                    ValidationCommand(
                        command=f"internal:web_runtime_smoke:{json.dumps(html_targets)}",
                        kind="test",
                        verification_scope="runtime",
                        source="default",
                        priority=22,
                        reason="Execute a bounded browser-like runtime smoke harness for the changed standalone web artifact.",
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
        commands: list[ValidationCommand],
        session: SessionState | None,
    ) -> list[ValidationCommand]:
        if any(self.command_scope(command) == "runtime" for command in commands):
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
        has_web_runtime_smoke = any(
            command.command.startswith("internal:web_runtime_smoke:")
            for command in commands
        )
        return [
            command.model_copy(update={"required": False})
            if (
                command.source == "default"
                and command.verification_scope in {"syntax", "static", "structural"}
                and not (
                    has_web_runtime_smoke
                    and command.command.startswith("internal:web_artifact:")
                )
            )
            else command
            for command in commands
        ]

    def _replace_generic_unittest_commands(
        self,
        commands: list[ValidationCommand],
        *,
        snapshot: WorkspaceSnapshot,
        changed_files: list[str],
        session: SessionState | None,
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
            session=session,
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
        session: SessionState | None,
    ) -> list[str]:
        candidate_paths = [
            path
            for path in changed_files
            if self._is_test_path(path) and Path(path).suffix.lower() == ".py"
        ]
        if not candidate_paths and session is not None:
            candidate_paths = [
                path
                for command in self._explicit_validation_commands(session)
                for path in self._paths_from_explicit_test_command(command.command)
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

    def _interactive_web_request(self, task: str) -> bool:
        token_space = self._normalized_keyword_space(task)
        markers = (
            " game ",
            " spiel ",
            " interaktiv ",
            " interactive ",
            " keyboard ",
            " tastatur ",
            " keydown ",
            " click ",
            " klick ",
            " drag ",
            " score ",
            " punktestand ",
            " restart ",
            " neustart ",
            " form ",
            " dialog ",
            " modal ",
        )
        return any(marker in token_space for marker in markers)

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
        current_failure_has_specific_evidence = self._text_contains_failure_evidence(
            str(failed_run.excerpt or "").strip()
        )
        for diagnostic in reversed(session.diagnostics):
            diagnostic_command = self.command_identity(diagnostic.command or "")
            if diagnostic_command and diagnostic_command == normalized_command:
                related.append(diagnostic)
                continue
            if artifact_set and artifact_set.intersection(diagnostic.file_hints):
                related.append(diagnostic)
        if not related:
            return []

        current_iteration = int(failed_run.iteration or 0)
        if current_iteration > 0:
            same_iteration = [
                diagnostic
                for diagnostic in related
                if int(getattr(diagnostic, "iteration", 0) or 0) == current_iteration
            ]
            if same_iteration:
                related = same_iteration
            elif current_failure_has_specific_evidence:
                return []
            else:
                older_relevant = [
                    diagnostic
                    for diagnostic in related
                    if 0 < int(getattr(diagnostic, "iteration", 0) or 0) <= current_iteration
                ]
                if older_relevant:
                    latest_iteration = max(int(diagnostic.iteration or 0) for diagnostic in older_relevant)
                    related = [
                        diagnostic
                        for diagnostic in older_relevant
                        if int(diagnostic.iteration or 0) == latest_iteration
                    ]
        else:
            iterated = [
                diagnostic
                for diagnostic in related
                if int(getattr(diagnostic, "iteration", 0) or 0) > 0
            ]
            if iterated:
                latest_iteration = max(int(diagnostic.iteration or 0) for diagnostic in iterated)
                related = [
                    diagnostic
                    for diagnostic in iterated
                    if int(diagnostic.iteration or 0) == latest_iteration
                ]

        return list(reversed(related[-4:]))

    def _artifact_paths_for_failed_run(
        self,
        session: SessionState,
        failed_run: ValidationRunRecord,
    ) -> list[str]:
        paths: list[str] = []
        task_state = session.task_state
        snapshot = session.workspace_snapshot
        no_tests_executed = self._no_tests_executed(failed_run)
        if failed_run.verification_scope == "runtime" and task_state is not None and not no_tests_executed:
            task_paths = [
                artifact.path
                for artifact in task_state.target_artifacts
                if artifact.path and artifact.role != "supporting_context"
            ]
            existing_task_paths = self._existing_workspace_candidate_paths(session, task_paths)
            paths.extend(existing_task_paths or self._declared_workspace_candidate_paths(session, task_paths))
            if snapshot is not None:
                paths.extend(
                    self._existing_workspace_candidate_paths(
                        session,
                        snapshot.entrypoints[:4],
                    )
                )
        if no_tests_executed:
            if task_state is not None:
                test_paths = [
                    artifact.path
                    for artifact in task_state.target_artifacts
                    if artifact.path
                    and (
                        artifact.role == "validation_target"
                        or artifact.kind == "test"
                    )
                ]
                existing_test_paths = self._existing_workspace_candidate_paths(session, test_paths)
                paths.extend(existing_test_paths or self._declared_workspace_candidate_paths(session, test_paths))
            paths.extend(
                item.path
                for item in session.changed_files
                if self._is_test_path(item.path)
            )
            paths.extend(self._missing_unittest_package_inits(session, failed_run, paths))
        paths.extend(self._paths_from_validation_command(failed_run.command))
        fresh_paths = self._unique_paths(paths)
        if not fresh_paths and session.active_repair_context is not None:
            paths.extend(session.active_repair_context.artifact_paths)
        paths.extend(item.path for item in session.changed_files)
        return self._unique_paths(paths)

    def _prioritize_runtime_artifact_paths(
        self,
        artifact_paths: list[str],
        failed_run: ValidationRunRecord,
        *,
        prefer_test_runtime_target: bool = False,
        prefer_content_runtime_target: bool = False,
    ) -> list[str]:
        if failed_run.verification_scope != "runtime" or self._no_tests_executed(failed_run):
            return artifact_paths
        content_paths = [
            path
            for path in artifact_paths
            if path and not self._is_test_path(path)
        ]
        implementation_paths = [
            path
            for path in artifact_paths
            if path and not self._is_test_path(path) and not self._is_documentation_path(path)
        ]
        documentation_paths = [
            path
            for path in artifact_paths
            if path and self._is_documentation_path(path)
        ]
        validation_paths = [
            path
            for path in artifact_paths
            if path and self._is_test_path(path)
        ]
        if prefer_test_runtime_target:
            ordered = [*validation_paths, *implementation_paths, *documentation_paths]
            return ordered or artifact_paths
        if prefer_content_runtime_target:
            ordered = [*content_paths, *validation_paths]
            return ordered or artifact_paths
        ordered = [*implementation_paths, *documentation_paths, *validation_paths]
        return ordered or artifact_paths

    def _symbol_resolved_workspace_paths(
        self,
        session: SessionState,
        implicated_symbols: list[str],
    ) -> list[str]:
        snapshot = session.workspace_snapshot
        if snapshot is None:
            return []
        symbol_index = getattr(snapshot, "symbol_index", {}) or {}
        if not symbol_index:
            return []
        wanted = {
            str(symbol or "").strip().lower()
            for symbol in implicated_symbols
            if str(symbol or "").strip()
            and not self._is_generic_runtime_failure_symbol(str(symbol or "").strip())
        }
        if not wanted:
            return []
        hotspots = {str(path or "").strip() for path in getattr(snapshot, "import_hotspots", []) if str(path or "").strip()}
        services = {str(path or "").strip() for path in getattr(snapshot, "service_files", []) if str(path or "").strip()}
        entrypoints = {str(path or "").strip() for path in getattr(snapshot, "entrypoints", []) if str(path or "").strip()}
        candidates: list[str] = []
        for path, symbols in symbol_index.items():
            normalized_path = str(path or "").strip()
            if not normalized_path:
                continue
            lowered_symbols = {
                str(symbol or "").strip().lower()
                for symbol in list(symbols or [])
                if str(symbol or "").strip()
            }
            if lowered_symbols.intersection(wanted):
                candidates.append(normalized_path)
        return sorted(
            self._existing_workspace_candidate_paths(session, candidates),
            key=lambda path: (
                1 if self._is_test_path(path) else 0,
                1 if self._is_documentation_path(path) else 0,
                0 if path in hotspots else 1,
                0 if path in services else 1,
                0 if path in entrypoints else 1,
                path,
            ),
        )[:6]

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
        if lowered[0] in {"python", "python3"} and len(tokens) >= 2 and lowered[1] != "-m":
            candidate = str(tokens[1] or "").strip().replace("\\", "/").removeprefix("./")
            return [candidate] if candidate.endswith(".py") else []
        if len(tokens) >= 3 and lowered[:3] in (["python", "-m", "unittest"], ["python3", "-m", "unittest"]):
            return self._unittest_targets_to_paths(tokens[3:])
        if len(tokens) >= 3 and lowered[:3] in (["python", "-m", "pytest"], ["python3", "-m", "pytest"]):
            return self._pytest_targets_to_paths(tokens[3:])
        if lowered[0] == "pytest":
            return self._pytest_targets_to_paths(tokens[1:])
        return []

    def _unittest_targets_to_paths(self, targets: list[str]) -> list[str]:
        normalized_targets = [str(target or "").strip() for target in targets if str(target or "").strip()]
        if normalized_targets and normalized_targets[0].lower() == "discover":
            start_directory: str | None = None
            for index, token in enumerate(normalized_targets[1:], start=1):
                lowered = token.lower()
                if lowered in {"-s", "--start-directory"} and index + 1 < len(normalized_targets):
                    start_directory = normalized_targets[index + 1]
                    break
            if not start_directory:
                return []
            cleaned_start = str(start_directory or "").strip().replace("\\", "/").removeprefix("./")
            if not cleaned_start or cleaned_start.lower() in {"tests", "test"}:
                return []
            if cleaned_start.endswith(".py"):
                return [cleaned_start]
            return [f"{cleaned_start}.py"]

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
        if kind not in {"web_artifact", "web_runtime_smoke"}:
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
        focused_lines: list[str] = []
        for text in [
            str(failed_run.excerpt or "").strip(),
            *(str(item.excerpt or "").strip() for item in diagnostics),
            *(str(item.summary or "").strip() for item in diagnostics),
        ]:
            lines = self._evidence_lines_from_text(text)
            for index, line in enumerate(lines):
                if not self._line_contains_failure_evidence(line):
                    continue
                if line not in focused_lines:
                    focused_lines.append(line)
                if "assertionerror" in line.lower():
                    for diff_line in self._adjacent_assertion_diff_lines(lines, start_index=index):
                        if diff_line not in focused_lines:
                            focused_lines.append(diff_line)
        if focused_lines:
            return "\n".join(self._preserve_assertion_context_at_excerpt_boundary(focused_lines, max_lines=8))
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
            return self._trim_failure_summary_text(excerpt, max_chars=240)
        summary = str(failed_run.summary or "").strip()
        if summary:
            return summary[:240]
        return "Validation failed without a readable summary."

    def _preserve_assertion_context_at_excerpt_boundary(
        self,
        lines: list[str],
        *,
        max_lines: int,
    ) -> list[str]:
        clipped = [str(line or "").rstrip() for line in lines[:max_lines]]
        while len(clipped) < len(lines):
            last_line = str(clipped[-1] or "").strip() if clipped else ""
            next_line = str(lines[len(clipped)] or "").strip()
            if not last_line or not next_line:
                break
            if last_line.startswith("AssertionError:") and next_line.startswith(("-", "+", "?")):
                clipped.append(str(lines[len(clipped)] or "").rstrip())
                continue
            if last_line.startswith(("-", "+", "?")) and next_line.startswith(("+", "?")):
                clipped.append(str(lines[len(clipped)] or "").rstrip())
                continue
            break
        return clipped

    def _trim_failure_summary_text(self, text: str, *, max_chars: int) -> str:
        lines = [str(line or "").rstrip() for line in str(text or "").splitlines() if str(line or "").strip()]
        if not lines:
            return ""
        selected: list[str] = []
        used = 0
        for line in lines:
            projected = used + len(line) + (1 if selected else 0)
            if selected and projected > max_chars:
                break
            if not selected and len(line) > max_chars:
                return line[:max_chars]
            selected.append(line)
            used = projected
        if selected:
            return "\n".join(selected)
        return str(text or "").strip()[:max_chars]

    def _repair_requirements(
        self,
        failed_run: ValidationRunRecord,
        diagnostics: list[DiagnosticRecord],
        *,
        artifact_paths: list[str],
        expected_features: list[str],
        missing_features: list[str],
        missing_unittest_package_inits: list[str] | None = None,
        primary_target_override: str | None = None,
    ) -> list[str]:
        requirements: list[str] = []
        scope = failed_run.verification_scope
        primary_target = str(primary_target_override or "").strip() or (
            artifact_paths[0] if artifact_paths else "the implicated artifact"
        )
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
                undefined_symbol = self._undefined_runtime_symbol_name(
                    "\n".join(
                        part
                        for part in [
                            str(failed_run.excerpt or "").strip(),
                            str(failed_run.summary or "").strip(),
                        ]
                        if part
                    )
                )
                if undefined_symbol:
                    requirements.append(
                        f"Bind or import '{undefined_symbol}' before its failing use in {primary_target}, or remove that use if it is unnecessary."
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

    def _unique_semantic_pairs(
        self,
        values: list[tuple[str | None, str | None]],
    ) -> list[tuple[str | None, str | None]]:
        unique: list[tuple[str | None, str | None]] = []
        for observed, expected in values:
            pair = (
                str(observed).strip() if observed is not None else None,
                str(expected).strip() if expected is not None else None,
            )
            if pair == (None, None) or pair in unique:
                continue
            unique.append(pair)
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
