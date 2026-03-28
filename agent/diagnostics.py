from __future__ import annotations

import re
from pathlib import Path

from agent.models import DiagnosticRecord, ToolRunResult
from runtime.workspace import WorkspaceManager


PATH_LINE_PATTERNS = [
    re.compile(r'File "(?P<path>[^"]+)", line (?P<line>\d+)'),
    re.compile(r"(?P<path>[\w./\\-]+\.\w+):(?P<line>\d+)(?::\d+)?"),
]


class FailureAnalyzer:
    def __init__(self, workspace: WorkspaceManager, *, max_excerpt: int = 2_000):
        self.workspace = workspace
        self.max_excerpt = max_excerpt

    def analyze(self, result: ToolRunResult, *, iteration: int) -> list[DiagnosticRecord]:
        if result.success and not result.data.get("blocked") and not result.data.get("timeout"):
            return []

        command = str(result.data.get("command") or "")
        text = self._combined_output(result)
        summary = self._summarize(result.message, text)
        file_hints, line_hints = self._extract_file_hints(text)
        category, severity = self._categorize(result, command)
        action_hints = self._action_hints(category, command, text=text)

        diagnostic = DiagnosticRecord(
            source=result.tool_name,
            category=category,
            severity=severity,
            summary=summary,
            tool_name=result.tool_name,
            command=command or None,
            exit_code=result.data.get("exit_code"),
            file_hints=file_hints,
            line_hints=line_hints,
            action_hints=action_hints,
            excerpt=text[: self.max_excerpt] if text else None,
            iteration=iteration,
        )
        return [diagnostic]

    def _categorize(self, result: ToolRunResult, command: str) -> tuple[str, str]:
        lowered = f"{result.message}\n{command}".lower()
        if result.data.get("blocked"):
            if "approval" in lowered:
                return ("approval", "warning")
            if "safe mode" in lowered or "requires full access" in lowered:
                return ("access_control", "warning")
            return ("safety", "warning")
        if result.data.get("timeout"):
            return ("timeout", "error")
        if result.tool_name in {"run_tests", "run_shell"}:
            if any(token in command for token in ("pytest", "unittest")):
                return ("test_failure", "error")
            if any(token in command for token in ("ruff", "eslint", "flake8")):
                return ("lint_failure", "error")
            if any(token in command for token in ("mypy", "pyright", "tsc", "typecheck")):
                return ("typecheck_failure", "error")
            if "build" in command:
                return ("build_failure", "error")
            if "deploy" in command or "release" in command:
                return ("release_failure", "error")
            return ("command_failure", "error")
        if "validation" in lowered:
            return ("tool_validation", "warning")
        return ("tool_failure", "error")

    def _combined_output(self, result: ToolRunResult) -> str:
        parts = [
            str(result.data.get("stdout") or ""),
            str(result.data.get("stderr") or ""),
        ]
        text = "\n".join(part for part in parts if part).strip()
        if text:
            return text
        return str(result.message or "").strip()

    def _summarize(self, message: str, text: str) -> str:
        candidates = [message.strip()]
        candidates.extend(line.strip() for line in text.splitlines() if line.strip())
        for candidate in candidates:
            if candidate:
                return candidate[:240]
        return "Command failed without a readable diagnostic."


    def _extract_file_hints(self, text: str) -> tuple[list[str], list[int]]:
        file_hints: list[str] = []
        line_hints: list[int] = []
        seen_files: set[str] = set()

        for pattern in PATH_LINE_PATTERNS:
            for match in pattern.finditer(text):
                raw_path = match.group("path")
                raw_line = match.groupdict().get("line")
                if not raw_path:
                    continue
                display = self._normalize_path(raw_path)
                if not display:
                    continue
                if display not in seen_files:
                    seen_files.add(display)
                    file_hints.append(display)
                if raw_line and raw_line.isdigit():
                    line_hints.append(int(raw_line))
        return file_hints[:10], line_hints[:20]

    def _normalize_path(self, raw_path: str) -> str | None:
        text = str(raw_path or "").strip()
        if not text or (text.startswith("<") and text.endswith(">")):
            return None
        path = Path(text)
        candidate = path if path.is_absolute() else self.workspace.root / path
        resolved = candidate.expanduser().resolve(strict=False)
        if not self.workspace.is_within_root(resolved):
            return None
        return self.workspace.display_path(resolved)

    def _action_hints(self, category: str, command: str, *, text: str = "") -> list[str]:
        hints = {
            "test_failure": [
                "Read the failing test output and inspect the hinted source files before editing.",
                "Prefer a targeted fix over broad refactors, then rerun the same test command.",
            ],
            "lint_failure": [
                "Inspect the reported file and line hints, fix the style or static issue, then rerun lint.",
            ],
            "typecheck_failure": [
                "Check the referenced types, imports, and function signatures before rerunning typechecks.",
            ],
            "build_failure": [
                "Inspect build config, imports, and generated assets that the error references.",
            ],
            "approval": [
                "Choose a lower-risk command or rerun in full access mode if the task truly requires it.",
            ],
            "access_control": [
                "Stay within the configured workspace scope or rerun with a stronger access mode.",
            ],
            "timeout": [
                "Narrow the command scope or raise the timeout if the command is expected to be slow.",
            ],
        }.get(category, [])

        lowered_text = str(text or "").lower()
        if category == "test_failure" and any(
            marker in lowered_text for marker in ("ran 0 tests", "collected 0 items", "no tests ran")
        ):
            hints.append(
                "Inspect test discovery or package layout and ensure the requested command actually executes the intended tests."
            )

        if not hints and command:
            hints = [
                "Inspect the command output for file hints, apply a focused fix, then retry the command.",
            ]
        return hints
