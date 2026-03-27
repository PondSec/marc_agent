from __future__ import annotations

import json
import py_compile
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

from config.settings import AppConfig
from llm.schemas import RunShellArgs, RunTestsArgs
from runtime.workspace import WorkspaceManager
from tools.safety import SafetyManager


class ShellTools:
    def __init__(
        self,
        config: AppConfig,
        workspace: WorkspaceManager,
        safety: SafetyManager,
    ):
        self.config = config
        self.workspace = workspace
        self.safety = safety

    def run_shell(self, args: RunShellArgs) -> dict:
        return self._run_command(args.command, args.cwd, args.timeout)

    def run_tests(self, args: RunTestsArgs) -> dict:
        if args.command.startswith("internal:"):
            return self._run_internal_validation(args.command, args.cwd)
        result = self._run_command(args.command, args.cwd, args.timeout)
        result["message"] = (
            f"Validation command exited with {result['exit_code']}."
            if "exit_code" in result
            else result["message"]
        )
        return result

    def _run_internal_validation(self, command: str, cwd: str) -> dict:
        working_dir = self.workspace.resolve_directory(cwd)
        if not working_dir.exists():
            return {
                "success": False,
                "message": f"Working directory does not exist: {working_dir}",
                "risk_level": "low",
                "blocked": True,
                "command": command,
            }

        kind, _, payload = command.partition(":")
        kind, _, payload = payload.partition(":")
        try:
            relative_paths = json.loads(payload or "[]")
        except json.JSONDecodeError:
            return {
                "success": False,
                "message": "Internal validation payload could not be decoded.",
                "risk_level": "low",
                "command": command,
            }

        if kind == "python_syntax":
            return self._run_python_syntax_validation(command, working_dir, relative_paths)
        if kind == "python_cli_smoke":
            return self._run_python_cli_smoke_validation(command, working_dir, relative_paths)
        if kind == "html_refs":
            return self._run_html_reference_validation(command, working_dir, relative_paths)
        return {
            "success": False,
            "message": f"Unknown internal validation kind: {kind}",
            "risk_level": "low",
            "command": command,
        }

    def _run_command(self, command: str, cwd: str, timeout: int | None) -> dict:
        assessment = self.safety.assess_shell_command(command)
        if not assessment.allowed:
            return {
                "success": False,
                "message": "; ".join(assessment.reasons),
                "risk_level": assessment.risk_level,
                "blocked": True,
                "command": command,
            }

        working_dir = self.workspace.resolve_directory(cwd)
        if not working_dir.exists():
            return {
                "success": False,
                "message": f"Working directory does not exist: {working_dir}",
                "risk_level": assessment.risk_level,
                "blocked": True,
                "command": command,
            }
        if self.config.dry_run:
            return {
                "success": True,
                "message": f"Dry run: would execute '{command}' in {working_dir}",
                "risk_level": assessment.risk_level,
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
                "command": command,
            }

        effective_timeout = timeout or self.config.shell_timeout
        try:
            completed = subprocess.run(
                ["/bin/bash", "-lc", command],
                cwd=working_dir,
                text=True,
                capture_output=True,
                timeout=effective_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "message": f"Command timed out after {effective_timeout} seconds.",
                "risk_level": assessment.risk_level,
                "timeout": True,
                "command": command,
            }

        stdout = completed.stdout[-self.config.max_read_chars :]
        stderr = completed.stderr[-self.config.max_read_chars :]
        return {
            "success": completed.returncode == 0,
            "message": f"Command exited with {completed.returncode}.",
            "risk_level": assessment.risk_level,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": completed.returncode,
            "command": command,
        }

    def _run_python_syntax_validation(self, command: str, working_dir: Path, relative_paths: list[str]) -> dict:
        failures: list[str] = []
        checked = 0
        for relative_path in relative_paths:
            target = self.workspace.resolve_path(working_dir / relative_path)
            if not target.exists():
                failures.append(f"Missing file: {relative_path}")
                continue
            try:
                py_compile.compile(str(target), doraise=True)
                checked += 1
            except py_compile.PyCompileError as exc:
                failures.append(str(exc))

        success = not failures and checked > 0
        return {
            "success": success,
            "message": "Validation command exited with 0." if success else "Validation command exited with 1.",
            "risk_level": "low",
            "stdout": f"Checked {checked} Python file(s).",
            "stderr": "\n".join(failures),
            "exit_code": 0 if success else 1,
            "command": command,
        }

    def _run_python_cli_smoke_validation(self, command: str, working_dir: Path, relative_paths: list[str]) -> dict:
        failures: list[str] = []
        outputs: list[str] = []
        checked = 0
        effective_timeout = min(max(int(self.config.shell_timeout), 1), 8)

        for relative_path in relative_paths[:2]:
            target = self.workspace.resolve_path(working_dir / relative_path)
            if not target.exists():
                failures.append(f"Missing file: {relative_path}")
                continue
            if target.suffix.lower() != ".py":
                failures.append(f"Not a Python file: {relative_path}")
                continue

            try:
                completed = subprocess.run(
                    [sys.executable, str(target)],
                    cwd=working_dir,
                    text=True,
                    capture_output=True,
                    input=self._default_python_smoke_input(target),
                    timeout=effective_timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                stdout = str(exc.stdout or "")[-self.config.max_read_chars :]
                stderr = str(exc.stderr or "")[-self.config.max_read_chars :]
                return {
                    "success": False,
                    "message": f"Validation command timed out after {effective_timeout} seconds.",
                    "risk_level": "low",
                    "stdout": stdout,
                    "stderr": stderr,
                    "timeout": True,
                    "command": command,
                }

            checked += 1
            stdout = completed.stdout[-self.config.max_read_chars :]
            stderr = completed.stderr[-self.config.max_read_chars :]
            outputs.append(f"$ {relative_path}\n{stdout}".strip())
            if completed.returncode != 0:
                failure_text = stderr or stdout or f"Exited with {completed.returncode}."
                failures.append(f"{relative_path}: {failure_text}".strip())

        success = not failures and checked > 0
        return {
            "success": success,
            "message": "Validation command exited with 0." if success else "Validation command exited with 1.",
            "risk_level": "low",
            "stdout": "\n\n".join(part for part in outputs if part),
            "stderr": "\n".join(failures),
            "exit_code": 0 if success else 1,
            "command": command,
        }

    def _run_html_reference_validation(self, command: str, working_dir: Path, relative_paths: list[str]) -> dict:
        parser = _HTMLReferenceParser()
        missing_refs: list[str] = []
        checked = 0
        for relative_path in relative_paths:
            target = self.workspace.resolve_path(working_dir / relative_path)
            if not target.exists():
                missing_refs.append(f"Missing HTML file: {relative_path}")
                continue
            parser.reset_refs()
            parser.feed(target.read_text(encoding="utf-8"))
            checked += 1
            for reference in parser.references:
                resolved = (target.parent / reference).resolve()
                if not resolved.exists():
                    missing_refs.append(f"{relative_path} -> {reference}")

        success = not missing_refs and checked > 0
        return {
            "success": success,
            "message": "Validation command exited with 0." if success else "Validation command exited with 1.",
            "risk_level": "low",
            "stdout": f"Checked {checked} HTML file(s).",
            "stderr": "\n".join(missing_refs),
            "exit_code": 0 if success else 1,
            "command": command,
        }

    def _default_python_smoke_input(self, target: Path) -> str:
        try:
            content = target.read_text(encoding="utf-8")
        except OSError:
            return "\n"

        input_calls = content.count("input(")
        if input_calls <= 0:
            return ""

        seed_values = [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "n",
            "q",
            "quit",
            "exit",
            "0",
            "Alice",
            "Bob",
        ]
        line_count = min(max(input_calls + 4, 9), len(seed_values))
        return "\n".join(seed_values[:line_count]) + "\n"


class _HTMLReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def reset_refs(self) -> None:
        self.references = []
        self.reset()

    def handle_starttag(self, tag: str, attrs) -> None:
        del tag
        for key, value in attrs:
            if key not in {"src", "href"} or not value:
                continue
            cleaned = str(value).split("#", 1)[0].split("?", 1)[0].strip()
            if not cleaned or cleaned.startswith(("http://", "https://", "data:", "mailto:", "javascript:", "#")):
                continue
            self.references.append(cleaned)
