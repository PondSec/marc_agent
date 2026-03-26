from __future__ import annotations

import subprocess

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
        result = self._run_command(args.command, args.cwd, args.timeout)
        result["message"] = (
            f"Validation command exited with {result['exit_code']}."
            if "exit_code" in result
            else result["message"]
        )
        return result

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
