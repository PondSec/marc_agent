from __future__ import annotations

import subprocess

from config.settings import AppConfig
from llm.schemas import RunShellArgs
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
        assessment = self.safety.assess_shell_command(args.command)
        if not assessment.allowed:
            return {
                "success": False,
                "message": "; ".join(assessment.reasons),
                "risk_level": assessment.risk_level,
                "blocked": True,
            }

        cwd = self.workspace.resolve_directory(args.cwd)
        if not cwd.exists():
            return {
                "success": False,
                "message": f"Working directory does not exist: {cwd}",
                "risk_level": assessment.risk_level,
                "blocked": True,
            }
        if self.config.dry_run:
            return {
                "success": True,
                "message": f"Dry run: would execute '{args.command}' in {cwd}",
                "risk_level": assessment.risk_level,
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
                "command": args.command,
            }

        timeout = args.timeout or self.config.shell_timeout
        try:
            completed = subprocess.run(
                ["/bin/bash", "-lc", args.command],
                cwd=cwd,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "message": f"Command timed out after {timeout} seconds.",
                "risk_level": assessment.risk_level,
                "timeout": True,
                "command": args.command,
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
            "command": args.command,
        }
