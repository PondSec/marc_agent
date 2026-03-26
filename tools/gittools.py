from __future__ import annotations

import subprocess

from config.settings import AccessMode, AppConfig
from llm.schemas import EmptyArgs, GitCreateBranchArgs, GitDiffArgs, GitLogArgs
from runtime.workspace import WorkspaceManager


class GitTools:
    def __init__(self, config: AppConfig, workspace: WorkspaceManager):
        self.config = config
        self.workspace = workspace

    def git_status(self, args: EmptyArgs) -> dict:
        return self._run_git(["status", "--short"])

    def git_diff(self, args: GitDiffArgs) -> dict:
        command = ["diff", f"--unified={args.context_lines}"]
        if args.cached:
            command.append("--cached")
        if args.path:
            command.extend(["--", args.path])
        return self._run_git(command)

    def git_log(self, args: GitLogArgs) -> dict:
        return self._run_git(
            ["log", f"-n{args.limit}", "--oneline", "--decorate", "--graph"]
        )

    def git_create_branch(self, args: GitCreateBranchArgs) -> dict:
        if self.config.access_mode != AccessMode.FULL.value:
            return {
                "success": False,
                "message": "Branch creation requires full access mode.",
                "risk_level": "medium",
                "blocked": True,
            }
        if self.config.dry_run:
            return {
                "success": True,
                "message": f"Dry run: would create branch {args.name}.",
                "risk_level": "medium",
            }
        return self._run_git(["checkout", "-b", args.name], risk_level="medium")

    def _run_git(self, args: list[str], risk_level: str | None = None) -> dict:
        if not self._is_repo():
            return {"success": False, "message": "Workspace is not a git repository."}
        completed = subprocess.run(
            ["git", *args],
            cwd=self.workspace.root,
            text=True,
            capture_output=True,
            check=False,
        )
        return {
            "success": completed.returncode == 0,
            "message": completed.stdout.strip()
            or completed.stderr.strip()
            or "git command finished",
            "stdout": completed.stdout[-self.config.max_read_chars :],
            "stderr": completed.stderr[-self.config.max_read_chars :],
            "exit_code": completed.returncode,
            "risk_level": risk_level,
        }

    def _is_repo(self) -> bool:
        completed = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=self.workspace.root,
            text=True,
            capture_output=True,
            check=False,
        )
        return completed.returncode == 0
