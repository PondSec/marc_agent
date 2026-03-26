from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from config.settings import AppConfig
from runtime.workspace import WorkspaceManager


class CommandAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str
    risk_level: str
    allowed: bool
    reasons: list[str]


class SafetyError(RuntimeError):
    pass


class SafetyManager:
    HARD_BLOCK_PATTERNS = [
        re.compile(r"\bsudo\b"),
        re.compile(r"\bshutdown\b"),
        re.compile(r"\breboot\b"),
        re.compile(r"\bmkfs\b"),
        re.compile(r"\bdd\s+if="),
        re.compile(r"\brm\s+-rf\s+/"),
        re.compile(r"\bgit\s+reset\s+--hard\b"),
        re.compile(r"\bgit\s+clean\s+-fdx?\b"),
        re.compile(r"\bgit\s+push\b.*--force"),
    ]
    DANGEROUS_PATTERNS = [
        re.compile(r"\brm\s+-rf\b"),
        re.compile(r"\bchmod\s+-R\b"),
        re.compile(r"\bmv\b.+\s+/"),
        re.compile(r">\s*/"),
        re.compile(r"\bkillall\b"),
    ]
    NETWORK_PATTERNS = [
        re.compile(r"\bcurl\b"),
        re.compile(r"\bwget\b"),
        re.compile(r"\bnc\b"),
        re.compile(r"\bnmap\b"),
        re.compile(r"\bssh\b"),
        re.compile(r"\bscp\b"),
        re.compile(r"\brsync\b.+:"),
    ]
    MUTATING_PATTERNS = [
        re.compile(r"\bpytest\b"),
        re.compile(r"\bnpm\s+(test|run)\b"),
        re.compile(r"\bpython\b"),
        re.compile(r"\bmake\b"),
        re.compile(r"\bgit\s+checkout\s+-b\b"),
    ]

    def __init__(self, config: AppConfig, workspace: WorkspaceManager):
        self.config = config
        self.workspace = workspace

    def ensure_read_allowed(self, path: str | Path) -> Path:
        return self.workspace.resolve_path(path)

    def ensure_write_allowed(self, path: str | Path) -> Path:
        target = self.workspace.resolve_path(path)
        if self.config.read_only:
            raise SafetyError("Write blocked because read-only mode is enabled.")
        return target

    def assess_shell_command(self, command: str) -> CommandAssessment:
        reasons: list[str] = []
        risk = "low"
        allowed = True

        if any(pattern.search(command) for pattern in self.HARD_BLOCK_PATTERNS):
            return CommandAssessment(
                command=command,
                risk_level="blocked",
                allowed=False,
                reasons=["Command matches a hard safety block rule."],
            )

        if not self.config.allow_network and any(
            pattern.search(command) for pattern in self.NETWORK_PATTERNS
        ):
            return CommandAssessment(
                command=command,
                risk_level="blocked",
                allowed=False,
                reasons=["Network access is disabled by configuration."],
            )

        if any(pattern.search(command) for pattern in self.DANGEROUS_PATTERNS):
            risk = "high"
            reasons.append("Command matches a dangerous command rule.")

        if command.strip().startswith("git checkout -b"):
            risk = "medium"
            reasons.append("Branch creation mutates repository state.")

        if self.config.read_only and any(
            token in command for token in ("git checkout -b", "touch ", "mkdir ", "tee ", "sed -i", ">>", ">")
        ):
            return CommandAssessment(
                command=command,
                risk_level="blocked",
                allowed=False,
                reasons=["Read-only mode blocks mutating shell commands."],
            )

        if self.config.approval_mode and risk in {"medium", "high"}:
            allowed = False
            reasons.append("Approval mode requires confirmation for this command.")

        if risk == "high" and not self.config.allow_dangerous_commands:
            allowed = False
            reasons.append("Dangerous commands are disabled by configuration.")

        return CommandAssessment(
            command=command,
            risk_level=risk,
            allowed=allowed,
            reasons=reasons or ["Command accepted."],
        )
