from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from config.settings import AccessMode, AppConfig
from runtime.workspace import WorkspaceManager


class CommandAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str
    risk_level: str
    allowed: bool
    reasons: list[str]


class ToolAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    risk_level: str
    allowed: bool
    reasons: list[str]


class SafetyError(RuntimeError):
    pass


class SafetyManager:
    READ_TOOLS = {
        "inspect_workspace",
        "list_files",
        "search_in_files",
        "read_file",
        "show_diff",
        "git_status",
        "git_diff",
        "git_log",
    }
    WRITE_TOOLS = {
        "write_file",
        "append_file",
        "create_file",
        "delete_file",
        "replace_in_file",
        "patch_file",
        "git_create_branch",
    }
    SHELL_TOOLS = {"run_shell", "run_tests"}
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
    INSPECTION_PATTERNS = [
        re.compile(r"^\s*git\s+(status|diff|log)\b"),
        re.compile(r"^\s*(pwd|ls|find|cat|head|tail)\b"),
        re.compile(r"^\s*(rg|grep|sed\s+-n)\b"),
    ]
    VERIFICATION_PATTERNS = [
        re.compile(r"^internal:(python_syntax|html_refs)\b"),
        re.compile(r"\bpytest\b"),
        re.compile(r"\bpython(?:3)?\s+-m\s+pytest\b"),
        re.compile(r"\bpython(?:3)?\s+-m\s+py_compile\b"),
        re.compile(r"\bpython(?:3)?\s+-m\s+unittest\b"),
        re.compile(r"\buv\s+run\s+pytest\b"),
        re.compile(r"\bruff\b"),
        re.compile(r"\bmypy\b"),
        re.compile(r"\bnode\s+--check\b"),
        re.compile(r"\bnpm\s+(test|run\s+(test|lint|build|typecheck))\b"),
        re.compile(r"\bpnpm\s+(test|lint|build|typecheck)\b"),
        re.compile(r"\byarn\s+(test|lint|build|typecheck)\b"),
        re.compile(r"\bgo\s+test\b"),
        re.compile(r"\bcargo\s+(test|check)\b"),
    ]
    MUTATING_PATTERNS = [
        re.compile(r"\bgit\s+(checkout\s+-b|switch\s+-c|add|commit)\b"),
        re.compile(r"\bmkdir\b"),
        re.compile(r"\btouch\b"),
        re.compile(r"\bsed\s+-i\b"),
        re.compile(r"\btee\b"),
        re.compile(r"\bcp\b"),
        re.compile(r"\bmv\b"),
        re.compile(r">\s*[^|]"),
        re.compile(r">>\s*"),
        re.compile(r"\bpip(?:3)?\s+install\b"),
        re.compile(r"\bnpm\s+(install|ci)\b"),
        re.compile(r"\bpnpm\s+install\b"),
        re.compile(r"\byarn\s+install\b"),
        re.compile(r"\bdocker\b"),
        re.compile(r"\bpython(?:3)?\s+manage\.py\b"),
        re.compile(r"\balembic\b"),
    ]

    def __init__(self, config: AppConfig, workspace: WorkspaceManager):
        self.config = config
        self.workspace = workspace

    def ensure_read_allowed(self, path: str | Path) -> Path:
        return self.workspace.resolve_path(path)

    def ensure_write_allowed(self, path: str | Path) -> Path:
        target = self.workspace.resolve_path(path)
        if self.config.read_only:
            raise SafetyError("Write blocked because safe mode is enabled.")
        return target

    def assess_tool_call(self, tool_name: str, args: Any | None = None) -> ToolAssessment:
        if tool_name in self.READ_TOOLS:
            return ToolAssessment(
                tool_name=tool_name,
                risk_level="low",
                allowed=True,
                reasons=["Read-only inspection tool accepted."],
            )

        if tool_name in {"delete_file", "git_create_branch"}:
            if self.config.access_mode != AccessMode.FULL.value:
                return ToolAssessment(
                    tool_name=tool_name,
                    risk_level="medium",
                    allowed=False,
                    reasons=[
                        f"{tool_name} requires full access mode because it mutates repository state.",
                    ],
                )

        if tool_name in self.WRITE_TOOLS:
            if self.config.read_only:
                return ToolAssessment(
                    tool_name=tool_name,
                    risk_level="blocked",
                    allowed=False,
                    reasons=["Safe mode blocks file and git mutations."],
                )
            return ToolAssessment(
                tool_name=tool_name,
                risk_level="medium" if tool_name == "git_create_branch" else "low",
                allowed=True,
                reasons=["Mutating tool accepted for the current access mode."],
            )

        if tool_name in self.SHELL_TOOLS:
            command = getattr(args, "command", "") if args is not None else ""
            assessment = self.assess_shell_command(command)
            return ToolAssessment(
                tool_name=tool_name,
                risk_level=assessment.risk_level,
                allowed=assessment.allowed,
                reasons=assessment.reasons,
            )

        return ToolAssessment(
            tool_name=tool_name,
            risk_level="blocked",
            allowed=False,
            reasons=[f"Unknown tool policy for {tool_name}."],
        )

    def assess_shell_command(self, command: str) -> CommandAssessment:
        reasons: list[str] = []

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

        risk = self._classify_command_risk(command)
        mode = self.config.access_mode

        if mode == AccessMode.SAFE.value and risk != "low":
            reasons.append(
                "Safe mode permits only low-risk inspection and verification commands."
            )
            return CommandAssessment(
                command=command,
                risk_level=risk,
                allowed=False,
                reasons=reasons,
            )

        if mode == AccessMode.APPROVAL.value and risk in {"medium", "high"}:
            reasons.append(
                "Approval mode requires confirmation for medium or high risk shell commands."
            )
            return CommandAssessment(
                command=command,
                risk_level=risk,
                allowed=False,
                reasons=reasons,
            )

        if risk == "high" and not self.config.allow_dangerous_commands:
            reasons.append("Dangerous commands are disabled by configuration.")
            return CommandAssessment(
                command=command,
                risk_level=risk,
                allowed=False,
                reasons=reasons,
            )

        if not reasons:
            reasons.append("Command accepted.")
        return CommandAssessment(
            command=command,
            risk_level=risk,
            allowed=True,
            reasons=reasons,
        )

    def _classify_command_risk(self, command: str) -> str:
        if any(pattern.search(command) for pattern in self.DANGEROUS_PATTERNS):
            return "high"
        if any(pattern.search(command) for pattern in self.MUTATING_PATTERNS):
            return "medium"
        if any(pattern.search(command) for pattern in self.INSPECTION_PATTERNS):
            return "low"
        if any(pattern.search(command) for pattern in self.VERIFICATION_PATTERNS):
            return "low"
        return "medium"
