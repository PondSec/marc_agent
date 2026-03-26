from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


AgentPhase = Literal[
    "planning",
    "exploring",
    "editing",
    "verifying",
    "repairing",
    "blocked",
    "completed",
]
ValidationStatus = Literal["not_run", "passed", "failed", "blocked"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PlanItem(StrictModel):
    step: str
    status: Literal["pending", "in_progress", "completed", "blocked"] = "pending"


class WorkspaceSnapshot(StrictModel):
    root: str
    file_count: int
    language_counts: dict[str, int] = Field(default_factory=dict)
    top_directories: list[str] = Field(default_factory=list)
    important_files: list[str] = Field(default_factory=list)
    file_briefs: dict[str, str] = Field(default_factory=dict)
    likely_commands: list[str] = Field(default_factory=list)
    repo_summary: str = ""


class FileChangeRecord(StrictModel):
    path: str
    operation: str
    diff: str | None = None


class ToolCallRecord(StrictModel):
    iteration: int
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)
    success: bool
    summary: str
    phase: AgentPhase | None = None
    thought_summary: str | None = None
    expected_outcome: str | None = None
    output_excerpt: str | None = None
    risk_level: str | None = None
    timestamp: str = Field(default_factory=utc_now)


class ToolRunResult(StrictModel):
    tool_name: str
    success: bool
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    risk_level: str | None = None
    changed_files: list[FileChangeRecord] = Field(default_factory=list)


class SessionState(StrictModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    task: str
    status: Literal["queued", "running", "completed", "failed", "partial"] = "running"
    workspace_root: str
    access_mode: str = "approval"
    current_phase: AgentPhase = "planning"
    validation_status: ValidationStatus = "not_run"
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    iterations: int = 0
    repair_attempts: int = 0
    plan_summary: str | None = None
    plan: list[PlanItem] = Field(default_factory=list)
    candidate_files: list[str] = Field(default_factory=list)
    verification_commands: list[str] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)
    helper_artifacts: list[str] = Field(default_factory=list)
    workspace_snapshot: WorkspaceSnapshot | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    changed_files: list[FileChangeRecord] = Field(default_factory=list)
    executed_commands: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    runtime_options: dict[str, Any] = Field(default_factory=dict)
    stop_reason: str | None = None
    last_error: str | None = None
    final_response: str | None = None

    def touch(self) -> None:
        self.updated_at = utc_now()
