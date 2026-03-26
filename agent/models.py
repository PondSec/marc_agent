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
    "reporting",
    "blocked",
    "completed",
]
ValidationStatus = Literal["not_run", "passed", "failed", "blocked"]
WorkflowStage = Literal[
    "discover",
    "plan",
    "act",
    "verify",
    "repair",
    "report",
    "blocked",
    "completed",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PlanItem(StrictModel):
    step: str
    status: Literal["pending", "in_progress", "completed", "blocked"] = "pending"


class FileInsight(StrictModel):
    path: str
    category: str
    score: int
    reasons: list[str] = Field(default_factory=list)
    summary: str | None = None


class ValidationCommand(StrictModel):
    command: str
    cwd: str = "."
    kind: str = "check"
    source: str = "heuristic"
    priority: int = 100
    reason: str | None = None
    required: bool = True


class ValidationRunRecord(StrictModel):
    command: str
    cwd: str = "."
    kind: str | None = None
    status: Literal["passed", "failed", "blocked", "timeout"]
    exit_code: int | None = None
    risk_level: str | None = None
    iteration: int | None = None
    edit_generation: int = 0
    summary: str | None = None
    excerpt: str | None = None
    timestamp: str = Field(default_factory=utc_now)


class DiagnosticRecord(StrictModel):
    source: str
    category: str
    severity: Literal["info", "warning", "error"] = "error"
    summary: str
    tool_name: str | None = None
    command: str | None = None
    exit_code: int | None = None
    file_hints: list[str] = Field(default_factory=list)
    line_hints: list[int] = Field(default_factory=list)
    action_hints: list[str] = Field(default_factory=list)
    excerpt: str | None = None
    iteration: int | None = None
    timestamp: str = Field(default_factory=utc_now)


class SessionReport(StrictModel):
    summary: str
    status: str
    stop_reason: str | None = None
    changed_files: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    validation: list[ValidationRunRecord] = Field(default_factory=list)
    diagnostics: list[DiagnosticRecord] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    helper_artifacts: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    report_path: str | None = None


class WorkspaceSnapshot(StrictModel):
    root: str
    file_count: int
    language_counts: dict[str, int] = Field(default_factory=dict)
    top_directories: list[str] = Field(default_factory=list)
    important_files: list[str] = Field(default_factory=list)
    focus_files: list[str] = Field(default_factory=list)
    file_briefs: dict[str, str] = Field(default_factory=dict)
    file_insights: list[FileInsight] = Field(default_factory=list)
    manifests: list[str] = Field(default_factory=list)
    configs: list[str] = Field(default_factory=list)
    test_files: list[str] = Field(default_factory=list)
    build_files: list[str] = Field(default_factory=list)
    deploy_files: list[str] = Field(default_factory=list)
    entrypoints: list[str] = Field(default_factory=list)
    repo_map: list[str] = Field(default_factory=list)
    project_labels: list[str] = Field(default_factory=list)
    likely_commands: list[str] = Field(default_factory=list)
    validation_commands: list[ValidationCommand] = Field(default_factory=list)
    workflow_commands: list[ValidationCommand] = Field(default_factory=list)
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
    workflow_stage: WorkflowStage = "plan"
    validation_status: ValidationStatus = "not_run"
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    iterations: int = 0
    repair_attempts: int = 0
    edit_generation: int = 0
    plan_summary: str | None = None
    plan: list[PlanItem] = Field(default_factory=list)
    candidate_files: list[str] = Field(default_factory=list)
    verification_commands: list[str] = Field(default_factory=list)
    validation_plan: list[ValidationCommand] = Field(default_factory=list)
    validation_runs: list[ValidationRunRecord] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)
    helper_artifacts: list[str] = Field(default_factory=list)
    workspace_snapshot: WorkspaceSnapshot | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    changed_files: list[FileChangeRecord] = Field(default_factory=list)
    executed_commands: list[str] = Field(default_factory=list)
    diagnostics: list[DiagnosticRecord] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    runtime_options: dict[str, Any] = Field(default_factory=dict)
    stop_reason: str | None = None
    last_error: str | None = None
    final_response: str | None = None
    report: SessionReport | None = None

    def touch(self) -> None:
        self.updated_at = utc_now()
