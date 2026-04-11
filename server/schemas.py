from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskCreateRequest(StrictModel):
    prompt: str = Field(..., min_length=1)
    session_id: str | None = None
    workspace_id: str | None = None
    enqueue_if_busy: bool = False
    access_mode: Literal["safe", "approval", "full"] | None = None
    dry_run: bool | None = None
    read_only: bool | None = None
    approval_mode: bool | None = None
    verbose: bool | None = None
    model_name: str | None = None
    agent_profile: str | None = None
    execution_profile: Literal["fast", "balanced", "deep"] | None = None


class SessionUpdateRequest(StrictModel):
    archived: bool | None = None
    stop_requested: bool | None = None


class SessionSummary(StrictModel):
    id: str
    task: str
    title: str | None = None
    status: str
    current_phase: str
    workflow_stage: str
    validation_status: str
    access_mode: str
    workspace_id: str | None = None
    workspace_label: str | None = None
    created_at: str
    updated_at: str
    iterations: int
    changed_file_count: int
    tool_call_count: int
    message_count: int = 0
    last_message_preview: str | None = None
    archived: bool = False
    archived_at: str | None = None
    stop_requested: bool = False
    stop_reason: str | None = None
    runtime_options: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(StrictModel):
    ok: bool
    active_sessions: list[str]


class LogRecord(StrictModel):
    timestamp: str
    event: str
    payload: dict[str, Any] = Field(default_factory=dict)


class WorkspaceInspectResponse(StrictModel):
    text: str
    snapshot: dict[str, Any]


class WorkspaceRecord(StrictModel):
    id: str
    name: str
    path: str
    created_at: str
    updated_at: str
    git_sync_source: str | None = None
    git_branch: str | None = None
    git_remote_name: str | None = None
    last_git_sync_at: str | None = None


class WorkspaceCreateRequest(StrictModel):
    name: str = Field(..., min_length=1, max_length=80)
    path: str = Field(..., min_length=1)
    git_sync_source: str | None = Field(default=None, min_length=1)
    git_branch: str | None = Field(default=None, min_length=1, max_length=120)
    git_remote_name: str | None = Field(default=None, min_length=1, max_length=80)
    sync_on_create: bool = False


class WorkspaceUpdateRequest(StrictModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    path: str | None = Field(default=None, min_length=1)
    git_sync_source: str | None = Field(default=None, min_length=1)
    git_branch: str | None = Field(default=None, min_length=1, max_length=120)
    git_remote_name: str | None = Field(default=None, min_length=1, max_length=80)
    sync_on_save: bool | None = None


class GitRepositoryCandidate(StrictModel):
    name: str
    path: str
    remote_url: str | None = None
    current_branch: str | None = None
    local_branches: list[str] = Field(default_factory=list)
    remote_branches: list[str] = Field(default_factory=list)
    has_uncommitted_changes: bool = False


class GitSourceInspectionResponse(StrictModel):
    source: str
    source_kind: Literal["local_path", "remote_url"]
    resolved_path: str | None = None
    remote_url: str | None = None
    current_branch: str | None = None
    default_branch: str | None = None
    local_branches: list[str] = Field(default_factory=list)
    remote_branches: list[str] = Field(default_factory=list)
    has_uncommitted_changes: bool = False


class WorkspaceGitStatus(StrictModel):
    workspace_id: str
    workspace_path: str
    is_repo: bool
    configured_source: str | None = None
    remote_name: str | None = None
    remote_url: str | None = None
    current_branch: str | None = None
    configured_branch: str | None = None
    default_branch: str | None = None
    local_branches: list[str] = Field(default_factory=list)
    remote_branches: list[str] = Field(default_factory=list)
    has_uncommitted_changes: bool = False
    ahead_by: int = 0
    behind_by: int = 0
    last_synced_at: str | None = None


class WorkspaceGitSyncRequest(StrictModel):
    git_sync_source: str | None = Field(default=None, min_length=1)
    git_branch: str | None = Field(default=None, min_length=1, max_length=120)
    git_remote_name: str | None = Field(default=None, min_length=1, max_length=80)


class WorkspaceGitSyncResponse(StrictModel):
    workspace: WorkspaceRecord
    git: WorkspaceGitStatus


class TerminalSessionCreateRequest(StrictModel):
    workspace_id: str | None = None
    cwd: str | None = Field(default=None, min_length=1)


class TerminalSessionInputRequest(StrictModel):
    data: str = Field(..., min_length=1, max_length=20_000)


class TerminalSessionResponse(StrictModel):
    id: str
    cwd: str
    shell: str
    created_at: str
    updated_at: str
    status: Literal["running", "exited"]
    cursor: int
    output: str = ""
    exit_code: int | None = None
    reset: bool = False


class ModelInventory(StrictModel):
    name: str
    size: int | None = None
    modified_at: str | None = None
    family: str | None = None
    parameter_size: str | None = None


class RecommendedModelStatus(StrictModel):
    name: str
    label: str
    summary: str
    installed: bool = False
    status: Literal["installed", "missing", "queued", "pulling", "verifying", "failed"]
    progress: float | None = None
    completed_bytes: int | None = None
    total_bytes: int | None = None
    message: str | None = None
    error: str | None = None
    updated_at: str | None = None


class ModelCatalogResponse(StrictModel):
    installed_models: list[ModelInventory] = Field(default_factory=list)
    recommended_models: list[RecommendedModelStatus] = Field(default_factory=list)
