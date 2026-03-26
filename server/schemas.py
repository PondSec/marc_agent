from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskCreateRequest(StrictModel):
    prompt: str = Field(..., min_length=1)
    session_id: str | None = None
    dry_run: bool | None = None
    read_only: bool | None = None
    approval_mode: bool | None = None
    verbose: bool | None = None


class SessionSummary(StrictModel):
    id: str
    task: str
    status: str
    updated_at: str
    iterations: int
    changed_file_count: int
    tool_call_count: int
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
