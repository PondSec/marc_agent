from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentActionType(str, Enum):
    CALL_TOOL = "call_tool"
    FINAL = "final"


class AgentDecision(StrictModel):
    thought_summary: str = Field(..., description="Short reasoning summary.")
    action_type: AgentActionType = Field(..., description="Next action type.")
    tool_name: str | None = Field(None, description="Tool name when action_type=call_tool.")
    tool_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the selected tool.",
    )
    expected_outcome: str = Field(..., description="What the tool call should achieve.")
    final_response: str | None = Field(
        None,
        description="Filled only when action_type=final.",
    )


class PlanningResponse(StrictModel):
    summary: str = Field(..., description="High-level summary of the task.")
    steps: list[str] = Field(..., description="Ordered plan steps.")
    files_to_inspect: list[str] = Field(
        default_factory=list, description="Likely relevant files or directories."
    )
    tests_to_run: list[str] = Field(
        default_factory=list, description="Likely validation commands."
    )


class EmptyArgs(StrictModel):
    pass


class InspectWorkspaceArgs(StrictModel):
    focus: str | None = Field(
        default=None,
        description="Optional task focus like auth, billing, API, or tests.",
    )


class ListFilesArgs(StrictModel):
    path: str = Field(default=".", description="Directory path relative to the workspace root.")
    glob: str | None = Field(default=None, description="Optional glob filter.")
    recursive: bool = Field(default=True, description="Whether to recurse into subdirectories.")
    max_results: int = Field(default=200, description="Maximum number of returned paths.")


class ReadFileArgs(StrictModel):
    path: str = Field(..., description="File path relative to the workspace root.")
    start_line: int | None = Field(default=None, description="Optional first line to read.")
    end_line: int | None = Field(default=None, description="Optional last line to read.")


class WriteFileArgs(StrictModel):
    path: str = Field(..., description="File path relative to the workspace root.")
    content: str = Field(..., description="Full file content to write.")


class AppendFileArgs(StrictModel):
    path: str = Field(..., description="File path relative to the workspace root.")
    content: str = Field(..., description="Text to append.")


class CreateFileArgs(StrictModel):
    path: str = Field(..., description="File path relative to the workspace root.")
    content: str = Field(default="", description="Initial file content.")
    overwrite: bool = Field(default=False, description="Allow replacing an existing file.")


class DeleteFileArgs(StrictModel):
    path: str = Field(..., description="File path relative to the workspace root.")


class ReplaceInFileArgs(StrictModel):
    path: str = Field(..., description="File path relative to the workspace root.")
    find: str = Field(..., description="Literal text to replace.")
    replace: str = Field(..., description="Replacement text.")
    count: int = Field(default=0, description="Maximum replacements; 0 means all.")


class TextPatch(StrictModel):
    old: str = Field(..., description="Existing text that must be present.")
    new: str = Field(..., description="Replacement text.")
    expected_count: int = Field(
        default=1, description="Expected number of occurrences to replace."
    )


class PatchFileArgs(StrictModel):
    path: str = Field(..., description="File path relative to the workspace root.")
    patches: list[TextPatch] = Field(..., description="Ordered patch operations.")


class ShowDiffArgs(StrictModel):
    path: str = Field(..., description="File path relative to the workspace root.")
    new_content: str = Field(..., description="Candidate content to compare against disk.")


class SearchInFilesArgs(StrictModel):
    query: str = Field(..., description="Literal or regex search query.")
    path: str = Field(default=".", description="Directory path relative to the workspace root.")
    regex: bool = Field(default=False, description="Interpret query as a regex.")
    glob: str | None = Field(default=None, description="Optional glob filter.")
    case_sensitive: bool = Field(default=False, description="Case-sensitive matching.")
    max_results: int = Field(default=100, description="Maximum number of matches.")


class RunShellArgs(StrictModel):
    command: str = Field(..., description="Shell command to execute.")
    cwd: str = Field(default=".", description="Working directory relative to the workspace root.")
    timeout: int | None = Field(default=None, description="Timeout in seconds.")


class GitDiffArgs(StrictModel):
    path: str | None = Field(default=None, description="Optional path filter.")
    cached: bool = Field(default=False, description="Show staged changes.")
    context_lines: int = Field(default=3, description="Diff context line count.")


class GitLogArgs(StrictModel):
    limit: int = Field(default=5, description="Maximum number of commits to show.")


class GitCreateBranchArgs(StrictModel):
    name: str = Field(..., description="Branch name to create and switch to.")
