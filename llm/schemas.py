from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentActionType(str, Enum):
    CALL_TOOL = "call_tool"
    FINAL = "final"


class RouteIntent(str, Enum):
    INSPECT = "inspect"
    CREATE = "create"
    UPDATE = "update"
    DEBUG = "debug"
    DELETE = "delete"
    SEARCH = "search"
    EXPLAIN = "explain"
    PLAN = "plan"
    UNKNOWN = "unknown"


class RouteActionName(str, Enum):
    RESPOND_DIRECTLY = "respond_directly"
    ASK_CLARIFICATION = "ask_clarification"
    INSPECT_WORKSPACE = "inspect_workspace"
    SEARCH_WORKSPACE = "search_workspace"
    READ_RELEVANT_FILES = "read_relevant_files"
    DIAGNOSE_ISSUE = "diagnose_issue"
    CREATE_ARTIFACT = "create_artifact"
    UPDATE_ARTIFACT = "update_artifact"
    DELETE_ARTIFACT = "delete_artifact"
    PLAN_WORK = "plan_work"
    RUN_VALIDATION = "run_validation"
    SUMMARIZE_RESULT = "summarize_result"


def _compact_strings(values: list[str], *, limit: int | None = None) -> list[str]:
    unique: list[str] = []
    for raw in values:
        text = str(raw or "").strip()
        if not text or text in unique:
            continue
        unique.append(text)
    if limit is not None:
        return unique[:limit]
    return unique


class RouteEntities(StrictModel):
    target_type: str | None = Field(default=None, description="Type of the primary target.")
    target_name: str | None = Field(default=None, description="Name of the primary target.")
    target_paths: list[str] = Field(
        default_factory=list,
        description="Concrete files or directories when they are already known.",
    )
    attributes: list[str] = Field(default_factory=list, description="Useful extracted attributes.")
    constraints: list[str] = Field(default_factory=list, description="Execution constraints or limits.")

    @model_validator(mode="after")
    def normalize(self) -> RouteEntities:
        self.target_type = str(self.target_type or "").strip() or None
        self.target_name = str(self.target_name or "").strip() or None
        self.target_paths = _compact_strings(self.target_paths, limit=8)
        self.attributes = _compact_strings(self.attributes, limit=8)
        self.constraints = _compact_strings(self.constraints, limit=8)
        return self


class RouteActionStep(StrictModel):
    step: int = Field(..., ge=1, description="1-based action order.")
    action: RouteActionName = Field(..., description="Executor action from the approved catalog.")
    reason: str = Field(..., min_length=1, description="Why this step is needed.")

    @model_validator(mode="after")
    def normalize(self) -> RouteActionStep:
        self.reason = str(self.reason or "").strip()
        return self


class RouterOutput(StrictModel):
    user_goal: str = Field(..., min_length=1, description="What the user is ultimately trying to achieve.")
    intent: RouteIntent = Field(..., description="High-level request intent.")
    entities: RouteEntities = Field(default_factory=RouteEntities)
    requested_outcome: str = Field(
        ...,
        min_length=1,
        description="Concrete expected result or deliverable.",
    )
    action_plan: list[RouteActionStep] = Field(
        default_factory=list,
        description="Ordered execution plan using the approved action catalog.",
    )
    needs_clarification: bool = Field(..., description="Whether the agent must ask follow-up questions.")
    clarification_questions: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="One to three targeted questions when clarification is required.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1.")
    safe_to_execute: bool = Field(..., description="Whether the executor may proceed without extra input.")
    repo_context_needed: bool = Field(
        default=True,
        description="Whether repository inspection is needed before the final answer or mutation.",
    )
    search_terms: list[str] = Field(
        default_factory=list,
        description="High-signal search phrases derived by the router.",
    )
    relevant_extensions: list[str] = Field(
        default_factory=list,
        description="Likely relevant file extensions such as .py or .ts.",
    )
    direct_response: str | None = Field(
        default=None,
        description="Optional direct reply when no repository work is needed.",
    )

    @model_validator(mode="after")
    def normalize(self) -> RouterOutput:
        self.user_goal = str(self.user_goal or "").strip()
        self.requested_outcome = str(self.requested_outcome or "").strip()
        self.clarification_questions = _compact_strings(self.clarification_questions, limit=3)
        self.search_terms = _compact_strings(self.search_terms, limit=6)
        self.relevant_extensions = _compact_strings(self.relevant_extensions, limit=6)
        ordered_steps = sorted(self.action_plan, key=lambda item: item.step)
        for index, item in enumerate(ordered_steps, start=1):
            item.step = index
        self.action_plan = ordered_steps
        self.direct_response = str(self.direct_response or "").strip() or None

        if not self.action_plan:
            raise ValueError("action_plan must contain at least one step")
        if self.needs_clarification:
            if not self.clarification_questions:
                raise ValueError(
                    "clarification_questions must contain one to three entries when clarification is required"
                )
            if self.safe_to_execute:
                raise ValueError("safe_to_execute must be false when clarification is required")
        if self.intent == RouteIntent.UNKNOWN and self.safe_to_execute:
            raise ValueError("unknown intent cannot be marked safe_to_execute")
        if self.direct_response and self.needs_clarification:
            raise ValueError("direct_response cannot be set while clarification is required")
        return self


def router_output_schema() -> dict[str, Any]:
    return RouterOutput.model_json_schema()


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
        default_factory=list,
        description="Likely relevant files or directories.",
    )
    tests_to_run: list[str] = Field(
        default_factory=list,
        description="Likely validation commands.",
    )
    completion_criteria: list[str] = Field(
        default_factory=list,
        description="Conditions that should be true before returning final output.",
    )


class EmptyArgs(StrictModel):
    pass


class InspectWorkspaceArgs(StrictModel):
    focus: str | None = Field(
        default=None,
        description="Optional task focus like auth, billing, API, or tests.",
    )


class ListFilesArgs(StrictModel):
    path: str = Field(
        default=".",
        description="Directory path relative to the workspace root, or absolute in full access mode.",
    )
    glob: str | None = Field(default=None, description="Optional glob filter.")
    recursive: bool = Field(default=True, description="Whether to recurse into subdirectories.")
    max_results: int = Field(default=200, description="Maximum number of returned paths.")


class ReadFileArgs(StrictModel):
    path: str = Field(
        ...,
        description="File path relative to the workspace root, or absolute in full access mode.",
    )
    start_line: int | None = Field(default=None, description="Optional first line to read.")
    end_line: int | None = Field(default=None, description="Optional last line to read.")


class WriteFileArgs(StrictModel):
    path: str = Field(
        ...,
        description="File path relative to the workspace root, or absolute in full access mode.",
    )
    content: str = Field(..., description="Full file content to write.")


class AppendFileArgs(StrictModel):
    path: str = Field(
        ...,
        description="File path relative to the workspace root, or absolute in full access mode.",
    )
    content: str = Field(..., description="Text to append.")


class CreateFileArgs(StrictModel):
    path: str = Field(
        ...,
        description="File path relative to the workspace root, or absolute in full access mode.",
    )
    content: str = Field(default="", description="Initial file content.")
    overwrite: bool = Field(default=False, description="Allow replacing an existing file.")


class DeleteFileArgs(StrictModel):
    path: str = Field(
        ...,
        description="File path relative to the workspace root, or absolute in full access mode.",
    )


class ReplaceInFileArgs(StrictModel):
    path: str = Field(
        ...,
        description="File path relative to the workspace root, or absolute in full access mode.",
    )
    find: str = Field(..., description="Literal text to replace.")
    replace: str = Field(..., description="Replacement text.")
    count: int = Field(default=0, description="Maximum replacements; 0 means all.")


class TextPatch(StrictModel):
    old: str = Field(..., description="Existing text that must be present.")
    new: str = Field(..., description="Replacement text.")
    expected_count: int = Field(
        default=1,
        description="Expected number of occurrences to replace.",
    )


class PatchFileArgs(StrictModel):
    path: str = Field(
        ...,
        description="File path relative to the workspace root, or absolute in full access mode.",
    )
    patches: list[TextPatch] = Field(..., description="Ordered patch operations.")


class ShowDiffArgs(StrictModel):
    path: str = Field(
        ...,
        description="File path relative to the workspace root, or absolute in full access mode.",
    )
    new_content: str = Field(..., description="Candidate content to compare against disk.")


class SearchInFilesArgs(StrictModel):
    query: str = Field(..., description="Literal or regex search query.")
    path: str = Field(
        default=".",
        description="Directory path relative to the workspace root, or absolute in full access mode.",
    )
    regex: bool = Field(default=False, description="Interpret query as a regex.")
    glob: str | None = Field(default=None, description="Optional glob filter.")
    case_sensitive: bool = Field(default=False, description="Case-sensitive matching.")
    max_results: int = Field(default=100, description="Maximum number of matches.")


class RunShellArgs(StrictModel):
    command: str = Field(..., description="Shell command to execute.")
    cwd: str = Field(
        default=".",
        description="Working directory relative to the workspace root, or absolute in full access mode.",
    )
    timeout: int | None = Field(default=None, description="Timeout in seconds.")


class RunTestsArgs(StrictModel):
    command: str = Field(..., description="Targeted test, lint, typecheck, or build command.")
    cwd: str = Field(
        default=".",
        description="Working directory relative to the workspace root, or absolute in full access mode.",
    )
    timeout: int | None = Field(default=None, description="Timeout in seconds.")
    expected_stdout: str | None = Field(
        default=None,
        description="Optional expected stdout contract for direct runtime validation commands.",
    )


class GitDiffArgs(StrictModel):
    path: str | None = Field(default=None, description="Optional path filter.")
    cached: bool = Field(default=False, description="Show staged changes.")
    context_lines: int = Field(default=3, description="Diff context line count.")


class GitLogArgs(StrictModel):
    limit: int = Field(default=5, description="Maximum number of commits to show.")


class GitCreateBranchArgs(StrictModel):
    name: str = Field(..., description="Branch name to create and switch to.")
