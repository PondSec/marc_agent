from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from agent.task_state import RequestDigest, TaskState
from agent.task_schema import TaskUnderstanding
from llm.schemas import RouterOutput


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
MemoryType = Literal["working", "episodic", "project", "failure", "conversation"]
MemoryUseCase = Literal[
    "task_continuation",
    "repair_assistance",
    "similar_task_lookup",
    "project_context",
    "user_recall",
]
BootstrapStatus = Literal["none", "bootstrap_failed", "bootstrap_reset_required"]
ValidationStatus = Literal[
    "not_run",
    "passed",
    "failed",
    "blocked",
    "bootstrap_failed",
    "bootstrap_reset_required",
]
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
ToolExecutionMode = Literal["read_only", "exclusive", "mutating", "destructive"]


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
    verification_scope: Literal["syntax", "static", "structural", "semantic", "runtime"] = "static"
    source: str = "heuristic"
    priority: int = 100
    reason: str | None = None
    required: bool = True
    expected_stdout: str | None = None


class ValidationRunRecord(StrictModel):
    command: str
    cwd: str = "."
    kind: str | None = None
    verification_scope: Literal["syntax", "static", "structural", "semantic", "runtime"] = "static"
    status: Literal["passed", "failed", "blocked", "timeout"]
    exit_code: int | None = None
    risk_level: str | None = None
    iteration: int | None = None
    edit_generation: int = 0
    summary: str | None = None
    excerpt: str | None = None
    failure_signature: str | None = None
    root_cause_summary: str | None = None
    bootstrap_status: BootstrapStatus = "none"
    timestamp: str = Field(default_factory=utc_now)


class RepairAttemptSummary(StrictModel):
    target: str | None = None
    strategy: str
    result: str
    reason: str | None = None


class RepairBrief(StrictModel):
    failure_type: str | None = None
    failure_signature: str | None = None
    primary_target: str | None = None
    locked_target: str | None = None
    root_cause_summary: str | None = None
    bootstrap_status: BootstrapStatus = "none"
    bootstrap_reason: str | None = None
    expected_semantics: list[str] = Field(default_factory=list)
    observed_semantics: list[str] = Field(default_factory=list)
    implicated_symbols: list[str] = Field(default_factory=list)
    implicated_region_hint: str | None = None
    repair_constraints: list[str] = Field(default_factory=list)
    recent_failed_attempts: list[RepairAttemptSummary] = Field(default_factory=list)
    allowed_files: list[str] = Field(default_factory=list)
    forbidden_files: list[str] = Field(default_factory=list)


class ValidationFailureEvidence(StrictModel):
    command: str
    verification_scope: Literal["syntax", "static", "structural", "semantic", "runtime"] = "static"
    status: Literal["failed", "blocked", "timeout"] = "failed"
    artifact_paths: list[str] = Field(default_factory=list)
    summary: str
    excerpt: str | None = None
    failure_summary: str
    expected_features: list[str] = Field(default_factory=list)
    missing_features: list[str] = Field(default_factory=list)
    file_hints: list[str] = Field(default_factory=list)
    line_hints: list[int] = Field(default_factory=list)
    action_hints: list[str] = Field(default_factory=list)
    repair_requirements: list[str] = Field(default_factory=list)
    evidence_signature: str | None = None
    root_cause_summary: str | None = None
    bootstrap_status: BootstrapStatus = "none"
    repair_brief: RepairBrief | None = None


class SemanticChangeReview(StrictModel):
    requirements_satisfied: bool
    summary: str
    confidence: float = 0.0
    missing_requirements: list[str] = Field(default_factory=list)
    suspicious_issues: list[str] = Field(default_factory=list)
    file_hints: list[str] = Field(default_factory=list)
    repair_hints: list[str] = Field(default_factory=list)


class ProposedUpdateReview(StrictModel):
    safe_to_write: bool
    summary: str
    confidence: float = 0.0
    blocking_issues: list[str] = Field(default_factory=list)
    preservation_risks: list[str] = Field(default_factory=list)
    repair_hints: list[str] = Field(default_factory=list)


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


class RepairAttemptRecord(StrictModel):
    artifact_path: str | None = None
    validation_command: str | None = None
    verification_scope: Literal["syntax", "static", "structural", "semantic", "runtime"] | None = None
    strategy: str
    result: Literal["mutation_planned", "no_effective_change", "generation_failed", "blocked"]
    reason: str
    evidence_signature: str | None = None
    failure_signature: str | None = None
    region_hint: str | None = None
    root_cause_summary: str | None = None
    productive_change: bool | None = None
    before_hash: str | None = None
    after_hash: str | None = None
    change_labels: list[str] = Field(default_factory=list)
    post_validation_failure_signature: str | None = None
    behavior_changed: bool | None = None
    independent_verification: bool | None = None
    iteration: int | None = None
    timestamp: str = Field(default_factory=utc_now)


class FollowUpContext(StrictModel):
    previous_task: str | None = None
    previous_root_goal: str | None = None
    previous_active_goal: str | None = None
    previous_next_action: str | None = None
    previous_intent: str | None = None
    previous_requested_outcome: str | None = None
    previous_final_response: str | None = None
    previous_interpreted_goal: str | None = None
    previous_recommended_mode: str | None = None
    previous_confidence: float | None = None
    previous_assumptions: list[str] = Field(default_factory=list)
    previous_constraints: list[str] = Field(default_factory=list)
    target_paths: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    read_files: list[str] = Field(default_factory=list)
    recent_commands: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    diagnostics: list[DiagnosticRecord] = Field(default_factory=list)
    validation_runs: list[ValidationRunRecord] = Field(default_factory=list)
    last_error: str | None = None


class MemoryProvenance(StrictModel):
    source_type: Literal[
        "session",
        "workspace_snapshot",
        "validation",
        "diagnostic",
        "repair",
        "conversation",
        "report",
    ] = "session"
    session_id: str | None = None
    project_id: str | None = None
    workspace_root: str | None = None
    detail: str | None = None
    file_paths: list[str] = Field(default_factory=list)
    command: str | None = None
    created_at: str = Field(default_factory=utc_now)


class MemorySummary(StrictModel):
    title: str
    summary: str
    key_points: list[str] = Field(default_factory=list)
    why_relevant: str | None = None


class MemoryEntryBase(StrictModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:16])
    memory_type: MemoryType
    project_id: str | None = None
    workspace_root: str | None = None
    session_id: str | None = None
    summary: MemorySummary
    provenance: MemoryProvenance
    tags: list[str] = Field(default_factory=list)
    file_paths: list[str] = Field(default_factory=list)
    symbol_names: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    last_accessed_at: str | None = None
    importance: float = 0.5
    confidence: float = 0.5
    dedupe_key: str | None = None
    duplicate_count: int = 0
    retention: Literal["transient", "short", "medium", "long"] = "short"
    ttl_days: int | None = None


class WorkingMemoryEntry(MemoryEntryBase):
    memory_type: Literal["working"] = "working"
    current_goal: str | None = None
    current_subtask: str | None = None
    primary_target: str | None = None
    verification_target: str | None = None
    request_excerpt: str | None = None
    request_requirements: list[str] = Field(default_factory=list)
    request_chunks: list[str] = Field(default_factory=list)
    request_digest: RequestDigest | None = None
    active_constraints: list[str] = Field(default_factory=list)
    active_failure_signature: str | None = None
    recent_attempts: list[str] = Field(default_factory=list)
    recent_successes: list[str] = Field(default_factory=list)
    recent_failures: list[str] = Field(default_factory=list)
    relevant_files: list[str] = Field(default_factory=list)
    relevant_symbols: list[str] = Field(default_factory=list)
    last_effective_strategy: str | None = None
    last_ineffective_strategy: str | None = None
    compact_state_summary: str = ""


class EpisodicMemoryEntry(MemoryEntryBase):
    memory_type: Literal["episodic"] = "episodic"
    problem_type: str | None = None
    strategy_used: list[str] = Field(default_factory=list)
    result: Literal["completed", "partial", "failed", "blocked"] = "completed"
    what_worked: list[str] = Field(default_factory=list)
    what_failed: list[str] = Field(default_factory=list)
    important_constraints: list[str] = Field(default_factory=list)
    final_outcome: str | None = None
    changed_files: list[str] = Field(default_factory=list)
    failure_signatures: list[str] = Field(default_factory=list)


class ProjectMemoryEntry(MemoryEntryBase):
    memory_type: Literal["project"] = "project"
    repo_summary: str = ""
    module_roles: dict[str, str] = Field(default_factory=dict)
    directory_map: list[str] = Field(default_factory=list)
    entrypoints: list[str] = Field(default_factory=list)
    service_files: list[str] = Field(default_factory=list)
    import_hotspots: list[str] = Field(default_factory=list)
    common_file_relationships: list[str] = Field(default_factory=list)
    test_mappings: list[str] = Field(default_factory=list)
    symbol_index: dict[str, list[str]] = Field(default_factory=dict)
    file_relationships: dict[str, list[str]] = Field(default_factory=dict)
    module_summaries: dict[str, str] = Field(default_factory=dict)
    architecture_notes: list[str] = Field(default_factory=list)
    known_hotspots: list[str] = Field(default_factory=list)
    conventions: list[str] = Field(default_factory=list)
    workflow_hints: list[str] = Field(default_factory=list)
    co_change_hints: list[str] = Field(default_factory=list)
    subsystem_summaries: dict[str, str] = Field(default_factory=dict)


class FailureMemoryEntry(MemoryEntryBase):
    memory_type: Literal["failure"] = "failure"
    failure_signature: str
    expected_semantics: list[str] = Field(default_factory=list)
    observed_semantics: list[str] = Field(default_factory=list)
    chosen_targets: list[str] = Field(default_factory=list)
    tried_strategies: list[str] = Field(default_factory=list)
    successful_repair_patterns: list[str] = Field(default_factory=list)
    bad_retry_patterns: list[str] = Field(default_factory=list)
    review_rejection_reasons: list[str] = Field(default_factory=list)
    no_effective_change_count: int = 0
    last_result: str | None = None


class RememberedFact(StrictModel):
    subject: Literal["user", "assistant"]
    attribute: str
    value: str
    summary: str


class ConversationMemoryEntry(MemoryEntryBase):
    memory_type: Literal["conversation"] = "conversation"
    request_summary: str
    delivered_summary: str | None = None
    projects_touched: list[str] = Field(default_factory=list)
    decision_notes: list[str] = Field(default_factory=list)
    implemented_features: list[str] = Field(default_factory=list)
    referenced_sessions: list[str] = Field(default_factory=list)
    remembered_facts: list[RememberedFact] = Field(default_factory=list)


class RetrievedMemoryItem(StrictModel):
    entry_id: str
    memory_type: MemoryType
    project_id: str | None = None
    session_id: str | None = None
    entry: (
        WorkingMemoryEntry
        | EpisodicMemoryEntry
        | ProjectMemoryEntry
        | FailureMemoryEntry
        | ConversationMemoryEntry
        | None
    ) = None
    summary: MemorySummary
    provenance: MemoryProvenance
    file_paths: list[str] = Field(default_factory=list)
    symbol_names: list[str] = Field(default_factory=list)
    failure_signature: str | None = None
    score: float = 0.0
    similarity: float = 0.0
    recency: float = 0.0
    project_relevance: float = 0.0
    failure_relevance: float = 0.0
    exact_entity_relevance: float = 0.0
    confidence: float = 0.0
    duplicate_count: int = 0
    is_stale: bool = False
    reasons: list[str] = Field(default_factory=list)


class RetrievalRequest(StrictModel):
    query: str
    use_case: MemoryUseCase = "task_continuation"
    project_id: str | None = None
    workspace_root: str | None = None
    session_id: str | None = None
    recall_subject: Literal["user", "assistant"] | None = None
    recall_attributes: list[str] = Field(default_factory=list)
    target_paths: list[str] = Field(default_factory=list)
    symbol_names: list[str] = Field(default_factory=list)
    error_terms: list[str] = Field(default_factory=list)
    failure_signature: str | None = None
    current_goal: str | None = None
    current_subtask: str | None = None
    changed_files: list[str] = Field(default_factory=list)
    include_types: list[MemoryType] = Field(
        default_factory=lambda: ["episodic", "project", "failure", "conversation"]
    )
    max_hits: int = 6
    max_per_type: int = 2
    summary_budget_chars: int = 900
    allow_cross_project: bool = False


MemoryQuery = RetrievalRequest


class MemoryRetrievalResult(StrictModel):
    request: RetrievalRequest
    selected: list[RetrievedMemoryItem] = Field(default_factory=list)
    summary: str = ""
    recall_brief: str = ""
    suggested_files: list[str] = Field(default_factory=list)
    suggested_symbols: list[str] = Field(default_factory=list)
    repo_map_hints: list[str] = Field(default_factory=list)
    related_sessions: list[str] = Field(default_factory=list)
    related_projects: list[str] = Field(default_factory=list)
    total_candidates: int = 0
    total_hits: int = 0
    hit_count_by_type: dict[str, int] = Field(default_factory=dict)
    latency_ms: float = 0.0
    prompt_char_cost: int = 0
    duplicate_rate: float = 0.0
    stale_recall_rate: float = 0.0
    useful_recall_rate: float = 0.0


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
    runtime_executions: list[dict[str, Any]] = Field(default_factory=list)
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
    test_mappings: list[str] = Field(default_factory=list)
    service_files: list[str] = Field(default_factory=list)
    import_hotspots: list[str] = Field(default_factory=list)
    symbol_index: dict[str, list[str]] = Field(default_factory=dict)
    file_relationships: dict[str, list[str]] = Field(default_factory=dict)
    module_summaries: dict[str, str] = Field(default_factory=dict)
    subsystem_summaries: dict[str, str] = Field(default_factory=dict)
    project_labels: list[str] = Field(default_factory=list)
    likely_commands: list[str] = Field(default_factory=list)
    validation_commands: list[ValidationCommand] = Field(default_factory=list)
    workflow_commands: list[ValidationCommand] = Field(default_factory=list)
    repo_summary: str = ""


class FileChangeRecord(StrictModel):
    path: str
    operation: str
    diff: str | None = None


class ToolExecutionMeta(StrictModel):
    category: str = "general"
    read_only: bool = False
    mutating: bool = False
    destructive: bool = False
    concurrency_safe: bool = False
    verification_tool: bool = False
    execution_mode: ToolExecutionMode = "exclusive"


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
    tool_meta: ToolExecutionMeta | None = None
    timestamp: str = Field(default_factory=utc_now)


class ToolRunResult(StrictModel):
    tool_name: str
    success: bool
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    risk_level: str | None = None
    changed_files: list[FileChangeRecord] = Field(default_factory=list)
    tool_meta: ToolExecutionMeta | None = None


class ChatMessage(StrictModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    role: Literal["user", "assistant", "system"] = "user"
    content: str
    created_at: str = Field(default_factory=utc_now)


class SessionState(StrictModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    task: str
    title: str | None = None
    status: Literal["queued", "running", "completed", "failed", "partial"] = "running"
    workspace_root: str
    project_id: str | None = None
    workspace_id: str | None = None
    workspace_label: str | None = None
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
    task_analysis: dict[str, Any] | None = None
    task_state: TaskState | None = None
    task_understanding: TaskUnderstanding | None = None
    router_result: RouterOutput | None = None
    plan: list[PlanItem] = Field(default_factory=list)
    candidate_files: list[str] = Field(default_factory=list)
    verification_commands: list[str] = Field(default_factory=list)
    validation_plan: list[ValidationCommand] = Field(default_factory=list)
    validation_runs: list[ValidationRunRecord] = Field(default_factory=list)
    active_repair_context: ValidationFailureEvidence | None = None
    repair_history: list[RepairAttemptRecord] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)
    helper_artifacts: list[str] = Field(default_factory=list)
    workspace_snapshot: WorkspaceSnapshot | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    changed_files: list[FileChangeRecord] = Field(default_factory=list)
    executed_commands: list[str] = Field(default_factory=list)
    diagnostics: list[DiagnosticRecord] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    runtime_executions: list[dict[str, Any]] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    follow_up_context: FollowUpContext | None = None
    working_memory: WorkingMemoryEntry | None = None
    memory_context: MemoryRetrievalResult | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    runtime_options: dict[str, Any] = Field(default_factory=dict)
    stop_requested: bool = False
    archived: bool = False
    archived_at: str | None = None
    stop_reason: str | None = None
    last_error: str | None = None
    final_response: str | None = None
    report: SessionReport | None = None

    def touch(self) -> None:
        self.updated_at = utc_now()

    def append_message(self, role: Literal["user", "assistant", "system"], content: str) -> None:
        text = str(content or "").strip()
        if not text:
            return
        self.messages.append(ChatMessage(role=role, content=text))
