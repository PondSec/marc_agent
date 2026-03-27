from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.semantic_runtime import SemanticResolution


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


AmbiguityLevel = Literal["low", "medium", "high"]
RiskLevel = Literal["low", "medium", "high"]
RecommendedMode = Literal[
    "clarify",
    "inspect",
    "search",
    "create",
    "modify",
    "debug",
    "refactor",
    "test",
    "explain",
    "plan",
]
IntentCategory = Literal[
    "build",
    "modify",
    "debug",
    "refactor",
    "explain",
    "analyze",
    "test",
    "search",
    "configure",
    "plan",
    "unknown",
]
ConversationRelation = Literal[
    "new_task",
    "same_task_follow_up",
    "refinement",
    "problem_report",
    "constraint_update",
    "clarification",
    "correction",
    "unknown",
]


def _compact_strings(values: list[str], *, limit: int | None = None) -> list[str]:
    unique: list[str] = []
    for raw in values:
        text = str(raw or "").strip()
        if not text or text in unique:
            continue
        unique.append(text)
    return unique[:limit] if limit is not None else unique


class TaskArtifact(StrictModel):
    path: str | None = None
    name: str | None = None
    kind: str | None = None
    role: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def normalize(self) -> TaskArtifact:
        self.path = str(self.path or "").strip() or None
        self.name = str(self.name or "").strip() or None
        self.kind = str(self.kind or "").strip() or None
        self.role = str(self.role or "").strip() or None
        return self


class TaskPlanStep(StrictModel):
    step: int = Field(..., ge=1)
    summary: str
    action_hint: str | None = None
    requires_tools: bool = True

    @model_validator(mode="after")
    def normalize(self) -> TaskPlanStep:
        self.summary = str(self.summary or "").strip()
        self.action_hint = str(self.action_hint or "").strip() or None
        return self


class TaskUnderstanding(StrictModel):
    original_request: str
    interpreted_goal: str
    intent_category: IntentCategory = "unknown"
    conversation_relation: ConversationRelation = "unknown"
    subgoals: list[str] = Field(default_factory=list)
    target_artifacts: list[TaskArtifact] = Field(default_factory=list)
    relevant_context: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    user_observations: list[str] = Field(default_factory=list)
    supplied_evidence: list[str] = Field(default_factory=list)
    ambiguity_level: AmbiguityLevel = "medium"
    risk_level: RiskLevel = "medium"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    recommended_mode: RecommendedMode = "inspect"
    execution_plan: list[TaskPlanStep] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_questions: list[str] = Field(default_factory=list)
    semantic_resolution: SemanticResolution = "full_model"
    secondary_semantics_limited: bool = False

    @model_validator(mode="after")
    def normalize(self) -> TaskUnderstanding:
        self.original_request = str(self.original_request or "").strip()
        self.interpreted_goal = str(self.interpreted_goal or "").strip()
        self.subgoals = _compact_strings(self.subgoals, limit=8)
        self.relevant_context = _compact_strings(self.relevant_context, limit=8)
        self.constraints = _compact_strings(self.constraints, limit=8)
        self.missing_info = _compact_strings(self.missing_info, limit=6)
        self.assumptions = _compact_strings(self.assumptions, limit=8)
        self.user_observations = _compact_strings(self.user_observations, limit=8)
        self.supplied_evidence = _compact_strings(self.supplied_evidence, limit=8)
        self.clarification_questions = _compact_strings(self.clarification_questions, limit=3)
        ordered_steps = sorted(self.execution_plan, key=lambda item: item.step)
        for index, item in enumerate(ordered_steps, start=1):
            item.step = index
        self.execution_plan = ordered_steps

        if self.needs_clarification and not self.clarification_questions:
            raise ValueError("clarification_questions must be present when needs_clarification is true")
        if self.needs_clarification and self.confidence > 0.85:
            raise ValueError("high confidence tasks should not request clarification")
        if not self.original_request:
            raise ValueError("original_request is required")
        if not self.interpreted_goal:
            raise ValueError("interpreted_goal is required")
        return self
