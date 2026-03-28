from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.semantic_runtime import SemanticResolution
from agent.task_schema import TaskArtifact, TaskPlanStep, TaskUnderstanding


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


GoalRelation = Literal[
    "new_task",
    "continue",
    "refine",
    "correct",
    "report_problem",
    "scope_change",
    "validation_request",
    "rollback_request",
    "approval",
    "rejection",
    "update_constraints",
    "clarify",
    "unknown",
]
UserIntent = Literal[
    "repair",
    "implement",
    "refactor",
    "harden",
    "validate",
    "inspect",
    "explain",
    "search",
    "plan",
    "correct",
    "unknown",
]
ExecutionStrategy = Literal[
    "debug_repair",
    "feature_implementation",
    "refactor",
    "hardening",
    "validation_inspection",
    "rollback_correction",
]
NextAction = Literal[
    "inspect",
    "search",
    "create",
    "modify",
    "debug",
    "test",
    "explain",
    "plan",
    "clarify",
]
EvidenceKind = Literal["message", "log", "diff", "diagnostic", "test", "command", "file", "unknown"]
SemanticInferenceMode = Literal["full", "conservative"]

GOAL_RELATION_ALIASES: dict[str, GoalRelation] = {
    "same_task_follow_up": "continue",
    "refinement": "refine",
    "correction": "correct",
    "problem_report": "report_problem",
    "constraint_update": "scope_change",
    "validation": "validation_request",
}
USER_INTENT_ALIASES: dict[str, UserIntent] = {
    "update": "implement",
    "modify": "implement",
    "create": "implement",
    "build": "implement",
    "debug": "repair",
    "test": "validate",
    "analyze": "inspect",
}
NEXT_ACTION_ALIASES: dict[str, NextAction] = {
    "update": "modify",
    "modify_file": "modify",
    "create_file": "create",
    "build": "create",
    "validate": "test",
    "analyze": "inspect",
}
EXECUTION_STRATEGY_ALIASES: dict[str, ExecutionStrategy] = {
    "update": "feature_implementation",
    "modify": "feature_implementation",
    "create": "feature_implementation",
    "implement": "feature_implementation",
    "repair": "debug_repair",
    "debug": "debug_repair",
    "test": "validation_inspection",
    "validate": "validation_inspection",
    "inspect": "validation_inspection",
    "search": "validation_inspection",
    "explain": "validation_inspection",
    "plan": "validation_inspection",
}


def _compact_strings(values: list[str], *, limit: int | None = None) -> list[str]:
    unique: list[str] = []
    for raw in values:
        text = str(raw or "").strip()
        if not text or text in unique:
            continue
        unique.append(text)
    return unique[:limit] if limit is not None else unique


def _merge_artifacts(*groups: list[TaskArtifact]) -> list[TaskArtifact]:
    merged: list[TaskArtifact] = []
    seen: set[tuple[str | None, str | None, str | None, str | None]] = set()
    for group in groups:
        for item in group:
            key = (item.path, item.name, item.kind, item.role)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged[:8]


def _signal_text(
    latest_user_turn: str,
    root_goal: str,
    active_goal: str,
    output_expectation: str,
    open_problem: str | None,
    verification_target: str | None,
    constraints: list[str],
    assumptions: list[str],
    supplied_evidence: list[str],
) -> str:
    return " ".join(
        part.strip().lower()
        for part in [
            latest_user_turn,
            root_goal,
            active_goal,
            output_expectation,
            open_problem or "",
            verification_target or "",
            *constraints,
            *assumptions,
            *supplied_evidence,
        ]
        if str(part or "").strip()
    )


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _normalize_alias(value: Any, aliases: dict[str, str]) -> Any:
    text = str(value or "").strip().lower()
    if not text:
        return None
    return aliases.get(text, text)


def _infer_user_intent(
    *,
    latest_user_turn: str,
    goal_relation: GoalRelation,
    next_best_action: NextAction,
    root_goal: str,
    active_goal: str,
    output_expectation: str,
    open_problem: str | None,
    verification_target: str | None,
    constraints: list[str],
    assumptions: list[str],
    supplied_evidence: list[str],
    evidence: list[EvidenceItem],
) -> UserIntent:
    text = _signal_text(
        latest_user_turn,
        root_goal,
        active_goal,
        output_expectation,
        open_problem,
        verification_target,
        constraints,
        assumptions,
        supplied_evidence,
    )
    evidence_kinds = {item.kind for item in evidence}
    if goal_relation == "new_task" and next_best_action == "create" and not evidence_kinds:
        return "implement"
    if goal_relation in {"correct", "rollback_request", "scope_change"}:
        return "correct"
    if next_best_action == "debug" or open_problem or evidence_kinds & {"diagnostic", "test", "log"}:
        return "repair"
    if _contains_any(text, ("refactor", "cleanup", "clean up", "cleaner", "maintain", "readability", "modular")):
        return "refactor"
    if _contains_any(
        text,
        (
            "secure",
            "security",
            "safer",
            "safe",
            "sicher",
            "sicherer",
            "sicherheit",
            "absichern",
            "harden",
            "robust",
            "reliable",
            "resilien",
            "fragile",
        ),
    ):
        return "harden"
    if goal_relation == "validation_request" or next_best_action == "test":
        return "validate"
    mapping: dict[NextAction, UserIntent] = {
        "create": "implement",
        "modify": "implement",
        "inspect": "inspect",
        "search": "search",
        "explain": "explain",
        "plan": "plan",
        "clarify": "unknown",
        "test": "validate",
        "debug": "repair",
    }
    return mapping.get(next_best_action, "unknown")


def _infer_execution_strategy(
    *,
    latest_user_turn: str,
    goal_relation: GoalRelation,
    current_user_intent: UserIntent,
    next_best_action: NextAction,
    root_goal: str,
    active_goal: str,
    output_expectation: str,
    open_problem: str | None,
    verification_target: str | None,
    constraints: list[str],
    assumptions: list[str],
    supplied_evidence: list[str],
    evidence: list[EvidenceItem],
) -> ExecutionStrategy:
    text = _signal_text(
        latest_user_turn,
        root_goal,
        active_goal,
        output_expectation,
        open_problem,
        verification_target,
        constraints,
        assumptions,
        supplied_evidence,
    )
    evidence_kinds = {item.kind for item in evidence}
    if (
        goal_relation == "new_task"
        and next_best_action == "create"
        and current_user_intent in {"implement", "repair"}
        and not evidence_kinds
    ):
        return "feature_implementation"
    if goal_relation in {"correct", "rollback_request", "scope_change"} or current_user_intent == "correct":
        return "rollback_correction"
    if (
        next_best_action == "debug"
        or goal_relation == "report_problem"
        or open_problem is not None
        or evidence_kinds & {"diagnostic", "test", "log"}
    ):
        return "debug_repair"
    if current_user_intent == "refactor" or _contains_any(
        text,
        ("refactor", "cleanup", "clean up", "cleaner", "maintain", "readability", "modular"),
    ):
        return "refactor"
    if current_user_intent == "harden" or _contains_any(
        text,
        (
            "secure",
            "security",
            "safer",
            "safe",
            "sicher",
            "sicherer",
            "sicherheit",
            "absichern",
            "harden",
            "robust",
            "reliable",
            "fragile",
        ),
    ):
        return "hardening"
    if current_user_intent in {"validate", "inspect", "explain", "search", "plan"} or (
        goal_relation == "validation_request"
        or next_best_action in {"test", "inspect", "search", "explain", "plan"}
    ):
        return "validation_inspection"
    if current_user_intent in {"repair", "implement"} or next_best_action in {"create", "modify"}:
        return "feature_implementation"
    return "feature_implementation"


def _infer_user_intent_conservative(
    *,
    goal_relation: GoalRelation,
    next_best_action: NextAction,
    open_problem: str | None,
    evidence: list[EvidenceItem],
) -> UserIntent:
    evidence_kinds = {item.kind for item in evidence}
    if goal_relation == "new_task" and next_best_action == "create" and not evidence_kinds:
        return "implement"
    if goal_relation in {"correct", "rollback_request", "scope_change"}:
        return "correct"
    if next_best_action == "debug" or open_problem or evidence_kinds & {"diagnostic", "test", "log"}:
        return "repair"
    if goal_relation == "validation_request" or next_best_action == "test":
        return "validate"
    mapping: dict[NextAction, UserIntent] = {
        "create": "implement",
        "modify": "implement",
        "inspect": "inspect",
        "search": "search",
        "explain": "explain",
        "plan": "plan",
        "clarify": "unknown",
        "test": "validate",
        "debug": "repair",
    }
    return mapping.get(next_best_action, "unknown")


def _infer_execution_strategy_conservative(
    *,
    goal_relation: GoalRelation,
    current_user_intent: UserIntent | None,
    next_best_action: NextAction,
    open_problem: str | None,
    evidence: list[EvidenceItem],
) -> ExecutionStrategy | None:
    evidence_kinds = {item.kind for item in evidence}
    if (
        goal_relation == "new_task"
        and next_best_action == "create"
        and current_user_intent in {None, "implement", "repair"}
        and not evidence_kinds
    ):
        return "feature_implementation"
    if goal_relation in {"correct", "rollback_request", "scope_change"} or current_user_intent == "correct":
        return "rollback_correction"
    if (
        next_best_action == "debug"
        or goal_relation == "report_problem"
        or open_problem is not None
        or evidence_kinds & {"diagnostic", "test", "log"}
    ):
        return "debug_repair"
    if current_user_intent == "refactor":
        return "refactor"
    if current_user_intent == "harden":
        return "hardening"
    if current_user_intent in {"validate", "inspect", "explain", "search", "plan"} or (
        goal_relation == "validation_request"
        or next_best_action in {"test", "inspect", "search", "explain", "plan"}
    ):
        return "validation_inspection"
    if current_user_intent in {"repair", "implement"} or next_best_action in {"create", "modify"}:
        return "feature_implementation"
    return None


class EvidenceItem(StrictModel):
    kind: EvidenceKind = "unknown"
    summary: str
    source: str | None = None
    artifact_path: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def coerce_shorthand(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {"summary": "", "confidence": 0.0}
            return {
                "kind": "unknown",
                "summary": text,
                "confidence": 0.35,
            }
        return value

    @model_validator(mode="after")
    def normalize(self) -> EvidenceItem:
        self.summary = str(self.summary or "").strip()
        self.source = str(self.source or "").strip() or None
        self.artifact_path = str(self.artifact_path or "").strip() or None
        return self


class TaskState(StrictModel):
    latest_user_turn: str
    root_goal: str
    active_goal: str
    goal_relation: GoalRelation = "unknown"
    output_expectation: str
    current_user_intent: UserIntent | None = None
    execution_strategy: ExecutionStrategy | None = None
    open_problem: str | None = None
    verification_target: str | None = None
    target_artifacts: list[TaskArtifact] = Field(default_factory=list)
    active_artifacts: list[TaskArtifact] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    supplied_evidence: list[str] = Field(default_factory=list)
    relevant_context: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)
    ambiguity_level: Literal["low", "medium", "high"] = "medium"
    risk_level: Literal["low", "medium", "high"] = "medium"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    next_action: NextAction = "inspect"
    next_best_action: NextAction | None = None
    execution_outline: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_questions: list[str] = Field(default_factory=list)
    semantic_inference_mode: SemanticInferenceMode = Field(default="full", exclude=True, repr=False)
    semantic_resolution: SemanticResolution = "full_model"
    secondary_semantics_limited: bool = False

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if "goal_relation" in payload:
            normalized = _normalize_alias(payload.get("goal_relation"), GOAL_RELATION_ALIASES)
            if normalized:
                payload["goal_relation"] = normalized
        if "current_user_intent" in payload:
            normalized = _normalize_alias(payload.get("current_user_intent"), USER_INTENT_ALIASES)
            if normalized:
                payload["current_user_intent"] = normalized
        if "next_action" in payload:
            normalized = _normalize_alias(payload.get("next_action"), NEXT_ACTION_ALIASES)
            if normalized:
                payload["next_action"] = normalized
        if "next_best_action" in payload:
            normalized = _normalize_alias(payload.get("next_best_action"), NEXT_ACTION_ALIASES)
            if normalized:
                payload["next_best_action"] = normalized
        if "execution_strategy" in payload:
            normalized = _normalize_alias(payload.get("execution_strategy"), EXECUTION_STRATEGY_ALIASES)
            if normalized:
                payload["execution_strategy"] = normalized
        return payload

    @model_validator(mode="after")
    def normalize(self) -> TaskState:
        self.latest_user_turn = str(self.latest_user_turn or "").strip()
        self.root_goal = str(self.root_goal or "").strip()
        self.active_goal = str(self.active_goal or "").strip()
        self.output_expectation = str(self.output_expectation or "").strip()
        self.open_problem = str(self.open_problem or "").strip() or None
        self.verification_target = str(self.verification_target or "").strip() or None
        self.target_artifacts = _merge_artifacts(self.target_artifacts, self.active_artifacts)
        self.active_artifacts = _merge_artifacts(self.active_artifacts, self.target_artifacts)
        inferred_supplied_evidence = [item.summary for item in self.evidence if item.summary]
        self.supplied_evidence = _compact_strings(
            [*self.supplied_evidence, *inferred_supplied_evidence],
            limit=8,
        )
        self.relevant_context = _compact_strings(self.relevant_context, limit=8)
        self.constraints = _compact_strings(self.constraints, limit=8)
        self.assumptions = _compact_strings(self.assumptions, limit=8)
        self.missing_info = _compact_strings(self.missing_info, limit=6)
        self.execution_outline = _compact_strings(self.execution_outline, limit=6)
        self.clarification_questions = _compact_strings(self.clarification_questions, limit=3)
        self.next_best_action = self.next_best_action or self.next_action
        if self._should_prefer_feature_bootstrap():
            self.current_user_intent = "implement"
            self.execution_strategy = "feature_implementation"
            self.next_action = "create"
            self.next_best_action = "create"
        if (
            self.execution_strategy is None
            and self.current_user_intent in {
                "repair",
                "implement",
                "refactor",
                "harden",
                "validate",
                "inspect",
                "explain",
                "search",
                "plan",
            }
            and self.goal_relation not in {"correct", "scope_change", "rollback_request", "clarify"}
            and self.next_best_action != "clarify"
        ):
            self.execution_strategy = _infer_execution_strategy(
                latest_user_turn=self.latest_user_turn,
                goal_relation=self.goal_relation,
                current_user_intent=self.current_user_intent,
                next_best_action=self.next_best_action,
                root_goal=self.root_goal,
                active_goal=self.active_goal,
                output_expectation=self.output_expectation,
                open_problem=self.open_problem,
                verification_target=self.verification_target,
                constraints=self.constraints,
                assumptions=self.assumptions,
                supplied_evidence=self.supplied_evidence,
                evidence=self.evidence,
            )
        if self.needs_clarification and not self.clarification_questions:
            raise ValueError("clarification_questions must be present when needs_clarification is true")
        if not self.latest_user_turn:
            raise ValueError("latest_user_turn is required")
        if not self.root_goal:
            raise ValueError("root_goal is required")
        if not self.active_goal:
            raise ValueError("active_goal is required")
        if not self.output_expectation:
            raise ValueError("output_expectation is required")
        return self

    def _should_prefer_feature_bootstrap(self) -> bool:
        evidence_kinds = {item.kind for item in self.evidence}
        return (
            self.goal_relation == "new_task"
            and self.next_best_action == "create"
            and self.current_user_intent in {None, "implement", "repair"}
            and not evidence_kinds
            and not self.supplied_evidence
        )

    def to_task_understanding(self) -> TaskUnderstanding:
        relation_map = {
            "continue": "same_task_follow_up",
            "refine": "refinement",
            "correct": "correction",
            "report_problem": "problem_report",
            "scope_change": "constraint_update",
            "validation_request": "refinement",
            "rollback_request": "correction",
            "approval": "clarification",
            "rejection": "correction",
            "update_constraints": "constraint_update",
            "clarify": "clarification",
            "new_task": "new_task",
            "unknown": "unknown",
        }
        recommended_mode = self.next_best_action or self.next_action
        if self.execution_strategy == "debug_repair":
            intent_category = "debug"
            recommended_mode = "debug"
        elif self.execution_strategy == "refactor":
            intent_category = "refactor"
            recommended_mode = "refactor"
        elif self.execution_strategy == "hardening":
            intent_category = "configure"
            recommended_mode = "modify"
        elif self.execution_strategy == "rollback_correction":
            intent_category = "modify"
            recommended_mode = "modify"
        elif self.execution_strategy == "validation_inspection":
            intent_map = {
                "test": "test",
                "search": "search",
                "inspect": "analyze",
                "explain": "explain",
                "plan": "plan",
                "clarify": "unknown",
            }
            intent_category = intent_map.get(recommended_mode, "analyze")
        else:
            intent_map = {
                "create": "build",
                "modify": "modify",
                "debug": "debug",
                "test": "test",
                "search": "search",
                "inspect": "analyze",
                "explain": "explain",
                "plan": "plan",
                "clarify": "unknown",
            }
            intent_category = intent_map.get(recommended_mode, "modify")
        plan = [
            TaskPlanStep(
                step=index,
                summary=summary,
                action_hint=recommended_mode if index == 1 else None,
                requires_tools=recommended_mode not in {"explain", "clarify", "plan"},
            )
            for index, summary in enumerate(self.execution_outline or [self.active_goal], start=1)
        ]
        artifacts = self.active_artifacts or self.target_artifacts
        relevant_context = list(self.relevant_context)
        if self.execution_strategy:
            relevant_context.insert(0, f"Execution strategy: {self.execution_strategy}")
        if self.current_user_intent:
            relevant_context.insert(1 if self.execution_strategy else 0, f"Current user intent: {self.current_user_intent}")
        if self.open_problem:
            relevant_context.append(f"Open problem: {self.open_problem}")
        return TaskUnderstanding(
            original_request=self.latest_user_turn,
            interpreted_goal=self.active_goal,
            intent_category=intent_category,
            conversation_relation=relation_map.get(self.goal_relation, "unknown"),
            subgoals=list(self.execution_outline[:6]),
            target_artifacts=artifacts,
            relevant_context=relevant_context,
            constraints=[
                *self.constraints,
                *([f"Verification target: {self.verification_target}"] if self.verification_target else []),
            ],
            missing_info=self.missing_info,
            assumptions=self.assumptions,
            user_observations=[],
            supplied_evidence=self.supplied_evidence[:8],
            ambiguity_level=self.ambiguity_level,
            risk_level=self.risk_level,
            confidence=self.confidence,
            recommended_mode=recommended_mode,
            execution_plan=plan,
            needs_clarification=self.needs_clarification,
            clarification_questions=self.clarification_questions,
            semantic_resolution=self.semantic_resolution,
            secondary_semantics_limited=self.secondary_semantics_limited,
        )
