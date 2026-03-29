from __future__ import annotations

"""Conservative guardrails for when semantic model understanding is unavailable.

This module is intentionally not a replacement router or task reasoner. It only:
1. preserves obvious primary intent from the current turn when directly evident,
2. anchors same-task follow-ups when there is one safe active anchor, and
3. blocks or asks for clarification instead of inventing deeper semantics.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Literal

from agent.semantic_defaults import (
    extract_scope_constraints,
    has_follow_up_reference,
    infer_artifact_name_hint,
    infer_requested_extension,
    infer_scope_tokens,
    is_clear_low_risk_build_request,
    looks_like_additive_request,
    looks_like_correction_request,
    looks_like_debug_request,
    looks_like_explanation_request,
    looks_like_hardening_request,
    looks_like_problem_report,
    looks_like_scope_narrowing_request,
    looks_like_update_request,
    looks_like_validation_request,
    normalize_text,
)
from agent.semantic_runtime import SemanticResolution
from agent.task_schema import TaskArtifact, TaskUnderstanding
from agent.task_state import EvidenceItem, TaskState


GuardrailPrimaryIntent = Literal[
    "create",
    "update",
    "debug",
    "explain",
    "validate",
    "search",
    "plan",
    "unknown",
]

_PLAN_REQUEST_MARKERS = (
    "plan",
    "roadmap",
    "vorgehen",
    "naechsten schritte",
    "nächsten schritte",
    "wie wuerdest",
    "wie würdest",
)
_SEARCH_REQUEST_MARKERS = (
    "find",
    "such",
    "search",
    "locate",
    "wo ist",
    "wo steckt",
    "zeige mir wo",
)
_EXPLICIT_CHANGE_MARKERS = (
    "aender",
    "änder",
    "add",
    "change",
    "fueg",
    "füge",
    "hinzu",
    "modify",
    "update",
)
_DEICTIC_MARKERS = (
    "das",
    "da",
    "daran",
    "darauf",
    "dies",
    "dieses",
    "es",
    "hier",
    "it",
    "that",
    "this",
)
_EMPTY_WORKSPACE_CREATE_MARKERS = (
    "erstell",
    "erzeuge",
    "baue",
    "bau ",
    "build",
    "create",
    "schreib",
    "lege ",
    "ich brauche",
    "i need",
    "need a",
    "need an",
)
_PATH_RE = re.compile(
    r"([\w./-]+\.(py|js|ts|tsx|jsx|json|md|html|css|sh|toml|ya?ml|go|rs|java|kt|rb))",
    flags=re.IGNORECASE,
)
_CONVENTIONAL_ARTIFACT_PATHS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\breadme(?:\.md)?\b", flags=re.IGNORECASE), "README.md"),
)


@dataclass(frozen=True, slots=True)
class MinimalSemanticSignal:
    intent: GuardrailPrimaryIntent
    goal_relation: str
    confidence: float
    use_context: bool
    needs_clarification: bool
    requested_extension: str | None = None
    artifact_name_hint: str | None = None
    explicit_path: str | None = None


def build_minimal_task_state(
    user_input: str,
    *,
    session=None,
    snapshot=None,
    semantic_resolution: SemanticResolution = "minimal_inference",
) -> TaskState:
    request = str(user_input or "").strip() or "Unclear request"
    context = _context_anchor(session)
    signal = _minimal_semantic_signal(request, context=context)
    inferred_snapshot_targets = _snapshot_target_artifacts(request, snapshot)
    signal_targets = _target_artifacts_for_signal(
        request,
        signal=signal,
        anchor_artifacts=context["artifacts"],
    )
    if signal.intent == "update" and signal.needs_clarification and inferred_snapshot_targets:
        signal = MinimalSemanticSignal(
            intent=signal.intent,
            goal_relation=signal.goal_relation,
            confidence=max(signal.confidence, 0.6),
            use_context=signal.use_context,
            needs_clarification=False,
            requested_extension=signal.requested_extension,
            artifact_name_hint=signal.artifact_name_hint,
            explicit_path=signal.explicit_path,
        )
    anchor_artifacts = context["artifacts"]
    active_artifacts = anchor_artifacts[:6] if signal.use_context else []
    if signal.intent == "create" and any(str(item.path or "").strip() for item in signal_targets):
        target_artifacts = signal_targets
    else:
        target_artifacts = inferred_snapshot_targets or signal_targets
    if _prefer_create_in_empty_workspace(
        request,
        snapshot=snapshot,
        context=context,
        signal=signal,
        target_artifacts=target_artifacts,
    ):
        signal = MinimalSemanticSignal(
            intent="create",
            goal_relation="new_task",
            confidence=max(signal.confidence, 0.74),
            use_context=False,
            needs_clarification=False,
            requested_extension=signal.requested_extension,
            artifact_name_hint=signal.artifact_name_hint,
            explicit_path=signal.explicit_path,
        )
    constraints = extract_scope_constraints(request)
    if signal.goal_relation == "scope_change" and active_artifacts and constraints:
        filtered = _filter_artifacts_for_constraints(active_artifacts, constraints)
        if filtered:
            active_artifacts = filtered
            target_artifacts = filtered
    evidence = context["evidence"][:6] if signal.use_context else []
    root_goal = context["root_goal"] if signal.use_context and context["root_goal"] else request
    previous_goal = context["active_goal"] or context["root_goal"] or request
    current_user_intent: str | None = None
    execution_strategy: str | None = None
    open_problem: str | None = None
    verification_target: str | None = None
    risk_level = "medium"
    ambiguity_level = "low"
    next_action = "inspect"
    execution_outline: list[str] = []
    assumptions: list[str] = []
    relevant_context: list[str] = []
    missing_info: list[str] = []
    output_expectation = "Clarify the exact target before acting."
    active_goal = request

    if signal.use_context and context["root_goal"]:
        relevant_context.append(f"Existing root goal: {context['root_goal']}")
    if signal.use_context and active_artifacts:
        assumptions.append("Current-turn wording is treated as a same-task reference only because there is one safe active anchor.")
    elif inferred_snapshot_targets:
        assumptions.append("The target artifacts were inferred conservatively from strong lexical overlap between the request and workspace filenames.")

    if signal.needs_clarification:
        current_user_intent = "unknown"
        ambiguity_level = "high"
        next_action = "clarify"
        missing_info.append(_clarification_missing_info(signal, context=context))
        execution_outline = ["Ask for the exact artifact, scope, or task boundary before acting."]
    elif signal.intent == "create":
        current_user_intent = "implement"
        execution_strategy = "feature_implementation"
        next_action = "create"
        risk_level = "low"
        output_expectation = (
            "Create a small runnable implementation with a conventional default artifact and minimal scope."
        )
        verification_target = (
            "Create the initial implementation and run the most relevant validation or entry command."
        )
        execution_outline = [
            "Choose the smallest conventional artifact or scaffold that fits the request.",
            "Implement the requested behavior in minimal runnable scope.",
            "Validate the created artifact with the most relevant command if available.",
        ]
        if signal.goal_relation == "refine":
            active_goal = f"Extend the active task by adding the requested artifact or surface around {previous_goal}."
            assumptions.append("The new artifact should stay aligned with the active task rather than replacing it.")
        else:
            active_goal = request
            assumptions.append("A conventional default artifact is acceptable unless the workspace strongly suggests another entrypoint.")
    elif signal.intent == "debug":
        current_user_intent = "repair"
        execution_strategy = "debug_repair"
        next_action = "debug"
        active_goal = (
            f"Diagnose and resolve the current issue affecting {previous_goal}."
            if signal.use_context
            else "Diagnose and fix the issue described in the latest request."
        )
        output_expectation = (
            "Diagnose the issue, apply the smallest safe fix if warranted, and report the verification result honestly."
        )
        open_problem = _open_problem_summary(request, evidence)
        verification_target = (
            "Reproduce the failing path, apply the smallest safe fix, and rerun the most relevant validation."
        )
        execution_outline = [
            "Inspect the strongest artifact or evidence first.",
            "Reproduce or validate the failing path before changing code.",
            "Apply the smallest fix supported by the evidence and rerun verification.",
        ]
    elif signal.intent == "explain":
        current_user_intent = "explain"
        execution_strategy = "validation_inspection"
        next_action = "explain"
        risk_level = "low"
        active_goal = (
            f"Explain the active task state around {previous_goal}."
            if signal.use_context
            else request
        )
        output_expectation = "Explain the relevant behavior or code path clearly and honestly."
        execution_outline = [
            "Inspect the most relevant context if needed.",
            "Explain the result in clear user-facing language.",
        ]
    elif signal.intent == "validate":
        current_user_intent = "validate"
        execution_strategy = "validation_inspection"
        next_action = "test"
        active_goal = previous_goal if signal.use_context else request
        output_expectation = (
            "Run the most relevant validation for the active implementation and report the result honestly."
        )
        verification_target = output_expectation
        execution_outline = [
            "Choose the most relevant validation command.",
            "Run the validation and capture concrete failure output.",
            "Report whether validation passed, failed, or was blocked.",
        ]
    elif signal.intent == "plan":
        current_user_intent = "plan"
        execution_strategy = "validation_inspection"
        next_action = "plan"
        risk_level = "low"
        active_goal = previous_goal if signal.use_context else request
        output_expectation = "Provide a practical implementation plan."
        execution_outline = [
            "Summarize the concrete goal and scope.",
            "List the smallest sensible implementation steps.",
            "Call out risks, assumptions, and validation targets.",
        ]
    elif signal.intent == "search":
        current_user_intent = "search"
        execution_strategy = "validation_inspection"
        next_action = "search"
        risk_level = "low"
        active_goal = previous_goal if signal.use_context else request
        output_expectation = "Locate the relevant implementation area in the workspace."
        execution_outline = [
            "Search the workspace for the best semantic match.",
            "Inspect the strongest candidate before concluding.",
        ]
    elif signal.intent == "update":
        current_user_intent = "implement"
        execution_strategy = "feature_implementation"
        next_action = "modify"
        active_goal = (
            f"Continue the active implementation around {previous_goal} with the latest requested change."
            if signal.use_context
            else request
        )
        output_expectation = (
            "Apply the requested change with the smallest safe scope and verify the updated behavior honestly."
        )
        verification_target = (
            "Apply the requested change and rerun the most relevant validation for the updated behavior."
        )
        execution_outline = [
            "Inspect the active implementation before editing.",
            "Apply the focused change in the smallest sensible scope.",
            "Verify that the updated behavior still matches the requested goal.",
        ]
    else:
        next_action = "clarify"
        ambiguity_level = "high"
        missing_info.append(_clarification_missing_info(signal, context=context))
        execution_outline = ["Ask for the exact artifact, scope, or task boundary before acting."]

    if signal.goal_relation == "scope_change":
        current_user_intent = "correct"
        active_goal = _scope_change_goal(previous_goal, constraints)
        next_action = "modify"
        output_expectation = _scope_change_outcome(constraints)
        verification_target = _scope_change_verification(constraints)
        execution_strategy = None
        execution_outline = [
            "Inspect the active task boundary and the narrowed scope.",
            "Apply the requested change only inside that scope.",
            "Verify that only the constrained surface was affected.",
        ]
    elif signal.goal_relation == "correct":
        current_user_intent = "correct"
        next_action = "modify"
        execution_strategy = None
        active_goal = previous_goal
        output_expectation = "Correct the current scope or target boundary without widening the task."
        verification_target = "Apply the narrowest correction and verify the corrected boundary."
        execution_outline = [
            "Inspect the current implementation boundary.",
            "Apply the narrowest correction that matches the latest instruction.",
            "Verify that the corrected scope now matches the user intent.",
        ]

    confidence = max(min(signal.confidence, 0.95), 0.0)
    clarification_questions = (
        [_clarification_question(signal, context=context)]
        if signal.needs_clarification or next_action == "clarify"
        else []
    )

    return TaskState(
        latest_user_turn=request,
        root_goal=root_goal,
        active_goal=active_goal,
        goal_relation=signal.goal_relation,
        output_expectation=output_expectation,
        current_user_intent=current_user_intent,
        execution_strategy=execution_strategy,
        open_problem=open_problem,
        verification_target=verification_target,
        target_artifacts=target_artifacts[:6],
        active_artifacts=active_artifacts[:6],
        evidence=evidence,
        relevant_context=relevant_context[:8],
        constraints=constraints[:4],
        assumptions=assumptions[:8],
        missing_info=missing_info[:6],
        ambiguity_level=ambiguity_level,
        risk_level=risk_level,
        confidence=confidence,
        next_action=next_action,
        next_best_action=next_action,
        execution_outline=execution_outline[:6],
        needs_clarification=bool(clarification_questions),
        clarification_questions=clarification_questions[:3],
        semantic_inference_mode="conservative",
        semantic_resolution=semantic_resolution,
        secondary_semantics_limited=True,
    )


def build_minimal_task_understanding(
    user_input: str,
    *,
    session=None,
    snapshot=None,
    semantic_resolution: SemanticResolution = "minimal_inference",
) -> TaskUnderstanding:
    state = build_minimal_task_state(
        user_input,
        session=session,
        snapshot=snapshot,
        semantic_resolution=semantic_resolution,
    )
    understanding = state.to_task_understanding()
    understanding.semantic_resolution = semantic_resolution
    understanding.secondary_semantics_limited = True
    return understanding


def build_minimal_router_output(
    user_input: str,
    *,
    session=None,
    snapshot=None,
    logger=None,
    semantic_resolution: SemanticResolution = "minimal_inference",
):
    from agent.decision import ExecutionDecisionPolicy

    task_state = build_minimal_task_state(
        user_input,
        session=session,
        snapshot=snapshot,
        semantic_resolution=semantic_resolution,
    )
    return ExecutionDecisionPolicy(logger=logger).build_route(
        task_state,
        snapshot=snapshot,
        session=session,
    )


def _context_anchor(session) -> dict[str, Any]:
    root_goal = ""
    active_goal = ""
    requested_outcome = ""
    artifacts = _collect_context_artifacts(session)
    evidence = _collect_context_evidence(session)
    if session is not None and getattr(session, "task_state", None) is not None:
        task_state = session.task_state
        root_goal = str(task_state.root_goal or "").strip()
        active_goal = str(task_state.active_goal or "").strip()
        requested_outcome = str(task_state.output_expectation or "").strip()
    follow_up = getattr(session, "follow_up_context", None) if session is not None else None
    if follow_up is not None:
        root_goal = root_goal or str(follow_up.previous_root_goal or "").strip()
        active_goal = active_goal or str(
            follow_up.previous_active_goal
            or follow_up.previous_interpreted_goal
            or follow_up.previous_task
            or ""
        ).strip()
        requested_outcome = requested_outcome or str(follow_up.previous_requested_outcome or "").strip()
    return {
        "root_goal": root_goal,
        "active_goal": active_goal,
        "requested_outcome": requested_outcome,
        "artifacts": artifacts,
        "artifact_paths": [item.path for item in artifacts if item.path][:6],
        "evidence": evidence,
    }


def _minimal_semantic_signal(request: str, *, context: dict[str, Any]) -> MinimalSemanticSignal:
    normalized = normalize_text(request)
    explicit_path = _extract_explicit_path(request)
    requested_extension = infer_requested_extension(request)
    artifact_name_hint = infer_artifact_name_hint(request)
    explicit_follow_up = has_follow_up_reference(request)
    deictic = _looks_like_deictic_request(normalized)
    scope_change = looks_like_scope_narrowing_request(request)
    correction = looks_like_correction_request(request)
    anchor_paths = context["artifact_paths"]
    single_anchor = len(anchor_paths) == 1
    multiple_anchors = len(anchor_paths) > 1

    create_like = is_clear_low_risk_build_request(request)
    update_like = (
        looks_like_update_request(request)
        or looks_like_hardening_request(request)
        or looks_like_additive_request(request)
        or scope_change
        or correction
        or (single_anchor and _has_explicit_change_marker(normalized))
    )
    validate_like = looks_like_validation_request(request) and not looks_like_debug_request(request)

    intent: GuardrailPrimaryIntent
    if looks_like_explanation_request(request):
        intent = "explain"
    elif _looks_like_plan_request(normalized):
        intent = "plan"
    elif looks_like_debug_request(request):
        intent = "debug"
    elif _looks_like_search_request(normalized):
        intent = "search"
    elif create_like:
        intent = "create"
    elif update_like:
        intent = "update"
    elif validate_like:
        intent = "validate"
    else:
        intent = "unknown"

    if multiple_anchors and not explicit_path and (explicit_follow_up or deictic or correction) and not scope_change:
        return MinimalSemanticSignal(
            intent=intent if intent != "unknown" else "unknown",
            goal_relation="clarify",
            confidence=0.28,
            use_context=False,
            needs_clarification=True,
            requested_extension=requested_extension,
            artifact_name_hint=artifact_name_hint,
            explicit_path=explicit_path,
        )

    if scope_change and (single_anchor or explicit_follow_up or explicit_path or context["root_goal"]):
        return MinimalSemanticSignal(
            intent="update",
            goal_relation="scope_change",
            confidence=0.66,
            use_context=bool(single_anchor or explicit_path or context["root_goal"]),
            needs_clarification=False,
            requested_extension=requested_extension,
            artifact_name_hint=artifact_name_hint,
            explicit_path=explicit_path,
        )
    if correction and (single_anchor or explicit_follow_up or explicit_path):
        return MinimalSemanticSignal(
            intent="update",
            goal_relation="correct",
            confidence=0.64,
            use_context=bool(single_anchor or explicit_path or context["root_goal"]),
            needs_clarification=False,
            requested_extension=requested_extension,
            artifact_name_hint=artifact_name_hint,
            explicit_path=explicit_path,
        )

    if intent == "create":
        relation = "refine" if explicit_follow_up and single_anchor else "new_task"
        confidence = 0.82 if relation == "refine" else 0.8
        return MinimalSemanticSignal(
            intent=intent,
            goal_relation=relation,
            confidence=confidence,
            use_context=relation != "new_task",
            needs_clarification=False,
            requested_extension=requested_extension,
            artifact_name_hint=artifact_name_hint,
            explicit_path=explicit_path,
        )

    if intent == "debug":
        use_context = bool(single_anchor and (explicit_follow_up or deictic) and not explicit_path)
        relation = "report_problem" if use_context or looks_like_problem_report(request) else "new_task"
        return MinimalSemanticSignal(
            intent=intent,
            goal_relation=relation,
            confidence=0.76 if explicit_path or use_context else 0.68,
            use_context=use_context,
            needs_clarification=False,
            requested_extension=requested_extension,
            artifact_name_hint=artifact_name_hint,
            explicit_path=explicit_path,
        )

    if intent in {"explain", "validate", "search", "plan"}:
        use_context = bool(single_anchor and (explicit_follow_up or deictic) and not explicit_path)
        relation = "validation_request" if intent == "validate" and use_context else "continue" if use_context else "new_task"
        confidence = 0.76 if intent in {"explain", "search", "plan"} else 0.72
        return MinimalSemanticSignal(
            intent=intent,
            goal_relation=relation,
            confidence=confidence,
            use_context=use_context,
            needs_clarification=False,
            requested_extension=requested_extension,
            artifact_name_hint=artifact_name_hint,
            explicit_path=explicit_path,
        )

    if intent == "update":
        use_context = bool(
            single_anchor
            and (
                explicit_follow_up
                or deictic
                or looks_like_additive_request(request)
                or looks_like_hardening_request(request)
                or correction
                or scope_change
            )
            and not explicit_path
        )
        relation = (
            "refine"
            if use_context and (looks_like_additive_request(request) or looks_like_hardening_request(request))
            else "continue"
            if use_context
            else "new_task"
        )
        confidence = 0.7 if explicit_path else 0.58 if use_context else 0.34
        return MinimalSemanticSignal(
            intent=intent,
            goal_relation=relation,
            confidence=confidence,
            use_context=use_context,
            needs_clarification=not (explicit_path or use_context),
            requested_extension=requested_extension,
            artifact_name_hint=artifact_name_hint,
            explicit_path=explicit_path,
        )

    if (
        intent == "unknown"
        and single_anchor
        and (explicit_follow_up or deictic)
        and any(item.kind in {"diagnostic", "test", "log"} for item in context["evidence"])
    ):
        return MinimalSemanticSignal(
            intent="debug",
            goal_relation="report_problem",
            confidence=0.52,
            use_context=True,
            needs_clarification=False,
            requested_extension=requested_extension,
            artifact_name_hint=artifact_name_hint,
            explicit_path=explicit_path,
        )

    return MinimalSemanticSignal(
        intent="unknown",
        goal_relation="clarify",
        confidence=0.24,
        use_context=False,
        needs_clarification=True,
        requested_extension=requested_extension,
        artifact_name_hint=artifact_name_hint,
        explicit_path=explicit_path,
    )


def _target_artifacts_for_signal(
    request: str,
    *,
    signal: MinimalSemanticSignal,
    anchor_artifacts: list[TaskArtifact],
) -> list[TaskArtifact]:
    explicit_paths = _extract_explicit_paths(request)
    if explicit_paths:
        has_non_doc_primary = any(
            not _looks_like_test_artifact(explicit_path)
            and Path(explicit_path).suffix.lower() not in {".md", ".markdown", ".rst", ".txt"}
            for explicit_path in explicit_paths
        )
        artifacts: list[TaskArtifact] = []
        for index, explicit_path in enumerate(explicit_paths[:6]):
            if _looks_like_test_artifact(explicit_path):
                role = "validation_target"
            elif has_non_doc_primary and Path(explicit_path).suffix.lower() in {".md", ".markdown", ".rst", ".txt"}:
                role = "supporting_context"
            else:
                role = "primary_target"
            artifacts.append(
                TaskArtifact(
                    path=explicit_path,
                    name=Path(explicit_path).name,
                    kind="test" if _looks_like_test_artifact(explicit_path) else Path(explicit_path).suffix.lower() or "file",
                    role=role,
                    confidence=0.86 if index == 0 else 0.8,
                )
            )
        return artifacts
    if signal.intent == "create":
        name = signal.artifact_name_hint or _default_artifact_name(signal.requested_extension)
        if not name:
            return []
        return [
            TaskArtifact(
                path=None,
                name=name,
                kind=signal.requested_extension or "artifact",
                role="primary_target",
                confidence=0.66 if signal.goal_relation == "new_task" else 0.72,
            )
        ]
    if signal.use_context and anchor_artifacts:
        return anchor_artifacts[:6]
    if signal.intent in {"debug", "update", "explain", "validate"}:
        explicit = _extract_explicit_name(request)
        if explicit:
            return [
                TaskArtifact(
                    path=None,
                    name=explicit,
                    kind="artifact",
                    role="primary_target",
                    confidence=0.52,
                )
            ]
    return []


def _collect_context_artifacts(session) -> list[TaskArtifact]:
    artifacts: list[TaskArtifact] = []
    seen: set[str] = set()

    def add(path: str | None, *, role: str, confidence: float) -> None:
        text = str(path or "").strip()
        if not text or text in seen:
            return
        seen.add(text)
        artifacts.append(
            TaskArtifact(
                path=text,
                name=Path(text).name,
                kind=Path(text).suffix.lower() or "file",
                role=role,
                confidence=confidence,
            )
        )

    if session is None:
        return artifacts
    for item in getattr(session, "changed_files", [])[-4:]:
        add(getattr(item, "path", None), role="active_context", confidence=0.68)
    task_state = getattr(session, "task_state", None)
    if task_state is not None:
        for artifact in task_state.target_artifacts[:6]:
            add(artifact.path or artifact.name, role=artifact.role or "active_context", confidence=max(artifact.confidence, 0.72))
    follow_up = getattr(session, "follow_up_context", None)
    if follow_up is not None:
        for path in follow_up.target_paths[:6]:
            add(path, role="active_context", confidence=0.64)
        for path in follow_up.changed_files[:6]:
            add(path, role="active_context", confidence=0.62)
        for path in follow_up.read_files[:4]:
            add(path, role="supporting_context", confidence=0.56)
    for path in getattr(session, "candidate_files", [])[:6]:
        add(path, role="supporting_context", confidence=0.5)
    return artifacts[:8]


def _collect_context_evidence(session) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    seen: set[tuple[str, str, str]] = set()

    def add(item: EvidenceItem) -> None:
        key = (item.kind, item.summary, item.artifact_path or "")
        if key in seen:
            return
        seen.add(key)
        evidence.append(item)

    if session is None:
        return evidence
    task_state = getattr(session, "task_state", None)
    if task_state is not None:
        for item in task_state.evidence[:6]:
            add(item)
    follow_up = getattr(session, "follow_up_context", None)
    if follow_up is not None:
        for item in follow_up.diagnostics[-6:]:
            add(
                EvidenceItem(
                    kind="diagnostic",
                    summary=item.summary,
                    source=item.source,
                    artifact_path=item.file_hints[0] if item.file_hints else None,
                    confidence=0.82,
                )
            )
        for run in follow_up.validation_runs[-4:]:
            if run.status not in {"failed", "timeout", "blocked"}:
                continue
            add(
                EvidenceItem(
                    kind="test",
                    summary=run.excerpt or run.summary or f"{run.command} failed",
                    source=run.command,
                    confidence=0.78,
                )
            )
        if follow_up.last_error:
            add(
                EvidenceItem(
                    kind="diagnostic",
                    summary=follow_up.last_error,
                    source="follow_up_context",
                    confidence=0.8,
                )
            )
    return evidence[:8]


def _clarification_missing_info(signal: MinimalSemanticSignal, *, context: dict[str, Any]) -> str:
    if len(context["artifact_paths"]) > 1:
        return "There are multiple active artifacts and the current target cannot be inferred safely."
    if signal.intent in {"update", "debug"}:
        return "The concrete artifact or failing path is still ambiguous."
    return "The exact target is still ambiguous."


def _clarification_question(signal: MinimalSemanticSignal, *, context: dict[str, Any]) -> str:
    if len(context["artifact_paths"]) > 1:
        return "Welche der aktuell aktiven Dateien oder Bereiche meinst du genau?"
    if signal.intent == "debug":
        return "Welche Datei, welchen Fehlerpfad oder welchen aktiven Task soll ich genau debuggen?"
    if signal.intent == "update":
        return "Welche bestehende Datei oder welchen Bereich soll ich genau aendern?"
    return "Welchen konkreten Bereich oder welches Artefakt soll ich als naechstes bearbeiten?"


def _scope_change_goal(previous_goal: str, constraints: list[str]) -> str:
    if "Backend only." in constraints:
        return f"Continue {previous_goal} only in the backend scope."
    if "Frontend only." in constraints:
        return f"Continue {previous_goal} only in the frontend scope."
    return previous_goal


def _scope_change_outcome(constraints: list[str]) -> str:
    if "Backend only." in constraints:
        return "A backend-scoped change that leaves frontend surfaces untouched."
    if "Frontend only." in constraints:
        return "A frontend-scoped change that leaves backend surfaces untouched."
    return "A narrowed-scope change aligned with the latest user instruction."


def _scope_change_verification(constraints: list[str]) -> str:
    if "Backend only." in constraints:
        return "Only backend artifacts should be updated and verified."
    if "Frontend only." in constraints:
        return "Only frontend artifacts should be updated and verified."
    return "Only artifacts inside the narrowed scope should be updated and verified."


def _filter_artifacts_for_constraints(
    artifacts: list[TaskArtifact],
    constraints: list[str],
) -> list[TaskArtifact]:
    backend_only = "Backend only." in constraints
    frontend_only = "Frontend only." in constraints
    if not backend_only and not frontend_only:
        return artifacts
    filtered: list[TaskArtifact] = []
    for artifact in artifacts:
        candidate = str(artifact.path or artifact.name or "").lower()
        if backend_only and any(token in candidate for token in ("backend", "api", "server", "auth.py")):
            filtered.append(artifact)
        if frontend_only and any(token in candidate for token in ("frontend", "ui", "client", "web", ".html", ".css")):
            filtered.append(artifact)
    return filtered or artifacts


def _open_problem_summary(request: str, evidence: list[EvidenceItem]) -> str:
    for item in evidence:
        if item.summary:
            return item.summary
    return request

def _looks_like_plan_request(normalized: str) -> bool:
    return _contains_marker_phrase(normalized, _PLAN_REQUEST_MARKERS)


def _looks_like_search_request(normalized: str) -> bool:
    return _contains_marker_phrase(normalized, _SEARCH_REQUEST_MARKERS)


def _contains_marker_phrase(normalized: str, markers: tuple[str, ...]) -> bool:
    if not normalized:
        return False
    token_space = f" {re.sub(r'[^a-z0-9_äöüß]+', ' ', normalized).strip()} "
    if token_space == "  ":
        return False
    for marker in markers:
        marker_tokens = re.sub(r"[^a-z0-9_äöüß]+", " ", str(marker or "").strip().lower()).strip()
        if marker_tokens and f" {marker_tokens} " in token_space:
            return True
    return False


def _has_explicit_change_marker(normalized: str) -> bool:
    tokens = [token for token in re.split(r"[^a-z0-9_äöüß]+", normalized) if token]
    return any(token in _EXPLICIT_CHANGE_MARKERS for token in tokens)


def _looks_like_deictic_request(normalized: str) -> bool:
    if not normalized:
        return False
    tokens = [token for token in re.split(r"[^a-z0-9_äöüß]+", normalized) if token]
    return any(token in _DEICTIC_MARKERS for token in tokens) or normalized.startswith(("mach weiter", "do that", "continue"))


def _extract_explicit_path(text: str) -> str | None:
    paths = _extract_explicit_paths(text)
    return paths[0] if paths else None


def _extract_explicit_paths(text: str) -> list[str]:
    paths: list[str] = []
    for match in _PATH_RE.finditer(str(text or "")):
        candidate = match.group(1).lstrip("./")
        if candidate and candidate not in paths:
            paths.append(candidate)
    source_text = str(text or "")
    for pattern, conventional_path in _CONVENTIONAL_ARTIFACT_PATHS:
        if not pattern.search(source_text):
            continue
        if conventional_path in paths:
            continue
        insert_at = next(
            (index for index, candidate in enumerate(paths) if _looks_like_test_artifact(candidate)),
            len(paths),
        )
        paths.insert(insert_at, conventional_path)
    return paths[:8]


def _extract_explicit_name(text: str) -> str | None:
    candidate = infer_artifact_name_hint(text)
    return candidate if candidate and len(candidate) >= 3 else None


def _looks_like_test_artifact(path: str) -> bool:
    lowered = str(path or "").lower()
    name = Path(lowered).name
    return "/tests/" in f"/{lowered}" or name.startswith("test_") or name.endswith("_test.py")


def _is_non_actionable_test_support_path(path: str) -> bool:
    text = str(path or "").strip()
    if not text:
        return False
    normalized = text.replace("\\", "/")
    return Path(normalized).name.lower() == "__init__.py" and "/tests/" in f"/{normalized.lower()}"


def _snapshot_target_artifacts(request: str, snapshot) -> list[TaskArtifact]:
    if snapshot is None:
        return []

    request_tokens = [token for token in infer_scope_tokens(request) if len(token) >= 3]
    if not request_tokens:
        return []
    request_lower = str(request or "").lower()
    request_space = f" {re.sub(r'[^0-9a-zäöüß]+', ' ', request_lower).strip()} "
    explicit_request_paths = _explicit_snapshot_request_paths(request_lower, request_space, snapshot)
    request_targets_cli_entrypoint = _request_targets_package_cli_entrypoint(request_lower)
    request_targets_tests = _request_targets_existing_tests(request_lower)
    request_targets_runtime_repair = request_targets_tests and looks_like_debug_request(request)

    candidate_paths: list[str] = []
    for group in (
        getattr(snapshot, "focus_files", []),
        getattr(snapshot, "important_files", []),
        getattr(snapshot, "manifests", []),
        getattr(snapshot, "test_files", []),
        getattr(snapshot, "entrypoints", []),
    ):
        for item in group[:12]:
            path = str(item or "").strip()
            if _is_non_actionable_test_support_path(path):
                continue
            if path and path not in candidate_paths:
                candidate_paths.append(path)

    scored: list[tuple[float, str]] = []
    for path in candidate_paths:
        normalized_path = normalize_text(path).replace("_", " ")
        path_tokens = [token for token in re.split(r"[^a-z0-9äöüß]+", normalized_path) if token]
        score = 0.0
        for request_token in request_tokens:
            for path_token in path_tokens:
                if request_token == path_token:
                    score += 2.0
                elif len(request_token) >= 4 and (request_token in path_token or path_token in request_token):
                    score += 1.0
        basename = Path(path).name.lower()
        if request_targets_cli_entrypoint and basename == "__main__.py":
            score += 2.5
        if path in explicit_request_paths:
            score += 1.5
        if score >= 1.0:
            scored.append((score, path))

    if not scored:
        fallback_paths = explicit_request_paths[:4]
        return _task_artifacts_for_paths(snapshot, fallback_paths)

    best_score = max(score for score, _ in scored)
    entrypoints = set(getattr(snapshot, "entrypoints", []) or [])
    manifests = set(getattr(snapshot, "manifests", []) or [])
    test_files = {
        str(path or "").strip()
        for path in (getattr(snapshot, "test_files", []) or [])
        if str(path or "").strip() and not _is_non_actionable_test_support_path(str(path or "").strip())
    }

    def path_priority(path: str) -> tuple[int, str]:
        suffix = Path(path).suffix.lower()
        if request_targets_cli_entrypoint and Path(path).name.lower() == "__main__.py":
            return (-1, path)
        if path in entrypoints:
            return (0, path)
        if path in manifests or suffix in {".md", ".rst", ".txt"}:
            return (1, path)
        if path in test_files or "/tests/" in f"/{path}":
            return (2, path)
        return (0, path)

    selected = [
        path
        for score, path in sorted(scored, key=lambda item: (*path_priority(item[1]), -item[0]))
        if score >= max(1.0, best_score - 1.0)
    ]
    if request_targets_cli_entrypoint:
        implementation_like = [
            path
            for _, path in sorted(scored, key=lambda item: (*path_priority(item[1]), -item[0]))
            if path not in manifests and path not in test_files and "/tests/" not in f"/{path}"
        ]
        selected = [*implementation_like[:2], *selected]
    elif request_targets_runtime_repair:
        implementation_like = _runtime_repair_implementation_paths(snapshot, selected)
        implementation_like = [
            *implementation_like,
            *[
                path
            for _, path in sorted(scored, key=lambda item: (*path_priority(item[1]), -item[0]))
            if path not in manifests and path not in test_files and "/tests/" not in f"/{path}"
            ],
        ]
        selected = [*implementation_like[:2], *selected]
    if request_targets_tests:
        relevant_tests = _relevant_snapshot_test_paths(
            scored=scored,
            snapshot=snapshot,
            request_tokens=request_tokens,
            selected=selected,
        )
        selected = [*selected, *relevant_tests[:2]]
    explicit_support_paths = [
        path
        for path in explicit_request_paths
        if path in manifests or Path(path).suffix.lower() in {".md", ".rst", ".txt"}
    ]
    selected_non_tests = [
        path
        for path in selected
        if path not in test_files and "/tests/" not in f"/{path}"
    ]
    selected_tests = [
        path
        for path in selected
        if path in test_files or "/tests/" in f"/{path}"
    ]
    ordered_paths: list[str] = []
    for candidate in [
        *selected_non_tests,
        *explicit_support_paths,
        *selected_tests,
        *explicit_request_paths,
    ]:
        if candidate and candidate not in ordered_paths:
            ordered_paths.append(candidate)
    selected = ordered_paths[:4]
    return _task_artifacts_for_paths(snapshot, selected)


def _task_artifacts_for_paths(snapshot, selected: list[str]) -> list[TaskArtifact]:
    artifacts: list[TaskArtifact] = []
    manifests = set(getattr(snapshot, "manifests", []) or [])
    test_files = set(getattr(snapshot, "test_files", []) or [])
    for path in selected:
        confidence = 0.58
        role = "primary_target"
        suffix = Path(path).suffix.lower()
        if path in test_files or "/tests/" in f"/{path}":
            role = "validation_target"
        elif path in manifests or suffix in {".md", ".rst", ".txt"}:
            role = "supporting_context"
        artifacts.append(
            TaskArtifact(
                path=path,
                name=Path(path).name,
                kind=Path(path).suffix.lower() or "file",
                role=role,
                confidence=confidence,
            )
        )
    return artifacts


def _explicit_snapshot_request_paths(
    request_lower: str,
    request_space: str,
    snapshot,
) -> list[str]:
    candidate_paths: list[str] = []
    for group in (
        getattr(snapshot, "focus_files", []),
        getattr(snapshot, "important_files", []),
        getattr(snapshot, "manifests", []),
        getattr(snapshot, "test_files", []),
        getattr(snapshot, "entrypoints", []),
    ):
        for item in group[:12]:
            path = str(item or "").strip()
            if path and path not in candidate_paths:
                candidate_paths.append(path)

    generic_stems = {"app", "cli", "doc", "docs", "guide", "index", "main", "test", "tests"}
    explicit: list[str] = []
    for path in candidate_paths:
        path_lower = path.lower()
        basename = Path(path).name.lower()
        stem = Path(path).stem.lower()
        normalized_path = re.sub(r"[^0-9a-zäöüß]+", " ", path_lower).strip()
        normalized_basename = re.sub(r"[^0-9a-zäöüß]+", " ", basename).strip()
        normalized_stem = re.sub(r"[^0-9a-zäöüß]+", " ", stem).strip()

        matches_request = False
        if basename and basename in request_lower:
            matches_request = True
        elif normalized_path and f" {normalized_path} " in request_space:
            matches_request = True
        elif normalized_basename and f" {normalized_basename} " in request_space:
            matches_request = True
        elif (
            normalized_stem
            and normalized_stem not in generic_stems
            and f" {normalized_stem} " in request_space
        ):
            matches_request = True

        if matches_request and path not in explicit:
            explicit.append(path)
    return explicit


def _request_targets_package_cli_entrypoint(request_lower: str) -> bool:
    lowered = str(request_lower or "").lower()
    if "python -m" in lowered or "python3 -m" in lowered:
        return True
    if "command line" in lowered or "kommandozeile" in lowered:
        return True
    if "argparse" in lowered:
        return True
    if " cli " in f" {re.sub(r'[^0-9a-zäöüß]+', ' ', lowered).strip()} ":
        return True
    return bool(re.search(r"(^|[^0-9a-z])--[a-z0-9][\\w-]*", lowered))


def _request_targets_existing_tests(request_lower: str) -> bool:
    lowered = str(request_lower or "").lower()
    if any(marker in lowered for marker in ["unittest", "pytest", "unit test", "unit tests", "testcase", "test case"]):
        return True
    normalized = f" {re.sub(r'[^0-9a-zäöüß]+', ' ', lowered).strip()} "
    return " tests " in normalized or " test " in normalized


def _relevant_snapshot_test_paths(
    *,
    scored: list[tuple[float, str]],
    snapshot,
    request_tokens: list[str],
    selected: list[str],
) -> list[str]:
    test_files = [
        str(path or "").strip()
        for path in getattr(snapshot, "test_files", []) or []
        if str(path or "").strip() and not _is_non_actionable_test_support_path(str(path or "").strip())
    ]
    if not test_files:
        return []

    scored_tests = [
        path
        for _, path in sorted(scored, key=lambda item: -item[0])
        if path in test_files or "/tests/" in f"/{path}"
    ]
    if scored_tests:
        return scored_tests

    selected_tokens: set[str] = set()
    for candidate in selected:
        normalized = normalize_text(candidate).replace("_", " ")
        selected_tokens.update(token for token in re.split(r"[^a-z0-9äöüß]+", normalized) if len(token) >= 3)
    selected_tokens.update(token for token in request_tokens if len(token) >= 3)

    ranked: list[tuple[int, str]] = []
    for path in test_files:
        normalized = normalize_text(path).replace("_", " ")
        tokens = {token for token in re.split(r"[^a-z0-9äöüß]+", normalized) if len(token) >= 3}
        overlap = len(tokens & selected_tokens)
        ranked.append((overlap, path))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [path for _, path in ranked]


def _runtime_repair_implementation_paths(snapshot, selected: list[str]) -> list[str]:
    candidate_paths: list[str] = []
    for group in (
        getattr(snapshot, "focus_files", []),
        getattr(snapshot, "important_files", []),
        getattr(snapshot, "entrypoints", []),
    ):
        for item in group[:12]:
            path = str(item or "").strip()
            if path and path not in candidate_paths:
                candidate_paths.append(path)

    if not candidate_paths:
        return []

    requested_stems: list[str] = []
    for path in selected:
        name = Path(path).name
        stem = Path(name).stem
        if stem.startswith("test_") and len(stem) > 5:
            requested_stems.append(stem.removeprefix("test_"))
        elif stem.endswith("_test") and len(stem) > 5:
            requested_stems.append(stem.removesuffix("_test"))

    ordered: list[str] = []
    for candidate in candidate_paths:
        if candidate in ordered or _looks_like_test_artifact(candidate):
            continue
        candidate_stem = Path(candidate).stem.lower()
        if any(stem and candidate_stem == stem.lower() for stem in requested_stems):
            ordered.append(candidate)
    return ordered


def _prefer_create_in_empty_workspace(
    request: str,
    *,
    snapshot,
    context: dict[str, Any],
    signal: MinimalSemanticSignal,
    target_artifacts: list[TaskArtifact],
) -> bool:
    if snapshot is None or getattr(snapshot, "file_count", 0) != 0:
        return False
    if context.get("artifact_paths"):
        return False
    if signal.goal_relation != "new_task" or signal.needs_clarification:
        return False
    if signal.intent in {"debug", "validate", "search", "plan", "explain"}:
        return False
    normalized = normalize_text(request)
    has_create_signal = any(marker in normalized for marker in _EMPTY_WORKSPACE_CREATE_MARKERS)
    explicit_target = any(
        Path(str(artifact.path or artifact.name or "")).suffix
        for artifact in target_artifacts
        if str(artifact.path or artifact.name or "").strip()
    ) or infer_requested_extension(request) is not None
    return has_create_signal and explicit_target


def _default_artifact_name(extension: str | None) -> str | None:
    defaults = {
        ".py": "main",
        ".js": "app",
        ".ts": "app",
        ".tsx": "app",
        ".html": "index",
        ".css": "styles",
    }
    return defaults.get(extension or "")
