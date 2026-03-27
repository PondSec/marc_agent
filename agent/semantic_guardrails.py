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
_PATH_RE = re.compile(
    r"([\w./-]+\.(py|js|ts|tsx|jsx|json|md|html|css|sh|toml|ya?ml|go|rs|java|kt|rb))",
    flags=re.IGNORECASE,
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
    del snapshot
    request = str(user_input or "").strip() or "Unclear request"
    context = _context_anchor(session)
    signal = _minimal_semantic_signal(request, context=context)
    anchor_artifacts = context["artifacts"]
    active_artifacts = anchor_artifacts[:6] if signal.use_context else []
    target_artifacts = _target_artifacts_for_signal(
        request,
        signal=signal,
        anchor_artifacts=anchor_artifacts,
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

    intent: GuardrailPrimaryIntent
    if looks_like_explanation_request(request):
        intent = "explain"
    elif _looks_like_plan_request(normalized):
        intent = "plan"
    elif _looks_like_search_request(normalized):
        intent = "search"
    elif looks_like_validation_request(request) and not looks_like_debug_request(request):
        intent = "validate"
    elif looks_like_debug_request(request):
        intent = "debug"
    elif is_clear_low_risk_build_request(request):
        intent = "create"
    elif (
        looks_like_update_request(request)
        or looks_like_hardening_request(request)
        or looks_like_additive_request(request)
        or scope_change
        or correction
        or (single_anchor and _has_explicit_change_marker(normalized))
    ):
        intent = "update"
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
    explicit_path = signal.explicit_path
    if explicit_path:
        return [
            TaskArtifact(
                path=explicit_path,
                name=Path(explicit_path).name,
                kind=Path(explicit_path).suffix.lower() or "file",
                role="primary_target",
                confidence=0.86,
            )
        ]
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
    return any(marker in normalized for marker in _PLAN_REQUEST_MARKERS)


def _looks_like_search_request(normalized: str) -> bool:
    return any(marker in normalized for marker in _SEARCH_REQUEST_MARKERS)


def _has_explicit_change_marker(normalized: str) -> bool:
    tokens = [token for token in re.split(r"[^a-z0-9_äöüß]+", normalized) if token]
    return any(token in _EXPLICIT_CHANGE_MARKERS for token in tokens)


def _looks_like_deictic_request(normalized: str) -> bool:
    if not normalized:
        return False
    tokens = [token for token in re.split(r"[^a-z0-9_äöüß]+", normalized) if token]
    return any(token in _DEICTIC_MARKERS for token in tokens) or normalized.startswith(("mach weiter", "do that", "continue"))


def _extract_explicit_path(text: str) -> str | None:
    match = _PATH_RE.search(text)
    if not match:
        return None
    return match.group(1).lstrip("./")


def _extract_explicit_name(text: str) -> str | None:
    candidate = infer_artifact_name_hint(text)
    return candidate if candidate and len(candidate) >= 3 else None


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
