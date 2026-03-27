from __future__ import annotations

import json

from agent.models import SessionState, ValidationFailureEvidence, WorkspaceSnapshot
from agent.task_state import TaskState
from agent.task_schema import TaskUnderstanding
from config.settings import AGENT_FULL_NAME, AGENT_NAME
from llm.schemas import RouteActionName, RouterOutput


REPAIR_BLOCKED_SENTINEL = "__REPAIR_BLOCKED__"


def system_prompt() -> str:
    return (
        f"You are {AGENT_NAME} ({AGENT_FULL_NAME}), a local autonomous coding agent. "
        "Use the validated router output as the source of truth. "
        "Never select tools directly from the raw user prompt. "
        "Treat follow-up messages as continuing the current task unless the user clearly changes topic. "
        "For vague bug reports or comments like 'that is broken' or 'the terminal looks buggy', reconstruct the active task state, inspect available evidence, diagnose before editing, and ask only targeted follow-up questions when evidence is still missing. "
        "Inspect before editing, prefer the smallest sufficient change, and respect access mode."
    )


def router_system_prompt() -> str:
    actions = ", ".join(action.value for action in RouteActionName)
    return (
        f"You are {AGENT_NAME} ({AGENT_FULL_NAME}), a goal-oriented routing model. "
        "Return valid JSON only. "
        "Infer the user's true intent semantically, not from keyword matching. "
        "Be resilient to paraphrases, slang, typos, indirect wishes, and mixed German/English phrasing. "
        "Resolve deictic follow-ups like 'that', 'there', 'the error', 'hier', 'da', and 'das' against the current task state and follow-up context. "
        "Focus on the user's end goal, the minimum safe action plan, missing information, and whether execution is safe. "
        f"Allowed action names are: {actions}. "
        "If information is missing or the request is risky, ask one to three precise clarification questions instead of guessing."
    )


def task_understanding_system_prompt() -> str:
    return (
        f"You are {AGENT_NAME} ({AGENT_FULL_NAME}), an intent interpretation and goal extraction model "
        "for a local coding agent. "
        "Return valid JSON only. "
        "Infer the user's real objective semantically, not from literal verbs or keyword routing. "
        "Treat follow-up messages as continuing the current task unless the user clearly changes topic. "
        "Resolve references like 'that', 'there', 'the bug', 'das', 'da', 'hier', and 'backend' against the current task state. "
        "Prefer reasonable assumptions when the likely intent is strong. "
        "Ask for clarification only when the risk of acting on the wrong target is materially high. "
        "Produce a normalized task object that captures the goal, artifacts, assumptions, ambiguity, missing info, plan, and confidence."
    )


def task_state_system_prompt() -> str:
    return (
        f"You are {AGENT_NAME} ({AGENT_FULL_NAME}), the task-state update model for a local coding agent. "
        "Return valid JSON only. "
        "Your job is to update the agent's working task state from the latest user turn, prior task state, thread context, artifacts, diffs, diagnostics, and terminal evidence. "
        "Do not route by keywords. "
        "Determine whether the user is continuing, refining, correcting, constraining, or replacing the current task. "
        "Bind references like 'that', 'this part', 'the bug', 'das', 'hier', and 'backend' to concrete artifacts whenever possible. "
        "Always choose one execution strategy and one next_best_action that follow this discipline: inspect current state, gather evidence, act in the smallest sensible step, then verify."
    )


def task_state_update_prompt(
    task: str,
    snapshot: WorkspaceSnapshot | None,
    session: SessionState | None = None,
) -> str:
    schema_shape = {
        "latest_user_turn": "string",
        "root_goal": "string",
        "active_goal": "string",
        "goal_relation": "new_task | continue | refine | correct | report_problem | scope_change | validation_request | rollback_request | approval | rejection | update_constraints | clarify | unknown",
        "output_expectation": "string",
        "current_user_intent": "repair | implement | refactor | harden | validate | inspect | explain | search | plan | correct | unknown | null",
        "execution_strategy": "debug_repair | feature_implementation | refactor | hardening | validation_inspection | rollback_correction | null",
        "open_problem": "string | null",
        "verification_target": "string | null",
        "target_artifacts": [
            {
                "path": "string | null",
                "name": "string | null",
                "kind": "file | module | feature | flow | command | test | doc | service | null",
                "role": "primary_target | supporting_context | validation_target | active_context | null",
                "confidence": 0.0,
            }
        ],
        "active_artifacts": [
            {
                "path": "string | null",
                "name": "string | null",
                "kind": "file | module | feature | flow | command | test | doc | service | null",
                "role": "primary_target | supporting_context | validation_target | active_context | null",
                "confidence": 0.0,
            }
        ],
        "evidence": [
            {
                "kind": "message | log | diff | diagnostic | test | command | file | unknown",
                "summary": "string",
                "source": "string | null",
                "artifact_path": "string | null",
                "confidence": 0.0,
            }
        ],
        "supplied_evidence": ["string"],
        "relevant_context": ["string"],
        "constraints": ["string"],
        "assumptions": ["string"],
        "missing_info": ["string"],
        "ambiguity_level": "low | medium | high",
        "risk_level": "low | medium | high",
        "confidence": 0.0,
        "next_action": "inspect | search | create | modify | debug | test | explain | plan | clarify",
        "next_best_action": "inspect | search | create | modify | debug | test | explain | plan | clarify | null",
        "execution_outline": ["string"],
        "needs_clarification": False,
        "clarification_questions": ["string"],
    }
    lines = [
        "Update the central task state for this turn.",
        f"Latest user request: {_trim_text(task, 900)}",
        f"Recent conversation: {_format_objects(_compact_recent_messages(session))}",
        f"Recent tool calls: {_format_objects(_compact_recent_calls(session))}",
        f"Follow-up context: {json.dumps(_compact_follow_up_context(session), ensure_ascii=False)}",
        f"Previous task state: {json.dumps(_compact_task_state(session.task_state if session is not None else None), ensure_ascii=False)}",
        f"Previous task understanding: {json.dumps(_compact_task_understanding(session.task_understanding if session is not None else None), ensure_ascii=False)}",
        f"Recent diagnostics: {_format_objects(_compact_recent_diagnostics(session))}",
        f"Workspace context: {json.dumps(_compact_workspace_snapshot(snapshot, detail='router'), ensure_ascii=False)}",
        "State update rules:",
        "- First decide whether this turn continues or changes the active task.",
        "- Choose one execution_strategy from the committed task state, not from raw prompt keywords.",
        "- current_user_intent should reflect what the user is trying to do at this phase of the task.",
        "- Extract concrete evidence from diagnostics, terminal errors, changed files, and referenced artifacts.",
        "- For bugs or regressions, prefer inspect/debug/test before modify.",
        "- For scope corrections or rollbacks, narrow or revert only the necessary part of the prior work.",
        "- Update constraints and assumptions explicitly.",
        "- Keep root_goal stable across refinements unless the user clearly starts a new task.",
        "- If a compatible active artifact already exists and the user is extending its behavior, prefer modify over create unless the user clearly asks for a distinct new artifact or file surface.",
        "- next_action and next_best_action should be the single best next move, not a full route tree.",
        "- Preserve the execution order: inspect current state and active artifacts, gather evidence, act in the smallest sensible step, then verify against verification_target.",
        "- Ask clarification only if acting now would likely hit the wrong artifact or cause destructive behavior.",
        f"Return JSON only with this structure: {json.dumps(schema_shape, ensure_ascii=False)}",
    ]
    return "\n".join(lines)


def task_understanding_prompt(
    task: str,
    snapshot: WorkspaceSnapshot | None,
    session: SessionState | None = None,
) -> str:
    workspace = _compact_workspace_snapshot(snapshot, detail="router")
    prior_understanding = _compact_task_understanding(
        session.task_understanding if session is not None else None
    )
    schema_shape = {
        "original_request": "string",
        "interpreted_goal": "string",
        "intent_category": "build | modify | debug | refactor | explain | analyze | test | search | configure | plan | unknown",
        "conversation_relation": "new_task | same_task_follow_up | refinement | problem_report | constraint_update | clarification | correction | unknown",
        "subgoals": ["string"],
        "target_artifacts": [
            {
                "path": "string | null",
                "name": "string | null",
                "kind": "file | module | feature | flow | command | test | doc | service | null",
                "role": "primary_target | supporting_context | validation_target | active_context | null",
                "confidence": 0.0,
            }
        ],
        "relevant_context": ["string"],
        "constraints": ["string"],
        "missing_info": ["string"],
        "assumptions": ["string"],
        "user_observations": ["string"],
        "supplied_evidence": ["string"],
        "ambiguity_level": "low | medium | high",
        "risk_level": "low | medium | high",
        "confidence": 0.0,
        "recommended_mode": "clarify | inspect | search | create | modify | debug | refactor | test | explain | plan",
        "execution_plan": [
            {
                "step": 1,
                "summary": "string",
                "action_hint": "inspect | search | create | modify | debug | test | explain | plan | null",
                "requires_tools": True,
            }
        ],
        "needs_clarification": False,
        "clarification_questions": ["string"],
    }
    lines = [
        "Normalize the user's latest request into a task understanding object.",
        f"Latest user request: {_trim_text(task, 900)}",
        f"Recent conversation: {_format_objects(_compact_recent_messages(session))}",
        f"Recent tool calls: {_format_objects(_compact_recent_calls(session))}",
        f"Follow-up context: {json.dumps(_compact_follow_up_context(session), ensure_ascii=False)}",
        f"Current task state: {json.dumps(_compact_task_state(session.task_state if session is not None else None), ensure_ascii=False)}",
        f"Previous task understanding: {json.dumps(prior_understanding, ensure_ascii=False)}",
        f"Workspace context: {json.dumps(workspace, ensure_ascii=False)}",
        "Interpretation rules:",
        "- Extract the user's actual end goal, not just the surface wording.",
        "- Preserve continuity across turns when the user refines, corrects, or critiques earlier work.",
        "- Identify likely target artifacts even when the user names them indirectly.",
        "- Separate explicit constraints from assumptions.",
        "- If the user reports a vague problem, treat their message as an observation that should trigger diagnosis rather than blind fixing.",
        "- Use low ambiguity only when the target and outcome are both materially clear.",
        "- Use low confidence or needs_clarification only when acting now would likely hit the wrong target or be destructive.",
        "- Prefer short, concrete execution steps that can guide planning.",
        f"Return JSON only with this structure: {json.dumps(schema_shape, ensure_ascii=False)}",
    ]
    return "\n".join(lines)


def router_prompt(
    task: str,
    snapshot: WorkspaceSnapshot | None,
    session: SessionState | None = None,
) -> str:
    workspace = _compact_workspace_snapshot(snapshot, detail="router")
    recent_messages = _compact_recent_messages(session)
    recent_calls = _compact_recent_calls(session)
    follow_up_context = _compact_follow_up_context(session)
    recent_diagnostics = _compact_recent_diagnostics(session)
    changed_files = [item.path for item in session.changed_files[-6:]] if session else []
    action_catalog = [
        {
            "action": RouteActionName.RESPOND_DIRECTLY.value,
            "when": "Use when the request can be answered without any tool call.",
        },
        {
            "action": RouteActionName.ASK_CLARIFICATION.value,
            "when": "Use when required parameters, scope, or risk decisions are missing.",
        },
        {
            "action": RouteActionName.INSPECT_WORKSPACE.value,
            "when": "Use when repository structure or context must be gathered first.",
        },
        {
            "action": RouteActionName.SEARCH_WORKSPACE.value,
            "when": "Use when the target must be located semantically in the codebase.",
        },
        {
            "action": RouteActionName.READ_RELEVANT_FILES.value,
            "when": "Use when specific files should be inspected before replying or changing code.",
        },
        {
            "action": RouteActionName.DIAGNOSE_ISSUE.value,
            "when": "Use when a vague failure, bug report, terminal issue, or broken follow-up must be investigated before editing.",
        },
        {
            "action": RouteActionName.CREATE_ARTIFACT.value,
            "when": "Use when the user wants something new to be added.",
        },
        {
            "action": RouteActionName.UPDATE_ARTIFACT.value,
            "when": "Use when existing code or content should be changed.",
        },
        {
            "action": RouteActionName.DELETE_ARTIFACT.value,
            "when": "Use when something should be removed.",
        },
        {
            "action": RouteActionName.PLAN_WORK.value,
            "when": "Use when the user primarily wants a plan or implementation strategy.",
        },
        {
            "action": RouteActionName.RUN_VALIDATION.value,
            "when": "Use when execution should include tests, linting, or another verification step.",
        },
        {
            "action": RouteActionName.SUMMARIZE_RESULT.value,
            "when": "Use as the final step after inspection or mutation work.",
        },
    ]
    schema_shape = {
        "user_goal": "string",
        "intent": "inspect | create | update | debug | delete | search | explain | plan | unknown",
        "entities": {
            "target_type": "string | null",
            "target_name": "string | null",
            "target_paths": ["string"],
            "attributes": ["string"],
            "constraints": ["string"],
        },
        "requested_outcome": "string",
        "action_plan": [
            {
                "step": 1,
                "action": "one_of_the_allowed_action_names",
                "reason": "string",
            }
        ],
        "needs_clarification": True,
        "clarification_questions": ["string"],
        "confidence": 0.0,
        "safe_to_execute": False,
        "repo_context_needed": True,
        "search_terms": ["string"],
        "relevant_extensions": [".py"],
        "direct_response": "string | null",
    }
    lines = [
        "Interpret the user's latest request.",
        f"User request: {_trim_text(task, 600)}",
        f"Recent conversation: {_format_objects(recent_messages)}",
        f"Recent tool calls: {_format_objects(recent_calls)}",
        f"Recently changed files: {_format_list(changed_files)}",
        f"Follow-up context: {json.dumps(follow_up_context, ensure_ascii=False)}",
        f"Recent diagnostics: {_format_objects(recent_diagnostics)}",
        f"Workspace context: {json.dumps(workspace, ensure_ascii=False)}",
        f"Allowed actions: {json.dumps(action_catalog, ensure_ascii=False)}",
        "Routing rules:",
        "- Infer intent from meaning, not literal verbs.",
        "- Extract the user's end goal and likely target entities.",
        "- Treat vague follow-ups like 'da', 'das', 'hier', 'der Fehler', 'kaputt', 'buggy', 'komisch', and 'geht nicht' as references to the current task unless the user clearly changes topic.",
        "- When the user reports a vague bug or terminal issue and follow-up context exists, prefer a debug route with diagnosis before any update.",
        "- Map bug reports to evidence-seeking actions: inspect the active artifact, inspect diagnostics, rerun the most relevant validation or terminal command, then update only if the evidence supports it.",
        "- Detect when the request is ambiguous, unsafe, or missing a required parameter.",
        "- Prefer the smallest practical action plan.",
        "- Use intent=unknown when the request cannot be interpreted safely enough.",
        "- For direct conversational answers, set direct_response and use respond_directly.",
        "- For ambiguous requests, set needs_clarification=true, safe_to_execute=false, and add 1-3 precise questions tied to the missing artifact, symptom, or evidence.",
        "- For executable requests, set needs_clarification=false and safe_to_execute=true only when enough information is present.",
        f"Return JSON only with this structure: {json.dumps(schema_shape, ensure_ascii=False)}",
    ]
    return "\n".join(lines)


def router_repair_prompt(
    invalid_payload: dict[str, object],
    errors: list[dict[str, object]],
) -> str:
    return "\n".join(
        [
            "Repair the invalid router JSON output and return valid JSON only.",
            f"Invalid payload: {json.dumps(invalid_payload, ensure_ascii=False, default=str)}",
            f"Validation errors: {json.dumps(errors, ensure_ascii=False, default=str)}",
            "Do not add prose. Keep the same user intent and fix only the schema or safety issues.",
        ]
    )


def choose_path_prompt(route: RouterOutput, session: SessionState) -> str:
    snapshot = _compact_workspace_snapshot(session.workspace_snapshot, detail="decision")
    return "\n".join(
        [
            "Choose one relative workspace file path for a new implementation.",
            f"Route: {json.dumps(_compact_route(route), ensure_ascii=False)}",
            f"Workspace context: {json.dumps(snapshot, ensure_ascii=False)}",
            f"Already read files: {_format_list(_read_paths(session))}",
            "Return only the path, with no explanation or markdown.",
        ]
    )


def generate_content_prompt(
    route: RouterOutput,
    session: SessionState,
    *,
    path: str,
    current_content: str | None = None,
    repair_context: ValidationFailureEvidence | None = None,
    repair_strategy: str | None = None,
) -> str:
    sections = [
        "Produce the full file content for the requested task.",
        f"Validated route: {json.dumps(_compact_route(route), ensure_ascii=False)}",
        f"Task state: {json.dumps(_compact_task_state(session.task_state), ensure_ascii=False)}",
        f"Task understanding: {json.dumps(_compact_task_understanding(session.task_understanding), ensure_ascii=False)}",
        f"Target path: {path}",
        f"Workspace context: {json.dumps(_compact_workspace_snapshot(session.workspace_snapshot, detail='decision'), ensure_ascii=False)}",
        f"Inspected context: {_inspected_context(session)}",
        f"Diagnostic context: {_diagnostic_context(session)}",
        f"Follow-up context: {json.dumps(_compact_follow_up_context(session), ensure_ascii=False)}",
    ]
    if repair_context is not None:
        sections.extend(
            [
                f"Validation-guided repair context: {json.dumps(_compact_repair_context(repair_context), ensure_ascii=False)}",
                _repair_rules(repair_strategy),
            ]
        )
    if current_content is not None:
        sections.extend(
            [
                "Current file content:",
                current_content,
                "Return the full updated file content only. No markdown fences. No explanation.",
            ]
        )
    else:
        sections.append("Return the full new file content only. No markdown fences. No explanation.")
    return "\n\n".join(sections)


def generate_content_retry_prompt(
    route: RouterOutput,
    session: SessionState | None = None,
    *,
    path: str,
    current_content: str | None = None,
    repair_context: ValidationFailureEvidence | None = None,
    repair_strategy: str | None = None,
) -> str:
    sections = [
        "Produce the full file content for exactly one file.",
        f"User goal: {_trim_text(route.user_goal, 240)}",
        f"Requested outcome: {_trim_text(route.requested_outcome, 240)}",
        f"Task state: {json.dumps(_compact_task_state(session.task_state if session is not None else None), ensure_ascii=False)}",
        f"Task understanding: {json.dumps(_compact_task_understanding(session.task_understanding if session is not None else None), ensure_ascii=False)}",
        f"Target path: {path}",
        f"Search hints: {_format_list(route.search_terms[:6])}",
    ]
    if session is not None:
        sections.extend(
            [
                f"Diagnostic context: {_diagnostic_context(session)}",
                f"Follow-up context: {json.dumps(_compact_follow_up_context(session), ensure_ascii=False)}",
            ]
        )
    if repair_context is not None:
        sections.extend(
            [
                f"Validation-guided repair context: {json.dumps(_compact_repair_context(repair_context), ensure_ascii=False)}",
                _repair_rules(repair_strategy),
            ]
        )
    if current_content is not None:
        sections.extend(
            [
                "Current file content:",
                current_content,
                "Update this file to satisfy the request. Return the full updated file content only.",
            ]
        )
    else:
        sections.append("Create the file from scratch. Return the full new file content only.")
    sections.append("Do not add markdown fences or explanations.")
    return "\n\n".join(sections)


def generate_content_continuation_prompt(
    route: RouterOutput,
    session: SessionState | None = None,
    *,
    path: str,
    partial_content: str,
    current_content: str | None = None,
    repair_context: ValidationFailureEvidence | None = None,
    repair_strategy: str | None = None,
) -> str:
    sections = [
        "Finish the full file content for exactly one file.",
        "A previous slow generation already produced a partial draft. Use that progress instead of starting over blindly.",
        f"User goal: {_trim_text(route.user_goal, 240)}",
        f"Requested outcome: {_trim_text(route.requested_outcome, 240)}",
        f"Task state: {json.dumps(_compact_task_state(session.task_state if session is not None else None), ensure_ascii=False)}",
        f"Task understanding: {json.dumps(_compact_task_understanding(session.task_understanding if session is not None else None), ensure_ascii=False)}",
        f"Target path: {path}",
        f"Search hints: {_format_list(route.search_terms[:6])}",
    ]
    if session is not None:
        sections.extend(
            [
                f"Diagnostic context: {_diagnostic_context(session)}",
                f"Follow-up context: {json.dumps(_compact_follow_up_context(session), ensure_ascii=False)}",
            ]
        )
    if repair_context is not None:
        sections.extend(
            [
                f"Validation-guided repair context: {json.dumps(_compact_repair_context(repair_context), ensure_ascii=False)}",
                _repair_rules(repair_strategy),
            ]
        )
    if current_content is not None:
        sections.extend(
            [
                "Current file content:",
                current_content,
            ]
        )
    sections.extend(
        [
            "Partial draft from the previous attempt:",
            partial_content,
            "Return the complete final file content only. Preserve correct parts of the partial draft, finish the missing parts, and do not add markdown fences or explanations.",
        ]
    )
    return "\n\n".join(sections)


def final_response_prompt(route: RouterOutput, session: SessionState) -> str:
    recent_notes = session.notes[-12:]
    recent_calls = _compact_recent_calls(session)
    report_context = {
        "route": _compact_route(route),
        "task_state": _compact_task_state(session.task_state),
        "task_understanding": _compact_task_understanding(session.task_understanding),
        "changed_files": [item.path for item in session.changed_files[-8:]],
        "validation_status": session.validation_status,
        "recent_tool_calls": recent_calls,
        "recent_diagnostics": _compact_recent_diagnostics(session),
        "follow_up_context": _compact_follow_up_context(session),
        "notes": recent_notes,
        "blockers": session.blockers[-6:],
    }
    return "\n".join(
        [
            "Write a concise user-facing response for the completed or blocked task.",
            f"Context: {json.dumps(report_context, ensure_ascii=False)}",
            "Mention the main outcome, changed files or inspected files when relevant, and any blocker or validation status.",
            "Do not emit JSON.",
        ]
    )


def planning_prompt(
    route: RouterOutput,
    snapshot: WorkspaceSnapshot | None,
) -> str:
    return planning_prompt_with_route(route, snapshot)


def planning_prompt_with_route(
    route: RouterOutput,
    snapshot: WorkspaceSnapshot | None,
) -> str:
    compact_snapshot = _compact_workspace_snapshot(snapshot, detail="decision")
    return "\n".join(
        [
            "Summarize the validated router plan into practical execution steps.",
            f"Route: {json.dumps(_compact_route(route), ensure_ascii=False)}",
            f"Workspace context: {json.dumps(compact_snapshot, ensure_ascii=False)}",
        ]
    )


def _compact_recent_messages(session: SessionState | None) -> list[dict[str, str]]:
    if session is None:
        return []
    return [
        {
            "role": item.role,
            "content": _trim_text(item.content, 220),
        }
        for item in session.messages[-6:]
    ]


def _compact_recent_calls(session: SessionState | None) -> list[dict[str, object]]:
    if session is None:
        return []
    return [
        {
            "tool": item.tool_name,
            "success": item.success,
            "summary": _trim_text(item.summary, 120),
        }
        for item in session.tool_calls[-6:]
    ]


def _compact_follow_up_context(session: SessionState | None) -> dict[str, object]:
    if session is None or session.follow_up_context is None:
        return {}
    follow_up = session.follow_up_context
    return {
        "previous_task": _trim_text(follow_up.previous_task or "", 220),
        "previous_root_goal": _trim_text(follow_up.previous_root_goal or "", 220),
        "previous_active_goal": _trim_text(follow_up.previous_active_goal or "", 220),
        "previous_next_action": follow_up.previous_next_action,
        "previous_intent": follow_up.previous_intent,
        "previous_requested_outcome": _trim_text(follow_up.previous_requested_outcome or "", 220),
        "previous_final_response": _trim_text(follow_up.previous_final_response or "", 220),
        "previous_interpreted_goal": _trim_text(follow_up.previous_interpreted_goal or "", 220),
        "previous_recommended_mode": follow_up.previous_recommended_mode,
        "previous_confidence": follow_up.previous_confidence,
        "previous_assumptions": [_trim_text(item, 140) for item in follow_up.previous_assumptions[:6]],
        "previous_constraints": [_trim_text(item, 140) for item in follow_up.previous_constraints[:6]],
        "target_paths": follow_up.target_paths[:8],
        "changed_files": follow_up.changed_files[:8],
        "read_files": follow_up.read_files[:8],
        "recent_commands": follow_up.recent_commands[:6],
        "last_error": _trim_text(follow_up.last_error or "", 240),
        "notes": [_trim_text(item, 140) for item in follow_up.notes[-6:]],
    }


def _compact_recent_diagnostics(session: SessionState | None) -> list[dict[str, object]]:
    return [
        {
            "category": item.category,
            "severity": item.severity,
            "summary": _trim_text(item.summary, 180),
            "files": item.file_hints[:4],
            "command": _trim_text(item.command or "", 120),
        }
        for item in _recent_diagnostics(session)
    ]


def _recent_diagnostics(session: SessionState | None):
    if session is None:
        return []
    diagnostics = list(session.diagnostics[-6:])
    if session.follow_up_context is not None:
        diagnostics.extend(session.follow_up_context.diagnostics[-6:])
    return diagnostics[-6:]


def _compact_workspace_snapshot(
    snapshot: WorkspaceSnapshot | None,
    *,
    detail: str = "standard",
) -> dict[str, object]:
    if snapshot is None:
        return {}
    important_limit = 10 if detail == "router" else 6
    focus_limit = 8 if detail == "router" else 4
    payload: dict[str, object] = {
        "project_labels": snapshot.project_labels[:8],
        "top_directories": snapshot.top_directories[:8],
        "manifests": snapshot.manifests[:6],
        "entrypoints": snapshot.entrypoints[:6],
        "focus_files": snapshot.focus_files[:focus_limit],
        "important_files": snapshot.important_files[:important_limit],
        "repo_summary": _trim_text(snapshot.repo_summary, 320),
        "likely_commands": snapshot.likely_commands[:4],
    }
    if detail != "router":
        payload["file_briefs"] = {
            path: _trim_text(snapshot.file_briefs.get(path, ""), 120)
            for path in snapshot.important_files[:4]
            if snapshot.file_briefs.get(path)
        }
    return payload


def _compact_route(route: RouterOutput | None) -> dict[str, object]:
    if route is None:
        return {}
    return {
        "user_goal": route.user_goal,
        "intent": route.intent,
        "entities": route.entities.model_dump(),
        "requested_outcome": route.requested_outcome,
        "action_plan": [item.model_dump() for item in route.action_plan],
        "needs_clarification": route.needs_clarification,
        "clarification_questions": route.clarification_questions,
        "confidence": route.confidence,
        "safe_to_execute": route.safe_to_execute,
        "repo_context_needed": route.repo_context_needed,
        "search_terms": route.search_terms,
        "relevant_extensions": route.relevant_extensions,
        "direct_response": route.direct_response,
    }


def _compact_task_understanding(task: TaskUnderstanding | None) -> dict[str, object]:
    if task is None:
        return {}
    return {
        "interpreted_goal": _trim_text(task.interpreted_goal, 220),
        "intent_category": task.intent_category,
        "conversation_relation": task.conversation_relation,
        "recommended_mode": task.recommended_mode,
        "confidence": task.confidence,
        "ambiguity_level": task.ambiguity_level,
        "risk_level": task.risk_level,
        "target_artifacts": [item.model_dump() for item in task.target_artifacts[:6]],
        "constraints": task.constraints[:6],
        "assumptions": task.assumptions[:6],
        "missing_info": task.missing_info[:4],
        "execution_plan": [item.model_dump() for item in task.execution_plan[:5]],
    }


def _compact_task_state(state: TaskState | None) -> dict[str, object]:
    if state is None:
        return {}
    return {
        "root_goal": _trim_text(state.root_goal, 220),
        "active_goal": _trim_text(state.active_goal, 220),
        "goal_relation": state.goal_relation,
        "output_expectation": _trim_text(state.output_expectation, 220),
        "current_user_intent": state.current_user_intent,
        "execution_strategy": state.execution_strategy,
        "open_problem": _trim_text(state.open_problem or "", 220),
        "verification_target": _trim_text(state.verification_target or "", 220),
        "next_action": state.next_action,
        "next_best_action": state.next_best_action,
        "confidence": state.confidence,
        "ambiguity_level": state.ambiguity_level,
        "risk_level": state.risk_level,
        "target_artifacts": [item.model_dump() for item in state.target_artifacts[:6]],
        "active_artifacts": [item.model_dump() for item in state.active_artifacts[:6]],
        "evidence": [item.model_dump() for item in state.evidence[:6]],
        "supplied_evidence": state.supplied_evidence[:6],
        "constraints": state.constraints[:6],
        "assumptions": state.assumptions[:6],
        "missing_info": state.missing_info[:4],
        "execution_outline": state.execution_outline[:5],
    }


def _compact_repair_context(context: ValidationFailureEvidence) -> dict[str, object]:
    return {
        "command": _trim_text(context.command, 180),
        "verification_scope": context.verification_scope,
        "status": context.status,
        "artifact_paths": context.artifact_paths[:6],
        "summary": _trim_text(context.summary, 180),
        "failure_summary": _trim_text(context.failure_summary, 220),
        "excerpt": _trim_text(context.excerpt or "", 320),
        "expected_features": context.expected_features[:8],
        "missing_features": context.missing_features[:8],
        "file_hints": context.file_hints[:6],
        "line_hints": context.line_hints[:8],
        "action_hints": [_trim_text(item, 160) for item in context.action_hints[:4]],
        "repair_requirements": [_trim_text(item, 200) for item in context.repair_requirements[:6]],
        "evidence_signature": context.evidence_signature,
    }


def _repair_rules(repair_strategy: str | None) -> str:
    lines = [
        "Repair rules:",
        "- Use the failed validation evidence as a hard constraint for this update.",
        "- Make a concrete mutation that addresses the failed verification scope directly.",
        "- Do not return equivalent content or formatting-only changes.",
        f"- If the current evidence is still insufficient to derive a concrete fix, return exactly {REPAIR_BLOCKED_SENTINEL}.",
    ]
    if repair_strategy == "validation_escalated":
        lines.append(
            "- A previous repair attempt produced no effective change. Tighten the repair and change the file materially."
        )
    return "\n".join(lines)


def _inspected_context(session: SessionState) -> str:
    sections: list[str] = []
    for item in session.tool_calls:
        if item.tool_name != "read_file":
            continue
        path = str(item.tool_args.get("path", "")).strip()
        excerpt = (item.output_excerpt or "").strip()
        if not path or not excerpt:
            continue
        sections.append(f"{path}:\n{excerpt[:1200]}")
        if len(sections) >= 3:
            break
    if not sections and session.workspace_snapshot:
        sections.append(session.workspace_snapshot.repo_summary[:1000])
    return "\n\n".join(sections) or "none"


def _diagnostic_context(session: SessionState) -> str:
    sections: list[str] = []
    for item in _recent_diagnostics(session):
        command = f" command={item.command}" if item.command else ""
        files = f" files={','.join(item.file_hints[:4])}" if item.file_hints else ""
        excerpt = _trim_text(item.excerpt or "", 360)
        sections.append(
            f"[{item.severity}] {item.category}:{command}{files} summary={_trim_text(item.summary, 180)} excerpt={excerpt}"
        )
        if len(sections) >= 4:
            break
    if not sections and session.follow_up_context and session.follow_up_context.last_error:
        sections.append(_trim_text(session.follow_up_context.last_error, 360))
    return "\n".join(section for section in sections if section) or "none"


def _read_paths(session: SessionState) -> list[str]:
    return [
        str(item.tool_args.get("path") or "").strip()
        for item in session.tool_calls
        if item.tool_name == "read_file" and str(item.tool_args.get("path") or "").strip()
    ]


def _trim_text(text: str, limit: int) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _format_list(values: object) -> str:
    if not values:
        return "none"
    if isinstance(values, list):
        return ", ".join(str(value) for value in values if value) or "none"
    return str(values)


def _format_objects(values: list[object], formatter=None) -> str:
    if not values:
        return "none"
    if formatter is None:
        formatter = lambda item: item
    return json.dumps([formatter(item) for item in values], ensure_ascii=False)
