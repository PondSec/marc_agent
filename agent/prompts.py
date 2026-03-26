from __future__ import annotations

import json

from agent.models import SessionState, WorkspaceSnapshot
from config.settings import AGENT_FULL_NAME, AGENT_NAME
from llm.schemas import RouteActionName, RouterOutput


def system_prompt() -> str:
    return (
        f"You are {AGENT_NAME} ({AGENT_FULL_NAME}), a local autonomous coding agent. "
        "Use the validated router output as the source of truth. "
        "Never select tools directly from the raw user prompt. "
        "Inspect before editing, prefer the smallest sufficient change, and respect access mode."
    )


def router_system_prompt() -> str:
    actions = ", ".join(action.value for action in RouteActionName)
    return (
        f"You are {AGENT_NAME} ({AGENT_FULL_NAME}), a goal-oriented routing model. "
        "Return valid JSON only. "
        "Infer the user's true intent semantically, not from keyword matching. "
        "Be resilient to paraphrases, slang, typos, indirect wishes, and mixed German/English phrasing. "
        "Focus on the user's end goal, the minimum safe action plan, missing information, and whether execution is safe. "
        f"Allowed action names are: {actions}. "
        "If information is missing or the request is risky, ask one to three precise clarification questions instead of guessing."
    )


def router_prompt(
    task: str,
    snapshot: WorkspaceSnapshot | None,
    session: SessionState | None = None,
) -> str:
    workspace = _compact_workspace_snapshot(snapshot, detail="router")
    recent_messages = _compact_recent_messages(session)
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
        "intent": "inspect | create | update | delete | search | explain | plan | unknown",
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
        f"Recently changed files: {_format_list(changed_files)}",
        f"Workspace context: {json.dumps(workspace, ensure_ascii=False)}",
        f"Allowed actions: {json.dumps(action_catalog, ensure_ascii=False)}",
        "Routing rules:",
        "- Infer intent from meaning, not literal verbs.",
        "- Extract the user's end goal and likely target entities.",
        "- Detect when the request is ambiguous, unsafe, or missing a required parameter.",
        "- Prefer the smallest practical action plan.",
        "- Use intent=unknown when the request cannot be interpreted safely enough.",
        "- For direct conversational answers, set direct_response and use respond_directly.",
        "- For ambiguous requests, set needs_clarification=true, safe_to_execute=false, and add 1-3 precise questions.",
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
) -> str:
    sections = [
        "Produce the full file content for the requested task.",
        f"Validated route: {json.dumps(_compact_route(route), ensure_ascii=False)}",
        f"Target path: {path}",
        f"Workspace context: {json.dumps(_compact_workspace_snapshot(session.workspace_snapshot, detail='decision'), ensure_ascii=False)}",
        f"Inspected context: {_inspected_context(session)}",
    ]
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


def final_response_prompt(route: RouterOutput, session: SessionState) -> str:
    recent_notes = session.notes[-12:]
    recent_calls = _compact_recent_calls(session)
    report_context = {
        "route": _compact_route(route),
        "changed_files": [item.path for item in session.changed_files[-8:]],
        "validation_status": session.validation_status,
        "recent_tool_calls": recent_calls,
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
