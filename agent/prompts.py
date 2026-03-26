from __future__ import annotations

import json

from agent.models import SessionState, WorkspaceSnapshot
from config.settings import AGENT_FULL_NAME, AGENT_NAME
from llm.schemas import TaskAnalysis


def system_prompt() -> str:
    return (
        f"You are {AGENT_NAME} ({AGENT_FULL_NAME}), a local autonomous coding agent. "
        "Always return valid JSON only. "
        "If the user asks a simple greeting, clarification, or question that can be answered safely without repo work, reply directly instead of forcing tool use. "
        "Inspect before editing, prefer existing architecture, and avoid searching filler words from the prompt. "
        "For create requests, read only the minimum manifest, README, or nearby example needed before writing files. "
        "After code changes, run the relevant validation command before finishing unless there is a real blocker. "
        "Respect access_mode exactly and never ask for destructive remote actions."
    )


def planning_prompt(task: str, snapshot: WorkspaceSnapshot) -> str:
    return planning_prompt_with_analysis(task, snapshot, None)


def planning_prompt_with_analysis(
    task: str,
    snapshot: WorkspaceSnapshot,
    analysis: TaskAnalysis | None,
) -> str:
    compact_snapshot = _compact_workspace_snapshot(snapshot)
    analysis_data = _compact_task_analysis(analysis)
    return (
        "Create a practical implementation plan for this coding task.\n\n"
        f"Task:\n{task}\n\n"
        f"Task analysis:\n{json.dumps(analysis_data, indent=2)}\n\n"
        f"Workspace context:\n{json.dumps(compact_snapshot, indent=2)}\n\n"
        "Return JSON with keys: summary, steps, files_to_inspect, tests_to_run, completion_criteria.\n"
        "Prefer high-signal files, multi-step validation commands, and explicit completion conditions.\n"
        "If the task is about creating something new, inspect only the minimum files needed to match project conventions before editing.\n"
        "If helper tooling may be needed, include that in the plan."
    )


def task_analysis_prompt(
    task: str,
    snapshot: WorkspaceSnapshot | None,
) -> str:
    workspace = _compact_workspace_snapshot(snapshot, detail="minimal")
    lines = [
        "Analyze the user's latest request before choosing tools.",
        f"Task: {_trim_text(task, 320)}",
        f"Workspace labels: {_format_list(workspace.get('project_labels'))}",
        f"Top directories: {_format_list(workspace.get('top_directories'))}",
        f"Manifests: {_format_list(workspace.get('manifests'))}",
        f"Entrypoints: {_format_list(workspace.get('entrypoints'))}",
        f"Focus files: {_format_list(workspace.get('focus_files'))}",
        f"Repo summary: {_trim_text(str(workspace.get('repo_summary', '')), 180)}",
        "Rules: ignore greetings/filler words; prefer create for write/build/create requests; prefer fix for bug/error reports; prefer reply only for pure conversation; keep target_paths short and real; keep search_terms high-signal.",
        "Return JSON only with keys: summary, intent, requires_repo_context, should_create_files, deliverable, search_terms, target_paths, relevant_extensions, direct_response.",
    ]
    return "\n".join(lines)


def decision_prompt(
    task: str,
    session: SessionState,
    tool_manifest: str,
) -> str:
    del tool_manifest
    recent_calls = [
        {
            "tool": item.tool_name,
            "success": item.success,
            "summary": item.summary,
        }
        for item in session.tool_calls[-3:]
    ]
    analysis = _compact_task_analysis(session.task_analysis)
    workspace = _compact_workspace_snapshot(
        session.workspace_snapshot,
        detail="decision",
    )
    diagnostics = [
        f"{item.category}: {item.summary}"
        for item in session.diagnostics[-2:]
    ]
    validation_runs = [
        f"{item.status} {item.kind or 'check'} {item.command}"
        for item in session.validation_runs[-2:]
    ]
    lines = [
        "Decide the next best action for the coding task.",
        f"Task: {_trim_text(task, 320)}",
        f"Intent: {analysis.get('intent')}",
        f"Task summary: {_trim_text(str(analysis.get('summary', '')), 180)}",
        f"Deliverable: {analysis.get('deliverable') or 'none'}",
        f"Access mode: {session.access_mode}",
        f"Phase: {session.current_phase} | workflow_stage: {session.workflow_stage}",
        f"Validation status: {session.validation_status} | repair_attempts: {session.repair_attempts}",
        f"Project labels: {_format_list(workspace.get('project_labels'))}",
        f"Manifests: {_format_list(workspace.get('manifests'))}",
        f"Entrypoints: {_format_list(workspace.get('entrypoints'))}",
        f"Candidate files: {_format_list(session.candidate_files[:6])}",
        f"Recent calls: {_format_objects(recent_calls)}",
        f"Recent diagnostics: {_format_list(diagnostics)}",
        f"Changed files: {_format_list([item.path for item in session.changed_files[-4:]])}",
        f"Validation plan: {_format_objects(session.validation_plan[:3], formatter=_format_validation_command)}",
        f"Validation runs: {_format_list(validation_runs)}",
        f"Blockers: {_format_list(session.blockers[:3])}",
        f"Last error: {_trim_text(session.last_error or '', 180)}",
        f"Tools: {_format_list(_compact_tool_manifest())}",
        "Rules: inspect before editing; prefer the smallest sufficient file set; reply directly only for pure conversation; do not search one prompt token at a time; for create intent read the minimum manifest/README/example first; after edits run validation before finalizing.",
        "Return JSON only with keys: thought_summary, action_type, tool_name, tool_args, expected_outcome, final_response.",
    ]
    return "\n".join(lines)


def _compact_validation_command(command) -> dict[str, object]:
    return {
        "command": command.command,
        "cwd": command.cwd,
        "kind": command.kind,
        "required": command.required,
        "reason": command.reason,
    }


def _compact_workspace_snapshot(snapshot: WorkspaceSnapshot | None) -> dict[str, object]:
    return _compact_workspace_snapshot(snapshot, detail="standard")


def _compact_workspace_snapshot(
    snapshot: WorkspaceSnapshot | None,
    *,
    detail: str = "standard",
) -> dict[str, object]:
    if snapshot is None:
        return {}
    important_limit = 6 if detail == "decision" else 8
    focus_limit = 4 if detail == "minimal" else 6
    important_files = snapshot.important_files[:important_limit]
    focus_files = snapshot.focus_files[:focus_limit]
    file_briefs = {
        path: _trim_text(snapshot.file_briefs.get(path, ""), 120)
        for path in [*focus_files[:2], *important_files[:3]]
        if snapshot.file_briefs.get(path) and detail != "minimal"
    }
    payload = {
        "project_labels": snapshot.project_labels[:8],
        "top_directories": snapshot.top_directories[:6],
        "manifests": snapshot.manifests[:4],
        "entrypoints": snapshot.entrypoints[:4],
        "focus_files": focus_files,
        "important_files": important_files,
        "repo_summary": _trim_text(snapshot.repo_summary, 320 if detail == "decision" else 220),
    }
    if file_briefs:
        payload["file_briefs"] = file_briefs
    if detail != "minimal":
        payload["likely_commands"] = snapshot.likely_commands[:4]
    if detail == "standard":
        payload["validation_commands"] = [
            _compact_validation_command(item)
            for item in snapshot.validation_commands[:4]
        ]
    return payload


def _compact_task_analysis(analysis: TaskAnalysis | None) -> dict[str, object]:
    if analysis is None:
        return {}
    return {
        "summary": analysis.summary,
        "intent": analysis.intent,
        "requires_repo_context": analysis.requires_repo_context,
        "should_create_files": analysis.should_create_files,
        "deliverable": analysis.deliverable,
        "search_terms": analysis.search_terms[:4],
        "target_paths": analysis.target_paths[:6],
        "relevant_extensions": analysis.relevant_extensions[:4],
        "direct_response": analysis.direct_response,
    }


def _compact_tool_manifest() -> list[str]:
    return [
        "inspect_workspace(focus)",
        "list_files(path,glob,recursive,max_results)",
        "search_in_files(query,path,glob,regex,case_sensitive,max_results)",
        "read_file(path,start_line,end_line)",
        "create_file(path,content,overwrite)",
        "write_file(path,content)",
        "append_file(path,content)",
        "replace_in_file(path,find,replace,count)",
        "patch_file(path,patches)",
        "show_diff(path,new_content)",
        "run_shell(command,cwd,timeout)",
        "run_tests(command,cwd,timeout)",
        "git_status{}",
        "git_diff(path,cached,context_lines)",
        "git_log(limit)",
        "git_create_branch(name)",
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
        formatter = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return " | ".join(formatter(value) for value in values)


def _format_validation_command(command) -> str:
    return f"{command.kind}:{command.command}"
