from __future__ import annotations

import json

from agent.models import SessionState, WorkspaceSnapshot


def system_prompt() -> str:
    return (
        "You are a local coding agent. "
        "You must behave like a practical software engineering assistant, not a chatbot. "
        "Always return valid JSON only. "
        "Prefer targeted exploration over reading the whole repository. "
        "Use inspect_workspace, list_files, and search_in_files before loading many files. "
        "Only use shell commands for testing, linting, builds, or safe repository inspection. "
        "Never request destructive commands, remote pushes, or writes outside the workspace. "
        "When enough evidence is collected, finish with a concise, implementation-focused summary."
    )


def planning_prompt(task: str, snapshot: WorkspaceSnapshot) -> str:
    return (
        "Create a practical implementation plan for this coding task.\n\n"
        f"Task:\n{task}\n\n"
        f"Workspace summary:\n{snapshot.repo_summary}\n\n"
        f"Important files:\n{json.dumps(snapshot.important_files[:12], indent=2)}\n\n"
        f"Likely commands:\n{json.dumps(snapshot.likely_commands, indent=2)}\n\n"
        "Return JSON with keys: summary, steps, files_to_inspect, tests_to_run."
    )


def decision_prompt(
    task: str,
    session: SessionState,
    tool_manifest: str,
) -> str:
    recent_calls = [
        {
            "tool": item.tool_name,
            "args": item.tool_args,
            "success": item.success,
            "summary": item.summary,
            "output_excerpt": item.output_excerpt,
        }
        for item in session.tool_calls[-6:]
    ]
    changed_files = [item.path for item in session.changed_files[-10:]]
    prompt = {
        "task": task,
        "workspace_summary": session.workspace_snapshot.model_dump() if session.workspace_snapshot else {},
        "plan": [item.model_dump() for item in session.plan],
        "iteration": session.iterations,
        "recent_tool_calls": recent_calls,
        "changed_files": changed_files,
        "notes": session.notes[-6:],
        "tool_manifest": tool_manifest,
        "instructions": {
            "strategy": [
                "Inspect before editing.",
                "Prefer the smallest sufficient file set.",
                "Run tests or targeted validation after code changes.",
                "Finish when the task is completed or when a blocker must be reported.",
            ],
            "output_contract": {
                "thought_summary": "string",
                "action_type": "call_tool or final",
                "tool_name": "string or null",
                "tool_args": "object",
                "expected_outcome": "string",
                "final_response": "string or null",
            },
        },
    }
    return (
        "Decide the next best action for the coding task. "
        "Return JSON only.\n\n"
        + json.dumps(prompt, indent=2)
    )
