from __future__ import annotations

import json

from agent.models import SessionState, WorkspaceSnapshot
from config.settings import AGENT_FULL_NAME, AGENT_NAME


def system_prompt() -> str:
    return (
        f"You are {AGENT_NAME} ({AGENT_FULL_NAME}), a local autonomous coding agent. "
        "Behave like a serious implementation system, not a chatbot. "
        "Always return valid JSON only. "
        "Prefer inspect -> search -> read -> edit -> verify -> repair -> final. "
        "Do not finish after code changes until at least one targeted validation command succeeds, "
        "unless you hit a meaningful blocker and explain it clearly. "
        "Use inspect_workspace, list_files, and search_in_files before broad file reads. "
        "Respect access_mode exactly. "
        "If a format, asset, log, or build process is hard to inspect directly, you may create a small "
        "helper script, parser, converter, test harness, or temporary analysis tool inside the workspace "
        "to unblock the real task, run it, read the result, and continue. "
        "Prefer minimal, focused helpers and explain why they exist in the final summary. "
        "Never request destructive commands, remote pushes, or writes outside the workspace."
    )


def planning_prompt(task: str, snapshot: WorkspaceSnapshot) -> str:
    return (
        "Create a practical implementation plan for this coding task.\n\n"
        f"Task:\n{task}\n\n"
        f"Workspace summary:\n{snapshot.repo_summary}\n\n"
        f"Important files:\n{json.dumps(snapshot.important_files[:12], indent=2)}\n\n"
        f"Likely commands:\n{json.dumps(snapshot.likely_commands, indent=2)}\n\n"
        "Return JSON with keys: summary, steps, files_to_inspect, tests_to_run, completion_criteria.\n"
        "Prefer high-signal files, likely verification commands, and explicit completion conditions.\n"
        "If helper tooling may be needed, include that in the plan."
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
            "phase": item.phase,
            "summary": item.summary,
            "expected_outcome": item.expected_outcome,
            "output_excerpt": item.output_excerpt,
        }
        for item in session.tool_calls[-6:]
    ]
    changed_files = [item.path for item in session.changed_files[-10:]]
    prompt = {
        "task": task,
        "access_mode": session.access_mode,
        "current_phase": session.current_phase,
        "validation_status": session.validation_status,
        "repair_attempts": session.repair_attempts,
        "workspace_summary": (
            session.workspace_snapshot.model_dump() if session.workspace_snapshot else {}
        ),
        "plan_summary": session.plan_summary,
        "plan": [item.model_dump() for item in session.plan],
        "candidate_files": session.candidate_files[:20],
        "verification_commands": session.verification_commands,
        "completion_criteria": session.completion_criteria,
        "iteration": session.iterations,
        "recent_tool_calls": recent_calls,
        "changed_files": changed_files,
        "notes": session.notes[-8:],
        "blockers": session.blockers,
        "last_error": session.last_error,
        "tool_manifest": tool_manifest,
        "instructions": {
            "strategy": [
                "Inspect before editing.",
                "Prefer the smallest sufficient file set.",
                "Patch the existing architecture instead of duplicating logic.",
                "After edits, run targeted validation and keep iterating until it passes or a real blocker remains.",
                "When blocked by format or tooling limitations, build a tiny helper script or parser only if it materially advances the task.",
                "Do not finalize while unverified code changes remain.",
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
        "Decide the next best action for the coding task. Return JSON only.\n\n"
        + json.dumps(prompt, indent=2)
    )
