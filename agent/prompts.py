from __future__ import annotations

import json

from agent.models import SessionState, WorkspaceSnapshot
from config.settings import AGENT_FULL_NAME, AGENT_NAME


def system_prompt() -> str:
    return (
        f"You are {AGENT_NAME} ({AGENT_FULL_NAME}), a local autonomous coding agent. "
        "Behave like a serious implementation system, not a chatbot. "
        "Always return valid JSON only. "
        "Prefer discover -> plan -> act -> verify -> repair -> report. "
        "Use inspect_workspace, list_files, and search_in_files before broad file reads. "
        "Prioritize manifests, configs, routes, services, models, middleware, tests, scripts, CI, build, and deploy files when they are relevant. "
        "Do not finish after code changes until the validation plan is complete or you hit a meaningful blocker and explain it clearly. "
        "Respect access_mode exactly. "
        "In full access mode you may use absolute paths outside the workspace when the task truly requires system-wide access. "
        "After validation failures, read the error output, inspect the hinted files, repair precisely, and rerun the blocked validation step. "
        "If a format, asset, log, or build process is hard to inspect directly, you may create a small "
        "helper script, parser, converter, test harness, or temporary analysis tool inside the workspace "
        "to unblock the real task, run it, read the result, and continue. "
        "Prefer minimal, focused helpers and explain why they exist in the final summary. "
        "Never request destructive commands or remote pushes."
    )


def planning_prompt(task: str, snapshot: WorkspaceSnapshot) -> str:
    return (
        "Create a practical implementation plan for this coding task.\n\n"
        f"Task:\n{task}\n\n"
        f"Workspace summary:\n{snapshot.repo_summary}\n\n"
        f"Important files:\n{json.dumps(snapshot.important_files[:12], indent=2)}\n\n"
        f"Focus files:\n{json.dumps(snapshot.focus_files[:12], indent=2)}\n\n"
        f"Repo map:\n{json.dumps(snapshot.repo_map[:12], indent=2)}\n\n"
        f"Validation plan:\n{json.dumps([item.model_dump() for item in snapshot.validation_commands[:8]], indent=2)}\n\n"
        f"Workflow commands:\n{json.dumps([item.model_dump() for item in snapshot.workflow_commands[:8]], indent=2)}\n\n"
        "Return JSON with keys: summary, steps, files_to_inspect, tests_to_run, completion_criteria.\n"
        "Prefer high-signal files, multi-step validation commands, and explicit completion conditions.\n"
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
        "workflow_stage": session.workflow_stage,
        "validation_status": session.validation_status,
        "repair_attempts": session.repair_attempts,
        "workspace_summary": (
            session.workspace_snapshot.model_dump() if session.workspace_snapshot else {}
        ),
        "plan_summary": session.plan_summary,
        "plan": [item.model_dump() for item in session.plan],
        "candidate_files": session.candidate_files[:20],
        "verification_commands": session.verification_commands,
        "validation_plan": [item.model_dump() for item in session.validation_plan[:8]],
        "validation_runs": [item.model_dump() for item in session.validation_runs[-8:]],
        "completion_criteria": session.completion_criteria,
        "iteration": session.iterations,
        "recent_tool_calls": recent_calls,
        "changed_files": changed_files,
        "diagnostics": [item.model_dump() for item in session.diagnostics[-8:]],
        "notes": session.notes[-8:],
        "blockers": session.blockers,
        "last_error": session.last_error,
        "tool_manifest": tool_manifest,
        "instructions": {
            "strategy": [
                "Inspect before editing.",
                "Prefer the smallest sufficient file set.",
                "Patch the existing architecture instead of duplicating logic.",
                "After edits, run the validation plan and keep iterating until required checks pass or a real blocker remains.",
                "Use diagnostics and file hints from failed commands to guide repairs.",
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
