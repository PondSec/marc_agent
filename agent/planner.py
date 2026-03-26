from __future__ import annotations

from pathlib import Path

from agent.models import SessionState, WorkspaceSnapshot
from agent.prompts import decision_prompt, planning_prompt, system_prompt
from llm.ollama_client import OllamaClient
from llm.schemas import AgentActionType, AgentDecision, PlanningResponse


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "eine",
    "einen",
    "und",
    "oder",
    "fuer",
    "mit",
    "mein",
    "dein",
    "repo",
    "projekt",
    "agent",
    "coding",
}


class Planner:
    def __init__(self, llm: OllamaClient, tool_manifest: str):
        self.llm = llm
        self.tool_manifest = tool_manifest

    def create_plan(self, task: str, snapshot: WorkspaceSnapshot) -> PlanningResponse:
        try:
            data = self.llm.generate_json(
                planning_prompt(task, snapshot),
                system=system_prompt(),
            )
            response = PlanningResponse.model_validate(data)
            if not response.steps:
                raise ValueError("Planning response did not include steps.")
            return response
        except Exception:
            return self._fallback_plan(task, snapshot)

    def decide_next_action(self, task: str, session: SessionState) -> AgentDecision:
        try:
            data = self.llm.generate_json(
                decision_prompt(task, session, self.tool_manifest),
                system=system_prompt(),
            )
            return AgentDecision.model_validate(data)
        except Exception:
            return self._fallback_decision(session)

    def _fallback_plan(
        self,
        task: str,
        snapshot: WorkspaceSnapshot,
    ) -> PlanningResponse:
        steps = [
            "Discover the repo shape, manifests, tests, scripts, and architecture boundaries.",
            "Plan the smallest file set that should be read or searched before editing.",
            "Act on the existing implementation points instead of creating parallel architecture.",
            "Verify with the project-aware validation plan, not just a single command.",
            "Repair from concrete diagnostics until validation passes or a blocker is clear.",
            "Report changed files, commands, diagnostics, diffs, and stop reason.",
        ]
        if self._looks_like_analysis_task(task):
            steps = [
                "Discover the repository layout, entrypoints, and major subsystems.",
                "Read the highest-signal files for the requested analysis.",
                "Report strengths, weaknesses, and the highest-priority gaps.",
            ]
        if self._might_need_helper(task):
            steps.insert(
                min(4, len(steps)),
                "If direct inspection is insufficient, create a small helper script or parser to unblock the task.",
            )
        completion = [
            "Relevant files were inspected before editing.",
            "Any code changes were validated with the project-aware command plan or the blocker is explicit.",
            "Final output includes changed files, commands, diagnostics, and remaining risks.",
        ]
        return PlanningResponse(
            summary=(
                "Work from repository discovery into focused edits, multi-step verification, repair, and reporting."
            ),
            steps=steps,
            files_to_inspect=snapshot.focus_files[:6] or snapshot.important_files[:10],
            tests_to_run=[item.command for item in snapshot.validation_commands[:4]]
            or snapshot.likely_commands[:4],
            completion_criteria=completion,
        )

    def _fallback_decision(self, session: SessionState) -> AgentDecision:
        tool_names = [item.tool_name for item in session.tool_calls]
        read_paths = {
            item.tool_args.get("path")
            for item in session.tool_calls
            if item.tool_name == "read_file"
        }
        searched_queries = {
            str(item.tool_args.get("query", "")).lower()
            for item in session.tool_calls
            if item.tool_name == "search_in_files"
        }
        snapshot = session.workspace_snapshot
        analysis_task = self._looks_like_analysis_task(session.task)
        focus_terms = self._focus_terms(session.task)
        diagnostic_files = [
            path for item in session.diagnostics[-6:] for path in item.file_hints
        ]

        if not tool_names:
            return AgentDecision(
                thought_summary="Need a repository map and validation plan before selecting files.",
                action_type=AgentActionType.CALL_TOOL,
                tool_name="inspect_workspace",
                tool_args={"focus": session.task},
                expected_outcome="Identify relevant files, workflows, and validation commands.",
                final_response=None,
            )

        if Path(session.workspace_root, ".git").exists() and "git_status" not in tool_names:
            return AgentDecision(
                thought_summary="Checking git status avoids blind edits on a dirty worktree.",
                action_type=AgentActionType.CALL_TOOL,
                tool_name="git_status",
                tool_args={},
                expected_outcome="See current repository status before further work.",
                final_response=None,
            )

        if focus_terms:
            for term in focus_terms:
                if term not in searched_queries:
                    return AgentDecision(
                        thought_summary="Search task-specific keywords before broad file reads.",
                        action_type=AgentActionType.CALL_TOOL,
                        tool_name="search_in_files",
                        tool_args={"query": term, "path": ".", "max_results": 30},
                        expected_outcome="Find candidate files related to the task.",
                        final_response=None,
                    )

        candidates = session.candidate_files or (snapshot.important_files if snapshot else [])
        for candidate in [*diagnostic_files, *candidates[:16]]:
            if candidate not in read_paths:
                return AgentDecision(
                    thought_summary="Read the highest-signal file instead of guessing.",
                    action_type=AgentActionType.CALL_TOOL,
                    tool_name="read_file",
                    tool_args={"path": candidate},
                    expected_outcome="Collect concrete implementation or failure context.",
                    final_response=None,
                )

        if session.changed_files and session.validation_status in {"not_run", "failed"}:
            command = self._pick_validation_command(session)
            if command:
                return AgentDecision(
                    thought_summary="Changed code must run through the remaining validation plan.",
                    action_type=AgentActionType.CALL_TOOL,
                    tool_name="run_tests",
                    tool_args={"command": command, "cwd": ".", "timeout": 120},
                    expected_outcome="Validate current changes or reproduce remaining failures.",
                    final_response=None,
                )

        if session.validation_status == "failed" and session.repair_attempts >= 1:
            if diagnostic_files:
                unread = next((path for path in diagnostic_files if path not in read_paths), None)
                if unread:
                    return AgentDecision(
                        thought_summary="Inspect the file hinted by the failing validation before giving up.",
                        action_type=AgentActionType.CALL_TOOL,
                        tool_name="read_file",
                        tool_args={"path": unread},
                        expected_outcome="Understand the failing area for a repair attempt.",
                        final_response=None,
                    )
            return AgentDecision(
                thought_summary="Validation is still failing and deterministic fallback cannot repair further.",
                action_type=AgentActionType.FINAL,
                tool_name=None,
                tool_args={},
                expected_outcome="Report the blocker and current repository state.",
                final_response=self.summarize_session(session),
            )

        if analysis_task and (session.tool_calls or session.notes):
            return AgentDecision(
                thought_summary="Enough inspection context exists for a concrete analysis summary.",
                action_type=AgentActionType.FINAL,
                tool_name=None,
                tool_args={},
                expected_outcome="Provide an architecture and gap analysis.",
                final_response=self.summarize_session(session),
            )

        return AgentDecision(
            thought_summary="No better deterministic next step is available.",
            action_type=AgentActionType.FINAL,
            tool_name=None,
            tool_args={},
            expected_outcome="Provide a concise task summary.",
            final_response=self.summarize_session(session),
        )

    def summarize_session(self, session: SessionState) -> str:
        changed = ", ".join(item.path for item in session.changed_files) or "no files changed"
        commands = ", ".join(session.executed_commands) or "no commands executed"
        blockers = " | ".join(session.blockers[-3:]) if session.blockers else "none"
        notes = " | ".join(session.notes[-4:]) if session.notes else "none"
        diagnostics = (
            " | ".join(item.summary for item in session.diagnostics[-3:])
            if session.diagnostics
            else "none"
        )
        validations = (
            ", ".join(
                f"{item.kind or 'check'}:{item.status}:{item.command}"
                for item in session.validation_runs[-4:]
            )
            if session.validation_runs
            else "none"
        )
        return (
            f"Status={session.status}; phase={session.current_phase}; workflow_stage={session.workflow_stage}; "
            f"access_mode={session.access_mode}; validation={session.validation_status}; validations={validations}; "
            f"changed_files={changed}; commands={commands}; blockers={blockers}; diagnostics={diagnostics}; notes={notes}."
        )

    def _pick_validation_command(self, session: SessionState) -> str | None:
        passed = {
            run.command
            for run in session.validation_runs
            if run.edit_generation == session.edit_generation and run.status == "passed"
        }
        for item in session.validation_plan:
            if item.command not in passed:
                return item.command
        for command in session.verification_commands:
            if command and command not in passed:
                return command
        snapshot = session.workspace_snapshot
        if snapshot:
            for item in snapshot.validation_commands:
                if item.command not in passed:
                    return item.command
            for command in snapshot.likely_commands:
                if command and command not in passed:
                    return command
        return None

    def _looks_like_analysis_task(self, task: str) -> bool:
        lowered = task.lower()
        analysis_tokens = {
            "analyse",
            "analysiere",
            "bewerte",
            "review",
            "explain",
            "erklaer",
            "summarize",
            "zusammen",
            "understand",
        }
        action_tokens = {
            "fix",
            "behebe",
            "implement",
            "fuege",
            "refactor",
            "baue",
            "schreibe",
            "erweitere",
            "verbessere",
        }
        return any(token in lowered for token in analysis_tokens) and not any(
            token in lowered for token in action_tokens
        )

    def _might_need_helper(self, task: str) -> bool:
        lowered = task.lower()
        helper_tokens = {
            "log",
            "asset",
            "format",
            "parser",
            "convert",
            "analyse",
            "analyze",
            "build",
            "trace",
        }
        return any(token in lowered for token in helper_tokens)

    def _focus_terms(self, task: str) -> list[str]:
        tokens = []
        for raw in task.lower().replace("/", " ").replace("-", " ").split():
            token = raw.strip(".,:;()[]{}!?\"'")
            if len(token) < 3 or token in STOPWORDS:
                continue
            if token not in tokens:
                tokens.append(token)
        return tokens[:8]
