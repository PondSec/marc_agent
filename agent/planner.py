from __future__ import annotations

from pathlib import Path

from agent.models import PlanItem, SessionState, WorkspaceSnapshot
from agent.prompts import decision_prompt, planning_prompt, system_prompt
from llm.ollama_client import OllamaClient
from llm.schemas import AgentDecision, AgentActionType, PlanningResponse


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
            "Inspect the workspace and isolate the smallest useful set of relevant files.",
            "Read the highest-signal files before editing.",
            "Apply focused changes that preserve existing patterns.",
            "Run targeted validation commands and inspect failures.",
            "Repair issues, rerun validation, and summarize impact, diffs, commands, and risks.",
        ]
        if self._looks_like_analysis_task(task):
            steps = [
                "Inspect the repository layout and prioritize key files.",
                "Read the highest-signal files for the requested analysis.",
                "Summarize strengths, gaps, and likely next actions.",
            ]
        if self._might_need_helper(task):
            steps.insert(
                min(3, len(steps)),
                "If direct inspection is insufficient, create a small helper script or parser to unblock the task.",
            )
        completion = [
            "Relevant files were inspected before editing.",
            "Any code changes were validated with a targeted command or explained if blocked.",
            "Final output names changed files, commands, and remaining risks.",
        ]
        return PlanningResponse(
            summary=(
                "Prioritize repository understanding, targeted edits, validation, and repair."
            ),
            steps=steps,
            files_to_inspect=snapshot.important_files[:10],
            tests_to_run=snapshot.likely_commands[:4],
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

        if not tool_names:
            return AgentDecision(
                thought_summary="Need a repository map before selecting files.",
                action_type=AgentActionType.CALL_TOOL,
                tool_name="inspect_workspace",
                tool_args={"focus": session.task},
                expected_outcome="Identify relevant files and likely validation commands.",
                final_response=None,
            )

        if Path(session.workspace_root, ".git").exists() and "git_status" not in tool_names:
            return AgentDecision(
                thought_summary="Checking repo state helps avoid blind edits.",
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
                        thought_summary="Search task-specific keywords before broad reads.",
                        action_type=AgentActionType.CALL_TOOL,
                        tool_name="search_in_files",
                        tool_args={"query": term, "path": ".", "max_results": 30},
                        expected_outcome="Find candidate files related to the task.",
                        final_response=None,
                    )

        candidates = session.candidate_files or (snapshot.important_files if snapshot else [])
        for candidate in candidates[:12]:
            if candidate not in read_paths:
                return AgentDecision(
                    thought_summary="Read a high-priority file instead of guessing.",
                    action_type=AgentActionType.CALL_TOOL,
                    tool_name="read_file",
                    tool_args={"path": candidate},
                    expected_outcome="Collect concrete implementation context.",
                    final_response=None,
                )

        if session.changed_files and session.validation_status in {"not_run", "failed"}:
            command = self._pick_validation_command(session)
            if command:
                return AgentDecision(
                    thought_summary="Changed code should be validated before finalizing.",
                    action_type=AgentActionType.CALL_TOOL,
                    tool_name="run_tests",
                    tool_args={"command": command, "cwd": ".", "timeout": 120},
                    expected_outcome="Validate current changes or reproduce remaining failures.",
                    final_response=None,
                )

        if session.validation_status == "failed" and session.repair_attempts >= 1:
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
        return (
            f"Status={session.status}; phase={session.current_phase}; access_mode={session.access_mode}; "
            f"validation={session.validation_status}; changed_files={changed}; commands={commands}; "
            f"blockers={blockers}; notes={notes}."
        )

    def _pick_validation_command(self, session: SessionState) -> str | None:
        for command in session.verification_commands:
            if command:
                return command
        snapshot = session.workspace_snapshot
        if snapshot:
            for command in snapshot.likely_commands:
                if command:
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
        terms: list[str] = []
        for raw in task.lower().replace("/", " ").replace("-", " ").split():
            token = raw.strip(".,:;()[]{}!?\"'")
            if len(token) < 3 or token in STOPWORDS:
                continue
            if token not in terms:
                terms.append(token)
        return terms[:4]
