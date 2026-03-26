from __future__ import annotations

from agent.models import PlanItem, SessionState, WorkspaceSnapshot
from agent.prompts import decision_prompt, planning_prompt, system_prompt
from llm.ollama_client import OllamaClient, OllamaClientError
from llm.schemas import AgentDecision, AgentActionType, PlanningResponse


class Planner:
    def __init__(self, llm: OllamaClient, tool_manifest: str):
        self.llm = llm
        self.tool_manifest = tool_manifest

    def create_plan(self, task: str, snapshot: WorkspaceSnapshot) -> list[PlanItem]:
        try:
            data = self.llm.generate_json(
                planning_prompt(task, snapshot),
                system=system_prompt(),
            )
            response = PlanningResponse.model_validate(data)
            return [PlanItem(step=step) for step in response.steps]
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
    ) -> list[PlanItem]:
        plan = [
            PlanItem(step="Inspect the workspace and locate the most relevant files."),
            PlanItem(step="Read the smallest useful set of files to understand the task."),
            PlanItem(step="Apply focused code changes."),
            PlanItem(step="Run targeted validation commands."),
            PlanItem(step="Summarize the result, impacted files, and remaining risks."),
        ]
        if "test" in task.lower():
            plan.insert(3, PlanItem(step="Reproduce the failure before fixing it."))
        if snapshot.likely_commands:
            plan.append(
                PlanItem(
                    step=f"Prefer validation via: {', '.join(snapshot.likely_commands[:2])}."
                )
            )
        return plan

    def _fallback_decision(self, session: SessionState) -> AgentDecision:
        tool_names = [item.tool_name for item in session.tool_calls]
        read_paths = {
            item.tool_args.get("path")
            for item in session.tool_calls
            if item.tool_name == "read_file"
        }
        snapshot = session.workspace_snapshot

        if not tool_names:
            return AgentDecision(
                thought_summary="Need a quick repository map before selecting files.",
                action_type=AgentActionType.CALL_TOOL,
                tool_name="inspect_workspace",
                tool_args={"focus": session.task},
                expected_outcome="Identify the most relevant files and likely commands.",
                final_response=None,
            )

        if snapshot:
            for candidate in snapshot.important_files:
                if candidate not in read_paths:
                    return AgentDecision(
                        thought_summary="Read a high-priority file instead of guessing.",
                        action_type=AgentActionType.CALL_TOOL,
                        tool_name="read_file",
                        tool_args={"path": candidate},
                        expected_outcome="Collect concrete implementation context.",
                        final_response=None,
                    )

        if session.changed_files and "run_shell" not in tool_names:
            command = snapshot.likely_commands[0] if snapshot and snapshot.likely_commands else "pytest -q"
            return AgentDecision(
                thought_summary="Changes exist, so the next step is validation.",
                action_type=AgentActionType.CALL_TOOL,
                tool_name="run_shell",
                tool_args={"command": command, "cwd": ".", "timeout": 120},
                expected_outcome="Verify the current changes with a targeted command.",
                final_response=None,
            )

        return AgentDecision(
            thought_summary="No better automatic step is available, so return a summary.",
            action_type=AgentActionType.FINAL,
            tool_name=None,
            tool_args={},
            expected_outcome="Provide a concise task summary.",
            final_response=self.summarize_session(session),
        )

    def summarize_session(self, session: SessionState) -> str:
        changed = ", ".join(item.path for item in session.changed_files) or "no files changed"
        commands = ", ".join(session.executed_commands) or "no commands executed"
        return (
            f"Task status: {session.status}. Changed files: {changed}. "
            f"Executed commands: {commands}. "
            f"Notes: {' | '.join(session.notes[-4:]) if session.notes else 'none'}."
        )
