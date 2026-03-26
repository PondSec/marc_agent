from __future__ import annotations

from agent.models import SessionState, ToolCallRecord
from agent.planner import Planner
from agent.reporting import SessionReporter
from config.settings import AppConfig
from llm.schemas import AgentActionType


class ScriptedLLM:
    def __init__(self, json_payloads=None):
        self.json_payloads = list(json_payloads or [])

    def generate_json(self, *args, **kwargs):
        if not self.json_payloads:
            raise RuntimeError("No JSON payload configured")
        return self.json_payloads.pop(0)

    def generate(self, *args, **kwargs):
        raise RuntimeError("No text generation configured")


def test_planner_direct_response_is_user_facing(tmp_path):
    planner = Planner(
        ScriptedLLM(
            json_payloads=[
                {
                    "user_goal": "Answer who the agent is.",
                    "intent": "explain",
                    "entities": {
                        "target_type": None,
                        "target_name": None,
                        "target_paths": [],
                        "attributes": [],
                        "constraints": [],
                    },
                    "requested_outcome": "Provide a short capability explanation.",
                    "action_plan": [
                        {
                            "step": 1,
                            "action": "respond_directly",
                            "reason": "No repository work is required.",
                        }
                    ],
                    "needs_clarification": False,
                    "clarification_questions": [],
                    "confidence": 0.94,
                    "safe_to_execute": True,
                    "repo_context_needed": False,
                    "search_terms": [],
                    "relevant_extensions": [],
                    "direct_response": "Ich bin dein lokaler Coding-Agent fuer diesen Workspace.",
                }
            ]
        ),
        "",
    )
    session = SessionState(task="Wer bist du?", workspace_root=str(tmp_path))

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.FINAL
    assert "lokaler Coding-Agent" in (decision.final_response or "")


def test_reporter_replaces_machine_summary_with_user_facing_response(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    reporter = SessionReporter(config)
    session = SessionState(
        task="Lies die README und fasse das Repo zusammen",
        workspace_root=str(tmp_path),
        status="completed",
        current_phase="reporting",
        workflow_stage="report",
        validation_status="not_run",
    )
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="read_file",
            tool_args={"path": "README.md"},
            success=True,
            summary="Read README.md.",
            phase="exploring",
        )
    )
    session.notes.append("read_file: Read README.md.")
    session.report = reporter.build_report(session)

    response = reporter.render_final_response(
        session,
        draft_response=(
            "Status=completed; phase=reporting; workflow_stage=report; "
            "access_mode=approval; validation=not_run; validations=none; "
            "changed_files=no files changed; commands=no commands executed; "
            "blockers=none; diagnostics=none; notes=read_file: Read README.md."
        ),
    )

    assert "Status=" not in response
    assert "README.md" in response
    assert "Ich habe" in response
