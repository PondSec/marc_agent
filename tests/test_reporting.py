from __future__ import annotations

from agent.models import SessionState, ToolCallRecord
from agent.planner import Planner
from agent.reporting import SessionReporter
from config.settings import AppConfig
from llm.schemas import AgentActionType


class BrokenLLM:
    def generate_json(self, *args, **kwargs):
        raise RuntimeError("llm unavailable")


def test_planner_fallback_replies_to_simple_greeting_without_tooling(tmp_path):
    planner = Planner(BrokenLLM(), "")
    session = SessionState(task="hallo?", workspace_root=str(tmp_path))

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.FINAL
    assert "Hallo" in (decision.final_response or "")
    assert "analysieren" in (decision.final_response or "")


def test_planner_replies_to_intro_question_without_repo_scan(tmp_path):
    planner = Planner(BrokenLLM(), "")
    session = SessionState(task="hallo wer bist du?", workspace_root=str(tmp_path))

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.FINAL
    assert "Coding-Agent" in (decision.final_response or "")


def test_planner_focus_terms_ignore_greeting_and_keep_signal_phrase():
    planner = Planner(BrokenLLM(), "")

    terms = planner._focus_terms(
        "hallo Marc kannst du ein python code schreiben wo ich Tick Tack Toe gegen ein computer ki spiele?"
    )

    assert "hallo" not in terms
    assert "marc" not in terms
    assert "kannst" not in terms
    assert "tick tack toe" in terms


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
