from __future__ import annotations

import pytest

from agent.models import FileChangeRecord, SessionState, ToolCallRecord, ValidationRunRecord
from agent.planner import Planner
from agent.reporting import SessionReporter
from agent.task_state import TaskState
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
    payload = {
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
    planner = Planner(
        ScriptedLLM(json_payloads=[payload]),
        "",
    )
    session = SessionState(
        task="Wer bist du?",
        workspace_root=str(tmp_path),
        task_state=TaskState(
            latest_user_turn="Wer bist du?",
            root_goal="Answer the user's intro question.",
            active_goal="Explain briefly what the local coding agent can do.",
            goal_relation="new_task",
            output_expectation="A short direct capability explanation.",
            open_problem=None,
            verification_target="Return a concise direct answer.",
            target_artifacts=[],
            evidence=[],
            relevant_context=[],
            constraints=[],
            assumptions=[],
            missing_info=[],
            ambiguity_level="low",
            risk_level="low",
            confidence=0.95,
            next_action="explain",
            execution_outline=["Answer directly"],
            needs_clarification=False,
            clarification_questions=[],
        ),
    )
    session.router_result = planner.validate_router_output(payload)

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
        task_state=TaskState(
            latest_user_turn="Lies die README und fasse das Repo zusammen",
            root_goal="Summarize the repository.",
            active_goal="Read the README and summarize the repo.",
            goal_relation="new_task",
            output_expectation="A concise repo summary.",
            open_problem=None,
            verification_target="Mention the inspected README.",
            target_artifacts=[],
            evidence=[],
            relevant_context=[],
            constraints=[],
            assumptions=[],
            missing_info=[],
            ambiguity_level="low",
            risk_level="low",
            confidence=0.9,
            next_action="inspect",
            execution_outline=["Inspect README.md", "Summarize the repo"],
            needs_clarification=False,
            clarification_questions=[],
        ),
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


def test_reporter_requires_task_state_without_semantic_fallback(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    reporter = SessionReporter(config)
    session = SessionState(task="Hallo", workspace_root=str(tmp_path))

    with pytest.raises(RuntimeError, match="task_state"):
        reporter.render_final_response(session, draft_response="Hallo")


def test_reporter_marks_partial_unvalidated_changes_honestly(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    reporter = SessionReporter(config)
    session = SessionState(
        task="mach login sicherer",
        workspace_root=str(tmp_path),
        status="partial",
        current_phase="reporting",
        workflow_stage="report",
        validation_status="not_run",
        task_state=TaskState(
            latest_user_turn="mach login sicherer",
            root_goal="Implement login for this app.",
            active_goal="Harden the login flow without broad rewrites.",
            goal_relation="refine",
            output_expectation="A safer login flow.",
            open_problem=None,
            verification_target="Verify the login still works.",
            target_artifacts=[],
            evidence=[],
            relevant_context=[],
            constraints=[],
            assumptions=[],
            missing_info=[],
            ambiguity_level="low",
            risk_level="medium",
            confidence=0.84,
            next_action="modify",
            execution_outline=["Inspect auth flow", "Tighten weak spots", "Verify login path"],
            needs_clarification=False,
            clarification_questions=[],
        ),
    )
    session.changed_files.append(FileChangeRecord(path="app/auth.py", operation="modify"))
    session.report = reporter.build_report(session)

    response = reporter.render_final_response(session)

    assert "sauber validierten Abschluss" in response
    assert "kein sinnvoller Check" in response
    assert "Geaendert: app/auth.py." in response


def test_reporter_describes_model_start_failure_more_precisely(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    reporter = SessionReporter(config)
    session = SessionState(
        task="Bitte schreibe ein Snake Spiel in HTML",
        workspace_root=str(tmp_path),
        status="partial",
        current_phase="reporting",
        workflow_stage="report",
        validation_status="not_run",
        stop_reason="model_start_failed",
        task_state=TaskState(
            latest_user_turn="Bitte schreibe ein Snake Spiel in HTML",
            root_goal="Create a standalone web artifact.",
            active_goal="Create the requested standalone HTML artifact.",
            goal_relation="new_task",
            output_expectation="Generate the requested file content.",
            open_problem=None,
            verification_target="Create the requested artifact honestly.",
            target_artifacts=[],
            evidence=[],
            relevant_context=[],
            constraints=[],
            assumptions=[],
            missing_info=[],
            ambiguity_level="low",
            risk_level="medium",
            confidence=0.8,
            next_action="create",
            execution_outline=["Create the artifact"],
            needs_clarification=False,
            clarification_questions=[],
        ),
    )
    session.blockers.append(
        "Repeated model start failure for snake.html: qwen3:14b, qwen3:8b produced no first chunk, and no safe local recovery path applied."
    )
    session.report = reporter.build_report(session)

    response = reporter.render_final_response(session)

    assert "nicht einmal sauber starten" in response
    assert "Repeated model start failure" in response


def test_reporter_marks_degraded_starter_scaffold_honestly(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    reporter = SessionReporter(config)
    session = SessionState(
        task="Bitte erstelle ein HTML starter scaffold",
        workspace_root=str(tmp_path),
        status="completed",
        current_phase="reporting",
        workflow_stage="report",
        validation_status="not_run",
        task_state=TaskState(
            latest_user_turn="Bitte erstelle ein HTML starter scaffold",
            root_goal="Create a starter HTML artifact.",
            active_goal="Create a minimal starter HTML artifact.",
            goal_relation="new_task",
            output_expectation="A minimal starter scaffold.",
            open_problem=None,
            verification_target="Create the starter scaffold honestly.",
            target_artifacts=[],
            evidence=[],
            relevant_context=[],
            constraints=[],
            assumptions=[],
            missing_info=[],
            ambiguity_level="low",
            risk_level="low",
            confidence=0.88,
            next_action="create",
            execution_outline=["Create the starter scaffold"],
            needs_clarification=False,
            clarification_questions=[],
        ),
    )
    session.changed_files.append(FileChangeRecord(path="index.html", operation="create"))
    session.runtime_executions.append(
        {
            "operation_name": "content_generation",
            "task_class": "content_generation",
            "final_state": "degraded_success",
            "capability_tier": "tier_d",
            "recovery_strategy": "starter_scaffold",
            "degraded": True,
            "honest_blocked": False,
            "artifact_bytes_generated": 128,
            "validation_possible": True,
            "summary": "Artifact generation degraded to a minimal starter scaffold after repeated startup failures.",
            "attempts": [],
        }
    )
    session.report = reporter.build_report(session)

    response = reporter.render_final_response(session)

    assert "Starter-Grundgeruest" in response
    assert "keine vollstaendige" in response
    assert session.report.runtime_executions


def test_reporter_explains_missing_functional_validation_even_if_static_checks_passed(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    reporter = SessionReporter(config)
    session = SessionState(
        task="fix tic tac toe bug",
        workspace_root=str(tmp_path),
        status="partial",
        current_phase="reporting",
        workflow_stage="report",
        validation_status="passed",
        stop_reason="functional_validation_missing",
        task_state=TaskState(
            latest_user_turn="fix tic tac toe bug",
            root_goal="Build a Tic Tac Toe game in Python.",
            active_goal="Diagnose and fix the broken input handling.",
            goal_relation="report_problem",
            output_expectation="Diagnose the bug, apply the smallest safe fix, and rerun the interaction path.",
            open_problem="Moves are always rejected.",
            verification_target="Reproduce the interactive failure and rerun it after the fix.",
            target_artifacts=[],
            evidence=[],
            relevant_context=[],
            constraints=[],
            assumptions=[],
            missing_info=[],
            ambiguity_level="low",
            risk_level="medium",
            confidence=0.82,
            next_action="debug",
            execution_outline=["Read the script", "Reproduce the issue", "Fix and rerun it"],
            needs_clarification=False,
            clarification_questions=[],
        ),
    )
    session.changed_files.append(FileChangeRecord(path="tic_tac_toe.py", operation="modify"))
    session.report = reporter.build_report(session)

    response = reporter.render_final_response(session)

    assert "functional reproduction or smoke test is still missing" in response


def test_reporter_describes_structural_web_checks_more_honestly(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    reporter = SessionReporter(config)
    session = SessionState(
        task="Add a menu and highscore to snake.html",
        workspace_root=str(tmp_path),
        status="partial",
        current_phase="reporting",
        workflow_stage="report",
        validation_status="passed",
        stop_reason="functional_validation_missing",
        task_state=TaskState(
            latest_user_turn="Add a menu and highscore to snake.html",
            root_goal="Extend the small standalone web artifact.",
            active_goal="Update snake.html in place and check the result honestly.",
            goal_relation="refine",
            output_expectation="Apply the requested web follow-up and report the actual validation level.",
            open_problem=None,
            verification_target="Run the strongest available web validation without overstating coverage.",
            target_artifacts=[],
            evidence=[],
            relevant_context=[],
            constraints=[],
            assumptions=[],
            missing_info=[],
            ambiguity_level="low",
            risk_level="medium",
            confidence=0.85,
            next_action="modify",
            execution_outline=["Read snake.html", "Update the artifact", "Run web checks"],
            needs_clarification=False,
            clarification_questions=[],
        ),
    )
    session.changed_files.append(FileChangeRecord(path="snake.html", operation="modify"))
    session.validation_runs.append(
        ValidationRunRecord(
            command='internal:web_artifact:[{"path":"snake.html","expected_features":["menu","highscore"]}]',
            verification_scope="structural",
            status="passed",
        )
    )
    session.report = reporter.build_report(session)

    response = reporter.render_final_response(session)

    assert "structural web checks were confirmed" in response
    assert "functional browser or runtime smoke test is still missing" in response


def test_reporter_ignores_overconfident_draft_when_validation_is_only_structural(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    reporter = SessionReporter(config)
    session = SessionState(
        task="Add a menu to snake.html",
        workspace_root=str(tmp_path),
        status="partial",
        current_phase="reporting",
        workflow_stage="report",
        validation_status="passed",
        stop_reason="functional_validation_missing",
        task_state=TaskState(
            latest_user_turn="Add a menu to snake.html",
            root_goal="Extend the small standalone web artifact.",
            active_goal="Update snake.html in place and report the validation limits honestly.",
            goal_relation="refine",
            output_expectation="A concise user-facing summary that does not overclaim validation.",
            open_problem=None,
            verification_target="Report the strongest confirmed validation level without implying full runtime verification.",
            target_artifacts=[],
            evidence=[],
            relevant_context=[],
            constraints=[],
            assumptions=[],
            missing_info=[],
            ambiguity_level="low",
            risk_level="medium",
            confidence=0.85,
            next_action="modify",
            execution_outline=["Read snake.html", "Update the artifact", "Summarize honestly"],
            needs_clarification=False,
            clarification_questions=[],
        ),
    )
    session.changed_files.append(FileChangeRecord(path="snake.html", operation="modify"))
    session.validation_runs.append(
        ValidationRunRecord(
            command='internal:web_artifact:[{"path":"snake.html","expected_features":["menu"]}]',
            verification_scope="structural",
            status="passed",
        )
    )
    session.report = reporter.build_report(session)

    response = reporter.render_final_response(
        session,
        draft_response="I implemented the task and validated it.",
    )

    assert "cannot claim a clean validated completion yet" in response


def test_reporter_explains_missing_general_requirements_review(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    reporter = SessionReporter(config)
    session = SessionState(
        task="Fix the Python CLI output",
        workspace_root=str(tmp_path),
        status="partial",
        current_phase="reporting",
        workflow_stage="report",
        validation_status="passed",
        stop_reason="requirements_review_missing",
        task_state=TaskState(
            latest_user_turn="Fix the Python CLI output",
            root_goal="Fix the CLI output.",
            active_goal="Update the CLI and verify the requested output path honestly.",
            goal_relation="new_task",
            output_expectation="The CLI prints the corrected status line.",
            open_problem=None,
            verification_target="Confirm the changed output path matches the request.",
            target_artifacts=[],
            evidence=[],
            relevant_context=[],
            constraints=[],
            assumptions=[],
            missing_info=[],
            ambiguity_level="low",
            risk_level="medium",
            confidence=0.84,
            next_action="modify",
            execution_outline=["Read the CLI", "Apply the output fix", "Review the changed behavior"],
            needs_clarification=False,
            clarification_questions=[],
        ),
    )
    session.changed_files.append(FileChangeRecord(path="app/main.py", operation="modify"))
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m pytest",
            verification_scope="runtime",
            status="passed",
        )
    )
    session.report = reporter.build_report(session)

    response = reporter.render_final_response(
        session,
        draft_response="I implemented the task and validated it.",
    )

    assert "general requirements review" in response
    assert "cannot claim a clean validated completion yet" in response


def test_reporter_localizes_machine_summary_fallback_to_english(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    reporter = SessionReporter(config)
    session = SessionState(
        task="Read the README and summarize the repo",
        workspace_root=str(tmp_path),
        status="completed",
        current_phase="reporting",
        workflow_stage="report",
        validation_status="not_run",
        task_state=TaskState(
            latest_user_turn="Read the README and summarize the repo",
            root_goal="Summarize the repository.",
            active_goal="Read the README and summarize the repo.",
            goal_relation="new_task",
            output_expectation="A concise repo summary.",
            open_problem=None,
            verification_target="Mention the inspected README.",
            target_artifacts=[],
            evidence=[],
            relevant_context=[],
            constraints=[],
            assumptions=[],
            missing_info=[],
            ambiguity_level="low",
            risk_level="low",
            confidence=0.9,
            next_action="inspect",
            execution_outline=["Inspect README.md", "Summarize the repo"],
            needs_clarification=False,
            clarification_questions=[],
        ),
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

    assert "I " in response
    assert "README.md" in response
    assert "Ich habe" not in response
