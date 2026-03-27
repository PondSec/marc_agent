from __future__ import annotations

from agent.core import AgentCore
from agent.models import FileChangeRecord, SessionState, ValidationCommand, ValidationRunRecord
from agent.task_state import TaskState
from config.settings import AppConfig


def test_core_marks_unvalidated_changed_files_as_partial(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="Implementiere etwas",
        workspace_root=str(tmp_path),
        validation_status="not_run",
    )
    session.changed_files.append(FileChangeRecord(path="game.py", operation="create"))

    status = core._resolve_final_status(session, final_action=True)
    stop_reason = core._derive_stop_reason(session)

    assert status == "partial"
    assert stop_reason == "validation_missing"


def test_core_marks_generation_failure_without_changes_as_partial(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="Erweitere tic_tac_toe.py um ein Menü",
        workspace_root=str(tmp_path),
        validation_status="not_run",
        stop_reason="generation_failed",
    )
    session.blockers.append("No reliable update content could be generated for tic_tac_toe.py.")

    status = core._resolve_final_status(session, final_action=True)

    assert status == "partial"


def test_core_marks_debug_follow_up_without_runtime_reproduction_as_partial(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="fix tic tac toe bug",
        workspace_root=str(tmp_path),
        validation_status="passed",
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
            current_user_intent="repair",
            execution_strategy="debug_repair",
            next_action="debug",
            execution_outline=["Read the active script", "Reproduce the issue", "Fix it and rerun the path"],
            needs_clarification=False,
            clarification_questions=[],
        ),
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command='internal:python_syntax:["tic_tac_toe.py"]',
            verification_scope="syntax",
            status="passed",
        )
    )

    status = core._resolve_final_status(session, final_action=True)
    stop_reason = core._derive_stop_reason(session)

    assert status == "partial"
    assert stop_reason == "reproduction_missing"


def test_core_marks_debug_fix_with_only_syntax_validation_as_partial(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="fix tic tac toe bug",
        workspace_root=str(tmp_path),
        validation_status="passed",
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
            current_user_intent="repair",
            execution_strategy="debug_repair",
            next_action="debug",
            execution_outline=["Read the active script", "Reproduce the issue", "Fix it and rerun the path"],
            needs_clarification=False,
            clarification_questions=[],
        ),
    )
    session.changed_files.append(FileChangeRecord(path="tic_tac_toe.py", operation="modify"))
    session.validation_runs.append(
        ValidationRunRecord(
            command='internal:python_syntax:["tic_tac_toe.py"]',
            verification_scope="syntax",
            status="passed",
        )
    )

    status = core._resolve_final_status(session, final_action=True)
    stop_reason = core._derive_stop_reason(session)

    assert status == "partial"
    assert stop_reason == "functional_validation_missing"


def test_core_marks_small_web_artifact_with_only_structural_checks_as_partial(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="Ergaenze snake.html um Menü und Highscore",
        workspace_root=str(tmp_path),
        validation_status="passed",
    )
    session.changed_files.append(FileChangeRecord(path="snake.html", operation="modify"))
    session.validation_runs.append(
        ValidationRunRecord(
            command='internal:web_artifact:[{"path":"snake.html","expected_features":["menu","highscore"]}]',
            verification_scope="structural",
            status="passed",
        )
    )

    status = core._resolve_final_status(session, final_action=True)
    stop_reason = core._derive_stop_reason(session)

    assert status == "partial"
    assert stop_reason == "functional_validation_missing"


def test_core_does_not_offer_identical_failed_validation_again_without_progress(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="Implementiere etwas",
        workspace_root=str(tmp_path),
        edit_generation=1,
        validation_plan=[ValidationCommand(command="python -m pytest", kind="test")],
        verification_commands=["python -m pytest"],
    )
    session.changed_files.append(FileChangeRecord(path="app/main.py", operation="write"))
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m pytest",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=1,
            iteration=3,
        )
    )

    assert core._pick_validation_command(session) is None
