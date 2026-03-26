from __future__ import annotations

from agent.models import SessionState, ValidationCommand, ValidationRunRecord
from agent.verification import ValidationPlanner


def test_validation_planner_tracks_pending_commands_by_edit_generation():
    planner = ValidationPlanner()
    session = SessionState(
        task="Implement auth",
        workspace_root="/tmp/demo",
        validation_plan=[
            ValidationCommand(command="python -m pytest", kind="test", priority=10),
            ValidationCommand(command="ruff check .", kind="lint", priority=20),
        ],
        changed_files=[],
        edit_generation=2,
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m pytest",
            kind="test",
            status="passed",
            edit_generation=2,
        )
    )

    pending = planner.pending_commands(session)

    assert [item.command for item in pending] == ["ruff check ."]
    assert planner.rollup_status(session) == "not_run"


def test_validation_planner_marks_failure_when_current_generation_failed():
    planner = ValidationPlanner()
    session = SessionState(
        task="Implement auth",
        workspace_root="/tmp/demo",
        validation_plan=[
            ValidationCommand(command="python -m pytest", kind="test", priority=10),
        ],
        changed_files=[],
        edit_generation=1,
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m pytest",
            kind="test",
            status="failed",
            edit_generation=1,
        )
    )

    assert planner.rollup_status(session) == "failed"
