from __future__ import annotations

import json
import shutil

from agent.models import (
    DiagnosticRecord,
    FileChangeRecord,
    SessionState,
    ValidationCommand,
    ValidationRunRecord,
    WorkspaceSnapshot,
)
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


def test_validation_planner_synthesizes_default_python_and_html_checks(monkeypatch):
    planner = ValidationPlanner()
    snapshot = WorkspaceSnapshot(
        root="/tmp/demo",
        file_count=0,
        language_counts={},
        top_directories=[],
        important_files=[],
        focus_files=[],
        file_briefs={},
        manifests=[],
        configs=[],
        test_files=[],
        build_files=[],
        deploy_files=[],
        entrypoints=[],
        repo_map=[],
        project_labels=[],
        likely_commands=[],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Empty workspace.",
    )
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/node" if name == "node" else None)

    plan = planner.build_plan(
        "bau mir ein kleines starter projekt",
        snapshot,
        changed_files=["game.py", "index.html", "snake.js"],
    )

    commands = [item.command for item in plan]
    assert any(command.startswith("internal:python_cli_smoke:") for command in commands)
    assert any(command.startswith("internal:python_syntax:") for command in commands)
    assert any(command.startswith("internal:web_artifact:") for command in commands)
    assert any(command.startswith("internal:html_refs:") for command in commands)
    assert any(command.startswith("node --check") for command in commands)


def test_validation_planner_prefers_runtime_smoke_for_small_python_entry_artifact():
    planner = ValidationPlanner()
    snapshot = WorkspaceSnapshot(
        root="/tmp/demo",
        file_count=1,
        language_counts={"python": 1},
        top_directories=[],
        important_files=["tic_tac_toe.py"],
        focus_files=["tic_tac_toe.py"],
        file_briefs={},
        manifests=[],
        configs=[],
        test_files=[],
        build_files=[],
        deploy_files=[],
        entrypoints=[],
        repo_map=[],
        project_labels=["python"],
        likely_commands=[],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Single Python starter script.",
    )

    plan = planner.build_plan(
        "fix the interactive bug in the existing python script",
        snapshot,
        changed_files=["tic_tac_toe.py"],
    )

    assert plan[0].command.startswith("internal:python_cli_smoke:")
    assert plan[0].verification_scope == "runtime"
    syntax_checks = [item for item in plan if item.command.startswith("internal:python_syntax:")]
    assert syntax_checks
    assert syntax_checks[0].required is False


def test_validation_planner_does_not_mark_unchecked_changes_as_passed():
    planner = ValidationPlanner()
    session = SessionState(
        task="Implement starter artifact",
        workspace_root="/tmp/demo",
        validation_plan=[],
        changed_files=[],
        edit_generation=1,
    )
    session.changed_files.append(FileChangeRecord(path="game.py", operation="create"))

    assert planner.rollup_status(session) == "not_run"


def test_validation_planner_adds_structural_web_checks_with_expected_features():
    planner = ValidationPlanner()
    snapshot = WorkspaceSnapshot(
        root="/tmp/demo",
        file_count=2,
        language_counts={"html": 1, "javascript": 1},
        top_directories=[],
        important_files=["snake.html", "snake.js"],
        focus_files=["snake.html"],
        file_briefs={},
        manifests=[],
        configs=[],
        test_files=[],
        build_files=[],
        deploy_files=[],
        entrypoints=[],
        repo_map=[],
        project_labels=["web"],
        likely_commands=[],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Small standalone web artifact.",
    )

    plan = planner.build_plan(
        "Baue ein Menü und einen Highscore dazu",
        snapshot,
        changed_files=["snake.html", "snake.js"],
    )

    structural = next(item for item in plan if item.command.startswith("internal:web_artifact:"))
    payload = json.loads(structural.command.partition("internal:web_artifact:")[2])

    assert structural.verification_scope == "structural"
    assert payload[0]["path"] == "snake.html"
    assert "menu" in payload[0]["expected_features"]
    assert "highscore" in payload[0]["expected_features"]


def test_validation_planner_builds_structured_failure_evidence_for_web_validation():
    planner = ValidationPlanner()
    session = SessionState(
        task="Baue ein Menü und einen Highscore dazu",
        workspace_root="/tmp/demo",
        edit_generation=1,
    )
    session.changed_files.append(FileChangeRecord(path="snake.html", operation="modify"))
    failed_run = ValidationRunRecord(
        command='internal:web_artifact:[{"path":"snake.html","expected_features":["menu","highscore"]}]',
        kind="check",
        verification_scope="structural",
        status="failed",
        edit_generation=1,
        summary="Structural web validation failed.",
        excerpt="snake.html: missing expected web features (menu, highscore)",
    )
    session.validation_runs.append(failed_run)
    session.diagnostics.append(
        DiagnosticRecord(
            source="run_tests",
            category="command_failure",
            summary="snake.html is still missing the required menu and highscore markers",
            tool_name="run_tests",
            command=failed_run.command,
            file_hints=["snake.html"],
            action_hints=["Inspect the failing output and repair the reported artifact before rerunning the check."],
            excerpt="snake.html: missing expected web features (menu, highscore)",
        )
    )

    evidence = planner.build_failure_evidence(session, failed_run)

    assert evidence.verification_scope == "structural"
    assert evidence.artifact_paths == ["snake.html"]
    assert evidence.expected_features == ["menu", "highscore"]
    assert evidence.missing_features == ["menu", "highscore"]
    assert "snake.html is missing validation-required features" in evidence.failure_summary
    assert any("Add or restore the structural features" in item for item in evidence.repair_requirements)
    assert any("Do not stop at an equivalent" in item for item in evidence.repair_requirements)
