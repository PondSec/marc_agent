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
from agent.task_state import TaskState
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


def test_validation_planner_treats_unittest_importability_failure_as_discovery_gap(tmp_path):
    planner = ValidationPlanner()
    session = SessionState(
        task="Create a word counting CLI with tests.",
        workspace_root=str(tmp_path),
        edit_generation=1,
        task_state=TaskState(
            latest_user_turn="Create wordfreq.py, README.md, and tests/test_wordfreq.py.",
            root_goal="Create the initial CLI implementation.",
            active_goal="Create the CLI, docs, and tests.",
            goal_relation="new_task",
            output_expectation="A working CLI with docs and unittest coverage.",
            verification_target="python -m unittest discover -s tests -v",
            next_action="create",
            target_artifacts=[
                {"path": "wordfreq.py", "kind": "file", "role": "primary_target", "confidence": 1.0},
                {"path": "README.md", "kind": "file", "role": "primary_target", "confidence": 1.0},
                {"path": "tests/test_wordfreq.py", "kind": "test", "role": "validation_target", "confidence": 1.0},
            ],
        ),
    )
    session.changed_files.append(FileChangeRecord(path="wordfreq.py", operation="create"))
    failed_run = ValidationRunRecord(
        command="python -m unittest discover -s tests -v",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=1,
        summary="Validation command exited with 1.",
        excerpt="ImportError: Start directory is not importable: 'tests'",
    )

    evidence = planner.build_failure_evidence(session, failed_run)

    assert evidence.artifact_paths[:2] == ["tests/test_wordfreq.py", "tests/__init__.py"]
    assert "tests/__init__.py" in evidence.file_hints
    assert any("test discovery" in item.lower() for item in evidence.repair_requirements)


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


def test_validation_planner_includes_explicit_user_requested_validation_command():
    planner = ValidationPlanner()
    snapshot = WorkspaceSnapshot(
        root="/tmp/demo",
        file_count=4,
        language_counts={"python": 3, "markdown": 1},
        top_directories=["tests"],
        important_files=["cli.py", "README.md", "tests/test_cli.py"],
        focus_files=["cli.py", "tests/test_cli.py"],
        file_briefs={},
        manifests=[],
        configs=[],
        test_files=["tests/test_cli.py"],
        build_files=[],
        deploy_files=[],
        entrypoints=["cli.py"],
        repo_map=[],
        project_labels=["python"],
        likely_commands=[],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Small CLI project with unittest coverage.",
    )
    session = SessionState(
        task="Fuege --state-root hinzu und fuehre danach python -m unittest aus.",
        workspace_root="/tmp/demo",
        task_state=TaskState(
            latest_user_turn="Fuege --state-root hinzu und fuehre danach python -m unittest aus.",
            root_goal="Extend the CLI safely.",
            active_goal="Add the new option and keep existing behavior.",
            goal_relation="continue",
            output_expectation="Updated CLI and tests.",
            verification_target="python -m unittest",
            next_action="modify",
        ),
    )

    plan = planner.build_plan(
        session.task,
        snapshot,
        changed_files=["cli.py", "tests/test_cli.py"],
        session=session,
    )

    assert plan[0].command == "python -m unittest"
    assert plan[0].verification_scope == "runtime"
    assert plan[0].required is True


def test_validation_planner_extracts_multiline_explicit_validation_command_without_following_bullet_text():
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
    session = SessionState(
        task=(
            "Erstelle ein kleines Projekt.\n"
            "- Lege app.py an.\n"
            "- Pruefe am Ende mit python -m unittest discover -s tests -v.\n"
            "- Wenn du Beispieldateien brauchst, darfst du sie anlegen.\n"
        ),
        workspace_root="/tmp/demo",
    )

    plan = planner.build_plan(
        session.task,
        snapshot,
        changed_files=["app.py"],
        session=session,
    )

    assert any(item.command == "python -m unittest discover -s tests -v" for item in plan)
    assert not any("Wenn du Beispieldateien brauchst" in item.command for item in plan)


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


def test_validation_planner_does_not_infer_canvas_from_keyboard_accessible_copy():
    planner = ValidationPlanner()
    snapshot = WorkspaceSnapshot(
        root="/tmp/demo",
        file_count=2,
        language_counts={"html": 1, "javascript": 1},
        top_directories=[],
        important_files=["index.html", "app.js"],
        focus_files=["index.html"],
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
        "Ergaenze einen keyboard-accessible Theme-Umschalter mit localStorage und Statusmeldung",
        snapshot,
        changed_files=["index.html", "app.js"],
    )

    structural = next(item for item in plan if item.command.startswith("internal:web_artifact:"))
    payload = json.loads(structural.command.partition("internal:web_artifact:")[2])

    assert "canvas" not in payload[0]["expected_features"]


def test_validation_planner_tracks_semantic_review_runs_separately_from_runtime_checks():
    planner = ValidationPlanner()
    session = SessionState(
        task="Fix the Python CLI bug",
        workspace_root="/tmp/demo",
        edit_generation=2,
    )
    session.changed_files.append(FileChangeRecord(path="game.py", operation="modify"))
    session.validation_runs.append(
        ValidationRunRecord(
            command='internal:semantic_review:[{"path":"game.py"}]',
            verification_scope="semantic",
            status="passed",
            edit_generation=2,
        )
    )

    assert planner.has_semantic_review(session) is True
    assert planner.has_semantic_review_success(session) is True
    assert planner.semantic_review_command(["game.py", "game.py"]) == 'internal:semantic_review:[{"path":"game.py"}]'


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


def test_validation_planner_prioritizes_test_artifacts_for_no_test_execution_failures():
    planner = ValidationPlanner()
    session = SessionState(
        task="Fuege --state-root zur CLI hinzu und fuehre danach python -m unittest aus.",
        workspace_root="/tmp/demo",
        edit_generation=1,
        task_state=TaskState(
            latest_user_turn="Fuege --state-root zur CLI hinzu und fuehre danach python -m unittest aus.",
            root_goal="Extend the CLI safely.",
            active_goal="Add the new option and update tests.",
            goal_relation="continue",
            output_expectation="Updated CLI, docs, and tests.",
            verification_target="python -m unittest",
            next_action="modify",
            target_artifacts=[
                {"path": "cli.py", "kind": "file", "role": "primary_target", "confidence": 1.0},
                {"path": "tests/test_cli.py", "kind": "test", "role": "validation_target", "confidence": 1.0},
            ],
        ),
    )
    session.changed_files.append(FileChangeRecord(path="cli.py", operation="modify"))
    session.changed_files.append(FileChangeRecord(path="README.md", operation="modify"))
    session.changed_files.append(FileChangeRecord(path="tests/test_cli.py", operation="modify"))
    failed_run = ValidationRunRecord(
        command="python -m unittest",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=1,
        summary="Validation command exited with 5.",
        excerpt="Ran 0 tests in 0.000s\n\nNO TESTS RAN",
    )

    evidence = planner.build_failure_evidence(session, failed_run)

    assert evidence.artifact_paths[0] == "tests/test_cli.py"
    assert "tests/__init__.py" in evidence.file_hints
    assert any("test discovery" in item for item in evidence.repair_requirements)
    assert any("tests/__init__.py" in item for item in evidence.repair_requirements)
    assert not any(item == "Change cli.py so the failing runtime or test path can complete successfully." for item in evidence.repair_requirements)
