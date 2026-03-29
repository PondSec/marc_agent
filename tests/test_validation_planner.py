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


def test_validation_planner_treats_generic_unittest_as_satisfied_after_targeted_module_passes():
    planner = ValidationPlanner()
    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root="/tmp/demo",
        validation_plan=[
            ValidationCommand(command="python -m unittest", kind="test", verification_scope="runtime"),
        ],
        verification_commands=["python -m unittest tests.test_cli", "python -m unittest"],
        changed_files=[FileChangeRecord(path="greet_cli/__main__.py", operation="write")],
        edit_generation=1,
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_cli",
            kind="test",
            verification_scope="runtime",
            status="passed",
            edit_generation=1,
        )
    )

    assert planner.pending_commands(session) == []
    assert planner.rollup_status(session) == "passed"


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


def test_validation_planner_promotes_workspace_traceback_frame_as_runtime_repair_target(tmp_path):
    planner = ValidationPlanner()
    workspace = tmp_path / "greet_cli"
    workspace.mkdir()
    (workspace / "__main__.py").write_text(
        "from .cli import greet\n\n"
        "def main(argv=None):\n"
        "    return greet('Ada')\n",
        encoding="utf-8",
    )
    (workspace / "cli.py").write_text(
        "def greet(name):\n    return f\"Hello, {name}!\"\n",
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_cli.py").write_text("pass\n", encoding="utf-8")

    session = SessionState(
        task="Add an uppercase CLI flag and keep the greeting helper working.",
        workspace_root=str(tmp_path),
        changed_files=[
            FileChangeRecord(path="greet_cli/cli.py", operation="modify"),
            FileChangeRecord(path="tests/test_cli.py", operation="modify"),
        ],
        edit_generation=2,
    )
    failed_run = ValidationRunRecord(
        command="python -m unittest tests.test_cli",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=2,
        summary="Validation command exited with 1.",
        excerpt=(
            ".usage: python3 -m unittest [-h] [name]\n"
            "python3 -m unittest: error: unrecognized arguments: --uppercase\n"
            "Traceback (most recent call last):\n"
            '  File "/home/demo/tests/test_cli.py", line 22, in test_greet_with_uppercase_flag\n'
            "    __main__.main(['--uppercase', 'Ada'])\n"
            f'  File "{tmp_path / "greet_cli" / "__main__.py"}", line 7, in main\n'
            "    args = parser.parse_args(argv)\n"
            "SystemExit: 2\n"
        ),
    )

    evidence = planner.build_failure_evidence(session, failed_run)

    assert evidence.artifact_paths[0] == "greet_cli/__main__.py"
    assert evidence.file_hints[0] == "greet_cli/__main__.py"
    assert "greet_cli/cli.py" in evidence.file_hints
    assert "tests/test_cli.py" in evidence.file_hints
    assert 7 in evidence.line_hints
    assert any("greet_cli/__main__.py" in item for item in evidence.repair_requirements)


def test_validation_planner_prioritizes_non_test_task_targets_for_runtime_failures(tmp_path):
    planner = ValidationPlanner()
    session = SessionState(
        task="Fix the failing normalization bug without weakening the tests.",
        workspace_root=str(tmp_path),
        edit_generation=1,
        task_state=TaskState(
            latest_user_turn=(
                "There is a bug in this repo causing the name normalization tests to fail. "
                "Find the problem, fix the implementation without changing the intended behavior, "
                "do not weaken the tests, and run the relevant tests."
            ),
            root_goal="Fix the normalization bug.",
            active_goal="Repair textutils/normalize.py so the tests pass.",
            goal_relation="new_task",
            output_expectation="The implementation is fixed and the tests pass.",
            current_user_intent="repair",
            execution_strategy="debug_repair",
            verification_target="python -m unittest tests.test_normalize",
            next_action="debug",
            target_artifacts=[
                {"path": "textutils/normalize.py", "kind": "file", "role": "primary_target", "confidence": 1.0},
                {"path": "tests/test_normalize.py", "kind": "test", "role": "validation_target", "confidence": 1.0},
            ],
        ),
    )
    failed_run = ValidationRunRecord(
        command="python -m unittest tests.test_normalize",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=1,
        summary="Validation command exited with 1.",
        excerpt=(
            "Traceback (most recent call last):\n"
            '  File "/home/demo/tests/test_normalize.py", line 11, in test_trims_and_collapses_whitespace\n'
            '    self.assertEqual(normalize_name("  ada   lovelace  "), "Ada Lovelace")\n'
            "AssertionError: '  Ada   Lovelace  ' != 'Ada Lovelace'\n"
        ),
    )

    evidence = planner.build_failure_evidence(session, failed_run)

    assert evidence.artifact_paths[0] == "textutils/normalize.py"
    assert "tests/test_normalize.py" in evidence.artifact_paths
    assert any("textutils/normalize.py" in item for item in evidence.repair_requirements)
    assert not any(
        item == "Change tests/test_normalize.py so the failing runtime or test path can complete successfully."
        for item in evidence.repair_requirements
    )


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


def test_validation_planner_targets_changed_unittest_module_instead_of_generic_command():
    planner = ValidationPlanner()
    snapshot = WorkspaceSnapshot(
        root="/tmp/demo",
        file_count=4,
        language_counts={"python": 3, "markdown": 1},
        top_directories=["tests"],
        important_files=["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
        focus_files=["tests/test_cli.py"],
        file_briefs={},
        manifests=[],
        configs=[],
        test_files=["tests/test_cli.py"],
        build_files=[],
        deploy_files=[],
        entrypoints=["greet_cli/__main__.py"],
        repo_map=[],
        project_labels=["python"],
        likely_commands=["python -m unittest"],
        validation_commands=[
            ValidationCommand(
                command="python -m unittest",
                kind="test",
                verification_scope="runtime",
                source="python-test-files",
                priority=10,
                reason="unittest-style Python tests detected in the repository.",
            )
        ],
        workflow_commands=[],
        repo_summary="Small CLI project with unittest coverage.",
    )

    plan = planner.build_plan(
        "Create the requested CLI and run the tests.",
        snapshot,
        changed_files=["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
    )

    assert plan[0].command == "python -m unittest tests.test_cli"
    assert all(item.command != "python -m unittest" for item in plan)


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


def test_validation_planner_builds_failure_evidence_from_targeted_unittest_command(tmp_path):
    planner = ValidationPlanner()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tmp_path / "greet_cli").mkdir()
    session = SessionState(
        task="Create a CLI package and run the tests.",
        workspace_root=str(tmp_path),
        validation_status="failed",
        edit_generation=1,
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="greet_cli/__main__.py", operation="create"),
            FileChangeRecord(path="greet_cli/cli.py", operation="create"),
            FileChangeRecord(path="README.md", operation="create"),
            FileChangeRecord(path="tests/test_cli.py", operation="create"),
        ]
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_cli",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=1,
            iteration=8,
            summary="Validation command exited with 1.",
            excerpt="FAIL: expected Hello, Ada!",
        )
    )

    evidence = planner.build_failure_evidence(session, session.validation_runs[-1])

    assert evidence.artifact_paths[0] == "greet_cli/__main__.py"
    assert "tests/test_cli.py" in evidence.artifact_paths


def test_validation_planner_collects_missing_fixture_path_from_runtime_traceback(tmp_path):
    planner = ValidationPlanner()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tmp_path / "wordfreq.py").write_text("def wordfreq(path):\n    return []\n", encoding="utf-8")
    (tests_dir / "test_wordfreq.py").write_text("pass\n", encoding="utf-8")
    session = SessionState(
        task="Create wordfreq.py and its tests.",
        workspace_root=str(tmp_path),
        validation_status="failed",
        edit_generation=1,
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="wordfreq.py", operation="create"),
            FileChangeRecord(path="tests/test_wordfreq.py", operation="create"),
        ]
    )
    failed_run = ValidationRunRecord(
        command="python -m unittest tests.test_wordfreq",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=1,
        iteration=3,
        summary="Validation command exited with 1.",
        excerpt=(
            "Traceback (most recent call last):\n"
            f'  File "{tmp_path / "tests" / "test_wordfreq.py"}", line 6, in test_wordfreq\n'
            "    result = wordfreq('tests/test_data.txt')\n"
            f'  File "{tmp_path / "wordfreq.py"}", line 2, in wordfreq\n'
            "    with open(file_path, 'r') as file:\n"
            "FileNotFoundError: [Errno 2] No such file or directory: 'tests/test_data.txt'\n"
        ),
    )

    evidence = planner.build_failure_evidence(session, failed_run)

    assert "tests/test_data.txt" in evidence.artifact_paths
    assert "tests/test_data.txt" in evidence.file_hints


def test_validation_planner_strips_trailing_sentence_from_inline_unittest_command():
    planner = ValidationPlanner()

    normalized = planner._normalize_explicit_validation_command(
        "python -m unittest tests.test_wordfreq. Finish only when the tests pass."
    )

    assert normalized == "python -m unittest tests.test_wordfreq"


def test_validation_planner_strips_trailing_passes_token_from_inline_unittest_command():
    planner = ValidationPlanner()

    normalized = planner._normalize_explicit_validation_command(
        "python -m unittest tests.test_wordfreq passes"
    )

    assert normalized == "python -m unittest tests.test_wordfreq"


def test_validation_planner_strips_trailing_passes_sentence_from_inline_unittest_command():
    planner = ValidationPlanner()

    normalized = planner._normalize_explicit_validation_command(
        "python -m unittest tests.test_wordfreq passes."
    )

    assert normalized == "python -m unittest tests.test_wordfreq"


def test_validation_planner_build_plan_normalizes_finish_only_when_unittest_phrase():
    planner = ValidationPlanner()
    snapshot = WorkspaceSnapshot(
        root="/tmp/demo",
        file_count=2,
        language_counts={"python": 2},
        top_directories=["tests"],
        important_files=["wordfreq.py", "tests/test_wordfreq.py"],
        focus_files=["wordfreq.py", "tests/test_wordfreq.py"],
        file_briefs={},
        manifests=[],
        configs=[],
        test_files=["tests/test_wordfreq.py"],
        build_files=[],
        deploy_files=[],
        entrypoints=["wordfreq.py"],
        repo_map=[],
        project_labels=["python"],
        likely_commands=[],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Small Python CLI with unittest coverage.",
    )
    session = SessionState(
        task="Create wordfreq.py and finish only when python -m unittest tests.test_wordfreq passes.",
        workspace_root="/tmp/demo",
        task_state=TaskState(
            latest_user_turn="Create wordfreq.py and finish only when python -m unittest tests.test_wordfreq passes.",
            root_goal="Create the CLI.",
            active_goal="Create the CLI and make the test pass.",
            goal_relation="new_task",
            output_expectation="A working CLI with a targeted unittest.",
            verification_target="Finish only when python -m unittest tests.test_wordfreq passes.",
            next_action="create",
            target_artifacts=[
                {"path": "wordfreq.py", "kind": "file", "role": "primary_target", "confidence": 1.0},
                {"path": "tests/test_wordfreq.py", "kind": "test", "role": "validation_target", "confidence": 1.0},
            ],
        ),
    )

    plan = planner.build_plan(
        session.task,
        snapshot,
        changed_files=["wordfreq.py", "tests/test_wordfreq.py"],
        session=session,
    )

    assert [item.command for item in plan] == ["python -m unittest tests.test_wordfreq"]


def test_validation_planner_does_not_map_plain_prose_word_to_unittest_path():
    planner = ValidationPlanner()

    assert planner._path_from_unittest_target("passes") is None


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
