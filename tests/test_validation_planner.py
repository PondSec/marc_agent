from __future__ import annotations

import json
import shutil

from agent.models import (
    DiagnosticRecord,
    FileChangeRecord,
    RepairBrief,
    RepairAttemptRecord,
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


def test_validation_planner_prefers_symbol_resolved_runtime_implementation_over_documentation_noise(tmp_path):
    planner = ValidationPlanner()
    for relative_path, content in {
        "README.md": "# Inventory App\n\nA small service with auth and rate limiting.\n",
        "docs/repo-map.md": "# Repo Map\n",
        "tests/test_repo_map.py": (
            "import unittest\n"
            "from inventory_app.api import handle_request\n"
            "from inventory_app.auth import AuthGate\n"
            "from inventory_app.service import list_inventory\n"
        ),
        "inventory_app/api.py": (
            "from .auth import AuthGate\n"
            "from .rate_limit import RateLimiter\n"
            "from .service import list_inventory\n\n"
            "def handle_request(headers, attempts):\n"
            "    return {'status': 200, 'items': list_inventory()}\n"
        ),
        "inventory_app/auth.py": "class AuthGate:\n    pass\n",
        "inventory_app/service.py": "def list_inventory():\n    return ['item']\n",
    }.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    session = SessionState(
        task="Create docs/repo-map.md for this inventory repo and run python -m unittest tests.test_repo_map.",
        workspace_root=str(tmp_path),
        edit_generation=1,
        workspace_snapshot=WorkspaceSnapshot(
            root=str(tmp_path),
            file_count=6,
            important_files=[
                "README.md",
                "docs/repo-map.md",
                "inventory_app/api.py",
                "inventory_app/auth.py",
                "inventory_app/service.py",
                "tests/test_repo_map.py",
            ],
            focus_files=["docs/repo-map.md", "inventory_app/api.py", "tests/test_repo_map.py"],
            test_files=["tests/test_repo_map.py"],
            import_hotspots=["inventory_app/api.py"],
            symbol_index={
                "inventory_app/api.py": ["handle_request"],
                "inventory_app/auth.py": ["AuthGate"],
                "inventory_app/service.py": ["list_inventory"],
            },
        ),
        task_state=TaskState(
            latest_user_turn="Create docs/repo-map.md for this inventory repo and run python -m unittest tests.test_repo_map.",
            root_goal="Create the repo map and verify it.",
            active_goal="Create docs/repo-map.md and run the targeted unittest.",
            goal_relation="new_task",
            output_expectation="A concise repo map in docs/repo-map.md that passes the requested test.",
            verification_target="python -m unittest tests.test_repo_map",
            next_action="create",
            target_artifacts=[
                {"path": "docs/repo-map.md", "kind": "doc", "role": "primary_target", "confidence": 1.0},
                {"path": "tests/test_repo_map.py", "kind": "test", "role": "validation_target", "confidence": 1.0},
            ],
        ),
        changed_files=[FileChangeRecord(path="docs/repo-map.md", operation="create")],
    )
    failed_run = ValidationRunRecord(
        command="python -m unittest tests.test_repo_map",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=1,
        summary="Validation command exited with 1.",
        excerpt=(
            "Traceback (most recent call last):\n"
            f'  File "{tmp_path / "tests" / "test_repo_map.py"}", line 8, in test_repo_map_exists_and_mentions_key_files\n'
            "    self.assertIn('README.md', open('README.md').read())\n"
            "AssertionError: 'README.md' not found in '# Inventory App\\n\\nA small service with auth and rate limiting.\\n'\n"
            f'  File "{tmp_path / "tests" / "test_repo_map.py"}", line 14, in test_handle_request_authenticates_user\n'
            "    response = handle_request(request)\n"
            "TypeError: handle_request() missing 1 required positional argument: 'attempts'\n"
        ),
    )

    evidence = planner.build_failure_evidence(session, failed_run)

    assert evidence.artifact_paths[0] == "inventory_app/api.py"
    assert evidence.file_hints[0] == "inventory_app/api.py"
    assert evidence.repair_brief is not None
    assert evidence.repair_brief.primary_target == "inventory_app/api.py"
    assert any("inventory_app/api.py" in item for item in evidence.repair_requirements)
    assert not any(item.startswith("Change README.md") for item in evidence.repair_requirements)


def test_validation_planner_collects_bare_workspace_file_references_from_runtime_assertions(tmp_path):
    planner = ValidationPlanner()
    for relative_path in [
        "index.html",
        "about.html",
        "projects.html",
        "contact.html",
        "styles.css",
        "tests/test_site.py",
    ]:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    session = SessionState(
        task="Extend the starter portfolio site with projects and contact pages.",
        workspace_root=str(tmp_path),
        edit_generation=1,
        task_state=TaskState(
            latest_user_turn=(
                "Build a multi-page portfolio site with projects.html and contact.html, "
                "keep navigation consistent, and update the shared stylesheet."
            ),
            root_goal="Finish the portfolio site.",
            active_goal="Create the new pages and repair the failing site validation.",
            goal_relation="new_task",
            output_expectation="The site pages and shared styling pass the targeted tests.",
            current_user_intent="repair",
            verification_target="python -m unittest tests.test_site",
            next_action="debug",
            target_artifacts=[
                {"path": "projects.html", "kind": "file", "role": "primary_target", "confidence": 1.0},
                {"path": "contact.html", "kind": "file", "role": "primary_target", "confidence": 1.0},
                {"path": "styles.css", "kind": "file", "role": "primary_target", "confidence": 1.0},
                {"path": "tests/test_site.py", "kind": "test", "role": "validation_target", "confidence": 1.0},
            ],
        ),
        changed_files=[
            FileChangeRecord(path="projects.html", operation="create"),
            FileChangeRecord(path="contact.html", operation="create"),
            FileChangeRecord(path="styles.css", operation="write"),
        ],
    )
    failed_run = ValidationRunRecord(
        command="python -m unittest tests.test_site",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=1,
        summary="Validation command exited with 1.",
        excerpt=(
            "FAIL: test_navigation_links_all_pages (tests.test_site.TestPortfolioSite.test_navigation_links_all_pages)\n"
            "AssertionError: 'href=\"projects.html\"' not found in '<!doctype html>...': index.html\n"
            "FAIL: test_projects_page_has_cards (tests.test_site.TestPortfolioSite.test_projects_page_has_cards)\n"
            "AssertionError: 'Case Studies' not found in '<!DOCTYPE html>...': projects.html\n"
            "FAIL: test_contact_page_has_form_fields (tests.test_site.TestPortfolioSite.test_contact_page_has_form_fields)\n"
            "AssertionError: 'class=\"contact-form\"' not found in '<!DOCTYPE html>...': contact.html\n"
            "FAIL: test_styles_cover_projects_and_contact (tests.test_site.TestPortfolioSite.test_styles_cover_projects_and_contact)\n"
            "AssertionError: '.site-grid' not found in 'body { ... }': styles.css\n"
        ),
    )

    evidence = planner.build_failure_evidence(session, failed_run)

    assert "index.html" in evidence.artifact_paths
    assert "index.html" in evidence.file_hints
    assert evidence.repair_brief is not None
    assert "index.html" in evidence.repair_brief.allowed_files


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


def test_validation_planner_prefers_test_target_for_runtime_harness_nameerror(tmp_path):
    planner = ValidationPlanner()
    pkg = tmp_path / "wordfreq"
    pkg.mkdir()
    (pkg / "__main__.py").write_text("from .cli import main\n", encoding="utf-8")
    (pkg / "cli.py").write_text("def main():\n    return 0\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_wordfreq.py").write_text("import unittest\n", encoding="utf-8")

    session = SessionState(
        task="Create a small wordfreq CLI with tests.",
        workspace_root=str(tmp_path),
        edit_generation=1,
        task_state=TaskState(
            latest_user_turn=(
                "Create a wordfreq CLI with package files, tests, and a README. "
                "Run python -m unittest tests.test_wordfreq at the end."
            ),
            root_goal="Create the initial CLI implementation.",
            active_goal="Create the CLI, tests, and docs.",
            goal_relation="new_task",
            output_expectation="A working CLI with passing tests.",
            verification_target="python -m unittest tests.test_wordfreq",
            next_action="create",
            target_artifacts=[
                {"path": "wordfreq/__main__.py", "kind": "file", "role": "primary_target", "confidence": 1.0},
                {"path": "wordfreq/cli.py", "kind": "file", "role": "primary_target", "confidence": 1.0},
                {"path": "tests/test_wordfreq.py", "kind": "test", "role": "validation_target", "confidence": 1.0},
            ],
        ),
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="wordfreq/__main__.py", operation="create"),
            FileChangeRecord(path="wordfreq/cli.py", operation="create"),
            FileChangeRecord(path="tests/test_wordfreq.py", operation="create"),
        ]
    )
    failed_run = ValidationRunRecord(
        command="python -m unittest tests.test_wordfreq",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=1,
        summary="Validation command exited with 1.",
        excerpt=(
            "ERROR: test_file_input (tests.test_wordfreq.TestWordFreq.test_file_input)\n"
            "Traceback (most recent call last):\n"
            f'  File "{tmp_path / "tests" / "test_wordfreq.py"}", line 19, in test_file_input\n'
            "    result = subprocess.run([sys.executable, '-m', 'wordfreq', sample_path], check=True)\n"
            "NameError: name 'sys' is not defined\n"
        ),
    )

    evidence = planner.build_failure_evidence(session, failed_run)

    assert evidence.artifact_paths[0] == "tests/test_wordfreq.py"
    assert evidence.repair_brief is not None
    assert evidence.repair_brief.primary_target == "tests/test_wordfreq.py"
    assert evidence.repair_requirements[0] == (
        "Change tests/test_wordfreq.py so the failing runtime or test path can complete successfully."
    )
    assert "wordfreq/__main__.py" in evidence.artifact_paths


def test_validation_planner_builds_undefined_symbol_semantics_for_runtime_failure(tmp_path):
    planner = ValidationPlanner()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_wordfreq.py").write_text(
        "import unittest\n"
        "from wordfreq.cli import read_text\n\n"
        "class TestWordFreq(unittest.TestCase):\n"
        "    def test_read_text_stdin(self):\n"
        "        import io\n"
        "        sys.stdin = io.StringIO('hello world hello')\n",
        encoding="utf-8",
    )
    pkg = tmp_path / "wordfreq"
    pkg.mkdir()
    (pkg / "cli.py").write_text("def read_text(source):\n    return source\n", encoding="utf-8")

    session = SessionState(
        task="Repair the failing stdin test for wordfreq.",
        workspace_root=str(tmp_path),
        workspace_snapshot=WorkspaceSnapshot(
            root=str(tmp_path),
            file_count=2,
            important_files=["tests/test_wordfreq.py", "wordfreq/cli.py"],
            test_files=["tests/test_wordfreq.py"],
        ),
    )
    failed_run = ValidationRunRecord(
        command="python -m unittest tests.test_wordfreq",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=1,
        summary="Validation command exited with 1.",
        excerpt=(
            "ERROR: test_read_text_stdin (tests.test_wordfreq.TestWordFreq.test_read_text_stdin)\n"
            "Traceback (most recent call last):\n"
            f'  File "{tmp_path / "tests" / "test_wordfreq.py"}", line 6, in test_read_text_stdin\n'
            "    sys.stdin = io.StringIO('hello world hello')\n"
            "NameError: name 'sys' is not defined. Did you forget to import 'sys'?\n"
        ),
    )

    evidence = planner.build_failure_evidence(session, failed_run)

    assert evidence.repair_brief is not None
    assert evidence.repair_brief.primary_target == "tests/test_wordfreq.py"
    assert evidence.repair_brief.implicated_symbols[0] == "sys"
    assert any(
        "The symbol 'sys' should be bound or imported before it is used." == item
        for item in evidence.repair_brief.expected_semantics
    )
    assert any(
        "The current runtime path uses 'sys' before it is bound or imported." in item
        for item in evidence.repair_brief.observed_semantics
    )
    assert any(
        "Bind or import 'sys' before its failing use in tests/test_wordfreq.py" in item
        for item in evidence.repair_requirements
    )


def test_validation_planner_builds_repair_brief_with_semantics_and_stable_signature(tmp_path):
    planner = ValidationPlanner()
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    (pkg / "__main__.py").write_text("def main(argv=None):\n    pass\n", encoding="utf-8")
    (pkg / "cli.py").write_text("def greet(name):\n    return name\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_cli.py").write_text("pass\n", encoding="utf-8")

    session = SessionState(
        task="Repair the CLI greeting formatting.",
        workspace_root=str(tmp_path),
        edit_generation=1,
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="greet_cli/__main__.py", operation="modify"),
            FileChangeRecord(path="greet_cli/cli.py", operation="modify"),
            FileChangeRecord(path="tests/test_cli.py", operation="modify"),
        ]
    )
    excerpt = (
        "Traceback (most recent call last):\n"
        f'  File "{tmp_path / "tests" / "test_cli.py"}", line 9, in test_greet_with_title\n'
        "    self.assertEqual(mock_stdout.getvalue().strip(), 'Hello, Mr. Ada!')\n"
        f'  File "{tmp_path / "greet_cli" / "__main__.py"}", line 7, in main\n'
        "    print(greeting)\n"
        "AssertionError: 'Mr. Hello, Ada!' != 'Hello, Mr. Ada!'\n"
        "- Mr. Hello, Ada!\n"
        "+ Hello, Mr. Ada!\n"
    )
    failed_run_a = ValidationRunRecord(
        command="python -m unittest tests.test_cli",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=1,
        summary="Validation command exited with 1.",
        excerpt=excerpt,
    )
    failed_run_b = ValidationRunRecord(
        command="python -m unittest tests.test_cli",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=1,
        summary="The targeted CLI assertion still fails.",
        excerpt=excerpt,
    )

    evidence_a = planner.build_failure_evidence(session, failed_run_a)
    evidence_b = planner.build_failure_evidence(session, failed_run_b)

    assert evidence_a.repair_brief is not None
    assert evidence_a.repair_brief.primary_target == "greet_cli/__main__.py"
    assert evidence_a.repair_brief.expected_semantics == ["Validation should produce: Hello, Mr. Ada!"]
    assert evidence_a.repair_brief.observed_semantics == ["Validation currently produces: Mr. Hello, Ada!"]
    assert evidence_a.repair_brief.locked_target == "greet_cli/__main__.py"
    assert "tests/test_cli.py" in evidence_a.repair_brief.forbidden_files
    assert evidence_a.repair_brief.failure_signature == evidence_b.repair_brief.failure_signature


def test_validation_planner_repair_brief_tracks_recent_failed_attempts_for_same_signature(tmp_path):
    planner = ValidationPlanner()
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    (pkg / "__main__.py").write_text("def main(argv=None):\n    pass\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_cli.py").write_text("pass\n", encoding="utf-8")

    session = SessionState(
        task="Repair the CLI greeting formatting.",
        workspace_root=str(tmp_path),
        edit_generation=2,
    )
    session.changed_files.append(FileChangeRecord(path="greet_cli/__main__.py", operation="modify"))
    failed_run = ValidationRunRecord(
        command="python -m unittest tests.test_cli",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=2,
        summary="Validation command exited with 1.",
        excerpt=(
            "Traceback (most recent call last):\n"
            f'  File "{tmp_path / "tests" / "test_cli.py"}", line 9, in test_greet_with_title\n'
            "    self.assertEqual(mock_stdout.getvalue().strip(), 'Hello, Mr. Ada!')\n"
            f'  File "{tmp_path / "greet_cli" / "__main__.py"}", line 7, in main\n'
            "    print(greeting)\n"
            "AssertionError: 'Mr. Hello, Ada!' != 'Hello, Mr. Ada!'\n"
            "- Mr. Hello, Ada!\n"
            "+ Hello, Mr. Ada!\n"
        ),
    )

    evidence = planner.build_failure_evidence(session, failed_run)
    assert evidence.repair_brief is not None
    session.repair_history.append(
        RepairAttemptRecord(
            artifact_path="greet_cli/__main__.py",
            validation_command=failed_run.command,
            verification_scope="runtime",
            strategy="validation_targeted",
            result="no_effective_change",
            reason="The generated edit left the implicated behavior unchanged.",
            failure_signature=evidence.repair_brief.failure_signature,
        )
    )

    updated = planner.build_failure_evidence(session, failed_run)

    assert updated.repair_brief is not None
    assert updated.repair_brief.locked_target == "greet_cli/__main__.py"
    assert updated.repair_brief.recent_failed_attempts
    assert updated.repair_brief.recent_failed_attempts[0].target == "greet_cli/__main__.py"
    assert updated.repair_brief.recent_failed_attempts[0].result == "no_effective_change"


def test_validation_planner_prioritizes_absolute_workspace_script_path_from_called_process_error(tmp_path):
    planner = ValidationPlanner()
    package_dir = tmp_path / "wordaudit"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("from .report import duplicate_words\n", encoding="utf-8")
    (package_dir / "report.py").write_text("def duplicate_words(lines):\n    return []\n", encoding="utf-8")
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "build_duplicates.py").write_text(
        "from wordaudit import duplicate_words\n",
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_path = tests_dir / "test_report.py"
    test_path.write_text("pass\n", encoding="utf-8")

    session = SessionState(
        task="Repair the wordaudit runtime flow.",
        workspace_root=str(tmp_path),
        edit_generation=1,
        task_state=TaskState(
            latest_user_turn=(
                "Refactor the existing wordaudit workflow so comment lines are ignored by both "
                "the library and scripts/build_duplicates.py."
            ),
            root_goal="Repair the wordaudit runtime flow.",
            active_goal="Repair the runtime flow.",
            goal_relation="new_task",
            output_expectation="Working library and script behavior with validation.",
            verification_target="python -m unittest tests.test_report",
            next_action="modify",
            target_artifacts=[
                {"path": "wordaudit/report.py", "kind": "file", "role": "primary_target", "confidence": 1.0},
                {"path": "scripts/build_duplicates.py", "kind": "file", "role": "primary_target", "confidence": 1.0},
                {"path": "tests/test_report.py", "kind": "test", "role": "validation_target", "confidence": 1.0},
            ],
        ),
    )
    session.changed_files.append(FileChangeRecord(path="wordaudit/report.py", operation="modify"))
    session.changed_files.append(FileChangeRecord(path="scripts/build_duplicates.py", operation="modify"))

    failed_run = ValidationRunRecord(
        command="python -m unittest tests.test_report",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=1,
        summary="Validation command exited with 1.",
        excerpt=(
            ".E\n"
            "======================================================================\n"
            f'ERROR: test_script_output_ignores_comment_lines ({test_path})\n'
            "----------------------------------------------------------------------\n"
            "Traceback (most recent call last):\n"
            f'  File "{test_path}", line 22, in test_script_output_ignores_comment_lines\n'
            "    result = subprocess.run(..., check=True)\n"
            "subprocess.CalledProcessError: "
            f"Command '['/usr/bin/python3', '{scripts_dir / 'build_duplicates.py'}', '/tmp/tmpwords.txt']' "
            "returned non-zero exit status 1.\n"
        ),
    )

    evidence = planner.build_failure_evidence(session, failed_run)

    assert evidence.artifact_paths[0] == "scripts/build_duplicates.py"
    assert evidence.file_hints[0] == "scripts/build_duplicates.py"
    assert evidence.repair_brief.primary_target == "scripts/build_duplicates.py"
    assert "wordaudit/report.py" in evidence.artifact_paths


def test_validation_planner_called_process_error_tracks_command_exit_without_noise_symbol(tmp_path):
    planner = ValidationPlanner()
    package_dir = tmp_path / "wordaudit"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("from .report import duplicate_words\n", encoding="utf-8")
    (package_dir / "report.py").write_text("def duplicate_words(lines):\n    return []\n", encoding="utf-8")
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "build_duplicates.py").write_text(
        "from wordaudit import duplicate_words\n",
        encoding="utf-8",
    )

    session = SessionState(
        task="Repair the wordaudit runtime flow.",
        workspace_root=str(tmp_path),
        edit_generation=1,
    )
    session.changed_files.append(FileChangeRecord(path="scripts/build_duplicates.py", operation="modify"))
    failed_run = ValidationRunRecord(
        command="python -m unittest tests.test_report",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=1,
        summary="Validation command exited with 1.",
        excerpt=(
            "Traceback (most recent call last):\n"
            "raise CalledProcessError(retcode, process.args,\n"
            "subprocess.CalledProcessError: "
            f"Command '['/usr/bin/python3', '{scripts_dir / 'build_duplicates.py'}', '/tmp/tmpwords.txt']' "
            "returned non-zero exit status 1.\n"
            "FAILED (errors=1)\n"
        ),
    )

    evidence = planner.build_failure_evidence(session, failed_run)

    assert evidence.repair_brief is not None
    assert evidence.repair_brief.primary_target == "scripts/build_duplicates.py"
    assert "CalledProcessError" not in evidence.repair_brief.implicated_symbols
    assert evidence.repair_brief.expected_semantics == ["The exercised runtime command should exit successfully."]
    assert evidence.repair_brief.observed_semantics == [
        (
            "The current runtime command exits non-zero when invoking: "
            f"['/usr/bin/python3', '{scripts_dir / 'build_duplicates.py'}', '/tmp/tmpwords.txt'] (exit status 1)"
        )
    ]


def test_validation_planner_classifies_direct_script_import_failure_as_bootstrap_failed(tmp_path):
    planner = ValidationPlanner()
    package_dir = tmp_path / "wordaudit"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("from .report import duplicate_words\n", encoding="utf-8")
    (package_dir / "report.py").write_text("def duplicate_words(lines):\n    return []\n", encoding="utf-8")
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    script_path = scripts_dir / "build_duplicates.py"
    script_path.write_text("from wordaudit import duplicate_words\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_path = tests_dir / "test_report.py"
    test_path.write_text("pass\n", encoding="utf-8")

    session = SessionState(
        task="Repair the wordaudit runtime flow.",
        workspace_root=str(tmp_path),
        edit_generation=1,
    )
    session.changed_files.append(FileChangeRecord(path="scripts/build_duplicates.py", operation="modify"))

    failed_run = ValidationRunRecord(
        command="python -m unittest tests.test_report",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=1,
        summary="Validation command exited with 1.",
        excerpt=(
            "Traceback (most recent call last):\n"
            f'  File "{script_path}", line 1, in <module>\n'
            "    from wordaudit import duplicate_words\n"
            "ModuleNotFoundError: No module named 'wordaudit'\n"
            "subprocess.CalledProcessError: "
            f"Command '['/usr/bin/python3', '{script_path}', '/tmp/tmpwords.txt']' "
            "returned non-zero exit status 1.\n"
            f'  File "{test_path}", line 22, in test_script_output_ignores_comment_lines\n'
            "    result = subprocess.run(..., check=True)\n"
        ),
    )

    evidence = planner.build_failure_evidence(session, failed_run)

    assert evidence.bootstrap_status == "bootstrap_failed"
    assert evidence.root_cause_summary is not None
    assert "bootstrap" in evidence.root_cause_summary.lower()
    assert evidence.repair_brief is not None
    assert evidence.repair_brief.bootstrap_status == "bootstrap_failed"
    assert evidence.repair_brief.primary_target == "scripts/build_duplicates.py"


def test_validation_planner_prefers_library_target_for_behavioral_runtime_failure_before_script_wrapper(tmp_path):
    planner = ValidationPlanner()
    package_dir = tmp_path / "wordaudit"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("from .report import duplicate_words\n", encoding="utf-8")
    (package_dir / "report.py").write_text("def duplicate_words(lines):\n    return []\n", encoding="utf-8")
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    script_path = scripts_dir / "build_duplicates.py"
    script_path.write_text("from wordaudit import duplicate_words\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_path = tests_dir / "test_report.py"
    test_path.write_text("pass\n", encoding="utf-8")

    session = SessionState(
        task="Repair the wordaudit runtime flow.",
        workspace_root=str(tmp_path),
        edit_generation=1,
        task_state=TaskState(
            latest_user_turn="Fix the duplicate-word reporting workflow in the library and script.",
            root_goal="Repair the wordaudit runtime flow.",
            active_goal="Repair the runtime flow.",
            goal_relation="new_task",
            output_expectation="Working library and script behavior with validation.",
            verification_target="python -m unittest tests.test_report",
            next_action="debug",
            target_artifacts=[
                {"path": "wordaudit/report.py", "kind": "file", "role": "primary_target", "confidence": 1.0},
                {"path": "scripts/build_duplicates.py", "kind": "file", "role": "primary_target", "confidence": 1.0},
                {"path": "tests/test_report.py", "kind": "test", "role": "validation_target", "confidence": 1.0},
            ],
        ),
    )
    session.changed_files.append(FileChangeRecord(path="wordaudit/report.py", operation="modify"))
    session.changed_files.append(FileChangeRecord(path="scripts/build_duplicates.py", operation="modify"))

    failed_run = ValidationRunRecord(
        command="python -m unittest tests.test_report",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=1,
        summary="Validation command exited with 1.",
        excerpt=(
            "FAIL: test_duplicate_words_ignores_comment_lines (tests.test_report.DuplicateReportTests.test_duplicate_words_ignores_comment_lines)\n"
            f'  File "{test_path}", line 16, in test_duplicate_words_ignores_comment_lines\n'
            '    self.assertEqual(duplicate_words(lines), ["alpha", "beta"])\n'
            "AssertionError: Lists differ: ['# skipped', 'alpha', 'beta'] != ['alpha', 'beta']\n"
            "subprocess.CalledProcessError: "
            f"Command '['/usr/bin/python3', '{script_path}', '/tmp/tmpwords.txt']' "
            "returned non-zero exit status 1.\n"
        ),
    )

    evidence = planner.build_failure_evidence(session, failed_run)

    assert evidence.repair_brief is not None
    assert evidence.repair_brief.failure_type == "assertion_mismatch"
    assert evidence.repair_brief.primary_target == "wordaudit/report.py"
    assert "scripts/build_duplicates.py" in evidence.repair_brief.allowed_files


def test_validation_planner_requires_bootstrap_reset_after_two_same_failures_without_behavior_change():
    planner = ValidationPlanner()
    session = SessionState(
        task="Repair the direct script bootstrap path.",
        workspace_root="/tmp/demo",
    )
    repair_brief = RepairBrief(
        failure_type="import_failure",
        failure_signature="runtime:import_failure:abc123",
        primary_target="scripts/build_duplicates.py",
        locked_target="scripts/build_duplicates.py",
        bootstrap_status="bootstrap_failed",
        root_cause_summary="scripts/build_duplicates.py still fails during bootstrap import.",
    )
    session.repair_history.extend(
        [
            RepairAttemptRecord(
                artifact_path="scripts/build_duplicates.py",
                validation_command="python -m unittest tests.test_report",
                verification_scope="runtime",
                strategy="validation_targeted",
                result="no_effective_change",
                reason="comment-only change",
                failure_signature="runtime:import_failure:abc123",
                productive_change=False,
                change_labels=["comment_only"],
            ),
            RepairAttemptRecord(
                artifact_path="scripts/build_duplicates.py",
                validation_command="python -m unittest tests.test_report",
                verification_scope="runtime",
                strategy="validation_escalated",
                result="mutation_planned",
                reason="substantive mutation prepared",
                failure_signature="runtime:import_failure:abc123",
                productive_change=True,
                independent_verification=False,
                behavior_changed=False,
                post_validation_failure_signature="runtime:import_failure:abc123",
            ),
        ]
    )

    assert planner.same_failure_without_behavior_change_count(session, repair_brief) == 2
    assert planner.should_require_bootstrap_reset(session, repair_brief) is True


def test_validation_planner_ignores_separator_noise_in_assertion_semantics(tmp_path):
    planner = ValidationPlanner()
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    (pkg / "__main__.py").write_text("def main(argv=None):\n    pass\n", encoding="utf-8")
    (pkg / "cli.py").write_text("def greet(name):\n    return name\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_cli.py").write_text("pass\n", encoding="utf-8")

    session = SessionState(
        task="Repair the CLI greeting formatting.",
        workspace_root=str(tmp_path),
        edit_generation=1,
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="greet_cli/__main__.py", operation="modify"),
            FileChangeRecord(path="tests/test_cli.py", operation="modify"),
        ]
    )
    failed_run = ValidationRunRecord(
        command="python -m unittest tests.test_cli",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=1,
        summary="Validation command exited with 1.",
        excerpt=(
            "..FF\n"
            "======================================================================\n"
            "FAIL: test_title_flag (tests.test_cli.TestCLI.test_title_flag)\n"
            "----------------------------------------------------------------------\n"
            f'  File "{tmp_path / "tests" / "test_cli.py"}", line 19, in test_title_flag\n'
            '    self.assertEqual(self.run_cli(["Ada", "--title", "Dr."]), "Dr. Hello, Ada!")\n'
            "AssertionError: 'Dr.: Hello, Ada!' != 'Dr. Hello, Ada!'\n"
            "- Dr.: Hello, Ada!\n"
            "+ Dr. Hello, Ada!\n"
            "\n"
            "----------------------------------------------------------------------\n"
        ),
    )

    evidence = planner.build_failure_evidence(session, failed_run)

    assert evidence.repair_brief is not None
    assert evidence.repair_brief.expected_semantics == ["Validation should produce: Dr. Hello, Ada!"]
    assert evidence.repair_brief.observed_semantics == ["Validation currently produces: Dr.: Hello, Ada!"]


def test_validation_planner_prefers_implementation_region_hint_over_test_line(tmp_path):
    planner = ValidationPlanner()
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    (pkg / "__main__.py").write_text("def main(argv=None):\n    print('hi')\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_cli.py").write_text("pass\n", encoding="utf-8")

    session = SessionState(
        task="Repair the CLI greeting formatting.",
        workspace_root=str(tmp_path),
        edit_generation=1,
    )
    session.changed_files.append(FileChangeRecord(path="greet_cli/__main__.py", operation="modify"))
    failed_run = ValidationRunRecord(
        command="python -m unittest tests.test_cli",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=1,
        summary="Validation command exited with 1.",
        excerpt=(
            "Traceback (most recent call last):\n"
            f'  File "{tmp_path / "tests" / "test_cli.py"}", line 19, in test_title_flag\n'
            '    self.assertEqual(self.run_cli(["Ada", "--title", "Dr."]), "Dr. Hello, Ada!")\n'
            f'  File "{tmp_path / "greet_cli" / "__main__.py"}", line 11, in main\n'
            "    print(greeting)\n"
            "AssertionError: 'Dr.: Hello, Ada!' != 'Dr. Hello, Ada!'\n"
            "- Dr.: Hello, Ada!\n"
            "+ Dr. Hello, Ada!\n"
        ),
    )

    evidence = planner.build_failure_evidence(session, failed_run)

    assert evidence.repair_brief is not None
    assert evidence.repair_brief.primary_target == "greet_cli/__main__.py"
    assert evidence.repair_brief.implicated_region_hint == "greet_cli/__main__.py:line 11"


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


def test_validation_planner_infers_interactive_snake_features():
    planner = ValidationPlanner()
    snapshot = WorkspaceSnapshot(
        root="/tmp/demo",
        file_count=1,
        language_counts={"html": 1},
        top_directories=[],
        important_files=["index.html"],
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
        "Erstelle ein kleines spielbares Snake-Spiel als HTML-Datei mit Tastatursteuerung, Punktestand und Game-Over-Neustart.",
        snapshot,
        changed_files=["index.html"],
    )

    structural = next(item for item in plan if item.command.startswith("internal:web_artifact:"))
    payload = json.loads(structural.command.partition("internal:web_artifact:")[2])

    assert "score" in payload[0]["expected_features"]
    assert "keyboard_controls" in payload[0]["expected_features"]
    assert "game_over" in payload[0]["expected_features"]
    assert "start_controls" in payload[0]["expected_features"]


def test_validation_planner_accepts_small_standalone_web_structural_proxy():
    planner = ValidationPlanner()
    session = SessionState(
        task="Erstelle ein kleines spielbares Snake-Spiel als HTML-Datei mit Tastatursteuerung, Punktestand und Game-Over-Neustart.",
        workspace_root="/tmp/demo",
        edit_generation=1,
        workspace_snapshot=WorkspaceSnapshot(
            root="/tmp/demo",
            file_count=1,
            language_counts={"html": 1},
            top_directories=[],
            important_files=["index.html"],
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
        ),
    )
    session.changed_files.append(FileChangeRecord(path="index.html", operation="create"))
    session.validation_runs.append(
        ValidationRunRecord(
            command='internal:web_artifact:[{"path":"index.html","expected_features":["score","keyboard_controls","game_over","start_controls"]}]',
            verification_scope="structural",
            status="passed",
            edit_generation=1,
        )
    )

    assert planner.web_structural_proxy_sufficient(session) is True


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


def test_validation_planner_prioritizes_structural_failure_paths_over_generic_artifact_order(tmp_path):
    planner = ValidationPlanner()
    for name in ["index.html", "about.html", "projects.html", "contact.html", "styles.css"]:
        (tmp_path / name).write_text("", encoding="utf-8")

    session = SessionState(
        task="Repair the generated portfolio site.",
        workspace_root=str(tmp_path),
        edit_generation=1,
    )
    failed_run = ValidationRunRecord(
        command=(
            'internal:web_artifact:['
            '{"path":"index.html","expected_features":["menu"]},'
            '{"path":"about.html","expected_features":["team"]},'
            '{"path":"projects.html","expected_features":["cards"]},'
            '{"path":"contact.html","expected_features":["form"]},'
            '{"path":"styles.css","expected_features":["layout"]}'
            "]"
        ),
        kind="check",
        verification_scope="structural",
        status="failed",
        edit_generation=1,
        summary="Structural web validation failed.",
        excerpt=(
            "index.html: refs ok; JS parse skipped: node unavailable (0 source(s)); expected features: menu; markers: no obvious interactive markers.\n"
            "about.html: refs ok; JS parse skipped: node unavailable (0 source(s)); expected features: menu; markers: no obvious interactive markers.\n"
            "contact.html: refs ok; JS parse skipped: node unavailable (0 source(s)); expected features: menu; markers: form, input, button.\n"
            "Structural web checks only; no browser/runtime smoke test was executed.\n"
            "projects.html -> projekt1.jpg\n"
            "projects.html -> projekt2.jpg\n"
            "projects.html -> projekt3.jpg"
        ),
    )

    evidence = planner.build_failure_evidence(session, failed_run)

    assert evidence.artifact_paths[0] == "projects.html"
    assert evidence.excerpt.startswith("projects.html -> projekt1.jpg")
    assert evidence.repair_brief.primary_target == "projects.html"
    assert evidence.failure_summary.startswith("projects.html -> projekt1.jpg")
    assert any("projects.html" in item for item in evidence.repair_requirements)


def test_validation_planner_uses_structural_diagnostic_file_mentions_as_primary_target(tmp_path):
    planner = ValidationPlanner()
    for name in ["index.html", "about.html", "projects.html", "contact.html", "styles.css"]:
        (tmp_path / name).write_text("", encoding="utf-8")

    session = SessionState(
        task="Repair the generated portfolio site.",
        workspace_root=str(tmp_path),
        edit_generation=1,
    )
    failed_run = ValidationRunRecord(
        command=(
            'internal:web_artifact:['
            '{"path":"index.html","expected_features":["menu"]},'
            '{"path":"about.html","expected_features":["team"]},'
            '{"path":"projects.html","expected_features":["cards"]},'
            '{"path":"contact.html","expected_features":["contact-form"]},'
            '{"path":"styles.css","expected_features":["layout"]}'
            "]"
        ),
        kind="check",
        verification_scope="structural",
        status="failed",
        edit_generation=1,
        summary="Structural web validation failed.",
        excerpt="Portfolio validation still reports missing required features.",
    )
    session.diagnostics.append(
        DiagnosticRecord(
            source="run_tests",
            category="command_failure",
            summary="contact.html is still missing the required contact-form marker.",
            tool_name="run_tests",
            command=failed_run.command,
            file_hints=["contact.html"],
            excerpt="contact.html: missing expected web features (contact-form)",
        )
    )

    evidence = planner.build_failure_evidence(session, failed_run)

    assert evidence.artifact_paths[0] == "contact.html"
    assert evidence.repair_brief.primary_target == "contact.html"
    assert evidence.missing_features == ["contact-form"]
    assert evidence.failure_summary == "contact.html is missing validation-required features: contact-form."
    assert any("contact.html" in item for item in evidence.repair_requirements)


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
