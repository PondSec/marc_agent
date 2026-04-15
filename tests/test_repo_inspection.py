from __future__ import annotations

import pytest

from agent.memory import RepoMemoryStore
from config.settings import AppConfig
from runtime.workspace import WorkspaceManager


def test_repo_snapshot_detects_manifests_validation_and_repo_map(tmp_path):
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("import json\n\ndef main():\n    return 'ok'\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "demo"
dependencies = ["pytest>=8", "ruff>=0.5"]

[tool.pytest.ini_options]
addopts = "-q"

[tool.ruff]
line-length = 100
""".strip(),
        encoding="utf-8",
    )

    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    memory = RepoMemoryStore(config, WorkspaceManager(tmp_path))

    snapshot = memory.build_snapshot("auth and tests")

    assert "pyproject.toml" in snapshot.manifests
    assert any(item.command == "python -m pytest" for item in snapshot.validation_commands)
    assert any(item.command == "ruff check ." for item in snapshot.validation_commands)
    assert snapshot.repo_map
    assert "src/app.py" in snapshot.import_hotspots
    assert snapshot.test_mappings == ["tests/test_app.py -> src/app.py"]
    assert snapshot.symbol_index["src/app.py"] == ["main"]
    assert "python" in snapshot.project_labels
    assert any(item.path == "tests/test_app.py" for item in snapshot.file_insights)


def test_repo_snapshot_ignores_external_symlinked_virtualenv(tmp_path, tmp_path_factory):
    external_dir = tmp_path_factory.mktemp("repo_inspection_external")
    external_python = external_dir / "python3.14"
    external_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    symlink_path = tmp_path / ".venv" / "bin" / "python3.14"
    try:
        symlink_path.symlink_to(external_python)
    except OSError:
        pytest.skip("symlink creation is not available on this platform")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main():\n    return 'ok'\n", encoding="utf-8")

    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    memory = RepoMemoryStore(config, WorkspaceManager(tmp_path))

    snapshot = memory.build_snapshot("python")

    assert snapshot.file_count == 1
    assert snapshot.important_files == ["src/app.py"]


def test_repo_snapshot_detects_unittest_command_from_python_test_files_without_manifest(tmp_path):
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    (tmp_path / "cli.py").write_text("def main():\n    return 0\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_cli.py").write_text(
        (
            "import unittest\n\n"
            "class CliTests(unittest.TestCase):\n"
            "    def test_ok(self):\n"
            "        self.assertTrue(True)\n"
        ),
        encoding="utf-8",
    )

    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    memory = RepoMemoryStore(config, WorkspaceManager(tmp_path))

    snapshot = memory.build_snapshot("cli unittest")

    assert any(item.command == "python -m unittest" for item in snapshot.validation_commands)


def test_repo_snapshot_detects_pytest_command_from_pytest_style_test_files_without_manifest(tmp_path):
    (tmp_path / "checkout_app").mkdir()
    (tmp_path / "checkout_app" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "checkout_app" / "totals.py").write_text(
        "def build_checkout_summary(order):\n"
        "    return {'item_count': len(order.get(\"items\", []))}\n",
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_totals.py").write_text(
        "from checkout_app.totals import build_checkout_summary\n\n"
        "def test_empty_order_has_zero_items():\n"
        "    assert build_checkout_summary({'items': []})['item_count'] == 0\n",
        encoding="utf-8",
    )

    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    memory = RepoMemoryStore(config, WorkspaceManager(tmp_path))

    snapshot = memory.build_snapshot("checkout crash tests")

    assert any(item.command == "python -m pytest" for item in snapshot.validation_commands)


def test_repo_snapshot_infers_entrypoint_from_main_symbol_when_name_is_not_builtin(tmp_path):
    (tmp_path / "normalize_cli.py").write_text(
        "def main(argv=None):\n"
        "    return ' '.join(argv or [])\n",
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_normalize.py").write_text(
        "import unittest\n\n"
        "class NormalizeTests(unittest.TestCase):\n"
        "    def test_ok(self):\n"
        "        self.assertTrue(True)\n",
        encoding="utf-8",
    )

    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    memory = RepoMemoryStore(config, WorkspaceManager(tmp_path))

    snapshot = memory.build_snapshot("cli unittest")

    assert "normalize_cli.py" in snapshot.entrypoints
    assert snapshot.symbol_index["normalize_cli.py"] == ["main"]


def test_repo_snapshot_collects_service_files_and_symbols_for_repo_map(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "router.py").write_text(
        "from .service import fetch_user\n\n"
        "def route_request(user_id):\n"
        "    return fetch_user(user_id)\n",
        encoding="utf-8",
    )
    (tmp_path / "app" / "service.py").write_text(
        "class AuthService:\n"
        "    pass\n\n"
        "def fetch_user(user_id):\n"
        "    return {'id': user_id}\n",
        encoding="utf-8",
    )

    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    memory = RepoMemoryStore(config, WorkspaceManager(tmp_path))

    snapshot = memory.build_snapshot("service auth router")

    assert "app/router.py" in snapshot.service_files
    assert "app/service.py" in snapshot.service_files
    assert snapshot.symbol_index["app/service.py"][:2] == ["AuthService", "fetch_user"]


def test_repo_snapshot_focus_matches_late_repo_specific_terms_from_large_prompt(tmp_path):
    (tmp_path / "agent").mkdir()
    (tmp_path / "server").mkdir()
    (tmp_path / "agent" / "planner.py").write_text("def plan():\n    return 'ok'\n", encoding="utf-8")
    (tmp_path / "agent" / "prompts.py").write_text("def build_prompt():\n    return 'prompt'\n", encoding="utf-8")
    (tmp_path / "agent" / "layered_memory.py").write_text("class LayeredMemory:\n    pass\n", encoding="utf-8")
    (tmp_path / "agent" / "task_state.py").write_text("class TaskState:\n    pass\n", encoding="utf-8")
    (tmp_path / "server" / "app.py").write_text("def create_app():\n    return object()\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")

    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    memory = RepoMemoryStore(config, WorkspaceManager(tmp_path))

    prompt = (
        "Analysiere dieses Projekt gruendlich und belastbar. "
        "Ich moechte eine Architektur-Zusammenfassung, die spaete Anforderungen nicht verliert. "
        "Erklaere danach planner, prompts, layered_memory, task_state und server."
    )
    snapshot = memory.build_snapshot(prompt)

    assert "agent/planner.py" in snapshot.focus_files
    assert "agent/prompts.py" in snapshot.focus_files
    assert "agent/layered_memory.py" in snapshot.focus_files
    assert "agent/task_state.py" in snapshot.focus_files
    assert "server/app.py" in snapshot.focus_files
