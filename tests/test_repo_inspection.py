from __future__ import annotations

import pytest

from agent.memory import RepoMemoryStore
from config.settings import AppConfig
from runtime.workspace import WorkspaceManager


def test_repo_snapshot_detects_manifests_validation_and_repo_map(tmp_path):
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main():\n    return 'ok'\n", encoding="utf-8")
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
