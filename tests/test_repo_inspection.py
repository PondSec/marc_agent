from __future__ import annotations

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
