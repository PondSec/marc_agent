from __future__ import annotations

import pytest

from runtime.workspace import WorkspaceError, WorkspaceManager


def test_workspace_rejects_path_escape(tmp_path):
    workspace = WorkspaceManager(tmp_path)
    with pytest.raises(WorkspaceError):
        workspace.resolve_path("../outside.txt")


def test_workspace_lists_only_internal_files(tmp_path):
    workspace = WorkspaceManager(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("ignored", encoding="utf-8")

    files = workspace.iter_files()

    assert [path.relative_to(tmp_path).as_posix() for path in files] == ["src/app.py"]
