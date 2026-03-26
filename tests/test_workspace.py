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


def test_workspace_full_mode_can_resolve_absolute_path_outside_root(tmp_path, tmp_path_factory):
    external_dir = tmp_path_factory.mktemp("workspace_external")
    external_file = external_dir / "outside.txt"
    external_file.write_text("ok", encoding="utf-8")
    workspace = WorkspaceManager(tmp_path, allow_outside_root=True)

    resolved = workspace.resolve_path(str(external_file))

    assert resolved == external_file.resolve()
    assert workspace.display_path(resolved) == str(external_file.resolve())


def test_workspace_iter_files_skips_external_symlink_targets(tmp_path, tmp_path_factory):
    external_dir = tmp_path_factory.mktemp("workspace_external_symlink")
    external_python = external_dir / "python3.14"
    external_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    workspace = WorkspaceManager(tmp_path)
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    symlink_path = tmp_path / ".venv" / "bin" / "python3.14"
    try:
        symlink_path.symlink_to(external_python)
    except OSError:
        pytest.skip("symlink creation is not available on this platform")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')", encoding="utf-8")

    files = workspace.iter_files()

    assert [path.relative_to(tmp_path).as_posix() for path in files] == ["src/app.py"]
