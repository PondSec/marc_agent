from __future__ import annotations

from config.settings import AppConfig
from llm.schemas import PatchFileArgs, TextPatch
from runtime.workspace import WorkspaceManager
from tools.filesystem import FileSystemTools
from tools.safety import SafetyManager


def test_patch_file_updates_content_and_diff(tmp_path):
    target = tmp_path / "service.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")

    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    workspace = WorkspaceManager(tmp_path)
    safety = SafetyManager(config, workspace)
    filesystem = FileSystemTools(config, workspace, safety)

    result = filesystem.patch_file(
        PatchFileArgs(
            path="service.py",
            patches=[TextPatch(old="VALUE = 1", new="VALUE = 2", expected_count=1)],
        )
    )

    assert result["success"] is True
    assert "VALUE = 2" in target.read_text(encoding="utf-8")
    assert "-VALUE = 1" in result["diff"]
    assert "+VALUE = 2" in result["diff"]
