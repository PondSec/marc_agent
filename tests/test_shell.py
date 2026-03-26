from __future__ import annotations

from config.settings import AppConfig
from llm.schemas import RunShellArgs
from runtime.workspace import WorkspaceManager
from tools.safety import SafetyManager
from tools.shell import ShellTools


def test_shell_command_timeout(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path), shell_timeout=1, access_mode="full").normalized()
    config.ensure_state_dirs()
    workspace = WorkspaceManager(tmp_path)
    safety = SafetyManager(config, workspace)
    shell = ShellTools(config, workspace, safety)

    result = shell.run_shell(
        RunShellArgs(
            command='python3 -c "import time; time.sleep(2)"',
            cwd=".",
            timeout=1,
        )
    )

    assert result["success"] is False
    assert result["timeout"] is True
