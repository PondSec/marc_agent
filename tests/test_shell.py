from __future__ import annotations

import json

from config.settings import AppConfig
from llm.schemas import RunShellArgs, RunTestsArgs
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


def test_shell_internal_validation_handles_python_and_html_defaults(tmp_path):
    (tmp_path / "game.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "index.html").write_text('<script src="snake.js"></script>\n', encoding="utf-8")
    (tmp_path / "snake.js").write_text("console.log('ok');\n", encoding="utf-8")

    config = AppConfig(workspace_root=str(tmp_path), access_mode="full").normalized()
    config.ensure_state_dirs()
    workspace = WorkspaceManager(tmp_path)
    safety = SafetyManager(config, workspace)
    shell = ShellTools(config, workspace, safety)

    python_result = shell.run_tests(
        RunTestsArgs(
            command=f'internal:python_syntax:{json.dumps(["game.py"])}',
            cwd=".",
        )
    )
    html_result = shell.run_tests(
        RunTestsArgs(
            command=f'internal:html_refs:{json.dumps(["index.html"])}',
            cwd=".",
        )
    )

    assert python_result["success"] is True
    assert html_result["success"] is True


def test_shell_internal_python_cli_smoke_runs_interactive_script(tmp_path):
    (tmp_path / "tic_tac_toe.py").write_text(
        (
            "choice = input('move: ')\n"
            "print(f'you chose {choice}')\n"
        ),
        encoding="utf-8",
    )

    config = AppConfig(workspace_root=str(tmp_path), access_mode="full").normalized()
    config.ensure_state_dirs()
    workspace = WorkspaceManager(tmp_path)
    safety = SafetyManager(config, workspace)
    shell = ShellTools(config, workspace, safety)

    result = shell.run_tests(
        RunTestsArgs(
            command=f'internal:python_cli_smoke:{json.dumps(["tic_tac_toe.py"])}',
            cwd=".",
        )
    )

    assert result["success"] is True
    assert "you chose 1" in result["stdout"]
