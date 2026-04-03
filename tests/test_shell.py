from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest

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


def test_shell_structural_web_validation_checks_dom_refs_and_js(monkeypatch, tmp_path):
    (tmp_path / "snake.html").write_text(
        (
            "<html><body>"
            "<nav id='menu'><button>Start</button></nav>"
            "<div id='highscore'>Highscore</div>"
            "<script src='snake.js'></script>"
            "</body></html>"
        ),
        encoding="utf-8",
    )
    (tmp_path / "snake.js").write_text("const best = localStorage.getItem('highscore');\n", encoding="utf-8")

    config = AppConfig(workspace_root=str(tmp_path), access_mode="full").normalized()
    config.ensure_state_dirs()
    workspace = WorkspaceManager(tmp_path)
    safety = SafetyManager(config, workspace)
    shell = ShellTools(config, workspace, safety)

    monkeypatch.setattr("tools.shell.shutil.which", lambda name: "/usr/bin/node" if name == "node" else None)
    monkeypatch.setattr(
        "tools.shell.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr=""),
    )

    result = shell.run_tests(
        RunTestsArgs(
            command='internal:web_artifact:[{"path":"snake.html","expected_features":["menu","highscore"]}]',
            cwd=".",
        )
    )

    assert result["success"] is True
    assert "Structural web checks only" in result["stdout"]
    assert "expected features: menu, highscore" in result["stdout"]


def test_shell_structural_web_validation_accepts_interactive_snake_markers(monkeypatch, tmp_path):
    (tmp_path / "index.html").write_text(
        (
            "<html><body>"
            "<canvas id='gameCanvas' width='400' height='400'></canvas>"
            "<div id='scoreBoard'>Score: 0</div>"
            "<script>"
            "document.addEventListener('keydown', (event) => {"
            "  if (event.key === 'ArrowUp') window.direction = 'up';"
            "});"
            "function endGame(){ alert('Game Over!'); document.location.reload(); }"
            "</script>"
            "</body></html>"
        ),
        encoding="utf-8",
    )

    config = AppConfig(workspace_root=str(tmp_path), access_mode="full").normalized()
    config.ensure_state_dirs()
    workspace = WorkspaceManager(tmp_path)
    safety = SafetyManager(config, workspace)
    shell = ShellTools(config, workspace, safety)

    monkeypatch.setattr("tools.shell.shutil.which", lambda name: "/usr/bin/node" if name == "node" else None)
    monkeypatch.setattr(
        "tools.shell.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr=""),
    )

    result = shell.run_tests(
        RunTestsArgs(
            command='internal:web_artifact:[{"path":"index.html","expected_features":["score","keyboard_controls","game_over","start_controls"]}]',
            cwd=".",
        )
    )

    assert result["success"] is True
    assert "expected features: score, keyboard_controls, game_over, start_controls" in result["stdout"]


def test_shell_structural_web_validation_catches_dom_id_mismatch_across_referenced_files(monkeypatch, tmp_path):
    (tmp_path / "index.html").write_text(
        (
            "<html><head><link rel='stylesheet' href='styles.css'></head><body>"
            "<label class='theme-switcher'><input type='checkbox' id='themeSwitch' /></label>"
            "<p id='feedbackStatus' aria-live='polite'></p>"
            "<script src='app.js'></script>"
            "</body></html>"
        ),
        encoding="utf-8",
    )
    (tmp_path / "styles.css").write_text(
        "[data-theme='dark'] { color-scheme: dark; }\n",
        encoding="utf-8",
    )
    (tmp_path / "app.js").write_text(
        (
            "const themeSwitcher = document.getElementById('themeSwitcher');\n"
            "themeSwitcher?.addEventListener('change', (event) => {\n"
            "  localStorage.setItem('theme', event.target.value);\n"
            "});\n"
        ),
        encoding="utf-8",
    )

    config = AppConfig(workspace_root=str(tmp_path), access_mode="full").normalized()
    config.ensure_state_dirs()
    workspace = WorkspaceManager(tmp_path)
    safety = SafetyManager(config, workspace)
    shell = ShellTools(config, workspace, safety)

    monkeypatch.setattr("tools.shell.shutil.which", lambda name: "/usr/bin/node" if name == "node" else None)
    monkeypatch.setattr(
        "tools.shell.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr=""),
    )

    result = shell.run_tests(
        RunTestsArgs(
            command='internal:web_artifact:[{"path":"index.html","expected_features":[]}]',
            cwd=".",
        )
    )

    assert result["success"] is False
    assert "missing DOM ids referenced by JS (themeSwitcher)" in result["stderr"]


def test_shell_web_runtime_smoke_executes_small_interactive_page(tmp_path):
    node_binary = shutil.which("node")
    if not node_binary:
        pytest.skip("node is required for web runtime smoke validation")

    (tmp_path / "index.html").write_text(
        (
            "<html><body>"
            "<form id='feedbackForm'>"
            "<textarea id='feedbackInput' name='feedback'></textarea>"
            "<button type='submit'>Save</button>"
            "</form>"
            "<p id='feedbackStatus'></p>"
            "<script src='app.js'></script>"
            "</body></html>"
        ),
        encoding="utf-8",
    )
    (tmp_path / "app.js").write_text(
        (
            "const form = document.getElementById('feedbackForm');\n"
            "const input = document.getElementById('feedbackInput');\n"
            "const status = document.getElementById('feedbackStatus');\n"
            "form?.addEventListener('submit', (event) => {\n"
            "  event.preventDefault();\n"
            "  const value = input?.value.trim() || '';\n"
            "  localStorage.setItem('feedback', value);\n"
            "  status.textContent = value || 'empty';\n"
            "  form.reset();\n"
            "});\n"
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
            command='internal:web_runtime_smoke:[{"path":"index.html","expected_features":[]}]',
            cwd=".",
        )
    )

    assert result["success"] is True
    assert "runtime smoke passed" in result["stdout"]


def test_shell_web_runtime_smoke_reports_bootstrap_errors(tmp_path):
    node_binary = shutil.which("node")
    if not node_binary:
        pytest.skip("node is required for web runtime smoke validation")

    (tmp_path / "index.html").write_text(
        (
            "<html><body>"
            "<button id='launchButton'>Launch</button>"
            "<script src='app.js'></script>"
            "</body></html>"
        ),
        encoding="utf-8",
    )
    (tmp_path / "app.js").write_text(
        (
            "document.addEventListener('DOMContentLoaded', () => {\n"
            "  missingBootstrapCall();\n"
            "});\n"
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
            command='internal:web_runtime_smoke:[{"path":"index.html","expected_features":[]}]',
            cwd=".",
        )
    )

    assert result["success"] is False
    assert "missingBootstrapCall" in result["stderr"]


def test_shell_web_runtime_smoke_supports_insert_before_and_dynamic_button_events(tmp_path):
    node_binary = shutil.which("node")
    if not node_binary:
        pytest.skip("node is required for web runtime smoke validation")

    (tmp_path / "index.html").write_text(
        (
            "<html><body>"
            "<section class='panel'><p>Ready</p></section>"
            "<script src='app.js'></script>"
            "</body></html>"
        ),
        encoding="utf-8",
    )
    (tmp_path / "app.js").write_text(
        (
            "document.addEventListener('DOMContentLoaded', () => {\n"
            "  const panel = document.querySelector('.panel');\n"
            "  const button = document.createElement('button');\n"
            "  button.textContent = 'Toggle';\n"
            "  button.addEventListener('click', () => {\n"
            "    document.body.dataset.clicked = 'yes';\n"
            "  });\n"
            "  panel.insertBefore(button, panel.firstChild);\n"
            "  setTimeout(() => {\n"
            "    if (document.body.dataset.clicked !== 'yes') {\n"
            "      throw new Error('dynamic button was not clicked');\n"
            "    }\n"
            "  }, 0);\n"
            "});\n"
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
            command='internal:web_runtime_smoke:[{"path":"index.html","expected_features":[]}]',
            cwd=".",
        )
    )

    assert result["success"] is True
    assert "runtime smoke passed" in result["stdout"]


def test_shell_marks_zero_test_unittest_run_as_insufficient_validation(monkeypatch, tmp_path):
    config = AppConfig(workspace_root=str(tmp_path), access_mode="full").normalized()
    config.ensure_state_dirs()
    workspace = WorkspaceManager(tmp_path)
    safety = SafetyManager(config, workspace)
    shell = ShellTools(config, workspace, safety)
    monkeypatch.setattr(
        shell,
        "_run_command",
        lambda command, cwd, timeout: {
            "success": True,
            "message": "Command exited with 0.",
            "risk_level": "low",
            "stdout": "Ran 0 tests in 0.000s\n\nOK\n",
            "stderr": "",
            "exit_code": 0,
            "command": command,
        },
    )

    result = shell.run_tests(
        RunTestsArgs(
            command="python -m unittest",
            cwd=".",
        )
    )

    assert result["success"] is False
    assert result["insufficient_verification"] is True
    assert "did not execute any tests" in result["message"]
    assert "Ran 0 tests" in result["stdout"]


def test_shell_runs_python_module_commands_with_current_runtime_interpreter(monkeypatch, tmp_path):
    config = AppConfig(workspace_root=str(tmp_path), access_mode="full").normalized()
    config.ensure_state_dirs()
    workspace = WorkspaceManager(tmp_path)
    safety = SafetyManager(config, workspace)
    shell = ShellTools(config, workspace, safety)
    called: dict[str, object] = {}

    def fake_run(args, **kwargs):
        called["args"] = args
        called["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr("tools.shell.subprocess.run", fake_run)

    result = shell.run_tests(
        RunTestsArgs(
            command="python -m unittest",
            cwd=".",
        )
    )

    assert result["success"] is True
    assert result["resolved_command"].startswith(sys.executable)
    assert called["args"][0:2] == ["/bin/bash", "-lc"]
    assert sys.executable in called["args"][2]
