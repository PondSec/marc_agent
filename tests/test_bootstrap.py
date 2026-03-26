from __future__ import annotations

import subprocess
from pathlib import Path

import bootstrap_runtime
from bootstrap_runtime import (
    build_pip_install_command,
    ensure_ollama_runtime,
    ensure_runtime_dependencies,
    runtime_python_path,
    runtime_venv_path,
)


def test_build_pip_install_command_uses_user_scope_when_requested(monkeypatch):
    monkeypatch.setenv("MARC_A1_PIP_SCOPE", "user")

    command = build_pip_install_command(Path("requirements-runtime.txt"))

    assert "--user" in command
    assert command[-2:] == ["-r", "requirements-runtime.txt"]


def test_build_pip_install_command_skips_user_scope_when_global(monkeypatch):
    monkeypatch.setenv("MARC_A1_PIP_SCOPE", "global")

    command = build_pip_install_command(Path("requirements-runtime.txt"))

    assert "--user" not in command
    assert command[-2:] == ["-r", "requirements-runtime.txt"]


def test_runtime_venv_path_uses_repo_local_state_dir(tmp_path):
    target = runtime_venv_path(tmp_path)

    assert target == tmp_path / ".marc_a1" / "runtime-venv"


def test_runtime_python_path_matches_platform_layout(tmp_path):
    python_path = runtime_python_path(tmp_path / ".marc_a1" / "runtime-venv")

    if bootstrap_runtime.os.name == "nt":
        assert python_path == tmp_path / ".marc_a1" / "runtime-venv" / "Scripts" / "python.exe"
    else:
        assert python_path == tmp_path / ".marc_a1" / "runtime-venv" / "bin" / "python"


def test_display_path_for_log_falls_back_to_absolute_outside_project_root(tmp_path):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    external_path = tmp_path / "python" / "bin" / "python3.14"
    external_path.parent.mkdir(parents=True)

    display = bootstrap_runtime._display_path_for_log(external_path, project_root)

    assert display == external_path.as_posix()


def test_ensure_runtime_dependencies_falls_back_to_runtime_venv(monkeypatch, tmp_path):
    requirements = tmp_path / "requirements-runtime.txt"
    requirements.write_text("fastapi>=0.1\n", encoding="utf-8")

    run_calls: list[list[str]] = []
    activated: list[Path] = []
    state = {"checks": 0}

    def fake_modules_available(modules):
        state["checks"] += 1
        return state["checks"] >= 2

    def fake_run(command, *, capture_output=False):
        run_calls.append(command)
        if command[:4] == [bootstrap_runtime.sys.executable, "-m", "pip", "install"]:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="error: externally-managed-environment",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def fake_ensure_runtime_venv(venv_dir):
        return runtime_python_path(venv_dir)

    def fake_activate(venv_dir):
        activated.append(venv_dir)

    monkeypatch.setattr(bootstrap_runtime, "_modules_available", fake_modules_available)
    monkeypatch.setattr(bootstrap_runtime, "_ensure_pip", lambda: None)
    monkeypatch.setattr(bootstrap_runtime, "_run_command", fake_run)
    monkeypatch.setattr(bootstrap_runtime, "_ensure_runtime_venv", fake_ensure_runtime_venv)
    monkeypatch.setattr(bootstrap_runtime, "_activate_runtime_venv", fake_activate)
    monkeypatch.setattr(bootstrap_runtime, "__file__", str(tmp_path / "bootstrap_runtime.py"))
    monkeypatch.setattr(
        bootstrap_runtime,
        "_modules_available_in_interpreter",
        lambda python, modules: False,
    )

    ensure_runtime_dependencies(requirements_file=str(requirements.relative_to(tmp_path)))

    assert any(command[:4] == [bootstrap_runtime.sys.executable, "-m", "pip", "install"] for command in run_calls)
    assert any(str(runtime_python_path(runtime_venv_path(tmp_path))) == command[0] for command in run_calls)
    assert activated == [runtime_venv_path(tmp_path)]


def test_ensure_runtime_dependencies_reuses_existing_runtime_venv(monkeypatch, tmp_path):
    runtime_venv = runtime_venv_path(tmp_path)
    runtime_python = runtime_python_path(runtime_venv)
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    activated: list[Path] = []

    monkeypatch.setattr(bootstrap_runtime, "__file__", str(tmp_path / "bootstrap_runtime.py"))
    monkeypatch.setattr(bootstrap_runtime, "_modules_available", lambda modules: False if not activated else True)
    monkeypatch.setattr(
        bootstrap_runtime,
        "_modules_available_in_interpreter",
        lambda python, modules: str(python) == str(runtime_python),
    )
    monkeypatch.setattr(
        bootstrap_runtime,
        "_activate_runtime_venv",
        lambda venv_dir: activated.append(venv_dir),
    )
    monkeypatch.setattr(
        bootstrap_runtime,
        "_run_command",
        lambda command, *, capture_output=False: (_ for _ in ()).throw(
            AssertionError("pip should not run when an existing runtime venv is reusable")
        ),
    )

    ensure_runtime_dependencies(requirements_file="requirements-runtime.txt")

    assert activated == [runtime_venv]


def test_ensure_ollama_runtime_installs_and_starts_ollama_on_windows(monkeypatch, tmp_path):
    local_appdata = tmp_path / "LocalAppData"
    candidate = local_appdata / "Programs" / "Ollama" / "ollama.exe"
    install_calls: list[str] = []
    start_calls: list[Path] = []
    ready_checks = iter([False, False])

    monkeypatch.setattr(bootstrap_runtime.os, "name", "nt", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))
    monkeypatch.setattr(bootstrap_runtime.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        bootstrap_runtime,
        "_windows_ollama_candidate_paths",
        lambda: [candidate],
    )
    monkeypatch.setattr(
        bootstrap_runtime,
        "_ollama_api_ready",
        lambda host: next(ready_checks, False),
    )
    monkeypatch.setattr(bootstrap_runtime, "_wait_for_ollama_api", lambda host, timeout_seconds: True)

    def fake_install():
        install_calls.append("install")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text("", encoding="utf-8")

    monkeypatch.setattr(bootstrap_runtime, "_install_ollama_windows", fake_install)
    monkeypatch.setattr(
        bootstrap_runtime,
        "_start_ollama_server",
        lambda binary: start_calls.append(binary),
    )

    ensure_ollama_runtime("http://127.0.0.1:11434")

    assert install_calls == ["install"]
    assert start_calls == [candidate]


def test_ensure_ollama_runtime_raises_clear_error_without_local_ollama(monkeypatch):
    monkeypatch.setattr(bootstrap_runtime.os, "name", "posix", raising=False)
    monkeypatch.setattr(bootstrap_runtime.shutil, "which", lambda name: None)
    monkeypatch.setattr(bootstrap_runtime, "_ollama_api_ready", lambda host: False)
    monkeypatch.setattr(bootstrap_runtime, "_find_ollama_binary", lambda: None)

    try:
        ensure_ollama_runtime("http://127.0.0.1:11434")
    except RuntimeError as exc:
        assert "ollama serve" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when Ollama is missing")
