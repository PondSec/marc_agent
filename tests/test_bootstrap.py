from __future__ import annotations

from pathlib import Path

from bootstrap_runtime import build_pip_install_command


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
