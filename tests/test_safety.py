from __future__ import annotations

from config.settings import AppConfig
from runtime.workspace import WorkspaceManager
from tools.safety import SafetyManager


def make_safety(tmp_path, **kwargs) -> SafetyManager:
    config = AppConfig(workspace_root=str(tmp_path), **kwargs)
    config.ensure_state_dirs()
    workspace = WorkspaceManager(tmp_path)
    return SafetyManager(config, workspace)


def test_blocks_destructive_command(tmp_path):
    safety = make_safety(tmp_path)
    assessment = safety.assess_shell_command("rm -rf /")

    assert assessment.allowed is False
    assert assessment.risk_level == "blocked"


def test_blocks_network_when_disabled(tmp_path):
    safety = make_safety(tmp_path, allow_network=False)
    assessment = safety.assess_shell_command("curl https://example.com")

    assert assessment.allowed is False
    assert assessment.risk_level == "blocked"


def test_allows_pytest_command(tmp_path):
    safety = make_safety(tmp_path)
    assessment = safety.assess_shell_command("pytest -q")

    assert assessment.allowed is True
    assert assessment.risk_level == "low"
