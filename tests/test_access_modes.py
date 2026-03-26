from __future__ import annotations

from config.settings import AppConfig
from llm.schemas import WriteFileArgs
from runtime.logger import AgentLogger
from runtime.tool_dispatcher import ToolDispatcher
from runtime.workspace import WorkspaceManager
from agent.memory import RepoMemoryStore
from tools.filesystem import FileSystemTools
from tools.gittools import GitTools
from tools.registry import build_default_registry
from tools.safety import SafetyManager
from tools.search import SearchTools
from tools.shell import ShellTools


def build_dispatcher(tmp_path, *, access_mode: str) -> ToolDispatcher:
    config = AppConfig(workspace_root=str(tmp_path), access_mode=access_mode).normalized()
    config.ensure_state_dirs()
    workspace = WorkspaceManager(tmp_path)
    safety = SafetyManager(config, workspace)
    memory = RepoMemoryStore(config, workspace)
    filesystem = FileSystemTools(config, workspace, safety)
    search = SearchTools(config, workspace, memory)
    shell = ShellTools(config, workspace, safety)
    gittools = GitTools(config, workspace)
    registry = build_default_registry(filesystem, search, shell, gittools)
    logger = AgentLogger(config.log_dir_path, f"test-{access_mode}", verbose=False)
    return ToolDispatcher(registry, logger, safety)


def test_safe_mode_blocks_write_tools(tmp_path):
    dispatcher = build_dispatcher(tmp_path, access_mode="safe")

    result = dispatcher.dispatch(
        "write_file",
        WriteFileArgs(path="demo.txt", content="hello").model_dump(),
        iteration=1,
    )

    assert result.success is False
    assert result.data["blocked"] is True
    assert "safe mode" in result.message.lower()


def test_full_access_allows_medium_shell_commands(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path), access_mode="full").normalized()
    config.ensure_state_dirs()
    safety = SafetyManager(config, WorkspaceManager(tmp_path))

    assessment = safety.assess_shell_command("mkdir build")

    assert assessment.allowed is True
    assert assessment.risk_level == "medium"


def test_approval_mode_blocks_medium_shell_commands(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path), access_mode="approval").normalized()
    config.ensure_state_dirs()
    safety = SafetyManager(config, WorkspaceManager(tmp_path))

    assessment = safety.assess_shell_command("mkdir build")

    assert assessment.allowed is False
    assert assessment.risk_level == "medium"
