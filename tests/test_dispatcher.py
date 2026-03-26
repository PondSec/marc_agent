from __future__ import annotations

from agent.memory import RepoMemoryStore
from config.settings import AppConfig
from runtime.logger import AgentLogger
from runtime.tool_dispatcher import ToolDispatcher
from runtime.workspace import WorkspaceManager
from tools.filesystem import FileSystemTools
from tools.gittools import GitTools
from tools.registry import build_default_registry
from tools.safety import SafetyManager
from tools.search import SearchTools
from tools.shell import ShellTools


def build_dispatcher(tmp_path) -> ToolDispatcher:
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    workspace = WorkspaceManager(tmp_path)
    safety = SafetyManager(config, workspace)
    memory = RepoMemoryStore(config, workspace)
    filesystem = FileSystemTools(config, workspace, safety)
    search = SearchTools(config, workspace, memory)
    shell = ShellTools(config, workspace, safety)
    gittools = GitTools(config, workspace)
    registry = build_default_registry(filesystem, search, shell, gittools)
    logger = AgentLogger(config.log_dir_path, "test-session", verbose=False)
    return ToolDispatcher(registry, logger, safety)


def test_dispatcher_rejects_invalid_tool_args(tmp_path):
    dispatcher = build_dispatcher(tmp_path)

    result = dispatcher.dispatch("read_file", {}, iteration=1)

    assert result.success is False
    assert "validation" in result.message.lower()
