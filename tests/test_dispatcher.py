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
    assert result.tool_meta is not None
    assert result.tool_meta.read_only is True
    assert result.tool_meta.concurrency_safe is True


def test_dispatcher_marks_read_only_tools_as_concurrency_safe(tmp_path):
    workspace_file = tmp_path / "notes.txt"
    workspace_file.write_text("hello\n", encoding="utf-8")
    dispatcher = build_dispatcher(tmp_path)

    result = dispatcher.dispatch("read_file", {"path": "notes.txt"}, iteration=1)

    assert result.success is True
    assert result.tool_meta is not None
    assert result.tool_meta.category == "read"
    assert result.tool_meta.read_only is True
    assert result.tool_meta.concurrency_safe is True
    assert result.tool_meta.execution_mode == "read_only"


def test_dispatcher_marks_mutating_tools_as_exclusive(tmp_path):
    dispatcher = build_dispatcher(tmp_path)

    result = dispatcher.dispatch(
        "create_file",
        {"path": "created.txt", "content": "hello\n", "overwrite": False},
        iteration=1,
    )

    assert result.success is True
    assert result.tool_meta is not None
    assert result.tool_meta.category == "write"
    assert result.tool_meta.read_only is False
    assert result.tool_meta.concurrency_safe is False
    assert result.tool_meta.execution_mode == "mutating"


def test_dispatcher_marks_validation_tools_in_runtime_metadata(tmp_path):
    dispatcher = build_dispatcher(tmp_path)

    result = dispatcher.dispatch(
        "run_tests",
        {"command": "python -m pytest -q", "cwd": ".", "timeout": 5},
        iteration=1,
    )

    assert result.tool_meta is not None
    assert result.tool_meta.verification_tool is True
