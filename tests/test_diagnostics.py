from __future__ import annotations

from agent.diagnostics import FailureAnalyzer
from agent.models import ToolRunResult
from runtime.workspace import WorkspaceManager


def test_failure_analyzer_extracts_category_and_file_hints(tmp_path):
    analyzer = FailureAnalyzer(WorkspaceManager(tmp_path))
    result = ToolRunResult(
        tool_name="run_tests",
        success=False,
        message="Validation command exited with 1.",
        data={
            "command": "python -m pytest",
            "stderr": "tests/test_api.py:12: AssertionError: expected 200\n",
            "exit_code": 1,
        },
    )

    diagnostics = analyzer.analyze(result, iteration=3)

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.category == "test_failure"
    assert diagnostic.file_hints == ["tests/test_api.py"]
    assert diagnostic.line_hints == [12]
    assert diagnostic.severity == "error"
