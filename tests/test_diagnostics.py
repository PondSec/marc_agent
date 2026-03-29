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


def test_failure_analyzer_adds_discovery_hint_when_no_tests_ran(tmp_path):
    analyzer = FailureAnalyzer(WorkspaceManager(tmp_path))
    result = ToolRunResult(
        tool_name="run_tests",
        success=False,
        message="Validation command did not execute any tests.",
        data={
            "command": "python -m unittest",
            "stdout": "Ran 0 tests in 0.000s\n\nOK\n",
            "stderr": "Validation command did not execute any tests.",
            "exit_code": 0,
        },
    )

    diagnostics = analyzer.analyze(result, iteration=4)

    assert len(diagnostics) == 1
    assert diagnostics[0].category == "test_failure"
    assert any("test discovery" in hint for hint in diagnostics[0].action_hints)


def test_failure_analyzer_ignores_pseudo_and_external_traceback_paths(tmp_path):
    analyzer = FailureAnalyzer(WorkspaceManager(tmp_path))
    result = ToolRunResult(
        tool_name="run_tests",
        success=False,
        message="Validation command exited with 1.",
        data={
            "command": "python -m unittest discover -s tests -v",
            "stderr": (
                'Traceback (most recent call last):\n'
                '  File "<frozen runpy>", line 198, in _run_module_as_main\n'
                '  File "/usr/lib/python3.14/unittest/main.py", line 242, in _do_discovery\n'
                "ImportError: Start directory is not importable: 'tests'\n"
            ),
            "exit_code": 1,
        },
    )

    diagnostics = analyzer.analyze(result, iteration=5)

    assert len(diagnostics) == 1
    assert diagnostics[0].file_hints == []
