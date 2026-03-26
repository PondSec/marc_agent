from __future__ import annotations

from agent.models import FileChangeRecord, SessionState, ToolCallRecord, WorkspaceSnapshot
from agent.planner import Planner
from agent.prompts import decision_prompt
from llm.schemas import AgentActionType, TaskIntent


class BrokenLLM:
    def generate_json(self, *args, **kwargs):
        raise RuntimeError("llm unavailable")

    def generate(self, *args, **kwargs):
        raise RuntimeError("llm unavailable")


class CreateDraftLLM(BrokenLLM):
    def __init__(self):
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return "tic_tac_toe.py"
        return "print('tic tac toe')\n"


class ModelFirstCreateLLM(BrokenLLM):
    def __init__(self):
        self.analysis_prompts: list[str] = []

    def generate_json(self, prompt, *args, **kwargs):
        self.analysis_prompts.append(prompt)
        return {
            "summary": "Create a Python Tic Tac Toe implementation.",
            "intent": "create",
            "requires_repo_context": True,
            "should_create_files": True,
            "deliverable": "New implementation",
            "search_terms": ["tic tac toe"],
            "target_paths": [],
            "relevant_extensions": [".py"],
            "direct_response": None,
        }


class ContextAwareFollowUpLLM(BrokenLLM):
    def __init__(self):
        self.analysis_prompts: list[str] = []

    def generate_json(self, prompt, *args, **kwargs):
        self.analysis_prompts.append(prompt)
        return {
            "summary": "Modify the recently created Tic Tac Toe implementation.",
            "intent": "modify",
            "requires_repo_context": True,
            "should_create_files": False,
            "deliverable": "Updated implementation",
            "search_terms": [],
            "target_paths": ["tic_tac_toe.py"],
            "relevant_extensions": [".py"],
            "direct_response": None,
        }


def build_snapshot(tmp_path) -> WorkspaceSnapshot:
    return WorkspaceSnapshot(
        root=str(tmp_path),
        file_count=4,
        language_counts={"python": 2, "markdown": 1, "toml": 1},
        top_directories=["app", "tests"],
        important_files=["pyproject.toml", "README.md", "app/main.py", "tests/test_main.py"],
        focus_files=[],
        file_briefs={
            "pyproject.toml": "Project metadata",
            "README.md": "Usage guide",
            "app/main.py": "Main entrypoint",
        },
        manifests=["pyproject.toml", "README.md"],
        configs=["pyproject.toml"],
        test_files=["tests/test_main.py"],
        build_files=[],
        deploy_files=[],
        entrypoints=["app/main.py"],
        repo_map=["app/", "tests/"],
        project_labels=["python"],
        likely_commands=["python -m pytest"],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Small Python application with tests.",
    )


def test_task_analysis_classifies_create_request_without_filler_terms(tmp_path):
    planner = Planner(BrokenLLM(), "")
    analysis = planner.analyze_task(
        "hallo schreib bitte ein python script das ein Taschenrechner ist",
        build_snapshot(tmp_path),
    )

    assert analysis.intent == TaskIntent.CREATE
    assert analysis.should_create_files is True
    assert "hallo" not in analysis.search_terms
    assert "schreib" not in analysis.search_terms
    assert ".py" in analysis.relevant_extensions


def test_create_request_with_summary_suffix_still_classifies_as_create(tmp_path):
    planner = Planner(BrokenLLM(), "")
    snapshot = WorkspaceSnapshot(
        root=str(tmp_path),
        file_count=0,
        language_counts={},
        top_directories=[],
        important_files=[],
        focus_files=[],
        file_briefs={},
        manifests=[],
        configs=[],
        test_files=[],
        build_files=[],
        deploy_files=[],
        entrypoints=[],
        repo_map=[],
        project_labels=[],
        likely_commands=[],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Empty workspace.",
    )

    analysis = planner.analyze_task(
        "bitte programmiere fuer mich ein Tic Tac Toe spiel in python fuer mich. "
        "gib danach eine Zusammenfassung was du genau getan hast",
        snapshot,
    )

    assert analysis.intent == TaskIntent.CREATE
    assert analysis.should_create_files is True
    assert "tic tac toe" in analysis.search_terms
    assert "tic tac" not in analysis.search_terms
    assert "programmiere" not in analysis.search_terms
    assert "mich" not in analysis.search_terms
    assert "zusammenfassung" not in analysis.search_terms


def test_analyze_task_prefers_model_for_free_form_create_prompt(tmp_path):
    llm = ModelFirstCreateLLM()
    planner = Planner(llm, "")
    snapshot = WorkspaceSnapshot(
        root=str(tmp_path),
        file_count=0,
        language_counts={},
        top_directories=[],
        important_files=[],
        focus_files=[],
        file_briefs={},
        manifests=[],
        configs=[],
        test_files=[],
        build_files=[],
        deploy_files=[],
        entrypoints=[],
        repo_map=[],
        project_labels=[],
        likely_commands=[],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Empty workspace.",
    )

    analysis = planner.analyze_task("bau mir ein python tic tac toe", snapshot)

    assert analysis.intent == TaskIntent.CREATE
    assert llm.analysis_prompts


def test_analyze_task_uses_recent_thread_context_for_follow_up(tmp_path):
    llm = ContextAwareFollowUpLLM()
    planner = Planner(llm, "")
    snapshot = WorkspaceSnapshot(
        root=str(tmp_path),
        file_count=1,
        language_counts={"python": 1},
        top_directories=[],
        important_files=["tic_tac_toe.py"],
        focus_files=["tic_tac_toe.py"],
        file_briefs={"tic_tac_toe.py": "Tic Tac Toe game"},
        manifests=[],
        configs=[],
        test_files=[],
        build_files=[],
        deploy_files=[],
        entrypoints=["tic_tac_toe.py"],
        repo_map=["tic_tac_toe.py"],
        project_labels=["python"],
        likely_commands=[],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Single Python game file.",
    )
    session = SessionState(
        task="mach das anders",
        workspace_root=str(tmp_path),
        workspace_snapshot=snapshot,
    )
    session.append_message("user", "programmiere bitte ein python tic tac toe")
    session.append_message("assistant", "Ich habe tic_tac_toe.py erstellt.")
    session.append_message("user", "mach das anders")
    session.changed_files.append(FileChangeRecord(path="tic_tac_toe.py", operation="created"))

    analysis = planner.analyze_task(session.task, snapshot, session=session)

    assert analysis.intent == TaskIntent.MODIFY
    assert analysis.target_paths == ["tic_tac_toe.py"]
    assert "programmiere bitte ein python tic tac toe" in llm.analysis_prompts[0]
    assert "tic_tac_toe.py" in llm.analysis_prompts[0]


def test_fallback_create_flow_reads_manifest_before_searching_prompt_words(tmp_path):
    planner = Planner(BrokenLLM(), "")
    snapshot = build_snapshot(tmp_path)
    session = SessionState(
        task="hallo schreib bitte ein python script das ein Taschenrechner ist",
        workspace_root=str(tmp_path),
        workspace_snapshot=snapshot,
    )
    session.task_analysis = planner.analyze_task(session.task, snapshot)
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="inspect_workspace",
            tool_args={"focus": session.task},
            success=True,
            summary="Workspace inspected successfully.",
            phase="exploring",
        )
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "read_file"
    assert decision.tool_args["path"] == "pyproject.toml"


def test_decision_prompt_stays_compact_with_snapshot_context(tmp_path):
    planner = Planner(BrokenLLM(), "")
    snapshot = build_snapshot(tmp_path)
    session = SessionState(
        task="Bitte behebe den Fehler im Python-Projekt",
        workspace_root=str(tmp_path),
        workspace_snapshot=snapshot,
    )
    session.task_analysis = planner.analyze_task(session.task, snapshot)
    session.plan_summary = "Inspect error, patch file, rerun targeted checks."
    session.candidate_files = ["app/main.py", "tests/test_main.py"]
    session.verification_commands = ["python -m pytest"]

    prompt = decision_prompt(session.task, session, "tool manifest placeholder")

    assert len(prompt) < 9000


def test_empty_workspace_create_request_drafts_file_after_inspect(tmp_path):
    planner = Planner(CreateDraftLLM(), "")
    snapshot = WorkspaceSnapshot(
        root=str(tmp_path),
        file_count=0,
        language_counts={},
        top_directories=[],
        important_files=[],
        focus_files=[],
        file_briefs={},
        manifests=[],
        configs=[],
        test_files=[],
        build_files=[],
        deploy_files=[],
        entrypoints=[],
        repo_map=[],
        project_labels=[],
        likely_commands=[],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Empty workspace.",
    )
    session = SessionState(
        task="kannst du bitte ein python tic tac toe spiel schreiben?",
        workspace_root=str(tmp_path),
        workspace_snapshot=snapshot,
    )
    session.task_analysis = planner.analyze_task(session.task, snapshot)
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="inspect_workspace",
            tool_args={"focus": session.task},
            success=True,
            summary="Workspace inspected successfully.",
            phase="exploring",
        )
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "create_file"
    assert decision.tool_args["path"] == "tic_tac_toe.py"
    assert "tic tac toe" in decision.tool_args["content"].lower()


def test_empty_workspace_programmiere_request_drafts_file_after_inspect(tmp_path):
    planner = Planner(CreateDraftLLM(), "")
    snapshot = WorkspaceSnapshot(
        root=str(tmp_path),
        file_count=0,
        language_counts={},
        top_directories=[],
        important_files=[],
        focus_files=[],
        file_briefs={},
        manifests=[],
        configs=[],
        test_files=[],
        build_files=[],
        deploy_files=[],
        entrypoints=[],
        repo_map=[],
        project_labels=[],
        likely_commands=[],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Empty workspace.",
    )
    task = (
        "bitte programmiere fuer mich ein Tic Tac Toe spiel in python fuer mich. "
        "gib danach eine Zusammenfassung was du genau getan hast"
    )
    session = SessionState(
        task=task,
        workspace_root=str(tmp_path),
        workspace_snapshot=snapshot,
    )
    session.task_analysis = planner.analyze_task(session.task, snapshot)
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="inspect_workspace",
            tool_args={"focus": session.task},
            success=True,
            summary="Workspace inspected successfully.",
            phase="exploring",
        )
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "create_file"
    assert decision.tool_args["path"] == "tic_tac_toe.py"


def test_follow_up_modify_request_reads_recent_changed_file_first(tmp_path):
    (tmp_path / "tic_tac_toe.py").write_text("def main():\n    pass\n", encoding="utf-8")
    llm = ContextAwareFollowUpLLM()
    planner = Planner(llm, "")
    snapshot = WorkspaceSnapshot(
        root=str(tmp_path),
        file_count=1,
        language_counts={"python": 1},
        top_directories=[],
        important_files=["tic_tac_toe.py"],
        focus_files=["tic_tac_toe.py"],
        file_briefs={"tic_tac_toe.py": "Tic Tac Toe game"},
        manifests=[],
        configs=[],
        test_files=[],
        build_files=[],
        deploy_files=[],
        entrypoints=["tic_tac_toe.py"],
        repo_map=["tic_tac_toe.py"],
        project_labels=["python"],
        likely_commands=[],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Single Python game file.",
    )
    session = SessionState(
        task="mach das anders",
        workspace_root=str(tmp_path),
        workspace_snapshot=snapshot,
    )
    session.append_message("user", "programmiere bitte ein python tic tac toe")
    session.append_message("assistant", "Ich habe tic_tac_toe.py erstellt.")
    session.append_message("user", "mach das anders")
    session.changed_files.append(FileChangeRecord(path="tic_tac_toe.py", operation="created"))
    session.task_analysis = planner.analyze_task(session.task, snapshot, session=session)

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "read_file"
    assert decision.tool_args["path"] == "tic_tac_toe.py"


def test_summarize_session_prefers_changed_files_over_generic_workspace_message(tmp_path):
    planner = Planner(BrokenLLM(), "")
    session = SessionState(
        task="schreib bitte ein python script das ein taschenrechner ist",
        workspace_root=str(tmp_path),
        validation_status="not_run",
    )
    session.changed_files.append(
        {
            "path": "calculator.py",
            "operation": "create",
        }
    )

    summary = planner.summarize_session(session)

    assert "umgesetzt" in summary
