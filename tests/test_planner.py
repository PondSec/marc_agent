from __future__ import annotations

from pathlib import Path

from agent.models import FileChangeRecord, SessionState, ToolCallRecord, ValidationCommand, WorkspaceSnapshot
from agent.planner import Planner
from llm.schemas import AgentActionType, RouteIntent


class ScriptedLLM:
    def __init__(self, json_payloads=None, text_payloads=None):
        self.json_payloads = list(json_payloads or [])
        self.text_payloads = list(text_payloads or [])

    def generate_json(self, *args, **kwargs):
        if not self.json_payloads:
            raise RuntimeError("No JSON payload configured")
        return self.json_payloads.pop(0)

    def generate(self, *args, **kwargs):
        if not self.text_payloads:
            raise RuntimeError("No text payload configured")
        return self.text_payloads.pop(0)


def build_snapshot(tmp_path: Path) -> WorkspaceSnapshot:
    return WorkspaceSnapshot(
        root=str(tmp_path),
        file_count=4,
        language_counts={"python": 2, "markdown": 1, "toml": 1},
        top_directories=["app", "tests"],
        important_files=["pyproject.toml", "README.md", "app/main.py", "tests/test_main.py"],
        focus_files=["app/main.py"],
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


def route_payload(
    *,
    intent: str,
    action_plan,
    target_paths=None,
    direct_response=None,
    needs_clarification=False,
    clarification_questions=None,
    safe_to_execute=True,
    repo_context_needed=True,
    target_name=None,
):
    return {
        "user_goal": "Handle the user request safely.",
        "intent": intent,
        "entities": {
            "target_type": "file" if target_paths else None,
            "target_name": target_name,
            "target_paths": target_paths or [],
            "attributes": [],
            "constraints": [],
        },
        "requested_outcome": "Produce the requested result.",
        "action_plan": action_plan,
        "needs_clarification": needs_clarification,
        "clarification_questions": clarification_questions or [],
        "confidence": 0.91 if safe_to_execute else 0.31,
        "safe_to_execute": safe_to_execute,
        "repo_context_needed": repo_context_needed,
        "search_terms": [target_name] if target_name else [],
        "relevant_extensions": [Path(target_paths[0]).suffix] if target_paths else [".py"],
        "direct_response": direct_response,
    }


def test_planner_returns_direct_response_from_router(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="explain",
                action_plan=[
                    {
                        "step": 1,
                        "action": "respond_directly",
                        "reason": "The user only wants a short explanation.",
                    }
                ],
                direct_response="Ich bin dein lokaler Agent fuer Analyse, Planung und Code-Aenderungen.",
                repo_context_needed=False,
            )
        ]
    )
    planner = Planner(llm, "")
    session = SessionState(task="Wer bist du?", workspace_root=str(tmp_path))

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.FINAL
    assert "lokaler Agent" in (decision.final_response or "")


def test_planner_asks_for_clarification_when_router_requires_it(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="unknown",
                action_plan=[
                    {
                        "step": 1,
                        "action": "ask_clarification",
                        "reason": "The target is ambiguous.",
                    }
                ],
                needs_clarification=True,
                clarification_questions=["Welche Datei oder welchen Bereich meinst du genau?"],
                safe_to_execute=False,
                repo_context_needed=False,
            )
        ]
    )
    planner = Planner(llm, "")
    session = SessionState(task="Mach das mal anders", workspace_root=str(tmp_path))

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.FINAL
    assert "Praezisierung" in (decision.final_response or "")
    assert "Welche Datei" in (decision.final_response or "")


def test_planner_reads_target_file_before_update(tmp_path):
    target = tmp_path / "app" / "main.py"
    target.parent.mkdir()
    target.write_text("print('hello')\n", encoding="utf-8")

    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="update",
                action_plan=[
                    {
                        "step": 1,
                        "action": "read_relevant_files",
                        "reason": "Inspect the current target file first.",
                    },
                    {
                        "step": 2,
                        "action": "update_artifact",
                        "reason": "Apply the requested update.",
                    },
                ],
                target_paths=["app/main.py"],
                target_name="app/main.py",
            )
        ]
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Aktualisiere app/main.py",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "read_file"
    assert decision.tool_args["path"] == "app/main.py"


def test_planner_generates_write_after_update_target_is_read(tmp_path):
    target = tmp_path / "app" / "main.py"
    target.parent.mkdir()
    target.write_text("print('hello')\n", encoding="utf-8")

    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="update",
                action_plan=[
                    {
                        "step": 1,
                        "action": "update_artifact",
                        "reason": "Apply the requested update.",
                    }
                ],
                target_paths=["app/main.py"],
                target_name="app/main.py",
            )
        ],
        text_payloads=["print('updated')\n"],
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Aktualisiere app/main.py",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="read_file",
            tool_args={"path": "app/main.py"},
            success=True,
            summary="Read app/main.py.",
            output_excerpt="print('hello')\n",
        )
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "write_file"
    assert decision.tool_args["path"] == "app/main.py"
    assert "updated" in decision.tool_args["content"]


def test_planner_uses_route_to_create_new_file(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="create",
                action_plan=[
                    {
                        "step": 1,
                        "action": "create_artifact",
                        "reason": "Create the requested new module.",
                    },
                    {
                        "step": 2,
                        "action": "run_validation",
                        "reason": "Validate the generated code.",
                    },
                ],
                target_paths=["tools/new_agent.py"],
                target_name="tools/new_agent.py",
            )
        ],
        text_payloads=["print('new agent')\n"],
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Erstelle tools/new_agent.py",
        workspace_root=str(tmp_path),
        workspace_snapshot=WorkspaceSnapshot(
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
        ),
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "create_file"
    assert decision.tool_args["path"] == "tools/new_agent.py"


def test_planner_runs_validation_after_changes(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="update",
                action_plan=[
                    {
                        "step": 1,
                        "action": "run_validation",
                        "reason": "Validate the changed code.",
                    }
                ],
                target_paths=["app/main.py"],
            )
        ]
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Aktualisiere app/main.py",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
        verification_commands=["python -m pytest"],
        validation_plan=[ValidationCommand(command="python -m pytest")],
    )
    session.changed_files.append(FileChangeRecord(path="app/main.py", operation="write"))

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "run_tests"
    assert decision.tool_args["command"] == "python -m pytest"


def test_planner_returns_plan_intent_as_final_response(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="plan",
                action_plan=[
                    {
                        "step": 1,
                        "action": "plan_work",
                        "reason": "Outline the migration in clear stages.",
                    }
                ],
                repo_context_needed=False,
            )
        ]
    )
    planner = Planner(llm, "")
    session = SessionState(task="Mach mir einen Plan", workspace_root=str(tmp_path))

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.FINAL
    assert "Vorgehen" in (decision.final_response or "")


def test_planner_deletes_only_after_target_has_been_read(tmp_path):
    target = tmp_path / "obsolete.py"
    target.write_text("print('old')\n", encoding="utf-8")

    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="delete",
                action_plan=[
                    {
                        "step": 1,
                        "action": "delete_artifact",
                        "reason": "Remove the obsolete file safely.",
                    }
                ],
                target_paths=["obsolete.py"],
                target_name="obsolete.py",
            )
        ]
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Loesche obsolete.py",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="read_file",
            tool_args={"path": "obsolete.py"},
            success=True,
            summary="Read obsolete.py.",
            output_excerpt="print('old')\n",
        )
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "delete_file"
    assert decision.tool_args["path"] == "obsolete.py"


def test_analyze_task_returns_router_output_model(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="search",
                action_plan=[
                    {
                        "step": 1,
                        "action": "search_workspace",
                        "reason": "Locate matching code.",
                    }
                ],
                target_name="auth middleware",
            )
        ]
    )
    planner = Planner(llm, "")

    route = planner.analyze_task("Wo ist die Auth-Middleware?", build_snapshot(tmp_path))

    assert route.intent == RouteIntent.SEARCH
    assert route.entities.target_name == "auth middleware"
