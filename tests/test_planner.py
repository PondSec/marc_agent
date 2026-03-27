from __future__ import annotations

from pathlib import Path

import pytest

from agent.models import (
    DiagnosticRecord,
    FileChangeRecord,
    FollowUpContext,
    SessionState,
    ToolCallRecord,
    ValidationCommand,
    ValidationRunRecord,
    WorkspaceSnapshot,
)
from agent.planner import Planner
from agent.task_state import TaskState
from llm.schemas import AgentActionType, RouteIntent


class ScriptedLLM:
    def __init__(self, json_payloads=None, text_payloads=None, generate_fail_times: int = 0, generate_fail_message: str = "timed out"):
        self.json_payloads = list(json_payloads or [])
        self.text_payloads = list(text_payloads or [])
        self.generate_calls: list[dict] = []
        self.generate_json_calls: list[dict] = []
        self.generate_fail_times = generate_fail_times
        self.generate_fail_message = generate_fail_message

    def generate_json(self, *args, **kwargs):
        self.generate_json_calls.append({"args": args, "kwargs": kwargs})
        if not self.json_payloads:
            raise RuntimeError("No JSON payload configured")
        return self.json_payloads.pop(0)

    def generate(self, *args, **kwargs):
        self.generate_calls.append({"args": args, "kwargs": kwargs})
        if self.generate_fail_times > 0:
            self.generate_fail_times -= 1
            raise RuntimeError(self.generate_fail_message)
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


def empty_snapshot(tmp_path: Path) -> WorkspaceSnapshot:
    return WorkspaceSnapshot(
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
    search_terms=None,
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
        "search_terms": search_terms if search_terms is not None else ([target_name] if target_name else []),
        "relevant_extensions": [Path(target_paths[0]).suffix] if target_paths else [".py"],
        "direct_response": direct_response,
    }


def commit_task_state_and_route(
    planner: Planner,
    session: SessionState,
    payload: dict[str, object],
    *,
    goal_relation: str = "new_task",
    open_problem: str | None = None,
    verification_target: str | None = None,
) -> None:
    def semantic_text(value: object, fallback: str) -> str:
        text = str(value or "").strip()
        if text in {"Handle the user request safely.", "Produce the requested result."}:
            return fallback
        return text or fallback

    entities = payload.get("entities", {})
    target_paths = list(entities.get("target_paths") or [])
    target_name = str(entities.get("target_name") or "").strip() or None
    artifacts: list[dict[str, object]] = [
        {
            "path": path,
            "name": path,
            "kind": "file",
            "role": "primary_target",
            "confidence": 0.85,
        }
        for path in target_paths
    ]
    if not artifacts and target_name:
        artifacts.append(
            {
                "path": None,
                "name": target_name,
                "kind": "artifact",
                "role": "primary_target",
                "confidence": 0.7,
            }
        )
    next_action_map = {
        "create": "create",
        "update": "modify",
        "debug": "debug",
        "search": "search",
        "inspect": "inspect",
        "explain": "explain",
        "plan": "plan",
        "delete": "modify",
        "unknown": "clarify",
    }
    intent = str(payload.get("intent") or "").strip().lower()
    clarification_questions = [str(item) for item in payload.get("clarification_questions") or []]
    session.task_state = TaskState(
        latest_user_turn=session.task,
        root_goal=semantic_text(payload.get("user_goal"), session.task),
        active_goal=semantic_text(payload.get("requested_outcome"), session.task),
        goal_relation=goal_relation,
        output_expectation=semantic_text(payload.get("requested_outcome"), session.task),
        open_problem=open_problem,
        verification_target=verification_target,
        target_artifacts=artifacts,
        evidence=[],
        relevant_context=[],
        constraints=[],
        assumptions=[],
        missing_info=clarification_questions if payload.get("needs_clarification") else [],
        ambiguity_level="high" if payload.get("needs_clarification") else "low",
        risk_level="medium",
        confidence=float(payload.get("confidence") or 0.8),
        next_action=next_action_map.get(intent, "inspect"),
        execution_outline=[
            str(item.get("reason") or "").strip()
            for item in payload.get("action_plan") or []
            if str(item.get("reason") or "").strip()
        ]
        or [str(payload.get("requested_outcome") or session.task)],
        needs_clarification=bool(payload.get("needs_clarification")),
        clarification_questions=clarification_questions,
    )
    session.router_result = planner.validate_router_output(payload)


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
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(task="Wer bist du?", workspace_root=str(tmp_path))
    commit_task_state_and_route(planner, session, payload)

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
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(task="Mach das mal anders", workspace_root=str(tmp_path))
    commit_task_state_and_route(planner, session, payload)

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
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="Aktualisiere app/main.py",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    commit_task_state_and_route(planner, session, payload)

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
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="Aktualisiere app/main.py",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    commit_task_state_and_route(planner, session, payload)
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


def test_planner_reads_follow_up_target_before_diagnosing_vague_bug_report(tmp_path):
    target = tmp_path / "app" / "main.py"
    target.parent.mkdir()
    target.write_text("print('hello')\n", encoding="utf-8")

    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="debug",
                action_plan=[
                    {
                        "step": 1,
                        "action": "read_relevant_files",
                        "reason": "Inspect the active file first.",
                    },
                    {
                        "step": 2,
                        "action": "diagnose_issue",
                        "reason": "Reproduce the problem before editing.",
                    },
                    {
                        "step": 3,
                        "action": "update_artifact",
                        "reason": "Apply the focused fix.",
                    },
                ],
                target_paths=["app/main.py"],
                target_name="app/main.py",
            )
        ]
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="da ist ein fehler im terminal",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
        follow_up_context=FollowUpContext(
            previous_task="baue die app",
            target_paths=["app/main.py"],
            changed_files=["app/main.py"],
            recent_commands=["python -m pytest"],
            last_error="AssertionError",
        ),
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        goal_relation="report_problem",
        open_problem="AssertionError",
        verification_target="Reproduce the failing terminal path before editing.",
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "read_file"
    assert decision.tool_args["path"] == "app/main.py"


def test_planner_reruns_last_failing_command_for_debug_follow_up(tmp_path):
    target = tmp_path / "app" / "main.py"
    target.parent.mkdir()
    target.write_text("print('hello')\n", encoding="utf-8")

    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="debug",
                action_plan=[
                    {
                        "step": 1,
                        "action": "diagnose_issue",
                        "reason": "Reproduce the problem before editing.",
                    },
                    {
                        "step": 2,
                        "action": "update_artifact",
                        "reason": "Apply the focused fix.",
                    },
                ],
                target_paths=["app/main.py"],
                target_name="app/main.py",
            )
        ]
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="da stimmt was nicht",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
        follow_up_context=FollowUpContext(
            previous_task="baue die app",
            target_paths=["app/main.py"],
            changed_files=["app/main.py"],
            read_files=["app/main.py"],
            recent_commands=["python -m pytest"],
            validation_runs=[
                ValidationRunRecord(
                    command="python -m pytest",
                    status="failed",
                    summary="Validation command exited with 1.",
                    excerpt="AssertionError: boom",
                )
            ],
            last_error="AssertionError: boom",
        ),
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        goal_relation="report_problem",
        open_problem="AssertionError: boom",
        verification_target="Re-run python -m pytest until the failure is resolved.",
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
    assert decision.tool_name == "run_tests"
    assert decision.tool_args["command"] == "python -m pytest"


def test_planner_includes_diagnostics_in_update_prompt_for_debug_follow_up(tmp_path):
    target = tmp_path / "app" / "main.py"
    target.parent.mkdir()
    target.write_text("print('hello')\n", encoding="utf-8")

    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="debug",
                action_plan=[
                    {
                        "step": 1,
                        "action": "diagnose_issue",
                        "reason": "Use the available diagnostics.",
                    },
                    {
                        "step": 2,
                        "action": "update_artifact",
                        "reason": "Apply the focused fix.",
                    },
                ],
                target_paths=["app/main.py"],
                target_name="app/main.py",
            )
        ],
        text_payloads=["print('fixed')\n"],
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="das ist kaputt",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        goal_relation="report_problem",
        open_problem="AssertionError: wrong greeting",
        verification_target="Reproduce and fix the failing greeting path.",
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
    session.diagnostics.append(
        DiagnosticRecord(
            source="run_tests",
            category="test_failure",
            summary="AssertionError: wrong greeting",
            tool_name="run_tests",
            command="python -m pytest",
            file_hints=["app/main.py"],
            excerpt="AssertionError: wrong greeting",
        )
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "write_file"
    assert decision.tool_args["path"] == "app/main.py"
    assert "fixed" in decision.tool_args["content"]
    assert "Diagnostic context:" in llm.generate_calls[0]["args"][0]
    assert "AssertionError: wrong greeting" in llm.generate_calls[0]["args"][0]


@pytest.mark.parametrize(
    ("prompt", "expected_path", "content"),
    [
        ("mach ein tic tac toe in python", "tic_tac_toe.py", "print('tic tac toe')\n"),
        (
            "ich moechte ein schiffe versenken spiel in python haben",
            "schiffe_versenken.py",
            "print('battleship')\n",
        ),
        ("bau mir ein snake spiel in javascript", "snake.js", "console.log('snake');\n"),
        ("erstelle eine kleine flask api mit login", "flask_api_login.py", "print('api')\n"),
    ],
)
def test_planner_starts_clear_new_create_requests_without_clarification(
    tmp_path,
    prompt: str,
    expected_path: str,
    content: str,
):
    llm = ScriptedLLM(text_payloads=[content])
    planner = Planner(llm, "")
    session = SessionState(
        task=prompt,
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
    )

    session.task_state = planner.update_task_state(prompt, session.workspace_snapshot, session=session)
    session.router_result = planner.route_task_state(
        session.task_state,
        session.workspace_snapshot,
        session=session,
    )

    decision = planner.decide_next_action(session.task, session)

    assert session.task_state.next_action == "create"
    assert session.router_result.intent == RouteIntent.CREATE
    assert session.router_result.needs_clarification is False
    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "create_file"
    assert decision.tool_args["path"] == expected_path


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
    payload = llm.json_payloads[0]
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
    commit_task_state_and_route(planner, session, payload)

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "create_file"
    assert decision.tool_args["path"] == "tools/new_agent.py"


def test_planner_skips_path_llm_in_empty_workspace_and_disables_content_retries(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="create",
                action_plan=[
                    {
                        "step": 1,
                        "action": "create_artifact",
                        "reason": "Create the requested standalone script.",
                    }
                ],
                target_name="tic tac toe",
                repo_context_needed=False,
            )
        ],
        text_payloads=["print('tic tac toe')\n"],
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="Programmiere mir ein Tic Tac Toe in Python",
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
    commit_task_state_and_route(planner, session, payload)

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "create_file"
    assert decision.tool_args["path"] == "tic_tac_toe.py"
    assert "tic tac toe" in decision.tool_args["content"]
    assert len(llm.generate_calls) == 1
    assert llm.generate_calls[0]["kwargs"]["retries"] == 0
    assert llm.generate_calls[0]["kwargs"]["timeout"] >= 75


def test_planner_uses_search_terms_for_general_default_filename(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="create",
                action_plan=[
                    {
                        "step": 1,
                        "action": "create_artifact",
                        "reason": "Create the requested standalone script.",
                    }
                ],
                target_name=None,
                search_terms=["kanban", "board", "python"],
                repo_context_needed=False,
            )
        ],
        text_payloads=["print('kanban')\n"],
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="Bau mir ein Python Kanban Board",
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
    commit_task_state_and_route(planner, session, payload)

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "create_file"
    assert decision.tool_args["path"] == "kanban_board.py"


def test_planner_uses_tic_tac_toe_template_when_generation_times_out(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="create",
                action_plan=[
                    {
                        "step": 1,
                        "action": "create_artifact",
                        "reason": "Create the requested standalone script.",
                    }
                ],
                target_name="tic tac toe",
                repo_context_needed=False,
            )
        ],
        text_payloads=[],
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="Programmiere mir ein Tic Tac Toe in Python",
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
    commit_task_state_and_route(planner, session, payload)

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "create_file"
    assert decision.tool_args["path"] == "tic_tac_toe.py"
    assert "Willkommen zu Tic Tac Toe!" in decision.tool_args["content"]
    assert "Computer waehlt Feld" in decision.tool_args["content"]


def test_planner_retries_with_compact_prompt_before_template_fallback(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="update",
                action_plan=[
                    {
                        "step": 1,
                        "action": "update_artifact",
                        "reason": "Apply the requested UI update.",
                    }
                ],
                target_paths=["app.py"],
                target_name="app.py",
            )
        ],
        text_payloads=["print('gui version')\n"],
        generate_fail_times=1,
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="Bau eine GUI dazu",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    commit_task_state_and_route(planner, session, payload)
    target = tmp_path / "app.py"
    target.write_text("print('terminal version')\n", encoding="utf-8")
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="read_file",
            tool_args={"path": "app.py"},
            success=True,
            summary="Read app.py.",
            output_excerpt=target.read_text(encoding="utf-8"),
        )
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "write_file"
    assert decision.tool_args["path"] == "app.py"
    assert decision.tool_args["content"] == "print('gui version')"
    assert len(llm.generate_calls) == 2
    assert llm.generate_calls[1]["kwargs"]["retries"] == 0
    assert llm.generate_calls[1]["kwargs"]["timeout"] >= 75
    assert llm.generate_calls[1]["kwargs"]["total_timeout"] >= 180
    assert llm.generate_calls[1]["kwargs"]["num_ctx"] <= 4096


def test_planner_updates_tic_tac_toe_to_play_against_computer_on_timeout(tmp_path):
    target = tmp_path / "tic_tac_toe.py"
    target.write_text(
        "def play_game():\n    print('two players')\n",
        encoding="utf-8",
    )

    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="update",
                action_plan=[
                    {
                        "step": 1,
                        "action": "update_artifact",
                        "reason": "Switch the game to a computer opponent.",
                    }
                ],
                target_paths=["tic_tac_toe.py"],
                target_name="tic tac toe",
            )
        ],
        text_payloads=[],
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="Ich moechte gegen einen Computer spielen statt mit 2 Spielern.",
        workspace_root=str(tmp_path),
        workspace_snapshot=WorkspaceSnapshot(
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
            repo_map=[],
            project_labels=["python"],
            likely_commands=[],
            validation_commands=[],
            workflow_commands=[],
            repo_summary="Small Python workspace.",
        ),
    )
    commit_task_state_and_route(planner, session, payload)
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="read_file",
            tool_args={"path": "tic_tac_toe.py"},
            success=True,
            summary="Read tic_tac_toe.py.",
            output_excerpt=target.read_text(encoding="utf-8"),
        )
    )
    session.candidate_files = ["tic_tac_toe.py"]

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "write_file"
    assert decision.tool_args["path"] == "tic_tac_toe.py"
    assert "Du spielst gegen den Computer." in decision.tool_args["content"]
    assert "Computer waehlt Feld" in decision.tool_args["content"]


def test_planner_stops_recreating_file_after_successful_mutation(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="create",
                action_plan=[
                    {
                        "step": 1,
                        "action": "create_artifact",
                        "reason": "Create the requested standalone script.",
                    },
                    {
                        "step": 2,
                        "action": "summarize_result",
                        "reason": "Return the creation outcome.",
                    },
                ],
                target_paths=["tic_tac_toe.py"],
                target_name="tic_tac_toe.py",
                repo_context_needed=False,
            )
        ],
        text_payloads=["Fertig."],
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="Programmiere mir ein Tic Tac Toe in Python",
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
    commit_task_state_and_route(planner, session, payload)
    session.changed_files.append(FileChangeRecord(path="tic_tac_toe.py", operation="create"))

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.FINAL
    assert decision.final_response == "Fertig."


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
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="Aktualisiere app/main.py",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
        verification_commands=["python -m pytest"],
        validation_plan=[ValidationCommand(command="python -m pytest")],
    )
    commit_task_state_and_route(planner, session, payload)
    session.changed_files.append(FileChangeRecord(path="app/main.py", operation="write"))

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "run_tests"
    assert decision.tool_args["command"] == "python -m pytest"


def test_planner_plan_completion_criteria_uses_task_state_verification_target(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="update",
                action_plan=[
                    {
                        "step": 1,
                        "action": "read_relevant_files",
                        "reason": "Inspect the active implementation before editing it.",
                    },
                    {
                        "step": 2,
                        "action": "update_artifact",
                        "reason": "Apply the focused change implied by the interpreted goal.",
                    },
                    {
                        "step": 3,
                        "action": "run_validation",
                        "reason": "Verify against the committed target: Re-run the auth validation path.",
                    },
                ],
                target_paths=["app/main.py"],
                target_name="app/main.py",
            )
        ]
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="Harden auth handling",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Re-run the auth validation path.",
    )

    plan = planner.create_plan(session.task, session, session.router_result)

    assert any("inspected before mutation" in item.lower() for item in plan.completion_criteria)
    assert any("re-run the auth validation path" in item.lower() for item in plan.completion_criteria)


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
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(task="Mach mir einen Plan", workspace_root=str(tmp_path))
    commit_task_state_and_route(planner, session, payload)

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
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="Loesche obsolete.py",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    commit_task_state_and_route(planner, session, payload)
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
    llm = ScriptedLLM()
    planner = Planner(llm, "")
    session = SessionState(
        task="Wo ist die Auth-Middleware?",
        workspace_root=str(tmp_path),
        task_state=TaskState(
            latest_user_turn="Wo ist die Auth-Middleware?",
            root_goal="Locate the auth middleware.",
            active_goal="Locate the auth middleware in the current workspace.",
            goal_relation="new_task",
            output_expectation="The most relevant auth middleware location.",
            open_problem=None,
            verification_target="Return the most relevant location.",
            target_artifacts=[
                {
                    "path": None,
                    "name": "auth middleware",
                    "kind": "artifact",
                    "role": "primary_target",
                    "confidence": 0.8,
                }
            ],
            evidence=[],
            relevant_context=[],
            constraints=[],
            assumptions=[],
            missing_info=[],
            ambiguity_level="low",
            risk_level="low",
            confidence=0.87,
            next_action="search",
            execution_outline=["Search the workspace for auth middleware", "Inspect the best matches"],
            needs_clarification=False,
            clarification_questions=[],
        ),
    )

    route = planner.analyze_task(session.task, build_snapshot(tmp_path), session=session)

    assert route.intent == RouteIntent.SEARCH
    assert route.entities.target_name == "auth middleware"


def test_planner_requires_task_state_without_silent_fallback(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(task="fix it", workspace_root=str(tmp_path))

    with pytest.raises(RuntimeError, match="task_state"):
        planner.decide_next_action(session.task, session)


def test_planner_uses_deterministic_final_response_when_final_llm_roundtrip_times_out(tmp_path):
    payload = route_payload(
        intent="update",
        action_plan=[
            {
                "step": 1,
                "action": "summarize_result",
                "reason": "Return the final result to the user.",
            }
        ],
        target_paths=["snake.js"],
        target_name="snake.js",
    )
    llm = ScriptedLLM(
        json_payloads=[payload],
        text_payloads=[],
        generate_fail_times=1,
        generate_fail_message="timed out",
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Fasse die Aenderung zusammen",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    commit_task_state_and_route(planner, session, payload)
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="read_file",
            tool_args={"path": "snake.js"},
            success=True,
            summary="Read snake.js.",
            output_excerpt="console.log('snake');\n",
        )
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.FINAL
    assert "snake.js" in (decision.final_response or "")
    assert "untersucht" in (decision.final_response or "")
    assert len(llm.generate_calls) == 1
    assert llm.generate_calls[0]["kwargs"]["num_ctx"] <= 1536
    assert llm.generate_calls[0]["kwargs"]["total_timeout"] >= 60


def test_planner_localizes_deterministic_final_response_to_english(tmp_path):
    payload = route_payload(
        intent="update",
        action_plan=[
            {
                "step": 1,
                "action": "summarize_result",
                "reason": "Return the final result to the user.",
            }
        ],
        target_paths=["snake.js"],
        target_name="snake.js",
    )
    llm = ScriptedLLM(
        json_payloads=[payload],
        text_payloads=[],
        generate_fail_times=1,
        generate_fail_message="timed out",
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Summarize the change",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    commit_task_state_and_route(planner, session, payload)
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="read_file",
            tool_args={"path": "snake.js"},
            success=True,
            summary="Read snake.js.",
            output_excerpt="console.log('snake');\n",
        )
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.FINAL
    assert "I mainly inspected" in (decision.final_response or "")
    assert "Ich habe" not in (decision.final_response or "")
