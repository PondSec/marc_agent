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
from agent.task_schema import TaskArtifact
from agent.task_state import TaskState
from config.settings import AppConfig
from llm.ollama_client import OllamaGenerationError
from llm.schemas import AgentActionType, RouteIntent
from runtime.logger import AgentLogger


class ScriptedLLM:
    def __init__(
        self,
        json_payloads=None,
        text_payloads=None,
        generate_fail_times: int = 0,
        generate_fail_message: str = "timed out",
        generate_side_effects=None,
        progress_events=None,
        config: AppConfig | None = None,
    ):
        self.json_payloads = list(json_payloads or [])
        self.text_payloads = list(text_payloads or [])
        self.generate_calls: list[dict] = []
        self.generate_json_calls: list[dict] = []
        self.generate_fail_times = generate_fail_times
        self.generate_fail_message = generate_fail_message
        self.generate_side_effects = list(generate_side_effects or [])
        self.progress_events = list(progress_events or [])
        self.config = config or AppConfig(workspace_root=".")

    def generate_json(self, *args, **kwargs):
        self.generate_json_calls.append({"args": args, "kwargs": kwargs})
        if not self.json_payloads:
            raise RuntimeError("No JSON payload configured")
        return self.json_payloads.pop(0)

    def generate(self, *args, **kwargs):
        self.generate_calls.append({"args": args, "kwargs": kwargs})
        callback = kwargs.get("progress_callback")
        if self.progress_events:
            for event in self.progress_events.pop(0):
                if callback is not None:
                    callback(dict(event))
        if self.generate_side_effects:
            effect = self.generate_side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            return effect
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


def test_planner_marks_generation_failure_honestly_before_any_validation(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("print('old version')\n", encoding="utf-8")

    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="update",
                action_plan=[
                    {
                        "step": 1,
                        "action": "update_artifact",
                        "reason": "Apply the requested update.",
                    },
                    {
                        "step": 2,
                        "action": "run_validation",
                        "reason": "Validate the changed code.",
                    },
                ],
                target_paths=["app.py"],
                target_name="app.py",
            )
        ],
        generate_fail_times=5,
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="bau noch ein menü dazu",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    commit_task_state_and_route(planner, session, payload, goal_relation="refine")
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

    assert decision.action_type == AgentActionType.FINAL
    assert decision.tool_name is None
    assert session.blockers == ["No reliable update content could be generated for app.py."]
    assert session.stop_reason == "generation_failed"
    assert session.validation_status == "not_run"
    assert session.changed_files == []


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


def test_planner_prefers_python_smoke_reproduction_over_prior_syntax_only_check(tmp_path):
    target = tmp_path / "tic_tac_toe.py"
    target.write_text("choice = input('move: ')\nprint(choice)\n", encoding="utf-8")

    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="debug",
                action_plan=[
                    {
                        "step": 1,
                        "action": "diagnose_issue",
                        "reason": "Reproduce the reported issue before editing.",
                    },
                    {
                        "step": 2,
                        "action": "update_artifact",
                        "reason": "Apply the focused fix.",
                    },
                ],
                target_paths=["tic_tac_toe.py"],
                target_name="tic_tac_toe.py",
            )
        ]
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="man kann das spiel nicht vernünftig spielen, bitte fixen",
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path).model_copy(
            update={
                "file_count": 1,
                "language_counts": {"python": 1},
                "important_files": ["tic_tac_toe.py"],
                "focus_files": ["tic_tac_toe.py"],
                "project_labels": ["python"],
            }
        ),
        follow_up_context=FollowUpContext(
            previous_task="schreib bitte in python ein Tic tac toe spiel",
            target_paths=["tic_tac_toe.py"],
            changed_files=["tic_tac_toe.py"],
            read_files=["tic_tac_toe.py"],
            recent_commands=['internal:python_syntax:["tic_tac_toe.py"]'],
        ),
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        goal_relation="report_problem",
        open_problem="The terminal board output and move parsing are broken.",
        verification_target="Reproduce the failing interactive path, then fix and rerun it.",
    )
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

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "run_tests"
    assert decision.tool_args["command"].startswith("internal:python_cli_smoke:")


def test_planner_reenters_debug_repair_after_failed_runtime_validation(tmp_path):
    target = tmp_path / "tic_tac_toe.py"
    target.write_text("print('broken')\n", encoding="utf-8")

    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="debug",
                action_plan=[
                    {
                        "step": 1,
                        "action": "diagnose_issue",
                        "reason": "Use the failing runtime evidence.",
                    },
                    {
                        "step": 2,
                        "action": "update_artifact",
                        "reason": "Apply the focused fix.",
                    },
                ],
                target_paths=["tic_tac_toe.py"],
                target_name="tic_tac_toe.py",
            )
        ],
        text_payloads=["print('fixed')\n"],
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="fix the tic tac toe bug",
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path).model_copy(
            update={
                "file_count": 1,
                "language_counts": {"python": 1},
                "important_files": ["tic_tac_toe.py"],
                "focus_files": ["tic_tac_toe.py"],
                "project_labels": ["python"],
            }
        ),
        validation_status="failed",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        goal_relation="report_problem",
        open_problem="Moves are always rejected.",
        verification_target="Reproduce the bug, fix it, and rerun the interactive path.",
    )
    session.changed_files.append(FileChangeRecord(path="tic_tac_toe.py", operation="write"))
    session.validation_runs.append(
        ValidationRunRecord(
            command='internal:python_cli_smoke:["tic_tac_toe.py"]',
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=1,
            summary="Validation command exited with 1.",
            excerpt="EOFError",
        )
    )
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
    session.diagnostics.append(
        DiagnosticRecord(
            source="run_tests",
            category="command_failure",
            summary="EOFError while replaying the interactive path",
            tool_name="run_tests",
            command='internal:python_cli_smoke:["tic_tac_toe.py"]',
            file_hints=["tic_tac_toe.py"],
            excerpt="EOFError",
        )
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "write_file"
    assert decision.tool_args["path"] == "tic_tac_toe.py"
    assert "fixed" in decision.tool_args["content"]


def test_planner_failed_web_validation_triggers_read_then_repair_for_new_artifact(tmp_path):
    target = tmp_path / "snake_game.html"
    target.write_text("<html><body><h1>Snake</h1></body></html>\n", encoding="utf-8")

    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="create",
                action_plan=[
                    {
                        "step": 1,
                        "action": "create_artifact",
                        "reason": "Create the requested standalone HTML game.",
                    },
                    {
                        "step": 2,
                        "action": "run_validation",
                        "reason": "Validate the generated artifact.",
                    },
                ],
                target_paths=["snake_game.html"],
                target_name="snake_game.html",
                repo_context_needed=False,
            )
        ],
        text_payloads=[
            "<html><body><canvas id='game'></canvas><script>startGame();</script></body></html>\n",
        ],
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="bitte programmiere ein Snake Spiel in HTML",
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command='internal:web_artifact:[{"path":"snake_game.html","expected_features":[]}]',
                verification_scope="structural",
            )
        ],
        verification_commands=['internal:web_artifact:[{"path":"snake_game.html","expected_features":[]}]'],
    )
    commit_task_state_and_route(planner, session, payload)
    session.changed_files.append(FileChangeRecord(path="snake_game.html", operation="create"))
    session.validation_runs.append(
        ValidationRunRecord(
            command='internal:web_artifact:[{"path":"snake_game.html","expected_features":[]}]',
            kind="check",
            verification_scope="structural",
            status="failed",
            edit_generation=1,
            iteration=2,
            summary="Structural web validation failed.",
            excerpt="snake_game.html: missing expected web features",
        )
    )

    first_decision = planner.decide_next_action(session.task, session)

    assert first_decision.action_type == AgentActionType.CALL_TOOL
    assert first_decision.tool_name == "read_file"
    assert first_decision.tool_args["path"] == "snake_game.html"

    session.tool_calls.append(
        ToolCallRecord(
            iteration=3,
            tool_name="read_file",
            tool_args={"path": "snake_game.html"},
            success=True,
            summary="Read snake_game.html.",
            output_excerpt=target.read_text(encoding="utf-8"),
        )
    )

    second_decision = planner.decide_next_action(session.task, session)

    assert second_decision.action_type == AgentActionType.CALL_TOOL
    assert second_decision.tool_name == "write_file"
    assert second_decision.tool_args["path"] == "snake_game.html"
    assert second_decision.tool_args["content"]


def test_planner_does_not_repeat_identical_validation_without_progress(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Aktualisiere app/main.py",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
        edit_generation=1,
        validation_plan=[ValidationCommand(command="python -m pytest", kind="test")],
        verification_commands=["python -m pytest"],
    )
    session.changed_files.append(FileChangeRecord(path="app/main.py", operation="write"))
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m pytest",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=1,
            iteration=4,
            summary="Validation command exited with 1.",
        )
    )

    assert planner._pick_validation_command(session) is None


def test_planner_allows_same_validation_after_new_context(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Aktualisiere app/main.py",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
        edit_generation=1,
        validation_plan=[ValidationCommand(command="python -m pytest", kind="test")],
        verification_commands=["python -m pytest"],
    )
    session.changed_files.append(FileChangeRecord(path="app/main.py", operation="write"))
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m pytest",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=1,
            iteration=4,
            summary="Validation command exited with 1.",
        )
    )
    session.tool_calls.append(
        ToolCallRecord(
            iteration=5,
            tool_name="read_file",
            tool_args={"path": "app/main.py"},
            success=True,
            summary="Read app/main.py.",
            output_excerpt="print('broken')\n",
        )
    )

    assert planner._pick_validation_command(session) == "python -m pytest"


def test_planner_blocks_instead_of_reverifying_when_failed_validation_has_no_repair_target(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="update",
                action_plan=[
                    {
                        "step": 1,
                        "action": "run_validation",
                        "reason": "Validate the changed artifact.",
                    },
                ],
                target_paths=["broken.txt"],
                target_name="broken.txt",
                repo_context_needed=False,
            )
        ]
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="aktualisiere broken.txt",
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command="custom validate broken.txt",
                verification_scope="static",
            )
        ],
        verification_commands=["custom validate broken.txt"],
    )
    commit_task_state_and_route(planner, session, payload)
    session.changed_files.append(FileChangeRecord(path="broken.txt", operation="write"))
    session.validation_runs.append(
        ValidationRunRecord(
            command="custom validate broken.txt",
            kind="check",
            verification_scope="static",
            status="failed",
            edit_generation=1,
            iteration=2,
            summary="Validation command exited with 1.",
            excerpt="broken.txt is invalid",
        )
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.FINAL
    assert session.blockers
    assert "repairable target" in session.blockers[-1]


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


def test_planner_reuses_active_artifact_instead_of_junk_follow_up_filename(tmp_path):
    target = tmp_path / "tic_tac_toe.py"
    target.write_text("print('old version')\n", encoding="utf-8")

    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="create",
                action_plan=[
                    {
                        "step": 1,
                        "action": "create_artifact",
                        "reason": "Create the routed follow-up artifact.",
                    }
                ],
                target_name="du aber men habe",
                repo_context_needed=False,
            )
        ],
        text_payloads=["print('new version')\n"],
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="kannst du jetzt machen das ich aber ein menü habe mit 2 modis einmal gegen computer und einmal 2 spieler modus",
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
        follow_up_context=FollowUpContext(
            previous_task="ich möchte ein Tic tac Toe spiel in python haben",
            target_paths=["tic_tac_toe.py"],
            changed_files=["tic_tac_toe.py"],
            read_files=["tic_tac_toe.py"],
        ),
        candidate_files=["tic_tac_toe.py"],
    )
    session.task_state = TaskState(
        latest_user_turn=session.task,
        root_goal="Build a Tic Tac Toe game in Python.",
        active_goal="Extend the active Tic Tac Toe implementation with the latest follow-up change.",
        goal_relation="refine",
        output_expectation="Extend the active implementation in place.",
        verification_target="Update the active artifact and rerun the most relevant validation.",
        target_artifacts=[
            TaskArtifact(
                path=None,
                name="du aber men habe",
                kind="artifact",
                role="primary_target",
                confidence=0.62,
            )
        ],
        active_artifacts=[
            TaskArtifact(
                path="tic_tac_toe.py",
                name="tic_tac_toe.py",
                kind="file",
                role="active_context",
                confidence=0.9,
            )
        ],
        evidence=[],
        relevant_context=[],
        constraints=[],
        assumptions=[],
        missing_info=[],
        ambiguity_level="low",
        risk_level="medium",
        confidence=0.78,
        next_action="create",
        execution_outline=["Inspect active artifact", "Apply follow-up change", "Verify result"],
        needs_clarification=False,
        clarification_questions=[],
    )
    session.router_result = planner.validate_router_output(payload)

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "write_file"
    assert decision.tool_args["path"] == "tic_tac_toe.py"
    assert "du_aber" not in decision.tool_args["path"]


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


def test_planner_avoids_compact_retry_when_context_is_not_the_likely_issue(tmp_path):
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
        generate_side_effects=[
            OllamaGenerationError("timed out", reason="startup_timeout", retryable=False),
            "print('gui version')\n",
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen3-coder:30b",
            router_model_name="qwen2.5-coder:14b",
        ),
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
    assert llm.generate_calls[1]["kwargs"]["model"] == "qwen2.5-coder:14b"
    assert llm.generate_calls[1]["kwargs"]["timeout"] >= 75
    assert llm.generate_calls[1]["kwargs"]["total_timeout"] >= 210
    assert llm.generate_calls[1]["kwargs"]["num_ctx"] <= 4096
    assert "Produce the full file content for exactly one file." not in llm.generate_calls[1]["args"][0]
    assert "Validated route:" in llm.generate_calls[1]["args"][0]


def test_planner_uses_compact_retry_only_when_context_pressure_is_plausible(tmp_path):
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
        generate_side_effects=[
            OllamaGenerationError("context window exceeded", reason="provider_error"),
            "print('gui version')\n",
        ],
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
    target.write_text("x" * 7000, encoding="utf-8")
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
    assert len(llm.generate_calls) == 2
    assert llm.generate_calls[1]["kwargs"]["model"] is None
    assert llm.generate_calls[1]["kwargs"]["num_ctx"] == 2048
    assert "Produce the full file content for exactly one file." in llm.generate_calls[1]["args"][0]


def test_planner_uses_lightweight_model_first_for_small_web_follow_up_updates(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="update",
                action_plan=[
                    {
                        "step": 1,
                        "action": "update_artifact",
                        "reason": "Extend the existing web artifact in place.",
                    }
                ],
                target_paths=["snake.html"],
                target_name="snake.html",
            )
        ],
        text_payloads=["<html><body><div id='menu'></div><div id='highscore'></div></body></html>\n"],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen3-coder:30b",
            router_model_name="qwen2.5-coder:14b",
        ),
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="Baue ein Menü und einen Highscore dazu",
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
    )
    commit_task_state_and_route(planner, session, payload)
    target = tmp_path / "snake.html"
    target.write_text("<html><body><canvas id='game'></canvas></body></html>\n", encoding="utf-8")
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="read_file",
            tool_args={"path": "snake.html"},
            success=True,
            summary="Read snake.html.",
            output_excerpt=target.read_text(encoding="utf-8"),
        )
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "write_file"
    assert llm.generate_calls[0]["kwargs"]["model"] == "qwen2.5-coder:14b"


def test_planner_preserves_partial_progress_before_switching_models(tmp_path):
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
        generate_side_effects=[
            OllamaGenerationError(
                "timed out waiting for model completion",
                reason="total_timeout",
                partial_text="print('gui",
                characters=10,
            ),
            "print('gui version')\n",
        ],
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
    assert decision.tool_args["content"] == "print('gui version')"
    assert len(llm.generate_calls) == 2
    assert llm.generate_calls[1]["kwargs"]["model"] is None
    assert "Partial draft from the previous attempt:" in llm.generate_calls[1]["args"][0]
    assert "print('gui" in llm.generate_calls[1]["args"][0]


def test_planner_only_uses_fallback_model_after_same_model_recovery_fails(tmp_path):
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
        generate_side_effects=[
            OllamaGenerationError(
                "timed out waiting for model completion",
                reason="total_timeout",
                partial_text="print('gui",
                characters=10,
            ),
            OllamaGenerationError(
                "timed out waiting for model completion",
                reason="total_timeout",
                partial_text="print('gui vers",
                characters=15,
            ),
            "print('gui version')\n",
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen3-coder:30b",
            router_model_name="qwen2.5-coder:14b",
        ),
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
    assert decision.tool_args["content"] == "print('gui version')"
    assert len(llm.generate_calls) == 3
    assert llm.generate_calls[1]["kwargs"]["model"] is None
    assert llm.generate_calls[2]["kwargs"]["model"] == "qwen2.5-coder:14b"


def test_planner_emits_progress_events_for_long_content_generation(tmp_path):
    logger = AgentLogger(tmp_path, "planner-progress")
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
                target_name="hello.py",
                repo_context_needed=False,
            )
        ],
        text_payloads=["print('hello')\n"],
        progress_events=[
            [
                {"type": "status", "stage": "request_started", "model": "qwen3-coder:30b"},
                {"type": "heartbeat", "phase": "waiting_for_start", "elapsed": 9.0, "idle_for": 9.0, "characters": 0},
                {"type": "heartbeat", "elapsed": 12.0, "idle_for": 4.0, "characters": 0},
                {"type": "chunk", "elapsed": 13.0, "characters": 20},
            ]
        ],
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "", logger=logger)
    session = SessionState(
        task="Schreib mir hello.py",
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
    )
    commit_task_state_and_route(planner, session, payload)

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    logs = (tmp_path / "planner-progress.jsonl").read_text(encoding="utf-8")
    assert "content_generation_progress" in logs
    assert "content_generation_started" in logs
    assert "content_generation_finished" in logs
    assert "waiting_for_start" in logs
    assert "request_started" in logs


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

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "run_tests"
    assert decision.tool_args["command"].startswith("internal:python_cli_smoke:")


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
