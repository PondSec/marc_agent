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
from agent.prompts import _artifact_scoped_focus, generate_content_prompt, proposed_update_review_prompt
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


def test_planner_routes_failed_semantic_review_into_repair_cycle(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            {
                "requirements_satisfied": False,
                "summary": "The Python CLI no longer prints the corrected status line the user requested.",
                "confidence": 0.92,
                "missing_requirements": ["Print the corrected status line after processing the command"],
                "suspicious_issues": ["The updated function writes to result_message, but the CLI still prints old_message"],
                "file_hints": ["app/main.py"],
                "repair_hints": ["Reconnect the final CLI print path to the updated result_message value."],
            }
        ]
    )
    planner = Planner(llm, "")
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "read_relevant_files", "reason": "Inspect the target."},
            {"step": 2, "action": "update_artifact", "reason": "Apply the requested change."},
            {"step": 3, "action": "run_validation", "reason": "Validate the result."},
            {"step": 4, "action": "summarize_result", "reason": "Summarize honestly."},
        ],
        target_paths=["app/main.py"],
        target_name="app/main.py",
    )
    session = SessionState(
        task="Fix the Python CLI output so it prints the corrected status line",
        workspace_root=str(tmp_path),
        validation_status="passed",
    )
    commit_task_state_and_route(planner, session, payload, verification_target="Print the corrected status line.")
    session.changed_files.append(FileChangeRecord(path="app/main.py", operation="modify"))
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m pytest",
            kind="test",
            verification_scope="runtime",
            status="passed",
        )
    )
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("print(old_message)\n", encoding="utf-8")

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "read_file"
    assert decision.tool_args == {"path": "app/main.py"}
    assert session.validation_status == "failed"
    assert session.active_repair_context is not None
    assert session.active_repair_context.verification_scope == "semantic"
    assert any(
        "task-to-code gaps reported by the semantic review" in item
        for item in session.active_repair_context.repair_requirements
    )


def test_planner_prefers_lightweight_semantic_review_for_small_validated_change_set(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            {
                "requirements_satisfied": True,
                "summary": "The requested CLI, docs, and tests changes are aligned.",
                "confidence": 0.88,
                "missing_requirements": [],
                "suspicious_issues": [],
                "file_hints": ["cli.py", "README.md", "tests/test_cli.py", "tests/__init__.py"],
                "repair_hints": [],
            }
        ]
    )
    planner = Planner(llm, "")
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "read_relevant_files", "reason": "Inspect the target."},
            {"step": 2, "action": "update_artifact", "reason": "Apply the requested change."},
            {"step": 3, "action": "run_validation", "reason": "Validate the result."},
            {"step": 4, "action": "summarize_result", "reason": "Summarize honestly."},
        ],
        target_paths=["cli.py", "README.md", "tests/test_cli.py"],
        target_name="cli.py",
    )
    session = SessionState(
        task="Fuege --state-root hinzu, aktualisiere README und erweitere den unittest.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
        validation_status="passed",
    )
    commit_task_state_and_route(planner, session, payload, verification_target="python -m unittest")
    session.changed_files.extend(
        [
            FileChangeRecord(path="cli.py", operation="modify"),
            FileChangeRecord(path="README.md", operation="modify"),
            FileChangeRecord(path="tests/test_cli.py", operation="modify"),
            FileChangeRecord(path="tests/__init__.py", operation="create"),
        ]
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest",
            kind="test",
            verification_scope="runtime",
            status="passed",
        )
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "cli.py").write_text("print('cli')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    (tmp_path / "tests" / "test_cli.py").write_text("print('tests')\n", encoding="utf-8")
    (tmp_path / "tests" / "__init__.py").write_text("", encoding="utf-8")

    planner._run_semantic_change_review(session.router_result, session)

    prompt = llm.generate_json_calls[0]["args"][0]

    assert llm.generate_json_calls[0]["kwargs"]["model"] == "qwen2.5-coder:14b"
    assert llm.generate_json_calls[0]["kwargs"]["num_ctx"] == 2048
    assert "adds unrequested new sections, paragraphs, examples, commands, tests, or guidance" in prompt
    assert "Treat explicit literal examples from the request as hard constraints" in prompt
    assert session.validation_runs[-1].verification_scope == "semantic"
    assert session.validation_runs[-1].status == "passed"


def test_planner_keeps_primary_semantic_review_for_large_change_sets(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            {
                "requirements_satisfied": True,
                "summary": "The broader change set is internally consistent.",
                "confidence": 0.81,
                "missing_requirements": [],
                "suspicious_issues": [],
                "file_hints": [],
                "repair_hints": [],
            }
        ]
    )
    planner = Planner(llm, "")
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "read_relevant_files", "reason": "Inspect the targets."},
            {"step": 2, "action": "update_artifact", "reason": "Apply the requested change."},
            {"step": 3, "action": "run_validation", "reason": "Validate the result."},
            {"step": 4, "action": "summarize_result", "reason": "Summarize honestly."},
        ],
        target_paths=[f"module_{index}.py" for index in range(7)],
        target_name="module_0.py",
    )
    session = SessionState(
        task="Implement a wider multi-file update.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
        validation_status="passed",
    )
    commit_task_state_and_route(planner, session, payload, verification_target="python -m pytest")
    session.changed_files.extend(
        FileChangeRecord(path=f"module_{index}.py", operation="modify")
        for index in range(7)
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m pytest",
            kind="test",
            verification_scope="runtime",
            status="passed",
        )
    )
    for index in range(7):
        (tmp_path / f"module_{index}.py").write_text("value = 1\n", encoding="utf-8")

    planner._run_semantic_change_review(session.router_result, session)

    assert llm.generate_json_calls[0]["kwargs"]["model"] == "qwen3-coder:30b"


def test_planner_uses_compact_ai_review_for_small_existing_file_updates(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            {
                "safe_to_write": True,
                "summary": "The proposed update stays focused and preserves the existing CLI behavior.",
                "confidence": 0.84,
                "blocking_issues": [],
                "preservation_risks": [],
                "repair_hints": [],
            }
        ]
    )
    planner = Planner(llm, "")
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "read_relevant_files", "reason": "Inspect the target."},
            {"step": 2, "action": "update_artifact", "reason": "Apply the requested change."},
            {"step": 3, "action": "run_validation", "reason": "Validate the result."},
            {"step": 4, "action": "summarize_result", "reason": "Summarize honestly."},
        ],
        target_paths=["cli.py", "README.md", "tests/test_cli.py"],
        target_name="cli.py",
    )
    session = SessionState(
        task="Fuege eine kleine CLI-Option hinzu und halte die bestehenden Optionen stabil.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    commit_task_state_and_route(planner, session, payload, verification_target="python -m unittest")
    session.changed_files.append(FileChangeRecord(path="README.md", operation="modify", diff="+ usage\n"))
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="read_file",
            tool_args={"path": "tests/test_cli.py"},
            success=True,
            summary="Read tests/test_cli.py.",
            output_excerpt="from cli import build_parser\n",
        )
    )

    review = planner._review_generated_update(
        session.router_result,
        session,
        path="cli.py",
        current_content="def build_parser():\n    return None\n",
        proposed_content="def build_parser():\n    parser = None\n    return parser\n",
    )

    prompt = llm.generate_json_calls[0]["args"][0]

    assert review.safe_to_write is True
    assert llm.generate_json_calls[0]["kwargs"]["model"] == "qwen2.5-coder:14b"
    assert llm.generate_json_calls[0]["kwargs"]["num_ctx"] == 2048
    assert llm.generate_json_calls[0]["kwargs"]["total_timeout"] == 45
    assert '"task_understanding"' not in prompt
    assert '"follow_up_context"' not in prompt


def test_planner_prefers_lightweight_retry_after_review_rejection_for_small_updates(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            {
                "safe_to_write": False,
                "summary": "The proposal broadens scope by rewriting unrelated README guidance.",
                "confidence": 0.61,
                "blocking_issues": ["The draft adds unrelated guidance instead of only fixing the example order."],
                "preservation_risks": [],
                "repair_hints": ["Only reorder the existing usage example."],
            },
            {
                "safe_to_write": True,
                "summary": "The retry stays focused on the requested README example order.",
                "confidence": 0.89,
                "blocking_issues": [],
                "preservation_risks": [],
                "repair_hints": [],
            },
        ],
        text_payloads=[
            "# Demo\n\n```bash\npython cli.py --config settings.json --verbose --state-root /tmp/state\n```\n",
            "# Demo\n\n```bash\npython cli.py --config settings.json --state-root /tmp/state --verbose\n```\n",
        ],
    )
    planner = Planner(llm, "")
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Apply the requested change."},
            {"step": 2, "action": "run_validation", "reason": "Validate the result."},
            {"step": 3, "action": "summarize_result", "reason": "Summarize honestly."},
        ],
        target_paths=["README.md", "tests/test_cli.py"],
        target_name="README.md",
    )
    session = SessionState(
        task="Korrigiere nur die Beispielreihenfolge in der README.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    commit_task_state_and_route(planner, session, payload, verification_target="python -m unittest")

    result = planner._generate_file_content(
        session.router_result,
        session,
        path="README.md",
        current_content="# Demo\n\n```bash\npython cli.py --config settings.json --verbose --state-root /tmp/state\n```\n",
    )

    retry_prompt = llm.generate_calls[1]["args"][0]

    assert result.content is not None
    assert "--state-root /tmp/state --verbose" in result.content
    assert len(llm.generate_calls) == 2
    assert llm.generate_calls[1]["kwargs"]["model"] == "qwen2.5-coder:14b"
    assert llm.generate_calls[1]["kwargs"]["num_ctx"] == 2048
    assert llm.generate_calls[1]["kwargs"]["total_timeout"] == 120
    assert "Task understanding:" not in retry_prompt
    assert "Follow-up context:" not in retry_prompt
    assert "Task focus:" in retry_prompt
    assert "Self-review feedback on the previous proposal:" in retry_prompt


def test_planner_keeps_primary_retry_first_for_large_review_repair_updates(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            {
                "safe_to_write": False,
                "summary": "The proposal rewrites unrelated sections.",
                "confidence": 0.58,
                "blocking_issues": ["The change is too broad."],
                "preservation_risks": [],
                "repair_hints": ["Keep the wider document structure intact."],
            },
            {
                "safe_to_write": True,
                "summary": "The retry is now scoped appropriately.",
                "confidence": 0.83,
                "blocking_issues": [],
                "preservation_risks": [],
                "repair_hints": [],
            },
        ],
        text_payloads=[
            "section = 'bad rewrite'\n",
            "section = 'narrower rewrite'\n",
        ],
    )
    planner = Planner(llm, "")
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Apply the requested change."},
        ],
        target_paths=["docs/guide.py"],
        target_name="docs/guide.py",
    )
    session = SessionState(
        task="Aktualisiere die groessere Guide-Datei.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    commit_task_state_and_route(planner, session, payload)
    large_content = "\n".join(f"line_{index} = {index}" for index in range(400))

    result = planner._generate_file_content(
        session.router_result,
        session,
        path="docs/guide.py",
        current_content=large_content,
    )

    assert result.content == "section = 'narrower rewrite'"
    assert llm.generate_calls[1]["kwargs"]["model"] == "qwen3-coder:30b"
    assert llm.generate_calls[1]["kwargs"]["num_ctx"] == 4096
    assert llm.generate_calls[1]["kwargs"]["total_timeout"] == 180


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


def test_planner_updates_after_all_explicit_targets_are_read_before_touching_snapshot_candidates(tmp_path):
    (tmp_path / "cli.py").write_text("def main():\n    return 0\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_cli.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    snapshot = WorkspaceSnapshot(
        root=str(tmp_path),
        file_count=3,
        language_counts={"python": 2, "markdown": 1},
        top_directories=["tests"],
        important_files=["cli.py", "README.md", "tests/test_cli.py"],
        focus_files=["cli.py", "README.md", "tests/test_cli.py"],
        file_briefs={},
        manifests=["README.md"],
        configs=[],
        test_files=["tests/test_cli.py"],
        build_files=[],
        deploy_files=[],
        entrypoints=["cli.py"],
        repo_map=["tests/"],
        project_labels=["python"],
        likely_commands=["python -m pytest"],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Small CLI project with a supporting README and test file.",
    )

    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="update",
                action_plan=[
                    {
                        "step": 1,
                        "action": "read_relevant_files",
                        "reason": "Inspect the explicit targets before editing them.",
                    },
                    {
                        "step": 2,
                        "action": "update_artifact",
                        "reason": "Apply the requested update.",
                    },
                ],
                target_paths=["cli.py", "README.md"],
                target_name="cli.py",
            )
        ],
        text_payloads=["def main():\n    return 1\n"],
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="Fuege eine neue CLI-Option hinzu und aktualisiere die README.",
        workspace_root=str(tmp_path),
        workspace_snapshot=snapshot,
    )
    commit_task_state_and_route(planner, session, payload)
    session.tool_calls.extend(
        [
            ToolCallRecord(
                iteration=1,
                tool_name="read_file",
                tool_args={"path": "cli.py"},
                success=True,
                summary="Read cli.py.",
                output_excerpt=(tmp_path / "cli.py").read_text(encoding="utf-8"),
            ),
            ToolCallRecord(
                iteration=2,
                tool_name="read_file",
                tool_args={"path": "README.md"},
                success=True,
                summary="Read README.md.",
                output_excerpt=(tmp_path / "README.md").read_text(encoding="utf-8"),
            ),
        ]
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "write_file"
    assert decision.tool_args["path"] == "cli.py"
    assert "return 1" in decision.tool_args["content"]


def test_planner_prefers_lightweight_model_for_low_risk_small_python_update(tmp_path):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    target = app_dir / "main.py"
    original = "def main():\n    return 'old'\n"
    updated = "def main():\n    return 'updated'\n"
    target.write_text(original, encoding="utf-8")

    llm = ScriptedLLM(
        json_payloads=[
            {
                "safe_to_write": True,
                "summary": "The update is focused and preserves the rest of the small Python file.",
                "confidence": 0.88,
                "blocking_issues": [],
                "preservation_risks": [],
                "repair_hints": [],
            }
        ],
        text_payloads=[updated],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen3-coder:30b",
            router_model_name="qwen2.5-coder:14b",
        ),
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Aktualisiere die Python-Funktion in app/main.py fokussiert.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {
                "step": 1,
                "action": "read_relevant_files",
                "reason": "Inspect the current implementation before editing it.",
            },
            {
                "step": 2,
                "action": "update_artifact",
                "reason": "Apply the focused Python change.",
            },
        ],
        target_paths=["app/main.py"],
        target_name="app/main.py",
    )
    commit_task_state_and_route(planner, session, payload)
    session.task_state.risk_level = "low"
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="read_file",
            tool_args={"path": "app/main.py"},
            success=True,
            summary="Read app/main.py.",
            output_excerpt=original,
        )
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "write_file"
    assert decision.tool_args["path"] == "app/main.py"
    assert decision.tool_args["content"].strip() == updated.strip()
    assert llm.generate_calls[0]["kwargs"]["model"] == "qwen2.5-coder:14b"
    assert llm.generate_calls[0]["kwargs"]["num_ctx"] == 2048
    assert llm.generate_calls[0]["kwargs"]["timeout"] >= 60


def test_planner_keeps_primary_generation_for_high_risk_python_update(tmp_path):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    target = app_dir / "main.py"
    original = "def main():\n    return 'old'\n"
    updated = "def main():\n    return 'updated'\n"
    target.write_text(original, encoding="utf-8")

    llm = ScriptedLLM(
        json_payloads=[
            {
                "safe_to_write": True,
                "summary": "The update is acceptable.",
                "confidence": 0.83,
                "blocking_issues": [],
                "preservation_risks": [],
                "repair_hints": [],
            }
        ],
        text_payloads=[updated],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen3-coder:30b",
            router_model_name="qwen2.5-coder:14b",
        ),
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Aendere app/main.py mit hoeherem Risiko.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {
                "step": 1,
                "action": "read_relevant_files",
                "reason": "Inspect the current implementation before editing it.",
            },
            {
                "step": 2,
                "action": "update_artifact",
                "reason": "Apply the requested Python change.",
            },
        ],
        target_paths=["app/main.py"],
        target_name="app/main.py",
    )
    commit_task_state_and_route(planner, session, payload)
    session.task_state.risk_level = "high"
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="read_file",
            tool_args={"path": "app/main.py"},
            success=True,
            summary="Read app/main.py.",
            output_excerpt=original,
        )
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "write_file"
    assert decision.tool_args["path"] == "app/main.py"
    assert decision.tool_args["content"].strip() == updated.strip()
    assert llm.generate_calls[0]["kwargs"]["model"] is None
    assert llm.generate_calls[0]["kwargs"]["num_ctx"] == 4096


def test_planner_review_prompt_includes_recent_changed_file_evidence_for_cross_file_update(tmp_path):
    cli_path = tmp_path / "cli.py"
    readme_path = tmp_path / "README.md"
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_path = tests_dir / "test_cli.py"
    cli_path.write_text(
        "def build_parser():\n"
        "    parser.add_argument('--config')\n"
        "    parser.add_argument('--state-root')\n"
        "    parser.add_argument('--verbose', action='store_true')\n",
        encoding="utf-8",
    )
    readme_path.write_text(
        "# Demo\n\n"
        "```bash\n"
        "python cli.py --config settings.json --verbose\n"
        "```\n",
        encoding="utf-8",
    )
    test_path.write_text("def test_cli_options():\n    assert True\n", encoding="utf-8")

    llm = ScriptedLLM(
        json_payloads=[
            {
                "safe_to_write": True,
                "summary": "The README update matches the already changed CLI behavior.",
                "confidence": 0.84,
                "blocking_issues": [],
                "preservation_risks": [],
                "repair_hints": [],
            }
        ],
        text_payloads=[
            "# Demo\n\n"
            "```bash\n"
            "python cli.py --config settings.json --state-root .marc --verbose\n"
            "```\n"
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen3-coder:30b",
            router_model_name="qwen2.5-coder:14b",
        ),
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Aktualisiere die README passend zur neuen CLI-Option.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {
                "step": 1,
                "action": "read_relevant_files",
                "reason": "Inspect the README before editing it.",
            },
            {
                "step": 2,
                "action": "update_artifact",
                "reason": "Document the already changed CLI behavior.",
            },
        ],
        target_paths=["README.md"],
        target_name="README.md",
    )
    payload["entities"]["target_paths"] = ["cli.py", "README.md", "tests/test_cli.py"]
    payload["entities"]["constraints"] = [
        "CLI functionality should remain unchanged",
        "New unit test for parser.parse_args(['--state-root', '/tmp/state']) is required",
        "Update README.md with correct example order",
    ]
    commit_task_state_and_route(planner, session, payload)
    session.task_state.constraints = [
        "CLI functionality should remain unchanged",
        "New unit test for parser.parse_args(['--state-root', '/tmp/state']) is required",
        "Update README.md with correct example order",
    ]
    session.changed_files.append(FileChangeRecord(path="cli.py", operation="write"))
    session.tool_calls.extend(
        [
            ToolCallRecord(
                iteration=1,
                tool_name="read_file",
                tool_args={"path": "cli.py"},
                success=True,
                summary="Read cli.py.",
                output_excerpt=cli_path.read_text(encoding="utf-8"),
            ),
            ToolCallRecord(
                iteration=2,
                tool_name="read_file",
                tool_args={"path": "README.md"},
                success=True,
                summary="Read README.md.",
                output_excerpt=readme_path.read_text(encoding="utf-8"),
            ),
            ToolCallRecord(
                iteration=3,
                tool_name="read_file",
                tool_args={"path": "tests/test_cli.py"},
                success=True,
                summary="Read tests/test_cli.py.",
                output_excerpt=test_path.read_text(encoding="utf-8"),
            ),
        ]
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "write_file"
    review_prompt = llm.generate_json_calls[0]["args"][0]
    assert llm.generate_json_calls[0]["kwargs"]["model"] == "qwen2.5-coder:14b"
    assert llm.generate_json_calls[0]["kwargs"]["num_ctx"] == 2048
    assert llm.generate_json_calls[0]["kwargs"]["total_timeout"] == 45
    assert "Supporting artifact evidence:" in review_prompt
    assert "Changed supporting artifacts already written:" in review_prompt
    assert "Recent supporting context (may still be pending updates):" in review_prompt
    assert "Judge whether writing this file now is safe as one step in the task" in review_prompt
    assert "adds unrequested new sections, explanatory prose, examples, commands, tests, or guidance" in review_prompt
    assert "Treat explicit literal examples from the request as hard constraints" in review_prompt
    assert "Use path_scope.current_write_requirements as the hard requirements for this file" in review_prompt
    assert "Treat path_scope.other_pending_requirements as non-blocking for this file" in review_prompt
    assert "Treat documentation command examples as illustrative usage" in review_prompt
    assert '"path_scope"' in review_prompt
    assert '"current_write_requirements"' in review_prompt
    assert '"other_pending_requirements"' in review_prompt
    assert "pending_target_artifacts" in review_prompt
    assert "cli.py" in review_prompt
    assert "--state-root" in review_prompt
    assert "Update README.md with correct example order" in review_prompt
    assert "New unit test for parser.parse_args(['--state-root', '/tmp/state']) is required" in review_prompt
    assert '"task_understanding"' not in review_prompt
    assert '"follow_up_context"' not in review_prompt


def test_planner_surfaces_explicit_constraints_in_compact_generation_prompt(tmp_path):
    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        "# Demo\n\n"
        "```bash\n"
        "python cli.py --config settings.json --verbose --state-root /path/to/state/root\n"
        "```\n",
        encoding="utf-8",
    )
    llm = ScriptedLLM(
        json_payloads=[
            {
                "safe_to_write": True,
                "summary": "The README update stays narrowly focused.",
                "confidence": 0.88,
                "blocking_issues": [],
                "preservation_risks": [],
                "repair_hints": [],
            }
        ],
        text_payloads=[
            "# Demo\n\n"
            "```bash\n"
            "python cli.py --config settings.json --state-root /path/to/state/root --verbose\n"
            "```\n"
        ],
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Korrigiere die Reihenfolge im README-Beispiel.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Apply the focused README change."},
        ],
        target_paths=["README.md"],
        target_name="README.md",
    )
    payload["entities"]["constraints"] = [
        "README example order should be updated to --config settings.json --state-root /path/to/state/root --verbose.",
        "New unit test for parser.parse_args(['--state-root', '/tmp/state']) is required.",
    ]
    payload["entities"]["target_paths"] = ["README.md", "tests/test_cli.py"]
    commit_task_state_and_route(planner, session, payload)
    session.task_state.constraints = [
        "README example order should be updated to --config settings.json --state-root /path/to/state/root --verbose.",
        "New unit test for parser.parse_args(['--state-root', '/tmp/state']) is required.",
    ]
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="read_file",
            tool_args={"path": "README.md"},
            success=True,
            summary="Read README.md.",
            output_excerpt=readme_path.read_text(encoding="utf-8"),
        )
    )

    decision = planner.decide_next_action(session.task, session)

    prompt = llm.generate_calls[0]["args"][0]

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "write_file"
    assert "Explicit constraints:" in prompt
    assert "File-scoped focus:" in prompt
    assert '"current_write_requirements"' in prompt
    assert '"other_pending_requirements"' in prompt
    assert "--config settings.json --state-root /path/to/state/root --verbose" in prompt
    assert "New unit test for parser.parse_args(['--state-root', '/tmp/state']) is required." in prompt


def test_planner_extracts_user_literal_examples_into_file_scoped_focus(tmp_path):
    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        "# Demo\n\n"
        "```bash\n"
        "python cli.py --config settings.json --verbose --state-root /path/to/state/root\n"
        "```\n",
        encoding="utf-8",
    )
    llm = ScriptedLLM(
        json_payloads=[
            {
                "safe_to_write": True,
                "summary": "The README update stays narrowly focused.",
                "confidence": 0.88,
                "blocking_issues": [],
                "preservation_risks": [],
                "repair_hints": [],
            }
        ],
        text_payloads=[
            "# Demo\n\n"
            "```bash\n"
            "python cli.py --config settings.json --state-root /path/to/state/root --verbose\n"
            "```\n"
        ],
    )
    planner = Planner(llm, "")
    session = SessionState(
        task=(
            "Bitte belasse die CLI-Funktionalitaet wie sie ist, "
            "ergaenze aber einen unittest, der parser.parse_args(['--state-root', '/tmp/state']) prueft, "
            "korrigiere die README-Beispielreihenfolge zu --config settings.json --state-root /path/to/state/root --verbose "
            "und fuehre danach wieder python -m unittest aus."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Apply the focused README change."},
        ],
        target_paths=["README.md", "tests/test_cli.py"],
        target_name="README.md",
    )
    payload["entities"]["constraints"] = [
        "CLI functionality should remain unchanged.",
        "Update README.md with correct example order.",
    ]
    commit_task_state_and_route(planner, session, payload)
    session.task_state.constraints = [
        "CLI functionality should remain unchanged.",
        "Update README.md with correct example order.",
    ]
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="read_file",
            tool_args={"path": "README.md"},
            success=True,
            summary="Read README.md.",
            output_excerpt=readme_path.read_text(encoding="utf-8"),
        )
    )

    decision = planner.decide_next_action(session.task, session)

    prompt = llm.generate_calls[0]["args"][0]

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "write_file"
    assert "File-scoped focus:" in prompt
    assert '"literal_constraints"' in prompt
    assert "--config settings.json --state-root /path/to/state/root --verbose" in prompt


def test_artifact_scoped_focus_prefers_primary_target_and_derives_behavior_requirements(tmp_path):
    current = (
        "from __future__ import annotations\n\n\n"
        "def normalize_name(value: str) -> str:\n"
        "    cleaned = value.strip().lower()\n"
        '    return "".join(char for char in cleaned if char.isalnum())\n'
    )
    (tmp_path / "text_utils.py").write_text(current, encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_text_utils.py").write_text(
        "from text_utils import normalize_name\n",
        encoding="utf-8",
    )

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task=(
            "Fix normalize_name in text_utils.py so it trims outer whitespace, lowercases text, "
            "converts internal whitespace runs to single hyphens, and preserves existing hyphens. "
            "Add or update unit tests. Validate with python -m unittest."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "read_relevant_files", "reason": "Inspect the active implementation before editing it."},
            {"step": 2, "action": "update_artifact", "reason": "Apply the focused change implied by the interpreted goal."},
        ],
        target_paths=["text_utils.py", "tests/test_text_utils.py"],
        target_name="text_utils.py",
    )
    commit_task_state_and_route(planner, session, payload, verification_target="python -m unittest")
    session.task_state.output_expectation = (
        "normalize_name function updated to trim outer whitespace, lowercase text, "
        "convert internal whitespace runs to single hyphens, and preserve existing hyphens. "
        "Unit tests added/updated."
    )
    session.task_state.target_artifacts = [
        TaskArtifact(path="text_utils.py", name="normalize_name", kind="function", role="primary_target", confidence=1.0),
        TaskArtifact(path="tests/test_text_utils.py", name="unit tests for normalize_name", kind="test", role="validation_target", confidence=1.0),
        TaskArtifact(path="text_utils.py", name="normalize_name", kind="function", role="primary_context", confidence=1.0),
        TaskArtifact(path="tests/test_text_utils.py", name="unit tests for normalize_name", kind="test", role="supporting_context", confidence=1.0),
    ]

    focus = _artifact_scoped_focus(
        session.router_result,
        session,
        "text_utils.py",
        current_content=current,
    )

    assert focus["artifact_role"] == "primary_target"
    assert any("trim outer whitespace" in item.lower() for item in focus["current_write_requirements"])
    assert any("lowercase text" in item.lower() for item in focus["current_write_requirements"])
    assert any("single hyphens" in item.lower() for item in focus["current_write_requirements"])
    assert any("unit tests" in item.lower() for item in focus["other_pending_requirements"])


def test_compact_prompts_keep_latest_request_details_for_behavior_change_updates(tmp_path):
    current = (
        "from __future__ import annotations\n\n\n"
        "def normalize_name(value: str) -> str:\n"
        "    cleaned = value.strip().lower()\n"
        '    return "".join(char for char in cleaned if char.isalnum())\n'
    )
    (tmp_path / "text_utils.py").write_text(current, encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_text_utils.py").write_text(
        "from text_utils import normalize_name\n",
        encoding="utf-8",
    )

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task=(
            "Fix normalize_name in text_utils.py so it trims outer whitespace, lowercases text, "
            "converts internal whitespace runs to single hyphens, and preserves existing hyphens. "
            "Add or update unit tests. Validate with python -m unittest."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Apply the focused change implied by the interpreted goal."},
        ],
        target_paths=["text_utils.py", "tests/test_text_utils.py"],
        target_name="text_utils.py",
    )
    commit_task_state_and_route(planner, session, payload, verification_target="python -m unittest")
    session.task_state.output_expectation = (
        "normalize_name function updated to trim outer whitespace, lowercase text, "
        "convert internal whitespace runs to single hyphens, and preserve existing hyphens. "
        "Unit tests added/updated."
    )
    session.task_state.target_artifacts = [
        TaskArtifact(path="text_utils.py", name="normalize_name", kind="function", role="primary_target", confidence=1.0),
        TaskArtifact(path="tests/test_text_utils.py", name="unit tests for normalize_name", kind="test", role="validation_target", confidence=1.0),
        TaskArtifact(path="text_utils.py", name="normalize_name", kind="function", role="primary_context", confidence=1.0),
        TaskArtifact(path="tests/test_text_utils.py", name="unit tests for normalize_name", kind="test", role="supporting_context", confidence=1.0),
    ]
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="read_file",
            tool_args={"path": "text_utils.py"},
            success=True,
            summary="Read text_utils.py.",
            output_excerpt=current,
        )
    )

    prompt = generate_content_prompt(
        session.router_result,
        session,
        path="text_utils.py",
        current_content=current,
        mode="compact",
    )
    review_prompt = proposed_update_review_prompt(
        session.router_result,
        session,
        path="text_utils.py",
        supporting_artifact_context="none",
        current_excerpt=current,
        proposed_excerpt=current,
        diff_excerpt="fake diff",
        mode="compact",
    )

    assert "Latest user request:" in prompt
    assert "converts internal whitespace runs to single hyphens" in prompt
    assert '"current_write_requirements"' in prompt
    assert "lowercase text" in prompt
    assert "Latest user request:" in review_prompt
    assert "Do not treat an explicitly requested behavior change for this file as scope broadening" in review_prompt
    assert "trim outer whitespace" in review_prompt


def test_planner_blocks_missing_explicit_literal_constraint_before_write(tmp_path):
    readme_path = tmp_path / "README.md"
    current = (
        "# Demo\n\n"
        "```bash\n"
        "python cli.py --config settings.json --verbose --state-root /path/to/state/root\n"
        "```\n"
    )
    readme_path.write_text(current, encoding="utf-8")
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task=(
            "Bitte belasse die CLI-Funktionalitaet wie sie ist, "
            "ergaenze aber einen unittest, der parser.parse_args(['--state-root', '/tmp/state']) prueft, "
            "korrigiere die README-Beispielreihenfolge zu --config settings.json --state-root /path/to/state/root --verbose "
            "und fuehre danach wieder python -m unittest aus."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Apply the focused README change."},
        ],
        target_paths=["README.md", "tests/test_cli.py"],
        target_name="README.md",
    )
    payload["entities"]["constraints"] = ["Update README.md with correct example order."]
    commit_task_state_and_route(planner, session, payload)
    session.task_state.constraints = ["Update README.md with correct example order."]
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="read_file",
            tool_args={"path": "README.md"},
            success=True,
            summary="Read README.md.",
            output_excerpt=current,
        )
    )

    review = planner._explicit_constraint_integrity_review(
        session.router_result,
        session,
        path="README.md",
        current_content=current,
        proposed_content=(
            "# Demo\n\n"
            "```bash\n"
            "python cli.py --state-root /tmp/state --config settings.json --verbose\n"
            "```\n"
        ),
    )

    assert review is not None
    assert review.safe_to_write is False
    assert "--config settings.json --state-root /path/to/state/root --verbose" in review.blocking_issues[0]


def test_planner_blocks_unrequested_markdown_heading_before_write(tmp_path):
    current = (
        "# Demo\n\n"
        "```bash\n"
        "python cli.py --config settings.json --verbose --state-root /path/to/state/root\n"
        "```\n"
    )
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Korrigiere nur das README-Beispiel und sonst nichts.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Apply the focused README change."},
        ],
        target_paths=["README.md"],
        target_name="README.md",
    )
    payload["entities"]["constraints"] = [
        "README example order should be updated to --config settings.json --state-root /path/to/state/root --verbose."
    ]
    commit_task_state_and_route(planner, session, payload)
    session.task_state.constraints = [
        "README example order should be updated to --config settings.json --state-root /path/to/state/root --verbose."
    ]

    review = planner._explicit_constraint_integrity_review(
        session.router_result,
        session,
        path="README.md",
        current_content=current,
        proposed_content=(
            "# Demo\n\n"
            "```bash\n"
            "python cli.py --config settings.json --state-root /path/to/state/root --verbose\n"
            "```\n\n"
            "## Unit Tests\n\n"
            "Run `python -m unittest`.\n"
        ),
    )

    assert review is not None
    assert review.safe_to_write is False
    assert "markdown headings" in review.blocking_issues[0].lower()


def test_planner_retries_update_after_prewrite_review_rejects_broad_rewrite(tmp_path):
    target = tmp_path / "cli.py"
    original = (
        "import argparse\n"
        "\n"
        "from bootstrap_runtime import ensure_runtime_dependencies\n"
        "\n"
        "\n"
        "def build_parser():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--config')\n"
        "    parser.add_argument('--verbose', action='store_true')\n"
        "    return parser\n"
        "\n"
        "\n"
        "def main():\n"
        "    ensure_runtime_dependencies()\n"
        "    parser = build_parser()\n"
        "    return parser.parse_args()\n"
    )
    bad_rewrite = (
        "import argparse\n"
        "\n"
        "\n"
        "def build_parser():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--state-root')\n"
        "    return parser\n"
    )
    focused_update = (
        "import argparse\n"
        "\n"
        "from bootstrap_runtime import ensure_runtime_dependencies\n"
        "\n"
        "\n"
        "def build_parser():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--config')\n"
        "    parser.add_argument('--state-root')\n"
        "    parser.add_argument('--verbose', action='store_true')\n"
        "    return parser\n"
        "\n"
        "\n"
        "def main():\n"
        "    ensure_runtime_dependencies()\n"
        "    parser = build_parser()\n"
        "    return parser.parse_args()\n"
    )
    target.write_text(original, encoding="utf-8")

    llm = ScriptedLLM(
        json_payloads=[
            {
                "safe_to_write": False,
                "summary": "The proposal removes existing config and runtime bootstrap behavior unrelated to the requested CLI flag addition.",
                "confidence": 0.92,
                "blocking_issues": [
                    "Existing --config option was removed without request evidence.",
                ],
                "preservation_risks": [
                    "Runtime bootstrap import and call disappeared from the proposed rewrite.",
                ],
                "repair_hints": [
                    "Keep existing CLI options and bootstrap behavior while adding the new flag.",
                ],
            },
            {
                "safe_to_write": True,
                "summary": "The second proposal adds the new CLI flag while preserving the existing behavior.",
                "confidence": 0.88,
                "blocking_issues": [],
                "preservation_risks": [],
                "repair_hints": [],
            },
        ],
        text_payloads=[bad_rewrite, focused_update],
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Fuege --state-root zur CLI hinzu, aber lasse bestehendes Verhalten intakt.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {
                "step": 1,
                "action": "read_relevant_files",
                "reason": "Inspect the current CLI implementation first.",
            },
            {
                "step": 2,
                "action": "update_artifact",
                "reason": "Add the requested flag without regressing existing behavior.",
            },
        ],
        target_paths=["cli.py"],
        target_name="cli.py",
    )
    commit_task_state_and_route(planner, session, payload)
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="read_file",
            tool_args={"path": "cli.py"},
            success=True,
            summary="Read cli.py.",
            output_excerpt=original,
        )
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "write_file"
    assert decision.tool_args["path"] == "cli.py"
    assert decision.tool_args["content"].strip() == focused_update.strip()
    assert len(llm.generate_calls) == 2
    assert "Self-review feedback on the previous proposal" in llm.generate_calls[1]["args"][0]
    assert "Keep existing CLI options and bootstrap behavior" in llm.generate_calls[1]["args"][0]


def test_planner_retries_markdown_update_when_generated_file_has_unclosed_fence(tmp_path):
    readme_path = tmp_path / "README.md"
    original = (
        "# Demo\n\n"
        "## Usage\n\n"
        "```bash\n"
        "python cli.py --config settings.json --verbose\n"
        "```\n"
    )
    malformed = (
        "# Demo\n\n"
        "## Usage\n\n"
        "```bash\n"
        "python cli.py --config settings.json --verbose --state-root /tmp/state\n"
    )
    repaired = (
        "# Demo\n\n"
        "## Usage\n\n"
        "```bash\n"
        "python cli.py --config settings.json --verbose --state-root /tmp/state\n"
        "```\n"
    )
    readme_path.write_text(original, encoding="utf-8")

    llm = ScriptedLLM(
        json_payloads=[
            {
                "safe_to_write": True,
                "summary": "The repaired README update is safe to write.",
                "confidence": 0.9,
                "blocking_issues": [],
                "preservation_risks": [],
                "repair_hints": [],
            }
        ],
        text_payloads=[malformed, repaired],
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Aktualisiere die README fuer die neue state-root Option.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {
                "step": 1,
                "action": "read_relevant_files",
                "reason": "Inspect the README before editing it.",
            },
            {
                "step": 2,
                "action": "update_artifact",
                "reason": "Document the new CLI option.",
            },
        ],
        target_paths=["README.md"],
        target_name="README.md",
    )
    commit_task_state_and_route(planner, session, payload)
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="read_file",
            tool_args={"path": "README.md"},
            success=True,
            summary="Read README.md.",
            output_excerpt=original,
        )
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "write_file"
    assert decision.tool_args["path"] == "README.md"
    assert decision.tool_args["content"].strip() == repaired.strip()
    assert len(llm.generate_calls) == 2
    assert len(llm.generate_json_calls) == 1
    assert "unclosed fenced code block" in llm.generate_calls[1]["args"][0]


def test_strip_code_fences_preserves_markdown_internal_fences():
    planner = Planner(ScriptedLLM(), "")
    markdown = (
        "# Demo\n\n"
        "## Usage\n\n"
        "```bash\n"
        "python cli.py --config settings.json --verbose --state-root /tmp/state\n"
        "```\n"
    )

    cleaned = planner._strip_code_fences(markdown)

    assert cleaned == markdown.strip()


def test_planner_blocks_update_when_prewrite_review_keeps_finding_regression_risk(tmp_path):
    target = tmp_path / "cli.py"
    original = (
        "import argparse\n"
        "\n"
        "from bootstrap_runtime import ensure_runtime_dependencies\n"
        "\n"
        "\n"
        "def build_parser():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--config')\n"
        "    parser.add_argument('--verbose', action='store_true')\n"
        "    return parser\n"
        "\n"
        "\n"
        "def main():\n"
        "    ensure_runtime_dependencies()\n"
        "    parser = build_parser()\n"
        "    return parser.parse_args()\n"
    )
    bad_rewrite = (
        "import argparse\n"
        "\n"
        "\n"
        "def build_parser():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--state-root')\n"
        "    return parser\n"
    )
    target.write_text(original, encoding="utf-8")

    llm = ScriptedLLM(
        json_payloads=[
            {
                "safe_to_write": False,
                "summary": "The proposal removes unrelated existing CLI behavior and bootstrap handling.",
                "confidence": 0.91,
                "blocking_issues": [
                    "Existing --config option disappeared.",
                ],
                "preservation_risks": [
                    "Bootstrap handling was removed from the file.",
                ],
                "repair_hints": [
                    "Preserve current options and startup behavior while adding the new flag.",
                ],
            },
            {
                "safe_to_write": False,
                "summary": "The retry is still too broad and still drops unrelated behavior.",
                "confidence": 0.86,
                "blocking_issues": [
                    "The retry still rewrites the file into a stripped-down variant.",
                ],
                "preservation_risks": [
                    "Unrelated existing behavior remains missing.",
                ],
                "repair_hints": [
                    "Make a smaller mutation instead of rewriting the file.",
                ],
            },
        ],
        text_payloads=[bad_rewrite, bad_rewrite],
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Fuege --state-root zur CLI hinzu, aber ohne Regressions.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {
                "step": 1,
                "action": "read_relevant_files",
                "reason": "Inspect the current CLI implementation first.",
            },
            {
                "step": 2,
                "action": "update_artifact",
                "reason": "Add the requested flag without regressing existing behavior.",
            },
        ],
        target_paths=["cli.py"],
        target_name="cli.py",
    )
    commit_task_state_and_route(planner, session, payload)
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="read_file",
            tool_args={"path": "cli.py"},
            success=True,
            summary="Read cli.py.",
            output_excerpt=original,
        )
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.FINAL
    assert "ai review" in decision.final_response.lower().replace("-", " ")
    assert session.diagnostics[-1].category == "preservation_risk"
    assert session.diagnostics[-1].file_hints == ["cli.py"]
    assert session.stop_reason == "update_review_rejected"


def test_planner_updates_remaining_explicit_target_before_validation(tmp_path):
    (tmp_path / "index.html").write_text("<!doctype html><button>Toggle</button>\n", encoding="utf-8")
    (tmp_path / "app.js").write_text("console.log('old');\n", encoding="utf-8")
    (tmp_path / "styles.css").write_text("body { color: black; }\n", encoding="utf-8")

    snapshot = WorkspaceSnapshot(
        root=str(tmp_path),
        file_count=3,
        language_counts={"html": 1, "javascript": 1, "css": 1},
        top_directories=[],
        important_files=["index.html", "app.js", "styles.css"],
        focus_files=["index.html", "app.js", "styles.css"],
        file_briefs={},
        manifests=[],
        configs=[],
        test_files=[],
        build_files=[],
        deploy_files=[],
        entrypoints=[],
        repo_map=[],
        project_labels=["web"],
        likely_commands=[],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Small multi-file web workspace.",
    )

    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="update",
                action_plan=[
                    {
                        "step": 1,
                        "action": "read_relevant_files",
                        "reason": "Inspect the current implementation before editing it.",
                    },
                    {
                        "step": 2,
                        "action": "update_artifact",
                        "reason": "Apply the requested update.",
                    },
                    {
                        "step": 3,
                        "action": "run_validation",
                        "reason": "Validate the changed files after the update is complete.",
                    },
                ],
                target_paths=["index.html", "app.js", "styles.css"],
                target_name="index.html",
            )
        ],
        text_payloads=["console.log('theme enabled');\n"],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen3-coder:30b",
            router_model_name="qwen2.5-coder:14b",
        ),
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="Ergaenze einen Theme-Umschalter mit localStorage und Statusmeldung.",
        workspace_root=str(tmp_path),
        workspace_snapshot=snapshot,
    )
    commit_task_state_and_route(planner, session, payload)
    session.tool_calls.extend(
        [
            ToolCallRecord(
                iteration=1,
                tool_name="read_file",
                tool_args={"path": "index.html"},
                success=True,
                summary="Read index.html.",
                output_excerpt=(tmp_path / "index.html").read_text(encoding="utf-8"),
            ),
            ToolCallRecord(
                iteration=2,
                tool_name="read_file",
                tool_args={"path": "app.js"},
                success=True,
                summary="Read app.js.",
                output_excerpt=(tmp_path / "app.js").read_text(encoding="utf-8"),
            ),
            ToolCallRecord(
                iteration=3,
                tool_name="read_file",
                tool_args={"path": "styles.css"},
                success=True,
                summary="Read styles.css.",
                output_excerpt=(tmp_path / "styles.css").read_text(encoding="utf-8"),
            ),
        ]
    )
    session.changed_files.append(
        FileChangeRecord(
            path="index.html",
            operation="write",
            diff="--- a/index.html\n+++ b/index.html\n@@\n-<button>Toggle</button>\n+<button id='themeToggle'>Toggle</button>\n",
        )
    )
    session.edit_generation = 1

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "write_file"
    assert decision.tool_args["path"] == "app.js"
    assert llm.generate_calls[0]["kwargs"]["model"] == "qwen2.5-coder:14b"


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


def test_planner_includes_validation_evidence_in_repair_prompt_after_failed_web_validation(tmp_path):
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
            "<html><body><nav id='menu'><button>Start</button></nav><div id='highscore'>Highscore</div><canvas id='game'></canvas></body></html>\n",
        ],
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="bitte programmiere ein Snake Spiel in HTML mit Menü und Highscore",
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command='internal:web_artifact:[{"path":"snake_game.html","expected_features":["menu","highscore"]}]',
                verification_scope="structural",
            )
        ],
        verification_commands=['internal:web_artifact:[{"path":"snake_game.html","expected_features":["menu","highscore"]}]'],
    )
    commit_task_state_and_route(planner, session, payload)
    session.changed_files.append(FileChangeRecord(path="snake_game.html", operation="create"))
    session.validation_runs.append(
        ValidationRunRecord(
            command='internal:web_artifact:[{"path":"snake_game.html","expected_features":["menu","highscore"]}]',
            kind="check",
            verification_scope="structural",
            status="failed",
            edit_generation=1,
            iteration=2,
            summary="Structural web validation failed.",
            excerpt="snake_game.html: missing expected web features (menu, highscore)",
        )
    )

    first_decision = planner.decide_next_action(session.task, session)

    assert first_decision.tool_name == "read_file"

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
    prompt = llm.generate_calls[0]["args"][0]

    assert second_decision.tool_name == "write_file"
    assert "Validation-guided repair context:" in prompt
    assert "missing expected web features" in prompt
    assert "menu" in prompt
    assert "highscore" in prompt
    assert "Do not return equivalent content or formatting-only changes." in prompt


def test_planner_escalates_after_no_effective_validation_guided_repair(tmp_path):
    target = tmp_path / "snake_game.html"
    original = "<html><body><h1>Snake</h1></body></html>\n"
    target.write_text(original, encoding="utf-8")

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
            original,
            "<html><body><nav id='menu'></nav><div id='highscore'>Highscore</div><canvas id='game'></canvas></body></html>\n",
        ],
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="bitte programmiere ein Snake Spiel in HTML mit Menü und Highscore",
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command='internal:web_artifact:[{"path":"snake_game.html","expected_features":["menu","highscore"]}]',
                verification_scope="structural",
            )
        ],
        verification_commands=['internal:web_artifact:[{"path":"snake_game.html","expected_features":["menu","highscore"]}]'],
    )
    commit_task_state_and_route(planner, session, payload)
    session.changed_files.append(FileChangeRecord(path="snake_game.html", operation="create"))
    session.validation_runs.append(
        ValidationRunRecord(
            command='internal:web_artifact:[{"path":"snake_game.html","expected_features":["menu","highscore"]}]',
            kind="check",
            verification_scope="structural",
            status="failed",
            edit_generation=1,
            iteration=2,
            summary="Structural web validation failed.",
            excerpt="snake_game.html: missing expected web features (menu, highscore)",
        )
    )
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

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "write_file"
    assert "highscore" in decision.tool_args["content"]
    assert len(llm.generate_calls) == 2
    assert "Validation-guided repair context:" in llm.generate_calls[0]["args"][0]
    assert "A previous repair attempt produced no effective change." in llm.generate_calls[1]["args"][0]
    assert session.repair_history[0].strategy == "validation_targeted"
    assert session.repair_history[0].result == "no_effective_change"
    assert session.repair_history[1].strategy == "validation_escalated"
    assert session.repair_history[1].result == "mutation_planned"


def test_planner_rejects_equivalent_repair_after_escalation(tmp_path):
    target = tmp_path / "snake_game.html"
    original = "<html><body><h1>Snake</h1></body></html>\n"
    target.write_text(original, encoding="utf-8")

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
            original,
            "<html><body><h1>Snake</h1></body></html>\n\n",
        ],
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="bitte programmiere ein Snake Spiel in HTML mit Menü und Highscore",
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command='internal:web_artifact:[{"path":"snake_game.html","expected_features":["menu","highscore"]}]',
                verification_scope="structural",
            )
        ],
        verification_commands=['internal:web_artifact:[{"path":"snake_game.html","expected_features":["menu","highscore"]}]'],
    )
    commit_task_state_and_route(planner, session, payload)
    session.changed_files.append(FileChangeRecord(path="snake_game.html", operation="create"))
    session.validation_runs.append(
        ValidationRunRecord(
            command='internal:web_artifact:[{"path":"snake_game.html","expected_features":["menu","highscore"]}]',
            kind="check",
            verification_scope="structural",
            status="failed",
            edit_generation=1,
            iteration=2,
            summary="Structural web validation failed.",
            excerpt="snake_game.html: missing expected web features (menu, highscore)",
        )
    )
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

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.FINAL
    assert session.stop_reason == "no_effective_change"
    assert len(llm.generate_calls) == 2
    assert any(item.result == "no_effective_change" for item in session.repair_history)
    assert session.blockers
    assert "did not produce a substantive repair change" in session.blockers[-1]


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


def test_planner_does_not_rerun_same_validation_after_read_only_context(tmp_path):
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

    assert planner._pick_validation_command(session) is None


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


def test_planner_creates_missing_validation_support_artifact_from_failure_evidence(tmp_path):
    cli = tmp_path / "cli.py"
    cli.write_text("def main():\n    return 'ok'\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_cli = tests_dir / "test_cli.py"
    test_cli.write_text(
        "import unittest\n\nfrom cli import main\n\n\nclass CliTests(unittest.TestCase):\n    def test_main(self):\n        self.assertEqual(main(), 'ok')\n",
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
                        "reason": "Repair the CLI implementation.",
                    },
                    {
                        "step": 2,
                        "action": "run_validation",
                        "reason": "Rerun python -m unittest.",
                    },
                ],
                target_paths=["cli.py"],
                target_name="cli.py",
            )
        ],
        text_payloads=["# Package marker for unittest discovery.\n"],
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="fix the CLI so python -m unittest works",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 3,
                "important_files": ["cli.py", "tests/test_cli.py", "README.md"],
                "focus_files": ["cli.py", "tests/test_cli.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest"],
            }
        ),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest"],
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest and fix the failing discovery path.",
    )
    session.changed_files.append(FileChangeRecord(path="cli.py", operation="write"))
    session.changed_files.append(FileChangeRecord(path="tests/test_cli.py", operation="write"))
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=1,
            iteration=2,
            summary="Validation command exited with 5.",
            excerpt="NO TESTS RAN",
        )
    )

    first_decision = planner.decide_next_action(session.task, session)

    assert first_decision.action_type == AgentActionType.CALL_TOOL
    assert first_decision.tool_name == "read_file"
    assert first_decision.tool_args["path"] == "tests/test_cli.py"
    assert session.active_repair_context is not None
    assert "tests/__init__.py" in session.active_repair_context.repair_requirements[1]

    session.tool_calls.append(
        ToolCallRecord(
            iteration=3,
            tool_name="read_file",
            tool_args={"path": "tests/test_cli.py"},
            success=True,
            summary="Read tests/test_cli.py.",
            output_excerpt=test_cli.read_text(encoding="utf-8"),
        )
    )

    second_decision = planner.decide_next_action(session.task, session)

    assert second_decision.action_type == AgentActionType.CALL_TOOL
    assert second_decision.tool_name == "create_file"
    assert second_decision.tool_args["path"] == "tests/__init__.py"
    assert second_decision.tool_args["overwrite"] is False
    assert second_decision.tool_args["content"].strip()


def test_planner_prefers_lightweight_model_for_missing_validation_support_artifact(tmp_path):
    cli = tmp_path / "cli.py"
    cli.write_text("def main():\n    return 'ok'\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_cli = tests_dir / "test_cli.py"
    test_cli.write_text(
        "import unittest\n\nfrom cli import main\n\n\nclass CliTests(unittest.TestCase):\n    def test_main(self):\n        self.assertEqual(main(), 'ok')\n",
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
                        "reason": "Repair the CLI implementation.",
                    },
                    {
                        "step": 2,
                        "action": "run_validation",
                        "reason": "Rerun python -m unittest.",
                    },
                ],
                target_paths=["cli.py"],
                target_name="cli.py",
            )
        ],
        text_payloads=["# Package marker for unittest discovery.\n"],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen3-coder:30b",
            router_model_name="qwen2.5-coder:14b",
        ),
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="fix the CLI so python -m unittest works",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 3,
                "important_files": ["cli.py", "tests/test_cli.py", "README.md"],
                "focus_files": ["cli.py", "tests/test_cli.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest"],
            }
        ),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest"],
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest and fix the failing discovery path.",
    )
    session.changed_files.append(FileChangeRecord(path="cli.py", operation="write"))
    session.changed_files.append(FileChangeRecord(path="tests/test_cli.py", operation="write"))
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=1,
            iteration=2,
            summary="Validation command exited with 5.",
            excerpt="NO TESTS RAN",
        )
    )
    session.tool_calls.append(
        ToolCallRecord(
            iteration=3,
            tool_name="read_file",
            tool_args={"path": "tests/test_cli.py"},
            success=True,
            summary="Read tests/test_cli.py.",
            output_excerpt=test_cli.read_text(encoding="utf-8"),
        )
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "create_file"
    assert decision.tool_args["path"] == "tests/__init__.py"
    assert llm.generate_calls[0]["kwargs"]["model"] == "qwen2.5-coder:14b"
    assert llm.generate_calls[0]["kwargs"]["num_ctx"] == 2048


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


def test_planner_finishes_explicit_create_targets_before_running_validation(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="create",
                action_plan=[
                    {
                        "step": 1,
                        "action": "create_artifact",
                        "reason": "Create the requested project files.",
                    },
                    {
                        "step": 2,
                        "action": "run_validation",
                        "reason": "Validate the generated project.",
                    },
                ],
                target_paths=["wordfreq.py", "README.md", "tests/test_wordfreq.py"],
                target_name="wordfreq.py",
                repo_context_needed=False,
            )
        ],
        text_payloads=["# Wordfreq\n\nUsage: python wordfreq.py sample.txt\n"],
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="Create wordfreq.py, README.md, and tests/test_wordfreq.py.",
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
        validation_plan=[
            ValidationCommand(
                command="python -m unittest discover -s tests -v",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest discover -s tests -v"],
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="python -m unittest discover -s tests -v",
    )
    session.changed_files.append(FileChangeRecord(path="wordfreq.py", operation="create"))

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "create_file"
    assert decision.tool_args["path"] == "README.md"


def test_planner_prefers_lightweight_model_for_remaining_create_targets_after_primary_start_failure(tmp_path):
    payload = route_payload(
        intent="create",
        action_plan=[
            {
                "step": 1,
                "action": "create_artifact",
                "reason": "Create the requested project files.",
            },
            {
                "step": 2,
                "action": "run_validation",
                "reason": "Validate the generated project.",
            },
        ],
        target_paths=["wordfreq.py", "README.md", "tests/test_wordfreq.py"],
        target_name="wordfreq.py",
        repo_context_needed=False,
    )
    llm = ScriptedLLM(
        generate_side_effects=[
            OllamaGenerationError(
                "timed out waiting for the model to start streaming after 110.0 seconds",
                reason="startup_timeout",
                retryable=False,
                model_name="qwen3-coder:30b",
                startup_timeout_seconds=110,
            ),
            "print('hello')\n",
            "# Wordfreq\n",
        ],
        config=AppConfig(
            workspace_root=".",
            model_name="qwen3-coder:30b",
            router_model_name="qwen2.5-coder:14b",
        ),
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Create wordfreq.py, README.md, and tests/test_wordfreq.py.",
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
        validation_plan=[
            ValidationCommand(
                command="python -m unittest discover -s tests -v",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest discover -s tests -v"],
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="python -m unittest discover -s tests -v",
    )

    first_decision = planner.decide_next_action(session.task, session)

    assert first_decision.action_type == AgentActionType.CALL_TOOL
    assert first_decision.tool_name == "create_file"
    assert first_decision.tool_args["path"] == "wordfreq.py"
    session.changed_files.append(FileChangeRecord(path="wordfreq.py", operation="create"))

    second_decision = planner.decide_next_action(session.task, session)

    assert second_decision.action_type == AgentActionType.CALL_TOOL
    assert second_decision.tool_name == "create_file"
    assert second_decision.tool_args["path"] == "README.md"
    assert llm.generate_calls[2]["kwargs"]["model"] == "qwen2.5-coder:14b"


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
    assert llm.generate_calls[0]["kwargs"]["num_ctx"] == 2048
    assert "Produce the full file content for exactly one file." in llm.generate_calls[0]["args"][0]
    assert "Validated route:" not in llm.generate_calls[0]["args"][0]


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


def test_planner_classifies_repeated_no_start_as_model_start_failure(tmp_path):
    logger = AgentLogger(tmp_path, "planner-no-start")
    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="create",
                action_plan=[
                    {
                        "step": 1,
                        "action": "create_artifact",
                        "reason": "Create the requested standalone web artifact.",
                    }
                ],
                target_paths=["snake.html"],
                target_name="snake.html",
                repo_context_needed=False,
            )
        ],
        generate_side_effects=[
            OllamaGenerationError("timed out", reason="startup_timeout", retryable=False),
            OllamaGenerationError("timed out", reason="startup_timeout", retryable=False),
        ],
        progress_events=[
            [
                {"type": "status", "stage": "request_started", "model": "qwen3-coder:30b"},
                {"type": "status", "stage": "waiting_for_first_chunk", "model": "qwen3-coder:30b"},
                {"type": "status", "stage": "startup_timeout_warning", "model": "qwen3-coder:30b"},
            ],
            [
                {"type": "status", "stage": "request_started", "model": "qwen2.5-coder:14b"},
                {"type": "status", "stage": "waiting_for_first_chunk", "model": "qwen2.5-coder:14b"},
            ],
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen3-coder:30b",
            router_model_name="qwen2.5-coder:14b",
        ),
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "", logger=logger)
    session = SessionState(
        task="bitte schreib ein Snake Spiel in HTML",
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
    )
    commit_task_state_and_route(planner, session, payload)

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.FINAL
    assert session.stop_reason == "model_start_failed"
    assert "Repeated model start failure for snake.html" in session.blockers[-1]
    logs = (tmp_path / "planner-no-start.jsonl").read_text(encoding="utf-8")
    assert "waiting_for_first_chunk" in logs
    assert "content_generation_recovery_unavailable" in logs
    assert "final_block_reason" in logs
    assert "model_start_failed" in logs


def test_planner_uses_starter_scaffold_recovery_for_low_risk_no_start_task(tmp_path):
    logger = AgentLogger(tmp_path, "planner-starter-recovery")
    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="create",
                action_plan=[
                    {
                        "step": 1,
                        "action": "create_artifact",
                        "reason": "Create the requested starter artifact.",
                    }
                ],
                target_paths=["index.html"],
                target_name="index.html",
                repo_context_needed=False,
            )
        ],
        generate_side_effects=[
            OllamaGenerationError("timed out", reason="startup_timeout", retryable=False),
            OllamaGenerationError("timed out", reason="startup_timeout", retryable=False),
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen3-coder:30b",
            router_model_name="qwen2.5-coder:14b",
        ),
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "", logger=logger)
    session = SessionState(
        task="Bitte erstelle ein HTML starter scaffold",
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
    )
    commit_task_state_and_route(planner, session, payload)

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "create_file"
    assert decision.tool_args["path"] == "index.html"
    assert "<!doctype html>" in decision.tool_args["content"]
    assert "Starter scaffold ready." in decision.tool_args["content"]
    logs = (tmp_path / "planner-starter-recovery.jsonl").read_text(encoding="utf-8")
    assert "content_generation_recovery_started" in logs
    assert "content_generation_recovery_finished" in logs
    assert "starter_scaffold" in logs


def test_planner_does_not_write_scaffold_for_non_starter_no_start_request(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="create",
                action_plan=[
                    {
                        "step": 1,
                        "action": "create_artifact",
                        "reason": "Create the requested standalone web artifact.",
                    }
                ],
                target_paths=["snake.html"],
                target_name="snake.html",
                repo_context_needed=False,
            )
        ],
        generate_side_effects=[
            OllamaGenerationError("timed out", reason="startup_timeout", retryable=False),
            OllamaGenerationError("timed out", reason="startup_timeout", retryable=False),
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
        task="Bitte schreibe ein Snake Spiel in HTML",
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
    )
    commit_task_state_and_route(planner, session, payload)

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.FINAL
    assert decision.tool_name is None
    assert session.stop_reason == "model_start_failed"
    assert session.changed_files == []
    assert "starter scaffold" not in (decision.final_response or "").lower()


def test_planner_marks_failure_after_partial_progress_separately_from_no_start(tmp_path):
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
            OllamaGenerationError(
                "provider aborted after progress",
                reason="provider_error",
                partial_text="print('gui version",
                characters=18,
                retryable=False,
            ),
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

    assert decision.action_type == AgentActionType.FINAL
    assert session.stop_reason == "generation_failed_after_progress"
    assert "failed after partial progress" in session.blockers[-1]


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
