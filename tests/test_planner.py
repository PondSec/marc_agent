from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.models import (
    DiagnosticRecord,
    FileChangeRecord,
    FollowUpContext,
    ProposedUpdateReview,
    RepairAttemptRecord,
    SessionState,
    ToolCallRecord,
    ValidationCommand,
    ValidationRunRecord,
    WorkspaceSnapshot,
)
from agent.planner import (
    DEFERRED_UPDATE_TARGET_NOTE_PREFIX,
    ContentGenerationFailure,
    ContentGenerationResult,
    GenerationRecoveryAttempt,
    Planner,
    ValidationFailureEvidence,
)
from agent.prompts import (
    _artifact_scoped_focus,
    _targeted_runtime_failure_focus_lines,
    _targeted_runtime_prompt_hints,
    generate_content_prompt,
    generate_content_retry_prompt,
    proposed_update_review_prompt,
)
from agent.task_schema import TaskArtifact
from agent.task_state import TaskState
from config.settings import AppConfig
from llm.ollama_client import OllamaGenerationError
from llm.runtime_resilience import ExecutionFailure
from llm.schemas import AgentActionType, AgentDecision, RouteIntent
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

    assert llm.generate_json_calls[0]["kwargs"]["model"] == "qwen3:8b"
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

    assert llm.generate_json_calls[0]["kwargs"]["model"] == "qwen3:14b"


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
    assert llm.generate_json_calls[0]["kwargs"]["model"] == "qwen3:8b"
    assert llm.generate_json_calls[0]["kwargs"]["num_ctx"] == 2048
    assert llm.generate_json_calls[0]["kwargs"]["total_timeout"] == 45
    assert '"task_understanding"' not in prompt
    assert '"follow_up_context"' not in prompt


def test_planner_falls_back_locally_after_compact_review_start_failure(tmp_path):
    llm = ScriptedLLM()
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

    assert review.safe_to_write is True
    assert len(llm.generate_json_calls) == 1
    assert llm.generate_json_calls[0]["kwargs"]["model"] == "qwen3:8b"


def test_planner_uses_compact_primary_review_for_validation_guided_repairs(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            {
                "safe_to_write": True,
                "summary": "The repair stays focused on the failed runtime path.",
                "confidence": 0.9,
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
            {"step": 1, "action": "update_artifact", "reason": "Repair the failing runtime behavior."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted test command."},
        ],
        target_paths=["greet_cli/__main__.py"],
        target_name="__main__.py",
    )
    session = SessionState(
        task="Fix the failing CLI runtime behavior.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 4,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "focus_files": ["tests/test_cli.py", "greet_cli/__main__.py"],
                "test_files": ["tests/test_cli.py"],
            }
        ),
        validation_status="failed",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="python -m unittest tests.test_cli",
    )
    session.active_repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/__main__.py", "tests/test_cli.py"],
        summary="Validation command exited with 1.",
        excerpt="TypeError: main() takes 0 positional arguments but 1 was given",
        failure_summary="TypeError: main() takes 0 positional arguments but 1 was given",
        expected_features=[],
        missing_features=[],
        file_hints=["greet_cli/__main__.py", "tests/test_cli.py"],
        line_hints=[10],
        action_hints=["Prefer a targeted fix over broad refactors."],
        repair_requirements=["Change greet_cli/__main__.py so the failing runtime path succeeds."],
        evidence_signature="repair-review-compact",
    )

    review = planner._review_generated_update(
        session.router_result,
        session,
        path="greet_cli/__main__.py",
        current_content="def main():\n    pass\n",
        proposed_content="def main(argv=None):\n    return argv\n",
    )

    prompt = llm.generate_json_calls[0]["args"][0]

    assert review.safe_to_write is True
    assert llm.generate_json_calls[0]["kwargs"]["model"] == "qwen3:14b"
    assert llm.generate_json_calls[0]["kwargs"]["num_ctx"] == 2048
    assert llm.generate_json_calls[0]["kwargs"]["total_timeout"] == 45
    assert '"safe_to_write"' in prompt


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
    assert llm.generate_calls[1]["kwargs"]["model"] == "qwen3:8b"
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
    assert llm.generate_calls[1]["kwargs"]["model"] == "qwen3:14b"
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


def test_planner_prefers_lightweight_model_for_small_empty_workspace_create(tmp_path):
    llm = ScriptedLLM(
        text_payloads=[
            "import sys\n\nprint('placeholder')\n",
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen3-coder:30b",
            router_model_name="qwen2.5-coder:14b",
        ),
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Erstelle app.py und README.md fuer ein kleines CLI-Beispiel.",
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="create",
        action_plan=[
            {
                "step": 1,
                "action": "create_artifact",
                "reason": "Create the requested starter files.",
            }
        ],
        target_paths=["app.py", "README.md"],
        target_name="app.py",
        repo_context_needed=False,
    )
    commit_task_state_and_route(planner, session, payload)
    session.task_state.risk_level = "low"
    session.task_state.confidence = 0.86

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "create_file"
    assert decision.tool_args["path"] == "app.py"
    assert llm.generate_calls[0]["kwargs"]["model"] == "qwen2.5-coder:14b"
    assert llm.generate_calls[0]["kwargs"]["num_ctx"] == 2048
    assert "Produce the full file content for exactly one file." in llm.generate_calls[0]["args"][0]


def test_planner_uses_compact_primary_prompt_for_low_risk_multi_file_create_when_lightweight_is_too_narrow(tmp_path):
    llm = ScriptedLLM(
        text_payloads=[
            "import sys\n\nprint('placeholder')\n",
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen3:14b",
            router_model_name="qwen3:8b",
        ),
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Erstelle ein kleines Python-CLI-Paket mit README, Tests und Einstiegspunkt.",
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="create",
        action_plan=[
            {
                "step": 1,
                "action": "create_artifact",
                "reason": "Create the requested starter files.",
            }
        ],
        target_paths=["greet_cli/__init__.py", "greet_cli/__main__.py", "README.md", "tests/test_cli.py"],
        target_name="greet_cli/__main__.py",
        repo_context_needed=False,
    )
    commit_task_state_and_route(planner, session, payload)
    session.task_state.risk_level = "low"
    session.task_state.confidence = 0.88

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "create_file"
    assert decision.tool_args["path"] in {
        "greet_cli/__init__.py",
        "greet_cli/__main__.py",
        "README.md",
        "tests/test_cli.py",
    }
    assert llm.generate_calls[0]["kwargs"]["model"] is None
    assert llm.generate_calls[0]["kwargs"]["num_ctx"] == 2048
    assert "Validated route:" not in llm.generate_calls[0]["args"][0]


def test_planner_keeps_primary_model_for_small_create_when_semantics_are_limited(tmp_path):
    llm = ScriptedLLM(
        text_payloads=[
            "def main():\n    print('hello')\n",
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen3:14b",
            router_model_name="qwen3:8b",
        ),
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Erstelle greet_cli/cli.py, README.md, tests/test_cli.py und einen Einstiegspunkt fuer ein kleines CLI.",
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="create",
        action_plan=[
            {
                "step": 1,
                "action": "create_artifact",
                "reason": "Create the requested starter files.",
            }
        ],
        target_paths=["greet_cli/cli.py", "tests/test_cli.py"],
        target_name="greet_cli/cli.py",
        repo_context_needed=False,
    )
    commit_task_state_and_route(planner, session, payload)
    session.task_state.risk_level = "low"
    session.task_state.confidence = 0.9
    session.task_state.semantic_resolution = "minimal_inference"
    session.task_state.secondary_semantics_limited = True

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "create_file"
    assert decision.tool_args["path"] == "greet_cli/cli.py"
    assert llm.generate_calls[0]["kwargs"]["model"] is None
    assert llm.generate_calls[0]["kwargs"]["num_ctx"] == 2048


def test_planner_uses_compact_primary_prompt_for_semantics_limited_multi_file_create(tmp_path):
    llm = ScriptedLLM(
        text_payloads=[
            "from .cli import main\n\nif __name__ == '__main__':\n    main()\n",
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:14b",
            router_model_name="qwen2.5-coder:7b",
        ),
    )
    planner = Planner(llm, "")
    session = SessionState(
        task=(
            "Create greet_cli/__main__.py, greet_cli/cli.py, README.md, and tests/test_cli.py "
            "for a tiny Python CLI package."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="create",
        action_plan=[
            {
                "step": 1,
                "action": "create_artifact",
                "reason": "Create the requested starter files.",
            }
        ],
        target_paths=["greet_cli/__main__.py", "greet_cli/cli.py", "README.md", "tests/test_cli.py"],
        target_name="greet_cli/__main__.py",
        repo_context_needed=True,
    )
    commit_task_state_and_route(planner, session, payload)
    session.task_state.risk_level = "low"
    session.task_state.confidence = 0.8
    session.task_state.next_action = "create"
    session.task_state.next_best_action = "create"
    session.task_state.semantic_resolution = "minimal_inference"
    session.task_state.secondary_semantics_limited = True

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "create_file"
    assert decision.tool_args["path"] == "greet_cli/__main__.py"
    assert llm.generate_calls[0]["kwargs"]["model"] is None
    assert llm.generate_calls[0]["kwargs"]["num_ctx"] == 2048
    assert "Validated route:" not in llm.generate_calls[0]["args"][0]


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


def test_generate_prompt_preserves_quoted_output_literals_for_create_tasks(tmp_path):
    llm = ScriptedLLM(
        text_payloads=["print('Hallo, Welt!')\n"],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen3-coder:30b",
            router_model_name="qwen2.5-coder:14b",
        ),
    )
    planner = Planner(llm, "")
    session = SessionState(
        task=(
            "Erstelle app.py so, dass bei einem Namen exakt 'Hallo, <Name>!' ausgegeben wird "
            "und ohne Argument exakt 'Hallo, Welt!'."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="create",
        action_plan=[
            {"step": 1, "action": "create_artifact", "reason": "Create the requested CLI file."},
        ],
        target_paths=["app.py"],
        target_name="app.py",
        repo_context_needed=False,
    )
    commit_task_state_and_route(planner, session, payload)
    session.task_state.risk_level = "low"
    session.task_state.confidence = 0.86

    decision = planner.decide_next_action(session.task, session)
    prompt = llm.generate_calls[0]["args"][0]

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "create_file"
    assert "Hallo, <Name>!" in prompt
    assert "Hallo, Welt!" in prompt


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


def test_planner_uses_compact_primary_generation_for_small_existing_updates_without_distinct_router_model(tmp_path):
    target = tmp_path / "cli.py"
    current = "def greet(name):\n    return f\"Hello, {name}!\"\n"
    target.write_text(current, encoding="utf-8")

    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="update",
                action_plan=[
                    {
                        "step": 1,
                        "action": "update_artifact",
                        "reason": "Apply the focused CLI behavior update.",
                    }
                ],
                target_paths=["cli.py", "tests/test_cli.py"],
                target_name="cli.py",
            )
        ],
        text_payloads=[
            "def greet(name, uppercase=False):\n"
            "    greeting = f\"Hello, {name}!\"\n"
            "    return greeting.upper() if uppercase else greeting\n"
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:7b",
            router_model_name="qwen2.5-coder:7b",
        ),
    )
    payload = llm.json_payloads[0]
    payload["confidence"] = 0.6
    planner = Planner(llm, "")
    session = SessionState(
        task="Add an uppercase option to the greeting helper while keeping the default greeting unchanged.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="python -m unittest tests.test_cli",
    )
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="read_file",
            tool_args={"path": "cli.py"},
            success=True,
            summary="Read cli.py.",
            output_excerpt=current,
        )
    )

    decision = planner.decide_next_action(session.task, session)

    prompt = llm.generate_calls[0]["args"][0]
    kwargs = llm.generate_calls[0]["kwargs"]

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert kwargs["model"] is None
    assert kwargs["strict_timeouts"] is True
    assert kwargs["num_ctx"] == 2048
    assert "Latest user request:" in prompt
    assert '"current_write_requirements"' in prompt
    assert "Validated route:" not in prompt
    assert "Task understanding:" not in prompt


def test_planner_infers_snapshot_explicit_readme_target_when_route_omits_it(tmp_path):
    (tmp_path / "greet_cli").mkdir()
    (tmp_path / "greet_cli" / "cli.py").write_text("def greet(name):\n    return name\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Greet CLI\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_cli.py").write_text("pass\n", encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Fuege --uppercase hinzu, aktualisiere die README und erweitere den unittest.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "important_files": ["greet_cli/cli.py", "README.md", "tests/test_cli.py"],
                "focus_files": ["greet_cli/cli.py", "tests/test_cli.py"],
                "manifests": ["README.md"],
                "test_files": ["tests/test_cli.py"],
                "entrypoints": ["greet_cli/cli.py"],
                "repo_summary": "Small CLI project with README and tests.",
            }
        ),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Apply the requested feature update."},
        ],
        target_paths=["greet_cli/cli.py", "tests/test_cli.py"],
        target_name="greet_cli/cli.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="python -m unittest tests.test_cli",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="greet_cli/cli.py", operation="modify"),
            FileChangeRecord(path="tests/test_cli.py", operation="modify"),
        ]
    )

    explicit_targets = planner._explicit_target_paths(session.router_result, session)

    assert "README.md" in explicit_targets
    assert planner._has_pending_explicit_update_targets(session.router_result, session) is True
    assert planner._next_update_target(session.router_result, session) == "README.md"


def test_fallback_semantic_review_flags_pending_snapshot_explicit_target(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Fuege --uppercase hinzu und aktualisiere die README.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "important_files": ["greet_cli/cli.py", "README.md", "tests/test_cli.py"],
                "focus_files": ["greet_cli/cli.py", "tests/test_cli.py"],
                "manifests": ["README.md"],
                "test_files": ["tests/test_cli.py"],
                "entrypoints": ["greet_cli/cli.py"],
            }
        ),
        validation_status="passed",
        changed_files=[
            FileChangeRecord(path="greet_cli/cli.py", operation="modify"),
            FileChangeRecord(path="tests/test_cli.py", operation="modify"),
        ],
    )

    review = planner._fallback_semantic_change_review(session)

    assert review.requirements_satisfied is False
    assert "README.md" in review.summary
    assert review.file_hints == ["README.md"]


def test_snapshot_explicit_target_paths_ignore_validation_only_test_command_reference(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    task = (
        "Fix this existing Python repo so python -m unittest tests.test_wordfreq passes. "
        "Read the repo, change only what is needed, and preserve the intended behavior."
    )
    session = SessionState(
        task=task,
        workspace_root=str(tmp_path),
        workspace_snapshot=WorkspaceSnapshot(
            root=str(tmp_path),
            file_count=4,
            language_counts={"python": 2, "text": 1},
            top_directories=["tests"],
            important_files=["wordfreq.py", "tests/test_wordfreq.py", "tests/test_data.txt", "README.md"],
            focus_files=["wordfreq.py", "tests/test_wordfreq.py"],
            file_briefs={},
            manifests=["README.md"],
            configs=[],
            test_files=["tests/test_wordfreq.py"],
            build_files=[],
            deploy_files=[],
            entrypoints=["wordfreq.py"],
            repo_map=["wordfreq.py", "tests/"],
            project_labels=["python"],
            likely_commands=["python -m unittest tests.test_wordfreq"],
            validation_commands=[],
            workflow_commands=[],
            repo_summary="Small Python utility with unittest coverage for word counting.",
        ),
        task_state=TaskState(
            latest_user_turn=task,
            root_goal="Fix the existing Python repo.",
            active_goal="Repair the failing runtime behavior with the smallest safe change.",
            goal_relation="new_task",
            output_expectation="The requested unittest command passes without changing unrelated behavior.",
            current_user_intent="repair",
            execution_strategy="debug_repair",
            verification_target="python -m unittest tests.test_wordfreq",
            target_artifacts=[
                TaskArtifact(path="wordfreq.py", name="wordfreq.py", kind="file", role="primary_target", confidence=0.95),
                TaskArtifact(
                    path="tests/test_wordfreq.py",
                    name="tests/test_wordfreq.py",
                    kind="test",
                    role="validation_target",
                    confidence=0.95,
                ),
            ],
            evidence=[],
            relevant_context=[],
            constraints=[],
            assumptions=[],
            missing_info=[],
            ambiguity_level="low",
            risk_level="medium",
            confidence=0.84,
            next_action="debug",
            execution_outline=["Run the failing unittest command, repair the relevant artifact, and rerun it."],
            needs_clarification=False,
            clarification_questions=[],
        ),
    )

    explicit_targets = planner._snapshot_explicit_target_paths(session)

    assert "tests/test_wordfreq.py" not in explicit_targets


def test_snapshot_explicit_target_paths_keep_path_named_outside_validation_command(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    task = (
        "Fix this repo, add concrete assertions in tests/test_wordfreq.py, and then run "
        "python -m unittest tests.test_wordfreq until it passes."
    )
    session = SessionState(
        task=task,
        workspace_root=str(tmp_path),
        workspace_snapshot=WorkspaceSnapshot(
            root=str(tmp_path),
            file_count=4,
            language_counts={"python": 2, "text": 1},
            top_directories=["tests"],
            important_files=["wordfreq.py", "tests/test_wordfreq.py", "README.md"],
            focus_files=["wordfreq.py", "tests/test_wordfreq.py"],
            file_briefs={},
            manifests=["README.md"],
            configs=[],
            test_files=["tests/test_wordfreq.py"],
            build_files=[],
            deploy_files=[],
            entrypoints=["wordfreq.py"],
            repo_map=["wordfreq.py", "tests/"],
            project_labels=["python"],
            likely_commands=["python -m unittest tests.test_wordfreq"],
            validation_commands=[],
            workflow_commands=[],
            repo_summary="Small Python utility with unittest coverage for word counting.",
        ),
        task_state=TaskState(
            latest_user_turn=task,
            root_goal="Repair the repo and update the requested test file.",
            active_goal="Change the explicitly named test artifact and confirm the command passes.",
            goal_relation="new_task",
            output_expectation="The requested test file contains concrete assertions and the command passes.",
            current_user_intent="repair",
            execution_strategy="debug_repair",
            verification_target="python -m unittest tests.test_wordfreq",
            target_artifacts=[
                TaskArtifact(path="wordfreq.py", name="wordfreq.py", kind="file", role="primary_target", confidence=0.9),
                TaskArtifact(
                    path="tests/test_wordfreq.py",
                    name="tests/test_wordfreq.py",
                    kind="test",
                    role="validation_target",
                    confidence=0.95,
                ),
            ],
            evidence=[],
            relevant_context=[],
            constraints=[],
            assumptions=[],
            missing_info=[],
            ambiguity_level="low",
            risk_level="medium",
            confidence=0.84,
            next_action="debug",
            execution_outline=["Update the explicitly named test file and rerun the requested unittest command."],
            needs_clarification=False,
            clarification_questions=[],
        ),
    )

    explicit_targets = planner._snapshot_explicit_target_paths(session)

    assert "tests/test_wordfreq.py" in explicit_targets


def test_fallback_semantic_review_ignores_validation_only_test_command_reference_after_runtime_pass(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    task = (
        "Fix this existing Python repo so python -m unittest tests.test_wordfreq passes. "
        "Read the repo, change only what is needed, and preserve the intended behavior."
    )
    session = SessionState(
        task=task,
        workspace_root=str(tmp_path),
        workspace_snapshot=WorkspaceSnapshot(
            root=str(tmp_path),
            file_count=4,
            language_counts={"python": 2, "text": 1},
            top_directories=["tests"],
            important_files=["wordfreq.py", "tests/test_wordfreq.py", "tests/test_data.txt", "README.md"],
            focus_files=["wordfreq.py", "tests/test_wordfreq.py"],
            file_briefs={},
            manifests=["README.md"],
            configs=[],
            test_files=["tests/test_wordfreq.py"],
            build_files=[],
            deploy_files=[],
            entrypoints=["wordfreq.py"],
            repo_map=["wordfreq.py", "tests/"],
            project_labels=["python"],
            likely_commands=["python -m unittest tests.test_wordfreq"],
            validation_commands=[],
            workflow_commands=[],
            repo_summary="Small Python utility with unittest coverage for word counting.",
        ),
        validation_status="passed",
        changed_files=[FileChangeRecord(path="tests/test_data.txt", operation="modify")],
        validation_runs=[
            ValidationRunRecord(
                command="python -m unittest tests.test_wordfreq",
                verification_scope="runtime",
                status="passed",
            )
        ],
        task_state=TaskState(
            latest_user_turn=task,
            root_goal="Fix the existing Python repo.",
            active_goal="Repair the failing runtime behavior with the smallest safe change.",
            goal_relation="new_task",
            output_expectation="The requested unittest command passes without changing unrelated behavior.",
            current_user_intent="repair",
            execution_strategy="debug_repair",
            verification_target="python -m unittest tests.test_wordfreq",
            target_artifacts=[
                TaskArtifact(path="wordfreq.py", name="wordfreq.py", kind="file", role="primary_target", confidence=0.95),
                TaskArtifact(
                    path="tests/test_wordfreq.py",
                    name="tests/test_wordfreq.py",
                    kind="test",
                    role="validation_target",
                    confidence=0.95,
                ),
            ],
            evidence=[],
            relevant_context=[],
            constraints=[],
            assumptions=[],
            missing_info=[],
            ambiguity_level="low",
            risk_level="medium",
            confidence=0.84,
            next_action="debug",
            execution_outline=["Run the failing unittest command, repair the relevant artifact, and rerun it."],
            needs_clarification=False,
            clarification_questions=[],
        ),
    )

    review = planner._fallback_semantic_change_review(session)

    assert review.requirements_satisfied is True
    assert "tests/test_wordfreq.py" not in review.file_hints


def test_planner_blocks_moving_cli_launcher_logic_into_helper_module(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    main_path = pkg / "__main__.py"
    main_path.write_text(
        "from .cli import greet\n\n"
        "def main(argv=None):\n"
        "    print(greet('Ada'))\n",
        encoding="utf-8",
    )
    cli_path = pkg / "cli.py"
    current = "def greet(name):\n    return f'Hello, {name}!'\n"
    cli_path.write_text(current, encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task=(
            "Extend the existing greet_cli package so the CLI accepts a --uppercase flag that prints the greeting in uppercase. "
            "Update only what is needed, update the README usage example, add or update unittests, "
            "run python -m unittest tests.test_cli, and finish only when the tests pass."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "README.md", "tests/test_cli.py"],
                "focus_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "manifests": ["README.md"],
                "test_files": ["tests/test_cli.py"],
                "entrypoints": ["greet_cli/__main__.py", "greet_cli/cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
    )
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="read_file",
            tool_args={"path": "greet_cli/__main__.py"},
            success=True,
            summary="Read greet_cli/__main__.py.",
            output_excerpt=main_path.read_text(encoding="utf-8"),
        )
    )

    review = planner._helper_entrypoint_scope_review(
        session,
        path="greet_cli/cli.py",
        current_content=current,
        proposed_content=(
            "import argparse\n\n"
            "def greet(name, uppercase=False):\n"
            "    greeting = f\"Hello, {name}!\"\n"
            "    if uppercase:\n"
            "        return greeting.upper()\n"
            "    return greeting\n\n"
            "if __name__ == '__main__':\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('name')\n"
            "    parser.add_argument('--uppercase', action='store_true')\n"
            "    args = parser.parse_args()\n"
            "    print(greet(args.name, args.uppercase))\n"
        ),
    )

    assert review is not None
    assert review.safe_to_write is False
    assert "helper module" in review.summary.lower()
    assert "greet_cli/__main__.py" in review.blocking_issues[0]


def test_validation_target_focus_keeps_behavior_requirements_for_create_tasks(tmp_path):
    (tmp_path / "greet_cli").mkdir()
    (tmp_path / "greet_cli" / "__main__.py").write_text(
        "from .cli import greet\nprint(greet('Ada'))\n",
        encoding="utf-8",
    )
    (tmp_path / "greet_cli" / "cli.py").write_text(
        "def greet(name):\n    return f'Hello, {name}!'\n",
        encoding="utf-8",
    )

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task=(
            "Create greet_cli/__main__.py, greet_cli/cli.py, README.md, and tests/test_cli.py for a tiny Python CLI package. "
            "The CLI should print \"Hello, <name>!\" when run as python -m greet_cli Ada and default to "
            "\"Hello, world!\" when no name is provided. Use argparse, document usage in README.md, and add unittests "
            "that verify both outputs."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="create",
        action_plan=[
            {
                "step": 1,
                "action": "create_artifact",
                "reason": "Create the requested starter files.",
            }
        ],
        target_paths=["greet_cli/__main__.py", "greet_cli/cli.py", "README.md", "tests/test_cli.py"],
        target_name="tests/test_cli.py",
    )
    commit_task_state_and_route(planner, session, payload)
    session.task_state.risk_level = "low"
    session.task_state.confidence = 0.8
    session.task_state.target_artifacts = [
        TaskArtifact(path="greet_cli/__main__.py", name="__main__.py", kind=".py", role="primary_target", confidence=1.0),
        TaskArtifact(path="greet_cli/cli.py", name="cli.py", kind=".py", role="primary_context", confidence=1.0),
        TaskArtifact(path="tests/test_cli.py", name="test_cli.py", kind="test", role="validation_target", confidence=1.0),
    ]

    focus = _artifact_scoped_focus(
        session.router_result,
        session,
        "tests/test_cli.py",
        current_content=None,
    )

    assert any("hello, <name>!" in item.lower() for item in focus["current_write_requirements"])
    assert any("hello, world!" in item.lower() for item in focus["current_write_requirements"])


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


def test_planner_blocks_markdown_instruction_echo_before_write(tmp_path):
    current = (
        "# Greet CLI\n\n"
        "Usage:\n\n"
        "```bash\n"
        "python -m greet_cli Ada\n"
        "```\n"
    )
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task=(
            "Add a --uppercase flag to this CLI without moving the argument parsing out of greet_cli/__main__.py. "
            "Keep greet_cli/cli.py as a small helper that only returns the greeting string. "
            "Update the tests and README so the new flag is covered and documented, then run the relevant tests."
        ),
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
    commit_task_state_and_route(planner, session, payload)

    review = planner._explicit_constraint_integrity_review(
        session.router_result,
        session,
        path="README.md",
        current_content=current,
        proposed_content=(
            "# Greet CLI\n\n"
            "Usage:\n\n"
            "```bash\n"
            "python -m greet_cli Ada\n"
            "```\n\n"
            "Add a --uppercase flag to this CLI without moving the argument parsing out of greet_cli/__main__.py. "
            "Keep greet_cli/cli.py as a small helper that only returns the greeting string.\n"
        ),
    )

    assert review is not None
    assert review.safe_to_write is False
    assert "copies task instructions into the document" in review.summary.lower()
    assert "Add a --uppercase flag to this CLI" in review.blocking_issues[0]


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


def test_planner_defers_review_blocked_target_and_continues_then_validates(tmp_path, monkeypatch):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    (pkg / "cli.py").write_text("def greet(name):\n    return f\"Hello, {name}!\"\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Greet CLI\n", encoding="utf-8")

    snapshot = WorkspaceSnapshot(
        root=str(tmp_path),
        file_count=2,
        language_counts={"python": 1, "markdown": 1},
        top_directories=["greet_cli"],
        important_files=["greet_cli/cli.py", "README.md"],
        focus_files=["greet_cli/cli.py", "README.md"],
        file_briefs={},
        manifests=["README.md"],
        configs=[],
        test_files=[],
        build_files=[],
        deploy_files=[],
        entrypoints=[],
        repo_map=["greet_cli/"],
        project_labels=["python"],
        likely_commands=["python -m unittest tests.test_cli"],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Small CLI package.",
    )

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Add an uppercase flag, keep helpers scoped, and document the usage.",
        workspace_root=str(tmp_path),
        workspace_snapshot=snapshot,
        verification_commands=["python -m unittest tests.test_cli"],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {
                "step": 1,
                "action": "read_relevant_files",
                "reason": "Inspect the target files first.",
            },
            {
                "step": 2,
                "action": "update_artifact",
                "reason": "Apply the requested updates.",
            },
            {
                "step": 3,
                "action": "run_validation",
                "reason": "Verify the resulting CLI behavior.",
            },
        ],
        target_paths=["greet_cli/cli.py", "README.md"],
        target_name="greet_cli/cli.py",
    )
    commit_task_state_and_route(planner, session, payload)
    session.changed_files.append(
        FileChangeRecord(
            path="greet_cli/__main__.py",
            operation="write",
        )
    )
    session.tool_calls.extend(
        [
            ToolCallRecord(
                iteration=1,
                tool_name="read_file",
                tool_args={"path": "greet_cli/cli.py"},
                success=True,
                summary="Read greet_cli/cli.py.",
                output_excerpt=(pkg / "cli.py").read_text(encoding="utf-8"),
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

    def fake_generate(route, active_session, *, path, current_content=None, repair_context=None, repair_strategy=None):
        assert route.intent == RouteIntent.UPDATE
        assert active_session is session
        del current_content, repair_context, repair_strategy
        if path == "greet_cli/cli.py":
            return ContentGenerationResult(
                source="failed",
                failure=ContentGenerationFailure(
                    stop_reason="update_review_rejected",
                    failure_class="update_review_rejected",
                    blocker_message="Pre-write review rejected the proposed update for greet_cli/cli.py.",
                    user_message="Blocked until a narrower mutation is available.",
                ),
            )
        if path == "README.md":
            return ContentGenerationResult(content="# Greet CLI\n\nUse `--uppercase`.\n")
        raise AssertionError(path)

    monkeypatch.setattr(planner, "_generate_file_content", fake_generate)

    first_decision = planner.decide_next_action(session.task, session)

    assert first_decision.action_type == AgentActionType.CALL_TOOL
    assert first_decision.tool_name == "write_file"
    assert first_decision.tool_args["path"] == "README.md"
    assert f"{DEFERRED_UPDATE_TARGET_NOTE_PREFIX}greet_cli/cli.py" in session.notes

    session.changed_files.append(
        FileChangeRecord(
            path="README.md",
            operation="write",
        )
    )

    second_decision = planner.decide_next_action(session.task, session)

    assert second_decision.action_type == AgentActionType.CALL_TOOL
    assert second_decision.tool_name == "run_tests"
    assert second_decision.tool_args["command"] == "python -m unittest tests.test_cli"


def test_planner_defers_identical_explicit_update_target_and_continues_to_remaining_targets(tmp_path, monkeypatch):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    (pkg / "__main__.py").write_text(
        "from .cli import greet\n\n"
        "def main(argv=None):\n"
        "    print(greet('Ada'))\n",
        encoding="utf-8",
    )
    (pkg / "cli.py").write_text(
        "def greet(name: str, uppercase: bool = False) -> str:\n"
        "    greeting = f\"Hello, {name}!\"\n"
        "    return greeting.upper() if uppercase else greeting\n",
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_cli = tests_dir / "test_cli.py"
    current_test = (
        "import io\n"
        "import unittest\n"
        "from unittest.mock import patch\n\n"
        "from greet_cli import __main__\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    def run_cli(self, argv):\n"
        "        with patch('sys.stdout', new_callable=io.StringIO) as stdout:\n"
        "            __main__.main(argv)\n"
        "            return stdout.getvalue().strip()\n\n"
        "    def test_default_greeting(self):\n"
        "        self.assertEqual(self.run_cli(['Ada']), 'Hello, Ada!')\n\n"
        "    def test_uppercase_flag(self):\n"
        "        self.assertEqual(self.run_cli(['--uppercase', 'Ada']), 'HELLO, ADA!')\n"
    )
    test_cli.write_text(current_test, encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("# greet_cli\n\nRun with `python -m greet_cli Ada`.\n", encoding="utf-8")

    snapshot = WorkspaceSnapshot(
        root=str(tmp_path),
        file_count=4,
        language_counts={"python": 3, "markdown": 1},
        top_directories=["greet_cli", "tests"],
        important_files=["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py", "README.md"],
        focus_files=["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
        file_briefs={},
        manifests=["README.md"],
        configs=[],
        test_files=["tests/test_cli.py"],
        build_files=[],
        deploy_files=[],
        entrypoints=["greet_cli/__main__.py", "greet_cli/cli.py"],
        repo_map=["greet_cli/", "tests/"],
        project_labels=["python"],
        likely_commands=["python -m unittest tests.test_cli"],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Small CLI package.",
    )

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Extend the CLI with an uppercase flag and document the usage.",
        workspace_root=str(tmp_path),
        workspace_snapshot=snapshot,
        verification_commands=["python -m unittest tests.test_cli"],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "read_relevant_files", "reason": "Inspect the target files first."},
            {"step": 2, "action": "update_artifact", "reason": "Apply the requested updates."},
            {"step": 3, "action": "run_validation", "reason": "Verify the resulting CLI behavior."},
        ],
        target_paths=["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py", "README.md"],
        target_name="greet_cli/__main__.py",
    )
    commit_task_state_and_route(planner, session, payload)
    session.changed_files.extend(
        [
            FileChangeRecord(path="greet_cli/__main__.py", operation="write"),
            FileChangeRecord(path="greet_cli/cli.py", operation="write"),
        ]
    )
    session.tool_calls.extend(
        [
            ToolCallRecord(
                iteration=1,
                tool_name="read_file",
                tool_args={"path": "greet_cli/__main__.py"},
                success=True,
                summary="Read greet_cli/__main__.py.",
                output_excerpt=(pkg / "__main__.py").read_text(encoding="utf-8"),
            ),
            ToolCallRecord(
                iteration=2,
                tool_name="read_file",
                tool_args={"path": "greet_cli/cli.py"},
                success=True,
                summary="Read greet_cli/cli.py.",
                output_excerpt=(pkg / "cli.py").read_text(encoding="utf-8"),
            ),
            ToolCallRecord(
                iteration=3,
                tool_name="read_file",
                tool_args={"path": "tests/test_cli.py"},
                success=True,
                summary="Read tests/test_cli.py.",
                output_excerpt=current_test,
            ),
            ToolCallRecord(
                iteration=4,
                tool_name="read_file",
                tool_args={"path": "README.md"},
                success=True,
                summary="Read README.md.",
                output_excerpt=readme.read_text(encoding="utf-8"),
            ),
        ]
    )

    def fake_generate(route, active_session, *, path, current_content=None, repair_context=None, repair_strategy=None):
        assert route.intent == RouteIntent.UPDATE
        assert active_session is session
        del repair_context, repair_strategy
        if path == "tests/test_cli.py":
            return ContentGenerationResult(content=current_content)
        if path == "README.md":
            return ContentGenerationResult(content="# greet_cli\n\nRun with `python -m greet_cli --uppercase Ada`.\n")
        raise AssertionError(path)

    monkeypatch.setattr(planner, "_generate_file_content", fake_generate)

    first_decision = planner.decide_next_action(session.task, session)

    assert first_decision.action_type == AgentActionType.CALL_TOOL
    assert first_decision.tool_name == "write_file"
    assert first_decision.tool_args["path"] == "README.md"
    assert f"{DEFERRED_UPDATE_TARGET_NOTE_PREFIX}tests/test_cli.py" in session.notes

    session.changed_files.append(FileChangeRecord(path="README.md", operation="write"))

    second_decision = planner.decide_next_action(session.task, session)

    assert second_decision.action_type == AgentActionType.CALL_TOOL
    assert second_decision.tool_name == "run_tests"
    assert second_decision.tool_args["command"] == "python -m unittest tests.test_cli"


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


def test_planner_prefers_explicit_runtime_validation_command_over_internal_smoke_for_debug(tmp_path):
    cli_main = tmp_path / "greet_cli" / "__main__.py"
    cli_main.parent.mkdir()
    cli_main.write_text("from .cli import main\n\nif __name__ == '__main__':\n    main()\n", encoding="utf-8")
    cli_impl = tmp_path / "greet_cli" / "cli.py"
    cli_impl.write_text("def main():\n    return 0\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_cli.py").write_text("import unittest\n", encoding="utf-8")

    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="debug",
                action_plan=[
                    {
                        "step": 1,
                        "action": "diagnose_issue",
                        "reason": "Reproduce the reported problem before editing.",
                    },
                    {
                        "step": 2,
                        "action": "update_artifact",
                        "reason": "Apply the smallest focused fix.",
                    },
                ],
                target_paths=["tests/test_cli.py"],
                target_name="tests/test_cli.py",
            )
        ]
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task=(
            "Fix the failing Python CLI repo in the current workspace. Read the files, change only what is needed, "
            "run python -m unittest tests.test_cli, and finish only when the tests pass."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=WorkspaceSnapshot(
            root=str(tmp_path),
            file_count=4,
            language_counts={"python": 3},
            top_directories=["greet_cli", "tests"],
            important_files=["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
            focus_files=["tests/test_cli.py"],
            file_briefs={},
            manifests=[],
            configs=[],
            test_files=["tests/test_cli.py"],
            build_files=[],
            deploy_files=[],
            entrypoints=["greet_cli/__main__.py"],
            repo_map=[],
            project_labels=["python"],
            likely_commands=["python -m unittest"],
            validation_commands=[
                ValidationCommand(
                    command="python -m unittest",
                    kind="test",
                    verification_scope="runtime",
                    source="python-test-files",
                    priority=10,
                    reason="unittest-style Python tests detected in the repository.",
                )
            ],
            workflow_commands=[],
            repo_summary="Small Python CLI with unittest coverage.",
        ),
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        goal_relation="report_problem",
        open_problem="The CLI tests are failing.",
        verification_target="Reproduce the failing path, apply the fix, and rerun the relevant validation.",
    )
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="read_file",
            tool_args={"path": "tests/test_cli.py"},
            success=True,
            summary="Read tests/test_cli.py.",
            output_excerpt="import unittest\n",
        )
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "run_tests"
    assert decision.tool_args["command"] == "python -m unittest tests.test_cli"


def test_planner_enters_validation_repair_for_existing_repo_before_reusing_original_debug_target(tmp_path, monkeypatch):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    main = pkg / "__main__.py"
    main.write_text(
        "import argparse\nfrom .cli import greet\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser(description='Greet someone by name.')\n"
        "    parser.add_argument('name', nargs='?', default='world')\n"
        "    args = parser.parse_args()\n"
        "    print(greet(args.name))\n",
        encoding="utf-8",
    )
    cli = pkg / "cli.py"
    cli.write_text("def greet(name):\n    return f'Hello, {name}!'\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_content = (
        "import unittest\nfrom unittest.mock import patch\nfrom io import StringIO\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    @patch('sys.stdout', new_callable=StringIO)\n"
        "    def test_greet_with_name(self, mock_stdout):\n"
        "        from greet_cli import __main__\n"
        "        with patch('__main__.sys.argv', ['python', '-m', 'greet_cli', 'Ada']):\n"
        "            __main__.main()\n"
    )
    test_path = tests_dir / "test_cli.py"
    test_path.write_text(test_content, encoding="utf-8")

    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="debug",
                action_plan=[
                    {
                        "step": 1,
                        "action": "diagnose_issue",
                        "reason": "Reproduce the reported problem before editing.",
                    },
                    {
                        "step": 2,
                        "action": "update_artifact",
                        "reason": "Apply the smallest focused fix.",
                    },
                ],
                target_paths=["tests/test_cli.py"],
                target_name="tests/test_cli.py",
            )
        ]
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task=(
            "Fix the failing Python CLI repo in the current workspace. Read the files, change only what is needed, "
            "run python -m unittest tests.test_cli, and finish only when the tests pass."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=WorkspaceSnapshot(
            root=str(tmp_path),
            file_count=4,
            language_counts={"python": 3},
            top_directories=["greet_cli", "tests"],
            important_files=["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
            focus_files=["tests/test_cli.py"],
            file_briefs={},
            manifests=[],
            configs=[],
            test_files=["tests/test_cli.py"],
            build_files=[],
            deploy_files=[],
            entrypoints=["greet_cli/__main__.py"],
            repo_map=[],
            project_labels=["python"],
            likely_commands=["python -m unittest"],
            validation_commands=[
                ValidationCommand(
                    command="python -m unittest",
                    kind="test",
                    verification_scope="runtime",
                    source="python-test-files",
                    priority=10,
                    reason="unittest-style Python tests detected in the repository.",
                )
            ],
            workflow_commands=[],
            repo_summary="Small Python CLI with unittest coverage.",
        ),
        validation_status="failed",
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_cli",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_cli"],
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        goal_relation="report_problem",
        open_problem="The CLI tests are failing.",
        verification_target="Reproduce the failing path, apply the fix, and rerun the relevant validation.",
    )
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="read_file",
            tool_args={"path": "tests/test_cli.py"},
            success=True,
            summary="Read tests/test_cli.py.",
            output_excerpt=test_content,
        )
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_cli",
            kind="test",
            verification_scope="runtime",
            status="failed",
            iteration=2,
            summary="Validation command exited with 1.",
            excerpt=(
                "usage: python [-h] [name]\n"
                "python: error: unrecognized arguments: -m\n"
                "ERROR: test_greet_with_name (tests.test_cli.TestCLI.test_greet_with_name)\n"
                '  File "/tmp/tests/test_cli.py", line 9, in test_greet_with_name\n'
                "    __main__.main()\n"
                '  File "/tmp/greet_cli/__main__.py", line 7, in main\n'
                "    args = parser.parse_args()\n"
            ),
        )
    )
    session.diagnostics.append(
        DiagnosticRecord(
            source="run_tests",
            category="command_failure",
            summary="python -m unittest tests.test_cli fails because __main__.py parses the patched argv incorrectly.",
            tool_name="run_tests",
            command="python -m unittest tests.test_cli",
            file_hints=["tests/test_cli.py", "greet_cli/__main__.py"],
            line_hints=[9, 7],
            action_hints=[
                "Read the failing test output and inspect the hinted source files before editing.",
                "Prefer a targeted fix over broad refactors, then rerun the same test command.",
            ],
            excerpt=(
                '  File "/tmp/tests/test_cli.py", line 9, in test_greet_with_name\n'
                "    __main__.main()\n"
                '  File "/tmp/greet_cli/__main__.py", line 7, in main\n'
                "    args = parser.parse_args()\n"
                "SystemExit: 2\n"
            ),
        )
    )

    first_decision = planner.decide_next_action(session.task, session)

    assert first_decision.action_type == AgentActionType.CALL_TOOL
    assert first_decision.tool_name == "read_file"
    assert first_decision.tool_args["path"] == "greet_cli/__main__.py"

    session.tool_calls.append(
        ToolCallRecord(
            iteration=2,
            tool_name="read_file",
            tool_args={"path": "greet_cli/__main__.py"},
            success=True,
            summary="Read greet_cli/__main__.py.",
            output_excerpt=main.read_text(encoding="utf-8"),
        )
    )

    def fake_generate_file_content(*_args, **kwargs):
        assert kwargs["path"] == "greet_cli/__main__.py"
        return SimpleNamespace(
            content=main.read_text(encoding="utf-8").replace(
                "args = parser.parse_args()\n",
                "args = parser.parse_args(sys.argv[2:] if len(sys.argv) > 1 and sys.argv[1] == '-m' else None)\n",
            ),
            failure=None,
        )

    monkeypatch.setattr(planner, "_generate_file_content", fake_generate_file_content)

    second_decision = planner.decide_next_action(session.task, session)

    assert second_decision.action_type == AgentActionType.CALL_TOOL
    assert second_decision.tool_name == "write_file"
    assert second_decision.tool_args["path"] == "greet_cli/__main__.py"


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


def test_planner_skips_generic_unittest_after_targeted_module_passes(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    snapshot = build_snapshot(tmp_path).model_copy(
        update={
            "important_files": ["greet_cli/__main__.py", "tests/test_cli.py"],
            "focus_files": ["greet_cli/__main__.py", "tests/test_cli.py"],
            "test_files": ["tests/test_cli.py"],
            "likely_commands": ["python -m unittest"],
            "validation_commands": [
                ValidationCommand(
                    command="python -m unittest",
                    kind="test",
                    verification_scope="runtime",
                    source="python-test-files",
                )
            ],
        }
    )
    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root=str(tmp_path),
        workspace_snapshot=snapshot,
        edit_generation=1,
        validation_plan=[
            ValidationCommand(command="python -m unittest", kind="test", verification_scope="runtime")
        ],
        verification_commands=["python -m unittest tests.test_cli", "python -m unittest"],
    )
    session.changed_files.append(FileChangeRecord(path="greet_cli/__main__.py", operation="write"))
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_cli",
            kind="test",
            verification_scope="runtime",
            status="passed",
            edit_generation=1,
            iteration=5,
            summary="Validation command exited with 0.",
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


def test_planner_repairs_targeted_unittest_failure_from_the_test_module_first(tmp_path):
    cli = tmp_path / "greet_cli" / "cli.py"
    cli.parent.mkdir()
    cli.write_text("def greet(name):\n    return f'Hello, {name}!'\n", encoding="utf-8")
    main = tmp_path / "greet_cli" / "__main__.py"
    main.write_text("from .cli import greet\nprint(greet('Ada'))\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_cli = tests_dir / "test_cli.py"
    test_cli.write_text(
        "import unittest\n\nclass CliTests(unittest.TestCase):\n    pass\n",
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
                        "reason": "Rerun the targeted unittest module.",
                    },
                ],
                target_paths=["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                target_name="greet_cli/cli.py",
            )
        ],
    )
    payload = llm.json_payloads[0]
    planner = Planner(llm, "")
    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 4,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py", "README.md"],
                "focus_files": ["greet_cli/cli.py", "tests/test_cli.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest"],
            }
        ),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_cli",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_cli"],
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli and fix the failing CLI behavior.",
    )
    session.changed_files.append(FileChangeRecord(path="greet_cli/__main__.py", operation="write"))
    session.changed_files.append(FileChangeRecord(path="greet_cli/cli.py", operation="write"))
    session.changed_files.append(FileChangeRecord(path="tests/test_cli.py", operation="write"))
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_cli",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=1,
            iteration=8,
            summary="Validation command exited with 1.",
            excerpt="FAIL: expected Hello, Ada!",
        )
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "read_file"
    assert decision.tool_args["path"] == "tests/test_cli.py"


def test_planner_repair_does_not_treat_basename_target_name_as_a_second_file(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_cli = tests_dir / "test_cli.py"
    current = (
        "import unittest\n"
        "from greet_cli import __main__\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    def test_greet_with_name(self):\n"
        "        __main__.main(['Ada'])\n"
    )
    test_cli.write_text(current, encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 2,
                "important_files": ["tests/test_cli.py"],
                "focus_files": ["tests/test_cli.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_cli",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_cli"],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failing test module."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["tests/test_cli.py"],
        target_name="test_cli.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli and fix the failing CLI behavior.",
    )
    session.changed_files.append(FileChangeRecord(path="tests/test_cli.py", operation="write"))
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_cli",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=1,
            iteration=8,
            summary="Validation command exited with 1.",
            excerpt=(
                "Traceback (most recent call last):\n"
                '  File "/tmp/tests/test_cli.py", line 5, in test_greet_with_name\n'
                "    __main__.main(['Ada'])\n"
                "TypeError: main() takes 0 positional arguments but 1 was given\n"
            ),
        )
    )
    session.tool_calls.append(
        ToolCallRecord(
            iteration=9,
            tool_name="read_file",
            tool_args={"path": "tests/test_cli.py"},
            success=True,
            summary="Read tests/test_cli.py.",
            output_excerpt=current,
        )
    )

    failed_run = session.validation_runs[-1]
    repair_context = planner.validation_planner.build_failure_evidence(session, failed_run)
    repair_route = planner._repair_route_after_failed_validation(
        session.router_result,
        session,
        failed_run,
        repair_context,
    )

    assert repair_route is not None
    assert planner._explicit_target_paths(repair_route) == ["tests/test_cli.py"]
    assert planner._next_update_target(repair_route, session) == "tests/test_cli.py"
    assert planner._read_candidates(
        repair_route,
        session,
        planner._candidate_paths(repair_route, session),
    ) == ["tests/test_cli.py"]


def test_planner_blocks_repair_reverify_when_no_mutation_exists(tmp_path, monkeypatch):
    broken = tmp_path / "tests"
    broken.mkdir()
    (broken / "test_cli.py").write_text("def test_cli():\n    assert False\n", encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 2,
                "important_files": ["tests/test_cli.py"],
                "focus_files": ["tests/test_cli.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli", "python -m unittest"],
            }
        ),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_cli",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_cli", "python -m unittest"],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failing test module."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["tests/test_cli.py"],
        target_name="test_cli.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli and fix the failing CLI behavior.",
    )
    session.changed_files.append(FileChangeRecord(path="tests/test_cli.py", operation="write"))
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_cli",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=1,
            iteration=8,
            summary="Validation command exited with 1.",
            excerpt="TypeError: main() takes 0 positional arguments but 1 was given\n",
        )
    )

    def fake_execute_action_from_plan(_route, _session):
        return AgentDecision(
            thought_summary="Rerun validation immediately.",
            action_type=AgentActionType.CALL_TOOL,
            tool_name="run_tests",
            tool_args={"command": "python -m unittest", "cwd": ".", "timeout": 120},
            expected_outcome="Run the next validation step for the current changes.",
            final_response=None,
        )

    monkeypatch.setattr(planner, "execute_action_from_plan", fake_execute_action_from_plan)

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.FINAL
    assert session.stop_reason == "repair_blocked_after_validation_failure"
    assert session.blockers


def test_compact_repair_update_prompt_omits_heavy_validation_metadata(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    main = pkg / "__main__.py"
    main.write_text(
        "import argparse\nfrom .cli import greet\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name', nargs='?', default='world')\n"
        "    args = parser.parse_args()\n"
        "    print(greet(args.name))\n",
        encoding="utf-8",
    )
    cli = pkg / "cli.py"
    cli.write_text("def greet(name):\n    return f'Hello, {name}!'\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_cli = tests_dir / "test_cli.py"
    test_content = (
        "import unittest\n"
        "from unittest.mock import patch\n"
        "from io import StringIO\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    @patch('sys.stdout', new_callable=StringIO)\n"
        "    def test_greet_with_name(self, mock_stdout):\n"
        "        from greet_cli import __main__\n"
        "        __main__.main(['Ada'])\n"
        "        self.assertEqual(mock_stdout.getvalue().strip(), 'Hello, Ada!')\n"
    )
    test_cli.write_text(test_content, encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 4,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "focus_files": ["tests/test_cli.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_cli",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_cli"],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failing test module."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["tests/test_cli.py"],
        target_name="test_cli.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli and fix the failing CLI behavior.",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="greet_cli/__main__.py", operation="write"),
            FileChangeRecord(path="greet_cli/cli.py", operation="write"),
            FileChangeRecord(path="tests/test_cli.py", operation="write"),
        ]
    )
    session.tool_calls.extend(
        [
            ToolCallRecord(
                iteration=1,
                tool_name="read_file",
                tool_args={"path": "greet_cli/__main__.py"},
                success=True,
                summary="Read greet_cli/__main__.py.",
                output_excerpt=main.read_text(encoding="utf-8"),
            ),
            ToolCallRecord(
                iteration=2,
                tool_name="read_file",
                tool_args={"path": "greet_cli/cli.py"},
                success=True,
                summary="Read greet_cli/cli.py.",
                output_excerpt=cli.read_text(encoding="utf-8"),
            ),
            ToolCallRecord(
                iteration=3,
                tool_name="read_file",
                tool_args={"path": "tests/test_cli.py"},
                success=True,
                summary="Read tests/test_cli.py.",
                output_excerpt=test_content,
            ),
        ]
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_cli",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=1,
            iteration=8,
            summary="Validation command exited with 1.",
            excerpt=(
                "Traceback (most recent call last):\n"
                '  File "/tmp/tests/test_cli.py", line 8, in test_greet_with_name\n'
                "    __main__.main(['Ada'])\n"
                "TypeError: main() takes 0 positional arguments but 1 was given\n"
            ),
        )
    )

    failed_run = session.validation_runs[-1]
    repair_context = planner.validation_planner.build_failure_evidence(session, failed_run)
    repair_route = planner._repair_route_after_failed_validation(
        session.router_result,
        session,
        failed_run,
        repair_context,
    )

    assert repair_route is not None
    prompt = generate_content_prompt(
        repair_route,
        session,
        path="tests/test_cli.py",
        current_content=test_content,
        repair_context=repair_context,
        repair_strategy="validation_targeted",
        mode="compact",
    )

    assert "evidence_signature" not in prompt
    assert "Task focus:" not in prompt
    assert "User goal:" not in prompt
    assert "Supporting file hints:" in prompt
    assert "Validation-guided repair context:" in prompt
    assert len(prompt) < 5000


def test_compact_repair_update_prompt_filters_other_target_requirements(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    main = pkg / "__main__.py"
    main.write_text(
        "import argparse\nfrom .cli import greet\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name', nargs='?', default='world')\n"
        "    args = parser.parse_args()\n"
        "    print(greet(args.name))\n",
        encoding="utf-8",
    )
    cli = pkg / "cli.py"
    cli.write_text("def greet(name):\n    return f'Hello, {name}!'\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_content = (
        "import unittest\n"
        "from unittest.mock import patch\n"
        "from io import StringIO\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    @patch('sys.stdout', new_callable=StringIO)\n"
        "    def test_greet_with_name(self, mock_stdout):\n"
        "        from greet_cli import __main__\n"
        "        __main__.main(['Ada'])\n"
        "        self.assertEqual(mock_stdout.getvalue().strip(), 'Hello, Ada!')\n"
    )
    (tests_dir / "test_cli.py").write_text(test_content, encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 4,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "focus_files": ["tests/test_cli.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_cli",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_cli"],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failing CLI entrypoint."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["greet_cli/__main__.py"],
        target_name="__main__.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli and fix the failing CLI behavior.",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="greet_cli/__main__.py", operation="write"),
            FileChangeRecord(path="greet_cli/cli.py", operation="write"),
            FileChangeRecord(path="tests/test_cli.py", operation="write"),
        ]
    )
    session.tool_calls.extend(
        [
            ToolCallRecord(
                iteration=1,
                tool_name="read_file",
                tool_args={"path": "greet_cli/__main__.py"},
                success=True,
                summary="Read greet_cli/__main__.py.",
                output_excerpt=main.read_text(encoding="utf-8"),
            ),
            ToolCallRecord(
                iteration=2,
                tool_name="read_file",
                tool_args={"path": "greet_cli/cli.py"},
                success=True,
                summary="Read greet_cli/cli.py.",
                output_excerpt=cli.read_text(encoding="utf-8"),
            ),
            ToolCallRecord(
                iteration=3,
                tool_name="read_file",
                tool_args={"path": "tests/test_cli.py"},
                success=True,
                summary="Read tests/test_cli.py.",
                output_excerpt=test_content,
            ),
        ]
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_cli",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=1,
            iteration=8,
            summary="Validation command exited with 1.",
            excerpt=(
                "Traceback (most recent call last):\n"
                '  File "/tmp/tests/test_cli.py", line 8, in test_greet_with_name\n'
                "    __main__.main(['Ada'])\n"
                "TypeError: main() takes 0 positional arguments but 1 was given\n"
            ),
        )
    )

    failed_run = session.validation_runs[-1]
    repair_context = planner.validation_planner.build_failure_evidence(session, failed_run)
    repair_route = planner._repair_route_after_failed_validation(
        session.router_result,
        session,
        failed_run,
        repair_context,
    )

    prompt = generate_content_prompt(
        repair_route,
        session,
        path="greet_cli/__main__.py",
        current_content=main.read_text(encoding="utf-8"),
        repair_context=repair_context,
        repair_strategy="validation_escalated",
        mode="compact",
    )

    assert "Target path: greet_cli/__main__.py" in prompt
    assert "Change tests/test_cli.py so the failing runtime or test path can complete successfully." not in prompt
    assert "Validation-guided repair context:" in prompt
    assert "Failure focus:" in prompt
    assert "TypeError: main() takes 0 positional arguments but 1 was given" in prompt


def test_compact_repair_update_prompt_adds_sys_argv_launcher_hint_for_argparse_runtime_failures(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    main = pkg / "__main__.py"
    main.write_text(
        "import argparse\nfrom .cli import greet\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser(description='Greet someone by name.')\n"
        "    parser.add_argument('name', nargs='?', default='world')\n"
        "    args = parser.parse_args()\n"
        "    print(greet(args.name))\n",
        encoding="utf-8",
    )
    cli = pkg / "cli.py"
    cli.write_text("def greet(name):\n    return f'Hello, {name}!'\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_content = (
        "import unittest\n"
        "from unittest.mock import patch\n"
        "from io import StringIO\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    @patch('sys.stdout', new_callable=StringIO)\n"
        "    def test_greet_with_name(self, mock_stdout):\n"
        "        from greet_cli import __main__\n"
        "        with patch('__main__.sys.argv', ['python', '-m', 'greet_cli', 'Ada']):\n"
        "            __main__.main()\n"
        "    @patch('sys.stdout', new_callable=StringIO)\n"
        "    def test_greet_without_name(self, mock_stdout):\n"
        "        from greet_cli import __main__\n"
        "        with patch('__main__.sys.argv', ['python', '-m', 'greet_cli']):\n"
        "            __main__.main()\n"
    )
    (tests_dir / "test_cli.py").write_text(test_content, encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 4,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "focus_files": ["tests/test_cli.py", "greet_cli/__main__.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_cli",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_cli"],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failing CLI entrypoint."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["greet_cli/__main__.py"],
        target_name="__main__.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli and fix the failing CLI behavior.",
    )
    session.tool_calls.extend(
        [
            ToolCallRecord(
                iteration=1,
                tool_name="read_file",
                tool_args={"path": "greet_cli/__main__.py"},
                success=True,
                summary="Read greet_cli/__main__.py.",
                output_excerpt=main.read_text(encoding="utf-8"),
            ),
            ToolCallRecord(
                iteration=2,
                tool_name="read_file",
                tool_args={"path": "tests/test_cli.py"},
                success=True,
                summary="Read tests/test_cli.py.",
                output_excerpt=test_content,
            ),
        ]
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_cli",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=1,
            iteration=8,
            summary="Validation command exited with 1.",
            excerpt=(
                "usage: python [-h] [name]\n"
                "python: error: unrecognized arguments: -m Ada\n"
                "SystemExit: 2\n"
            ),
        )
    )

    failed_run = session.validation_runs[-1]
    repair_context = planner.validation_planner.build_failure_evidence(session, failed_run)
    repair_route = planner._repair_route_after_failed_validation(
        session.router_result,
        session,
        failed_run,
        repair_context,
    )

    prompt = generate_content_prompt(
        repair_route,
        session,
        path="greet_cli/__main__.py",
        current_content=main.read_text(encoding="utf-8"),
        repair_context=repair_context,
        repair_strategy="validation_targeted",
        mode="compact",
    )

    assert "Do not pass 'python', '-m', or the module name into argparse" in prompt
    assert "argparse should see ['Ada'] in the named case and [] in the default-name case" in prompt


def test_compact_repair_update_prompt_adds_mandatory_mutation_anchors_after_unchanged_review(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    main = pkg / "__main__.py"
    current_content = (
        "import argparse\nfrom .cli import greet\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser(description='Greet someone by name.')\n"
        "    parser.add_argument('name', nargs='?', default='world')\n"
        "    args = parser.parse_args()\n"
        "    print(greet(args.name))\n"
    )
    main.write_text(current_content, encoding="utf-8")
    cli = pkg / "cli.py"
    cli.write_text("def greet(name):\n    return f'Hello, {name}!'\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_content = (
        "import unittest\n"
        "from unittest.mock import patch\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    def test_greet_with_name(self):\n"
        "        from greet_cli import __main__\n"
        "        with patch('__main__.sys.argv', ['python', '-m', 'greet_cli', 'Ada']):\n"
        "            __main__.main()\n"
    )
    (tests_dir / "test_cli.py").write_text(test_content, encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "focus_files": ["tests/test_cli.py", "greet_cli/__main__.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
        validation_status="failed",
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failing CLI entrypoint."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["greet_cli/__main__.py"],
        target_name="__main__.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli and fix the failing CLI behavior.",
    )
    session.tool_calls.extend(
        [
            ToolCallRecord(
                iteration=1,
                tool_name="read_file",
                tool_args={"path": "greet_cli/__main__.py"},
                success=True,
                summary="Read greet_cli/__main__.py.",
                output_excerpt=current_content,
            ),
            ToolCallRecord(
                iteration=2,
                tool_name="read_file",
                tool_args={"path": "tests/test_cli.py"},
                success=True,
                summary="Read tests/test_cli.py.",
                output_excerpt=test_content,
            ),
        ]
    )
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/__main__.py", "tests/test_cli.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "usage: python [-h] [name]\n"
            "python: error: unrecognized arguments: -m Ada\n"
            '  File "/tmp/greet_cli/__main__.py", line 7, in main\n'
            "    args = parser.parse_args()\n"
            "SystemExit: 2\n"
        ),
        failure_summary="The CLI entrypoint rejects the patched python -m style argv.",
        expected_features=[],
        missing_features=[],
        file_hints=["greet_cli/__main__.py", "tests/test_cli.py"],
        line_hints=[7],
        action_hints=[],
        repair_requirements=[],
        evidence_signature="sig-unchanged-review",
    )

    prompt = generate_content_prompt(
        session.router_result,
        session,
        path="greet_cli/__main__.py",
        current_content=current_content,
        repair_context=repair_context,
        repair_strategy="validation_targeted",
        review_feedback=ProposedUpdateReview(
            safe_to_write=False,
            summary="Lines unchanged.",
            confidence=0.84,
            blocking_issues=["The proposal leaves the implicated identifier lines unchanged: main near args = parser.parse_args()"],
            preservation_risks=[],
            repair_hints=["Change the implicated callable instead of editing unrelated helper code."],
        ),
        mode="compact",
    )

    assert "Mandatory mutation anchors:" in prompt
    assert "At least one listed current target line or behavior anchor must change" in prompt
    assert "7:     args = parser.parse_args()" in prompt
    assert "Previous proposal was rejected because: The proposal leaves the implicated identifier lines unchanged" in prompt
    assert "Required repair direction: Change the implicated callable instead of editing unrelated helper code." in prompt


def test_compact_repair_update_prompt_prefers_created_test_context_and_traceback_focus(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    main = pkg / "__main__.py"
    main.write_text(
        "import argparse\nfrom .cli import greet\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name', nargs='?', default='world')\n"
        "    args = parser.parse_args()\n"
        "    print(greet(args.name))\n",
        encoding="utf-8",
    )
    cli = pkg / "cli.py"
    cli.write_text("def greet(name):\n    return f'Hello, {name}!'\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("# Greet CLI\n\nUsage docs.\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_content = (
        "import unittest\n"
        "from unittest.mock import patch\n"
        "from io import StringIO\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    @patch('sys.stdout', new_callable=StringIO)\n"
        "    def test_greet_with_name(self, mock_stdout):\n"
        "        from greet_cli import __main__\n"
        "        __main__.main(['Ada'])\n"
        "        self.assertEqual(mock_stdout.getvalue().strip(), 'Hello, Ada!')\n"
    )
    (tests_dir / "test_cli.py").write_text(test_content, encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 4,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py", "README.md"],
                "focus_files": ["tests/test_cli.py", "greet_cli/__main__.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_cli",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_cli"],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failing CLI entrypoint."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["greet_cli/__main__.py"],
        target_name="__main__.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli and fix the failing CLI behavior.",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="greet_cli/__main__.py", operation="write"),
            FileChangeRecord(path="greet_cli/cli.py", operation="write"),
            FileChangeRecord(path="tests/test_cli.py", operation="write"),
        ]
    )
    session.tool_calls.extend(
        [
            ToolCallRecord(
                iteration=1,
                tool_name="read_file",
                tool_args={"path": "greet_cli/__main__.py"},
                success=True,
                summary="Read greet_cli/__main__.py.",
                output_excerpt=main.read_text(encoding="utf-8"),
            ),
            ToolCallRecord(
                iteration=2,
                tool_name="read_file",
                tool_args={"path": "greet_cli/cli.py"},
                success=True,
                summary="Read greet_cli/cli.py.",
                output_excerpt=cli.read_text(encoding="utf-8"),
            ),
            ToolCallRecord(
                iteration=3,
                tool_name="read_file",
                tool_args={"path": "README.md"},
                success=True,
                summary="Read README.md.",
                output_excerpt=readme.read_text(encoding="utf-8"),
            ),
            ToolCallRecord(
                iteration=4,
                tool_name="create_file",
                tool_args={"path": "tests/test_cli.py", "content": test_content, "overwrite": False},
                success=True,
                summary="Created tests/test_cli.py.",
                output_excerpt=test_content,
            ),
        ]
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_cli",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=1,
            iteration=8,
            summary="Validation command exited with 1.",
            excerpt=(
                "Traceback (most recent call last):\n"
                '  File "/tmp/tests/test_cli.py", line 8, in test_greet_with_name\n'
                "    __main__.main(['Ada'])\n"
                "TypeError: main() takes 0 positional arguments but 1 was given\n"
            ),
        )
    )

    failed_run = session.validation_runs[-1]
    repair_context = planner.validation_planner.build_failure_evidence(session, failed_run)
    repair_route = planner._repair_route_after_failed_validation(
        session.router_result,
        session,
        failed_run,
        repair_context,
    )

    prompt = generate_content_prompt(
        repair_route,
        session,
        path="greet_cli/__main__.py",
        current_content=main.read_text(encoding="utf-8"),
        repair_context=repair_context,
        repair_strategy="validation_escalated",
        mode="compact",
    )

    assert "Supporting file hints: tests/test_cli.py:" in prompt
    assert "import unittest" in prompt
    assert "@@ -0,0 +" not in prompt
    assert "README.md:" not in prompt
    assert "Failure focus:" in prompt
    assert "__main__.main(['Ada'])" in prompt
    assert "TypeError: main() takes 0 positional arguments but 1 was given" in prompt


def test_compact_repair_prompt_focuses_supporting_test_lines_around_runtime_hints(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    main = pkg / "__main__.py"
    main.write_text(
        "import argparse\nfrom .cli import greet\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name', nargs='?', default='world')\n"
        "    args = parser.parse_args()\n"
        "    print(greet(args.name))\n",
        encoding="utf-8",
    )
    cli = pkg / "cli.py"
    cli.write_text("def greet(name):\n    return f'Hello, {name}!'\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_content = (
        "import unittest\n"
        "from unittest.mock import patch\n"
        "from io import StringIO\n\n"
        "class TestCLI(unittest.TestCase):\n\n"
        "    @patch('sys.stdout', new_callable=StringIO)\n"
        "    def test_greet_with_name(self, mock_stdout):\n"
        "        from greet_cli import __main__\n"
        "        with patch('__main__.sys.argv', ['python', '-m', 'greet_cli', 'Ada']):\n"
        "            __main__.main()\n"
        "        self.assertEqual(mock_stdout.getvalue().strip(), \"Hello, Ada!\")\n\n"
        "    @patch('sys.stdout', new_callable=StringIO)\n"
        "    def test_greet_without_name(self, mock_stdout):\n"
        "        from greet_cli import __main__\n"
        "        with patch('__main__.sys.argv', ['python', '-m', 'greet_cli']):\n"
        "            __main__.main()\n"
        "        self.assertEqual(mock_stdout.getvalue().strip(), \"Hello, world!\")\n"
    )
    (tests_dir / "test_cli.py").write_text(test_content, encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 4,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "focus_files": ["tests/test_cli.py", "greet_cli/__main__.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_cli",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_cli"],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failing CLI entrypoint."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["greet_cli/__main__.py"],
        target_name="__main__.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli and fix the failing CLI behavior.",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="greet_cli/__main__.py", operation="write"),
            FileChangeRecord(path="greet_cli/cli.py", operation="write"),
            FileChangeRecord(path="tests/test_cli.py", operation="write"),
        ]
    )
    session.tool_calls.append(
        ToolCallRecord(
            iteration=4,
            tool_name="create_file",
            tool_args={"path": "tests/test_cli.py", "content": test_content, "overwrite": False},
            success=True,
            summary="Created tests/test_cli.py.",
            output_excerpt=test_content,
        )
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_cli",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=1,
            iteration=8,
            summary="Validation command exited with 1.",
            excerpt=(
                "usage: python [-h] [name]\n"
                "python: error: unrecognized arguments: -m\n"
                "  File \"/tmp/tests/test_cli.py\", line 11, in test_greet_with_name\n"
                "    __main__.main()\n"
                "  File \"/tmp/greet_cli/__main__.py\", line 7, in main\n"
                "    args = parser.parse_args()\n"
            ),
        )
    )

    failed_run = session.validation_runs[-1]
    repair_context = planner.validation_planner.build_failure_evidence(session, failed_run)
    repair_route = planner._repair_route_after_failed_validation(
        session.router_result,
        session,
        failed_run,
        repair_context,
    )

    prompt = generate_content_prompt(
        repair_route,
        session,
        path="greet_cli/__main__.py",
        current_content=main.read_text(encoding="utf-8"),
        repair_context=repair_context,
        repair_strategy="validation_escalated",
        mode="compact",
    )

    assert "9:         from greet_cli import __main__" in prompt
    assert "10:         with patch('__main__.sys.argv', ['python', '-m', 'greet_cli', 'Ada']):" in prompt
    assert "17:         with patch('__main__.sys.argv', ['python', '-m', 'greet_cli']):" in prompt
    assert "1: import unittest" not in prompt


def test_compact_repair_retry_prompt_includes_optional_argv_and_launcher_strip_hints(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    main = pkg / "__main__.py"
    main.write_text(
        "import argparse\nfrom .cli import greet\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name', nargs='?', default='world')\n"
        "    args = parser.parse_args()\n"
        "    print(greet(args.name))\n",
        encoding="utf-8",
    )
    cli = pkg / "cli.py"
    cli.write_text("def greet(name):\n    return f'Hello, {name}!'\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_content = (
        "from unittest.mock import patch\n\n"
        "def test_cli_runtime():\n"
        "    from greet_cli import __main__\n"
        "    with patch('__main__.sys.argv', ['python', '-m', 'greet_cli', 'Ada']):\n"
        "        __main__.main()\n"
    )
    (tests_dir / "test_cli.py").write_text(test_content, encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root=str(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failing CLI entrypoint."},
        ],
        target_paths=["greet_cli/__main__.py"],
        target_name="__main__.py",
    )
    commit_task_state_and_route(planner, session, payload)
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="read_file",
            tool_args={"path": "tests/test_cli.py"},
            success=True,
            summary="Read tests/test_cli.py.",
            output_excerpt=test_content,
        )
    )
    review_feedback = ProposedUpdateReview(
        safe_to_write=False,
        summary="The proposed repair still routes launcher tokens into argparse.",
        confidence=0.95,
        blocking_issues=[
            "The proposal for greet_cli/__main__.py still ignores the patched __main__.sys.argv runtime input for the direct __main__.main() invocation.",
        ],
        preservation_risks=[],
        repair_hints=[
            "When the failing tests call __main__.main() while patching __main__.sys.argv, derive the CLI arguments from the patched runtime argv when no explicit arguments were passed.",
        ],
    )
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/__main__.py", "tests/test_cli.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "usage: python [-h] [name]\n"
            "python: error: unrecognized arguments: -m Ada\n"
            '  File "/tmp/tests/test_cli.py", line 5, in test_cli_runtime\n'
            "    __main__.main()\n"
            '  File "/tmp/greet_cli/__main__.py", line 7, in main\n'
            "    args = parser.parse_args()\n"
        ),
        failure_summary="python: error: unrecognized arguments: -m Ada",
        expected_features=[],
        missing_features=[],
        file_hints=["greet_cli/__main__.py", "tests/test_cli.py"],
        line_hints=[5, 7],
        action_hints=["Prefer a targeted fix over broad refactors."],
        repair_requirements=["Change greet_cli/__main__.py so the failing runtime path succeeds."],
        evidence_signature="sig",
    )

    prompt = generate_content_retry_prompt(
        session.router_result,
        session,
        path="greet_cli/__main__.py",
        current_content=main.read_text(encoding="utf-8"),
        repair_context=repair_context,
        repair_strategy="validation_targeted",
        review_feedback=review_feedback,
        mode="compact",
    )

    assert "If the updated file references sys.argv, add import sys before using it." in prompt
    assert "Do not use sys.argv[1:] or sys.argv[2:] here." in prompt
    assert "Keep main() callable without positional arguments." in prompt
    assert "strip a leading python -m launcher prefix" in prompt
    assert "derive the CLI arguments from the patched runtime argv" in prompt


def test_generate_content_prompt_includes_runtime_fixture_data_hints(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_wordfreq.py").write_text(
        "import unittest\n"
        "from wordfreq import count_words\n\n"
        "class TestWordFreq(unittest.TestCase):\n"
        "    def test_count_words(self):\n"
        "        expected = {'hello': 2, 'world': 1}\n"
        "        self.assertEqual(count_words('tests/test_data.txt'), expected)\n",
        encoding="utf-8",
    )

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Create a small Python CLI in wordfreq.py and make the unittest pass.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 1,
                "important_files": ["tests/test_wordfreq.py"],
                "focus_files": ["tests/test_wordfreq.py"],
                "test_files": ["tests/test_wordfreq.py"],
                "likely_commands": ["python -m unittest tests.test_wordfreq"],
            }
        ),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failing fixture file."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["tests/test_data.txt"],
        target_name="test_data.txt",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_wordfreq and finish only when it passes.",
    )
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_wordfreq",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["wordfreq.py", "tests/test_wordfreq.py", "tests/test_data.txt"],
        summary="Validation command exited with 1.",
        excerpt=(
            "Traceback (most recent call last):\n"
            f'  File "{tests_dir / "test_wordfreq.py"}", line 7, in test_count_words\n'
            "    self.assertEqual(count_words('tests/test_data.txt'), expected)\n"
            "AssertionError: {'this': 1, 'sample': 1} != {'hello': 2, 'world': 1}\n"
        ),
        failure_summary="tests/test_data.txt still produces extra words during the runtime validation.",
        expected_features=[],
        missing_features=[],
        file_hints=["wordfreq.py", "tests/test_wordfreq.py", "tests/test_data.txt"],
        line_hints=[7],
        action_hints=[],
        repair_requirements=[],
        evidence_signature="sig-runtime-fixture-hints",
    )

    prompt = generate_content_prompt(
        session.router_result,
        session,
        path="tests/test_data.txt",
        current_content="This is placeholder fixture text.\nHello hello world\n",
        repair_context=repair_context,
        repair_strategy="validation_targeted",
        mode="compact",
    )

    assert "Runtime support file hints:" in prompt
    assert "This file is runtime input or fixture data" in prompt
    assert "Write only the minimal raw content needed" in prompt
    assert "Do not add explanatory prose" in prompt


def test_fallback_review_allows_large_runtime_fixture_rewrite(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Repair the failing runtime fixture.",
        workspace_root=str(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failing fixture file."},
        ],
        target_paths=["tests/test_data.txt"],
        target_name="test_data.txt",
    )
    commit_task_state_and_route(planner, session, payload)
    session.active_repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_wordfreq",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["tests/test_data.txt"],
        summary="Validation command exited with 1.",
        excerpt=(
            "Traceback (most recent call last):\n"
            "  File \"/tmp/tests/test_wordfreq.py\", line 8, in test_count_words\n"
            "    self.assertEqual(result, expected)\n"
            "AssertionError: {'this': 1, 'sample': 1} != {'hello': 2, 'world': 1}\n"
        ),
        failure_summary="tests/test_data.txt still produces extra words during the runtime validation.",
        expected_features=[],
        missing_features=[],
        file_hints=["tests/test_data.txt"],
        line_hints=[8],
        action_hints=[],
        repair_requirements=[],
        evidence_signature="sig-runtime-fixture-reduction",
    )

    review = planner._fallback_proposed_update_review(
        session.router_result,
        session=session,
        path="tests/test_data.txt",
        current_content=(
            "This is a test data file for wordfreq.py.\n"
            "It contains some words to be counted.\n"
            "Punctuation should be ignored.\n"
            "Case insensitivity matters.\n"
        ),
        proposed_content="Hello hello world\n",
    )

    assert review.safe_to_write is True
    assert "fixture or support data" in review.summary


def test_validation_relevance_review_blocks_runtime_fixture_that_keeps_sample_prose(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_wordfreq",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["wordfreq.py", "tests/test_wordfreq.py", "tests/test_data.txt"],
        summary="Validation command exited with 1.",
        excerpt=(
            "Traceback (most recent call last):\n"
            "  File \"tests/test_wordfreq.py\", line 7, in test_count_words\n"
            "    self.assertEqual(result, expected)\n"
            "AssertionError: ['This', 'is', 'placeholder', 'fixture', 'text.'] != {'hello': 2, 'world': 1}\n"
        ),
        failure_summary="AssertionError: the runtime fixture still produces extra words instead of the expected result.",
        expected_features=[],
        missing_features=[],
        file_hints=["wordfreq.py", "tests/test_wordfreq.py", "tests/test_data.txt"],
        line_hints=[7],
        action_hints=[],
        repair_requirements=[],
        evidence_signature="sig-runtime-fixture-prose",
    )

    review = planner._validation_repair_relevance_review(
        path="tests/test_data.txt",
        current_content=(
            "This is a sample text file for testing wordfreq.py.\n"
            "It contains some words, punctuation, and repeated words to test the functionality of the script.\n"
            "Hello hello world\n"
        ),
        proposed_content=(
            "This is a sample text file for testing wordfreq.py.\n"
            "It contains some words, punctuation, and repeated words to test the functionality of the script.\n"
            "hello hello world\n"
        ),
        repair_context=repair_context,
    )

    assert review is not None
    assert review.safe_to_write is False
    assert "runtime support file" in review.summary.lower()


def test_validation_relevance_review_allows_minimal_runtime_fixture_content(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_wordfreq",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["wordfreq.py", "tests/test_wordfreq.py", "tests/test_data.txt"],
        summary="Validation command exited with 1.",
        excerpt=(
            "Traceback (most recent call last):\n"
            "  File \"tests/test_wordfreq.py\", line 7, in test_count_words\n"
            "    self.assertEqual(result, expected)\n"
            "AssertionError: ['This', 'is', 'placeholder', 'fixture', 'text.'] != {'hello': 2, 'world': 1}\n"
        ),
        failure_summary="AssertionError: the runtime fixture still produces extra words instead of the expected result.",
        expected_features=[],
        missing_features=[],
        file_hints=["wordfreq.py", "tests/test_wordfreq.py", "tests/test_data.txt"],
        line_hints=[7],
        action_hints=[],
        repair_requirements=[],
        evidence_signature="sig-runtime-fixture-minimal",
    )

    review = planner._validation_repair_relevance_review(
        path="tests/test_data.txt",
        current_content=(
            "This is a sample text file for testing wordfreq.py.\n"
            "It contains some words, punctuation, and repeated words to test the functionality of the script.\n"
            "Hello hello world\n"
        ),
        proposed_content="hello hello world\n",
        repair_context=repair_context,
    )

    assert review is None


def test_compact_create_prompt_marks_other_requested_files_out_of_scope(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Create wordfreq.py, README.md, and tests/test_wordfreq.py for a tiny CLI.",
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="create",
        action_plan=[
            {"step": 1, "action": "create_artifact", "reason": "Create the requested CLI file first."},
        ],
        target_paths=["wordfreq.py", "README.md", "tests/test_wordfreq.py"],
        target_name="wordfreq.py",
    )
    commit_task_state_and_route(planner, session, payload, verification_target="python -m unittest tests.test_wordfreq")

    prompt = generate_content_prompt(
        session.router_result,
        session,
        path="wordfreq.py",
        mode="compact",
    )

    assert "Only write wordfreq.py." in prompt
    assert "out of scope for this output" in prompt
    assert "README.md" in prompt
    assert "tests/test_wordfreq.py" in prompt


def test_compact_create_retry_prompt_marks_other_requested_files_out_of_scope(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Create wordfreq.py, README.md, and tests/test_wordfreq.py for a tiny CLI.",
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="create",
        action_plan=[
            {"step": 1, "action": "create_artifact", "reason": "Create the requested CLI file first."},
        ],
        target_paths=["wordfreq.py", "README.md", "tests/test_wordfreq.py"],
        target_name="wordfreq.py",
    )
    commit_task_state_and_route(planner, session, payload, verification_target="python -m unittest tests.test_wordfreq")
    review_feedback = ProposedUpdateReview(
        safe_to_write=False,
        summary="The previous draft mixed multiple requested files together.",
        confidence=0.94,
        blocking_issues=["The proposed wordfreq.py draft included content that belongs in README.md and tests/test_wordfreq.py."],
        preservation_risks=[],
        repair_hints=["Return only the Python source for wordfreq.py."],
    )

    prompt = generate_content_retry_prompt(
        session.router_result,
        session,
        path="wordfreq.py",
        review_feedback=review_feedback,
        mode="compact",
    )

    assert "Only write wordfreq.py." in prompt
    assert "out of scope for this output" in prompt
    assert "README.md" in prompt
    assert "tests/test_wordfreq.py" in prompt


def test_targeted_runtime_prompt_hints_detect_launcher_failure_even_when_supporting_excerpt_is_truncated():
    hints = _targeted_runtime_prompt_hints(
        path="greet_cli/__main__.py",
        current_content=(
            "import argparse\n"
            "from .cli import greet\n\n"
            "def main():\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('name', nargs='?', default='world')\n"
            "    args = parser.parse_args()\n"
            "    print(greet(args.name))\n"
        ),
        supporting_context=(
            "tests/test_cli.py:\n"
            "9:         from greet_cli import __main__\n"
            "10:         with patch('__main__.sys.argv', ['pytho…\n"
        ),
        targeted_context={
            "failure_summary": "python: error: unrecognized arguments: -m Ada",
            "excerpt": (
                '  File "/tmp/tests/test_cli.py", line 11, in test_greet_with_name\n'
                "    __main__.main()\n"
                '  File "/tmp/greet_cli/__main__.py", line 7, in main\n'
                "    args = parser.parse_args()\n"
            ),
            "failure_focus": [
                "ERROR: test_greet_with_name (tests.test_cli.TestCLI.test_greet_with_name)",
                "__main__.main()",
                "args = parser.parse_args()",
            ],
            "file_hints": ["greet_cli/__main__.py", "tests/test_cli.py"],
        },
    )

    assert any("only the trailing CLI arguments should reach the parser" in hint for hint in hints)
    assert any("Keep main() callable without positional arguments." in hint for hint in hints)
    assert any("strip a leading python -m launcher prefix" in hint for hint in hints)


def test_targeted_runtime_failure_focus_keeps_adjacent_code_lines_for_runtime_repairs():
    focus = _targeted_runtime_failure_focus_lines(
        (
            "usage: python [-h] [name]\n"
            "python: error: unrecognized arguments: -m\n"
            '  File "/home/marc/workspace/e2e-python-cli/tests/test_cli.py", line 11, in test_greet_with_name\n'
            "    __main__.main()\n"
            '  File "/home/marc/workspace/e2e-python-cli/greet_cli/__main__.py", line 7, in main\n'
            "    args = parser.parse_args()\n"
            "ERROR: test_greet_with_name (tests.test_cli.TestCLI.test_greet_with_name)\n"
        ),
        target_path="greet_cli/__main__.py",
        limit=6,
    )

    assert 'File "/home/marc/workspace/e2e-python-cli/greet_cli/__main__.py", line 7, in main' in focus
    assert "args = parser.parse_args()" in focus
    assert "__main__.main()" in focus


def test_planner_repair_switches_to_another_candidate_after_no_effective_change(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    (pkg / "__main__.py").write_text(
        "from .cli import greet\n\n"
        "def main():\n"
        "    print(greet('Ada'))\n",
        encoding="utf-8",
    )
    (pkg / "cli.py").write_text("def greet(name):\n    return f'Hello, {name}!'\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_cli.py").write_text(
        "import unittest\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    def test_greet_with_name(self):\n"
        "        from greet_cli import __main__\n"
        "        __main__.main(['Ada'])\n",
        encoding="utf-8",
    )

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 4,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "focus_files": ["tests/test_cli.py", "greet_cli/__main__.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_cli",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_cli"],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failing test module."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["tests/test_cli.py"],
        target_name="test_cli.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli and fix the failing CLI behavior.",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="greet_cli/__main__.py", operation="write"),
            FileChangeRecord(path="greet_cli/cli.py", operation="write"),
            FileChangeRecord(path="tests/test_cli.py", operation="write"),
        ]
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_cli",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=1,
            iteration=8,
            summary="Validation command exited with 1.",
            excerpt=(
                "Traceback (most recent call last):\n"
                '  File "/tmp/tests/test_cli.py", line 5, in test_greet_with_name\n'
                "    __main__.main(['Ada'])\n"
                "TypeError: main() takes 0 positional arguments but 1 was given\n"
            ),
        )
    )

    failed_run = session.validation_runs[-1]
    repair_context = planner.validation_planner.build_failure_evidence(session, failed_run)
    session.repair_history.append(
        RepairAttemptRecord(
            artifact_path="tests/test_cli.py",
            validation_command=repair_context.command,
            verification_scope=repair_context.verification_scope,
            strategy="validation_targeted",
            result="no_effective_change",
            reason="identical content",
            evidence_signature=repair_context.evidence_signature,
            iteration=9,
        )
    )

    next_target = planner._repair_target_after_failed_validation(
        session.router_result,
        session,
        failed_run,
        repair_context,
    )

    assert next_target == "greet_cli/__main__.py"


def test_planner_switches_target_immediately_after_identical_repair_generation(tmp_path, monkeypatch):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    main = pkg / "__main__.py"
    main.write_text(
        "from .cli import greet\n\n"
        "def main():\n"
        "    print(greet('Ada'))\n",
        encoding="utf-8",
    )
    cli = pkg / "cli.py"
    cli.write_text("def greet(name):\n    return f'Hello, {name}!'\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_cli = tests_dir / "test_cli.py"
    current = (
        "import unittest\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    def test_greet_with_name(self):\n"
        "        from greet_cli import __main__\n"
        "        __main__.main(['Ada'])\n"
    )
    test_cli.write_text(current, encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 4,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "focus_files": ["tests/test_cli.py", "greet_cli/__main__.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_cli",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_cli"],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failing test module."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["tests/test_cli.py"],
        target_name="test_cli.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli and fix the failing CLI behavior.",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="greet_cli/__main__.py", operation="write"),
            FileChangeRecord(path="greet_cli/cli.py", operation="write"),
            FileChangeRecord(path="tests/test_cli.py", operation="write"),
        ]
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_cli",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=1,
            iteration=8,
            summary="Validation command exited with 1.",
            excerpt=(
                "Traceback (most recent call last):\n"
                '  File "/tmp/tests/test_cli.py", line 5, in test_greet_with_name\n'
                "    __main__.main(['Ada'])\n"
                "TypeError: main() takes 0 positional arguments but 1 was given\n"
            ),
        )
    )
    session.tool_calls.append(
        ToolCallRecord(
            iteration=9,
            tool_name="read_file",
            tool_args={"path": "tests/test_cli.py"},
            success=True,
            summary="Read tests/test_cli.py.",
            output_excerpt=current,
        )
    )

    failed_run = session.validation_runs[-1]
    repair_context = planner.validation_planner.build_failure_evidence(session, failed_run)
    session.active_repair_context = repair_context
    repair_route = planner._repair_route_after_failed_validation(
        session.router_result,
        session,
        failed_run,
        repair_context,
    )

    assert repair_route is not None

    def fake_generate_file_content(*_args, **kwargs):
        return SimpleNamespace(content=kwargs["current_content"], failure=None)

    monkeypatch.setattr(planner, "_generate_file_content", fake_generate_file_content)

    decision = planner._draft_update_decision(repair_route, session, "tests/test_cli.py")

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "read_file"
    assert decision.tool_args["path"] == "greet_cli/__main__.py"


def test_planner_prefers_implementation_target_before_validation_target_for_runtime_failure(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    (pkg / "__main__.py").write_text(
        "from .cli import greet\n\n"
        "def main():\n"
        "    print(greet('Ada'))\n",
        encoding="utf-8",
    )
    (pkg / "cli.py").write_text("def greet(name):\n    return f'Hello, {name}!'\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_cli.py").write_text(
        "import unittest\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    def test_greet_with_name(self):\n"
        "        from greet_cli import __main__\n"
        "        __main__.main(['Ada'])\n",
        encoding="utf-8",
    )

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 4,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "focus_files": ["tests/test_cli.py", "greet_cli/__main__.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_cli",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_cli"],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failing CLI implementation."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["tests/test_cli.py"],
        target_name="test_cli.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli and fix the failing CLI behavior.",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="greet_cli/__main__.py", operation="write"),
            FileChangeRecord(path="greet_cli/cli.py", operation="write"),
            FileChangeRecord(path="tests/test_cli.py", operation="write"),
        ]
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_cli",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=1,
            iteration=8,
            summary="Validation command exited with 1.",
            excerpt=(
                "Traceback (most recent call last):\n"
                '  File "/tmp/tests/test_cli.py", line 5, in test_greet_with_name\n'
                "    __main__.main(['Ada'])\n"
                "TypeError: main() takes 0 positional arguments but 1 was given\n"
            ),
        )
    )

    failed_run = session.validation_runs[-1]
    repair_context = planner.validation_planner.build_failure_evidence(session, failed_run)

    next_target = planner._repair_target_after_failed_validation(
        session.router_result,
        session,
        failed_run,
        repair_context,
    )

    assert next_target == "greet_cli/__main__.py"


def test_planner_repairs_missing_runtime_fixture_before_reediting_implementation(tmp_path):
    (tmp_path / "wordfreq.py").write_text(
        "def wordfreq(file_path):\n"
        "    with open(file_path, 'r') as file:\n"
        "        return file.read()\n",
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_wordfreq.py").write_text(
        "import unittest\n"
        "from wordfreq import wordfreq\n\n"
        "class TestWordFreq(unittest.TestCase):\n"
        "    def test_wordfreq(self):\n"
        "        result = wordfreq('tests/test_data.txt')\n",
        encoding="utf-8",
    )

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Create wordfreq.py, README.md, and tests/test_wordfreq.py.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 2,
                "important_files": ["wordfreq.py", "tests/test_wordfreq.py"],
                "focus_files": ["wordfreq.py", "tests/test_wordfreq.py"],
                "test_files": ["tests/test_wordfreq.py"],
                "likely_commands": ["python -m unittest tests.test_wordfreq"],
            }
        ),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_wordfreq",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_wordfreq"],
    )
    payload = route_payload(
        intent="create",
        action_plan=[
            {"step": 1, "action": "create_artifact", "reason": "Create the requested CLI and tests."},
            {"step": 2, "action": "run_validation", "reason": "Run the targeted unittest module."},
        ],
        target_paths=["wordfreq.py", "README.md", "tests/test_wordfreq.py"],
        target_name="wordfreq.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_wordfreq and finish only when it passes.",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="wordfreq.py", operation="create"),
            FileChangeRecord(path="README.md", operation="create"),
            FileChangeRecord(path="tests/test_wordfreq.py", operation="create"),
        ]
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_wordfreq",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=1,
            iteration=6,
            summary="Validation command exited with 1.",
            excerpt=(
                "Traceback (most recent call last):\n"
                f'  File "{tmp_path / "tests" / "test_wordfreq.py"}", line 6, in test_wordfreq\n'
                "    result = wordfreq('tests/test_data.txt')\n"
                f'  File "{tmp_path / "wordfreq.py"}", line 2, in wordfreq\n'
                "    with open(file_path, 'r') as file:\n"
                "FileNotFoundError: [Errno 2] No such file or directory: 'tests/test_data.txt'\n"
            ),
        )
    )

    failed_run = session.validation_runs[-1]
    repair_context = planner.validation_planner.build_failure_evidence(session, failed_run)

    next_target = planner._repair_target_after_failed_validation(
        session.router_result,
        session,
        failed_run,
        repair_context,
    )

    assert next_target == "tests/test_data.txt"


def test_planner_keeps_recent_runtime_fixture_target_after_follow_up_assertion_failure(tmp_path):
    (tmp_path / "wordfreq.py").write_text(
        "def count_words(file_path):\n"
        "    with open(file_path, 'r') as file:\n"
        "        return file.read().split()\n",
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_wordfreq.py").write_text(
        "import unittest\n"
        "from wordfreq import count_words\n\n"
        "class TestWordFreq(unittest.TestCase):\n"
        "    def test_count_words(self):\n"
        "        expected = {'hello': 2, 'world': 1}\n"
        "        result = count_words('tests/test_data.txt')\n"
        "        self.assertEqual(result, expected)\n",
        encoding="utf-8",
    )
    (tests_dir / "test_data.txt").write_text(
        "This is placeholder fixture text.\n",
        encoding="utf-8",
    )

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Create a small Python CLI in wordfreq.py and make the unittest pass.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 3,
                "important_files": ["wordfreq.py", "tests/test_wordfreq.py", "tests/test_data.txt"],
                "focus_files": ["tests/test_wordfreq.py", "tests/test_data.txt", "wordfreq.py"],
                "test_files": ["tests/test_wordfreq.py"],
                "likely_commands": ["python -m unittest tests.test_wordfreq"],
            }
        ),
        validation_status="failed",
        edit_generation=2,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_wordfreq",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_wordfreq"],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failed validation."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["wordfreq.py"],
        target_name="wordfreq.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_wordfreq and finish only when it passes.",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="wordfreq.py", operation="create"),
            FileChangeRecord(path="tests/test_wordfreq.py", operation="create"),
            FileChangeRecord(path="tests/test_data.txt", operation="create"),
        ]
    )
    session.active_repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_wordfreq",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["wordfreq.py", "tests/test_wordfreq.py", "tests/test_data.txt"],
        summary="Validation command exited with 1.",
        excerpt=(
            "Traceback (most recent call last):\n"
            f'  File "{tmp_path / "tests" / "test_wordfreq.py"}", line 7, in test_count_words\n'
            "    result = count_words('tests/test_data.txt')\n"
            f'  File "{tmp_path / "wordfreq.py"}", line 2, in count_words\n'
            "    with open(file_path, 'r') as file:\n"
            "FileNotFoundError: [Errno 2] No such file or directory: 'tests/test_data.txt'\n"
        ),
        failure_summary="count_words failed because tests/test_data.txt is missing.",
        expected_features=[],
        missing_features=[],
        file_hints=["wordfreq.py", "tests/test_wordfreq.py", "tests/test_data.txt"],
        line_hints=[7, 2],
        action_hints=[],
        repair_requirements=[],
        evidence_signature="sig-wordfreq-missing-fixture",
    )
    session.repair_history.append(
        RepairAttemptRecord(
            artifact_path="tests/test_data.txt",
            validation_command="python -m unittest tests.test_wordfreq",
            verification_scope="runtime",
            strategy="validation_targeted",
            result="mutation_planned",
            reason="Prepared missing repair artifact tests/test_data.txt from validation evidence.",
            evidence_signature="sig-wordfreq-missing-fixture",
            iteration=4,
        )
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_wordfreq",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=2,
            iteration=6,
            summary="Validation command exited with 1.",
            excerpt=(
                "Traceback (most recent call last):\n"
                f'  File "{tmp_path / "tests" / "test_wordfreq.py"}", line 8, in test_count_words\n'
                "    self.assertEqual(result, expected)\n"
                "AssertionError: ['This', 'is', 'placeholder', 'fixture', 'text.'] != {'hello': 2, 'world': 1}\n"
            ),
        )
    )

    failed_run = session.validation_runs[-1]
    repair_context = planner.validation_planner.build_failure_evidence(session, failed_run)

    next_target = planner._repair_target_after_failed_validation(
        session.router_result,
        session,
        failed_run,
        repair_context,
    )

    assert next_target == "tests/test_data.txt"


def test_planner_prefers_runtime_fixture_target_on_initial_assertion_mismatch(tmp_path):
    (tmp_path / "wordfreq.py").write_text(
        "def count_words(file_path):\n"
        "    with open(file_path, 'r') as file:\n"
        "        return file.read().split()\n",
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_wordfreq.py").write_text(
        "import unittest\n"
        "from wordfreq import count_words\n\n"
        "class TestWordFreq(unittest.TestCase):\n"
        "    def test_count_words(self):\n"
        "        expected = {'hello': 2, 'world': 1}\n"
        "        result = count_words('tests/test_data.txt')\n"
        "        self.assertEqual(result, expected)\n",
        encoding="utf-8",
    )
    (tests_dir / "test_data.txt").write_text(
        "This is placeholder fixture text.\n",
        encoding="utf-8",
    )

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Fix the failing wordfreq runtime validation.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 3,
                "important_files": ["wordfreq.py", "tests/test_wordfreq.py", "tests/test_data.txt"],
                "focus_files": ["wordfreq.py", "tests/test_wordfreq.py", "tests/test_data.txt"],
                "test_files": ["tests/test_wordfreq.py"],
                "likely_commands": ["python -m unittest tests.test_wordfreq"],
            }
        ),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_wordfreq",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_wordfreq"],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failed validation."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["wordfreq.py"],
        target_name="wordfreq.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_wordfreq and finish only when it passes.",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="wordfreq.py", operation="create"),
            FileChangeRecord(path="tests/test_wordfreq.py", operation="create"),
            FileChangeRecord(path="tests/test_data.txt", operation="create"),
        ]
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_wordfreq",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=1,
            iteration=4,
            summary="Validation command exited with 1.",
            excerpt=(
                "Traceback (most recent call last):\n"
                f'  File "{tmp_path / "tests" / "test_wordfreq.py"}", line 7, in test_count_words\n'
                "    self.assertEqual(result, expected)\n"
                "AssertionError: ['This', 'is', 'placeholder', 'fixture', 'text.'] != {'hello': 2, 'world': 1}\n"
            ),
        )
    )

    failed_run = session.validation_runs[-1]
    repair_context = planner.validation_planner.build_failure_evidence(session, failed_run)

    next_target = planner._repair_target_after_failed_validation(
        session.router_result,
        session,
        failed_run,
        repair_context,
    )

    assert next_target == "tests/test_data.txt"


def test_planner_prefers_failure_specific_stylesheet_over_first_created_page_after_website_validation_failure(tmp_path):
    for relative_path, content in {
        "index.html": "<!doctype html><title>Home</title>\n",
        "about.html": "<!doctype html><title>About</title>\n",
        "projects.html": "<!doctype html><title>Projects</title>\n",
        "contact.html": "<!doctype html><title>Contact</title>\n",
        "styles.css": "body { color: #222; }\n",
    }.items():
        absolute = tmp_path / relative_path
        absolute.write_text(content, encoding="utf-8")

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")
    (tests_dir / "test_site.py").write_text(
        "import unittest\n\n"
        "class TestSite(unittest.TestCase):\n"
        "    def test_styles_include_variables_and_responsive_rule(self):\n"
        "        self.assertTrue(True)\n",
        encoding="utf-8",
    )

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task=(
            "Create a multi-page personal portfolio website in this repo. "
            "Add index.html, about.html, projects.html, contact.html, and styles.css. "
            "Use one shared navigation across pages, make it responsive, and finish only "
            "when python -m unittest tests.test_site passes."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 7,
                "important_files": [
                    "index.html",
                    "about.html",
                    "projects.html",
                    "contact.html",
                    "styles.css",
                    "tests/test_site.py",
                ],
                "focus_files": [
                    "index.html",
                    "about.html",
                    "projects.html",
                    "contact.html",
                    "styles.css",
                ],
                "test_files": ["tests/test_site.py"],
                "likely_commands": ["python -m unittest tests.test_site"],
            }
        ),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_site",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_site"],
    )
    payload = route_payload(
        intent="create",
        action_plan=[
            {"step": 1, "action": "create_artifact", "reason": "Create the requested website files."},
            {"step": 2, "action": "run_validation", "reason": "Run the targeted unittest module."},
        ],
        target_paths=["index.html", "about.html", "projects.html", "contact.html", "styles.css"],
        target_name="index.html",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_site and finish only when it passes.",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="index.html", operation="create"),
            FileChangeRecord(path="about.html", operation="create"),
            FileChangeRecord(path="projects.html", operation="create"),
            FileChangeRecord(path="contact.html", operation="create"),
            FileChangeRecord(path="styles.css", operation="create"),
        ]
    )

    failed_run = ValidationRunRecord(
        command="python -m unittest tests.test_site",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=1,
        iteration=5,
        summary="Validation command exited with 1.",
        excerpt=(
            "FAIL: test_styles_include_variables_and_responsive_rule "
            "(tests.test_site.TestSite.test_styles_include_variables_and_responsive_rule)\n"
            "Traceback (most recent call last):\n"
            '  File "/tmp/tests/test_site.py", line 41, in test_styles_include_variables_and_responsive_rule\n'
            "    self.assertIn(':root', css)\n"
            "AssertionError: ':root' not found in 'body { color: #222; }\\n\\n"
            ".contact-form { display: flex; }\\n\\n"
            "@media (max-width: 600px) { nav { flex-direction: column; } }\\n'\n"
        ),
    )
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_site",
        verification_scope="runtime",
        status="failed",
        artifact_paths=[
            "index.html",
            "about.html",
            "projects.html",
            "contact.html",
            "styles.css",
            "tests/test_site.py",
        ],
        summary="Validation command exited with 1.",
        excerpt=failed_run.excerpt,
        failure_summary="The stylesheet validation still fails because the responsive CSS contract is incomplete.",
        expected_features=[],
        missing_features=[],
        file_hints=[
            "index.html",
            "about.html",
            "projects.html",
            "contact.html",
            "styles.css",
            "tests/test_site.py",
        ],
        line_hints=[41],
        action_hints=[
            "Keep the multi-page site structure intact and repair only the failing stylesheet behavior."
        ],
        repair_requirements=[
            "Change index.html so the failing runtime or test path can complete successfully.",
            "Do not stop at an equivalent or formatting-only rewrite.",
        ],
        evidence_signature="sig-website-css-repair",
    )

    next_target = planner._repair_target_after_failed_validation(
        session.router_result,
        session,
        failed_run,
        repair_context,
    )

    assert next_target == "styles.css"


def test_planner_reads_related_test_before_creating_missing_runtime_fixture(tmp_path):
    (tmp_path / "wordfreq.py").write_text(
        "def count_words(file_path):\n"
        "    with open(file_path, 'r') as file:\n"
        "        return file.read()\n",
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_wordfreq.py"
    test_file.write_text(
        "import unittest\n"
        "from wordfreq import count_words\n\n"
        "class TestWordFreq(unittest.TestCase):\n"
        "    def test_count_words(self):\n"
        "        expected = {'hello': 2, 'world': 1}\n"
        "        self.assertEqual(count_words('tests/test_data.txt'), expected)\n",
        encoding="utf-8",
    )

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Create a small Python CLI in wordfreq.py and make the unittest pass.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 2,
                "important_files": ["wordfreq.py", "tests/test_wordfreq.py"],
                "focus_files": ["tests/test_wordfreq.py", "wordfreq.py"],
                "test_files": ["tests/test_wordfreq.py"],
                "likely_commands": ["python -m unittest tests.test_wordfreq"],
            }
        ),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_wordfreq",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_wordfreq"],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failed validation."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["tests/test_data.txt"],
        target_name="test_data.txt",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_wordfreq and finish only when it passes.",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="wordfreq.py", operation="create"),
            FileChangeRecord(path="tests/test_wordfreq.py", operation="create"),
        ]
    )
    session.active_repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_wordfreq",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["wordfreq.py", "tests/test_wordfreq.py", "tests/test_data.txt"],
        summary="Validation command exited with 1.",
        excerpt=(
            "Traceback (most recent call last):\n"
            f'  File "{test_file}", line 7, in test_count_words\n'
            "    self.assertEqual(count_words('tests/test_data.txt'), expected)\n"
            f'  File "{tmp_path / "wordfreq.py"}", line 2, in count_words\n'
            "    with open(file_path, 'r') as file:\n"
            "FileNotFoundError: [Errno 2] No such file or directory: 'tests/test_data.txt'\n"
        ),
        failure_summary="count_words failed because tests/test_data.txt is missing.",
        expected_features=[],
        missing_features=[],
        file_hints=["wordfreq.py", "tests/test_wordfreq.py", "tests/test_data.txt"],
        line_hints=[7, 2],
        action_hints=[],
        repair_requirements=[],
        evidence_signature="sig-wordfreq-missing-fixture-bootstrap",
    )

    decision = planner._draft_update_decision(session.router_result, session, "tests/test_data.txt")

    assert decision is not None
    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "read_file"
    assert decision.tool_args["path"] == "tests/test_wordfreq.py"


def test_planner_keeps_runtime_repair_on_last_implicated_implementation_target(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    (pkg / "__main__.py").write_text(
        "import argparse\nfrom .cli import greet\n\n"
        "def main(name=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name', nargs='?', default='world')\n"
        "    args = parser.parse_args()\n"
        "    print(greet(args.name))\n",
        encoding="utf-8",
    )
    (pkg / "cli.py").write_text("def greet(name):\n    return f'Hello, {name}!'\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_cli.py").write_text(
        "import unittest\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    def test_greet_with_name(self):\n"
        "        from greet_cli import __main__\n"
        "        __main__.main(['Ada'])\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# docs\n", encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 4,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py", "README.md"],
                "focus_files": ["tests/test_cli.py", "greet_cli/__main__.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_cli",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_cli"],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failing CLI implementation."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["tests/test_cli.py"],
        target_name="test_cli.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli and fix the failing CLI behavior.",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="greet_cli/__main__.py", operation="write"),
            FileChangeRecord(path="greet_cli/cli.py", operation="write"),
            FileChangeRecord(path="README.md", operation="write"),
            FileChangeRecord(path="tests/test_cli.py", operation="write"),
        ]
    )
    session.repair_history.append(
        RepairAttemptRecord(
            artifact_path="greet_cli/__main__.py",
            validation_command="python -m unittest tests.test_cli",
            verification_scope="runtime",
            strategy="validation_targeted",
            result="mutation_planned",
            reason="updated entrypoint signature",
            evidence_signature="older-runtime-evidence",
            iteration=9,
        )
    )
    failed_run = ValidationRunRecord(
        command="python -m unittest tests.test_cli",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=1,
        iteration=10,
        summary="Validation command exited with 1.",
        excerpt=(
            "FF\n======================================================================\n"
            "FAIL: test_greet_with_name (tests.test_cli.TestCLI.test_greet_with_name)\n"
            "Traceback (most recent call last):\n"
            '  File "/tmp/tests/test_cli.py", line 10, in test_greet_with_name\n'
            "    __main__.main(['Ada'])\n"
            '  File "/tmp/greet_cli/__main__.py", line 7, in main\n'
            "    args = parser.parse_args()\n"
            "AssertionError: 'Hello, tests.test_cli!' != 'Hello, Ada!'\n"
        ),
    )
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/cli.py", "tests/test_cli.py", "greet_cli/__main__.py", "README.md"],
        summary="Validation command exited with 1.",
        excerpt=(
            "FF\n======================================================================\n"
            "FAIL: test_greet_with_name (tests.test_cli.TestCLI.test_greet_with_name)\n"
            "Traceback (most recent call last):\n"
            '  File "/tmp/tests/test_cli.py", line 10, in test_greet_with_name\n'
            "    __main__.main(['Ada'])\n"
            '  File "/tmp/greet_cli/__main__.py", line 7, in main\n'
            "    args = parser.parse_args()\n"
            "AssertionError: 'Hello, tests.test_cli!' != 'Hello, Ada!'\n"
        ),
        failure_summary="FAIL: test_greet_with_name still fails after the first runtime repair.",
        expected_features=[],
        missing_features=[],
        file_hints=["greet_cli/cli.py", "tests/test_cli.py", "greet_cli/__main__.py", "README.md"],
        line_hints=[10, 7],
        action_hints=[
            "Read the failing test output and inspect the hinted source files before editing.",
            "Prefer a targeted fix over broad refactors, then rerun the same test command.",
        ],
        repair_requirements=[
            "Change tests/test_cli.py so the failing runtime or test path can complete successfully.",
            "Do not stop at an equivalent or formatting-only rewrite.",
        ],
        evidence_signature="new-runtime-evidence",
    )

    next_target = planner._repair_target_after_failed_validation(
        session.router_result,
        session,
        failed_run,
        repair_context,
    )

    assert next_target == "greet_cli/__main__.py"


def test_planner_keeps_runtime_repair_on_last_mutated_implementation_target_when_new_failure_only_mentions_tests(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    (pkg / "__main__.py").write_text(
        "import argparse\nfrom .cli import greet\n\n"
        "def main(name=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name', nargs='?', default='world')\n"
        "    args = parser.parse_args([name] if name else [])\n"
        "    print(greet(args.name))\n",
        encoding="utf-8",
    )
    (pkg / "cli.py").write_text("def greet(name):\n    return f'Hello, {name}!'\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_cli.py").write_text(
        "import unittest\nfrom unittest.mock import patch\nfrom io import StringIO\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    @patch('sys.stdout', new_callable=StringIO)\n"
        "    def test_greet_with_name(self, mock_stdout):\n"
        "        from greet_cli import __main__\n"
        "        with patch('__main__.sys.argv', ['python', '-m', 'greet_cli', 'Ada']):\n"
        "            __main__.main()\n"
        "        self.assertEqual(mock_stdout.getvalue().strip(), 'Hello, Ada!')\n",
        encoding="utf-8",
    )

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 3,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "focus_files": ["tests/test_cli.py", "greet_cli/__main__.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
        validation_status="failed",
        edit_generation=2,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_cli",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_cli"],
    )
    payload = route_payload(
        intent="debug",
        action_plan=[
            {"step": 1, "action": "read_relevant_files", "reason": "Inspect the failing runtime path."},
            {"step": 2, "action": "update_artifact", "reason": "Repair the implicated CLI implementation."},
            {"step": 3, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["tests/test_cli.py"],
        target_name="test_cli.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli and fix the failing CLI behavior.",
    )
    session.repair_history.append(
        RepairAttemptRecord(
            artifact_path="greet_cli/__main__.py",
            validation_command="python -m unittest tests.test_cli",
            verification_scope="runtime",
            strategy="validation_targeted",
            result="mutation_planned",
            reason="updated entrypoint argument handling",
            evidence_signature="older-runtime-evidence",
            iteration=6,
        )
    )

    failed_run = ValidationRunRecord(
        command="python -m unittest tests.test_cli",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=2,
        iteration=7,
        summary="Validation command exited with 1.",
        excerpt=(
            "FAIL: test_greet_with_name (tests.test_cli.TestCLI.test_greet_with_name)\n"
            "Traceback (most recent call last):\n"
            '  File "/tmp/tests/test_cli.py", line 8, in test_greet_with_name\n'
            "    self.assertEqual(mock_stdout.getvalue().strip(), 'Hello, Ada!')\n"
            "AssertionError: 'Hello, world!' != 'Hello, Ada!'\n"
        ),
    )
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["tests/test_cli.py"],
        summary="Validation command exited with 1.",
        excerpt=failed_run.excerpt,
        failure_summary="The named greeting still prints the default name after the first runtime repair.",
        expected_features=[],
        missing_features=[],
        file_hints=["tests/test_cli.py"],
        line_hints=[8],
        action_hints=[
            "Read the failing assertion and keep the repair focused on the runtime behavior that still produces the wrong greeting."
        ],
        repair_requirements=[
            "Preserve the explicit-name and default-name cases while fixing the runtime behavior."
        ],
        evidence_signature="assertion-only-runtime",
    )

    next_target = planner._repair_target_after_failed_validation(
        session.router_result,
        session,
        failed_run,
        repair_context,
    )

    assert next_target == "greet_cli/__main__.py"


def test_planner_uses_traceback_workspace_path_for_runtime_repair_target(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Extend the CLI with an uppercase flag and keep the package behavior coherent.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    (tmp_path / "greet_cli").mkdir()
    (tmp_path / "greet_cli" / "__main__.py").write_text(
        "from .cli import greet\n\n"
        "def main(argv=None):\n"
        "    return greet('Ada')\n",
        encoding="utf-8",
    )
    (tmp_path / "greet_cli" / "cli.py").write_text(
        "def greet(name):\n    return f\"Hello, {name}!\"\n",
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_cli.py").write_text("pass\n", encoding="utf-8")
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Apply the requested feature update."},
        ],
        target_paths=["greet_cli/cli.py", "tests/test_cli.py"],
        target_name="greet_cli/cli.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="python -m unittest tests.test_cli",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="greet_cli/cli.py", operation="modify"),
            FileChangeRecord(path="tests/test_cli.py", operation="modify"),
        ]
    )

    failed_run = ValidationRunRecord(
        command="python -m unittest tests.test_cli",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=2,
        summary="Validation command exited with 1.",
        excerpt=(
            ".usage: python3 -m unittest [-h] [name]\n"
            "python3 -m unittest: error: unrecognized arguments: --uppercase\n"
            "Traceback (most recent call last):\n"
            f'  File "{tmp_path / "tests" / "test_cli.py"}", line 22, in test_greet_with_uppercase_flag\n'
            "    __main__.main(['--uppercase', 'Ada'])\n"
            f'  File "{tmp_path / "greet_cli" / "__main__.py"}", line 7, in main\n'
            "    args = parser.parse_args(argv)\n"
            "SystemExit: 2\n"
        ),
    )

    repair_context = planner.validation_planner.build_failure_evidence(session, failed_run)
    next_target = planner._repair_target_after_failed_validation(
        session.router_result,
        session,
        failed_run,
        repair_context,
    )

    assert repair_context.artifact_paths[0] == "greet_cli/__main__.py"
    assert next_target == "greet_cli/__main__.py"


def test_planner_keeps_attempted_runtime_implementation_ahead_of_unattempted_test_target(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root=str(tmp_path),
    )
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["tests/test_cli.py", "greet_cli/__main__.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "FAIL: test_greet_with_name (tests.test_cli.TestCLI.test_greet_with_name)\n"
            "Traceback (most recent call last):\n"
            '  File "/tmp/tests/test_cli.py", line 11, in test_greet_with_name\n'
            "    __main__.main()\n"
            '  File "/tmp/greet_cli/__main__.py", line 7, in main\n'
            "    args = parser.parse_args()\n"
            "AssertionError: 'Hello, world!' != 'Hello, Ada!'\n"
        ),
        failure_summary="The CLI still prints the default greeting for the named case.",
        expected_features=[],
        missing_features=[],
        file_hints=["tests/test_cli.py", "greet_cli/__main__.py"],
        line_hints=[11, 7],
        action_hints=[],
        repair_requirements=[],
        evidence_signature="sig-runtime-ordering",
    )
    session.repair_history.append(
        RepairAttemptRecord(
            artifact_path="greet_cli/__main__.py",
            validation_command="python -m unittest tests.test_cli",
            verification_scope="runtime",
            strategy="validation_targeted",
            result="mutation_planned",
            reason="updated the CLI implementation once already",
            evidence_signature="sig-runtime-ordering",
            iteration=4,
        )
    )

    ordered = planner._repair_candidates_with_unattempted_first(
        session,
        repair_context,
        ["greet_cli/__main__.py", "tests/test_cli.py"],
    )

    assert ordered == ["greet_cli/__main__.py", "tests/test_cli.py"]


def test_planner_retries_same_implementation_target_after_identical_repair_generation(tmp_path, monkeypatch):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    main = pkg / "__main__.py"
    current_main = (
        "from .cli import greet\n\n"
        "def main():\n"
        "    print(greet('Ada'))\n"
    )
    main.write_text(current_main, encoding="utf-8")
    cli = pkg / "cli.py"
    cli.write_text("def greet(name):\n    return f'Hello, {name}!'\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_cli = tests_dir / "test_cli.py"
    test_cli.write_text(
        "import unittest\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    def test_greet_with_name(self):\n"
        "        from greet_cli import __main__\n"
        "        __main__.main(['Ada'])\n",
        encoding="utf-8",
    )

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 4,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "focus_files": ["tests/test_cli.py", "greet_cli/__main__.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_cli",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_cli"],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failing CLI implementation."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["tests/test_cli.py"],
        target_name="test_cli.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli and fix the failing CLI behavior.",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="greet_cli/__main__.py", operation="write"),
            FileChangeRecord(path="greet_cli/cli.py", operation="write"),
            FileChangeRecord(path="tests/test_cli.py", operation="write"),
        ]
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_cli",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=1,
            iteration=8,
            summary="Validation command exited with 1.",
            excerpt=(
                "Traceback (most recent call last):\n"
                '  File "/tmp/tests/test_cli.py", line 5, in test_greet_with_name\n'
                "    __main__.main(['Ada'])\n"
                "TypeError: main() takes 0 positional arguments but 1 was given\n"
            ),
        )
    )

    failed_run = session.validation_runs[-1]
    repair_context = planner.validation_planner.build_failure_evidence(session, failed_run)
    session.active_repair_context = repair_context
    session.repair_history.append(
        RepairAttemptRecord(
            artifact_path="tests/test_cli.py",
            validation_command=repair_context.command,
            verification_scope=repair_context.verification_scope,
            strategy="validation_targeted",
            result="no_effective_change",
            reason="identical content",
            evidence_signature=repair_context.evidence_signature,
            iteration=9,
        )
    )
    repair_route = planner._repair_route_after_failed_validation(
        session.router_result,
        session,
        failed_run,
        repair_context,
    )

    assert repair_route is not None

    seen_strategies: list[str] = []

    def fake_generate_file_content(*_args, **kwargs):
        seen_strategies.append(str(kwargs["repair_strategy"]))
        if kwargs["repair_strategy"] == "validation_targeted":
            return SimpleNamespace(content=kwargs["current_content"], failure=None)
        return SimpleNamespace(content=kwargs["current_content"] + "\n# fix: accept argv\n", failure=None)

    monkeypatch.setattr(planner, "_generate_file_content", fake_generate_file_content)

    decision = planner._draft_update_decision(repair_route, session, "greet_cli/__main__.py")

    assert seen_strategies == ["validation_targeted", "validation_escalated"]
    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "write_file"
    assert decision.tool_args["path"] == "greet_cli/__main__.py"


def test_planner_switches_to_remaining_repair_target_after_generation_failure(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    main = pkg / "__main__.py"
    main.write_text(
        "from .cli import greet\n\n"
        "def main():\n"
        "    print(greet('Ada'))\n",
        encoding="utf-8",
    )
    cli = pkg / "cli.py"
    cli.write_text("def greet(name):\n    return f'Hello, {name}!'\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_cli = tests_dir / "test_cli.py"
    current = (
        "import unittest\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    def test_greet_with_name(self):\n"
        "        from greet_cli import __main__\n"
        "        __main__.main(['Ada'])\n"
    )
    test_cli.write_text(current, encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 4,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "focus_files": ["tests/test_cli.py", "greet_cli/__main__.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_cli",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_cli"],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failing test module."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["tests/test_cli.py"],
        target_name="test_cli.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli and fix the failing CLI behavior.",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="greet_cli/__main__.py", operation="write"),
            FileChangeRecord(path="greet_cli/cli.py", operation="write"),
            FileChangeRecord(path="tests/test_cli.py", operation="write"),
        ]
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_cli",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=1,
            iteration=8,
            summary="Validation command exited with 1.",
            excerpt=(
                "Traceback (most recent call last):\n"
                '  File "/tmp/tests/test_cli.py", line 5, in test_greet_with_name\n'
                "    __main__.main(['Ada'])\n"
                "TypeError: main() takes 0 positional arguments but 1 was given\n"
            ),
        )
    )
    session.tool_calls.append(
        ToolCallRecord(
            iteration=9,
            tool_name="read_file",
            tool_args={"path": "tests/test_cli.py"},
            success=True,
            summary="Read tests/test_cli.py.",
            output_excerpt=current,
        )
    )

    failed_run = session.validation_runs[-1]
    repair_context = planner.validation_planner.build_failure_evidence(session, failed_run)
    session.active_repair_context = repair_context
    session.repair_history.extend(
        [
            RepairAttemptRecord(
                artifact_path="tests/test_cli.py",
                validation_command=repair_context.command,
                verification_scope=repair_context.verification_scope,
                strategy="validation_targeted",
                result="no_effective_change",
                reason="identical content",
                evidence_signature=repair_context.evidence_signature,
                iteration=9,
            ),
            RepairAttemptRecord(
                artifact_path="greet_cli/cli.py",
                validation_command=repair_context.command,
                verification_scope=repair_context.verification_scope,
                strategy="validation_targeted",
                result="generation_failed",
                reason="model_start_failed",
                evidence_signature=repair_context.evidence_signature,
                iteration=9,
            ),
            RepairAttemptRecord(
                artifact_path="greet_cli/cli.py",
                validation_command=repair_context.command,
                verification_scope=repair_context.verification_scope,
                strategy="validation_escalated",
                result="generation_failed",
                reason="model_start_failed",
                evidence_signature=repair_context.evidence_signature,
                iteration=9,
            ),
        ]
    )

    decision = planner._alternative_repair_target_decision(
        session.router_result,
        session,
        repair_context,
        current_target="greet_cli/cli.py",
    )

    assert decision is not None
    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "read_file"
    assert decision.tool_args["path"] == "greet_cli/__main__.py"


def test_compact_runtime_repair_retry_prompt_includes_neighbor_implementation_context(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    (pkg / "__main__.py").write_text(
        "from .cli import greet\n\n"
        "def main(argv=None):\n"
        "    print(greet('Ada'))\n",
        encoding="utf-8",
    )
    (pkg / "cli.py").write_text(
        "def greet(name: str, prefix: str = '', repeat: int = 1, uppercase: bool = False) -> str:\n"
        "    greeting = f'{prefix}{name}'\n"
        "    if uppercase:\n"
        "        greeting = greeting.upper()\n"
        "    return greeting * repeat\n",
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_cli.py").write_text(
        "import unittest\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    def test_default_greeting(self):\n"
        "        self.assertEqual('Ada', 'Hello, Ada!')\n",
        encoding="utf-8",
    )

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Extend this existing CLI with prefix and repeat support while keeping uppercase working.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 4,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "focus_files": ["tests/test_cli.py", "greet_cli/__main__.py", "greet_cli/cli.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failing CLI behavior."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["greet_cli/__main__.py", "greet_cli/cli.py", "README.md", "tests/test_cli.py"],
        target_name="greet_cli/__main__.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli and fix the failing CLI behavior.",
    )
    session.tool_calls.extend(
        [
            ToolCallRecord(
                iteration=1,
                tool_name="read_file",
                tool_args={"path": "tests/test_cli.py"},
                success=True,
                summary="Read tests/test_cli.py.",
                output_excerpt=(tests_dir / "test_cli.py").read_text(encoding="utf-8"),
            ),
            ToolCallRecord(
                iteration=2,
                tool_name="read_file",
                tool_args={"path": "greet_cli/__main__.py"},
                success=True,
                summary="Read greet_cli/__main__.py.",
                output_excerpt=(pkg / "__main__.py").read_text(encoding="utf-8"),
            ),
        ]
    )
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/cli.py", "tests/test_cli.py", "greet_cli/__main__.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "FAIL: test_default_greeting (tests.test_cli.TestCLI.test_default_greeting)\n"
            "AssertionError: 'Ada' != 'Hello, Ada!'\n"
        ),
        failure_summary="test_default_greeting expects Hello, Ada! but the CLI only prints Ada.",
        file_hints=["tests/test_cli.py", "greet_cli/__main__.py"],
        line_hints=[4],
        action_hints=["Inspect the failing expectation and the runtime path before editing."],
        repair_requirements=["Repair greet_cli/cli.py so the failing runtime validation passes."],
        evidence_signature="runtime-neighbor-context",
    )
    review_feedback = ProposedUpdateReview(
        safe_to_write=False,
        summary="The proposed update moves package entrypoint logic into a helper module.",
        confidence=0.96,
        blocking_issues=[
            "The proposal adds argparse and/or __main__ launcher logic to greet_cli/cli.py even though the package already has a sibling entrypoint at greet_cli/__main__.py."
        ],
        preservation_risks=[],
        repair_hints=[
            "Keep greet_cli/cli.py focused on reusable helper behavior and place CLI parsing or launcher handling in greet_cli/__main__.py."
        ],
    )

    prompt = generate_content_retry_prompt(
        session.router_result,
        session,
        path="greet_cli/cli.py",
        current_content=(pkg / "cli.py").read_text(encoding="utf-8"),
        repair_context=repair_context,
        repair_strategy="validation_targeted",
        review_feedback=review_feedback,
        mode="compact",
    )

    assert "Supporting file hints: tests/test_cli.py:" in prompt
    assert "greet_cli/__main__.py:" in prompt


def test_compact_runtime_repair_retry_prompt_prefers_current_workspace_support_content(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    current_main = (
        "from __future__ import annotations\n\n"
        "import argparse\n\n"
        "from .cli import greet\n\n\n"
        "def main(argv=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name')\n"
        "    parser.add_argument('--prefix', type=str)\n"
        "    parser.add_argument('--repeat', type=int, default=1)\n"
        "    parser.add_argument('--uppercase', action='store_true')\n"
        "    args = parser.parse_args(argv)\n\n"
        "    greeting = greet(args.name)\n"
        "    if args.prefix:\n"
        "        greeting = f\"{args.prefix} {greeting}\"\n"
        "    if args.uppercase:\n"
        "        greeting = greeting.upper()\n\n"
        "    for _ in range(args.repeat):\n"
        "        print(greeting)\n"
    )
    stale_main = (
        "from __future__ import annotations\n\n"
        "import argparse\n\n"
        "from .cli import greet\n\n\n"
        "def main(argv=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name')\n"
        "    args = parser.parse_args(argv)\n"
        "    print(greet(args.name))\n"
    )
    (pkg / "__main__.py").write_text(current_main, encoding="utf-8")
    current_cli = (
        "def greet(name: str, prefix: str = '', repeat: int = 1, uppercase: bool = False) -> str:\n"
        "    greeting = f'{prefix}{name}'\n"
        "    if uppercase:\n"
        "        greeting = greeting.upper()\n"
        "    return greeting * repeat\n"
    )
    (pkg / "cli.py").write_text(current_cli, encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_content = (
        "import io\n"
        "import unittest\n"
        "from unittest.mock import patch\n\n"
        "from greet_cli import __main__\n\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    def run_cli(self, argv):\n"
        "        with patch('sys.stdout', new_callable=io.StringIO) as stdout:\n"
        "            __main__.main(argv)\n"
        "            return stdout.getvalue().strip()\n\n"
        "    def test_default_greeting(self):\n"
        "        self.assertEqual(self.run_cli(['Ada']), 'Hello, Ada!')\n"
    )
    (tests_dir / "test_cli.py").write_text(test_content, encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Extend this existing CLI so it supports --prefix and --repeat while keeping uppercase working.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 4,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "focus_files": ["tests/test_cli.py", "greet_cli/__main__.py", "greet_cli/cli.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failing CLI behavior."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
        target_name="greet_cli/cli.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli and fix the failing CLI behavior.",
    )
    session.tool_calls.extend(
        [
            ToolCallRecord(
                iteration=1,
                tool_name="read_file",
                tool_args={"path": "tests/test_cli.py"},
                success=True,
                summary="Read tests/test_cli.py.",
                output_excerpt=test_content,
            ),
            ToolCallRecord(
                iteration=2,
                tool_name="read_file",
                tool_args={"path": "greet_cli/__main__.py"},
                success=True,
                summary="Read greet_cli/__main__.py.",
                output_excerpt=stale_main,
            ),
        ]
    )
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/cli.py", "tests/test_cli.py", "greet_cli/__main__.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "FAIL: test_default_greeting (tests.test_cli.TestCLI.test_default_greeting)\n"
            "AssertionError: 'Ada' != 'Hello, Ada!'\n"
        ),
        failure_summary="test_default_greeting expects Hello, Ada! but the CLI only prints Ada.",
        file_hints=["tests/test_cli.py", "greet_cli/__main__.py"],
        line_hints=[12],
        action_hints=["Inspect the failing expectation and the runtime path before editing."],
        repair_requirements=["Repair greet_cli/cli.py so the failed runtime validation passes."],
        evidence_signature="runtime-current-support-context",
    )
    review_feedback = ProposedUpdateReview(
        safe_to_write=False,
        summary="The proposed update keeps the helper wrong and misses the active CLI wiring.",
        confidence=0.91,
        blocking_issues=[
            "The helper update does not repair the cross-file runtime behavior expected by the failing tests."
        ],
        preservation_risks=[],
        repair_hints=[
            "Use the current runtime neighbor files as the contract for what the helper should return."
        ],
    )

    prompt = generate_content_retry_prompt(
        session.router_result,
        session,
        path="greet_cli/cli.py",
        current_content=current_cli,
        repair_context=repair_context,
        repair_strategy="validation_targeted",
        review_feedback=review_feedback,
        mode="compact",
    )

    assert "greet_cli/__main__.py:" in prompt
    assert "parser.add_argument('--prefix', type=str)" in prompt
    assert "print(greet(args.name))" not in prompt


def test_compact_update_prompt_prefers_current_workspace_related_file_content(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    stale_main = (
        "from __future__ import annotations\n\n"
        "import argparse\n\n"
        "from .cli import greet\n\n\n"
        "def main(argv=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name')\n"
        "    args = parser.parse_args(argv)\n"
        "    print(greet(args.name))\n"
    )
    current_main = (
        "from __future__ import annotations\n\n"
        "import argparse\n\n"
        "from .cli import greet\n\n\n"
        "def main(argv=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name')\n"
        "    parser.add_argument('--prefix', type=str)\n"
        "    parser.add_argument('--repeat', type=int, default=1)\n"
        "    parser.add_argument('--uppercase', action='store_true')\n"
        "    args = parser.parse_args(argv)\n\n"
        "    greeting = greet(args.name)\n"
        "    if args.prefix:\n"
        "        greeting = f\"{args.prefix} {greeting}\"\n"
        "    if args.uppercase:\n"
        "        greeting = greeting.upper()\n\n"
        "    for _ in range(args.repeat):\n"
        "        print(greeting)\n"
    )
    (pkg / "__main__.py").write_text(current_main, encoding="utf-8")
    current_cli = "def greet(name: str) -> str:\n    return f\"Hello, {name}!\"\n"
    (pkg / "cli.py").write_text(current_cli, encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("# greet_cli\n", encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Extend this existing CLI so it supports --prefix and --repeat while keeping uppercase working.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 3,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "README.md"],
                "focus_files": ["greet_cli/__main__.py", "greet_cli/cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Extend the existing CLI behavior."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the relevant tests."},
        ],
        target_paths=["greet_cli/__main__.py", "greet_cli/cli.py", "README.md", "tests/test_cli.py"],
        target_name="greet_cli/cli.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli after updating the CLI.",
    )
    session.tool_calls.extend(
        [
            ToolCallRecord(
                iteration=1,
                tool_name="read_file",
                tool_args={"path": "greet_cli/__main__.py"},
                success=True,
                summary="Read greet_cli/__main__.py.",
                output_excerpt=stale_main,
            ),
            ToolCallRecord(
                iteration=2,
                tool_name="write_file",
                tool_args={"path": "greet_cli/__main__.py", "content": current_main},
                success=True,
                summary="Wrote greet_cli/__main__.py.",
                output_excerpt=current_main,
            ),
            ToolCallRecord(
                iteration=3,
                tool_name="read_file",
                tool_args={"path": "README.md"},
                success=True,
                summary="Read README.md.",
                output_excerpt=readme.read_text(encoding="utf-8"),
            ),
        ]
    )

    prompt = generate_content_prompt(
        session.router_result,
        session,
        path="greet_cli/cli.py",
        current_content=current_cli,
        mode="compact",
    )

    assert "greet_cli/__main__.py:" in prompt
    assert "parser.add_argument('--prefix', type=str)" in prompt
    assert "print(greet(args.name))" not in prompt


def test_compact_update_prompt_prefers_runtime_tests_over_readme_context(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    current_main = (
        "from __future__ import annotations\n\n"
        "import argparse\n\n"
        "from .cli import greet\n\n\n"
        "def main(argv=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name')\n"
        "    parser.add_argument('--prefix', type=str)\n"
        "    parser.add_argument('--repeat', type=int, default=1)\n"
        "    parser.add_argument('--uppercase', action='store_true')\n"
        "    args = parser.parse_args(argv)\n\n"
        "    greeting = greet(args.name)\n"
        "    if args.prefix:\n"
        "        greeting = f\"{args.prefix} {greeting}\"\n"
        "    if args.uppercase:\n"
        "        greeting = greeting.upper()\n\n"
        "    for _ in range(args.repeat):\n"
        "        print(greeting)\n"
    )
    (pkg / "__main__.py").write_text(current_main, encoding="utf-8")
    current_cli = "def greet(name: str) -> str:\n    return f\"Hello, {name}!\"\n"
    (pkg / "cli.py").write_text(current_cli, encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_content = (
        "import io\n"
        "import unittest\n"
        "from unittest.mock import patch\n\n"
        "from greet_cli import __main__\n\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    def run_cli(self, argv):\n"
        "        with patch('sys.stdout', new_callable=io.StringIO) as stdout:\n"
        "            __main__.main(argv)\n"
        "            return stdout.getvalue().strip()\n\n"
        "    def test_prefix_flag(self):\n"
        "        self.assertEqual(self.run_cli(['--prefix', 'Mr.', 'Ada']), 'Hello, Mr. Ada!')\n"
    )
    (tests_dir / "test_cli.py").write_text(test_content, encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(
        "# greet_cli\n\nRun the package with:\n\n```sh\npython -m greet_cli Ada\n```\n",
        encoding="utf-8",
    )

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Extend this existing CLI so it supports --prefix and --repeat while keeping uppercase working.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 4,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py", "README.md"],
                "focus_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Extend the existing CLI behavior."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the relevant tests."},
        ],
        target_paths=["greet_cli/__main__.py", "greet_cli/cli.py", "README.md", "tests/test_cli.py"],
        target_name="greet_cli/cli.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli after updating the CLI.",
    )
    session.tool_calls.extend(
        [
            ToolCallRecord(
                iteration=1,
                tool_name="read_file",
                tool_args={"path": "greet_cli/__main__.py"},
                success=True,
                summary="Read greet_cli/__main__.py.",
                output_excerpt=current_main,
            ),
            ToolCallRecord(
                iteration=2,
                tool_name="read_file",
                tool_args={"path": "README.md"},
                success=True,
                summary="Read README.md.",
                output_excerpt=readme.read_text(encoding="utf-8"),
            ),
            ToolCallRecord(
                iteration=3,
                tool_name="read_file",
                tool_args={"path": "tests/test_cli.py"},
                success=True,
                summary="Read tests/test_cli.py.",
                output_excerpt=test_content,
            ),
        ]
    )

    prompt = generate_content_prompt(
        session.router_result,
        session,
        path="greet_cli/cli.py",
        current_content=current_cli,
        mode="compact",
    )

    assert "greet_cli/__main__.py:" in prompt
    assert "tests/test_cli.py:" in prompt
    assert "Hello, Mr. Ada!" in prompt
    assert "README.md:" not in prompt


def test_compact_runtime_repair_supporting_test_excerpt_keeps_assertion_line(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    (pkg / "__main__.py").write_text(
        "from .cli import greet\n\n"
        "def main(argv=None):\n"
        "    print(greet('Ada'))\n",
        encoding="utf-8",
    )
    (pkg / "cli.py").write_text(
        "def greet(name: str, prefix: str = '', repeat: int = 1, uppercase: bool = False) -> str:\n"
        "    greeting = f'{prefix}{name}'\n"
        "    if uppercase:\n"
        "        greeting = greeting.upper()\n"
        "    return greeting * repeat\n",
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_content = (
        "import io\n"
        "import unittest\n"
        "from unittest.mock import patch\n\n"
        "from greet_cli import __main__\n\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    def run_cli(self, argv):\n"
        "        with patch('sys.stdout', new_callable=io.StringIO) as stdout:\n"
        "            __main__.main(argv)\n"
        "            return stdout.getvalue().strip()\n\n"
        "    def test_default_greeting(self):\n"
        "        self.assertEqual(self.run_cli(['Ada']), 'Hello, Ada!')\n"
    )
    (tests_dir / "test_cli.py").write_text(test_content, encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Extend this existing CLI so it supports --prefix and --repeat while keeping uppercase working.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 4,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "focus_files": ["tests/test_cli.py", "greet_cli/__main__.py", "greet_cli/cli.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failing CLI behavior."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
        target_name="greet_cli/cli.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli and fix the failing CLI behavior.",
    )
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/cli.py", "tests/test_cli.py", "greet_cli/__main__.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "FAIL: test_default_greeting (tests.test_cli.TestCLI.test_default_greeting)\n"
            "AssertionError: 'Ada' != 'Hello, Ada!'\n"
            "self.assertEqual(self.run_cli(['Ada']), 'Hello, Ada!')\n"
        ),
        failure_summary="test_default_greeting expects Hello, Ada! but the CLI only prints Ada.",
        file_hints=["tests/test_cli.py", "greet_cli/__main__.py"],
        line_hints=[12],
        action_hints=["Inspect the failing expectation and the runtime path before editing."],
        repair_requirements=["Repair greet_cli/cli.py so the failed runtime validation passes."],
        evidence_signature="runtime-support-assertion-line",
    )
    review_feedback = ProposedUpdateReview(
        safe_to_write=False,
        summary="The helper still misses the expected greeting contract.",
        confidence=0.92,
        blocking_issues=["The updated helper still does not satisfy the failing assertion."],
        preservation_risks=[],
        repair_hints=["Preserve the existing greeting wrapper while adding the new flags."],
    )

    prompt = generate_content_retry_prompt(
        session.router_result,
        session,
        path="greet_cli/cli.py",
        current_content=(pkg / "cli.py").read_text(encoding="utf-8"),
        repair_context=repair_context,
        repair_strategy="validation_targeted",
        review_feedback=review_feedback,
        mode="compact",
    )

    assert "Supporting file hints: tests/test_cli.py:" in prompt
    assert "self.assertEqual(self.run_cli(['Ada']), 'Hello, Ada!')" in prompt


def test_planner_pivots_runtime_helper_repair_after_identical_change(tmp_path, monkeypatch):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    main = pkg / "__main__.py"
    main.write_text(
        "from .cli import greet\n\n"
        "def main(argv=None):\n"
        "    print(greet('Ada'))\n",
        encoding="utf-8",
    )
    cli = pkg / "cli.py"
    current_cli = (
        "def greet(name: str, prefix: str = '', repeat: int = 1, uppercase: bool = False) -> str:\n"
        "    greeting = f'{prefix}{name}'\n"
        "    if uppercase:\n"
        "        greeting = greeting.upper()\n"
        "    return greeting * repeat\n"
    )
    cli.write_text(current_cli, encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_cli.py").write_text(
        "import unittest\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    def test_default_greeting(self):\n"
        "        self.assertEqual('Ada', 'Hello, Ada!')\n",
        encoding="utf-8",
    )
    readme = tmp_path / "README.md"
    readme.write_text("# greet_cli\n", encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Extend this existing CLI so it supports --prefix and --repeat while keeping uppercase working.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 5,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py", "README.md"],
                "focus_files": ["tests/test_cli.py", "greet_cli/__main__.py", "greet_cli/cli.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_cli",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_cli"],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failing CLI behavior."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["greet_cli/__main__.py", "greet_cli/cli.py", "README.md", "tests/test_cli.py"],
        target_name="greet_cli/__main__.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli and fix the failing CLI behavior.",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="greet_cli/__main__.py", operation="write"),
            FileChangeRecord(path="greet_cli/cli.py", operation="write"),
            FileChangeRecord(path="README.md", operation="write"),
            FileChangeRecord(path="tests/test_cli.py", operation="write"),
        ]
    )
    session.tool_calls.extend(
        [
            ToolCallRecord(
                iteration=1,
                tool_name="read_file",
                tool_args={"path": "tests/test_cli.py"},
                success=True,
                summary="Read tests/test_cli.py.",
                output_excerpt=(tests_dir / "test_cli.py").read_text(encoding="utf-8"),
            ),
            ToolCallRecord(
                iteration=2,
                tool_name="read_file",
                tool_args={"path": "greet_cli/__main__.py"},
                success=True,
                summary="Read greet_cli/__main__.py.",
                output_excerpt=main.read_text(encoding="utf-8"),
            ),
        ]
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_cli",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=1,
            iteration=9,
            summary="Validation command exited with 1.",
            excerpt=(
                "FAIL: test_default_greeting (tests.test_cli.TestCLI.test_default_greeting)\n"
                "AssertionError: 'Ada' != 'Hello, Ada!'\n"
            ),
        )
    )
    failed_run = session.validation_runs[-1]
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/cli.py", "tests/test_cli.py", "greet_cli/__main__.py", "README.md"],
        summary="Validation command exited with 1.",
        excerpt=(
            "FAIL: test_default_greeting (tests.test_cli.TestCLI.test_default_greeting)\n"
            "AssertionError: 'Ada' != 'Hello, Ada!'\n"
        ),
        failure_summary="test_default_greeting expects Hello, Ada! but the CLI only prints Ada.",
        file_hints=["tests/test_cli.py", "greet_cli/__main__.py", "greet_cli/cli.py"],
        line_hints=[4],
        action_hints=[
            "Read the failing test output and inspect the hinted source files before editing.",
            "Prefer a targeted fix over broad refactors, then rerun the same test command.",
        ],
        repair_requirements=[
            "Repair greet_cli/cli.py so the failed runtime validation passes.",
            "Do not stop at an equivalent or formatting-only rewrite.",
        ],
        evidence_signature="runtime-cli-regression",
    )
    session.active_repair_context = repair_context

    repair_route = planner._repair_route_after_failed_validation(
        session.router_result,
        session,
        failed_run,
        repair_context,
    )
    assert repair_route is not None

    monkeypatch.setattr(
        planner,
        "_generate_file_content",
        lambda *_args, **kwargs: SimpleNamespace(content=current_cli, failure=None),
    )

    decision = planner._draft_update_decision(repair_route, session, "greet_cli/cli.py")

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name in {"read_file", "write_file"}
    assert decision.tool_args["path"] == "greet_cli/__main__.py"


def test_validation_repair_relevance_review_rejects_unrelated_runtime_fix(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["tests/test_cli.py", "greet_cli/__main__.py", "greet_cli/cli.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "Traceback (most recent call last):\n"
            "    __main__.main(['Ada'])\n"
            "TypeError: main() takes 0 positional arguments but 1 was given\n"
        ),
        failure_summary="TypeError: main() takes 0 positional arguments but 1 was given",
        expected_features=[],
        missing_features=[],
        file_hints=["tests/test_cli.py", "greet_cli/__main__.py", "greet_cli/cli.py"],
        line_hints=[10, 16],
        action_hints=[],
        repair_requirements=[],
        evidence_signature="sig",
    )

    review = planner._validation_repair_relevance_review(
        path="greet_cli/cli.py",
        current_content="def greet(name):\n    return f\"Hello, {name}!\"\n",
        proposed_content="import argparse\n\ndef greet(name):\n    return f\"Hello, {name}!\"\n",
        repair_context=repair_context,
    )

    assert review is not None
    assert review.safe_to_write is False
    assert "implicated" in review.summary.lower()
    assert any("main" in issue for issue in review.blocking_issues)


def test_validation_repair_relevance_review_prefers_target_frame_identifiers(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/__main__.py", "tests/test_cli.py", "greet_cli/cli.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "FAIL: test_greet_with_name (tests.test_cli.TestCLI.test_greet_with_name)\n"
            "Traceback (most recent call last):\n"
            '  File "/tmp/tests/test_cli.py", line 10, in test_greet_with_name\n'
            "    self.assertEqual(mock_stdout.getvalue().strip(), 'Hello, Ada!')\n"
            '  File "/tmp/greet_cli/__main__.py", line 7, in main\n'
            "    parser.parse_args()\n"
            "AssertionError: 'Hello, tests.test_cli!' != 'Hello, Ada!'\n"
        ),
        failure_summary="FAIL: test_greet_with_name still fails after the first runtime repair.",
        expected_features=[],
        missing_features=[],
        file_hints=["greet_cli/__main__.py", "tests/test_cli.py", "greet_cli/cli.py"],
        line_hints=[10, 7],
        action_hints=[],
        repair_requirements=[],
        evidence_signature="sig-targeted-runtime",
    )

    review = planner._validation_repair_relevance_review(
        path="greet_cli/__main__.py",
        current_content=(
            "import argparse\n\n"
            "def main():\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.parse_args()\n"
        ),
        proposed_content=(
            "import argparse\n\n"
            "def main(name=None):\n"
            "    if name is None:\n"
            "        parser = argparse.ArgumentParser()\n"
            "        parser.parse_args()\n"
        ),
        repair_context=repair_context,
    )

    assert review is None


def test_validation_repair_relevance_review_ignores_test_assertion_helpers_for_impl_targets(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/__main__.py", "tests/test_cli.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "FAIL: test_greet_with_name (tests.test_cli.TestCLI.test_greet_with_name)\n"
            "Traceback (most recent call last):\n"
            '  File "/tmp/tests/test_cli.py", line 12, in test_greet_with_name\n'
            "    self.assertEqual(mock_stdout.getvalue().strip(), 'Hello, Ada!')\n"
            "AssertionError: 'Hello, world!' != 'Hello, Ada!'\n"
        ),
        failure_summary="The CLI still prints the default greeting for the named case.",
        expected_features=[],
        missing_features=[],
        file_hints=["greet_cli/__main__.py", "tests/test_cli.py"],
        line_hints=[12],
        action_hints=[],
        repair_requirements=[],
        evidence_signature="sig-assertion-only-runtime",
    )

    review = planner._validation_repair_relevance_review(
        path="greet_cli/__main__.py",
        current_content=(
            "import argparse\nfrom .cli import greet\n\n"
            "def main(name=None):\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('name', nargs='?', default='world')\n"
            "    args = parser.parse_args([name] if name else [])\n"
            "    print(greet(args.name))\n"
        ),
        proposed_content=(
            "import argparse\nimport sys\nfrom .cli import greet\n\n"
            "def main(name=None):\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('name', nargs='?', default='world')\n"
            "    argv = sys.argv[3:] if sys.argv[:3] == ['python', '-m', __package__] else ([name] if name else None)\n"
            "    args = parser.parse_args(argv)\n"
            "    print(greet(args.name))\n"
        ),
        repair_context=repair_context,
    )

    assert review is None


def test_validation_repair_relevance_review_surfaces_target_runtime_evidence(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/__main__.py", "tests/test_cli.py", "greet_cli/cli.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "usage: python [-h] [name]\n"
            "python: error: unrecognized arguments: -m\n"
            '  File "/tmp/tests/test_cli.py", line 11, in test_greet_with_name\n'
            "    __main__.main()\n"
            '  File "/tmp/greet_cli/__main__.py", line 7, in main\n'
            "    args = parser.parse_args()\n"
        ),
        failure_summary="The CLI entrypoint rejects the runtime invocation used by the tests.",
        expected_features=[],
        missing_features=[],
        file_hints=["greet_cli/__main__.py", "tests/test_cli.py", "greet_cli/cli.py"],
        line_hints=[11, 7],
        action_hints=[],
        repair_requirements=[],
        evidence_signature="sig-runtime-evidence",
    )

    review = planner._validation_repair_relevance_review(
        path="greet_cli/__main__.py",
        current_content=(
            "import argparse\n\n"
            "def main():\n"
            "    parser = argparse.ArgumentParser()\n"
            "    args = parser.parse_args()\n"
        ),
        proposed_content=(
            "import argparse\nimport sys\n\n"
            "def main():\n"
            "    parser = argparse.ArgumentParser()\n"
            "    args = parser.parse_args()\n"
        ),
        repair_context=repair_context,
    )

    assert review is not None
    assert any("args = parser.parse_args()" in issue for issue in review.blocking_issues)
    assert any("argument handling" in hint.lower() for hint in review.repair_hints)


def test_validation_repair_relevance_review_allows_python_body_change_with_same_function_signature(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_normalize",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["textutils/normalize.py", "tests/test_normalize.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "Traceback (most recent call last):\n"
            '  File "/tmp/tests/test_normalize.py", line 11, in test_trims_and_collapses_whitespace\n'
            '    self.assertEqual(normalize_name("  ada   lovelace  "), "Ada Lovelace")\n'
            "AssertionError: '  Ada   Lovelace  ' != 'Ada Lovelace'\n"
        ),
        failure_summary="normalize_name still keeps outer and repeated whitespace.",
        expected_features=[],
        missing_features=[],
        file_hints=["textutils/normalize.py", "tests/test_normalize.py"],
        line_hints=[11],
        action_hints=[],
        repair_requirements=[],
        evidence_signature="sig-normalize-runtime",
    )

    review = planner._validation_repair_relevance_review(
        path="textutils/normalize.py",
        current_content=(
            "def normalize_name(value: str) -> str:\n"
            "    parts = value.split(\" \")\n"
            "    return \" \".join(part.capitalize() for part in parts)\n"
        ),
        proposed_content=(
            "def normalize_name(value: str) -> str:\n"
            "    parts = value.split()\n"
            "    return \" \".join(part.capitalize() for part in parts)\n"
        ),
        repair_context=repair_context,
    )

    assert review is None


def test_validation_repair_relevance_review_ignores_test_filename_substring_for_runtime_target(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_wordfreq",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["wordfreq.py", "tests/test_wordfreq.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "Traceback (most recent call last):\n"
            '  File "/home/demo/tests/test_wordfreq.py", line 14, in test_wordfreq\n'
            "    with open('output.txt', 'r') as f:\n"
            "AssertionError: output mismatch\n"
        ),
        failure_summary="The generated output did not match the expected counts.",
        expected_features=[],
        missing_features=[],
        file_hints=["wordfreq.py", "tests/test_wordfreq.py"],
        line_hints=[14],
        action_hints=[],
        repair_requirements=[],
        evidence_signature="sig-wordfreq-runtime",
    )

    review = planner._validation_repair_relevance_review(
        path="wordfreq.py",
        current_content=(
            "def wordfreq(path):\n"
            "    with open(path, 'r', encoding='utf-8') as handle:\n"
            "        return handle.read()\n"
        ),
        proposed_content=(
            "def wordfreq(path):\n"
            "    with open(path, 'r', encoding='utf-8') as handle:\n"
            "        return handle.read().strip()\n"
        ),
        repair_context=repair_context,
    )

    assert review is None


def test_runtime_target_evidence_lines_ignore_test_filename_substring_for_non_test_target(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_wordfreq",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["wordfreq.py", "tests/test_wordfreq.py", "tests/test_data.txt"],
        summary="Validation command exited with 1.",
        excerpt=(
            "Traceback (most recent call last):\n"
            f'  File "{tmp_path / "tests" / "test_wordfreq.py"}", line 6, in test_wordfreq\n'
            "    result = wordfreq('tests/test_data.txt')\n"
            f'  File "{tmp_path / "wordfreq.py"}", line 2, in wordfreq\n'
            "    with open(file_path, 'r') as file:\n"
            "FileNotFoundError: [Errno 2] No such file or directory: 'tests/test_data.txt'\n"
        ),
        failure_summary="wordfreq.py failed because the referenced test fixture is missing.",
        expected_features=[],
        missing_features=[],
        file_hints=["wordfreq.py", "tests/test_wordfreq.py", "tests/test_data.txt"],
        line_hints=[6, 2],
        action_hints=[],
        repair_requirements=[],
        evidence_signature="sig-wordfreq-evidence-lines",
    )

    evidence_lines = planner._runtime_target_evidence_lines("wordfreq.py", repair_context)

    assert evidence_lines
    assert any('File "' in line and "wordfreq.py" in line for line in evidence_lines)
    assert all("test_wordfreq.py" not in line for line in evidence_lines)


def test_review_generated_update_rejects_python_m_launcher_slice_that_keeps_module_name(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    main = pkg / "__main__.py"
    current_content = (
        "import argparse\nfrom .cli import greet\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser(description='Greet someone by name.')\n"
        "    parser.add_argument('name', nargs='?', default='world')\n"
        "    args = parser.parse_args()\n"
        "    print(greet(args.name))\n"
    )
    main.write_text(current_content, encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_content = (
        "import unittest\n"
        "from unittest.mock import patch\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    def test_greet_with_name(self):\n"
        "        from greet_cli import __main__\n"
        "        with patch('__main__.sys.argv', ['python', '-m', 'greet_cli', 'Ada']):\n"
        "            __main__.main()\n"
    )
    (tests_dir / "test_cli.py").write_text(test_content, encoding="utf-8")

    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root=str(tmp_path),
        validation_status="failed",
    )
    session.tool_calls.extend(
        [
            ToolCallRecord(
                iteration=1,
                tool_name="read_file",
                tool_args={"path": "greet_cli/__main__.py"},
                success=True,
                summary="Read greet_cli/__main__.py.",
                output_excerpt=current_content,
            ),
            ToolCallRecord(
                iteration=2,
                tool_name="read_file",
                tool_args={"path": "tests/test_cli.py"},
                success=True,
                summary="Read tests/test_cli.py.",
                output_excerpt=test_content,
            ),
        ]
    )
    session.active_repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/__main__.py", "tests/test_cli.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "usage: python [-h] [name]\n"
            "python: error: unrecognized arguments: -m Ada\n"
            "SystemExit: 2\n"
        ),
        failure_summary="The CLI entrypoint rejects the patched python -m style argv.",
        expected_features=[],
        missing_features=[],
        file_hints=["greet_cli/__main__.py", "tests/test_cli.py"],
        line_hints=[7, 10],
        action_hints=[],
        repair_requirements=[],
        evidence_signature="sig-python-m-launcher",
    )

    review = planner._review_generated_update(
        route=route_payload(
            intent="update",
            action_plan=[{"step": 1, "action": "update_artifact", "reason": "Repair the CLI entrypoint."}],
            target_paths=["greet_cli/__main__.py"],
            target_name="__main__.py",
        ),
        session=session,
        path="greet_cli/__main__.py",
        current_content=current_content,
        proposed_content=(
            "import argparse\nfrom .cli import greet\nimport sys\n\n"
            "def main():\n"
            "    parser = argparse.ArgumentParser(description='Greet someone by name.')\n"
            "    parser.add_argument('name', nargs='?', default='world')\n"
            "    args = parser.parse_args(sys.argv[2:] if sys.argv[:2] == ['python', '-m'] else sys.argv)\n"
            "    print(greet(args.name))\n"
        ),
    )

    assert review.safe_to_write is False
    assert "launcher tokens" in review.summary.lower()
    assert any("sys.argv[2:]" in issue or "module name" in issue for issue in review.blocking_issues)


def test_review_generated_update_rejects_python_m_launcher_fix_that_leaves_parse_args_line_unchanged(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    main = pkg / "__main__.py"
    current_content = (
        "import argparse\nfrom .cli import greet\n\n"
        "def main(name=None):\n"
        "    parser = argparse.ArgumentParser(description='Greet someone by name.')\n"
        "    parser.add_argument('name', nargs='?', default='world')\n"
        "    args = parser.parse_args()\n"
        "    print(greet(args.name))\n"
    )
    main.write_text(current_content, encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_content = (
        "import unittest\n"
        "from unittest.mock import patch\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    def test_greet_with_name(self):\n"
        "        from greet_cli import __main__\n"
        "        with patch('__main__.sys.argv', ['python', '-m', 'greet_cli', 'Ada']):\n"
        "            __main__.main()\n"
    )
    (tests_dir / "test_cli.py").write_text(test_content, encoding="utf-8")

    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root=str(tmp_path),
        validation_status="failed",
    )
    session.tool_calls.extend(
        [
            ToolCallRecord(
                iteration=1,
                tool_name="read_file",
                tool_args={"path": "greet_cli/__main__.py"},
                success=True,
                summary="Read greet_cli/__main__.py.",
                output_excerpt=current_content,
            ),
            ToolCallRecord(
                iteration=2,
                tool_name="read_file",
                tool_args={"path": "tests/test_cli.py"},
                success=True,
                summary="Read tests/test_cli.py.",
                output_excerpt=test_content,
            ),
        ]
    )
    session.active_repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/__main__.py", "tests/test_cli.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "usage: python [-h] [name]\n"
            "python: error: unrecognized arguments: -m Ada\n"
            '  File "/tmp/greet_cli/__main__.py", line 7, in main\n'
            "    args = parser.parse_args()\n"
            "SystemExit: 2\n"
        ),
        failure_summary="The CLI entrypoint rejects the patched python -m style argv.",
        expected_features=[],
        missing_features=[],
        file_hints=["greet_cli/__main__.py", "tests/test_cli.py"],
        line_hints=[7, 10],
        action_hints=[],
        repair_requirements=[],
        evidence_signature="sig-python-m-unchanged-parse-args",
    )

    review = planner._review_generated_update(
        route=route_payload(
            intent="update",
            action_plan=[{"step": 1, "action": "update_artifact", "reason": "Repair the CLI entrypoint."}],
            target_paths=["greet_cli/__main__.py"],
            target_name="__main__.py",
        ),
        session=session,
        path="greet_cli/__main__.py",
        current_content=current_content,
        proposed_content=(
            "import argparse\nfrom .cli import greet\n\n"
            "def main(name=None):\n"
            "    parser = argparse.ArgumentParser(description='Greet someone by name.')\n"
            "    parser.add_argument('name', nargs='?', default='world')\n"
            "    args = parser.parse_args()\n"
            "    print(greet(args.name))\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main('Ada')\n"
        ),
    )

    assert review.safe_to_write is False
    assert "launcher tokens" in review.summary.lower()
    assert any("parse_args()" in issue for issue in review.blocking_issues)


def test_review_generated_update_rejects_python_m_launcher_fix_that_ignores_patched_sys_argv(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    current_content = (
        "import argparse\nfrom .cli import greet\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser(description='Greet someone by name.')\n"
        "    parser.add_argument('name', nargs='?', default='world')\n"
        "    args = parser.parse_args()\n"
        "    print(greet(args.name))\n"
    )
    (pkg / "__main__.py").write_text(current_content, encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_content = (
        "import unittest\n"
        "from unittest.mock import patch\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    def test_greet_with_name(self):\n"
        "        from greet_cli import __main__\n"
        "        with patch('__main__.sys.argv', ['python', '-m', 'greet_cli', 'Ada']):\n"
        "            __main__.main()\n"
    )
    (tests_dir / "test_cli.py").write_text(test_content, encoding="utf-8")

    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root=str(tmp_path),
        validation_status="failed",
    )
    session.tool_calls.extend(
        [
            ToolCallRecord(
                iteration=1,
                tool_name="read_file",
                tool_args={"path": "greet_cli/__main__.py"},
                success=True,
                summary="Read greet_cli/__main__.py.",
                output_excerpt=current_content,
            ),
            ToolCallRecord(
                iteration=2,
                tool_name="read_file",
                tool_args={"path": "tests/test_cli.py"},
                success=True,
                summary="Read tests/test_cli.py.",
                output_excerpt=test_content,
            ),
        ]
    )
    session.active_repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/__main__.py", "tests/test_cli.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "FAIL: test_greet_with_name (tests.test_cli.TestCLI.test_greet_with_name)\n"
            "Traceback (most recent call last):\n"
            '  File "/tmp/tests/test_cli.py", line 7, in test_greet_with_name\n'
            "    __main__.main()\n"
            '  File "/tmp/greet_cli/__main__.py", line 7, in main\n'
            "    args = parser.parse_args([name] if name else [])\n"
            "AssertionError: 'Hello, world!' != 'Hello, Ada!'\n"
        ),
        failure_summary="The CLI still ignores the named runtime input from the patched python -m argv.",
        expected_features=[],
        missing_features=[],
        file_hints=["greet_cli/__main__.py", "tests/test_cli.py"],
        line_hints=[7],
        action_hints=[],
        repair_requirements=[],
        evidence_signature="sig-python-m-ignores-patched-argv",
    )

    review = planner._review_generated_update(
        route=route_payload(
            intent="update",
            action_plan=[{"step": 1, "action": "update_artifact", "reason": "Repair the CLI entrypoint."}],
            target_paths=["greet_cli/__main__.py"],
            target_name="__main__.py",
        ),
        session=session,
        path="greet_cli/__main__.py",
        current_content=current_content,
        proposed_content=(
            "import argparse\nfrom .cli import greet\n\n"
            "def main(name=None):\n"
            "    parser = argparse.ArgumentParser(description='Greet someone by name.')\n"
            "    parser.add_argument('name', nargs='?', default='world')\n"
            "    args = parser.parse_args([name] if name else [])\n"
            "    print(greet(args.name))\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
    )

    assert review.safe_to_write is False
    assert "launcher tokens" in review.summary.lower()
    assert any("patched __main__.sys.argv" in issue for issue in review.blocking_issues)


def test_review_generated_update_rejects_python_m_launcher_fix_that_uses_sys_argv_one_slice(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    current_content = (
        "import argparse\nfrom .cli import greet\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser(description='Greet someone by name.')\n"
        "    parser.add_argument('name', nargs='?', default='world')\n"
        "    args = parser.parse_args()\n"
        "    print(greet(args.name))\n"
    )
    proposed_content = (
        "import argparse\nfrom .cli import greet\n\n"
        "def main(argv=None):\n"
        "    if argv is None:\n"
        "        import sys\n"
        "        argv = sys.argv[1:]\n"
        "    parser = argparse.ArgumentParser(description='Greet someone by name.')\n"
        "    parser.add_argument('name', nargs='?', default='world')\n"
        "    args = parser.parse_args(argv)\n"
        "    print(greet(args.name))\n"
    )
    (pkg / "__main__.py").write_text(current_content, encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_content = (
        "import unittest\n"
        "from unittest.mock import patch\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    def test_greet_with_name(self):\n"
        "        from greet_cli import __main__\n"
        "        with patch('__main__.sys.argv', ['python', '-m', 'greet_cli', 'Ada']):\n"
        "            __main__.main()\n"
    )
    (tests_dir / "test_cli.py").write_text(test_content, encoding="utf-8")

    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root=str(tmp_path),
        validation_status="failed",
    )
    session.tool_calls.extend(
        [
            ToolCallRecord(
                iteration=1,
                tool_name="read_file",
                tool_args={"path": "greet_cli/__main__.py"},
                success=True,
                summary="Read greet_cli/__main__.py.",
                output_excerpt=current_content,
            ),
            ToolCallRecord(
                iteration=2,
                tool_name="read_file",
                tool_args={"path": "tests/test_cli.py"},
                success=True,
                summary="Read tests/test_cli.py.",
                output_excerpt=test_content,
            ),
        ]
    )
    session.active_repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/__main__.py", "tests/test_cli.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "usage: python [-h] [name]\n"
            "python: error: unrecognized arguments: -m Ada\n"
            '  File "/tmp/greet_cli/__main__.py", line 9, in main\n'
            "    args = parser.parse_args(argv)\n"
            "SystemExit: 2\n"
        ),
        failure_summary="The CLI still forwards the python -m launcher flag into argparse.",
        expected_features=[],
        missing_features=[],
        file_hints=["greet_cli/__main__.py", "tests/test_cli.py"],
        line_hints=[9],
        action_hints=[],
        repair_requirements=[],
        evidence_signature="sig-python-m-one-slice",
    )

    review = planner._review_generated_update(
        route=route_payload(
            intent="update",
            action_plan=[{"step": 1, "action": "update_artifact", "reason": "Repair the CLI entrypoint."}],
            target_paths=["greet_cli/__main__.py"],
            target_name="__main__.py",
        ),
        session=session,
        path="greet_cli/__main__.py",
        current_content=current_content,
        proposed_content=proposed_content,
    )

    assert review.safe_to_write is False
    assert "launcher tokens" in review.summary.lower()
    assert any("sys.argv[1:]" in issue or "launcher flag" in issue for issue in review.blocking_issues)


def test_review_generated_update_rejects_python_m_launcher_fix_that_requires_main_argument(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    current_content = (
        "import argparse\nfrom .cli import greet\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser(description='Greet someone by name.')\n"
        "    parser.add_argument('name', nargs='?', default='world')\n"
        "    args = parser.parse_args()\n"
        "    print(greet(args.name))\n"
    )
    (pkg / "__main__.py").write_text(current_content, encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_content = (
        "import unittest\n"
        "from unittest.mock import patch\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    def test_greet_with_name(self):\n"
        "        from greet_cli import __main__\n"
        "        with patch('__main__.sys.argv', ['python', '-m', 'greet_cli', 'Ada']):\n"
        "            __main__.main()\n"
    )
    (tests_dir / "test_cli.py").write_text(test_content, encoding="utf-8")

    session = SessionState(
        task="Fix the failing CLI unittest.",
        workspace_root=str(tmp_path),
        validation_status="failed",
    )
    session.tool_calls.extend(
        [
            ToolCallRecord(
                iteration=1,
                tool_name="read_file",
                tool_args={"path": "greet_cli/__main__.py"},
                success=True,
                summary="Read greet_cli/__main__.py.",
                output_excerpt=current_content,
            ),
            ToolCallRecord(
                iteration=2,
                tool_name="read_file",
                tool_args={"path": "tests/test_cli.py"},
                success=True,
                summary="Read tests/test_cli.py.",
                output_excerpt=test_content,
            ),
        ]
    )
    session.active_repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/__main__.py", "tests/test_cli.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "FAIL: test_greet_with_name (tests.test_cli.TestCLI.test_greet_with_name)\n"
            "Traceback (most recent call last):\n"
            '  File "/tmp/tests/test_cli.py", line 7, in test_greet_with_name\n'
            "    __main__.main()\n"
            "TypeError: main() missing 1 required positional argument: 'name'\n"
        ),
        failure_summary="The proposed repair made main() incompatible with the direct test invocation.",
        expected_features=[],
        missing_features=[],
        file_hints=["greet_cli/__main__.py", "tests/test_cli.py"],
        line_hints=[7],
        action_hints=[],
        repair_requirements=[],
        evidence_signature="sig-python-m-required-main-arg",
    )

    review = planner._review_generated_update(
        route=route_payload(
            intent="update",
            action_plan=[{"step": 1, "action": "update_artifact", "reason": "Repair the CLI entrypoint."}],
            target_paths=["greet_cli/__main__.py"],
            target_name="__main__.py",
        ),
        session=session,
        path="greet_cli/__main__.py",
        current_content=current_content,
        proposed_content=(
            "import argparse\nfrom .cli import greet\n\n"
            "def main(name):\n"
            "    parser = argparse.ArgumentParser(description='Greet someone by name.')\n"
            "    parser.add_argument('name', nargs='?', default='world')\n"
            "    args = parser.parse_args([name])\n"
            "    print(greet(args.name))\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    import sys\n"
            "    main(*sys.argv[1:])\n"
        ),
    )

    assert review.safe_to_write is False
    assert "launcher tokens" in review.summary.lower()
    assert any("without arguments" in issue for issue in review.blocking_issues)


def test_planner_prefers_primary_model_for_validation_guided_repairs(tmp_path):
    planner = Planner(
        ScriptedLLM(
            config=AppConfig(
                workspace_root=str(tmp_path),
                model_name="qwen2.5-coder:14b",
                router_model_name="qwen2.5-coder:7b",
            )
        ),
        "",
    )
    session = SessionState(task="Fix app.py", workspace_root=str(tmp_path))
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Repair the failing runtime behavior."}],
        target_paths=["app.py"],
        target_name="app.py",
    )
    commit_task_state_and_route(planner, session, payload)
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_app",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["app.py", "tests/test_app.py"],
        summary="Validation command exited with 1.",
        excerpt="TypeError: main() takes 0 positional arguments but 1 was given",
        failure_summary="TypeError: main() takes 0 positional arguments but 1 was given",
        expected_features=[],
        missing_features=[],
        file_hints=["app.py", "tests/test_app.py"],
        line_hints=[12],
        action_hints=["Prefer a targeted fix over broad refactors."],
        repair_requirements=["Change app.py so the failing runtime path succeeds."],
        evidence_signature="sig",
    )

    model_name = planner._content_generation_model_name(
        session.router_result,
        session,
        path="app.py",
        current_content="def main():\n    pass\n",
        repair_context=repair_context,
    )

    assert model_name is None


def test_planner_uses_primary_model_for_task_state_updates(tmp_path):
    planner = Planner(
        ScriptedLLM(
            config=AppConfig(
                workspace_root=str(tmp_path),
                model_name="qwen2.5-coder:14b",
                router_model_name="qwen2.5-coder:7b",
                ollama_num_ctx=4096,
            )
        ),
        "",
    )

    assert planner.task_state_updater.model_name == "qwen2.5-coder:14b"
    assert planner.task_state_updater.num_ctx == 4096


def test_review_guided_retry_prefers_primary_model_for_validation_repairs(tmp_path, monkeypatch):
    llm = ScriptedLLM(
        text_payloads=[
            "def main(argv=None):\n    return argv\n",
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:14b",
            router_model_name="qwen2.5-coder:7b",
        ),
    )
    planner = Planner(llm, "")
    session = SessionState(task="Fix app.py", workspace_root=str(tmp_path))
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Repair the failing runtime behavior."}],
        target_paths=["app.py"],
        target_name="app.py",
    )
    commit_task_state_and_route(planner, session, payload)
    monkeypatch.setattr(
        planner,
        "_review_generated_update",
        lambda *_args, **_kwargs: ProposedUpdateReview(
            safe_to_write=True,
            summary="ok",
            confidence=0.9,
            blocking_issues=[],
            preservation_risks=[],
            repair_hints=[],
        ),
    )
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_app",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["app.py", "tests/test_app.py"],
        summary="Validation command exited with 1.",
        excerpt="TypeError: main() takes 0 positional arguments but 1 was given",
        failure_summary="TypeError: main() takes 0 positional arguments but 1 was given",
        expected_features=[],
        missing_features=[],
        file_hints=["app.py", "tests/test_app.py"],
        line_hints=[12],
        action_hints=["Prefer a targeted fix over broad refactors."],
        repair_requirements=["Change app.py so the failing runtime path succeeds."],
        evidence_signature="sig",
    )

    result = planner._retry_update_after_review_failure(
        session.router_result,
        session,
        path="app.py",
        current_content="def main():\n    pass\n",
        review_feedback=ProposedUpdateReview(
            safe_to_write=False,
            summary="Lines unchanged.",
            confidence=0.8,
            blocking_issues=["The proposal leaves the implicated identifier lines unchanged."],
            preservation_risks=[],
            repair_hints=["Change the implicated callable."],
        ),
        repair_context=repair_context,
        repair_strategy="validation_targeted",
        prior_attempts=[],
    )

    assert result.content == "def main(argv=None):\n    return argv"
    assert llm.generate_calls
    assert llm.generate_calls[0]["kwargs"]["model"] == "qwen2.5-coder:14b"


def test_planner_review_retry_keeps_same_model_for_compact_and_full_follow_up_attempts(tmp_path, monkeypatch):
    llm = ScriptedLLM(
        text_payloads=[
            "def main():\n    return 'compact'\n",
            "def main(argv=None):\n    return argv\n",
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:7b",
            router_model_name="qwen2.5-coder:7b",
        ),
    )
    planner = Planner(llm, "")
    session = SessionState(task="Fix app.py", workspace_root=str(tmp_path))
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Repair the failing runtime behavior."}],
        target_paths=["app.py"],
        target_name="app.py",
    )
    commit_task_state_and_route(planner, session, payload)

    review_responses = iter(
        [
            ProposedUpdateReview(
                safe_to_write=False,
                summary="Compact retry still leaves the runtime argv repair incomplete.",
                confidence=0.9,
                blocking_issues=["The proposal references sys.argv without importing sys."],
                preservation_risks=[],
                repair_hints=["If you reference sys.argv, add import sys before using it."],
            ),
            ProposedUpdateReview(
                safe_to_write=True,
                summary="ok",
                confidence=0.9,
                blocking_issues=[],
                preservation_risks=[],
                repair_hints=[],
            ),
        ]
    )
    monkeypatch.setattr(planner, "_review_generated_update", lambda *_args, **_kwargs: next(review_responses))

    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_app",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["app.py", "tests/test_app.py"],
        summary="Validation command exited with 1.",
        excerpt="TypeError: main() takes 0 positional arguments but 1 was given",
        failure_summary="TypeError: main() takes 0 positional arguments but 1 was given",
        expected_features=[],
        missing_features=[],
        file_hints=["app.py", "tests/test_app.py"],
        line_hints=[12],
        action_hints=["Prefer a targeted fix over broad refactors."],
        repair_requirements=["Change app.py so the failing runtime path succeeds."],
        evidence_signature="sig",
    )

    result = planner._retry_update_after_review_failure(
        session.router_result,
        session,
        path="app.py",
        current_content="def main():\n    pass\n",
        review_feedback=ProposedUpdateReview(
            safe_to_write=False,
            summary="Lines unchanged.",
            confidence=0.8,
            blocking_issues=["The proposal leaves the implicated identifier lines unchanged."],
            preservation_risks=[],
            repair_hints=["Change the implicated callable."],
        ),
        repair_context=repair_context,
        repair_strategy="validation_targeted",
        prior_attempts=[],
    )

    assert result.content == "def main(argv=None):\n    return argv"
    assert len(llm.generate_calls) == 2
    assert llm.generate_calls[0]["kwargs"]["model"] == "qwen2.5-coder:7b"
    assert llm.generate_calls[0]["kwargs"]["strict_timeouts"] is True
    assert "The proposal leaves the implicated identifier lines unchanged." in llm.generate_calls[0]["args"][0]
    assert llm.generate_calls[1]["kwargs"]["model"] == "qwen2.5-coder:7b"
    assert llm.generate_calls[1]["kwargs"]["strict_timeouts"] is True
    assert "references sys.argv without importing sys." in llm.generate_calls[1]["args"][0]
    assert "add import sys before using it" in llm.generate_calls[1]["args"][0]


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


def test_generated_content_integrity_review_blocks_placeholder_python_test_without_assertions(tmp_path):
    planner = Planner(ScriptedLLM(), "")

    review = planner._generated_content_integrity_review(
        path="tests/test_wordfreq.py",
        proposed_content=(
            "import unittest\n\n"
            "from wordfreq import wordfreq\n\n\n"
            "class WordfreqTests(unittest.TestCase):\n"
            "    def test_counts_words(self):\n"
            "        result = wordfreq('sample.txt')\n"
            "        # Add assertions to check the output.\n"
            "        print(result)\n"
        ),
    )

    assert review is not None
    assert review.safe_to_write is False
    assert "test file" in review.summary.lower()
    assert any("placeholder" in issue.lower() or "assert" in issue.lower() for issue in review.blocking_issues)


def test_generated_content_integrity_review_allows_python_test_with_meaningful_assertions(tmp_path):
    planner = Planner(ScriptedLLM(), "")

    review = planner._generated_content_integrity_review(
        path="tests/test_wordfreq.py",
        proposed_content=(
            "import unittest\n\n"
            "from wordfreq import wordfreq\n\n\n"
            "class WordfreqTests(unittest.TestCase):\n"
            "    def test_counts_words(self):\n"
            "        result = wordfreq('sample.txt')\n"
            "        self.assertEqual(result[0], ('ada', 2))\n"
        ),
    )

    assert review is None


def test_generated_content_integrity_review_blocks_tautological_python_test_assertion(tmp_path):
    planner = Planner(ScriptedLLM(), "")

    review = planner._generated_content_integrity_review(
        path="tests/test_wordfreq.py",
        proposed_content=(
            "import unittest\n\n"
            "from wordfreq import wordfreq\n\n\n"
            "class WordfreqTests(unittest.TestCase):\n"
            "    def test_counts_words(self):\n"
            "        wordfreq('sample.txt')\n"
            "        self.assertTrue(True)\n"
        ),
    )

    assert review is not None
    assert review.safe_to_write is False
    assert "meaningful assertion" in review.summary.lower()
    assert any("tautological" in issue.lower() or "meaningful assertion" in issue.lower() for issue in review.blocking_issues)


def test_planner_retries_create_after_placeholder_test_review_rejection(tmp_path):
    llm = ScriptedLLM(
        text_payloads=[
            (
                "import unittest\n\n"
                "from wordfreq import wordfreq\n\n\n"
                "class WordfreqTests(unittest.TestCase):\n"
                "    def test_counts_words(self):\n"
                "        result = wordfreq('sample.txt')\n"
                "        # Add assertions to check the output.\n"
                "        print(result)\n"
            ),
            (
                "import unittest\n\n"
                "from wordfreq import wordfreq\n\n\n"
                "class WordfreqTests(unittest.TestCase):\n"
                "    def test_counts_words(self):\n"
                "        result = wordfreq('sample.txt')\n"
                "        self.assertEqual(result[0], ('ada', 2))\n"
            ),
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:14b",
            router_model_name="qwen2.5-coder:7b",
        ),
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Create a real unittest file for wordfreq and make sure it actually checks the result.",
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="create",
        action_plan=[
            {"step": 1, "action": "create_artifact", "reason": "Create the requested test file."},
        ],
        target_paths=["tests/test_wordfreq.py"],
        target_name="tests/test_wordfreq.py",
    )
    commit_task_state_and_route(planner, session, payload, verification_target="python -m unittest tests.test_wordfreq")

    result = planner._generate_file_content(
        session.router_result,
        session,
        path="tests/test_wordfreq.py",
        current_content=None,
    )

    retry_prompt = llm.generate_calls[1]["args"][0]

    assert result.content is not None
    assert "self.assertEqual" in result.content
    assert len(llm.generate_calls) == 2
    assert llm.generate_calls[1]["kwargs"]["strict_timeouts"] is True
    assert llm.generate_calls[1]["kwargs"]["num_ctx"] == 2048
    assert "Self-review feedback on the previous proposal:" in retry_prompt
    assert "meaningful assertion" in retry_prompt or "placeholder" in retry_prompt.lower()


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


def test_planner_switches_to_primary_model_when_lightweight_no_start_cannot_go_faster(tmp_path):
    planner = Planner(
        ScriptedLLM(
            config=AppConfig(
                workspace_root=str(tmp_path),
                model_name="qwen2.5-coder:14b",
                router_model_name="qwen2.5-coder:7b",
            )
        ),
        "",
    )

    attempts = planner._content_generation_recovery_attempts(
        ExecutionFailure(
            failure_class="startup_timeout",
            state="failed_startup",
            had_progress=False,
            first_output_received=False,
            model_identifier="qwen2.5-coder:7b",
            backend_identifier="ollama",
            context_pressure_estimate="low",
            retryable=False,
            raw_reason="startup_timeout",
        )
    )

    assert attempts
    assert attempts[0].strategy == "compact_fallback_model"
    assert attempts[0].model_name == "qwen2.5-coder:14b"
    assert attempts[0].capability_tier == "tier_a"
    assert attempts[0].prompt_kind == "compact"


def test_planner_uses_compact_prompt_for_fallback_model_after_primary_no_start(tmp_path):
    planner = Planner(
        ScriptedLLM(
            config=AppConfig(
                workspace_root=str(tmp_path),
                model_name="qwen2.5-coder:14b",
                router_model_name="qwen2.5-coder:7b",
            )
        ),
        "",
    )

    attempts = planner._content_generation_recovery_attempts(
        ExecutionFailure(
            failure_class="startup_timeout",
            state="failed_startup",
            had_progress=False,
            first_output_received=False,
            model_identifier="qwen2.5-coder:14b",
            backend_identifier="ollama",
            context_pressure_estimate="low",
            retryable=False,
            raw_reason="startup_timeout",
        )
    )

    assert attempts
    assert attempts[0].model_name == "qwen2.5-coder:7b"
    assert attempts[0].prompt_kind == "compact"
    assert attempts[0].strategy == "compact_fallback_model"


def test_planner_uses_compact_same_model_retry_after_retryable_no_start(tmp_path):
    planner = Planner(
        ScriptedLLM(
            config=AppConfig(
                workspace_root=str(tmp_path),
                model_name="qwen2.5-coder:7b",
                router_model_name="qwen2.5-coder:7b",
            )
        ),
        "",
    )

    attempts = planner._content_generation_recovery_attempts(
        ExecutionFailure(
            failure_class="startup_timeout",
            state="failed_startup",
            had_progress=False,
            first_output_received=False,
            model_identifier="qwen2.5-coder:7b",
            backend_identifier="ollama",
            context_pressure_estimate="low",
            retryable=True,
            raw_reason="startup_timeout",
        )
    )

    assert attempts
    assert attempts[0].strategy == "retry_same_model"
    assert attempts[0].prompt_kind == "compact"
    assert attempts[0].model_name is None


def test_planner_extends_compact_primary_generation_budget_for_repairs(tmp_path):
    planner = Planner(ScriptedLLM(config=AppConfig(workspace_root=str(tmp_path))), "")
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/__main__.py", "tests/test_cli.py"],
        summary="Validation command exited with 1.",
        excerpt="AssertionError: \"Hello, ['Ada']!\" != 'Hello, Ada!'",
        failure_summary="AssertionError: the CLI prints the wrong greeting when argv is provided.",
        expected_features=[],
        missing_features=[],
        file_hints=["greet_cli/__main__.py", "tests/test_cli.py"],
        line_hints=[10],
        action_hints=["Prefer a targeted runtime fix and rerun the same test command."],
        repair_requirements=["Repair greet_cli/__main__.py so both CLI tests pass."],
        evidence_signature="repair-time-budget",
    )

    timeout_seconds, total_timeout_seconds = planner._content_generation_time_budget(
        prompt_variant="compact",
        repair_context=repair_context,
    )

    assert timeout_seconds == 60
    assert total_timeout_seconds == 210


def test_planner_extends_compact_resume_budget_after_timeout_progress(tmp_path):
    planner = Planner(ScriptedLLM(config=AppConfig(workspace_root=str(tmp_path))), "")

    timeout_seconds, total_timeout_seconds, num_ctx = planner._content_generation_runtime_for_attempt(
        GenerationRecoveryAttempt(
            strategy="resume_same_model",
            prompt_kind="resume",
            model_name=None,
            capability_tier="tier_a",
        ),
        ExecutionFailure(
            failure_class="total_timeout",
            state="failed_after_progress",
            had_progress=True,
            first_output_received=True,
            model_identifier="qwen2.5-coder:14b",
            backend_identifier="ollama",
            context_pressure_estimate="low",
            retryable=True,
            raw_reason="total_timeout",
        ),
    )

    assert timeout_seconds == 60
    assert total_timeout_seconds == 210
    assert num_ctx == 3072


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
