from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.models import (
    DiagnosticRecord,
    FileChangeRecord,
    FollowUpContext,
    ProposedUpdateReview,
    RepairAttemptSummary,
    RepairBrief,
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
    _repair_target_line_hints,
    _repair_required_literal_anchors,
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
from llm.schemas import AgentActionType, AgentDecision, RouteIntent, RouterOutput
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


def test_choose_create_path_prefers_index_html_for_empty_workspace_web_requests(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    llm = ScriptedLLM(config=config)
    planner = Planner(llm, "", logger=AgentLogger(tmp_path, "planner"))
    session = SessionState(task="build a snake game", workspace_root=str(tmp_path))
    session.workspace_snapshot = empty_snapshot(tmp_path)
    route = RouterOutput.model_validate(
        {
            "user_goal": "Build a snake game as a single HTML file.",
            "intent": "create",
            "entities": {
                "target_type": "artifact",
                "target_name": "snake",
                "target_paths": [],
                "attributes": [],
                "constraints": [],
            },
            "requested_outcome": "Build a snake game as a single HTML file.",
            "action_plan": [
                {
                    "step": 1,
                    "action": "create_artifact",
                    "reason": "Create the requested artifact.",
                }
            ],
            "needs_clarification": False,
            "clarification_questions": [],
            "confidence": 0.9,
            "safe_to_execute": True,
            "repo_context_needed": True,
            "search_terms": ["snake game", "html"],
            "relevant_extensions": [".html"],
            "direct_response": None,
        }
    )

    assert planner._choose_create_path(route, session) == "index.html"


def test_next_update_target_stays_on_locked_repair_target_before_other_explicit_targets(tmp_path):
    llm = ScriptedLLM()
    planner = Planner(llm, "")
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "repo-map.md").write_text("# Repo Map\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Inventory App\n", encoding="utf-8")

    session = SessionState(
        task="Create docs/repo-map.md and run python -m unittest tests.test_repo_map.",
        workspace_root=str(tmp_path),
        changed_files=[FileChangeRecord(path="docs/repo-map.md", operation="create")],
        validation_status="failed",
        task_state=TaskState(
            latest_user_turn="Create docs/repo-map.md and run python -m unittest tests.test_repo_map.",
            root_goal="Document the inventory repo.",
            active_goal="Repair docs/repo-map.md after validation failed.",
            goal_relation="continue",
            output_expectation="A repo map that passes the targeted unittest.",
            verification_target="python -m unittest tests.test_repo_map",
            next_action="update",
            target_artifacts=[
                {"path": "docs/repo-map.md", "kind": "doc", "role": "primary_target", "confidence": 1.0},
                {"path": "tests/test_repo_map.py", "kind": "test", "role": "validation_target", "confidence": 1.0},
            ],
        ),
    )
    session.active_repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_repo_map",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["docs/repo-map.md", "README.md", "tests/test_repo_map.py"],
        summary="Validation command exited with 1.",
        excerpt="AssertionError: 'tests/test_auth.py' not found in docs/repo-map.md output",
        failure_summary="docs/repo-map.md is still missing required repo-map references.",
        file_hints=["docs/repo-map.md", "README.md", "tests/test_repo_map.py"],
        repair_requirements=["Change docs/repo-map.md so the failing validation passes."],
        evidence_signature="runtime:assertion_mismatch:test",
        repair_brief=RepairBrief(
            failure_type="assertion_mismatch",
            failure_signature="runtime:assertion_mismatch:test",
            primary_target="docs/repo-map.md",
            locked_target="docs/repo-map.md",
            expected_semantics=["Validation should produce: tests/test_auth.py"],
            observed_semantics=["Validation currently produces: repo map text without tests/test_auth.py"],
            allowed_files=["docs/repo-map.md"],
            forbidden_files=["README.md", "tests/test_repo_map.py"],
        ),
    )
    route = RouterOutput.model_validate(
        {
            "user_goal": "Repair the repo map after validation failed.",
            "intent": "update",
            "entities": {
                "target_type": "file",
                "target_name": "docs/repo-map.md",
                "target_paths": ["docs/repo-map.md", "README.md"],
                "attributes": [],
                "constraints": [],
            },
            "requested_outcome": "Repair docs/repo-map.md so the failed validation passes.",
            "action_plan": [
                {"step": 1, "action": "update_artifact", "reason": "Repair the failed validation target."}
            ],
            "needs_clarification": False,
            "clarification_questions": [],
            "confidence": 0.9,
            "safe_to_execute": True,
            "repo_context_needed": True,
            "search_terms": ["docs/repo-map.md"],
            "relevant_extensions": [".md"],
            "direct_response": None,
        }
    )

    assert planner._next_update_target(route, session) == "docs/repo-map.md"


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


def test_fallback_semantic_review_accepts_small_standalone_web_structural_proxy(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Erstelle ein kleines spielbares Snake-Spiel als HTML-Datei mit Tastatursteuerung, Punktestand und Game-Over-Neustart.",
        workspace_root=str(tmp_path),
        workspace_snapshot=WorkspaceSnapshot(
            root=str(tmp_path),
            file_count=1,
            language_counts={"html": 1},
            top_directories=[],
            important_files=["index.html"],
            focus_files=["index.html"],
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
            repo_summary="Small standalone web artifact.",
        ),
        validation_status="passed",
        changed_files=[FileChangeRecord(path="index.html", operation="create")],
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command='internal:web_artifact:[{"path":"index.html","expected_features":["score","keyboard_controls","game_over","start_controls"]}]',
            kind="check",
            verification_scope="structural",
            status="passed",
            edit_generation=0,
        )
    )

    review = planner._fallback_semantic_change_review(session)

    assert review.requirements_satisfied is True
    assert "standalone web artifact" in review.summary


def test_planner_skips_model_backed_semantic_review_for_small_standalone_web_proxy(tmp_path):
    llm = ScriptedLLM(
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:7b",
            router_model_name="qwen2.5-coder:7b",
        )
    )
    planner = Planner(llm, "")
    payload = route_payload(
        intent="create",
        action_plan=[
            {"step": 1, "action": "create_artifact", "reason": "Create the requested artifact."},
            {"step": 2, "action": "run_validation", "reason": "Validate the result."},
            {"step": 3, "action": "summarize_result", "reason": "Summarize honestly."},
        ],
        target_paths=["index.html"],
        target_name="index.html",
    )
    session = SessionState(
        task="Erstelle ein kleines spielbares Snake-Spiel als HTML-Datei mit Tastatursteuerung, Punktestand und Game-Over-Neustart.",
        workspace_root=str(tmp_path),
        workspace_snapshot=WorkspaceSnapshot(
            root=str(tmp_path),
            file_count=1,
            language_counts={"html": 1},
            top_directories=[],
            important_files=["index.html"],
            focus_files=["index.html"],
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
            repo_summary="Small standalone web artifact.",
        ),
        validation_status="passed",
    )
    commit_task_state_and_route(planner, session, payload, verification_target="Verify the generated web artifact.")
    session.changed_files.append(FileChangeRecord(path="index.html", operation="create"))
    session.validation_runs.append(
        ValidationRunRecord(
            command='internal:web_artifact:[{"path":"index.html","expected_features":["score","keyboard_controls","game_over","start_controls"]}]',
            kind="check",
            verification_scope="structural",
            status="passed",
            edit_generation=0,
        )
    )

    planner._run_semantic_change_review(session.router_result, session)

    assert llm.generate_json_calls == []
    assert session.validation_runs[-1].verification_scope == "semantic"
    assert session.validation_runs[-1].status == "passed"
    assert session.runtime_executions[-1]["recovery_strategy"] == "deterministic_fallback"


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
    assert llm.generate_json_calls[0]["kwargs"]["strict_timeouts"] is True
    assert '"task_understanding"' not in prompt
    assert '"follow_up_context"' not in prompt


def test_planner_uses_local_review_fallback_for_small_updates_without_fallback_model(tmp_path):
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
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:7b",
            router_model_name="qwen2.5-coder:7b",
        ),
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

    assert review.safe_to_write is True
    assert llm.generate_json_calls == []
    assert session.runtime_executions[-1]["recovery_strategy"] == "deterministic_fallback"
    assert session.runtime_executions[-1]["task_class"] == "proposed_update_review"


def test_planner_uses_local_review_fallback_for_compact_single_model_repairs(tmp_path):
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
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:7b",
            router_model_name="qwen2.5-coder:7b",
        ),
    )
    planner = Planner(llm, "")
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Fix the failing target."},
            {"step": 2, "action": "run_validation", "reason": "Validate the repair."},
        ],
        target_paths=["greet_cli/__main__.py"],
        target_name="greet_cli/__main__.py",
    )
    session = SessionState(
        task="Repair the CLI greeting formatting without changing unrelated files.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    commit_task_state_and_route(planner, session, payload, verification_target="python -m unittest tests.test_cli")
    session.active_repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/__main__.py", "tests/test_cli.py"],
        summary="Validation command exited with 1.",
        excerpt="AssertionError: 'Dr.: Hello, Ada!' != 'Dr. Hello, Ada!'",
        failure_summary="The CLI currently inserts punctuation that the expected output does not contain.",
        repair_requirements=["Change greet_cli/__main__.py so the failing runtime path can complete successfully."],
        repair_brief=RepairBrief(
            failure_type="assertion_mismatch",
            failure_signature="runtime:assertion_mismatch:compactrepair",
            primary_target="greet_cli/__main__.py",
            locked_target="greet_cli/__main__.py",
            expected_semantics=["Validation should produce: Dr. Hello, Ada!"],
            observed_semantics=["Validation currently produces: Dr.: Hello, Ada!"],
            implicated_region_hint="greet_cli/__main__.py:line 5",
            repair_constraints=["Change greet_cli/__main__.py so the failing runtime path can complete successfully."],
            allowed_files=["greet_cli/__main__.py"],
            forbidden_files=["tests/test_cli.py"],
        ),
    )

    review = planner._review_generated_update(
        session.router_result,
        session,
        path="greet_cli/__main__.py",
        current_content=(
            "def main(argv=None):\n"
            "    greeting = 'Hello, Ada!'\n"
            "    title = 'Dr.'\n"
            "    if title:\n"
            "        greeting = f\"{title}: {greeting}\"\n"
            "    print(greeting)\n"
        ),
        proposed_content=(
            "def main(argv=None):\n"
            "    greeting = 'Hello, Ada!'\n"
            "    title = 'Dr.'\n"
            "    if title:\n"
            "        greeting = f\"{title} {greeting}\"\n"
            "    print(greeting)\n"
        ),
    )

    assert review.safe_to_write is True
    assert llm.generate_json_calls == []
    assert session.runtime_executions[-1]["recovery_strategy"] == "deterministic_fallback"
    assert session.runtime_executions[-1]["task_class"] == "proposed_update_review"


def test_planner_repair_target_scope_review_blocks_drift_before_ai_review(tmp_path):
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
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:7b",
            router_model_name="qwen2.5-coder:7b",
        ),
    )
    planner = Planner(llm, "")
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Fix the failing target."}],
        target_paths=["greet_cli/__main__.py"],
        target_name="greet_cli/__main__.py",
    )
    session = SessionState(
        task="Repair the CLI greeting formatting without changing tests.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    commit_task_state_and_route(planner, session, payload, verification_target="python -m unittest tests.test_cli")
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/__main__.py", "tests/test_cli.py"],
        summary="Validation command exited with 1.",
        excerpt="AssertionError: 'Dr.: Hello, Ada!' != 'Dr. Hello, Ada!'",
        failure_summary="The CLI currently inserts punctuation that the expected output does not contain.",
        repair_requirements=["Change greet_cli/__main__.py so the failing runtime path can complete successfully."],
        repair_brief=RepairBrief(
            failure_type="assertion_mismatch",
            failure_signature="runtime:assertion_mismatch:lockscope",
            primary_target="greet_cli/__main__.py",
            locked_target="greet_cli/__main__.py",
            allowed_files=["greet_cli/__main__.py"],
            forbidden_files=["tests/test_cli.py"],
        ),
    )

    review = planner._pre_write_update_review(
        session.router_result,
        session,
        path="tests/test_cli.py",
        current_content="def test_cli():\n    assert run_cli(['Ada']) == 'Hello, Ada!'\n",
        proposed_content="def test_cli():\n    assert run_cli(['Ada']) == 'Dr. Hello, Ada!'\n",
        repair_context=repair_context,
    )

    assert review.safe_to_write is False
    assert "locked" in review.summary.lower() or "scope" in review.summary.lower()
    assert llm.generate_json_calls == []


def test_planner_broad_single_model_repair_can_still_escalate_to_ai_review(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            {
                "safe_to_write": True,
                "summary": "The broader rewrite still preserves the required behavior.",
                "confidence": 0.75,
                "blocking_issues": [],
                "preservation_risks": [],
                "repair_hints": [],
            }
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:7b",
            router_model_name="qwen2.5-coder:7b",
        ),
    )
    planner = Planner(llm, "")
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Fix the failing target."}],
        target_paths=["greet_cli/__main__.py"],
        target_name="greet_cli/__main__.py",
    )
    session = SessionState(
        task="Repair the CLI greeting formatting even if the implementation needs a larger rewrite.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    commit_task_state_and_route(planner, session, payload, verification_target="python -m unittest tests.test_cli")
    session.active_repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/__main__.py", "tests/test_cli.py"],
        summary="Validation command exited with 1.",
        excerpt="AssertionError: 'Dr.: Hello, Ada!' != 'Dr. Hello, Ada!'",
        failure_summary="The CLI currently inserts punctuation that the expected output does not contain.",
        repair_requirements=["Change greet_cli/__main__.py so the failing runtime path can complete successfully."],
        repair_brief=RepairBrief(
            failure_type="assertion_mismatch",
            failure_signature="runtime:assertion_mismatch:broadrepair",
            primary_target="greet_cli/__main__.py",
            locked_target="greet_cli/__main__.py",
            allowed_files=["greet_cli/__main__.py"],
        ),
    )

    current_content = "\n".join(f"line_{index} = {index}" for index in range(1, 15)) + "\n"
    proposed_content = "\n".join(f"rewritten_line_{index} = {index}" for index in range(1, 15)) + "\n"

    review = planner._review_generated_update(
        session.router_result,
        session,
        path="greet_cli/__main__.py",
        current_content=current_content,
        proposed_content=proposed_content,
    )

    assert review.safe_to_write is True
    assert len(llm.generate_json_calls) == 1


def test_planner_uses_local_review_fallback_for_locked_single_model_repair_without_explicit_targets(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            {
                "safe_to_write": True,
                "summary": "This JSON review should be skipped for a focused locked repair.",
                "confidence": 0.8,
                "blocking_issues": [],
                "preservation_risks": [],
                "repair_hints": [],
            }
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:7b",
            router_model_name="qwen2.5-coder:7b",
        ),
    )
    planner = Planner(llm, "")
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Fix the failing target."}],
        target_paths=None,
        target_name=None,
    )
    session = SessionState(
        task="Repair the CLI entrypoint without drifting into docs or tests.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    commit_task_state_and_route(planner, session, payload, verification_target="python -m unittest tests.test_cli")
    session.changed_files = [
        FileChangeRecord(path="greet_cli/__main__.py", operation="write", diff=""),
        FileChangeRecord(path="README.md", operation="write", diff=""),
        FileChangeRecord(path="tests/test_cli.py", operation="write", diff=""),
    ]
    session.active_repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/__main__.py", "greet_cli/cli.py", "README.md", "tests/test_cli.py"],
        summary="Validation command exited with 1.",
        excerpt="AssertionError: 'Dr.: Hello, Ada!' != 'Dr. Hello, Ada!'",
        failure_summary="The CLI currently inserts punctuation that the expected output does not contain.",
        repair_requirements=["Change greet_cli/__main__.py so the failing runtime path can complete successfully."],
        repair_brief=RepairBrief(
            failure_type="assertion_mismatch",
            failure_signature="runtime:assertion_mismatch:livecompactrepair",
            primary_target="greet_cli/__main__.py",
            locked_target="greet_cli/__main__.py",
            expected_semantics=["Validation should produce: Dr. Hello, Ada!"],
            observed_semantics=["Validation currently produces: Dr.: Hello, Ada!"],
            implicated_region_hint="greet_cli/__main__.py",
            repair_constraints=[
                "Change greet_cli/__main__.py so the failing runtime path can complete successfully.",
                "Keep one clear contract boundary between the entrypoint and helper.",
            ],
            allowed_files=["greet_cli/__main__.py", "greet_cli/cli.py"],
            forbidden_files=["README.md", "tests/test_cli.py"],
        ),
    )

    current_content = (
        "def main(argv=None):\n"
        "    greeting = 'Hello, Ada!'\n"
        "    title = 'Dr.'\n"
        "    if title:\n"
        "        greeting = f\"{title}: {greeting}\"\n"
        "    print(greeting)\n"
    )
    proposed_content = (
        "def main(argv=None):\n"
        "    greeting = 'Hello, Ada!'\n"
        "    title = 'Dr.'\n"
        "    if title:\n"
        "        greeting = f\"{title} {greeting}\"\n"
        "    print(greeting)\n"
    )

    review = planner._review_generated_update(
        session.router_result,
        session,
        path="greet_cli/__main__.py",
        current_content=current_content,
        proposed_content=proposed_content,
    )

    assert review.safe_to_write is True
    assert llm.generate_json_calls == []
    assert session.runtime_executions[-1]["recovery_strategy"] == "deterministic_fallback"


def test_planner_repair_no_effective_change_blocks_model_review(tmp_path):
    llm = ScriptedLLM(
        json_payloads=[
            {
                "safe_to_write": True,
                "summary": "This JSON review should not run when the repair makes no change.",
                "confidence": 0.7,
                "blocking_issues": [],
                "preservation_risks": [],
                "repair_hints": [],
            }
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:7b",
            router_model_name="qwen2.5-coder:7b",
        ),
    )
    planner = Planner(llm, "")
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Fix the failing target."}],
        target_paths=None,
        target_name=None,
    )
    session = SessionState(
        task="Repair the CLI entrypoint with a real behavior change.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    commit_task_state_and_route(planner, session, payload, verification_target="python -m unittest tests.test_cli")
    session.active_repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/__main__.py", "tests/test_cli.py"],
        summary="Validation command exited with 1.",
        excerpt="AssertionError: 'Dr.: Hello, Ada!' != 'Dr. Hello, Ada!'",
        failure_summary="The CLI currently inserts punctuation that the expected output does not contain.",
        repair_requirements=["Change greet_cli/__main__.py so the failing runtime path can complete successfully."],
        repair_brief=RepairBrief(
            failure_type="assertion_mismatch",
            failure_signature="runtime:assertion_mismatch:noeffectivechange",
            primary_target="greet_cli/__main__.py",
            locked_target="greet_cli/__main__.py",
            allowed_files=["greet_cli/__main__.py"],
            forbidden_files=["tests/test_cli.py"],
        ),
    )

    current_content = (
        "def main(argv=None):\n"
        "    greeting = 'Hello, Ada!'\n"
        "    title = 'Dr.'\n"
        "    if title:\n"
        "        greeting = f\"{title}: {greeting}\"\n"
        "    print(greeting)\n"
    )

    review = planner._pre_write_update_review(
        session.router_result,
        session,
        path="greet_cli/__main__.py",
        current_content=current_content,
        proposed_content=current_content,
        repair_context=session.active_repair_context,
    )

    assert review.safe_to_write is False
    assert (
        "unchanged" in review.summary.lower()
        or "effective change" in review.summary.lower()
        or "no-op" in review.summary.lower()
    )
    assert llm.generate_json_calls == []


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
    assert len(llm.generate_json_calls) == 2
    assert llm.generate_json_calls[0]["kwargs"]["model"] == "qwen3:8b"
    assert llm.generate_json_calls[0]["kwargs"]["strict_timeouts"] is True
    assert llm.generate_json_calls[1]["kwargs"]["model"] == "qwen3:14b"
    assert llm.generate_json_calls[1]["kwargs"]["strict_timeouts"] is True


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


def test_fallback_semantic_review_ignores_deferred_snapshot_explicit_target(tmp_path):
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
            FileChangeRecord(path="README.md", operation="modify"),
            FileChangeRecord(path="tests/test_cli.py", operation="modify"),
        ],
        notes=[f"{DEFERRED_UPDATE_TARGET_NOTE_PREFIX}greet_cli/cli.py"],
    )

    review = planner._fallback_semantic_change_review(session)

    assert review.requirements_satisfied is True
    assert "no additional concrete task-to-code mismatch" in review.summary


def test_fallback_semantic_review_ignores_supporting_doc_snapshot_target_for_passed_repair(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task=(
            "Fix the existing calcstats bug so moving_average returns one average per full window only, "
            "keeps raising ValueError for invalid window sizes, updates the README example if needed, "
            "and runs python -m unittest tests.test_stats before finishing."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "important_files": ["calcstats/stats.py", "README.md", "tests/test_stats.py"],
                "focus_files": ["calcstats/stats.py", "tests/test_stats.py"],
                "manifests": ["README.md"],
                "test_files": ["tests/test_stats.py"],
                "entrypoints": ["calcstats/stats.py"],
            }
        ),
        validation_status="passed",
        changed_files=[FileChangeRecord(path="calcstats/stats.py", operation="modify")],
    )
    payload = route_payload(
        intent="debug",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the failing implementation."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["calcstats/stats.py", "README.md", "tests/test_stats.py"],
        target_name="calcstats/stats.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        goal_relation="report_problem",
        open_problem="Fix the moving_average bug without regressing the validated implementation.",
        verification_target="python -m unittest tests.test_stats",
    )
    session.task_state.current_user_intent = "repair"
    artifact_roles = {item.path: item for item in session.task_state.target_artifacts if item.path}
    artifact_roles["calcstats/stats.py"].role = "primary_target"
    artifact_roles["README.md"].role = "supporting_context"
    artifact_roles["tests/test_stats.py"].role = "validation_target"

    review = planner._fallback_semantic_change_review(session)

    assert review.requirements_satisfied is True
    assert "no additional concrete task-to-code mismatch" in review.summary


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


def test_planner_accepts_typed_python_signature_for_explicit_literal_constraint(tmp_path):
    current = (
        "from __future__ import annotations\n\n"
        "import re\n"
        "from collections import Counter\n\n\n"
        "def count_words(file_path: str) -> dict[str, int]:\n"
        "    with open(file_path, 'r', encoding='utf-8') as handle:\n"
        "        words = re.findall(r'\\\\b\\\\w+\\\\b', handle.read().lower())\n"
        "    return dict(Counter(words))\n"
    )
    proposed = (
        current
        + "\n\n"
        + "def top_words(file_path: str, limit: int) -> list[tuple[str, int]]:\n"
        + "    counts = count_words(file_path)\n"
        + "    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]\n"
    )
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task=(
            "Keep count_words in wordfreq.py intact and add a new helper top_words(file_path, limit) in wordfreq.py "
            "that returns the most frequent words sorted by count descending and alphabetically for ties."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Add the requested helper to wordfreq.py."},
        ],
        target_paths=["wordfreq.py", "README.md", "tests/test_wordfreq.py"],
        target_name="wordfreq.py",
    )
    commit_task_state_and_route(planner, session, payload)

    review = planner._explicit_constraint_integrity_review(
        session.router_result,
        session,
        path="wordfreq.py",
        current_content=current,
        proposed_content=proposed,
    )

    assert review is None


def test_artifact_scoped_focus_scopes_helper_signature_literal_to_owning_python_file(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    main_current = (
        "from .cli import greet\n\n"
        "def main(argv=None):\n"
        "    print(greet('Ada'))\n"
    )
    cli_current = "def greet(name):\n    return f'Hello, {name}!'\n"
    (pkg / "__main__.py").write_text(main_current, encoding="utf-8")
    (pkg / "cli.py").write_text(cli_current, encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task=(
            "You are working in an existing Python CLI repo. Keep the default greeting behavior intact. "
            "Extend greet_cli/cli.py so greet(name, shout=False) returns an uppercased greeting when shout=True. "
            "Update greet_cli/__main__.py to add a --shout flag that routes through the helper, and update README.md with the new flag."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Apply the requested CLI update."},
        ],
        target_paths=["greet_cli/__main__.py", "greet_cli/cli.py", "README.md", "tests/test_cli.py"],
        target_name="greet_cli/__main__.py",
    )
    commit_task_state_and_route(planner, session, payload)

    main_focus = _artifact_scoped_focus(
        session.router_result,
        session,
        "greet_cli/__main__.py",
        current_content=main_current,
    )
    cli_focus = _artifact_scoped_focus(
        session.router_result,
        session,
        "greet_cli/cli.py",
        current_content=cli_current,
    )

    assert "greet(name, shout=False)" not in main_focus["literal_constraints"]
    assert "greet(name, shout=False)" in cli_focus["literal_constraints"]


def test_artifact_scoped_focus_adds_cross_file_import_consistency_requirement(tmp_path):
    pkg = tmp_path / "texttools"
    pkg.mkdir()
    init_current = "from .normalize import normalize_words, normalize_words_keep_case\n"
    normalize_current = "def normalize_words(text, *, keep_case=False):\n    return text.split()\n"
    (pkg / "__init__.py").write_text(init_current, encoding="utf-8")
    (pkg / "normalize.py").write_text(normalize_current, encoding="utf-8")
    (tmp_path / "normalize_cli.py").write_text("from texttools import normalize_words\n", encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    snapshot = build_snapshot(tmp_path).model_copy(
        update={
            "important_files": ["texttools/__init__.py", "texttools/normalize.py", "normalize_cli.py"],
            "focus_files": ["texttools/normalize.py", "texttools/__init__.py", "normalize_cli.py"],
            "file_briefs": {
                "texttools/__init__.py": "from .normalize import normalize_words, normalize_words_keep_case",
                "texttools/normalize.py": "def normalize_words(text, *, keep_case=False):",
                "normalize_cli.py": "from texttools import normalize_words",
            },
            "symbol_index": {
                "texttools/normalize.py": ["normalize_words"],
            },
            "import_hotspots": ["texttools/__init__.py", "normalize_cli.py"],
        }
    )
    session = SessionState(
        task="Add keep_case support to the package and CLI without breaking imports.",
        workspace_root=str(tmp_path),
        workspace_snapshot=snapshot,
    )
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Finish the keep-case feature."}],
        target_paths=["texttools/normalize.py", "normalize_cli.py", "README.md", "tests/test_normalize.py"],
        target_name="texttools/normalize.py",
    )
    commit_task_state_and_route(planner, session, payload)

    focus = _artifact_scoped_focus(
        session.router_result,
        session,
        "texttools/__init__.py",
        current_content=init_current,
    )

    assert any("normalize_words_keep_case" in item for item in focus["current_write_requirements"])
    assert any("texttools/normalize.py" in item for item in focus["current_write_requirements"])
    assert any("consistent with texttools/normalize.py" in item.lower() for item in focus["current_write_requirements"])


def test_generate_content_prompt_surfaces_cross_file_import_consistency_requirement(tmp_path):
    pkg = tmp_path / "texttools"
    pkg.mkdir()
    init_current = "from .normalize import normalize_words, normalize_words_keep_case\n"
    normalize_current = "def normalize_words(text, *, keep_case=False):\n    return text.split()\n"
    (pkg / "__init__.py").write_text(init_current, encoding="utf-8")
    (pkg / "normalize.py").write_text(normalize_current, encoding="utf-8")
    (tmp_path / "normalize_cli.py").write_text("from texttools import normalize_words\n", encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    snapshot = build_snapshot(tmp_path).model_copy(
        update={
            "important_files": ["texttools/__init__.py", "texttools/normalize.py", "normalize_cli.py"],
            "focus_files": ["texttools/normalize.py", "texttools/__init__.py", "normalize_cli.py"],
            "file_briefs": {
                "texttools/__init__.py": "from .normalize import normalize_words, normalize_words_keep_case",
                "texttools/normalize.py": "def normalize_words(text, *, keep_case=False):",
                "normalize_cli.py": "from texttools import normalize_words",
            },
            "symbol_index": {
                "texttools/normalize.py": ["normalize_words"],
            },
            "import_hotspots": ["texttools/__init__.py", "normalize_cli.py"],
        }
    )
    session = SessionState(
        task="Add keep_case support to the package and CLI without breaking imports.",
        workspace_root=str(tmp_path),
        workspace_snapshot=snapshot,
    )
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Finish the keep-case feature."}],
        target_paths=["texttools/normalize.py", "normalize_cli.py", "README.md", "tests/test_normalize.py"],
        target_name="texttools/normalize.py",
    )
    commit_task_state_and_route(planner, session, payload)

    prompt = generate_content_prompt(
        session.router_result,
        session,
        path="texttools/__init__.py",
        current_content=init_current,
    )

    assert "File-local requirements:" in prompt
    assert "normalize_words_keep_case" in prompt
    assert "texttools/normalize.py" in prompt


def test_artifact_scoped_focus_adds_test_contract_requirement_for_python_module(tmp_path):
    pkg = tmp_path / "texttools"
    pkg.mkdir()
    normalize_current = "def normalize_words(text, *, keep_case=False):\n    return text.lower()\n"
    (pkg / "normalize.py").write_text(normalize_current, encoding="utf-8")
    (pkg / "__init__.py").write_text("from .normalize import normalize_words\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_normalize.py").write_text(
        "import unittest\n\n"
        "from texttools import normalize_words, normalize_words_keep_case\n\n"
        "class TestNormalize(unittest.TestCase):\n"
        "    def test_keep_case_variant_preserves_original_case(self):\n"
        "        self.assertEqual(normalize_words_keep_case('Hello, WORLD!'), 'Hello WORLD')\n",
        encoding="utf-8",
    )

    planner = Planner(ScriptedLLM(), "")
    snapshot = build_snapshot(tmp_path).model_copy(
        update={
            "important_files": ["texttools/normalize.py", "texttools/__init__.py", "tests/test_normalize.py"],
            "focus_files": ["texttools/normalize.py", "texttools/__init__.py"],
            "test_files": ["tests/test_normalize.py"],
            "symbol_index": {
                "texttools/normalize.py": ["normalize_words"],
            },
        }
    )
    session = SessionState(
        task="Add keep_case support to texttools/normalize.py and make the unittest pass.",
        workspace_root=str(tmp_path),
        workspace_snapshot=snapshot,
    )
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Implement the keep-case feature."}],
        target_paths=["texttools/normalize.py", "normalize_cli.py", "README.md", "tests/test_normalize.py"],
        target_name="texttools/normalize.py",
    )
    commit_task_state_and_route(planner, session, payload)

    focus = _artifact_scoped_focus(
        session.router_result,
        session,
        "texttools/normalize.py",
        current_content=normalize_current,
    )

    assert any("tests/test_normalize.py" in item for item in focus["current_write_requirements"])
    assert any("normalize_words_keep_case" in item for item in focus["current_write_requirements"])


def test_generate_content_prompt_adds_test_contract_requirement_for_cli_entrypoint(tmp_path):
    pkg = tmp_path / "texttools"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "from .normalize import normalize_words, normalize_words_keep_case\n",
        encoding="utf-8",
    )
    (pkg / "normalize.py").write_text(
        "def normalize_words(text, *, keep_case=False):\n    return text.lower()\n"
        "def normalize_words_keep_case(text):\n    return text\n",
        encoding="utf-8",
    )
    cli_current = (
        "from texttools import normalize_words\n\n"
        "def main(argv=None):\n"
        "    return normalize_words('Hello, WORLD!')\n"
    )
    (tmp_path / "normalize_cli.py").write_text(cli_current, encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_normalize.py").write_text(
        "import unittest\n\n"
        "from normalize_cli import main\n\n"
        "class TestNormalize(unittest.TestCase):\n"
        "    def test_cli_supports_keep_case_flag(self):\n"
        "        self.assertEqual(main(['--keep-case']), 'Hello WORLD')\n",
        encoding="utf-8",
    )

    planner = Planner(ScriptedLLM(), "")
    snapshot = build_snapshot(tmp_path).model_copy(
        update={
            "important_files": ["normalize_cli.py", "texttools/normalize.py", "tests/test_normalize.py"],
            "focus_files": ["normalize_cli.py", "texttools/normalize.py"],
            "test_files": ["tests/test_normalize.py"],
            "entrypoints": ["normalize_cli.py"],
            "symbol_index": {
                "normalize_cli.py": ["main"],
                "texttools/normalize.py": ["normalize_words", "normalize_words_keep_case"],
            },
        }
    )
    session = SessionState(
        task="Add keep_case support and update the CLI so the targeted unittest passes.",
        workspace_root=str(tmp_path),
        workspace_snapshot=snapshot,
    )
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Implement the keep-case feature."}],
        target_paths=["texttools/normalize.py", "normalize_cli.py", "README.md", "tests/test_normalize.py"],
        target_name="normalize_cli.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="python -m unittest tests.test_normalize",
    )

    prompt = generate_content_prompt(
        session.router_result,
        session,
        path="normalize_cli.py",
        current_content=cli_current,
    )

    assert "File-local requirements:" in prompt
    assert "main(['--keep-case'])" in prompt
    assert "'Hello WORLD'" in prompt


def test_planner_does_not_reject_entrypoint_update_for_helper_signature_literal_scoped_elsewhere(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    main_current = (
        "from .cli import greet\n\n"
        "def main(argv=None):\n"
        "    print(greet('Ada'))\n"
    )
    proposed_main = (
        "from __future__ import annotations\n\n"
        "import argparse\n\n"
        "from .cli import greet\n\n\n"
        "def build_parser() -> argparse.ArgumentParser:\n"
        "    parser = argparse.ArgumentParser(description='Simple greeter')\n"
        "    parser.add_argument('name')\n"
        "    parser.add_argument('--shout', action='store_true')\n"
        "    return parser\n\n\n"
        "def main(argv=None) -> int:\n"
        "    parser = build_parser()\n"
        "    args = parser.parse_args(argv)\n"
        "    print(greet(args.name, shout=args.shout))\n"
        "    return 0\n"
    )

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task=(
            "You are working in an existing Python CLI repo. Keep the default greeting behavior intact. "
            "Extend greet_cli/cli.py so greet(name, shout=False) returns an uppercased greeting when shout=True. "
            "Update greet_cli/__main__.py to add a --shout flag that routes through the helper, and update README.md with the new flag."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Apply the requested CLI update."},
        ],
        target_paths=["greet_cli/__main__.py", "greet_cli/cli.py", "README.md", "tests/test_cli.py"],
        target_name="greet_cli/__main__.py",
    )
    commit_task_state_and_route(planner, session, payload)

    review = planner._explicit_constraint_integrity_review(
        session.router_result,
        session,
        path="greet_cli/__main__.py",
        current_content=main_current,
        proposed_content=proposed_main,
    )

    assert review is None


def test_planner_allows_example_heading_for_requested_readme_example(tmp_path):
    current = "# Word Frequency\n\nCount words from a text file.\n"
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Update README.md to document top_words with a short example.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Document the new helper in README.md."},
        ],
        target_paths=["README.md", "wordfreq.py"],
        target_name="README.md",
    )
    commit_task_state_and_route(planner, session, payload)

    review = planner._explicit_constraint_integrity_review(
        session.router_result,
        session,
        path="README.md",
        current_content=current,
        proposed_content=(
            "# Word Frequency\n\n"
            "Count words from a text file.\n\n"
            "## Example\n\n"
            "`top_words('tests/test_data.txt', limit=2)` returns the most frequent entries.\n"
        ),
    )

    assert review is None


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


def test_assess_effective_mutation_rejects_python_comment_only_change():
    planner = Planner(ScriptedLLM(), "")

    mutation = planner._assess_effective_mutation(
        "app/main.py",
        "def greet(name):\n    return f'Hello, {name}!'\n",
        "def greet(name):  # greeting helper\n    return f'Hello, {name}!'\n",
    )

    assert mutation.effective is False
    assert mutation.reason == "comment-only change"
    assert mutation.before_hash != mutation.after_hash
    assert "comment_only" in mutation.change_labels


def test_assess_effective_mutation_rejects_metadata_only_change():
    planner = Planner(ScriptedLLM(), "")

    mutation = planner._assess_effective_mutation(
        "pyproject.toml",
        "[project]\nname = 'demo'\nversion = '0.1.0'\nrequires-python = '>=3.11'\n",
        "[project]\nname = 'demo'\nversion = '0.1.1'\nrequires-python = '>=3.11'\n",
    )

    assert mutation.effective is False
    assert mutation.reason == "metadata-only change"
    assert mutation.before_hash != mutation.after_hash
    assert "metadata_only" in mutation.change_labels


def test_fallback_semantic_change_review_requires_root_cause_productive_change_and_independent_verification(
    tmp_path,
):
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Repair the CLI bootstrap path.",
        workspace_root=str(tmp_path),
        validation_status="passed",
        edit_generation=1,
    )
    session.changed_files.append(FileChangeRecord(path="app/main.py", operation="modify"))
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_cli",
            kind="test",
            verification_scope="runtime",
            status="passed",
            edit_generation=1,
        )
    )
    session.repair_history.append(
        RepairAttemptRecord(
            artifact_path="app/main.py",
            validation_command="python -m unittest tests.test_cli",
            verification_scope="runtime",
            strategy="validation_targeted",
            result="mutation_planned",
            reason="substantive mutation prepared",
            productive_change=False,
            root_cause_summary=None,
            independent_verification=False,
            behavior_changed=False,
            failure_signature="runtime:import_failure:abc123",
            post_validation_failure_signature="runtime:import_failure:abc123",
        )
    )

    review = planner._fallback_semantic_change_review(session)

    assert review.requirements_satisfied is False
    assert "Concrete root cause explanation for the latest repair" in review.missing_requirements
    assert "Productive code change for the latest repair" in review.missing_requirements
    assert "Independent verification after the latest repair" in review.missing_requirements
    assert any("same failure signature" in issue.lower() for issue in review.suspicious_issues)


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


def test_compact_repair_update_prompt_avoids_cross_file_line_hint_drift(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    main = pkg / "__main__.py"
    current_content = (
        "from __future__ import annotations\n\n"
        "import argparse\n\n"
        "from .cli import greet\n\n"
        "def main(argv=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name')\n"
        "    parser.add_argument('--title')\n"
        "    args = parser.parse_args(argv)\n"
        "    greeting = greet(args.name)\n"
        "    if args.title:\n"
        "        greeting = f\"{args.title}: {greeting}\"\n"
        "    print(greeting)\n"
    )
    main.write_text(current_content, encoding="utf-8")
    cli = pkg / "cli.py"
    cli.write_text("def greet(name):\n    return f'Hello, {name}!'\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_cli.py").write_text("pass\n", encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Fix the title formatting bug in the existing CLI.",
        workspace_root=str(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Repair the failing CLI entrypoint."}],
        target_paths=["greet_cli/__main__.py"],
        target_name="__main__.py",
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
        artifact_paths=["greet_cli/__main__.py", "tests/test_cli.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "Traceback (most recent call last):\n"
            '  File "/tmp/tests/test_cli.py", line 25, in test_title_and_repeat_flags\n'
            '    self.assertEqual(self.run_cli(["Ada", "--title", "Dr.", "--repeat", "2"]), "Dr. Hello, Ada!\\nDr. Hello, Ada!")\n'
            "AssertionError: 'Dr.: Hello, Ada!\\nDr.: Hello, Ada!' != 'Dr. Hello, Ada!\\nDr. Hello, Ada!'\n"
            "- Dr.: Hello, Ada!\n"
            "+ Dr. Hello, Ada!\n"
        ),
        failure_summary="The CLI adds extra punctuation when formatting titled greetings.",
        file_hints=["greet_cli/__main__.py", "tests/test_cli.py"],
        line_hints=[25],
        repair_requirements=[],
        evidence_signature="sig-title-drift",
        repair_brief=RepairBrief(
            failure_type="assertion_mismatch",
            failure_signature="runtime:assertion_mismatch:deadbeef",
            primary_target="greet_cli/__main__.py",
            locked_target="greet_cli/__main__.py",
            expected_semantics=["Validation should produce: Dr. Hello, Ada!"],
            observed_semantics=["Validation currently produces: Dr.: Hello, Ada!"],
            implicated_symbols=[],
            implicated_region_hint="greet_cli/__main__.py",
            repair_constraints=[],
            recent_failed_attempts=[],
            allowed_files=["greet_cli/__main__.py", "greet_cli/cli.py"],
            forbidden_files=["tests/test_cli.py"],
        ),
    )

    prompt = generate_content_prompt(
        session.router_result,
        session,
        path="greet_cli/__main__.py",
        current_content=current_content,
        repair_context=repair_context,
        repair_strategy="validation_targeted",
        mode="compact",
    )

    assert "Change the implicated current lines in greet_cli/__main__.py:" in prompt
    assert "13:     if args.title:" in prompt
    assert '14:         greeting = f"{args.title}: {greeting}"' in prompt
    assert "Change the implicated current lines in greet_cli/__main__.py:\n25:" not in prompt


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


def test_compact_repair_prompt_includes_assertion_diff_lines_for_runtime_mismatch(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    main = pkg / "__main__.py"
    main.write_text(
        "import argparse\nfrom .cli import greet\n\n"
        "def main(argv=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name', nargs='?', default='world')\n"
        "    parser.add_argument('--title', default='')\n"
        "    args = parser.parse_args(argv)\n"
        "    greeting = greet(args.name)\n"
        "    if args.title:\n"
        "        greeting = f'{args.title} {greeting}'\n"
        "    print(greeting)\n",
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
        "    def test_greet_with_title(self, mock_stdout):\n"
        "        from greet_cli import __main__\n"
        "        __main__.main(['Ada', '--title', 'Mr.'])\n"
        "        self.assertEqual(mock_stdout.getvalue().strip(), 'Hello, Mr. Ada!')\n"
    )
    (tests_dir / "test_cli.py").write_text(test_content, encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Repair the CLI greeting formatting.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 4,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py", "README.md"],
                "focus_files": ["greet_cli/__main__.py", "tests/test_cli.py"],
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
                '  File "/tmp/tests/test_cli.py", line 9, in test_greet_with_title\n'
                "    self.assertEqual(mock_stdout.getvalue().strip(), 'Hello, Mr. Ada!')\n"
                "AssertionError: 'Mr. Hello, Ada!' != 'Hello, Mr. Ada!'\n"
                "- Mr. Hello, Ada!\n"
                "+ Hello, Mr. Ada!\n"
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

    assert "Failure focus:" in prompt
    assert "AssertionError: 'Mr. Hello, Ada!' != 'Hello, Mr. Ada!'" in prompt
    assert "- Mr. Hello, Ada!" in prompt
    assert "+ Hello, Mr. Ada!" in prompt


def test_compact_repair_prompt_includes_repair_brief_semantics_and_file_constraints(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    main = pkg / "__main__.py"
    main.write_text(
        "def main(argv=None):\n"
        "    greeting = 'Mr. Hello, Ada!'\n"
        "    print(greeting)\n",
        encoding="utf-8",
    )
    session = SessionState(
        task="Repair the CLI greeting formatting.",
        workspace_root=str(tmp_path),
    )
    planner = Planner(ScriptedLLM(), "")
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
        summary="Validation command exited with 1.",
        excerpt="AssertionError: 'Mr. Hello, Ada!' != 'Hello, Mr. Ada!'",
        failure_summary="The CLI output ordering is wrong for the title-aware greeting.",
        file_hints=["greet_cli/__main__.py", "tests/test_cli.py", "README.md"],
        repair_requirements=["Change greet_cli/__main__.py so the failing runtime path can complete successfully."],
        evidence_signature="sig-repair-brief",
        repair_brief=RepairBrief(
            failure_type="assertion_mismatch",
            failure_signature="runtime:assertion_mismatch:abcd1234",
            primary_target="greet_cli/__main__.py",
            locked_target="greet_cli/__main__.py",
            expected_semantics=["Validation should produce: Hello, Mr. Ada!"],
            observed_semantics=["Validation currently produces: Mr. Hello, Ada!"],
            implicated_symbols=["main"],
            implicated_region_hint="greet_cli/__main__.py:line 2",
            repair_constraints=["Change greet_cli/__main__.py so the failing runtime path can complete successfully."],
            recent_failed_attempts=[
                RepairAttemptSummary(
                    target="greet_cli/__main__.py",
                    strategy="validation_targeted",
                    result="no_effective_change",
                    reason="The generated edit left the implicated behavior unchanged.",
                )
            ],
            allowed_files=["greet_cli/__main__.py", "greet_cli/cli.py"],
            forbidden_files=["README.md", "tests/test_cli.py"],
        ),
    )
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Repair the failing CLI entrypoint."}],
        target_paths=["greet_cli/__main__.py"],
        target_name="__main__.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli and fix the failing CLI behavior.",
    )

    prompt = generate_content_prompt(
        session.router_result,
        session,
        path="greet_cli/__main__.py",
        current_content=main.read_text(encoding="utf-8"),
        repair_context=repair_context,
        repair_strategy="validation_escalated",
        mode="compact",
    )

    assert "Primary repair target: greet_cli/__main__.py" in prompt
    assert "Expected semantics: Validation should produce: Hello, Mr. Ada!" in prompt
    assert "Observed semantics: Validation currently produces: Mr. Hello, Ada!" in prompt
    assert "Minimal semantic delta:" in prompt
    assert "Allowed repair files: greet_cli/__main__.py, greet_cli/cli.py" in prompt
    assert "Avoid drifting into other files without strong new evidence: README.md, tests/test_cli.py" in prompt
    assert "Recent failed repair attempts:" in prompt


def test_compact_repair_prompt_surfaces_minimal_semantic_delta_for_behavior_mismatch(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    current_content = (
        "from __future__ import annotations\n\n"
        "import argparse\n\n"
        "from .cli import greet\n\n"
        "def main(argv=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name')\n"
        "    parser.add_argument('--title')\n"
        "    args = parser.parse_args(argv)\n"
        "    greeting = greet(args.name)\n"
        "    if args.title:\n"
        "        greeting = f\"{args.title}: {greeting}\"\n"
        "    print(greeting)\n"
    )
    (pkg / "__main__.py").write_text(current_content, encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Fix the title formatting bug in the existing CLI.",
        workspace_root=str(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Repair the failing CLI entrypoint."}],
        target_paths=["greet_cli/__main__.py"],
        target_name="__main__.py",
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
        artifact_paths=["greet_cli/__main__.py", "tests/test_cli.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "AssertionError: 'Dr.: Hello, Ada!\\nDr.: Hello, Ada!' != 'Dr. Hello, Ada!\\nDr. Hello, Ada!'\n"
            "- Dr.: Hello, Ada!\n"
            "+ Dr. Hello, Ada!\n"
        ),
        failure_summary="The CLI adds extra punctuation when formatting titled greetings.",
        file_hints=["greet_cli/__main__.py", "tests/test_cli.py"],
        line_hints=[25],
        repair_requirements=[],
        evidence_signature="sig-title-delta",
        repair_brief=RepairBrief(
            failure_type="assertion_mismatch",
            failure_signature="runtime:assertion_mismatch:delta1234",
            primary_target="greet_cli/__main__.py",
            locked_target="greet_cli/__main__.py",
            expected_semantics=["Validation should produce: Dr. Hello, Ada!\nDr. Hello, Ada!"],
            observed_semantics=["Validation currently produces: Dr.: Hello, Ada!\nDr.: Hello, Ada!"],
            implicated_symbols=[],
            implicated_region_hint="greet_cli/__main__.py",
            repair_constraints=[],
            recent_failed_attempts=[],
            allowed_files=["greet_cli/__main__.py", "greet_cli/cli.py"],
            forbidden_files=["tests/test_cli.py"],
        ),
    )

    prompt = generate_content_prompt(
        session.router_result,
        session,
        path="greet_cli/__main__.py",
        current_content=current_content,
        repair_context=repair_context,
        repair_strategy="validation_targeted",
        mode="compact",
    )

    assert (
        "Minimal semantic delta: Remove observed-only text ':' between shared prefix 'Dr.' and shared suffix 'Hello, Ada!'."
        in prompt
    )
    assert "Change the implicated current lines in greet_cli/__main__.py:" in prompt
    assert "13:     if args.title:" in prompt
    assert '14:         greeting = f"{args.title}: {greeting}"' in prompt
    assert (
        "Apply this exact semantic delta in the behavior produced by this file: Remove observed-only text ':' between shared prefix 'Dr.' and shared suffix 'Hello, Ada!'."
        in prompt
    )


def test_runtime_repair_prefers_behavioral_local_anchor_over_wrapper_line_for_semantic_delta(tmp_path):
    pkg = tmp_path / "texttools"
    pkg.mkdir()
    current_content = (
        "def normalize_words(text: str, keep_case: bool = False) -> str:\n"
        "    if keep_case:\n"
        "        return ' '.join(text.replace(',', ' ').split())\n"
        "    else:\n"
        "        return ' '.join(text.replace(',', ' ').split()).lower()\n\n"
        "def normalize_words_keep_case(text: str) -> str:\n"
        "    return normalize_words(text, keep_case=True)\n"
    )
    (pkg / "normalize.py").write_text(current_content, encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_normalize.py").write_text(
        "from texttools.normalize import normalize_words, normalize_words_keep_case\n"
        "from normalize_cli import main\n",
        encoding="utf-8",
    )

    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_normalize",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["texttools/normalize.py", "tests/test_normalize.py", "normalize_cli.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "AssertionError: 'Hello WORLD!' != 'Hello WORLD'\n"
            "- Hello WORLD!\n"
            "+ Hello WORLD\n"
        ),
        failure_summary="texttools/normalize.py still produces the wrong behavior: expected Hello WORLD but observed Hello WORLD!.",
        file_hints=["texttools/normalize.py", "tests/test_normalize.py", "normalize_cli.py"],
        line_hints=[9, 12, 15],
        repair_requirements=[],
        evidence_signature="sig-normalize-semantic-anchor",
        repair_brief=RepairBrief(
            failure_type="assertion_mismatch",
            failure_signature="runtime:assertion_mismatch:normalize1234",
            primary_target="texttools/normalize.py",
            locked_target="texttools/normalize.py",
            expected_semantics=["Validation should produce: Hello WORLD"],
            observed_semantics=["Validation currently produces: Hello WORLD!"],
            implicated_symbols=[
                "test_cli_supports_keep_case_flag",
                "test_default_normalization_lowercases_words",
                "test_keep_case_variant_preserves_original_case",
                "assertEqual",
                "normalize_words_keep_case",
            ],
            implicated_region_hint="",
            repair_constraints=[],
            recent_failed_attempts=[],
            allowed_files=["texttools/normalize.py", "normalize_cli.py", "texttools/__init__.py"],
            forbidden_files=["tests/test_normalize.py", "README.md"],
        ),
    )

    hints = _repair_target_line_hints(
        path="texttools/normalize.py",
        current_content=current_content,
        repair_context=repair_context,
    )

    assert 3 in hints
    assert 5 in hints
    assert hints.index(3) < hints.index(8) if 8 in hints else True


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


def test_compact_repair_prompt_scopes_multi_file_website_failures_per_target(tmp_path):
    projects = tmp_path / "projects.html"
    projects.write_text(
        "<!doctype html>\n<html><body><main><h1>Projects</h1></main></body></html>\n",
        encoding="utf-8",
    )
    contact = tmp_path / "contact.html"
    contact.write_text(
        "<!doctype html>\n<html><body><main><form></form></main></body></html>\n",
        encoding="utf-8",
    )
    styles = tmp_path / "styles.css"
    styles.write_text("body { margin: 0; }\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_content = (
        "import unittest\n\n"
        "class TestSite(unittest.TestCase):\n"
        "    def read(self, name):\n"
        "        with open(name, 'r', encoding='utf-8') as handle:\n"
        "            return handle.read()\n\n"
        "    def test_projects_page(self):\n"
        "        html = self.read('projects.html')\n"
        "        self.assertIn('Case Studies', html)\n\n"
        "    def test_contact_page(self):\n"
        "        html = self.read('contact.html')\n"
        "        self.assertIn('class=\"contact-form\"', html)\n\n"
        "    def test_stylesheet_layout(self):\n"
        "        css = self.read('styles.css')\n"
        "        self.assertIn('.site-grid', css)\n"
    )
    (tests_dir / "test_site.py").write_text(test_content, encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Repair the portfolio website so the targeted page-level validation passes.",
        workspace_root=str(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Repair the targeted website file."}],
        target_paths=["projects.html"],
        target_name="projects.html",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_site and repair the failing website behavior.",
    )

    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_site",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["projects.html", "contact.html", "styles.css", "tests/test_site.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "FAIL: test_contact_page (tests.test_site.TestSite.test_contact_page)\n"
            '  File "/tmp/tests/test_site.py", line 12, in test_contact_page\n'
            "    self.assertIn('class=\"contact-form\"', self.read('contact.html'))\n"
            "AssertionError: 'class=\"contact-form\"' not found in '<form></form>'\n"
            "\n"
            "FAIL: test_projects_page (tests.test_site.TestSite.test_projects_page)\n"
            '  File "/tmp/tests/test_site.py", line 8, in test_projects_page\n'
            "    self.assertIn('Case Studies', self.read('projects.html'))\n"
            "AssertionError: 'Case Studies' not found in '<h1>Projects</h1>'\n"
            "\n"
            "FAIL: test_stylesheet_layout (tests.test_site.TestSite.test_stylesheet_layout)\n"
            '  File "/tmp/tests/test_site.py", line 16, in test_stylesheet_layout\n'
            "    self.assertIn('.site-grid', self.read('styles.css'))\n"
            "AssertionError: '.site-grid' not found in 'body { margin: 0; }'\n"
        ),
        failure_summary="The generated portfolio site still misses required page-specific and stylesheet content.",
        file_hints=["projects.html", "contact.html", "styles.css", "tests/test_site.py"],
        repair_requirements=[],
        evidence_signature="sig-website-scoped-repair",
        repair_brief=RepairBrief(
            failure_type="assertion_mismatch",
            failure_signature="runtime:assertion_mismatch:website1234",
            primary_target="projects.html",
            locked_target="projects.html",
            expected_semantics=["projects.html should include Case Studies."],
            observed_semantics=["projects.html currently omits Case Studies."],
            implicated_symbols=[],
            implicated_region_hint="projects.html",
            repair_constraints=["Only repair the requested target file for this write step."],
            recent_failed_attempts=[],
            allowed_files=["projects.html"],
            forbidden_files=["contact.html", "styles.css"],
        ),
    )

    prompt = generate_content_prompt(
        session.router_result,
        session,
        path="projects.html",
        current_content=projects.read_text(encoding="utf-8"),
        repair_context=repair_context,
        repair_strategy="validation_targeted",
        mode="compact",
    )

    assert "Case Studies" in prompt
    assert "self.assertIn('Case Studies', self.read('projects.html'))" in prompt
    assert 'class="contact-form"' not in prompt
    assert ".site-grid" not in prompt
    assert "self.read('contact.html')" not in prompt
    assert "self.read('styles.css')" not in prompt


def test_compact_repair_prompt_adds_exact_literal_anchor_for_quote_variant(tmp_path):
    about = tmp_path / "about.html"
    about.write_text(
        "<!doctype html>\n<html><body><header class='about-hero'></header></body></html>\n",
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_site.py").write_text("pass\n", encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Repair the portfolio about page so the targeted page-level validation passes.",
        workspace_root=str(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Repair the targeted website file."}],
        target_paths=["about.html"],
        target_name="about.html",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_site and repair the failing website behavior.",
    )

    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_site",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["about.html", "tests/test_site.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "FAIL: test_about_page (tests.test_site.TestSite.test_about_page)\n"
            '  File "/tmp/tests/test_site.py", line 8, in test_about_page\n'
            "    self.assertIn('class=\"about-hero\"', self.read('about.html'))\n"
            "AssertionError: 'class=\"about-hero\"' not found in \"<header class='about-hero'></header>\"\n"
        ),
        failure_summary="The about page still misses the required hero marker.",
        file_hints=["about.html", "tests/test_site.py"],
        repair_requirements=[],
        evidence_signature="sig-website-literal-anchor",
    )

    anchors = _repair_required_literal_anchors(
        path="about.html",
        current_content=about.read_text(encoding="utf-8"),
        repair_context=repair_context,
    )

    assert anchors
    assert "exact required literal" in anchors[0]
    assert "near-match literal" in anchors[0]
    assert 'class="about-hero"' in anchors[0]
    assert "class='about-hero'" in anchors[0]


def test_compact_repair_prompt_adds_count_based_literal_anchor(tmp_path):
    projects = tmp_path / "projects.html"
    projects.write_text(
        "<!doctype html>\n<html><body><div class='project-card'></div></body></html>\n",
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_site.py").write_text("pass\n", encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Repair the projects page so the project cards satisfy validation.",
        workspace_root=str(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Repair the targeted website file."}],
        target_paths=["projects.html"],
        target_name="projects.html",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_site and repair the failing website behavior.",
    )

    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_site",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["projects.html", "tests/test_site.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "FAIL: test_projects_page (tests.test_site.TestSite.test_projects_page)\n"
            '  File "/tmp/tests/test_site.py", line 12, in test_projects_page\n'
            "    self.assertGreaterEqual(self.read('projects.html').count('class=\"project-card\"'), 3)\n"
            "AssertionError: 1 not greater than or equal to 3\n"
        ),
        failure_summary="The projects page still exposes too few project cards.",
        file_hints=["projects.html", "tests/test_site.py"],
        repair_requirements=[],
        evidence_signature="sig-website-count-anchor",
    )

    anchors = _repair_required_literal_anchors(
        path="projects.html",
        current_content=projects.read_text(encoding="utf-8"),
        repair_context=repair_context,
    )

    assert anchors
    assert "appears at least 3 times" in anchors[0]
    assert 'class="project-card"' in anchors[0]
    assert "class='project-card'" in anchors[0]


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


def test_targeted_runtime_prompt_hints_prioritize_bootstrap_for_direct_script_execution():
    hints = _targeted_runtime_prompt_hints(
        path="scripts/build_duplicates.py",
        current_content=(
            "import sys\n"
            "\n"
            "from wordaudit import duplicate_words\n"
            "\n"
            "\n"
            "def main(argv=None):\n"
            "    argv = list(sys.argv[1:] if argv is None else argv)\n"
            "    with open(argv[0], 'r', encoding='utf-8') as handle:\n"
            "        duplicates = duplicate_words(handle.readlines())\n"
            "    for word in duplicates:\n"
            "        print(word)\n"
        ),
        supporting_context=(
            "tests/test_report.py:\n"
            '10: SCRIPT = ROOT / "scripts" / "build_duplicates.py"\n'
        ),
        targeted_context={
            "failure_summary": (
                "subprocess.CalledProcessError: "
                "Command '['/usr/bin/python3', '/tmp/demo/scripts/build_duplicates.py', '/tmp/tmpwords.txt']' "
                "returned non-zero exit status 1."
            ),
            "excerpt": (
                "subprocess.CalledProcessError: "
                "Command '['/usr/bin/python3', '/tmp/demo/scripts/build_duplicates.py', '/tmp/tmpwords.txt']' "
                "returned non-zero exit status 1."
            ),
            "failure_focus": [],
            "file_hints": ["tests/test_report.py", "wordaudit/report.py"],
        },
    )

    assert any("runs this file directly by script path" in hint for hint in hints)
    assert any("script directory is guaranteed on sys.path" in hint for hint in hints)
    assert any("cannot fix an import failure here" in hint for hint in hints)
    assert any("place it above the current top-level project import" in hint for hint in hints)


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


def test_targeted_runtime_failure_focus_includes_assertion_diff_lines():
    focus = _targeted_runtime_failure_focus_lines(
        (
            "FAIL: test_greet_with_title (tests.test_cli.TestCLI.test_greet_with_title)\n"
            '  File "/home/marc/workspace/e2e-python-cli/tests/test_cli.py", line 12, in test_greet_with_title\n'
            "    self.assertEqual(mock_stdout.getvalue().strip(), 'Hello, Mr. Ada!')\n"
            "AssertionError: 'Mr. Hello, Ada!' != 'Hello, Mr. Ada!'\n"
            "- Mr. Hello, Ada!\n"
            "+ Hello, Mr. Ada!\n"
            "?     -------\n"
        ),
        target_path="greet_cli/__main__.py",
        limit=6,
    )

    assert "AssertionError: 'Mr. Hello, Ada!' != 'Hello, Mr. Ada!'" in focus
    assert "- Mr. Hello, Ada!" in focus
    assert "+ Hello, Mr. Ada!" in focus


def test_targeted_runtime_failure_focus_scopes_multi_file_website_failures_to_target():
    focus = _targeted_runtime_failure_focus_lines(
        (
            "FAIL: test_contact_page (tests.test_site.TestSite.test_contact_page)\n"
            '  File "/tmp/tests/test_site.py", line 12, in test_contact_page\n'
            "    self.assertIn('class=\"contact-form\"', self.read('contact.html'))\n"
            "AssertionError: 'class=\"contact-form\"' not found in '<form></form>'\n"
            "\n"
            "FAIL: test_projects_page (tests.test_site.TestSite.test_projects_page)\n"
            '  File "/tmp/tests/test_site.py", line 8, in test_projects_page\n'
            "    self.assertIn('Case Studies', self.read('projects.html'))\n"
            "AssertionError: 'Case Studies' not found in '<h1>Projects</h1>'\n"
            "\n"
            "FAIL: test_stylesheet_layout (tests.test_site.TestSite.test_stylesheet_layout)\n"
            '  File "/tmp/tests/test_site.py", line 16, in test_stylesheet_layout\n'
            "    self.assertIn('.site-grid', self.read('styles.css'))\n"
            "AssertionError: '.site-grid' not found in 'body { margin: 0; }'\n"
        ),
        target_path="projects.html",
        other_paths=["contact.html", "styles.css", "tests/test_site.py"],
        limit=6,
    )

    assert any("Case Studies" in line for line in focus)
    assert all('class="contact-form"' not in line for line in focus)
    assert all(".site-grid" not in line for line in focus)


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


def test_planner_no_effective_change_keeps_locked_primary_target_before_pivot():
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Repair the CLI greeting formatting.",
        workspace_root="/tmp/demo",
    )
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
        summary="Validation command exited with 1.",
        excerpt="AssertionError: 'Mr. Hello, Ada!' != 'Hello, Mr. Ada!'",
        failure_summary="The CLI output ordering is wrong for the title-aware greeting.",
        file_hints=["greet_cli/__main__.py", "tests/test_cli.py"],
        repair_requirements=["Change greet_cli/__main__.py so the failing runtime path can complete successfully."],
        evidence_signature="sig-runtime-locked",
        repair_brief=RepairBrief(
            failure_type="assertion_mismatch",
            failure_signature="runtime:assertion_mismatch:locked1234",
            primary_target="greet_cli/__main__.py",
            locked_target="greet_cli/__main__.py",
            expected_semantics=["Validation should produce: Hello, Mr. Ada!"],
            observed_semantics=["Validation currently produces: Mr. Hello, Ada!"],
            implicated_symbols=["main"],
            implicated_region_hint="greet_cli/__main__.py:line 7",
            allowed_files=["greet_cli/__main__.py", "greet_cli/cli.py"],
            forbidden_files=["tests/test_cli.py"],
        ),
    )

    session.repair_history.append(
        RepairAttemptRecord(
            artifact_path="greet_cli/__main__.py",
            validation_command=repair_context.command,
            verification_scope="runtime",
            strategy="validation_targeted",
            result="no_effective_change",
            reason="identical content",
            failure_signature="runtime:assertion_mismatch:locked1234",
        )
    )

    assert planner._should_pivot_after_no_effective_change(session, "greet_cli/__main__.py", repair_context) is False

    session.repair_history.append(
        RepairAttemptRecord(
            artifact_path="greet_cli/__main__.py",
            validation_command=repair_context.command,
            verification_scope="runtime",
            strategy="validation_escalated",
            result="no_effective_change",
            reason="same failure signature after edit",
            failure_signature="runtime:assertion_mismatch:locked1234",
        )
    )

    assert planner._should_pivot_after_no_effective_change(session, "greet_cli/__main__.py", repair_context) is True


def test_planner_direct_script_wrapper_pivots_after_first_no_effective_change():
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Repair the duplicate-word reporting flow.",
        workspace_root="/tmp/demo",
    )
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_report",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["scripts/build_duplicates.py", "wordaudit/report.py", "tests/test_report.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "AssertionError: Lists differ: ['# skipped', 'alpha', 'beta'] != ['alpha', 'beta']\n"
            "subprocess.CalledProcessError: "
            "Command '['/usr/bin/python3', '/tmp/demo/scripts/build_duplicates.py', '/tmp/tmpwords.txt']' "
            "returned non-zero exit status 1.\n"
        ),
        failure_summary="The library behavior is still wrong and the direct script wrapper now exits non-zero.",
        file_hints=["scripts/build_duplicates.py", "wordaudit/report.py", "tests/test_report.py"],
        repair_requirements=["Change the library behavior before rechecking the wrapper script."],
        evidence_signature="sig-runtime-direct-script-wrapper",
        repair_brief=RepairBrief(
            failure_type="assertion_mismatch",
            failure_signature="runtime:assertion_mismatch:directscriptpivot",
            primary_target="scripts/build_duplicates.py",
            locked_target="scripts/build_duplicates.py",
            expected_semantics=["Validation should produce: ['alpha', 'beta']"],
            observed_semantics=["Validation currently produces: ['# skipped', 'alpha', 'beta']"],
            allowed_files=["scripts/build_duplicates.py", "wordaudit/report.py"],
            forbidden_files=["tests/test_report.py"],
        ),
    )

    session.repair_history.append(
        RepairAttemptRecord(
            artifact_path="scripts/build_duplicates.py",
            validation_command=repair_context.command,
            verification_scope="runtime",
            strategy="validation_targeted",
            result="no_effective_change",
            reason="identical wrapper rewrite",
            failure_signature="runtime:assertion_mismatch:directscriptpivot",
        )
    )

    assert planner._should_pivot_after_no_effective_change(session, "scripts/build_duplicates.py", repair_context) is True


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


def test_nonblocking_update_target_deferral_is_honored_before_any_files_change(tmp_path, monkeypatch):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    (pkg / "__main__.py").write_text(
        "import argparse\nfrom .cli import greet\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name', nargs='?', default='world')\n"
        "    args = parser.parse_args()\n"
        "    print(greet(args.name))\n",
        encoding="utf-8",
    )
    (pkg / "cli.py").write_text("def greet(name):\n    return f'Hello, {name}!'\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# greet-cli\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_cli.py").write_text("def test_cli():\n    assert True\n", encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task=(
            "Extend the CLI with --prefix and --repeat, keep --uppercase working, "
            "update the README, and extend the unittest module."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 4,
                "important_files": [
                    "greet_cli/__main__.py",
                    "greet_cli/cli.py",
                    "README.md",
                    "tests/test_cli.py",
                ],
                "focus_files": ["greet_cli/__main__.py", "greet_cli/cli.py"],
                "manifests": ["README.md"],
                "test_files": ["tests/test_cli.py"],
                "entrypoints": ["greet_cli/__main__.py"],
                "repo_summary": "Small Python CLI project with entrypoint, helper, README, and tests.",
            }
        ),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Apply the requested feature update."},
            {"step": 2, "action": "run_validation", "reason": "Validate the result."},
        ],
        target_paths=[
            "greet_cli/__main__.py",
            "greet_cli/cli.py",
            "README.md",
            "tests/test_cli.py",
        ],
        target_name="greet_cli/__main__.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="python -m unittest tests.test_cli",
    )

    def fake_execute_action_from_plan(_route, _session):
        return AgentDecision(
            thought_summary="Continue with the next explicit target.",
            action_type=AgentActionType.CALL_TOOL,
            tool_name="read_file",
            tool_args={"path": planner._next_update_target(_route, _session)},
            expected_outcome="Inspect the next file that still needs an update.",
            final_response=None,
        )

    monkeypatch.setattr(planner, "execute_action_from_plan", fake_execute_action_from_plan)

    decision = planner._continue_after_nonblocking_update_target_failure(
        session.router_result,
        session,
        target="greet_cli/__main__.py",
        stop_reason="update_review_rejected",
        repair_context=None,
    )

    assert decision is not None
    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "read_file"
    assert decision.tool_args["path"] == "greet_cli/cli.py"
    assert f"{DEFERRED_UPDATE_TARGET_NOTE_PREFIX}greet_cli/__main__.py" in session.notes
    assert planner._has_pending_explicit_update_targets(session.router_result, session) is True


def test_planner_no_effective_change_prefers_code_candidate_before_supporting_docs(tmp_path, monkeypatch):
    package_dir = tmp_path / "texttools"
    package_dir.mkdir()
    (package_dir / "normalize.py").write_text("def normalize_words(text):\n    return text.split()\n", encoding="utf-8")
    (tmp_path / "normalize_cli.py").write_text(
        "from texttools.normalize import normalize_words\n\n"
        "def main(argv=None):\n"
        "    return normalize_words('Hello WORLD')\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# texttools\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_normalize.py").write_text("def test_normalize():\n    assert True\n", encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task=(
            "Add a keep_case option to texttools/normalize.py, support it in normalize_cli.py, "
            "update README.md if needed, and run python -m unittest tests.test_normalize."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 4,
                "important_files": [
                    "texttools/normalize.py",
                    "normalize_cli.py",
                    "README.md",
                    "tests/test_normalize.py",
                ],
                "focus_files": ["texttools/normalize.py", "normalize_cli.py"],
                "manifests": ["README.md"],
                "test_files": ["tests/test_normalize.py"],
                "repo_summary": "Small text normalization project with a helper module, CLI wrapper, README, and tests.",
            }
        ),
        candidate_files=[
            "tests/test_normalize.py",
            "texttools/normalize.py",
            "normalize_cli.py",
            "README.md",
        ],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Implement the keep_case feature."},
            {"step": 2, "action": "update_artifact", "reason": "Update the related CLI wrapper and docs."},
            {"step": 3, "action": "run_validation", "reason": "Run the requested unittest module."},
        ],
        target_paths=[
            "texttools/normalize.py",
            "README.md",
            "tests/test_normalize.py",
        ],
        target_name="texttools/normalize.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="python -m unittest tests.test_normalize",
    )
    session.candidate_files = [
        "tests/test_normalize.py",
        "texttools/normalize.py",
        "normalize_cli.py",
        "README.md",
    ]
    session.task_state.target_artifacts = [
        TaskArtifact(path="texttools/normalize.py", name="normalize_words", kind=".py", role="primary_target", confidence=0.99),
        TaskArtifact(path="README.md", name="README.md", kind=".md", role="supporting_context", confidence=0.9),
        TaskArtifact(path="tests/test_normalize.py", name="test_normalize.py", kind="test", role="validation_target", confidence=0.9),
    ]

    def fake_execute_action_from_plan(_route, _session):
        return AgentDecision(
            thought_summary="Continue with the next relevant update target.",
            action_type=AgentActionType.CALL_TOOL,
            tool_name="read_file",
            tool_args={"path": planner._next_update_target(_route, _session)},
            expected_outcome="Inspect the next file that still needs a technical update.",
            final_response=None,
        )

    monkeypatch.setattr(planner, "execute_action_from_plan", fake_execute_action_from_plan)

    decision = planner._continue_after_nonblocking_update_target_failure(
        session.router_result,
        session,
        target="texttools/normalize.py",
        stop_reason="no_effective_change",
        repair_context=None,
    )

    assert decision is not None
    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "read_file"
    assert decision.tool_args["path"] == "normalize_cli.py"
    assert planner._next_update_target(session.router_result, session) == "normalize_cli.py"


def test_planner_keeps_remaining_explicit_support_target_instead_of_broadening_after_code_updates(tmp_path):
    package_dir = tmp_path / "texttools"
    package_dir.mkdir()
    (package_dir / "normalize.py").write_text(
        "def normalize_words(text, *, lowercase=True, keep_case=False):\n    return text.split()\n",
        encoding="utf-8",
    )
    (package_dir / "__init__.py").write_text(
        "from .normalize import normalize_words\n",
        encoding="utf-8",
    )
    (tmp_path / "normalize_cli.py").write_text(
        "from texttools import normalize_words\n\n"
        "def main(argv=None):\n"
        "    return normalize_words('Hello WORLD')\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# texttools\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_normalize.py").write_text("def test_normalize():\n    assert True\n", encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task=(
            "Add a keep_case option to texttools/normalize.py, support it in normalize_cli.py, "
            "update README.md if needed, and run python -m unittest tests.test_normalize."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 5,
                "important_files": [
                    "texttools/normalize.py",
                    "normalize_cli.py",
                    "texttools/__init__.py",
                    "README.md",
                    "tests/test_normalize.py",
                ],
                "focus_files": ["texttools/normalize.py", "normalize_cli.py"],
                "manifests": ["README.md"],
                "test_files": ["tests/test_normalize.py"],
                "entrypoints": ["normalize_cli.py"],
                "repo_summary": "Small text normalization project with helper module, package export, CLI wrapper, README, and tests.",
            }
        ),
        candidate_files=[
            "texttools/normalize.py",
            "normalize_cli.py",
            "texttools/__init__.py",
            "README.md",
            "tests/test_normalize.py",
        ],
        changed_files=[
            FileChangeRecord(path="texttools/normalize.py", operation="modify", diff="+ keep_case\n"),
            FileChangeRecord(path="normalize_cli.py", operation="modify", diff="+ --keep-case\n"),
        ],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Implement the keep_case feature."},
            {"step": 2, "action": "update_artifact", "reason": "Update the related CLI wrapper and docs."},
            {"step": 3, "action": "run_validation", "reason": "Run the requested unittest module."},
        ],
        target_paths=[
            "texttools/normalize.py",
            "normalize_cli.py",
            "README.md",
            "tests/test_normalize.py",
        ],
        target_name="texttools/normalize.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="python -m unittest tests.test_normalize",
    )
    session.task_state.target_artifacts = [
        TaskArtifact(path="texttools/normalize.py", name="normalize.py", kind=".py", role="primary_target", confidence=0.99),
        TaskArtifact(path="normalize_cli.py", name="normalize_cli.py", kind=".py", role="primary_target", confidence=0.98),
        TaskArtifact(path="README.md", name="README.md", kind=".md", role="supporting_context", confidence=0.9),
        TaskArtifact(path="tests/test_normalize.py", name="test_normalize.py", kind="test", role="validation_target", confidence=0.9),
    ]

    assert planner._ordered_remaining_update_targets(session.router_result, session) == ["README.md"]
    assert planner._next_update_target(session.router_result, session) == "README.md"


def test_planner_keeps_review_blocked_pending_target_deferred_during_runtime_repair(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    (pkg / "__main__.py").write_text(
        "from __future__ import annotations\n\n"
        "import argparse\n\n"
        "from .cli import greet\n\n\n"
        "def main(argv=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name')\n"
        "    parser.add_argument('--uppercase', action='store_true')\n"
        "    args = parser.parse_args(argv)\n\n"
        "    greeting = greet(args.name)\n"
        "    if args.uppercase:\n"
        "        greeting = greeting.upper()\n"
        "    print(greeting)\n",
        encoding="utf-8",
    )
    (pkg / "cli.py").write_text(
        "def greet(name: str) -> str:\n"
        "    return f\"Hello, {name}!\"\n",
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_cli.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# greet_cli\n", encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task=(
            "Extend this existing CLI so it supports an optional --prefix TEXT flag and an optional "
            "--repeat N flag while keeping the existing --uppercase behavior working."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 4,
                "important_files": [
                    "greet_cli/__main__.py",
                    "greet_cli/cli.py",
                    "README.md",
                    "tests/test_cli.py",
                ],
                "focus_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "manifests": ["README.md"],
                "test_files": ["tests/test_cli.py"],
                "entrypoints": ["greet_cli/__main__.py"],
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
        notes=[f"{DEFERRED_UPDATE_TARGET_NOTE_PREFIX}greet_cli/cli.py"],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Apply the requested feature update."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=[
            "greet_cli/__main__.py",
            "greet_cli/cli.py",
            "README.md",
            "tests/test_cli.py",
        ],
        target_name="greet_cli/__main__.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli after updating the CLI.",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="greet_cli/__main__.py", operation="write"),
            FileChangeRecord(path="README.md", operation="write"),
            FileChangeRecord(path="tests/test_cli.py", operation="write"),
        ]
    )
    failed_run = ValidationRunRecord(
        command="python -m unittest tests.test_cli",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=1,
        iteration=8,
        summary="Validation command exited with 1.",
        excerpt=(
            ".FFFFF.\n"
            "======================================================================\n"
            "FAIL: test_prefix_flag (tests.test_cli.TestCLI.test_prefix_flag)\n"
            "Traceback (most recent call last):\n"
            f'  File "{tmp_path / "tests" / "test_cli.py"}", line 21, in test_prefix_flag\n'
            "    self.assertEqual(self.run_cli(['--prefix', 'Mr.', 'Ada']), 'Hello, Mr. Ada!')\n"
            "AssertionError: 'Mr. Hello, Ada!' != 'Hello, Mr. Ada!'\n"
            f'  File "{tmp_path / "greet_cli" / "__main__.py"}", line 14, in main\n'
            "    print(greeting * args.repeat)\n"
        ),
    )
    session.validation_runs.append(failed_run)
    repair_context = planner.validation_planner.build_failure_evidence(session, failed_run)

    assert repair_context.artifact_paths[0] == "greet_cli/__main__.py"
    assert session.active_repair_context is None
    assert planner._active_deferred_update_targets(session) == {"greet_cli/cli.py"}
    assert planner._has_pending_explicit_update_targets(session.router_result, session) is False

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "read_file"
    assert decision.tool_args["path"] == "greet_cli/__main__.py"


def test_planner_runtime_repair_ignores_test_cli_substring_when_repairing_entrypoint(tmp_path):
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    (pkg / "__main__.py").write_text(
        "from __future__ import annotations\n\n"
        "import argparse\n\n"
        "from .cli import greet\n\n\n"
        "def main(argv=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name')\n"
        "    parser.add_argument('--uppercase', action='store_true')\n"
        "    args = parser.parse_args(argv)\n\n"
        "    greeting = greet(args.name)\n"
        "    if args.uppercase:\n"
        "        greeting = greeting.upper()\n"
        "    print(greeting)\n",
        encoding="utf-8",
    )
    (pkg / "cli.py").write_text(
        "def greet(name: str) -> str:\n"
        "    return f\"Hello, {name}!\"\n",
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_cli.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# greet_cli\n", encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task=(
            "Extend this existing CLI so it supports an optional --prefix TEXT flag and an optional "
            "--repeat N flag while keeping the existing --uppercase behavior working."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 4,
                "important_files": [
                    "greet_cli/__main__.py",
                    "greet_cli/cli.py",
                    "README.md",
                    "tests/test_cli.py",
                ],
                "focus_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "manifests": ["README.md"],
                "test_files": ["tests/test_cli.py"],
                "entrypoints": ["greet_cli/__main__.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
        validation_status="failed",
        edit_generation=3,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_cli",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_cli"],
        notes=[f"{DEFERRED_UPDATE_TARGET_NOTE_PREFIX}greet_cli/cli.py"],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Apply the requested feature update."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=[
            "greet_cli/__main__.py",
            "greet_cli/cli.py",
            "README.md",
            "tests/test_cli.py",
        ],
        target_name="greet_cli/__main__.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli after updating the CLI.",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="greet_cli/__main__.py", operation="write"),
            FileChangeRecord(path="README.md", operation="write"),
            FileChangeRecord(path="tests/test_cli.py", operation="write"),
        ]
    )
    failed_run = ValidationRunRecord(
        command="python -m unittest tests.test_cli",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=3,
        iteration=8,
        summary="Validation command exited with 1.",
        excerpt=(
            ".FFFFF.\n"
            "======================================================================\n"
            "FAIL: test_prefix_flag (tests.test_cli.TestCLI.test_prefix_flag)\n"
            "Traceback (most recent call last):\n"
            f'  File "{tmp_path / "tests" / "test_cli.py"}", line 21, in test_prefix_flag\n'
            "    self.assertEqual(self.run_cli(['--prefix', 'Mr.', 'Ada']), 'Hello, Mr. Ada!')\n"
            "AssertionError: 'Mr. Hello, Ada!' != 'Hello, Mr. Ada!'\n"
            f'  File "{tmp_path / "greet_cli" / "__main__.py"}", line 14, in main\n'
            "    print(greeting * args.repeat)\n"
        ),
    )
    session.validation_runs.append(failed_run)
    repair_context = planner.validation_planner.build_failure_evidence(session, failed_run)
    session.active_repair_context = repair_context

    assert planner._repair_candidate_matches_failure_text("greet_cli/__main__.py", repair_context) is True
    assert planner._repair_candidate_matches_failure_text("greet_cli/cli.py", repair_context) is False
    assert planner._repair_candidate_is_explicitly_referenced("greet_cli/cli.py", repair_context) is False

    next_target = planner._repair_target_after_failed_validation(
        session.router_result,
        session,
        failed_run,
        repair_context,
    )

    assert next_target == "greet_cli/__main__.py"


def test_planner_prefers_non_test_import_source_before_test_module_for_unittest_import_wrapper(tmp_path):
    pkg = tmp_path / "texttools"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from .normalize import normalize_words\n", encoding="utf-8")
    (pkg / "normalize.py").write_text(
        "def normalize_words(text, *, lowercase=True):\n"
        "    return text.split()\n",
        encoding="utf-8",
    )
    cli = tmp_path / "normalize_cli.py"
    cli.write_text(
        "from texttools import normalize_words\n\n"
        "def main(argv=None):\n"
        "    print(' '.join(normalize_words(' '.join(argv or []))))\n",
        encoding="utf-8",
    )
    readme = tmp_path / "README.md"
    readme.write_text("# Texttools\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_path = tests_dir / "test_normalize.py"
    test_path.write_text(
        "from texttools import normalize_words, normalize_words_keep_case\n",
        encoding="utf-8",
    )

    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="update",
                action_plan=[
                    {"step": 1, "action": "update_artifact", "reason": "Repair the keep-case implementation."},
                    {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
                ],
                target_paths=["texttools/normalize.py", "normalize_cli.py", "README.md", "tests/test_normalize.py"],
                target_name="texttools/normalize.py",
            )
        ],
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Add keep_case support and fix the failing unittest import path.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 5,
                "important_files": [
                    "texttools/__init__.py",
                    "texttools/normalize.py",
                    "normalize_cli.py",
                    "README.md",
                    "tests/test_normalize.py",
                ],
                "focus_files": ["texttools/normalize.py", "normalize_cli.py", "tests/test_normalize.py"],
                "test_files": ["tests/test_normalize.py"],
                "likely_commands": ["python -m unittest tests.test_normalize"],
            }
        ),
        validation_status="failed",
        edit_generation=1,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_normalize",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_normalize"],
    )
    commit_task_state_and_route(
        planner,
        session,
        llm.json_payloads[0],
        verification_target="Run python -m unittest tests.test_normalize and fix the keep-case runtime flow.",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="texttools/normalize.py", operation="write"),
            FileChangeRecord(path="normalize_cli.py", operation="write"),
            FileChangeRecord(path="README.md", operation="write"),
        ]
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_normalize",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=1,
            iteration=8,
            summary="Validation command exited with 1.",
            excerpt=(
                "ImportError: Failed to import test module: test_normalize\n"
                "Traceback (most recent call last):\n"
                f'  File "{test_path}", line 1, in <module>\n'
                "    from texttools import normalize_words, normalize_words_keep_case\n"
                "ImportError: cannot import name 'normalize_words_keep_case' "
                f"from 'texttools' ({pkg / '__init__.py'})\n"
                "FAILED (errors=1)\n"
            ),
        )
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "read_file"
    assert decision.tool_args["path"] == "texttools/__init__.py"


def test_planner_prefers_import_provider_after_export_patch_runtime_failure(tmp_path):
    pkg = tmp_path / "texttools"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "from .normalize import normalize_words, normalize_words_keep_case\n",
        encoding="utf-8",
    )
    normalize_path = pkg / "normalize.py"
    normalize_path.write_text(
        "def normalize_words(text, *, keep_case=False):\n"
        "    return text.split()\n",
        encoding="utf-8",
    )
    cli = tmp_path / "normalize_cli.py"
    cli.write_text("from texttools import normalize_words\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("# Texttools\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_path = tests_dir / "test_normalize.py"
    test_path.write_text(
        "from normalize_cli import main\n"
        "from texttools import normalize_words, normalize_words_keep_case\n",
        encoding="utf-8",
    )

    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="update",
                action_plan=[
                    {"step": 1, "action": "update_artifact", "reason": "Repair the keep-case implementation."},
                    {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
                ],
                target_paths=["texttools/normalize.py", "normalize_cli.py", "README.md", "tests/test_normalize.py"],
                target_name="texttools/normalize.py",
            )
        ],
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Repair the follow-up import failure for keep_case support.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 5,
                "important_files": [
                    "texttools/__init__.py",
                    "texttools/normalize.py",
                    "normalize_cli.py",
                    "README.md",
                    "tests/test_normalize.py",
                ],
                "focus_files": ["texttools/normalize.py", "normalize_cli.py", "tests/test_normalize.py"],
                "test_files": ["tests/test_normalize.py"],
                "likely_commands": ["python -m unittest tests.test_normalize"],
            }
        ),
        validation_status="failed",
        edit_generation=2,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_normalize",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_normalize"],
    )
    commit_task_state_and_route(
        planner,
        session,
        llm.json_payloads[0],
        verification_target="Run python -m unittest tests.test_normalize and fix the keep-case runtime flow.",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="texttools/normalize.py", operation="write"),
            FileChangeRecord(path="normalize_cli.py", operation="write"),
            FileChangeRecord(path="README.md", operation="write"),
            FileChangeRecord(path="texttools/__init__.py", operation="write"),
        ]
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_normalize",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=2,
            iteration=11,
            summary="Validation command exited with 1.",
            excerpt=(
                "ImportError: Failed to import test module: test_normalize\n"
                "Traceback (most recent call last):\n"
                f'  File "{test_path}", line 2, in <module>\n'
                "    from texttools import normalize_words, normalize_words_keep_case\n"
                "ImportError: cannot import name 'normalize_words_keep_case' "
                f"from 'texttools.normalize' ({normalize_path})\n"
                "FAILED (errors=1)\n"
            ),
        )
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "read_file"
    assert decision.tool_args["path"] == "texttools/normalize.py"


def test_planner_prefers_import_provider_after_recent_support_patch_runtime_failure(tmp_path):
    pkg = tmp_path / "texttools"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "from .normalize import normalize_words, normalize_words_keep_case\n",
        encoding="utf-8",
    )
    normalize_path = pkg / "normalize.py"
    normalize_path.write_text(
        "def normalize_words(text, *, keep_case=False):\n"
        "    return text.split()\n",
        encoding="utf-8",
    )
    cli = tmp_path / "normalize_cli.py"
    cli.write_text("from texttools import normalize_words\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("# Texttools\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_path = tests_dir / "test_normalize.py"
    test_path.write_text(
        "from normalize_cli import main\n"
        "from texttools import normalize_words, normalize_words_keep_case\n",
        encoding="utf-8",
    )

    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="update",
                action_plan=[
                    {"step": 1, "action": "update_artifact", "reason": "Repair the keep-case implementation."},
                    {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
                ],
                target_paths=["texttools/normalize.py", "normalize_cli.py", "README.md", "tests/test_normalize.py"],
                target_name="texttools/normalize.py",
            )
        ],
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Repair the follow-up import failure for keep_case support.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 5,
                "important_files": [
                    "texttools/__init__.py",
                    "texttools/normalize.py",
                    "normalize_cli.py",
                    "README.md",
                    "tests/test_normalize.py",
                ],
                "focus_files": ["texttools/normalize.py", "normalize_cli.py", "tests/test_normalize.py"],
                "test_files": ["tests/test_normalize.py"],
                "likely_commands": ["python -m unittest tests.test_normalize"],
            }
        ),
        validation_status="failed",
        edit_generation=2,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_normalize",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_normalize"],
    )
    commit_task_state_and_route(
        planner,
        session,
        llm.json_payloads[0],
        verification_target="Run python -m unittest tests.test_normalize and fix the keep-case runtime flow.",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="texttools/normalize.py", operation="write"),
            FileChangeRecord(path="normalize_cli.py", operation="write"),
            FileChangeRecord(path="README.md", operation="write"),
            FileChangeRecord(path="texttools/__init__.py", operation="write"),
        ]
    )
    session.repair_history.append(
        RepairAttemptRecord(
            artifact_path="texttools/__init__.py",
            validation_command="python -m unittest tests.test_normalize",
            verification_scope="runtime",
            strategy="validation_targeted",
            result="mutation_planned",
            reason="updated export first",
            evidence_signature="older-runtime-evidence",
            iteration=10,
        )
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_normalize",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=2,
            iteration=11,
            summary="Validation command exited with 1.",
            excerpt=(
                "ImportError: Failed to import test module: test_normalize\n"
                "Traceback (most recent call last):\n"
                f'  File "{test_path}", line 2, in <module>\n'
                "    from texttools import normalize_words, normalize_words_keep_case\n"
                f'  File "{cli}", line 1, in <module>\n'
                "    from texttools import normalize_words\n"
                "ImportError: cannot import name 'normalize_words_keep_case' "
                f"from 'texttools.normalize' ({normalize_path})\n"
                "FAILED (errors=1)\n"
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

    assert next_target == "texttools/normalize.py"


def test_planner_current_repair_context_pivots_back_to_provider_after_support_noop(tmp_path):
    pkg = tmp_path / "texttools"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "from .normalize import normalize_words, normalize_words_keep_case\n",
        encoding="utf-8",
    )
    normalize_path = pkg / "normalize.py"
    normalize_path.write_text(
        "def normalize_words(text, *, keep_case=False):\n"
        "    return text.split()\n",
        encoding="utf-8",
    )
    cli = tmp_path / "normalize_cli.py"
    cli.write_text("from texttools import normalize_words\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("# Texttools\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_path = tests_dir / "test_normalize.py"
    test_path.write_text(
        "from normalize_cli import main\n"
        "from texttools import normalize_words, normalize_words_keep_case\n",
        encoding="utf-8",
    )

    llm = ScriptedLLM(
        json_payloads=[
            route_payload(
                intent="update",
                action_plan=[
                    {"step": 1, "action": "update_artifact", "reason": "Repair the keep-case implementation."},
                    {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
                ],
                target_paths=["texttools/normalize.py", "normalize_cli.py", "README.md", "tests/test_normalize.py"],
                target_name="texttools/normalize.py",
            )
        ],
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Repair the follow-up import failure for keep_case support.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 5,
                "important_files": [
                    "texttools/__init__.py",
                    "texttools/normalize.py",
                    "normalize_cli.py",
                    "README.md",
                    "tests/test_normalize.py",
                ],
                "focus_files": ["texttools/normalize.py", "normalize_cli.py", "tests/test_normalize.py"],
                "test_files": ["tests/test_normalize.py"],
                "likely_commands": ["python -m unittest tests.test_normalize"],
            }
        ),
        validation_status="failed",
        edit_generation=2,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_normalize",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_normalize"],
    )
    commit_task_state_and_route(
        planner,
        session,
        llm.json_payloads[0],
        verification_target="Run python -m unittest tests.test_normalize and fix the keep-case runtime flow.",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="texttools/normalize.py", operation="write"),
            FileChangeRecord(path="normalize_cli.py", operation="write"),
            FileChangeRecord(path="README.md", operation="write"),
            FileChangeRecord(path="texttools/__init__.py", operation="write"),
        ]
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m unittest tests.test_normalize",
            kind="test",
            verification_scope="runtime",
            status="failed",
            edit_generation=2,
            iteration=11,
            summary="Validation command exited with 1.",
            excerpt=(
                "ImportError: Failed to import test module: test_normalize\n"
                "Traceback (most recent call last):\n"
                f'  File "{test_path}", line 2, in <module>\n'
                "    from texttools import normalize_words, normalize_words_keep_case\n"
                f'  File "{cli}", line 1, in <module>\n'
                "    from texttools import normalize_words\n"
                "ImportError: cannot import name 'normalize_words_keep_case' "
                f"from 'texttools.normalize' ({normalize_path})\n"
                "FAILED (errors=1)\n"
            ),
        )
    )

    failed_run = session.validation_runs[-1]
    first_context = planner.validation_planner.build_failure_evidence(session, failed_run)
    assert first_context.repair_brief is not None
    session.repair_history.extend(
        [
            RepairAttemptRecord(
                artifact_path="texttools/__init__.py",
                validation_command=failed_run.command,
                verification_scope="runtime",
                strategy="validation_targeted",
                result="mutation_planned",
                reason="updated export first",
                evidence_signature="older-runtime-evidence",
                iteration=10,
            ),
            RepairAttemptRecord(
                artifact_path="texttools/__init__.py",
                validation_command=failed_run.command,
                verification_scope="runtime",
                strategy="validation_escalated",
                result="no_effective_change",
                reason="file hash unchanged",
                failure_signature=first_context.repair_brief.failure_signature,
                iteration=12,
            ),
        ]
    )

    assert planner._next_update_target(session.router_result, session) == "texttools/normalize.py"


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


def test_planner_prefers_test_target_for_runtime_harness_nameerror(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Create a wordfreq CLI and repair failing tests.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 4,
                "important_files": ["wordfreq/__main__.py", "wordfreq/cli.py", "tests/test_wordfreq.py"],
                "focus_files": ["tests/test_wordfreq.py", "wordfreq/__main__.py"],
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
    (tmp_path / "wordfreq").mkdir()
    (tmp_path / "wordfreq" / "__main__.py").write_text("from .cli import main\n", encoding="utf-8")
    (tmp_path / "wordfreq" / "cli.py").write_text("def main():\n    return 0\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_wordfreq.py").write_text("import unittest\n", encoding="utf-8")
    payload = route_payload(
        intent="create",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Create the requested CLI and tests."},
            {"step": 2, "action": "run_validation", "reason": "Run the targeted unittest module."},
        ],
        target_paths=["wordfreq/__main__.py", "wordfreq/cli.py", "tests/test_wordfreq.py"],
        target_name="wordfreq/__main__.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="python -m unittest tests.test_wordfreq",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="wordfreq/__main__.py", operation="write"),
            FileChangeRecord(path="wordfreq/cli.py", operation="write"),
            FileChangeRecord(path="tests/test_wordfreq.py", operation="write"),
        ]
    )
    failed_run = ValidationRunRecord(
        command="python -m unittest tests.test_wordfreq",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=1,
        iteration=6,
        summary="Validation command exited with 1.",
        excerpt=(
            "ERROR: test_file_input (tests.test_wordfreq.TestWordFreq.test_file_input)\n"
            "Traceback (most recent call last):\n"
            f'  File "{tmp_path / "tests" / "test_wordfreq.py"}", line 19, in test_file_input\n'
            "    result = subprocess.run([sys.executable, '-m', 'wordfreq', sample_path], check=True)\n"
            "NameError: name 'sys' is not defined\n"
        ),
    )

    repair_context = planner.validation_planner.build_failure_evidence(session, failed_run)
    next_target = planner._repair_target_after_failed_validation(
        session.router_result,
        session,
        failed_run,
        repair_context,
    )

    assert repair_context.repair_brief is not None
    assert repair_context.repair_brief.primary_target == "tests/test_wordfreq.py"
    assert next_target == "tests/test_wordfreq.py"


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


def test_runtime_support_candidates_do_not_lead_without_fixture_signal(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
        summary="Validation command exited with 1.",
        excerpt="AssertionError: 'Ada' != 'Hello, Ada!'",
        failure_summary="The CLI still returns the wrong greeting.",
        expected_features=[],
        missing_features=[],
        file_hints=["greet_cli/__main__.py", "tests/test_cli.py"],
        line_hints=[12],
        action_hints=[],
        repair_requirements=[],
        evidence_signature="sig-no-fixture-signal",
    )

    assert planner._runtime_support_candidates_should_lead(
        ["tests/test_data.txt"],
        repair_context,
    ) is False


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


def test_actionable_explicit_update_targets_drop_inferred_validation_targets(tmp_path):
    (tmp_path / "wordfreq.py").write_text(
        "def count_words(file_path):\n    return {}\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# Word Frequency\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_wordfreq.py").write_text(
        "import unittest\n",
        encoding="utf-8",
    )
    (tests_dir / "test_data.txt").write_text("hello hello world hello world pond\n", encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task=(
            "Keep count_words in wordfreq.py intact. Add a new helper top_words(file_path, limit) "
            "in wordfreq.py. Update README.md to document top_words with a short example. "
            "Do not rewrite the tests unless it is truly necessary for the requested feature. "
            "Run python -m unittest tests.test_wordfreq and finish only when it passes."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 4,
                "important_files": [
                    "wordfreq.py",
                    "README.md",
                    "tests/test_wordfreq.py",
                    "tests/test_data.txt",
                ],
                "focus_files": [
                    "wordfreq.py",
                    "README.md",
                    "tests/test_wordfreq.py",
                    "tests/test_data.txt",
                ],
                "test_files": ["tests/test_wordfreq.py"],
                "likely_commands": ["python -m unittest tests.test_wordfreq"],
            }
        ),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Implement the requested helper."},
            {"step": 2, "action": "update_artifact", "reason": "Document the new helper in README."},
            {"step": 3, "action": "run_validation", "reason": "Run the targeted unittest module."},
        ],
        target_paths=["wordfreq.py", "README.md", "tests/test_data.txt", "tests/test_wordfreq.py"],
        target_name="wordfreq.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_wordfreq and finish only when it passes.",
    )
    session.task_state.target_artifacts = [
        TaskArtifact(path="wordfreq.py", name="wordfreq.py", kind=".py", role="primary_target", confidence=1.0),
        TaskArtifact(path="README.md", name="README.md", kind=".md", role="supporting_context", confidence=0.92),
        TaskArtifact(
            path="tests/test_data.txt",
            name="tests/test_data.txt",
            kind=".txt",
            role="validation_target",
            confidence=0.82,
        ),
        TaskArtifact(
            path="tests/test_wordfreq.py",
            name="tests/test_wordfreq.py",
            kind="test",
            role="validation_target",
            confidence=0.84,
        ),
    ]

    assert planner._actionable_explicit_target_paths(session.router_result, session) == [
        "wordfreq.py",
        "README.md",
    ]

    session.changed_files.extend(
        [
            FileChangeRecord(path="wordfreq.py", operation="write"),
            FileChangeRecord(path="README.md", operation="write"),
        ]
    )

    assert planner._has_pending_explicit_update_targets(session.router_result, session) is False


def test_actionable_explicit_update_targets_keep_requested_validation_target(tmp_path):
    (tmp_path / "README.md").write_text("# Word Frequency\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_wordfreq.py").write_text(
        "import unittest\n",
        encoding="utf-8",
    )

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Update tests/test_wordfreq.py and README.md to document the new top_words behavior.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 2,
                "important_files": ["README.md", "tests/test_wordfreq.py"],
                "focus_files": ["tests/test_wordfreq.py", "README.md"],
                "test_files": ["tests/test_wordfreq.py"],
            }
        ),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Update the requested test file."},
            {"step": 2, "action": "update_artifact", "reason": "Update the requested README file."},
        ],
        target_paths=["README.md", "tests/test_wordfreq.py"],
        target_name="tests/test_wordfreq.py",
    )
    commit_task_state_and_route(planner, session, payload)
    session.task_state.target_artifacts = [
        TaskArtifact(path="README.md", name="README.md", kind=".md", role="supporting_context", confidence=0.9),
        TaskArtifact(
            path="tests/test_wordfreq.py",
            name="tests/test_wordfreq.py",
            kind="test",
            role="validation_target",
            confidence=0.95,
        ),
    ]

    assert planner._actionable_explicit_target_paths(session.router_result, session) == [
        "README.md",
        "tests/test_wordfreq.py",
    ]


def test_actionable_explicit_update_targets_drop_unrelated_ambiguous_package_init(tmp_path):
    package_dir = tmp_path / "textutils"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("from .normalize import normalize_spaces\n", encoding="utf-8")
    (package_dir / "normalize.py").write_text("def normalize_spaces(text):\n    return text\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# textutils\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")
    (tests_dir / "test_normalize.py").write_text("import unittest\n", encoding="utf-8")

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task=(
            "Implement slugify in textutils/normalize.py, export it from textutils/__init__.py, "
            "update README.md if needed, run python -m unittest tests.test_normalize, and finish when it passes."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "important_files": [
                    "textutils/normalize.py",
                    "textutils/__init__.py",
                    "tests/test_normalize.py",
                    "tests/__init__.py",
                    "README.md",
                ],
                "focus_files": [
                    "textutils/normalize.py",
                    "textutils/__init__.py",
                    "tests/test_normalize.py",
                    "tests/__init__.py",
                ],
                "test_files": ["tests/test_normalize.py", "tests/__init__.py"],
                "likely_commands": ["python -m unittest tests.test_normalize"],
            }
        ),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Implement slugify."},
            {"step": 2, "action": "update_artifact", "reason": "Export it from the package init."},
            {"step": 3, "action": "run_validation", "reason": "Run the requested unittest module."},
        ],
        target_paths=[
            "textutils/normalize.py",
            "textutils/__init__.py",
            "README.md",
            "tests/test_normalize.py",
            "tests/__init__.py",
        ],
        target_name="textutils/normalize.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_normalize and finish only when it passes.",
    )
    session.task_state.target_artifacts = [
        TaskArtifact(path="textutils/normalize.py", name="normalize.py", kind=".py", role="primary_target", confidence=0.98),
        TaskArtifact(path="textutils/__init__.py", name="__init__.py", kind=".py", role="primary_target", confidence=0.92),
        TaskArtifact(path="README.md", name="README.md", kind=".md", role="supporting_context", confidence=0.9),
        TaskArtifact(path="tests/test_normalize.py", name="test_normalize.py", kind="test", role="validation_target", confidence=0.86),
        TaskArtifact(path="tests/__init__.py", name="__init__.py", kind=".py", role="validation_target", confidence=0.8),
    ]

    assert planner._actionable_explicit_target_paths(session.router_result, session) == [
        "textutils/normalize.py",
        "textutils/__init__.py",
        "README.md",
    ]


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


def test_runtime_repair_pivots_from_locked_python_support_file_to_impl_candidate(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Fix the failing calcstats runtime test.",
        workspace_root=str(tmp_path),
    )
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_stats",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["calcstats/__init__.py", "calcstats/stats.py", "tests/test_stats.py"],
        summary="Validation command exited with 1.",
        excerpt="AssertionError: Lists differ: [6.0, 9.0, 12.0] != [6.0, 9.0]",
        failure_summary="window_sums returns too many trailing window sums.",
        expected_features=[],
        missing_features=[],
        file_hints=["calcstats/__init__.py", "calcstats/stats.py"],
        line_hints=[18],
        action_hints=[],
        repair_requirements=[],
        evidence_signature="sig-calcstats-window-sums",
        repair_brief=RepairBrief(
            failure_type="assertion_mismatch",
            failure_signature="window-sums-mismatch",
            primary_target="calcstats/__init__.py",
            locked_target="calcstats/__init__.py",
            expected_semantics=["return one sum per accepted trailing window"],
            observed_semantics=["returns an extra trailing sum"],
            implicated_symbols=["window_sums"],
            implicated_region_hint="window_sums trailing-window loop",
            allowed_files=["calcstats/__init__.py", "calcstats/stats.py"],
            forbidden_files=["tests/test_stats.py"],
        ),
    )
    session.repair_history.append(
        RepairAttemptRecord(
            artifact_path="calcstats/__init__.py",
            validation_command="python -m unittest tests.test_stats",
            verification_scope="runtime",
            strategy="validation_targeted",
            result="mutation_planned",
            reason="updated the export file first",
            evidence_signature="sig-calcstats-window-sums",
            failure_signature="window-sums-mismatch",
            iteration=5,
        )
    )

    ordered = planner._repair_candidates_with_unattempted_first(
        session,
        repair_context,
        ["calcstats/__init__.py", "calcstats/stats.py", "tests/test_stats.py"],
    )

    assert ordered[:2] == ["calcstats/stats.py", "tests/test_stats.py"]
    assert ordered[-1] == "calcstats/__init__.py"


def test_locked_runtime_support_scope_can_yield_to_behavioral_impl_fix(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(task="Fix runtime mismatch", workspace_root=str(tmp_path))
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_stats",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["calcstats/__init__.py", "calcstats/stats.py"],
        summary="Validation command exited with 1.",
        excerpt="AssertionError: Lists differ: [6.0, 9.0, 12.0] != [6.0, 9.0]",
        failure_summary="runtime output still includes an extra trailing window sum",
        file_hints=["calcstats/__init__.py", "calcstats/stats.py"],
        evidence_signature="sig-calcstats-window-sums",
        repair_brief=RepairBrief(
            failure_type="assertion_mismatch",
            failure_signature="window-sums-mismatch",
            primary_target="calcstats/__init__.py",
            locked_target="calcstats/__init__.py",
            expected_semantics=["return one sum per accepted trailing window"],
            observed_semantics=["returns an extra trailing sum"],
            allowed_files=["calcstats/__init__.py", "calcstats/stats.py"],
        ),
    )
    session.repair_history.append(
        RepairAttemptRecord(
            artifact_path="calcstats/__init__.py",
            validation_command="python -m unittest tests.test_stats",
            verification_scope="runtime",
            strategy="validation_targeted",
            result="mutation_planned",
            reason="updated export file once already",
            evidence_signature="sig-calcstats-window-sums",
            failure_signature="window-sums-mismatch",
            iteration=3,
        )
    )

    review = planner._repair_target_scope_review(
        session=session,
        path="calcstats/stats.py",
        repair_context=repair_context,
    )

    assert review is None


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
    assert decision.action_type == AgentActionType.FINAL
    assert session.stop_reason == "no_effective_change"
    assert "comment-only change" in str(session.last_error or "")


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


def test_validation_repair_relevance_review_rejects_unresolved_undefined_symbol(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_wordfreq",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["tests/test_wordfreq.py", "wordfreq/cli.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "Traceback (most recent call last):\n"
            '  File "/tmp/tests/test_wordfreq.py", line 13, in test_read_text_stdin\n'
            '    sys.stdin = io.StringIO("hello world hello")\n'
            "NameError: name 'sys' is not defined. Did you forget to import 'sys'?\n"
        ),
        failure_summary="NameError: name 'sys' is not defined.",
        expected_features=[],
        missing_features=[],
        file_hints=["tests/test_wordfreq.py", "wordfreq/cli.py"],
        line_hints=[13],
        action_hints=[],
        repair_requirements=["Change tests/test_wordfreq.py so the failing runtime or test path can complete successfully."],
        evidence_signature="sig-runtime-nameerror",
    )

    review = planner._validation_repair_relevance_review(
        path="tests/test_wordfreq.py",
        current_content=(
            "import unittest\n\n"
            "class TestWordFreq(unittest.TestCase):\n"
            "    def test_read_text_stdin(self):\n"
            "        import io\n"
            "        sys.stdin = io.StringIO(\"hello world hello\")\n"
        ),
        proposed_content=(
            "import unittest\n\n"
            "class TestWordFreq(unittest.TestCase):\n"
            "    def test_read_text_stdin(self):\n"
            "        import io\n"
            "        sys.stdin = io.StringIO(\"hello world hello\\n\")\n"
        ),
        repair_context=repair_context,
    )

    assert review is not None
    assert review.safe_to_write is False
    assert "undefined runtime symbol" in review.summary.lower()
    assert any("binds/imports 'sys'" in issue for issue in review.blocking_issues)


def test_fallback_proposed_update_review_reuses_runtime_symbol_review_after_model_timeout(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(task="Repair tests/test_wordfreq.py", workspace_root=str(tmp_path))
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_wordfreq",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["tests/test_wordfreq.py", "wordfreq/cli.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "Traceback (most recent call last):\n"
            '  File "/tmp/tests/test_wordfreq.py", line 13, in test_read_text_stdin\n'
            '    sys.stdin = io.StringIO("hello world hello")\n'
            "NameError: name 'sys' is not defined. Did you forget to import 'sys'?\n"
        ),
        failure_summary="NameError: name 'sys' is not defined.",
        expected_features=[],
        missing_features=[],
        file_hints=["tests/test_wordfreq.py", "wordfreq/cli.py"],
        line_hints=[13],
        action_hints=[],
        repair_requirements=["Change tests/test_wordfreq.py so the failing runtime or test path can complete successfully."],
        evidence_signature="sig-runtime-nameerror-fallback",
    )
    session.active_repair_context = repair_context
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Repair the failing test module."}],
        target_paths=["tests/test_wordfreq.py"],
        target_name="tests/test_wordfreq.py",
    )
    commit_task_state_and_route(planner, session, payload)

    review = planner._fallback_proposed_update_review(
        session.router_result,
        session,
        path="tests/test_wordfreq.py",
        current_content=(
            "import unittest\n\n"
            "class TestWordFreq(unittest.TestCase):\n"
            "    def test_read_text_stdin(self):\n"
            "        import io\n"
            "        sys.stdin = io.StringIO(\"hello world hello\")\n"
        ),
        proposed_content=(
            "import unittest\n\n"
            "class TestWordFreq(unittest.TestCase):\n"
            "    def test_read_text_stdin(self):\n"
            "        import io\n"
            "        sys.stdin = io.StringIO(\"hello world hello\\n\")\n"
        ),
    )

    assert review.safe_to_write is False
    assert "undefined runtime symbol" in review.summary.lower()


def test_should_skip_model_backed_repair_review_allows_explicit_allowed_scope_when_brief_target_lags(tmp_path):
    llm = ScriptedLLM(
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:7b",
            router_model_name="qwen2.5-coder:7b",
        )
    )
    planner = Planner(llm, "")
    session = SessionState(task="Repair normalize_cli.py", workspace_root=str(tmp_path))
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Repair the CLI wrapper."}],
        target_paths=["texttools/normalize.py", "normalize_cli.py"],
        target_name="normalize_cli.py",
    )
    commit_task_state_and_route(planner, session, payload)
    session.active_repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_normalize",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["normalize_cli.py", "texttools/normalize.py", "tests/test_normalize.py"],
        summary="Validation command exited with 1.",
        excerpt="AssertionError: '' != 'Hello WORLD'",
        failure_summary="The CLI entrypoint still returns an empty result instead of preserving case.",
        expected_features=[],
        missing_features=[],
        file_hints=["normalize_cli.py", "texttools/normalize.py", "tests/test_normalize.py"],
        line_hints=[9],
        action_hints=[],
        repair_requirements=["Change normalize_cli.py so the failing runtime path produces the expected output."],
        evidence_signature="sig-normalize-keep-case",
        repair_brief=RepairBrief(
            failure_type="assertion_mismatch",
            failure_signature="runtime:assertion_mismatch:normalize-keep-case",
            primary_target="texttools/normalize.py",
            locked_target="texttools/normalize.py",
            expected_semantics=["Validation should produce: Hello WORLD"],
            observed_semantics=["Validation currently produces: ''"],
            implicated_symbols=["main", "normalize_words_keep_case"],
            implicated_region_hint="normalize_cli.py",
            repair_constraints=["Keep the repair on the CLI path that produces the observed output."],
            recent_failed_attempts=[],
            allowed_files=["texttools/normalize.py", "normalize_cli.py"],
            forbidden_files=["README.md", "tests/test_normalize.py"],
        ),
    )

    current_content = (
        "from texttools import normalize_words\n\n"
        "def main(argv=None):\n"
        "    return normalize_words('Hello, WORLD!')\n"
    )
    proposed_content = (
        "from texttools import normalize_words, normalize_words_keep_case\n\n"
        "def main(argv=None):\n"
        "    keep_case = bool(argv and '--keep-case' in argv)\n"
        "    return normalize_words_keep_case('Hello, WORLD!') if keep_case else normalize_words('Hello, WORLD!')\n"
    )

    assert (
        planner._should_skip_model_backed_repair_review(
            session.router_result,
            session,
            path="normalize_cli.py",
            current_content=current_content,
            proposed_content=proposed_content,
            reserve_model=None,
        )
        is True
    )


def test_should_skip_model_backed_repair_review_ignores_changed_and_test_evidence_scope(tmp_path):
    llm = ScriptedLLM(
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:7b",
            router_model_name="qwen2.5-coder:7b",
        )
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Repair the texttools package export after a failing runtime validation.",
        workspace_root=str(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Repair the locked package export."}],
        target_paths=["texttools/__init__.py"],
        target_name="texttools/__init__.py",
    )
    commit_task_state_and_route(planner, session, payload)
    session.changed_files = [
        FileChangeRecord(path="texttools/normalize.py", operation="update"),
        FileChangeRecord(path="normalize_cli.py", operation="update"),
    ]
    session.active_repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_normalize",
        verification_scope="runtime",
        status="failed",
        artifact_paths=[
            "texttools/__init__.py",
            "texttools/normalize.py",
            "normalize_cli.py",
            "tests/test_normalize.py",
        ],
        summary="Validation command exited with 1.",
        excerpt=(
            "ImportError: cannot import name 'normalize_words_keep_case' from 'texttools' "
            "(/workspace/texttools/__init__.py)"
        ),
        failure_summary="The package export for keep-case normalization is still missing.",
        expected_features=[],
        missing_features=[],
        file_hints=[
            "texttools/__init__.py",
            "texttools/normalize.py",
            "normalize_cli.py",
            "tests/test_normalize.py",
        ],
        line_hints=[1],
        action_hints=[],
        repair_requirements=["Change texttools/__init__.py so the missing import path becomes available."],
        evidence_signature="sig-texttools-export-runtime",
        repair_brief=RepairBrief(
            failure_type="import_failure",
            failure_signature="runtime:import_failure:texttools-export",
            primary_target="texttools/__init__.py",
            locked_target="texttools/__init__.py",
            expected_semantics=[],
            observed_semantics=[],
            implicated_symbols=["normalize_words_keep_case"],
            implicated_region_hint="texttools/__init__.py",
            repair_constraints=["Stay on the locked package export target."],
            recent_failed_attempts=[],
            allowed_files=[
                "texttools/__init__.py",
                "texttools/normalize.py",
                "normalize_cli.py",
                "tests/test_normalize.py",
            ],
            forbidden_files=["README.md"],
        ),
    )

    current_content = "from .normalize import normalize_words\n"
    proposed_content = (
        "from .normalize import normalize_words, normalize_words_keep_case\n"
    )

    assert (
        planner._should_skip_model_backed_repair_review(
            session.router_result,
            session,
            path="texttools/__init__.py",
            current_content=current_content,
            proposed_content=proposed_content,
            reserve_model=None,
        )
        is True
    )


def test_should_skip_model_backed_repair_review_disables_skip_after_first_failed_locked_attempt(tmp_path):
    llm = ScriptedLLM(
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:7b",
            router_model_name="qwen2.5-coder:7b",
        )
    )
    planner = Planner(llm, "")
    session = SessionState(task="Repair texttools/normalize.py", workspace_root=str(tmp_path))
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Repair the locked normalization behavior."}],
        target_paths=["texttools/normalize.py"],
        target_name="texttools/normalize.py",
    )
    commit_task_state_and_route(planner, session, payload)
    session.active_repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_normalize",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["texttools/normalize.py", "normalize_cli.py", "tests/test_normalize.py"],
        summary="Validation command exited with 1.",
        excerpt="AssertionError: 'Hello WORLD!' != 'Hello WORLD'",
        failure_summary="The keep-case normalization still leaves punctuation in the observed output.",
        expected_features=[],
        missing_features=[],
        file_hints=["texttools/normalize.py", "normalize_cli.py", "tests/test_normalize.py"],
        line_hints=[2],
        action_hints=[],
        repair_requirements=["Change texttools/normalize.py so the keep-case output drops the observed-only punctuation."],
        evidence_signature="sig-normalize-behavior",
        repair_brief=RepairBrief(
            failure_type="assertion_mismatch",
            failure_signature="runtime:assertion_mismatch:normalize-behavior",
            primary_target="texttools/normalize.py",
            locked_target="texttools/normalize.py",
            expected_semantics=["Validation should produce: Hello WORLD"],
            observed_semantics=["Validation currently produces: Hello WORLD!"],
            implicated_symbols=["normalize_words_keep_case"],
            implicated_region_hint="texttools/normalize.py",
            repair_constraints=["Stay on the locked normalization target and remove the observed-only punctuation."],
            recent_failed_attempts=[],
            allowed_files=["texttools/normalize.py", "normalize_cli.py"],
            forbidden_files=["README.md", "tests/test_normalize.py"],
        ),
    )
    session.repair_history.append(
        RepairAttemptRecord(
            artifact_path="texttools/normalize.py",
            validation_command="python -m unittest tests.test_normalize",
            verification_scope="runtime",
            strategy="validation_targeted",
            result="no_effective_change",
            reason="same observed punctuation remained",
            failure_signature="runtime:assertion_mismatch:normalize-behavior",
        )
    )

    current_content = (
        "def normalize_words(text: str, keep_case: bool = False) -> str:\n"
        "    if keep_case:\n"
        "        return ' '.join(text.replace(',', ' ').split())\n"
        "    return ' '.join(text.replace(',', ' ').split()).lower()\n"
    )
    proposed_content = (
        "def normalize_words(text: str, keep_case: bool = False) -> str:\n"
        "    cleaned = ' '.join(text.replace(',', ' ').split())\n"
        "    if keep_case:\n"
        "        return cleaned\n"
        "    return cleaned.lower()\n"
    )

    assert (
        planner._should_skip_model_backed_repair_review(
            session.router_result,
            session,
            path="texttools/normalize.py",
            current_content=current_content,
            proposed_content=proposed_content,
            reserve_model=None,
        )
        is False
    )


def test_review_generated_update_blocks_when_model_backed_review_times_out(tmp_path, monkeypatch):
    llm = ScriptedLLM(
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:7b",
            router_model_name="qwen2.5-coder:7b",
        )
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Repair normalize_cli.py so keep-case output is correct.",
        workspace_root=str(tmp_path),
        validation_status="failed",
    )
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Repair the CLI wrapper."}],
        target_paths=["normalize_cli.py"],
        target_name="normalize_cli.py",
    )
    commit_task_state_and_route(planner, session, payload)
    session.active_repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_normalize",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["normalize_cli.py", "tests/test_normalize.py"],
        summary="Validation command exited with 1.",
        excerpt="AssertionError: '' != 'Hello WORLD'",
        failure_summary="The CLI entrypoint still returns an empty result instead of preserving case.",
        expected_features=[],
        missing_features=[],
        file_hints=["normalize_cli.py", "tests/test_normalize.py"],
        line_hints=[9],
        action_hints=[],
        repair_requirements=["Change normalize_cli.py so the failing runtime path produces the expected output."],
        evidence_signature="sig-normalize-review-timeout",
        repair_brief=RepairBrief(
            failure_type="assertion_mismatch",
            failure_signature="runtime:assertion_mismatch:normalize-review-timeout",
            primary_target="normalize_cli.py",
            locked_target="normalize_cli.py",
            expected_semantics=["Validation should produce: Hello WORLD"],
            observed_semantics=["Validation currently produces: ''"],
            implicated_symbols=["main", "normalize_words_keep_case"],
            implicated_region_hint="normalize_cli.py",
            repair_constraints=["Keep the repair on normalize_cli.py."],
            recent_failed_attempts=[],
            allowed_files=["normalize_cli.py"],
            forbidden_files=["README.md", "tests/test_normalize.py"],
        ),
    )

    def fail_review(*_args, **_kwargs):
        raise OllamaGenerationError(
            "timed out waiting for the model to start streaming after 45.0 seconds",
            reason="startup_timeout",
            retryable=False,
            model_name="qwen2.5-coder:7b",
            startup_timeout_seconds=45,
            total_timeout_seconds=45,
        )

    monkeypatch.setattr(planner, "_should_skip_model_backed_repair_review", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(llm, "generate_json", fail_review)

    review = planner._review_generated_update(
        route=session.router_result,
        session=session,
        path="normalize_cli.py",
        current_content=(
            "from texttools import normalize_words\n\n"
            "def main(argv=None):\n"
            "    return normalize_words('Hello, WORLD!')\n"
        ),
        proposed_content=(
            "from texttools import normalize_words, normalize_words_keep_case\n\n"
            "def main(argv=None):\n"
            "    keep_case = bool(argv and '--keep-case' in argv)\n"
            "    return normalize_words_keep_case('Hello, WORLD!') if keep_case else normalize_words('Hello, WORLD!')\n"
        ),
    )

    assert review.safe_to_write is False
    assert "attempted review" in review.summary.lower()
    assert any("deterministic fallback checks alone" in issue for issue in review.blocking_issues)


def test_primary_compact_repair_review_uses_relaxed_runtime_budget(tmp_path, monkeypatch):
    llm = ScriptedLLM(
        json_payloads=[
            {
                "safe_to_write": True,
                "summary": "The focused repair changes the implicated runtime path and stays scoped.",
                "confidence": 0.91,
                "blocking_issues": [],
                "preservation_risks": [],
                "repair_hints": [],
            }
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:7b",
            router_model_name="qwen2.5-coder:7b",
        ),
    )
    planner = Planner(llm, "")
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Fix the failing target."},
            {"step": 2, "action": "run_validation", "reason": "Validate the repair."},
        ],
        target_paths=["normalize_cli.py"],
        target_name="normalize_cli.py",
    )
    session = SessionState(
        task="Repair the runtime output without changing unrelated files.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )
    commit_task_state_and_route(planner, session, payload, verification_target="python -m unittest tests.test_normalize")
    session.active_repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_normalize",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["normalize_cli.py", "tests/test_normalize.py"],
        summary="Validation command exited with 1.",
        excerpt="AssertionError: 'Hello WORLD!' != 'Hello WORLD'",
        failure_summary="The keep-case path still preserves punctuation that the expected output removes.",
        repair_requirements=["Change normalize_cli.py so the runtime output drops the trailing punctuation."],
        repair_brief=RepairBrief(
            failure_type="assertion_mismatch",
            failure_signature="runtime:assertion_mismatch:review-runtime-budget",
            primary_target="normalize_cli.py",
            locked_target="normalize_cli.py",
            expected_semantics=["Validation should produce: Hello WORLD"],
            observed_semantics=["Validation currently produces: Hello WORLD!"],
            implicated_symbols=["main"],
            implicated_region_hint="normalize_cli.py",
            repair_constraints=["Keep the repair on normalize_cli.py."],
            allowed_files=["normalize_cli.py"],
            forbidden_files=["README.md", "tests/test_normalize.py"],
        ),
    )
    monkeypatch.setattr(planner, "_should_skip_model_backed_repair_review", lambda *_args, **_kwargs: False)

    review = planner._review_generated_update(
        session.router_result,
        session,
        path="normalize_cli.py",
        current_content=(
            "from texttools import normalize_words, normalize_words_keep_case\n\n"
            "def main(argv=None):\n"
            "    keep_case = bool(argv and '--keep-case' in argv)\n"
            "    return normalize_words_keep_case('Hello, WORLD!') if keep_case else normalize_words('Hello, WORLD!')\n"
        ),
        proposed_content=(
            "from texttools import normalize_words, normalize_words_keep_case\n\n"
            "def main(argv=None):\n"
            "    keep_case = bool(argv and '--keep-case' in argv)\n"
            "    source = 'Hello, WORLD!'\n"
            "    return normalize_words_keep_case(source).replace('!', '') if keep_case else normalize_words(source)\n"
        ),
    )

    assert review.safe_to_write is True
    kwargs = llm.generate_json_calls[0]["kwargs"]
    assert kwargs["model"] == "qwen2.5-coder:7b"
    assert kwargs["num_ctx"] == 2048
    assert kwargs["timeout"] == 45
    assert kwargs["total_timeout"] == 90
    assert kwargs["strict_timeouts"] is False


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


def test_pre_write_update_review_rejects_helper_that_duplicates_entrypoint_cli_flags(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    main_content = (
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
    current_cli = "def greet(name: str) -> str:\n    return f\"Hello, {name}!\"\n"
    proposed_cli = (
        "def greet(name: str, prefix: str = '', repeat: int = 1, uppercase: bool = False) -> str:\n"
        "    greeting = f\"{prefix}Hello, {name}!\"\n"
        "    if uppercase:\n"
        "        greeting = greeting.upper()\n"
        "    return greeting * repeat\n"
    )
    (pkg / "__main__.py").write_text(main_content, encoding="utf-8")
    (pkg / "cli.py").write_text(current_cli, encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_cli.py").write_text(
        "import io\n"
        "import unittest\n"
        "from unittest.mock import patch\n\n"
        "from greet_cli import __main__\n\n\n"
        "class TestCLI(unittest.TestCase):\n"
        "    def run_cli(self, argv):\n"
        "        with patch('sys.stdout', new_callable=io.StringIO) as stdout:\n"
        "            __main__.main(argv)\n"
        "            return stdout.getvalue().strip()\n",
        encoding="utf-8",
    )

    session = SessionState(
        task="Extend this existing CLI so it supports --prefix and --repeat while keeping uppercase working.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 3,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "focus_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Extend the CLI behavior."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the relevant tests."},
        ],
        target_paths=["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
        target_name="greet_cli/cli.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli after updating the CLI.",
    )

    review = planner._pre_write_update_review(
        session.router_result,
        session,
        path="greet_cli/cli.py",
        current_content=current_cli,
        proposed_content=proposed_cli,
        repair_context=None,
    )

    assert review.safe_to_write is False
    assert "duplicates cli wrapper responsibilities" in review.summary.lower()
    assert "prefix, repeat, uppercase" in review.blocking_issues[0]


def test_pre_write_update_review_rejects_entrypoint_that_composes_helper_input_from_cli_values(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    current_main = (
        "from __future__ import annotations\n\n"
        "import argparse\n\n"
        "from .cli import greet\n\n\n"
        "def main(argv=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name')\n"
        "    parser.add_argument('--prefix', type=str, default='')\n"
        "    parser.add_argument('--repeat', type=int, default=1)\n"
        "    parser.add_argument('--uppercase', action='store_true')\n"
        "    args = parser.parse_args(argv)\n\n"
        "    for _ in range(args.repeat):\n"
        "        print(greet(args.name, uppercase=args.uppercase))\n"
    )
    proposed_main = (
        "from __future__ import annotations\n\n"
        "import argparse\n\n"
        "from .cli import greet\n\n\n"
        "def main(argv=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name')\n"
        "    parser.add_argument('--prefix', type=str, default='')\n"
        "    parser.add_argument('--repeat', type=int, default=1)\n"
        "    parser.add_argument('--uppercase', action='store_true')\n"
        "    args = parser.parse_args(argv)\n\n"
        "    for _ in range(args.repeat):\n"
        "        print(greet(f\"{args.prefix} {args.name}\", uppercase=args.uppercase))\n"
    )
    (pkg / "__main__.py").write_text(current_main, encoding="utf-8")
    (pkg / "cli.py").write_text("def greet(name: str, uppercase: bool = False) -> str:\n    return name\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_cli.py").write_text("pass\n", encoding="utf-8")

    session = SessionState(
        task="Extend this existing CLI so it supports --prefix and --repeat while keeping uppercase working.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 3,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "focus_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Extend the CLI behavior."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the relevant tests."},
        ],
        target_paths=["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
        target_name="greet_cli/__main__.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli after updating the CLI.",
    )

    review = planner._pre_write_update_review(
        session.router_result,
        session,
        path="greet_cli/__main__.py",
        current_content=current_main,
        proposed_content=proposed_main,
        repair_context=None,
    )

    assert review.safe_to_write is False
    assert "blurs the contract" in review.summary.lower()
    assert any("helper inputs" in issue.lower() for issue in review.blocking_issues)


def test_pre_write_update_review_rejects_entrypoint_that_wraps_helper_output_with_cli_formatting(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    current_main = (
        "from __future__ import annotations\n\n"
        "import argparse\n\n"
        "from .cli import greet\n\n\n"
        "def main(argv=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name')\n"
        "    parser.add_argument('--prefix', type=str, default='')\n"
        "    parser.add_argument('--repeat', type=int, default=1)\n"
        "    parser.add_argument('--uppercase', action='store_true')\n"
        "    args = parser.parse_args(argv)\n\n"
        "    for _ in range(args.repeat):\n"
        "        print(greet(args.name, uppercase=args.uppercase))\n"
    )
    proposed_main = (
        "from __future__ import annotations\n\n"
        "import argparse\n\n"
        "from .cli import greet\n\n\n"
        "def main(argv=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name')\n"
        "    parser.add_argument('--prefix', type=str, default='')\n"
        "    parser.add_argument('--repeat', type=int, default=1)\n"
        "    parser.add_argument('--uppercase', action='store_true')\n"
        "    args = parser.parse_args(argv)\n\n"
        "    full_greeting = f\"{args.prefix} {greet(args.name, uppercase=args.uppercase)}\"\n"
        "    for _ in range(args.repeat):\n"
        "        print(full_greeting)\n"
    )
    (pkg / "__main__.py").write_text(current_main, encoding="utf-8")
    (pkg / "cli.py").write_text("def greet(name: str, uppercase: bool = False) -> str:\n    return name\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_cli.py").write_text("pass\n", encoding="utf-8")

    session = SessionState(
        task="Extend this existing CLI so it supports --prefix and --repeat while keeping uppercase working.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 3,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "focus_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Extend the CLI behavior."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the relevant tests."},
        ],
        target_paths=["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
        target_name="greet_cli/__main__.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli after updating the CLI.",
    )

    review = planner._pre_write_update_review(
        session.router_result,
        session,
        path="greet_cli/__main__.py",
        current_content=current_main,
        proposed_content=proposed_main,
        repair_context=None,
    )

    assert review.safe_to_write is False
    assert "blurs the contract" in review.summary.lower()
    assert any("helper result" in issue.lower() for issue in review.blocking_issues)


def test_pre_write_update_review_uses_latest_session_write_for_sibling_entrypoint_contract(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    original_main = (
        "from __future__ import annotations\n\n"
        "import argparse\n\n"
        "from .cli import greet\n\n\n"
        "def main(argv=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name')\n"
        "    args = parser.parse_args(argv)\n\n"
        "    print(greet(args.name))\n"
    )
    updated_main = (
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
    current_cli = "def greet(name: str) -> str:\n    return f\"Hello, {name}!\"\n"
    proposed_cli = (
        "def greet(name: str, prefix: str = '', repeat: int = 1, uppercase: bool = False) -> str:\n"
        "    greeting = f\"{prefix}Hello, {name}!\"\n"
        "    if uppercase:\n"
        "        greeting = greeting.upper()\n"
        "    return greeting * repeat\n"
    )
    (pkg / "__main__.py").write_text(original_main, encoding="utf-8")
    (pkg / "cli.py").write_text(current_cli, encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_cli.py").write_text("pass\n", encoding="utf-8")

    session = SessionState(
        task=(
            "Extend this existing CLI so it supports an optional --prefix TEXT flag and an optional "
            "--repeat N flag while keeping the existing --uppercase behavior working."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 3,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "focus_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
        tool_calls=[
            ToolCallRecord(
                iteration=5,
                tool_name="write_file",
                tool_args={"path": "greet_cli/__main__.py", "content": updated_main},
                success=True,
                summary="Wrote greet_cli/__main__.py.",
            )
        ],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Extend the CLI behavior."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the relevant tests."},
        ],
        target_paths=["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
        target_name="greet_cli/cli.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli after updating the CLI.",
    )

    review = planner._pre_write_update_review(
        session.router_result,
        session,
        path="greet_cli/cli.py",
        current_content=current_cli,
        proposed_content=proposed_cli,
        repair_context=None,
    )

    assert review.safe_to_write is False
    assert "duplicates cli wrapper responsibilities" in review.summary.lower()
    assert "repeat" in review.blocking_issues[0].lower()


def test_pre_write_update_review_rejects_entrypoint_that_passes_duplicate_cli_values_to_helper(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    pkg = tmp_path / "greet_cli"
    pkg.mkdir()
    current_main = (
        "from __future__ import annotations\n\n"
        "import argparse\n\n"
        "from .cli import greet\n\n\n"
        "def main(argv=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name')\n"
        "    parser.add_argument('--prefix', type=str, help='Optional prefix to prepend to the greeting')\n"
        "    parser.add_argument('--repeat', type=int, default=1, help='Number of times to repeat the greeting')\n"
        "    parser.add_argument('--uppercase', action='store_true', help='Convert the greeting to uppercase')\n"
        "    args = parser.parse_args(argv)\n\n"
        "    greeting = greet(args.name)\n"
        "    if args.prefix:\n"
        "        greeting = f\"{args.prefix} {greeting}\"\n"
        "    if args.uppercase:\n"
        "        greeting = greeting.upper()\n\n"
        "    for _ in range(args.repeat):\n"
        "        print(greeting)\n"
    )
    proposed_main = (
        "from __future__ import annotations\n\n"
        "import argparse\n\n"
        "from .cli import greet\n\n\n"
        "def main(argv=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name')\n"
        "    parser.add_argument('--prefix', type=str, default='', help='Optional prefix to prepend to the greeting')\n"
        "    parser.add_argument('--repeat', type=int, default=1, help='Number of times to repeat the greeting')\n"
        "    parser.add_argument('--uppercase', action='store_true', help='Convert the greeting to uppercase')\n"
        "    args = parser.parse_args(argv)\n\n"
        "    greeting = greet(args.name, prefix=args.prefix, repeat=args.repeat, uppercase=args.uppercase)\n\n"
        "    for _ in range(args.repeat):\n"
        "        print(greeting)\n"
    )
    (pkg / "__main__.py").write_text(current_main, encoding="utf-8")
    (pkg / "cli.py").write_text(
        "def greet(name: str, prefix: str = '', repeat: int = 1, uppercase: bool = False) -> str:\n"
        "    greeting = f\"{prefix}Hello, {name}!\"\n"
        "    if uppercase:\n"
        "        greeting = greeting.upper()\n"
        "    return greeting * repeat\n",
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_cli.py").write_text("pass\n", encoding="utf-8")

    session = SessionState(
        task=(
            "Extend this existing CLI so it supports an optional --prefix TEXT flag and an optional "
            "--repeat N flag while keeping the existing --uppercase behavior working."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "file_count": 3,
                "important_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "focus_files": ["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
                "test_files": ["tests/test_cli.py"],
                "likely_commands": ["python -m unittest tests.test_cli"],
            }
        ),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the CLI behavior."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the relevant tests."},
        ],
        target_paths=["greet_cli/__main__.py", "greet_cli/cli.py", "tests/test_cli.py"],
        target_name="greet_cli/__main__.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_cli after updating the CLI.",
    )

    review = planner._pre_write_update_review(
        session.router_result,
        session,
        path="greet_cli/__main__.py",
        current_content=current_main,
        proposed_content=proposed_main,
        repair_context=None,
    )

    assert review.safe_to_write is False
    assert "blurs the contract" in review.summary.lower()
    assert any("same cli values outside the helper call" in issue.lower() for issue in review.blocking_issues)


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


def test_planner_review_retry_keeps_same_model_and_escalates_follow_up_to_full_for_validation_repairs(tmp_path, monkeypatch):
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
    assert llm.generate_calls[0]["kwargs"]["num_ctx"] == 2048
    assert "The proposal leaves the implicated identifier lines unchanged." in llm.generate_calls[0]["args"][0]
    assert llm.generate_calls[1]["kwargs"]["model"] == "qwen2.5-coder:7b"
    assert llm.generate_calls[1]["kwargs"]["strict_timeouts"] is False
    assert llm.generate_calls[1]["kwargs"]["num_ctx"] == 4096
    assert "references sys.argv without importing sys." in llm.generate_calls[1]["args"][0]
    assert "add import sys before using it" in llm.generate_calls[1]["args"][0]


def test_review_guided_retry_stays_compact_for_small_focused_updates(tmp_path, monkeypatch):
    llm = ScriptedLLM(
        text_payloads=[
            "def main():\n    return 'draft'\n",
            "def main(argv=None):\n    return argv or []\n",
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:7b",
            router_model_name="qwen2.5-coder:7b",
        ),
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Update app.py so main accepts optional argv while keeping the edit focused.",
        workspace_root=str(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Refine the focused update."}],
        target_paths=["app.py"],
        target_name="app.py",
    )
    commit_task_state_and_route(planner, session, payload)

    reviews = iter(
        [
            ProposedUpdateReview(
                safe_to_write=False,
                summary="The retry still leaves argv handling incomplete.",
                confidence=0.9,
                blocking_issues=["The proposal still ignores the optional argv path."],
                preservation_risks=[],
                repair_hints=["Handle argv=None inside main() instead of hard-coding a fixed return value."],
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
    monkeypatch.setattr(planner, "_pre_write_update_review", lambda *_args, **_kwargs: next(reviews))

    result = planner._retry_update_after_review_failure(
        session.router_result,
        session,
        path="app.py",
        current_content="def main():\n    return []\n",
        review_feedback=ProposedUpdateReview(
            safe_to_write=False,
            summary="The proposal replaces the focused update with a stub.",
            confidence=0.82,
            blocking_issues=["Keep the change focused on argv handling."],
            preservation_risks=[],
            repair_hints=["Revise only the main() logic that needs argv support."],
        ),
        repair_context=None,
        repair_strategy=None,
        prior_attempts=[],
    )

    assert result.content == "def main(argv=None):\n    return argv or []"
    assert len(llm.generate_calls) == 2
    assert all(call["kwargs"]["strict_timeouts"] is True for call in llm.generate_calls)
    assert all(call["kwargs"]["num_ctx"] == 2048 for call in llm.generate_calls)
    assert "The retry still leaves argv handling incomplete." in llm.generate_calls[1]["args"][0]
    assert "Handle argv=None inside main() instead of hard-coding a fixed return value." in llm.generate_calls[1]["args"][0]


def test_review_guided_retry_prompt_surfaces_minimal_semantic_delta(tmp_path, monkeypatch):
    llm = ScriptedLLM(
        text_payloads=[
            "from __future__ import annotations\n\n"
            "import argparse\n\n"
            "from .cli import greet\n\n"
            "def main(argv=None):\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('name')\n"
            "    parser.add_argument('--title')\n"
            "    args = parser.parse_args(argv)\n"
            "    greeting = greet(args.name)\n"
            "    if args.title:\n"
            "        greeting = f\"{args.title} {greeting}\"\n"
            "    print(greeting)\n"
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:7b",
            router_model_name="qwen2.5-coder:7b",
        ),
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Repair the title formatting bug in the existing CLI.",
        workspace_root=str(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Repair the failing CLI entrypoint."}],
        target_paths=["greet_cli/__main__.py"],
        target_name="__main__.py",
    )
    commit_task_state_and_route(planner, session, payload)
    monkeypatch.setattr(
        planner,
        "_pre_write_update_review",
        lambda *_args, **_kwargs: ProposedUpdateReview(
            safe_to_write=True,
            summary="ok",
            confidence=0.9,
            blocking_issues=[],
            preservation_risks=[],
            repair_hints=[],
        ),
    )
    current_content = (
        "from __future__ import annotations\n\n"
        "import argparse\n\n"
        "from .cli import greet\n\n"
        "def main(argv=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('name')\n"
        "    parser.add_argument('--title')\n"
        "    args = parser.parse_args(argv)\n"
        "    greeting = greet(args.name)\n"
        "    if args.title:\n"
        "        greeting = f\"{args.title}: {greeting}\"\n"
        "    print(greeting)\n"
    )
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_cli",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["greet_cli/__main__.py", "tests/test_cli.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "AssertionError: 'Dr.: Hello, Ada!\\nDr.: Hello, Ada!' != 'Dr. Hello, Ada!\\nDr. Hello, Ada!'\n"
            "- Dr.: Hello, Ada!\n"
            "+ Dr. Hello, Ada!\n"
        ),
        failure_summary="The CLI adds extra punctuation when formatting titled greetings.",
        file_hints=["greet_cli/__main__.py", "tests/test_cli.py"],
        repair_requirements=[],
        evidence_signature="sig-review-delta",
        repair_brief=RepairBrief(
            failure_type="assertion_mismatch",
            failure_signature="runtime:assertion_mismatch:delta5678",
            primary_target="greet_cli/__main__.py",
            locked_target="greet_cli/__main__.py",
            expected_semantics=["Validation should produce: Dr. Hello, Ada!\nDr. Hello, Ada!"],
            observed_semantics=["Validation currently produces: Dr.: Hello, Ada!\nDr.: Hello, Ada!"],
            implicated_symbols=[],
            implicated_region_hint="greet_cli/__main__.py",
            repair_constraints=[],
            recent_failed_attempts=[],
            allowed_files=["greet_cli/__main__.py", "greet_cli/cli.py"],
            forbidden_files=["tests/test_cli.py"],
        ),
    )

    result = planner._retry_update_after_review_failure(
        session.router_result,
        session,
        path="greet_cli/__main__.py",
        current_content=current_content,
        review_feedback=ProposedUpdateReview(
            safe_to_write=False,
            summary="The proposal still leaves the title punctuation mismatch in place.",
            confidence=0.89,
            blocking_issues=["The proposal still preserves the observed punctuation mismatch in the titled greeting output."],
            preservation_risks=[],
            repair_hints=["Change the output construction in the locked target so the observed-only punctuation disappears."],
        ),
        repair_context=repair_context,
        repair_strategy="validation_targeted",
        prior_attempts=[],
    )

    assert "greeting = f\"{args.title} {greeting}\"" in result.content
    assert len(llm.generate_calls) == 1
    prompt = llm.generate_calls[0]["args"][0]
    assert (
        "Minimal semantic delta: Remove observed-only text ':' between shared prefix 'Dr.' and shared suffix 'Hello, Ada!'."
        in prompt
    )
    assert "Change the implicated current lines in greet_cli/__main__.py:" in prompt
    assert '14:         greeting = f"{args.title}: {greeting}"' in prompt
    assert (
        "Apply this exact semantic delta in the behavior produced by this file: Remove observed-only text ':' between shared prefix 'Dr.' and shared suffix 'Hello, Ada!'."
        in prompt
    )


def test_review_feedback_marks_semantic_stalls_as_noop_for_repair_escalation(tmp_path):
    llm = ScriptedLLM(
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:7b",
            router_model_name="qwen2.5-coder:7b",
        )
    )
    planner = Planner(llm, "")

    unchanged_lines_review = ProposedUpdateReview(
        safe_to_write=False,
        summary="Lines unchanged.",
        confidence=0.8,
        blocking_issues=["The proposal leaves the implicated identifier lines unchanged."],
        preservation_risks=[],
        repair_hints=["Change the implicated callable."],
    )
    observed_mismatch_review = ProposedUpdateReview(
        safe_to_write=False,
        summary="The proposal still leaves the title punctuation mismatch in place.",
        confidence=0.89,
        blocking_issues=[
            "The proposal still preserves the observed punctuation mismatch in the titled greeting output."
        ],
        preservation_risks=[],
        repair_hints=["Change the output construction in the locked target so the observed-only punctuation disappears."],
    )

    assert planner._review_feedback_is_noop(unchanged_lines_review) is True
    assert planner._review_feedback_is_noop(observed_mismatch_review) is True


def test_review_guided_retry_prompt_surfaces_undefined_runtime_symbol_guidance(tmp_path, monkeypatch):
    llm = ScriptedLLM(
        text_payloads=[
            "import sys\n"
            "import unittest\n\n"
            "class TestWordFreq(unittest.TestCase):\n"
            "    def test_read_text_stdin(self):\n"
            "        import io\n"
            "        sys.stdin = io.StringIO('hello world hello')\n"
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:7b",
            router_model_name="qwen2.5-coder:7b",
        ),
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Repair the failing stdin test for wordfreq.",
        workspace_root=str(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Repair the failing test module."}],
        target_paths=["tests/test_wordfreq.py"],
        target_name="tests/test_wordfreq.py",
    )
    commit_task_state_and_route(planner, session, payload)
    monkeypatch.setattr(
        planner,
        "_pre_write_update_review",
        lambda *_args, **_kwargs: ProposedUpdateReview(
            safe_to_write=True,
            summary="ok",
            confidence=0.9,
            blocking_issues=[],
            preservation_risks=[],
            repair_hints=[],
        ),
    )

    current_content = (
        "import unittest\n\n"
        "class TestWordFreq(unittest.TestCase):\n"
        "    def test_read_text_stdin(self):\n"
        "        import io\n"
        "        sys.stdin = io.StringIO('hello world hello')\n"
    )
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_wordfreq",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["tests/test_wordfreq.py", "wordfreq/cli.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "Traceback (most recent call last):\n"
            '  File "/tmp/tests/test_wordfreq.py", line 6, in test_read_text_stdin\n'
            "        sys.stdin = io.StringIO('hello world hello')\n"
            "NameError: name 'sys' is not defined. Did you forget to import 'sys'?\n"
        ),
        failure_summary="NameError: name 'sys' is not defined.",
        file_hints=["tests/test_wordfreq.py", "wordfreq/cli.py"],
        line_hints=[6],
        repair_requirements=[
            "Change tests/test_wordfreq.py so the failing runtime or test path can complete successfully.",
            "Bind or import 'sys' before its failing use in tests/test_wordfreq.py, or remove that use if it is unnecessary.",
        ],
        evidence_signature="sig-runtime-nameerror-prompt",
        repair_brief=RepairBrief(
            failure_type="runtime_failure",
            failure_signature="runtime:runtime_failure:nameerrorprompt",
            primary_target="tests/test_wordfreq.py",
            locked_target="tests/test_wordfreq.py",
            expected_semantics=["The symbol 'sys' should be bound or imported before it is used."],
            observed_semantics=[
                "The current runtime path uses 'sys' before it is bound or imported. Current use: sys.stdin = io.StringIO('hello world hello')"
            ],
            implicated_symbols=["sys", "test_read_text_stdin", "StringIO"],
            implicated_region_hint="tests/test_wordfreq.py:line 6",
            repair_constraints=[
                "Bind or import 'sys' before its failing use in tests/test_wordfreq.py, or remove that use if it is unnecessary."
            ],
            recent_failed_attempts=[],
            allowed_files=["tests/test_wordfreq.py", "wordfreq/cli.py"],
            forbidden_files=["README.md"],
        ),
    )

    result = planner._retry_update_after_review_failure(
        session.router_result,
        session,
        path="tests/test_wordfreq.py",
        current_content=current_content,
        review_feedback=ProposedUpdateReview(
            safe_to_write=False,
            summary="The proposed repair still leaves the undefined runtime symbol unresolved.",
            confidence=0.91,
            blocking_issues=[
                "The runtime failure still reports 'sys' as undefined in tests/test_wordfreq.py, but the proposal neither binds/imports 'sys' nor removes its failing usage."
            ],
            preservation_risks=[],
            repair_hints=[
                "Either import or otherwise bind 'sys' in tests/test_wordfreq.py, or remove the failing usage from the implicated line."
            ],
        ),
        repair_context=repair_context,
        repair_strategy="validation_targeted",
        prior_attempts=[],
    )

    assert "import sys" in result.content
    prompt = llm.generate_calls[0]["args"][0]
    assert "Expected semantics: The symbol 'sys' should be bound or imported before it is used." in prompt
    assert "Observed semantics: The current runtime path uses 'sys' before it is bound or imported." in prompt
    assert "Repair constraints: Bind or import 'sys' before its failing use in tests/test_wordfreq.py" in prompt
    assert "Repair focus: region=tests/test_wordfreq.py:line 6 symbols=sys" in prompt
    assert "Resolve the undefined symbol 'sys' in tests/test_wordfreq.py" in prompt
    assert "6:         sys.stdin = io.StringIO('hello world hello')" in prompt
    assert "Either import or otherwise bind 'sys' before its current use" in prompt


def test_runtime_repair_review_ignores_called_process_error_summary_noise(tmp_path):
    llm = ScriptedLLM(
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:7b",
            router_model_name="qwen2.5-coder:7b",
        )
    )
    planner = Planner(llm, "")
    current_content = (
        "import sys\n\n"
        "from wordaudit import duplicate_words\n\n\n"
        "def main(argv=None):\n"
        "    argv = list(sys.argv[1:] if argv is None else argv)\n"
        "    with open(argv[0], 'r', encoding='utf-8') as handle:\n"
        "        lines = handle.readlines()\n"
        "        duplicates = duplicate_words([line for line in lines if not line.strip().startswith('#')])\n"
        "    for word in duplicates:\n"
        "        print(word)\n"
    )
    proposed_content = (
        "import sys\n"
        "from pathlib import Path\n\n"
        "ROOT = Path(__file__).resolve().parents[1]\n"
        "if str(ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(ROOT))\n\n"
        "from wordaudit import duplicate_words\n\n\n"
        "def main(argv=None):\n"
        "    argv = list(sys.argv[1:] if argv is None else argv)\n"
        "    with open(argv[0], 'r', encoding='utf-8') as handle:\n"
        "        lines = handle.readlines()\n"
        "        duplicates = duplicate_words([line for line in lines if not line.strip().startswith('#')])\n"
        "    for word in duplicates:\n"
        "        print(word)\n"
    )
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_report",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["scripts/build_duplicates.py", "wordaudit/report.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "Traceback (most recent call last):\n"
            "raise CalledProcessError(retcode, process.args,\n"
            "subprocess.CalledProcessError: "
            "Command '['/usr/bin/python3', '/tmp/demo/scripts/build_duplicates.py', '/tmp/tmpwords.txt']' "
            "returned non-zero exit status 1.\n"
            "FAILED (errors=1)\n"
        ),
        failure_summary=(
            "Traceback (most recent call last):\n"
            "raise CalledProcessError(retcode, process.args,\n"
            "subprocess.CalledProcessError: "
            "Command '['/usr/bin/python3', '/tmp/demo/scripts/build_duplicates.py', '/tmp/tmpwords.txt']' "
            "returned non-zero exit status 1.\n"
            "FAILED (errors=1)\n"
        ),
        file_hints=["scripts/build_duplicates.py", "wordaudit/report.py"],
        repair_requirements=[
            "Change scripts/build_duplicates.py so the failing runtime or test path can complete successfully."
        ],
        evidence_signature="sig-calledprocess-noise",
        repair_brief=RepairBrief(
            failure_type="runtime_failure",
            failure_signature="runtime:runtime_failure:calledprocessnoise",
            primary_target="scripts/build_duplicates.py",
            locked_target="scripts/build_duplicates.py",
            expected_semantics=["The exercised runtime command should exit successfully."],
            observed_semantics=[
                "The current runtime command exits non-zero when invoking: ['/usr/bin/python3', '/tmp/demo/scripts/build_duplicates.py', '/tmp/tmpwords.txt'] (exit status 1)"
            ],
            implicated_symbols=[],
            implicated_region_hint="scripts/build_duplicates.py",
            repair_constraints=[
                "Change scripts/build_duplicates.py so the failing runtime or test path can complete successfully."
            ],
            recent_failed_attempts=[],
            allowed_files=["scripts/build_duplicates.py", "wordaudit/report.py"],
            forbidden_files=["README.md", "tests/test_report.py"],
        ),
    )

    review = planner._validation_repair_relevance_review(
        path="scripts/build_duplicates.py",
        current_content=current_content,
        proposed_content=proposed_content,
        repair_context=repair_context,
    )

    assert review is None


def test_compact_runtime_repair_prompt_anchors_direct_python_script_execution(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Repair the dataflow script runtime path.",
        workspace_root=str(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the runtime script path."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["scripts/build_duplicates.py", "wordaudit/report.py", "README.md"],
        target_name="build_duplicates.py",
    )
    commit_task_state_and_route(planner, session, payload, verification_target="python -m unittest tests.test_report")

    current_content = (
        "import sys\n\n"
        "from wordaudit import duplicate_words\n\n\n"
        "def main(argv=None):\n"
        "    argv = list(sys.argv[1:] if argv is None else argv)\n"
        "    with open(argv[0], 'r', encoding='utf-8') as handle:\n"
        "        lines = handle.readlines()\n"
        "        duplicates = duplicate_words([line for line in lines if not line.strip().startswith('#')])\n"
        "    for word in duplicates:\n"
        "        print(word)\n"
    )
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_report",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["scripts/build_duplicates.py", "wordaudit/report.py", "README.md", "tests/test_report.py"],
        summary="Validation command exited with 1.",
        excerpt=(
            "Traceback (most recent call last):\n"
            "raise CalledProcessError(retcode, process.args,\n"
            "subprocess.CalledProcessError: "
            "Command '['/usr/bin/python3', '/tmp/demo/scripts/build_duplicates.py', '/tmp/tmpwords.txt']' "
            "returned non-zero exit status 1.\n"
            "FAILED (errors=1)\n"
        ),
        failure_summary=(
            "Traceback (most recent call last):\n"
            "raise CalledProcessError(retcode, process.args,\n"
            "subprocess.CalledProcessError: "
            "Command '['/usr/bin/python3', '/tmp/demo/scripts/build_duplicates.py', '/tmp/tmpwords.txt']' "
            "returned non-zero exit status 1.\n"
            "FAILED (errors=1)\n"
        ),
        file_hints=["scripts/build_duplicates.py", "wordaudit/report.py", "README.md", "tests/test_report.py"],
        repair_requirements=[
            "Change scripts/build_duplicates.py so the failing runtime or test path can complete successfully."
        ],
        evidence_signature="sig-direct-script-anchor",
        repair_brief=RepairBrief(
            failure_type="runtime_failure",
            failure_signature="runtime:runtime_failure:directscriptanchor",
            primary_target="scripts/build_duplicates.py",
            locked_target="scripts/build_duplicates.py",
            expected_semantics=["The exercised runtime command should exit successfully."],
            observed_semantics=[
                "The current runtime command exits non-zero when invoking: ['/usr/bin/python3', '/tmp/demo/scripts/build_duplicates.py', '/tmp/tmpwords.txt'] (exit status 1)"
            ],
            implicated_symbols=[],
            implicated_region_hint="scripts/build_duplicates.py",
            repair_constraints=[
                "Change scripts/build_duplicates.py so the failing runtime or test path can complete successfully."
            ],
            recent_failed_attempts=[],
            allowed_files=["scripts/build_duplicates.py", "wordaudit/report.py"],
            forbidden_files=["README.md", "tests/test_report.py"],
        ),
    )

    prompt = generate_content_prompt(
        session.router_result,
        session,
        path="scripts/build_duplicates.py",
        current_content=current_content,
        repair_context=repair_context,
        repair_strategy="validation_targeted",
        mode="compact",
    )

    assert "This file is executed directly as a Python script in the failing command." in prompt
    assert "Treat the top-of-file import/bootstrap path as the primary repair surface." in prompt
    assert "Keep scripts/build_duplicates.py runnable in that mode" in prompt
    assert "1: import sys" in prompt
    assert "3: from wordaudit import duplicate_words" in prompt
    assert "A change inside main() alone cannot fix a top-level import failure" in prompt
    assert "add that bootstrap before imports like" in prompt
    assert "Change the implicated current lines in scripts/build_duplicates.py:" not in prompt
    assert "The failing command runs this file directly by script path." in prompt


def test_prioritize_runtime_target_categories_demotes_documentation_for_runtime_repairs(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    repair_context = ValidationFailureEvidence(
        command="python -m unittest tests.test_report",
        verification_scope="runtime",
        status="failed",
        artifact_paths=["scripts/build_duplicates.py", "README.md", "tests/test_report.py"],
        summary="Validation command exited with 1.",
        excerpt="CalledProcessError: scripts/build_duplicates.py exited with status 1.",
        failure_summary="The runtime script still exits non-zero.",
        file_hints=["scripts/build_duplicates.py", "README.md", "tests/test_report.py"],
        repair_requirements=[
            "Change scripts/build_duplicates.py so the failing runtime or test path can complete successfully."
        ],
        evidence_signature="sig-runtime-doc-demotion",
    )

    ordered = planner._prioritize_runtime_target_categories(
        ["README.md", "scripts/build_duplicates.py", "tests/test_report.py"],
        repair_context,
    )

    assert ordered[:2] == ["scripts/build_duplicates.py", "tests/test_report.py"]
    assert ordered[-1] == "README.md"


def test_runtime_repair_target_after_failed_validation_skips_documentation_after_technical_noops(tmp_path):
    pkg = tmp_path / "texttools"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "from .normalize import normalize_words\n",
        encoding="utf-8",
    )
    normalize_path = pkg / "normalize.py"
    normalize_path.write_text(
        "def normalize_words(text, *, keep_case=False):\n"
        "    return text.split()\n",
        encoding="utf-8",
    )
    (tmp_path / "normalize_cli.py").write_text(
        "from texttools import normalize_words\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# texttools\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_path = tests_dir / "test_normalize.py"
    test_path.write_text(
        "from texttools import normalize_words, normalize_words_keep_case\n",
        encoding="utf-8",
    )

    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task="Add keep_case support and finish the failing runtime repair.",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path).model_copy(
            update={
                "important_files": [
                    "texttools/normalize.py",
                    "texttools/__init__.py",
                    "normalize_cli.py",
                    "README.md",
                    "tests/test_normalize.py",
                ],
                "focus_files": [
                    "texttools/normalize.py",
                    "texttools/__init__.py",
                    "normalize_cli.py",
                    "tests/test_normalize.py",
                ],
                "test_files": ["tests/test_normalize.py"],
                "likely_commands": ["python -m unittest tests.test_normalize"],
            }
        ),
        validation_status="failed",
        edit_generation=2,
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_normalize",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_normalize"],
    )
    payload = route_payload(
        intent="update",
        action_plan=[
            {"step": 1, "action": "update_artifact", "reason": "Repair the keep-case implementation."},
            {"step": 2, "action": "run_validation", "reason": "Rerun the targeted unittest module."},
        ],
        target_paths=["texttools/normalize.py", "normalize_cli.py", "README.md", "tests/test_normalize.py"],
        target_name="texttools/normalize.py",
    )
    commit_task_state_and_route(
        planner,
        session,
        payload,
        verification_target="Run python -m unittest tests.test_normalize and finish only when it passes.",
    )
    session.changed_files.extend(
        [
            FileChangeRecord(path="texttools/normalize.py", operation="write"),
            FileChangeRecord(path="texttools/__init__.py", operation="write"),
            FileChangeRecord(path="normalize_cli.py", operation="write"),
        ]
    )
    failed_run = ValidationRunRecord(
        command="python -m unittest tests.test_normalize",
        kind="test",
        verification_scope="runtime",
        status="failed",
        edit_generation=2,
        iteration=12,
        summary="Validation command exited with 1.",
        excerpt=(
            "ImportError: Failed to import test module: test_normalize\n"
            "Traceback (most recent call last):\n"
            f'  File "{test_path}", line 1, in <module>\n'
            "    from texttools import normalize_words, normalize_words_keep_case\n"
            "ImportError: cannot import name 'normalize_words_keep_case' "
            f"from 'texttools.normalize' ({normalize_path})\n"
            "FAILED (errors=1)\n"
        ),
    )
    session.validation_runs.append(failed_run)
    repair_context = planner.validation_planner.build_failure_evidence(session, failed_run)
    assert repair_context is not None
    session.active_repair_context = repair_context
    for path in ("texttools/normalize.py", "texttools/__init__.py", "normalize_cli.py"):
        session.repair_history.append(
            RepairAttemptRecord(
                artifact_path=path,
                validation_command=repair_context.command,
                verification_scope=repair_context.verification_scope,
                strategy="validation_targeted",
                result="no_effective_change",
                reason="file hash unchanged",
                evidence_signature=repair_context.evidence_signature,
                failure_signature=getattr(repair_context.repair_brief, "failure_signature", None),
                iteration=12,
            )
        )

    next_target = planner._repair_target_after_failed_validation(
        session.router_result,
        session,
        failed_run,
        repair_context,
    )

    assert next_target == "texttools/normalize.py"


def test_review_guided_retry_can_escalate_to_full_for_broad_updates(tmp_path, monkeypatch):
    llm = ScriptedLLM(
        text_payloads=[
            "def main(argv=None):\n    return argv or []\n",
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:7b",
            router_model_name="qwen2.5-coder:7b",
        ),
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Update app.py for a broader refactor.",
        workspace_root=str(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Revise a broad update."}],
        target_paths=["app.py"],
        target_name="app.py",
    )
    commit_task_state_and_route(planner, session, payload)
    monkeypatch.setattr(
        planner,
        "_pre_write_update_review",
        lambda *_args, **_kwargs: ProposedUpdateReview(
            safe_to_write=True,
            summary="ok",
            confidence=0.9,
            blocking_issues=[],
            preservation_risks=[],
            repair_hints=[],
        ),
    )

    broad_current_content = "\n".join(f"line_{index} = {index}" for index in range(320))

    result = planner._retry_update_after_review_failure(
        session.router_result,
        session,
        path="app.py",
        current_content=broad_current_content,
        review_feedback=ProposedUpdateReview(
            safe_to_write=False,
            summary="The proposal needs a broader revision.",
            confidence=0.75,
            blocking_issues=["The broad refactor still misses requested coverage."],
            preservation_risks=[],
            repair_hints=["Pull more surrounding context into the next draft."],
        ),
        repair_context=None,
        repair_strategy=None,
        prior_attempts=[],
    )

    assert result.content == "def main(argv=None):\n    return argv or []"
    assert len(llm.generate_calls) == 1
    assert llm.generate_calls[0]["kwargs"]["strict_timeouts"] is False
    assert llm.generate_calls[0]["kwargs"]["num_ctx"] == 4096


def test_review_guided_retry_skips_duplicate_same_model_prompt_followup(tmp_path, monkeypatch):
    llm = ScriptedLLM(
        text_payloads=[
            "def main():\n    return 'draft'\n",
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:7b",
            router_model_name="qwen2.5-coder:7b",
        ),
    )
    planner = Planner(llm, "")
    session = SessionState(
        task="Update app.py so main accepts optional argv while keeping the edit focused.",
        workspace_root=str(tmp_path),
    )
    payload = route_payload(
        intent="update",
        action_plan=[{"step": 1, "action": "update_artifact", "reason": "Refine the focused update."}],
        target_paths=["app.py"],
        target_name="app.py",
    )
    commit_task_state_and_route(planner, session, payload)

    repeated_review = ProposedUpdateReview(
        safe_to_write=False,
        summary="The proposal still ignores the optional argv path.",
        confidence=0.9,
        blocking_issues=["The proposal still ignores the optional argv path."],
        preservation_risks=[],
        repair_hints=["Handle argv=None inside main() instead of hard-coding a fixed return value."],
    )
    monkeypatch.setattr(planner, "_pre_write_update_review", lambda *_args, **_kwargs: repeated_review)

    result = planner._retry_update_after_review_failure(
        session.router_result,
        session,
        path="app.py",
        current_content="def main():\n    return []\n",
        review_feedback=repeated_review,
        repair_context=None,
        repair_strategy=None,
        prior_attempts=[],
    )

    assert result.content is None
    assert result.review == repeated_review
    assert len(llm.generate_calls) == 1


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


def test_planner_finishes_productive_create_targets_before_docs_and_validation(tmp_path):
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
        text_payloads=[
            "import unittest\n\n\nclass WordfreqTests(unittest.TestCase):\n    def test_example(self):\n        self.assertEqual('hello'.upper(), 'HELLO')\n"
        ],
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
    assert decision.tool_args["path"] == "tests/test_wordfreq.py"


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
            "import unittest\n\n\nclass WordfreqTests(unittest.TestCase):\n    def test_example(self):\n        self.assertEqual('hello'.upper(), 'HELLO')\n",
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
    assert second_decision.tool_args["path"] == "tests/test_wordfreq.py"
    assert llm.generate_calls[2]["kwargs"]["model"] == "qwen2.5-coder:14b"


def test_planner_defers_supporting_create_docs_until_first_validation(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task=(
            "Create calcstats/stats.py, calcstats/__main__.py, README.md, and tests/test_stats.py, "
            "then finish only after python -m unittest tests.test_stats passes."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_stats",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_stats"],
    )
    payload = route_payload(
        intent="create",
        action_plan=[
            {"step": 1, "action": "create_artifact", "reason": "Create the requested starter files."},
            {"step": 2, "action": "run_validation", "reason": "Validate the generated package."},
        ],
        target_paths=["calcstats/stats.py", "calcstats/__main__.py", "README.md", "tests/test_stats.py"],
        target_name="calcstats/stats.py",
    )
    commit_task_state_and_route(planner, session, payload, verification_target="python -m unittest tests.test_stats")
    session.task_state.target_artifacts = [
        TaskArtifact(path="calcstats/stats.py", name="calcstats/stats.py", kind="file", role="primary_target", confidence=1.0),
        TaskArtifact(path="calcstats/__main__.py", name="calcstats/__main__.py", kind="file", role="primary_target", confidence=1.0),
        TaskArtifact(path="tests/test_stats.py", name="tests/test_stats.py", kind="test", role="validation_target", confidence=1.0),
        TaskArtifact(path="README.md", name="README.md", kind="doc", role="supporting_context", confidence=1.0),
    ]
    session.changed_files = [
        FileChangeRecord(path="calcstats/stats.py", operation="create"),
        FileChangeRecord(path="calcstats/__main__.py", operation="create"),
        FileChangeRecord(path="tests/test_stats.py", operation="create"),
    ]

    assert planner._active_deferred_create_targets(session.router_result, session) == {"README.md"}
    assert planner._has_pending_explicit_create_targets(session.router_result, session) is False

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "run_tests"
    assert decision.tool_args["command"] == "python -m unittest tests.test_stats"


def test_planner_reactivates_supporting_create_docs_after_first_passed_validation(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task=(
            "Create calcstats/stats.py, calcstats/__main__.py, README.md, and tests/test_stats.py, "
            "then finish only after python -m unittest tests.test_stats passes."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_stats",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_stats"],
    )
    payload = route_payload(
        intent="create",
        action_plan=[
            {"step": 1, "action": "create_artifact", "reason": "Create the requested starter files."},
            {"step": 2, "action": "run_validation", "reason": "Validate the generated package."},
        ],
        target_paths=["calcstats/stats.py", "calcstats/__main__.py", "README.md", "tests/test_stats.py"],
        target_name="calcstats/stats.py",
    )
    commit_task_state_and_route(planner, session, payload, verification_target="python -m unittest tests.test_stats")
    session.task_state.target_artifacts = [
        TaskArtifact(path="calcstats/stats.py", name="calcstats/stats.py", kind="file", role="primary_target", confidence=1.0),
        TaskArtifact(path="calcstats/__main__.py", name="calcstats/__main__.py", kind="file", role="primary_target", confidence=1.0),
        TaskArtifact(path="tests/test_stats.py", name="tests/test_stats.py", kind="test", role="validation_target", confidence=1.0),
        TaskArtifact(path="README.md", name="README.md", kind="doc", role="supporting_context", confidence=1.0),
    ]
    session.changed_files = [
        FileChangeRecord(path="calcstats/stats.py", operation="create"),
        FileChangeRecord(path="calcstats/__main__.py", operation="create"),
        FileChangeRecord(path="tests/test_stats.py", operation="create"),
    ]
    session.validation_runs = [
        ValidationRunRecord(
            command="python -m unittest tests.test_stats",
            status="passed",
            kind="test",
            verification_scope="runtime",
        )
    ]
    session.validation_status = "passed"

    assert planner._active_deferred_create_targets(session.router_result, session) == set()
    assert planner._has_pending_explicit_create_targets(session.router_result, session) is True
    assert planner._choose_create_path(session.router_result, session) == "README.md"


def test_planner_preserves_explicit_primary_markdown_create_target_over_validation_file(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task=(
            "Create docs/repo-map.md for this inventory repo, then run "
            "python -m unittest tests.test_repo_map before finishing."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_repo_map",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_repo_map"],
    )
    payload = route_payload(
        intent="create",
        action_plan=[
            {"step": 1, "action": "create_artifact", "reason": "Create the requested repo map."},
            {"step": 2, "action": "run_validation", "reason": "Validate the generated document."},
        ],
        target_paths=["docs/repo-map.md", "tests/test_repo_map.py"],
        target_name="docs/repo-map.md",
    )
    commit_task_state_and_route(planner, session, payload, verification_target="python -m unittest tests.test_repo_map")
    session.task_state.target_artifacts = [
        TaskArtifact(path="docs/repo-map.md", name="repo-map.md", kind="doc", role="primary_target", confidence=1.0),
        TaskArtifact(path="tests/test_repo_map.py", name="tests/test_repo_map.py", kind="test", role="validation_target", confidence=1.0),
    ]

    assert planner._explicit_create_target_roles(session.router_result, session) == {
        "docs/repo-map.md": "primary_target",
        "tests/test_repo_map.py": "validation_target",
    }
    assert planner._ordered_create_targets(session.router_result, session) == ["docs/repo-map.md"]
    assert planner._choose_create_path(session.router_result, session) == "docs/repo-map.md"


def test_planner_does_not_treat_validation_target_as_pending_create_artifact(tmp_path):
    planner = Planner(ScriptedLLM(), "")
    session = SessionState(
        task=(
            "Create docs/repo-map.md for this inventory repo, then run "
            "python -m unittest tests.test_repo_map before finishing."
        ),
        workspace_root=str(tmp_path),
        workspace_snapshot=empty_snapshot(tmp_path),
        validation_plan=[
            ValidationCommand(
                command="python -m unittest tests.test_repo_map",
                kind="test",
                verification_scope="runtime",
            )
        ],
        verification_commands=["python -m unittest tests.test_repo_map"],
    )
    payload = route_payload(
        intent="create",
        action_plan=[
            {"step": 1, "action": "create_artifact", "reason": "Create the requested repo map."},
            {"step": 2, "action": "run_validation", "reason": "Validate the generated document."},
        ],
        target_paths=["docs/repo-map.md", "tests/test_repo_map.py"],
        target_name="docs/repo-map.md",
    )
    commit_task_state_and_route(planner, session, payload, verification_target="python -m unittest tests.test_repo_map")
    session.task_state.target_artifacts = [
        TaskArtifact(path="docs/repo-map.md", name="repo-map.md", kind="doc", role="primary_target", confidence=1.0),
        TaskArtifact(path="tests/test_repo_map.py", name="tests/test_repo_map.py", kind="test", role="validation_target", confidence=1.0),
    ]
    session.changed_files = [FileChangeRecord(path="docs/repo-map.md", operation="create")]

    assert planner._ordered_create_targets(session.router_result, session) == ["docs/repo-map.md"]
    assert planner._has_pending_explicit_create_targets(session.router_result, session) is False


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
    assert attempts[0].strategy == "switch_to_primary_model"
    assert attempts[0].model_name is None
    assert attempts[0].capability_tier == "tier_a"
    assert attempts[0].prompt_kind == "full"


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
    assert attempts[0].prompt_kind == "full"
    assert attempts[0].strategy == "fallback_model"


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
    assert attempts[0].prompt_kind == "full"
    assert attempts[0].model_name is None


def test_planner_prefers_compact_same_model_retry_for_high_pressure_single_model_no_start(tmp_path):
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
            context_pressure_estimate="high",
            retryable=True,
            raw_reason="startup_timeout",
        )
    )

    assert attempts
    assert attempts[0].strategy == "compact_same_model"
    assert attempts[0].prompt_kind == "compact"
    assert attempts[0].model_name is None
    assert any(attempt.strategy == "retry_same_model" and attempt.prompt_kind == "full" for attempt in attempts[1:])


def test_planner_recovers_from_retryable_no_start_with_same_model_retry(tmp_path):
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
                "timed out waiting for the model to start streaming after 100.1 seconds",
                reason="startup_timeout",
                retryable=True,
                model_name="qwen2.5-coder:7b",
                startup_timeout_seconds=100,
                total_timeout_seconds=150,
            ),
            "print('gui version')\n",
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:7b",
            router_model_name="qwen2.5-coder:7b",
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
    assert llm.generate_calls[1]["kwargs"]["model"] is None
    assert llm.generate_calls[1]["kwargs"]["timeout"] >= 60
    assert llm.generate_calls[1]["kwargs"]["total_timeout"] >= 210
    assert llm.generate_calls[1]["kwargs"]["num_ctx"] == 4096


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
    assert total_timeout_seconds == 270
    assert num_ctx == 3072


def test_planner_retries_resume_once_after_no_start_during_partial_progress_recovery(tmp_path):
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
                "timed out waiting for model completion after 150.0 seconds",
                reason="total_timeout",
                partial_text="print('gui",
                characters=10,
            ),
            OllamaGenerationError(
                "timed out waiting for the model to start streaming after 100.1 seconds",
                reason="startup_timeout",
                retryable=True,
                model_name="qwen2.5-coder:7b",
                startup_timeout_seconds=100,
                total_timeout_seconds=210,
            ),
            "print('gui version')\n",
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:7b",
            router_model_name="qwen2.5-coder:7b",
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
    assert len(llm.generate_calls) == 3
    assert llm.generate_calls[1]["kwargs"]["model"] is None
    assert llm.generate_calls[2]["kwargs"]["model"] is None
    assert "Partial draft from the previous attempt:" in llm.generate_calls[1]["args"][0]
    assert "print('gui" in llm.generate_calls[2]["args"][0]
    assert llm.generate_calls[2]["kwargs"]["timeout"] >= 60
    assert llm.generate_calls[2]["kwargs"]["total_timeout"] >= 270


def test_planner_preserves_progress_budget_for_compact_retry_after_resume_no_start(tmp_path):
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
                "timed out waiting for model completion after 150.0 seconds",
                reason="total_timeout",
                partial_text="print('gui",
                characters=10,
            ),
            OllamaGenerationError(
                "timed out waiting for the model to start streaming after 140.1 seconds",
                reason="startup_timeout",
                retryable=True,
                model_name="qwen2.5-coder:7b",
                startup_timeout_seconds=140,
                total_timeout_seconds=270,
            ),
            OllamaGenerationError(
                "timed out waiting for the model to start streaming after 140.1 seconds",
                reason="startup_timeout",
                retryable=True,
                model_name="qwen2.5-coder:7b",
                startup_timeout_seconds=140,
                total_timeout_seconds=270,
            ),
            "print('gui version')\n",
        ],
        config=AppConfig(
            workspace_root=str(tmp_path),
            model_name="qwen2.5-coder:7b",
            router_model_name="qwen2.5-coder:7b",
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
    assert len(llm.generate_calls) == 4
    assert llm.generate_calls[3]["kwargs"]["model"] is None
    assert llm.generate_calls[3]["kwargs"]["timeout"] >= 60
    assert llm.generate_calls[3]["kwargs"]["total_timeout"] >= 270


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
