from __future__ import annotations

from pathlib import Path

import pytest

from agent.core import AgentCore
from agent.decision import ExecutionDecisionPolicy
from agent.models import ChatMessage, FollowUpContext, SessionState, WorkspaceSnapshot
from agent.planner import Planner
from agent.prompts import task_state_update_prompt
from agent.state_updater import TaskStateUpdater
from agent.task_state import EvidenceItem, TaskState
from agent.task_schema import TaskArtifact, TaskPlanStep, TaskUnderstanding
from agent.understanding import TaskInterpreter
from config.settings import AppConfig
from llm.ollama_client import OllamaGenerationError
from llm.schemas import AgentActionType, RouteActionName, RouteIntent
from runtime.logger import AgentLogger


class ScriptedLLM:
    def __init__(self, json_payloads=None, *, fail: bool = False, fail_times: int = 0, fail_message: str = "timed out"):
        self.json_payloads = list(json_payloads or [])
        self.fail = fail
        self.fail_times = fail_times
        self.fail_message = fail_message
        self.generate_json_calls: list[dict] = []

    def generate_json(self, *args, **kwargs):
        self.generate_json_calls.append({"args": args, "kwargs": kwargs})
        if self.fail:
            raise RuntimeError(self.fail_message)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError(self.fail_message)
        if not self.json_payloads:
            raise RuntimeError("No JSON payload configured")
        return self.json_payloads.pop(0)

    def generate(self, *args, **kwargs):
        raise RuntimeError("Text generation not configured for this test")


class ModelSelectiveLLM(ScriptedLLM):
    def __init__(
        self,
        payload_by_model: dict[str, dict],
        *,
        failing_models: set[str] | None = None,
        fail_message: str = "timed out",
        config=None,
    ):
        super().__init__(json_payloads=[])
        self.payload_by_model = dict(payload_by_model)
        self.failing_models = set(failing_models or set())
        self.fail_message = fail_message
        self.config = config

    def generate_json(self, *args, **kwargs):
        self.generate_json_calls.append({"args": args, "kwargs": kwargs})
        model = str(kwargs.get("model") or "")
        if model in self.failing_models:
            raise RuntimeError(self.fail_message)
        payload = self.payload_by_model.get(model)
        if payload is None:
            raise RuntimeError(f"No JSON payload configured for model {model!r}")
        return payload


class StartupTimeoutLLM(ScriptedLLM):
    def __init__(self, *, config=None):
        super().__init__(json_payloads=[])
        self.config = config

    def generate_json(self, *args, **kwargs):
        self.generate_json_calls.append({"args": args, "kwargs": kwargs})
        raise OllamaGenerationError(
            "timed out waiting for the model to start streaming after 38.0 seconds",
            reason="startup_timeout",
            retryable=False,
            model_name=str(kwargs.get("model") or "") or None,
            startup_timeout_seconds=38,
            inactivity_timeout_seconds=18,
            total_timeout_seconds=38,
        )


def build_snapshot(tmp_path: Path) -> WorkspaceSnapshot:
    return WorkspaceSnapshot(
        root=str(tmp_path),
        file_count=4,
        language_counts={"python": 4},
        top_directories=["app", "tests"],
        important_files=["app/main.py", "app/auth.py", "app/upload.py", "tests/test_auth.py"],
        focus_files=["app/main.py"],
        file_briefs={
            "app/main.py": "Application entrypoint.",
            "app/auth.py": "Authentication and authorization helpers.",
            "app/upload.py": "Upload flow implementation.",
        },
        manifests=["pyproject.toml"],
        configs=["pyproject.toml"],
        test_files=["tests/test_auth.py"],
        build_files=[],
        deploy_files=[],
        entrypoints=["app/main.py"],
        repo_map=["app/", "tests/"],
        service_files=["app/auth.py"],
        import_hotspots=["app/auth.py"],
        symbol_index={
            "app/main.py": ["main"],
            "app/auth.py": ["check_role", "login_user"],
            "app/upload.py": ["upload_file"],
        },
        project_labels=["python"],
        likely_commands=["python -m pytest"],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Python application with auth and upload flows.",
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
        service_files=[],
        import_hotspots=[],
        symbol_index={},
        project_labels=[],
        likely_commands=[],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Empty workspace.",
    )


def calcstats_snapshot(tmp_path: Path) -> WorkspaceSnapshot:
    return WorkspaceSnapshot(
        root=str(tmp_path),
        file_count=4,
        language_counts={"python": 4},
        top_directories=["calcstats", "tests"],
        important_files=["calcstats/stats.py", "calcstats/__init__.py", "tests/test_stats.py", "README.md"],
        focus_files=["calcstats/stats.py"],
        file_briefs={
            "calcstats/stats.py": "Numeric report helpers.",
            "tests/test_stats.py": "Validation for the stats helpers.",
        },
        manifests=["README.md"],
        configs=[],
        test_files=["tests/test_stats.py"],
        build_files=[],
        deploy_files=[],
        entrypoints=[],
        repo_map=["calcstats/", "tests/"],
        test_mappings=["tests/test_stats.py -> calcstats/stats.py"],
        service_files=[],
        import_hotspots=["calcstats/__init__.py"],
        symbol_index={
            "calcstats/stats.py": ["moving_average"],
            "calcstats/__init__.py": ["moving_average"],
        },
        project_labels=["python"],
        likely_commands=["python -m unittest tests.test_stats"],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Python utilities for numeric reports.",
    )


def task_state_from_understanding(understanding: TaskUnderstanding) -> TaskState:
    relation_map = {
        "new_task": "new_task",
        "same_task_follow_up": "continue",
        "refinement": "refine",
        "correction": "correct",
        "problem_report": "report_problem",
        "constraint_update": "scope_change",
        "clarification": "clarify",
        "unknown": "unknown",
    }
    next_action_map = {
        "create": "create",
        "build": "create",
        "modify": "modify",
        "refactor": "modify",
        "debug": "debug",
        "test": "test",
        "explain": "explain",
        "analyze": "inspect",
        "inspect": "inspect",
        "search": "search",
        "plan": "plan",
        "clarify": "clarify",
    }
    evidence = [
        EvidenceItem(kind="message", summary=item, confidence=0.6)
        for item in understanding.supplied_evidence
    ]
    return TaskState(
        latest_user_turn=understanding.original_request,
        root_goal=understanding.interpreted_goal,
        active_goal=understanding.interpreted_goal,
        goal_relation=relation_map.get(understanding.conversation_relation, "unknown"),
        output_expectation=understanding.interpreted_goal,
        open_problem=None,
        verification_target=None,
        target_artifacts=understanding.target_artifacts,
        evidence=evidence,
        relevant_context=understanding.relevant_context,
        constraints=understanding.constraints,
        assumptions=understanding.assumptions,
        missing_info=understanding.missing_info,
        ambiguity_level=understanding.ambiguity_level,
        risk_level=understanding.risk_level,
        confidence=understanding.confidence,
        next_action=next_action_map.get(understanding.recommended_mode, "inspect"),
        execution_outline=[step.summary for step in understanding.execution_plan] or [understanding.interpreted_goal],
        needs_clarification=understanding.needs_clarification,
        clarification_questions=understanding.clarification_questions,
    )


def understanding_payload(
    prompt: str,
    *,
    interpreted_goal: str,
    intent_category: str,
    recommended_mode: str,
    confidence: float,
    conversation_relation: str = "new_task",
    target_artifacts=None,
    constraints=None,
    assumptions=None,
    missing_info=None,
    needs_clarification: bool = False,
    clarification_questions=None,
):
    return {
        "original_request": prompt,
        "interpreted_goal": interpreted_goal,
        "intent_category": intent_category,
        "conversation_relation": conversation_relation,
        "subgoals": ["Inspect the current implementation", "Apply the smallest safe change"],
        "target_artifacts": target_artifacts or [],
        "relevant_context": ["Current workspace context is available."],
        "constraints": constraints or [],
        "missing_info": missing_info or [],
        "assumptions": assumptions or [],
        "user_observations": [],
        "supplied_evidence": [],
        "ambiguity_level": "medium" if confidence < 0.85 else "low",
        "risk_level": "medium",
        "confidence": confidence,
        "recommended_mode": recommended_mode,
        "execution_plan": [
            {
                "step": 1,
                "summary": "Inspect the most relevant implementation area.",
                "action_hint": "inspect",
                "requires_tools": True,
            },
            {
                "step": 2,
                "summary": "Apply the requested change.",
                "action_hint": recommended_mode,
                "requires_tools": True,
            },
        ],
        "needs_clarification": needs_clarification,
        "clarification_questions": clarification_questions or [],
    }


@pytest.mark.parametrize(
    ("prompt", "expected_goal"),
    [
        ("make this cleaner", "Improve maintainability and code clarity of the current implementation."),
        (
            "refactor this so it's easier to maintain",
            "Improve maintainability and code clarity of the current implementation.",
        ),
    ],
)
def test_task_interpreter_normalizes_paraphrases_to_same_goal(prompt: str, expected_goal: str):
    llm = ScriptedLLM(
        [
            understanding_payload(
                prompt,
                interpreted_goal=expected_goal,
                intent_category="refactor",
                recommended_mode="refactor",
                confidence=0.86,
                target_artifacts=[
                    {
                        "path": "app/main.py",
                        "name": "app/main.py",
                        "kind": "file",
                        "role": "primary_target",
                        "confidence": 0.82,
                    }
                ],
            )
        ]
    )
    interpreter = TaskInterpreter(llm)

    understanding = interpreter.interpret(prompt)

    assert understanding.interpreted_goal == expected_goal
    assert understanding.intent_category == "refactor"
    assert understanding.recommended_mode == "refactor"
    assert understanding.target_artifacts[0].path == "app/main.py"


def test_execution_policy_maps_semantic_understanding_to_update_route(tmp_path):
    understanding = TaskUnderstanding(
        original_request="I want the login to be safer",
        interpreted_goal="Strengthen the login flow security without changing its main user journey.",
        intent_category="configure",
        conversation_relation="new_task",
        subgoals=["Inspect the login/auth flow", "Tighten insecure behavior"],
        target_artifacts=[
            TaskArtifact(
                path="app/auth.py",
                name="app/auth.py",
                kind="file",
                role="primary_target",
                confidence=0.83,
            )
        ],
        relevant_context=["The workspace contains an auth module."],
        constraints=["Preserve the existing login flow."],
        missing_info=[],
        assumptions=["The backend auth module is the right target."],
        user_observations=[],
        supplied_evidence=[],
        ambiguity_level="low",
        risk_level="medium",
        confidence=0.79,
        recommended_mode="modify",
        execution_plan=[
            TaskPlanStep(step=1, summary="Inspect auth.py", action_hint="inspect"),
            TaskPlanStep(step=2, summary="Harden the login behavior", action_hint="modify"),
        ],
        needs_clarification=False,
        clarification_questions=[],
    )

    route = ExecutionDecisionPolicy().build_route(
        task_state_from_understanding(understanding),
        snapshot=build_snapshot(tmp_path),
        session=SessionState(task=understanding.original_request, workspace_root=str(tmp_path)),
    )

    assert route.intent == RouteIntent.UPDATE
    assert route.entities.target_paths == ["app/auth.py"]
    assert route.action_plan[0].action == RouteActionName.READ_RELEVANT_FILES
    assert any(step.action == RouteActionName.UPDATE_ARTIFACT for step in route.action_plan)
    assert any(step.action == RouteActionName.RUN_VALIDATION for step in route.action_plan)


def test_execution_policy_uses_follow_up_context_for_refinement(tmp_path):
    understanding = TaskUnderstanding(
        original_request="do it more securely",
        interpreted_goal="Tighten the security of the current auth implementation.",
        intent_category="modify",
        conversation_relation="refinement",
        subgoals=["Continue from the current auth work"],
        target_artifacts=[],
        relevant_context=["Previous auth work exists in the session context."],
        constraints=[],
        missing_info=[],
        assumptions=["Continue with the previously modified backend auth file."],
        user_observations=[],
        supplied_evidence=[],
        ambiguity_level="medium",
        risk_level="medium",
        confidence=0.77,
        recommended_mode="modify",
        execution_plan=[
            TaskPlanStep(step=1, summary="Inspect the active auth file", action_hint="inspect"),
            TaskPlanStep(step=2, summary="Harden the implementation", action_hint="modify"),
        ],
        needs_clarification=False,
        clarification_questions=[],
    )
    session = SessionState(
        task=understanding.original_request,
        workspace_root=str(tmp_path),
        follow_up_context=FollowUpContext(
            previous_task="add auth to this app",
            previous_interpreted_goal="Add basic auth to the backend app.",
            previous_recommended_mode="modify",
            target_paths=["app/auth.py"],
            changed_files=["app/auth.py"],
        ),
    )

    route = ExecutionDecisionPolicy().build_route(
        task_state_from_understanding(understanding),
        snapshot=build_snapshot(tmp_path),
        session=session,
    )

    assert route.intent == RouteIntent.UPDATE
    assert route.entities.target_paths[0] == "app/auth.py"
    assert route.safe_to_execute is True


def test_execution_policy_routes_implement_request_to_update_even_if_mode_is_inspect(tmp_path):
    task_state = TaskState(
        latest_user_turn="add a theme toggle to index.html, styles.css, and app.js",
        root_goal="Add a theme toggle to the landing page.",
        active_goal="Add a theme toggle to the landing page.",
        goal_relation="new_task",
        output_expectation="A working theme toggle integrated into the landing page.",
        current_user_intent="implement",
        execution_strategy=None,
        open_problem=None,
        verification_target="The theme toggle should work and persist.",
        target_artifacts=[
            TaskArtifact(path="index.html", name="index.html", kind="file", role="primary_target", confidence=0.9),
            TaskArtifact(path="styles.css", name="styles.css", kind="file", role="primary_target", confidence=0.9),
            TaskArtifact(path="app.js", name="app.js", kind="file", role="primary_target", confidence=0.9),
        ],
        evidence=[],
        relevant_context=[],
        constraints=["Only modify the named files."],
        assumptions=[],
        missing_info=[],
        ambiguity_level="low",
        risk_level="medium",
        confidence=0.9,
        next_action="inspect",
        next_best_action="inspect",
        execution_outline=["Inspect the existing files", "Implement the theme toggle", "Validate the result"],
        needs_clarification=False,
        clarification_questions=[],
    )

    route = ExecutionDecisionPolicy().build_route(
        task_state,
        snapshot=build_snapshot(tmp_path),
        session=SessionState(task=task_state.latest_user_turn, workspace_root=str(tmp_path)),
    )

    assert route.intent == RouteIntent.UPDATE
    assert route.entities.target_paths == ["index.html", "styles.css", "app.js"]
    assert route.action_plan[0].action == RouteActionName.READ_RELEVANT_FILES
    assert any(step.action == RouteActionName.UPDATE_ARTIFACT for step in route.action_plan)


def test_execution_policy_requests_clarification_on_low_confidence_high_risk():
    understanding = TaskUnderstanding(
        original_request="actually revert that part",
        interpreted_goal="Revert a previously changed part of the implementation.",
        intent_category="modify",
        conversation_relation="correction",
        subgoals=["Identify the exact change to revert"],
        target_artifacts=[],
        relevant_context=["There was previous work, but the exact scope is unclear."],
        constraints=[],
        missing_info=["Which exact file or change should be reverted?"],
        assumptions=[],
        user_observations=[],
        supplied_evidence=[],
        ambiguity_level="high",
        risk_level="high",
        confidence=0.33,
        recommended_mode="clarify",
        execution_plan=[
            TaskPlanStep(step=1, summary="Ask for the exact revert scope", action_hint="plan"),
        ],
        needs_clarification=True,
        clarification_questions=["Which exact file or change should I revert?"],
    )

    route = ExecutionDecisionPolicy().build_route(task_state_from_understanding(understanding))

    assert route.intent == RouteIntent.UNKNOWN
    assert route.safe_to_execute is False
    assert route.needs_clarification is True
    assert "revert" in route.clarification_questions[0].lower()


@pytest.mark.parametrize(
    ("prompt", "expected_extension"),
    [
        ("mach ein tic tac toe in python", ".py"),
        ("ich moechte ein schiffe versenken spiel in python haben", ".py"),
        ("bau mir ein snake spiel in javascript", ".js"),
        ("erstelle eine kleine flask api mit login", ".py"),
    ],
)
def test_execution_policy_allows_clear_create_requests_without_explicit_filename(prompt: str, expected_extension: str):
    task_state = TaskState(
        latest_user_turn=prompt,
        root_goal=prompt,
        active_goal=prompt,
        goal_relation="new_task",
        output_expectation="Create a small runnable implementation with sensible defaults.",
        open_problem=None,
        verification_target="Create the initial implementation and validate the most relevant path.",
        target_artifacts=[],
        evidence=[],
        relevant_context=[],
        constraints=[],
        assumptions=["A conventional default artifact is acceptable for this request."],
        missing_info=[],
        ambiguity_level="low",
        risk_level="low",
        confidence=0.82,
        next_action="create",
        execution_outline=["Choose a conventional default artifact", "Implement it", "Verify it"],
        needs_clarification=False,
        clarification_questions=[],
    )

    route = ExecutionDecisionPolicy().build_route(task_state)

    assert route.intent == RouteIntent.CREATE
    assert route.needs_clarification is False
    assert route.safe_to_execute is True
    assert route.entities.target_name
    assert expected_extension in route.relevant_extensions


def test_task_state_updater_tracks_continuation_with_evidence():
    prompt = "why is this failing?"
    llm = ScriptedLLM(
        [
            {
                "latest_user_turn": prompt,
                "root_goal": "Fix the failing upload flow.",
                "active_goal": "Diagnose the failing upload flow and identify the highest-evidence cause.",
                "goal_relation": "report_problem",
                "output_expectation": "A diagnosis and, if supported, a focused fix.",
                "open_problem": "AttributeError in upload handler",
                "verification_target": "Rerun the failing upload path and confirm it passes.",
                "target_artifacts": [
                    {
                        "path": "app/upload.py",
                        "name": "app/upload.py",
                        "kind": "file",
                        "role": "primary_target",
                        "confidence": 0.8,
                    }
                ],
                "evidence": [
                    {
                        "kind": "diagnostic",
                        "summary": "AttributeError in upload handler",
                        "source": "pytest",
                        "artifact_path": "app/upload.py",
                        "confidence": 0.9,
                    }
                ],
                "relevant_context": ["Previous validation already failed in the upload flow."],
                "constraints": [],
                "assumptions": ["The latest traceback points at app/upload.py."],
                "missing_info": [],
                "ambiguity_level": "low",
                "risk_level": "medium",
                "confidence": 0.86,
                "next_action": "debug",
                "execution_outline": [
                    "Inspect the implicated upload handler.",
                    "Reproduce or validate the failing path.",
                    "Patch the highest-evidence cause.",
                ],
                "needs_clarification": False,
                "clarification_questions": [],
            }
        ]
    )
    updater = TaskStateUpdater(llm)

    task_state = updater.update_task_state(prompt)

    assert task_state.root_goal == "Fix the failing upload flow."
    assert task_state.goal_relation == "report_problem"
    assert task_state.next_action == "debug"
    assert task_state.open_problem == "AttributeError in upload handler"
    assert task_state.evidence[0].artifact_path == "app/upload.py"


def test_task_state_updater_accepts_string_evidence_items_from_model():
    prompt = "add a state-root option without removing existing CLI behavior"
    llm = ScriptedLLM(
        [
            {
                "latest_user_turn": prompt,
                "root_goal": "Add a CLI state-root option safely.",
                "active_goal": "Add a state-root override without regressing the CLI.",
                "goal_relation": "new_task",
                "output_expectation": "A focused CLI update with preserved behavior.",
                "open_problem": None,
                "verification_target": "CLI options should keep existing behavior while adding the override.",
                "target_artifacts": [
                    {
                        "path": "cli.py",
                        "name": "cli.py",
                        "kind": "file",
                        "role": "primary_target",
                        "confidence": 0.88,
                    }
                ],
                "evidence": [
                    "The existing configuration flow should be preserved while introducing new CLI parameters."
                ],
                "relevant_context": [],
                "constraints": ["Preserve existing CLI behavior."],
                "assumptions": [],
                "missing_info": [],
                "ambiguity_level": "low",
                "risk_level": "medium",
                "confidence": 0.84,
                "next_action": "modify",
                "execution_outline": [
                    "Inspect the CLI entry point.",
                    "Add the override in the smallest coherent change.",
                    "Verify the CLI behavior.",
                ],
                "needs_clarification": False,
                "clarification_questions": [],
            }
        ]
    )
    updater = TaskStateUpdater(llm)

    task_state = updater.update_task_state(prompt)

    assert task_state.target_artifacts[0].path == "cli.py"
    assert task_state.evidence[0].summary == (
        "The existing configuration flow should be preserved while introducing new CLI parameters."
    )
    assert task_state.evidence[0].kind == "unknown"
    assert task_state.supplied_evidence == [
        "The existing configuration flow should be preserved while introducing new CLI parameters."
    ]


def test_task_state_normalize_does_not_infer_phase_two_operational_fields():
    state = TaskState(
        latest_user_turn="why is this failing?",
        root_goal="Fix the failing upload flow.",
        active_goal="Diagnose the failing upload flow and identify the highest-evidence cause.",
        goal_relation="report_problem",
        output_expectation="A diagnosis and, if supported, a focused fix.",
        open_problem="AttributeError in upload handler",
        verification_target="Rerun the failing upload path and confirm it passes.",
        target_artifacts=[
            TaskArtifact(
                path="app/upload.py",
                name="app/upload.py",
                kind="file",
                role="primary_target",
                confidence=0.82,
            )
        ],
        evidence=[
            EvidenceItem(
                kind="diagnostic",
                summary="AttributeError in upload handler",
                source="pytest",
                artifact_path="app/upload.py",
                confidence=0.92,
            )
        ],
        relevant_context=[],
        constraints=[],
        assumptions=[],
        missing_info=[],
        ambiguity_level="low",
        risk_level="medium",
        confidence=0.88,
        next_action="debug",
        execution_outline=["Inspect app/upload.py", "Reproduce the failure", "Fix and verify"],
        needs_clarification=False,
        clarification_questions=[],
    )

    assert state.current_user_intent is None
    assert state.execution_strategy is None
    assert state.next_best_action == "debug"
    assert state.active_artifacts[0].path == "app/upload.py"
    assert state.supplied_evidence == ["AttributeError in upload handler"]


@pytest.mark.parametrize(
    ("prompt", "expected_extension"),
    [
        ("mach ein tic tac toe in python", ".py"),
        ("ich moechte ein schiffe versenken spiel in python haben", ".py"),
        ("bau mir ein snake spiel in javascript", ".js"),
        ("erstelle eine kleine flask api mit login", ".py"),
    ],
)
def test_task_state_fallback_treats_clear_new_build_requests_as_executable_creates(
    tmp_path,
    prompt: str,
    expected_extension: str,
):
    updater = TaskStateUpdater(ScriptedLLM())

    task_state = updater.update_task_state(prompt, snapshot=build_snapshot(tmp_path))

    assert task_state.goal_relation == "new_task"
    assert task_state.next_action == "create"
    assert task_state.needs_clarification is False
    assert task_state.confidence >= 0.68
    assert task_state.target_artifacts
    assert task_state.target_artifacts[0].kind == expected_extension


@pytest.mark.parametrize("prompt", ["mach das besser", "aendere das", "baue das um"])
def test_task_state_fallback_still_clarifies_vague_requests_without_active_context(tmp_path, prompt: str):
    updater = TaskStateUpdater(ScriptedLLM())

    task_state = updater.update_task_state(prompt, snapshot=build_snapshot(tmp_path))

    assert task_state.next_action == "clarify"
    assert task_state.needs_clarification is True


def test_task_state_timeout_fallback_preserves_clear_create_request_without_invented_specialization(tmp_path):
    updater = TaskStateUpdater(ScriptedLLM(fail=True, fail_message="timed out"))
    session = SessionState(
        task="mach login sicherer",
        workspace_root=str(tmp_path),
        task_state=TaskState(
            latest_user_turn="mach login sicherer",
            root_goal="Improve login resilience.",
            active_goal="Harden the login flow without broad rewrites.",
            goal_relation="refine",
            output_expectation="A safer login flow.",
            current_user_intent="harden",
            execution_strategy="hardening",
            open_problem=None,
            verification_target="Apply the highest-value hardening change and verify the intended behavior still works.",
            target_artifacts=[
                TaskArtifact(path="app/auth.py", name="app/auth.py", kind="file", role="primary_target", confidence=0.84)
            ],
            evidence=[],
            relevant_context=[],
            constraints=[],
            assumptions=[],
            missing_info=[],
            ambiguity_level="low",
            risk_level="medium",
            confidence=0.86,
            next_action="modify",
            execution_outline=["Inspect auth.py", "Harden it", "Verify login path"],
            needs_clarification=False,
            clarification_questions=[],
        ),
        follow_up_context=FollowUpContext(
            previous_task="mach login sicherer",
            previous_root_goal="Improve login resilience.",
            previous_active_goal="Harden the login flow without broad rewrites.",
            previous_next_action="modify",
            previous_requested_outcome="A safer login flow.",
            target_paths=["app/auth.py"],
            changed_files=["app/auth.py"],
            read_files=["app/auth.py"],
        ),
    )

    task_state = updater.update_task_state(
        "ich brauche ein snake spiel in html",
        snapshot=build_snapshot(tmp_path),
        session=session,
    )

    assert task_state.goal_relation == "new_task"
    assert task_state.next_action == "create"
    assert task_state.current_user_intent == "implement"
    assert task_state.execution_strategy == "feature_implementation"
    assert task_state.needs_clarification is False
    assert "hardening" not in task_state.output_expectation.lower()
    assert "hardening" not in (task_state.verification_target or "").lower()
    assert task_state.execution_strategy != "hardening"
    assert task_state.semantic_resolution == "minimal_inference"


def test_task_state_primary_model_preserves_full_model_semantics_for_clear_create_request(tmp_path):
    config = type("Cfg", (), {"router_model_name": "router-a", "model_name": "reserve-b"})()
    llm = ModelSelectiveLLM(
        {
            "router-a": {
                "latest_user_turn": "ich brauche ein snake spiel in html",
                "root_goal": "Build a small Snake game in HTML.",
                "active_goal": "Create the first playable Snake implementation.",
                "goal_relation": "new_task",
                "output_expectation": "Create a small runnable implementation with a conventional default artifact and minimal scope.",
                "current_user_intent": "implement",
                "execution_strategy": "feature_implementation",
                "open_problem": None,
                "verification_target": "Create the initial implementation and run the most relevant validation or entry command.",
                "target_artifacts": [
                    {
                        "path": None,
                        "name": "snake",
                        "kind": ".html",
                        "role": "primary_target",
                        "confidence": 0.8,
                    }
                ],
                "active_artifacts": [],
                "evidence": [],
                "relevant_context": [],
                "constraints": [],
                "assumptions": ["A conventional default artifact is acceptable."],
                "missing_info": [],
                "ambiguity_level": "low",
                "risk_level": "low",
                "confidence": 0.9,
                "next_action": "create",
                "next_best_action": "create",
                "execution_outline": ["Choose a conventional default artifact.", "Implement the smallest runnable version."],
                "needs_clarification": False,
                "clarification_questions": [],
            }
        },
        config=config,
    )

    task_state = TaskStateUpdater(llm).update_task_state(
        "ich brauche ein snake spiel in html",
        snapshot=empty_snapshot(tmp_path),
    )

    assert task_state.goal_relation == "new_task"
    assert task_state.next_action == "create"
    assert task_state.current_user_intent == "implement"
    assert task_state.execution_strategy == "feature_implementation"
    assert task_state.semantic_resolution == "full_model"


def test_task_state_uses_reserve_model_before_minimal_inference_for_clear_create_request(tmp_path):
    config = type("Cfg", (), {"router_model_name": "router-a", "model_name": "reserve-b"})()
    reserve_payload = {
        "latest_user_turn": "ich brauche ein snake spiel in html",
        "root_goal": "Build a small Snake game in HTML.",
        "active_goal": "Create the first playable Snake implementation.",
        "goal_relation": "new_task",
        "output_expectation": "Create a small runnable implementation with a conventional default artifact and minimal scope.",
        "current_user_intent": "implement",
        "execution_strategy": "feature_implementation",
        "open_problem": None,
        "verification_target": "Create the initial implementation and run the most relevant validation or entry command.",
        "target_artifacts": [
            {
                "path": None,
                "name": "snake",
                "kind": ".html",
                "role": "primary_target",
                "confidence": 0.78,
            }
        ],
        "active_artifacts": [],
        "evidence": [],
        "relevant_context": [],
        "constraints": [],
        "assumptions": ["A conventional default artifact is acceptable."],
        "missing_info": [],
        "ambiguity_level": "low",
        "risk_level": "low",
        "confidence": 0.84,
        "next_action": "create",
        "next_best_action": "create",
        "execution_outline": ["Choose a conventional default artifact.", "Implement the smallest runnable version."],
        "needs_clarification": False,
        "clarification_questions": [],
    }
    llm = ModelSelectiveLLM(
        {"reserve-b": reserve_payload},
        failing_models={"router-a"},
        config=config,
    )

    task_state = TaskStateUpdater(llm).update_task_state(
        "ich brauche ein snake spiel in html",
        snapshot=empty_snapshot(tmp_path),
    )

    assert task_state.goal_relation == "new_task"
    assert task_state.next_action == "create"
    assert task_state.current_user_intent == "implement"
    assert task_state.execution_strategy == "feature_implementation"
    assert task_state.semantic_resolution == "reduced_model"
    assert any(call["kwargs"].get("model") == "reserve-b" for call in llm.generate_json_calls)
    assert llm.generate_json_calls[1]["kwargs"]["timeout"] == 18
    assert llm.generate_json_calls[1]["kwargs"]["total_timeout"] == 72
    assert llm.generate_json_calls[1]["kwargs"]["num_ctx"] == 2048


def test_task_state_uses_larger_primary_model_as_reserve_when_smaller_router_times_out(tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen3:14b",
        router_model_name="qwen3:8b",
    )
    reserve_payload = {
        "latest_user_turn": "ich brauche ein snake spiel in html",
        "root_goal": "Build a small Snake game in HTML.",
        "active_goal": "Create the first playable Snake implementation.",
        "goal_relation": "new_task",
        "output_expectation": "Create a small runnable implementation with a conventional default artifact and minimal scope.",
        "current_user_intent": "implement",
        "execution_strategy": "feature_implementation",
        "open_problem": None,
        "verification_target": "Create the initial implementation and run the most relevant validation or entry command.",
        "target_artifacts": [
            {
                "path": None,
                "name": "snake",
                "kind": ".html",
                "role": "primary_target",
                "confidence": 0.78,
            }
        ],
        "active_artifacts": [],
        "evidence": [],
        "relevant_context": [],
        "constraints": [],
        "assumptions": ["A conventional default artifact is acceptable."],
        "missing_info": [],
        "ambiguity_level": "low",
        "risk_level": "low",
        "confidence": 0.84,
        "next_action": "create",
        "next_best_action": "create",
        "execution_outline": ["Choose a conventional default artifact.", "Implement the smallest runnable version."],
        "needs_clarification": False,
        "clarification_questions": [],
    }
    llm = ModelSelectiveLLM(
        {"qwen3:14b": reserve_payload},
        failing_models={"qwen3:8b"},
        config=config,
    )

    task_state = TaskStateUpdater(llm).update_task_state(
        "ich brauche ein snake spiel in html",
        snapshot=empty_snapshot(tmp_path),
    )

    assert task_state.goal_relation == "new_task"
    assert task_state.next_action == "create"
    assert task_state.current_user_intent == "implement"
    assert task_state.execution_strategy == "feature_implementation"
    assert task_state.semantic_resolution == "reduced_model"
    assert [call["kwargs"].get("model") for call in llm.generate_json_calls[:2]] == [
        "qwen3:8b",
        "qwen3:14b",
    ]
    assert llm.generate_json_calls[1]["kwargs"]["timeout"] == 18
    assert llm.generate_json_calls[1]["kwargs"]["total_timeout"] == 72
    assert llm.generate_json_calls[1]["kwargs"]["num_ctx"] == 2048


def test_task_state_short_circuits_to_deterministic_fallback_after_fresh_compact_no_start(tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen3:14b",
        router_model_name="qwen3:8b",
    )
    logger = AgentLogger(tmp_path, "task-state-short-circuit")
    llm = StartupTimeoutLLM(config=config)
    updater = TaskStateUpdater(llm, logger=logger)

    task_state = updater.update_task_state(
        "ich brauche ein snake spiel in html",
        snapshot=empty_snapshot(tmp_path),
    )

    assert task_state.goal_relation == "new_task"
    assert task_state.next_action == "create"
    assert task_state.semantic_resolution == "minimal_inference"
    assert [call["kwargs"].get("model") for call in llm.generate_json_calls] == ["qwen3:8b"]
    log_text = logger.log_path.read_text(encoding="utf-8")
    assert "task_state_recovery_short_circuit" in log_text


def test_task_state_uses_local_short_circuit_when_primary_and_router_models_match(tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen2.5-coder:7b",
        router_model_name="qwen2.5-coder:7b",
    )
    logger = AgentLogger(tmp_path, "task-state-local-short-circuit")
    llm = ScriptedLLM()
    llm.config = config
    updater = TaskStateUpdater(llm, logger=logger)

    task_state = updater.update_task_state(
        "fix den fehler in app/upload.py",
        snapshot=build_snapshot(tmp_path),
    )

    assert task_state.current_user_intent == "repair"
    assert task_state.next_action == "debug"
    assert task_state.semantic_resolution == "minimal_inference"


def test_task_state_local_short_circuit_keeps_relevant_test_artifact_for_cli_feature_requests(tmp_path):
    snapshot = WorkspaceSnapshot(
        root=str(tmp_path),
        file_count=4,
        language_counts={"python": 3, "markdown": 1},
        top_directories=["greet_cli", "tests"],
        important_files=["greet_cli/cli.py", "greet_cli/__main__.py", "README.md", "tests/test_cli.py"],
        focus_files=["greet_cli/cli.py", "greet_cli/__main__.py", "tests/test_cli.py"],
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
        repo_summary="Small CLI package runnable with python -m greet_cli.",
    )
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen2.5-coder:7b",
        router_model_name="qwen2.5-coder:7b",
    )
    llm = ScriptedLLM()
    llm.config = config
    updater = TaskStateUpdater(llm)

    task_state = updater.update_task_state(
        (
            "Add a --uppercase flag to this CLI without moving the argument parsing out of greet_cli/__main__.py. "
            "Keep greet_cli/cli.py as a small helper that only returns the greeting string. "
            "Update the tests and README so the new flag is covered and documented, then run the relevant tests."
        ),
        snapshot=snapshot,
    )

    artifact_roles = {artifact.path: artifact.role for artifact in task_state.target_artifacts}

    assert task_state.semantic_resolution == "minimal_inference"
    assert {artifact.path for artifact in task_state.target_artifacts} >= {
        "greet_cli/__main__.py",
        "greet_cli/cli.py",
        "README.md",
        "tests/test_cli.py",
    }
    assert artifact_roles["tests/test_cli.py"] == "validation_target"
    assert llm.generate_json_calls == []


def test_task_state_local_short_circuit_infers_readme_for_empty_workspace_create_request(tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen2.5-coder:7b",
        router_model_name="qwen2.5-coder:7b",
    )
    llm = ScriptedLLM()
    llm.config = config
    updater = TaskStateUpdater(llm)

    task_state = updater.update_task_state(
        (
            "Create a small Python CLI called wordfreq.py that reads a text file path argument and prints the most "
            "common words with counts, ignoring case and punctuation. Add unittest coverage in tests/test_wordfreq.py "
            "with real assertions, add a short README with usage, and run python -m unittest tests.test_wordfreq."
        ),
        snapshot=empty_snapshot(tmp_path),
    )

    artifact_roles = {artifact.path: artifact.role for artifact in task_state.target_artifacts}

    assert task_state.semantic_resolution == "minimal_inference"
    assert {artifact.path for artifact in task_state.target_artifacts} >= {
        "wordfreq.py",
        "README.md",
        "tests/test_wordfreq.py",
    }
    assert artifact_roles["README.md"] == "supporting_context"
    assert artifact_roles["tests/test_wordfreq.py"] == "validation_target"
    assert llm.generate_json_calls == []


def test_task_state_local_short_circuit_inferrs_named_package_code_before_docs_in_empty_workspace(tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen2.5-coder:7b",
        router_model_name="qwen2.5-coder:7b",
    )
    llm = ScriptedLLM()
    llm.config = config
    updater = TaskStateUpdater(llm)

    prompt = (
        "Create a small Python package named calcstats with library helpers and a CLI entrypoint. "
        "The CLI should accept integers from the command line and support --sum, --mean, and --median output modes. "
        "Add a unittest suite in tests/test_stats.py that covers both the library functions and the CLI behavior, "
        "write a concise README with usage examples, and finish only after python -m unittest tests.test_stats passes."
    )
    task_state = updater.update_task_state(prompt, snapshot=empty_snapshot(tmp_path))
    route = ExecutionDecisionPolicy().build_route(task_state, snapshot=empty_snapshot(tmp_path))

    artifact_roles = {artifact.path: artifact.role for artifact in task_state.target_artifacts if artifact.path}

    assert task_state.semantic_resolution == "minimal_inference"
    assert [artifact.path for artifact in task_state.target_artifacts[:3]] == [
        "calcstats/stats.py",
        "calcstats/__main__.py",
        "calcstats/__init__.py",
    ]
    assert {artifact.path for artifact in task_state.target_artifacts} >= {
        "calcstats/stats.py",
        "calcstats/__main__.py",
        "calcstats/__init__.py",
        "README.md",
        "tests/test_stats.py",
    }
    assert artifact_roles["README.md"] == "supporting_context"
    assert artifact_roles["tests/test_stats.py"] == "validation_target"
    assert route.entities.target_name == "calcstats/stats.py"
    assert route.entities.target_paths[:3] == [
        "calcstats/stats.py",
        "calcstats/__main__.py",
        "calcstats/__init__.py",
    ]
    assert llm.generate_json_calls == []


def test_task_state_local_short_circuit_preserves_explicit_multi_file_website_targets(tmp_path):
    snapshot = WorkspaceSnapshot(
        root=str(tmp_path),
        file_count=6,
        language_counts={"html": 4, "css": 1, "python": 1},
        top_directories=["tests"],
        important_files=["tests/test_site.py", "about.html", "contact.html", "index.html", "projects.html", "styles.css"],
        focus_files=[],
        file_briefs={},
        manifests=[],
        configs=[],
        test_files=["tests/test_site.py"],
        build_files=[],
        deploy_files=[],
        entrypoints=[],
        repo_map=["tests/"],
        project_labels=["website"],
        likely_commands=["python -m unittest tests.test_site"],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Starter portfolio website.",
    )
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen2.5-coder:7b",
        router_model_name="qwen2.5-coder:7b",
    )
    llm = ScriptedLLM()
    llm.config = config
    updater = TaskStateUpdater(llm)

    task_state = updater.update_task_state(
        (
            "You are working in an existing starter portfolio website. "
            "Turn it into a polished multi-page portfolio without breaking the current pages. "
            "Keep index.html and about.html coherent, update projects.html so it contains a visible Case Studies section, "
            "update contact.html so the form uses class=\"contact-form\", update styles.css so it defines a .site-grid layout "
            "and keeps the look consistent across pages, then run python -m unittest tests.test_site and finish only when it passes."
        ),
        snapshot=snapshot,
    )
    route = ExecutionDecisionPolicy().build_route(task_state, snapshot=snapshot)
    artifact_roles = {artifact.path: artifact.role for artifact in task_state.target_artifacts}

    assert task_state.semantic_resolution == "minimal_inference"
    assert {artifact.path for artifact in task_state.target_artifacts} >= {
        "index.html",
        "about.html",
        "projects.html",
        "contact.html",
        "styles.css",
        "tests/test_site.py",
    }
    assert artifact_roles["tests/test_site.py"] == "validation_target"
    assert {path for path in route.entities.target_paths} >= {
        "index.html",
        "about.html",
        "projects.html",
        "contact.html",
        "styles.css",
        "tests/test_site.py",
    }
    assert llm.generate_json_calls == []


def test_task_state_local_short_circuit_prioritizes_debug_for_bugfix_prompts_with_search_words(tmp_path):
    snapshot = WorkspaceSnapshot(
        root=str(tmp_path),
        file_count=4,
        language_counts={"python": 3, "markdown": 1},
        top_directories=["textutils", "tests"],
        important_files=["README.md", "tests/test_normalize.py", "textutils/__init__.py", "textutils/normalize.py"],
        focus_files=["tests/test_normalize.py", "textutils/normalize.py"],
        file_briefs={},
        manifests=["README.md"],
        configs=[],
        test_files=["tests/test_normalize.py"],
        build_files=[],
        deploy_files=[],
        entrypoints=[],
        repo_map=["textutils/", "tests/"],
        project_labels=["python"],
        likely_commands=["python -m unittest tests.test_normalize"],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Small Python helper package with unittest coverage for name normalization.",
    )
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen2.5-coder:7b",
        router_model_name="qwen2.5-coder:7b",
    )
    llm = ScriptedLLM()
    llm.config = config
    updater = TaskStateUpdater(llm)

    task_state = updater.update_task_state(
        (
            "There is a bug in this repo causing the name normalization tests to fail. "
            "Find the problem, fix the implementation without changing the intended behavior, "
            "do not weaken the tests, and run the relevant tests."
        ),
        snapshot=snapshot,
    )
    route = ExecutionDecisionPolicy().build_route(task_state, snapshot=snapshot)

    assert task_state.semantic_resolution == "minimal_inference"
    assert task_state.current_user_intent == "repair"
    assert task_state.execution_strategy == "debug_repair"
    assert task_state.next_action == "debug"
    assert task_state.target_artifacts[0].path == "textutils/normalize.py"
    assert {artifact.path for artifact in task_state.target_artifacts} >= {
        "textutils/normalize.py",
        "tests/test_normalize.py",
    }
    assert route.intent == RouteIntent.DEBUG
    assert llm.generate_json_calls == []


def test_task_state_a2_shared_model_stack_requires_semantic_model_for_mutation_tasks(tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen2.5-coder:7b",
        router_model_name="qwen2.5-coder:7b",
    )
    llm = ScriptedLLM(
        json_payloads=[
            {
                "latest_user_turn": "Fix the upload bug in app/upload.py without weakening the tests.",
                "root_goal": "Fix the upload bug in app/upload.py without weakening the tests.",
                "active_goal": "Diagnose and fix the upload bug in the current repo.",
                "goal_relation": "report_problem",
                "output_expectation": "Diagnose the issue, apply the smallest safe fix, and verify the result.",
                "current_user_intent": "repair",
                "execution_strategy": "debug_repair",
                "open_problem": "Upload bug in app/upload.py.",
                "verification_target": "Reproduce the bug and rerun the relevant tests.",
                "target_artifacts": [
                    {"path": "app/upload.py", "name": "upload.py", "kind": ".py", "role": "primary_target", "confidence": 0.92},
                    {"path": "tests/test_auth.py", "name": "test_auth.py", "kind": ".py", "role": "validation_target", "confidence": 0.78},
                ],
                "active_artifacts": [
                    {"path": "app/upload.py", "name": "upload.py", "kind": ".py", "role": "primary_target", "confidence": 0.92},
                ],
                "evidence": [],
                "supplied_evidence": [],
                "relevant_context": [],
                "constraints": [],
                "assumptions": [],
                "missing_info": [],
                "ambiguity_level": "low",
                "risk_level": "medium",
                "confidence": 0.83,
                "next_action": "debug",
                "next_best_action": "debug",
                "execution_outline": [
                    "Read the strongest target first.",
                    "Reproduce the failure before editing.",
                    "Apply the smallest safe fix and rerun tests.",
                ],
                "needs_clarification": False,
                "clarification_questions": [],
            }
        ]
    )
    llm.config = config
    updater = TaskStateUpdater(llm)
    session = SessionState(
        task="Fix the upload bug in app/upload.py without weakening the tests.",
        workspace_root=str(tmp_path),
        runtime_options={"agent_profile": "a2"},
    )

    task_state = updater.update_task_state(
        "Fix the upload bug in app/upload.py without weakening the tests.",
        snapshot=build_snapshot(tmp_path),
        session=session,
    )

    assert task_state.semantic_resolution == "full_model"
    assert task_state.execution_strategy == "debug_repair"
    assert task_state.target_artifacts[0].path == "app/upload.py"
    assert llm.generate_json_calls


def test_task_state_a2_blocks_mutation_when_semantic_model_is_unavailable(tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen2.5-coder:7b",
        router_model_name="qwen2.5-coder:7b",
    )
    llm = StartupTimeoutLLM(config=config)
    updater = TaskStateUpdater(llm)
    session = SessionState(
        task="Fix the upload bug in app/upload.py without weakening the tests.",
        workspace_root=str(tmp_path),
        runtime_options={"agent_profile": "a2"},
    )

    task_state = updater.update_task_state(
        "Fix the upload bug in app/upload.py without weakening the tests.",
        snapshot=build_snapshot(tmp_path),
        session=session,
    )

    assert task_state.semantic_resolution == "blocked"
    assert task_state.needs_clarification is True
    assert task_state.next_action == "clarify"
    assert task_state.target_artifacts == []
    assert llm.generate_json_calls


def test_task_state_a2_no_start_runs_recovery_before_blocking(tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen3:14b",
        router_model_name="qwen2.5-coder:7b",
    )
    logger = AgentLogger(tmp_path, "task-state-a2-no-start-recovery")
    llm = StartupTimeoutLLM(config=config)
    updater = TaskStateUpdater(llm, logger=logger)
    session = SessionState(
        task="Fix the upload bug in app/upload.py without weakening the tests.",
        workspace_root=str(tmp_path),
        runtime_options={"agent_profile": "a2"},
    )

    task_state = updater.update_task_state(
        "Fix the upload bug in app/upload.py without weakening the tests.",
        snapshot=build_snapshot(tmp_path),
        session=session,
    )

    models = [call["kwargs"].get("model") for call in llm.generate_json_calls]

    assert task_state.semantic_resolution == "blocked"
    assert len(models) > 1
    assert "qwen2.5-coder:7b" in models
    log_text = logger.log_path.read_text(encoding="utf-8")
    assert "task_state_recovery_short_circuit" not in log_text


def test_task_state_a2_preserves_full_retry_mode_for_semantic_recovery(tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen3:14b",
        router_model_name="qwen2.5-coder:7b",
    )
    llm = StartupTimeoutLLM(config=config)
    updater = TaskStateUpdater(llm)
    session = SessionState(
        task="Fix the upload bug in app/upload.py without weakening the tests.",
        workspace_root=str(tmp_path),
        runtime_options={"agent_profile": "a2"},
    )

    updater.update_task_state(
        "Fix the upload bug in app/upload.py without weakening the tests.",
        snapshot=build_snapshot(tmp_path),
        session=session,
    )

    assert len(llm.generate_json_calls) >= 2
    first_call = llm.generate_json_calls[0]["kwargs"]
    assert first_call["timeout"] == 18
    assert first_call["num_ctx"] == 2048
    assert any(
        call["kwargs"]["timeout"] == updater.timeout
        and call["kwargs"]["num_ctx"] == updater.num_ctx
        for call in llm.generate_json_calls[1:]
    )


def test_task_state_timeout_fallback_preserves_clear_explain_request(tmp_path):
    updater = TaskStateUpdater(ScriptedLLM(fail=True, fail_message="timed out"))

    task_state = updater.update_task_state(
        "erklär mir wie die upload route funktioniert",
        snapshot=build_snapshot(tmp_path),
    )
    route = ExecutionDecisionPolicy().build_route(task_state, snapshot=build_snapshot(tmp_path))

    assert task_state.goal_relation == "new_task"
    assert task_state.current_user_intent == "explain"
    assert task_state.execution_strategy == "validation_inspection"
    assert task_state.next_action == "explain"
    assert route.intent == RouteIntent.EXPLAIN
    assert route.needs_clarification is False
    assert task_state.semantic_resolution == "minimal_inference"


def test_task_interpreter_timeout_fallback_preserves_clear_create_request(tmp_path):
    interpreter = TaskInterpreter(ScriptedLLM(fail=True, fail_message="timed out"))

    understanding = interpreter.interpret(
        "ich brauche ein snake spiel in html",
        session=SessionState(task="ich brauche ein snake spiel in html", workspace_root=str(tmp_path)),
    )

    assert understanding.intent_category == "build"
    assert understanding.conversation_relation == "new_task"
    assert understanding.recommended_mode == "create"
    assert understanding.needs_clarification is False
    assert understanding.semantic_resolution == "minimal_inference"


def test_task_state_timeout_fallback_preserves_clear_debug_request(tmp_path):
    updater = TaskStateUpdater(ScriptedLLM(fail=True, fail_message="timed out"))

    task_state = updater.update_task_state(
        "fix den fehler in app/auth.py",
        snapshot=build_snapshot(tmp_path),
    )
    route = ExecutionDecisionPolicy().build_route(task_state, snapshot=build_snapshot(tmp_path))

    assert task_state.goal_relation == "report_problem"
    assert task_state.current_user_intent == "repair"
    assert task_state.execution_strategy == "debug_repair"
    assert task_state.next_action == "debug"
    assert task_state.target_artifacts[0].path == "app/auth.py"
    assert route.intent == RouteIntent.DEBUG
    assert route.needs_clarification is False
    assert task_state.semantic_resolution == "minimal_inference"


def test_task_state_timeout_fallback_prioritizes_update_over_validation_for_compound_change_request(tmp_path):
    snapshot = WorkspaceSnapshot(
        root=str(tmp_path),
        file_count=4,
        language_counts={"python": 3, "markdown": 1},
        top_directories=["tests"],
        important_files=["cli.py", "README.md", "tests/test_cli.py", "bootstrap_runtime.py"],
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
        likely_commands=["python -m unittest"],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Small CLI project with README and unittest coverage.",
    )
    updater = TaskStateUpdater(ScriptedLLM(fail=True, fail_message="timed out"))
    prompt = (
        "Bitte fuege --state-root zur CLI hinzu. "
        "Behalte --config und --verbose unveraendert bei, aktualisiere README, "
        "erweitere den unittest fuer die neue Option und fuehre danach python -m unittest aus."
    )

    task_state = updater.update_task_state(prompt, snapshot=snapshot)
    route = ExecutionDecisionPolicy().build_route(task_state, snapshot=snapshot)

    assert task_state.current_user_intent == "implement"
    assert task_state.next_action == "modify"
    assert task_state.needs_clarification is False
    assert task_state.target_artifacts[0].path == "cli.py"
    assert {artifact.path for artifact in task_state.target_artifacts} >= {"cli.py", "README.md", "tests/test_cli.py"}
    assert "Apply the requested change" in (task_state.verification_target or "")
    assert route.intent == RouteIntent.UPDATE
    assert route.needs_clarification is False


def test_task_state_timeout_fallback_prioritizes_package_main_for_cli_feature_requests(tmp_path):
    snapshot = WorkspaceSnapshot(
        root=str(tmp_path),
        file_count=4,
        language_counts={"python": 3, "markdown": 1},
        top_directories=["greet_cli", "tests"],
        important_files=["greet_cli/cli.py", "greet_cli/__main__.py", "README.md", "tests/test_cli.py"],
        focus_files=["greet_cli/cli.py", "greet_cli/__main__.py", "tests/test_cli.py"],
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
        repo_summary="Small CLI package runnable with python -m greet_cli.",
    )
    updater = TaskStateUpdater(ScriptedLLM(fail=True, fail_message="timed out"))
    prompt = (
        "Extend the existing greet_cli package so the CLI accepts a --uppercase flag that prints the greeting in uppercase. "
        "Update only what is needed, update the README usage example, add or update unittests, "
        "run python -m unittest tests.test_cli, and finish only when the tests pass."
    )

    task_state = updater.update_task_state(prompt, snapshot=snapshot)
    route = ExecutionDecisionPolicy().build_route(task_state, snapshot=snapshot)
    artifact_roles = {artifact.path: artifact.role for artifact in task_state.target_artifacts}

    assert task_state.current_user_intent == "implement"
    assert task_state.next_action == "modify"
    assert task_state.needs_clarification is False
    assert task_state.target_artifacts[0].path == "greet_cli/__main__.py"
    assert {artifact.path for artifact in task_state.target_artifacts} >= {
        "greet_cli/__main__.py",
        "greet_cli/cli.py",
        "README.md",
        "tests/test_cli.py",
    }
    assert artifact_roles["tests/test_cli.py"] == "validation_target"
    assert artifact_roles["README.md"] == "supporting_context"
    assert route.intent == RouteIntent.UPDATE
    assert route.entities.target_name == "greet_cli/__main__.py"


def test_task_state_timeout_fallback_prefers_cli_helper_over_package_init_for_cli_feature_requests(tmp_path):
    snapshot = WorkspaceSnapshot(
        root=str(tmp_path),
        file_count=5,
        language_counts={"python": 4, "markdown": 1},
        top_directories=["greet_cli", "tests"],
        important_files=["README.md", "tests/test_cli.py", "greet_cli/__main__.py", "greet_cli/cli.py", "greet_cli/__init__.py"],
        focus_files=["tests/test_cli.py", "greet_cli/__main__.py", "greet_cli/cli.py", "greet_cli/__init__.py"],
        file_briefs={},
        manifests=["README.md"],
        configs=[],
        test_files=["tests/test_cli.py"],
        build_files=[],
        deploy_files=[],
        entrypoints=["greet_cli/__main__.py", "greet_cli/cli.py"],
        repo_map=["greet_cli/", "tests/"],
        project_labels=["python"],
        likely_commands=["python3 -m unittest tests.test_cli"],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Small CLI package runnable with python -m greet_cli.",
    )
    updater = TaskStateUpdater(ScriptedLLM(fail=True, fail_message="timed out"))
    prompt = (
        "Extend this existing CLI so it supports --prefix TEXT and --repeat N while keeping --uppercase working. "
        "Update the implementation, the README, and the unittest coverage. "
        "The repeated output should print the full greeting on separate lines. "
        "Finish only when python3 -m unittest tests.test_cli passes."
    )

    task_state = updater.update_task_state(prompt, snapshot=snapshot)

    assert [artifact.path for artifact in task_state.target_artifacts[:2]] == [
        "greet_cli/__main__.py",
        "greet_cli/cli.py",
    ]
    assert {artifact.path for artifact in task_state.target_artifacts} >= {
        "greet_cli/__main__.py",
        "greet_cli/cli.py",
        "README.md",
        "tests/test_cli.py",
    }
    assert "greet_cli/__init__.py" not in {artifact.path for artifact in task_state.target_artifacts}


def test_task_state_model_normalizes_route_style_aliases():
    state = TaskState.model_validate(
        {
            "latest_user_turn": "Update README.md only.",
            "root_goal": "Update the README example.",
            "active_goal": "Modify README.md with the requested command.",
            "goal_relation": "same_task_follow_up",
            "output_expectation": "README.md contains the exact requested example command.",
            "current_user_intent": "update",
            "execution_strategy": "update",
            "next_action": "update",
            "next_best_action": "update",
            "ambiguity_level": "low",
            "risk_level": "low",
            "confidence": 0.92,
        }
    )

    assert state.goal_relation == "continue"
    assert state.current_user_intent == "implement"
    assert state.execution_strategy == "feature_implementation"
    assert state.next_action == "modify"
    assert state.next_best_action == "modify"


def test_task_state_updater_routes_explicit_repair_contract_to_update_flow(tmp_path):
    payload = {
        "latest_user_turn": (
            "Fix normalize_name in text_utils.py so it trims outer whitespace, lowercases text, "
            "converts internal whitespace runs to single hyphens, and preserves existing hyphens. "
            "Add or update unit tests. Validate with python -m unittest."
        ),
        "root_goal": "Implement the requested normalize_name behavior.",
        "active_goal": "Modify text_utils.py and tests/test_text_utils.py for the requested behavior.",
        "goal_relation": "new_task",
        "output_expectation": "normalize_name matches the requested behavior and unit tests pass.",
        "current_user_intent": "repair",
        "execution_strategy": None,
        "verification_target": "python -m unittest",
        "target_artifacts": [
            {"path": "text_utils.py", "name": "normalize_name", "kind": "function", "role": "primary_target", "confidence": 1.0},
            {"path": "tests/test_text_utils.py", "name": "unit tests", "kind": "test", "role": "validation_target", "confidence": 1.0},
        ],
        "relevant_context": ["Update the implementation directly and rerun the tests."],
        "constraints": [],
        "assumptions": [],
        "missing_info": [],
        "ambiguity_level": "low",
        "risk_level": "low",
        "confidence": 0.98,
        "next_action": "modify",
        "next_best_action": "modify",
        "execution_outline": [
            "Read the target implementation and test.",
            "Apply the requested behavior change.",
            "Run python -m unittest.",
        ],
        "needs_clarification": False,
        "clarification_questions": [],
    }

    task_state = TaskStateUpdater(ScriptedLLM(json_payloads=[payload])).update_task_state(
        payload["latest_user_turn"],
        snapshot=build_snapshot(tmp_path),
    )
    route = ExecutionDecisionPolicy().build_route(task_state, snapshot=build_snapshot(tmp_path))

    assert task_state.current_user_intent == "repair"
    assert task_state.execution_strategy == "feature_implementation"
    assert route.intent == RouteIntent.UPDATE
    assert route.action_plan[1].action.value == "update_artifact"


def test_task_state_updater_keeps_empty_workspace_create_requests_out_of_debug_flow(tmp_path):
    payload = {
        "latest_user_turn": "Erstelle in diesem leeren Projekt ein kleines Python-CLI zum Wortzaehlen.",
        "root_goal": "Create a small Python CLI for word counting in an empty project.",
        "active_goal": "Set up the project structure and implement the basic functionality of the word counting CLI.",
        "goal_relation": "new_task",
        "output_expectation": "A working Python CLI that counts words in a text file, ignoring case and punctuation.",
        "current_user_intent": "implement",
        "execution_strategy": "debug_repair",
        "open_problem": "No existing project structure to build upon; need to create the necessary files and implement the functionality.",
        "verification_target": "Running `python wordfreq.py sample.txt` should output the most frequent words with their counts.",
        "target_artifacts": [
            {"path": "wordfreq.py", "name": "wordfreq.py", "kind": "file", "role": "primary_target", "confidence": 1.0},
            {"path": "README.md", "name": "README.md", "kind": "doc", "role": "supporting_context", "confidence": 1.0},
            {"path": "tests/test_wordfreq.py", "name": "test_wordfreq.py", "kind": "test", "role": "validation_target", "confidence": 1.0},
        ],
        "active_artifacts": [],
        "evidence": [],
        "relevant_context": [],
        "constraints": ["Ignore case and punctuation.", "Use `unittest` for testing."],
        "assumptions": ["The user wants a simple implementation without additional features."],
        "missing_info": [],
        "ambiguity_level": "low",
        "risk_level": "low",
        "confidence": 0.98,
        "next_action": "create",
        "next_best_action": "create",
        "execution_outline": [
            "Choose the smallest conventional artifact or scaffold that fits the request.",
            "Implement the requested behavior in minimal runnable scope.",
            "Validate the created artifact with the most relevant command if available.",
        ],
        "needs_clarification": False,
        "clarification_questions": [],
    }

    task_state = TaskStateUpdater(ScriptedLLM(json_payloads=[payload])).update_task_state(
        payload["latest_user_turn"],
        snapshot=empty_snapshot(tmp_path),
    )
    route = ExecutionDecisionPolicy().build_route(task_state, snapshot=empty_snapshot(tmp_path))

    assert task_state.current_user_intent == "implement"
    assert task_state.execution_strategy == "feature_implementation"
    assert task_state.next_action == "create"
    assert route.intent == RouteIntent.CREATE
    assert route.action_plan[0].action.value == "create_artifact"


def test_task_state_updater_reconciles_explain_misclassification_for_existing_repo_feature_request(tmp_path):
    snapshot = WorkspaceSnapshot(
        root=str(tmp_path),
        file_count=4,
        language_counts={"python": 3, "markdown": 1},
        top_directories=["textutils", "tests"],
        important_files=["textutils/__init__.py", "textutils/normalize.py", "tests/test_normalize.py", "README.md"],
        focus_files=["textutils/normalize.py", "textutils/__init__.py", "tests/test_normalize.py"],
        file_briefs={},
        manifests=["README.md"],
        configs=[],
        test_files=["tests/test_normalize.py"],
        build_files=[],
        deploy_files=[],
        entrypoints=["textutils/normalize.py"],
        repo_map=["textutils/", "tests/"],
        project_labels=["python"],
        likely_commands=["python -m unittest tests.test_normalize"],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Small Python utility package with unittest coverage.",
    )
    prompt = (
        "Extend the existing textutils module with a slugify(text) helper that lowercases, replaces whitespace "
        "with hyphens, removes punctuation, export it from __init__.py, add unittests, update the README examples, "
        "and run python -m unittest tests.test_normalize before finishing."
    )
    payload = {
        "latest_user_turn": prompt,
        "root_goal": "Review the existing textutils behavior.",
        "active_goal": "Inspect how slug handling should work in this repository.",
        "goal_relation": "new_task",
        "output_expectation": "Explain the current textutils behavior and summarize the relevant files.",
        "current_user_intent": "explain",
        "execution_strategy": "validation_inspection",
        "open_problem": None,
        "verification_target": None,
        "target_artifacts": [
            {"path": "textutils/normalize.py", "name": "normalize.py", "kind": "file", "role": "primary_target", "confidence": 0.92},
            {"path": "README.md", "name": "README.md", "kind": "doc", "role": "supporting_context", "confidence": 0.74},
        ],
        "active_artifacts": [],
        "evidence": [],
        "relevant_context": ["Inspect the existing implementation before answering."],
        "constraints": [],
        "assumptions": [],
        "missing_info": [],
        "ambiguity_level": "low",
        "risk_level": "low",
        "confidence": 0.83,
        "next_action": "inspect",
        "next_best_action": "explain",
        "execution_outline": [
            "Inspect the relevant files.",
            "Summarize the current implementation.",
        ],
        "needs_clarification": False,
        "clarification_questions": [],
    }

    task_state = TaskStateUpdater(ScriptedLLM(json_payloads=[payload])).update_task_state(
        prompt,
        snapshot=snapshot,
    )
    route = ExecutionDecisionPolicy().build_route(task_state, snapshot=snapshot)

    assert task_state.current_user_intent == "implement"
    assert task_state.execution_strategy == "feature_implementation"
    assert task_state.next_action == "modify"
    assert task_state.next_best_action == "modify"
    assert task_state.target_artifacts[0].path == "textutils/normalize.py"
    assert {artifact.path for artifact in task_state.target_artifacts} >= {
        "textutils/normalize.py",
        "textutils/__init__.py",
        "README.md",
        "tests/test_normalize.py",
    }
    assert "Apply the requested change" in task_state.output_expectation
    assert "Apply the requested change" in (task_state.verification_target or "")
    assert route.intent == RouteIntent.UPDATE


def test_task_state_avoids_unrelated_test_package_init_when_request_targets_other_init(tmp_path):
    snapshot = WorkspaceSnapshot(
        root=str(tmp_path),
        file_count=5,
        language_counts={"python": 4, "markdown": 1},
        top_directories=["textutils", "tests"],
        important_files=[
            "textutils/__init__.py",
            "textutils/normalize.py",
            "tests/test_normalize.py",
            "tests/__init__.py",
            "README.md",
        ],
        focus_files=[
            "textutils/normalize.py",
            "textutils/__init__.py",
            "tests/test_normalize.py",
            "tests/__init__.py",
        ],
        file_briefs={},
        manifests=["README.md"],
        configs=[],
        test_files=["tests/test_normalize.py", "tests/__init__.py"],
        build_files=[],
        deploy_files=[],
        entrypoints=["textutils/normalize.py"],
        repo_map=["textutils/", "tests/"],
        project_labels=["python"],
        likely_commands=["python -m unittest tests.test_normalize"],
        validation_commands=[],
        workflow_commands=[],
        repo_summary="Small Python utility package with unittest coverage.",
    )
    prompt = (
        "Implement the missing public helper slugify(text) in textutils/normalize.py, "
        "export it from textutils/__init__.py, update README.md with a short usage example if needed, "
        "run python -m unittest tests.test_normalize, and finish only after the tests are green."
    )

    task_state = TaskStateUpdater(ScriptedLLM(fail=True, fail_message="timed out")).update_task_state(
        prompt,
        snapshot=snapshot,
    )

    artifact_paths = {artifact.path for artifact in task_state.target_artifacts}

    assert "textutils/normalize.py" in artifact_paths
    assert "textutils/__init__.py" in artifact_paths
    assert "tests/test_normalize.py" in artifact_paths
    assert "README.md" in artifact_paths
    assert "tests/__init__.py" not in artifact_paths


def test_task_state_timeout_fallback_preserves_explicit_file_create_request_in_empty_workspace(tmp_path):
    updater = TaskStateUpdater(ScriptedLLM(fail=True, fail_message="timed out"))

    task_state = updater.update_task_state(
        (
            "Erstelle in diesem leeren Projekt ein kleines Python-CLI zum Wortzaehlen. "
            "Lege wordfreq.py, README.md und tests/test_wordfreq.py an."
        ),
        snapshot=empty_snapshot(tmp_path),
    )
    route = ExecutionDecisionPolicy().build_route(task_state, snapshot=empty_snapshot(tmp_path))

    assert task_state.goal_relation == "new_task"
    assert task_state.current_user_intent == "implement"
    assert task_state.execution_strategy == "feature_implementation"
    assert task_state.next_action == "create"
    assert [artifact.path for artifact in task_state.target_artifacts] == [
        "wordfreq.py",
        "README.md",
        "tests/test_wordfreq.py",
    ]
    assert route.intent == RouteIntent.CREATE
    assert route.entities.target_paths == [
        "wordfreq.py",
        "README.md",
        "tests/test_wordfreq.py",
    ]
    assert route.action_plan[0].action.value == "create_artifact"


def test_task_state_updater_falls_back_when_model_payload_is_invalid(tmp_path):
    invalid_payload = {
        "latest_user_turn": "Update README.md to document the CLI.",
        "root_goal": "Update the README.",
        "active_goal": "Modify README.md with the new CLI usage.",
        "goal_relation": "new_task",
        "output_expectation": "README.md reflects the updated CLI.",
        "current_user_intent": "nonsense",
        "next_action": "modify",
        "ambiguity_level": "low",
        "risk_level": "low",
        "confidence": 0.8,
    }

    task_state = TaskStateUpdater(ScriptedLLM(json_payloads=[invalid_payload])).update_task_state(
        "Update README.md to document the CLI.",
        snapshot=build_snapshot(tmp_path),
    )

    assert task_state.semantic_resolution == "minimal_inference"
    assert task_state.current_user_intent == "implement"
    assert task_state.next_action == "modify"


def test_task_state_timeout_fallback_clarifies_vague_request_without_specialized_strategy(tmp_path):
    updater = TaskStateUpdater(ScriptedLLM(fail=True, fail_message="timed out"))
    session = SessionState(
        task="mach login sicherer",
        workspace_root=str(tmp_path),
        follow_up_context=FollowUpContext(
            previous_task="mach login sicherer",
            previous_root_goal="Improve login resilience.",
            previous_active_goal="Harden the login flow without broad rewrites.",
            previous_next_action="modify",
            previous_requested_outcome="A safer login flow.",
            target_paths=["app/auth.py", "app/session.py"],
            changed_files=["app/auth.py", "app/session.py"],
            read_files=["app/auth.py", "app/session.py"],
        ),
    )

    task_state = updater.update_task_state(
        "mach da was",
        snapshot=build_snapshot(tmp_path),
        session=session,
    )

    assert task_state.next_action == "clarify"
    assert task_state.needs_clarification is True
    assert task_state.current_user_intent == "unknown"
    assert task_state.execution_strategy is None


def test_task_state_fallback_uses_existing_state_evidence_instead_of_prompt_text(tmp_path):
    updater = TaskStateUpdater(ScriptedLLM())
    prompt = "mach weiter"

    noisy_session = SessionState(
        task=prompt,
        workspace_root=str(tmp_path),
        task_state=TaskState(
            latest_user_turn="fix the upload flow",
            root_goal="Fix the upload flow.",
            active_goal="Stabilize the upload flow implementation.",
            goal_relation="new_task",
            output_expectation="A working upload flow.",
            open_problem="Upload regression after the last change.",
            verification_target="Rerun the failing upload command.",
            target_artifacts=[
                TaskArtifact(
                    path="app/upload.py",
                    name="app/upload.py",
                    kind="file",
                    role="primary_target",
                    confidence=0.85,
                )
            ],
            evidence=[],
            relevant_context=[],
            constraints=[],
            assumptions=[],
            missing_info=[],
            ambiguity_level="low",
            risk_level="medium",
            confidence=0.82,
            next_action="modify",
            execution_outline=["Inspect upload.py", "Patch the issue", "Verify it"],
            needs_clarification=False,
            clarification_questions=[],
        ),
        follow_up_context=FollowUpContext(
            previous_task="fix the upload flow",
            previous_root_goal="Fix the upload flow.",
            previous_active_goal="Stabilize the upload flow implementation.",
            previous_next_action="modify",
            target_paths=["app/upload.py"],
            changed_files=["app/upload.py"],
            last_error="AttributeError: uploader is None",
        ),
    )

    quiet_session = SessionState(
        task=prompt,
        workspace_root=str(tmp_path),
        task_state=TaskState(
            latest_user_turn="fix the upload flow",
            root_goal="Fix the upload flow.",
            active_goal="Stabilize the upload flow implementation.",
            goal_relation="new_task",
            output_expectation="A working upload flow.",
            open_problem=None,
            verification_target=None,
            target_artifacts=[],
            evidence=[],
            relevant_context=[],
            constraints=[],
            assumptions=[],
            missing_info=[],
            ambiguity_level="medium",
            risk_level="medium",
            confidence=0.5,
            next_action="modify",
            execution_outline=["Continue the task"],
            needs_clarification=False,
            clarification_questions=[],
        ),
    )

    noisy_state = updater.update_task_state(prompt, snapshot=build_snapshot(tmp_path), session=noisy_session)
    quiet_state = updater.update_task_state(prompt, snapshot=build_snapshot(tmp_path), session=quiet_session)

    assert noisy_state.next_action == "debug"
    assert noisy_state.goal_relation == "report_problem"
    assert noisy_state.target_artifacts[0].path == "app/upload.py"
    assert "AttributeError" in (noisy_state.open_problem or "")
    assert quiet_state.next_action in {"search", "clarify"}
    assert quiet_state.next_action != noisy_state.next_action


def test_task_state_fallback_clarifies_when_multiple_active_artifacts_compete(tmp_path):
    updater = TaskStateUpdater(ScriptedLLM())
    session = SessionState(
        task="fix it",
        workspace_root=str(tmp_path),
        task_state=TaskState(
            latest_user_turn="fix it",
            root_goal="Improve the system safely.",
            active_goal="Continue the current implementation work.",
            goal_relation="continue",
            output_expectation="A safe follow-up change.",
            open_problem=None,
            verification_target=None,
            target_artifacts=[
                TaskArtifact(path="app/ui.py", name="app/ui.py", kind="file", role="primary_target", confidence=0.7),
                TaskArtifact(path="app/api.py", name="app/api.py", kind="file", role="primary_target", confidence=0.7),
            ],
            evidence=[],
            relevant_context=[],
            constraints=[],
            assumptions=[],
            missing_info=[],
            ambiguity_level="medium",
            risk_level="medium",
            confidence=0.6,
            next_action="modify",
            execution_outline=["Continue the current task"],
            needs_clarification=False,
            clarification_questions=[],
        ),
    )

    task_state = updater.update_task_state("mach nur den letzten Teil rueckgaengig", snapshot=build_snapshot(tmp_path), session=session)

    assert task_state.next_action == "clarify"
    assert task_state.needs_clarification is True
    assert "mehrere" in task_state.missing_info[0].lower() or "multiple" in task_state.missing_info[0].lower()


def test_task_state_fallback_clarifies_when_requested_symbol_is_absent_from_repo_map(tmp_path):
    updater = TaskStateUpdater(ScriptedLLM())
    prompt = (
        "Fix the existing calcstats bug so median_value returns the correct median for even-length "
        "numeric lists and run python -m unittest tests.test_stats."
    )

    task_state = updater.update_task_state(prompt, snapshot=calcstats_snapshot(tmp_path))
    route = ExecutionDecisionPolicy().build_route(
        task_state,
        snapshot=calcstats_snapshot(tmp_path),
        session=SessionState(task=prompt, workspace_root=str(tmp_path)),
    )

    assert task_state.next_action == "clarify"
    assert task_state.needs_clarification is True
    assert any("median_value" in item for item in task_state.missing_info)
    assert not any(artifact.path == "calcstats/stats.py" for artifact in task_state.target_artifacts)
    assert route.needs_clarification is True
    assert route.action_plan[0].action == RouteActionName.ASK_CLARIFICATION


def test_task_state_fallback_keeps_repo_target_when_requested_symbol_exists_in_repo_map(tmp_path):
    updater = TaskStateUpdater(ScriptedLLM())
    prompt = (
        "Fix the existing calcstats bug so moving_average returns one average per full window only "
        "and run python -m unittest tests.test_stats."
    )

    task_state = updater.update_task_state(prompt, snapshot=calcstats_snapshot(tmp_path))
    route = ExecutionDecisionPolicy().build_route(
        task_state,
        snapshot=calcstats_snapshot(tmp_path),
        session=SessionState(task=prompt, workspace_root=str(tmp_path)),
    )

    assert task_state.needs_clarification is False
    assert task_state.next_action == "debug"
    assert task_state.target_artifacts
    assert task_state.target_artifacts[0].path == "calcstats/stats.py"
    assert route.needs_clarification is False
    assert route.intent == RouteIntent.DEBUG


def test_task_state_timeout_fallback_keeps_clear_build_request_executable_even_with_follow_up_context(tmp_path):
    logger = AgentLogger(tmp_path, "task-state-timeout-build")
    updater = TaskStateUpdater(ScriptedLLM(), logger=logger)
    session = SessionState(
        task="vorheriger task",
        workspace_root=str(tmp_path),
        follow_up_context=FollowUpContext(
            previous_task="bau mir ein snake spiel in html und javascript",
            previous_root_goal="Build a small Snake game in HTML and JavaScript.",
            previous_active_goal="Create the first playable Snake implementation.",
            previous_next_action="create",
            previous_requested_outcome="Create a small runnable implementation with a conventional default artifact and minimal scope.",
            target_paths=["snake.js"],
            changed_files=["snake.js"],
            read_files=["snake.js"],
        ),
    )

    task_state = updater.update_task_state(
        "ich möchte ein schiffe versenken spiel in python haben",
        snapshot=empty_snapshot(tmp_path),
        session=session,
    )
    route = ExecutionDecisionPolicy(logger=logger).build_route(
        task_state,
        snapshot=empty_snapshot(tmp_path),
        session=session,
    )

    assert task_state.goal_relation == "new_task"
    assert task_state.next_action == "create"
    assert task_state.current_user_intent == "implement"
    assert task_state.needs_clarification is False
    assert task_state.output_expectation.startswith("Create a small runnable implementation")
    assert route.intent == RouteIntent.CREATE
    assert route.needs_clarification is False
    assert route.safe_to_execute is True
    assert route.action_plan[0].action == RouteActionName.CREATE_ARTIFACT

    logs = (tmp_path / "task-state-timeout-build.jsonl").read_text(encoding="utf-8")
    assert "task_state_fallback" in logs
    assert '"goal_relation": "new_task"' in logs
    assert '"chosen_intent": "create"' in logs


def test_task_state_timeout_fallback_treats_html_follow_up_as_continuation_create_not_new_task(tmp_path):
    logger = AgentLogger(tmp_path, "task-state-timeout-follow-up")
    updater = TaskStateUpdater(ScriptedLLM(), logger=logger)
    session = SessionState(
        task="bau mir ein snake spiel in html und javascript",
        workspace_root=str(tmp_path),
        follow_up_context=FollowUpContext(
            previous_task="bau mir ein snake spiel in html und javascript",
            previous_root_goal="Build a small Snake game in HTML and JavaScript.",
            previous_active_goal="Create the first playable Snake implementation.",
            previous_next_action="create",
            previous_requested_outcome="Create a small runnable implementation with a conventional default artifact and minimal scope.",
            target_paths=["snake.js"],
            changed_files=["snake.js"],
            read_files=["snake.js"],
        ),
    )

    task_state = updater.update_task_state(
        "jetzt bau dazu eine html datei zum anzeigen des Spiels",
        snapshot=build_snapshot(tmp_path),
        session=session,
    )
    route = ExecutionDecisionPolicy(logger=logger).build_route(
        task_state,
        snapshot=build_snapshot(tmp_path),
        session=session,
    )

    assert task_state.root_goal == "Build a small Snake game in HTML and JavaScript."
    assert task_state.goal_relation == "refine"
    assert task_state.next_action == "create"
    assert task_state.needs_clarification is False
    assert task_state.active_artifacts[0].path == "snake.js"
    assert route.intent == RouteIntent.CREATE
    assert route.needs_clarification is False
    assert "snake.js" not in route.entities.target_paths
    assert route.action_plan[0].action == RouteActionName.INSPECT_WORKSPACE

    logs = (tmp_path / "task-state-timeout-follow-up.jsonl").read_text(encoding="utf-8")
    assert "task_state_fallback" in logs
    assert '"goal_relation": "refine"' in logs
    assert '"chosen_intent": "create"' in logs


def test_task_state_timeout_fallback_clarifies_ambiguous_follow_up_instead_of_claiming_same_task(tmp_path):
    logger = AgentLogger(tmp_path, "task-state-timeout-tictactoe-menu")
    updater = TaskStateUpdater(ScriptedLLM(), logger=logger)
    session = SessionState(
        task="kannst du jetzt machen das ich aber ein menü habe mit 2 modis einmal gegen computer und einmal 2 spieler modus",
        workspace_root=str(tmp_path),
        follow_up_context=FollowUpContext(
            previous_task="ich möchte ein Tic tac Toe spiel in python haben",
            previous_root_goal="Build a Tic Tac Toe game in Python.",
            previous_active_goal="Create the first playable Tic Tac Toe implementation.",
            previous_next_action="create",
            previous_requested_outcome="Create a small runnable implementation with a conventional default artifact and minimal scope.",
            target_paths=["tic_tac_toe.py"],
            changed_files=["tic_tac_toe.py"],
            read_files=["tic_tac_toe.py"],
            recent_commands=['internal:python_syntax:["tic_tac_toe.py"]'],
        ),
        candidate_files=["tic_tac_toe.py"],
    )

    task_state = updater.update_task_state(
        session.task,
        snapshot=build_snapshot(tmp_path),
        session=session,
    )
    route = ExecutionDecisionPolicy(logger=logger).build_route(
        task_state,
        snapshot=build_snapshot(tmp_path),
        session=session,
    )

    assert task_state.root_goal == session.task
    assert task_state.goal_relation == "clarify"
    assert task_state.current_user_intent == "unknown"
    assert task_state.next_action == "clarify"
    assert task_state.needs_clarification is True
    assert route.intent == RouteIntent.UNKNOWN
    assert route.needs_clarification is True
    assert route.safe_to_execute is False

    logs = (tmp_path / "task-state-timeout-tictactoe-menu.jsonl").read_text(encoding="utf-8")
    assert "task_state_fallback" in logs
    assert '"goal_relation": "clarify"' in logs
    assert '"chosen_intent": "unknown"' in logs


@pytest.mark.parametrize("path", ["snake.js", "tic_tac_toe.py"])
def test_task_state_timeout_fallback_treats_generic_feature_extension_as_modify(tmp_path, path: str):
    updater = TaskStateUpdater(ScriptedLLM())
    session = SessionState(
        task="füge noch einen modus hinzu",
        workspace_root=str(tmp_path),
        follow_up_context=FollowUpContext(
            previous_task=f"build {path}",
            previous_root_goal=f"Build the first playable version in {path}.",
            previous_active_goal=f"Create the initial implementation in {path}.",
            previous_next_action="create",
            previous_requested_outcome="Create the initial playable version.",
            target_paths=[path],
            changed_files=[path],
            read_files=[path],
        ),
        candidate_files=[path],
    )

    task_state = updater.update_task_state(
        session.task,
        snapshot=build_snapshot(tmp_path),
        session=session,
    )
    route = ExecutionDecisionPolicy().build_route(
        task_state,
        snapshot=build_snapshot(tmp_path),
        session=session,
    )

    assert task_state.goal_relation in {"continue", "refine"}
    assert task_state.next_action == "modify"
    assert [artifact.path for artifact in task_state.target_artifacts[:1]] == [path]
    assert route.intent == RouteIntent.UPDATE
    assert route.entities.target_paths == [path]


def test_task_state_timeout_fallback_treats_hardening_follow_up_as_targeted_update(tmp_path):
    logger = AgentLogger(tmp_path, "task-state-timeout-hardening")
    updater = TaskStateUpdater(ScriptedLLM(), logger=logger)
    session = SessionState(
        task="mach login",
        workspace_root=str(tmp_path),
        follow_up_context=FollowUpContext(
            previous_task="mach login",
            previous_root_goal="Implement login for this app.",
            previous_active_goal="Create a working login flow.",
            previous_next_action="modify",
            previous_requested_outcome="A working login flow.",
            target_paths=["app/auth.py"],
            changed_files=["app/auth.py"],
            read_files=["app/auth.py"],
        ),
    )

    task_state = updater.update_task_state(
        "mach es sicherer",
        snapshot=build_snapshot(tmp_path),
        session=session,
    )
    route = ExecutionDecisionPolicy(logger=logger).build_route(
        task_state,
        snapshot=build_snapshot(tmp_path),
        session=session,
    )

    assert task_state.root_goal == "Implement login for this app."
    assert task_state.goal_relation == "refine"
    assert task_state.current_user_intent == "implement"
    assert task_state.execution_strategy == "feature_implementation"
    assert task_state.next_action == "modify"
    assert task_state.needs_clarification is False
    assert "hardening" not in task_state.output_expectation.lower()
    assert "hardening" not in (task_state.verification_target or "").lower()
    assert route.intent == RouteIntent.UPDATE
    assert route.action_plan[0].action == RouteActionName.READ_RELEVANT_FILES

    logs = (tmp_path / "task-state-timeout-hardening.jsonl").read_text(encoding="utf-8")
    assert "task_state_fallback" in logs
    assert '"execution_strategy": "feature_implementation"' in logs


def test_task_state_timeout_fallback_treats_backend_only_follow_up_as_scope_change(tmp_path):
    logger = AgentLogger(tmp_path, "task-state-timeout-backend-only")
    updater = TaskStateUpdater(ScriptedLLM(), logger=logger)
    session = SessionState(
        task="bau login",
        workspace_root=str(tmp_path),
        follow_up_context=FollowUpContext(
            previous_task="bau login",
            previous_root_goal="Add login to this app.",
            previous_active_goal="Add login across frontend and backend.",
            previous_next_action="modify",
            previous_requested_outcome="A working full-stack login flow.",
            target_paths=["frontend/login.html", "backend/auth.py"],
            changed_files=["frontend/login.html", "backend/auth.py"],
            read_files=["frontend/login.html", "backend/auth.py"],
        ),
    )

    task_state = updater.update_task_state(
        "nein, nur im backend",
        snapshot=build_snapshot(tmp_path),
        session=session,
    )
    route = ExecutionDecisionPolicy(logger=logger).build_route(
        task_state,
        snapshot=build_snapshot(tmp_path),
        session=session,
    )

    assert task_state.root_goal == "Add login to this app."
    assert task_state.goal_relation == "scope_change"
    assert task_state.current_user_intent == "correct"
    assert task_state.execution_strategy is None
    assert task_state.constraints == ["Backend only."]
    assert [artifact.path for artifact in task_state.target_artifacts] == ["backend/auth.py"]
    assert task_state.needs_clarification is False
    assert "backend-scoped" in task_state.output_expectation.lower()
    assert route.intent == RouteIntent.UPDATE
    assert route.entities.target_paths[0] == "backend/auth.py"
    assert route.action_plan[0].action == RouteActionName.READ_RELEVANT_FILES

    logs = (tmp_path / "task-state-timeout-backend-only.jsonl").read_text(encoding="utf-8")
    assert "task_state_fallback" in logs
    assert '"goal_relation": "scope_change"' in logs


def test_task_state_update_prompt_contains_prior_state_and_existing_evidence(tmp_path):
    llm = ScriptedLLM(
        [
            {
                "latest_user_turn": "das ist immer noch kaputt",
                "root_goal": "Fix the backend auth flow.",
                "active_goal": "Diagnose the backend auth failure.",
                "goal_relation": "report_problem",
                "output_expectation": "A diagnosis of the current auth failure.",
                "open_problem": "401s after the auth refactor",
                "verification_target": "Re-run the auth validation path.",
                "target_artifacts": [
                    {
                        "path": "app/auth.py",
                        "name": "app/auth.py",
                        "kind": "file",
                        "role": "primary_target",
                        "confidence": 0.84,
                    }
                ],
                "evidence": [
                    {
                        "kind": "diagnostic",
                        "summary": "AssertionError in auth test",
                        "source": "pytest",
                        "artifact_path": "app/auth.py",
                        "confidence": 0.9,
                    }
                ],
                "relevant_context": ["Follow-up after an auth refactor."],
                "constraints": ["Backend only."],
                "assumptions": ["The auth module is still the active artifact."],
                "missing_info": [],
                "ambiguity_level": "low",
                "risk_level": "medium",
                "confidence": 0.88,
                "next_action": "debug",
                "execution_outline": ["Inspect auth.py", "Reproduce auth failure", "Fix and verify"],
                "needs_clarification": False,
                "clarification_questions": [],
            }
        ]
    )
    updater = TaskStateUpdater(llm)
    session = SessionState(
        task="das ist immer noch kaputt",
        workspace_root=str(tmp_path),
        task_state=TaskState(
            latest_user_turn="mach es sicherer",
            root_goal="Fix the backend auth flow.",
            active_goal="Harden backend auth safely.",
            goal_relation="continue",
            output_expectation="A safer backend auth implementation.",
            open_problem="401s after the auth refactor",
            verification_target="Re-run the auth validation path.",
            target_artifacts=[
                TaskArtifact(path="app/auth.py", name="app/auth.py", kind="file", role="primary_target", confidence=0.8)
            ],
            evidence=[],
            relevant_context=[],
            constraints=["Backend only."],
            assumptions=[],
            missing_info=[],
            ambiguity_level="low",
            risk_level="medium",
            confidence=0.8,
            next_action="modify",
            execution_outline=["Inspect auth.py", "Update auth.py", "Verify auth tests"],
            needs_clarification=False,
            clarification_questions=[],
        ),
        follow_up_context=FollowUpContext(
            previous_task="mach es sicherer",
            previous_root_goal="Fix the backend auth flow.",
            previous_active_goal="Harden backend auth safely.",
            previous_next_action="modify",
            target_paths=["app/auth.py"],
            changed_files=["app/auth.py"],
            last_error="AssertionError: unauthorized request",
        ),
    )

    updater.update_task_state("das ist immer noch kaputt", snapshot=build_snapshot(tmp_path), session=session)
    prompt = llm.generate_json_calls[0]["args"][0]

    assert "Fix the backend auth flow." in prompt
    assert "app/auth.py" in prompt
    assert "AssertionError: unauthorized request" in prompt
    assert "Backend only." in prompt


def test_task_state_update_prompt_mentions_phase_two_strategy_fields(tmp_path):
    prompt = task_state_update_prompt("fix the auth flow", snapshot=build_snapshot(tmp_path))

    assert "execution_strategy" in prompt
    assert "next_best_action" in prompt
    assert "inspect current state and active artifacts" in prompt
    assert "Be terse" in prompt
    assert "Omit optional keys" in prompt


def test_task_state_updater_uses_compact_generation_for_fresh_session(tmp_path):
    llm = ScriptedLLM(
        [
            {
                "latest_user_turn": "add a theme toggle",
                "root_goal": "Add a theme toggle to the landing page.",
                "active_goal": "Add a theme toggle to the landing page.",
                "output_expectation": "A working theme toggle for the landing page.",
                "target_artifacts": [
                    {
                        "path": "index.html",
                        "name": "index.html",
                        "kind": "file",
                        "role": "primary_target",
                        "confidence": 0.8,
                    }
                ],
                "confidence": 0.82,
                "next_action": "modify",
            }
        ]
    )
    updater = TaskStateUpdater(llm)
    session = SessionState(
        task="add a theme toggle",
        workspace_root=str(tmp_path),
        messages=[ChatMessage(role="user", content="add a theme toggle")],
    )

    updater.update_task_state("add a theme toggle", snapshot=build_snapshot(tmp_path), session=session)

    call = llm.generate_json_calls[0]
    prompt = call["args"][0]
    assert "Recent conversation:" not in prompt
    assert call["kwargs"]["timeout"] == 18
    assert call["kwargs"]["total_timeout"] == 72
    assert call["kwargs"]["num_ctx"] == 2048


def test_task_state_contract_handles_backend_correction():
    llm = ScriptedLLM(
        [
            {
                "latest_user_turn": "nein, ich meinte das backend",
                "root_goal": "Add auth to this app.",
                "active_goal": "Move the auth change to the backend implementation instead of the UI.",
                "goal_relation": "correct",
                "output_expectation": "A backend-scoped auth change.",
                "open_problem": "The last change targeted the wrong surface.",
                "verification_target": "Only backend auth files should be updated and verified.",
                "target_artifacts": [
                    {
                        "path": "app/api/auth.py",
                        "name": "app/api/auth.py",
                        "kind": "file",
                        "role": "primary_target",
                        "confidence": 0.9,
                    }
                ],
                "evidence": [],
                "relevant_context": ["Previous work touched the UI, but the user corrected the target."],
                "constraints": ["Backend only."],
                "assumptions": [],
                "missing_info": [],
                "ambiguity_level": "low",
                "risk_level": "medium",
                "confidence": 0.9,
                "next_action": "modify",
                "execution_outline": ["Inspect backend auth implementation", "Move the change to backend", "Verify backend path"],
                "needs_clarification": False,
                "clarification_questions": [],
            }
        ]
    )

    state = TaskStateUpdater(llm).update_task_state("nein, ich meinte das backend")

    assert state.goal_relation == "correct"
    assert state.target_artifacts[0].path == "app/api/auth.py"
    assert state.constraints == ["Backend only."]


def test_execution_policy_uses_task_state_strategy_for_refactor_ordering(tmp_path):
    task_state = TaskState(
        latest_user_turn="make auth.py cleaner",
        root_goal="Improve maintainability of the auth module.",
        active_goal="Refactor the auth module to reduce duplication while preserving behavior.",
        goal_relation="refine",
        output_expectation="A behavior-preserving auth refactor.",
        current_user_intent="refactor",
        execution_strategy="refactor",
        open_problem=None,
        verification_target="Re-run the auth tests after the refactor.",
        target_artifacts=[
            TaskArtifact(
                path="app/auth.py",
                name="app/auth.py",
                kind="file",
                role="primary_target",
                confidence=0.86,
            )
        ],
        evidence=[],
        supplied_evidence=[],
        relevant_context=[],
        constraints=[],
        assumptions=[],
        missing_info=[],
        ambiguity_level="low",
        risk_level="medium",
        confidence=0.84,
        next_action="modify",
        next_best_action="modify",
        execution_outline=["Inspect auth.py", "Refactor incrementally", "Re-run auth tests"],
        needs_clarification=False,
        clarification_questions=[],
    )

    route = ExecutionDecisionPolicy().build_route(
        task_state,
        snapshot=build_snapshot(tmp_path),
        session=SessionState(task=task_state.latest_user_turn, workspace_root=str(tmp_path)),
    )

    assert route.intent == RouteIntent.UPDATE
    assert route.action_plan[0].action == RouteActionName.READ_RELEVANT_FILES
    assert "behavior" in route.requested_outcome.lower()
    assert any("preserving behavior" in step.reason.lower() for step in route.action_plan)


def test_execution_policy_uses_task_state_strategy_for_scope_correction(tmp_path):
    task_state = TaskState(
        latest_user_turn="no, I meant the backend only",
        root_goal="Add auth to this app.",
        active_goal="Move the auth change to the backend implementation only.",
        goal_relation="correct",
        output_expectation="A backend-scoped auth change.",
        current_user_intent="correct",
        execution_strategy="rollback_correction",
        open_problem="The last change targeted the wrong surface.",
        verification_target="Only backend auth files should be updated and verified.",
        target_artifacts=[
            TaskArtifact(
                path="app/api/auth.py",
                name="app/api/auth.py",
                kind="file",
                role="primary_target",
                confidence=0.9,
            )
        ],
        evidence=[],
        supplied_evidence=[],
        relevant_context=[],
        constraints=["Backend only."],
        assumptions=[],
        missing_info=[],
        ambiguity_level="low",
        risk_level="medium",
        confidence=0.9,
        next_action="modify",
        next_best_action="modify",
        execution_outline=["Inspect backend auth implementation", "Move the change to backend", "Verify backend path"],
        needs_clarification=False,
        clarification_questions=[],
    )

    route = ExecutionDecisionPolicy().build_route(
        task_state,
        snapshot=build_snapshot(tmp_path),
        session=SessionState(task=task_state.latest_user_turn, workspace_root=str(tmp_path)),
    )

    assert route.intent == RouteIntent.UPDATE
    assert route.action_plan[0].action == RouteActionName.READ_RELEVANT_FILES
    assert "scope" in route.requested_outcome.lower() or "prior change" in route.requested_outcome.lower()
    assert any("necessary part" in step.reason.lower() for step in route.action_plan)


def test_planner_uses_task_state_before_any_other_interpretation(tmp_path):
    target = tmp_path / "app" / "auth.py"
    target.parent.mkdir()
    target.write_text("print('auth')\n", encoding="utf-8")

    llm = ScriptedLLM()
    planner = Planner(llm, "")
    session = SessionState(
        task="I want the login to be safer",
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
        task_state=TaskState(
            latest_user_turn="I want the login to be safer",
            root_goal="Make the login safer.",
            active_goal="Strengthen the login flow security without changing its main user journey.",
            goal_relation="new_task",
            output_expectation="A safer login implementation with verification.",
            target_artifacts=[
                TaskArtifact(
                    path="app/auth.py",
                    name="app/auth.py",
                    kind="file",
                    role="primary_target",
                    confidence=0.82,
                )
            ],
            evidence=[
                EvidenceItem(
                    kind="file",
                    summary="The auth module is the most relevant backend artifact.",
                    artifact_path="app/auth.py",
                    confidence=0.7,
                )
            ],
            relevant_context=["Auth module exists."],
            constraints=[],
            assumptions=["app/auth.py is the correct backend target."],
            missing_info=[],
            ambiguity_level="low",
            risk_level="medium",
            confidence=0.8,
            next_action="modify",
            execution_outline=["Inspect auth.py", "Update auth.py", "Verify the change"],
            needs_clarification=False,
            clarification_questions=[],
        ),
    )

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.CALL_TOOL
    assert decision.tool_name == "read_file"
    assert decision.tool_args["path"] == "app/auth.py"
    assert llm.generate_json_calls == []


def test_core_initialization_uses_task_state_as_only_semantic_source_in_main_path(tmp_path):
    prompt = "I want the login to be safer"
    llm = ScriptedLLM(
        [
            {
                "latest_user_turn": prompt,
                "root_goal": "Make the login safer.",
                "active_goal": "Strengthen the login flow security without changing its main user journey.",
                "goal_relation": "new_task",
                "output_expectation": "A safer login implementation with verification.",
                "open_problem": None,
                "verification_target": "Inspect auth.py and verify the resulting login behavior.",
                "target_artifacts": [
                    {
                        "path": "app/auth.py",
                        "name": "app/auth.py",
                        "kind": "file",
                        "role": "primary_target",
                        "confidence": 0.82,
                    }
                ],
                "evidence": [
                    {
                        "kind": "file",
                        "summary": "The auth module is the most relevant backend artifact.",
                        "artifact_path": "app/auth.py",
                        "confidence": 0.7,
                    }
                ],
                "relevant_context": ["Auth module exists."],
                "constraints": [],
                "assumptions": ["app/auth.py is the correct backend target."],
                "missing_info": [],
                "ambiguity_level": "low",
                "risk_level": "medium",
                "confidence": 0.83,
                "next_action": "modify",
                "execution_outline": ["Inspect auth.py", "Update auth.py", "Verify the change"],
                "needs_clarification": False,
                "clarification_questions": [],
            }
        ]
    )
    planner = Planner(llm, "")
    config = AppConfig(workspace_root=str(tmp_path), state_root_override=str(tmp_path / ".state"))
    core = AgentCore(config)
    session = SessionState(
        task=prompt,
        workspace_root=str(tmp_path),
        workspace_snapshot=build_snapshot(tmp_path),
    )

    core._initialize_session(prompt, session, planner)

    assert session.task_state is not None
    assert session.router_result is not None
    assert session.router_result.intent == RouteIntent.UPDATE
    assert session.router_result.entities.target_paths == ["app/auth.py"]
    assert len(llm.generate_json_calls) == 1
