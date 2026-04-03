from __future__ import annotations

from agent.core import AgentCore
from agent.layered_memory import AgentMemoryStore
from agent.memory import RepoMemoryStore
from agent.models import (
    FileChangeRecord,
    RepairAttemptRecord,
    RepairBrief,
    SessionState,
    ToolCallRecord,
    ToolExecutionMeta,
    ToolRunResult,
    ValidationCommand,
    ValidationFailureEvidence,
    ValidationRunRecord,
)
from agent.task_state import TaskState
from config.settings import AppConfig
from llm.schemas import AgentActionType, AgentDecision, RouteActionName, RouteIntent, RouterOutput


def test_core_marks_unvalidated_changed_files_as_partial(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="Implementiere etwas",
        workspace_root=str(tmp_path),
        validation_status="not_run",
    )
    session.changed_files.append(FileChangeRecord(path="game.py", operation="create"))

    status = core._resolve_final_status(session, final_action=True)
    stop_reason = core._derive_stop_reason(session)

    assert status == "partial"
    assert stop_reason == "validation_missing"


def test_core_uses_a2_memory_profile_by_default(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)

    assert isinstance(core._build_memory_store(), AgentMemoryStore)


def test_core_can_switch_back_to_a1_memory_profile(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="Inspect the repo",
        workspace_root=str(tmp_path),
        runtime_options={"agent_profile": "a1"},
    )

    assert isinstance(core._build_memory_store(session), RepoMemoryStore)


def test_core_validation_decision_preserves_expected_stdout_contract(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)

    decision = core._validation_decision(
        "Run the CLI contract.",
        ValidationCommand(
            command="python normalize_cli.py --keep-case hello world",
            kind="test",
            verification_scope="runtime",
            expected_stdout="hello world",
        ),
    )

    assert decision.tool_args["command"] == "python normalize_cli.py --keep-case hello world"
    assert decision.tool_args["expected_stdout"] == "hello world"


def test_core_runtime_options_preserve_agent_profile_overrides(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="Inspect the repo",
        workspace_root=str(tmp_path),
        runtime_options={"agent_profile": "a1", "execution_profile": "fast"},
    )

    options = core._runtime_options(session)

    assert options["agent_profile"] == "a1"
    assert options["execution_profile"] == "fast"


def test_core_requires_semantic_review_before_marking_changed_run_complete(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="Implement the requested Python fix",
        workspace_root=str(tmp_path),
        validation_status="passed",
    )
    session.changed_files.append(FileChangeRecord(path="app/main.py", operation="modify"))
    session.validation_runs.append(
        ValidationRunRecord(
            command="python -m pytest",
            verification_scope="runtime",
            status="passed",
        )
    )

    status = core._resolve_final_status(session, final_action=True)
    stop_reason = core._derive_stop_reason(session)

    assert status == "partial"
    assert stop_reason == "requirements_review_missing"


def test_core_marks_generation_failure_without_changes_as_partial(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="Erweitere tic_tac_toe.py um ein Menü",
        workspace_root=str(tmp_path),
        validation_status="not_run",
        stop_reason="generation_failed",
    )
    session.blockers.append("No reliable update content could be generated for tic_tac_toe.py.")

    status = core._resolve_final_status(session, final_action=True)

    assert status == "partial"


def test_core_allows_progressive_repair_write_after_limit_when_behavior_improved(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    config.max_repair_attempts = 3
    core = AgentCore(config)
    session = SessionState(
        task="Repair normalize_cli.py",
        workspace_root=str(tmp_path),
        validation_status="failed",
        repair_attempts=3,
        active_repair_context=ValidationFailureEvidence(
            command="python -m unittest tests.test_normalize",
            verification_scope="runtime",
            status="failed",
            artifact_paths=["normalize_cli.py"],
            summary="CLI stdout contract still fails.",
            failure_summary="Expected Hello WORLD.",
            evidence_signature="sig-2",
            repair_brief=RepairBrief(
                failure_signature="runtime:assertion_mismatch:new",
                primary_target="normalize_cli.py",
                locked_target="normalize_cli.py",
                allowed_files=["normalize_cli.py"],
            ),
        ),
    )
    session.repair_history.append(
        RepairAttemptRecord(
            artifact_path="normalize_cli.py",
            validation_command="python -m unittest tests.test_normalize",
            verification_scope="runtime",
            strategy="targeted_repair",
            result="mutation_planned",
            reason="stdout path repaired",
            evidence_signature="sig-1",
            failure_signature="runtime:assertion_mismatch:old",
            independent_verification=False,
            behavior_changed=True,
        )
    )
    decision = AgentDecision(
        thought_summary="Repair normalize_cli.py from the latest failed runtime evidence.",
        action_type=AgentActionType.CALL_TOOL,
        tool_name="write_file",
        tool_args={"path": "normalize_cli.py", "content": "print('Hello WORLD')\n"},
        expected_outcome="Apply the targeted repair.",
    )

    enforced = core._enforce_iteration_rules(session, decision)

    assert enforced == decision
    assert session.stop_reason is None
    assert session.validation_status == "failed"


def test_core_stops_after_repair_limit_when_latest_failed_repair_did_not_change_behavior(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    config.max_repair_attempts = 3
    core = AgentCore(config)
    session = SessionState(
        task="Repair normalize_cli.py",
        workspace_root=str(tmp_path),
        validation_status="failed",
        repair_attempts=3,
        active_repair_context=ValidationFailureEvidence(
            command="python -m unittest tests.test_normalize",
            verification_scope="runtime",
            status="failed",
            artifact_paths=["normalize_cli.py"],
            summary="CLI stdout contract still fails.",
            failure_summary="Expected Hello WORLD.",
            evidence_signature="sig-3",
            repair_brief=RepairBrief(
                failure_signature="runtime:assertion_mismatch:same",
                primary_target="normalize_cli.py",
                locked_target="normalize_cli.py",
                allowed_files=["normalize_cli.py"],
            ),
        ),
    )
    session.repair_history.append(
        RepairAttemptRecord(
            artifact_path="normalize_cli.py",
            validation_command="python -m unittest tests.test_normalize",
            verification_scope="runtime",
            strategy="targeted_repair",
            result="mutation_planned",
            reason="stdout path repaired",
            evidence_signature="sig-2",
            failure_signature="runtime:assertion_mismatch:same",
            independent_verification=False,
            behavior_changed=False,
        )
    )
    decision = AgentDecision(
        thought_summary="Repair normalize_cli.py from the latest failed runtime evidence.",
        action_type=AgentActionType.CALL_TOOL,
        tool_name="write_file",
        tool_args={"path": "normalize_cli.py", "content": "print('Hello WORLD')\n"},
        expected_outcome="Apply the targeted repair.",
    )

    enforced = core._enforce_iteration_rules(session, decision)

    assert enforced.action_type == AgentActionType.FINAL
    assert session.stop_reason == "max_repair_attempts_reached"


def test_core_uses_bootstrap_reset_stop_reason_and_partial_status(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="Repair the bootstrap path",
        workspace_root=str(tmp_path),
        validation_status="bootstrap_reset_required",
    )
    session.changed_files.append(FileChangeRecord(path="scripts/build_duplicates.py", operation="modify"))

    stop_reason = core._derive_stop_reason(session)
    status = core._resolve_final_status(session, final_action=True)

    assert stop_reason == "needs_human_or_bootstrap_reset"
    assert status == "partial"


def test_core_marks_model_start_failure_without_changes_as_partial(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="Erstelle snake.html",
        workspace_root=str(tmp_path),
        validation_status="not_run",
        stop_reason="model_start_failed",
    )
    session.blockers.append(
        "Repeated model start failure for snake.html: qwen3-coder:30b, qwen2.5-coder:14b produced no first chunk, and no safe local recovery path applied."
    )

    status = core._resolve_final_status(session, final_action=True)

    assert status == "partial"


def test_core_uses_tool_metadata_for_phase_detection(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="Mutate a generated artifact",
        workspace_root=str(tmp_path),
    )
    session.tool_calls.append(
        ToolCallRecord(
            iteration=1,
            tool_name="custom_mutator",
            success=True,
            summary="Updated generated artifact.",
            tool_meta=ToolExecutionMeta(
                category="write",
                mutating=True,
                execution_mode="mutating",
            ),
        )
    )

    assert core._determine_phase(session) == "editing"


def test_core_append_note_uses_concise_tool_summary_for_file_update(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="Update normalize_cli.py",
        workspace_root=str(tmp_path),
    )
    result = ToolRunResult(
        tool_name="write_file",
        success=True,
        message="Wrote normalize_cli.py.",
        data={"path": "normalize_cli.py"},
        changed_files=[FileChangeRecord(path="normalize_cli.py", operation="write")],
    )

    core._append_note(session, result)

    assert session.notes == ["Updated normalize_cli.py"]


def test_core_append_note_keeps_failure_reason_for_failed_tool_runs(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="Run tests",
        workspace_root=str(tmp_path),
    )
    result = ToolRunResult(
        tool_name="run_tests",
        success=False,
        message="Validation command exited with 1.",
        data={"command": "python -m unittest tests.test_stats", "exit_code": 1},
    )

    core._append_note(session, result)

    assert session.notes == [
        "Ran python -m unittest tests.test_stats failed: Validation command exited with 1. (exit=1)"
    ]


def test_core_marks_uncertain_degraded_semantic_fallback_as_partial_not_analysis_complete(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="mach da was",
        workspace_root=str(tmp_path),
        validation_status="not_run",
        task_state=TaskState(
            latest_user_turn="mach da was",
            root_goal="mach da was",
            active_goal="mach da was",
            goal_relation="clarify",
            output_expectation="Clarify the active target and decide the safest next step.",
            open_problem=None,
            verification_target=None,
            target_artifacts=[],
            evidence=[],
            relevant_context=[],
            constraints=[],
            assumptions=[],
            missing_info=["The exact target is still ambiguous."],
            ambiguity_level="high",
            risk_level="medium",
            confidence=0.3,
            next_action="clarify",
            execution_outline=["Ask for the concrete target artifact or area."],
            needs_clarification=True,
            clarification_questions=["Welchen konkreten Bereich oder welches Artefakt soll ich als naechstes bearbeiten?"],
            semantic_inference_mode="conservative",
        ),
        router_result=RouterOutput(
            user_goal="mach da was",
            intent=RouteIntent.UNKNOWN,
            requested_outcome="Clarify the user's exact target before doing any work.",
            action_plan=[
                {
                    "step": 1,
                    "action": RouteActionName.ASK_CLARIFICATION,
                    "reason": "The degraded semantic fallback is not reliable enough to proceed.",
                }
            ],
            needs_clarification=True,
            clarification_questions=["Was genau soll ich fuer dich erreichen?"],
            confidence=0.22,
            safe_to_execute=False,
            repo_context_needed=False,
            search_terms=[],
            relevant_extensions=[],
            direct_response=None,
        ),
    )
    session.runtime_executions.append(
        {
            "operation_name": "task_state_generation",
            "task_class": "task_state_generation",
            "final_state": "degraded_success",
            "capability_tier": "tier_d",
            "recovery_strategy": "deterministic_fallback",
            "degraded": True,
            "honest_blocked": False,
            "artifact_bytes_generated": 0,
            "validation_possible": False,
            "summary": "Task understanding used the deterministic fallback after a startup timeout.",
            "attempts": [],
        }
    )

    stop_reason = core._derive_stop_reason(session)
    status = core._resolve_final_status(session, final_action=True)

    assert stop_reason == "clarification_required"
    assert status == "partial"


def test_core_distinguishes_safe_degraded_semantic_completion_from_full_analysis_complete(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="erklaer mir kurz die auth logik",
        workspace_root=str(tmp_path),
        validation_status="not_run",
        task_state=TaskState(
            latest_user_turn="erklaer mir kurz die auth logik",
            root_goal="erklaer mir kurz die auth logik",
            active_goal="erklaer mir kurz die auth logik",
            goal_relation="new_task",
            output_expectation="Explain the relevant behavior or code path clearly and honestly.",
            current_user_intent="explain",
            execution_strategy="validation_inspection",
            open_problem=None,
            verification_target=None,
            target_artifacts=[],
            evidence=[],
            relevant_context=[],
            constraints=[],
            assumptions=[],
            missing_info=[],
            ambiguity_level="low",
            risk_level="low",
            confidence=0.78,
            next_action="explain",
            execution_outline=["Explain the result in clear user-facing language."],
            needs_clarification=False,
            clarification_questions=[],
            semantic_inference_mode="conservative",
        ),
        router_result=RouterOutput(
            user_goal="erklaer mir kurz die auth logik",
            intent=RouteIntent.EXPLAIN,
            requested_outcome="Explain the relevant code or concept.",
            action_plan=[
                {
                    "step": 1,
                    "action": RouteActionName.RESPOND_DIRECTLY,
                    "reason": "The interpreted request can be answered directly.",
                }
            ],
            needs_clarification=False,
            clarification_questions=[],
            confidence=0.78,
            safe_to_execute=True,
            repo_context_needed=False,
            search_terms=[],
            relevant_extensions=[],
            direct_response="Kurz erklaert.",
        ),
    )
    session.runtime_executions.append(
        {
            "operation_name": "task_state_generation",
            "task_class": "task_state_generation",
            "final_state": "degraded_success",
            "capability_tier": "tier_d",
            "recovery_strategy": "deterministic_fallback",
            "degraded": True,
            "honest_blocked": False,
            "artifact_bytes_generated": 0,
            "validation_possible": False,
            "summary": "Task understanding used the deterministic fallback after a startup timeout.",
            "attempts": [],
        }
    )

    stop_reason = core._derive_stop_reason(session)

    assert stop_reason == "minimal_semantic_complete"


def test_core_distinguishes_reduced_semantic_completion_from_full_analysis_complete(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="erklaer mir kurz die auth logik",
        workspace_root=str(tmp_path),
        validation_status="not_run",
        task_state=TaskState(
            latest_user_turn="erklaer mir kurz die auth logik",
            root_goal="erklaer mir kurz die auth logik",
            active_goal="erklaer mir kurz die auth logik",
            goal_relation="new_task",
            output_expectation="Explain the relevant behavior or code path clearly and honestly.",
            current_user_intent="explain",
            execution_strategy="validation_inspection",
            open_problem=None,
            verification_target=None,
            target_artifacts=[],
            evidence=[],
            relevant_context=[],
            constraints=[],
            assumptions=[],
            missing_info=[],
            ambiguity_level="low",
            risk_level="low",
            confidence=0.82,
            next_action="explain",
            execution_outline=["Explain the result in clear user-facing language."],
            needs_clarification=False,
            clarification_questions=[],
            semantic_resolution="reserve_model",
        ),
        router_result=RouterOutput(
            user_goal="erklaer mir kurz die auth logik",
            intent=RouteIntent.EXPLAIN,
            requested_outcome="Explain the relevant code or concept.",
            action_plan=[
                {
                    "step": 1,
                    "action": RouteActionName.RESPOND_DIRECTLY,
                    "reason": "The interpreted request can be answered directly.",
                }
            ],
            needs_clarification=False,
            clarification_questions=[],
            confidence=0.82,
            safe_to_execute=True,
            repo_context_needed=False,
            search_terms=[],
            relevant_extensions=[],
            direct_response="Kurz erklaert.",
        ),
    )
    session.runtime_executions.append(
        {
            "operation_name": "task_state_generation",
            "task_class": "task_state_generation",
            "final_state": "completed",
            "capability_tier": "tier_b",
            "recovery_strategy": "switch_to_faster_model",
            "degraded": True,
            "honest_blocked": False,
            "artifact_bytes_generated": 0,
            "validation_possible": False,
            "summary": "Task understanding recovered on a reserve semantic model.",
            "attempts": [],
            "semantic_resolution": "reserve_model",
        }
    )

    stop_reason = core._derive_stop_reason(session)

    assert stop_reason == "reduced_semantic_complete"


def test_core_marks_debug_follow_up_without_runtime_reproduction_as_partial(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="fix tic tac toe bug",
        workspace_root=str(tmp_path),
        validation_status="passed",
        task_state=TaskState(
            latest_user_turn="fix tic tac toe bug",
            root_goal="Build a Tic Tac Toe game in Python.",
            active_goal="Diagnose and fix the broken input handling.",
            goal_relation="report_problem",
            output_expectation="Diagnose the bug, apply the smallest safe fix, and rerun the interaction path.",
            open_problem="Moves are always rejected.",
            verification_target="Reproduce the interactive failure and rerun it after the fix.",
            target_artifacts=[],
            evidence=[],
            relevant_context=[],
            constraints=[],
            assumptions=[],
            missing_info=[],
            ambiguity_level="low",
            risk_level="medium",
            confidence=0.82,
            current_user_intent="repair",
            execution_strategy="debug_repair",
            next_action="debug",
            execution_outline=["Read the active script", "Reproduce the issue", "Fix it and rerun the path"],
            needs_clarification=False,
            clarification_questions=[],
        ),
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command='internal:python_syntax:["tic_tac_toe.py"]',
            verification_scope="syntax",
            status="passed",
        )
    )

    status = core._resolve_final_status(session, final_action=True)
    stop_reason = core._derive_stop_reason(session)

    assert status == "partial"
    assert stop_reason == "reproduction_missing"


def test_core_marks_debug_fix_with_only_syntax_validation_as_partial(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="fix tic tac toe bug",
        workspace_root=str(tmp_path),
        validation_status="passed",
        task_state=TaskState(
            latest_user_turn="fix tic tac toe bug",
            root_goal="Build a Tic Tac Toe game in Python.",
            active_goal="Diagnose and fix the broken input handling.",
            goal_relation="report_problem",
            output_expectation="Diagnose the bug, apply the smallest safe fix, and rerun the interaction path.",
            open_problem="Moves are always rejected.",
            verification_target="Reproduce the interactive failure and rerun it after the fix.",
            target_artifacts=[],
            evidence=[],
            relevant_context=[],
            constraints=[],
            assumptions=[],
            missing_info=[],
            ambiguity_level="low",
            risk_level="medium",
            confidence=0.82,
            current_user_intent="repair",
            execution_strategy="debug_repair",
            next_action="debug",
            execution_outline=["Read the active script", "Reproduce the issue", "Fix it and rerun the path"],
            needs_clarification=False,
            clarification_questions=[],
        ),
    )
    session.changed_files.append(FileChangeRecord(path="tic_tac_toe.py", operation="modify"))
    session.validation_runs.append(
        ValidationRunRecord(
            command='internal:python_syntax:["tic_tac_toe.py"]',
            verification_scope="syntax",
            status="passed",
        )
    )

    status = core._resolve_final_status(session, final_action=True)
    stop_reason = core._derive_stop_reason(session)

    assert status == "partial"
    assert stop_reason == "functional_validation_missing"


def test_core_marks_small_web_artifact_with_only_structural_checks_as_partial(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="Ergaenze snake.html um Menü und Highscore",
        workspace_root=str(tmp_path),
        validation_status="passed",
    )
    session.changed_files.append(FileChangeRecord(path="snake.html", operation="modify"))
    session.validation_runs.append(
        ValidationRunRecord(
            command='internal:web_artifact:[{"path":"snake.html","expected_features":["menu","highscore"]}]',
            verification_scope="structural",
            status="passed",
        )
    )

    status = core._resolve_final_status(session, final_action=True)
    stop_reason = core._derive_stop_reason(session)

    assert status == "partial"
    assert stop_reason == "functional_validation_missing"


def test_core_accepts_small_web_artifact_after_structural_and_semantic_signoff(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="Erstelle ein kleines spielbares Snake-Spiel als HTML-Datei mit Tastatursteuerung, Punktestand und Game-Over-Neustart.",
        workspace_root=str(tmp_path),
        validation_status="passed",
    )
    session.changed_files.append(FileChangeRecord(path="index.html", operation="create"))
    session.validation_runs.append(
        ValidationRunRecord(
            command='internal:web_artifact:[{"path":"index.html","expected_features":["score","keyboard_controls","game_over","start_controls"]}]',
            verification_scope="structural",
            status="passed",
        )
    )
    session.validation_runs.append(
        ValidationRunRecord(
            command='internal:semantic_review:[{"path":"index.html"}]',
            verification_scope="semantic",
            status="passed",
        )
    )

    status = core._resolve_final_status(session, final_action=True)
    stop_reason = core._derive_stop_reason(session)

    assert status == "completed"
    assert stop_reason == "validated"


def test_core_does_not_offer_identical_failed_validation_again_without_progress(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="Implementiere etwas",
        workspace_root=str(tmp_path),
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
            iteration=3,
        )
    )

    assert core._pick_validation_command(session) is None
