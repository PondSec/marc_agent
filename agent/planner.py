from __future__ import annotations

import ast
from dataclasses import dataclass, field
import difflib
import hashlib
import io
import json
from pathlib import Path
import re
import time
import tomllib
import tokenize

from pydantic import ValidationError

from agent.decision import ExecutionDecisionPolicy
from agent.models import (
    DiagnosticRecord,
    ProposedUpdateReview,
    RepairAttemptRecord,
    SemanticChangeReview,
    SessionState,
    ValidationCommand,
    ValidationFailureEvidence,
    ValidationRunRecord,
    WorkspaceSnapshot,
)
from agent.prompts import (
    REPAIR_BLOCKED_SENTINEL,
    _artifact_scoped_focus,
    _direct_main_runtime_contract,
    _direct_python_script_runtime_contract,
    _line_focused_excerpt,
    _repair_semantic_value_text,
    _repair_target_line_hints,
    _repair_semantic_delta_lines,
    choose_path_prompt,
    final_response_prompt,
    generate_content_continuation_prompt,
    generate_content_prompt,
    generate_content_retry_prompt,
    proposed_update_review_prompt,
    proposed_update_review_system_prompt,
    semantic_change_review_prompt,
    semantic_change_review_system_prompt,
)
from agent.router import IntentRouter
from agent.semantic_runtime import availability_recovery_model
from agent.state_updater import TaskStateUpdater
from agent.task_state import TaskState
from agent.verification import ValidationPlanner
from llm.ollama_client import OllamaGenerationError
from llm.provider import LLMProvider
from llm.runtime_resilience import (
    ExecutionAttemptRecord,
    ExecutionFailure,
    ExecutionRecoveryPolicy,
    build_execution_run_record,
    classify_execution_failure,
    estimate_context_pressure,
    invoke_model,
)
from llm.schemas import (
    AgentActionType,
    AgentDecision,
    PlanningResponse,
    RouteActionName,
    RouteActionStep,
    RouteIntent,
    RouterOutput,
)
from runtime.logger import AgentLogger


WRITE_INTENTS = {RouteIntent.CREATE, RouteIntent.UPDATE, RouteIntent.DEBUG, RouteIntent.DELETE}
WRITE_TOOL_NAMES = {
    "write_file",
    "append_file",
    "create_file",
    "delete_file",
    "replace_in_file",
    "patch_file",
}
TARGETED_REPAIR_STRATEGY = "validation_targeted"
ESCALATED_REPAIR_STRATEGY = "validation_escalated"
LIGHTWEIGHT_UPDATE_SUFFIXES = {
    ".c",
    ".cc",
    ".cfg",
    ".conf",
    ".cpp",
    ".css",
    ".cxx",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".less",
    ".lua",
    ".md",
    ".mjs",
    ".php",
    ".py",
    ".pyi",
    ".rb",
    ".rs",
    ".rst",
    ".scss",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
MAX_LIGHTWEIGHT_UPDATE_CHARS = 8_000
MAX_LIGHTWEIGHT_UPDATE_LINES = 260
MAX_LIGHTWEIGHT_UPDATE_TARGETS = 6
MAX_LIGHTWEIGHT_UPDATE_WORKSPACE_FILES = 60
MAX_LIGHTWEIGHT_UPDATE_CHANGED_FILES = 6
MIN_LIGHTWEIGHT_UPDATE_CONFIDENCE = 0.72
MIN_COMPACT_PRIMARY_UPDATE_CONFIDENCE = 0.55
DEFERRED_UPDATE_TARGET_NOTE_PREFIX = "deferred_update_target:"
WEB_CONTRACT_HTML_SUFFIXES = {".html", ".htm"}
WEB_CONTRACT_CSS_SUFFIXES = {".css", ".less", ".scss"}
WEB_CONTRACT_SCRIPT_SUFFIXES = {".cjs", ".js", ".jsx", ".mjs", ".ts", ".tsx"}
WEB_CONTRACT_SUFFIXES = (
    WEB_CONTRACT_HTML_SUFFIXES
    | WEB_CONTRACT_CSS_SUFFIXES
    | WEB_CONTRACT_SCRIPT_SUFFIXES
)


@dataclass(slots=True)
class GenerationRecoveryAttempt:
    strategy: str
    prompt_kind: str
    model_name: str | None = None
    capability_tier: str = "tier_a"


@dataclass(slots=True)
class GenerationRetryResult:
    content: str | None = None
    attempts: list[ExecutionAttemptRecord] = field(default_factory=list)
    capability_tier: str | None = None
    recovery_strategy: str | None = None


@dataclass(slots=True)
class UpdateReviewRetryResult:
    content: str | None = None
    attempts: list[ExecutionAttemptRecord] = field(default_factory=list)
    review: ProposedUpdateReview | None = None
    capability_tier: str | None = None
    recovery_strategy: str | None = None
    effective_repair_strategy: str | None = None


@dataclass(slots=True)
class ContentGenerationFailure:
    stop_reason: str
    failure_class: str
    blocker_message: str
    user_message: str
    attempts: list[ExecutionAttemptRecord] = field(default_factory=list)


@dataclass(slots=True)
class ContentGenerationResult:
    content: str | None = None
    source: str = "model"
    failure: ContentGenerationFailure | None = None
    repair_strategy_used: str | None = None


@dataclass(slots=True)
class DeterministicUpdateRecovery:
    content: str
    review: ProposedUpdateReview
    recovery_strategy: str
    capability_tier: str = "tier_d"


@dataclass(slots=True)
class MutationAssessment:
    effective: bool
    reason: str
    before_hash: str
    after_hash: str
    change_labels: list[str] = field(default_factory=list)
    changed_line_count: int = 0


@dataclass(slots=True)
class WebContractInventory:
    html_ids: set[str] = field(default_factory=set)
    html_id_classes: dict[str, set[str]] = field(default_factory=dict)
    html_root_classes: set[str] = field(default_factory=set)
    css_id_selectors: set[str] = field(default_factory=set)
    css_class_selectors: set[str] = field(default_factory=set)
    css_root_state_classes: set[str] = field(default_factory=set)
    js_id_refs: set[str] = field(default_factory=set)
    js_declared_ids: set[str] = field(default_factory=set)
    js_class_tokens: set[str] = field(default_factory=set)
    js_root_state_classes: set[str] = field(default_factory=set)


@dataclass(slots=True)
class WebContractFinding:
    kind: str
    token: str
    source_paths: tuple[str, ...]
    summary: str


@dataclass(slots=True)
class PythonSequencePrefixContract:
    variable_expression: str
    slice_length: int
    literal_tokens: tuple[str, ...]
    lineno: int


class Planner:
    def __init__(
        self,
        llm: LLMProvider,
        tool_manifest: str,
        *,
        logger: AgentLogger | None = None,
    ):
        self.llm = llm
        self.tool_manifest = tool_manifest
        self.logger = logger
        llm_config = getattr(self.llm, "config", None)
        self.router = IntentRouter(
            llm,
            logger=logger,
            model_name=getattr(llm_config, "router_model_name", None),
            timeout=max(int(getattr(llm_config, "router_timeout", self._llm_timeout(12))), 12),
            num_ctx=max(int(getattr(llm_config, "router_num_ctx", 2048)), 512),
            retries=max(int(getattr(llm_config, "router_retries", 1)), 0),
        )
        self.task_state_updater = TaskStateUpdater(
            llm,
            logger=logger,
            model_name=(
                getattr(llm_config, "router_model_name", None)
                or getattr(llm_config, "model_name", None)
            ),
            timeout=max(int(getattr(llm_config, "router_timeout", self._llm_timeout(18))), 18),
            num_ctx=max(
                int(
                    getattr(
                        llm_config,
                        "router_num_ctx",
                        getattr(llm_config, "ollama_num_ctx", self._llm_num_ctx(4096)),
                    )
                ),
                1024,
            ),
        )
        self.decision_policy = ExecutionDecisionPolicy(logger=logger)
        self.validation_planner = ValidationPlanner()

    def update_task_state(
        self,
        task: str,
        snapshot: WorkspaceSnapshot | None,
        session: SessionState | None = None,
    ) -> TaskState:
        return self.task_state_updater.update_task_state(task, snapshot=snapshot, session=session)

    def route_task_state(
        self,
        task_state: TaskState,
        snapshot: WorkspaceSnapshot | None,
        session: SessionState | None = None,
    ) -> RouterOutput:
        return self.decision_policy.build_route(
            task_state,
            snapshot=snapshot,
            session=session,
        )

    def interpret_user_request(
        self,
        task: str,
        snapshot: WorkspaceSnapshot | None,
        session: SessionState,
    ) -> RouterOutput:
        del task
        task_state = self._require_task_state(session)
        return self.route_task_state(
            task_state,
            snapshot,
            session=session,
        )

    def validate_router_output(self, payload: dict[str, object]) -> RouterOutput:
        return self.router.validate_router_output(payload)

    def analyze_task(
        self,
        task: str,
        snapshot: WorkspaceSnapshot | None,
        session: SessionState,
    ) -> RouterOutput:
        return self.interpret_user_request(task, snapshot, session=session)

    def create_plan(
        self,
        task: str,
        session: SessionState,
        route: RouterOutput | None = None,
    ) -> PlanningResponse:
        del task
        task_state = self._require_task_state(session)
        snapshot = session.workspace_snapshot
        resolved_route = route or session.router_result or self.route_task_state(
            task_state,
            snapshot,
            session=session,
        )
        session.router_result = resolved_route
        steps = [item.reason for item in resolved_route.action_plan]
        if resolved_route.safe_to_execute and resolved_route.intent in WRITE_INTENTS:
            if not any(item.action == RouteActionName.RUN_VALIDATION for item in resolved_route.action_plan):
                steps.append("Validate the resulting change with the most relevant project command.")
        completion = [
            "The next step follows committed task state and the validated route, not the raw prompt.",
            "Relevant artifacts are inspected before mutation work.",
            "Missing information triggers targeted clarification instead of blind execution.",
        ]
        if task_state.execution_strategy == "debug_repair":
            completion.append("Diagnosis uses current evidence before any repair is declared complete.")
        if resolved_route.intent in WRITE_INTENTS:
            if task_state.verification_target:
                completion.append(f"Validation checks the committed target: {task_state.verification_target}")
            else:
                completion.append("Any code or file change is validated or clearly reported as blocked.")
        else:
            if task_state.verification_target:
                completion.append(f"Findings are checked against the committed target: {task_state.verification_target}")
            else:
                completion.append("The response explains the result or plan in user-facing language.")
        tests = snapshot.likely_commands[:4] if snapshot is not None else []
        return PlanningResponse(
            summary=resolved_route.requested_outcome,
            steps=steps or [resolved_route.user_goal],
            files_to_inspect=resolved_route.entities.target_paths[:8],
            tests_to_run=tests,
            completion_criteria=completion,
        )

    def decide_next_action(self, task: str, session: SessionState) -> AgentDecision:
        del task
        self._require_task_state(session)
        route = session.router_result or self.interpret_user_request(
            session.task,
            session.workspace_snapshot,
            session=session,
        )
        session.router_result = route
        session.task_analysis = {
            "task_state": (
                session.task_state.model_dump()
                if session.task_state is not None
                else None
            ),
            "task_understanding": (
                session.task_understanding.model_dump()
                if session.task_understanding is not None
                else None
            ),
            "router_result": route.model_dump(),
        }

        if route.needs_clarification:
            return self._final_decision(
                "The router requires clarification before any execution.",
                self._clarification_response(route, session=session),
            )

        if not route.safe_to_execute:
            return self._final_decision(
                "Execution is not safe enough to continue without more user input.",
                self._unsafe_response(route, session=session),
            )

        if route.direct_response and not route.repo_context_needed and not session.tool_calls:
            return self._final_decision(
                "The validated route can be answered directly without repository work.",
                route.direct_response,
            )

        memory_recall_response = self._memory_recall_response(route, session)
        if memory_recall_response is not None and not session.tool_calls and not session.changed_files:
            return self._final_decision(
                "The request is a memory recall query that can be answered from retrieved memory.",
                memory_recall_response,
            )

        if session.changed_files and (
            self._has_pending_explicit_update_targets(route, session)
            or self._has_pending_explicit_create_targets(route, session)
        ):
            decision = self.execute_action_from_plan(route, session)
            self._log(
                "execution_plan_selected",
                intent=route.intent,
                safe_to_execute=route.safe_to_execute,
                action_type=decision.action_type,
                tool_name=decision.tool_name,
                source="pending_explicit_targets",
            )
            return decision

        if session.validation_status in {"failed", "bootstrap_failed", "bootstrap_reset_required"} and session.validation_runs:
            repair_decision = self._repair_after_failed_validation(route, session)
            if repair_decision is not None:
                return repair_decision

        if session.changed_files:
            command = self._pick_validation_command(session)
            if command is not None:
                return self._validation_decision(
                    "Changed files must go through the remaining validation plan.",
                    command.command,
                    expected_stdout=command.expected_stdout,
                )
            if self._requirements_review_missing(session):
                self._run_semantic_change_review(route, session)
                if session.validation_status in {"failed", "bootstrap_failed", "bootstrap_reset_required"}:
                    repair_decision = self._repair_after_failed_validation(route, session)
                    if repair_decision is not None:
                        return repair_decision
            return self._final_decision(
                "The routed mutation is complete and no validation command remains.",
                self._compose_user_response(route, session),
            )

        decision = self.execute_action_from_plan(route, session)
        self._log(
            "execution_plan_selected",
            intent=route.intent,
            safe_to_execute=route.safe_to_execute,
            action_type=decision.action_type,
            tool_name=decision.tool_name,
        )
        return decision

    def execute_action_from_plan(
        self,
        route: RouterOutput,
        session: SessionState,
    ) -> AgentDecision:
        read_paths = set(self._read_paths(session))
        searched_queries = set(self._searched_queries(session))
        inspected = any(item.tool_name == "inspect_workspace" for item in session.tool_calls)
        candidate_paths = self._candidate_paths(route, session)
        read_candidates = self._read_candidates(route, session, candidate_paths)

        for step in route.action_plan:
            if step.action == RouteActionName.ASK_CLARIFICATION:
                return self._final_decision(
                    "The next plan step is clarification.",
                    self._clarification_response(route),
                )

            if step.action == RouteActionName.RESPOND_DIRECTLY:
                return self._final_decision(
                    "The plan resolves with a direct response.",
                    route.direct_response or self._compose_user_response(route, session),
                )

            if step.action == RouteActionName.INSPECT_WORKSPACE and not inspected:
                return AgentDecision(
                    thought_summary=step.reason,
                    action_type=AgentActionType.CALL_TOOL,
                    tool_name="inspect_workspace",
                    tool_args={"focus": route.user_goal},
                    expected_outcome="Collect repository structure and validation context.",
                    final_response=None,
                )

            if step.action == RouteActionName.SEARCH_WORKSPACE:
                query = self._best_search_query(route)
                if query and query not in searched_queries:
                    return AgentDecision(
                        thought_summary=step.reason,
                        action_type=AgentActionType.CALL_TOOL,
                        tool_name="search_in_files",
                        tool_args={"query": query, "path": ".", "max_results": 30},
                        expected_outcome="Locate candidate files that match the routed target.",
                        final_response=None,
                    )

            if step.action == RouteActionName.READ_RELEVANT_FILES:
                candidate = self._next_unread_candidate(read_candidates, read_paths)
                if candidate is not None:
                    return AgentDecision(
                        thought_summary=step.reason,
                        action_type=AgentActionType.CALL_TOOL,
                        tool_name="read_file",
                        tool_args={"path": candidate},
                        expected_outcome="Inspect the most relevant file before proceeding.",
                        final_response=None,
                    )
                query = self._best_search_query(route)
                if not candidate_paths and query and query not in searched_queries:
                    return AgentDecision(
                        thought_summary="A search is needed before concrete files can be read.",
                        action_type=AgentActionType.CALL_TOOL,
                        tool_name="search_in_files",
                        tool_args={"query": query, "path": ".", "max_results": 30},
                        expected_outcome="Find files that likely satisfy the routed goal.",
                        final_response=None,
                    )

            if step.action == RouteActionName.DIAGNOSE_ISSUE:
                diagnosis = self._diagnose_issue_decision(route, session, candidate_paths, read_paths)
                if diagnosis is not None:
                    return diagnosis

            if step.action == RouteActionName.CREATE_ARTIFACT:
                bootstrap = self._next_create_bootstrap(route, session, read_paths)
                if bootstrap is not None:
                    return AgentDecision(
                        thought_summary="Read a nearby manifest or example before creating new code.",
                        action_type=AgentActionType.CALL_TOOL,
                        tool_name="read_file",
                        tool_args={"path": bootstrap},
                        expected_outcome="Match existing conventions before generating a new artifact.",
                        final_response=None,
                    )
                draft = self._draft_create_decision(route, session)
                if draft is not None:
                    return draft

            if step.action == RouteActionName.UPDATE_ARTIFACT:
                target = self._next_update_target(route, session)
                if target is None:
                    query = self._best_search_query(route)
                    if query and query not in searched_queries:
                        return AgentDecision(
                            thought_summary="The update target still needs to be located in the workspace.",
                            action_type=AgentActionType.CALL_TOOL,
                            tool_name="search_in_files",
                            tool_args={"query": query, "path": ".", "max_results": 30},
                            expected_outcome="Find the file that should be updated.",
                            final_response=None,
                        )
                    return self._final_decision(
                        "The update target is still ambiguous.",
                        self._clarification_response(
                            route.model_copy(
                                update={
                                    "needs_clarification": True,
                                    "clarification_questions": [
                                        "Welche Datei oder welcher konkrete Bereich soll aktualisiert werden?"
                                    ],
                                    "safe_to_execute": False,
                                }
                            ),
                            session=session,
                        ),
                    )
                repair_context = self._repair_context_for_target(route, session, target)
                current_content = self._current_file_content(session, target)
                if current_content is not None and target not in read_paths:
                    return AgentDecision(
                        thought_summary="Read the target file before generating an update.",
                        action_type=AgentActionType.CALL_TOOL,
                        tool_name="read_file",
                        tool_args={"path": target},
                        expected_outcome="Inspect the current implementation before editing it.",
                        final_response=None,
                    )
                bootstrap = self._repair_bootstrap_candidate(
                    session,
                    target,
                    read_paths,
                    repair_context,
                )
                if bootstrap is not None:
                    return AgentDecision(
                        thought_summary="Read the most relevant supporting artifact before creating the repair.",
                        action_type=AgentActionType.CALL_TOOL,
                        tool_name="read_file",
                        tool_args={"path": bootstrap},
                        expected_outcome="Inspect the surrounding validation context before creating the repair artifact.",
                        final_response=None,
                    )
                draft = self._draft_update_decision(route, session, target)
                if draft is not None:
                    return draft

            if step.action == RouteActionName.DELETE_ARTIFACT:
                target = self._primary_target_path(route, session)
                if target is None:
                    query = self._best_search_query(route)
                    if query and query not in searched_queries:
                        return AgentDecision(
                            thought_summary="Locate the deletion target before removing anything.",
                            action_type=AgentActionType.CALL_TOOL,
                            tool_name="search_in_files",
                            tool_args={"query": query, "path": ".", "max_results": 30},
                            expected_outcome="Find the file or artifact that should be deleted.",
                            final_response=None,
                        )
                    return self._final_decision(
                        "Deletion requires a concrete target path.",
                        self._clarification_response(
                            route.model_copy(
                                update={
                                    "needs_clarification": True,
                                    "clarification_questions": [
                                        "Welche Datei oder welches Artefakt soll ich loeschen?"
                                    ],
                                    "safe_to_execute": False,
                                }
                            ),
                            session=session,
                        ),
                    )
                if target not in read_paths:
                    return AgentDecision(
                        thought_summary="Read the target once before deleting it.",
                        action_type=AgentActionType.CALL_TOOL,
                        tool_name="read_file",
                        tool_args={"path": target},
                        expected_outcome="Confirm the deletion target before removing it.",
                        final_response=None,
                    )
                return AgentDecision(
                    thought_summary=step.reason,
                    action_type=AgentActionType.CALL_TOOL,
                    tool_name="delete_file",
                    tool_args={"path": target},
                    expected_outcome="Delete the routed target artifact.",
                    final_response=None,
                )

            if step.action == RouteActionName.PLAN_WORK:
                return self._final_decision(
                    "The user primarily asked for a plan.",
                    self._render_plan_response(route, session=session),
                )

            if step.action == RouteActionName.RUN_VALIDATION and session.changed_files:
                command = self._pick_validation_command(session)
                if command is not None:
                    return self._validation_decision(
                        step.reason,
                        command.command,
                        expected_stdout=command.expected_stdout,
                    )

            if step.action == RouteActionName.SUMMARIZE_RESULT:
                return self._final_decision(
                    "The routed execution plan is ready to summarize.",
                    self._compose_user_response(route, session),
                )

        if route.repo_context_needed and not session.tool_calls:
            return AgentDecision(
                thought_summary="The route needs repository context before a safe answer.",
                action_type=AgentActionType.CALL_TOOL,
                tool_name="inspect_workspace",
                tool_args={"focus": route.user_goal},
                expected_outcome="Collect repository structure and relevant entrypoints.",
                final_response=None,
            )

        return self._final_decision(
            "No further tool step is required by the routed plan.",
            self._compose_user_response(route, session),
        )

    def summarize_session(self, session: SessionState) -> str:
        self._require_task_state(session)
        language = self._session_language(session)
        route = session.router_result
        if route is not None and route.direct_response and not session.tool_calls:
            return route.direct_response
        if session.changed_files and self._functional_validation_missing(session):
            return self._localized_text(
                language,
                de="Ich habe die Aufgabe umgesetzt, aber nur statische oder strukturelle Checks bestaetigt und noch keinen funktionalen Abschluss erreicht.",
                en="I implemented the task, but only static or structural checks were confirmed and I do not have a functional sign-off yet.",
            )
        if session.changed_files and self._requirements_review_missing(session):
            return self._localized_text(
                language,
                de="Ich habe die Aufgabe umgesetzt, aber die allgemeine Anforderungspruefung ueber die geaenderten Artefakte ist noch nicht abgeschlossen.",
                en="I implemented the task, but the general requirements review across the changed artifacts is not complete yet.",
            )
        if session.changed_files and session.validation_status == "passed":
            return self._localized_text(
                language,
                de="Ich habe die Aufgabe umgesetzt und validiert.",
                en="I implemented the task and validated it.",
            )
        if session.changed_files and session.validation_status == "bootstrap_failed":
            return self._localized_text(
                language,
                de="Ich habe die Aufgabe umgesetzt, aber die Validierung scheitert aktuell noch im Bootstrap- oder Discovery-Pfad.",
                en="I implemented the task, but validation is currently still failing in the bootstrap or discovery path.",
            )
        if session.changed_files and session.validation_status == "bootstrap_reset_required":
            return self._localized_text(
                language,
                de="Ich habe die Aufgabe umgesetzt, aber der Bootstrap-Pfad braucht vor weiteren Retries einen Reset oder menschliche Entscheidung.",
                en="I implemented the task, but the bootstrap path now needs a reset or human decision before more retries.",
            )
        if session.changed_files and session.validation_status == "failed":
            return self._localized_text(
                language,
                de="Ich habe die Aufgabe umgesetzt, aber die Validierung ist noch nicht sauber.",
                en="I implemented the task, but validation is still failing.",
            )
        if session.changed_files:
            return self._localized_text(
                language,
                de="Ich habe die Aufgabe umgesetzt, aber noch nicht sauber validiert.",
                en="I implemented the task, but it is not cleanly validated yet.",
            )
        if route is not None and route.intent == RouteIntent.DEBUG and session.diagnostics:
            return self._localized_text(
                language,
                de="Ich habe das Problem diagnostiziert und die relevanten Fehlerspuren gesammelt.",
                en="I diagnosed the problem and gathered the relevant failure evidence.",
            )
        if route is not None and route.intent == RouteIntent.PLAN:
            return self._render_plan_response(route, session=session)
        if route is not None and route.needs_clarification:
            return self._clarification_response(route, session=session)
        return self._localized_text(
            language,
            de="Ich habe den Workspace untersucht, aber noch kein belastbares Abschlussergebnis erreicht.",
            en="I inspected the workspace, but I have not reached a reliable final result yet.",
        )

    def _deterministic_final_response(self, route: RouterOutput, session: SessionState) -> str:
        changed = [item.path for item in session.changed_files[:4]]
        inspected = self._read_paths(session)[:4]
        language = self._session_language(session)
        lines: list[str] = []

        if session.changed_files:
            lines.append(
                self._localized_text(
                    language,
                    de="Ich habe die angefragte Aenderung umgesetzt.",
                    en="I applied the requested change.",
                )
            )
            if changed:
                lines.append(
                    self._localized_text(
                        language,
                        de=f"Geaendert: {', '.join(changed)}.",
                        en=f"Changed: {', '.join(changed)}.",
                    )
                )
            if self._functional_validation_missing(session):
                lines.append(
                    self._localized_text(
                        language,
                        de="Validierung: nur statische oder strukturelle Checks wurden bestaetigt; ein funktionaler Smoke-Test fehlt noch.",
                        en="Validation: only static or structural checks were confirmed; a functional smoke test is still missing.",
                    )
                )
            elif self._requirements_review_missing(session):
                lines.append(
                    self._localized_text(
                        language,
                        de="Validierung: die allgemeine Anforderungspruefung ueber die geaenderten Artefakte ist noch offen.",
                        en="Validation: the general requirements review across the changed artifacts is still pending.",
                    )
                )
            elif session.validation_status == "passed":
                lines.append(self._localized_text(language, de="Validierung: bestanden.", en="Validation: passed."))
            elif session.validation_status == "bootstrap_failed":
                lines.append(
                    self._localized_text(
                        language,
                        de="Validierung: Bootstrap-/Discovery-Fehler, normaler Repair-Pfad ist noch nicht freigegeben.",
                        en="Validation: bootstrap/discovery failure; the normal repair path is not cleared yet.",
                    )
                )
            elif session.validation_status == "bootstrap_reset_required":
                lines.append(
                    self._localized_text(
                        language,
                        de="Validierung: bootstrap_reset_required.",
                        en="Validation: bootstrap_reset_required.",
                    )
                )
            elif session.validation_status == "failed":
                lines.append(self._localized_text(language, de="Validierung: fehlgeschlagen.", en="Validation: failed."))
            elif session.validation_status == "blocked":
                lines.append(self._localized_text(language, de="Validierung: blockiert.", en="Validation: blocked."))
            else:
                lines.append(
                    self._localized_text(
                        language,
                        de="Validierung: kein sinnvoller automatischer Check wurde in diesem Lauf bestaetigt.",
                        en="Validation: no meaningful automatic check was confirmed in this run.",
                    )
                )
        elif route.intent == RouteIntent.DEBUG and session.diagnostics:
            lines.append(
                self._localized_text(
                    language,
                    de="Ich habe die relevanten Fehlerspuren gesammelt und das Problem eingegrenzt.",
                    en="I gathered the relevant failure evidence and narrowed down the problem.",
                )
            )
        elif inspected:
            lines.append(
                self._localized_text(
                    language,
                    de=f"Ich habe vor allem {', '.join(inspected)} untersucht.",
                    en=f"I mainly inspected {', '.join(inspected)}.",
                )
            )
        else:
            lines.append(self.summarize_session(session))

        if session.blockers:
            lines.append(
                self._localized_text(
                    language,
                    de=f"Blocker: {session.blockers[-1]}.",
                    en=f"Blocker: {session.blockers[-1]}.",
                )
            )
        elif session.validation_status in {"failed", "bootstrap_failed", "bootstrap_reset_required"} and session.validation_runs:
            lines.append(
                self._localized_text(
                    language,
                    de=f"Letzter Check: {session.validation_runs[-1].command}.",
                    en=f"Last check: {session.validation_runs[-1].command}.",
                )
            )

        return "\n\n".join(part for part in lines if part)

    def _draft_create_decision(
        self,
        route: RouterOutput,
        session: SessionState,
    ) -> AgentDecision | None:
        path = self._choose_create_path(route, session)
        if not path:
            return None
        current_content = self._current_file_content(session, path)
        generation = self._generate_file_content(
            route,
            session,
            path=path,
            current_content=current_content,
        )
        if not generation.content:
            failure = generation.failure or self._build_content_generation_failure(
                route,
                session,
                path=path,
                current_content=current_content,
                repair_context=None,
                attempts=[],
            )
            self._record_generation_blocker(
                session,
                failure.blocker_message,
                stop_reason=failure.stop_reason,
                failure_class=failure.failure_class,
            )
            return self._final_decision(
                "The new artifact could not be generated from the routed goal.",
                failure.user_message,
            )
        absolute_target = Path(session.workspace_root, path)
        tool_name = "write_file" if absolute_target.exists() else "create_file"
        tool_args = {"path": path, "content": generation.content}
        if tool_name == "create_file":
            tool_args["overwrite"] = False
        return AgentDecision(
            thought_summary=f"Create the routed artifact in {path}.",
            action_type=AgentActionType.CALL_TOOL,
            tool_name=tool_name,
            tool_args=tool_args,
            expected_outcome="Add the requested artifact to the workspace.",
            final_response=None,
        )

    def _draft_update_decision(
        self,
        route: RouterOutput,
        session: SessionState,
        target: str,
    ) -> AgentDecision | None:
        repair_context = self._repair_context_for_target(route, session, target)
        current_content = self._current_file_content(session, target)
        if current_content is None:
            if repair_context is None or not self._repair_target_can_be_created(
                session,
                target,
                repair_context,
            ):
                return None
            bootstrap = self._repair_bootstrap_candidate(
                session,
                target,
                set(self._read_paths(session)),
                repair_context,
            )
            if bootstrap is not None:
                return AgentDecision(
                    thought_summary="Read the most relevant related file before creating the missing repair artifact.",
                    action_type=AgentActionType.CALL_TOOL,
                    tool_name="read_file",
                    tool_args={"path": bootstrap},
                    expected_outcome="Use the failing test or adjacent artifact as concrete context for the missing repair file.",
                    final_response=None,
                )
            return self._draft_missing_repair_artifact_decision(
                route,
                session,
                target,
                repair_context,
            )
        deterministic_repair_decision = self._deterministic_runtime_repair_decision(
            route,
            session,
            path=target,
            current_content=current_content,
            repair_context=repair_context,
        )
        if deterministic_repair_decision is not None:
            return deterministic_repair_decision
        strategies = self._repair_generation_strategies(session, repair_context, target)
        if repair_context is not None and not strategies:
            final_failure_reason = (
                f"The validation-guided repair for {target} already exhausted the targeted and escalated strategies without new evidence."
            )
            self._record_generation_blocker(
                session,
                final_failure_reason,
                stop_reason="repair_blocked_after_validation_failure",
                failure_class="repair_blocked_after_validation_failure",
            )
            return self._final_decision(
                "The current validation-guided repair is blocked until new evidence is gathered.",
                "Ich habe bereits einen gezielten und einen verschaerften Repair-Versuch ohne wirksame Mutation ausgeschoepft. Ohne neue Evidenz kann ich keinen ehrlichen weiteren Fix ableiten.",
            )
        final_failure_reason = f"No reliable update content could be generated for {target}."
        final_stop_reason = "generation_failed"
        final_failure_class = "generation_failed"
        final_user_message = self._localized_text(
            self._session_language(session),
            de="Ich konnte noch keine belastbare Aktualisierung fuer die Zieldatei erzeugen.",
            en="I could not produce a reliable update for the target file yet.",
        )

        for strategy in strategies:
            generation = self._generate_file_content(
                route,
                session,
                path=target,
                current_content=current_content,
                repair_context=repair_context,
                repair_strategy=strategy,
            )
            if not generation.content:
                failure = generation.failure or self._build_content_generation_failure(
                    route,
                    session,
                    path=target,
                    current_content=current_content,
                    repair_context=repair_context,
                    attempts=[],
                )
                final_failure_reason = failure.blocker_message
                final_stop_reason = failure.stop_reason
                final_failure_class = failure.failure_class
                final_user_message = failure.user_message
                effective_strategy = getattr(generation, "repair_strategy_used", None) or strategy
                if repair_context is not None:
                    if failure.failure_class != "no_effective_change":
                        self._record_repair_attempt(
                            session,
                            repair_context,
                            target=target,
                            strategy=effective_strategy,
                            result="generation_failed",
                            reason=failure.blocker_message,
                        )
                if repair_context is not None and strategy != strategies[-1] and effective_strategy == strategy:
                    continue
                deferred_decision = self._continue_after_nonblocking_update_target_failure(
                    route,
                    session,
                    target=target,
                    stop_reason=failure.stop_reason,
                    repair_context=repair_context,
                )
                if deferred_decision is not None:
                    return deferred_decision
                break

            content = generation.content
            effective_strategy = getattr(generation, "repair_strategy_used", None) or strategy
            if repair_context is not None and content.strip() == REPAIR_BLOCKED_SENTINEL:
                final_failure_reason = (
                    f"The validation-guided repair for {target} could not derive a concrete fix from the current evidence."
                )
                final_stop_reason = "repair_blocked_after_validation_failure"
                final_failure_class = "repair_blocked_after_validation_failure"
                final_user_message = self._localized_text(
                    self._session_language(session),
                    de="Ich konnte aus der aktuellen Validierungs-Evidenz keinen ehrlichen weiteren Fix ableiten.",
                    en="I could not derive an honest follow-up fix from the current validation evidence.",
                )
                self._record_repair_attempt(
                    session,
                    repair_context,
                    target=target,
                    strategy=effective_strategy,
                    result="blocked",
                    reason=final_failure_reason,
                )
                if strategy != strategies[-1]:
                    continue
                break

            mutation = self._assess_effective_mutation(target, current_content, content)
            if mutation.effective:
                if repair_context is not None:
                    self._record_repair_attempt(
                        session,
                        repair_context,
                        target=target,
                        strategy=effective_strategy,
                        result="mutation_planned",
                        reason=mutation.reason,
                        mutation=mutation,
                    )
                return AgentDecision(
                    thought_summary=f"Update {target} according to the routed goal.",
                    action_type=AgentActionType.CALL_TOOL,
                    tool_name="write_file",
                    tool_args={"path": target, "content": content},
                    expected_outcome="Apply the requested update to the target artifact.",
                    final_response=None,
                )

            final_failure_reason = (
                f"The routed update for {target} did not produce a substantive repair change ({mutation.reason})."
            )
            final_stop_reason = "no_effective_change"
            final_failure_class = "no_effective_change"
            final_user_message = self._localized_text(
                self._session_language(session),
                de="Ich konnte keine belastbare inhaltliche Aenderung fuer die Zieldatei ableiten.",
                en="I could not derive a reliable substantive change for the target file.",
            )
            if repair_context is not None:
                self._log(
                    "repair_noop_detected",
                    path=target,
                    strategy=strategy,
                    reason=mutation.reason,
                    before_hash=mutation.before_hash,
                    after_hash=mutation.after_hash,
                    change_labels=mutation.change_labels,
                )
                self._record_repair_attempt(
                    session,
                    repair_context,
                    target=target,
                    strategy=effective_strategy,
                    result="no_effective_change",
                    reason=mutation.reason,
                    mutation=mutation,
                )
                if self._should_pivot_after_no_effective_change(session, target, repair_context):
                    alternative_decision = self._alternative_repair_target_decision(
                        route,
                        session,
                        repair_context,
                        current_target=target,
                    )
                    if alternative_decision is not None:
                        return alternative_decision
                if strategy != strategies[-1]:
                    continue
            deferred_decision = self._continue_after_nonblocking_update_target_failure(
                route,
                session,
                target=target,
                stop_reason=final_stop_reason,
                repair_context=repair_context,
            )
            if deferred_decision is not None:
                return deferred_decision
            break

        self._record_generation_blocker(
            session,
            final_failure_reason,
            stop_reason=final_stop_reason,
            failure_class=final_failure_class,
        )
        return self._final_decision(
            "The update content could not produce a repairable mutation.",
            final_user_message,
        )

    def _draft_missing_repair_artifact_decision(
        self,
        route: RouterOutput,
        session: SessionState,
        target: str,
        repair_context: ValidationFailureEvidence,
    ) -> AgentDecision:
        strategies = self._repair_generation_strategies(session, repair_context, target)
        if not strategies:
            final_failure_reason = (
                f"The validation-guided repair for {target} already exhausted the targeted and escalated strategies without new evidence."
            )
            self._record_generation_blocker(
                session,
                final_failure_reason,
                stop_reason="repair_blocked_after_validation_failure",
                failure_class="repair_blocked_after_validation_failure",
            )
            return self._final_decision(
                "The current validation-guided repair is blocked until new evidence is gathered.",
                "Ich habe bereits einen gezielten und einen verschaerften Repair-Versuch ohne wirksame Mutation ausgeschoepft. Ohne neue Evidenz kann ich keinen ehrlichen weiteren Fix ableiten.",
            )

        final_failure_reason = f"No reliable repair content could be generated for missing artifact {target}."
        final_stop_reason = "generation_failed"
        final_failure_class = "generation_failed"
        final_user_message = self._localized_text(
            self._session_language(session),
            de="Ich konnte noch keinen belastbaren Reparaturinhalt fuer das fehlende Zielartefakt erzeugen.",
            en="I could not produce reliable repair content for the missing target artifact yet.",
        )

        for strategy in strategies:
            generation = self._generate_file_content(
                route,
                session,
                path=target,
                current_content=None,
                repair_context=repair_context,
                repair_strategy=strategy,
            )
            if not generation.content:
                failure = generation.failure or self._build_content_generation_failure(
                    route,
                    session,
                    path=target,
                    current_content=None,
                    repair_context=repair_context,
                    attempts=[],
                )
                final_failure_reason = failure.blocker_message
                final_stop_reason = failure.stop_reason
                final_failure_class = failure.failure_class
                final_user_message = failure.user_message
                self._record_repair_attempt(
                    session,
                    repair_context,
                    target=target,
                    strategy=strategy,
                    result="generation_failed",
                    reason=failure.blocker_message,
                )
                if strategy != strategies[-1]:
                    continue
                alternative_decision = self._alternative_repair_target_decision(
                    route,
                    session,
                    repair_context,
                    current_target=target,
                )
                if alternative_decision is not None:
                    return alternative_decision
                break

            content = generation.content
            if content.strip() == REPAIR_BLOCKED_SENTINEL:
                final_failure_reason = (
                    f"The validation-guided repair for {target} could not derive a concrete fix from the current evidence."
                )
                final_stop_reason = "repair_blocked_after_validation_failure"
                final_failure_class = "repair_blocked_after_validation_failure"
                final_user_message = self._localized_text(
                    self._session_language(session),
                    de="Ich konnte aus der aktuellen Validierungs-Evidenz keinen ehrlichen weiteren Fix ableiten.",
                    en="I could not derive an honest follow-up fix from the current validation evidence.",
                )
                self._record_repair_attempt(
                    session,
                    repair_context,
                    target=target,
                    strategy=strategy,
                    result="blocked",
                    reason=final_failure_reason,
                )
                if strategy != strategies[-1]:
                    continue
                break

            self._record_repair_attempt(
                session,
                repair_context,
                target=target,
                strategy=strategy,
                result="mutation_planned",
                reason=f"Prepared missing repair artifact {target} from validation evidence.",
            )
            return AgentDecision(
                thought_summary=f"Create the validation-guided repair artifact in {target}.",
                action_type=AgentActionType.CALL_TOOL,
                tool_name="create_file",
                tool_args={"path": target, "content": content, "overwrite": False},
                expected_outcome="Add the missing artifact required to repair the failed validation.",
                final_response=None,
            )

        self._record_generation_blocker(
            session,
            final_failure_reason,
            stop_reason=final_stop_reason,
            failure_class=final_failure_class,
        )
        return self._final_decision(
            "The repair content for the missing artifact could not be generated reliably.",
            final_user_message,
        )

    def _repair_context_for_target(
        self,
        route: RouterOutput,
        session: SessionState,
        target: str,
    ) -> ValidationFailureEvidence | None:
        del route
        context = session.active_repair_context
        if context is None:
            failed_run = self.validation_planner.latest_failed_run(
                session,
                current_generation_only=False,
            )
            if failed_run is None:
                return None
            context = self.validation_planner.build_failure_evidence(session, failed_run)
            session.active_repair_context = context
        implicated_paths = self._unique_paths([*context.artifact_paths, *context.file_hints])
        if implicated_paths and target not in implicated_paths:
            return None
        return context

    def _alternative_repair_target_decision(
        self,
        route: RouterOutput,
        session: SessionState,
        repair_context: ValidationFailureEvidence,
        *,
        current_target: str,
    ) -> AgentDecision | None:
        failed_run = self.validation_planner.latest_failed_run(
            session,
            current_generation_only=False,
            command=repair_context.command,
        )
        if failed_run is None:
            return None
        alternative_route = self._repair_route_after_failed_validation(
            route,
            session,
            failed_run,
            repair_context,
        )
        if alternative_route is None:
            return None
        alternative_targets = [
            candidate
            for candidate in alternative_route.entities.target_paths
            if candidate and candidate != current_target
        ]
        if not alternative_targets:
            return None
        return self.execute_action_from_plan(alternative_route, session)

    def _should_pivot_after_no_effective_change(
        self,
        session: SessionState,
        target: str,
        repair_context: ValidationFailureEvidence | None,
    ) -> bool:
        text = str(target or "").strip()
        if not text or repair_context is None:
            return False
        locked_target = self._repair_brief_locked_target(repair_context)
        primary_target = self._repair_brief_primary_target(repair_context)
        if text in {locked_target, primary_target}:
            allow_early_direct_script_pivot = (
                self._is_direct_runtime_entrypoint_script_target(text, repair_context)
                and self._repair_attempt_failure_count(session, repair_context, text) >= 1
            )
            if not allow_early_direct_script_pivot and self._repair_attempt_failure_count(session, repair_context, text) < 2:
                return False
        if self.validation_planner._is_test_path(text):
            return True
        if Path(text).suffix.lower() in {".md", ".markdown", ".txt", ".rst"}:
            return True
        if repair_context.verification_scope != "runtime":
            return False

        alternative_paths = self._unique_paths(
            [*repair_context.artifact_paths, *repair_context.file_hints]
        )
        implementation_alternatives = [
            candidate
            for candidate in alternative_paths
            if candidate
            and candidate != text
            and not self.validation_planner._is_test_path(candidate)
            and Path(candidate).suffix.lower() not in {".md", ".markdown", ".txt", ".rst"}
        ]
        return bool(implementation_alternatives)

    def _repair_generation_strategies(
        self,
        session: SessionState,
        repair_context: ValidationFailureEvidence | None,
        target: str,
    ) -> list[str]:
        if repair_context is None:
            return ["default"]
        preferred_patterns, blocked_patterns = self._historical_repair_pattern_hints(
            session,
            repair_context,
            target,
        )
        if self._repair_attempt_needs_new_evidence(session, repair_context, target):
            return []
        if self._repair_attempt_seen(
            session,
            repair_context,
            target,
            strategy=ESCALATED_REPAIR_STRATEGY,
            results={"no_effective_change", "blocked", "generation_failed"},
        ):
            return []
        if self._repair_attempt_seen(
            session,
            repair_context,
            target,
            strategy=TARGETED_REPAIR_STRATEGY,
            results={"no_effective_change", "blocked", "generation_failed"},
        ):
            strategies = [ESCALATED_REPAIR_STRATEGY]
        else:
            strategies = [TARGETED_REPAIR_STRATEGY, ESCALATED_REPAIR_STRATEGY]
        filtered_strategies = [item for item in strategies if item not in blocked_patterns]
        if filtered_strategies:
            strategies = filtered_strategies
        elif self._repair_attempted_in_session(session, repair_context, target):
            return []
        if not strategies:
            return []
        strategies.sort(
            key=lambda item: (
                0 if item in preferred_patterns else 1,
                0 if item == TARGETED_REPAIR_STRATEGY else 1,
            )
        )
        return strategies

    def _historical_repair_pattern_hints(
        self,
        session: SessionState,
        repair_context: ValidationFailureEvidence,
        target: str,
    ) -> tuple[set[str], set[str]]:
        memory_context = session.memory_context
        if memory_context is None:
            return set(), set()
        failure_signature = self._repair_failure_signature(repair_context)
        target_name = Path(str(target or "").strip()).name
        preferred: set[str] = set()
        blocked: set[str] = set()
        for item in memory_context.selected:
            if item.memory_type != "failure" or item.entry is None:
                continue
            entry = item.entry
            entry_signature = str(getattr(entry, "failure_signature", "") or "").strip()
            if failure_signature and entry_signature and entry_signature != failure_signature:
                continue
            chosen_targets = [
                str(candidate or "").strip()
                for candidate in list(getattr(entry, "chosen_targets", []) or [])
                if str(candidate or "").strip()
            ]
            if chosen_targets and target_name:
                entry_target_names = {Path(candidate).name for candidate in chosen_targets}
                if str(target or "").strip() not in chosen_targets and target_name not in entry_target_names:
                    continue
            preferred.update(
                str(candidate or "").strip()
                for candidate in list(getattr(entry, "successful_repair_patterns", []) or [])
                if str(candidate or "").strip()
            )
            blocked.update(
                str(candidate or "").strip()
                for candidate in list(getattr(entry, "bad_retry_patterns", []) or [])
                if str(candidate or "").strip()
            )
        return preferred, blocked

    def _repair_attempted_in_session(
        self,
        session: SessionState,
        repair_context: ValidationFailureEvidence,
        target: str,
    ) -> bool:
        return any(
            item.evidence_signature == repair_context.evidence_signature
            and item.artifact_path == target
            for item in session.repair_history
        )

    def _memory_recall_response(
        self,
        route: RouterOutput,
        session: SessionState,
    ) -> str | None:
        memory_context = session.memory_context
        if memory_context is None:
            return None
        request = getattr(memory_context, "request", None)
        if str(getattr(request, "use_case", "") or "").strip() != "user_recall":
            return None
        recall_brief = str(getattr(memory_context, "recall_brief", "") or "").strip()
        if not recall_brief:
            return None
        if route.intent not in {
            RouteIntent.EXPLAIN,
            RouteIntent.PLAN,
            RouteIntent.UNKNOWN,
            RouteIntent.SEARCH,
        }:
            return None
        if route.entities.target_paths:
            return None
        return recall_brief

    def _memory_guided_candidate_paths(
        self,
        session: SessionState,
        *,
        failure_only: bool = False,
    ) -> list[str]:
        memory_context = session.memory_context
        if memory_context is None:
            return []
        candidates: list[str] = []
        candidates.extend(memory_context.suggested_files[:8])
        for item in memory_context.selected:
            if failure_only and item.memory_type != "failure":
                continue
            candidates.extend(item.file_paths[:6])
            entry = item.entry
            if entry is None:
                continue
            candidates.extend(list(getattr(entry, "chosen_targets", []) or [])[:6])
            if failure_only:
                continue
            candidates.extend(list(getattr(entry, "changed_files", []) or [])[:4])
            candidates.extend(list(getattr(entry, "known_hotspots", []) or [])[:4])
        return self._unique_paths(candidates)

    def _repair_attempt_seen(
        self,
        session: SessionState,
        repair_context: ValidationFailureEvidence,
        target: str,
        *,
        strategy: str,
        results: set[str],
    ) -> bool:
        return any(
            item.strategy == strategy
            and item.result in results
            and item.evidence_signature == repair_context.evidence_signature
            and item.artifact_path == target
            for item in session.repair_history
        )

    def _repair_attempt_needs_new_evidence(
        self,
        session: SessionState,
        repair_context: ValidationFailureEvidence,
        target: str,
    ) -> bool:
        last_attempt = next(
            (
                item
                for item in reversed(session.repair_history)
                if item.evidence_signature == repair_context.evidence_signature
                and item.artifact_path == target
            ),
            None,
        )
        if last_attempt is None:
            return False
        if last_attempt.result == "mutation_planned":
            return False
        last_iteration = int(last_attempt.iteration or 0)
        if any(
            int(getattr(item, "iteration", 0) or 0) > last_iteration
            for item in session.diagnostics
        ):
            return False
        if any(
            int(getattr(item, "iteration", 0) or 0) > last_iteration
            and item.tool_name == "read_file"
            and str(item.tool_args.get("path") or "").strip() == target
            for item in session.tool_calls
        ):
            return False
        return self._repair_attempt_seen(
            session,
            repair_context,
            target,
            strategy=ESCALATED_REPAIR_STRATEGY,
            results={"no_effective_change", "blocked", "generation_failed"},
        )

    def _record_repair_attempt(
        self,
        session: SessionState,
        repair_context: ValidationFailureEvidence,
        *,
        target: str,
        strategy: str,
        result: str,
        reason: str,
        mutation: MutationAssessment | None = None,
    ) -> None:
        root_cause_summary = str(getattr(repair_context, "root_cause_summary", "") or "").strip()
        productive_change = mutation.effective if mutation is not None else result == "mutation_planned"
        session.repair_history.append(
            RepairAttemptRecord(
                artifact_path=target,
                validation_command=repair_context.command,
                verification_scope=repair_context.verification_scope,
                strategy=strategy,
                result=result,
                reason=reason,
                evidence_signature=repair_context.evidence_signature,
                failure_signature=self._repair_failure_signature(repair_context),
                region_hint=self._repair_region_hint(repair_context),
                root_cause_summary=root_cause_summary or None,
                productive_change=productive_change,
                before_hash=mutation.before_hash if mutation is not None else None,
                after_hash=mutation.after_hash if mutation is not None else None,
                change_labels=list(mutation.change_labels) if mutation is not None else [],
                iteration=session.iterations,
            )
        )
        session.repair_history = session.repair_history[-20:]

    def _repair_failure_signature(
        self,
        repair_context: ValidationFailureEvidence,
    ) -> str | None:
        brief = getattr(repair_context, "repair_brief", None)
        if brief is None:
            return None
        signature = str(getattr(brief, "failure_signature", "") or "").strip()
        return signature or None

    def _repair_region_hint(
        self,
        repair_context: ValidationFailureEvidence,
    ) -> str | None:
        brief = getattr(repair_context, "repair_brief", None)
        if brief is None:
            return None
        region = str(getattr(brief, "implicated_region_hint", "") or "").strip()
        return region or None

    def _repair_context_root_cause_summary(
        self,
        repair_context: ValidationFailureEvidence | None,
    ) -> str | None:
        if repair_context is None:
            return None
        summary = str(getattr(repair_context, "root_cause_summary", "") or "").strip()
        if summary:
            return summary
        brief = getattr(repair_context, "repair_brief", None)
        summary = str(getattr(brief, "root_cause_summary", "") or "").strip()
        return summary or None

    def _repair_context_bootstrap_status(
        self,
        repair_context: ValidationFailureEvidence | None,
    ) -> str:
        if repair_context is None:
            return "none"
        status = str(getattr(repair_context, "bootstrap_status", "") or "").strip()
        if status in {"bootstrap_failed", "bootstrap_reset_required"}:
            return status
        brief = getattr(repair_context, "repair_brief", None)
        status = str(getattr(brief, "bootstrap_status", "") or "").strip()
        if status in {"bootstrap_failed", "bootstrap_reset_required"}:
            return status
        return "none"

    def _proposed_update_review_text(
        self,
        review: ProposedUpdateReview,
    ) -> str:
        return " ".join(
            part.strip()
            for part in [review.summary, *review.blocking_issues, *review.repair_hints]
            if str(part or "").strip()
        ).lower()

    def _review_feedback_is_noop(
        self,
        review: ProposedUpdateReview,
    ) -> bool:
        text = self._proposed_update_review_text(review)
        if not text:
            return False
        markers = (
            "no-op",
            "effectively unchanged",
            "formal rewrite",
            "same failing state",
            "does not make an effective change",
            "does not make a productive change",
            "whitespace-only",
            "comment-only",
            "metadata-only",
            "equivalent repair",
            "leaves the implicated",
            "still leaves the implicated",
            "preserves the observed",
            "still preserves the observed",
            "leaves the observed",
        )
        return any(marker in text for marker in markers)

    def _review_feedback_noop_reason(
        self,
        review: ProposedUpdateReview,
    ) -> str:
        text = self._proposed_update_review_text(review)
        if "file hash unchanged" in text:
            return "file hash unchanged"
        if "whitespace-only" in text:
            return "whitespace-only change"
        if "comment-only" in text:
            return "comment-only change"
        if "metadata-only" in text:
            return "metadata-only change"
        if "formal rewrite" in text:
            return "formal-only rewrite"
        if "effectively unchanged" in text or "equivalent repair" in text:
            return "equivalent repair"
        return "equivalent repair"

    def _review_retry_repair_strategy(
        self,
        repair_strategy: str | None,
        review_feedback: ProposedUpdateReview,
        repair_context: ValidationFailureEvidence | None,
    ) -> str | None:
        if repair_context is None:
            return repair_strategy
        if self._review_feedback_is_noop(review_feedback):
            return ESCALATED_REPAIR_STRATEGY
        return repair_strategy

    def _prefer_reserve_after_initial_primary_noop(
        self,
        *,
        review_feedback: ProposedUpdateReview,
        repair_context: ValidationFailureEvidence | None,
        primary_model: str | None,
        reserve_model: str | None,
        prior_attempts: list[ExecutionAttemptRecord],
    ) -> bool:
        primary = str(primary_model or "").strip()
        reserve = str(reserve_model or "").strip()
        if (
            repair_context is None
            or not primary
            or not reserve
            or reserve == primary
            or not self._review_feedback_is_noop(review_feedback)
        ):
            return False
        return any(str(item.model_identifier or "").strip() == primary for item in prior_attempts)

    def _record_noop_review_attempt(
        self,
        session: SessionState,
        *,
        target: str,
        current_content: str | None,
        proposed_content: str,
        repair_context: ValidationFailureEvidence | None,
        strategy: str | None,
        review: ProposedUpdateReview,
    ) -> bool:
        if repair_context is None or current_content is None or not self._review_feedback_is_noop(review):
            return False

        mutation = self._assess_effective_mutation(target, current_content, proposed_content)
        if mutation.effective:
            mutation = MutationAssessment(
                effective=False,
                reason=self._review_feedback_noop_reason(review),
                before_hash=self._content_sha256(current_content),
                after_hash=self._content_sha256(proposed_content),
                change_labels=["equivalent_repair"],
                changed_line_count=self._compact_repair_change_line_count(
                    current_content=current_content,
                    proposed_content=proposed_content,
                ),
            )
        effective_strategy = strategy or TARGETED_REPAIR_STRATEGY
        self._log(
            "repair_noop_detected",
            path=target,
            strategy=effective_strategy,
            reason=mutation.reason,
            before_hash=mutation.before_hash,
            after_hash=mutation.after_hash,
            change_labels=mutation.change_labels,
            source="pre_write_review",
        )
        self._record_repair_attempt(
            session,
            repair_context,
            target=target,
            strategy=effective_strategy,
            result="no_effective_change",
            reason=mutation.reason,
            mutation=mutation,
        )
        return True

    def _repair_brief_primary_target(
        self,
        repair_context: ValidationFailureEvidence | None,
    ) -> str | None:
        if repair_context is None or getattr(repair_context, "repair_brief", None) is None:
            return None
        target = str(getattr(repair_context.repair_brief, "primary_target", "") or "").strip()
        return target or None

    def _repair_brief_locked_target(
        self,
        repair_context: ValidationFailureEvidence | None,
    ) -> str | None:
        if repair_context is None or getattr(repair_context, "repair_brief", None) is None:
            return None
        target = str(getattr(repair_context.repair_brief, "locked_target", "") or "").strip()
        if target:
            return target
        return self._repair_brief_primary_target(repair_context)

    def _repair_attempt_failure_count(
        self,
        session: SessionState,
        repair_context: ValidationFailureEvidence,
        target: str,
    ) -> int:
        normalized_target = str(target or "").strip()
        if not normalized_target:
            return 0
        failure_signature = self._repair_failure_signature(repair_context)
        evidence_signature = str(repair_context.evidence_signature or "").strip()
        count = 0
        for item in reversed(session.repair_history):
            if str(item.artifact_path or "").strip() != normalized_target:
                continue
            if item.result not in {"no_effective_change", "blocked", "generation_failed"}:
                continue
            if failure_signature and str(item.failure_signature or "").strip() == failure_signature:
                count += 1
                continue
            if not failure_signature and evidence_signature and str(item.evidence_signature or "").strip() == evidence_signature:
                count += 1
        return count

    def _repair_attempt_mutation_count(
        self,
        session: SessionState,
        repair_context: ValidationFailureEvidence,
        target: str,
    ) -> int:
        normalized_target = str(target or "").strip()
        if not normalized_target:
            return 0
        failure_signature = self._repair_failure_signature(repair_context)
        evidence_signature = str(repair_context.evidence_signature or "").strip()
        count = 0
        for item in reversed(session.repair_history):
            if str(item.artifact_path or "").strip() != normalized_target:
                continue
            if item.result != "mutation_planned":
                continue
            if failure_signature and str(item.failure_signature or "").strip() == failure_signature:
                count += 1
                continue
            if not failure_signature and evidence_signature and str(item.evidence_signature or "").strip() == evidence_signature:
                count += 1
        return count

    def _assess_effective_mutation(
        self,
        path: str,
        current_content: str,
        new_content: str,
    ) -> MutationAssessment:
        before_hash = self._content_sha256(current_content)
        after_hash = self._content_sha256(new_content)
        changed_line_count = self._compact_repair_change_line_count(
            current_content=current_content,
            proposed_content=new_content,
        )
        if before_hash == after_hash:
            return MutationAssessment(
                False,
                "file hash unchanged",
                before_hash=before_hash,
                after_hash=after_hash,
                change_labels=["hash_unchanged"],
                changed_line_count=changed_line_count,
            )
        if new_content.strip() == current_content.strip():
            return MutationAssessment(
                False,
                "identical content",
                before_hash=before_hash,
                after_hash=after_hash,
                change_labels=["equivalent_content"],
                changed_line_count=changed_line_count,
            )
        if self._normalized_mutation_content(new_content) == self._normalized_mutation_content(current_content):
            return MutationAssessment(
                False,
                "whitespace-only change",
                before_hash=before_hash,
                after_hash=after_hash,
                change_labels=["whitespace_only"],
                changed_line_count=changed_line_count,
            )
        if self._normalized_comment_stripped_content(path, new_content) == self._normalized_comment_stripped_content(
            path,
            current_content,
        ):
            return MutationAssessment(
                False,
                "comment-only change",
                before_hash=before_hash,
                after_hash=after_hash,
                change_labels=["comment_only"],
                changed_line_count=changed_line_count,
            )
        if self._normalized_metadata_stripped_content(path, new_content) == self._normalized_metadata_stripped_content(
            path,
            current_content,
        ):
            return MutationAssessment(
                False,
                "metadata-only change",
                before_hash=before_hash,
                after_hash=after_hash,
                change_labels=["metadata_only"],
                changed_line_count=changed_line_count,
            )
        return MutationAssessment(
            True,
            "substantive mutation prepared",
            before_hash=before_hash,
            after_hash=after_hash,
            change_labels=["productive_change"],
            changed_line_count=changed_line_count,
        )

    def _normalized_mutation_content(self, content: str) -> str:
        return re.sub(r"\s+", " ", str(content or "").strip())

    def _content_sha256(self, content: str) -> str:
        return hashlib.sha256(str(content or "").encode("utf-8")).hexdigest()

    def _normalized_comment_stripped_content(self, path: str, content: str) -> str:
        stripped = self._strip_comments_for_noop_detection(path, content)
        lines = [str(line or "").strip() for line in str(stripped or "").splitlines() if str(line or "").strip()]
        return "\n".join(lines)

    def _normalized_metadata_stripped_content(self, path: str, content: str) -> str:
        text = self._strip_comments_for_noop_detection(path, content)
        lines: list[str] = []
        raw_lines = [str(raw or "").rstrip() for raw in str(text or "").splitlines()]
        front_matter_end = self._front_matter_end_index(path, raw_lines)
        for index, raw in enumerate(raw_lines, start=1):
            if front_matter_end is not None and index <= front_matter_end:
                continue
            stripped = raw.strip()
            if not stripped or self._line_is_metadata_only(path, stripped, line_number=index):
                continue
            lines.append(stripped)
        return "\n".join(lines)

    def _strip_comments_for_noop_detection(self, path: str, content: str) -> str:
        suffix = Path(str(path or "").strip()).suffix.lower()
        text = str(content or "")
        if suffix in {".py", ".pyi"}:
            return self._strip_python_comments(text)
        if suffix in {".html", ".htm", ".xml", ".svg", ".md", ".markdown"}:
            text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        if suffix in {
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".c",
            ".cc",
            ".cpp",
            ".cxx",
            ".h",
            ".hpp",
            ".java",
            ".kt",
            ".css",
            ".scss",
            ".less",
            ".go",
            ".rs",
            ".php",
        }:
            text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        line_prefixes = self._comment_prefixes_for_path(path)
        kept_lines: list[str] = []
        for raw in text.splitlines():
            stripped = str(raw or "").strip()
            if not stripped:
                continue
            if any(stripped.startswith(prefix) for prefix in line_prefixes):
                continue
            kept_lines.append(str(raw or "").rstrip())
        return "\n".join(kept_lines)

    def _strip_python_comments(self, content: str) -> str:
        try:
            tokens = tokenize.generate_tokens(io.StringIO(str(content or "")).readline)
            kept = [(token.type, token.string) for token in tokens if token.type != tokenize.COMMENT]
            return tokenize.untokenize(kept)
        except (SyntaxError, tokenize.TokenError):
            return str(content or "")

    def _comment_prefixes_for_path(self, path: str) -> tuple[str, ...]:
        suffix = Path(str(path or "").strip()).suffix.lower()
        if suffix in {".py", ".pyi", ".sh", ".yaml", ".yml", ".toml", ".rb"}:
            return ("#",)
        if suffix in {".ini", ".cfg", ".conf", ".properties"}:
            return ("#", ";", "!")
        if suffix in {".sql", ".lua"}:
            return ("--",)
        if suffix in {
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".c",
            ".cc",
            ".cpp",
            ".cxx",
            ".h",
            ".hpp",
            ".java",
            ".kt",
            ".css",
            ".scss",
            ".less",
            ".go",
            ".rs",
            ".php",
        }:
            return ("//", "/*", "*", "*/")
        if suffix in {".html", ".htm", ".xml", ".svg", ".md", ".markdown"}:
            return ("<!--", "-->")
        return tuple()

    def _front_matter_end_index(self, path: str, lines: list[str]) -> int | None:
        suffix = Path(str(path or "").strip()).suffix.lower()
        if suffix not in {".md", ".markdown", ".mdx", ".html", ".htm", ".txt", ".rst"}:
            return None
        if not lines or str(lines[0] or "").strip() != "---":
            return None
        for index, raw in enumerate(lines[1:12], start=2):
            if str(raw or "").strip() == "---":
                return index
        return None

    def _line_is_metadata_only(self, path: str, stripped_line: str, *, line_number: int) -> bool:
        suffix = Path(str(path or "").strip()).suffix.lower()
        lowered = str(stripped_line or "").strip().lower()
        if not lowered:
            return False
        if line_number == 1 and lowered.startswith("#!"):
            return True
        if line_number <= 2 and "coding:" in lowered:
            return True
        if lowered.startswith("<meta ") or lowered.startswith("<meta\t"):
            return True
        return bool(
            re.match(
                r"^(?:__version__|__author__|__copyright__|version|name|description|author|authors|license|"
                r"generated(?:_at)?|updated(?:_at)?|last_updated|timestamp|date|created_at|modified_at)\s*[:=]",
                lowered,
            )
        )

    def _has_mutation_since_failed_validation(
        self,
        session: SessionState,
        failed_run,
    ) -> bool:
        if int(getattr(failed_run, "edit_generation", 0) or 0) < int(session.edit_generation or 0):
            return True
        failed_iteration = int(getattr(failed_run, "iteration", 0) or 0)
        return any(
            int(getattr(item, "iteration", 0) or 0) > failed_iteration
            and item.tool_name in WRITE_TOOL_NAMES
            for item in session.tool_calls
        )

    def _diagnose_issue_decision(
        self,
        route: RouterOutput,
        session: SessionState,
        candidate_paths: list[str],
        read_paths: set[str],
    ) -> AgentDecision | None:
        diagnostic_candidates = self._diagnostic_file_candidates(route, session, candidate_paths)
        unread_diagnostic = self._next_unread_candidate(diagnostic_candidates, read_paths)
        if unread_diagnostic is not None:
            return AgentDecision(
                thought_summary="Read the file most strongly implicated by the current or previous diagnostics.",
                action_type=AgentActionType.CALL_TOOL,
                tool_name="read_file",
                tool_args={"path": unread_diagnostic},
                expected_outcome="Inspect the file implicated by the failing output before editing it.",
                final_response=None,
            )

        if self._diagnostic_evidence_available(session):
            return None

        command = self._next_diagnostic_command(session)
        if command is not None and not self._command_already_ran(session, command):
            command_spec = self._diagnostic_command_spec(session, command)
            return AgentDecision(
                thought_summary="Reproduce the issue with the strongest available runtime or validation command before editing.",
                action_type=AgentActionType.CALL_TOOL,
                tool_name="run_tests",
                tool_args={
                    "command": command,
                    "cwd": ".",
                    "timeout": 120,
                    **(
                        {"expected_stdout": command_spec.expected_stdout}
                        if command_spec is not None and command_spec.expected_stdout is not None
                        else {}
                    ),
                },
                expected_outcome="Reproduce the reported issue and collect concrete diagnostics.",
                final_response=None,
            )

        if self._diagnosis_attempted_without_findings(session) or route.intent == RouteIntent.DEBUG:
            return self._final_decision(
                "There is not enough diagnostic evidence to apply a safe fix yet.",
                self._missing_issue_evidence_response(route, session),
            )
        return None

    def _repair_after_failed_validation(
        self,
        route: RouterOutput,
        session: SessionState,
    ) -> AgentDecision | None:
        failed_run = self.validation_planner.latest_failed_run(
            session,
            current_generation_only=False,
        )
        if failed_run is None:
            return None

        repair_context = self.validation_planner.build_failure_evidence(session, failed_run)
        session.active_repair_context = repair_context
        if session.validation_status == "bootstrap_reset_required":
            blocker = self._bootstrap_reset_blocker_message(repair_context)
            if blocker not in session.blockers:
                session.blockers.append(blocker)
                session.blockers = session.blockers[-10:]
            session.last_error = blocker
            session.stop_reason = "needs_human_or_bootstrap_reset"
            return self._final_decision(
                "The current bootstrap failure needs a reset or human intervention before another repair loop.",
                self._bootstrap_reset_blocked_response(route, session, repair_context),
            )

        repair_route = self._repair_route_after_failed_validation(
            route,
            session,
            failed_run,
            repair_context,
        )
        if repair_route is not None:
            repair_decision = self.execute_action_from_plan(repair_route, session)
            if (
                repair_decision.action_type == AgentActionType.CALL_TOOL
                and repair_decision.tool_name == "run_tests"
                and not self._has_mutation_since_failed_validation(session, failed_run)
            ):
                blocker = self._validation_repair_blocker_message(failed_run, repair_context)
                if blocker not in session.blockers:
                    session.blockers.append(blocker)
                    session.blockers = session.blockers[-10:]
                session.last_error = blocker
                session.stop_reason = "repair_blocked_after_validation_failure"
                return self._final_decision(
                    "Validation failed and there is still no substantive repair mutation to re-verify.",
                    self._validation_repair_blocked_response(route, session, failed_run, repair_context),
                )
            if repair_decision.action_type != AgentActionType.FINAL or session.stop_reason:
                return repair_decision
        else:
            blocker = self._validation_repair_blocker_message(
                failed_run,
                repair_context,
                repair_target_missing=True,
            )
            if blocker not in session.blockers:
                session.blockers.append(blocker)
                session.blockers = session.blockers[-10:]
            session.last_error = blocker
            session.stop_reason = "repair_blocked_after_validation_failure"
            return self._final_decision(
                "The failed validation cannot be repaired safely without a concrete artifact target.",
                self._validation_repair_blocked_response(
                    route,
                    session,
                    failed_run,
                    repair_context,
                    repair_target_missing=True,
                ),
            )

        if not self._has_mutation_since_failed_validation(session, failed_run):
            blocker = self._validation_repair_blocker_message(failed_run, repair_context)
            if blocker not in session.blockers:
                session.blockers.append(blocker)
                session.blockers = session.blockers[-10:]
            session.last_error = blocker
            session.stop_reason = "repair_blocked_after_validation_failure"
            return self._final_decision(
                "Validation failed and there is still no substantive repair mutation to re-verify.",
                self._validation_repair_blocked_response(route, session, failed_run, repair_context),
            )

        command = self._pick_validation_command(session)
        if command is not None:
            return self._validation_decision(
                "Only rerun validation after a substantive repair mutation has been prepared.",
                command.command,
                expected_stdout=command.expected_stdout,
            )

        blocker = self._validation_repair_blocker_message(failed_run, repair_context)
        if blocker not in session.blockers:
            session.blockers.append(blocker)
            session.blockers = session.blockers[-10:]
        session.last_error = blocker
        session.stop_reason = "repair_blocked_after_validation_failure"
        return self._final_decision(
            "The failed validation cannot be rerun safely without a repairable target or new evidence.",
            self._validation_repair_blocked_response(route, session, failed_run, repair_context),
        )

    def _repair_route_after_failed_validation(
        self,
        route: RouterOutput,
        session: SessionState,
        failed_run,
        repair_context: ValidationFailureEvidence,
    ) -> RouterOutput | None:
        target = self._repair_target_after_failed_validation(
            route,
            session,
            failed_run,
            repair_context,
        )
        if target is None:
            return None

        target_name = Path(target).name
        absolute_target = Path(session.workspace_root, target)
        bootstrap_phase = self._repair_context_bootstrap_status(repair_context) == "bootstrap_failed"
        root_cause_summary = self._repair_context_root_cause_summary(repair_context)
        if absolute_target.exists() and absolute_target.is_file():
            action_plan = [
                RouteActionStep(
                    step=1,
                    action=RouteActionName.READ_RELEVANT_FILES,
                    reason=(
                        "Inspect the bootstrap-critical artifact again before repairing the validation startup path."
                        if bootstrap_phase
                        else "Inspect the changed artifact again after the failed validation before repairing it."
                    ),
                ),
                RouteActionStep(
                    step=2,
                    action=RouteActionName.UPDATE_ARTIFACT,
                    reason=(
                        "Repair the failing bootstrap or discovery path before touching downstream behavior."
                        if bootstrap_phase
                        else "Repair the changed artifact using the failed validation evidence."
                    ),
                ),
                RouteActionStep(
                    step=3,
                    action=RouteActionName.RUN_VALIDATION,
                    reason=(
                        "Rerun validation to confirm the bootstrap path really changed."
                        if bootstrap_phase
                        else "Rerun the validation after the repair step."
                    ),
                ),
                RouteActionStep(
                    step=4,
                    action=RouteActionName.SUMMARIZE_RESULT,
                    reason="Report the repair result and remaining validation state honestly.",
                ),
            ]
        else:
            action_plan = [
                RouteActionStep(
                    step=1,
                    action=RouteActionName.UPDATE_ARTIFACT,
                    reason=(
                        "Create or restore the missing bootstrap-critical artifact before retrying validation."
                        if bootstrap_phase
                        else "Create or restore the missing artifact implicated by the failed validation."
                    ),
                ),
                RouteActionStep(
                    step=2,
                    action=RouteActionName.RUN_VALIDATION,
                    reason=(
                        "Rerun validation to confirm the bootstrap path really changed."
                        if bootstrap_phase
                        else "Rerun the validation after the repair step."
                    ),
                ),
                RouteActionStep(
                    step=3,
                    action=RouteActionName.SUMMARIZE_RESULT,
                    reason="Report the repair result and remaining validation state honestly.",
                ),
            ]
        failure_scope = repair_context.verification_scope
        failure_summary = repair_context.failure_summary
        constraint_lines = self._unique_paths(
            [
                *route.entities.constraints,
                *repair_context.repair_requirements[:4],
                "Do not return an equivalent file without a substantive fix.",
                "No-op repairs are forbidden: do not change only whitespace, comments, metadata, or unrelated regions.",
            ]
        )
        if bootstrap_phase:
            constraint_lines = self._unique_paths(
                [
                    *constraint_lines,
                    "Bootstrap phase: repair the startup/import/discovery path before changing downstream logic.",
                    "Do not claim progress unless the failing bootstrap signature changes or validation passes.",
                ]
            )
        attribute_lines = self._unique_paths(
            [
                *route.entities.attributes,
                f"failed_validation_scope:{failure_scope}",
                *(["repair_phase:bootstrap"] if bootstrap_phase else []),
                *(f"missing_feature:{item}" for item in repair_context.missing_features[:4]),
            ]
        )
        search_terms = self._unique_paths(
            [
                *route.search_terms,
                target_name,
                failure_scope,
                *repair_context.missing_features[:4],
            ]
        )
        next_intent = RouteIntent.DEBUG if route.intent == RouteIntent.DEBUG else RouteIntent.UPDATE
        return route.model_copy(
            update={
                "intent": next_intent,
                "requested_outcome": (
                    (
                        f"Repair the bootstrap path in {target_name} so the failed {failure_scope} validation can start cleanly. "
                        if bootstrap_phase
                        else f"Repair {target_name} so the failed {failure_scope} validation passes. "
                    )
                    + (f"Root cause: {root_cause_summary} " if root_cause_summary else "")
                    + f"Failure summary: {failure_summary} Preserve the original requested outcome."
                ),
                "entities": route.entities.model_copy(
                    update={
                        "target_name": target_name,
                        "target_paths": [target],
                        "attributes": attribute_lines[:8],
                        "constraints": constraint_lines[:8],
                    }
                ),
                "action_plan": action_plan,
                "search_terms": search_terms[:6],
            }
        )

    def _repair_target_after_failed_validation(
        self,
        route: RouterOutput,
        session: SessionState,
        failed_run,
        repair_context: ValidationFailureEvidence | None = None,
    ) -> str | None:
        candidates: list[str] = []
        if repair_context is not None:
            candidates.extend(self._prioritized_repair_target_candidates(session, repair_context))
            candidates.extend(self._memory_guided_candidate_paths(session, failure_only=True))
        candidates.extend(self._paths_from_internal_validation_command(str(failed_run.command or "")))
        for item in reversed(session.diagnostics):
            candidates.extend(item.file_hints)
        candidates.extend(route.entities.target_paths)
        candidates.extend(item.path for item in reversed(session.changed_files))
        candidates.extend(session.candidate_files)
        ordered_candidates = self._repair_candidates_with_unattempted_first(
            session,
            repair_context,
            self._unique_paths(candidates),
        )
        if repair_context is not None:
            non_documentation_candidates = [
                candidate
                for candidate in ordered_candidates
                if (
                    not self.validation_planner._is_documentation_path(candidate)
                    or (
                        repair_context.verification_scope == "runtime"
                        and self._is_runtime_support_repair_target(candidate, repair_context)
                    )
                )
            ]
            if non_documentation_candidates:
                ordered_candidates = non_documentation_candidates
        if (
            repair_context is not None
            and self._runtime_failure_needs_test_context_first(repair_context)
        ):
            read_paths = set(self._read_paths(session))
            for candidate in ordered_candidates:
                if not self.validation_planner._is_test_path(candidate):
                    continue
                if candidate in read_paths:
                    continue
                if not self._is_safe_workspace_target_candidate(session, candidate):
                    continue
                absolute = Path(session.workspace_root, candidate)
                if absolute.exists() and absolute.is_file():
                    return candidate
        if (
            repair_context is not None
            and self._repair_context_is_bootstrap_test_discovery(repair_context)
            and any(
                candidate in set(self._read_paths(session))
                for candidate in ordered_candidates
                if candidate and self.validation_planner._is_test_path(candidate)
            )
        ):
            for candidate in ordered_candidates:
                if not self._is_runtime_support_repair_target(candidate, repair_context):
                    continue
                if self._repair_target_can_be_created(session, candidate, repair_context):
                    return candidate
        for candidate in ordered_candidates:
            if not self._is_safe_workspace_target_candidate(session, candidate):
                continue
            absolute = Path(session.workspace_root, candidate)
            if absolute.exists() and absolute.is_file():
                return candidate
            if repair_context is not None and self._repair_target_can_be_created(
                session,
                candidate,
                repair_context,
            ):
                return candidate
        return None

    def _repair_context_is_bootstrap_test_discovery(
        self,
        repair_context: ValidationFailureEvidence | None,
    ) -> bool:
        if repair_context is None:
            return False
        brief = getattr(repair_context, "repair_brief", None)
        if brief is None:
            return False
        return (
            str(getattr(brief, "failure_type", "") or "").strip() == "test_discovery_gap"
            and str(getattr(brief, "bootstrap_status", "") or "").strip() == "bootstrap_failed"
        )

    def _repair_candidates_with_unattempted_first(
        self,
        session: SessionState,
        repair_context: ValidationFailureEvidence | None,
        candidates: list[str],
    ) -> list[str]:
        if repair_context is None or len(candidates) <= 1:
            return candidates

        attempted_results: dict[str, str] = {}
        for item in reversed(session.repair_history):
            if item.evidence_signature != repair_context.evidence_signature:
                continue
            if item.artifact_path in attempted_results:
                continue
            attempted_results[item.artifact_path] = item.result

        def _order_by_attempt_status(group: list[str]) -> list[str]:
            if not group:
                return []
            unattempted = [candidate for candidate in group if candidate not in attempted_results]
            if not unattempted:
                return group
            attempted = [candidate for candidate in group if candidate in attempted_results]
            return [*unattempted, *attempted]

        recent_support_target: str | None = None
        if repair_context.verification_scope == "runtime":
            runtime_support_candidates = [
                candidate
                for candidate in candidates
                if self._is_runtime_support_repair_target(candidate, repair_context)
            ]
            if runtime_support_candidates and self._runtime_support_candidates_should_lead(
                runtime_support_candidates,
                repair_context,
            ):
                locked_target = self._repair_brief_locked_target(repair_context)
                if locked_target not in runtime_support_candidates:
                    remaining = [
                        candidate for candidate in candidates if candidate not in runtime_support_candidates
                    ]
                    return [
                        *_order_by_attempt_status(runtime_support_candidates),
                        *_order_by_attempt_status(remaining),
                    ]

            recent_support_target = self._recent_runtime_support_repair_target(
                session,
                repair_context,
            )
            if (
                recent_support_target is not None
                and recent_support_target in candidates
                and not self._runtime_locked_target_should_yield(
                    session,
                    repair_context,
                    recent_support_target,
                )
                and recent_support_target in {*repair_context.artifact_paths, *repair_context.file_hints}
            ):
                remaining = [candidate for candidate in candidates if candidate != recent_support_target]
                return [recent_support_target, *_order_by_attempt_status(remaining)]

        locked_target = self._repair_brief_locked_target(repair_context)
        if locked_target and locked_target in candidates:
            if self._runtime_locked_target_should_yield(session, repair_context, locked_target):
                remaining = [candidate for candidate in candidates if candidate != locked_target]
                return [*_order_by_attempt_status(remaining), locked_target]
            remaining = [candidate for candidate in candidates if candidate != locked_target]
            return [locked_target, *_order_by_attempt_status(remaining)]

        if repair_context.verification_scope == "runtime":
            if recent_support_target is not None or attempted_results:
                support_candidates = [
                    candidate
                    for candidate in candidates
                    if candidate and candidate == recent_support_target
                ]
                implementation_candidates = [
                    candidate
                    for candidate in candidates
                    if (
                        candidate
                        and candidate != recent_support_target
                        and not self.validation_planner._is_test_path(candidate)
                    )
                ]
                validation_candidates = [
                    candidate
                    for candidate in candidates
                    if (
                        candidate
                        and candidate != recent_support_target
                        and self.validation_planner._is_test_path(candidate)
                    )
                ]
                ordered = [
                    *_order_by_attempt_status(support_candidates),
                    *_order_by_attempt_status(implementation_candidates),
                    *_order_by_attempt_status(validation_candidates),
                ]
                if ordered:
                    return ordered

        if not attempted_results:
            return candidates

        return _order_by_attempt_status(candidates)

    def _recent_runtime_support_repair_target(
        self,
        session: SessionState,
        repair_context: ValidationFailureEvidence,
    ) -> str | None:
        if repair_context.verification_scope != "runtime":
            return None
        for item in reversed(session.repair_history):
            if item.validation_command != repair_context.command:
                continue
            if item.verification_scope != repair_context.verification_scope:
                continue
            if item.result != "mutation_planned":
                return None
            candidate = str(item.artifact_path or "").strip()
            if not candidate:
                return None
            if self._is_runtime_support_repair_target(candidate, repair_context):
                return candidate
            return None
        return None

    def _is_runtime_support_repair_target(
        self,
        candidate: str,
        repair_context: ValidationFailureEvidence,
    ) -> bool:
        text = str(candidate or "").strip()
        if not text:
            return False
        path = Path(text)
        suffix = path.suffix.lower()
        if suffix in {".py", ".pyi"} and self._is_python_runtime_support_module(text, repair_context):
            return True
        if suffix == ".py" and self._is_direct_runtime_entrypoint_script_target(text, repair_context):
            return True
        if suffix == ".py":
            return False
        if self._repair_candidate_is_explicitly_referenced(text, repair_context):
            return True
        support_directories = {
            "data",
            "fixture",
            "fixtures",
            "sample",
            "samples",
            "sample-data",
            "sample_data",
            "test",
            "test-data",
            "test_data",
            "testdata",
            "tests",
        }
        return any(part.lower() in support_directories for part in path.parts[:-1])

    def _prioritized_repair_target_candidates(
        self,
        session: SessionState,
        repair_context: ValidationFailureEvidence,
    ) -> list[str]:
        candidates = self._prioritize_runtime_target_categories(
            self._unique_paths([*repair_context.artifact_paths, *repair_context.file_hints]),
            repair_context,
        )
        sticky_target = self._sticky_runtime_repair_target(session, repair_context)
        failure_specific_non_test = [
            candidate
            for candidate in candidates
            if (
                candidate
                and not self.validation_planner._is_test_path(candidate)
                and self._repair_candidate_matches_failure_text(candidate, repair_context)
            )
        ]
        failure_specific_test = [
            candidate
            for candidate in candidates
            if (
                candidate
                and self.validation_planner._is_test_path(candidate)
                and self._repair_candidate_matches_failure_text(candidate, repair_context)
            )
        ]
        explicitly_referenced = [
            candidate
            for candidate in candidates
            if self._repair_candidate_is_explicitly_referenced(candidate, repair_context)
        ]
        explicitly_referenced_missing = [
            candidate
            for candidate in explicitly_referenced
            if self._repair_target_can_be_created(session, candidate, repair_context)
        ]
        explicitly_referenced_existing = [
            candidate
            for candidate in explicitly_referenced
            if candidate not in explicitly_referenced_missing
        ]
        creatable_missing = [
            candidate
            for candidate in candidates
            if self._repair_target_can_be_created(session, candidate, repair_context)
        ]
        existing = self._repair_related_existing_context_paths(session, repair_context)
        support_candidates = self._unique_paths(
            [
                candidate
                for candidate in [*candidates, *explicitly_referenced_missing, *creatable_missing, *existing]
                if self._is_runtime_support_repair_target(candidate, repair_context)
            ]
        )
        support_should_lead = self._runtime_support_candidates_should_lead(
            support_candidates,
            repair_context,
        )
        failure_specific_support = [
            candidate
            for candidate in support_candidates
            if candidate and self._repair_candidate_matches_failure_text(candidate, repair_context)
        ]
        locked_target = self._repair_brief_locked_target(repair_context)
        primary_target = self._repair_brief_primary_target(repair_context)
        demote_locked = self._runtime_locked_target_should_yield(session, repair_context, locked_target)
        prioritized_locked_targets = [] if demote_locked else [locked_target, primary_target]
        if support_should_lead and support_candidates:
            prioritized_locked_targets = [
                candidate
                for candidate in prioritized_locked_targets
                if candidate not in support_candidates
            ]
        return self._unique_paths(
            [
                *(failure_specific_support if support_should_lead else []),
                *(support_candidates if support_should_lead else []),
                *prioritized_locked_targets,
                sticky_target,
                *self._memory_guided_candidate_paths(session, failure_only=True),
                *([] if support_should_lead else failure_specific_support),
                *([] if support_should_lead else support_candidates),
                *failure_specific_non_test,
                *explicitly_referenced_missing,
                *explicitly_referenced_existing,
                *creatable_missing,
                *candidates,
                *existing,
                *failure_specific_test,
                *([locked_target] if demote_locked and locked_target else []),
            ]
        )

    def _prioritize_runtime_target_categories(
        self,
        candidates: list[str],
        repair_context: ValidationFailureEvidence,
    ) -> list[str]:
        if repair_context.verification_scope != "runtime":
            return candidates
        documentation_like = [
            candidate
            for candidate in candidates
            if self.validation_planner._is_documentation_path(candidate)
        ]
        support_candidates = [
            candidate
            for candidate in candidates
            if self._is_runtime_support_repair_target(candidate, repair_context)
        ]
        implementation_like = [
            candidate
            for candidate in candidates
            if (
                candidate not in support_candidates
                and candidate not in documentation_like
                and not self.validation_planner._is_test_path(candidate)
            )
        ]
        validation_like = [
            candidate
            for candidate in candidates
            if self.validation_planner._is_test_path(candidate)
        ]
        if validation_like and self._runtime_failure_needs_test_context_first(repair_context):
            return [*validation_like, *implementation_like, *support_candidates, *documentation_like]
        if support_candidates and self._runtime_support_candidates_should_lead(
            support_candidates,
            repair_context,
        ):
            return [*support_candidates, *implementation_like, *validation_like, *documentation_like]
        if support_candidates:
            return [*implementation_like, *support_candidates, *validation_like, *documentation_like]
        return [*implementation_like, *validation_like, *documentation_like]

    def _runtime_failure_needs_test_context_first(
        self,
        repair_context: ValidationFailureEvidence,
    ) -> bool:
        if repair_context.verification_scope != "runtime":
            return False
        brief = getattr(repair_context, "repair_brief", None)
        if (
            brief is not None
            and str(getattr(brief, "failure_type", "") or "").strip() == "test_discovery_gap"
            and str(getattr(brief, "bootstrap_status", "") or "").strip() == "bootstrap_failed"
            and any(
                self.validation_planner._is_test_path(candidate)
                for candidate in [*repair_context.artifact_paths, *repair_context.file_hints]
            )
        ):
            return True
        failure_text = "\n".join(
            part
            for part in [
                str(repair_context.excerpt or "").strip(),
                str(repair_context.failure_summary or "").strip(),
                str(repair_context.summary or "").strip(),
            ]
            if part
        ).lower()
        if not failure_text:
            return False
        if re.search(r'file ".+?", line \d+', failure_text):
            return False
        if re.search(r"[\w./\\-]+\.py:\d+(?::\d+)?", failure_text):
            return False
        if any(
            marker in failure_text
            for marker in (
                "assertionerror",
                "typeerror",
                "valueerror",
                "runtimeerror",
                "nameerror",
                "attributeerror",
                "importerror",
                "modulenotfounderror",
                "syntaxerror",
                "indexerror",
                "keyerror",
                "filenotfounderror",
                "calledprocesserror",
                "systemexit",
                "traceback",
                "no such file or directory",
                " not found in ",
                " != ",
                "lists differ",
            )
        ):
            return False
        region_hint = str(getattr(brief, "implicated_region_hint", "") or "").strip().lower()
        target_markers: set[str] = set()
        for candidate in (
            str(getattr(brief, "locked_target", "") or "").strip(),
            str(getattr(brief, "primary_target", "") or "").strip(),
        ):
            target_markers.update(
                token
                for token in self._repair_candidate_reference_tokens(candidate)
                if token
            )
        region_hint_is_generic_target_label = bool(region_hint) and any(
            marker and marker in region_hint for marker in target_markers
        )
        if brief is not None and (
            brief.expected_semantics
            or brief.observed_semantics
            or brief.implicated_symbols
            or (region_hint and not region_hint_is_generic_target_label)
        ):
            return False
        if not any(
            self.validation_planner._is_test_path(candidate)
            for candidate in [*repair_context.artifact_paths, *repair_context.file_hints]
        ):
            return False
        return "fail" in failure_text or any(
            marker in failure_text
            for marker in (
                "no tests ran",
                "ran 0 tests",
                "collected 0 items",
                "test discovery",
                "loader._failed_test",
            )
        )

    def _runtime_support_candidates_should_lead(
        self,
        support_candidates: list[str],
        repair_context: ValidationFailureEvidence,
    ) -> bool:
        if repair_context.verification_scope != "runtime":
            return False
        failure_text = "\n".join(
            part
            for part in [
                str(repair_context.excerpt or "").strip(),
                str(repair_context.failure_summary or "").strip(),
                str(repair_context.summary or "").strip(),
            ]
            if part
        ).lower()
        failure_indicates_support_or_fixture_issue = any(
            marker in failure_text
            for marker in (
                "filenotfounderror",
                "no such file or directory",
                "fixture",
                "sample data",
                "sample text",
                "raw fixture",
                "test data",
                "test_data",
            )
        )
        if not failure_indicates_support_or_fixture_issue:
            return False
        explicit_support = [
            candidate
            for candidate in support_candidates
            if self._repair_candidate_is_explicitly_referenced(candidate, repair_context)
        ]
        if explicit_support:
            return True
        return len(support_candidates) == 1

    def _sticky_runtime_repair_target(
        self,
        session: SessionState,
        repair_context: ValidationFailureEvidence,
    ) -> str | None:
        if repair_context.verification_scope != "runtime":
            return None
        implicated = self._unique_paths([*repair_context.artifact_paths, *repair_context.file_hints])
        implicated_implementation = [
            candidate
            for candidate in implicated
            if candidate and not self.validation_planner._is_test_path(candidate)
        ]
        require_current_implication = bool(implicated_implementation)
        for item in reversed(session.repair_history):
            if item.validation_command != repair_context.command:
                continue
            if item.verification_scope != repair_context.verification_scope:
                continue
            if item.result != "mutation_planned":
                continue
            candidate = str(item.artifact_path or "").strip()
            if not candidate:
                continue
            if self.validation_planner._is_test_path(candidate):
                continue
            if self._runtime_locked_target_should_yield(session, repair_context, candidate):
                continue
            if require_current_implication and candidate not in implicated:
                continue
            return candidate
        return None

    def _is_python_runtime_support_module(
        self,
        candidate: str,
        repair_context: ValidationFailureEvidence,
    ) -> bool:
        text = str(candidate or "").strip()
        if not text:
            return False
        path = Path(text)
        name = path.name.lower()
        if name == "__init__.py":
            return True
        if name != "__main__.py":
            return False
        implementation_peers = self._runtime_implementation_candidates(
            repair_context,
            exclude={text},
        )
        return bool(implementation_peers)

    def _is_direct_runtime_entrypoint_script_target(
        self,
        candidate: str,
        repair_context: ValidationFailureEvidence,
    ) -> bool:
        text = str(candidate or "").strip()
        if repair_context.verification_scope != "runtime" or not text:
            return False
        target = text.replace("\\", "/").lower()
        failure_text = "\n".join(
            part
            for part in [
                str(repair_context.excerpt or "").strip(),
                str(repair_context.failure_summary or "").strip(),
                str(repair_context.summary or "").strip(),
            ]
            if part
        )
        script_target = self.validation_planner._called_process_python_script_target(failure_text)
        if not script_target:
            return False
        normalized_script = str(script_target or "").strip().replace("\\", "/").lower()
        if not normalized_script:
            return False
        if not (target == normalized_script or normalized_script.endswith(f"/{target}") or target.endswith(f"/{Path(normalized_script).name}")):
            return False
        path = Path(text)
        scriptish_directories = {"script", "scripts", "bin", "tool", "tools"}
        if not any(part.lower() in scriptish_directories for part in path.parts[:-1]):
            return False
        implementation_peers = self._runtime_implementation_candidates(
            repair_context,
            exclude={text},
        )
        return bool(implementation_peers)

    def _runtime_locked_target_should_yield(
        self,
        session: SessionState,
        repair_context: ValidationFailureEvidence | None,
        locked_target: str | None,
    ) -> bool:
        target = str(locked_target or "").strip()
        if repair_context is None or repair_context.verification_scope != "runtime" or not target:
            return False
        if not self._is_runtime_support_repair_target(target, repair_context):
            return False
        if self._runtime_support_candidates_should_lead([target], repair_context):
            return False
        failure_text = "\n".join(
            part
            for part in [
                str(getattr(getattr(repair_context, "repair_brief", None), "failure_type", "") or "").strip(),
                str(repair_context.failure_summary or "").strip(),
                str(repair_context.excerpt or "").strip(),
            ]
            if part
        ).lower()
        behavioral_failure = any(
            marker in failure_text
            for marker in (
                "assertionerror",
                "assertion mismatch",
                "lists differ",
                "expected",
                "observed",
                "output",
                "returned",
                "prints",
            )
        )
        brief = getattr(repair_context, "repair_brief", None)
        if brief is not None and (brief.expected_semantics or brief.observed_semantics):
            behavioral_failure = True
        implementation_alternatives = self._runtime_implementation_candidates(
            repair_context,
            exclude={target},
        )
        allowed_files = {
            str(item or "").strip()
            for item in getattr(brief, "allowed_files", [])
            if str(item or "").strip()
        }
        if allowed_files:
            implementation_alternatives = [
                candidate for candidate in implementation_alternatives if candidate in allowed_files
            ]
        if not implementation_alternatives:
            return False
        target_markers = {
            token
            for token in self._repair_candidate_reference_tokens(target)
            if token
        }
        region_hint = str(getattr(brief, "implicated_region_hint", "") or "").strip().lower()
        evidence_points_away_from_support_target = bool(
            self._is_python_runtime_support_module(target, repair_context)
            and implementation_alternatives
            and (
                any(str(symbol or "").strip() for symbol in getattr(brief, "implicated_symbols", []))
                or region_hint
            )
            and not any(marker and marker in region_hint for marker in target_markers)
        )
        if not behavioral_failure and not evidence_points_away_from_support_target:
            return False
        attempted_failures = self._repair_attempt_failure_count(session, repair_context, target)
        strongly_implicated_alternatives = [
            candidate
            for candidate in implementation_alternatives
            if (
                self._repair_candidate_is_strongly_referenced(candidate, repair_context)
                or self._repair_candidate_matches_failure_text(candidate, repair_context)
            )
        ]
        if attempted_failures >= 1:
            return True
        if evidence_points_away_from_support_target:
            return True
        if (
            self._is_python_runtime_support_module(target, repair_context)
            and implementation_alternatives
            and any(
                item.result == "mutation_planned"
                and str(item.artifact_path or "").strip() == target
                and str(item.validation_command or "").strip() == str(repair_context.command or "").strip()
                and str(item.verification_scope or "").strip() == str(repair_context.verification_scope or "").strip()
                and (
                    (
                        self._repair_failure_signature(repair_context)
                        and str(item.failure_signature or "").strip() == self._repair_failure_signature(repair_context)
                    )
                    or (
                        not self._repair_failure_signature(repair_context)
                        and str(item.evidence_signature or "").strip() == str(repair_context.evidence_signature or "").strip()
                    )
                )
                for item in session.repair_history
            )
        ):
            return True
        target_is_directly_implicated = self._repair_candidate_is_explicitly_referenced(
            target,
            repair_context,
        )
        return bool(strongly_implicated_alternatives) and not target_is_directly_implicated

    def _runtime_implementation_candidates(
        self,
        repair_context: ValidationFailureEvidence,
        *,
        exclude: set[str] | None = None,
    ) -> list[str]:
        skipped = {
            str(item or "").strip()
            for item in list(exclude or set())
            if str(item or "").strip()
        }
        candidates = self._unique_paths([*repair_context.artifact_paths, *repair_context.file_hints])
        return [
            candidate
            for candidate in candidates
            if candidate
            and candidate not in skipped
            and not self.validation_planner._is_test_path(candidate)
            and not self.validation_planner._is_documentation_path(candidate)
            and not self._is_runtime_support_repair_target(candidate, repair_context)
        ]

    def _repair_candidate_is_explicitly_referenced(
        self,
        candidate: str,
        repair_context: ValidationFailureEvidence,
    ) -> bool:
        reference_tokens = self._repair_candidate_reference_tokens(candidate)
        if not reference_tokens:
            return False
        evidence_texts = [
            str(repair_context.failure_summary or "").strip(),
            str(repair_context.summary or "").strip(),
            str(repair_context.excerpt or "").strip(),
            *(str(item or "").strip() for item in repair_context.repair_requirements),
            *(str(item or "").strip() for item in repair_context.action_hints),
        ]
        return self._repair_texts_reference_candidate(reference_tokens, evidence_texts)

    def _repair_candidate_matches_failure_text(
        self,
        candidate: str,
        repair_context: ValidationFailureEvidence,
    ) -> bool:
        brief = getattr(repair_context, "repair_brief", None)
        locked_target = str(getattr(brief, "locked_target", "") or "").strip()
        primary_target = str(getattr(brief, "primary_target", "") or "").strip()
        if candidate and candidate in {locked_target, primary_target}:
            return True
        reference_tokens = self._repair_candidate_reference_tokens(candidate)
        if not reference_tokens:
            return False
        failure_texts = [
            str(repair_context.failure_summary or "").strip(),
            str(repair_context.summary or "").strip(),
            str(repair_context.excerpt or "").strip(),
        ]
        return self._repair_texts_reference_candidate(reference_tokens, failure_texts)

    def _repair_candidate_is_strongly_referenced(
        self,
        candidate: str,
        repair_context: ValidationFailureEvidence,
    ) -> bool:
        text = str(candidate or "").strip()
        if not text:
            return False
        path = Path(text)
        dotted = text.replace("/", ".").removesuffix(".py")
        reference_tokens = self._unique_paths(
            [
                text.lower(),
                path.name.lower(),
                dotted.lower(),
            ]
        )
        evidence_texts = [
            str(repair_context.failure_summary or "").strip().lower(),
            str(repair_context.summary or "").strip().lower(),
            str(repair_context.excerpt or "").strip().lower(),
            *(str(item or "").strip().lower() for item in repair_context.repair_requirements),
            *(str(item or "").strip().lower() for item in repair_context.action_hints),
        ]
        return any(
            token
            and re.search(
                rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])",
                evidence_text,
            )
            for evidence_text in evidence_texts
            for token in reference_tokens
        )

    def _repair_candidate_reference_tokens(self, candidate: str) -> list[str]:
        text = str(candidate or "").strip()
        if not text:
            return []
        path = Path(text)
        basename = path.name
        stem = path.stem
        dotted = text.replace("/", ".").removesuffix(".py")
        allow_stem_token = (
            path.suffix.lower() not in {".html", ".htm"}
            and self._repair_reference_stem_is_specific(path)
        )
        return [
            token.lower()
            for token in (
                text,
                basename,
                stem if allow_stem_token else None,
                dotted,
            )
            if str(token or "").strip()
        ]

    def _repair_reference_stem_is_specific(self, path: Path) -> bool:
        stem = re.sub(r"[^0-9a-z]+", "", path.stem.lower())
        if len(stem) < 4:
            return False
        return stem not in {
            "app",
            "apps",
            "cli",
            "doc",
            "docs",
            "file",
            "files",
            "helper",
            "helpers",
            "index",
            "main",
            "module",
            "readme",
            "test",
            "tests",
            "tool",
            "tools",
            "util",
            "utils",
        }

    def _repair_texts_reference_candidate(
        self,
        reference_tokens: list[str],
        evidence_texts: list[str],
    ) -> bool:
        lowered_texts = [str(item or "").strip().lower() for item in evidence_texts if str(item or "").strip()]
        return any(
            any(
                re.search(
                    rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])",
                    text,
                )
                for token in reference_tokens
                if token
            )
            for text in lowered_texts
        )

    def _repair_related_existing_context_paths(
        self,
        session: SessionState,
        repair_context: ValidationFailureEvidence,
        *,
        exclude: str | None = None,
    ) -> list[str]:
        existing: list[str] = []
        for candidate in self._unique_paths([*repair_context.file_hints, *repair_context.artifact_paths]):
            if exclude is not None and candidate == exclude:
                continue
            if not self._is_safe_workspace_target_candidate(session, candidate):
                continue
            absolute = Path(session.workspace_root, candidate)
            if absolute.exists() and absolute.is_file():
                existing.append(candidate)
        return existing

    def _repair_target_can_be_created(
        self,
        session: SessionState,
        candidate: str,
        repair_context: ValidationFailureEvidence | None,
    ) -> bool:
        text = str(candidate or "").strip()
        if not text or repair_context is None:
            return False
        if not self._is_safe_workspace_target_candidate(session, text):
            return False
        if text not in repair_context.artifact_paths and text not in repair_context.file_hints:
            return False
        absolute = Path(session.workspace_root, text)
        if absolute.exists() or absolute.is_dir():
            return False
        return bool(
            self._repair_related_existing_context_paths(
                session,
                repair_context,
                exclude=text,
            )
        )

    def _is_safe_workspace_target_candidate(self, session: SessionState, candidate: str) -> bool:
        text = str(candidate or "").strip()
        if not text or (text.startswith("<") and text.endswith(">")):
            return False
        path = Path(text)
        absolute = path if path.is_absolute() else Path(session.workspace_root, path)
        try:
            absolute.resolve(strict=False).relative_to(Path(session.workspace_root).resolve())
        except ValueError:
            return False
        return True

    def _paths_from_internal_validation_command(self, command: str) -> list[str]:
        text = str(command or "").strip()
        if not text.startswith("internal:"):
            return []
        _, _, payload = text.partition(":")
        _, _, raw_json = payload.partition(":")
        try:
            data = json.loads(raw_json or "[]")
        except Exception:
            return []

        values = data if isinstance(data, list) else [data]
        paths: list[str] = []
        for item in values:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    paths.append(cleaned)
            elif isinstance(item, dict):
                cleaned = str(item.get("path") or "").strip()
                if cleaned:
                    paths.append(cleaned)
        return self._unique_paths(paths)

    def _validation_repair_blocker_message(
        self,
        failed_run,
        repair_context: ValidationFailureEvidence | None = None,
        *,
        repair_target_missing: bool = False,
    ) -> str:
        command = str(getattr(failed_run, "command", "") or "").strip() or "the last validation command"
        if repair_target_missing:
            suffix = (
                f" Known failure evidence: {repair_context.failure_summary}"
                if repair_context is not None and repair_context.failure_summary
                else ""
            )
            return (
                f"Validation failed for {command}, and I do not have a repairable target that I can safely mutate.{suffix}"
            )
        if repair_context is not None and repair_context.failure_summary:
            return (
                f"Validation failed for {command}. The current repair evidence still does not support a substantive fix: "
                f"{repair_context.failure_summary}"
            )
        return (
            f"Validation failed for {command}, and I do not have a repairable target or new evidence that would justify rerunning the same check."
        )

    def _validation_repair_blocked_response(
        self,
        route: RouterOutput,
        session: SessionState,
        failed_run,
        repair_context: ValidationFailureEvidence | None = None,
        *,
        repair_target_missing: bool = False,
    ) -> str:
        language = self._session_language(session)
        command = str(getattr(failed_run, "command", "") or "").strip() or "the last validation command"
        target = self._repair_target_after_failed_validation(
            route,
            session,
            failed_run,
            repair_context,
        )
        failure_summary = (
            str(repair_context.failure_summary or "").strip()
            if repair_context is not None
            else ""
        )
        if language == "de":
            lines = [
                "Ich habe den fehlgeschlagenen Validierungsschritt als Reparaturausloeser behandelt und keinen sinnlosen Verify-Loop gestartet.",
                f"Fehlgeschlagener Check: {command}.",
            ]
            if target:
                lines.append(f"Betroffenes Artefakt: {target}.")
            if failure_summary:
                lines.append(f"Bekannte Fehler-Evidenz: {failure_summary}")
            if repair_target_missing:
                lines.append("Mir fehlt aber ein reparierbares Artefaktziel fuer einen sicheren weiteren Schritt.")
            else:
                lines.append("Mir fehlt aber noch eine belastbare, nicht-aequivalente Reparatur, die diesen Fehler wirklich adressiert.")
            return "\n".join(lines)
        lines = [
            "I treated the failed validation as a repair trigger and did not start a useless verify loop.",
            f"Failed check: {command}.",
        ]
        if target:
            lines.append(f"Affected artifact: {target}.")
        if failure_summary:
            lines.append(f"Known failure evidence: {failure_summary}")
        if repair_target_missing:
            lines.append("I still lack a repairable artifact target for a safe next step.")
        else:
            lines.append("I still lack a substantive, non-equivalent repair that would address this failure safely.")
        return "\n".join(lines)

    def _bootstrap_reset_blocker_message(
        self,
        repair_context: ValidationFailureEvidence | None,
    ) -> str:
        root_cause = self._repair_context_root_cause_summary(repair_context)
        summary = str(getattr(repair_context, "failure_summary", "") or "").strip() if repair_context is not None else ""
        if root_cause and summary and summary != root_cause:
            return f"Bootstrap reset required: {root_cause} Latest failure evidence: {summary}"
        if root_cause:
            return f"Bootstrap reset required: {root_cause}"
        if summary:
            return f"Bootstrap reset required: {summary}"
        return "Bootstrap reset required before another repair loop can continue."

    def _bootstrap_reset_blocked_response(
        self,
        route: RouterOutput,
        session: SessionState,
        repair_context: ValidationFailureEvidence | None,
    ) -> str:
        del route
        language = self._session_language(session)
        root_cause = self._repair_context_root_cause_summary(repair_context)
        summary = str(getattr(repair_context, "failure_summary", "") or "").strip() if repair_context is not None else ""
        target = self._repair_brief_locked_target(repair_context) or self._repair_brief_primary_target(repair_context)
        if language == "de":
            lines = [
                "Ich stoppe den Repair-Loop, weil der Bootstrap-Pfad trotz wiederholter Versuche dasselbe Fehlerbild liefert.",
                "Status: bootstrap_reset_required.",
            ]
            if target:
                lines.append(f"Betroffenes Artefakt: {target}.")
            if root_cause:
                lines.append(f"Root Cause: {root_cause}")
            if summary and summary != root_cause:
                lines.append(f"Letzte Evidenz: {summary}")
            lines.append("Ein weiterer Retry ohne Reset, neue Evidenz oder menschliche Entscheidung wuerde nur dieselbe Schleife wiederholen.")
            return "\n".join(lines)
        lines = [
            "I am stopping the repair loop because the bootstrap path is still producing the same failure after repeated attempts.",
            "Status: bootstrap_reset_required.",
        ]
        if target:
            lines.append(f"Affected artifact: {target}.")
        if root_cause:
            lines.append(f"Root cause: {root_cause}")
        if summary and summary != root_cause:
            lines.append(f"Latest evidence: {summary}")
        lines.append("Another retry without a reset, new evidence, or human intervention would only repeat the same loop.")
        return "\n".join(lines)

    def _validation_decision(
        self,
        thought_summary: str,
        command: str,
        *,
        expected_stdout: str | None = None,
    ) -> AgentDecision:
        return AgentDecision(
            thought_summary=thought_summary,
            action_type=AgentActionType.CALL_TOOL,
            tool_name="run_tests",
            tool_args={
                "command": command,
                "cwd": ".",
                "timeout": 120,
                **({"expected_stdout": expected_stdout} if expected_stdout is not None else {}),
            },
            expected_outcome="Run the next validation step for the current changes.",
            final_response=None,
        )

    def _diagnostic_command_spec(
        self,
        session: SessionState,
        command: str,
    ) -> ValidationCommand | None:
        identity = self.validation_planner.command_identity(command)
        if not identity:
            return None
        for item in session.validation_plan:
            if self.validation_planner.command_identity(item.command) == identity:
                return item
        snapshot = session.workspace_snapshot
        if snapshot is not None:
            validation_plan = self.validation_planner.build_plan(
                session.task,
                snapshot,
                changed_files=[],
                session=session,
            )
            for item in validation_plan:
                if self.validation_planner.command_identity(item.command) == identity:
                    return item
        for item in self.validation_planner.build_diagnostic_plan(session):
            if self.validation_planner.command_identity(item.command) == identity:
                return item
        return None

    def _pick_validation_command(self, session: SessionState) -> ValidationCommand | None:
        passed = {
            self.validation_planner.command_identity(run.command)
            for run in session.validation_runs
            if run.edit_generation == session.edit_generation and run.status == "passed"
        }
        for item in session.validation_plan:
            identity = self.validation_planner.command_identity(item.command)
            if identity not in passed and self.validation_planner.can_repeat_command(session, item.command):
                return item
        for command in session.verification_commands:
            identity = self.validation_planner.command_identity(command)
            if command and identity not in passed and self.validation_planner.can_repeat_command(session, command):
                return ValidationCommand(command=command)
        snapshot = session.workspace_snapshot
        if snapshot is None:
            return None
        fallback_plan = self.validation_planner.build_plan(
            session.task,
            snapshot,
            changed_files=[item.path for item in session.changed_files],
            session=session,
        )
        for item in fallback_plan:
            identity = self.validation_planner.command_identity(item.command)
            if identity not in passed and self.validation_planner.can_repeat_command(session, item.command):
                return item
        for command in snapshot.likely_commands:
            identity = self.validation_planner.command_identity(command)
            if command and identity not in passed and self.validation_planner.can_repeat_command(session, command):
                return ValidationCommand(command=command)
        return None

    def _diagnostic_file_candidates(
        self,
        route: RouterOutput,
        session: SessionState,
        candidate_paths: list[str],
    ) -> list[str]:
        candidates: list[str] = []
        diagnostics = list(session.diagnostics[-8:])
        if session.follow_up_context is not None:
            diagnostics.extend(session.follow_up_context.diagnostics[-8:])
        for item in diagnostics:
            candidates.extend(item.file_hints)
        candidates.extend(route.entities.target_paths)
        if session.follow_up_context is not None:
            candidates.extend(session.follow_up_context.target_paths)
            candidates.extend(session.follow_up_context.changed_files)
            candidates.extend(session.follow_up_context.read_files)
        if candidates:
            return self._unique_paths(candidates)
        candidates.extend(candidate_paths)
        return self._unique_paths(candidates)

    def _diagnostic_evidence_available(self, session: SessionState) -> bool:
        if session.diagnostics:
            return True
        return any(
            item.tool_name in {"run_tests", "run_shell"} and not item.success
            for item in session.tool_calls
        )

    def _next_diagnostic_command(self, session: SessionState) -> str | None:
        def pick_failed(runs) -> str | None:
            for run in reversed(list(runs)):
                command = str(getattr(run, "command", "") or "").strip()
                status = str(getattr(run, "status", "") or "").strip()
                if command and status in {"failed", "timeout", "blocked"}:
                    return command
            return None

        command = pick_failed(session.validation_runs)
        if command:
            return command
        if session.follow_up_context is not None:
            command = pick_failed(session.follow_up_context.validation_runs)
            if command:
                return command
        snapshot = session.workspace_snapshot
        if snapshot is not None:
            validation_plan = self.validation_planner.build_plan(
                session.task,
                snapshot,
                changed_files=[],
                session=session,
            )
            fallback_runtime: str | None = None
            for item in validation_plan:
                command = str(item.command or "").strip()
                if not command or item.verification_scope != "runtime":
                    continue
                if not command.startswith("internal:"):
                    return command
                fallback_runtime = fallback_runtime or command
            if fallback_runtime:
                return fallback_runtime
        for item in self.validation_planner.build_diagnostic_plan(session):
            command = str(item.command or "").strip()
            if command:
                return command
        if session.follow_up_context is not None:
            for item in reversed(session.follow_up_context.recent_commands):
                command = str(item or "").strip()
                if command and self.validation_planner.command_scope(command) == "runtime":
                    return command
        if snapshot is None:
            return None
        for item in snapshot.likely_commands:
            command = str(item or "").strip()
            if command and self.validation_planner.command_scope(command) == "runtime":
                return command
        return None

    def _command_already_ran(self, session: SessionState, command: str) -> bool:
        normalized = str(command or "").strip()
        if not normalized:
            return False
        return any(
            item.tool_name in {"run_tests", "run_shell"}
            and str(item.tool_args.get("command") or "").strip() == normalized
            for item in session.tool_calls
        )

    def _diagnosis_attempted_without_findings(self, session: SessionState) -> bool:
        successful_runtime_checks = {
            run.command
            for run in session.validation_runs
            if run.verification_scope == "runtime" and run.status == "passed"
        }
        return bool(successful_runtime_checks) and not session.diagnostics

    def _missing_issue_evidence_response(self, route: RouterOutput, session: SessionState) -> str:
        follow_up = session.follow_up_context
        previous_task = str(follow_up.previous_task or "").strip() if follow_up else ""
        lines = [
            "Ich habe den zuletzt relevanten Kontext rekonstruiert und zuerst diagnostisch statt blind editiert.",
        ]
        if previous_task:
            lines.append(f"Vorheriger Task: {previous_task}")
        if session.validation_runs:
            lines.append(f"Gepruefter Befehl: {session.validation_runs[-1].command}")
        elif follow_up and follow_up.recent_commands:
            lines.append(f"Gepruefter Befehl: {follow_up.recent_commands[-1]}")
        lines.extend(
            [
                "",
                "Mir fehlt noch ein konkreter Hinweis, bevor ich sicher fixen kann:",
                "- Welche Ausgabe im Terminal wirkt falsch oder kaputt?",
                "- Wenn moeglich: die letzte Fehlermeldung oder 2-3 relevante Zeilen aus dem Terminal schicken.",
            ]
        )
        if route.entities.target_paths:
            lines.append(f"- Falls du es schon eingrenzen kannst: betrifft es {route.entities.target_paths[0]}?")
        return "\n".join(lines)

    def _candidate_paths(self, route: RouterOutput, session: SessionState) -> list[str]:
        candidates: list[str] = []
        candidates.extend(route.entities.target_paths)
        target_name_path = self._target_name_path_candidate(route)
        if target_name_path is not None:
            candidates.append(target_name_path)
        candidates.extend(self._memory_guided_candidate_paths(session))
        candidates.extend(session.candidate_files)
        if session.follow_up_context is not None:
            candidates.extend(session.follow_up_context.target_paths)
            candidates.extend(session.follow_up_context.changed_files)
            candidates.extend(session.follow_up_context.read_files)
            for item in session.follow_up_context.diagnostics[-6:]:
                candidates.extend(item.file_hints)
        if session.workspace_snapshot is not None:
            candidates.extend(session.workspace_snapshot.focus_files)
            candidates.extend(session.workspace_snapshot.important_files[:12])
        return self._prioritize_request_entrypoint_targets(
            session,
            self._unique_paths(candidates),
        )

    def _read_candidates(
        self,
        route: RouterOutput,
        session: SessionState,
        candidate_paths: list[str],
    ) -> list[str]:
        repair_context = self._current_repair_context_hint(session)
        if repair_context is not None and route.intent in {
            RouteIntent.UPDATE,
            RouteIntent.DEBUG,
            RouteIntent.DELETE,
            RouteIntent.EXPLAIN,
        }:
            repair_targets = [
                path for path in route.entities.target_paths if path and Path(path).suffix
            ]
            if repair_targets:
                return self._unique_paths(repair_targets)
        explicit_targets = self._explicit_target_paths(route, session)
        if explicit_targets and route.intent in {
            RouteIntent.UPDATE,
            RouteIntent.DEBUG,
            RouteIntent.DELETE,
            RouteIntent.EXPLAIN,
        }:
            return explicit_targets
        return candidate_paths

    def _primary_target_path(self, route: RouterOutput, session: SessionState) -> str | None:
        candidates = self._candidate_paths(route, session)
        return candidates[0] if candidates else None

    def _explicit_target_paths(
        self,
        route: RouterOutput,
        session: SessionState | None = None,
    ) -> list[str]:
        candidates = [path for path in route.entities.target_paths if path and Path(path).suffix]
        target_name_path = self._target_name_path_candidate(route)
        if target_name_path is not None and Path(target_name_path).suffix:
            candidates.append(target_name_path)
        if session is not None:
            candidates.extend(self._snapshot_explicit_target_paths(session))
        unique = self._unique_paths(candidates)
        if session is None:
            return unique
        return self._prioritize_request_entrypoint_targets(session, unique)

    def _ordered_create_targets(
        self,
        route: RouterOutput,
        session: SessionState,
    ) -> list[str]:
        explicit = self._actionable_explicit_target_paths(route, session)
        if route.intent != RouteIntent.CREATE or not explicit:
            return explicit

        artifact_roles = self._explicit_create_target_roles(route, session, explicit_targets=explicit)
        role_order = {
            "primary_target": 0,
            "active_context": 0,
            "validation_target": 1,
            "supporting_context": 2,
        }
        indexed_positions = {path: index for index, path in enumerate(explicit)}
        return sorted(
            explicit,
            key=lambda path: (
                role_order.get(artifact_roles.get(path, "primary_target"), 3),
                indexed_positions.get(path, 999),
            ),
        )

    def _explicit_create_target_roles(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        explicit_targets: list[str] | None = None,
    ) -> dict[str, str]:
        explicit = explicit_targets or self._explicit_target_paths(route, session)
        if route.intent != RouteIntent.CREATE or not explicit:
            return {}

        task_state = session.task_state
        artifact_roles: dict[str, str] = {}
        if task_state is not None:
            artifact_roles = {
                path: str(artifact.role or "").strip().lower()
                for artifact in task_state.target_artifacts
                for path in [str(artifact.path or "").strip()]
                if path
            }
        routed_primary_target = self._target_name_path_candidate(route)
        if not routed_primary_target and explicit:
            routed_primary_target = explicit[0]

        def inferred_role(path: str) -> str:
            explicit_role = artifact_roles.get(path, "")
            is_routed_primary_target = path == routed_primary_target
            if self._path_is_test_like(path) and not is_routed_primary_target:
                return "validation_target"
            suffix = Path(path).suffix.lower()
            if suffix in {".md", ".markdown", ".rst", ".txt"} and not is_routed_primary_target:
                return "supporting_context"
            if explicit_role in {
                "primary_target",
                "validation_target",
                "supporting_context",
                "active_context",
            }:
                return explicit_role
            return "primary_target"

        return {
            path: inferred_role(path)
            for path in explicit
        }

    def _update_candidate_roles(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        candidate_paths: list[str] | None = None,
    ) -> dict[str, str]:
        candidates = candidate_paths or self._actionable_explicit_target_paths(route, session)
        if route.intent != RouteIntent.UPDATE or not candidates:
            return {}

        task_state = session.task_state
        artifact_roles: dict[str, str] = {}
        if task_state is not None:
            artifact_roles = {
                path: str(artifact.role or "").strip().lower()
                for artifact in task_state.target_artifacts
                for path in [str(artifact.path or "").strip()]
                if path
            }
        routed_primary_target = self._target_name_path_candidate(route)
        if not routed_primary_target and candidates:
            routed_primary_target = candidates[0]

        def inferred_role(path: str) -> str:
            explicit_role = artifact_roles.get(path, "")
            is_routed_primary_target = path == routed_primary_target
            if self._path_is_test_like(path) and not is_routed_primary_target:
                return "validation_target"
            suffix = Path(path).suffix.lower()
            if suffix in {".md", ".markdown", ".rst", ".txt"} and not is_routed_primary_target:
                return "supporting_context"
            if explicit_role in {
                "primary_target",
                "validation_target",
                "supporting_context",
                "active_context",
            }:
                return explicit_role
            return "primary_target"

        return {
            path: inferred_role(path)
            for path in candidates
        }

    def _ordered_remaining_update_targets(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        exclude: set[str] | None = None,
    ) -> list[str]:
        explicit_targets = self._actionable_explicit_target_paths(route, session)
        if not explicit_targets:
            return []

        excluded = {item for item in (exclude or set()) if item}
        changed_paths = {item.path for item in session.changed_files}
        deferred_targets = self._active_deferred_update_targets(session)
        explicit_roles = self._update_candidate_roles(
            route,
            session,
            candidate_paths=explicit_targets,
        )
        role_order = {
            "primary_target": 0,
            "active_context": 0,
            "validation_target": 1,
            "supporting_context": 2,
        }
        explicit_index = {path: index for index, path in enumerate(explicit_targets)}
        remaining_explicit = sorted(
            [
                candidate
                for candidate in explicit_targets
                if candidate not in changed_paths
                and candidate not in deferred_targets
                and candidate not in excluded
            ],
            key=lambda candidate: (
                role_order.get(explicit_roles.get(candidate, "primary_target"), 3),
                explicit_index.get(candidate, 999),
            ),
        )
        changed_explicit_technical_targets = {
            candidate
            for candidate in explicit_targets
            if candidate in changed_paths
            and explicit_roles.get(candidate) not in {"supporting_context", "validation_target"}
        }
        if remaining_explicit and any(
            explicit_roles.get(candidate) not in {"supporting_context", "validation_target"}
            for candidate in remaining_explicit
        ):
            return remaining_explicit
        if remaining_explicit and changed_explicit_technical_targets:
            return remaining_explicit

        routed_primary_target = self._target_name_path_candidate(route)
        routed_primary_suffix = Path(routed_primary_target).suffix.lower() if routed_primary_target else ""
        routed_primary_is_documentation = routed_primary_suffix in {".md", ".markdown", ".rst", ".txt"}
        if routed_primary_is_documentation:
            return remaining_explicit

        grounded_candidates = self._grounded_update_candidate_paths(session)
        broader_candidates = [
            candidate
            for candidate in grounded_candidates
            if candidate not in explicit_targets
            and candidate not in changed_paths
            and candidate not in deferred_targets
            and candidate not in excluded
            and self._current_file_content(session, candidate) is not None
        ]
        broader_roles = self._update_candidate_roles(
            route,
            session,
            candidate_paths=broader_candidates,
        )
        technical_candidates = [
            candidate
            for candidate in broader_candidates
            if broader_roles.get(candidate) not in {"supporting_context", "validation_target"}
        ]
        return [*technical_candidates, *remaining_explicit]

    def _grounded_update_candidate_paths(self, session: SessionState) -> list[str]:
        candidates: list[str] = []
        task_state = session.task_state
        if task_state is not None:
            candidates.extend(
                str(getattr(artifact, "path", "") or "").strip()
                for artifact in task_state.target_artifacts
                if str(getattr(artifact, "path", "") or "").strip()
            )
        candidates.extend(session.candidate_files)
        if session.follow_up_context is not None:
            candidates.extend(session.follow_up_context.target_paths)
            candidates.extend(session.follow_up_context.changed_files)
            candidates.extend(session.follow_up_context.read_files)
            for item in session.follow_up_context.diagnostics[-6:]:
                candidates.extend(item.file_hints)
        candidates.extend(self._snapshot_match_candidate_paths(session.workspace_snapshot))
        return self._unique_paths(candidates)

    def _active_deferred_create_targets(
        self,
        route: RouterOutput,
        session: SessionState,
    ) -> set[str]:
        explicit_targets = self._ordered_create_targets(route, session)
        if route.intent != RouteIntent.CREATE or not explicit_targets:
            return set()

        artifact_roles = self._explicit_create_target_roles(route, session, explicit_targets=explicit_targets)
        supporting_targets = {
            path
            for path in explicit_targets
            if artifact_roles.get(path) == "supporting_context"
        }
        if not supporting_targets:
            return set()
        if not any(artifact_roles.get(path) != "supporting_context" for path in explicit_targets):
            return set()
        if not self.validation_planner.pending_commands(session):
            return set()
        if any(run.status == "passed" for run in session.validation_runs):
            return set()
        return supporting_targets

    def _actionable_explicit_target_paths(
        self,
        route: RouterOutput,
        session: SessionState,
    ) -> list[str]:
        explicit = self._explicit_target_paths(route, session)
        if not explicit:
            return explicit

        task_state = session.task_state
        if task_state is None:
            return explicit

        artifact_roles = {
            path: str(artifact.role or "").strip().lower()
            for artifact in task_state.target_artifacts
            for path in [str(artifact.path or "").strip()]
            if path
        }
        if not artifact_roles:
            return explicit

        has_non_validation_target = any(
            artifact_roles.get(path) != "validation_target"
            for path in explicit
        )
        if not has_non_validation_target:
            return explicit

        filtered = [
            path
            for path in explicit
            if artifact_roles.get(path) != "validation_target"
            or self._path_matches_explicit_request(session, path)
        ]
        return filtered or explicit

    def _snapshot_explicit_target_paths(self, session: SessionState) -> list[str]:
        snapshot = session.workspace_snapshot
        if snapshot is None:
            return []

        request_lower = self._request_text_for_explicit_target_matching(session)
        if not request_lower.strip():
            return []

        candidate_paths = self._snapshot_match_candidate_paths(snapshot)

        generic_stems = {"app", "cli", "doc", "docs", "guide", "index", "main", "test", "tests"}
        explicit: list[str] = []
        for path in candidate_paths:
            if self._path_matches_explicit_request(session, path, generic_stems=generic_stems):
                explicit.append(path)

        return self._unique_paths(explicit)

    def _generic_support_basename(self, path: str) -> bool:
        name = Path(str(path or "").strip()).name.lower()
        return name in {"__init__.py", "__main__.py", "conftest.py"}

    def _snapshot_match_candidate_paths(self, snapshot) -> list[str]:
        candidate_paths: list[str] = []
        if snapshot is None:
            return candidate_paths
        for group in (
            getattr(snapshot, "focus_files", []),
            getattr(snapshot, "important_files", []),
            getattr(snapshot, "manifests", []),
            getattr(snapshot, "test_files", []),
            getattr(snapshot, "entrypoints", []),
        ):
            for item in group[:12]:
                path = str(item or "").strip()
                if path and path not in candidate_paths:
                    candidate_paths.append(path)
        return candidate_paths

    def _basename_is_ambiguous_in_snapshot(self, session: SessionState, path: str) -> bool:
        basename = Path(str(path or "").strip()).name.lower()
        if not basename:
            return False
        matches = 0
        for candidate in self._snapshot_match_candidate_paths(session.workspace_snapshot):
            if Path(candidate).name.lower() != basename:
                continue
            matches += 1
            if matches > 1:
                return True
        return False

    def _should_require_contextual_path_match(self, session: SessionState, path: str) -> bool:
        return self._generic_support_basename(path) or self._basename_is_ambiguous_in_snapshot(session, path)

    def _semantic_review_pending_snapshot_targets(
        self,
        session: SessionState,
        *,
        deferred_targets: set[str],
    ) -> list[str]:
        pending_targets = [
            path
            for path in self._snapshot_explicit_target_paths(session)
            if path not in {item.path for item in session.changed_files}
            and path not in deferred_targets
        ]
        if not pending_targets:
            return []

        task_state = session.task_state
        current_intent = str(getattr(task_state, "current_user_intent", "") or "").strip().lower()
        if session.validation_status != "passed" or current_intent not in {"repair", "debug"}:
            return pending_targets

        supporting_doc_paths = {
            path
            for artifact in getattr(task_state, "target_artifacts", []) or []
            for path in [str(getattr(artifact, "path", "") or "").strip()]
            if path
            and str(getattr(artifact, "role", "") or "").strip().lower() == "supporting_context"
            and self.validation_planner._is_documentation_path(path)
        }
        if not supporting_doc_paths:
            return pending_targets
        return [path for path in pending_targets if path not in supporting_doc_paths]

    def _path_matches_explicit_request(
        self,
        session: SessionState,
        path: str,
        *,
        generic_stems: set[str] | None = None,
    ) -> bool:
        text = str(path or "").strip()
        if not text:
            return False

        request_lower = self._request_text_for_explicit_target_matching(session)
        request_space = f" {re.sub(r'[^0-9a-zäöüß]+', ' ', request_lower).strip()} "
        if not request_space.strip():
            return False

        generic_stems = generic_stems or {
            "app",
            "cli",
            "doc",
            "docs",
            "guide",
            "index",
            "main",
            "test",
            "tests",
        }
        path_lower = text.lower()
        basename = Path(text).name.lower()
        stem = Path(text).stem.lower()
        normalized_path = re.sub(r"[^0-9a-zäöüß]+", " ", path_lower).strip()
        normalized_basename = re.sub(r"[^0-9a-zäöüß]+", " ", basename).strip()
        normalized_stem = re.sub(r"[^0-9a-zäöüß]+", " ", stem).strip()
        require_context = self._should_require_contextual_path_match(session, text)

        if normalized_path and f" {normalized_path} " in request_space:
            return True
        if not require_context and basename and basename in request_lower:
            return True
        if not require_context and normalized_basename and f" {normalized_basename} " in request_space:
            return True
        return bool(
            not require_context
            and normalized_stem
            and normalized_stem not in generic_stems
            and f" {normalized_stem} " in request_space
        )

    def _request_text_for_explicit_target_matching(self, session: SessionState) -> str:
        request_text = ""
        if session.task_state is not None and session.task_state.latest_user_turn:
            request_text = str(session.task_state.latest_user_turn or "")
        elif session.task:
            request_text = str(session.task or "")
        if not request_text.strip():
            return ""

        sanitized = request_text
        for command in self.validation_planner._explicit_validation_commands(session):
            normalized = self.validation_planner.command_identity(command.command)
            if not normalized:
                continue
            sanitized = re.sub(re.escape(normalized), " ", sanitized, flags=re.IGNORECASE)
        return sanitized.lower()

    def _request_targets_cli_entrypoint(self, request_lower: str) -> bool:
        lowered = str(request_lower or "").lower()
        if "python -m" in lowered or "python3 -m" in lowered:
            return True
        if "command line" in lowered or "kommandozeile" in lowered:
            return True
        if "argparse" in lowered:
            return True
        normalized = f" {re.sub(r'[^0-9a-zäöüß]+', ' ', lowered).strip()} "
        if " cli " in normalized:
            return True
        return bool(re.search(r"(^|[^0-9a-z])--[a-z0-9][\w-]*", lowered))

    def _prioritize_request_entrypoint_targets(
        self,
        session: SessionState,
        candidates: list[str],
    ) -> list[str]:
        if len(candidates) <= 1:
            return candidates
        snapshot = session.workspace_snapshot
        if snapshot is None:
            return candidates
        request_lower = self._request_text_for_explicit_target_matching(session)
        if not self._request_targets_cli_entrypoint(request_lower):
            return candidates

        entrypoints = {
            str(path or "").strip()
            for path in getattr(snapshot, "entrypoints", []) or []
            if str(path or "").strip()
        }
        if not entrypoints:
            return candidates

        prioritized_entrypoints = [
            path
            for path in candidates
            if path in entrypoints
        ]
        if not prioritized_entrypoints:
            return candidates

        def priority(path: str) -> tuple[int, int]:
            basename = Path(path).name.lower()
            is_entrypoint = path in entrypoints
            return (
                0 if is_entrypoint and basename == "__main__.py" else 1 if is_entrypoint else 2,
                0 if self._path_matches_explicit_request(session, path) else 1,
            )

        indexed_positions = {path: index for index, path in enumerate(candidates)}
        return sorted(
            candidates,
            key=lambda path: (
                *priority(path),
                indexed_positions.get(path, 999),
            ),
        )

    def _target_name_path_candidate(self, route: RouterOutput) -> str | None:
        target_name = str(route.entities.target_name or "").strip()
        if not target_name or not self._looks_like_path(target_name):
            return None
        explicit_targets = [path for path in route.entities.target_paths if path and Path(path).suffix]
        target_basename = Path(target_name).name
        if explicit_targets and any(
            Path(candidate).name == target_basename and candidate != target_name
            for candidate in explicit_targets
        ):
            return None
        return target_name

    def _deferred_update_targets(self, session: SessionState) -> set[str]:
        deferred: set[str] = set()
        for note in session.notes[-20:]:
            text = str(note or "").strip()
            if not text.startswith(DEFERRED_UPDATE_TARGET_NOTE_PREFIX):
                continue
            path = text.removeprefix(DEFERRED_UPDATE_TARGET_NOTE_PREFIX).strip()
            if path:
                deferred.add(path)
        return deferred

    def _current_repair_context_hint(
        self,
        session: SessionState,
    ) -> ValidationFailureEvidence | None:
        if session.active_repair_context is not None:
            return session.active_repair_context
        if session.validation_status not in {"failed", "bootstrap_failed", "bootstrap_reset_required"} or not session.validation_runs:
            return None
        failed_run = self.validation_planner.latest_failed_run(session)
        if failed_run is None:
            return None
        return self.validation_planner.build_failure_evidence(session, failed_run)

    def _active_deferred_update_targets(self, session: SessionState) -> set[str]:
        deferred = self._deferred_update_targets(session)
        repair_context = self._current_repair_context_hint(session)
        if repair_context is None or not deferred:
            return deferred

        reactivated: set[str] = set()
        primary_target = next(
            (
                path
                for path in [*repair_context.artifact_paths, *repair_context.file_hints]
                if path
            ),
            None,
        )
        if primary_target:
            reactivated.add(primary_target)
        reactivated.update(
            candidate
            for candidate in deferred
            if self._repair_candidate_is_strongly_referenced(candidate, repair_context)
        )
        return {
            candidate
            for candidate in deferred
            if candidate not in reactivated
        }

    def _next_update_target(self, route: RouterOutput, session: SessionState) -> str | None:
        repair_context = self._current_repair_context_hint(session)
        if route.intent in {RouteIntent.UPDATE, RouteIntent.DEBUG} and repair_context is not None:
            failed_run = self.validation_planner.latest_failed_run(
                session,
                current_generation_only=False,
                command=repair_context.command,
            )
            if failed_run is not None:
                validation_target = self._repair_target_after_failed_validation(
                    route,
                    session,
                    failed_run,
                    repair_context,
                )
                if (
                    validation_target
                    and self._is_safe_workspace_target_candidate(session, validation_target)
                ):
                    absolute = Path(session.workspace_root, validation_target)
                    if absolute.exists() and absolute.is_file():
                        return validation_target
                    if self._repair_target_can_be_created(
                        session,
                        validation_target,
                        repair_context,
                    ):
                        return validation_target
            locked_target = self._repair_brief_locked_target(repair_context)
            primary_target = self._repair_brief_primary_target(repair_context)
            pinned_target = locked_target or primary_target
            if (
                pinned_target
                and not self._runtime_locked_target_should_yield(session, repair_context, pinned_target)
                and self._is_safe_workspace_target_candidate(session, pinned_target)
            ):
                absolute = Path(session.workspace_root, pinned_target)
                if absolute.exists() and absolute.is_file():
                    return pinned_target
                if self._repair_target_can_be_created(session, pinned_target, repair_context):
                    return pinned_target

        ordered_targets = self._ordered_remaining_update_targets(route, session)
        if ordered_targets:
            return ordered_targets[0]
        explicit_targets = self._actionable_explicit_target_paths(route, session)
        if explicit_targets:
            return explicit_targets[0]
        return self._primary_target_path(route, session)

    def _has_pending_explicit_update_targets(self, route: RouterOutput, session: SessionState) -> bool:
        if route.intent != RouteIntent.UPDATE:
            return False
        explicit_targets = self._actionable_explicit_target_paths(route, session)
        if len(explicit_targets) <= 1:
            return False
        changed_paths = {item.path for item in session.changed_files}
        deferred_targets = self._active_deferred_update_targets(session)
        return any(
            candidate not in changed_paths and candidate not in deferred_targets
            for candidate in explicit_targets
        )

    def _has_pending_explicit_create_targets(self, route: RouterOutput, session: SessionState) -> bool:
        if route.intent != RouteIntent.CREATE:
            return False
        explicit_targets = self._ordered_create_targets(route, session)
        if len(explicit_targets) <= 1:
            return False
        changed_paths = {item.path for item in session.changed_files}
        deferred_targets = self._active_deferred_create_targets(route, session)
        return any(
            candidate not in changed_paths and candidate not in deferred_targets
            for candidate in explicit_targets
        )

    def _next_unread_candidate(
        self,
        candidates: list[str],
        read_paths: set[str],
    ) -> str | None:
        for candidate in candidates:
            if candidate and candidate not in read_paths and Path(candidate).suffix:
                return candidate
        return None

    def _next_create_bootstrap(
        self,
        route: RouterOutput,
        session: SessionState,
        read_paths: set[str],
    ) -> str | None:
        for target in self._explicit_target_paths(route, session):
            repair_context = self._repair_context_for_target(route, session, target)
            bootstrap = self._repair_bootstrap_candidate(
                session,
                target,
                read_paths,
                repair_context,
            )
            if bootstrap is not None:
                return bootstrap
        snapshot = session.workspace_snapshot
        if snapshot is None or snapshot.file_count == 0:
            return None
        candidates = [
            *snapshot.manifests[:4],
            *snapshot.entrypoints[:4],
            *snapshot.focus_files[:4],
        ]
        for candidate in self._unique_paths(candidates):
            absolute = Path(session.workspace_root, candidate)
            if candidate not in read_paths and absolute.exists():
                return candidate
        return None

    def _repair_bootstrap_candidate(
        self,
        session: SessionState,
        target: str,
        read_paths: set[str],
        repair_context: ValidationFailureEvidence | None,
    ) -> str | None:
        if repair_context is None:
            return None
        if self._current_file_content(session, target) is not None:
            return None
        related_existing = self._repair_related_existing_context_paths(
            session,
            repair_context,
            exclude=target,
        )
        related_existing = self._ordered_repair_bootstrap_candidates(
            target,
            related_existing,
            repair_context,
        )
        if not related_existing:
            return None
        if any(candidate in read_paths for candidate in related_existing):
            return None
        return related_existing[0]

    def _ordered_repair_bootstrap_candidates(
        self,
        target: str,
        candidates: list[str],
        repair_context: ValidationFailureEvidence,
    ) -> list[str]:
        if repair_context.verification_scope != "runtime" or not candidates:
            return candidates
        if self.validation_planner._is_test_path(target):
            test_like = [candidate for candidate in candidates if self.validation_planner._is_test_path(candidate)]
            non_test = [candidate for candidate in candidates if not self.validation_planner._is_test_path(candidate)]
            ordered = [*test_like, *non_test]
            if ordered:
                return ordered
        return candidates

    def _best_search_query(self, route: RouterOutput) -> str | None:
        for candidate in [
            *route.search_terms,
            route.entities.target_name,
            route.requested_outcome,
            route.user_goal,
        ]:
            text = str(candidate or "").strip()
            if len(text) >= 3:
                return text[:120]
        return None

    def _choose_create_path(self, route: RouterOutput, session: SessionState) -> str:
        follow_up_override = self._follow_up_create_path_override(route, session)
        if follow_up_override:
            self._log("path_generation_skipped", path=follow_up_override, reason="active_artifact_follow_up")
            return follow_up_override
        explicit_targets = self._ordered_create_targets(route, session)
        if explicit_targets:
            changed_paths = {item.path for item in session.changed_files}
            deferred_targets = self._active_deferred_create_targets(route, session)
            for candidate in explicit_targets:
                if candidate in deferred_targets:
                    continue
                if candidate not in changed_paths:
                    return candidate
            return explicit_targets[0]
        for candidate in route.entities.target_paths:
            if candidate:
                return candidate
        if route.entities.target_name and self._looks_like_path(route.entities.target_name):
            return route.entities.target_name
        snapshot = session.workspace_snapshot
        if snapshot is not None and snapshot.file_count == 0:
            path = self._empty_workspace_default_path(route)
            self._log("path_generation_skipped", path=path, reason="empty_workspace_fast_path")
            return path
        prompt = choose_path_prompt(route, session)
        self._log("path_generation_started", target_name=route.entities.target_name)
        outcome = invoke_model(
            lambda progress: self.llm.generate(
                prompt,
                model=self._lightweight_generation_model_name(),
                timeout=max(self._llm_timeout(20), 20),
                total_timeout=max(self._llm_timeout(45), 45),
                num_ctx=1536,
                retries=0,
                progress_callback=progress,
            ),
            operation_name="path_generation",
            task_class="path_generation",
            attempt_number=1,
            capability_tier="tier_b" if self._lightweight_generation_model_name() else "tier_a",
            recovery_strategy="path_generation",
            prompt_variant="compact",
            model_identifier=self._lightweight_generation_model_name() or self._primary_generation_model_name(),
            backend_identifier=self._backend_identifier(),
            inactivity_timeout_seconds=max(self._llm_timeout(20), 20),
            total_timeout_seconds=max(self._llm_timeout(45), 45),
            context_pressure_estimate=estimate_context_pressure(prompt_chars=len(prompt)),
        )
        if outcome.exception is None:
            path = self._sanitize_generated_path(str(outcome.value or ""), self._preferred_extension(route))
            if path:
                self._append_runtime_execution(
                    session,
                    build_execution_run_record(
                        operation_name="path_generation",
                        task_class="path_generation",
                        final_state="completed",
                        capability_tier="tier_b" if self._lightweight_generation_model_name() else "tier_a",
                        recovery_strategy="path_generation",
                        degraded=bool(self._lightweight_generation_model_name()),
                        honest_blocked=False,
                        artifact_bytes_generated=0,
                        validation_possible=False,
                        summary="The target path was generated by the model runtime.",
                        attempts=[outcome.attempt],
                    ),
                )
                self._log("path_generation_finished", path=path)
                return path
        else:
            self._log(
                "path_generation_error",
                error=str(outcome.exception),
                route=route.model_dump(),
                failure=outcome.attempt.failure.to_dict()
                if outcome.attempt.failure is not None
                else None,
            )
        path = self._default_new_path(route)
        self._append_runtime_execution(
            session,
            build_execution_run_record(
                operation_name="path_generation",
                task_class="path_generation",
                final_state="degraded_success",
                capability_tier="tier_d",
                recovery_strategy="deterministic_fallback",
                degraded=True,
                honest_blocked=False,
                artifact_bytes_generated=0,
                validation_possible=False,
                summary="The target path fell back to the deterministic default because model generation did not finish cleanly.",
                attempts=[outcome.attempt],
            ),
        )
        self._log("path_generation_finished", path=path, source="default")
        return path

    def _follow_up_create_path_override(self, route: RouterOutput, session: SessionState) -> str | None:
        task_state = session.task_state
        if task_state is None or route.entities.target_paths:
            return None

        primary_targets = [
            artifact
            for artifact in task_state.target_artifacts
            if artifact.role == "primary_target"
        ]
        has_concrete_new_target = any(
            artifact.path or str(artifact.kind or "").startswith(".")
            for artifact in primary_targets
        )
        if has_concrete_new_target:
            return None

        preferred_extension = self._preferred_extension(route)
        candidate_paths = [
            candidate
            for candidate in self._candidate_paths(route, session)
            if Path(candidate).suffix
        ]
        if len(candidate_paths) != 1:
            return None

        candidate = candidate_paths[0]
        candidate_extension = Path(candidate).suffix
        if preferred_extension not in {"", ".txt"} and candidate_extension != preferred_extension:
            return None
        if task_state.goal_relation not in {"continue", "refine"}:
            return None
        return candidate

    def _generate_file_content(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str | None = None,
        repair_context: ValidationFailureEvidence | None = None,
        repair_strategy: str | None = None,
    ) -> ContentGenerationResult:
        model_name = self._content_generation_model_name(
            route,
            session,
            path=path,
            current_content=current_content,
            repair_context=repair_context,
        )
        prompt_variant = self._content_generation_prompt_variant(
            route,
            session,
            path=path,
            current_content=current_content,
            model_name=model_name,
            repair_context=repair_context,
        )
        prompt = generate_content_prompt(
            route,
            session,
            path=path,
            current_content=current_content,
            repair_context=repair_context,
            repair_strategy=repair_strategy,
            review_feedback=None,
            mode=prompt_variant,
        )
        effective_model = model_name or self._primary_generation_model_name()
        timeout_seconds, total_timeout_seconds = self._content_generation_time_budget(
            prompt_variant=prompt_variant,
            repair_context=repair_context,
        )
        num_ctx = self._content_generation_num_ctx(prompt_variant)
        context_pressure = self._context_pressure_estimate(
            prompt=prompt,
            current_content=current_content,
            exc=None,
        )
        primary_tier = "tier_b" if model_name and model_name != self._primary_generation_model_name() else "tier_a"
        primary_strategy = "planned_fast_model" if primary_tier == "tier_b" else "primary_model"
        prompt_trace_path = self._write_prompt_trace(
            session,
            operation_name="content_generation",
            path=path,
            prompt=prompt,
            model=effective_model,
            prompt_variant=prompt_variant,
            num_ctx=num_ctx,
            timeout_seconds=timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
        )
        attempts: list[ExecutionAttemptRecord] = []

        self._log(
            "content_generation_started",
            path=path,
            update=current_content is not None,
            model=effective_model,
            capability_tier=primary_tier,
            prompt_variant=prompt_variant,
            prompt_chars=len(prompt),
            prompt_lines=prompt.count("\n") + 1,
            prompt_sha256=self._prompt_sha256(prompt),
            num_ctx=num_ctx,
            inactivity_timeout_seconds=timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
            prompt_artifact=prompt_trace_path,
        )
        outcome = invoke_model(
            lambda progress: self.llm.generate(
                prompt,
                model=model_name,
                timeout=timeout_seconds,
                total_timeout=total_timeout_seconds,
                strict_timeouts=prompt_variant == "compact",
                num_ctx=num_ctx,
                retries=0,
                progress_callback=progress,
            ),
            operation_name="content_generation",
            task_class="content_generation",
            attempt_number=1,
            capability_tier=primary_tier,
            recovery_strategy=primary_strategy,
            prompt_variant=prompt_variant,
            model_identifier=effective_model,
            backend_identifier=self._backend_identifier(),
            inactivity_timeout_seconds=timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
            context_pressure_estimate=context_pressure,
            event_callback=self._progress_logger(
                "content_generation_progress",
                path=path,
                update=current_content is not None,
            ),
        )
        attempts.append(outcome.attempt)
        if outcome.exception is None:
            cleaned = self._strip_code_fences(str(outcome.value or "")).strip()
            if cleaned:
                approved_content = cleaned
                review = self._pre_write_update_review(
                    route,
                    session,
                    path=path,
                    current_content=current_content,
                    proposed_content=cleaned,
                    repair_context=repair_context,
                )
                if not review.safe_to_write:
                    self._log(
                        "proposed_update_review_rejected",
                        path=path,
                        summary=review.summary,
                        blocking_issues=review.blocking_issues[:4],
                    )
                    recorded_noop_review = self._record_noop_review_attempt(
                        session,
                        target=path,
                        current_content=current_content,
                        proposed_content=cleaned,
                        repair_context=repair_context,
                        strategy=repair_strategy,
                        review=review,
                    )
                    review_retry = self._retry_update_after_review_failure(
                        route,
                        session,
                        path=path,
                        current_content=current_content,
                        review_feedback=review,
                        repair_context=repair_context,
                        repair_strategy=repair_strategy,
                        prior_attempts=attempts,
                    )
                    attempts.extend(review_retry.attempts)
                    if review_retry.content is None:
                        final_review = review_retry.review or review
                        failure = self._build_update_review_failure(
                            session,
                            path,
                            final_review,
                            current_content=current_content,
                        )
                        self._record_proposed_update_review_failure(
                            session,
                            path=path,
                            review=final_review,
                        )
                        self._append_runtime_execution(
                            session,
                            build_execution_run_record(
                                operation_name="content_generation",
                                task_class="content_generation",
                                final_state="blocked",
                                capability_tier="tier_c",
                                recovery_strategy="quality_review_blocked_write",
                                degraded=False,
                                honest_blocked=True,
                                artifact_bytes_generated=0,
                                validation_possible=False,
                                summary=failure.blocker_message,
                                attempts=attempts,
                            ),
                        )
                        return ContentGenerationResult(
                            source="failed",
                            failure=failure,
                            repair_strategy_used=(
                                review_retry.effective_repair_strategy
                                or self._review_retry_repair_strategy(
                                    repair_strategy,
                                    final_review,
                                    repair_context,
                                )
                            ),
                    )
                    approved_content = review_retry.content
                    repair_strategy = review_retry.effective_repair_strategy or repair_strategy
                    primary_tier = review_retry.capability_tier or primary_tier
                    primary_strategy = review_retry.recovery_strategy or primary_strategy
                self._log(
                    "content_generation_finished",
                    path=path,
                    characters=len(approved_content),
                    update=current_content is not None,
                    source=primary_strategy,
                )
                self._append_runtime_execution(
                    session,
                    build_execution_run_record(
                        operation_name="content_generation",
                        task_class="content_generation",
                        final_state="completed",
                        capability_tier=primary_tier,
                        recovery_strategy=primary_strategy,
                        degraded=primary_tier != "tier_a",
                        honest_blocked=False,
                        artifact_bytes_generated=len(approved_content),
                        validation_possible=bool(approved_content),
                        summary="Artifact generation completed on the selected model tier.",
                        attempts=attempts,
                    ),
                )
                return ContentGenerationResult(
                    content=approved_content,
                    repair_strategy_used=repair_strategy,
                )
            attempts[-1] = self._empty_response_attempt(
                base_attempt=outcome.attempt,
                context_pressure=context_pressure,
            )
        else:
            issue = outcome.attempt.failure
            self._log(
                "content_generation_error",
                error=str(outcome.exception),
                path=path,
                reason=issue.reason if issue is not None else "runtime_error",
                had_progress=issue.had_progress if issue is not None else False,
                partial_characters=issue.characters if issue is not None else 0,
                context_pressure=issue.context_pressure_estimate if issue is not None else context_pressure,
                failure_class=issue.classification if issue is not None else "generation_failed",
            )

        issue = attempts[-1].failure
        retry_result = self._retry_content_generation(
            route,
            session=session,
            path=path,
            current_content=current_content,
            cause=issue,
            prompt=prompt,
            repair_context=repair_context,
            repair_strategy=repair_strategy,
            prior_attempts=attempts,
        )
        attempts.extend(retry_result.attempts)
        if retry_result.content is not None:
            approved_content = retry_result.content
            review = self._pre_write_update_review(
                route,
                session,
                path=path,
                current_content=current_content,
                proposed_content=retry_result.content,
                repair_context=repair_context,
            )
            if not review.safe_to_write:
                self._log(
                    "proposed_update_review_rejected",
                    path=path,
                    summary=review.summary,
                    blocking_issues=review.blocking_issues[:4],
                )
                recorded_noop_review = self._record_noop_review_attempt(
                    session,
                    target=path,
                    current_content=current_content,
                    proposed_content=retry_result.content,
                    repair_context=repair_context,
                    strategy=self._review_retry_repair_strategy(
                        repair_strategy,
                        review,
                        repair_context,
                    ),
                    review=review,
                )
                review_retry = self._retry_update_after_review_failure(
                    route,
                    session,
                    path=path,
                    current_content=current_content,
                    review_feedback=review,
                    repair_context=repair_context,
                    repair_strategy=repair_strategy,
                    prior_attempts=attempts,
                )
                attempts.extend(review_retry.attempts)
                if review_retry.content is None:
                    final_review = review_retry.review or review
                    failure = self._build_update_review_failure(
                        session,
                        path,
                        final_review,
                        current_content=current_content,
                    )
                    self._record_proposed_update_review_failure(
                        session,
                        path=path,
                        review=final_review,
                    )
                    self._append_runtime_execution(
                        session,
                        build_execution_run_record(
                            operation_name="content_generation",
                            task_class="content_generation",
                            final_state="blocked",
                            capability_tier="tier_c",
                            recovery_strategy="quality_review_blocked_write",
                            degraded=False,
                            honest_blocked=True,
                            artifact_bytes_generated=0,
                            validation_possible=False,
                            summary=failure.blocker_message,
                            attempts=attempts,
                        ),
                    )
                    return ContentGenerationResult(
                        source="failed",
                        failure=failure,
                        repair_strategy_used=(
                            review_retry.effective_repair_strategy
                            or self._review_retry_repair_strategy(
                                repair_strategy,
                                final_review,
                                repair_context,
                            )
                        ),
                )
                approved_content = review_retry.content
                repair_strategy = review_retry.effective_repair_strategy or repair_strategy
                retry_result.capability_tier = review_retry.capability_tier or retry_result.capability_tier
                retry_result.recovery_strategy = review_retry.recovery_strategy or retry_result.recovery_strategy
            self._log(
                "content_generation_finished",
                path=path,
                characters=len(approved_content),
                update=current_content is not None,
                source=retry_result.recovery_strategy or "retry",
            )
            self._append_runtime_execution(
                session,
                build_execution_run_record(
                    operation_name="content_generation",
                    task_class="content_generation",
                    final_state="completed",
                    capability_tier=retry_result.capability_tier or "tier_b",
                    recovery_strategy=retry_result.recovery_strategy or "retry_content_generation",
                    degraded=(retry_result.capability_tier or "tier_b") != "tier_a",
                    honest_blocked=False,
                    artifact_bytes_generated=len(approved_content),
                    validation_possible=bool(approved_content),
                    summary="Artifact generation recovered after a runtime startup or stall issue.",
                    attempts=attempts,
                ),
            )
            return ContentGenerationResult(
                content=approved_content,
                source="retry",
                repair_strategy_used=repair_strategy,
            )
        if self._startup_failure_exhausted(attempts):
            recovery = self._no_start_recovery_content(
                route,
                session,
                path=path,
                current_content=current_content,
                attempts=attempts,
            )
            if recovery is not None:
                return recovery
            self._log(
                "content_generation_recovery_unavailable",
                path=path,
                reason="startup_failure_exhausted",
                failure_class="startup_failure_exhausted",
            )
        else:
            fallback = self._template_fallback_content(
                route,
                session,
                path=path,
                current_content=current_content,
            )
            if fallback is not None:
                self._log("content_generation_fallback_started", path=path, source="template")
                self._log("content_generation_fallback_finished", path=path, source="template")
                self._append_runtime_execution(
                    session,
                    build_execution_run_record(
                        operation_name="content_generation",
                        task_class="content_generation",
                        final_state="degraded_success",
                        capability_tier="tier_d",
                        recovery_strategy="deterministic_template",
                        degraded=True,
                        honest_blocked=False,
                        artifact_bytes_generated=len(fallback),
                        validation_possible=bool(fallback),
                        summary="Artifact generation used a deterministic template fallback after backend execution instability.",
                        attempts=attempts,
                    ),
                )
                return ContentGenerationResult(
                    content=fallback,
                    source="template",
                    repair_strategy_used=repair_strategy,
                )
        failure = self._build_content_generation_failure(
            route,
            session,
            path=path,
            current_content=current_content,
            repair_context=repair_context,
            attempts=attempts,
        )
        self._append_runtime_execution(
            session,
            build_execution_run_record(
                operation_name="content_generation",
                task_class="content_generation",
                final_state="blocked",
                capability_tier="tier_e",
                recovery_strategy="honest_block",
                degraded=False,
                honest_blocked=True,
                artifact_bytes_generated=0,
                validation_possible=False,
                summary=failure.blocker_message,
                attempts=attempts,
            ),
        )
        return ContentGenerationResult(
            source="failed",
            failure=failure,
            repair_strategy_used=repair_strategy,
        )

    def _retry_content_generation(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str | None = None,
        cause: ExecutionFailure | None = None,
        prompt: str,
        repair_context: ValidationFailureEvidence | None = None,
        repair_strategy: str | None = None,
        prior_attempts: list[ExecutionAttemptRecord] | None = None,
    ) -> GenerationRetryResult:
        issue = cause or self._assess_generation_issue(
            None,
            prompt=prompt,
            current_content=current_content,
        )
        retained_progress_issue = issue if issue.failed_after_progress and issue.partial_text else None
        attempted_recovery_keys: set[tuple[str, str, str | None]] = set()
        forced_follow_up: GenerationRecoveryAttempt | None = None
        used_resume_retry_after_no_start = False
        retry_attempts: list[ExecutionAttemptRecord] = []
        while True:
            attempt: GenerationRecoveryAttempt | None = None
            if forced_follow_up is not None:
                forced_key = (
                    forced_follow_up.strategy,
                    forced_follow_up.prompt_kind,
                    forced_follow_up.model_name,
                )
                if forced_key not in attempted_recovery_keys:
                    attempt = forced_follow_up
                forced_follow_up = None
            if attempt is None:
                attempts = [
                    candidate
                    for candidate in self._content_generation_recovery_attempts(
                        issue,
                        history=[*(prior_attempts or []), *retry_attempts],
                    )
                    if (
                        candidate.strategy,
                        candidate.prompt_kind,
                        candidate.model_name,
                    )
                    not in attempted_recovery_keys
                ]
                if not attempts:
                    break
                attempt = attempts[0]

            attempted_recovery_keys.add(
                (
                    attempt.strategy,
                    attempt.prompt_kind,
                    attempt.model_name,
                )
            )
            effective_partial_text = issue.partial_text
            if not effective_partial_text and attempt.prompt_kind == "resume" and retained_progress_issue is not None:
                effective_partial_text = retained_progress_issue.partial_text
            retry_prompt = self._content_generation_prompt_for_attempt(
                attempt,
                route,
                session,
                path=path,
                current_content=current_content,
                prompt=prompt,
                partial_text=effective_partial_text,
                repair_context=repair_context,
                repair_strategy=repair_strategy,
            )
            timeout_seconds, total_timeout_seconds, num_ctx = self._content_generation_runtime_for_attempt(
                attempt,
                issue,
                retained_progress_issue=retained_progress_issue,
            )
            self._log(
                "content_generation_retry_started",
                path=path,
                strategy=attempt.strategy,
                model=attempt.model_name or self._primary_generation_model_name(),
                reason=issue.reason,
                had_progress=issue.had_progress,
                partial_characters=issue.characters,
                context_pressure=issue.context_pressure_estimate,
                failure_class=issue.classification,
                capability_tier=attempt.capability_tier,
            )
            outcome = invoke_model(
                lambda progress: self.llm.generate(
                    retry_prompt,
                    model=attempt.model_name,
                    timeout=timeout_seconds,
                    total_timeout=total_timeout_seconds,
                    strict_timeouts=attempt.prompt_kind == "compact",
                    num_ctx=num_ctx,
                    retries=0,
                    progress_callback=progress,
                ),
                operation_name="content_generation",
                task_class="content_generation",
                attempt_number=len(prior_attempts or []) + len(retry_attempts) + 1,
                capability_tier=attempt.capability_tier,
                recovery_strategy=attempt.strategy,
                prompt_variant=attempt.prompt_kind,
                model_identifier=attempt.model_name or self._primary_generation_model_name(),
                backend_identifier=self._backend_identifier(),
                inactivity_timeout_seconds=timeout_seconds,
                total_timeout_seconds=total_timeout_seconds,
                context_pressure_estimate=self._context_pressure_estimate(
                    prompt=retry_prompt,
                    current_content=current_content,
                    exc=None,
                ),
                event_callback=self._progress_logger(
                    "content_generation_progress",
                    path=path,
                    update=current_content is not None,
                    strategy=attempt.strategy,
                ),
            )
            retry_attempts.append(outcome.attempt)
            if outcome.exception is None:
                cleaned = self._strip_code_fences(str(outcome.value or "")).strip()
                if cleaned:
                    self._log(
                        "content_generation_retry_finished",
                        path=path,
                        strategy=attempt.strategy,
                        characters=len(cleaned),
                    )
                    return GenerationRetryResult(
                        content=cleaned,
                        attempts=retry_attempts,
                        capability_tier=attempt.capability_tier,
                        recovery_strategy=attempt.strategy,
                    )
                retry_attempts[-1] = self._empty_response_attempt(
                    base_attempt=outcome.attempt,
                    context_pressure=self._context_pressure_estimate(
                        prompt=retry_prompt,
                        current_content=current_content,
                        exc=None,
                    ),
                )
            retry_issue = retry_attempts[-1].failure
            self._log(
                "content_generation_retry_error",
                path=path,
                strategy=attempt.strategy,
                error=str(outcome.exception or "empty model response"),
                reason=retry_issue.reason if retry_issue is not None else "runtime_error",
                had_progress=retry_issue.had_progress if retry_issue is not None else False,
                partial_characters=retry_issue.characters if retry_issue is not None else 0,
                failure_class=retry_issue.classification if retry_issue is not None else "generation_failed",
                capability_tier=attempt.capability_tier,
            )
            if retry_issue is not None and retry_issue.failed_after_progress and retry_issue.partial_text:
                retained_progress_issue = retry_issue
            if (
                retry_issue is not None
                and retry_issue.no_start_failure
                and attempt.prompt_kind == "resume"
                and retained_progress_issue is not None
                and retained_progress_issue.partial_text
                and not used_resume_retry_after_no_start
            ):
                forced_follow_up = GenerationRecoveryAttempt(
                    strategy=f"{attempt.strategy}_retry",
                    prompt_kind="resume",
                    model_name=attempt.model_name,
                    capability_tier=attempt.capability_tier,
                )
                used_resume_retry_after_no_start = True
            issue = retry_issue or issue
        return GenerationRetryResult(attempts=retry_attempts)

    def _assess_generation_issue(
        self,
        exc: Exception | None,
        *,
        prompt: str,
        current_content: str | None = None,
    ) -> ExecutionFailure:
        synthetic_attempt = ExecutionAttemptRecord(
            operation_name="content_generation",
            task_class="content_generation",
            attempt_number=0,
            capability_tier="tier_a",
            recovery_strategy="assessment_only",
            prompt_variant="full",
            model_identifier=self._primary_generation_model_name(),
            backend_identifier=self._backend_identifier(),
        )
        return classify_execution_failure(
            exc or RuntimeError("generation runtime error"),
            attempt=synthetic_attempt,
            context_pressure_estimate=self._context_pressure_estimate(
                prompt=prompt,
                current_content=current_content,
                exc=exc,
            ),
            elapsed_seconds=getattr(exc, "elapsed", None),
        )

    def _context_pressure_estimate(
        self,
        *,
        prompt: str,
        current_content: str | None,
        exc: Exception | None,
    ) -> str:
        return estimate_context_pressure(
            prompt_chars=len(prompt),
            current_content_chars=len(current_content or ""),
            error_text=str(exc or ""),
        )

    def _empty_response_attempt(
        self,
        *,
        base_attempt: ExecutionAttemptRecord,
        context_pressure: str,
    ) -> ExecutionAttemptRecord:
        failure = ExecutionFailure(
            failure_class="empty_response",
            state="failed_backend_unavailable",
            had_progress=False,
            first_output_received=False,
            startup_timeout_seconds=base_attempt.startup_timeout_seconds,
            inactivity_timeout_seconds=base_attempt.inactivity_timeout_seconds,
            total_timeout_seconds=base_attempt.total_timeout_seconds,
            elapsed_seconds=base_attempt.elapsed_seconds,
            model_identifier=base_attempt.model_identifier,
            backend_identifier=base_attempt.backend_identifier,
            context_pressure_estimate=context_pressure,
            retryable=False,
            recommended_recovery_strategy="reduce_request_complexity",
            raw_reason="empty_response",
            detail="Model returned an empty response.",
        )
        return ExecutionAttemptRecord(
            operation_name=base_attempt.operation_name,
            task_class=base_attempt.task_class,
            attempt_number=base_attempt.attempt_number,
            capability_tier=base_attempt.capability_tier,
            recovery_strategy=base_attempt.recovery_strategy,
            prompt_variant=base_attempt.prompt_variant,
            model_identifier=base_attempt.model_identifier,
            backend_identifier=base_attempt.backend_identifier,
            state=failure.state,
            startup_timeout_seconds=base_attempt.startup_timeout_seconds,
            inactivity_timeout_seconds=base_attempt.inactivity_timeout_seconds,
            total_timeout_seconds=base_attempt.total_timeout_seconds,
            elapsed_seconds=base_attempt.elapsed_seconds,
            had_progress=False,
            first_output_received=False,
            output_characters=0,
            activity_count=base_attempt.activity_count,
            failure=failure,
        )

    def _normalize_generation_strategy_name(
        self,
        strategy: str,
        *,
        prompt_kind: str,
        model_name: str | None,
    ) -> str:
        normalized = str(strategy or "").strip()
        if normalized == "resume_after_progress":
            return "resume_fallback_model" if model_name else "resume_same_model"
        if normalized == "switch_to_faster_model":
            if prompt_kind == "compact":
                return "compact_fallback_model"
            return "fallback_model"
        if normalized == "reduce_request_complexity":
            return "compact_fallback_model" if model_name else "compact_same_model"
        if normalized == "minimal_viable_generation":
            return "compact_fallback_model" if model_name else "compact_same_model"
        if normalized == "retry_same_backend":
            return "retry_same_model"
        return normalized or "retry_same_model"

    def _content_generation_recovery_attempts(
        self,
        issue: ExecutionFailure,
        *,
        history: list[ExecutionAttemptRecord] | None = None,
    ) -> list[GenerationRecoveryAttempt]:
        recovery_model = self._lightweight_generation_model_name() or self._generation_recovery_model_name()
        policy = ExecutionRecoveryPolicy(
            task_class="content_generation",
            allow_same_backend_retry=True,
            allow_smaller_faster_model=bool(recovery_model),
            allow_resume_after_progress=True,
            allow_reduce_request_complexity=True,
            allow_deterministic_fallback=False,
            max_same_backend_retries=1,
            max_total_attempts=4,
        )
        decisions = policy.plan_recovery(
            issue,
            primary_model=self._primary_generation_model_name(),
            faster_model=recovery_model,
            history=list(history or []),
        )
        attempts: list[GenerationRecoveryAttempt] = []
        for decision in decisions:
            self._log(
                "content_generation_recovery_option",
                strategy=decision.candidate.strategy,
                capability_tier=decision.candidate.capability_tier,
                prompt_variant=decision.candidate.prompt_variant,
                model=decision.candidate.model_identifier,
                accepted=decision.accepted,
                reason=decision.reason,
            )
            if not decision.accepted or decision.candidate.local_only:
                continue
            prompt_kind = "full"
            if decision.candidate.prompt_variant == "resume":
                prompt_kind = "resume"
            elif decision.candidate.prompt_variant in {"compact", "minimal"}:
                prompt_kind = "compact"
            candidate_model_name = decision.candidate.model_identifier
            capability_tier = decision.candidate.capability_tier
            if (
                decision.candidate.strategy == "switch_to_faster_model"
                and candidate_model_name
                and candidate_model_name == str(issue.model_identifier or "").strip()
            ):
                continue
            if (
                decision.candidate.strategy in {
                    "retry_same_backend",
                    "resume_after_progress",
                    "reduce_request_complexity",
                    "minimal_viable_generation",
                }
                and candidate_model_name == self._primary_generation_model_name()
            ):
                candidate_model_name = None
            attempts.append(
                GenerationRecoveryAttempt(
                    strategy=self._normalize_generation_strategy_name(
                        decision.candidate.strategy,
                        prompt_kind=prompt_kind,
                        model_name=candidate_model_name,
                    ),
                    prompt_kind=prompt_kind,
                    model_name=candidate_model_name,
                    capability_tier=capability_tier,
                )
            )
        primary_model = self._primary_generation_model_name()
        lightweight_model = recovery_model
        current_model = str(issue.model_identifier or "").strip()
        if (
            issue.no_start_failure
            and primary_model
            and lightweight_model
            and current_model == lightweight_model
            and primary_model != current_model
            and not any(
                (attempt.model_name or primary_model) == primary_model
                for attempt in attempts
            )
            and not any(
                str(item.model_identifier or "").strip() == primary_model
                and str(item.prompt_variant or "").strip() == "full"
                for item in list(history or [])
            )
        ):
            attempts.append(
                GenerationRecoveryAttempt(
                    strategy="switch_to_primary_model",
                    prompt_kind="full",
                    model_name=None,
                    capability_tier="tier_a",
                )
            )
        if (
            issue.no_start_failure
            and issue.context_pressure_likely
            and (not lightweight_model or lightweight_model == primary_model)
        ):
            compact_same_model = next(
                (
                    attempt
                    for attempt in attempts
                    if attempt.prompt_kind == "compact"
                    and (attempt.model_name is None or attempt.model_name == primary_model)
                ),
                None,
            )
            full_same_model = next(
                (
                    attempt
                    for attempt in attempts
                    if attempt.prompt_kind == "full"
                    and attempt.strategy == "retry_same_model"
                    and (attempt.model_name is None or attempt.model_name == primary_model)
                ),
                None,
            )
            if compact_same_model is not None and full_same_model is not None:
                reordered: list[GenerationRecoveryAttempt] = [compact_same_model]
                for attempt in attempts:
                    if attempt is compact_same_model:
                        continue
                    reordered.append(attempt)
                attempts = reordered
        return attempts

    def _content_generation_prompt_for_attempt(
        self,
        attempt: GenerationRecoveryAttempt,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str | None,
        prompt: str,
        partial_text: str,
        repair_context: ValidationFailureEvidence | None,
        repair_strategy: str | None,
        review_feedback: ProposedUpdateReview | None = None,
    ) -> str:
        if attempt.prompt_kind == "resume":
            return generate_content_continuation_prompt(
                route,
                session,
                path=path,
                partial_content=partial_text,
                current_content=current_content,
                repair_context=repair_context,
                repair_strategy=repair_strategy,
                review_feedback=review_feedback,
            )
        if attempt.prompt_kind == "compact":
            return generate_content_retry_prompt(
                route,
                session,
                path=path,
                current_content=current_content,
                repair_context=repair_context,
                repair_strategy=repair_strategy,
                review_feedback=review_feedback,
            )
        return generate_content_prompt(
            route,
            session,
            path=path,
            current_content=current_content,
            repair_context=repair_context,
            repair_strategy=repair_strategy,
            review_feedback=review_feedback,
            mode="full",
        )

    def _content_generation_runtime_for_attempt(
        self,
        attempt: GenerationRecoveryAttempt,
        issue: ExecutionFailure,
        *,
        retained_progress_issue: ExecutionFailure | None = None,
    ) -> tuple[int, int, int]:
        progress_timeout_recovery = (
            issue.timeout_like and issue.had_progress
        ) or (
            issue.no_start_failure
            and retained_progress_issue is not None
            and retained_progress_issue.timeout_like
            and retained_progress_issue.had_progress
        )
        if attempt.prompt_kind == "resume":
            base_timeout = 60 if issue.timeout_like else 45
            total_timeout = 270 if progress_timeout_recovery else (210 if issue.timeout_like else 150)
            num_ctx = 3072 if attempt.model_name is None else 2048
        elif attempt.prompt_kind == "compact":
            base_timeout = 60 if issue.timeout_like else 45
            total_timeout = 270 if progress_timeout_recovery else (210 if issue.timeout_like else 150)
            num_ctx = 2048
        else:
            base_timeout = 75 if issue.timeout_like else 60
            total_timeout = 270 if progress_timeout_recovery else (210 if issue.timeout_like else 150)
            num_ctx = 4096 if attempt.model_name is None else 3072
        return (
            max(self._llm_timeout(base_timeout), base_timeout),
            max(self._llm_timeout(total_timeout), total_timeout),
            num_ctx,
        )

    def _content_generation_time_budget(
        self,
        *,
        prompt_variant: str,
        repair_context: ValidationFailureEvidence | None,
    ) -> tuple[int, int]:
        compact_prompt = str(prompt_variant or "").strip() == "compact"
        if compact_prompt:
            timeout_seconds = max(self._llm_timeout(60), 60)
            total_timeout = 210 if repair_context is not None else 150
        else:
            timeout_seconds = max(self._llm_timeout(75), 75)
            total_timeout = 240 if repair_context is not None else 210
        return timeout_seconds, max(self._llm_timeout(total_timeout), total_timeout)

    def _startup_failure_exhausted(
        self,
        attempts: list[ExecutionAttemptRecord],
    ) -> bool:
        issues = [attempt.failure for attempt in attempts if attempt.failure is not None]
        return len(issues) >= 2 and all(issue.no_start_failure for issue in issues)

    def _attempts_have_progress(
        self,
        attempts: list[ExecutionAttemptRecord],
    ) -> bool:
        return any(
            attempt.failure is not None
            and (
                attempt.failure.had_progress
                or attempt.failure.characters > 0
                or bool(attempt.failure.partial_text)
            )
            for attempt in attempts
        )

    def _generation_models_summary(
        self,
        attempts: list[ExecutionAttemptRecord],
    ) -> str:
        models: list[str] = []
        for attempt in attempts:
            model_name = str(attempt.model_identifier or "").strip()
            if model_name and model_name not in models:
                models.append(model_name)
        return ", ".join(models[:3])

    def _build_content_generation_failure(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str | None,
        repair_context: ValidationFailureEvidence | None,
        attempts: list[ExecutionAttemptRecord],
    ) -> ContentGenerationFailure:
        del route
        language = self._session_language(session)
        if self._startup_failure_exhausted(attempts):
            models = self._generation_models_summary(attempts) or "the primary and fallback models"
            blocker_message = (
                f"Repeated model start failure for {path}: {models} produced no first chunk, and no safe local recovery path applied."
            )
            user_message = self._localized_text(
                language,
                de=(
                    f"Ich konnte die Generierung fuer {path} nicht starten: "
                    "Weder das Primaer- noch das Fallback-Modell haben vor dem ersten Chunk geantwortet, "
                    "und fuer diese Aufgabe gab es keinen sicheren lokalen Recovery-Pfad."
                ),
                en=(
                    f"I could not start generation for {path}: "
                    "neither the primary nor the fallback model produced a first chunk, "
                    "and there was no safe local recovery path for this task."
                ),
            )
            return ContentGenerationFailure(
                stop_reason="model_start_failed",
                failure_class="startup_failure_exhausted",
                blocker_message=blocker_message,
                user_message=user_message,
                attempts=attempts,
            )

        if repair_context is not None:
            blocker_message = (
                f"Repair generation failed for {path}; no reliable content could be generated from the current validation evidence."
            )
            user_message = self._localized_text(
                language,
                de=(
                    "Ich konnte aus der aktuellen Validierungs-Evidenz keine belastbare Reparatur erzeugen. "
                    "Das Modell hat den Repair-Pfad nicht stabil zu einem verwertbaren Inhalt abgeschlossen."
                ),
                en=(
                    "I could not generate a reliable repair from the current validation evidence. "
                    "The model did not complete the repair path into usable content."
                ),
            )
            return ContentGenerationFailure(
                stop_reason="repair_generation_failed",
                failure_class="repair_generation_failed",
                blocker_message=blocker_message,
                user_message=user_message,
                attempts=attempts,
            )

        if self._attempts_have_progress(attempts):
            blocker_message = f"Content generation for {path} failed after partial progress."
            user_message = self._localized_text(
                language,
                de=(
                    f"Ich habe fuer {path} Teilfortschritt gesehen, aber keinen belastbaren Abschluss erzeugen koennen."
                ),
                en=(
                    f"I saw partial progress for {path}, but I could not complete it into a reliable final result."
                ),
            )
            return ContentGenerationFailure(
                stop_reason="generation_failed_after_progress",
                failure_class="generation_failed_after_progress",
                blocker_message=blocker_message,
                user_message=user_message,
                attempts=attempts,
            )

        blocker_message = (
            f"No reliable update content could be generated for {path}."
            if current_content is not None
            else f"No reliable content could be generated for {path}."
        )
        user_message = self._localized_text(
            language,
            de=(
                f"Ich konnte noch keinen belastbaren Inhalt fuer {path} erzeugen."
                if current_content is None
                else f"Ich konnte noch keine belastbare Aktualisierung fuer {path} erzeugen."
            ),
            en=(
                f"I could not produce reliable content for {path} yet."
                if current_content is None
                else f"I could not produce a reliable update for {path} yet."
            ),
        )
        return ContentGenerationFailure(
            stop_reason="generation_failed",
            failure_class="generation_failed",
            blocker_message=blocker_message,
            user_message=user_message,
            attempts=attempts,
        )

    def _no_start_recovery_content(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str | None,
        attempts: list[ExecutionAttemptRecord],
    ) -> ContentGenerationResult | None:
        direct_main_recovery = self._deterministic_direct_main_runtime_recovery(
            route,
            session,
            path=path,
            current_content=current_content,
            repair_context=session.active_repair_context,
        )
        if direct_main_recovery is not None:
            self._log(
                "content_generation_recovery_started",
                path=path,
                strategy=direct_main_recovery.recovery_strategy,
                failure_class="startup_failure_exhausted",
                models=self._generation_models_summary(attempts),
            )
            self._log(
                "content_generation_fallback_started",
                path=path,
                source="deterministic_direct_main_contract",
            )
            self._log(
                "content_generation_fallback_finished",
                path=path,
                source="deterministic_direct_main_contract",
            )
            self._log(
                "content_generation_recovery_finished",
                path=path,
                strategy=direct_main_recovery.recovery_strategy,
                source="deterministic_direct_main_contract",
            )
            self._append_runtime_execution(
                session,
                build_execution_run_record(
                    operation_name="content_generation",
                    task_class="content_generation",
                    final_state="degraded_success",
                    capability_tier=direct_main_recovery.capability_tier,
                    recovery_strategy=direct_main_recovery.recovery_strategy,
                    degraded=True,
                    honest_blocked=False,
                    artifact_bytes_generated=len(direct_main_recovery.content),
                    validation_possible=True,
                    summary="Artifact generation degraded to a deterministic direct-main runtime repair after repeated startup failures.",
                    attempts=attempts,
                ),
            )
            return ContentGenerationResult(
                content=direct_main_recovery.content,
                source="deterministic_direct_main_contract",
            )

        template = self._template_fallback_content(
            route,
            session,
            path=path,
            current_content=current_content,
        )
        if template is not None:
            self._log(
                "content_generation_recovery_started",
                path=path,
                strategy="deterministic_template",
                failure_class="startup_failure_exhausted",
                models=self._generation_models_summary(attempts),
            )
            self._log("content_generation_fallback_started", path=path, source="template")
            self._log("content_generation_fallback_finished", path=path, source="template")
            self._log(
                "content_generation_recovery_finished",
                path=path,
                strategy="deterministic_template",
                source="template",
            )
            self._append_runtime_execution(
                session,
                build_execution_run_record(
                    operation_name="content_generation",
                    task_class="content_generation",
                    final_state="degraded_success",
                    capability_tier="tier_d",
                    recovery_strategy="deterministic_template",
                    degraded=True,
                    honest_blocked=False,
                    artifact_bytes_generated=len(template),
                    validation_possible=bool(template),
                    summary="Artifact generation degraded to a deterministic template after repeated startup failures.",
                    attempts=attempts,
                ),
            )
            return ContentGenerationResult(
                content=template,
                source="template",
            )

        scaffold = self._starter_scaffold_content(
            route,
            session,
            path=path,
            current_content=current_content,
        )
        if scaffold is None:
            return None
        self._log(
            "content_generation_recovery_started",
            path=path,
            strategy="starter_scaffold",
            failure_class="startup_failure_exhausted",
            models=self._generation_models_summary(attempts),
        )
        self._log("content_generation_fallback_started", path=path, source="starter_scaffold")
        self._log("content_generation_fallback_finished", path=path, source="starter_scaffold")
        self._log(
            "content_generation_recovery_finished",
            path=path,
            strategy="starter_scaffold",
            source="starter_scaffold",
        )
        self._append_runtime_execution(
            session,
            build_execution_run_record(
                operation_name="content_generation",
                task_class="content_generation",
                final_state="degraded_success",
                capability_tier="tier_d",
                recovery_strategy="starter_scaffold",
                degraded=True,
                honest_blocked=False,
                artifact_bytes_generated=len(scaffold),
                validation_possible=bool(scaffold),
                summary="Artifact generation degraded to a minimal starter scaffold after repeated startup failures.",
                attempts=attempts,
            ),
        )
        return ContentGenerationResult(
            content=scaffold,
            source="starter_scaffold",
        )

    def _starter_scaffold_content(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str | None,
    ) -> str | None:
        if current_content is not None or not self._is_low_risk_starter_task(route, session, path=path):
            return None

        title = self._starter_title(route, path)
        suffix = Path(path).suffix.lower()
        if suffix == ".py":
            return (
                "def main():\n"
                f"    print(\"{title} starter scaffold ready.\")\n"
                "\n"
                "\n"
                "if __name__ == \"__main__\":\n"
                "    main()\n"
            )
        if suffix == ".js":
            return (
                "function main() {\n"
                f"  console.log(\"{title} starter scaffold ready.\");\n"
                "}\n"
                "\n"
                "main();\n"
            )
        if suffix == ".html":
            return (
                "<!doctype html>\n"
                "<html lang=\"en\">\n"
                "<head>\n"
                "  <meta charset=\"utf-8\">\n"
                "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
                f"  <title>{title}</title>\n"
                "  <style>\n"
                "    body { font-family: sans-serif; margin: 2rem; }\n"
                "    main { max-width: 48rem; }\n"
                "  </style>\n"
                "</head>\n"
                "<body>\n"
                "  <main>\n"
                f"    <h1>{title}</h1>\n"
                "    <p>Starter scaffold ready.</p>\n"
                "  </main>\n"
                "  <script>\n"
                f"    console.log(\"{title} starter scaffold ready.\");\n"
                "  </script>\n"
                "</body>\n"
                "</html>\n"
            )
        return None

    def _is_low_risk_starter_task(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
    ) -> bool:
        task_state = session.task_state
        if route.intent != RouteIntent.CREATE:
            return False
        if task_state is None or task_state.goal_relation != "new_task":
            return False
        if (
            task_state.execution_strategy is not None
            and task_state.execution_strategy != "feature_implementation"
        ):
            return False
        if route.repo_context_needed or route.needs_clarification or not route.safe_to_execute:
            return False
        if session.changed_files or session.validation_runs or session.diagnostics:
            return False
        snapshot = session.workspace_snapshot
        if snapshot is not None and snapshot.file_count > 25:
            return False
        if Path(path).suffix.lower() not in {".py", ".js", ".html"}:
            return False
        if len([item for item in route.entities.target_paths if item]) > 1:
            return False
        if len(Path(path).parts) > 2:
            return False
        return self._starter_scope_requested(route, session, path=path)

    def _starter_scope_requested(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
    ) -> bool:
        task_state = session.task_state
        normalized = " ".join(
            part.strip().lower()
            for part in [
                route.user_goal,
                route.requested_outcome,
                route.entities.target_name or "",
                path,
                task_state.root_goal if task_state is not None else "",
                task_state.active_goal if task_state is not None else "",
                task_state.output_expectation if task_state is not None else "",
            ]
            if str(part or "").strip()
        )
        starter_markers = (
            "starter",
            "scaffold",
            "skeleton",
            "template",
            "boilerplate",
            "grundgeruest",
            "grundgerüst",
            "geruest",
            "gerüst",
            "stub",
            "hello world",
        )
        if any(marker in normalized for marker in starter_markers):
            return True
        conventional_names = {"index.html", "main.py", "script.js", "app.py", "app.js"}
        return len(normalized) <= 120 and Path(path).name.lower() in conventional_names

    def _starter_title(
        self,
        route: RouterOutput,
        path: str,
    ) -> str:
        raw = str(route.entities.target_name or Path(path).stem).strip() or Path(path).stem
        cleaned = re.sub(r"[_-]+", " ", raw).strip()
        return cleaned[:80] or "Starter"

    def _current_file_content(self, session: SessionState, path: str) -> str | None:
        target = Path(session.workspace_root, path)
        if not target.exists() or target.is_dir():
            return None
        return target.read_text(encoding="utf-8")

    def _latest_session_write_content(self, session: SessionState, path: str) -> str | None:
        normalized_path = str(path or "").strip()
        if not normalized_path:
            return None

        for item in reversed(session.tool_calls):
            if not item.success:
                continue
            if item.tool_name not in {"write_file", "create_file", "replace_file"}:
                continue
            candidate_path = str(item.tool_args.get("path") or "").strip()
            if candidate_path != normalized_path:
                continue
            content = item.tool_args.get("content")
            if isinstance(content, str):
                return content
        return None

    def _session_or_current_file_content(self, session: SessionState, path: str) -> str | None:
        pending_content = self._latest_session_write_content(session, path)
        if pending_content is not None:
            return pending_content
        return self._current_file_content(session, path)

    def _path_is_web_contract_artifact(self, path: str) -> bool:
        return Path(path).suffix.lower() in WEB_CONTRACT_SUFFIXES

    def _path_is_web_contract_html(self, path: str) -> bool:
        return Path(path).suffix.lower() in WEB_CONTRACT_HTML_SUFFIXES

    def _path_is_web_contract_css(self, path: str) -> bool:
        return Path(path).suffix.lower() in WEB_CONTRACT_CSS_SUFFIXES

    def _path_is_web_contract_script(self, path: str) -> bool:
        return Path(path).suffix.lower() in WEB_CONTRACT_SCRIPT_SUFFIXES

    def _normalize_web_hook_token(self, token: str) -> str | None:
        candidate = str(token or "").strip()
        if not candidate or not re.fullmatch(r"[A-Za-z_][\w-]*", candidate):
            return None
        return candidate

    def _web_contract_inventory(self, path: str, content: str) -> WebContractInventory:
        inventory = WebContractInventory()
        suffix = Path(path).suffix.lower()

        if suffix in WEB_CONTRACT_HTML_SUFFIXES:
            for tag_match in re.finditer(r"<[A-Za-z][^>]*>", content):
                tag_text = str(tag_match.group(0) or "")
                id_match = re.search(r"\bid\s*=\s*(['\"])([^'\"]+)\1", tag_text, flags=re.IGNORECASE)
                class_match = re.search(r"\bclass\s*=\s*(['\"])([^'\"]+)\1", tag_text, flags=re.IGNORECASE)
                normalized_id = None
                if id_match is not None:
                    normalized_id = self._normalize_web_hook_token(str(id_match.group(2) or ""))
                    if normalized_id is not None:
                        inventory.html_ids.add(normalized_id)
                if normalized_id is None or class_match is None:
                    continue
                class_tokens: set[str] = set()
                for token in str(class_match.group(2) or "").split():
                    normalized = self._normalize_web_hook_token(token)
                    if normalized is not None:
                        class_tokens.add(normalized)
                if class_tokens:
                    inventory.html_id_classes.setdefault(normalized_id, set()).update(class_tokens)
            for _, raw_value in re.findall(r"\bid\s*=\s*(['\"])([^'\"]+)\1", content, flags=re.IGNORECASE):
                normalized = self._normalize_web_hook_token(raw_value)
                if normalized is not None:
                    inventory.html_ids.add(normalized)
            for _, raw_value in re.findall(
                r"<(?:html|body)\b[^>]*\bclass\s*=\s*(['\"])([^'\"]+)\1",
                content,
                flags=re.IGNORECASE,
            ):
                for token in raw_value.split():
                    normalized = self._normalize_web_hook_token(token)
                    if normalized is not None:
                        inventory.html_root_classes.add(normalized)

        if suffix in WEB_CONTRACT_CSS_SUFFIXES:
            selector_source = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
            for selector_text in re.findall(r"(?s)([^{}]+)\{", selector_source):
                for raw_value in re.findall(r"#([A-Za-z_][\w-]*)", selector_text):
                    normalized = self._normalize_web_hook_token(raw_value)
                    if normalized is not None:
                        inventory.css_id_selectors.add(normalized)
                for raw_value in re.findall(r"\.([A-Za-z_][\w-]*)", selector_text):
                    normalized = self._normalize_web_hook_token(raw_value)
                    if normalized is not None:
                        inventory.css_class_selectors.add(normalized)
                for raw_value in re.findall(
                    r"(?:^|[\s>+~,])(?:html|body|:root)\.([A-Za-z_][\w-]*)",
                    selector_text,
                ):
                    normalized = self._normalize_web_hook_token(raw_value)
                    if normalized is not None:
                        inventory.css_root_state_classes.add(normalized)

        if suffix in WEB_CONTRACT_SCRIPT_SUFFIXES:
            for raw_value in re.findall(r"getElementById\(\s*['\"]([^'\"]+)['\"]\s*\)", content):
                normalized = self._normalize_web_hook_token(raw_value)
                if normalized is not None:
                    inventory.js_id_refs.add(normalized)
            for _, selector_text in re.findall(
                r"(?:querySelector|querySelectorAll|closest|matches)\(\s*(['\"])(.*?)\1\s*\)",
                content,
                flags=re.DOTALL,
            ):
                for raw_value in re.findall(r"#([A-Za-z_][\w-]*)", selector_text):
                    normalized = self._normalize_web_hook_token(raw_value)
                    if normalized is not None:
                        inventory.js_id_refs.add(normalized)
                for raw_value in re.findall(r"\.([A-Za-z_][\w-]*)", selector_text):
                    normalized = self._normalize_web_hook_token(raw_value)
                    if normalized is not None:
                        inventory.js_class_tokens.add(normalized)
            for raw_value in re.findall(r"\.id\s*=\s*['\"]([^'\"]+)['\"]", content):
                normalized = self._normalize_web_hook_token(raw_value)
                if normalized is not None:
                    inventory.js_declared_ids.add(normalized)
            for raw_value in re.findall(
                r"setAttribute\(\s*['\"]id['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)",
                content,
            ):
                normalized = self._normalize_web_hook_token(raw_value)
                if normalized is not None:
                    inventory.js_declared_ids.add(normalized)
            for raw_value in re.findall(
                r"document\.(?:body|documentElement)\.classList\.(?:add|remove|toggle|replace|contains)\(\s*['\"]([^'\"]+)['\"]",
                content,
            ):
                normalized = self._normalize_web_hook_token(raw_value)
                if normalized is not None:
                    inventory.js_root_state_classes.add(normalized)
            for raw_value in re.findall(
                r"\.classList\.(?:add|remove|toggle|replace|contains)\(\s*['\"]([^'\"]+)['\"]",
                content,
            ):
                normalized = self._normalize_web_hook_token(raw_value)
                if normalized is not None:
                    inventory.js_class_tokens.add(normalized)

        return inventory

    def _web_contract_paths(
        self,
        session: SessionState,
        *,
        route: RouterOutput | None = None,
        current_path: str | None = None,
    ) -> tuple[list[str], set[str]]:
        resolved_route = route or session.router_result
        candidates: list[str] = []
        if resolved_route is not None:
            candidates.extend(self._explicit_target_paths(resolved_route, session))
        candidates.extend(str(item.path or "").strip() for item in session.changed_files)
        if current_path:
            candidates.append(current_path)
        relevant_paths = [
            candidate
            for candidate in self._unique_paths(candidates)
            if candidate and self._path_is_web_contract_artifact(candidate)
        ]
        if resolved_route is None:
            return relevant_paths, set()

        if resolved_route.intent == RouteIntent.CREATE:
            deferred_targets = self._active_deferred_create_targets(resolved_route, session)
        else:
            deferred_targets = self._active_deferred_update_targets(session)
        changed_paths = {str(item.path or "").strip() for item in session.changed_files if str(item.path or "").strip()}
        pending_paths = {
            candidate
            for candidate in self._explicit_target_paths(resolved_route, session)
            if (
                candidate
                and candidate != current_path
                and candidate not in changed_paths
                and candidate not in deferred_targets
                and self._path_is_web_contract_artifact(candidate)
            )
        }
        return relevant_paths, pending_paths

    def _web_contract_analysis(
        self,
        session: SessionState,
        *,
        route: RouterOutput | None = None,
        current_path: str | None = None,
        proposed_content: str | None = None,
    ) -> tuple[list[WebContractFinding], list[str], dict[str, WebContractInventory]]:
        relevant_paths, pending_paths = self._web_contract_paths(
            session,
            route=route,
            current_path=current_path,
        )
        if len(relevant_paths) < 2:
            return [], [], {}

        contents_by_path: dict[str, str] = {}
        for candidate in relevant_paths:
            if current_path is not None and candidate == current_path:
                content = proposed_content
            else:
                content = self._session_or_current_file_content(session, candidate)
            if isinstance(content, str) and content.strip():
                contents_by_path[candidate] = content
        if len(contents_by_path) < 2:
            return [], [], {}

        has_html = any(self._path_is_web_contract_html(path) for path in contents_by_path)
        has_css = any(self._path_is_web_contract_css(path) for path in contents_by_path)
        has_script = any(self._path_is_web_contract_script(path) for path in contents_by_path)
        if not has_html or not (has_css or has_script):
            return [], [], {}

        scope_paths = {
            str(item.path or "").strip()
            for item in session.changed_files
            if str(item.path or "").strip() in contents_by_path
        }
        if current_path is not None:
            scope_paths.add(current_path)
        if not scope_paths:
            return [], [], {}

        inventories: dict[str, WebContractInventory] = {
            path: self._web_contract_inventory(path, content)
            for path, content in contents_by_path.items()
        }

        def collect_by_token(
            paths: set[str],
            extractor,
        ) -> dict[str, set[str]]:
            index: dict[str, set[str]] = {}
            for path in paths:
                inventory = inventories.get(path)
                if inventory is None:
                    continue
                for token in extractor(inventory):
                    index.setdefault(token, set()).add(path)
            return index

        all_html_ids = set().union(*(item.html_ids for item in inventories.values()))
        all_html_root_classes = set().union(*(item.html_root_classes for item in inventories.values()))
        all_css_id_selectors = set().union(*(item.css_id_selectors for item in inventories.values()))
        all_js_id_refs = set().union(*(item.js_id_refs for item in inventories.values()))
        all_js_declared_ids = set().union(*(item.js_declared_ids for item in inventories.values()))
        all_js_root_state_classes = set().union(*(item.js_root_state_classes for item in inventories.values()))
        all_declared_ids = all_html_ids | all_js_declared_ids

        source_html_paths = {path for path in scope_paths if self._path_is_web_contract_html(path)}
        source_css_paths = {path for path in scope_paths if self._path_is_web_contract_css(path)}
        source_script_paths = {path for path in scope_paths if self._path_is_web_contract_script(path)}
        source_html_ids = collect_by_token(source_html_paths, lambda inventory: inventory.html_ids)
        source_css_id_selectors = collect_by_token(source_css_paths, lambda inventory: inventory.css_id_selectors)
        source_css_root_state_classes = collect_by_token(
            source_css_paths,
            lambda inventory: inventory.css_root_state_classes,
        )
        source_js_id_refs = collect_by_token(source_script_paths, lambda inventory: inventory.js_id_refs)

        pending_html = any(self._path_is_web_contract_html(path) for path in pending_paths)
        pending_css = any(self._path_is_web_contract_css(path) for path in pending_paths)
        pending_script = any(self._path_is_web_contract_script(path) for path in pending_paths)

        findings: list[WebContractFinding] = []
        file_hints: set[str] = set()

        for token, paths in sorted(source_css_root_state_classes.items()):
            if token in all_html_root_classes or token in all_js_root_state_classes:
                continue
            if pending_html or pending_script:
                continue
            source_preview = ", ".join(sorted(paths)[:2])
            findings.append(
                WebContractFinding(
                    kind="css_root_state",
                    token=token,
                    source_paths=tuple(sorted(paths)),
                    summary=(
                        f"{source_preview} introduces the root state selector '{token}', but no current sibling HTML or JS provides or toggles that state."
                    ),
                )
            )
            file_hints.update(paths)

        if has_script:
            for token, paths in sorted(source_html_ids.items()):
                if token in all_js_id_refs or token in all_css_id_selectors:
                    continue
                if pending_css or pending_script:
                    continue
                source_preview = ", ".join(sorted(paths)[:2])
                findings.append(
                    WebContractFinding(
                        kind="html_id_hook",
                        token=token,
                        source_paths=tuple(sorted(paths)),
                        summary=(
                            f"{source_preview} introduces the id hook '{token}', but no current sibling CSS or JS consumes it."
                        ),
                    )
                )
                file_hints.update(paths)

        for token, paths in sorted(source_css_id_selectors.items()):
            if token in all_declared_ids:
                continue
            if pending_html or pending_script:
                continue
            source_preview = ", ".join(sorted(paths)[:2])
            findings.append(
                WebContractFinding(
                    kind="css_id_selector",
                    token=token,
                    source_paths=tuple(sorted(paths)),
                    summary=(
                        f"{source_preview} references the id selector '#{token}', but no current sibling HTML or JS declares that hook."
                    ),
                )
            )
            file_hints.update(paths)

        for token, paths in sorted(source_js_id_refs.items()):
            if token in all_declared_ids:
                continue
            if pending_html or pending_script:
                continue
            source_preview = ", ".join(sorted(paths)[:2])
            findings.append(
                WebContractFinding(
                    kind="js_id_lookup",
                    token=token,
                    source_paths=tuple(sorted(paths)),
                    summary=(
                        f"{source_preview} looks up the id hook '{token}', but no current sibling HTML or JS declares it."
                    ),
                )
            )
            file_hints.update(paths)

        if not findings:
            return [], [], inventories
        return findings[:4], sorted(file_hints)[:6], inventories

    def _web_contract_findings(
        self,
        session: SessionState,
        *,
        route: RouterOutput | None = None,
        current_path: str | None = None,
        proposed_content: str | None = None,
    ) -> tuple[list[str], list[str]]:
        findings, file_hints, _ = self._web_contract_analysis(
            session,
            route=route,
            current_path=current_path,
            proposed_content=proposed_content,
        )
        return [item.summary for item in findings], file_hints

    def _web_contract_repair_hints(
        self,
        *,
        path: str,
        findings: list[WebContractFinding],
        inventories: dict[str, WebContractInventory],
    ) -> list[str]:
        current_inventory = inventories.get(path)
        sibling_class_tokens: set[str] = set()
        for sibling_path, inventory in inventories.items():
            if sibling_path == path:
                continue
            sibling_class_tokens.update(inventory.css_class_selectors)
            sibling_class_tokens.update(inventory.js_class_tokens)

        hints: list[str] = []
        for finding in findings:
            if finding.kind == "html_id_hook":
                related_classes = (
                    current_inventory.html_id_classes.get(finding.token, set())
                    if current_inventory is not None
                    else set()
                )
                consumed_classes = sorted(token for token in related_classes if token in sibling_class_tokens)
                if consumed_classes:
                    class_preview = ", ".join(consumed_classes[:2])
                    hints.append(
                        f"Reuse the existing class-based hook {class_preview} in {path} and remove the unconsumed id hook '{finding.token}' unless sibling CSS or JS also consumes that id."
                    )
                hints.append(
                    f"If you are repairing only {path}, remove or rename the unconsumed id hook '{finding.token}' instead of returning it unchanged."
                )
                continue
            if finding.kind == "css_root_state":
                hints.append(
                    f"Do not introduce the root-state token '{finding.token}' in {path} until sibling HTML or JS actually provides or toggles it."
                )
                continue
            if finding.kind == "css_id_selector":
                hints.append(
                    f"Drop or rename the selector '#{finding.token}' in {path} unless sibling HTML or JS declares that hook."
                )
                continue
            if finding.kind == "js_id_lookup":
                hints.append(
                    f"Remove the lookup for '{finding.token}' in {path} or declare that id in the shared HTML or JS contract before using it."
                )

        hints.append(
            "Keep one shared contract across HTML, CSS, and JS instead of introducing parallel hooks that describe the same UI behavior in different ways."
        )
        return list(dict.fromkeys(hints))[:4]

    def _pre_write_web_contract_review(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        proposed_content: str,
    ) -> ProposedUpdateReview | None:
        findings, _, inventories = self._web_contract_analysis(
            session,
            route=route,
            current_path=path,
            proposed_content=proposed_content,
        )
        if not findings:
            return None
        return ProposedUpdateReview(
            safe_to_write=False,
            summary=(
                "The proposed update introduces a cross-file web contract that is not yet wired through the sibling HTML, CSS, and JS artifacts."
            ),
            confidence=0.92,
            blocking_issues=[item.summary for item in findings],
            preservation_risks=[],
            repair_hints=self._web_contract_repair_hints(
                path=path,
                findings=findings,
                inventories=inventories,
            ),
        )

    def _semantic_web_contract_review(self, session: SessionState) -> SemanticChangeReview | None:
        findings, file_hints, inventories = self._web_contract_analysis(session)
        if not findings:
            return None
        repair_hints: list[str] = []
        implicated_paths = file_hints or [item.path for item in session.changed_files[:4]]
        for candidate in implicated_paths:
            normalized_path = str(candidate or "").strip()
            if not normalized_path:
                continue
            path_findings = [
                item
                for item in findings
                if normalized_path in item.source_paths
            ]
            if not path_findings:
                continue
            repair_hints.extend(
                self._web_contract_repair_hints(
                    path=normalized_path,
                    findings=path_findings,
                    inventories=inventories,
                )
            )
        return SemanticChangeReview(
            requirements_satisfied=False,
            summary="The changed web artifacts still contain a cross-file web contract mismatch.",
            confidence=0.93,
            missing_requirements=[
                "Resolve the introduced cross-file web contract mismatch across the changed HTML, CSS, and JS artifacts."
            ],
            suspicious_issues=[item.summary for item in findings],
            file_hints=file_hints or [item.path for item in session.changed_files[:4]],
            repair_hints=list(dict.fromkeys(repair_hints))[:4]
            or [
                "Wire the same ids, selectors, and root-state tokens through the relevant HTML, CSS, and JS files before declaring the task complete.",
                "Remove duplicate or unused hooks instead of adding parallel DOM elements or selectors for the same behavior.",
            ],
        )

    def _clarification_response(self, route: RouterOutput, *, session: SessionState | None = None) -> str:
        language = self._session_language(session) if session is not None else self._language_for_text(route.user_goal)
        questions = route.clarification_questions[:3]
        if not questions:
            return self._localized_text(
                language,
                de="Ich brauche noch eine kurze Praezisierung, bevor ich sicher weitermachen kann.",
                en="I need one short clarification before I can continue safely.",
            )
        lines = [
            self._localized_text(
                language,
                de="Ich brauche noch eine kurze Praezisierung, bevor ich loslege.",
                en="I need one short clarification before I start.",
            ),
            "",
        ]
        for question in questions:
            lines.append(f"- {question}")
        return "\n".join(lines)

    def _unsafe_response(self, route: RouterOutput, *, session: SessionState | None = None) -> str:
        language = self._session_language(session) if session is not None else self._language_for_text(route.user_goal)
        if route.needs_clarification:
            return self._clarification_response(route, session=session)
        return self._localized_text(
            language,
            de="Ich fuehre das noch nicht aus, weil der Router die Anfrage noch nicht als sicher genug eingestuft hat.",
            en="I am not executing that yet because the router did not classify the request as safe enough.",
        )

    def _render_plan_response(self, route: RouterOutput, *, session: SessionState | None = None) -> str:
        language = self._session_language(session) if session is not None else self._language_for_text(route.user_goal)
        lines = [
            self._localized_text(language, de=f"Ziel: {route.user_goal}", en=f"Goal: {route.user_goal}"),
            "",
            self._localized_text(language, de="Vorgehen:", en="Plan:"),
        ]
        for item in route.action_plan:
            lines.append(f"{item.step}. {item.reason}")
        return "\n".join(lines)

    def _compose_user_response(self, route: RouterOutput, session: SessionState) -> str:
        if route.direct_response and not session.tool_calls and not session.changed_files:
            return route.direct_response
        if route.intent == RouteIntent.PLAN:
            return self._render_plan_response(route)
        prompt = final_response_prompt(route, session)
        model_name = self._lightweight_generation_model_name()
        self._log(
            "final_response_generation_started",
            model=model_name or self._primary_generation_model_name(),
        )
        outcome = invoke_model(
            lambda progress: self.llm.generate(
                prompt,
                model=model_name,
                timeout=max(self._llm_timeout(20), 20),
                total_timeout=max(self._llm_timeout(60), 60),
                num_ctx=1024,
                retries=0,
                progress_callback=progress,
            ),
            operation_name="final_response_generation",
            task_class="final_response_generation",
            attempt_number=1,
            capability_tier="tier_b" if model_name else "tier_a",
            recovery_strategy="final_response_generation",
            prompt_variant="compact",
            model_identifier=model_name or self._primary_generation_model_name(),
            backend_identifier=self._backend_identifier(),
            inactivity_timeout_seconds=max(self._llm_timeout(20), 20),
            total_timeout_seconds=max(self._llm_timeout(60), 60),
            context_pressure_estimate=estimate_context_pressure(prompt_chars=len(prompt)),
            event_callback=self._progress_logger("final_response_generation_progress"),
        )
        if outcome.exception is None:
            response = self._strip_code_fences(str(outcome.value or "")).strip()
            if response:
                self._log("final_response_generation_finished", characters=len(response))
                self._append_runtime_execution(
                    session,
                    build_execution_run_record(
                        operation_name="final_response_generation",
                        task_class="final_response_generation",
                        final_state="completed",
                        capability_tier="tier_b" if model_name else "tier_a",
                        recovery_strategy="final_response_generation",
                        degraded=bool(model_name),
                        honest_blocked=False,
                        artifact_bytes_generated=len(response),
                        validation_possible=False,
                        summary="Final user response was generated by the model runtime.",
                        attempts=[outcome.attempt],
                    ),
                )
                return response
        else:
            issue = outcome.attempt.failure
            self._log(
                "final_response_generation_error",
                error=str(outcome.exception),
                reason=issue.reason if issue is not None else "runtime_error",
                had_progress=issue.had_progress if issue is not None else False,
                partial_characters=issue.characters if issue is not None else 0,
            )
        deterministic = self._deterministic_final_response(route, session)
        self._log(
            "final_response_generation_finished",
            characters=len(deterministic),
            source="deterministic",
        )
        self._append_runtime_execution(
            session,
            build_execution_run_record(
                operation_name="final_response_generation",
                task_class="final_response_generation",
                final_state="degraded_success",
                capability_tier="tier_d",
                recovery_strategy="deterministic_fallback",
                degraded=True,
                honest_blocked=False,
                artifact_bytes_generated=len(deterministic),
                validation_possible=False,
                summary="Final user response used the deterministic fallback because the backend generation step did not complete cleanly.",
                attempts=[outcome.attempt],
            ),
        )
        return deterministic

    def _final_decision(
        self,
        thought_summary: str,
        final_response: str,
    ) -> AgentDecision:
        return AgentDecision(
            thought_summary=thought_summary,
            action_type=AgentActionType.FINAL,
            tool_name=None,
            tool_args={},
            expected_outcome="Return a user-facing response.",
            final_response=final_response,
        )

    def _record_generation_blocker(
        self,
        session: SessionState,
        message: str,
        *,
        stop_reason: str,
        failure_class: str | None = None,
    ) -> None:
        text = str(message or "").strip()
        if not text:
            return
        if text not in session.blockers:
            session.blockers.append(text)
            session.blockers = session.blockers[-10:]
        session.last_error = text
        session.stop_reason = stop_reason
        self._log(
            "final_block_reason",
            stop_reason=stop_reason,
            failure_class=failure_class or stop_reason,
            message=text,
        )

    def _continue_after_nonblocking_update_target_failure(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        target: str,
        stop_reason: str,
        repair_context: ValidationFailureEvidence | None,
    ) -> AgentDecision | None:
        if repair_context is not None or stop_reason not in {
            "update_review_rejected",
            "no_effective_change",
        }:
            return None
        if route.intent != RouteIntent.UPDATE:
            return None

        deferred_targets = self._deferred_update_targets(session)
        if target in deferred_targets:
            return None

        alternative_targets = self._ordered_remaining_update_targets(
            route,
            session,
            exclude={target},
        )
        validation_command = self._pick_validation_command(session) if session.changed_files else None
        if not alternative_targets and validation_command is None:
            return None

        note = f"{DEFERRED_UPDATE_TARGET_NOTE_PREFIX}{target}"
        if note not in session.notes:
            session.notes.append(note)
            session.notes = session.notes[-40:]

        if alternative_targets:
            self._log(
                "update_target_deferred",
                path=target,
                reason=stop_reason,
                next_target=alternative_targets[0],
            )
            return self.execute_action_from_plan(route, session)

        self._log(
            "update_target_deferred",
            path=target,
            reason=stop_reason,
            next_action="validation",
            command=validation_command,
        )
        return self._validation_decision(
            "A review-blocked target was deferred so the remaining validation can reveal the next concrete repair step.",
            validation_command,
        )

    def _append_runtime_execution(self, session: SessionState | None, record: dict[str, object]) -> None:
        if session is None:
            return
        session.runtime_executions.append(dict(record))
        session.runtime_executions = session.runtime_executions[-20:]

    def _read_paths(self, session: SessionState) -> list[str]:
        return [
            str(item.tool_args.get("path") or "").strip()
            for item in session.tool_calls
            if item.tool_name == "read_file" and str(item.tool_args.get("path") or "").strip()
        ]

    def _searched_queries(self, session: SessionState) -> list[str]:
        return [
            str(item.tool_args.get("query") or "").strip()
            for item in session.tool_calls
            if item.tool_name == "search_in_files" and str(item.tool_args.get("query") or "").strip()
        ]

    def _preferred_extension(self, route: RouterOutput) -> str:
        for extension in route.relevant_extensions:
            normalized = str(extension or "").strip()
            if normalized.startswith("."):
                return normalized
        primary = self._primary_target_name(route)
        if "." in primary:
            return Path(primary).suffix or ".txt"
        return ".txt"

    def _primary_target_name(self, route: RouterOutput) -> str:
        return str(route.entities.target_name or route.requested_outcome or "generated_output")

    def _default_new_path(self, route: RouterOutput) -> str:
        target = self._path_seed_from_route(route).lower()
        aliases = {
            "tictactoe": "tic_tac_toe",
            "tic_tac_toe": "tic_tac_toe",
        }
        target = aliases.get(target, target)
        slug = re.sub(r"[^a-z0-9]+", "_", target).strip("_") or "generated_output"
        extension = self._preferred_extension(route)
        if not slug.endswith(extension):
            return f"{slug}{extension}"
        return slug

    def _empty_workspace_default_path(self, route: RouterOutput) -> str:
        default_path = self._default_new_path(route)
        preferred_extension = self._preferred_extension(route)
        explicit_target = str(route.entities.target_name or "").strip()
        if preferred_extension == ".html" and (
            not explicit_target or not self._looks_like_path(explicit_target)
        ):
            return "index.html"
        default_name = Path(default_path).name.lower()
        if default_name not in {"generated_output", "generated_output.txt", "app.py", "app.js", "app.ts"}:
            return default_path
        if not str(route.entities.target_name or "").strip() or not self._looks_like_path(str(route.entities.target_name or "").strip()):
            conventional = {
                ".html": "index.html",
                ".css": "styles.css",
                ".js": "app.js",
                ".ts": "app.ts",
            }.get(preferred_extension)
            if conventional is not None:
                return conventional
        return default_path

    def _path_seed_from_route(self, route: RouterOutput) -> str:
        explicit = str(route.entities.target_name or "").strip()
        if explicit:
            return explicit

        prioritized_source = " ".join(route.search_terms).strip()
        token_source = prioritized_source or " ".join([route.user_goal, route.requested_outcome])
        token_source = token_source.lower()
        stopwords = {
            "a",
            "an",
            "and",
            "app",
            "artifact",
            "bau",
            "bitte",
            "build",
            "create",
            "datei",
            "der",
            "die",
            "ein",
            "eine",
            "einen",
            "erstell",
            "file",
            "fuer",
            "game",
            "implementation",
            "implement",
            "in",
            "javascript",
            "mach",
            "mir",
            "mit",
            "module",
            "new",
            "or",
            "python",
            "produce",
            "requested",
            "request",
            "result",
            "safely",
            "schreib",
            "script",
            "spiel",
            "standalone",
            "the",
            "this",
            "tool",
            "typescript",
            "und",
            "user",
        }
        tokens: list[str] = []
        for raw in re.split(r"[^a-z0-9]+", token_source):
            token = raw.strip()
            if len(token) < 2 or token in stopwords or token in tokens:
                continue
            tokens.append(token)
        if tokens:
            return "_".join(tokens[:4])
        return self._primary_target_name(route)

    def _template_fallback_content(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str | None = None,
    ) -> str | None:
        if not path.endswith(".py"):
            return None

        if not self._looks_like_tic_tac_toe_request(
            route,
            session,
            path=path,
            current_content=current_content,
        ):
            return None

        return self._tic_tac_toe_template(
            versus_computer=self._wants_computer_opponent(
                route,
                session,
                current_content=current_content,
            )
        )

    def _looks_like_tic_tac_toe_request(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str | None = None,
    ) -> bool:
        task_state = session.task_state
        normalized = " ".join(
            [
                str(route.user_goal or "").lower(),
                str(task_state.root_goal if task_state is not None else "").lower(),
                str(task_state.active_goal if task_state is not None else "").lower(),
                str(task_state.output_expectation if task_state is not None else "").lower(),
                str(route.entities.target_name or "").lower(),
                " ".join(item.lower() for item in route.search_terms),
                str(path or "").lower(),
                str(current_content or "").lower(),
            ]
        )
        return any(
            marker in normalized
            for marker in ("tic tac toe", "tictactoe", "tic_tac_toe")
        )

    def _wants_computer_opponent(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        current_content: str | None = None,
    ) -> bool:
        task_state = session.task_state
        normalized_request = " ".join(
            [
                str(route.user_goal or "").lower(),
                str(route.requested_outcome or "").lower(),
                str(route.entities.target_name or "").lower(),
                " ".join(item.lower() for item in route.search_terms),
                str(task_state.root_goal if task_state is not None else "").lower(),
                str(task_state.active_goal if task_state is not None else "").lower(),
                str(task_state.output_expectation if task_state is not None else "").lower(),
            ]
        )
        normalized_existing = str(current_content or "").lower()
        if any(
            marker in normalized_request
            for marker in (
                "computer",
                "bot",
                "ki",
                "ai",
                "singleplayer",
                "single player",
                "gegner",
            )
        ):
            return True
        if any(
            marker in normalized_request
            for marker in (
                "2 spieler",
                "zwei spieler",
                "two player",
                "multiplayer",
                "zu zweit",
                "gegen einen freund",
                "gegen einen menschen",
            )
        ):
            return False
        if "computer" in normalized_existing or "bot" in normalized_existing:
            return True
        if any(
            marker in normalized_existing
            for marker in ("2 spieler", "zwei spieler", "two player", "spieler o")
        ):
            return False
        return True

    def _tic_tac_toe_template(self, *, versus_computer: bool) -> str:
        if versus_computer:
            return (
                "import random\n"
                "\n"
                "\n"
                "WINNING_LINES = [\n"
                "    (0, 1, 2),\n"
                "    (3, 4, 5),\n"
                "    (6, 7, 8),\n"
                "    (0, 3, 6),\n"
                "    (1, 4, 7),\n"
                "    (2, 5, 8),\n"
                "    (0, 4, 8),\n"
                "    (2, 4, 6),\n"
                "]\n"
                "\n"
                "\n"
                "def render_board(board):\n"
                "    cells = [str(index + 1) if cell == ' ' else cell for index, cell in enumerate(board)]\n"
                "    rows = [\n"
                "        f\" {cells[0]} | {cells[1]} | {cells[2]} \",\n"
                "        f\" {cells[3]} | {cells[4]} | {cells[5]} \",\n"
                "        f\" {cells[6]} | {cells[7]} | {cells[8]} \",\n"
                "    ]\n"
                "    return \"\\n---+---+---\\n\".join(rows)\n"
                "\n"
                "\n"
                "def check_winner(board, marker):\n"
                "    return any(all(board[position] == marker for position in line) for line in WINNING_LINES)\n"
                "\n"
                "\n"
                "def board_full(board):\n"
                "    return all(cell != ' ' for cell in board)\n"
                "\n"
                "\n"
                "def ask_move(board):\n"
                "    while True:\n"
                "        choice = input('Du bist X. Waehle ein Feld (1-9): ').strip()\n"
                "        if not choice.isdigit():\n"
                "            print('Bitte gib eine Zahl von 1 bis 9 ein.')\n"
                "            continue\n"
                "        position = int(choice) - 1\n"
                "        if position < 0 or position > 8:\n"
                "            print('Die Zahl muss zwischen 1 und 9 liegen.')\n"
                "            continue\n"
                "        if board[position] != ' ':\n"
                "            print('Dieses Feld ist schon belegt.')\n"
                "            continue\n"
                "        return position\n"
                "\n"
                "\n"
                "def choose_computer_move(board):\n"
                "    available = [index for index, value in enumerate(board) if value == ' ']\n"
                "    for marker in ('O', 'X'):\n"
                "        for move in available:\n"
                "            board[move] = marker\n"
                "            wins = check_winner(board, marker)\n"
                "            board[move] = ' '\n"
                "            if wins:\n"
                "                return move\n"
                "    if 4 in available:\n"
                "        return 4\n"
                "    corners = [move for move in available if move in {0, 2, 6, 8}]\n"
                "    if corners:\n"
                "        return random.choice(corners)\n"
                "    return random.choice(available)\n"
                "\n"
                "\n"
                "def play_game():\n"
                "    board = [' '] * 9\n"
                "    print('\\nDu spielst gegen den Computer.')\n"
                "    while True:\n"
                "        print()\n"
                "        print(render_board(board))\n"
                "        print()\n"
                "        player_move = ask_move(board)\n"
                "        board[player_move] = 'X'\n"
                "        if check_winner(board, 'X'):\n"
                "            print(render_board(board))\n"
                "            print('\\nDu gewinnst!')\n"
                "            return\n"
                "        if board_full(board):\n"
                "            print(render_board(board))\n"
                "            print('\\nUnentschieden!')\n"
                "            return\n"
                "\n"
                "        computer_move = choose_computer_move(board)\n"
                "        board[computer_move] = 'O'\n"
                "        print(f'Computer waehlt Feld {computer_move + 1}.')\n"
                "        if check_winner(board, 'O'):\n"
                "            print()\n"
                "            print(render_board(board))\n"
                "            print('\\nDer Computer gewinnt.')\n"
                "            return\n"
                "        if board_full(board):\n"
                "            print()\n"
                "            print(render_board(board))\n"
                "            print('\\nUnentschieden!')\n"
                "            return\n"
                "\n"
                "\n"
                "def main():\n"
                "    print('Willkommen zu Tic Tac Toe!')\n"
                "    while True:\n"
                "        play_game()\n"
                "        again = input('\\nNoch eine Runde? (j/n): ').strip().lower()\n"
                "        if again not in {'j', 'ja', 'y', 'yes'}:\n"
                "            print('Bis zum naechsten Mal!')\n"
                "            break\n"
                "\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            )

        return (
            "WINNING_LINES = [\n"
            "    (0, 1, 2),\n"
            "    (3, 4, 5),\n"
            "    (6, 7, 8),\n"
            "    (0, 3, 6),\n"
            "    (1, 4, 7),\n"
            "    (2, 5, 8),\n"
            "    (0, 4, 8),\n"
            "    (2, 4, 6),\n"
            "]\n"
            "\n"
            "\n"
            "def render_board(board):\n"
            "    cells = [str(index + 1) if cell == ' ' else cell for index, cell in enumerate(board)]\n"
            "    rows = [\n"
            "        f\" {cells[0]} | {cells[1]} | {cells[2]} \",\n"
            "        f\" {cells[3]} | {cells[4]} | {cells[5]} \",\n"
            "        f\" {cells[6]} | {cells[7]} | {cells[8]} \",\n"
            "    ]\n"
            "    return \"\\n---+---+---\\n\".join(rows)\n"
            "\n"
            "\n"
            "def check_winner(board, marker):\n"
            "    return any(all(board[position] == marker for position in line) for line in WINNING_LINES)\n"
            "\n"
            "\n"
            "def board_full(board):\n"
            "    return all(cell != ' ' for cell in board)\n"
            "\n"
            "\n"
            "def ask_move(board, player):\n"
            "    while True:\n"
            "        choice = input(f\"Spieler {player}, waehle ein Feld (1-9): \").strip()\n"
            "        if not choice.isdigit():\n"
            "            print('Bitte gib eine Zahl von 1 bis 9 ein.')\n"
            "            continue\n"
            "        position = int(choice) - 1\n"
            "        if position < 0 or position > 8:\n"
            "            print('Die Zahl muss zwischen 1 und 9 liegen.')\n"
            "            continue\n"
            "        if board[position] != ' ':\n"
            "            print('Dieses Feld ist schon belegt.')\n"
            "            continue\n"
            "        return position\n"
            "\n"
            "\n"
            "def play_game():\n"
            "    board = [' '] * 9\n"
            "    current_player = 'X'\n"
            "    while True:\n"
            "        print()\n"
            "        print(render_board(board))\n"
            "        print()\n"
            "        move = ask_move(board, current_player)\n"
            "        board[move] = current_player\n"
            "\n"
            "        if check_winner(board, current_player):\n"
            "            print()\n"
            "            print(render_board(board))\n"
            "            print(f\"\\nSpieler {current_player} gewinnt!\")\n"
            "            return\n"
            "\n"
            "        if board_full(board):\n"
            "            print()\n"
            "            print(render_board(board))\n"
            "            print('\\nUnentschieden!')\n"
            "            return\n"
            "\n"
            "        current_player = 'O' if current_player == 'X' else 'X'\n"
            "\n"
            "\n"
            "def main():\n"
            "    print('Willkommen zu Tic Tac Toe!')\n"
            "    while True:\n"
            "        play_game()\n"
            "        again = input('\\nNoch eine Runde? (j/n): ').strip().lower()\n"
            "        if again not in {'j', 'ja', 'y', 'yes'}:\n"
            "            print('Bis zum naechsten Mal!')\n"
            "            break\n"
            "\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )

    def _sanitize_generated_path(
        self,
        raw: str,
        preferred_extension: str,
    ) -> str | None:
        text = self._strip_code_fences(raw).strip().splitlines()[0].strip("`'\" ")
        if not text:
            return None
        text = text.lstrip("./").replace("\\", "/")
        parts = [part for part in text.split("/") if part not in {"", ".", ".."}]
        text = "/".join(parts)
        if not text:
            return None
        if preferred_extension and not text.endswith(preferred_extension):
            if "." not in Path(text).name:
                text = f"{text}{preferred_extension}"
        return text

    def _looks_like_path(self, value: str) -> bool:
        text = str(value or "").strip()
        return bool(re.search(r"[\w./-]+\.[a-z0-9]{1,8}$", text, re.IGNORECASE))

    def _unique_paths(self, values: list[str]) -> list[str]:
        unique: list[str] = []
        for raw in values:
            text = str(raw or "").strip()
            if not text or text in unique:
                continue
            unique.append(text)
        return unique[:24]

    def _strip_code_fences(self, text: str) -> str:
        cleaned = str(text or "").strip()
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            opening = lines[0].strip()
            closing = lines[-1].strip()
            if re.fullmatch(r"(```+|~~~+)[A-Za-z0-9_.+-]*", opening) and (
                closing == "```" or closing == "~~~"
            ):
                return "\n".join(lines[1:-1]).strip()
        return cleaned

    def _llm_timeout(self, minimum: int) -> int:
        configured = getattr(getattr(self.llm, "config", None), "llm_timeout", minimum)
        return max(int(configured), minimum)

    def _llm_num_ctx(self, minimum: int) -> int:
        configured = getattr(getattr(self.llm, "config", None), "ollama_num_ctx", minimum)
        return max(int(configured), minimum)

    def _primary_generation_model_name(self) -> str | None:
        config = getattr(self.llm, "config", None)
        if config is None:
            return None
        candidate = str(getattr(config, "model_name", "") or "").strip()
        return candidate or None

    def _backend_identifier(self) -> str:
        return "ollama"

    def _lightweight_generation_model_name(self) -> str | None:
        config = getattr(self.llm, "config", None)
        if config is None:
            return None
        candidate = str(getattr(config, "router_model_name", "") or "").strip()
        primary = str(getattr(config, "model_name", "") or "").strip()
        if not candidate or candidate == primary:
            return None
        return candidate

    def _live_generation_model_candidates(self) -> list[str]:
        list_models = getattr(self.llm, "list_models_safe", None)
        if not callable(list_models):
            return []
        try:
            raw_models = list_models()
        except Exception:
            return []
        candidates: list[str] = []
        for item in raw_models or []:
            if isinstance(item, dict):
                text = str(item.get("name") or item.get("model") or "").strip()
            else:
                text = str(item or "").strip()
            if text and text not in candidates:
                candidates.append(text)
        return candidates

    def _generation_recovery_model_name(self) -> str | None:
        primary = self._primary_generation_model_name()
        config = getattr(self.llm, "config", None)
        candidate_pool: list[str] = []
        if config is not None:
            raw_candidates = getattr(config, "model_candidates", ()) or ()
            if isinstance(raw_candidates, str):
                raw_candidates = [raw_candidates]
            for raw in raw_candidates:
                text = str(raw or "").strip()
                if text and text not in candidate_pool:
                    candidate_pool.append(text)
        for candidate in self._live_generation_model_candidates():
            if candidate not in candidate_pool:
                candidate_pool.append(candidate)
        recovery_model = availability_recovery_model(primary, candidate_pool)
        return str(recovery_model or "").strip() or None

    def _task_state_semantics_limited(self, session: SessionState) -> bool:
        task_state = session.task_state
        if task_state is None:
            return False
        if bool(getattr(task_state, "secondary_semantics_limited", False)):
            return True
        resolution = str(getattr(task_state, "semantic_resolution", "") or "").strip()
        return resolution in {"reduced_model", "minimal_inference", "blocked"}

    def _content_generation_model_name(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str | None,
        repair_context: ValidationFailureEvidence | None = None,
    ) -> str | None:
        lightweight = self._lightweight_generation_model_name()
        if lightweight is None:
            return None
        if repair_context is not None:
            if (
                current_content is None
                and self._should_prefer_lightweight_missing_artifact_generation(
                    route,
                    session,
                    path=path,
                    repair_context=repair_context,
                )
            ):
                return lightweight
            return None
        if current_content is None and self._should_prefer_lightweight_missing_artifact_generation(
            route,
            session,
            path=path,
            repair_context=repair_context,
        ):
            return lightweight
        if self._should_prefer_lightweight_update_generation(
            route,
            session,
            path=path,
            current_content=current_content,
        ):
            return lightweight
        return None

    def _focused_update_scope_supports_small_model(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
    ) -> bool:
        target_paths = self._actionable_explicit_target_paths(route, session)
        if not target_paths or path not in target_paths:
            return False
        if len(target_paths) > MAX_LIGHTWEIGHT_UPDATE_TARGETS:
            return False

        suffix = Path(path).suffix.lower()
        if suffix in {".html", ".htm", ".css", ".js", ".jsx", ".ts", ".tsx"}:
            return True

        companion_targets = [candidate for candidate in target_paths if candidate != path]
        return any(
            self._path_is_test_like(candidate)
            or self.validation_planner._is_documentation_path(candidate)
            for candidate in companion_targets
        )

    def _content_generation_prompt_variant(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str | None,
        model_name: str | None,
        repair_context: ValidationFailureEvidence | None,
    ) -> str:
        lightweight = self._lightweight_generation_model_name()
        if repair_context is not None:
            return "compact"
        if (
            current_content is not None
            and self._should_prefer_compact_primary_update_generation(
                route,
                session,
                path=path,
                current_content=current_content,
            )
        ):
            return "compact"
        if (
            model_name is None
            and current_content is None
            and repair_context is None
            and (
                self._should_prefer_compact_primary_create_generation(
                    route,
                    session,
                    path=path,
                )
                or self._should_force_compact_create_after_semantic_fallback(
                    route,
                    session,
                    path=path,
                )
            )
        ):
            return "compact"
        if (
            model_name is not None
            and model_name == lightweight
            and (
                (
                    current_content is not None
                    and self._should_prefer_lightweight_update_generation(
                        route,
                        session,
                        path=path,
                        current_content=current_content,
                    )
                )
                or (
                    current_content is None
                    and self._should_prefer_lightweight_missing_artifact_generation(
                        route,
                        session,
                        path=path,
                        repair_context=repair_context,
                    )
                )
            )
        ):
            return "compact"
        return "full"

    def _should_prefer_compact_primary_update_generation(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str | None,
    ) -> bool:
        if not self._supports_update_style_existing_artifact(route, current_content=current_content):
            return False
        if route.needs_clarification or not route.safe_to_execute:
            return False

        target_paths = self._actionable_explicit_target_paths(route, session)
        if target_paths and path not in target_paths:
            return False
        if len(target_paths) > MAX_LIGHTWEIGHT_UPDATE_TARGETS:
            return False
        if (
            not self._path_matches_explicit_request(session, path)
            and not list(route.entities.constraints or [])
            and not self._focused_update_scope_supports_small_model(
                route,
                session,
                path=path,
            )
        ):
            return False

        suffix = Path(path).suffix.lower()
        if suffix not in LIGHTWEIGHT_UPDATE_SUFFIXES:
            return False
        if len(current_content) > MAX_LIGHTWEIGHT_UPDATE_CHARS:
            return False
        if current_content.count("\n") + 1 > MAX_LIGHTWEIGHT_UPDATE_LINES:
            return False
        if len(session.changed_files) > MAX_LIGHTWEIGHT_UPDATE_CHANGED_FILES:
            return False

        snapshot = session.workspace_snapshot
        if snapshot is not None and snapshot.file_count > MAX_LIGHTWEIGHT_UPDATE_WORKSPACE_FILES:
            return False

        focus = _artifact_scoped_focus(route, session, path, current_content=current_content)
        write_requirements = focus.get("current_write_requirements") or []
        if not isinstance(write_requirements, list) or not write_requirements:
            return False
        if len(write_requirements) > 4:
            return False

        task_state = session.task_state
        route_confidence = float(route.confidence or 0.0)
        if task_state is None:
            return route_confidence >= MIN_COMPACT_PRIMARY_UPDATE_CONFIDENCE
        if task_state.needs_clarification:
            return False
        if task_state.ambiguity_level == "high" or task_state.risk_level == "high":
            return False
        return max(float(task_state.confidence or 0.0), route_confidence) >= MIN_COMPACT_PRIMARY_UPDATE_CONFIDENCE

    def _should_force_compact_create_after_semantic_fallback(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
    ) -> bool:
        task_state = session.task_state
        if task_state is None or not self._task_state_semantics_limited(session):
            return False
        if route.needs_clarification or not route.safe_to_execute:
            return False
        next_action = str(task_state.next_action or task_state.next_best_action or "").strip().lower()
        if next_action != "create":
            return False

        explicit_targets = self._explicit_target_paths(route, session)
        if explicit_targets and path not in explicit_targets:
            return False
        if len(explicit_targets) > MAX_LIGHTWEIGHT_UPDATE_TARGETS:
            return False

        absolute = Path(session.workspace_root, path)
        if absolute.exists() or absolute.is_dir():
            return False

        suffix = Path(path).suffix.lower()
        if suffix not in LIGHTWEIGHT_UPDATE_SUFFIXES:
            return False
        if len(session.changed_files) > MAX_LIGHTWEIGHT_UPDATE_CHANGED_FILES:
            return False

        snapshot = session.workspace_snapshot
        if snapshot is not None:
            if snapshot.file_count > MAX_LIGHTWEIGHT_UPDATE_WORKSPACE_FILES:
                return False
            if snapshot.file_count > 0 and len(snapshot.important_files) > 16:
                return False

        if task_state.needs_clarification:
            return False
        if task_state.ambiguity_level == "high" or task_state.risk_level == "high":
            return False
        return True

    def _should_prefer_compact_primary_create_generation(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
    ) -> bool:
        if route.intent != RouteIntent.CREATE or route.needs_clarification or not route.safe_to_execute:
            return False

        explicit_targets = self._explicit_target_paths(route, session)
        if explicit_targets and path not in explicit_targets:
            return False
        if len(explicit_targets) > MAX_LIGHTWEIGHT_UPDATE_TARGETS:
            return False

        absolute = Path(session.workspace_root, path)
        if absolute.exists() or absolute.is_dir():
            return False

        suffix = Path(path).suffix.lower()
        if suffix not in LIGHTWEIGHT_UPDATE_SUFFIXES:
            return False
        if len(session.changed_files) > MAX_LIGHTWEIGHT_UPDATE_CHANGED_FILES:
            return False

        snapshot = session.workspace_snapshot
        if snapshot is not None:
            if snapshot.file_count > MAX_LIGHTWEIGHT_UPDATE_WORKSPACE_FILES:
                return False
            if snapshot.file_count > 0 and len(snapshot.important_files) > 16:
                return False

        task_state = session.task_state
        if task_state is None:
            return float(route.confidence or 0.0) >= 0.86
        if task_state.needs_clarification:
            return False
        if task_state.ambiguity_level == "high" or task_state.risk_level != "low":
            return False
        if float(task_state.confidence or 0.0) < MIN_LIGHTWEIGHT_UPDATE_CONFIDENCE:
            return False
        if len(task_state.target_artifacts) > MAX_LIGHTWEIGHT_UPDATE_TARGETS:
            return False
        return True

    def _should_prefer_lightweight_missing_artifact_generation(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        repair_context: ValidationFailureEvidence | None,
    ) -> bool:
        lightweight = self._lightweight_generation_model_name()
        if lightweight is None:
            return False
        if route.needs_clarification or not route.safe_to_execute:
            return False
        if route.intent not in {RouteIntent.UPDATE, RouteIntent.DEBUG, RouteIntent.CREATE}:
            return False

        explicit_targets = self._explicit_target_paths(route, session)
        if repair_context is None:
            if explicit_targets and path not in explicit_targets:
                return False
            if len(explicit_targets) > MAX_LIGHTWEIGHT_UPDATE_TARGETS:
                return False

        prefer_after_runtime_recovery = (
            repair_context is None
            and route.intent == RouteIntent.CREATE
            and bool(session.changed_files)
            and self._recent_primary_content_generation_recovered_on_lightweight(session)
        )
        prefer_small_low_risk_create = (
            repair_context is None
            and self._should_prefer_lightweight_simple_create(
                route,
                session,
                path=path,
            )
        )
        if repair_context is None and not (
            prefer_after_runtime_recovery or prefer_small_low_risk_create
        ):
            return False

        if repair_context is not None:
            implicated_paths = self._unique_paths([*repair_context.artifact_paths, *repair_context.file_hints])
            if implicated_paths and path not in implicated_paths:
                return False

        absolute = Path(session.workspace_root, path)
        if absolute.exists() or absolute.is_dir():
            return False

        suffix = Path(path).suffix.lower()
        if suffix not in LIGHTWEIGHT_UPDATE_SUFFIXES:
            return False
        if len(session.changed_files) > MAX_LIGHTWEIGHT_UPDATE_CHANGED_FILES:
            return False

        snapshot = session.workspace_snapshot
        if snapshot is not None and snapshot.file_count > MAX_LIGHTWEIGHT_UPDATE_WORKSPACE_FILES:
            return False
        if repair_context is not None and not self._repair_related_existing_context_paths(session, repair_context, exclude=path):
            return False

        task_state = session.task_state
        if task_state is not None:
            if task_state.needs_clarification:
                return False
            if task_state.ambiguity_level == "high" or task_state.risk_level == "high":
                return False
            if float(task_state.confidence or 0.0) < MIN_LIGHTWEIGHT_UPDATE_CONFIDENCE:
                return False

        return True

    def _should_prefer_lightweight_simple_create(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
    ) -> bool:
        lightweight = self._lightweight_generation_model_name()
        if lightweight is None:
            return False
        if route.intent != RouteIntent.CREATE or route.needs_clarification or not route.safe_to_execute:
            return False
        if self._task_state_semantics_limited(session):
            return False

        explicit_targets = self._explicit_target_paths(route, session)
        if explicit_targets and path not in explicit_targets:
            return False
        if len(explicit_targets) > 3:
            return False

        absolute = Path(session.workspace_root, path)
        if absolute.exists() or absolute.is_dir():
            return False

        suffix = Path(path).suffix.lower()
        if suffix not in LIGHTWEIGHT_UPDATE_SUFFIXES:
            return False
        if len(session.changed_files) > MAX_LIGHTWEIGHT_UPDATE_CHANGED_FILES:
            return False

        snapshot = session.workspace_snapshot
        if snapshot is not None:
            if snapshot.file_count > 10:
                return False
            if snapshot.file_count > 0 and len(snapshot.important_files) > 6:
                return False

        task_state = session.task_state
        if task_state is not None:
            if task_state.needs_clarification:
                return False
            if task_state.ambiguity_level == "high" or task_state.risk_level != "low":
                return False
            if float(task_state.confidence or 0.0) < MIN_LIGHTWEIGHT_UPDATE_CONFIDENCE:
                return False
            if len(task_state.target_artifacts) > 3:
                return False
            return True

        return float(route.confidence or 0.0) >= MIN_LIGHTWEIGHT_UPDATE_CONFIDENCE

    def _recent_primary_content_generation_recovered_on_lightweight(
        self,
        session: SessionState,
    ) -> bool:
        primary = self._primary_generation_model_name()
        lightweight = self._lightweight_generation_model_name()
        if primary is None or lightweight is None:
            return False

        for execution in reversed(session.runtime_executions[-6:]):
            if str(execution.get("task_class") or "").strip() != "content_generation":
                continue
            attempts = execution.get("attempts") or []
            if not isinstance(attempts, list):
                continue

            saw_primary_no_start = False
            saw_lightweight_completion = False
            for attempt in attempts:
                if not isinstance(attempt, dict):
                    continue
                model_name = str(attempt.get("model_identifier") or "").strip()
                if model_name == primary and self._attempt_record_dict_has_no_start_failure(attempt):
                    saw_primary_no_start = True
                if (
                    model_name == lightweight
                    and attempt.get("failure") is None
                    and str(attempt.get("state") or "").strip() == "completed"
                ):
                    saw_lightweight_completion = True
            if saw_primary_no_start and saw_lightweight_completion:
                return True
        return False

    def _attempt_record_dict_has_no_start_failure(self, attempt: dict[str, object]) -> bool:
        failure = attempt.get("failure")
        if not isinstance(failure, dict):
            return False
        if str(failure.get("failure_class") or "").strip() != "startup_timeout":
            return False
        if bool(failure.get("first_output_received")) or bool(failure.get("had_progress")):
            return False
        if int(failure.get("characters") or 0) > 0:
            return False
        if str(failure.get("partial_text") or "").strip():
            return False
        return True

    def _supports_update_style_existing_artifact(
        self,
        route: RouterOutput,
        *,
        current_content: str | None,
    ) -> bool:
        if current_content is None:
            return False
        return route.intent in {RouteIntent.UPDATE, RouteIntent.DEBUG, RouteIntent.CREATE}

    def _should_prefer_lightweight_update_generation(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str | None,
    ) -> bool:
        lightweight = self._lightweight_generation_model_name()
        if lightweight is None:
            return False
        if (
            not self._supports_update_style_existing_artifact(route, current_content=current_content)
            or route.needs_clarification
            or not route.safe_to_execute
        ):
            return False

        target_paths = self._actionable_explicit_target_paths(route, session)
        if target_paths and path not in target_paths:
            return False
        if len(target_paths) > MAX_LIGHTWEIGHT_UPDATE_TARGETS:
            return False
        if (
            not self._path_matches_explicit_request(session, path)
            and not list(route.entities.constraints or [])
            and not self._focused_update_scope_supports_small_model(
                route,
                session,
                path=path,
            )
        ):
            return False

        suffix = Path(path).suffix.lower()
        if suffix not in LIGHTWEIGHT_UPDATE_SUFFIXES:
            return False
        if len(current_content) > MAX_LIGHTWEIGHT_UPDATE_CHARS:
            return False
        if current_content.count("\n") + 1 > MAX_LIGHTWEIGHT_UPDATE_LINES:
            return False
        if len(session.changed_files) > MAX_LIGHTWEIGHT_UPDATE_CHANGED_FILES:
            return False

        snapshot = session.workspace_snapshot
        if snapshot is not None and snapshot.file_count > MAX_LIGHTWEIGHT_UPDATE_WORKSPACE_FILES:
            return False

        task_state = session.task_state
        if task_state is not None:
            if task_state.needs_clarification:
                return False
            if task_state.ambiguity_level == "high" or task_state.risk_level == "high":
                return False
            if float(task_state.confidence or 0.0) < MIN_LIGHTWEIGHT_UPDATE_CONFIDENCE:
                return False

        focus = _artifact_scoped_focus(route, session, path, current_content=current_content)
        write_requirements = focus.get("current_write_requirements") or []
        if not isinstance(write_requirements, list) or not write_requirements:
            return False
        if len(write_requirements) > 4:
            return False

        return True

    def _should_keep_update_review_compact(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str | None,
    ) -> bool:
        if not self._supports_update_style_existing_artifact(route, current_content=current_content):
            return False
        if route.needs_clarification or not route.safe_to_execute:
            return False

        target_paths = self._actionable_explicit_target_paths(route, session)
        if target_paths and path not in target_paths:
            return False
        if len(target_paths) > MAX_LIGHTWEIGHT_UPDATE_TARGETS:
            return False

        suffix = Path(path).suffix.lower()
        if suffix not in LIGHTWEIGHT_UPDATE_SUFFIXES:
            return False
        if len(current_content) > MAX_LIGHTWEIGHT_UPDATE_CHARS:
            return False
        if current_content.count("\n") + 1 > MAX_LIGHTWEIGHT_UPDATE_LINES:
            return False
        if len(session.changed_files) > MAX_LIGHTWEIGHT_UPDATE_CHANGED_FILES:
            return False

        snapshot = session.workspace_snapshot
        if snapshot is not None and snapshot.file_count > MAX_LIGHTWEIGHT_UPDATE_WORKSPACE_FILES:
            return False

        task_state = session.task_state
        if task_state is not None:
            if task_state.needs_clarification:
                return False
            if task_state.ambiguity_level == "high" or task_state.risk_level == "high":
                return False

        return True

    def _content_generation_num_ctx(self, prompt_variant: str) -> int:
        if prompt_variant == "compact":
            return min(self._llm_num_ctx(2048), 2048)
        return min(self._llm_num_ctx(4096), 4096)

    def _should_prefer_lightweight_semantic_review(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        artifacts: list[dict[str, object]],
    ) -> bool:
        lightweight = self._lightweight_generation_model_name()
        if lightweight is None:
            return False
        if route.needs_clarification or not route.safe_to_execute:
            return False
        if route.intent not in {RouteIntent.UPDATE, RouteIntent.DEBUG, RouteIntent.CREATE}:
            return False
        if not session.changed_files or len(session.changed_files) > MAX_LIGHTWEIGHT_UPDATE_CHANGED_FILES:
            return False
        if len(artifacts) > MAX_LIGHTWEIGHT_UPDATE_CHANGED_FILES:
            return False
        if session.validation_status != "passed" and not self.validation_planner.has_runtime_success(session):
            return False

        target_paths = self._actionable_explicit_target_paths(route, session)
        if target_paths and len(target_paths) > MAX_LIGHTWEIGHT_UPDATE_TARGETS:
            return False

        snapshot = session.workspace_snapshot
        if snapshot is not None and snapshot.file_count > MAX_LIGHTWEIGHT_UPDATE_WORKSPACE_FILES:
            return False

        task_state = session.task_state
        if task_state is not None:
            if task_state.needs_clarification:
                return False
            if task_state.ambiguity_level == "high" or task_state.risk_level == "high":
                return False
            if float(task_state.confidence or 0.0) < MIN_LIGHTWEIGHT_UPDATE_CONFIDENCE:
                return False

        for change in session.changed_files:
            path = str(change.path or "").strip()
            if not path:
                return False
            suffix = Path(path).suffix.lower()
            if suffix not in LIGHTWEIGHT_UPDATE_SUFFIXES:
                return False
            current_content = self._current_file_content(session, path)
            if current_content is None:
                continue
            if len(current_content) > MAX_LIGHTWEIGHT_UPDATE_CHARS:
                return False
            if current_content.count("\n") + 1 > MAX_LIGHTWEIGHT_UPDATE_LINES:
                return False

        return True

    def _compact_repair_change_line_count(
        self,
        *,
        current_content: str,
        proposed_content: str,
    ) -> int:
        count = 0
        for raw in difflib.unified_diff(
            str(current_content or "").splitlines(),
            str(proposed_content or "").splitlines(),
            lineterm="",
        ):
            if raw.startswith(("---", "+++", "@@")):
                continue
            if raw.startswith(("+", "-")):
                count += 1
        return count

    def _review_rejection_looks_like_scope_broadening(
        self,
        review: ProposedUpdateReview,
    ) -> bool:
        text = " ".join(
            part.strip().lower()
            for part in [review.summary, *review.blocking_issues, *review.preservation_risks]
            if str(part or "").strip()
        )
        if not text:
            return False
        return any(
            marker in text
            for marker in (
                "scope",
                "too broad",
                "broadens",
                "broader",
                "unrequested",
                "unrelated existing behavior",
                "remove working behavior",
            )
        )

    def _review_reported_missing_literal(
        self,
        *,
        path: str,
        issue: str,
    ) -> str | None:
        normalized_path = str(path or "").strip()
        text = str(issue or "").strip()
        if not normalized_path or not text:
            return None
        prefix = f"The exact requested literal is missing from {normalized_path}:"
        if not text.startswith(prefix):
            return None
        literal = text[len(prefix) :].strip()
        return literal or None

    def _sanitize_model_backed_review(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str,
        review: ProposedUpdateReview,
    ) -> ProposedUpdateReview:
        if review.safe_to_write or not review.blocking_issues:
            return review
        scope = _artifact_scoped_focus(
            route,
            session,
            path,
            current_content=current_content,
        )
        supported_literals = {
            str(item or "").strip()
            for item in scope.get("literal_constraints", [])
            if str(item or "").strip()
        }
        retained_issues: list[str] = []
        discarded_literals: list[str] = []
        for issue in review.blocking_issues:
            literal = self._review_reported_missing_literal(path=path, issue=issue)
            if literal is None or literal in supported_literals:
                retained_issues.append(issue)
                continue
            discarded_literals.append(literal)
        if not discarded_literals:
            return review

        self._log(
            "proposed_update_review_unsupported_literal_discarded",
            path=path,
            literals=discarded_literals[:4],
            supported_literals=sorted(supported_literals)[:4],
        )

        lowered_discarded = [literal.lower() for literal in discarded_literals]
        retained_hints = [
            hint
            for hint in review.repair_hints
            if not (
                "exact requested literal" in str(hint or "").lower()
                or "placeholder values" in str(hint or "").lower()
                or any(literal in str(hint or "").lower() for literal in lowered_discarded)
            )
        ]

        if not retained_issues and not review.preservation_risks:
            return ProposedUpdateReview(
                safe_to_write=True,
                summary=(
                    "The proposed update passes the deterministic integrity checks; the model-backed review only "
                    "reported unsupported literal obligations that are not hard source constraints for this file."
                ),
                confidence=max(0.51, min(float(review.confidence or 0.0), 0.74)),
                blocking_issues=[],
                preservation_risks=[],
                repair_hints=retained_hints[:4]
                or [
                    "Keep the repair anchored to the evidenced behavior delta and preserve unrelated existing behavior.",
                ],
            )

        summary = review.summary
        lowered_summary = str(summary or "").lower()
        if "literal constraint" in lowered_summary or "exact requested literal" in lowered_summary:
            summary = "The model-backed review still found another concrete issue after unsupported literal findings were discarded."
        return ProposedUpdateReview(
            safe_to_write=False,
            summary=summary,
            confidence=review.confidence,
            blocking_issues=retained_issues[:4],
            preservation_risks=review.preservation_risks[:4],
            repair_hints=retained_hints[:4],
        )

    def _evidence_backed_behavior_adjustment_is_in_scope(
        self,
        *,
        session: SessionState,
        path: str,
        current_content: str,
        proposed_content: str,
        repair_context: ValidationFailureEvidence | None,
        review: ProposedUpdateReview,
    ) -> bool:
        if review.safe_to_write:
            return False
        if not self._review_rejection_looks_like_scope_broadening(review):
            return False
        if self._supporting_runtime_contract_requires_stdout_emission(
            session,
            repair_context=repair_context,
        ) and not self._repair_update_addresses_stdout_contract(
            current_content=current_content,
            proposed_content=proposed_content,
        ):
            return False
        return self._repair_update_is_evidence_backed_behavior_adjustment(
            path=path,
            current_content=current_content,
            proposed_content=proposed_content,
            repair_context=repair_context,
        )

    def _supporting_runtime_contract_requires_stdout_emission(
        self,
        session: SessionState,
        *,
        repair_context: ValidationFailureEvidence | None,
    ) -> bool:
        if repair_context is None or repair_context.verification_scope != "runtime":
            return False

        candidate_paths = list(
            dict.fromkeys(
                [
                    str(candidate or "").strip()
                    for candidate in getattr(repair_context, "file_hints", []) or []
                    if str(candidate or "").strip() and self._path_is_test_like(str(candidate or "").strip())
                ]
            )
        )
        if not candidate_paths:
            return False

        stdout_capture_markers = (
            "redirect_stdout(",
            ".getvalue().strip(",
            ".getvalue().strip()",
            "capsys.readouterr(",
            "patch('sys.stdout'",
            'patch("sys.stdout"',
        )
        for candidate in candidate_paths[:4]:
            excerpt = self._current_or_last_read_excerpt(session, path=candidate)
            if any(marker in excerpt for marker in stdout_capture_markers):
                return True
        return False

    def _repair_update_addresses_stdout_contract(
        self,
        *,
        current_content: str,
        proposed_content: str,
    ) -> bool:
        current_markers = self._stdout_emission_markers_in_content(current_content)
        proposed_markers = self._stdout_emission_markers_in_content(proposed_content)
        if not proposed_markers:
            return False
        if not current_markers:
            return True
        return proposed_markers != current_markers

    def _stdout_emission_markers_in_content(self, content: str) -> tuple[str, ...]:
        matches: list[str] = []
        for raw_line in str(content or "").splitlines():
            line = raw_line.strip()
            lowered = line.lower()
            if "print(" in line:
                matches.append(line)
            elif "sys.stdout.write(" in lowered or "stdout.write(" in lowered:
                matches.append(line)
        return tuple(matches)

    def _repair_update_is_evidence_backed_behavior_adjustment(
        self,
        *,
        path: str,
        current_content: str,
        proposed_content: str,
        repair_context: ValidationFailureEvidence | None,
    ) -> bool:
        if repair_context is None or repair_context.verification_scope != "runtime":
            return False
        if not _repair_semantic_delta_lines(repair_context, limit=2):
            return False

        normalized_path = str(path or "").strip()
        if not normalized_path:
            return False
        brief = getattr(repair_context, "repair_brief", None)
        locked_target = str(getattr(brief, "locked_target", "") or "").strip()
        primary_target = str(getattr(brief, "primary_target", "") or "").strip()
        if normalized_path not in {locked_target, primary_target}:
            return False

        changed_line_count = self._compact_repair_change_line_count(
            current_content=current_content,
            proposed_content=proposed_content,
        )
        if changed_line_count <= 0 or changed_line_count > 16:
            return False

        current_lines = max(len(str(current_content or "").splitlines()), 1)
        proposed_lines = max(len(str(proposed_content or "").splitlines()), 1)
        if abs(proposed_lines - current_lines) > 4:
            return False

        current_length = max(len(str(current_content or "")), 1)
        proposed_length = max(len(str(proposed_content or "")), 1)
        if proposed_length < int(current_length * 0.6) and proposed_lines < int(current_lines * 0.7):
            return False

        brief_symbols = [
            str(symbol or "").strip()
            for symbol in getattr(brief, "implicated_symbols", [])
            if str(symbol or "").strip()
        ]
        identifiers = self._repair_identifiers_for_target(
            normalized_path,
            repair_context,
            current_content=current_content,
            proposed_content=proposed_content,
        )
        relevant_identifiers = self._unique_paths(
            [
                identifier
                for identifier in [*identifiers, *brief_symbols]
                if identifier and (identifier in current_content or identifier in proposed_content)
            ]
        )
        if not relevant_identifiers:
            return False
        if not any(
            self._identifier_lines_changed(identifier, current_content, proposed_content)
            for identifier in relevant_identifiers
        ):
            return False

        return True

    def _should_skip_model_backed_repair_review(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str | None,
        proposed_content: str,
        reserve_model: str | None,
    ) -> bool:
        repair_context = session.active_repair_context
        if repair_context is None or current_content is None:
            return False
        if reserve_model is not None:
            return False
        if route.needs_clarification or not route.safe_to_execute:
            return False

        brief = getattr(repair_context, "repair_brief", None)
        locked_target = str(getattr(brief, "locked_target", "") or "").strip()
        primary_target = str(getattr(brief, "primary_target", "") or "").strip()
        allowed_files = {
            str(item or "").strip()
            for item in getattr(brief, "allowed_files", [])
            if str(item or "").strip()
        }
        forbidden_files = {
            str(item or "").strip()
            for item in getattr(brief, "forbidden_files", [])
            if str(item or "").strip()
        }
        normalized_path = str(path or "").strip()
        if not normalized_path:
            return False
        target_is_explicitly_allowed = bool(allowed_files and normalized_path in allowed_files)
        suffix = Path(normalized_path).suffix.lower()
        if suffix not in LIGHTWEIGHT_UPDATE_SUFFIXES:
            return False
        if len(current_content) > MAX_LIGHTWEIGHT_UPDATE_CHARS:
            return False
        if current_content.count("\n") + 1 > MAX_LIGHTWEIGHT_UPDATE_LINES:
            return False
        if locked_target and locked_target != normalized_path and not target_is_explicitly_allowed:
            return False
        if primary_target and primary_target != normalized_path and not target_is_explicitly_allowed:
            return False
        if self._repair_update_is_evidence_backed_behavior_adjustment(
            path=normalized_path,
            current_content=current_content,
            proposed_content=proposed_content,
            repair_context=repair_context,
        ):
            return False
        if allowed_files and normalized_path not in allowed_files:
            return False
        if normalized_path in forbidden_files:
            return False
        changed_paths = {
            str(getattr(item, "path", "") or "").strip()
            for item in getattr(session, "changed_files", []) or []
            if str(getattr(item, "path", "") or "").strip()
        }
        unresolved_competing_scope = {
            candidate
            for candidate in allowed_files
            if candidate != normalized_path
            and candidate not in changed_paths
            and not self.validation_planner._is_test_path(candidate)
            and not self.validation_planner._is_documentation_path(candidate)
        }
        if len(unresolved_competing_scope) > 1:
            return False
        if self._repair_attempt_failure_count(session, repair_context, normalized_path) >= 1:
            return False
        if self._repair_attempt_mutation_count(session, repair_context, normalized_path) >= 1:
            return False

        changed_line_count = self._compact_repair_change_line_count(
            current_content=current_content,
            proposed_content=proposed_content,
        )
        if changed_line_count <= 0:
            return False
        if changed_line_count > 24:
            return False

        return True

    def _repair_no_effective_change_review(
        self,
        *,
        path: str,
        current_content: str,
        proposed_content: str,
        repair_context: ValidationFailureEvidence | None,
    ) -> ProposedUpdateReview | None:
        if repair_context is None:
            return None

        mutation = self._assess_effective_mutation(path, current_content, proposed_content)
        if mutation.effective:
            return None

        normalized_path = str(path or "").strip()
        failure_signature = str(getattr(repair_context, "evidence_signature", "") or "").strip()
        brief = getattr(repair_context, "repair_brief", None)
        if brief is not None:
            candidate_signature = str(getattr(brief, "failure_signature", "") or "").strip()
            if candidate_signature:
                failure_signature = candidate_signature

        signature_suffix = f" for {failure_signature}" if failure_signature else ""
        blocking_issue = (
            f"The proposed repair does not make a productive change to {normalized_path}{signature_suffix} "
            f"({mutation.reason})."
        )
        locked_target = self._repair_brief_locked_target(repair_context)
        if locked_target:
            blocking_issue += f" The locked repair target is still {locked_target}."
        return ProposedUpdateReview(
            safe_to_write=False,
            summary=(
                "The proposed repair is a no-op or only a formal rewrite, so writing it would only repeat the "
                "same failing state."
            ),
            confidence=0.93,
            blocking_issues=[blocking_issue],
            preservation_risks=[],
            repair_hints=[
                *self._semantic_delta_noop_repair_hints(
                    path=path,
                    current_content=current_content,
                    repair_context=repair_context,
                ),
                "Change the locked repair target in a way that alters the failing behavior instead of resubmitting an equivalent file.",
                "Do not change only whitespace, comments, metadata, or unrelated lines; produce a real code-level fix.",
                "Use the expected-versus-observed repair brief to make a minimal but real semantic change.",
            ],
        )

    def _task_backed_no_effective_change_review(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str,
        proposed_content: str,
        repair_context: ValidationFailureEvidence | None,
    ) -> ProposedUpdateReview | None:
        if repair_context is not None:
            return None
        if not self._supports_update_style_existing_artifact(route, current_content=current_content):
            return None

        mutation = self._assess_effective_mutation(path, current_content, proposed_content)
        if mutation.effective:
            return None

        target_paths = self._actionable_explicit_target_paths(route, session)
        normalized_path = str(path or "").strip()
        if target_paths and normalized_path not in target_paths:
            return None

        focus = _artifact_scoped_focus(route, session, path, current_content=current_content)
        raw_requirements = focus.get("current_write_requirements") or []
        write_requirements = [
            str(item or "").strip()
            for item in raw_requirements
            if str(item or "").strip()
        ]
        verification_target = str(getattr(session.task_state, "verification_target", "") or "").strip()
        if not write_requirements and not verification_target:
            return None

        repair_hints: list[str] = []
        for requirement in write_requirements[:2]:
            repair_hints.append(f"Make a real change in {normalized_path} that addresses: {requirement}.")
        if verification_target:
            repair_hints.append(
                f"Anchor the next draft to the exercised path checked by {verification_target} instead of resubmitting equivalent code."
            )
        repair_hints.append(
            "Do not resubmit equivalent content; change the code lines that control the requested behavior or output."
        )
        return ProposedUpdateReview(
            safe_to_write=False,
            summary=(
                "The proposed update is a no-op or only an equivalent restatement, so writing it would preserve "
                "the same current behavior."
            ),
            confidence=0.9,
            blocking_issues=[
                f"The proposal for {normalized_path} does not make a productive change ({mutation.reason})."
            ],
            preservation_risks=[],
            repair_hints=repair_hints[:4],
        )

    def _semantic_delta_noop_repair_hints(
        self,
        *,
        path: str,
        current_content: str,
        repair_context: ValidationFailureEvidence,
    ) -> list[str]:
        hints: list[str] = []
        seen: set[str] = set()
        focused_line_hints = _repair_target_line_hints(
            path=path,
            current_content=current_content,
            repair_context=repair_context,
        )
        focused_line_excerpt = _line_focused_excerpt(
            current_content,
            line_hints=focused_line_hints,
            limit=220,
            before_radius=0,
            after_radius=0,
        )
        for delta in _repair_semantic_delta_lines(repair_context, limit=2):
            text = str(delta or "").strip()
            if not text:
                continue
            replace_match = re.search(
                r"Replace observed-only text ['\"](?P<observed>.+?)['\"] with expected text ['\"](?P<expected>.+?)['\"]",
                text,
            )
            remove_match = re.search(r"Remove observed-only text ['\"](?P<observed>.+?)['\"]", text)
            insert_match = re.search(r"Insert expected-only text ['\"](?P<expected>.+?)['\"]", text)
            hint: str | None = None
            if replace_match is not None:
                observed = str(replace_match.group("observed") or "").strip()
                expected = str(replace_match.group("expected") or "").strip()
                if observed and expected:
                    hint = (
                        f"The current implementation in {path} must stop returning {observed!r} at the mismatching "
                        f"output position and instead produce {expected!r} there."
                    )
            elif remove_match is not None:
                observed = str(remove_match.group("observed") or "").strip()
                if observed:
                    if focused_line_excerpt:
                        hint = (
                            f"Rewrite at least one implicated output line in {path}; leaving this excerpt unchanged "
                            f"will preserve the failing behavior:\n{focused_line_excerpt}"
                        )
                        if hint not in seen:
                            seen.add(hint)
                            hints.append(hint)
                    if observed not in str(current_content or ""):
                        hint = (
                            f"The current implementation in {path} never handles the observed-only literal {observed!r} "
                            "explicitly. Change the output construction or normalization so that literal no longer "
                            "survives in the returned value."
                        )
                    else:
                        hint = (
                            f"Change the output construction in {path} so the observed-only literal {observed!r} "
                            "disappears from the returned value."
                        )
            elif insert_match is not None:
                expected = str(insert_match.group("expected") or "").strip()
                if expected:
                    hint = (
                        f"The current implementation in {path} still omits the expected literal {expected!r}. "
                        "Change the produced result so that literal appears at the mismatching output position."
                    )
            if hint and hint not in seen:
                seen.add(hint)
                hints.append(hint)
        return hints[:2]

    def _repair_followup_retry_budget(self) -> tuple[int, int]:
        reserve_model = self._lightweight_generation_model_name()
        if reserve_model is None:
            # On single-model 7B deployments the decisive follow-up repair hop has no
            # real model switch available. Give that last same-model escalation enough
            # warm-start and completion budget to be meaningfully different from the
            # initial compact retry instead of timing out during startup again.
            return max(self._llm_timeout(90), 90), max(self._llm_timeout(420), 420)
        return max(self._llm_timeout(60), 60), max(self._llm_timeout(240), 240)

    def _compact_reserve_review_budget(self) -> tuple[int, int]:
        timeout = max(self._llm_timeout(25), 25)
        # Compact reserve-model reviews still carry summarized artifact excerpts and
        # often begin streaming close to the initial warm-start deadline on local CPU
        # stacks. Keep the prompt compact, but give the completion enough room to
        # finish so we do not escalate purely because the reviewer started late.
        return timeout, max(timeout + 60, 90)

    def _review_generated_update(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str,
        proposed_content: str,
    ) -> ProposedUpdateReview:
        repair_review = session.active_repair_context is not None
        argv_launcher_review = self._argv_launcher_runtime_review(
            session,
            path=path,
            current_content=current_content,
            proposed_content=proposed_content,
        )
        if argv_launcher_review is not None:
            return argv_launcher_review
        focused_compact_review = self._should_keep_update_review_compact(
            route,
            session,
            path=path,
            current_content=current_content,
        )
        compact_review = repair_review or focused_compact_review or self._should_prefer_lightweight_update_generation(
            route,
            session,
            path=path,
            current_content=current_content,
        )
        full_prompt = proposed_update_review_prompt(
            route,
            session,
            path=path,
            supporting_artifact_context=self._supporting_artifact_review_context(
                session,
                target_path=path,
            ),
            current_excerpt=self._review_excerpt(current_content, limit=5_200),
            proposed_excerpt=self._review_excerpt(proposed_content, limit=5_200),
            diff_excerpt=self._review_diff_excerpt(
                path,
                current_content=current_content,
                proposed_content=proposed_content,
                limit=5_200,
            ),
        )
        compact_prompt: str | None = None
        if compact_review:
            compact_prompt = proposed_update_review_prompt(
                route,
                session,
                path=path,
                supporting_artifact_context=self._supporting_artifact_review_context(
                    session,
                    target_path=path,
                    excerpt_limit=650,
                    max_sections=2,
                ),
                current_excerpt=self._review_excerpt(current_content, limit=2_400),
                proposed_excerpt=self._review_excerpt(proposed_content, limit=2_400),
                diff_excerpt=self._review_diff_excerpt(
                    path,
                    current_content=current_content,
                    proposed_content=proposed_content,
                    limit=1_800,
                ),
                mode="compact",
            )
        primary_model = self._primary_generation_model_name()
        reserve_model = self._lightweight_generation_model_name()
        if reserve_model is None and repair_review:
            reserve_model = self._generation_recovery_model_name()
            if reserve_model == primary_model:
                reserve_model = None
        local_review_reason: str | None = None
        if focused_compact_review and not repair_review and reserve_model is None:
            local_review_reason = "single_model_compact_update"
        elif self._should_skip_model_backed_repair_review(
            route,
            session,
            path=path,
            current_content=current_content,
            proposed_content=proposed_content,
            reserve_model=reserve_model,
        ):
            local_review_reason = "single_model_compact_repair"
        if local_review_reason is not None:
            review = self._fallback_proposed_update_review(
                route,
                session=session,
                path=path,
                current_content=current_content,
                proposed_content=proposed_content,
            )
            self._log(
                "proposed_update_review_skipped",
                path=path,
                reason=local_review_reason,
                model=primary_model,
            )
            self._append_runtime_execution(
                session,
                build_execution_run_record(
                    operation_name="proposed_update_review",
                    task_class="proposed_update_review",
                    final_state="degraded_success",
                    capability_tier="tier_d",
                    recovery_strategy="deterministic_fallback",
                    degraded=True,
                    honest_blocked=False,
                    artifact_bytes_generated=0,
                    validation_possible=True,
                    summary="A focused update skipped a second same-model review hop and used conservative local preservation checks instead.",
                    attempts=[],
                ),
            )
            return review
        review_attempts: list[tuple[str | None, str, str]] = []
        if repair_review:
            if reserve_model is not None:
                review_attempts.append((reserve_model, "tier_b", "reserve_model_review"))
            review_attempts.append((primary_model, "tier_a", "primary_model_review"))
        elif focused_compact_review:
            if reserve_model is not None:
                review_attempts.append((reserve_model, "tier_b", "reserve_model_review"))
            review_attempts.append((primary_model, "tier_a", "primary_model_review"))
        elif compact_review and reserve_model is not None:
            # On constrained hardware a second full-model review often costs far more
            # latency than the lightweight safety value it adds for small focused edits.
            # Fall back to conservative local preservation checks instead of escalating.
            review_attempts.append((reserve_model, "tier_b", "reserve_model_review"))
        else:
            if reserve_model is not None:
                review_attempts.append((reserve_model, "tier_b", "reserve_model_review"))
            review_attempts.append((primary_model, "tier_a", "primary_model_review"))

        attempts: list[ExecutionAttemptRecord] = []
        self._log(
            "proposed_update_review_started",
            path=path,
            model=review_attempts[0][0] or primary_model,
        )

        seen_models: set[str] = set()
        for model_name, capability_tier, strategy in review_attempts:
            normalized_model = str(model_name or "").strip()
            model_key = normalized_model or "__default__"
            if model_key in seen_models:
                continue
            seen_models.add(model_key)

            use_compact_prompt = (
                compact_prompt is not None
                and (
                    (repair_review and capability_tier == "tier_a")
                    or (focused_compact_review and capability_tier == "tier_a")
                    or (
                        reserve_model is not None
                        and model_name == reserve_model
                        and capability_tier == "tier_b"
                    )
                )
            )
            prompt = compact_prompt if use_compact_prompt else full_prompt
            same_model_compact_review = (
                use_compact_prompt
                and capability_tier == "tier_a"
                and normalized_model == str(primary_model or "").strip()
            )
            if use_compact_prompt:
                # Primary-model compact reviews still pay the full local startup cost on
                # constrained single-model hardware. Keep the prompt compact, but avoid
                # treating the review like a short strict-timeout hop when it is the same
                # model that just needed a longer warm start for generation.
                if same_model_compact_review:
                    if repair_review:
                        timeout, total_timeout = self._content_generation_time_budget(
                            prompt_variant="compact",
                            repair_context=session.active_repair_context,
                        )
                    else:
                        timeout = max(self._llm_timeout(60), 60)
                        total_timeout = max(self._llm_timeout(180), 180)
                else:
                    timeout, total_timeout = self._compact_reserve_review_budget()
            else:
                timeout = max(self._llm_timeout(18), 18)
                total_timeout = max(timeout + 12, timeout * 2)
            num_ctx = min(self._llm_num_ctx(2048), 2048) if use_compact_prompt else min(self._llm_num_ctx(6144), 6144)
            prompt_variant = "compact" if use_compact_prompt else "full"
            strict_timeouts = use_compact_prompt and not same_model_compact_review
            outcome = invoke_model(
                lambda progress, review_model=model_name: self.llm.generate_json(
                    prompt,
                    system=proposed_update_review_system_prompt(),
                    model=review_model,
                    retries=0,
                    timeout=timeout,
                    total_timeout=total_timeout,
                    strict_timeouts=strict_timeouts,
                    num_ctx=num_ctx,
                    progress_callback=progress,
                ),
                operation_name="proposed_update_review",
                task_class="proposed_update_review",
                attempt_number=len(attempts) + 1,
                capability_tier=capability_tier,
                recovery_strategy=strategy,
                prompt_variant=prompt_variant,
                model_identifier=model_name or primary_model,
                backend_identifier=self._backend_identifier(),
                inactivity_timeout_seconds=timeout,
                total_timeout_seconds=total_timeout,
                context_pressure_estimate=estimate_context_pressure(prompt_chars=len(prompt)),
                event_callback=self._progress_logger("proposed_update_review_progress", path=path),
            )
            attempts.append(outcome.attempt)
            if outcome.exception is not None:
                self._log(
                    "proposed_update_review_error",
                    path=path,
                    error=str(outcome.exception),
                    strategy=strategy,
                    model=model_name,
                )
                continue
            try:
                review = ProposedUpdateReview.model_validate(outcome.value)
            except ValidationError as exc:
                self._log(
                    "proposed_update_review_invalid",
                    path=path,
                    errors=exc.errors(),
                    payload=outcome.value,
                    strategy=strategy,
                    model=model_name,
                )
                continue
            review = self._sanitize_model_backed_review(
                route,
                session,
                path=path,
                current_content=current_content,
                review=review,
            )

            self._append_runtime_execution(
                session,
                build_execution_run_record(
                    operation_name="proposed_update_review",
                    task_class="proposed_update_review",
                    final_state="completed",
                    capability_tier=capability_tier,
                    recovery_strategy=strategy,
                    degraded=capability_tier != "tier_a",
                    honest_blocked=False,
                    artifact_bytes_generated=0,
                    validation_possible=True,
                    summary="An AI pre-write review checked whether the proposed file update stayed focused and preserved unrelated behavior.",
                    attempts=attempts,
                ),
            )
            return review

        review = self._fallback_proposed_update_review(
            route,
            session=session,
            path=path,
            current_content=current_content,
            proposed_content=proposed_content,
            model_backed_review_failed=True,
            review_failure_reason=(
                str(attempts[-1].failure.reason if attempts and attempts[-1].failure is not None else "").strip()
                or "model_review_unavailable"
            ),
        )
        self._append_runtime_execution(
            session,
            build_execution_run_record(
                operation_name="proposed_update_review",
                task_class="proposed_update_review",
                final_state="degraded_success",
                capability_tier="tier_d",
                recovery_strategy="deterministic_fallback",
                degraded=True,
                honest_blocked=False,
                artifact_bytes_generated=0,
                validation_possible=True,
                summary="The AI pre-write review fell back to conservative local heuristics after model-backed review did not complete cleanly.",
                attempts=attempts,
            ),
        )
        return review

    def _pre_write_update_review(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str | None,
        proposed_content: str,
        repair_context: ValidationFailureEvidence | None,
    ) -> ProposedUpdateReview:
        if current_content is None:
            return self._pre_write_create_review(
                route,
                session,
                path=path,
                proposed_content=proposed_content,
            )

        review_from_generated = False
        review = self._generated_content_integrity_review(path=path, proposed_content=proposed_content)
        if review is None:
            review = self._explicit_constraint_integrity_review(
                route,
                session,
                path=path,
                current_content=current_content,
                proposed_content=proposed_content,
            )
        if review is None:
            review = self._helper_entrypoint_scope_review(
                session,
                path=path,
                current_content=current_content,
                proposed_content=proposed_content,
            )
        if review is None:
            review = self._entrypoint_helper_contract_review(
                path=path,
                current_content=current_content,
                proposed_content=proposed_content,
            )
        if review is None:
            review = self._repair_target_scope_review(
                session=session,
                path=path,
                repair_context=repair_context,
            )
        if review is None:
            review = self._cli_wrapper_responsibility_review(
                session,
                path=path,
                proposed_content=proposed_content,
            )
        if review is None:
            review = self._direct_main_option_contract_review(
                session,
                path=path,
                current_content=current_content,
                proposed_content=proposed_content,
                repair_context=repair_context,
            )
        if review is None:
            review = self._direct_python_script_option_contract_review(
                path=path,
                proposed_content=proposed_content,
                repair_context=repair_context,
            )
        if review is None:
            review = self._pre_write_web_contract_review(
                route,
                session,
                path=path,
                proposed_content=proposed_content,
            )
        if review is None:
            review = self._validation_repair_relevance_review(
                path=path,
                current_content=current_content,
                proposed_content=proposed_content,
                repair_context=repair_context,
                session=session,
            )
        if review is None:
            review = self._repair_no_effective_change_review(
                path=path,
                current_content=current_content,
                proposed_content=proposed_content,
                repair_context=repair_context,
            )
        if review is None:
            review = self._review_generated_update(
                route,
                session,
                path=path,
                current_content=current_content,
                proposed_content=proposed_content,
            )
            review_from_generated = True
        review = self._sanitize_model_backed_review(
            route,
            session,
            path=path,
            current_content=current_content,
            review=review,
        )
        if (
            review_from_generated
            and
            not review.safe_to_write
            and self._evidence_backed_behavior_adjustment_is_in_scope(
                session=session,
                path=path,
                current_content=current_content,
                proposed_content=proposed_content,
                repair_context=repair_context,
                review=review,
            )
        ):
            self._log(
                "proposed_update_review_scope_override",
                path=path,
                summary=review.summary,
                reason="failure_evidence_backed_local_behavior_change",
            )
            review = ProposedUpdateReview(
                safe_to_write=True,
                summary=(
                    "The proposed repair stays local and makes a behavior adjustment that is directly supported "
                    "by the active runtime failure evidence."
                ),
                confidence=max(float(review.confidence or 0.0), 0.58),
                blocking_issues=[],
                preservation_risks=[],
                repair_hints=[
                    "Keep the repair anchored to the evidenced behavior delta and avoid unrelated follow-on changes.",
                ],
            )
        return review

    def _deterministic_direct_main_runtime_recovery(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str | None,
        repair_context: ValidationFailureEvidence | None,
        review_feedback: ProposedUpdateReview | None = None,
    ) -> DeterministicUpdateRecovery | None:
        if current_content is None or repair_context is None:
            return None
        if repair_context.verification_scope != "runtime":
            return None
        if Path(path).suffix.lower() not in {".py", ".pyi"}:
            return None
        if not self._path_exposes_direct_main_contract_target(
            session,
            path=path,
            current_content=current_content,
            proposed_content=current_content,
        ):
            return None

        review_text = self._proposed_update_review_text(review_feedback) if review_feedback is not None else ""

        option_tokens, _ = self._direct_main_option_contract_details(session, repair_context)
        _, positional_tokens = self._direct_main_option_contract_details(session, repair_context)
        if not option_tokens:
            return None

        payload_echo_patch = self._apply_direct_main_payload_echo_patch(
            current_content=current_content,
            option_tokens=option_tokens,
            positional_tokens=positional_tokens,
            repair_context=repair_context,
        )
        patched_content = payload_echo_patch
        if patched_content is None:
            if review_text and not any(
                marker in review_text
                for marker in ("direct main", "argv", "launcher", "option token")
            ):
                return None

            patched_content = self._apply_direct_main_contract_patch(
                current_content=current_content,
                option_tokens=option_tokens,
            )
        if patched_content is None or patched_content == current_content:
            return None
        try:
            ast.parse(patched_content)
        except SyntaxError:
            return None

        review = self._local_pre_write_update_review(
            route,
            session,
            path=path,
            current_content=current_content,
            proposed_content=patched_content,
            repair_context=repair_context,
        )
        if not review.safe_to_write:
            return None
        return DeterministicUpdateRecovery(
            content=patched_content,
            review=review,
            recovery_strategy="deterministic_direct_main_contract",
        )

    def _direct_python_script_expected_output(
        self,
        repair_context: ValidationFailureEvidence,
    ) -> str | None:
        if repair_context.verification_scope != "runtime":
            return None
        brief = getattr(repair_context, "repair_brief", None)
        if brief is None:
            return None
        expected_values = [
            _repair_semantic_value_text(item)
            for item in getattr(brief, "expected_semantics", [])
            if _repair_semantic_value_text(item)
        ]
        if not expected_values:
            return None
        expected_output = str(expected_values[0] or "").strip()
        return expected_output or None

    def _direct_python_script_verbatim_option_payload_layout(
        self,
        *,
        tail_tokens: list[str],
        repair_context: ValidationFailureEvidence,
    ) -> str | None:
        if len(tail_tokens) < 2:
            return None
        expected_output = self._direct_python_script_expected_output(repair_context)
        if not expected_output:
            return None

        option_payload = str(tail_tokens[0] or "").strip()
        remainder_tokens = [str(token or "").strip() for token in tail_tokens[1:] if str(token or "").strip()]
        if not option_payload or not remainder_tokens:
            return None

        remainder_output = " ".join(remainder_tokens).strip()
        candidates = (
            ("prefix_tight", f"{option_payload}{remainder_output}"),
            ("prefix_spaced", f"{option_payload} {remainder_output}".strip()),
            ("suffix_tight", f"{remainder_output}{option_payload}"),
            ("suffix_spaced", f"{remainder_output} {option_payload}".strip()),
        )
        for layout, candidate in candidates:
            if candidate == expected_output:
                return layout
        return None

    def _apply_direct_python_script_verbatim_option_payload_patch(
        self,
        *,
        current_content: str,
        option_tokens: list[str],
        tail_tokens: list[str],
        repair_context: ValidationFailureEvidence,
    ) -> str | None:
        if not option_tokens or len(tail_tokens) < 2:
            return None

        layout = self._direct_python_script_verbatim_option_payload_layout(
            tail_tokens=tail_tokens,
            repair_context=repair_context,
        )
        if layout is None:
            return None

        option_payload = str(tail_tokens[0] or "").strip()
        if not option_payload:
            return None

        bounds = self._python_function_block_bounds(current_content, function_name="main")
        if bounds is None:
            return None
        lines = str(current_content or "").splitlines(keepends=True)
        start, end = bounds
        block_lines = lines[start:end]
        condition_pattern = re.compile(
            r"^(?P<indent>\s*)if\s+(?:(?:len\((?P<len_arg>\w+)\)\s*>=\s*(?P<required>\d+)\s+and\s+))?"
            r"(?P<arg>\w+)\s*\[\s*:\s*(?P<slice>\d+)\s*\]\s*==\s*(?P<literal>\[[^\n]+\])\s*:\s*$"
        )
        option_count = len(option_tokens)
        required_len = option_count + 1

        for index, raw_line in enumerate(block_lines):
            match = condition_pattern.match(raw_line)
            if match is None:
                continue
            try:
                literal_value = ast.literal_eval(str(match.group("literal") or "").strip())
            except (SyntaxError, ValueError):
                continue
            if not isinstance(literal_value, (list, tuple)):
                continue

            literal_tokens = [str(item) for item in literal_value]
            slice_length = int(match.group("slice") or 0)
            if slice_length != len(literal_tokens):
                continue

            explicit_required = int(match.group("required") or 0)
            arg_name = str(match.group("arg") or "").strip()
            len_arg_name = str(match.group("len_arg") or "").strip()
            if not arg_name or (len_arg_name and len_arg_name != arg_name):
                continue

            dynamic_branch = literal_tokens == option_tokens
            hardcoded_payload_branch = literal_tokens == [*option_tokens, option_payload]
            if not dynamic_branch and not hardcoded_payload_branch:
                continue

            branch_indent = str(match.group("indent") or "")
            branch_body_indent = branch_indent + "    "
            branch_end = len(block_lines)
            for candidate in range(index + 1, len(block_lines)):
                line = block_lines[candidate]
                stripped = line.strip()
                if not stripped:
                    continue
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= len(branch_indent):
                    branch_end = candidate
                    break
            branch_lines = block_lines[index + 1 : branch_end]
            if not branch_lines:
                continue

            has_print = any("print(" in line for line in branch_lines)
            has_return = any(line.lstrip().startswith("return") for line in branch_lines)
            has_value_return = any(
                line.lstrip().startswith("return") and line.strip() != "return"
                for line in branch_lines
            )
            if not has_print and not has_value_return:
                continue

            effective_required_len = max(explicit_required, required_len)
            payload_expr = f"{arg_name}[{option_count}]"
            remainder_expr = f"' '.join({arg_name}[{option_count + 1}:])"
            if layout == "prefix_tight":
                output_expr = f"{payload_expr} + {remainder_expr}"
            elif layout == "prefix_spaced":
                output_expr = f"{payload_expr} + ' ' + {remainder_expr}"
            elif layout == "suffix_spaced":
                output_expr = f"{remainder_expr} + ' ' + {payload_expr}"
            else:
                output_expr = f"{remainder_expr} + {payload_expr}"

            new_condition = (
                f"{branch_indent}if len({arg_name}) >= {effective_required_len} and "
                f"{arg_name}[:{option_count}] == {option_tokens!r}:\n"
            )
            replacement_lines = [new_condition]
            if has_print:
                replacement_lines.append(f"{branch_body_indent}print({output_expr})\n")
                if has_return:
                    replacement_lines.append(f"{branch_body_indent}return\n")
            else:
                replacement_lines.append(f"{branch_body_indent}return {output_expr}\n")

            updated_lines = [
                *lines[:start],
                *block_lines[:index],
                *replacement_lines,
                *block_lines[branch_end:],
                *lines[end:],
            ]
            updated_content = "".join(updated_lines)
            if updated_content != current_content:
                return updated_content
        return None

    def _rewrite_direct_python_script_payload_branch(
        self,
        branch_lines: list[str],
        *,
        indent: str,
        arg_name: str,
        payload_index: int,
        payload_literal: str,
    ) -> list[str] | None:
        if not branch_lines:
            return None
        wrapper_source = "def _codex_branch():\n" + "".join(branch_lines)
        try:
            tree = ast.parse(wrapper_source)
        except SyntaxError:
            return None
        function = tree.body[0] if tree.body else None
        if not isinstance(function, ast.FunctionDef):
            return None

        def _payload_access() -> ast.Subscript:
            return ast.Subscript(
                value=ast.Name(id=arg_name, ctx=ast.Load()),
                slice=ast.Constant(value=payload_index),
                ctx=ast.Load(),
            )

        class PayloadLiteralTransformer(ast.NodeTransformer):
            def __init__(self) -> None:
                self.changed = False

            def visit_Constant(self, node: ast.Constant) -> ast.AST:
                value = node.value
                if not isinstance(value, str):
                    return node
                if value == payload_literal:
                    self.changed = True
                    return ast.copy_location(_payload_access(), node)
                if value == f"{payload_literal} ":
                    self.changed = True
                    return ast.copy_location(
                        ast.BinOp(
                            left=_payload_access(),
                            op=ast.Add(),
                            right=ast.Constant(value=" "),
                        ),
                        node,
                    )
                if value == f" {payload_literal}":
                    self.changed = True
                    return ast.copy_location(
                        ast.BinOp(
                            left=ast.Constant(value=" "),
                            op=ast.Add(),
                            right=_payload_access(),
                        ),
                        node,
                    )
                return node

        transformer = PayloadLiteralTransformer()
        transformed = transformer.visit(tree)
        if not transformer.changed:
            return None
        ast.fix_missing_locations(transformed)
        function = transformed.body[0] if transformed.body else None
        if not isinstance(function, ast.FunctionDef):
            return None
        rewritten_lines: list[str] = []
        for statement in function.body:
            statement_code = ast.unparse(statement)
            rewritten_lines.extend(
                f"{indent}{line}\n" if line else "\n"
                for line in statement_code.splitlines()
            )
        return rewritten_lines or None

    def _apply_direct_python_script_payload_literal_patch(
        self,
        *,
        current_content: str,
        option_tokens: list[str],
        tail_tokens: list[str],
        repair_context: ValidationFailureEvidence,
    ) -> str | None:
        if not option_tokens or not tail_tokens:
            return None
        expected_output = self._direct_python_script_expected_output(repair_context)
        first_tail_token = str(tail_tokens[0] or "").strip()
        if not expected_output or not first_tail_token or first_tail_token not in expected_output:
            return None

        bounds = self._python_function_block_bounds(current_content, function_name="main")
        if bounds is None:
            return None
        lines = str(current_content or "").splitlines(keepends=True)
        start, end = bounds
        block_lines = lines[start:end]
        condition_pattern = re.compile(
            r"^(?P<indent>\s*)if\s+(?P<arg>\w+)\s*\[\s*:\s*(?P<slice>\d+)\s*\]\s*==\s*(?P<literal>\[[^\n]+\])\s*:\s*$"
        )
        option_count = len(option_tokens)
        required_len = option_count + 1

        for index, raw_line in enumerate(block_lines):
            match = condition_pattern.match(raw_line)
            if match is None:
                continue
            try:
                literal_value = ast.literal_eval(str(match.group("literal") or "").strip())
            except (SyntaxError, ValueError):
                continue
            if not isinstance(literal_value, (list, tuple)):
                continue
            literal_tokens = [str(item) for item in literal_value]
            slice_length = int(match.group("slice") or 0)
            if (
                slice_length != len(literal_tokens)
                or len(literal_tokens) != required_len
                or literal_tokens[:option_count] != option_tokens
            ):
                continue

            payload_literal = str(literal_tokens[option_count] or "").strip()
            if not payload_literal:
                continue

            branch_indent = str(match.group("indent") or "")
            branch_body_indent = branch_indent + "    "
            arg_name = str(match.group("arg") or "").strip()
            if not arg_name:
                continue

            branch_end = len(block_lines)
            for candidate in range(index + 1, len(block_lines)):
                line = block_lines[candidate]
                stripped = line.strip()
                if not stripped:
                    continue
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= len(branch_indent):
                    branch_end = candidate
                    break
            branch_lines = block_lines[index + 1 : branch_end]
            rewritten_body = self._rewrite_direct_python_script_payload_branch(
                branch_lines,
                indent=branch_body_indent,
                arg_name=arg_name,
                payload_index=option_count,
                payload_literal=payload_literal,
            )
            if rewritten_body is None:
                continue

            new_condition = (
                f"{branch_indent}if len({arg_name}) >= {required_len} and "
                f"{arg_name}[:{option_count}] == {option_tokens!r}:\n"
            )
            updated_lines = [
                *lines[:start],
                *block_lines[:index],
                new_condition,
                *rewritten_body,
                *block_lines[branch_end:],
                *lines[end:],
            ]
            updated_content = "".join(updated_lines)
            if updated_content != current_content:
                return updated_content
        return None

    def _deterministic_direct_python_script_runtime_recovery(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str | None,
        repair_context: ValidationFailureEvidence | None,
        review_feedback: ProposedUpdateReview | None = None,
    ) -> DeterministicUpdateRecovery | None:
        if current_content is None or repair_context is None:
            return None
        if repair_context.verification_scope != "runtime":
            return None
        if Path(path).suffix.lower() not in {".py", ".pyi"}:
            return None

        option_tokens, tail_tokens = self._direct_python_script_option_contract_details(
            repair_context,
            path=path,
        )
        if not option_tokens or not tail_tokens:
            return None

        patched_content = self._apply_direct_python_script_verbatim_option_payload_patch(
            current_content=current_content,
            option_tokens=option_tokens,
            tail_tokens=tail_tokens,
            repair_context=repair_context,
        )
        if patched_content is None:
            patched_content = self._apply_direct_python_script_payload_literal_patch(
            current_content=current_content,
            option_tokens=option_tokens,
            tail_tokens=tail_tokens,
            repair_context=repair_context,
            )
        if patched_content is None or patched_content == current_content:
            return None
        try:
            ast.parse(patched_content)
        except SyntaxError:
            return None

        review = self._local_pre_write_update_review(
            route,
            session,
            path=path,
            current_content=current_content,
            proposed_content=patched_content,
            repair_context=repair_context,
        )
        if not review.safe_to_write:
            return None
        return DeterministicUpdateRecovery(
            content=patched_content,
            review=review,
            recovery_strategy="deterministic_direct_python_script_contract",
        )

    def _deterministic_runtime_repair_decision(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str | None,
        repair_context: ValidationFailureEvidence | None,
    ) -> AgentDecision | None:
        if current_content is None or repair_context is None:
            return None
        recovery = self._deterministic_direct_main_runtime_recovery(
            route,
            session,
            path=path,
            current_content=current_content,
            repair_context=repair_context,
        )
        if recovery is None:
            recovery = self._deterministic_direct_python_script_runtime_recovery(
                route,
                session,
                path=path,
                current_content=current_content,
                repair_context=repair_context,
            )
        if recovery is None:
            return None
        mutation = self._assess_effective_mutation(path, current_content, recovery.content)
        if not mutation.effective:
            return None
        self._log(
            "content_generation_recovery_started",
            path=path,
            strategy=recovery.recovery_strategy,
            failure_class="validation_repair_runtime_direct",
            models=[],
        )
        self._log(
            "content_generation_recovery_finished",
            path=path,
            strategy=recovery.recovery_strategy,
            source=recovery.recovery_strategy,
        )
        self._record_repair_attempt(
            session,
            repair_context,
            target=path,
            strategy=recovery.recovery_strategy,
            result="mutation_planned",
            reason=mutation.reason,
            mutation=mutation,
        )
        return AgentDecision(
            thought_summary=f"Apply the deterministic runtime repair for {path}.",
            action_type=AgentActionType.CALL_TOOL,
            tool_name="write_file",
            tool_args={"path": path, "content": recovery.content},
            expected_outcome="Apply the targeted repair derived directly from the failed runtime evidence.",
            final_response=None,
        )

    def _apply_direct_main_payload_echo_patch(
        self,
        *,
        current_content: str,
        option_tokens: list[str],
        positional_tokens: list[str],
        repair_context: ValidationFailureEvidence,
    ) -> str | None:
        expected_output = self._direct_main_expected_payload_output(
            repair_context,
            positional_tokens=positional_tokens,
        )
        if expected_output is None:
            return None

        bounds = self._python_function_block_bounds(current_content, function_name="main")
        if bounds is None:
            return None

        lines = str(current_content or "").splitlines(keepends=True)
        start, end = bounds
        block_lines = lines[start:end]
        option_set = {str(token or "").strip() for token in option_tokens if str(token or "").strip()}
        if not option_set:
            return None

        condition_index: int | None = None
        branch_end_index: int | None = None
        branch_arg_name = ""
        branch_indent = ""
        condition_pattern = re.compile(
            r"^(?P<indent>\s*)if\s+(?P<arg>\w+)\s+and\s+(?P=arg)\s*\[\s*0\s*\]\s*==\s*(?P<quote>['\"])(?P<option>.+?)(?P=quote)\s*:\s*$"
        )
        for index, raw_line in enumerate(block_lines):
            match = condition_pattern.match(raw_line)
            if match is None:
                continue
            option_literal = str(match.group("option") or "").strip()
            if option_literal not in option_set:
                continue
            condition_index = index
            branch_arg_name = str(match.group("arg") or "").strip()
            branch_indent = str(match.group("indent") or "")
            branch_body_indent = branch_indent + "    "
            branch_end_index = len(block_lines)
            for candidate in range(index + 1, len(block_lines)):
                line = block_lines[candidate]
                stripped = line.strip()
                if not stripped:
                    continue
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= len(branch_indent):
                    branch_end_index = candidate
                    break
            branch_lines = block_lines[index + 1 : branch_end_index]
            if not branch_lines:
                return None
            has_print = any("print(" in line for line in branch_lines)
            has_return = any(line.lstrip().startswith("return") for line in branch_lines)
            if not has_print and not has_return:
                return None
            normalized_block_lines = list(block_lines)
            for binding_index in range(index):
                normalized_line = self._normalize_direct_main_option_binding_line(
                    normalized_block_lines[binding_index],
                    binding_name=branch_arg_name,
                )
                if normalized_line != normalized_block_lines[binding_index]:
                    normalized_block_lines[binding_index] = normalized_line
                    break
            replacement_lines = [block_lines[index]]
            if has_print:
                replacement_lines.append(f"{branch_body_indent}print(' '.join({branch_arg_name}[1:]))\n")
                if has_return:
                    replacement_lines.append(f"{branch_body_indent}return\n")
            else:
                replacement_lines.append(f"{branch_body_indent}return ' '.join({branch_arg_name}[1:])\n")
            updated_lines = [
                *lines[:start],
                *normalized_block_lines[:index],
                *replacement_lines,
                *normalized_block_lines[branch_end_index:],
                *lines[end:],
            ]
            updated_content = "".join(updated_lines)
            if updated_content == current_content:
                return None
            return updated_content
        return None

    def _normalize_direct_main_option_binding_line(
        self,
        raw_line: str,
        *,
        binding_name: str,
    ) -> str:
        pattern = re.compile(
            rf"^(?P<indent>\s*){re.escape(binding_name)}\s*=\s*(?P<expr>.+?)\s*\[\s*1\s*:\s*\]\s*$"
        )
        stripped_line = raw_line.rstrip("\n")
        match = pattern.match(stripped_line)
        if match is None:
            return raw_line
        expression = str(match.group("expr") or "").strip()
        if "argv" not in expression:
            return raw_line
        indent = str(match.group("indent") or "")
        suffix = "\n" if raw_line.endswith("\n") else ""
        return f"{indent}{binding_name} = {expression}{suffix}"

    def _direct_main_expected_payload_output(
        self,
        repair_context: ValidationFailureEvidence,
        *,
        positional_tokens: list[str],
    ) -> str | None:
        if repair_context.verification_scope != "runtime" or not positional_tokens:
            return None
        brief = getattr(repair_context, "repair_brief", None)
        if brief is None:
            return None
        expected_values = [
            _repair_semantic_value_text(item)
            for item in getattr(brief, "expected_semantics", [])
            if _repair_semantic_value_text(item)
        ]
        if not expected_values:
            return None
        expected_output = str(expected_values[0] or "").strip()
        payload_output = " ".join(str(token or "").strip() for token in positional_tokens if str(token or "").strip()).strip()
        if not expected_output or not payload_output or expected_output != payload_output:
            return None
        return expected_output

    def _apply_direct_main_contract_patch(
        self,
        *,
        current_content: str,
        option_tokens: list[str],
    ) -> str | None:
        bounds = self._python_function_block_bounds(current_content, function_name="main")
        if bounds is None:
            return None

        lines = str(current_content or "").splitlines(keepends=True)
        start, end = bounds
        block = "".join(lines[start:end])
        if "argv" not in block:
            return None

        updated_block = self._rewrite_equivalent_option_literals(block, option_tokens=option_tokens)
        replacements: tuple[tuple[re.Pattern[str], str], ...] = (
            (re.compile(r"\bargv\s*\[\s*2\s*:\s*\]"), "argv[1:]"),
            (re.compile(r"\bargv\s*\[\s*2\s*\]"), "argv[1]"),
            (re.compile(r"\bargv\s*\[\s*1\s*\]"), "argv[0]"),
            (re.compile(r"\blen\(\s*argv\s*\)\s*>\s*2\b"), "len(argv) > 1"),
            (re.compile(r"\blen\(\s*argv\s*\)\s*>=\s*3\b"), "len(argv) >= 2"),
            (re.compile(r"\blen\(\s*argv\s*\)\s*==\s*2\b"), "len(argv) == 1"),
            (re.compile(r"\blen\(\s*argv\s*\)\s*>\s*1\b"), "len(argv) > 0"),
            (re.compile(r"\blen\(\s*argv\s*\)\s*>=\s*2\b"), "len(argv) >= 1"),
        )
        for pattern, replacement in replacements:
            updated_block = pattern.sub(replacement, updated_block)
        updated_block = re.sub(r"\bargv\s+and\s+len\(argv\)\s*>\s*0\b", "argv", updated_block)
        updated_block = re.sub(r"\bargv\s+and\s+len\(argv\)\s*>=\s*1\b", "argv", updated_block)
        if updated_block == block:
            return None

        updated_lines = [*lines[:start], updated_block, *lines[end:]]
        return "".join(updated_lines)

    def _canonical_cli_option_token(self, token: str) -> str:
        normalized = str(token or "").strip().lower()
        if normalized.startswith("--"):
            normalized = normalized[2:]
        return re.sub(r"[^a-z0-9]", "", normalized)

    def _rewrite_equivalent_option_literals(
        self,
        text: str,
        *,
        option_tokens: list[str],
    ) -> str:
        canonical_options = {
            self._canonical_cli_option_token(token): token
            for token in option_tokens
            if self._canonical_cli_option_token(token)
        }
        if not canonical_options:
            return text

        def _replace(match: re.Match[str]) -> str:
            quote = str(match.group("quote") or "")
            literal = str(match.group("literal") or "")
            replacement = canonical_options.get(self._canonical_cli_option_token(literal))
            if replacement is None or replacement == literal:
                return match.group(0)
            return f"{quote}{replacement}{quote}"

        return re.sub(
            r"(?P<quote>['\"])(?P<literal>[A-Za-z0-9][A-Za-z0-9_-]*)(?P=quote)",
            _replace,
            text,
        )

    def _python_function_block_bounds(
        self,
        content: str,
        *,
        function_name: str,
    ) -> tuple[int, int] | None:
        lines = str(content or "").splitlines(keepends=True)
        pattern = re.compile(rf"^(?P<indent>\s*)def\s+{re.escape(function_name)}\s*\(")
        start_index: int | None = None
        base_indent = 0
        for index, raw_line in enumerate(lines):
            match = pattern.match(raw_line)
            if match is None:
                continue
            start_index = index
            base_indent = len(str(match.group("indent") or ""))
            break
        if start_index is None:
            return None

        end_index = len(lines)
        for index in range(start_index + 1, len(lines)):
            raw_line = lines[index]
            stripped = raw_line.strip()
            if not stripped:
                continue
            current_indent = len(raw_line) - len(raw_line.lstrip())
            if current_indent <= base_indent and not raw_line.lstrip().startswith("#"):
                end_index = index
                break
        return start_index, end_index

    def _direct_main_option_contract_review(
        self,
        session: SessionState,
        *,
        path: str,
        current_content: str,
        proposed_content: str,
        repair_context: ValidationFailureEvidence | None,
    ) -> ProposedUpdateReview | None:
        if Path(path).suffix.lower() not in {".py", ".pyi"}:
            return None
        if not self._path_exposes_direct_main_contract_target(
            session,
            path=path,
            current_content=current_content,
            proposed_content=proposed_content,
        ):
            return None
        if repair_context is not None and repair_context.verification_scope != "runtime":
            return None

        if repair_context is not None:
            option_tokens, positional_tokens = self._direct_main_option_contract_details(
                session,
                repair_context,
            )
        else:
            option_tokens, positional_tokens = self._session_direct_main_option_contract_details(
                session,
            )
        if not option_tokens:
            return None

        lowered_current = str(current_content or "").lower()
        lowered_proposed = str(proposed_content or "").lower()
        target_evidence = (
            self._runtime_target_evidence_lines(path, repair_context)
            if repair_context is not None
            else []
        )
        missing_exact_tokens = [
            token for token in option_tokens if token.lower() not in lowered_proposed
        ]
        if missing_exact_tokens:
            option_preview = ", ".join(missing_exact_tokens[:3])
            positional_preview = ", ".join(repr(token) for token in positional_tokens[:3])
            repair_hints = [
                f"Handle the exact option token sequence {option_preview} in {path}; do not rewrite those tokens into stripped-hyphen or underscore-only lookalikes.",
                "Keep the repair grounded in the direct main([...]) contract instead of approximating the option spelling.",
            ]
            if positional_tokens:
                repair_hints.insert(
                    1,
                    f"After recognizing {option_preview}, derive behavior from the remaining argv payload {positional_preview} instead of hardcoding those sample values.",
                )
            if repair_context is not None:
                repair_hints.extend(
                    self._runtime_target_repair_hints(
                        path,
                        repair_context,
                        evidence_lines=target_evidence,
                    )
                )
            return ProposedUpdateReview(
                safe_to_write=False,
                summary=(
                    "The proposed repair still misses the exact direct main([...]) option tokens exercised by the failed runtime path."
                ),
                confidence=0.92,
                blocking_issues=[
                    (
                        f"The failing runtime path passes option tokens like {option_preview} into main([...]), "
                        f"but the proposal never recognizes those exact tokens in {path}."
                    )
                ],
                preservation_risks=[],
                repair_hints=repair_hints[:4],
            )

        argv_contract_misindex_review = self._direct_main_option_contract_argv_index_review(
            path=path,
            proposed_content=proposed_content,
            option_tokens=option_tokens,
            positional_tokens=positional_tokens,
            target_evidence=target_evidence,
            repair_context=repair_context,
        )
        if argv_contract_misindex_review is not None:
            return argv_contract_misindex_review

        current_contract_lines = self._runtime_argv_contract_lines(current_content)
        proposed_contract_lines = self._runtime_argv_contract_lines(proposed_content)
        if not current_contract_lines or current_contract_lines != proposed_contract_lines:
            return None

        if any(token.lower() in lowered_current for token in option_tokens):
            return None

        option_preview = ", ".join(option_tokens[:3])
        evidence_hint = f" near {' | '.join(target_evidence[:2])}" if target_evidence else ""
        repair_hints = [
            f"Change the argv-handling lines in {path} so the direct main([...]) invocation recognizes option tokens like {option_preview} before formatting stdout.",
            "Do not stop at output-only formatting changes if the provided runtime option tokens are still ignored.",
        ]
        if repair_context is not None:
            repair_hints.extend(
                self._runtime_target_repair_hints(
                    path,
                    repair_context,
                    evidence_lines=target_evidence,
                )
            )
        return ProposedUpdateReview(
            safe_to_write=False,
            summary=(
                "The proposed repair still ignores the direct main([...]) option contract exercised by the failed runtime path."
            ),
            confidence=0.9,
            blocking_issues=[
                (
                    f"The failing runtime path passes option tokens like {option_preview} into main([...]), "
                    f"but the proposal leaves the argv-handling lines in {path} unchanged{evidence_hint}."
                )
            ],
            preservation_risks=[],
            repair_hints=repair_hints[:4],
        )

    def _direct_main_option_contract_details(
        self,
        session: SessionState,
        repair_context: ValidationFailureEvidence,
        *,
        limit: int = 4,
    ) -> tuple[list[str], list[str]]:
        if repair_context.verification_scope != "runtime":
            return [], []

        command_paths = self._unique_paths(
            [
                str(candidate or "").strip()
                for candidate in self.validation_planner._paths_from_explicit_test_command(
                    str(getattr(repair_context, "command", "") or "").strip()
                )
                if str(candidate or "").strip() and self._path_is_test_like(str(candidate or "").strip())
            ]
        )
        fallback_paths = self._unique_paths(
            [
                *[
                    str(candidate or "").strip()
                    for candidate in getattr(repair_context, "file_hints", []) or []
                    if str(candidate or "").strip() and self._path_is_test_like(str(candidate or "").strip())
                ],
                *(
                    list(getattr(session.workspace_snapshot, "test_files", []) or [])
                    if session.workspace_snapshot is not None
                    else []
                ),
            ]
        )
        candidate_paths = command_paths or fallback_paths
        if not candidate_paths:
            return [], []

        tokens: list[str] = []
        positional_tokens: list[str] = []
        for candidate in candidate_paths[:4]:
            excerpt = self._current_or_last_read_excerpt(session, path=candidate)
            if "main([" not in excerpt:
                continue
            option_candidates, positional_candidates = _direct_main_runtime_contract(
                excerpt,
                limit=limit,
            )
            for token in option_candidates:
                if token and token not in tokens:
                    tokens.append(token)
                    if len(tokens) >= limit and positional_tokens:
                        return tokens[:limit], positional_tokens[:limit]
            if positional_candidates:
                positional_tokens = positional_candidates[:limit]
                if len(positional_tokens) >= limit:
                    return tokens[:limit], positional_tokens[:limit]
        return tokens[:limit], positional_tokens[:limit]

    def _session_direct_main_option_contract_details(
        self,
        session: SessionState,
        *,
        limit: int = 4,
    ) -> tuple[list[str], list[str]]:
        route_target_paths = (
            list(getattr(getattr(session, "router_result", None), "entities", None).target_paths or [])
            if getattr(getattr(session, "router_result", None), "entities", None) is not None
            else []
        )
        candidate_paths = self._unique_paths(
            [
                *[
                    str(item.tool_args.get("path") or "").strip()
                    for item in reversed(session.tool_calls)
                    if item.tool_name == "read_file"
                    and self._path_is_test_like(str(item.tool_args.get("path") or "").strip())
                ],
                *[
                    str(path or "").strip()
                    for path in route_target_paths
                    if str(path or "").strip() and self._path_is_test_like(str(path or "").strip())
                ],
                *(
                    list(getattr(session.workspace_snapshot, "test_files", []) or [])
                    if session.workspace_snapshot is not None
                    else []
                ),
            ]
        )
        if not candidate_paths:
            return [], []

        option_tokens: list[str] = []
        positional_tokens: list[str] = []
        for candidate in candidate_paths[:8]:
            excerpt = self._current_or_last_read_excerpt(session, path=candidate)
            if "main([" not in excerpt:
                continue
            option_candidates, positional_candidates = _direct_main_runtime_contract(
                excerpt,
                limit=limit,
            )
            for token in option_candidates:
                if token and token not in option_tokens:
                    option_tokens.append(token)
                    if len(option_tokens) >= limit:
                        break
            for token in positional_candidates:
                if token and token not in positional_tokens:
                    positional_tokens.append(token)
                    if len(positional_tokens) >= limit:
                        break
            if len(option_tokens) >= limit and len(positional_tokens) >= limit:
                break
        return option_tokens[:limit], positional_tokens[:limit]

    def _path_exposes_direct_main_contract_target(
        self,
        session: SessionState,
        *,
        path: str,
        current_content: str,
        proposed_content: str,
    ) -> bool:
        normalized_path = str(path or "").strip()
        if not normalized_path:
            return False
        if normalized_path == "__main__.py" or Path(normalized_path).name == "__main__.py":
            return True
        if normalized_path in (getattr(session.workspace_snapshot, "entrypoints", []) or []):
            return True
        combined = "\n".join([str(current_content or ""), str(proposed_content or "")])
        return bool(re.search(r"^\s*def\s+main\s*\(", combined, re.MULTILINE))

    def _direct_python_script_option_contract_details(
        self,
        repair_context: ValidationFailureEvidence,
        *,
        path: str,
        limit: int = 4,
    ) -> tuple[list[str], list[str]]:
        if repair_context.verification_scope != "runtime":
            return [], []
        return _direct_python_script_runtime_contract(
            repair_context.command,
            target_path=path,
            limit=limit,
        )

    def _python_constant_string_sequence(
        self,
        node: ast.AST,
    ) -> tuple[str, ...] | None:
        if not isinstance(node, (ast.List, ast.Tuple)):
            return None
        values: list[str] = []
        for element in node.elts:
            if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                return None
            values.append(str(element.value))
        return tuple(values)

    def _python_prefix_slice_contract(
        self,
        sequence_node: ast.AST,
        literal_node: ast.AST,
    ) -> PythonSequencePrefixContract | None:
        if not isinstance(sequence_node, ast.Subscript):
            return None
        slice_node = sequence_node.slice
        if not isinstance(slice_node, ast.Slice):
            return None
        if slice_node.lower is not None or slice_node.step is not None:
            return None
        upper = slice_node.upper
        if not isinstance(upper, ast.Constant) or not isinstance(upper.value, int):
            return None
        literal_tokens = self._python_constant_string_sequence(literal_node)
        if literal_tokens is None:
            return None
        try:
            variable_expression = ast.unparse(sequence_node.value).strip()
        except Exception:
            return None
        if not variable_expression:
            return None
        return PythonSequencePrefixContract(
            variable_expression=variable_expression,
            slice_length=int(upper.value),
            literal_tokens=literal_tokens,
            lineno=int(getattr(sequence_node, "lineno", 0) or 0),
        )

    def _python_prefix_sequence_contracts(
        self,
        content: str,
    ) -> list[PythonSequencePrefixContract]:
        try:
            tree = ast.parse(str(content or ""))
        except SyntaxError:
            return []
        contracts: list[PythonSequencePrefixContract] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq) or len(node.comparators) != 1:
                continue
            comparator = node.comparators[0]
            for sequence_node, literal_node in ((node.left, comparator), (comparator, node.left)):
                contract = self._python_prefix_slice_contract(sequence_node, literal_node)
                if contract is not None:
                    contracts.append(contract)
                    break
        return contracts

    def _python_sequence_accesses(
        self,
        content: str,
        *,
        variable_expression: str,
    ) -> tuple[set[int], set[int]]:
        try:
            tree = ast.parse(str(content or ""))
        except SyntaxError:
            return set(), set()
        index_accesses: set[int] = set()
        slice_starts: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue
            try:
                base_expression = ast.unparse(node.value).strip()
            except Exception:
                continue
            if base_expression != variable_expression:
                continue
            slice_node = node.slice
            if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, int):
                index_accesses.add(int(slice_node.value))
                continue
            if (
                isinstance(slice_node, ast.Slice)
                and isinstance(slice_node.lower, ast.Constant)
                and isinstance(slice_node.lower.value, int)
                and slice_node.upper is None
                and slice_node.step is None
            ):
                slice_starts.add(int(slice_node.lower.value))
        return index_accesses, slice_starts

    def _direct_python_script_option_contract_review(
        self,
        *,
        path: str,
        proposed_content: str,
        repair_context: ValidationFailureEvidence | None,
    ) -> ProposedUpdateReview | None:
        if repair_context is None or repair_context.verification_scope != "runtime":
            return None
        if Path(path).suffix.lower() not in {".py", ".pyi"}:
            return None

        option_tokens, tail_tokens = self._direct_python_script_option_contract_details(
            repair_context,
            path=path,
        )
        if not option_tokens:
            return None

        lowered_proposed = str(proposed_content or "").lower()
        target_evidence = self._runtime_target_evidence_lines(path, repair_context)
        option_preview = ", ".join(option_tokens[:3])
        tail_preview = ", ".join(repr(token) for token in tail_tokens[:3])
        missing_exact_tokens = [
            token for token in option_tokens if token.lower() not in lowered_proposed
        ]
        if missing_exact_tokens:
            repair_hints = [
                f"Handle the exact direct script option token sequence {option_preview} in {path}; do not rewrite those tokens into stripped-hyphen or underscore-only lookalikes.",
            ]
            if tail_tokens:
                repair_hints.append(
                    f"After recognizing {option_preview}, derive behavior from the remaining argv payload {tail_preview} instead of hardcoding those sample values."
                )
            repair_hints.extend(
                self._runtime_target_repair_hints(
                    path,
                    repair_context,
                    evidence_lines=target_evidence,
                )
            )
            return ProposedUpdateReview(
                safe_to_write=False,
                summary=(
                    "The proposed repair still misses the exact direct python script option tokens exercised by the failed runtime path."
                ),
                confidence=0.92,
                blocking_issues=[
                    (
                        f"The failing runtime command invokes {Path(path).name} with option tokens like {option_preview}, "
                        f"but the proposal never recognizes those exact tokens in {path}."
                    )
                ],
                preservation_risks=[],
                repair_hints=repair_hints[:4],
            )

        prefix_contracts = [
            contract
            for contract in self._python_prefix_sequence_contracts(proposed_content)
            if any(token in option_tokens for token in contract.literal_tokens)
        ]
        for contract in prefix_contracts:
            literal_preview = ", ".join(repr(token) for token in contract.literal_tokens)
            if (
                len(contract.literal_tokens) > len(option_tokens)
                and list(contract.literal_tokens[: len(option_tokens)]) == option_tokens
                and any(
                    token in tail_tokens
                    for token in contract.literal_tokens[len(option_tokens) :]
                )
            ):
                extra_tokens = contract.literal_tokens[len(option_tokens) :]
                extra_preview = ", ".join(repr(token) for token in extra_tokens[:3])
                repair_hints = [
                    f"Treat the direct script option prefix {option_preview} as the contract anchor in {path}; do not hardcode illustrative runtime payload tokens like {extra_preview} into the branch guard.",
                ]
                if tail_tokens:
                    repair_hints.append(
                        f"After recognizing {option_preview}, derive behavior from the remaining argv payload {tail_preview} instead of matching one sample payload token literally."
                    )
                repair_hints.extend(
                    self._runtime_target_repair_hints(
                        path,
                        repair_context,
                        evidence_lines=target_evidence,
                    )
                )
                return ProposedUpdateReview(
                    safe_to_write=False,
                    summary=(
                        "The proposed repair still hardcodes illustrative direct python script payload tokens into the exercised option-prefix branch."
                    ),
                    confidence=0.9,
                    blocking_issues=[
                        (
                            f"The proposal guards {contract.variable_expression} with {literal_preview} in {path}, "
                            f"which bakes the sample runtime payload token(s) {extra_preview} into the prefix match "
                            "instead of treating them as data that should flow through the repaired path."
                        )
                    ],
                    preservation_risks=[],
                    repair_hints=repair_hints[:4],
                )
            if contract.slice_length != len(contract.literal_tokens):
                repair_hints = [
                    f"If you compare {contract.variable_expression}[:N] against a literal option sequence in {path}, keep the slice length aligned with the compared literal tokens.",
                    f"Match the exercised option prefix {option_preview} on the real argv payload after {Path(path).name}; do not write a branch that can never satisfy the runtime contract.",
                ]
                repair_hints.extend(
                    self._runtime_target_repair_hints(
                        path,
                        repair_context,
                        evidence_lines=target_evidence,
                    )
                )
                return ProposedUpdateReview(
                    safe_to_write=False,
                    summary=(
                        "The proposed repair still misstates the direct python script argv prefix contract exercised by the failed runtime path."
                    ),
                    confidence=0.9,
                    blocking_issues=[
                        (
                            f"The proposal compares {contract.variable_expression}[:{contract.slice_length}] against "
                            f"{literal_preview} in {path}, but that branch can never match because the slice length "
                            "and literal token count differ."
                        )
                    ],
                    preservation_risks=[],
                    repair_hints=repair_hints[:4],
                )

        payload_start = len(option_tokens)
        for contract in prefix_contracts:
            if payload_start <= 0:
                break
            if list(contract.literal_tokens) != option_tokens[: len(contract.literal_tokens)]:
                continue
            if contract.slice_length != payload_start:
                continue
            index_accesses, slice_starts = self._python_sequence_accesses(
                proposed_content,
                variable_expression=contract.variable_expression,
            )
            suspicious_slice_starts = sorted(
                start for start in slice_starts if start > payload_start
            )
            if (
                suspicious_slice_starts
                and payload_start not in index_accesses
                and payload_start not in slice_starts
            ):
                suspicious_start = suspicious_slice_starts[0]
                repair_hints = [
                    f"Once {contract.variable_expression}[:{payload_start}] matches {option_preview}, consume the remaining payload from {contract.variable_expression}[{payload_start}:] or {contract.variable_expression}[{payload_start}] onward instead of skipping past the first non-option token.",
                ]
                if tail_tokens:
                    repair_hints.append(
                        f"The exercised runtime payload after {option_preview} begins with {tail_preview}; do not drop that first payload token when formatting stdout."
                    )
                repair_hints.extend(
                    self._runtime_target_repair_hints(
                        path,
                        repair_context,
                        evidence_lines=target_evidence,
                    )
                )
                return ProposedUpdateReview(
                    safe_to_write=False,
                    summary=(
                        "The proposed repair still drops the first direct python script payload token after matching the exercised option prefix."
                    ),
                    confidence=0.88,
                    blocking_issues=[
                        (
                            f"The proposal recognizes {option_preview} in {contract.variable_expression}[:{payload_start}] "
                            f"but then consumes {contract.variable_expression}[{suspicious_start}:], which skips the first "
                            "runtime payload token required by the failing path."
                        )
                    ],
                    preservation_risks=[],
                    repair_hints=repair_hints[:4],
                )
        return None

    def _direct_main_option_contract_argv_index_review(
        self,
        *,
        path: str,
        proposed_content: str,
        option_tokens: list[str],
        positional_tokens: list[str],
        target_evidence: list[str],
        repair_context: ValidationFailureEvidence | None,
    ) -> ProposedUpdateReview | None:
        if not option_tokens or not positional_tokens:
            return None

        lowered_proposed = str(proposed_content or "").lower()
        if not any(marker in lowered_proposed for marker in ("argv[1]", "argv[2:]", "argv[2]")):
            return None
        if "argv[0]" in lowered_proposed or "argv[1:]" in lowered_proposed:
            return None

        option_preview = ", ".join(option_tokens[:3])
        positional_preview = ", ".join(repr(token) for token in positional_tokens[:3])
        repair_hints = [
            f"The direct main([...]) contract already provides argv itself, so treat {option_preview} as the leading runtime tokens instead of assuming a launcher/program name at argv[0].",
            f"After handling {option_preview}, consume the remaining argv payload {positional_preview} directly instead of skipping past it with launcher-style indices like argv[2:].",
            "For direct main([...]) tests, do not read argv[1] as the first option token unless the function normalized argv earlier in the same code path.",
        ]
        if repair_context is not None:
            repair_hints.extend(
                self._runtime_target_repair_hints(
                    path,
                    repair_context,
                    evidence_lines=target_evidence,
                )
            )
        return ProposedUpdateReview(
            safe_to_write=False,
            summary=(
                "The proposed repair still treats direct main([...]) argv input as though argv[0] were a launcher or program name."
            ),
            confidence=0.91,
            blocking_issues=[
                (
                    f"The failing direct main([...]) path passes option tokens like {option_preview} and payload "
                    f"{positional_preview} into {path}, but the proposal still uses launcher-style argv indexing "
                    "such as argv[1] or argv[2:]."
                )
            ],
            preservation_risks=[],
            repair_hints=repair_hints[:4],
        )

    def _direct_main_option_contract_tokens(
        self,
        session: SessionState,
        repair_context: ValidationFailureEvidence,
        *,
        limit: int = 4,
    ) -> list[str]:
        tokens, _ = self._direct_main_option_contract_details(
            session,
            repair_context,
            limit=limit,
        )
        return tokens

    def _runtime_argv_contract_lines(
        self,
        content: str,
    ) -> tuple[str, ...]:
        lines: list[str] = []
        for raw_line in str(content or "").splitlines():
            stripped = raw_line.strip()
            lowered = stripped.lower()
            if not stripped:
                continue
            if "argv" in lowered or "parse_args(" in lowered:
                lines.append(stripped)
        return tuple(lines[:8])

    def _repair_target_scope_review(
        self,
        *,
        session: SessionState,
        path: str,
        repair_context: ValidationFailureEvidence | None,
    ) -> ProposedUpdateReview | None:
        if repair_context is None:
            return None
        brief = getattr(repair_context, "repair_brief", None)
        if brief is None:
            return None

        normalized_path = str(path or "").strip()
        if not normalized_path:
            return None

        allowed_files = {
            str(item or "").strip()
            for item in getattr(brief, "allowed_files", [])
            if str(item or "").strip()
        }
        forbidden_files = {
            str(item or "").strip()
            for item in getattr(brief, "forbidden_files", [])
            if str(item or "").strip()
        }
        locked_target = str(getattr(brief, "locked_target", "") or "").strip()
        primary_target = str(getattr(brief, "primary_target", "") or "").strip()

        if normalized_path in forbidden_files:
            return ProposedUpdateReview(
                safe_to_write=False,
                summary="The proposed repair drifts into a file that the repair brief explicitly marked as out of scope.",
                confidence=0.92,
                blocking_issues=[
                    f"The repair brief forbids editing {normalized_path} for this failure.",
                ],
                preservation_risks=[],
                repair_hints=[
                    "Stay on the primary implementation target from the repair brief.",
                ],
            )
        if allowed_files and normalized_path not in allowed_files:
            return ProposedUpdateReview(
                safe_to_write=False,
                summary="The proposed repair targets a file outside the repair brief's allowed scope.",
                confidence=0.9,
                blocking_issues=[
                    f"The repair brief only allows edits in: {', '.join(sorted(allowed_files)[:4])}.",
                ],
                preservation_risks=[],
                repair_hints=[
                    "Return to the locked or primary repair target instead of drifting into a different file.",
                ],
            )

        locked_scope = locked_target or primary_target
        if (
            locked_scope
            and locked_scope != normalized_path
            and not self._runtime_locked_target_should_yield(session, repair_context, locked_scope)
            and not self._is_runtime_support_repair_target(normalized_path, repair_context)
        ):
            return ProposedUpdateReview(
                safe_to_write=False,
                summary="The proposed repair moved away from the locked repair target without strong new evidence.",
                confidence=0.89,
                blocking_issues=[
                    f"The repair brief is locked to {locked_scope}, not {normalized_path}.",
                ],
                preservation_risks=[],
                repair_hints=[
                    "Keep the repair on the locked target unless new validation evidence clearly points somewhere else.",
                ],
            )
        return None

    def _pre_write_create_review(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        proposed_content: str,
    ) -> ProposedUpdateReview:
        review = self._generated_content_integrity_review(path=path, proposed_content=proposed_content)
        if review is None:
            review = self._explicit_create_constraint_review(
                route,
                session,
                path=path,
                proposed_content=proposed_content,
            )
        if review is not None:
            return review
        return ProposedUpdateReview(
            safe_to_write=True,
            summary=f"The proposed new file for {path} passes the local structural integrity checks.",
            confidence=0.56,
            blocking_issues=[],
            preservation_risks=[],
            repair_hints=[
                "Keep the new file concrete and complete.",
            ],
        )

    def _local_pre_write_update_review(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str,
        proposed_content: str,
        repair_context: ValidationFailureEvidence | None,
    ) -> ProposedUpdateReview:
        review = self._generated_content_integrity_review(path=path, proposed_content=proposed_content)
        if review is None:
            review = self._explicit_constraint_integrity_review(
                route,
                session,
                path=path,
                current_content=current_content,
                proposed_content=proposed_content,
            )
        if review is None:
            review = self._helper_entrypoint_scope_review(
                session,
                path=path,
                current_content=current_content,
                proposed_content=proposed_content,
            )
        if review is None:
            review = self._entrypoint_helper_contract_review(
                path=path,
                current_content=current_content,
                proposed_content=proposed_content,
            )
        if review is None:
            review = self._repair_target_scope_review(
                session=session,
                path=path,
                repair_context=repair_context,
            )
        if review is None:
            review = self._cli_wrapper_responsibility_review(
                session,
                path=path,
                proposed_content=proposed_content,
            )
        if review is None:
            review = self._direct_main_option_contract_review(
                session,
                path=path,
                current_content=current_content,
                proposed_content=proposed_content,
                repair_context=repair_context,
            )
        if review is None:
            review = self._direct_python_script_option_contract_review(
                path=path,
                proposed_content=proposed_content,
                repair_context=repair_context,
            )
        if review is None:
            review = self._pre_write_web_contract_review(
                route,
                session,
                path=path,
                proposed_content=proposed_content,
            )
        if review is None:
            review = self._validation_repair_relevance_review(
                path=path,
                current_content=current_content,
                proposed_content=proposed_content,
                repair_context=repair_context,
                session=session,
            )
        if review is None:
            review = self._repair_no_effective_change_review(
                path=path,
                current_content=current_content,
                proposed_content=proposed_content,
                repair_context=repair_context,
            )
        if review is None:
            review = self._fallback_proposed_update_review(
                route,
                session=session,
                path=path,
                current_content=current_content,
                proposed_content=proposed_content,
            )
        review = self._sanitize_model_backed_review(
            route,
            session,
            path=path,
            current_content=current_content,
            review=review,
        )
        return review

    def _retry_update_after_review_failure(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str | None,
        review_feedback: ProposedUpdateReview,
        repair_context: ValidationFailureEvidence | None,
        repair_strategy: str | None,
        prior_attempts: list[ExecutionAttemptRecord],
    ) -> UpdateReviewRetryResult:
        retry_attempts: list[ExecutionAttemptRecord] = []
        last_review = review_feedback
        primary_model = self._primary_generation_model_name()
        reserve_model = self._generation_recovery_model_name() or self._lightweight_generation_model_name()
        prefer_compact_create_retry = current_content is None
        prefer_lightweight_retry = current_content is not None and self._should_prefer_lightweight_update_generation(
            route,
            session,
            path=path,
            current_content=current_content,
        )
        prefer_primary_repair_retry = repair_context is not None
        prefer_reserve_after_initial_noop = self._prefer_reserve_after_initial_primary_noop(
            review_feedback=review_feedback,
            repair_context=repair_context,
            primary_model=primary_model,
            reserve_model=reserve_model,
            prior_attempts=prior_attempts,
        )
        keep_compact_update_retry = (
            repair_context is None
            and current_content is not None
            and self._should_keep_update_review_compact(
                route,
                session,
                path=path,
                current_content=current_content,
            )
        )

        deterministic_recovery = self._deterministic_direct_main_runtime_recovery(
            route,
            session,
            path=path,
            current_content=current_content,
            repair_context=repair_context,
            review_feedback=review_feedback,
        )
        if deterministic_recovery is None:
            deterministic_recovery = self._deterministic_direct_python_script_runtime_recovery(
                route,
                session,
                path=path,
                current_content=current_content,
                repair_context=repair_context,
                review_feedback=review_feedback,
            )
        if deterministic_recovery is not None:
            effective_repair_strategy = self._review_retry_repair_strategy(
                repair_strategy,
                review_feedback,
                repair_context,
            )
            self._log(
                "content_generation_recovery_started",
                path=path,
                strategy=deterministic_recovery.recovery_strategy,
                failure_class="proposed_update_review_rejected",
                models=self._generation_models_summary(prior_attempts),
            )
            self._log(
                "content_generation_recovery_finished",
                path=path,
                strategy=deterministic_recovery.recovery_strategy,
                source=deterministic_recovery.recovery_strategy,
            )
            return UpdateReviewRetryResult(
                content=deterministic_recovery.content,
                review=deterministic_recovery.review,
                capability_tier=deterministic_recovery.capability_tier,
                recovery_strategy=deterministic_recovery.recovery_strategy,
                effective_repair_strategy=effective_repair_strategy,
            )

        retry_models: list[tuple[str | None, str, str, int, int, int, str]] = []
        if prefer_compact_create_retry:
            retry_models.append(
                (
                    primary_model,
                    "tier_a",
                    "review_guided_retry",
                    max(self._llm_timeout(45), 45),
                    max(self._llm_timeout(120), 120),
                    min(self._llm_num_ctx(2048), 2048),
                    "compact",
                )
            )
            if not keep_compact_update_retry:
                retry_models.append(
                    (
                        primary_model,
                        "tier_a",
                        "review_guided_primary_fallback",
                        max(self._llm_timeout(60), 60),
                        max(self._llm_timeout(180), 180),
                        min(self._llm_num_ctx(3072), 3072),
                        "full",
                    )
                )
            if reserve_model is not None:
                retry_models.append(
                    (
                        reserve_model,
                        "tier_b",
                        "review_guided_fallback_model",
                        max(self._llm_timeout(45), 45),
                        max(self._llm_timeout(120), 120),
                        min(self._llm_num_ctx(2048), 2048),
                        "compact",
                    )
                )
        elif prefer_primary_repair_retry:
            followup_timeout_seconds, followup_total_timeout_seconds = self._repair_followup_retry_budget()
            retry_models.append(
                (
                    primary_model,
                    "tier_a",
                    "review_guided_retry",
                    max(self._llm_timeout(60), 60),
                    max(self._llm_timeout(210), 210),
                    min(self._llm_num_ctx(2048), 2048),
                    "compact",
                )
            )
            if prefer_reserve_after_initial_noop:
                if reserve_model is not None and reserve_model != primary_model:
                    retry_models.append(
                        (
                            reserve_model,
                            "tier_b",
                            "review_guided_fallback_model",
                            followup_timeout_seconds,
                            followup_total_timeout_seconds,
                            min(self._llm_num_ctx(3072), 3072),
                            "full",
                        )
                    )
            else:
                retry_models.append(
                    (
                        primary_model,
                        "tier_a",
                        "review_guided_retry_followup",
                        followup_timeout_seconds,
                        followup_total_timeout_seconds,
                        min(self._llm_num_ctx(4096), 4096),
                        "full",
                    )
                )
                if reserve_model is not None and reserve_model != primary_model:
                    retry_models.append(
                        (
                            reserve_model,
                            "tier_b",
                            "review_guided_fallback_model",
                            followup_timeout_seconds,
                            followup_total_timeout_seconds,
                            min(self._llm_num_ctx(3072), 3072),
                            "full",
                        )
                    )
        elif prefer_lightweight_retry and reserve_model is not None:
            retry_models.append(
                (
                    reserve_model,
                    "tier_b",
                    "review_guided_retry",
                    max(self._llm_timeout(45), 45),
                    max(self._llm_timeout(120), 120),
                    min(self._llm_num_ctx(2048), 2048),
                    "compact",
                )
            )
        elif keep_compact_update_retry:
            followup_timeout_seconds, followup_total_timeout_seconds = self._repair_followup_retry_budget()
            retry_models.append(
                (
                    primary_model,
                    "tier_a",
                    "review_guided_retry",
                    max(self._llm_timeout(45), 45),
                    max(self._llm_timeout(150), 150),
                    min(self._llm_num_ctx(2048), 2048),
                    "compact",
                )
            )
            retry_models.append(
                (
                    primary_model,
                    "tier_a",
                    "review_guided_retry_followup",
                    max(self._llm_timeout(45), 45),
                    max(self._llm_timeout(150), 150),
                    min(self._llm_num_ctx(2048), 2048),
                    "compact",
                )
            )
            if reserve_model is not None and reserve_model != primary_model:
                retry_models.append(
                    (
                        reserve_model,
                        "tier_b",
                        "review_guided_fallback_model",
                        followup_timeout_seconds,
                        followup_total_timeout_seconds,
                        min(self._llm_num_ctx(2048), 2048),
                        "compact",
                    )
                )
        if prefer_lightweight_retry and reserve_model is not None and keep_compact_update_retry:
            retry_models.append(
                (
                    primary_model,
                    "tier_a",
                    "review_guided_primary_followup",
                    max(self._llm_timeout(45), 45),
                    max(self._llm_timeout(150), 150),
                    min(self._llm_num_ctx(2048), 2048),
                    "compact",
                )
            )
        if not keep_compact_update_retry and not prefer_primary_repair_retry:
            retry_models.append(
                (
                    primary_model,
                    "tier_a",
                    "review_guided_retry"
                    if not prefer_lightweight_retry
                    else "review_guided_primary_fallback",
                    max(self._llm_timeout(60), 60),
                    max(self._llm_timeout(180), 180),
                    min(self._llm_num_ctx(4096), 4096),
                    "full",
                )
            )
        if reserve_model is not None and not prefer_lightweight_retry and not keep_compact_update_retry:
            retry_models.append(
                (
                    reserve_model,
                    "tier_b",
                    "review_guided_fallback_model",
                    max(self._llm_timeout(45), 45),
                    max(self._llm_timeout(120), 120),
                    min(self._llm_num_ctx(2048), 2048),
                    "compact",
                )
            )

        seen_retry_variants: set[tuple[str, str, str]] = set()
        seen_retry_prompts: set[tuple[str, str]] = set()
        for (
            model_name,
            capability_tier,
            strategy,
            timeout_seconds,
            total_timeout_seconds,
            num_ctx,
            prompt_variant,
        ) in retry_models:
            effective_repair_strategy = self._review_retry_repair_strategy(
                repair_strategy,
                last_review,
                repair_context,
            )
            normalized_model = str(model_name or "").strip()
            variant_key = (normalized_model or "__default__", prompt_variant, strategy)
            if variant_key in seen_retry_variants:
                continue
            seen_retry_variants.add(variant_key)

            prompt = generate_content_retry_prompt(
                route,
                session,
                path=path,
                current_content=current_content,
                repair_context=repair_context,
                repair_strategy=effective_repair_strategy,
                review_feedback=last_review,
                mode=prompt_variant,
            )
            prompt_sha = self._prompt_sha256(prompt)
            prompt_key = (normalized_model or primary_model or "__default__", prompt_sha)
            if prompt_key in seen_retry_prompts:
                self._log(
                    "content_generation_retry_skipped",
                    path=path,
                    strategy=strategy,
                    reason="duplicate_prompt",
                    model=model_name or primary_model,
                    capability_tier=capability_tier,
                    prompt_variant=prompt_variant,
                    prompt_sha256=prompt_sha,
                )
                continue
            seen_retry_prompts.add(prompt_key)
            prompt_trace_path = self._write_prompt_trace(
                session,
                operation_name=f"content_generation_{strategy}",
                path=path,
                prompt=prompt,
                model=model_name or primary_model,
                prompt_variant=prompt_variant,
                num_ctx=num_ctx,
                timeout_seconds=timeout_seconds,
                total_timeout_seconds=total_timeout_seconds,
            )
            self._log(
                "content_generation_retry_started",
                path=path,
                strategy=strategy,
                model=model_name or primary_model,
                reason="proposed_update_review_rejected",
                capability_tier=capability_tier,
                prompt_variant=prompt_variant,
                repair_strategy=effective_repair_strategy,
                prompt_chars=len(prompt),
                prompt_lines=prompt.count("\n") + 1,
                prompt_sha256=prompt_sha,
                prompt_artifact=prompt_trace_path,
            )
            outcome = invoke_model(
                lambda progress, retry_model=model_name: self.llm.generate(
                    prompt,
                    model=retry_model,
                    timeout=timeout_seconds,
                    total_timeout=total_timeout_seconds,
                    strict_timeouts=prompt_variant == "compact",
                    num_ctx=num_ctx,
                    retries=0,
                    progress_callback=progress,
                ),
                operation_name="content_generation",
                task_class="content_generation",
                attempt_number=len(prior_attempts) + len(retry_attempts) + 1,
                capability_tier=capability_tier,
                recovery_strategy=strategy,
                prompt_variant=prompt_variant,
                model_identifier=model_name or primary_model,
                backend_identifier=self._backend_identifier(),
                inactivity_timeout_seconds=timeout_seconds,
                total_timeout_seconds=total_timeout_seconds,
                context_pressure_estimate=self._context_pressure_estimate(
                    prompt=prompt,
                    current_content=current_content,
                    exc=None,
                ),
                event_callback=self._progress_logger(
                    "content_generation_progress",
                    path=path,
                    update=current_content is not None,
                    strategy=strategy,
                ),
            )
            retry_attempts.append(outcome.attempt)
            if outcome.exception is not None:
                retry_issue = outcome.attempt.failure
                self._log(
                    "content_generation_retry_error",
                    path=path,
                    strategy=strategy,
                    error=str(outcome.exception),
                    reason=retry_issue.reason if retry_issue is not None else "runtime_error",
                    had_progress=retry_issue.had_progress if retry_issue is not None else False,
                    partial_characters=retry_issue.characters if retry_issue is not None else 0,
                    failure_class=retry_issue.classification if retry_issue is not None else "generation_failed",
                    capability_tier=capability_tier,
                )
                continue

            cleaned = self._strip_code_fences(str(outcome.value or "")).strip()
            if not cleaned:
                retry_attempts[-1] = self._empty_response_attempt(
                    base_attempt=outcome.attempt,
                    context_pressure=self._context_pressure_estimate(
                        prompt=prompt,
                        current_content=current_content,
                        exc=None,
                    ),
                )
                self._log(
                    "content_generation_retry_error",
                    path=path,
                    strategy=strategy,
                    error="empty model response",
                    reason="empty_response",
                    had_progress=False,
                    partial_characters=0,
                    failure_class="empty_response",
                    capability_tier=capability_tier,
                )
                continue

            self._log(
                "content_generation_retry_finished",
                path=path,
                strategy=strategy,
                characters=len(cleaned),
            )
            review = self._pre_write_update_review(
                route,
                session,
                path=path,
                current_content=current_content,
                proposed_content=cleaned,
                repair_context=repair_context,
            )
            if review.safe_to_write:
                return UpdateReviewRetryResult(
                    content=cleaned,
                    attempts=retry_attempts,
                    review=review,
                    capability_tier=capability_tier,
                    recovery_strategy=strategy,
                    effective_repair_strategy=effective_repair_strategy,
                )
            self._log(
                "proposed_update_review_rejected",
                path=path,
                summary=review.summary,
                blocking_issues=review.blocking_issues[:4],
                strategy=strategy,
            )
            self._record_noop_review_attempt(
                session,
                target=path,
                current_content=current_content,
                proposed_content=cleaned,
                repair_context=repair_context,
                strategy=effective_repair_strategy,
                review=review,
            )
            last_review = review

        return UpdateReviewRetryResult(
            attempts=retry_attempts,
            review=last_review,
        )

    def _build_update_review_failure(
        self,
        session: SessionState,
        path: str,
        review: ProposedUpdateReview,
        *,
        current_content: str | None,
    ) -> ContentGenerationFailure:
        language = self._session_language(session)
        if current_content is not None and self._review_feedback_is_noop(review):
            reason = self._review_feedback_noop_reason(review)
            blocker_message = f"The routed update for {path} did not produce a substantive repair change ({reason})."
            user_message = self._localized_text(
                language,
                de="Ich konnte keine belastbare inhaltliche Aenderung fuer die Zieldatei ableiten.",
                en="I could not derive a reliable substantive change for the target file.",
            )
            return ContentGenerationFailure(
                stop_reason="no_effective_change",
                failure_class="no_effective_change",
                blocker_message=blocker_message,
                user_message=user_message,
            )
        target_kind = "update" if current_content is not None else "new file content"
        blocker_message = f"Pre-write review rejected the proposed {target_kind} for {path}: {review.summary}"
        user_message = self._localized_text(
            language,
            de=(
                f"Ich habe den vorgeschlagenen Inhalt fuer {path} verworfen, "
                "weil der KI-Vorab-Review einen zu breiten, regressiven oder unvollstaendigen Entwurf erkannt hat. "
                "Ich konnte noch keine belastbare engere Fassung ableiten."
            ),
            en=(
                f"I rejected the proposed content for {path} because the AI review flagged the pre-write draft as too broad, regressive, or incomplete. "
                "I could not derive a reliable narrower draft yet."
            ),
        )
        return ContentGenerationFailure(
            stop_reason="update_review_rejected",
            failure_class="update_review_rejected",
            blocker_message=blocker_message,
            user_message=user_message,
        )

    def _record_proposed_update_review_failure(
        self,
        session: SessionState,
        *,
        path: str,
        review: ProposedUpdateReview,
    ) -> None:
        excerpt_parts: list[str] = []
        if review.blocking_issues:
            excerpt_parts.append(f"Blocking issues: {', '.join(review.blocking_issues[:4])}.")
        if review.preservation_risks:
            excerpt_parts.append(f"Preservation risks: {', '.join(review.preservation_risks[:4])}.")
        excerpt = " ".join(excerpt_parts).strip() or None
        session.diagnostics.append(
            DiagnosticRecord(
                source="proposed_update_review",
                category="preservation_risk",
                severity="error",
                summary=review.summary,
                tool_name="proposed_update_review",
                command=f"internal:proposed_update_review:{path}",
                file_hints=[path],
                action_hints=review.repair_hints[:4],
                excerpt=excerpt,
                iteration=session.iterations,
            )
        )
        session.diagnostics = session.diagnostics[-20:]
        session.last_error = excerpt or review.summary

    def _fallback_proposed_update_review(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str,
        proposed_content: str,
        model_backed_review_failed: bool = False,
        review_failure_reason: str | None = None,
    ) -> ProposedUpdateReview:
        repair_context = session.active_repair_context
        review = self._validation_repair_relevance_review(
            path=path,
            current_content=current_content,
            proposed_content=proposed_content,
            repair_context=repair_context,
            session=session,
        )
        if review is not None:
            return review
        review = self._repair_no_effective_change_review(
            path=path,
            current_content=current_content,
            proposed_content=proposed_content,
            repair_context=repair_context,
        )
        if review is not None:
            return review
        review = self._task_backed_no_effective_change_review(
            route,
            session,
            path=path,
            current_content=current_content,
            proposed_content=proposed_content,
            repair_context=repair_context,
        )
        if review is not None:
            return review
        current_length = max(len(current_content), 1)
        proposed_length = max(len(proposed_content), 1)
        current_lines = max(len(current_content.splitlines()), 1)
        proposed_lines = max(len(proposed_content.splitlines()), 1)
        if self._runtime_support_rewrite_is_allowed(
            session,
            path=path,
            current_content=current_content,
            proposed_content=proposed_content,
        ):
            return ProposedUpdateReview(
                safe_to_write=True,
                summary=(
                    f"A larger rewrite is acceptable for {path} because the runtime validation evidence points to fixture or support data rather than behavior-preserving source code."
                ),
                confidence=0.52,
                repair_hints=[
                    "Keep only the minimal runtime data needed for the failing validation to pass.",
                ],
            )
        if proposed_length < int(current_length * 0.6) and proposed_lines < int(current_lines * 0.7):
            return ProposedUpdateReview(
                safe_to_write=False,
                summary=(
                    f"The proposed update for {path} appears to remove a large amount of existing content without a model-backed confirmation that the broader rewrite is required."
                ),
                confidence=0.72,
                blocking_issues=[
                    "Large unexplained reduction in existing file content during an update",
                ],
                preservation_risks=[
                    "Unrelated existing behavior may have been dropped by the proposed rewrite.",
                ],
                repair_hints=[
                    "Produce a smaller focused update that preserves unrelated existing behavior.",
                ],
            )
        if (
            model_backed_review_failed
            and repair_context is not None
            and repair_context.verification_scope == "structural"
            and (
                bool(getattr(repair_context, "missing_features", []))
                or self.validation_planner.command_identity(repair_context.command).startswith("internal:web_artifact:")
            )
        ):
            return ProposedUpdateReview(
                safe_to_write=True,
                summary=(
                    f"Model-backed pre-write review was unavailable, but the proposed update for {path} is already anchored by deterministic structural validation evidence."
                ),
                confidence=0.41,
                repair_hints=[
                    "Keep the update strictly aligned to the reported missing structural features and avoid unrelated rewrites.",
                ],
            )
        if (
            model_backed_review_failed
            and repair_context is not None
            and not self._should_skip_model_backed_repair_review(
                route,
                session,
                path=path,
                current_content=current_content,
                proposed_content=proposed_content,
                reserve_model=None,
            )
        ):
            locked_target = self._repair_brief_locked_target(repair_context) if repair_context is not None else ""
            target_hint = locked_target or str(path or "").strip()
            failure_detail = str(review_failure_reason or "").strip()
            blocking_issue = (
                f"The model-backed pre-write review for {path} did not complete cleanly, so the update "
                "cannot be written on deterministic fallback checks alone."
            )
            if failure_detail:
                blocking_issue += f" Last review error: {failure_detail}."
            return ProposedUpdateReview(
                safe_to_write=False,
                summary=(
                    "The model-backed pre-write review was unavailable after an attempted review, and the "
                    "local fallback cannot independently prove this update is safe."
                ),
                confidence=0.78,
                blocking_issues=[blocking_issue],
                preservation_risks=[
                    "Writing an unreviewed patch here could repeat the same failing behavior or introduce a new drift.",
                ],
                repair_hints=[
                    f"Retry with a smaller, better-anchored change that stays on {target_hint}.",
                    "If the same review path keeps timing out, block the write instead of applying the patch unchecked.",
                ],
            )
        return ProposedUpdateReview(
            safe_to_write=True,
            summary=(
                f"Model-backed pre-write review was unavailable, but the fallback preservation check did not detect an obvious destructive rewrite for {path}."
            ),
            confidence=0.32,
            repair_hints=[
                "Keep unrelated existing behavior intact while applying the requested change.",
            ],
        )

    def _runtime_support_rewrite_is_allowed(
        self,
        session: SessionState,
        *,
        path: str,
        current_content: str,
        proposed_content: str,
    ) -> bool:
        repair_context = session.active_repair_context
        if repair_context is None or repair_context.verification_scope != "runtime":
            return False
        if not self._is_runtime_support_repair_target(path, repair_context):
            return False
        if len(str(proposed_content or "")) >= len(str(current_content or "")):
            return False
        failure_text = "\n".join(
            part
            for part in [
                str(repair_context.excerpt or "").strip(),
                str(repair_context.failure_summary or "").strip(),
                str(repair_context.summary or "").strip(),
            ]
            if part
        ).lower()
        return "assertionerror" in failure_text or "self.assertequal" in failure_text

    def _review_excerpt(self, text: str, *, limit: int) -> str:
        normalized = str(text or "")
        if len(normalized) <= limit:
            return normalized
        marker = "\n...\n[omitted middle for review]\n...\n"
        if limit <= len(marker) + 20:
            return normalized[:limit]
        head = (limit - len(marker)) // 2
        tail = limit - len(marker) - head
        return normalized[:head].rstrip() + marker + normalized[-tail:].lstrip()

    def _supporting_artifact_review_context(
        self,
        session: SessionState,
        *,
        target_path: str,
        excerpt_limit: int = 1_000,
        max_sections: int = 3,
    ) -> str:
        changed_by_path = {
            str(item.path or "").strip(): item
            for item in reversed(session.changed_files)
            if str(item.path or "").strip()
        }

        candidates: list[str] = []
        candidates.extend(changed_by_path.keys())
        for item in reversed(session.tool_calls):
            if item.tool_name != "read_file":
                continue
            candidates.append(str(item.tool_args.get("path") or "").strip())

        changed_sections: list[str] = []
        recent_sections: list[str] = []
        seen = {target_path}
        for candidate in candidates:
            path = str(candidate or "").strip()
            if not path or path in seen:
                continue
            seen.add(path)

            excerpt = self._current_or_last_read_excerpt(session, path=path)
            if not excerpt:
                continue

            changed_record = changed_by_path.get(path)
            if changed_record is not None:
                change_evidence = str(changed_record.diff or "").strip()
                if not change_evidence:
                    change_evidence = excerpt
                changed_sections.append(
                    f"{path} (already changed in this task):\n"
                    f"{self._review_excerpt(change_evidence, limit=excerpt_limit)}"
                )
            else:
                recent_sections.append(
                    f"{path} (recent context only, may still be pending):\n"
                    f"{self._review_excerpt(excerpt, limit=excerpt_limit)}"
                )

            if len(changed_sections) + len(recent_sections) >= max_sections:
                break

        sections: list[str] = []
        if changed_sections:
            sections.append("Changed supporting artifacts already written:\n" + "\n\n".join(changed_sections))
        if recent_sections:
            sections.append(
                "Recent supporting context (may still be pending updates):\n"
                + "\n\n".join(recent_sections)
            )

        return "\n\n".join(sections) or "none"

    def _generated_content_integrity_review(
        self,
        *,
        path: str,
        proposed_content: str,
    ) -> ProposedUpdateReview | None:
        normalized = str(proposed_content or "")
        suffix = Path(path).suffix.lower()
        parsed_python_module: ast.AST | None = None

        if suffix in {".py", ".pyi"}:
            try:
                parsed_python_module = ast.parse(normalized)
            except SyntaxError as exc:
                detail = "The proposed file does not parse as valid Python."
                if exc.lineno is not None:
                    detail += f" Syntax error near line {exc.lineno}."
                return ProposedUpdateReview(
                    safe_to_write=False,
                    summary="The proposed update is structurally incomplete and cannot be written safely.",
                    confidence=0.99,
                    blocking_issues=[detail],
                    preservation_risks=[],
                    repair_hints=[
                        "Return the complete Python file with valid syntax and no truncated blocks.",
                    ],
                )
            if self._is_python_test_path(path):
                placeholder_issues = self._python_test_placeholder_issues(normalized)
                if placeholder_issues:
                    return ProposedUpdateReview(
                        safe_to_write=False,
                        summary="The proposed Python test file still contains placeholder instructions instead of a finished test.",
                        confidence=0.98,
                        blocking_issues=placeholder_issues,
                        preservation_risks=[],
                        repair_hints=[
                            "Replace placeholder comments with concrete test logic and explicit checks.",
                            "Make the test verify observable behavior directly instead of describing what should be asserted later.",
                        ],
                    )
                if parsed_python_module is not None and not self._python_test_has_meaningful_assertions(parsed_python_module):
                    return ProposedUpdateReview(
                        safe_to_write=False,
                        summary="The proposed Python test file does not contain a meaningful assertion.",
                        confidence=0.97,
                        blocking_issues=[
                            f"The proposed test file {path} only contains placeholder or tautological checks instead of a meaningful assertion.",
                        ],
                        preservation_risks=[],
                        repair_hints=[
                            "Add at least one concrete assertion that checks the intended behavior or failure mode.",
                            "Do not leave the test as setup-only scaffolding without verifying the result.",
                        ],
                    )

        if suffix == ".json":
            try:
                json.loads(normalized)
            except json.JSONDecodeError as exc:
                return ProposedUpdateReview(
                    safe_to_write=False,
                    summary="The proposed update is structurally incomplete and cannot be written safely.",
                    confidence=0.99,
                    blocking_issues=[
                        f"The proposed JSON is invalid near line {exc.lineno}, column {exc.colno}.",
                    ],
                    preservation_risks=[],
                    repair_hints=[
                        "Return complete valid JSON with balanced braces, brackets, and quoted strings.",
                    ],
                )

        if suffix == ".toml":
            try:
                tomllib.loads(normalized)
            except tomllib.TOMLDecodeError:
                return ProposedUpdateReview(
                    safe_to_write=False,
                    summary="The proposed update is structurally incomplete and cannot be written safely.",
                    confidence=0.99,
                    blocking_issues=["The proposed TOML is invalid or truncated."],
                    preservation_risks=[],
                    repair_hints=[
                        "Return complete valid TOML with closed strings, arrays, and tables.",
                    ],
                )

        if suffix in {".md", ".markdown"}:
            if self._has_unclosed_markdown_fence(normalized):
                return ProposedUpdateReview(
                    safe_to_write=False,
                    summary="The proposed update is structurally incomplete and cannot be written safely.",
                    confidence=0.99,
                    blocking_issues=[
                        "The proposed Markdown contains an unclosed fenced code block.",
                    ],
                    preservation_risks=[],
                    repair_hints=[
                        "Return the complete Markdown file with balanced fenced code blocks.",
                    ],
                )

        return None

    def _is_python_test_path(self, path: str) -> bool:
        normalized = str(path or "").strip().lower()
        if not normalized:
            return False
        name = Path(normalized).name
        if name in {"__init__.py", "conftest.py"}:
            return False
        if name.startswith("test_") or name.endswith("_test.py"):
            return True
        return "/tests/" in f"/{normalized}" and name.startswith("test")

    def _python_test_placeholder_issues(self, content: str) -> list[str]:
        issues: list[str] = []
        placeholder_patterns = (
            r"\badd\s+assertions?\b",
            r"\bassertions?\s+to\s+check\b",
            r"\bplaceholder\b",
            r"\btodo\b",
            r"\bfill\s+in\b",
        )
        for line_number, raw_line in enumerate(str(content or "").splitlines(), start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            if not (
                stripped.startswith("#")
                or stripped.startswith('"""')
                or stripped.startswith("'''")
            ):
                continue
            lowered = stripped.lower()
            if any(re.search(pattern, lowered) for pattern in placeholder_patterns):
                issues.append(
                    f"Line {line_number} still contains a placeholder testing note instead of an executable check: {stripped[:140]}"
                )
            if len(issues) >= 3:
                break
        return issues

    def _python_test_has_meaningful_assertions(self, module: ast.AST) -> bool:
        for node in ast.walk(module):
            if isinstance(node, ast.Assert):
                if self._python_assert_statement_is_meaningful(node):
                    return True
            if isinstance(node, ast.Call) and self._python_test_call_is_assertion(node):
                if self._python_assertion_call_is_meaningful(node):
                    return True
            if isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if self._python_test_call_is_assertion(item.context_expr) and self._python_assertion_call_is_meaningful(
                        item.context_expr
                    ):
                        return True
        return False

    def _python_test_call_is_assertion(self, expression: ast.AST) -> bool:
        if not isinstance(expression, ast.Call):
            return False
        func = expression.func
        if isinstance(func, ast.Attribute):
            attribute_name = str(func.attr or "").strip().lower()
            return attribute_name.startswith("assert") or attribute_name == "raises"
        if isinstance(func, ast.Name):
            function_name = str(func.id or "").strip().lower()
            return function_name.startswith("assert") or function_name == "raises"
        return False

    def _python_assert_statement_is_meaningful(self, node: ast.Assert) -> bool:
        return not self._python_test_expression_is_tautology(node.test)

    def _python_assertion_call_is_meaningful(self, expression: ast.AST) -> bool:
        if not isinstance(expression, ast.Call):
            return False
        func = expression.func
        if isinstance(func, ast.Attribute):
            assertion_name = str(func.attr or "").strip().lower()
        elif isinstance(func, ast.Name):
            assertion_name = str(func.id or "").strip().lower()
        else:
            return False

        args = list(expression.args or [])
        if assertion_name in {"asserttrue", "assert_"}:
            return bool(args) and not self._python_test_expression_is_tautology(args[0])
        if assertion_name == "assertfalse":
            return bool(args) and not self._python_test_expression_is_always_false(args[0])
        if assertion_name in {"assertequal", "assertis"} and len(args) >= 2:
            return not self._python_test_expressions_are_equivalent(args[0], args[1])
        if assertion_name == "assertisnone":
            return bool(args) and not self._python_test_expression_is_literal_none(args[0])
        if assertion_name == "assertisnotnone":
            return bool(args) and not self._python_test_expression_is_literal_non_none(args[0])
        return True

    def _python_test_expression_is_tautology(self, expression: ast.AST) -> bool:
        if self._python_test_expression_literal_truthiness(expression) is True:
            return True
        if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
            return self._python_test_expression_is_always_false(expression.operand)
        if isinstance(expression, ast.Compare) and len(expression.ops) == 1 and len(expression.comparators) == 1:
            left = expression.left
            right = expression.comparators[0]
            op = expression.ops[0]
            if isinstance(op, (ast.Eq, ast.Is)) and self._python_test_expressions_are_equivalent(left, right):
                return True
        return False

    def _python_test_expression_is_always_false(self, expression: ast.AST) -> bool:
        truthiness = self._python_test_expression_literal_truthiness(expression)
        if truthiness is False:
            return True
        if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
            return self._python_test_expression_is_tautology(expression.operand)
        if isinstance(expression, ast.Compare) and len(expression.ops) == 1 and len(expression.comparators) == 1:
            left = expression.left
            right = expression.comparators[0]
            op = expression.ops[0]
            if isinstance(op, (ast.NotEq, ast.IsNot)) and self._python_test_expressions_are_equivalent(left, right):
                return True
        return False

    def _python_test_expression_literal_truthiness(self, expression: ast.AST) -> bool | None:
        try:
            literal = ast.literal_eval(expression)
        except Exception:
            return None
        return bool(literal)

    def _python_test_expression_is_literal_none(self, expression: ast.AST) -> bool:
        return isinstance(expression, ast.Constant) and expression.value is None

    def _python_test_expression_is_literal_non_none(self, expression: ast.AST) -> bool:
        return isinstance(expression, ast.Constant) and expression.value is not None

    def _python_test_expressions_are_equivalent(self, left: ast.AST, right: ast.AST) -> bool:
        if ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False):
            return True
        try:
            left_literal = ast.literal_eval(left)
            right_literal = ast.literal_eval(right)
        except Exception:
            return False
        return left_literal == right_literal

    def _explicit_constraint_integrity_review(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str,
        proposed_content: str,
    ) -> ProposedUpdateReview | None:
        scope = _artifact_scoped_focus(
            route,
            session,
            path,
            current_content=current_content,
        )
        literal_constraints = [
            str(item or "").strip()
            for item in scope.get("literal_constraints", [])
            if str(item or "").strip()
        ]
        for literal in literal_constraints:
            if not self._proposed_content_preserves_literal_constraint(
                path=path,
                literal=literal,
                proposed_content=proposed_content,
            ):
                return ProposedUpdateReview(
                    safe_to_write=False,
                    summary="The proposed update does not preserve an explicit literal constraint from the request.",
                    confidence=0.99,
                    blocking_issues=[
                        f"The exact requested literal is missing from {path}: {literal}",
                    ],
                    preservation_risks=[],
                    repair_hints=[
                        "Keep the update narrow and include the exact requested literal without changing its order or placeholder values.",
                    ],
                )

        if Path(path).suffix.lower() == ".md":
            added_headings = self._unexpected_markdown_headings(
                route,
                session,
                current_content=current_content,
                proposed_content=proposed_content,
            )
            if added_headings:
                return ProposedUpdateReview(
                    safe_to_write=False,
                    summary="The proposed markdown update adds unrequested documentation sections.",
                    confidence=0.92,
                    blocking_issues=[
                        f"The proposal adds new markdown headings without request evidence: {', '.join(added_headings[:3])}",
                    ],
                    preservation_risks=[],
                    repair_hints=[
                        "Limit the markdown edit to the explicitly requested section or example and avoid adding new headings unless the request asks for them.",
                    ],
                )

            echoed_instruction = self._markdown_instruction_echo(
                route,
                session,
                current_content=current_content,
                proposed_content=proposed_content,
            )
            if echoed_instruction is not None:
                return ProposedUpdateReview(
                    safe_to_write=False,
                    summary="The proposed markdown update copies task instructions into the document instead of documenting the resulting behavior.",
                    confidence=0.93,
                    blocking_issues=[
                        f"The proposal copies user/task instruction text into {path}: {echoed_instruction}",
                    ],
                    preservation_risks=[],
                    repair_hints=[
                        "Document only the resulting user-facing behavior, commands, and examples.",
                        "Do not paste the user request or internal task instructions into the markdown output.",
                    ],
                )

        return None

    def _explicit_create_constraint_review(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        proposed_content: str,
    ) -> ProposedUpdateReview | None:
        scope = _artifact_scoped_focus(
            route,
            session,
            path,
            current_content="",
        )
        literal_constraints = [
            str(item or "").strip()
            for item in scope.get("literal_constraints", [])
            if str(item or "").strip()
        ]
        for literal in literal_constraints:
            if not self._proposed_content_preserves_literal_constraint(
                path=path,
                literal=literal,
                proposed_content=proposed_content,
            ):
                return ProposedUpdateReview(
                    safe_to_write=False,
                    summary="The proposed new file misses an explicit literal constraint from the request.",
                    confidence=0.99,
                    blocking_issues=[
                        f"The exact requested literal is missing from {path}: {literal}",
                    ],
                    preservation_risks=[],
                    repair_hints=[
                        "Keep the new file aligned with the exact requested literals when they are part of the required output.",
                    ],
                )

        if Path(path).suffix.lower() == ".md":
            echoed_instruction = self._markdown_instruction_echo(
                route,
                session,
                current_content="",
                proposed_content=proposed_content,
            )
            if echoed_instruction is not None:
                return ProposedUpdateReview(
                    safe_to_write=False,
                    summary="The proposed markdown file copies task instructions into the document instead of documenting the resulting behavior.",
                    confidence=0.93,
                    blocking_issues=[
                        f"The proposal copies user/task instruction text into {path}: {echoed_instruction}",
                    ],
                    preservation_risks=[],
                    repair_hints=[
                        "Document only the resulting user-facing behavior, commands, and examples.",
                        "Do not paste the user request or internal task instructions into the markdown output.",
                    ],
                )

        return None

    def _markdown_instruction_echo(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        current_content: str,
        proposed_content: str,
    ) -> str | None:
        proposed_space = re.sub(r"\s+", " ", str(proposed_content or "")).strip()
        current_space = re.sub(r"\s+", " ", str(current_content or "")).strip()
        if not proposed_space:
            return None

        candidate_sources = [
            str(session.task or "").strip(),
            str(route.user_goal or "").strip(),
            str(route.requested_outcome or "").strip(),
        ]
        candidate_phrases: list[str] = []
        for source in candidate_sources:
            if not source:
                continue
            normalized = re.sub(r"\s+", " ", source).strip()
            if len(normalized) >= 80:
                candidate_phrases.append(normalized)
            for piece in re.split(r"(?<=[.!?])\s+|:\s+|;\s+", normalized):
                compact = re.sub(r"\s+", " ", str(piece or "")).strip()
                if len(compact) >= 60:
                    candidate_phrases.append(compact)

        seen: set[str] = set()
        for candidate in candidate_phrases:
            lowered = candidate.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            if candidate in current_space:
                continue
            if candidate in proposed_space:
                if len(candidate) > 140:
                    return candidate[:137].rstrip() + "..."
                return candidate
        return None

    def _validation_repair_relevance_review(
        self,
        *,
        path: str,
        current_content: str,
        proposed_content: str,
        repair_context: ValidationFailureEvidence | None,
        session: SessionState | None = None,
    ) -> ProposedUpdateReview | None:
        if repair_context is None or repair_context.verification_scope != "runtime":
            return None
        if (
            self._is_runtime_support_repair_target(path, repair_context)
            and Path(path).suffix.lower() not in {".py", ".pyi"}
        ):
            support_review = self._runtime_support_file_relevance_review(
                path=path,
                current_content=current_content,
                proposed_content=proposed_content,
                repair_context=repair_context,
            )
            if support_review is not None:
                return support_review

        identifiers = self._repair_identifiers_for_target(
            path,
            repair_context,
            current_content=current_content,
            proposed_content=proposed_content,
            session=session,
        )
        if not identifiers:
            return None

        relevant_identifiers = [
            identifier
            for identifier in identifiers
            if identifier in current_content or identifier in proposed_content
        ]
        if Path(path).suffix.lower() in {".py", ".pyi"}:
            undefined_symbol_review = self._undefined_runtime_symbol_repair_review(
                path=path,
                current_content=current_content,
                proposed_content=proposed_content,
                repair_context=repair_context,
            )
            if undefined_symbol_review is not None:
                return undefined_symbol_review
        target_evidence = self._runtime_target_evidence_lines(path, repair_context)
        if not relevant_identifiers:
            evidence_hint = f" near {' | '.join(target_evidence[:2])}" if target_evidence else ""
            return ProposedUpdateReview(
                safe_to_write=False,
                summary="The proposed repair does not touch the identifiers implicated by the failed validation.",
                confidence=0.88,
                blocking_issues=[
                    f"The proposed update for {path} does not modify any identifier from the failure evidence: {', '.join(identifiers[:4])}{evidence_hint}",
                ],
                preservation_risks=[],
                repair_hints=[
                    "Change the function, method, or symbol directly implicated by the failing validation instead of editing unrelated helper code.",
                    *self._runtime_target_repair_hints(path, repair_context, evidence_lines=target_evidence),
                ],
            )

        if any(
            self._identifier_lines_changed(identifier, current_content, proposed_content)
            for identifier in relevant_identifiers
        ):
            return None

        blocking_issue = (
            f"The proposal for {path} leaves the implicated identifier lines unchanged: {', '.join(relevant_identifiers[:4])}"
        )
        if target_evidence:
            blocking_issue += f" near {' | '.join(target_evidence[:2])}"
        return ProposedUpdateReview(
            safe_to_write=False,
            summary="The proposed repair changes the file, but not the lines tied to the failed runtime behavior.",
            confidence=0.84,
            blocking_issues=[blocking_issue],
            preservation_risks=[],
            repair_hints=[
                "Update the relevant function signature or behavior line that the failing traceback points to.",
                *self._runtime_target_repair_hints(path, repair_context, evidence_lines=target_evidence),
            ],
        )

    def _undefined_runtime_symbol_repair_review(
        self,
        *,
        path: str,
        current_content: str,
        proposed_content: str,
        repair_context: ValidationFailureEvidence,
    ) -> ProposedUpdateReview | None:
        if Path(path).suffix.lower() not in {".py", ".pyi"}:
            return None
        undefined_symbol = self._undefined_runtime_symbol(repair_context)
        if not undefined_symbol:
            return None
        current_uses = [line.strip() for line in str(current_content or "").splitlines() if undefined_symbol in line]
        proposed_uses = [line.strip() for line in str(proposed_content or "").splitlines() if undefined_symbol in line]
        if not current_uses and not proposed_uses:
            return None
        if current_uses and not proposed_uses:
            return None
        if self._python_content_binds_name(proposed_content, undefined_symbol):
            return None
        evidence_lines = self._runtime_target_evidence_lines(path, repair_context)
        evidence_hint = f" near {' | '.join(evidence_lines[:2])}" if evidence_lines else ""
        return ProposedUpdateReview(
            safe_to_write=False,
            summary="The proposed repair still leaves the undefined runtime symbol unresolved.",
            confidence=0.91,
            blocking_issues=[
                (
                    f"The runtime failure still reports '{undefined_symbol}' as undefined in {path}, "
                    f"but the proposal neither binds/imports '{undefined_symbol}' nor removes its failing usage{evidence_hint}."
                )
            ],
            preservation_risks=[],
            repair_hints=[
                f"Either import or otherwise bind '{undefined_symbol}' in {path}, or remove the failing usage from the implicated line.",
                *self._runtime_target_repair_hints(path, repair_context, evidence_lines=evidence_lines),
            ],
        )

    def _undefined_runtime_symbol(self, repair_context: ValidationFailureEvidence | None) -> str | None:
        if repair_context is None or repair_context.verification_scope != "runtime":
            return None
        texts = [
            str(repair_context.excerpt or "").strip(),
            str(repair_context.failure_summary or "").strip(),
            str(repair_context.summary or "").strip(),
        ]
        patterns = (
            re.compile(r"NameError:\s+name ['\"](?P<name>[A-Za-z_][A-Za-z0-9_]*)['\"] is not defined"),
            re.compile(r"UnboundLocalError:\s+cannot access local variable ['\"](?P<name>[A-Za-z_][A-Za-z0-9_]*)['\"]"),
        )
        for text in texts:
            if not text:
                continue
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    return str(match.group("name") or "").strip()
        return None

    def _python_content_binds_name(
        self,
        content: str,
        name: str,
    ) -> bool:
        target = str(name or "").strip()
        if not target:
            return False
        try:
            module = ast.parse(str(content or ""))
        except SyntaxError:
            return False

        def _binds_target(node: ast.AST) -> bool:
            if isinstance(node, ast.Name):
                return node.id == target and isinstance(node.ctx, ast.Store)
            if isinstance(node, (ast.Tuple, ast.List)):
                return any(_binds_target(item) for item in node.elts)
            if isinstance(node, ast.Starred):
                return _binds_target(node.value)
            return False

        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    alias_name = alias.asname or alias.name.split(".", 1)[0]
                    if alias_name == target:
                        return True
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    alias_name = alias.asname or alias.name
                    if alias_name == target:
                        return True
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = getattr(node, "targets", None) or [getattr(node, "target", None)]
                if any(target_node is not None and _binds_target(target_node) for target_node in targets):
                    return True
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                if _binds_target(node.target):
                    return True
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                if any(item.optional_vars is not None and _binds_target(item.optional_vars) for item in node.items):
                    return True
            elif isinstance(node, ast.ExceptHandler):
                if str(getattr(node, "name", "") or "").strip() == target:
                    return True
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == target:
                    return True
                all_args = [
                    *getattr(node.args, "posonlyargs", []),
                    *getattr(node.args, "args", []),
                    *getattr(node.args, "kwonlyargs", []),
                ]
                if node.args.vararg is not None:
                    all_args.append(node.args.vararg)
                if node.args.kwarg is not None:
                    all_args.append(node.args.kwarg)
                if any(arg.arg == target for arg in all_args):
                    return True
            elif isinstance(node, ast.ClassDef) and node.name == target:
                return True
        return False

    def _runtime_support_file_relevance_review(
        self,
        *,
        path: str,
        current_content: str,
        proposed_content: str,
        repair_context: ValidationFailureEvidence,
    ) -> ProposedUpdateReview | None:
        failure_text = "\n".join(
            part
            for part in [
                str(repair_context.excerpt or "").strip(),
                str(repair_context.failure_summary or "").strip(),
                str(repair_context.summary or "").strip(),
            ]
            if part
        ).lower()
        if not any(marker in failure_text for marker in ("assertionerror", "self.assertequal")):
            return None

        lowered_current = str(current_content or "").lower()
        lowered_proposed = str(proposed_content or "").lower()
        prose_markers = (
            "sample text",
            "sample text file",
            "for testing",
            "functionality of the script",
            "it contains",
            "placeholder",
            "this is a sample",
            "this is placeholder",
        )
        preserved_prose = [marker for marker in prose_markers if marker in lowered_proposed]
        current_lines = [line.strip() for line in str(current_content or "").splitlines() if line.strip()]
        proposed_lines = [line.strip() for line in str(proposed_content or "").splitlines() if line.strip()]
        unchanged_lines = [line for line in current_lines if line and line in proposed_lines]
        keeps_most_existing_lines = bool(current_lines) and len(unchanged_lines) >= max(1, len(current_lines) - 1)
        if not preserved_prose and not keeps_most_existing_lines:
            return None
        if not any(marker in lowered_current for marker in prose_markers) and not preserved_prose:
            return None

        return ProposedUpdateReview(
            safe_to_write=False,
            summary="The proposed runtime support file still preserves descriptive sample prose instead of narrowing to the raw fixture data needed by the failing assertion.",
            confidence=0.91,
            blocking_issues=[
                (
                    f"The proposed update for {path} still keeps descriptive or placeholder sample text, "
                    "so the runtime assertion is unlikely to converge to the expected raw fixture content."
                )
            ],
            preservation_risks=[],
            repair_hints=[
                "Remove descriptive sample prose and keep only the minimal raw fixture data required for the expected assertion.",
                "Do not preserve placeholder narrative lines just because they were already present in the file.",
            ],
        )

    def _helper_entrypoint_scope_review(
        self,
        session: SessionState,
        *,
        path: str,
        current_content: str,
        proposed_content: str,
    ) -> ProposedUpdateReview | None:
        if Path(path).suffix.lower() not in {".py", ".pyi"}:
            return None
        if Path(path).name == "__main__.py":
            return None

        sibling_entrypoint = Path(path).with_name("__main__.py").as_posix()
        sibling_exists = self._session_or_current_file_content(session, sibling_entrypoint) is not None or (
            sibling_entrypoint in (session.workspace_snapshot.important_files if session.workspace_snapshot is not None else [])
        )
        if not sibling_exists:
            return None

        lowered_current = str(current_content or "").lower()
        lowered_proposed = str(proposed_content or "").lower()
        current_has_cli_launcher = self._file_contains_cli_launcher_logic(lowered_current)
        proposed_has_cli_launcher = self._file_contains_cli_launcher_logic(lowered_proposed)
        if current_has_cli_launcher or not proposed_has_cli_launcher:
            return None

        if not self._request_or_context_targets_package_entrypoint(
            session,
            helper_path=path,
            sibling_entrypoint=sibling_entrypoint,
        ):
            return None

        return ProposedUpdateReview(
            safe_to_write=False,
            summary="The proposed update moves package entrypoint logic into a helper module.",
            confidence=0.96,
            blocking_issues=[
                (
                    f"The proposal adds argparse and/or __main__ launcher logic to {path} even though "
                    f"the package already has a sibling entrypoint at {sibling_entrypoint}."
                )
            ],
            preservation_risks=[],
            repair_hints=[
                f"Keep {path} focused on reusable helper behavior and place CLI parsing or launcher handling in {sibling_entrypoint}.",
                f"If the new flag changes helper behavior, update the function signature or return value in {path} without adding a second entrypoint block.",
            ],
        )

    def _entrypoint_helper_contract_review(
        self,
        *,
        path: str,
        current_content: str,
        proposed_content: str,
    ) -> ProposedUpdateReview | None:
        if Path(path).suffix.lower() not in {".py", ".pyi"}:
            return None
        if Path(path).name != "__main__.py":
            return None

        try:
            current_module = ast.parse(str(current_content or ""))
            proposed_module = ast.parse(str(proposed_content or ""))
        except SyntaxError:
            return None

        current_markers = set(self._entrypoint_helper_contract_markers(current_module))
        proposed_markers = set(self._entrypoint_helper_contract_markers(proposed_module))
        new_markers = sorted(marker for marker in proposed_markers if marker not in current_markers)
        if not new_markers:
            return None

        blocking_issues: list[str] = []
        for marker_type, attrs in new_markers:
            attr_text = ", ".join(sorted(attrs)) or "CLI values"
            if marker_type == "compound_input":
                blocking_issues.append(
                    f"The proposal composes helper inputs from CLI values ({attr_text}) inside {path} instead of passing a stable helper contract."
                )
            elif marker_type == "duplicated_direct_arg":
                blocking_issues.append(
                    f"The proposal passes CLI values ({attr_text}) into the helper while {path} still uses the same CLI values outside the helper call."
                )
            elif marker_type == "wrapped_output":
                blocking_issues.append(
                    f"The proposal wraps the helper result with extra CLI-derived formatting ({attr_text}) inside {path}, which moves greeting semantics into the CLI wrapper."
                )

        if not blocking_issues:
            return None

        return ProposedUpdateReview(
            safe_to_write=False,
            summary="The proposed entrypoint update blurs the contract between CLI parsing and helper behavior.",
            confidence=0.94,
            blocking_issues=blocking_issues,
            preservation_risks=[],
            repair_hints=[
                "Keep the entrypoint focused on parsing argv, invoking the helper, and printing or repeating the final result.",
                "Pass parsed values into the helper as direct arguments instead of composing new greeting text around the helper call in __main__.py.",
            ],
        )

    def _entrypoint_helper_contract_markers(
        self,
        module: ast.AST,
    ) -> list[tuple[str, tuple[str, ...]]]:
        parse_arg_bindings = self._entrypoint_parse_arg_bindings(module)
        if not parse_arg_bindings:
            return []

        helper_function_names, helper_aliases = self._entrypoint_relative_import_bindings(module)
        if not helper_function_names and not helper_aliases:
            return []

        parent_map = self._ast_parent_map(module)
        external_attrs = self._entrypoint_external_cli_attrs(
            module,
            helper_function_names=helper_function_names,
            helper_aliases=helper_aliases,
            parse_arg_bindings=parse_arg_bindings,
            parent_map=parent_map,
        )
        markers: list[tuple[str, tuple[str, ...]]] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        statement_nodes = (
            ast.Assign,
            ast.AnnAssign,
            ast.AugAssign,
            ast.Expr,
            ast.Return,
            ast.For,
            ast.AsyncFor,
            ast.If,
            ast.While,
            ast.With,
            ast.AsyncWith,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.Module,
        )

        for node in ast.walk(module):
            if not isinstance(node, ast.Call):
                continue
            if not self._call_matches_helper_binding(
                node,
                helper_function_names=helper_function_names,
                helper_aliases=helper_aliases,
            ):
                continue

            helper_attrs: set[str] = set()
            direct_helper_attrs: set[str] = set()
            for arg_node in [*node.args, *(keyword.value for keyword in node.keywords if keyword.value is not None)]:
                attrs = self._parse_arg_attrs_in_node(arg_node, parse_arg_bindings)
                helper_attrs.update(attrs)
                is_direct = self._is_direct_parse_arg_reference(arg_node, parse_arg_bindings)
                if attrs and is_direct:
                    direct_helper_attrs.update(attrs)
                if attrs and not is_direct:
                    marker = ("compound_input", tuple(sorted(attrs)))
                    if marker not in seen:
                        seen.add(marker)
                        markers.append(marker)

            duplicated_direct_attrs = sorted(attr for attr in direct_helper_attrs if attr in external_attrs)
            if duplicated_direct_attrs:
                marker = ("duplicated_direct_arg", tuple(duplicated_direct_attrs))
                if marker not in seen:
                    seen.add(marker)
                    markers.append(marker)

            current = node
            while current in parent_map:
                parent = parent_map[current]
                if isinstance(parent, (ast.JoinedStr, ast.BinOp)):
                    outside_attrs = self._parse_arg_attrs_in_node(parent, parse_arg_bindings) - helper_attrs
                    if outside_attrs:
                        marker = ("wrapped_output", tuple(sorted(outside_attrs)))
                        if marker not in seen:
                            seen.add(marker)
                            markers.append(marker)
                        break
                if isinstance(parent, statement_nodes):
                    break
                current = parent
        return markers

    def _file_contains_cli_launcher_logic(self, lowered_content: str) -> bool:
        text = str(lowered_content or "")
        if not text:
            return False
        if "__name__ == \"__main__\"" in text or "__name__ == '__main__'" in text:
            return True
        return any(
            token in text
            for token in ("argparse.argumentparser", "parse_args(", "parse_known_args(")
        )

    def _request_or_context_targets_package_entrypoint(
        self,
        session: SessionState,
        *,
        helper_path: str,
        sibling_entrypoint: str,
    ) -> bool:
        request_lower = str(session.task or "").lower()
        normalized_request = f" {re.sub(r'[^0-9a-zäöüß]+', ' ', request_lower).strip()} "
        if "python -m" in request_lower or "python3 -m" in request_lower:
            return True
        if "argparse" in request_lower:
            return True
        if " cli " in normalized_request:
            return True
        if re.search(r"(^|[^0-9a-z])--[a-z0-9][\\w-]*", request_lower):
            return True

        sibling_excerpt = self._current_or_last_read_excerpt(session, path=sibling_entrypoint).lower()
        helper_stem = Path(helper_path).stem
        if "__main__.main(" in sibling_excerpt or "__main__.sys.argv" in sibling_excerpt:
            return True
        if helper_stem and f"from .{helper_stem} import" in sibling_excerpt:
            return True
        return sibling_entrypoint in (session.workspace_snapshot.entrypoints if session.workspace_snapshot is not None else [])

    def _cli_wrapper_responsibility_review(
        self,
        session: SessionState,
        *,
        path: str,
        proposed_content: str,
    ) -> ProposedUpdateReview | None:
        if Path(path).suffix.lower() not in {".py", ".pyi"}:
            return None
        if Path(path).name == "__main__.py":
            return None

        sibling_entrypoint = Path(path).with_name("__main__.py").as_posix()
        sibling_content = (
            self._session_or_current_file_content(session, sibling_entrypoint)
            or self._current_or_last_read_excerpt(session, path=sibling_entrypoint)
        )
        if not sibling_content:
            return None
        if not self._file_contains_cli_launcher_logic(str(sibling_content).lower()):
            return None
        if not self._request_or_context_targets_package_entrypoint(
            session,
            helper_path=path,
            sibling_entrypoint=sibling_entrypoint,
        ):
            return None

        helper_optional_params = self._helper_optional_cli_params(
            helper_path=path,
            sibling_content=sibling_content,
            proposed_content=proposed_content,
        )
        if not helper_optional_params:
            return None

        wrapper_attrs = self._entrypoint_wrapper_cli_attrs(
            helper_path=path,
            entrypoint_content=sibling_content,
        )
        overlap = sorted(
            {
                attr
                for attr in helper_optional_params.intersection(wrapper_attrs)
                if attr not in {"name", "names", "argv", "args"}
            }
        )
        if not overlap:
            return None

        overlap_text = ", ".join(overlap[:4])
        return ProposedUpdateReview(
            safe_to_write=False,
            summary="The proposed helper update duplicates CLI wrapper responsibilities that are still handled in the package entrypoint.",
            confidence=0.93,
            blocking_issues=[
                (
                    f"The proposal adds helper parameters for {overlap_text} in {path} while "
                    f"{sibling_entrypoint} still applies the same CLI options around the helper result."
                )
            ],
            preservation_risks=[],
            repair_hints=[
                (
                    "Keep one clear contract boundary: either let the helper own those options and "
                    "remove the outer wrapper transformations, or keep the helper focused on the base value."
                ),
                (
                    f"Use the current {sibling_entrypoint} logic as the cross-file contract and avoid "
                    "making the helper and entrypoint apply the same flag twice."
                ),
            ],
        )

    def _helper_optional_cli_params(
        self,
        *,
        helper_path: str,
        sibling_content: str,
        proposed_content: str,
    ) -> set[str]:
        try:
            sibling_module = ast.parse(str(sibling_content or ""))
            proposed_module = ast.parse(str(proposed_content or ""))
        except SyntaxError:
            return set()

        helper_function_names, _ = self._helper_import_bindings(
            sibling_module,
            helper_path=helper_path,
        )
        optional_params: set[str] = set()
        for node in proposed_module.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if helper_function_names and node.name not in helper_function_names:
                continue
            positional = [
                arg.arg
                for arg in [*node.args.posonlyargs, *node.args.args]
                if str(arg.arg or "").strip() not in {"self", "cls"}
            ]
            optional_params.update(item for item in positional[1:] if str(item or "").strip())
            optional_params.update(
                arg.arg
                for arg in node.args.kwonlyargs
                if str(arg.arg or "").strip() not in {"self", "cls"}
            )

        if optional_params or helper_function_names:
            return optional_params

        for node in proposed_module.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue
            positional = [
                arg.arg
                for arg in [*node.args.posonlyargs, *node.args.args]
                if str(arg.arg or "").strip() not in {"self", "cls"}
            ]
            optional_params.update(item for item in positional[1:] if str(item or "").strip())
            optional_params.update(
                arg.arg
                for arg in node.args.kwonlyargs
                if str(arg.arg or "").strip() not in {"self", "cls"}
            )
        return optional_params

    def _helper_import_bindings(
        self,
        sibling_module: ast.AST,
        *,
        helper_path: str,
    ) -> tuple[set[str], set[str]]:
        helper_stem = Path(helper_path).stem
        imported_function_names: set[str] = set()
        helper_aliases: set[str] = set()
        for node in ast.walk(sibling_module):
            if isinstance(node, ast.ImportFrom):
                module_name = str(node.module or "").split(".")[-1]
                if module_name != helper_stem:
                    continue
                for alias in node.names:
                    imported_function_names.add(alias.name)
                    helper_aliases.add(alias.asname or alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = str(alias.name or "").split(".")[-1]
                    if module_name != helper_stem:
                        continue
                    helper_aliases.add(alias.asname or module_name)
        return imported_function_names, helper_aliases

    def _entrypoint_relative_import_bindings(
        self,
        module: ast.AST,
    ) -> tuple[set[str], set[str]]:
        helper_function_names: set[str] = set()
        helper_aliases: set[str] = set()
        for node in ast.walk(module):
            if isinstance(node, ast.ImportFrom):
                if int(node.level or 0) < 1:
                    continue
                for alias in node.names:
                    helper_function_names.add(alias.name)
                    helper_aliases.add(alias.asname or alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = str(alias.name or "").strip()
                    if "." not in module_name:
                        continue
                    helper_aliases.add(alias.asname or module_name.split(".")[-1])
        return helper_function_names, helper_aliases

    def _entrypoint_parse_arg_bindings(
        self,
        module: ast.AST,
    ) -> set[str]:
        bindings: set[str] = set()
        for node in ast.walk(module):
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, ast.Call):
                continue
            func = node.value.func
            if not isinstance(func, ast.Attribute) or func.attr not in {"parse_args", "parse_known_args"}:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings.add(target.id)
                elif isinstance(target, ast.Tuple) and target.elts and isinstance(target.elts[0], ast.Name):
                    bindings.add(target.elts[0].id)
        return bindings

    def _ast_parent_map(self, module: ast.AST) -> dict[ast.AST, ast.AST]:
        parent_map: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(module):
            for child in ast.iter_child_nodes(node):
                parent_map[child] = node
        return parent_map

    def _call_matches_helper_binding(
        self,
        node: ast.Call,
        *,
        helper_function_names: set[str],
        helper_aliases: set[str],
    ) -> bool:
        func = node.func
        if isinstance(func, ast.Name):
            return func.id in helper_function_names or func.id in helper_aliases
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            return func.value.id in helper_aliases
        return False

    def _parse_arg_attrs_in_node(
        self,
        node: ast.AST,
        parse_arg_bindings: set[str],
    ) -> set[str]:
        attrs: set[str] = set()
        for child in ast.walk(node):
            if not isinstance(child, ast.Attribute):
                continue
            if not isinstance(child.value, ast.Name):
                continue
            if child.value.id not in parse_arg_bindings:
                continue
            attrs.add(str(child.attr or "").strip())
        return {item for item in attrs if item}

    def _is_direct_parse_arg_reference(
        self,
        node: ast.AST,
        parse_arg_bindings: set[str],
    ) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in parse_arg_bindings
            and bool(str(node.attr or "").strip())
        )

    def _entrypoint_external_cli_attrs(
        self,
        module: ast.AST,
        *,
        helper_function_names: set[str],
        helper_aliases: set[str],
        parse_arg_bindings: set[str],
        parent_map: dict[ast.AST, ast.AST],
    ) -> set[str]:
        def in_parse_args_call(node: ast.AST) -> bool:
            current = node
            while current in parent_map:
                current = parent_map[current]
                if not isinstance(current, ast.Call):
                    continue
                func = current.func
                return isinstance(func, ast.Attribute) and func.attr in {"parse_args", "parse_known_args"}
            return False

        def in_helper_call_arg(node: ast.AST) -> bool:
            current = node
            while current in parent_map:
                current = parent_map[current]
                if not isinstance(current, ast.Call):
                    continue
                return self._call_matches_helper_binding(
                    current,
                    helper_function_names=helper_function_names,
                    helper_aliases=helper_aliases,
                )
            return False

        attrs: set[str] = set()
        for node in ast.walk(module):
            if not isinstance(node, ast.Attribute):
                continue
            if not isinstance(node.value, ast.Name):
                continue
            if node.value.id not in parse_arg_bindings:
                continue
            if in_parse_args_call(node) or in_helper_call_arg(node):
                continue
            attrs.add(str(node.attr or "").strip())
        return {item for item in attrs if item}

    def _entrypoint_wrapper_cli_attrs(
        self,
        *,
        helper_path: str,
        entrypoint_content: str,
    ) -> set[str]:
        try:
            module = ast.parse(str(entrypoint_content or ""))
        except SyntaxError:
            return set()

        helper_function_names, helper_aliases = self._helper_import_bindings(
            module,
            helper_path=helper_path,
        )
        parent_map = self._ast_parent_map(module)
        parse_arg_bindings = self._entrypoint_parse_arg_bindings(module)
        return self._entrypoint_external_cli_attrs(
            module,
            helper_function_names=helper_function_names,
            helper_aliases=helper_aliases,
            parse_arg_bindings=parse_arg_bindings,
            parent_map=parent_map,
        )

    def _argv_launcher_runtime_review(
        self,
        session: SessionState,
        *,
        path: str,
        current_content: str,
        proposed_content: str,
    ) -> ProposedUpdateReview | None:
        repair_context = session.active_repair_context
        if repair_context is None or repair_context.verification_scope != "runtime":
            return None
        if Path(path).suffix.lower() not in {".py", ".pyi"}:
            return None

        lowered_current = str(current_content or "").lower()
        lowered_proposed = str(proposed_content or "").lower()
        if not any(
            token in lowered_current or token in lowered_proposed
            for token in ("argparse.argumentparser", "parse_args(", "parse_known_args(")
        ):
            return None

        supporting_context = self._supporting_artifact_review_context(
            session,
            target_path=path,
            excerpt_limit=500,
            max_sections=2,
        )
        lowered_support = str(supporting_context or "").lower()
        if "__main__.sys.argv" not in lowered_support or "'-m'" not in supporting_context:
            return None

        direct_main_launcher_invocation = "__main__.main()" in lowered_support
        references_runtime_argv = (
            "sys.argv" in lowered_proposed
            or "from sys import argv" in lowered_proposed
        )
        target_evidence = self._runtime_target_evidence_lines(path, repair_context)
        target_code_lines = [
            line.strip()
            for line in target_evidence
            if line
            and not line.startswith("File ")
            and not line.startswith("Traceback")
        ]
        argparse_failure = any(
            marker in str(repair_context.excerpt or "").lower()
            for marker in ("unrecognized arguments", "systemexit")
        )
        implicated_argparse_lines = [
            line
            for line in target_code_lines
            if "parse_args(" in line or "parse_known_args(" in line
        ]
        current_stripped_lines = {line.strip() for line in str(current_content or "").splitlines() if line.strip()}
        proposed_stripped_lines = {line.strip() for line in str(proposed_content or "").splitlines() if line.strip()}
        unchanged_argparse_lines = [
            line
            for line in implicated_argparse_lines
            if line in current_stripped_lines and line in proposed_stripped_lines
        ]

        blocking_issue: str | None = None
        repair_hint: str | None = None
        if argparse_failure and unchanged_argparse_lines:
            blocking_issue = (
                f"The proposal for {path} still leaves the implicated argparse call unchanged: {unchanged_argparse_lines[0]}"
            )
            repair_hint = (
                "Change the failing argparse invocation itself so it consumes only the real CLI arguments instead of the patched python -m launcher prefix."
            )
        elif direct_main_launcher_invocation and self._required_function_parameter_count(
            proposed_content,
            function_name="main",
        ) > 0:
            blocking_issue = (
                f"The proposal for {path} makes main() require positional arguments even though the failing tests call __main__.main() without arguments."
            )
            repair_hint = (
                "Keep main() callable without positional arguments and derive the patched runtime argv inside the function when the tests invoke __main__.main() directly."
            )
        elif direct_main_launcher_invocation and not references_runtime_argv:
            blocking_issue = (
                f"The proposal for {path} still ignores the patched __main__.sys.argv runtime input for the direct __main__.main() invocation."
            )
            repair_hint = (
                "When the failing tests call __main__.main() while patching __main__.sys.argv, derive the CLI arguments from the patched runtime argv when no explicit arguments were passed."
            )
        elif "sys.argv" in lowered_proposed and "import sys" not in lowered_proposed and "from sys import" not in lowered_proposed:
            blocking_issue = (
                f"The proposal for {path} references sys.argv without importing sys."
            )
            repair_hint = (
                "Import every newly referenced module or remove the unresolved module reference before writing the repair."
            )
        elif re.search(r"\bargv\s*=\s*sys\.argv\s*\[\s*1\s*:\s*\]", lowered_proposed):
            blocking_issue = (
                f"The proposal for {path} still passes the python -m launcher flag into argparse by deriving argv from sys.argv[1:]."
            )
            repair_hint = (
                "When the patched runtime argv looks like ['python', '-m', '<module>', ...], strip the full launcher prefix before argparse so only the trailing CLI arguments remain. For that launcher shape, sys.argv[3:] is the equivalent slice."
            )
        elif "parse_known_args(" in lowered_proposed and "sys.argv[" not in lowered_proposed:
            blocking_issue = (
                f"The proposal for {path} still lets argparse consume the launcher prefix instead of only the trailing CLI arguments."
            )
            repair_hint = (
                "Strip the full launcher prefix ['python', '-m', '<module>'] before argparse so the parser only receives the trailing CLI arguments."
            )
        elif re.search(r"sys\.argv\s*\[\s*2\s*:\s*\]", lowered_proposed) and not re.search(
            r"sys\.argv\s*\[\s*3\s*:\s*\]",
            lowered_proposed,
        ):
            blocking_issue = (
                f"The proposal for {path} still passes the module name into argparse by slicing sys.argv[2:] after a python -m style launcher prefix."
            )
            repair_hint = (
                "When the failing tests patch __main__.sys.argv as ['python', '-m', '<module>', ...], argparse should only receive the trailing CLI arguments after the module name. For that launcher shape, sys.argv[3:] is the equivalent slice."
            )

        if blocking_issue is None:
            return None

        return ProposedUpdateReview(
            safe_to_write=False,
            summary="The proposed repair still routes launcher tokens into argparse.",
            confidence=0.95,
            blocking_issues=[blocking_issue],
            preservation_risks=[],
            repair_hints=[repair_hint] if repair_hint else [],
        )

    def _runtime_target_evidence_lines(
        self,
        path: str,
        repair_context: ValidationFailureEvidence,
        *,
        limit: int = 3,
    ) -> list[str]:
        normalized_path = str(path or "").strip().lower()
        if not normalized_path:
            return []
        target_markers = self._repair_target_markers(normalized_path)
        evidence: list[str] = []
        texts = [
            str(repair_context.excerpt or "").strip(),
            str(repair_context.failure_summary or "").strip(),
            str(repair_context.summary or "").strip(),
        ]
        for text in texts:
            if not text:
                continue
            lines = text.splitlines()
            for index, raw in enumerate(lines):
                stripped = raw.strip()
                lowered = stripped.lower()
                if (
                    not stripped
                    or not (stripped.startswith("File ") or re.match(r"[\w./\\-]+\.py:\d+(?::\d+)?", stripped))
                    or not self._runtime_evidence_line_matches_target(
                        path,
                        stripped,
                        lowered_line=lowered,
                        target_markers=target_markers,
                    )
                ):
                    continue
                if stripped not in evidence:
                    evidence.append(stripped)
                if index + 1 < len(lines):
                    next_line = lines[index + 1].strip()
                    if (
                        next_line
                        and not next_line.startswith("Traceback")
                        and not next_line.startswith("File ")
                        and next_line not in evidence
                    ):
                        evidence.append(next_line)
                if len(evidence) >= limit:
                    return evidence[:limit]
        return evidence[:limit]

    def _runtime_evidence_line_matches_target(
        self,
        path: str,
        line: str,
        *,
        lowered_line: str,
        target_markers: set[str],
    ) -> bool:
        normalized_target = str(path or "").strip().replace("\\", "/").lower()
        if not normalized_target or not lowered_line:
            return False

        frame_match = self.validation_planner.TRACEBACK_FRAME_PATTERN.search(line)
        if frame_match is not None:
            frame_path = str(frame_match.group("path") or "").strip().replace("\\", "/").lower()
            if frame_path:
                frame_name = Path(frame_path).name
                frame_stem = Path(frame_path).stem
                target_name = Path(normalized_target).name
                target_stem = Path(normalized_target).stem
                frame_is_test = self.validation_planner._is_test_path(frame_path)
                target_is_test = self.validation_planner._is_test_path(normalized_target)
                if frame_is_test and not target_is_test:
                    return frame_path == normalized_target or frame_name == target_name or frame_stem == target_stem

        return any(
            marker and self._repair_target_line_matches_marker(lowered_line, marker)
            for marker in target_markers
        )

    def _runtime_target_repair_hints(
        self,
        path: str,
        repair_context: ValidationFailureEvidence,
        *,
        evidence_lines: list[str],
    ) -> list[str]:
        hints: list[str] = []
        if evidence_lines:
            code_lines = [
                line
                for line in evidence_lines
                if not line.startswith("File ") and not line.startswith("Traceback")
            ]
            if code_lines:
                hints.append(
                    f"Change the target behavior near {path}: {code_lines[0]}"
                )
        lowered_excerpt = str(repair_context.excerpt or "").lower()
        if "unrecognized arguments" in lowered_excerpt:
            hints.append(
                "Adjust the target's argument handling so the runtime invocation exercised by the failing tests is accepted instead of being treated as extra arguments."
            )
        return hints[:2]

    def _repair_identifiers_for_target(
        self,
        path: str,
        repair_context: ValidationFailureEvidence,
        *,
        current_content: str,
        proposed_content: str,
        session: SessionState | None = None,
    ) -> list[str]:
        candidates: list[str] = []
        candidates.extend(self._target_specific_repair_identifiers(path, repair_context))
        candidates.extend(self._repair_brief_implicated_identifiers(repair_context))
        candidates.extend(
            self._request_repair_identifiers(
                session,
                current_content=current_content,
                proposed_content=proposed_content,
            )
        )
        candidates.extend(self._repair_identifiers_from_failure_evidence(repair_context))

        identifiers = self._unique_paths(
            [
                identifier
                for identifier in candidates
                if identifier and not self._is_generic_runtime_repair_identifier(identifier)
            ]
        )
        if not identifiers:
            return []

        relevant = [
            identifier
            for identifier in identifiers
            if identifier in current_content or identifier in proposed_content
        ]
        if relevant:
            return relevant[:8]
        return identifiers[:8]

    def _repair_brief_implicated_identifiers(
        self,
        repair_context: ValidationFailureEvidence,
    ) -> list[str]:
        brief = getattr(repair_context, "repair_brief", None)
        if brief is None:
            return []
        return self._unique_paths(
            [
                str(identifier or "").strip()
                for identifier in getattr(brief, "implicated_symbols", []) or []
                if str(identifier or "").strip()
            ]
        )[:6]

    def _request_repair_identifiers(
        self,
        session: SessionState | None,
        *,
        current_content: str,
        proposed_content: str,
    ) -> list[str]:
        if session is None:
            return []

        texts: list[str] = [str(session.task or "").strip()]
        task_state = getattr(session, "task_state", None)
        if task_state is not None:
            texts.append(str(getattr(task_state, "latest_user_turn", "") or "").strip())
            texts.extend(str(item or "").strip() for item in getattr(task_state, "constraints", []) or [])

        content_markers = f"{current_content}\n{proposed_content}"
        identifiers: list[str] = []
        patterns = (
            r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b",
            r"(?<![A-Za-z0-9_])--?[A-Za-z0-9_][A-Za-z0-9_.-]*[A-Za-z0-9_](?![A-Za-z0-9_])",
            r"\b[A-Za-z_][A-Za-z0-9_]*(?=\()",
        )
        for text in texts:
            if not text:
                continue
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    candidate = str(match.group(0) or "").strip()
                    if not candidate or candidate not in content_markers:
                        continue
                    identifiers.append(candidate)
        return self._unique_paths(identifiers)[:8]

    def _target_specific_repair_identifiers(
        self,
        path: str,
        repair_context: ValidationFailureEvidence,
    ) -> list[str]:
        normalized_path = str(path or "").strip().lower()
        if not normalized_path:
            return []
        target_markers = self._repair_target_markers(normalized_path)
        identifiers: list[str] = []
        texts = [
            str(repair_context.excerpt or "").strip(),
            str(repair_context.failure_summary or "").strip(),
            str(repair_context.summary or "").strip(),
            *(str(item or "").strip() for item in repair_context.action_hints),
        ]
        for text in texts:
            if not text:
                continue
            lines = text.splitlines()
            for index, raw in enumerate(lines):
                stripped = raw.strip()
                lowered = stripped.lower()
                if not stripped:
                    continue
                if self._runtime_evidence_line_matches_target(
                    path,
                    stripped,
                    lowered_line=lowered,
                    target_markers=target_markers,
                ):
                    frame_match = re.search(r"\bin (?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*$", stripped)
                    if frame_match:
                        identifiers.append(frame_match.group("name"))
                    frame_like = (
                        self.validation_planner.TRACEBACK_FRAME_PATTERN.search(stripped) is not None
                        or re.match(r"[\w./\\-]+\.py:\d+(?::\d+)?", stripped) is not None
                    )
                    if frame_like and index + 1 < len(lines):
                        code_line = lines[index + 1].strip()
                        if (
                            code_line
                            and not code_line.startswith("File ")
                            and not code_line.startswith("Traceback")
                            and not self._is_runtime_failure_noise_line(code_line)
                        ):
                            identifiers.extend(
                                match.group(0)
                                for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", code_line)
                            )
        blocked = {"self", "cls", "unittest", "tests", "python", "args"}
        return self._unique_paths(
            [
                identifier
                for identifier in identifiers
                if (
                    identifier
                    and identifier not in blocked
                    and not identifier.startswith("test_")
                    and not self._is_generic_runtime_repair_identifier(identifier)
                )
            ]
        )[:8]

    def _repair_target_line_matches_marker(self, lowered_line: str, marker: str) -> bool:
        line = str(lowered_line or "").strip().lower()
        token = str(marker or "").strip().lower()
        if not line or not token:
            return False
        if "/" in token:
            return token in line
        if "." in token:
            return re.search(rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])", line) is not None
        return re.search(rf"\b{re.escape(token)}\b", line) is not None

    def _is_generic_runtime_repair_identifier(self, identifier: str) -> bool:
        lowered = str(identifier or "").strip().lower()
        if not lowered:
            return True
        if lowered in {
            "and",
            "as",
            "calledprocesserror",
            "class",
            "command",
            "def",
            "error",
            "errors",
            "else",
            "fail",
            "failed",
            "false",
            "file",
            "for",
            "from",
            "if",
            "import",
            "in",
            "line",
            "none",
            "open",
            "output",
            "process",
            "print",
            "r",
            "retcode",
            "return",
            "run",
            "self",
            "stderr",
            "stdout",
            "traceback",
            "true",
            "with",
        }:
            return True
        return False

    def _is_runtime_failure_noise_line(self, line: str) -> bool:
        stripped = str(line or "").strip()
        if not stripped:
            return True
        lowered = stripped.lower()
        if re.fullmatch(r"[=\-]{5,}", stripped):
            return True
        if lowered.startswith(("traceback", "during handling")):
            return True
        if re.match(r"^(?:fail|failed|error|errors|ok|ran \d+ tests?)\b", lowered):
            return True
        if "returned non-zero exit status" in lowered:
            return True
        return False

    def _repair_target_markers(self, path: str) -> set[str]:
        normalized = str(path or "").strip().lower()
        if not normalized:
            return set()
        markers: set[str] = {normalized}
        path_obj = Path(normalized)
        if path_obj.name:
            markers.add(path_obj.name)
        if path_obj.stem:
            markers.add(path_obj.stem)
        for part in path_obj.parts:
            token = str(part or "").strip().lower()
            if token:
                markers.add(token)
        return {marker for marker in markers if len(marker) >= 3}

    def _repair_identifiers_from_failure_evidence(
        self,
        repair_context: ValidationFailureEvidence,
    ) -> list[str]:
        texts = [
            str(repair_context.failure_summary or "").strip(),
            str(repair_context.summary or "").strip(),
            str(repair_context.excerpt or "").strip(),
            *(str(item or "").strip() for item in repair_context.action_hints),
        ]
        identifiers: list[str] = []
        for text in texts:
            if not text:
                continue
            for match in re.finditer(
                r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\(\)\s+takes\s+\d+\s+positional arguments?",
                text,
            ):
                identifiers.append(match.group("name"))
            for match in re.finditer(
                r"\b(?P<module>[A-Za-z_][A-Za-z0-9_]*)\.(?P<name>[A-Za-z_][A-Za-z0-9_]*)\(",
                text,
            ):
                if self._is_test_helper_failure_identifier(
                    match.group("module"),
                    match.group("name"),
                ):
                    continue
                identifiers.extend([match.group("module"), match.group("name")])
        blocked = {"self", "cls", "unittest", "tests"}
        return self._unique_paths(
            [
                identifier
                for identifier in identifiers
                if (
                    identifier
                    and identifier not in blocked
                    and not self._is_generic_runtime_repair_identifier(identifier)
                )
            ]
        )[:6]

    def _is_test_helper_failure_identifier(
        self,
        module_name: str,
        identifier_name: str,
    ) -> bool:
        module = str(module_name or "").strip().lower()
        identifier = str(identifier_name or "").strip().lower()
        if not module or not identifier:
            return False
        if module in {"self", "cls"} or module.startswith("mock_"):
            return True
        if module.startswith("test_") or identifier.startswith("test_"):
            return True
        if identifier.startswith("assert"):
            return True
        return False

    def _required_function_parameter_count(
        self,
        content: str,
        *,
        function_name: str,
    ) -> int:
        target = str(function_name or "").strip()
        if not target:
            return 0
        try:
            module = ast.parse(str(content or ""))
        except SyntaxError:
            return 0

        for node in module.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != target:
                continue
            positional = len(list(node.args.posonlyargs)) + len(list(node.args.args))
            defaults = len(list(node.args.defaults))
            return max(positional - defaults, 0)
        return 0

    def _identifier_lines_changed(
        self,
        identifier: str,
        current_content: str,
        proposed_content: str,
    ) -> bool:
        current_lines = [
            line.strip()
            for line in str(current_content or "").splitlines()
            if identifier in line
        ]
        proposed_lines = [
            line.strip()
            for line in str(proposed_content or "").splitlines()
            if identifier in line
        ]
        if current_lines != proposed_lines:
            return True
        return self._python_named_block_changed(
            identifier,
            current_content=current_content,
            proposed_content=proposed_content,
        ) or self._python_implicated_statement_changed(
            identifier,
            current_content=current_content,
            proposed_content=proposed_content,
        )

    def _python_implicated_statement_changed(
        self,
        identifier: str,
        *,
        current_content: str,
        proposed_content: str,
    ) -> bool:
        target = str(identifier or "").strip()
        if not target:
            return False
        try:
            current_module = ast.parse(str(current_content or ""))
            proposed_module = ast.parse(str(proposed_content or ""))
        except SyntaxError:
            return False

        current_lines = [
            index
            for index, raw in enumerate(str(current_content or "").splitlines(), start=1)
            if target in raw
        ]
        proposed_lines = [
            index
            for index, raw in enumerate(str(proposed_content or "").splitlines(), start=1)
            if target in raw
        ]
        if not current_lines or not proposed_lines:
            return False

        current_statements = self._python_smallest_statements_for_lines(
            current_module,
            line_numbers=current_lines,
        )
        proposed_statements = self._python_smallest_statements_for_lines(
            proposed_module,
            line_numbers=proposed_lines,
        )
        if not current_statements or not proposed_statements:
            return False

        current_dump = [ast.dump(node, include_attributes=False) for node in current_statements]
        proposed_dump = [ast.dump(node, include_attributes=False) for node in proposed_statements]
        return current_dump != proposed_dump

    def _python_smallest_statements_for_lines(
        self,
        module: ast.AST,
        *,
        line_numbers: list[int],
    ) -> list[ast.stmt]:
        selected: list[ast.stmt] = []
        seen: set[tuple[int, int, str]] = set()
        for line_number in line_numbers:
            candidate = self._python_smallest_statement_covering_line(module, line_number=line_number)
            if candidate is None:
                continue
            key = (
                int(getattr(candidate, "lineno", 0) or 0),
                int(getattr(candidate, "end_lineno", 0) or 0),
                candidate.__class__.__name__,
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(candidate)
        return selected

    def _python_smallest_statement_covering_line(
        self,
        module: ast.AST,
        *,
        line_number: int,
    ) -> ast.stmt | None:
        best: ast.stmt | None = None
        best_span: tuple[int, int] | None = None
        for node in ast.walk(module):
            if not isinstance(node, ast.stmt) or isinstance(node, ast.Module):
                continue
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if start is None or end is None:
                continue
            if not (start <= line_number <= end):
                continue
            span = (int(end) - int(start), int(start))
            if best is None or best_span is None or span < best_span:
                best = node
                best_span = span
        return best

    def _python_named_block_changed(
        self,
        identifier: str,
        *,
        current_content: str,
        proposed_content: str,
    ) -> bool:
        name = str(identifier or "").strip()
        if not name:
            return False
        try:
            current_module = ast.parse(str(current_content or ""))
            proposed_module = ast.parse(str(proposed_content or ""))
        except SyntaxError:
            return False

        def _matching_nodes(module: ast.AST) -> list[ast.AST]:
            nodes: list[ast.AST] = []
            for node in ast.walk(module):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                    nodes.append(node)
            return nodes

        current_nodes = _matching_nodes(current_module)
        proposed_nodes = _matching_nodes(proposed_module)
        if not current_nodes or not proposed_nodes:
            return False
        current_dump = [ast.dump(node, include_attributes=False) for node in current_nodes]
        proposed_dump = [ast.dump(node, include_attributes=False) for node in proposed_nodes]
        return current_dump != proposed_dump

    def _unexpected_markdown_headings(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        current_content: str,
        proposed_content: str,
    ) -> list[str]:
        current_headings = self._markdown_headings(current_content)
        proposed_headings = self._markdown_headings(proposed_content)
        if not proposed_headings:
            return []
        request_text = " ".join(
            part.strip().lower()
            for part in [
                route.user_goal,
                route.requested_outcome,
                str(session.task_state.latest_user_turn if session.task_state is not None else ""),
                *(
                    session.task_state.constraints
                    if session.task_state is not None
                    else []
                ),
            ]
            if str(part or "").strip()
        )
        unexpected: list[str] = []
        for heading in proposed_headings:
            if heading in current_headings:
                continue
            normalized = heading.lower()
            if self._markdown_heading_allowed_by_request(heading, request_text):
                continue
            if normalized and normalized not in request_text:
                unexpected.append(heading)
        return unexpected

    def _markdown_headings(self, content: str) -> set[str]:
        headings: set[str] = set()
        for raw_line in str(content or "").splitlines():
            line = raw_line.strip()
            if not line.startswith("#"):
                continue
            heading = line.lstrip("#").strip()
            if heading:
                headings.add(heading)
        return headings

    def _markdown_heading_allowed_by_request(self, heading: str, request_text: str) -> bool:
        normalized = str(heading or "").strip().lower()
        if not normalized:
            return False
        if normalized in request_text:
            return True
        request_tokens = {
            token
            for token in re.split(r"[^a-z0-9]+", request_text)
            if len(token) >= 3
        }
        if not request_tokens:
            return False
        heading_base = normalized.split(":", 1)[0].strip()
        heading_tokens = [
            token
            for token in re.split(r"[^a-z0-9]+", heading_base)
            if token
        ]
        if not heading_tokens:
            return False
        wants_example = any(
            token in request_tokens
            for token in {
                "example",
                "examples",
                "usage",
                "document",
                "documented",
                "documenting",
                "readme",
                "beispiel",
                "dokumentiere",
                "dokumentation",
            }
        )
        if not wants_example:
            return False
        return all(
            token in {"example", "examples", "usage", "output", "outputs"}
            or token in request_tokens
            for token in heading_tokens
        )

    def _proposed_content_preserves_literal_constraint(
        self,
        *,
        path: str,
        literal: str,
        proposed_content: str,
    ) -> bool:
        normalized = str(literal or "").strip()
        if not normalized:
            return True
        if normalized in proposed_content:
            return True
        suffix = Path(path).suffix.lower()
        if suffix in {".py", ".pyi"} and not self._path_is_test_like(path):
            return self._python_signature_literal_is_preserved(normalized, proposed_content)
        return False

    def _python_signature_literal_is_preserved(self, literal: str, proposed_content: str) -> bool:
        normalized = str(literal or "").strip()
        if "(" not in normalized or ")" not in normalized:
            return False
        callee = str(normalized.split("(", 1)[0] or "").strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", callee):
            return False
        parameter_names = self._literal_parameter_names(normalized)
        if not parameter_names:
            return False
        try:
            tree = ast.parse(proposed_content)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name != callee:
                continue
            defined_names = [arg.arg for arg in node.args.posonlyargs]
            defined_names.extend(arg.arg for arg in node.args.args)
            if node.args.vararg is not None:
                defined_names.append(node.args.vararg.arg)
            defined_names.extend(arg.arg for arg in node.args.kwonlyargs)
            if node.args.kwarg is not None:
                defined_names.append(node.args.kwarg.arg)
            if defined_names[: len(parameter_names)] == parameter_names:
                return True
        return False

    def _literal_parameter_names(self, literal: str) -> list[str]:
        normalized = str(literal or "").strip()
        if "(" not in normalized or ")" not in normalized:
            return []
        inside = str(normalized.split("(", 1)[1].rsplit(")", 1)[0] or "").strip()
        if not inside:
            return []
        parameters: list[str] = []
        for raw_part in inside.split(","):
            part = str(raw_part or "").strip()
            if not part:
                continue
            part = part.split(":", 1)[0].strip()
            part = part.lstrip("*").strip()
            part = part.split("=", 1)[0].strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part):
                parameters.append(part)
        return parameters

    def _path_is_test_like(self, path: str) -> bool:
        normalized = str(path or "").strip().replace("\\", "/").lower()
        name = Path(normalized).name
        return normalized.startswith("tests/") or "/tests/" in f"/{normalized}" or name.startswith("test_")

    def _has_unclosed_markdown_fence(self, text: str) -> bool:
        active_fence: str | None = None
        for raw_line in str(text or "").splitlines():
            line = raw_line.lstrip()
            match = re.match(r"(```+|~~~+)", line)
            if match is None:
                continue
            marker = match.group(1)
            if active_fence is None:
                active_fence = marker
                continue
            if marker[0] == active_fence[0] and len(marker) >= len(active_fence):
                active_fence = None
        return active_fence is not None

    def _current_or_last_read_excerpt(
        self,
        session: SessionState,
        *,
        path: str,
    ) -> str:
        absolute = Path(session.workspace_root, path)
        if absolute.exists() and absolute.is_file():
            try:
                return absolute.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                pass

        for item in reversed(session.tool_calls):
            if item.tool_name != "read_file":
                continue
            candidate = str(item.tool_args.get("path") or "").strip()
            if candidate != path:
                continue
            excerpt = (item.output_excerpt or "").strip()
            if excerpt:
                return excerpt
        return ""

    def _review_diff_excerpt(
        self,
        path: str,
        *,
        current_content: str,
        proposed_content: str,
        limit: int,
    ) -> str:
        diff = "".join(
            difflib.unified_diff(
                current_content.splitlines(keepends=True),
                proposed_content.splitlines(keepends=True),
                fromfile=f"{path}:current",
                tofile=f"{path}:proposed",
                n=3,
            )
        )
        if not diff.strip():
            return "(no diff)"
        return self._review_excerpt(diff, limit=limit)

    def _run_semantic_change_review(self, route: RouterOutput, session: SessionState) -> None:
        if not session.changed_files or self.validation_planner.has_semantic_review(session):
            return

        deterministic_web_review = self._semantic_web_contract_review(session)
        if deterministic_web_review is not None:
            self._append_runtime_execution(
                session,
                build_execution_run_record(
                    operation_name="semantic_change_review",
                    task_class="semantic_change_review",
                    final_state="completed",
                    capability_tier="tier_d",
                    recovery_strategy="deterministic_web_contract_review",
                    degraded=True,
                    honest_blocked=False,
                    artifact_bytes_generated=0,
                    validation_possible=True,
                    summary="A deterministic cross-file web contract review found unresolved DOM or selector mismatches before model-backed semantic review.",
                    attempts=[],
                ),
            )
            self._record_semantic_change_review(session, deterministic_web_review)
            return

        artifacts = self._semantic_review_artifacts(session)
        primary_model = self._primary_generation_model_name()
        reserve_model = self._lightweight_generation_model_name()
        if (
            reserve_model is None
            and session.validation_status == "passed"
            and self.validation_planner.web_structural_proxy_sufficient(session)
        ):
            review = self._fallback_semantic_change_review(session)
            if review.requirements_satisfied:
                self._log(
                    "semantic_change_review_skipped",
                    reason="single_model_small_web_structural_proxy",
                    model=primary_model,
                )
                self._append_runtime_execution(
                    session,
                    build_execution_run_record(
                        operation_name="semantic_change_review",
                        task_class="semantic_change_review",
                        final_state="degraded_success",
                        capability_tier="tier_d",
                        recovery_strategy="deterministic_fallback",
                        degraded=True,
                        honest_blocked=False,
                        artifact_bytes_generated=0,
                        validation_possible=True,
                        summary="A small standalone web change used structural verification plus local requirements review instead of a second same-model semantic review hop.",
                        attempts=[],
                    ),
                )
                self._record_semantic_change_review(session, review)
                return
        attempts: list[ExecutionAttemptRecord] = []
        prefer_lightweight = self._should_prefer_lightweight_semantic_review(
            route,
            session,
            artifacts=artifacts,
        )
        full_prompt = semantic_change_review_prompt(route, session, artifacts=artifacts)
        compact_prompt: str | None = None
        if prefer_lightweight:
            compact_prompt = semantic_change_review_prompt(
                route,
                session,
                artifacts=self._semantic_review_artifacts(
                    session,
                    content_limit=1_000,
                    diff_limit=500,
                ),
                mode="compact",
            )

        review_attempts: list[tuple[str | None, str, str, int, int, str]] = []
        if prefer_lightweight and reserve_model is not None:
            review_attempts.append(
                (
                    reserve_model,
                    "tier_b",
                    "reserve_model_generation",
                    max(self._llm_timeout(18), 18),
                    min(self._llm_num_ctx(2048), 2048),
                    "compact",
                )
            )
        review_attempts.append(
            (
                primary_model,
                "tier_a",
                "primary_model_generation",
                max(self._llm_timeout(24), 24),
                min(self._llm_num_ctx(4096), 4096),
                "full",
            )
        )
        if reserve_model is not None and not prefer_lightweight:
            review_attempts.append(
                (
                    reserve_model,
                    "tier_b",
                    "reserve_model_generation",
                    max(self._llm_timeout(18), 18),
                    min(self._llm_num_ctx(2048), 2048),
                    "full",
                )
            )

        self._log(
            "semantic_change_review_started",
            model=review_attempts[0][0] or primary_model,
            preferred_strategy=review_attempts[0][2] if review_attempts else "primary_model_generation",
            prompt_variant=review_attempts[0][5] if review_attempts else "full",
            changed_files=[item.path for item in session.changed_files[-6:]],
        )

        for model_name, capability_tier, strategy, timeout, num_ctx, prompt_variant in review_attempts:
            prompt = compact_prompt if prompt_variant == "compact" and compact_prompt is not None else full_prompt
            if prompt_variant == "compact":
                timeout, total_timeout = self._compact_reserve_review_budget()
            else:
                total_timeout = max(timeout + 20, timeout * 2)
            outcome = invoke_model(
                lambda progress, review_model=model_name: self.llm.generate_json(
                    prompt,
                    system=semantic_change_review_system_prompt(),
                    model=review_model,
                    retries=0,
                    timeout=timeout,
                    total_timeout=total_timeout,
                    strict_timeouts=prompt_variant == "compact",
                    num_ctx=num_ctx,
                    progress_callback=progress,
                ),
                operation_name="semantic_change_review",
                task_class="semantic_change_review",
                attempt_number=len(attempts) + 1,
                capability_tier=capability_tier,
                recovery_strategy=strategy,
                prompt_variant=prompt_variant,
                model_identifier=model_name or self._primary_generation_model_name(),
                backend_identifier=self._backend_identifier(),
                inactivity_timeout_seconds=timeout,
                total_timeout_seconds=total_timeout,
                context_pressure_estimate=estimate_context_pressure(prompt_chars=len(prompt)),
                event_callback=self._progress_logger("semantic_change_review_progress"),
            )
            attempts.append(outcome.attempt)
            if outcome.exception is not None:
                self._log(
                    "semantic_change_review_error",
                    error=str(outcome.exception),
                    strategy=strategy,
                    model=model_name,
                )
                if (
                    prefer_lightweight
                    and strategy == "reserve_model_generation"
                    and attempts[-1].failure is not None
                    and attempts[-1].failure.no_start_failure
                ):
                    break
                continue
            try:
                review = SemanticChangeReview.model_validate(outcome.value)
            except ValidationError as exc:
                self._log(
                    "semantic_change_review_invalid",
                    errors=exc.errors(),
                    payload=outcome.value,
                    strategy=strategy,
                    model=model_name,
                )
                continue

            self._append_runtime_execution(
                session,
                build_execution_run_record(
                    operation_name="semantic_change_review",
                    task_class="semantic_change_review",
                    final_state="completed",
                    capability_tier=capability_tier,
                    recovery_strategy=strategy,
                    degraded=capability_tier != "tier_a",
                    honest_blocked=False,
                    artifact_bytes_generated=0,
                    validation_possible=True,
                    summary="A cross-artifact semantic review checked whether the changed implementation satisfies the requested outcome.",
                    attempts=attempts,
                ),
            )
            self._record_semantic_change_review(session, review)
            return

        review = self._fallback_semantic_change_review(session)
        self._append_runtime_execution(
            session,
            build_execution_run_record(
                operation_name="semantic_change_review",
                task_class="semantic_change_review",
                final_state="degraded_success",
                capability_tier="tier_d",
                recovery_strategy="deterministic_fallback",
                degraded=True,
                honest_blocked=False,
                artifact_bytes_generated=0,
                validation_possible=True,
                summary="Semantic change review fell back to conservative local heuristics after model-backed review did not complete cleanly.",
                attempts=attempts,
            ),
        )
        self._record_semantic_change_review(session, review)

    def _record_semantic_change_review(self, session: SessionState, review: SemanticChangeReview) -> None:
        review = self._enforce_repair_review_gates(session, review)
        command = self.validation_planner.semantic_review_command([item.path for item in session.changed_files])
        excerpt_parts: list[str] = []
        if review.missing_requirements:
            excerpt_parts.append(f"Missing requirements: {', '.join(review.missing_requirements[:4])}.")
        if review.suspicious_issues:
            excerpt_parts.append(f"Suspicious issues: {', '.join(review.suspicious_issues[:4])}.")
        excerpt = " ".join(excerpt_parts).strip() or None

        session.validation_runs.append(
            ValidationRunRecord(
                command=command,
                cwd=".",
                kind="check",
                verification_scope="semantic",
                status="passed" if review.requirements_satisfied else "failed",
                edit_generation=session.edit_generation,
                iteration=session.iterations,
                summary=review.summary,
                excerpt=excerpt,
            )
        )
        session.validation_runs = session.validation_runs[-20:]
        session.validation_status = self.validation_planner.rollup_status(session)

        if review.requirements_satisfied:
            session.active_repair_context = None
            session.last_error = None
            session.notes.append(f"Semantic review passed: {review.summary}")
            session.notes = session.notes[-20:]
            return

        summary = review.summary or "Semantic review found unmet requested behavior."
        session.diagnostics.append(
            DiagnosticRecord(
                source="semantic_change_review",
                category="requirements_gap",
                severity="error",
                summary=summary,
                tool_name="semantic_change_review",
                command=command,
                file_hints=review.file_hints[:6] or [item.path for item in session.changed_files[:4]],
                action_hints=review.repair_hints[:4],
                excerpt=excerpt,
                iteration=session.iterations,
            )
        )
        session.diagnostics = session.diagnostics[-20:]
        failed_run = session.validation_runs[-1]
        session.active_repair_context = self.validation_planner.build_failure_evidence(session, failed_run)
        session.last_error = excerpt or summary

    def _semantic_review_artifacts(
        self,
        session: SessionState,
        *,
        content_limit: int = 1_800,
        diff_limit: int = 900,
    ) -> list[dict[str, object]]:
        artifacts: list[dict[str, object]] = []
        for change in session.changed_files[-6:]:
            entry: dict[str, object] = {
                "path": change.path,
                "operation": change.operation,
            }
            if change.diff:
                entry["diff_excerpt"] = self._clip_text(change.diff, diff_limit)
            current_content = self._current_file_content(session, change.path)
            if current_content is not None:
                entry["content_excerpt"] = self._clip_text(current_content, content_limit)
            artifacts.append(entry)
        return artifacts

    def _latest_verified_repair_attempt(self, session: SessionState) -> RepairAttemptRecord | None:
        return next(
            (
                attempt
                for attempt in reversed(session.repair_history)
                if attempt.result == "mutation_planned"
            ),
            None,
        )

    def _repair_review_gate_failure(self, session: SessionState) -> SemanticChangeReview | None:
        attempt = self._latest_verified_repair_attempt(session)
        if attempt is None:
            return None

        missing_requirements: list[str] = []
        suspicious_issues: list[str] = []
        repair_hints: list[str] = []
        file_hints = [str(attempt.artifact_path or "").strip()] if str(attempt.artifact_path or "").strip() else [
            item.path for item in session.changed_files[:4]
        ]

        if not str(attempt.root_cause_summary or "").strip():
            missing_requirements.append("Concrete root cause explanation for the latest repair")
            repair_hints.append("Record the concrete failure cause before approving the repair.")
        if attempt.productive_change is not True:
            missing_requirements.append("Productive code change for the latest repair")
            repair_hints.append("Reject whitespace-only, comment-only, metadata-only, or equivalent repairs.")
        if attempt.independent_verification is not True:
            missing_requirements.append("Independent verification after the latest repair")
            repair_hints.append("Rerun an independent validation step after the repair before approving completion.")
        if (
            attempt.independent_verification is False
            and attempt.behavior_changed is False
            and str(attempt.post_validation_failure_signature or "").strip()
        ):
            suspicious_issues.append(
                "The latest repair reran into the same failure signature without changing the observed behavior."
            )
            repair_hints.append("Stop retrying the same path until the bootstrap state is reset or new evidence appears.")

        if not missing_requirements and not suspicious_issues:
            return None

        return SemanticChangeReview(
            requirements_satisfied=False,
            summary=(
                "Repair review cannot pass yet because the latest repair still lacks a concrete cause, a productive "
                "change, or an independent verification."
            ),
            confidence=0.9,
            missing_requirements=list(dict.fromkeys(missing_requirements)),
            suspicious_issues=list(dict.fromkeys(suspicious_issues)),
            repair_hints=list(dict.fromkeys(repair_hints)),
            file_hints=file_hints[:4],
        )

    def _enforce_repair_review_gates(
        self,
        session: SessionState,
        review: SemanticChangeReview,
    ) -> SemanticChangeReview:
        gate_failure = self._repair_review_gate_failure(session)
        if gate_failure is None:
            return review
        if review.requirements_satisfied:
            return gate_failure
        return review.model_copy(
            update={
                "missing_requirements": list(
                    dict.fromkeys([*review.missing_requirements, *gate_failure.missing_requirements])
                ),
                "suspicious_issues": list(
                    dict.fromkeys([*review.suspicious_issues, *gate_failure.suspicious_issues])
                ),
                "repair_hints": list(dict.fromkeys([*review.repair_hints, *gate_failure.repair_hints])),
                "file_hints": list(dict.fromkeys([*review.file_hints, *gate_failure.file_hints])),
            }
        )

    def _fallback_semantic_change_review(self, session: SessionState) -> SemanticChangeReview:
        deferred_targets = self._active_deferred_update_targets(session)
        pending_snapshot_targets = self._semantic_review_pending_snapshot_targets(
            session,
            deferred_targets=deferred_targets,
        )
        if session.validation_status in {"failed", "bootstrap_failed", "bootstrap_reset_required"}:
            return SemanticChangeReview(
                requirements_satisfied=False,
                summary="Existing validation already reports unresolved issues in the changed implementation.",
                confidence=0.85,
                repair_hints=["Inspect the last failing validation evidence before claiming the task is complete."],
                file_hints=[item.path for item in session.changed_files[:4]],
            )
        if pending_snapshot_targets:
            pending_preview = ", ".join(pending_snapshot_targets[:4])
            return SemanticChangeReview(
                requirements_satisfied=False,
                summary=f"Explicitly referenced task targets are still unchanged: {pending_preview}.",
                confidence=0.88,
                missing_requirements=[f"Update the still-pending explicit target artifacts: {pending_preview}"],
                repair_hints=["Complete the explicitly requested file updates before declaring the task done."],
                file_hints=pending_snapshot_targets[:6],
            )
        deterministic_web_review = self._semantic_web_contract_review(session)
        if deterministic_web_review is not None:
            return deterministic_web_review
        if self.validation_planner.web_structural_proxy_sufficient(session):
            return SemanticChangeReview(
                requirements_satisfied=True,
                summary="Structural web validation plus the local requirements review covered the changed standalone web artifact without a separate browser runtime step.",
                confidence=0.66,
                file_hints=[item.path for item in session.changed_files[:4]],
            )
        if self._functional_validation_missing(session):
            return SemanticChangeReview(
                requirements_satisfied=False,
                summary="The changed implementation still lacks a confirmed functional verification path.",
                confidence=0.8,
                missing_requirements=["Functional verification of the changed behavior"],
                repair_hints=["Rerun the most relevant runtime or reproduction path for the changed behavior."],
                file_hints=[item.path for item in session.changed_files[:4]],
            )
        if session.validation_status == "passed":
            review = SemanticChangeReview(
                requirements_satisfied=True,
                summary="Available project-aware validation passed and the fallback review found no additional concrete task-to-code mismatch.",
                confidence=0.55,
                file_hints=[item.path for item in session.changed_files[:4]],
            )
            return self._enforce_repair_review_gates(session, review)
        return SemanticChangeReview(
            requirements_satisfied=False,
            summary="The semantic review fallback could not confirm that the changed artifacts satisfy the requested outcome.",
            confidence=0.35,
            missing_requirements=["Task-to-code coverage review of the changed artifacts"],
            repair_hints=["Inspect the changed files against the explicit requested behavior before declaring completion."],
            file_hints=[item.path for item in session.changed_files[:4]],
        )

    def _requirements_review_missing(self, session: SessionState) -> bool:
        if not session.changed_files:
            return False
        return not self.validation_planner.has_semantic_review(session)

    def _clip_text(self, text: str | None, limit: int) -> str:
        normalized = str(text or "").strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 1].rstrip() + "…"

    def _functional_validation_missing(self, session: SessionState) -> bool:
        if session.changed_files and self.validation_planner.runtime_verification_required(session):
            return not self.validation_planner.has_runtime_success(session)
        if session.changed_files and self.validation_planner.web_functional_verification_required(session):
            if (
                self.validation_planner.has_semantic_review_success(session)
                and self.validation_planner.web_structural_proxy_sufficient(session)
            ):
                return False
            return not self.validation_planner.has_runtime_success(session)
        return False

    def _is_timeout_error(self, exc: Exception | None) -> bool:
        if isinstance(exc, OllamaGenerationError):
            return exc.timed_out
        return exc is not None and "timed out" in str(exc).lower()

    def _session_language(self, session: SessionState) -> str:
        task_state = session.task_state
        if task_state is not None and task_state.latest_user_turn:
            return self._language_for_text(task_state.latest_user_turn)
        return self._language_for_text(session.task)

    def _language_for_text(self, text: str | None) -> str:
        normalized = str(text or "").lower()
        german_markers = (
            " lies ",
            " fasse ",
            " ich ",
            " bitte ",
            " mach",
            " bau",
            " aenderung",
            " pruef",
            "prüf",
            "fehler",
            "datei",
            "sicher",
            "backend",
            "frontend",
            "kannst",
            "moechte",
            "möchte",
            "jetzt",
            "dazu",
            "nur ",
            " zusammen",
        )
        padded = f" {normalized} "
        if any(marker in padded for marker in german_markers) or any(char in normalized for char in "äöüß"):
            return "de"
        return "en"

    def _localized_text(self, language: str, *, de: str, en: str) -> str:
        return de if language == "de" else en

    def _require_task_state(self, session: SessionState | None) -> TaskState:
        if session is None or session.task_state is None:
            raise RuntimeError(
                "Planner requires session.task_state. Update task state before routing, planning, or deciding."
            )
        return session.task_state

    def _log(self, event: str, **payload) -> None:
        if self.logger is None:
            return
        self.logger.log_event(event, **payload)

    @staticmethod
    def _prompt_sha256(prompt: str) -> str:
        return hashlib.sha256(str(prompt or "").encode("utf-8")).hexdigest()

    def _write_prompt_trace(
        self,
        session: SessionState,
        *,
        operation_name: str,
        path: str,
        prompt: str,
        model: str | None,
        prompt_variant: str,
        num_ctx: int,
        timeout_seconds: int,
        total_timeout_seconds: int,
    ) -> str | None:
        config = getattr(self.llm, "config", None)
        helper_dir = getattr(config, "helper_dir_path", None)
        if helper_dir is None:
            return None
        try:
            target_dir = Path(helper_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            safe_target = re.sub(r"[^A-Za-z0-9._-]+", "_", path).strip("._") or "target"
            target = target_dir / f"{session.id}-{operation_name}-{safe_target}.prompt.txt"
            metadata = {
                "session_id": session.id,
                "operation_name": operation_name,
                "target_path": path,
                "model": model,
                "prompt_variant": prompt_variant,
                "prompt_chars": len(prompt),
                "prompt_lines": prompt.count("\n") + 1,
                "prompt_sha256": self._prompt_sha256(prompt),
                "num_ctx": num_ctx,
                "inactivity_timeout_seconds": timeout_seconds,
                "total_timeout_seconds": total_timeout_seconds,
            }
            target.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2)
                + "\n\n=== PROMPT ===\n"
                + prompt,
                encoding="utf-8",
            )
            target_text = str(target)
            if target_text not in session.helper_artifacts:
                session.helper_artifacts.append(target_text)
            return target_text
        except OSError:
            return None

    def _progress_logger(self, event: str, **base_payload):
        if self.logger is None:
            return None
        last_emitted = {"heartbeat": 0.0, "chunk": 0.0}

        def callback(payload: dict[str, object]) -> None:
            kind = str(payload.get("type") or "").strip()
            now = time.monotonic()
            if kind == "heartbeat":
                if now - last_emitted["heartbeat"] < 8.0:
                    return
                last_emitted["heartbeat"] = now
            elif kind == "chunk":
                if now - last_emitted["chunk"] < 2.0:
                    return
                last_emitted["chunk"] = now
            self._log(event, **base_payload, **payload)

        return callback
