from __future__ import annotations

import ast
from dataclasses import dataclass, field
import difflib
import json
from pathlib import Path
import re
import time
import tomllib

from pydantic import ValidationError

from agent.decision import ExecutionDecisionPolicy
from agent.models import (
    DiagnosticRecord,
    ProposedUpdateReview,
    RepairAttemptRecord,
    SemanticChangeReview,
    SessionState,
    ValidationFailureEvidence,
    ValidationRunRecord,
    WorkspaceSnapshot,
)
from agent.prompts import (
    REPAIR_BLOCKED_SENTINEL,
    _artifact_scoped_focus,
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


@dataclass(slots=True)
class MutationAssessment:
    effective: bool
    reason: str


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
            model_name=getattr(llm_config, "router_model_name", None),
            timeout=max(int(getattr(llm_config, "router_timeout", self._llm_timeout(18))), 18),
            num_ctx=max(int(getattr(llm_config, "router_num_ctx", self._llm_num_ctx(2048))), 1024),
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

        if session.changed_files:
            if session.validation_status == "failed":
                repair_decision = self._repair_after_failed_validation(route, session)
                if repair_decision is not None:
                    return repair_decision
            command = self._pick_validation_command(session)
            if command is not None:
                return self._validation_decision(
                    "Changed files must go through the remaining validation plan.",
                    command,
                )
            if self._requirements_review_missing(session):
                self._run_semantic_change_review(route, session)
                if session.validation_status == "failed":
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
                    return self._validation_decision(step.reason, command)

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
        elif session.validation_status == "failed" and session.validation_runs:
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
        generation = self._generate_file_content(route, session, path=path)
        if not generation.content:
            failure = generation.failure or self._build_content_generation_failure(
                route,
                session,
                path=path,
                current_content=None,
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
            return self._draft_missing_repair_artifact_decision(
                route,
                session,
                target,
                repair_context,
            )
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
                if repair_context is not None:
                    self._record_repair_attempt(
                        session,
                        repair_context,
                        target=target,
                        strategy=strategy,
                        result="generation_failed",
                        reason=failure.blocker_message,
                    )
                if repair_context is not None and strategy != strategies[-1]:
                    continue
                break

            content = generation.content
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
                    strategy=strategy,
                    result="blocked",
                    reason=final_failure_reason,
                )
                if strategy != strategies[-1]:
                    continue
                break

            mutation = self._assess_effective_mutation(current_content, content)
            if mutation.effective:
                if repair_context is not None:
                    self._record_repair_attempt(
                        session,
                        repair_context,
                        target=target,
                        strategy=strategy,
                        result="mutation_planned",
                        reason=mutation.reason,
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
                self._record_repair_attempt(
                    session,
                    repair_context,
                    target=target,
                    strategy=strategy,
                    result="no_effective_change",
                    reason=mutation.reason,
                )
                if strategy != strategies[-1]:
                    continue
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

    def _repair_generation_strategies(
        self,
        session: SessionState,
        repair_context: ValidationFailureEvidence | None,
        target: str,
    ) -> list[str]:
        if repair_context is None:
            return ["default"]
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
            return [ESCALATED_REPAIR_STRATEGY]
        return [TARGETED_REPAIR_STRATEGY, ESCALATED_REPAIR_STRATEGY]

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
    ) -> None:
        session.repair_history.append(
            RepairAttemptRecord(
                artifact_path=target,
                validation_command=repair_context.command,
                verification_scope=repair_context.verification_scope,
                strategy=strategy,
                result=result,
                reason=reason,
                evidence_signature=repair_context.evidence_signature,
                iteration=session.iterations,
            )
        )
        session.repair_history = session.repair_history[-20:]

    def _assess_effective_mutation(
        self,
        current_content: str,
        new_content: str,
    ) -> MutationAssessment:
        if new_content.strip() == current_content.strip():
            return MutationAssessment(False, "identical content")
        if self._normalized_mutation_content(new_content) == self._normalized_mutation_content(current_content):
            return MutationAssessment(False, "whitespace-only change")
        return MutationAssessment(True, "substantive mutation prepared")

    def _normalized_mutation_content(self, content: str) -> str:
        return re.sub(r"\s+", " ", str(content or "").strip())

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
            return AgentDecision(
                thought_summary="Reproduce the issue with the strongest available runtime or validation command before editing.",
                action_type=AgentActionType.CALL_TOOL,
                tool_name="run_tests",
                tool_args={"command": command, "cwd": ".", "timeout": 120},
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

        repair_route = self._repair_route_after_failed_validation(
            route,
            session,
            failed_run,
            repair_context,
        )
        if repair_route is not None:
            repair_decision = self.execute_action_from_plan(repair_route, session)
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
                command,
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
        if absolute_target.exists() and absolute_target.is_file():
            action_plan = [
                RouteActionStep(
                    step=1,
                    action=RouteActionName.READ_RELEVANT_FILES,
                    reason="Inspect the changed artifact again after the failed validation before repairing it.",
                ),
                RouteActionStep(
                    step=2,
                    action=RouteActionName.UPDATE_ARTIFACT,
                    reason="Repair the changed artifact using the failed validation evidence.",
                ),
                RouteActionStep(
                    step=3,
                    action=RouteActionName.RUN_VALIDATION,
                    reason="Rerun the validation after the repair step.",
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
                    reason="Create or restore the missing artifact implicated by the failed validation.",
                ),
                RouteActionStep(
                    step=2,
                    action=RouteActionName.RUN_VALIDATION,
                    reason="Rerun the validation after the repair step.",
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
            ]
        )
        attribute_lines = self._unique_paths(
            [
                *route.entities.attributes,
                f"failed_validation_scope:{failure_scope}",
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
                    f"Repair {target_name} so the failed {failure_scope} validation passes. "
                    f"Failure summary: {failure_summary} Preserve the original requested outcome."
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
        candidates.extend(self._paths_from_internal_validation_command(str(failed_run.command or "")))
        for item in reversed(session.diagnostics):
            candidates.extend(item.file_hints)
        candidates.extend(route.entities.target_paths)
        candidates.extend(item.path for item in reversed(session.changed_files))
        candidates.extend(session.candidate_files)
        for candidate in self._unique_paths(candidates):
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

    def _prioritized_repair_target_candidates(
        self,
        session: SessionState,
        repair_context: ValidationFailureEvidence,
    ) -> list[str]:
        candidates = self._unique_paths([*repair_context.artifact_paths, *repair_context.file_hints])
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
        return self._unique_paths(
            [
                *explicitly_referenced_missing,
                *explicitly_referenced_existing,
                *creatable_missing,
                *existing,
                *candidates,
            ]
        )

    def _repair_candidate_is_explicitly_referenced(
        self,
        candidate: str,
        repair_context: ValidationFailureEvidence,
    ) -> bool:
        text = str(candidate or "").strip()
        if not text:
            return False
        basename = Path(text).name
        evidence_texts = [
            str(repair_context.failure_summary or "").strip(),
            str(repair_context.summary or "").strip(),
            str(repair_context.excerpt or "").strip(),
            *(str(item or "").strip() for item in repair_context.repair_requirements),
            *(str(item or "").strip() for item in repair_context.action_hints),
        ]
        return any(
            item and (text in item or (basename and basename in item))
            for item in evidence_texts
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

    def _validation_decision(
        self,
        thought_summary: str,
        command: str,
    ) -> AgentDecision:
        return AgentDecision(
            thought_summary=thought_summary,
            action_type=AgentActionType.CALL_TOOL,
            tool_name="run_tests",
            tool_args={"command": command, "cwd": ".", "timeout": 120},
            expected_outcome="Run the next validation step for the current changes.",
            final_response=None,
        )

    def _pick_validation_command(self, session: SessionState) -> str | None:
        passed = {
            self.validation_planner.command_identity(run.command)
            for run in session.validation_runs
            if run.edit_generation == session.edit_generation and run.status == "passed"
        }
        for item in session.validation_plan:
            identity = self.validation_planner.command_identity(item.command)
            if identity not in passed and self.validation_planner.can_repeat_command(session, item.command):
                return item.command
        for command in session.verification_commands:
            identity = self.validation_planner.command_identity(command)
            if command and identity not in passed and self.validation_planner.can_repeat_command(session, command):
                return command
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
                return item.command
        for command in snapshot.likely_commands:
            identity = self.validation_planner.command_identity(command)
            if command and identity not in passed and self.validation_planner.can_repeat_command(session, command):
                return command
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
        for item in self.validation_planner.build_diagnostic_plan(session):
            command = str(item.command or "").strip()
            if command:
                return command
        if session.follow_up_context is not None:
            for item in reversed(session.follow_up_context.recent_commands):
                command = str(item or "").strip()
                if command and self.validation_planner.command_scope(command) == "runtime":
                    return command
        snapshot = session.workspace_snapshot
        if snapshot is None:
            return None
        fallback_plan = self.validation_planner.build_plan(
            session.task,
            snapshot,
            changed_files=[],
            session=session,
        )
        for item in fallback_plan:
            command = str(item.command or "").strip()
            if command and item.verification_scope == "runtime":
                return command
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
        if route.entities.target_name and self._looks_like_path(route.entities.target_name):
            candidates.append(route.entities.target_name)
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
        return self._unique_paths(candidates)

    def _read_candidates(
        self,
        route: RouterOutput,
        session: SessionState,
        candidate_paths: list[str],
    ) -> list[str]:
        explicit_targets = self._explicit_target_paths(route)
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

    def _explicit_target_paths(self, route: RouterOutput) -> list[str]:
        candidates = [path for path in route.entities.target_paths if path and Path(path).suffix]
        if route.entities.target_name and self._looks_like_path(route.entities.target_name):
            target_name = route.entities.target_name
            if Path(target_name).suffix:
                candidates.append(target_name)
        return self._unique_paths(candidates)

    def _next_update_target(self, route: RouterOutput, session: SessionState) -> str | None:
        explicit_targets = self._explicit_target_paths(route)
        if explicit_targets:
            changed_paths = {item.path for item in session.changed_files}
            for candidate in explicit_targets:
                if candidate not in changed_paths:
                    return candidate
            return explicit_targets[0]
        return self._primary_target_path(route, session)

    def _has_pending_explicit_update_targets(self, route: RouterOutput, session: SessionState) -> bool:
        if route.intent != RouteIntent.UPDATE:
            return False
        explicit_targets = self._explicit_target_paths(route)
        if len(explicit_targets) <= 1:
            return False
        changed_paths = {item.path for item in session.changed_files}
        return any(candidate not in changed_paths for candidate in explicit_targets)

    def _has_pending_explicit_create_targets(self, route: RouterOutput, session: SessionState) -> bool:
        if route.intent != RouteIntent.CREATE:
            return False
        explicit_targets = self._explicit_target_paths(route)
        if len(explicit_targets) <= 1:
            return False
        changed_paths = {item.path for item in session.changed_files}
        return any(candidate not in changed_paths for candidate in explicit_targets)

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
        for target in self._explicit_target_paths(route):
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
        if not related_existing:
            return None
        if any(candidate in read_paths for candidate in related_existing):
            return None
        return related_existing[0]

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
        explicit_targets = self._explicit_target_paths(route)
        if explicit_targets:
            changed_paths = {item.path for item in session.changed_files}
            for candidate in explicit_targets:
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
            path = self._default_new_path(route)
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
        if prompt_variant == "compact":
            timeout_seconds = max(self._llm_timeout(60), 60)
            total_timeout_seconds = max(self._llm_timeout(150), 150)
        else:
            timeout_seconds = max(self._llm_timeout(75), 75)
            total_timeout_seconds = max(self._llm_timeout(210), 210)
        num_ctx = self._content_generation_num_ctx(prompt_variant)
        context_pressure = self._context_pressure_estimate(
            prompt=prompt,
            current_content=current_content,
            exc=None,
        )
        primary_tier = "tier_b" if model_name and model_name != self._primary_generation_model_name() else "tier_a"
        primary_strategy = "planned_fast_model" if primary_tier == "tier_b" else "primary_model"
        attempts: list[ExecutionAttemptRecord] = []

        self._log(
            "content_generation_started",
            path=path,
            update=current_content is not None,
            model=effective_model,
            capability_tier=primary_tier,
        )
        outcome = invoke_model(
            lambda progress: self.llm.generate(
                prompt,
                model=model_name,
                timeout=timeout_seconds,
                total_timeout=total_timeout_seconds,
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
                if current_content is not None:
                    review = self._generated_content_integrity_review(path=path, proposed_content=cleaned)
                    if review is None:
                        review = self._explicit_constraint_integrity_review(
                            route,
                            session,
                            path=path,
                            current_content=current_content,
                            proposed_content=cleaned,
                        )
                    if review is None:
                        review = self._review_generated_update(
                            route,
                            session,
                            path=path,
                            current_content=current_content,
                            proposed_content=cleaned,
                        )
                    if not review.safe_to_write:
                        self._log(
                            "proposed_update_review_rejected",
                            path=path,
                            summary=review.summary,
                            blocking_issues=review.blocking_issues[:4],
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
                            failure = self._build_update_review_failure(session, path, final_review)
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
                            )
                        approved_content = review_retry.content
                self._log(
                    "content_generation_finished",
                    path=path,
                    characters=len(approved_content),
                    update=current_content is not None,
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
                return ContentGenerationResult(content=approved_content)
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
            if current_content is not None:
                review = self._generated_content_integrity_review(
                    path=path,
                    proposed_content=retry_result.content,
                )
                if review is None:
                    review = self._explicit_constraint_integrity_review(
                        route,
                        session,
                        path=path,
                        current_content=current_content,
                        proposed_content=retry_result.content,
                    )
                if review is None:
                    review = self._review_generated_update(
                        route,
                        session,
                        path=path,
                        current_content=current_content,
                        proposed_content=retry_result.content,
                    )
                if not review.safe_to_write:
                    self._log(
                        "proposed_update_review_rejected",
                        path=path,
                        summary=review.summary,
                        blocking_issues=review.blocking_issues[:4],
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
                        failure = self._build_update_review_failure(session, path, final_review)
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
                        )
                    approved_content = review_retry.content
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
        attempts = self._content_generation_recovery_attempts(issue)
        if not attempts:
            return GenerationRetryResult()

        retry_attempts: list[ExecutionAttemptRecord] = []
        for attempt in attempts:
            retry_prompt = self._content_generation_prompt_for_attempt(
                attempt,
                route,
                session,
                path=path,
                current_content=current_content,
                prompt=prompt,
                partial_text=issue.partial_text,
                repair_context=repair_context,
                repair_strategy=repair_strategy,
            )
            timeout_seconds, total_timeout_seconds, num_ctx = self._content_generation_runtime_for_attempt(
                attempt,
                issue,
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
    ) -> list[GenerationRecoveryAttempt]:
        policy = ExecutionRecoveryPolicy(
            task_class="content_generation",
            allow_same_backend_retry=True,
            allow_smaller_faster_model=bool(self._lightweight_generation_model_name()),
            allow_resume_after_progress=True,
            allow_reduce_request_complexity=True,
            allow_deterministic_fallback=False,
            max_same_backend_retries=1,
            max_total_attempts=4,
        )
        decisions = policy.plan_recovery(
            issue,
            primary_model=self._primary_generation_model_name(),
            faster_model=self._lightweight_generation_model_name(),
            history=[],
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
                    capability_tier=decision.candidate.capability_tier,
                )
            )
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
    ) -> tuple[int, int, int]:
        if attempt.prompt_kind == "resume":
            base_timeout = 60 if issue.timeout_like else 45
            total_timeout = 180 if issue.timeout_like else 120
            num_ctx = 3072 if attempt.model_name is None else 2048
        elif attempt.prompt_kind == "compact":
            base_timeout = 60 if issue.timeout_like else 45
            total_timeout = 180 if issue.timeout_like else 120
            num_ctx = 2048
        else:
            base_timeout = 75 if issue.timeout_like else 60
            total_timeout = 210 if issue.timeout_like else 150
            num_ctx = 4096 if attempt.model_name is None else 3072
        return (
            max(self._llm_timeout(base_timeout), base_timeout),
            max(self._llm_timeout(total_timeout), total_timeout),
            num_ctx,
        )

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

        explicit_targets = self._explicit_target_paths(route)
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
        if repair_context is None and not prefer_after_runtime_recovery:
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

    def _should_prefer_lightweight_update_generation(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str | None,
    ) -> bool:
        lightweight = self._lightweight_generation_model_name()
        if lightweight is None or current_content is None:
            return False
        if route.intent != RouteIntent.UPDATE or route.needs_clarification or not route.safe_to_execute:
            return False

        target_paths = self._explicit_target_paths(route)
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
            if float(task_state.confidence or 0.0) < MIN_LIGHTWEIGHT_UPDATE_CONFIDENCE:
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

        target_paths = self._explicit_target_paths(route)
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

    def _review_generated_update(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str,
        proposed_content: str,
    ) -> ProposedUpdateReview:
        compact_review = self._should_prefer_lightweight_update_generation(
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
        reserve_model = self._lightweight_generation_model_name()
        primary_model = self._primary_generation_model_name()
        review_attempts: list[tuple[str | None, str, str]] = []
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
                and reserve_model is not None
                and model_name == reserve_model
                and capability_tier == "tier_b"
            )
            prompt = compact_prompt if use_compact_prompt else full_prompt
            timeout = max(self._llm_timeout(18), 18)
            total_timeout = 45 if use_compact_prompt else max(timeout + 12, timeout * 2)
            num_ctx = min(self._llm_num_ctx(2048), 2048) if use_compact_prompt else min(self._llm_num_ctx(6144), 6144)
            prompt_variant = "compact" if use_compact_prompt else "full"
            outcome = invoke_model(
                lambda progress, review_model=model_name: self.llm.generate_json(
                    prompt,
                    system=proposed_update_review_system_prompt(),
                    model=review_model,
                    retries=0,
                    timeout=timeout,
                    total_timeout=total_timeout,
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
            path=path,
            current_content=current_content,
            proposed_content=proposed_content,
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

    def _retry_update_after_review_failure(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str,
        review_feedback: ProposedUpdateReview,
        repair_context: ValidationFailureEvidence | None,
        repair_strategy: str | None,
        prior_attempts: list[ExecutionAttemptRecord],
    ) -> UpdateReviewRetryResult:
        retry_attempts: list[ExecutionAttemptRecord] = []
        last_review = review_feedback
        primary_model = self._primary_generation_model_name()
        reserve_model = self._lightweight_generation_model_name()
        prefer_lightweight_retry = self._should_prefer_lightweight_update_generation(
            route,
            session,
            path=path,
            current_content=current_content,
        )

        full_prompt = generate_content_retry_prompt(
            route,
            session,
            path=path,
            current_content=current_content,
            repair_context=repair_context,
            repair_strategy=repair_strategy,
            review_feedback=last_review,
            mode="full",
        )
        compact_prompt: str | None = None
        if prefer_lightweight_retry:
            compact_prompt = generate_content_retry_prompt(
                route,
                session,
                path=path,
                current_content=current_content,
                repair_context=repair_context,
                repair_strategy=repair_strategy,
                review_feedback=last_review,
                mode="compact",
            )

        retry_models: list[tuple[str | None, str, str, int, int, int, str]] = []
        if prefer_lightweight_retry and reserve_model is not None:
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
        retry_models.append(
            (
                primary_model,
                "tier_a",
                "review_guided_retry" if not prefer_lightweight_retry else "review_guided_primary_fallback",
                max(self._llm_timeout(60), 60),
                max(self._llm_timeout(180), 180),
                min(self._llm_num_ctx(4096), 4096),
                "full",
            )
        )
        if reserve_model is not None and not prefer_lightweight_retry:
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

        seen_models: set[str] = set()
        for (
            model_name,
            capability_tier,
            strategy,
            timeout_seconds,
            total_timeout_seconds,
            num_ctx,
            prompt_variant,
        ) in retry_models:
            normalized_model = str(model_name or "").strip()
            model_key = normalized_model or "__default__"
            if model_key in seen_models:
                continue
            seen_models.add(model_key)

            prompt = compact_prompt if prompt_variant == "compact" and compact_prompt is not None else full_prompt
            self._log(
                "content_generation_retry_started",
                path=path,
                strategy=strategy,
                model=model_name or primary_model,
                reason="proposed_update_review_rejected",
                capability_tier=capability_tier,
                prompt_variant=prompt_variant,
            )
            outcome = invoke_model(
                lambda progress, retry_model=model_name: self.llm.generate(
                    prompt,
                    model=retry_model,
                    timeout=timeout_seconds,
                    total_timeout=total_timeout_seconds,
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
                    update=True,
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
            review = self._review_generated_update(
                route,
                session,
                path=path,
                current_content=current_content,
                proposed_content=cleaned,
            )
            if review.safe_to_write:
                return UpdateReviewRetryResult(
                    content=cleaned,
                    attempts=retry_attempts,
                    review=review,
                    capability_tier=capability_tier,
                    recovery_strategy=strategy,
                )
            self._log(
                "proposed_update_review_rejected",
                path=path,
                summary=review.summary,
                blocking_issues=review.blocking_issues[:4],
                strategy=strategy,
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
    ) -> ContentGenerationFailure:
        language = self._session_language(session)
        blocker_message = f"Pre-write review rejected the proposed update for {path}: {review.summary}"
        user_message = self._localized_text(
            language,
            de=(
                f"Ich habe die vorgeschlagene Aktualisierung fuer {path} verworfen, "
                "weil der AI-Review einen zu breiten oder regressiven Eingriff erkannt hat. "
                "Ich konnte noch keine belastbare, engere Mutation ableiten."
            ),
            en=(
                f"I rejected the proposed update for {path} because the AI review flagged it as too broad or regressive. "
                "I could not derive a reliable narrower mutation yet."
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
        *,
        path: str,
        current_content: str,
        proposed_content: str,
    ) -> ProposedUpdateReview:
        current_length = max(len(current_content), 1)
        proposed_length = max(len(proposed_content), 1)
        current_lines = max(len(current_content.splitlines()), 1)
        proposed_lines = max(len(proposed_content.splitlines()), 1)
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

        if suffix in {".py", ".pyi"}:
            try:
                ast.parse(normalized)
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
            if literal not in proposed_content:
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

        return None

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

        artifacts = self._semantic_review_artifacts(session)
        primary_model = self._primary_generation_model_name()
        reserve_model = self._lightweight_generation_model_name()
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
            total_timeout = 40 if prompt_variant == "compact" else max(timeout + 20, timeout * 2)
            outcome = invoke_model(
                lambda progress, review_model=model_name: self.llm.generate_json(
                    prompt,
                    system=semantic_change_review_system_prompt(),
                    model=review_model,
                    retries=0,
                    timeout=timeout,
                    total_timeout=total_timeout,
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

    def _fallback_semantic_change_review(self, session: SessionState) -> SemanticChangeReview:
        if session.validation_status == "failed":
            return SemanticChangeReview(
                requirements_satisfied=False,
                summary="Existing validation already reports unresolved issues in the changed implementation.",
                confidence=0.85,
                repair_hints=["Inspect the last failing validation evidence before claiming the task is complete."],
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
            return SemanticChangeReview(
                requirements_satisfied=True,
                summary="Available project-aware validation passed and the fallback review found no additional concrete task-to-code mismatch.",
                confidence=0.55,
                file_hints=[item.path for item in session.changed_files[:4]],
            )
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
