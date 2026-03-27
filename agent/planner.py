from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time

from agent.decision import ExecutionDecisionPolicy
from agent.models import SessionState, WorkspaceSnapshot
from agent.prompts import (
    choose_path_prompt,
    final_response_prompt,
    generate_content_continuation_prompt,
    generate_content_prompt,
    generate_content_retry_prompt,
)
from agent.router import IntentRouter
from agent.state_updater import TaskStateUpdater
from agent.task_state import TaskState
from agent.verification import ValidationPlanner
from llm.ollama_client import OllamaGenerationError
from llm.provider import LLMProvider
from llm.schemas import (
    AgentActionType,
    AgentDecision,
    PlanningResponse,
    RouteActionName,
    RouteIntent,
    RouterOutput,
)
from runtime.logger import AgentLogger


WRITE_INTENTS = {RouteIntent.CREATE, RouteIntent.UPDATE, RouteIntent.DEBUG, RouteIntent.DELETE}


@dataclass(slots=True)
class GenerationIssue:
    reason: str
    timeout_like: bool
    context_pressure_likely: bool
    partial_text: str = ""
    had_progress: bool = False
    characters: int = 0


@dataclass(slots=True)
class GenerationRecoveryAttempt:
    strategy: str
    prompt_kind: str
    model_name: str | None = None


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

        if session.changed_files:
            if (
                session.validation_status == "failed"
                and session.task_state is not None
                and session.task_state.execution_strategy == "debug_repair"
            ):
                repair_decision = self.execute_action_from_plan(route, session)
                if repair_decision.action_type != AgentActionType.FINAL or session.stop_reason:
                    return repair_decision
            command = self._pick_validation_command(session)
            if command is not None:
                return self._validation_decision(
                    "Changed files must go through the remaining validation plan.",
                    command,
                )
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
                candidate = self._next_unread_candidate(candidate_paths, read_paths)
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
                target = self._primary_target_path(route, session)
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
                if target not in read_paths:
                    return AgentDecision(
                        thought_summary="Read the target file before generating an update.",
                        action_type=AgentActionType.CALL_TOOL,
                        tool_name="read_file",
                        tool_args={"path": target},
                        expected_outcome="Inspect the current implementation before editing it.",
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
            if session.validation_status == "passed":
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
        content = self._generate_file_content(route, session, path=path)
        if not content:
            self._record_generation_blocker(
                session,
                "No reliable content could be generated for the routed artifact.",
                stop_reason="generation_failed",
            )
            return self._final_decision(
                "The new artifact could not be generated from the routed goal.",
                "Ich konnte noch keinen belastbaren Inhalt fuer die neue Datei erzeugen.",
            )
        absolute_target = Path(session.workspace_root, path)
        tool_name = "write_file" if absolute_target.exists() else "create_file"
        tool_args = {"path": path, "content": content}
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
        current_content = self._current_file_content(session, target)
        if current_content is None:
            return None
        content = self._generate_file_content(
            route,
            session,
            path=target,
            current_content=current_content,
        )
        if not content:
            self._record_generation_blocker(
                session,
                f"No reliable update content could be generated for {target}.",
                stop_reason="generation_failed",
            )
            return self._final_decision(
                "The update content could not be generated from the routed goal.",
                "Ich konnte noch keine belastbare Aktualisierung fuer die Zieldatei erzeugen.",
            )
        if content.strip() == current_content.strip():
            self._record_generation_blocker(
                session,
                f"The routed update for {target} did not produce a real file change.",
                stop_reason="no_effective_change",
            )
            return self._final_decision(
                "The routed update does not change the current file content.",
                "Ich habe keine belastbare inhaltliche Aenderung fuer die Zieldatei ableiten koennen.",
            )
        return AgentDecision(
            thought_summary=f"Update {target} according to the routed goal.",
            action_type=AgentActionType.CALL_TOOL,
            tool_name="write_file",
            tool_args={"path": target, "content": content},
            expected_outcome="Apply the requested update to the target artifact.",
            final_response=None,
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
            run.command
            for run in session.validation_runs
            if run.edit_generation == session.edit_generation and run.status == "passed"
        }
        for item in session.validation_plan:
            if item.command not in passed:
                return item.command
        for command in session.verification_commands:
            if command and command not in passed:
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
            if item.command not in passed:
                return item.command
        for command in snapshot.likely_commands:
            if command and command not in passed:
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

    def _primary_target_path(self, route: RouterOutput, session: SessionState) -> str | None:
        candidates = self._candidate_paths(route, session)
        return candidates[0] if candidates else None

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
        try:
            self._log("path_generation_started", target_name=route.entities.target_name)
            response = self.llm.generate(
                choose_path_prompt(route, session),
                model=self._lightweight_generation_model_name(),
                timeout=max(self._llm_timeout(20), 20),
                total_timeout=max(self._llm_timeout(45), 45),
                num_ctx=1536,
                retries=0,
            )
            path = self._sanitize_generated_path(response, self._preferred_extension(route))
            if path:
                self._log("path_generation_finished", path=path)
                return path
        except Exception as exc:
            self._log("path_generation_error", error=str(exc), route=route.model_dump())
        path = self._default_new_path(route)
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
    ) -> str | None:
        prompt = generate_content_prompt(
            route,
            session,
            path=path,
            current_content=current_content,
        )
        try:
            self._log(
                "content_generation_started",
                path=path,
                update=current_content is not None,
                model=self._primary_generation_model_name(),
            )
            content = self.llm.generate(
                prompt,
                timeout=max(self._llm_timeout(75), 75),
                total_timeout=max(self._llm_timeout(210), 210),
                num_ctx=4096,
                retries=0,
                progress_callback=self._progress_logger(
                    "content_generation_progress",
                    path=path,
                    update=current_content is not None,
                ),
            )
        except Exception as exc:
            issue = self._assess_generation_issue(
                exc,
                prompt=prompt,
                current_content=current_content,
            )
            self._log(
                "content_generation_error",
                error=str(exc),
                path=path,
                reason=issue.reason,
                had_progress=issue.had_progress,
                partial_characters=issue.characters,
                context_pressure=issue.context_pressure_likely,
            )
            retried = self._retry_content_generation(
                route,
                session=session,
                path=path,
                current_content=current_content,
                cause=exc,
                prompt=prompt,
            )
            if retried is not None:
                self._log(
                    "content_generation_finished",
                    path=path,
                    characters=len(retried),
                    update=current_content is not None,
                    source="retry",
                )
                return retried
            fallback = self._template_fallback_content(
                route,
                session,
                path=path,
                current_content=current_content,
            )
            if fallback is None:
                return None
            self._log("content_generation_fallback_started", path=path, source="template")
            self._log("content_generation_fallback_finished", path=path, source="template")
            return fallback
        cleaned = self._strip_code_fences(content).strip()
        self._log(
            "content_generation_finished",
            path=path,
            characters=len(cleaned),
            update=current_content is not None,
        )
        return cleaned or None

    def _retry_content_generation(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str | None = None,
        cause: Exception | None = None,
        prompt: str,
    ) -> str | None:
        issue = self._assess_generation_issue(
            cause,
            prompt=prompt,
            current_content=current_content,
        )
        attempts = self._content_generation_recovery_attempts(issue)
        if not attempts:
            return None

        for attempt in attempts:
            retry_prompt = self._content_generation_prompt_for_attempt(
                attempt,
                route,
                session,
                path=path,
                current_content=current_content,
                prompt=prompt,
                partial_text=issue.partial_text,
            )
            timeout_seconds, total_timeout_seconds, num_ctx = self._content_generation_runtime_for_attempt(
                attempt,
                issue,
            )
            try:
                self._log(
                    "content_generation_retry_started",
                    path=path,
                    strategy=attempt.strategy,
                    model=attempt.model_name or self._primary_generation_model_name(),
                    reason=issue.reason,
                    had_progress=issue.had_progress,
                    partial_characters=issue.characters,
                    context_pressure=issue.context_pressure_likely,
                )
                text = self.llm.generate(
                    retry_prompt,
                    model=attempt.model_name,
                    timeout=timeout_seconds,
                    total_timeout=total_timeout_seconds,
                    num_ctx=num_ctx,
                    retries=0,
                    progress_callback=self._progress_logger(
                        "content_generation_progress",
                        path=path,
                        update=current_content is not None,
                        strategy=attempt.strategy,
                    ),
                )
                cleaned = self._strip_code_fences(text).strip()
                if cleaned:
                    self._log(
                        "content_generation_retry_finished",
                        path=path,
                        strategy=attempt.strategy,
                        characters=len(cleaned),
                    )
                    return cleaned
            except Exception as exc:
                retry_issue = self._assess_generation_issue(
                    exc,
                    prompt=retry_prompt,
                    current_content=current_content,
                )
                self._log(
                    "content_generation_retry_error",
                    path=path,
                    strategy=attempt.strategy,
                    error=str(exc),
                    reason=retry_issue.reason,
                    had_progress=retry_issue.had_progress,
                    partial_characters=retry_issue.characters,
                )
        return None

    def _assess_generation_issue(
        self,
        exc: Exception | None,
        *,
        prompt: str,
        current_content: str | None = None,
    ) -> GenerationIssue:
        partial_text = self._strip_code_fences(str(getattr(exc, "partial_text", "") or "")).strip()
        characters = int(getattr(exc, "characters", len(partial_text)) or len(partial_text))
        had_progress = bool(getattr(exc, "progress_seen", False)) or bool(partial_text) or characters > 0
        timeout_like = bool(getattr(exc, "timed_out", False)) or self._is_timeout_error(exc)
        reason = str(getattr(exc, "reason", "") or "").strip()
        if not reason:
            reason = "timeout" if timeout_like else "runtime_error"
        return GenerationIssue(
            reason=reason,
            timeout_like=timeout_like,
            context_pressure_likely=self._context_pressure_likely(
                prompt=prompt,
                current_content=current_content,
                exc=exc,
            ),
            partial_text=partial_text,
            had_progress=had_progress,
            characters=characters,
        )

    def _context_pressure_likely(
        self,
        *,
        prompt: str,
        current_content: str | None,
        exc: Exception | None,
    ) -> bool:
        lowered = str(exc or "").lower()
        context_markers = (
            "context",
            "token",
            "num_ctx",
            "prompt too long",
            "input too long",
            "out of memory",
            "kv cache",
        )
        if any(marker in lowered for marker in context_markers):
            return True
        if len(prompt) >= 12_000:
            return True
        if current_content is not None and len(current_content) >= 6_000:
            return True
        return False

    def _content_generation_recovery_attempts(
        self,
        issue: GenerationIssue,
    ) -> list[GenerationRecoveryAttempt]:
        attempts: list[GenerationRecoveryAttempt] = []
        fallback_model = self._lightweight_generation_model_name()

        if issue.partial_text:
            attempts.append(GenerationRecoveryAttempt("resume_same_model", "resume"))
        elif issue.context_pressure_likely:
            attempts.append(GenerationRecoveryAttempt("compact_same_model", "compact"))
        elif issue.reason == "startup_timeout" or (issue.timeout_like and not fallback_model):
            attempts.append(GenerationRecoveryAttempt("retry_same_model", "full"))
        elif not issue.timeout_like:
            return attempts

        if fallback_model:
            if issue.partial_text:
                attempts.append(
                    GenerationRecoveryAttempt("resume_fallback_model", "resume", fallback_model)
                )
            elif issue.context_pressure_likely:
                attempts.append(
                    GenerationRecoveryAttempt("compact_fallback_model", "compact", fallback_model)
                )
            elif issue.timeout_like:
                attempts.append(GenerationRecoveryAttempt("fallback_model", "full", fallback_model))
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
    ) -> str:
        if attempt.prompt_kind == "resume":
            return generate_content_continuation_prompt(
                route,
                session,
                path=path,
                partial_content=partial_text,
                current_content=current_content,
            )
        if attempt.prompt_kind == "compact":
            return generate_content_retry_prompt(
                route,
                session,
                path=path,
                current_content=current_content,
            )
        return prompt

    def _content_generation_runtime_for_attempt(
        self,
        attempt: GenerationRecoveryAttempt,
        issue: GenerationIssue,
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
        try:
            self._log(
                "final_response_generation_started",
                model=self._lightweight_generation_model_name() or self._primary_generation_model_name(),
            )
            response = self.llm.generate(
                prompt,
                model=self._lightweight_generation_model_name(),
                timeout=max(self._llm_timeout(20), 20),
                total_timeout=max(self._llm_timeout(60), 60),
                num_ctx=1024,
                retries=0,
                progress_callback=self._progress_logger("final_response_generation_progress"),
            ).strip()
            if response:
                self._log("final_response_generation_finished", characters=len(response))
                return self._strip_code_fences(response).strip()
        except Exception as exc:
            issue = self._assess_generation_issue(exc, prompt=prompt)
            self._log(
                "final_response_generation_error",
                error=str(exc),
                reason=issue.reason,
                had_progress=issue.had_progress,
                partial_characters=issue.characters,
            )
        deterministic = self._deterministic_final_response(route, session)
        self._log(
            "final_response_generation_finished",
            characters=len(deterministic),
            source="deterministic",
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
    ) -> None:
        text = str(message or "").strip()
        if not text:
            return
        if text not in session.blockers:
            session.blockers.append(text)
            session.blockers = session.blockers[-10:]
        session.last_error = text
        session.stop_reason = stop_reason

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
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.splitlines()[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.splitlines()[:-1])
        return cleaned.strip()

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

    def _lightweight_generation_model_name(self) -> str | None:
        config = getattr(self.llm, "config", None)
        if config is None:
            return None
        candidate = str(getattr(config, "router_model_name", "") or "").strip()
        primary = str(getattr(config, "model_name", "") or "").strip()
        if not candidate or candidate == primary:
            return None
        return candidate

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
