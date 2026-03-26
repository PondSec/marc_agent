from __future__ import annotations

from pathlib import Path
import re

from agent.models import SessionState, WorkspaceSnapshot
from agent.prompts import (
    choose_path_prompt,
    final_response_prompt,
    generate_content_prompt,
)
from agent.router import IntentRouter
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


WRITE_INTENTS = {RouteIntent.CREATE, RouteIntent.UPDATE, RouteIntent.DELETE}


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

    def interpret_user_request(
        self,
        task: str,
        snapshot: WorkspaceSnapshot | None,
        session: SessionState | None = None,
    ) -> RouterOutput:
        return self.router.interpret_user_request(task, snapshot, session=session)

    def validate_router_output(self, payload: dict[str, object]) -> RouterOutput:
        return self.router.validate_router_output(payload)

    def analyze_task(
        self,
        task: str,
        snapshot: WorkspaceSnapshot | None,
        session: SessionState | None = None,
    ) -> RouterOutput:
        return self.interpret_user_request(task, snapshot, session=session)

    def create_plan(
        self,
        task: str,
        snapshot: WorkspaceSnapshot | None,
        route: RouterOutput | None = None,
    ) -> PlanningResponse:
        route = route or self.interpret_user_request(task, snapshot)
        steps = [item.reason for item in route.action_plan]
        if route.safe_to_execute and route.intent in WRITE_INTENTS:
            if not any(item.action == RouteActionName.RUN_VALIDATION for item in route.action_plan):
                steps.append("Validate the resulting change with the most relevant project command.")
        completion = [
            "The final action follows the validated router output, not the raw prompt.",
            "Missing information triggers targeted clarification instead of blind execution.",
        ]
        if route.intent in WRITE_INTENTS:
            completion.append("Any code or file change is validated or clearly reported as blocked.")
        else:
            completion.append("The response explains the result or plan in user-facing language.")
        tests = snapshot.likely_commands[:4] if snapshot is not None else []
        return PlanningResponse(
            summary=route.requested_outcome,
            steps=steps or [route.user_goal],
            files_to_inspect=route.entities.target_paths[:8],
            tests_to_run=tests,
            completion_criteria=completion,
        )

    def decide_next_action(self, task: str, session: SessionState) -> AgentDecision:
        del task
        route = session.router_result or self.interpret_user_request(
            session.task,
            session.workspace_snapshot,
            session=session,
        )
        session.router_result = route
        session.task_analysis = route.model_dump()

        if route.needs_clarification:
            return self._final_decision(
                "The router requires clarification before any execution.",
                self._clarification_response(route),
            )

        if not route.safe_to_execute:
            return self._final_decision(
                "Execution is not safe enough to continue without more user input.",
                self._unsafe_response(route),
            )

        if route.direct_response and not route.repo_context_needed and not session.tool_calls:
            return self._final_decision(
                "The validated route can be answered directly without repository work.",
                route.direct_response,
            )

        if session.changed_files:
            command = self._pick_validation_command(session)
            if command is not None:
                return self._validation_decision(
                    "Changed files must go through the remaining validation plan.",
                    command,
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
                            )
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
                            )
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
                    self._render_plan_response(route),
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
        route = session.router_result
        if route is not None and route.direct_response and not session.tool_calls:
            return route.direct_response
        if session.changed_files and session.validation_status == "passed":
            return "Ich habe die Aufgabe umgesetzt und validiert."
        if session.changed_files and session.validation_status == "failed":
            return "Ich habe die Aufgabe umgesetzt, aber die Validierung ist noch nicht sauber."
        if session.changed_files:
            return "Ich habe die Aufgabe umgesetzt."
        if route is not None and route.intent == RouteIntent.PLAN:
            return self._render_plan_response(route)
        if route is not None and route.needs_clarification:
            return self._clarification_response(route)
        return "Ich habe den Workspace untersucht, aber noch kein belastbares Abschlussergebnis erreicht."

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
            return self._final_decision(
                "The update content could not be generated from the routed goal.",
                "Ich konnte noch keine belastbare Aktualisierung fuer die Zieldatei erzeugen.",
            )
        if content.strip() == current_content.strip():
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
        for command in snapshot.likely_commands:
            if command and command not in passed:
                return command
        return None

    def _candidate_paths(self, route: RouterOutput, session: SessionState) -> list[str]:
        candidates: list[str] = []
        candidates.extend(route.entities.target_paths)
        if route.entities.target_name and self._looks_like_path(route.entities.target_name):
            candidates.append(route.entities.target_name)
        candidates.extend(session.candidate_files)
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
        for candidate in route.entities.target_paths:
            if candidate:
                return candidate
        if route.entities.target_name and self._looks_like_path(route.entities.target_name):
            return route.entities.target_name
        try:
            response = self.llm.generate(
                choose_path_prompt(route, session),
                timeout=max(self._llm_timeout(20), 20),
                num_ctx=self._llm_num_ctx(2048),
            )
            path = self._sanitize_generated_path(response, self._preferred_extension(route))
            if path:
                return path
        except Exception as exc:
            self._log("path_generation_error", error=str(exc), route=route.model_dump())
        return self._default_new_path(route)

    def _generate_file_content(
        self,
        route: RouterOutput,
        session: SessionState,
        *,
        path: str,
        current_content: str | None = None,
    ) -> str | None:
        try:
            content = self.llm.generate(
                generate_content_prompt(
                    route,
                    session,
                    path=path,
                    current_content=current_content,
                ),
                timeout=max(self._llm_timeout(180), 180),
                num_ctx=self._llm_num_ctx(8192),
            )
        except Exception as exc:
            self._log("content_generation_error", error=str(exc), path=path)
            return None
        cleaned = self._strip_code_fences(content).strip()
        return cleaned or None

    def _current_file_content(self, session: SessionState, path: str) -> str | None:
        target = Path(session.workspace_root, path)
        if not target.exists() or target.is_dir():
            return None
        return target.read_text(encoding="utf-8")

    def _clarification_response(self, route: RouterOutput) -> str:
        questions = route.clarification_questions[:3]
        if not questions:
            return "Ich brauche noch eine kurze Praezisierung, bevor ich sicher weitermachen kann."
        lines = ["Ich brauche noch eine kurze Praezisierung, bevor ich loslege.", ""]
        for question in questions:
            lines.append(f"- {question}")
        return "\n".join(lines)

    def _unsafe_response(self, route: RouterOutput) -> str:
        if route.needs_clarification:
            return self._clarification_response(route)
        return (
            "Ich fuehre das noch nicht aus, weil der Router die Anfrage noch nicht als sicher genug eingestuft hat."
        )

    def _render_plan_response(self, route: RouterOutput) -> str:
        lines = [f"Ziel: {route.user_goal}", "", "Vorgehen:"]
        for item in route.action_plan:
            lines.append(f"{item.step}. {item.reason}")
        return "\n".join(lines)

    def _compose_user_response(self, route: RouterOutput, session: SessionState) -> str:
        if route.direct_response and not session.tool_calls and not session.changed_files:
            return route.direct_response
        if route.intent == RouteIntent.PLAN:
            return self._render_plan_response(route)
        try:
            response = self.llm.generate(
                final_response_prompt(route, session),
                timeout=max(self._llm_timeout(20), 20),
                num_ctx=self._llm_num_ctx(4096),
            ).strip()
            if response:
                return self._strip_code_fences(response).strip()
        except Exception as exc:
            self._log("final_response_generation_error", error=str(exc))
        return self.summarize_session(session)

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
        target = self._primary_target_name(route).lower()
        slug = re.sub(r"[^a-z0-9]+", "_", target).strip("_") or "generated_output"
        extension = self._preferred_extension(route)
        if not slug.endswith(extension):
            return f"{slug}{extension}"
        return slug

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

    def _log(self, event: str, **payload) -> None:
        if self.logger is None:
            return
        self.logger.log_event(event, **payload)
