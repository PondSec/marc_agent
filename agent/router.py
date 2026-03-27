from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from agent.models import SessionState, WorkspaceSnapshot
from agent.prompts import router_prompt, router_repair_prompt, router_system_prompt
from llm.provider import LLMProvider
from llm.runtime_resilience import (
    ExecutionRecoveryPolicy,
    build_execution_run_record,
    estimate_context_pressure,
    invoke_model,
)
from llm.schemas import RouteActionName, RouteIntent, RouterOutput
from runtime.logger import AgentLogger


class IntentRouter:
    def __init__(
        self,
        llm: LLMProvider,
        *,
        logger: AgentLogger | None = None,
        model_name: str | None = None,
        timeout: int = 12,
        num_ctx: int = 4096,
        retries: int = 1,
    ):
        self.llm = llm
        self.logger = logger
        self.model_name = model_name
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.retries = max(int(retries), 0)

    def interpret_user_request(
        self,
        user_input: str,
        snapshot: WorkspaceSnapshot | None,
        session: SessionState | None = None,
    ) -> RouterOutput:
        self._log("router_input", raw_user_input=user_input)
        fast_path = self._fast_path_route(
            user_input,
            snapshot=snapshot,
            session=session,
        )
        if fast_path is not None:
            self._log("router_fast_path", router_result=fast_path.model_dump())
            return fast_path
        payload: dict[str, Any] | None = None
        prompt = router_prompt(user_input, snapshot, session=session)
        context_pressure = estimate_context_pressure(prompt_chars=len(prompt))
        primary_model = self._primary_model_name()
        attempts = []
        outcome = invoke_model(
            lambda progress: self.llm.generate_json(
                prompt,
                system=router_system_prompt(),
                model=primary_model,
                retries=0,
                timeout=self.timeout,
                num_ctx=self.num_ctx,
                progress_callback=progress,
            ),
            operation_name="router_generation",
            task_class="router_generation",
            attempt_number=1,
            capability_tier="tier_a",
            recovery_strategy="primary_model_generation",
            prompt_variant="full",
            model_identifier=primary_model,
            backend_identifier="ollama",
            inactivity_timeout_seconds=self.timeout,
            total_timeout_seconds=max(self.timeout + 18, self.timeout * 2),
            context_pressure_estimate=context_pressure,
            event_callback=self._progress_logger("router_generation_progress"),
        )
        attempts.append(outcome.attempt)
        try:
            if outcome.exception is None:
                payload = outcome.value
                route = self.validate_router_output(payload)
                self._append_runtime_execution(
                    session,
                    build_execution_run_record(
                        operation_name="router_generation",
                        task_class="router_generation",
                        final_state="completed",
                        capability_tier="tier_a",
                        recovery_strategy="primary_model_generation",
                        degraded=False,
                        honest_blocked=False,
                        artifact_bytes_generated=0,
                        validation_possible=False,
                        summary="Route generation completed on the primary tier.",
                        attempts=attempts,
                    ),
                )
                return route
        except ValidationError as exc:
            self._log(
                "router_validation_failed",
                raw_router_output=payload or {},
                errors=exc.errors(),
            )
            repaired = self._repair_invalid_output(payload or {}, exc.errors(), session=session)
            if repaired is not None:
                return repaired
        if outcome.exception is not None:
            exc = outcome.exception
            self._log(
                "router_error",
                raw_router_output=payload or {},
                error=str(exc),
                failure=outcome.attempt.failure.to_dict()
                if outcome.attempt.failure is not None
                else None,
            )
            if self._is_timeout_error(exc):
                fast_timeout_route = self._emergency_route(
                    user_input,
                    snapshot=snapshot,
                    session=session,
                )
                if fast_timeout_route is not None and fast_timeout_route.intent == RouteIntent.CREATE:
                    self._append_runtime_execution(
                        session,
                        build_execution_run_record(
                            operation_name="router_generation",
                            task_class="router_generation",
                            final_state="degraded_success",
                            capability_tier="tier_d",
                            recovery_strategy="deterministic_fallback",
                            degraded=True,
                            honest_blocked=False,
                            artifact_bytes_generated=0,
                            validation_possible=False,
                            summary="The router used the deterministic emergency route after a startup-time runtime issue.",
                            attempts=attempts,
                        ),
                    )
                    self._log(
                        "router_timeout_fast_fallback",
                        router_result=fast_timeout_route.model_dump(),
                    )
                    return fast_timeout_route
            failure = outcome.attempt.failure
            policy = ExecutionRecoveryPolicy(
                task_class="router_generation",
                allow_same_backend_retry=True,
                allow_reduce_request_complexity=True,
                allow_deterministic_fallback=True,
                max_same_backend_retries=1,
                max_total_attempts=3,
            )
            decisions = policy.plan_recovery(
                failure,
                primary_model=primary_model,
                faster_model=None,
                history=attempts,
            ) if failure is not None else []
            for decision in decisions:
                self._log(
                    "router_recovery_option",
                    strategy=decision.candidate.strategy,
                    capability_tier=decision.candidate.capability_tier,
                    prompt_variant=decision.candidate.prompt_variant,
                    accepted=decision.accepted,
                    reason=decision.reason,
                )
                if not decision.accepted:
                    continue
                if decision.candidate.local_only:
                    break
                retried = self._retry_after_timeout(
                    user_input,
                    session=session,
                    strategy=decision.candidate.strategy,
                    capability_tier=decision.candidate.capability_tier,
                )
                if retried is not None:
                    route, retry_attempts = retried
                    attempts.extend(retry_attempts)
                    self._append_runtime_execution(
                        session,
                        build_execution_run_record(
                            operation_name="router_generation",
                            task_class="router_generation",
                            final_state="completed",
                            capability_tier=decision.candidate.capability_tier,
                            recovery_strategy=decision.candidate.strategy,
                            degraded=decision.candidate.capability_tier != "tier_a",
                            honest_blocked=False,
                            artifact_bytes_generated=0,
                            validation_possible=False,
                            summary="Route generation recovered after a runtime startup issue.",
                            attempts=attempts,
                        ),
                    )
                    return route
        fallback = self._fallback_route(user_input, snapshot=snapshot, session=session)
        self._append_runtime_execution(
            session,
            build_execution_run_record(
                operation_name="router_generation",
                task_class="router_generation",
                final_state="degraded_success",
                capability_tier="tier_d",
                recovery_strategy="deterministic_fallback",
                degraded=True,
                honest_blocked=False,
                artifact_bytes_generated=0,
                validation_possible=False,
                summary="The router fell back to deterministic route heuristics.",
                attempts=attempts,
            ),
        )
        self._log("router_fallback", router_result=fallback.model_dump())
        return fallback

    def validate_router_output(self, payload: dict[str, Any]) -> RouterOutput:
        route = RouterOutput.model_validate(payload)
        self._log(
            "router_validation_succeeded",
            validation_result="valid",
            router_result=route.model_dump(),
        )
        return route

    def _repair_invalid_output(
        self,
        invalid_payload: dict[str, Any],
        errors: list[dict[str, Any]],
        *,
        session: SessionState | None = None,
    ) -> RouterOutput | None:
        try:
            repair_prompt = router_repair_prompt(invalid_payload, errors)
            outcome = invoke_model(
                lambda progress: self.llm.generate_json(
                    repair_prompt,
                    system=router_system_prompt(),
                    model=self._primary_model_name(),
                    retries=0,
                    timeout=max(self.timeout, 16),
                    num_ctx=self.num_ctx,
                    progress_callback=progress,
                ),
                operation_name="router_repair_generation",
                task_class="router_generation",
                attempt_number=1,
                capability_tier="tier_c",
                recovery_strategy="repair_invalid_router_output",
                prompt_variant="repair",
                model_identifier=self._primary_model_name(),
                backend_identifier="ollama",
                inactivity_timeout_seconds=max(self.timeout, 16),
                total_timeout_seconds=max(self.timeout + 18, 32),
                context_pressure_estimate=estimate_context_pressure(prompt_chars=len(repair_prompt)),
                event_callback=self._progress_logger("router_generation_progress"),
            )
            if outcome.exception is not None:
                raise outcome.exception
            route = self.validate_router_output(outcome.value)
            self._append_runtime_execution(
                session,
                build_execution_run_record(
                    operation_name="router_repair_generation",
                    task_class="router_generation",
                    final_state="completed",
                    capability_tier="tier_c",
                    recovery_strategy="repair_invalid_router_output",
                    degraded=True,
                    honest_blocked=False,
                    artifact_bytes_generated=0,
                    validation_possible=False,
                    summary="The router repaired invalid structured output with a constrained follow-up generation.",
                    attempts=[outcome.attempt],
                ),
            )
            self._log("router_repair_succeeded", router_result=route.model_dump())
            return route
        except Exception as exc:
            self._log(
                "router_repair_failed",
                raw_router_output=invalid_payload,
                errors=errors,
                error=str(exc),
            )
            return None

    def _retry_after_timeout(
        self,
        user_input: str,
        *,
        session: SessionState | None = None,
        strategy: str = "minimal_prompt_after_timeout",
        capability_tier: str = "tier_c",
    ) -> tuple[RouterOutput, list[Any]] | None:
        self._log(
            "router_retry_started",
            strategy=strategy,
            retry_timeout=max(self.timeout + 8, 26),
            retry_num_ctx=min(self.num_ctx, 2048),
        )
        try:
            retry_prompt = router_prompt(user_input, None, session=None)
            outcome = invoke_model(
                lambda progress: self.llm.generate_json(
                    retry_prompt,
                    system=router_system_prompt(),
                    model=self._primary_model_name(),
                    retries=0,
                    timeout=max(self.timeout + 8, 26),
                    num_ctx=min(self.num_ctx, 2048),
                    progress_callback=progress,
                ),
                operation_name="router_generation",
                task_class="router_generation",
                attempt_number=1,
                capability_tier=capability_tier,
                recovery_strategy=strategy,
                prompt_variant="compact",
                model_identifier=self._primary_model_name(),
                backend_identifier="ollama",
                inactivity_timeout_seconds=max(self.timeout + 8, 26),
                total_timeout_seconds=max(self.timeout + 30, 52),
                context_pressure_estimate=estimate_context_pressure(prompt_chars=len(retry_prompt)),
                event_callback=self._progress_logger("router_generation_progress"),
            )
            if outcome.exception is not None:
                raise outcome.exception
            route = self.validate_router_output(outcome.value)
            self._log("router_retry_succeeded", router_result=route.model_dump())
            return route, [outcome.attempt]
        except Exception as exc:
            self._log("router_retry_failed", error=str(exc))
            return None

    def _fallback_route(
        self,
        user_input: str,
        *,
        snapshot: WorkspaceSnapshot | None = None,
        session: SessionState | None = None,
    ) -> RouterOutput:
        trimmed = str(user_input or "").strip() or "Unklarer Nutzerwunsch"
        direct_response = self._fallback_direct_response(trimmed)
        if direct_response is not None:
            return RouterOutput(
                user_goal=trimmed,
                intent=RouteIntent.EXPLAIN,
                requested_outcome="Provide a short direct answer without repository work.",
                action_plan=[
                    {
                        "step": 1,
                        "action": RouteActionName.RESPOND_DIRECTLY,
                        "reason": "This is a simple conversational request that does not require tool execution.",
                    }
                ],
                needs_clarification=False,
                clarification_questions=[],
                confidence=0.3,
                safe_to_execute=True,
                repo_context_needed=False,
                search_terms=[],
                relevant_extensions=[],
                direct_response=direct_response,
            )
        emergency = self._emergency_route(trimmed, snapshot=snapshot, session=session)
        if emergency is not None:
            return emergency
        return RouterOutput(
            user_goal=trimmed,
            intent=RouteIntent.UNKNOWN,
            requested_outcome="Clarify the user's exact target before doing any work.",
            action_plan=[
                {
                    "step": 1,
                    "action": RouteActionName.ASK_CLARIFICATION,
                    "reason": "The router could not derive a safe executable goal with enough confidence.",
                }
            ],
            needs_clarification=True,
            clarification_questions=[
                "Was genau soll ich fuer dich erreichen?",
                "Auf welches Objekt, welche Datei oder welchen Bereich soll ich mich konzentrieren?",
            ],
            confidence=0.0,
            safe_to_execute=False,
            repo_context_needed=False,
            search_terms=[],
            relevant_extensions=[],
            direct_response=None,
        )

    def _fast_path_route(
        self,
        user_input: str,
        *,
        snapshot: WorkspaceSnapshot | None = None,
        session: SessionState | None = None,
    ) -> RouterOutput | None:
        trimmed = str(user_input or "").strip()
        if not trimmed:
            return None
        emergency = self._emergency_route(trimmed, snapshot=snapshot, session=session)
        if emergency is None:
            return None
        normalized = " ".join(trimmed.lower().split()).strip()
        if emergency.intent == RouteIntent.CREATE and (
            not self._follow_up_target_paths(session)
            or self._has_explicit_new_artifact_request(normalized)
        ):
            return emergency
        if emergency.intent == RouteIntent.DEBUG and self._has_follow_up_context(session):
            return emergency
        if emergency.intent == RouteIntent.UPDATE and self._follow_up_target_paths(session):
            return emergency
        return None

    def _fallback_direct_response(self, user_input: str) -> str | None:
        normalized = " ".join(str(user_input or "").lower().split()).strip("!?., ")
        if not normalized:
            return None
        greetings = {
            "hallo",
            "hello",
            "hi",
            "hey",
            "moin",
            "servus",
            "guten morgen",
            "guten tag",
            "guten abend",
        }
        if normalized in greetings:
            return (
                "Hallo. Ich bin bereit.\n\n"
                "Wenn du magst, kann ich den Code analysieren, eine Aenderung planen oder etwas im Projekt umsetzen."
            )
        intro_fragments = (
            "wer bist du",
            "who are you",
            "was kannst du",
            "what can you do",
            "was machst du",
            "what do you do",
            "hilfe",
            "help",
        )
        if any(fragment in normalized for fragment in intro_fragments):
            return (
                "Ich bin dein lokaler Coding-Agent fuer diesen Workspace.\n\n"
                "Ich kann Code analysieren, Aenderungen planen und auf Basis des validierten Router-Outputs ausfuehren."
            )
        return None

    def _emergency_route(
        self,
        user_input: str,
        *,
        snapshot: WorkspaceSnapshot | None = None,
        session: SessionState | None = None,
    ) -> RouterOutput | None:
        normalized = " ".join(str(user_input or "").lower().split()).strip()
        if not normalized:
            return None

        path = self._extract_path(user_input)
        target_name = self._extract_target_name(normalized, path)
        relevant_extensions = self._detect_extensions(normalized, path)
        search_terms = [target_name] if target_name and len(target_name) >= 3 else []

        if self._looks_like_plan_request(normalized):
            return self._build_route(
                user_goal=user_input,
                intent=RouteIntent.PLAN,
                requested_outcome="Provide a practical implementation plan.",
                action_plan=[
                    (RouteActionName.PLAN_WORK, "The request clearly asks for a plan or approach."),
                ],
                repo_context_needed=False,
                target_name=target_name,
                target_paths=[path] if path else [],
                search_terms=search_terms,
                relevant_extensions=relevant_extensions,
                confidence=0.58,
            )

        if self._looks_like_explanation_request(normalized):
            return self._build_route(
                user_goal=user_input,
                intent=RouteIntent.EXPLAIN,
                requested_outcome="Explain the relevant code or concept.",
                action_plan=[
                    (RouteActionName.INSPECT_WORKSPACE, "The agent should inspect the workspace before explaining."),
                    (RouteActionName.READ_RELEVANT_FILES, "Relevant files should be read before summarizing."),
                    (RouteActionName.SUMMARIZE_RESULT, "The user expects an explanation rather than a mutation."),
                ],
                repo_context_needed=True,
                target_name=target_name,
                target_paths=[path] if path else [],
                search_terms=search_terms,
                relevant_extensions=relevant_extensions,
                confidence=0.56,
            )

        if self._looks_like_search_request(normalized):
            return self._build_route(
                user_goal=user_input,
                intent=RouteIntent.SEARCH,
                requested_outcome="Locate the relevant implementation area in the workspace.",
                action_plan=[
                    (RouteActionName.SEARCH_WORKSPACE, "The request is to locate or find something in the repo."),
                    (RouteActionName.READ_RELEVANT_FILES, "Any search hit should be inspected before answering."),
                    (RouteActionName.SUMMARIZE_RESULT, "Return the located result to the user."),
                ],
                repo_context_needed=True,
                target_name=target_name,
                target_paths=[path] if path else [],
                search_terms=search_terms or self._fallback_search_terms(normalized),
                relevant_extensions=relevant_extensions,
                confidence=0.63,
            )

        if self._looks_like_delete_request(normalized, path):
            if path is None:
                return None
            return self._build_route(
                user_goal=user_input,
                intent=RouteIntent.DELETE,
                requested_outcome="Delete the specified artifact safely.",
                action_plan=[
                    (RouteActionName.READ_RELEVANT_FILES, "The target should be confirmed once before deletion."),
                    (RouteActionName.DELETE_ARTIFACT, "The request is to remove the specified target."),
                    (RouteActionName.SUMMARIZE_RESULT, "Return the deletion outcome."),
                ],
                repo_context_needed=True,
                target_name=target_name or path,
                target_paths=[path],
                search_terms=search_terms or [path],
                relevant_extensions=relevant_extensions,
                confidence=0.74,
            )

        follow_up_debug = self._follow_up_debug_route(
            user_input,
            normalized=normalized,
            path=path,
            session=session,
            target_name=target_name,
            search_terms=search_terms,
            relevant_extensions=relevant_extensions,
        )
        if follow_up_debug is not None:
            return follow_up_debug

        follow_up_update = self._follow_up_update_route(
            user_input,
            normalized=normalized,
            path=path,
            session=session,
            target_name=target_name,
            search_terms=search_terms,
            relevant_extensions=relevant_extensions,
        )
        if follow_up_update is not None:
            return follow_up_update

        if self._looks_like_update_request(normalized, path, session=session):
            target_paths = [path] if path else self._follow_up_target_paths(session)
            if not target_paths:
                return None
            return self._build_route(
                user_goal=user_input,
                intent=RouteIntent.UPDATE,
                requested_outcome="Update the relevant existing artifact.",
                action_plan=[
                    (RouteActionName.READ_RELEVANT_FILES, "The current implementation should be inspected before editing."),
                    (RouteActionName.UPDATE_ARTIFACT, "The request is to change an existing artifact."),
                    (RouteActionName.RUN_VALIDATION, "The edited code should be validated before finishing."),
                    (RouteActionName.SUMMARIZE_RESULT, "Return the update outcome."),
                ],
                repo_context_needed=True,
                target_name=target_name or target_paths[0],
                target_paths=target_paths,
                search_terms=search_terms or [target_name or target_paths[0]],
                relevant_extensions=relevant_extensions,
                confidence=0.72,
            )

        if self._looks_like_create_request(normalized, path, session=session):
            target_paths = [path] if path else []
            if snapshot is not None and snapshot.file_count > 0:
                action_plan = [
                    (RouteActionName.INSPECT_WORKSPACE, "Project conventions should be inspected before creating new code."),
                    (RouteActionName.CREATE_ARTIFACT, "The request clearly asks for a new artifact or implementation."),
                    (RouteActionName.RUN_VALIDATION, "New code should be validated before finishing."),
                    (RouteActionName.SUMMARIZE_RESULT, "Return the creation outcome."),
                ]
            else:
                action_plan = [
                    (RouteActionName.CREATE_ARTIFACT, "The request clearly asks for a new artifact or implementation."),
                    (RouteActionName.RUN_VALIDATION, "New code should be validated before finishing."),
                    (RouteActionName.SUMMARIZE_RESULT, "Return the creation outcome."),
                ]
            return self._build_route(
                user_goal=user_input,
                intent=RouteIntent.CREATE,
                requested_outcome="Create the requested new artifact or implementation.",
                action_plan=action_plan,
                repo_context_needed=True,
                target_name=target_name,
                target_paths=target_paths,
                search_terms=search_terms or self._fallback_search_terms(normalized),
                relevant_extensions=relevant_extensions,
                confidence=0.69,
            )

        return None

    def _build_route(
        self,
        *,
        user_goal: str,
        intent: RouteIntent,
        requested_outcome: str,
        action_plan: list[tuple[RouteActionName, str]],
        repo_context_needed: bool,
        target_name: str | None,
        target_paths: list[str],
        search_terms: list[str],
        relevant_extensions: list[str],
        confidence: float,
    ) -> RouterOutput:
        target_type = None
        if target_paths:
            target_type = "file"
        elif target_name:
            target_type = "artifact"
        return RouterOutput(
            user_goal=user_goal,
            intent=intent,
            entities={
                "target_type": target_type,
                "target_name": target_name,
                "target_paths": target_paths,
                "attributes": [],
                "constraints": [],
            },
            requested_outcome=requested_outcome,
            action_plan=[
                {"step": index, "action": action, "reason": reason}
                for index, (action, reason) in enumerate(action_plan, start=1)
            ],
            needs_clarification=False,
            clarification_questions=[],
            confidence=confidence,
            safe_to_execute=True,
            repo_context_needed=repo_context_needed,
            search_terms=search_terms[:6],
            relevant_extensions=relevant_extensions[:6],
            direct_response=None,
        )

    def _is_timeout_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "timed out" in message or "timeout" in message

    def _extract_path(self, user_input: str) -> str | None:
        match = re.search(
            r"([\w./-]+\.(py|js|ts|tsx|jsx|json|md|html|css|sh|toml|ya?ml|go|rs|java|kt|rb))",
            user_input,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        return match.group(1).lstrip("./")

    def _detect_extensions(self, normalized: str, path: str | None) -> list[str]:
        mapping = {
            "python": ".py",
            "py": ".py",
            "javascript": ".js",
            "js": ".js",
            "typescript": ".ts",
            "ts": ".ts",
            "tsx": ".tsx",
            "jsx": ".jsx",
            "react": ".tsx",
            "html": ".html",
            "css": ".css",
            "json": ".json",
            "markdown": ".md",
            "md": ".md",
            "shell": ".sh",
            "bash": ".sh",
        }
        detected: list[str] = []
        if path:
            suffix = "." + path.split(".")[-1]
            if suffix not in detected:
                detected.append(suffix)
        for token, extension in mapping.items():
            if token in normalized and extension not in detected:
                detected.append(extension)
        return detected

    def _extract_target_name(self, normalized: str, path: str | None) -> str | None:
        if path:
            return path
        special_patterns = [
            (r"\btic[\s_-]*tac[\s_-]*toe\b", "tic tac toe"),
            (r"\btictactoe\b", "tic tac toe"),
            (r"\btick[\s_-]*tack[\s_-]*toe\b", "tic tac toe"),
            (r"\btodo[\s_-]*app\b", "todo app"),
            (r"\brest[\s_-]*api\b", "rest api"),
            (r"\bauth[\s_-]*middleware\b", "auth middleware"),
        ]
        for pattern, label in special_patterns:
            if re.search(pattern, normalized):
                return label
        match = re.search(r"(?:ein|eine|einen|a|an)\s+([a-z0-9 _-]{3,40})\s+(?:in|mit)\s+(python|javascript|typescript|react)\b", normalized)
        if match:
            return match.group(1).strip()
        return None

    def _looks_like_plan_request(self, normalized: str) -> bool:
        phrases = ("plan", "vorgehen", "roadmap", "naechsten schritte", "nächsten schritte", "wie wuerdest", "wie würdest")
        return any(phrase in normalized for phrase in phrases)

    def _looks_like_explanation_request(self, normalized: str) -> bool:
        phrases = (
            "erklaer",
            "erklär",
            "explain",
            "was macht",
            "wie funktioniert",
            "fasse",
            "zusammen",
            "review",
            "analysiere",
        )
        return any(phrase in normalized for phrase in phrases)

    def _looks_like_search_request(self, normalized: str) -> bool:
        phrases = (
            "wo ist",
            "wo steckt",
            "find",
            "such",
            "search",
            "locate",
            "zeige mir wo",
        )
        return any(phrase in normalized for phrase in phrases)

    def _looks_like_delete_request(self, normalized: str, path: str | None) -> bool:
        phrases = ("loesch", "lösch", "delete", "remove", "entfern", "weg", "raeum", "räum")
        return any(phrase in normalized for phrase in phrases) and path is not None

    def _looks_like_update_request(
        self,
        normalized: str,
        path: str | None,
        *,
        session: SessionState | None = None,
    ) -> bool:
        phrases = (
            "update",
            "aktualis",
            "aender",
            "änder",
            "anpass",
            "modify",
            "refactor",
            "fix",
            "beheb",
            "reparier",
            "verbesser",
        )
        if any(phrase in normalized for phrase in phrases):
            return path is not None or bool(self._follow_up_target_paths(session))
        follow_up_change_phrases = (
            "statt",
            "instead",
            "gegen einen computer",
            "gegen den computer",
            "gegen eine ki",
            "gegen einen bot",
            "singleplayer",
            "single player",
            "2 spieler",
            "zwei spieler",
            "nicht mit 2 spielern",
            "nicht mit zwei spielern",
        )
        if any(phrase in normalized for phrase in follow_up_change_phrases):
            return bool(self._follow_up_target_paths(session))
        if any(token in normalized for token in ("mach das anders", "mach es anders", "change that", "do it differently")):
            return bool(self._follow_up_target_paths(session))
        return False

    def _looks_like_create_request(
        self,
        normalized: str,
        path: str | None,
        *,
        session: SessionState | None = None,
    ) -> bool:
        del session
        create_phrases = (
            "programmier",
            "programmiere",
            "schreib",
            "schreibe",
            "bau mir",
            "erstell",
            "create",
            "implement",
            "generate",
            "ich moechte",
            "ich möchte",
            "ich will",
            "i want",
        )
        artifact_phrases = (
            "spiel",
            "game",
            "app",
            "api",
            "script",
            "skript",
            "tool",
            "service",
            "helper",
            "modul",
            "module",
            "datei",
            "file",
        )
        has_create_signal = any(phrase in normalized for phrase in create_phrases)
        has_artifact_signal = any(phrase in normalized for phrase in artifact_phrases) or bool(self._detect_extensions(normalized, path))
        if path is not None:
            return has_create_signal
        return has_create_signal and has_artifact_signal

    def _looks_like_issue_follow_up(
        self,
        normalized: str,
        *,
        session: SessionState | None = None,
    ) -> bool:
        if not self._has_follow_up_context(session):
            return False
        markers = (
            "fehler",
            "error",
            "bug",
            "buggy",
            "kaputt",
            "komisch",
            "wirkt komisch",
            "stimmt da nicht",
            "geht nicht",
            "broken",
            "weird",
            "doesn't work",
            "doesnt work",
            "does not work",
            "issue",
            "problem",
            "terminal",
            "konsole",
            "traceback",
            "stacktrace",
            "exception",
            "crash",
            "warnung",
            "warning",
        )
        return any(marker in normalized for marker in markers)

    def _follow_up_debug_route(
        self,
        user_input: str,
        *,
        normalized: str,
        path: str | None,
        session: SessionState | None,
        target_name: str | None,
        search_terms: list[str],
        relevant_extensions: list[str],
    ) -> RouterOutput | None:
        if not self._looks_like_issue_follow_up(normalized, session=session):
            return None
        target_paths = [path] if path else self._follow_up_target_paths(session)
        debug_extensions = list(relevant_extensions)
        if not debug_extensions:
            for target_path in target_paths:
                suffix = "." + target_path.split(".")[-1] if "." in target_path else ""
                if suffix and suffix not in debug_extensions:
                    debug_extensions.append(suffix)
        return self._build_route(
            user_goal=user_input,
            intent=RouteIntent.DEBUG,
            requested_outcome="Diagnose the reported issue in the current task, then apply a focused fix if the evidence supports it.",
            action_plan=[
                (RouteActionName.READ_RELEVANT_FILES, "The active implementation should be inspected before diagnosing."),
                (RouteActionName.DIAGNOSE_ISSUE, "The vague follow-up should be translated into concrete diagnostics before editing."),
                (RouteActionName.UPDATE_ARTIFACT, "If diagnostics reveal the source, apply a focused fix to the active artifact."),
                (RouteActionName.RUN_VALIDATION, "Rerun the most relevant validation after the fix."),
                (RouteActionName.SUMMARIZE_RESULT, "Report what was diagnosed, changed, and verified."),
            ],
            repo_context_needed=True,
            target_name=target_name or (target_paths[0] if target_paths else None),
            target_paths=target_paths,
            search_terms=search_terms or self._fallback_search_terms(normalized) or target_paths[:2],
            relevant_extensions=debug_extensions,
            confidence=0.82,
        )

    def _follow_up_update_route(
        self,
        user_input: str,
        *,
        normalized: str,
        path: str | None,
        session: SessionState | None,
        target_name: str | None,
        search_terms: list[str],
        relevant_extensions: list[str],
    ) -> RouterOutput | None:
        if path is not None:
            return None
        target_paths = self._follow_up_target_paths(session)
        if not target_paths:
            return None
        if self._has_explicit_new_artifact_request(normalized):
            return None
        return self._build_route(
            user_goal=user_input,
            intent=RouteIntent.UPDATE,
            requested_outcome="Update the most relevant previously touched artifact.",
            action_plan=[
                (RouteActionName.READ_RELEVANT_FILES, "The current implementation should be inspected before editing."),
                (RouteActionName.UPDATE_ARTIFACT, "The follow-up request most likely changes the previously touched artifact."),
                (RouteActionName.RUN_VALIDATION, "The edited code should be validated before finishing."),
                (RouteActionName.SUMMARIZE_RESULT, "Return the update outcome."),
            ],
            repo_context_needed=True,
            target_name=target_name or target_paths[0],
            target_paths=target_paths,
            search_terms=search_terms or self._fallback_search_terms(normalized) or [target_paths[0]],
            relevant_extensions=relevant_extensions,
            confidence=0.68,
        )

    def _has_explicit_new_artifact_request(self, normalized: str) -> bool:
        create_markers = (
            "neu",
            "new",
            "another",
            "zweite",
            "zweiten",
            "second",
            "extra",
            "zusatz",
            "zusätzlich",
            "additional",
            "separate",
            "separat",
        )
        artifact_markers = (
            "datei",
            "file",
            "script",
            "skript",
            "app",
            "tool",
            "service",
            "helper",
            "modul",
            "module",
            "komponente",
            "component",
        )
        return any(marker in normalized for marker in create_markers) and any(
            marker in normalized for marker in artifact_markers
        )

    def _has_follow_up_context(self, session: SessionState | None) -> bool:
        if session is None:
            return False
        if session.changed_files or session.candidate_files:
            return True
        follow_up = session.follow_up_context
        if follow_up is None:
            return False
        return bool(
            follow_up.target_paths
            or follow_up.changed_files
            or follow_up.read_files
            or follow_up.validation_runs
            or follow_up.diagnostics
            or follow_up.last_error
        )

    def _follow_up_target_paths(self, session: SessionState | None) -> list[str]:
        if session is None:
            return []
        paths = [item.path for item in session.changed_files[-4:] if getattr(item, "path", None)]
        follow_up = session.follow_up_context
        if follow_up is not None:
            paths.extend(follow_up.target_paths[:6])
            paths.extend(follow_up.changed_files[:6])
            paths.extend(follow_up.read_files[:6])
        if paths:
            return self._unique_paths(paths)[:6]
        return self._unique_paths(session.candidate_files[:6])

    def _unique_paths(self, values: list[str]) -> list[str]:
        unique: list[str] = []
        for raw in values:
            text = str(raw or "").strip()
            if not text or text in unique:
                continue
            unique.append(text)
        return unique

    def _fallback_search_terms(self, normalized: str) -> list[str]:
        stopwords = {
            "bitte",
            "mir",
            "ein",
            "eine",
            "einen",
            "das",
            "der",
            "die",
            "und",
            "oder",
            "ich",
            "moechte",
            "möchte",
            "will",
            "haben",
            "kann",
            "kannst",
            "du",
            "in",
            "mit",
            "fuer",
            "für",
        }
        terms: list[str] = []
        for raw in re.split(r"[^a-z0-9]+", normalized):
            token = raw.strip()
            if len(token) < 3 or token in stopwords or token in terms:
                continue
            terms.append(token)
        return terms[:4]

    def _append_runtime_execution(self, session: SessionState | None, record: dict[str, Any]) -> None:
        if session is None:
            return
        session.runtime_executions.append(record)
        session.runtime_executions = session.runtime_executions[-20:]

    def _primary_model_name(self) -> str | None:
        candidate = str(self.model_name or "").strip()
        if candidate:
            return candidate
        config = getattr(self.llm, "config", None)
        if config is None:
            return None
        return str(getattr(config, "router_model_name", "") or getattr(config, "model_name", "") or "").strip() or None

    def _log(self, event: str, **payload: Any) -> None:
        if self.logger is None:
            return
        self.logger.log_event(event, **payload)

    def _progress_logger(self, event: str):
        if self.logger is None:
            return None

        def callback(payload: dict[str, object]) -> None:
            self._log(event, **payload)

        return callback
