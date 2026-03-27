from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from agent.models import SessionState, WorkspaceSnapshot
from agent.prompts import router_prompt, router_repair_prompt, router_system_prompt
from agent.semantic_guardrails import build_minimal_router_output
from agent.semantic_runtime import (
    annotate_semantic_record,
    semantic_model_candidates,
    semantic_resolution_from_attempt,
)
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
        trimmed = str(user_input or "").strip()
        direct_response = self._fallback_direct_response(trimmed)
        if direct_response is not None:
            route = RouterOutput(
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
            self._log("router_fast_path", router_result=route.model_dump(), source="direct_response_only")
            return route
        payload: dict[str, Any] | None = None
        prompt = router_prompt(user_input, snapshot, session=session, mode="full")
        context_pressure = estimate_context_pressure(prompt_chars=len(prompt))
        model_candidates = self._model_candidates()
        primary_model = model_candidates[0] if model_candidates else None
        reserve_model = model_candidates[1] if len(model_candidates) > 1 else None
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
                    annotate_semantic_record(
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
                            summary="Route generation completed on the primary semantic model.",
                            attempts=attempts,
                        ),
                        semantic_resolution="full_model",
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
            failure = outcome.attempt.failure
            policy = ExecutionRecoveryPolicy(
                task_class="router_generation",
                allow_same_backend_retry=True,
                allow_smaller_faster_model=bool(reserve_model),
                allow_reduce_request_complexity=True,
                allow_minimal_generation=True,
                allow_deterministic_fallback=True,
                max_same_backend_retries=1,
                max_total_attempts=4,
            )
            decisions = policy.plan_recovery(
                failure,
                primary_model=primary_model,
                faster_model=reserve_model,
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
                    snapshot=snapshot,
                    session=session,
                    strategy=decision.candidate.strategy,
                    capability_tier=decision.candidate.capability_tier,
                    model_name=decision.candidate.model_identifier,
                    prompt_variant=decision.candidate.prompt_variant,
                )
                if retried is not None:
                    route, retry_attempts = retried
                    attempts.extend(retry_attempts)
                    resolution = semantic_resolution_from_attempt(
                        capability_tier=decision.candidate.capability_tier,
                        prompt_variant=decision.candidate.prompt_variant,
                        model_identifier=decision.candidate.model_identifier,
                        primary_model=primary_model,
                    )
                    self._append_runtime_execution(
                        session,
                        annotate_semantic_record(
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
                                summary=(
                                    "Route generation recovered on a reserve semantic model."
                                    if resolution == "reserve_model"
                                    else "Route generation recovered through reduced semantic reconstruction."
                                ),
                                attempts=attempts,
                            ),
                            semantic_resolution=resolution,
                        ),
                    )
                    return route
        fallback = self._fallback_route(user_input, snapshot=snapshot, session=session)
        self._append_runtime_execution(
            session,
            annotate_semantic_record(
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
                    summary="The router fell back to conservative minimal inference.",
                    attempts=attempts,
                ),
                semantic_resolution="minimal_inference",
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
                annotate_semantic_record(
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
                        summary="The router repaired invalid structured output with a constrained semantic retry.",
                        attempts=[outcome.attempt],
                    ),
                    semantic_resolution="reduced_model",
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
        snapshot: WorkspaceSnapshot | None = None,
        session: SessionState | None = None,
        strategy: str = "minimal_prompt_after_timeout",
        capability_tier: str = "tier_c",
        model_name: str | None = None,
        prompt_variant: str = "compact",
    ) -> tuple[RouterOutput, list[Any]] | None:
        retry_timeout = max(self.timeout + 8, 26)
        retry_num_ctx = self.num_ctx if prompt_variant == "full" else min(self.num_ctx, 2048)
        self._log(
            "router_retry_started",
            strategy=strategy,
            retry_timeout=retry_timeout,
            retry_num_ctx=retry_num_ctx,
        )
        try:
            retry_model = model_name or self._primary_model_name()
            retry_prompt = router_prompt(
                user_input,
                snapshot,
                session=session if prompt_variant == "full" else session,
                mode="full" if prompt_variant == "full" else "compact",
            )
            outcome = invoke_model(
                lambda progress: self.llm.generate_json(
                    retry_prompt,
                    system=router_system_prompt(),
                    model=retry_model,
                    retries=0,
                    timeout=retry_timeout,
                    num_ctx=retry_num_ctx,
                    progress_callback=progress,
                ),
                operation_name="router_generation",
                task_class="router_generation",
                attempt_number=1,
                capability_tier=capability_tier,
                recovery_strategy=strategy,
                prompt_variant=prompt_variant,
                model_identifier=retry_model,
                backend_identifier="ollama",
                inactivity_timeout_seconds=retry_timeout,
                total_timeout_seconds=max(retry_timeout + 22, retry_timeout * 2),
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
        return build_minimal_router_output(
            trimmed,
            session=session,
            snapshot=snapshot,
            logger=self.logger,
            semantic_resolution="minimal_inference",
        )

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
        normalized_padded = f" {normalized} "
        if any(
            normalized == fragment
            or f" {fragment} " in normalized_padded
            for fragment in intro_fragments
        ):
            return (
                "Ich bin dein lokaler Coding-Agent fuer diesen Workspace.\n\n"
                "Ich kann Code analysieren, Aenderungen planen und auf Basis des validierten Router-Outputs ausfuehren."
            )
        return None

    def _append_runtime_execution(self, session: SessionState | None, record: dict[str, Any]) -> None:
        if session is None:
            return
        session.runtime_executions.append(record)
        session.runtime_executions = session.runtime_executions[-20:]

    def _primary_model_name(self) -> str | None:
        candidates = self._model_candidates()
        return candidates[0] if candidates else None

    def _model_candidates(self) -> list[str]:
        return semantic_model_candidates(self.model_name, getattr(self.llm, "config", None))

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
