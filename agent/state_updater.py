from __future__ import annotations

import time
from typing import Any

from agent.prompts import task_state_system_prompt, task_state_update_prompt
from agent.semantic_guardrails import build_minimal_task_state
from agent.semantic_runtime import (
    annotate_semantic_record,
    secondary_semantics_limited,
    semantic_model_candidates,
    semantic_resolution_from_attempt,
)
from agent.task_state import TaskState
from llm.provider import LLMProvider
from llm.runtime_resilience import (
    ExecutionRecoveryPolicy,
    build_execution_run_record,
    estimate_context_pressure,
    invoke_model,
)
from runtime.logger import AgentLogger


class TaskStateUpdater:
    """Updates the central working task state from the latest turn and prior session context."""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        logger: AgentLogger | None = None,
        model_name: str | None = None,
        timeout: int = 20,
        num_ctx: int = 4096,
    ):
        self.llm = llm
        self.logger = logger
        self.model_name = model_name
        self.timeout = timeout
        self.num_ctx = num_ctx

    def update_task_state(
        self,
        user_input: str,
        *,
        snapshot=None,
        session=None,
    ) -> TaskState:
        payload: dict[str, Any] | None = None
        initial_mode = self._initial_prompt_mode(session)
        prompt = task_state_update_prompt(
            user_input,
            snapshot=snapshot,
            session=session,
            mode=initial_mode,
        )
        initial_timeout = self.timeout if initial_mode == "full" else max(12, min(self.timeout, 18))
        initial_num_ctx = self.num_ctx if initial_mode == "full" else min(self.num_ctx, 2048)
        context_pressure = estimate_context_pressure(prompt_chars=len(prompt))
        model_candidates = self._model_candidates()
        primary_model = model_candidates[0] if model_candidates else None
        reserve_model = model_candidates[1] if len(model_candidates) > 1 else None
        policy = ExecutionRecoveryPolicy(
            task_class="task_state_generation",
            allow_same_backend_retry=True,
            allow_smaller_faster_model=bool(reserve_model),
            allow_reduce_request_complexity=True,
            allow_minimal_generation=True,
            allow_deterministic_fallback=True,
            max_same_backend_retries=1,
            max_total_attempts=4,
        )
        attempts = []
        outcome = invoke_model(
            lambda progress: self.llm.generate_json(
                prompt,
                system=task_state_system_prompt(),
                model=primary_model,
                retries=0,
                timeout=initial_timeout,
                num_ctx=initial_num_ctx,
                progress_callback=progress,
            ),
            operation_name="task_state_generation",
            task_class="task_state_generation",
            attempt_number=1,
            capability_tier="tier_a",
            recovery_strategy="primary_model_generation",
            prompt_variant=initial_mode,
            model_identifier=primary_model,
            backend_identifier=self._backend_identifier(),
            inactivity_timeout_seconds=initial_timeout,
            total_timeout_seconds=max(initial_timeout * 2, initial_timeout + 20),
            context_pressure_estimate=context_pressure,
            event_callback=self._progress_logger("task_state_generation_progress"),
        )
        attempts.append(outcome.attempt)
        if outcome.exception is None:
            payload = outcome.value
            state = self._finalize_state(TaskState.model_validate(payload), semantic_resolution="full_model")
            self._append_runtime_execution(
                session,
                annotate_semantic_record(
                    build_execution_run_record(
                        operation_name="task_state_generation",
                        task_class="task_state_generation",
                        final_state="completed",
                        capability_tier="tier_a",
                        recovery_strategy="primary_model_generation",
                        degraded=False,
                        honest_blocked=False,
                        artifact_bytes_generated=0,
                        validation_possible=False,
                        summary="Task understanding completed on the primary semantic model.",
                        attempts=attempts,
                    ),
                    semantic_resolution="full_model",
                ),
            )
            self._log("task_state_updated", task_state=state.model_dump())
            return state

        failure = outcome.attempt.failure
        self._log(
            "task_state_generation_error",
            error=str(outcome.exception),
            failure=failure.to_dict() if failure is not None else None,
        )
        decisions = policy.plan_recovery(
            failure,
            primary_model=primary_model,
            faster_model=reserve_model,
            history=attempts,
        ) if failure is not None else []
        for decision in decisions:
            self._log(
                "task_state_recovery_option",
                strategy=decision.candidate.strategy,
                capability_tier=decision.candidate.capability_tier,
                prompt_variant=decision.candidate.prompt_variant,
                model=decision.candidate.model_identifier,
                accepted=decision.accepted,
                reason=decision.reason,
            )
            if not decision.accepted:
                continue
            if decision.candidate.local_only:
                break
            retry_prompt = task_state_update_prompt(
                user_input,
                snapshot=snapshot,
                session=session,
                mode="full" if decision.candidate.prompt_variant == "full" else "compact",
            )
            retry_timeout = self.timeout if decision.candidate.prompt_variant == "full" else max(12, min(self.timeout, 18))
            retry_num_ctx = self.num_ctx if decision.candidate.prompt_variant == "full" else min(self.num_ctx, 2048)
            retry_outcome = invoke_model(
                lambda progress, prompt_text=retry_prompt, model_name=decision.candidate.model_identifier: self.llm.generate_json(
                    prompt_text,
                    system=task_state_system_prompt(),
                    model=model_name,
                    retries=0,
                    timeout=retry_timeout,
                    num_ctx=retry_num_ctx,
                    progress_callback=progress,
                ),
                operation_name="task_state_generation",
                task_class="task_state_generation",
                attempt_number=len(attempts) + 1,
                capability_tier=decision.candidate.capability_tier,
                recovery_strategy=decision.candidate.strategy,
                prompt_variant=decision.candidate.prompt_variant,
                model_identifier=decision.candidate.model_identifier,
                backend_identifier=self._backend_identifier(),
                inactivity_timeout_seconds=retry_timeout,
                total_timeout_seconds=max(retry_timeout * 2, retry_timeout + 20),
                context_pressure_estimate=context_pressure,
                event_callback=self._progress_logger("task_state_generation_progress"),
            )
            attempts.append(retry_outcome.attempt)
            if retry_outcome.exception is None:
                payload = retry_outcome.value
                resolution = semantic_resolution_from_attempt(
                    capability_tier=decision.candidate.capability_tier,
                    prompt_variant=decision.candidate.prompt_variant,
                    model_identifier=decision.candidate.model_identifier,
                    primary_model=primary_model,
                )
                state = self._finalize_state(TaskState.model_validate(payload), semantic_resolution=resolution)
                self._append_runtime_execution(
                    session,
                    annotate_semantic_record(
                        build_execution_run_record(
                            operation_name="task_state_generation",
                            task_class="task_state_generation",
                            final_state="completed",
                            capability_tier=decision.candidate.capability_tier,
                            recovery_strategy=decision.candidate.strategy,
                            degraded=decision.candidate.capability_tier != "tier_a",
                            honest_blocked=False,
                            artifact_bytes_generated=0,
                            validation_possible=False,
                            summary=(
                                "Task understanding recovered on a reserve semantic model."
                                if resolution == "reserve_model"
                                else "Task understanding recovered through reduced semantic reconstruction."
                            ),
                            attempts=attempts,
                        ),
                        semantic_resolution=resolution,
                    ),
                )
                self._log("task_state_updated", task_state=state.model_dump(), source="recovered_model")
                return state
            self._log(
                "task_state_generation_retry_error",
                error=str(retry_outcome.exception),
                failure=retry_outcome.attempt.failure.to_dict()
                if retry_outcome.attempt.failure is not None
                else None,
            )

        self._log("task_state_fallback", error=str(outcome.exception), payload=payload or {})
        state = self._fallback_state(user_input, snapshot=snapshot, session=session)
        self._append_runtime_execution(
            session,
            annotate_semantic_record(
                build_execution_run_record(
                    operation_name="task_state_generation",
                    task_class="task_state_generation",
                    final_state="degraded_success",
                    capability_tier="tier_d",
                    recovery_strategy="deterministic_fallback",
                    degraded=True,
                    honest_blocked=False,
                    artifact_bytes_generated=0,
                    validation_possible=False,
                    summary="Task understanding fell back to conservative minimal inference after model execution failed.",
                    attempts=attempts,
                ),
                semantic_resolution="minimal_inference",
            ),
        )
        self._log("task_state_updated", task_state=state.model_dump(), source="fallback")
        return state

    def _initial_prompt_mode(self, session) -> str:
        if session is None:
            return "compact"
        if session.task_state is not None:
            return "full"
        if session.follow_up_context is not None:
            return "full"
        if session.tool_calls or session.diagnostics or session.changed_files or session.validation_runs:
            return "full"
        if session.messages:
            if len(session.messages) == 1 and session.messages[0].role == "user":
                return "compact"
            return "full"
        return "compact"

    def _fallback_state(self, user_input: str, *, snapshot=None, session=None) -> TaskState:
        return build_minimal_task_state(
            user_input,
            session=session,
            snapshot=snapshot,
            semantic_resolution="minimal_inference",
        )

    def _log(self, event: str, **payload: Any) -> None:
        if self.logger is None:
            return
        self.logger.log_event(event, **payload)

    def _append_runtime_execution(self, session, record: dict[str, Any]) -> None:
        if session is None:
            return
        session.runtime_executions.append(record)
        session.runtime_executions = session.runtime_executions[-20:]

    def _finalize_state(self, state: TaskState, *, semantic_resolution: str) -> TaskState:
        state.semantic_resolution = semantic_resolution
        state.secondary_semantics_limited = secondary_semantics_limited(semantic_resolution)
        state.semantic_inference_mode = "conservative" if semantic_resolution == "minimal_inference" else "full"
        return state

    def _model_candidates(self) -> list[str]:
        return semantic_model_candidates(self.model_name, getattr(self.llm, "config", None))

    def _backend_identifier(self) -> str:
        return "ollama"

    def _progress_logger(self, event: str):
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
            self._log(event, **payload)

        return callback
