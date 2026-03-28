from __future__ import annotations

from typing import Any

from agent.prompts import task_understanding_prompt, task_understanding_system_prompt
from agent.semantic_guardrails import build_minimal_task_understanding
from agent.semantic_runtime import (
    annotate_semantic_record,
    secondary_semantics_limited,
    semantic_model_candidates,
    semantic_resolution_from_attempt,
)
from agent.task_schema import TaskUnderstanding
from llm.provider import LLMProvider
from llm.runtime_resilience import (
    ExecutionRecoveryPolicy,
    build_execution_run_record,
    estimate_context_pressure,
    invoke_model,
)
from runtime.logger import AgentLogger


class TaskInterpreter:
    """Turns raw natural language into a normalized task object before execution planning."""

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

    def interpret(
        self,
        user_input: str,
        *,
        snapshot=None,
        session=None,
    ) -> TaskUnderstanding:
        payload: dict[str, Any] | None = None
        prompt = task_understanding_prompt(user_input, snapshot=snapshot, session=session)
        context_pressure = estimate_context_pressure(prompt_chars=len(prompt))
        initial_total_timeout = max(self.timeout * 2, self.timeout + 20)
        model_candidates = self._model_candidates()
        primary_model = model_candidates[0] if model_candidates else None
        reserve_model = model_candidates[1] if len(model_candidates) > 1 else None
        policy = ExecutionRecoveryPolicy(
            task_class="task_understanding",
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
                system=task_understanding_system_prompt(),
                model=primary_model,
                retries=0,
                timeout=self.timeout,
                total_timeout=initial_total_timeout,
                num_ctx=self.num_ctx,
                progress_callback=progress,
            ),
            operation_name="task_understanding",
            task_class="task_understanding",
            attempt_number=1,
            capability_tier="tier_a",
            recovery_strategy="primary_model_generation",
            prompt_variant="full",
            model_identifier=primary_model,
            backend_identifier="ollama",
            inactivity_timeout_seconds=self.timeout,
            total_timeout_seconds=initial_total_timeout,
            context_pressure_estimate=context_pressure,
            event_callback=self._progress_logger("task_understanding_generation_progress"),
        )
        attempts.append(outcome.attempt)
        if outcome.exception is None:
            payload = outcome.value
            understanding = self._finalize_understanding(
                TaskUnderstanding.model_validate(payload),
                semantic_resolution="full_model",
            )
            self._append_runtime_execution(
                session,
                annotate_semantic_record(
                    build_execution_run_record(
                        operation_name="task_understanding",
                        task_class="task_understanding",
                        final_state="completed",
                        capability_tier="tier_a",
                        recovery_strategy="primary_model_generation",
                        degraded=False,
                        honest_blocked=False,
                        artifact_bytes_generated=0,
                        validation_possible=False,
                        summary="Task interpretation completed on the primary semantic model.",
                        attempts=attempts,
                    ),
                    semantic_resolution="full_model",
                ),
            )
            self._log("task_understanding", understanding=understanding.model_dump())
            return understanding

        failure = outcome.attempt.failure
        self._log(
            "task_understanding_generation_error",
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
                "task_understanding_recovery_option",
                strategy=decision.candidate.strategy,
                capability_tier=decision.candidate.capability_tier,
                accepted=decision.accepted,
                reason=decision.reason,
            )
            if not decision.accepted:
                continue
            if decision.candidate.local_only:
                break
            retry_prompt = task_understanding_prompt(
                user_input,
                snapshot=snapshot,
                session=session,
                mode="full" if decision.candidate.prompt_variant == "full" else "compact",
            )
            retry_timeout = self.timeout if decision.candidate.prompt_variant == "full" else max(12, min(self.timeout, 18))
            retry_total_timeout = max(retry_timeout * 2, retry_timeout + 20)
            retry_num_ctx = self.num_ctx if decision.candidate.prompt_variant == "full" else min(self.num_ctx, 2048)
            retry_outcome = invoke_model(
                lambda progress, prompt_text=retry_prompt, model_name=decision.candidate.model_identifier: self.llm.generate_json(
                    prompt_text,
                    system=task_understanding_system_prompt(),
                    model=model_name,
                    retries=0,
                    timeout=retry_timeout,
                    total_timeout=retry_total_timeout,
                    num_ctx=retry_num_ctx,
                    progress_callback=progress,
                ),
                operation_name="task_understanding",
                task_class="task_understanding",
                attempt_number=len(attempts) + 1,
                capability_tier=decision.candidate.capability_tier,
                recovery_strategy=decision.candidate.strategy,
                prompt_variant=decision.candidate.prompt_variant,
                model_identifier=decision.candidate.model_identifier,
                backend_identifier="ollama",
                inactivity_timeout_seconds=retry_timeout,
                total_timeout_seconds=retry_total_timeout,
                context_pressure_estimate=context_pressure,
                event_callback=self._progress_logger("task_understanding_generation_progress"),
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
                understanding = self._finalize_understanding(
                    TaskUnderstanding.model_validate(payload),
                    semantic_resolution=resolution,
                )
                self._append_runtime_execution(
                    session,
                    annotate_semantic_record(
                        build_execution_run_record(
                            operation_name="task_understanding",
                            task_class="task_understanding",
                            final_state="completed",
                            capability_tier=decision.candidate.capability_tier,
                            recovery_strategy=decision.candidate.strategy,
                            degraded=decision.candidate.capability_tier != "tier_a",
                            honest_blocked=False,
                            artifact_bytes_generated=0,
                            validation_possible=False,
                            summary=(
                                "Task interpretation recovered on a reserve semantic model."
                                if resolution == "reserve_model"
                                else "Task interpretation recovered through reduced semantic reconstruction."
                            ),
                            attempts=attempts,
                        ),
                        semantic_resolution=resolution,
                    ),
                )
                self._log("task_understanding", understanding=understanding.model_dump(), source="recovered_model")
                return understanding
            self._log(
                "task_understanding_generation_retry_error",
                error=str(retry_outcome.exception),
                failure=retry_outcome.attempt.failure.to_dict()
                if retry_outcome.attempt.failure is not None
                else None,
            )

        self._log(
            "task_understanding_fallback",
            error=str(outcome.exception),
            payload=payload or {},
        )
        fallback = self._fallback_understanding(user_input, snapshot=snapshot, session=session)
        self._append_runtime_execution(
            session,
            annotate_semantic_record(
                build_execution_run_record(
                    operation_name="task_understanding",
                    task_class="task_understanding",
                    final_state="degraded_success",
                    capability_tier="tier_d",
                    recovery_strategy="deterministic_fallback",
                    degraded=True,
                    honest_blocked=False,
                    artifact_bytes_generated=0,
                    validation_possible=False,
                    summary="Task interpretation fell back to conservative minimal inference after model execution failed.",
                    attempts=attempts,
                ),
                semantic_resolution="minimal_inference",
            ),
        )
        self._log("task_understanding", understanding=fallback.model_dump(), source="fallback")
        return fallback

    def _fallback_understanding(self, user_input: str, *, snapshot=None, session=None) -> TaskUnderstanding:
        return build_minimal_task_understanding(
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

    def _finalize_understanding(
        self,
        understanding: TaskUnderstanding,
        *,
        semantic_resolution: str,
    ) -> TaskUnderstanding:
        understanding.semantic_resolution = semantic_resolution
        understanding.secondary_semantics_limited = secondary_semantics_limited(semantic_resolution)
        return understanding

    def _model_candidates(self) -> list[str]:
        return semantic_model_candidates(self.model_name, getattr(self.llm, "config", None))

    def _progress_logger(self, event: str):
        if self.logger is None:
            return None

        def callback(payload: dict[str, object]) -> None:
            self._log(event, **payload)

        return callback
