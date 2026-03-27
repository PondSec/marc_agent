from __future__ import annotations

from typing import Any

from agent.prompts import task_understanding_prompt, task_understanding_system_prompt
from agent.task_schema import TaskArtifact, TaskPlanStep, TaskUnderstanding
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
        primary_model = self._primary_model_name()
        policy = ExecutionRecoveryPolicy(
            task_class="task_understanding",
            allow_same_backend_retry=True,
            allow_deterministic_fallback=True,
            max_same_backend_retries=1,
            max_total_attempts=2,
        )
        attempts = []
        outcome = invoke_model(
            lambda progress: self.llm.generate_json(
                prompt,
                system=task_understanding_system_prompt(),
                model=primary_model,
                retries=0,
                timeout=self.timeout,
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
            total_timeout_seconds=max(self.timeout * 2, self.timeout + 20),
            context_pressure_estimate=context_pressure,
            event_callback=self._progress_logger("task_understanding_generation_progress"),
        )
        attempts.append(outcome.attempt)
        if outcome.exception is None:
            payload = outcome.value
            understanding = TaskUnderstanding.model_validate(payload)
            self._append_runtime_execution(
                session,
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
                    summary="Task interpretation completed on the primary generation tier.",
                    attempts=attempts,
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
            faster_model=None,
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
            retry_outcome = invoke_model(
                lambda progress: self.llm.generate_json(
                    prompt,
                    system=task_understanding_system_prompt(),
                    model=decision.candidate.model_identifier,
                    retries=0,
                    timeout=self.timeout,
                    num_ctx=min(self.num_ctx, 2048),
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
                inactivity_timeout_seconds=self.timeout,
                total_timeout_seconds=max(self.timeout * 2, self.timeout + 20),
                context_pressure_estimate=context_pressure,
                event_callback=self._progress_logger("task_understanding_generation_progress"),
            )
            attempts.append(retry_outcome.attempt)
            if retry_outcome.exception is None:
                payload = retry_outcome.value
                understanding = TaskUnderstanding.model_validate(payload)
                self._append_runtime_execution(
                    session,
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
                        summary="Task interpretation recovered after a runtime startup issue.",
                        attempts=attempts,
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
        fallback = self._fallback_understanding(user_input, session=session)
        self._append_runtime_execution(
            session,
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
                summary="Task interpretation used the deterministic fallback because the backend could not start cleanly.",
                attempts=attempts,
            ),
        )
        self._log("task_understanding", understanding=fallback.model_dump(), source="fallback")
        return fallback

    def _fallback_understanding(self, user_input: str, *, session=None) -> TaskUnderstanding:
        request = str(user_input or "").strip() or "Unclear request"
        recent_paths: list[str] = []
        if session is not None:
            if getattr(session, "follow_up_context", None) is not None:
                recent_paths.extend(session.follow_up_context.target_paths[:4])
            recent_paths.extend(session.candidate_files[:4])
        artifacts = [
            TaskArtifact(path=path, name=path.split("/")[-1], kind="file", role="active_context", confidence=0.55)
            for path in recent_paths[:4]
        ]
        plan = [
            TaskPlanStep(step=1, summary="Inspect the most relevant context before acting.", action_hint="inspect"),
            TaskPlanStep(step=2, summary="Choose the smallest safe implementation step.", action_hint="plan"),
        ]
        return TaskUnderstanding(
            original_request=request,
            interpreted_goal=request,
            intent_category="unknown",
            conversation_relation="same_task_follow_up" if session is not None else "unknown",
            subgoals=["Clarify the immediate task and inspect context."],
            target_artifacts=artifacts,
            relevant_context=["Recent session context is available."] if session is not None else [],
            constraints=[],
            missing_info=["The exact target is still ambiguous."],
            assumptions=["Continue from the current task unless the user clearly changed topic."],
            user_observations=[],
            supplied_evidence=[],
            ambiguity_level="high",
            risk_level="medium",
            confidence=0.35,
            recommended_mode="clarify",
            execution_plan=plan,
            needs_clarification=True,
            clarification_questions=["Welchen konkreten Bereich soll ich als naechstes bearbeiten?"],
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

    def _primary_model_name(self) -> str | None:
        candidate = str(self.model_name or "").strip()
        if candidate:
            return candidate
        config = getattr(self.llm, "config", None)
        if config is None:
            return None
        return str(getattr(config, "router_model_name", "") or getattr(config, "model_name", "") or "").strip() or None

    def _progress_logger(self, event: str):
        if self.logger is None:
            return None

        def callback(payload: dict[str, object]) -> None:
            self._log(event, **payload)

        return callback
