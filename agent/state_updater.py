from __future__ import annotations

import time
from typing import Any

from pydantic import ValidationError

from agent.prompts import task_state_system_prompt, task_state_update_prompt
from agent.semantic_guardrails import build_minimal_task_state
from agent.semantic_runtime import (
    annotate_semantic_record,
    availability_recovery_model,
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


ANALYSIS_LIKE_INTENTS = {"explain", "inspect", "search", "plan", "validate"}
ANALYSIS_LIKE_ACTIONS = {"inspect", "search", "explain", "plan", "test"}
MUTATION_LIKE_INTENTS = {"implement", "repair", "correct"}
MUTATION_LIKE_ACTIONS = {"create", "modify", "debug"}


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
        a2_semantic_mode = self._agent_profile(session) == "a2"
        initial_mode = self._initial_prompt_mode(session)
        local_state = self._fallback_state(user_input, snapshot=snapshot, session=session)
        strict_semantic_execution = self._requires_semantic_model_execution(session, local_state)
        if self._should_short_circuit_with_local_state(
            initial_mode=initial_mode,
            session=session,
            state=local_state,
        ):
            self._log(
                "task_state_local_short_circuit",
                strategy="deterministic_fallback",
                reason="fresh_compact_shared_model_stack",
            )
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
                        summary="Task understanding used conservative local inference to avoid an extra same-model startup hop.",
                        attempts=[],
                    ),
                    semantic_resolution="minimal_inference",
                ),
            )
            self._log("task_state_updated", task_state=local_state.model_dump(), source="local_short_circuit")
            return local_state
        prompt = task_state_update_prompt(
            user_input,
            snapshot=snapshot,
            session=session,
            mode=initial_mode,
        )
        initial_timeout, initial_total_timeout, initial_num_ctx = self._mode_runtime(initial_mode)
        context_pressure = estimate_context_pressure(prompt_chars=len(prompt))
        model_candidates = self._model_candidates()
        primary_model = model_candidates[0] if model_candidates else None
        reserve_models = model_candidates[1:]
        policy = ExecutionRecoveryPolicy(
            task_class="task_state_generation",
            allow_same_backend_retry=True,
            allow_smaller_faster_model=bool(reserve_models),
            allow_reduce_request_complexity=True,
            allow_minimal_generation=True,
            allow_deterministic_fallback=not a2_semantic_mode and not strict_semantic_execution,
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
                total_timeout=initial_total_timeout,
                strict_timeouts=initial_mode != "full",
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
            total_timeout_seconds=initial_total_timeout,
            context_pressure_estimate=context_pressure,
            event_callback=self._progress_logger("task_state_generation_progress"),
        )
        attempts.append(outcome.attempt)
        if outcome.exception is None:
            payload = outcome.value
            try:
                state = TaskState.model_validate(payload)
                reconciled = False
                if not a2_semantic_mode:
                    reconciled = self._reconcile_with_local_state(state, local_state)
                if reconciled:
                    self._log(
                        "task_state_semantic_reconciled",
                        reason="prefer_local_mutation_semantics",
                        model_intent=payload.get("current_user_intent"),
                        model_strategy=payload.get("execution_strategy"),
                        local_intent=local_state.current_user_intent,
                        local_strategy=local_state.execution_strategy,
                    )
                state = self._finalize_state(state, semantic_resolution="full_model")
            except ValidationError as exc:
                self._log(
                    "task_state_validation_failed",
                    raw_task_state=payload or {},
                    errors=exc.errors(),
                )
            else:
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
        recovery_model = self._recovery_model_for_failure(
            primary_model=primary_model,
            reserve_models=reserve_models,
            failure=failure,
        )
        decisions = policy.plan_recovery(
            failure,
            primary_model=primary_model,
            faster_model=recovery_model,
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
            retry_mode = self._recovery_prompt_mode(
                initial_mode=initial_mode,
                candidate_prompt_variant=decision.candidate.prompt_variant,
                preserve_full_retry=strict_semantic_execution,
            )
            retry_prompt = task_state_update_prompt(
                user_input,
                snapshot=snapshot,
                session=session,
                mode=retry_mode,
            )
            retry_timeout, retry_total_timeout, retry_num_ctx = self._mode_runtime(retry_mode)
            retry_outcome = invoke_model(
                lambda progress, prompt_text=retry_prompt, model_name=decision.candidate.model_identifier: self.llm.generate_json(
                    prompt_text,
                    system=task_state_system_prompt(),
                    model=model_name,
                    retries=0,
                    timeout=retry_timeout,
                    total_timeout=retry_total_timeout,
                    strict_timeouts=retry_mode != "full",
                    num_ctx=retry_num_ctx,
                    progress_callback=progress,
                ),
                operation_name="task_state_generation",
                task_class="task_state_generation",
                attempt_number=len(attempts) + 1,
                capability_tier=decision.candidate.capability_tier,
                recovery_strategy=decision.candidate.strategy,
                prompt_variant=retry_mode,
                model_identifier=decision.candidate.model_identifier,
                backend_identifier=self._backend_identifier(),
                inactivity_timeout_seconds=retry_timeout,
                total_timeout_seconds=retry_total_timeout,
                context_pressure_estimate=context_pressure,
                event_callback=self._progress_logger("task_state_generation_progress"),
            )
            attempts.append(retry_outcome.attempt)
            if retry_outcome.exception is None:
                payload = retry_outcome.value
                resolution = semantic_resolution_from_attempt(
                    capability_tier=decision.candidate.capability_tier,
                    prompt_variant=retry_mode,
                    model_identifier=decision.candidate.model_identifier,
                    primary_model=primary_model,
                )
                try:
                    state = TaskState.model_validate(payload)
                    reconciled = False
                    if not a2_semantic_mode:
                        reconciled = self._reconcile_with_local_state(state, local_state)
                    if reconciled:
                        self._log(
                            "task_state_semantic_reconciled",
                            reason="prefer_local_mutation_semantics",
                            model_intent=payload.get("current_user_intent"),
                            model_strategy=payload.get("execution_strategy"),
                            local_intent=local_state.current_user_intent,
                            local_strategy=local_state.execution_strategy,
                        )
                    state = self._finalize_state(state, semantic_resolution=resolution)
                except ValidationError as exc:
                    self._log(
                        "task_state_validation_failed",
                        raw_task_state=payload or {},
                        errors=exc.errors(),
                    )
                else:
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
        if a2_semantic_mode or strict_semantic_execution:
            state = self._blocked_state_for_missing_semantics(user_input, local_state)
            self._append_runtime_execution(
                session,
                annotate_semantic_record(
                    build_execution_run_record(
                        operation_name="task_state_generation",
                        task_class="task_state_generation",
                        final_state="blocked",
                        capability_tier="tier_e",
                        recovery_strategy="semantic_model_required",
                        degraded=False,
                        honest_blocked=True,
                        artifact_bytes_generated=0,
                        validation_possible=False,
                        summary="A2 refused to continue without semantic model understanding after exhausting semantic recovery.",
                        attempts=attempts,
                    ),
                    semantic_resolution="blocked",
                ),
            )
            self._log("task_state_updated", task_state=state.model_dump(), source="semantic_blocked")
            return state
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

    def _reconcile_with_local_state(self, state: TaskState, local_state: TaskState) -> bool:
        if not self._should_prefer_local_mutation_semantics(state, local_state):
            return False

        local_targets = list(local_state.target_artifacts or [])
        merged_targets = local_targets + [
            artifact
            for artifact in state.target_artifacts
            if artifact not in local_targets
        ]
        local_active = list(local_state.active_artifacts or local_targets)
        merged_active = local_active + [
            artifact
            for artifact in state.active_artifacts
            if artifact not in local_active
        ]
        local_context = list(local_state.relevant_context or [])
        merged_context = local_context + [
            item for item in state.relevant_context
            if item not in local_context
        ]
        local_constraints = list(local_state.constraints or [])
        merged_constraints = local_constraints + [
            item for item in state.constraints
            if item not in local_constraints
        ]
        local_outline = list(local_state.execution_outline or [])
        merged_outline = local_outline + [
            item for item in state.execution_outline
            if item not in local_outline
        ]
        if state.goal_relation in {"validation_request", "clarify", "unknown"}:
            state.goal_relation = local_state.goal_relation
        state.current_user_intent = local_state.current_user_intent
        state.execution_strategy = local_state.execution_strategy
        state.next_action = local_state.next_action
        state.next_best_action = local_state.next_best_action
        state.target_artifacts = merged_targets
        state.active_artifacts = merged_active
        state.relevant_context = merged_context[:8]
        state.constraints = merged_constraints[:8]
        state.execution_outline = merged_outline[:6]
        state.output_expectation = local_state.output_expectation
        state.verification_target = local_state.verification_target
        if not state.assumptions:
            state.assumptions = list(local_state.assumptions or [])
        state.risk_level = local_state.risk_level
        state.ambiguity_level = local_state.ambiguity_level
        state.confidence = max(float(state.confidence or 0.0), float(local_state.confidence or 0.0))
        return True

    def _should_prefer_local_mutation_semantics(self, state: TaskState, local_state: TaskState) -> bool:
        if state.needs_clarification or local_state.needs_clarification:
            return False
        state_intent = str(state.current_user_intent or "").strip()
        state_strategy = str(state.execution_strategy or "").strip()
        state_action = str(state.next_best_action or state.next_action or "").strip()
        local_intent = str(local_state.current_user_intent or "").strip()
        local_strategy = str(local_state.execution_strategy or "").strip()
        local_action = str(local_state.next_best_action or local_state.next_action or "").strip()

        state_is_analysis = (
            state_strategy == "validation_inspection"
            or state_intent in ANALYSIS_LIKE_INTENTS
            or state_action in ANALYSIS_LIKE_ACTIONS
        )
        local_is_mutation = (
            local_strategy in {"feature_implementation", "debug_repair", "rollback_correction"}
            or local_intent in MUTATION_LIKE_INTENTS
            or local_action in MUTATION_LIKE_ACTIONS
        )
        if not state_is_analysis or not local_is_mutation:
            return False
        if float(local_state.confidence or 0.0) < 0.65:
            return False
        if not local_state.target_artifacts:
            return False
        has_local_verification = bool(local_state.verification_target)
        has_validation_artifact = any(
            str(artifact.role or "").strip() == "validation_target"
            for artifact in local_state.target_artifacts
        )
        return has_local_verification or has_validation_artifact

    def _should_short_circuit_with_local_state(
        self,
        *,
        initial_mode: str,
        session,
        state: TaskState,
    ) -> bool:
        if self._agent_profile(session) == "a2":
            return False
        if self._requires_semantic_model_execution(session, state):
            return False
        if initial_mode != "compact":
            return False
        config = getattr(self.llm, "config", None)
        if config is None:
            return False
        primary_model = str(getattr(config, "model_name", "") or "").strip()
        router_model = str(getattr(config, "router_model_name", "") or "").strip()
        if not primary_model or not router_model or primary_model != router_model:
            return False
        if state.needs_clarification or state.ambiguity_level == "high":
            return False
        return float(state.confidence or 0.0) >= 0.65

    def _agent_profile(self, session) -> str:
        if session is None:
            return ""
        runtime_options = getattr(session, "runtime_options", {}) or {}
        raw = str(runtime_options.get("agent_profile") or "").strip().lower()
        if raw in {"a1", "a2"}:
            return raw
        return ""

    def _requires_semantic_model_execution(self, session, state: TaskState) -> bool:
        if self._agent_profile(session) != "a2":
            return False
        if state.needs_clarification:
            return False
        strategy = str(state.execution_strategy or "").strip()
        intent = str(state.current_user_intent or "").strip()
        action = str(state.next_best_action or state.next_action or "").strip()
        if strategy in {"feature_implementation", "debug_repair", "refactor", "hardening", "rollback_correction"}:
            return True
        if intent in MUTATION_LIKE_INTENTS:
            return True
        return action in MUTATION_LIKE_ACTIONS

    def _blocked_state_for_missing_semantics(self, user_input: str, state: TaskState) -> TaskState:
        blocked = state.model_copy(deep=True)
        blocked.latest_user_turn = str(user_input or "").strip() or blocked.latest_user_turn
        blocked.root_goal = blocked.latest_user_turn or blocked.root_goal
        blocked.active_goal = "Wait for semantic model understanding before mutating the repository."
        blocked.goal_relation = "clarify"
        blocked.output_expectation = "Do not change repository files until semantic task understanding is available."
        blocked.current_user_intent = None
        blocked.execution_strategy = None
        blocked.open_problem = None
        blocked.verification_target = None
        blocked.target_artifacts = []
        blocked.active_artifacts = []
        blocked.evidence = []
        blocked.supplied_evidence = []
        blocked.relevant_context = []
        blocked.constraints = []
        blocked.assumptions = [
            "A2 requires semantic model understanding for create, update, and debug execution instead of deterministic lexical fallback."
        ]
        blocked.missing_info = [
            "Semantic model understanding is currently unavailable for this mutation task."
        ]
        blocked.ambiguity_level = "high"
        blocked.risk_level = "high"
        blocked.confidence = min(float(blocked.confidence or 0.0), 0.2)
        blocked.next_action = "clarify"
        blocked.next_best_action = "clarify"
        blocked.execution_outline = [
            "Retry semantic task understanding before executing repository changes."
        ]
        blocked.needs_clarification = True
        blocked.clarification_questions = [
            "Der semantische Modellpfad ist gerade nicht verfuegbar. Starte den Lauf erneut, sobald das Modell wieder sauber antwortet."
        ]
        return self._finalize_state(blocked, semantic_resolution="blocked")

    def _mode_runtime(self, mode: str) -> tuple[int, int, int]:
        if mode == "full":
            timeout = self.timeout
            total_timeout = max(timeout * 2, timeout + 20)
            num_ctx = self.num_ctx
            return timeout, total_timeout, num_ctx
        timeout = max(12, min(self.timeout, 18))
        # Compact task-state prompts stay small, but JSON responses can still
        # stream slowly on CPU-only systems. Give them a wider completion budget
        # without relaxing the inactivity budget.
        total_timeout = max(timeout * 4, timeout + 40)
        num_ctx = min(self.num_ctx, 2048)
        return timeout, total_timeout, num_ctx

    def _recovery_prompt_mode(
        self,
        *,
        initial_mode: str,
        candidate_prompt_variant: str,
        preserve_full_retry: bool = False,
    ) -> str:
        normalized = str(candidate_prompt_variant or "").strip().lower()
        if normalized == "full":
            return "full"
        if normalized in {"resume", "minimal"}:
            return normalized
        return "compact"

    def _recovery_model_for_failure(
        self,
        *,
        primary_model: str | None,
        reserve_models: list[str] | None,
        failure,
    ) -> str | None:
        candidates = list(reserve_models or [])
        if failure is None:
            return candidates[0] if candidates else None
        if failure.no_start_failure:
            return availability_recovery_model(primary_model, candidates)
        return candidates[0] if candidates else None

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
