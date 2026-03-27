from __future__ import annotations

"""Shared runtime resilience primitives for model/backend execution.

Failure classes:
- startup_timeout: the backend never produced a first output event/chunk.
- inactivity_timeout: output started, then progress stalled beyond idle limits.
- total_timeout: output started but the overall budget expired.
- backend_unavailable: the configured backend could not be reached at all.
- backend_overloaded: the backend responded but appears queued, overloaded, or cold-start constrained.
- provider_error/runtime_error: the backend started and then failed for another reason.

Recovery policy:
- Tier A keeps full-quality generation on the preferred model/backend.
- Tier B switches to a smaller or faster compatible model/backend.
- Tier C reduces request complexity to a minimal viable model generation.
- Tier D allows deterministic local fallbacks only when the task class marks them as safe.
- Tier E is the honest block path when no safe recovery exists.

Degraded execution rules:
- Tier C and Tier D are allowed only if they preserve the task's semantic contract.
- Deterministic fallbacks must never pretend that full generation or validation happened.
- Honest blocking is preferred over unsafe degradation.
"""

from dataclasses import asdict, dataclass
import json
import time
from typing import Any, Callable, Literal


CapabilityTier = Literal["tier_a", "tier_b", "tier_c", "tier_d", "tier_e"]
ExecutionAttemptState = Literal[
    "pending",
    "launching",
    "waiting_for_first_output",
    "streaming",
    "stalled_after_progress",
    "completed",
    "failed_startup",
    "failed_inactivity",
    "failed_total_timeout",
    "failed_backend_unavailable",
    "failed_cancelled",
]
FailureClass = Literal[
    "startup_timeout",
    "inactivity_timeout",
    "total_timeout",
    "backend_unavailable",
    "backend_overloaded",
    "provider_error",
    "cancelled",
    "empty_response",
    "runtime_error",
]
ContextPressureEstimate = Literal["low", "medium", "high"]


@dataclass(slots=True)
class ExecutionFailure:
    failure_class: FailureClass
    state: ExecutionAttemptState
    had_progress: bool
    first_output_received: bool
    startup_timeout_seconds: int | None = None
    inactivity_timeout_seconds: int | None = None
    total_timeout_seconds: int | None = None
    elapsed_seconds: float | None = None
    idle_seconds: float | None = None
    model_identifier: str | None = None
    backend_identifier: str | None = None
    context_pressure_estimate: ContextPressureEstimate | None = None
    retryable: bool = True
    recommended_recovery_strategy: str | None = None
    raw_reason: str | None = None
    detail: str | None = None
    partial_text: str = ""
    characters: int = 0
    activity_count: int = 0

    @property
    def reason(self) -> str:
        return str(self.raw_reason or self.failure_class)

    @property
    def timed_out(self) -> bool:
        return self.failure_class in {"startup_timeout", "inactivity_timeout", "total_timeout"}

    @property
    def timeout_like(self) -> bool:
        return self.timed_out

    @property
    def context_pressure_likely(self) -> bool:
        return self.context_pressure_estimate in {"medium", "high"}

    @property
    def no_start_failure(self) -> bool:
        return (
            self.failure_class == "startup_timeout"
            and not self.first_output_received
            and not self.had_progress
            and self.characters <= 0
            and not self.partial_text
        )

    @property
    def failed_after_progress(self) -> bool:
        return (
            self.first_output_received
            or self.had_progress
            or self.characters > 0
            or bool(self.partial_text)
        )

    @property
    def classification(self) -> str:
        if self.no_start_failure:
            return "no_start_failure"
        if self.failed_after_progress:
            return "generation_failed_after_progress"
        if self.failure_class == "startup_timeout":
            return "startup_timeout"
        return "generation_failed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExecutionAttemptRecord:
    operation_name: str
    task_class: str
    attempt_number: int
    capability_tier: CapabilityTier
    recovery_strategy: str
    prompt_variant: str = "full"
    model_identifier: str | None = None
    backend_identifier: str | None = None
    state: ExecutionAttemptState = "pending"
    startup_timeout_seconds: int | None = None
    inactivity_timeout_seconds: int | None = None
    total_timeout_seconds: int | None = None
    elapsed_seconds: float | None = None
    had_progress: bool = False
    first_output_received: bool = False
    output_characters: int = 0
    activity_count: int = 0
    failure: ExecutionFailure | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RecoveryCandidate:
    strategy: str
    capability_tier: CapabilityTier
    model_identifier: str | None = None
    prompt_variant: str = "full"
    local_only: bool = False

    def key(self) -> tuple[str, str | None, str, bool]:
        return (
            self.strategy,
            self.model_identifier,
            self.prompt_variant,
            self.local_only,
        )


@dataclass(slots=True)
class RecoveryDecision:
    candidate: RecoveryCandidate
    accepted: bool
    reason: str


@dataclass(slots=True)
class InvocationOutcome:
    attempt: ExecutionAttemptRecord
    value: Any | None = None
    exception: Exception | None = None


@dataclass(slots=True)
class ExecutionRecoveryPolicy:
    task_class: str
    allow_same_backend_retry: bool = False
    allow_smaller_faster_model: bool = False
    allow_resume_after_progress: bool = False
    allow_reduce_request_complexity: bool = False
    allow_minimal_generation: bool = False
    allow_deterministic_fallback: bool = False
    allow_honest_block: bool = True
    max_same_backend_retries: int = 1
    max_total_attempts: int = 4

    def plan_recovery(
        self,
        failure: ExecutionFailure,
        *,
        primary_model: str | None,
        faster_model: str | None,
        history: list[ExecutionAttemptRecord],
    ) -> list[RecoveryDecision]:
        decisions: list[RecoveryDecision] = []
        attempted_keys = {
            (
                item.recovery_strategy,
                item.model_identifier,
                item.prompt_variant,
                False,
            )
            for item in history
        }
        if len(history) >= self.max_total_attempts:
            if self.allow_honest_block:
                decisions.append(
                    RecoveryDecision(
                        candidate=RecoveryCandidate(
                            "honest_block",
                            "tier_e",
                            model_identifier=None,
                            prompt_variant="blocked",
                            local_only=True,
                        ),
                        accepted=True,
                        reason="maximum_attempts_reached",
                    )
                )
            return decisions

        same_backend_retries = sum(1 for item in history if item.recovery_strategy == "retry_same_backend")

        def _accept(
            candidate: RecoveryCandidate,
            *,
            condition: bool,
            reason: str,
        ) -> None:
            duplicate = candidate.key() in attempted_keys
            if duplicate:
                decisions.append(
                    RecoveryDecision(
                        candidate=candidate,
                        accepted=False,
                        reason="candidate_already_attempted",
                    )
                )
                return
            decisions.append(
                RecoveryDecision(
                    candidate=candidate,
                    accepted=condition,
                    reason=reason if not condition else "accepted",
                )
            )

        if failure.no_start_failure:
            _accept(
                RecoveryCandidate(
                    "retry_same_backend",
                    "tier_a",
                    model_identifier=primary_model,
                    prompt_variant="full",
                ),
                condition=self.allow_same_backend_retry
                and failure.retryable
                and same_backend_retries < self.max_same_backend_retries,
                reason="startup_retry_not_supported_or_not_transient",
            )
            _accept(
                RecoveryCandidate(
                    "switch_to_faster_model",
                    "tier_b",
                    model_identifier=faster_model,
                    prompt_variant="full",
                ),
                condition=self.allow_smaller_faster_model and bool(faster_model),
                reason="no_faster_model_available",
            )
            if self.allow_minimal_generation or (
                self.allow_reduce_request_complexity and failure.context_pressure_likely
            ):
                _accept(
                    RecoveryCandidate(
                        "minimal_viable_generation" if self.allow_minimal_generation else "reduce_request_complexity",
                        "tier_c",
                        model_identifier=faster_model or primary_model,
                        prompt_variant="minimal" if self.allow_minimal_generation else "compact",
                    ),
                    condition=True,
                    reason="accepted",
                )
        elif failure.failed_after_progress:
            if self.allow_resume_after_progress and failure.partial_text:
                _accept(
                    RecoveryCandidate(
                        "resume_after_progress",
                        "tier_a",
                        model_identifier=primary_model,
                        prompt_variant="resume",
                    ),
                    condition=True,
                    reason="accepted",
                )
            _accept(
                RecoveryCandidate(
                    "reduce_request_complexity",
                    "tier_c",
                    model_identifier=faster_model or primary_model,
                    prompt_variant="compact",
                ),
                condition=self.allow_reduce_request_complexity and failure.context_pressure_likely,
                reason="context_pressure_not_plausible",
            )
            _accept(
                RecoveryCandidate(
                    "switch_to_faster_model",
                    "tier_b",
                    model_identifier=faster_model,
                    prompt_variant="full",
                ),
                condition=self.allow_smaller_faster_model and bool(faster_model),
                reason="no_faster_model_available",
            )
        else:
            _accept(
                RecoveryCandidate(
                    "reduce_request_complexity",
                    "tier_c",
                    model_identifier=primary_model,
                    prompt_variant="compact",
                ),
                condition=self.allow_reduce_request_complexity and failure.context_pressure_likely,
                reason="context_pressure_not_plausible",
            )
            _accept(
                RecoveryCandidate(
                    "switch_to_faster_model",
                    "tier_b",
                    model_identifier=faster_model,
                    prompt_variant="full",
                ),
                condition=self.allow_smaller_faster_model and bool(faster_model),
                reason="no_faster_model_available",
            )
            _accept(
                RecoveryCandidate(
                    "reduce_request_complexity",
                    "tier_c",
                    model_identifier=faster_model or primary_model,
                    prompt_variant="compact",
                ),
                condition=self.allow_reduce_request_complexity and not failure.context_pressure_likely,
                reason="reduced_request_recovery_not_supported",
            )

        if self.allow_deterministic_fallback:
            _accept(
                RecoveryCandidate(
                    "deterministic_fallback",
                    "tier_d",
                    model_identifier=None,
                    prompt_variant="deterministic",
                    local_only=True,
                ),
                condition=True,
                reason="accepted",
            )

        if self.allow_honest_block:
            _accept(
                RecoveryCandidate(
                    "honest_block",
                    "tier_e",
                    model_identifier=None,
                    prompt_variant="blocked",
                    local_only=True,
                ),
                condition=True,
                reason="accepted",
            )
        return decisions


def estimate_context_pressure(
    *,
    prompt_chars: int,
    current_content_chars: int = 0,
    error_text: str | None = None,
) -> ContextPressureEstimate:
    lowered = str(error_text or "").lower()
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
        return "high"
    if prompt_chars >= 12_000 or current_content_chars >= 6_000:
        return "high"
    if prompt_chars >= 7_000 or current_content_chars >= 3_000:
        return "medium"
    return "low"


class ExecutionAttemptStateMachine:
    def __init__(
        self,
        record: ExecutionAttemptRecord,
        *,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.record = record
        self._event_callback = event_callback
        self._started_at = time.monotonic()
        self._last_heartbeat_at = 0.0
        self._last_chunk_at = 0.0
        self._last_status_key: tuple[str, str] | None = None
        self.record.state = "launching"

    def progress_callback(self, payload: dict[str, Any]) -> None:
        kind = str(payload.get("type") or "").strip()
        if kind == "status":
            stage = str(payload.get("stage") or "").strip()
            if stage == "request_started":
                self.record.state = "launching"
            elif stage in {"waiting_for_first_chunk", "startup_timeout_warning"}:
                self.record.state = "waiting_for_first_output"
            self._sync_timeouts(payload)
            status_key = (kind, stage)
            if status_key == self._last_status_key:
                return
            self._last_status_key = status_key
        elif kind == "heartbeat":
            if self.record.first_output_received or self.record.had_progress:
                self.record.state = "stalled_after_progress"
            else:
                self.record.state = "waiting_for_first_output"
            now = time.monotonic()
            if now - self._last_heartbeat_at < 8.0:
                return
            self._last_heartbeat_at = now
        elif kind == "chunk":
            self.record.state = "streaming"
            self.record.first_output_received = True
            self.record.had_progress = True
            self.record.output_characters = max(
                self.record.output_characters,
                _coerce_int(payload.get("characters")) or self.record.output_characters,
            )
            now = time.monotonic()
            if now - self._last_chunk_at < 2.0:
                return
            self._last_chunk_at = now
        enriched = {
            **payload,
            "attempt": self.record.attempt_number,
            "attempt_state": self.record.state,
            "capability_tier": self.record.capability_tier,
            "recovery_strategy": self.record.recovery_strategy,
            "prompt_variant": self.record.prompt_variant,
            "model": payload.get("model", self.record.model_identifier),
            "backend": payload.get("backend", self.record.backend_identifier),
            "task_class": self.record.task_class,
        }
        self._emit(enriched)

    def complete(self, value: Any) -> ExecutionAttemptRecord:
        self.record.state = "completed"
        self.record.elapsed_seconds = round(time.monotonic() - self._started_at, 1)
        self.record.output_characters = max(
            self.record.output_characters,
            measure_output_characters(value),
        )
        return self.record

    def fail(
        self,
        exc: Exception,
        *,
        context_pressure_estimate: ContextPressureEstimate | None = None,
    ) -> ExecutionAttemptRecord:
        failure = classify_execution_failure(
            exc,
            attempt=self.record,
            context_pressure_estimate=context_pressure_estimate,
            elapsed_seconds=round(time.monotonic() - self._started_at, 1),
        )
        self.record.failure = failure
        self.record.state = failure.state
        self.record.elapsed_seconds = failure.elapsed_seconds
        self.record.had_progress = failure.had_progress
        self.record.first_output_received = failure.first_output_received
        self.record.output_characters = max(self.record.output_characters, failure.characters)
        self.record.activity_count = max(self.record.activity_count, failure.activity_count)
        return self.record

    def _sync_timeouts(self, payload: dict[str, Any]) -> None:
        if self.record.startup_timeout_seconds is None:
            self.record.startup_timeout_seconds = _coerce_int(payload.get("startup_timeout"))
        if self.record.inactivity_timeout_seconds is None:
            self.record.inactivity_timeout_seconds = _coerce_int(payload.get("inactivity_timeout"))
        if self.record.total_timeout_seconds is None:
            self.record.total_timeout_seconds = _coerce_int(payload.get("total_timeout"))

    def _emit(self, payload: dict[str, Any]) -> None:
        if self._event_callback is None:
            return
        self._event_callback(payload)


def invoke_model(
    invoke: Callable[[Callable[[dict[str, Any]], None] | None], Any],
    *,
    operation_name: str,
    task_class: str,
    attempt_number: int,
    capability_tier: CapabilityTier,
    recovery_strategy: str,
    prompt_variant: str,
    model_identifier: str | None,
    backend_identifier: str | None,
    startup_timeout_seconds: int | None = None,
    inactivity_timeout_seconds: int | None = None,
    total_timeout_seconds: int | None = None,
    context_pressure_estimate: ContextPressureEstimate | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> InvocationOutcome:
    record = ExecutionAttemptRecord(
        operation_name=operation_name,
        task_class=task_class,
        attempt_number=attempt_number,
        capability_tier=capability_tier,
        recovery_strategy=recovery_strategy,
        prompt_variant=prompt_variant,
        model_identifier=model_identifier,
        backend_identifier=backend_identifier,
        startup_timeout_seconds=startup_timeout_seconds,
        inactivity_timeout_seconds=inactivity_timeout_seconds,
        total_timeout_seconds=total_timeout_seconds,
    )
    machine = ExecutionAttemptStateMachine(record, event_callback=event_callback)
    try:
        value = invoke(machine.progress_callback)
    except Exception as exc:  # noqa: BLE001
        machine.fail(exc, context_pressure_estimate=context_pressure_estimate)
        return InvocationOutcome(attempt=record, value=None, exception=exc)
    machine.complete(value)
    return InvocationOutcome(attempt=record, value=value, exception=None)


def classify_execution_failure(
    exc: Exception,
    *,
    attempt: ExecutionAttemptRecord,
    context_pressure_estimate: ContextPressureEstimate | None,
    elapsed_seconds: float | None,
) -> ExecutionFailure:
    message = str(exc or "")
    lowered = message.lower()
    partial_text = str(getattr(exc, "partial_text", "") or "")
    characters = _coerce_int(getattr(exc, "characters", None)) or len(partial_text)
    activity_count = _coerce_int(getattr(exc, "activity_count", None)) or 0
    raw_reason = str(getattr(exc, "reason", "") or "").strip() or None
    first_output_received = bool(
        getattr(exc, "first_output_received", False)
        or attempt.first_output_received
        or partial_text
        or characters > 0
    )
    had_progress = bool(
        getattr(exc, "progress_seen", False)
        or attempt.had_progress
        or first_output_received
        or partial_text
        or characters > 0
    )

    failure_class: FailureClass
    if raw_reason == "startup_timeout" or (
        not first_output_received
        and ("first chunk" in lowered or "start streaming" in lowered or "model to start" in lowered)
    ):
        failure_class = "startup_timeout"
    elif raw_reason == "inactivity_timeout" or (
        had_progress and "timed out" in lowered and "progress" in lowered
    ):
        failure_class = "inactivity_timeout"
    elif raw_reason == "total_timeout" or (
        had_progress and "timed out" in lowered and "completion" in lowered
    ):
        failure_class = "total_timeout"
    elif raw_reason == "backend_unavailable" or any(
        marker in lowered
        for marker in (
            "could not reach",
            "connection refused",
            "connection reset",
            "name or service not known",
            "host is down",
            "backend unavailable",
        )
    ):
        failure_class = "backend_unavailable"
    elif raw_reason == "backend_overloaded" or any(
        marker in lowered
        for marker in (
            "overloaded",
            "too many requests",
            "rate limited",
            "queue",
            "queued",
            "busy",
            "warming up",
            "cold start",
            "capacity",
            "unavailable due to load",
        )
    ):
        failure_class = "backend_overloaded"
    elif raw_reason == "cancelled" or "cancel" in lowered:
        failure_class = "cancelled"
    elif raw_reason == "empty_response":
        failure_class = "empty_response"
    elif raw_reason == "provider_error":
        failure_class = "provider_error"
    else:
        failure_class = "runtime_error"

    state: ExecutionAttemptState
    if failure_class == "startup_timeout":
        state = "failed_startup"
    elif failure_class == "inactivity_timeout":
        state = "failed_inactivity"
    elif failure_class == "total_timeout":
        state = "failed_total_timeout"
    elif failure_class in {"backend_unavailable", "backend_overloaded", "provider_error", "runtime_error", "empty_response"}:
        state = "failed_backend_unavailable"
    else:
        state = "failed_cancelled"

    retryable = bool(getattr(exc, "retryable", True))
    recommended = recommended_recovery_strategy_for_failure(
        failure_class,
        had_progress=had_progress,
        first_output_received=first_output_received,
        retryable=retryable,
        context_pressure_estimate=context_pressure_estimate,
        partial_text=partial_text,
    )
    return ExecutionFailure(
        failure_class=failure_class,
        state=state,
        had_progress=had_progress,
        first_output_received=first_output_received,
        startup_timeout_seconds=_coerce_int(
            getattr(exc, "startup_timeout_seconds", None)
        )
        or attempt.startup_timeout_seconds,
        inactivity_timeout_seconds=_coerce_int(
            getattr(exc, "inactivity_timeout_seconds", None)
        )
        or attempt.inactivity_timeout_seconds,
        total_timeout_seconds=_coerce_int(
            getattr(exc, "total_timeout_seconds", None)
        )
        or attempt.total_timeout_seconds,
        elapsed_seconds=_coerce_float(getattr(exc, "elapsed", None)) or elapsed_seconds,
        idle_seconds=_coerce_float(getattr(exc, "idle_for", None)),
        model_identifier=str(getattr(exc, "model_name", "") or attempt.model_identifier or "").strip() or attempt.model_identifier,
        backend_identifier=str(getattr(exc, "backend_identifier", "") or attempt.backend_identifier or "").strip() or attempt.backend_identifier,
        context_pressure_estimate=context_pressure_estimate,
        retryable=retryable,
        recommended_recovery_strategy=recommended,
        raw_reason=raw_reason,
        detail=message,
        partial_text=partial_text,
        characters=characters,
        activity_count=activity_count,
    )


def recommended_recovery_strategy_for_failure(
    failure_class: FailureClass,
    *,
    had_progress: bool,
    first_output_received: bool,
    retryable: bool,
    context_pressure_estimate: ContextPressureEstimate | None,
    partial_text: str,
) -> str:
    if failure_class == "startup_timeout":
        if retryable:
            return "retry_same_backend"
        return "switch_to_faster_model"
    if failure_class in {"inactivity_timeout", "total_timeout"}:
        if partial_text or first_output_received or had_progress:
            return "resume_after_progress"
        if context_pressure_estimate in {"medium", "high"}:
            return "reduce_request_complexity"
        return "switch_to_faster_model"
    if failure_class in {"backend_unavailable", "backend_overloaded"}:
        return "switch_to_faster_model"
    if failure_class == "empty_response":
        return "reduce_request_complexity"
    if failure_class == "cancelled":
        return "honest_block"
    return "deterministic_fallback" if not had_progress else "reduce_request_complexity"


def build_execution_run_record(
    *,
    operation_name: str,
    task_class: str,
    final_state: str,
    capability_tier: CapabilityTier,
    recovery_strategy: str,
    degraded: bool,
    honest_blocked: bool,
    artifact_bytes_generated: int,
    validation_possible: bool | None,
    summary: str | None,
    attempts: list[ExecutionAttemptRecord],
) -> dict[str, Any]:
    return {
        "operation_name": operation_name,
        "task_class": task_class,
        "final_state": final_state,
        "capability_tier": capability_tier,
        "recovery_strategy": recovery_strategy,
        "degraded": degraded,
        "honest_blocked": honest_blocked,
        "artifact_bytes_generated": max(int(artifact_bytes_generated), 0),
        "validation_possible": validation_possible,
        "summary": summary,
        "attempts": [item.to_dict() for item in attempts],
    }


def measure_output_characters(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value, ensure_ascii=False))
    except TypeError:
        return len(str(value))


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
