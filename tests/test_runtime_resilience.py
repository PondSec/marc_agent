from __future__ import annotations

import threading

import llm.runtime_resilience as runtime_resilience
from llm.runtime_resilience import ExecutionFailure, ExecutionRecoveryPolicy


def test_no_start_recovery_prefers_full_retry_then_full_reserve_before_minimal_generation():
    policy = ExecutionRecoveryPolicy(
        task_class="task_state_generation",
        allow_same_backend_retry=True,
        allow_smaller_faster_model=True,
        allow_reduce_request_complexity=True,
        allow_minimal_generation=True,
        allow_deterministic_fallback=False,
        max_same_backend_retries=1,
        max_total_attempts=4,
    )
    failure = ExecutionFailure(
        failure_class="startup_timeout",
        state="failed_startup",
        had_progress=False,
        first_output_received=False,
        model_identifier="qwen2.5-coder:7b",
        backend_identifier="ollama",
        context_pressure_estimate="low",
        retryable=True,
        raw_reason="startup_timeout",
    )

    decisions = policy.plan_recovery(
        failure,
        primary_model="qwen2.5-coder:7b",
        faster_model="qwen2.5-coder:14b",
        history=[],
    )
    accepted = [
        (
            decision.candidate.strategy,
            decision.candidate.prompt_variant,
            decision.candidate.model_identifier,
        )
        for decision in decisions
        if decision.accepted
    ]

    assert accepted[:4] == [
        ("retry_same_backend", "full", "qwen2.5-coder:7b"),
        ("switch_to_faster_model", "full", "qwen2.5-coder:14b"),
        ("minimal_viable_generation", "minimal", "qwen2.5-coder:14b"),
        ("honest_block", "blocked", None),
    ]


class FakeClock:
    def __init__(self) -> None:
        self._value = 0.0
        self._lock = threading.Lock()

    def monotonic(self) -> float:
        with self._lock:
            return self._value

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self._value += max(float(seconds), 0.0)


def test_invoke_model_watchdog_finishes_hanging_generation_without_progress(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(runtime_resilience.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(runtime_resilience.time, "sleep", clock.sleep)
    blocker = threading.Event()

    def hanging_invoke(progress):
        progress(
            {
                "type": "status",
                "stage": "request_started",
                "model": "qwen2.5-coder:7b",
                "startup_timeout": 2,
                "inactivity_timeout": 2,
                "total_timeout": 5,
            }
        )
        blocker.wait(timeout=60)
        return "never"

    outcome = runtime_resilience.invoke_model(
        hanging_invoke,
        operation_name="content_generation",
        task_class="content_generation",
        attempt_number=1,
        capability_tier="tier_a",
        recovery_strategy="primary_model",
        prompt_variant="full",
        model_identifier="qwen2.5-coder:7b",
        backend_identifier="ollama",
        startup_timeout_seconds=2,
        inactivity_timeout_seconds=2,
        total_timeout_seconds=5,
    )

    assert outcome.value is None
    assert outcome.exception is not None
    assert outcome.attempt.state == "failed_startup"
    assert outcome.attempt.failure is not None
    assert outcome.attempt.failure.failure_class == "startup_timeout"
    assert outcome.attempt.failure.no_start_failure is True


def test_invoke_model_allows_delayed_response_before_timeout(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(runtime_resilience.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(runtime_resilience.time, "sleep", clock.sleep)

    def delayed_success(progress):
        progress(
            {
                "type": "status",
                "stage": "request_started",
                "model": "qwen2.5-coder:7b",
                "startup_timeout": 3,
                "inactivity_timeout": 2,
                "total_timeout": 6,
            }
        )
        runtime_resilience.time.sleep(1.0)
        progress(
            {
                "type": "status",
                "stage": "response_headers_received",
                "model": "qwen2.5-coder:7b",
                "startup_timeout": 3,
                "inactivity_timeout": 2,
                "total_timeout": 6,
            }
        )
        progress(
            {
                "type": "status",
                "stage": "waiting_for_first_chunk",
                "model": "qwen2.5-coder:7b",
                "startup_timeout": 3,
                "inactivity_timeout": 2,
                "total_timeout": 6,
            }
        )
        runtime_resilience.time.sleep(0.5)
        progress(
            {
                "type": "status",
                "stage": "first_chunk_received",
                "model": "qwen2.5-coder:7b",
                "startup_timeout": 3,
                "inactivity_timeout": 2,
                "total_timeout": 6,
            }
        )
        progress({"type": "chunk", "characters": 5, "model": "qwen2.5-coder:7b"})
        runtime_resilience.time.sleep(0.5)
        return "hello"

    outcome = runtime_resilience.invoke_model(
        delayed_success,
        operation_name="content_generation",
        task_class="content_generation",
        attempt_number=1,
        capability_tier="tier_a",
        recovery_strategy="primary_model",
        prompt_variant="full",
        model_identifier="qwen2.5-coder:7b",
        backend_identifier="ollama",
        startup_timeout_seconds=3,
        inactivity_timeout_seconds=2,
        total_timeout_seconds=6,
    )

    assert outcome.exception is None
    assert outcome.value == "hello"
    assert outcome.attempt.state == "completed"
    assert outcome.attempt.output_characters >= 5


def test_invoke_model_watchdog_uses_provider_reported_effective_timeouts(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(runtime_resilience.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(runtime_resilience.time, "sleep", clock.sleep)
    blocker = threading.Event()

    def hanging_invoke(progress):
        progress(
            {
                "type": "status",
                "stage": "request_started",
                "model": "qwen2.5-coder:14b",
                "startup_timeout": 10,
                "inactivity_timeout": 6,
                "total_timeout": 12,
            }
        )
        blocker.wait(timeout=60)
        return "never"

    outcome = runtime_resilience.invoke_model(
        hanging_invoke,
        operation_name="task_state_generation",
        task_class="task_state_generation",
        attempt_number=1,
        capability_tier="tier_a",
        recovery_strategy="retry_same_backend",
        prompt_variant="full",
        model_identifier="qwen2.5-coder:14b",
        backend_identifier="ollama",
        startup_timeout_seconds=4,
        inactivity_timeout_seconds=2,
        total_timeout_seconds=5,
    )

    assert outcome.value is None
    assert outcome.exception is not None
    assert outcome.attempt.failure is not None
    assert outcome.attempt.failure.failure_class == "startup_timeout"
    assert 10.0 <= outcome.attempt.failure.elapsed_seconds < 11.0
    assert outcome.attempt.failure.startup_timeout_seconds == 10
    assert outcome.attempt.failure.inactivity_timeout_seconds == 6
    assert outcome.attempt.failure.total_timeout_seconds == 12


def test_invoke_model_watchdog_tracks_terminal_outcomes_across_multiple_attempts(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(runtime_resilience.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(runtime_resilience.time, "sleep", clock.sleep)
    blockers = [threading.Event(), threading.Event()]

    def no_start(progress):
        progress(
            {
                "type": "status",
                "stage": "request_started",
                "model": "qwen2.5-coder:7b",
                "startup_timeout": 2,
                "inactivity_timeout": 2,
                "total_timeout": 5,
            }
        )
        blockers[0].wait(timeout=60)
        return "never"

    def slow_success(progress):
        progress(
            {
                "type": "status",
                "stage": "request_started",
                "model": "qwen2.5-coder:7b",
                "startup_timeout": 2,
                "inactivity_timeout": 2,
                "total_timeout": 5,
            }
        )
        runtime_resilience.time.sleep(1.0)
        progress({"type": "chunk", "characters": 2, "model": "qwen2.5-coder:7b"})
        return "ok"

    def stall_after_progress(progress):
        progress(
            {
                "type": "status",
                "stage": "request_started",
                "model": "qwen2.5-coder:7b",
                "startup_timeout": 2,
                "inactivity_timeout": 2,
                "total_timeout": 5,
            }
        )
        progress({"type": "chunk", "characters": 7, "model": "qwen2.5-coder:7b"})
        blockers[1].wait(timeout=60)
        return "never"

    first = runtime_resilience.invoke_model(
        no_start,
        operation_name="content_generation",
        task_class="content_generation",
        attempt_number=1,
        capability_tier="tier_a",
        recovery_strategy="primary_model",
        prompt_variant="full",
        model_identifier="qwen2.5-coder:7b",
        backend_identifier="ollama",
        startup_timeout_seconds=2,
        inactivity_timeout_seconds=2,
        total_timeout_seconds=5,
    )
    second = runtime_resilience.invoke_model(
        slow_success,
        operation_name="content_generation",
        task_class="content_generation",
        attempt_number=2,
        capability_tier="tier_a",
        recovery_strategy="review_guided_retry",
        prompt_variant="compact",
        model_identifier="qwen2.5-coder:7b",
        backend_identifier="ollama",
        startup_timeout_seconds=2,
        inactivity_timeout_seconds=2,
        total_timeout_seconds=5,
    )
    third = runtime_resilience.invoke_model(
        stall_after_progress,
        operation_name="content_generation",
        task_class="content_generation",
        attempt_number=3,
        capability_tier="tier_a",
        recovery_strategy="review_guided_retry_followup",
        prompt_variant="full",
        model_identifier="qwen2.5-coder:7b",
        backend_identifier="ollama",
        startup_timeout_seconds=2,
        inactivity_timeout_seconds=2,
        total_timeout_seconds=5,
    )

    assert first.attempt.state == "failed_startup"
    assert second.attempt.state == "completed"
    assert third.attempt.state == "failed_inactivity"
    assert third.attempt.failure is not None
    assert third.attempt.failure.failure_class == "inactivity_timeout"
