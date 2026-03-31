from __future__ import annotations

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
