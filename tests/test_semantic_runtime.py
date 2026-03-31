from __future__ import annotations

from config.settings import AppConfig
from agent.semantic_runtime import availability_recovery_model, semantic_model_candidates


def test_semantic_model_candidates_skip_larger_reserve_model_when_primary_is_already_smaller():
    config = AppConfig(
        workspace_root=".",
        model_name="qwen3-coder:30b",
        router_model_name="qwen2.5-coder:14b",
    )

    candidates = semantic_model_candidates("qwen2.5-coder:14b", config)

    assert candidates == ["qwen2.5-coder:14b"]


def test_semantic_model_candidates_keep_smaller_router_model_as_reserve_for_larger_primary():
    config = AppConfig(
        workspace_root=".",
        model_name="qwen3-coder:30b",
        router_model_name="qwen2.5-coder:14b",
    )

    candidates = semantic_model_candidates("qwen3-coder:30b", config)

    assert candidates == ["qwen3-coder:30b", "qwen2.5-coder:14b"]


def test_semantic_model_candidates_keep_unknown_models_when_size_cannot_be_estimated():
    config = AppConfig(
        workspace_root=".",
        model_name="model-beta",
        router_model_name="model-alpha",
    )

    candidates = semantic_model_candidates("model-alpha", config)

    assert candidates == ["model-alpha", "model-beta"]


def test_semantic_model_candidates_use_discovered_live_pool_for_same_model_stack():
    config = AppConfig(
        workspace_root=".",
        model_name="qwen2.5-coder:7b",
        router_model_name="qwen2.5-coder:7b",
        model_candidates=("qwen2.5-coder:7b", "qwen2.5-coder:14b", "qwen3-coder:30b"),
    )

    candidates = semantic_model_candidates("qwen2.5-coder:7b", config)

    assert candidates[:3] == ["qwen2.5-coder:7b", "qwen2.5-coder:14b", "qwen3-coder:30b"]


def test_availability_recovery_model_skips_larger_reserve_candidate():
    assert availability_recovery_model("qwen2.5-coder:7b", "qwen2.5-coder:14b") is None


def test_availability_recovery_model_keeps_smaller_reserve_candidate():
    assert availability_recovery_model("qwen2.5-coder:14b", "qwen2.5-coder:7b") == "qwen2.5-coder:7b"
