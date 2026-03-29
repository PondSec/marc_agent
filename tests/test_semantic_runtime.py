from __future__ import annotations

from config.settings import AppConfig
from agent.semantic_runtime import semantic_model_candidates


def test_semantic_model_candidates_keep_larger_primary_model_as_reserve_for_smaller_router():
    config = AppConfig(
        workspace_root=".",
        model_name="qwen3:14b",
        router_model_name="qwen3:8b",
    )

    candidates = semantic_model_candidates("qwen3:8b", config)

    assert candidates == ["qwen3:8b", "qwen3:14b"]


def test_semantic_model_candidates_keep_smaller_router_model_as_reserve_for_larger_primary():
    config = AppConfig(
        workspace_root=".",
        model_name="qwen3:14b",
        router_model_name="qwen3:8b",
    )

    candidates = semantic_model_candidates("qwen3:14b", config)

    assert candidates == ["qwen3:14b", "qwen3:8b"]


def test_semantic_model_candidates_keep_unknown_models_when_size_cannot_be_estimated():
    config = AppConfig(
        workspace_root=".",
        model_name="model-beta",
        router_model_name="model-alpha",
    )

    candidates = semantic_model_candidates("model-alpha", config)

    assert candidates == ["model-alpha", "model-beta"]
