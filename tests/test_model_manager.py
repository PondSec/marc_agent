from __future__ import annotations

from config.settings import AppConfig
from server.model_manager import ModelManager


class _StubClient:
    def is_available(self) -> bool:
        return True

    def list_models_safe(self) -> list[dict]:
        return []


class _WarmupClient(_StubClient):
    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def list_models_safe(self) -> list[dict]:
        return [
            {"name": "qwen3:14b"},
            {"name": "qwen3:8b"},
        ]

    def generate(self, prompt: str, *, model: str, timeout: int, num_ctx: int, retries: int):
        del prompt, timeout, retries
        self.calls.append((model, num_ctx))
        return "OK"


def test_ensure_recommended_respects_auto_install_flag(tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        auto_install_recommended_models=False,
    ).normalized()
    manager = ModelManager(config)
    manager.client = _StubClient()

    catalog = manager.ensure_recommended()

    assert manager._queue == []
    assert manager._worker is None
    assert all(item["status"] == "missing" for item in catalog["recommended_models"])


def test_warmup_preferred_models_leaves_router_model_hot_for_first_request(tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen3:14b",
        router_model_name="qwen3:8b",
        ollama_num_ctx=4096,
        router_num_ctx=2048,
    ).normalized()
    manager = ModelManager(config)
    manager.client = _WarmupClient()

    manager.warmup_preferred_models()

    assert manager.client.calls == [("qwen3:14b", 4096), ("qwen3:8b", 2048)]


def test_warmup_preferred_models_uses_max_ctx_when_primary_and_router_match(tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen2.5-coder:7b",
        router_model_name="qwen2.5-coder:7b",
        ollama_num_ctx=4096,
        router_num_ctx=2048,
    ).normalized()
    manager = ModelManager(config)
    manager.client = _WarmupClient()
    manager.client.list_models_safe = lambda: [{"name": "qwen2.5-coder:7b"}]

    manager.warmup_preferred_models()

    assert manager.client.calls == [
        ("qwen2.5-coder:7b", 4096),
        ("qwen2.5-coder:7b", 2048),
    ]
