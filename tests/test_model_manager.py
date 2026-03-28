from __future__ import annotations

from config.settings import AppConfig
from server.model_manager import ModelManager


class _StubClient:
    def is_available(self) -> bool:
        return True

    def list_models_safe(self) -> list[dict]:
        return []


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
