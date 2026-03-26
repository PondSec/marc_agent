from __future__ import annotations

import json
from urllib import request

from config.settings import AppConfig
from llm.ollama_client import OllamaClient


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.payload


def test_ollama_client_retries_timeout_then_succeeds(monkeypatch, tmp_path):
    attempts = {"count": 0}
    config = AppConfig(
        workspace_root=str(tmp_path),
        llm_request_retries=2,
        llm_retry_backoff_ms=0,
    )
    client = OllamaClient(config)

    def fake_urlopen(req, timeout):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TimeoutError("timed out")
        return FakeResponse({"response": "ok"})

    monkeypatch.setattr("llm.ollama_client.request.urlopen", fake_urlopen)

    result = client.generate("test")

    assert result == "ok"
    assert attempts["count"] == 3


def test_ollama_client_uses_model_override(monkeypatch, tmp_path):
    captured: list[dict] = []
    config = AppConfig(workspace_root=str(tmp_path))
    client = OllamaClient(config)

    def fake_urlopen(req: request.Request, timeout):
        captured.append(json.loads(req.data.decode("utf-8")))
        return FakeResponse({"response": '{"ok": true}'})

    monkeypatch.setattr("llm.ollama_client.request.urlopen", fake_urlopen)

    payload = client.generate_json(
        "route this",
        model="qwen2.5-coder:14b",
    )

    assert payload == {"ok": True}
    assert captured[0]["model"] == "qwen2.5-coder:14b"
