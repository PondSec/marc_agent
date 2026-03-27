from __future__ import annotations

import json
import socket
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


class FakeStreamingResponse:
    def __init__(self, payloads: list[dict]):
        self.payloads = [
            (json.dumps(item) + "\n").encode("utf-8")
            for item in payloads
        ]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(self.payloads)


class FakeReadlineStreamingResponse:
    def __init__(self, events):
        self.events = list(events)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def readline(self):
        if not self.events:
            return b""
        event = self.events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event


class FakeMonotonic:
    def __init__(self, values):
        self.values = list(values)
        self.last = self.values[-1] if self.values else 0.0

    def __call__(self):
        if self.values:
            self.last = self.values.pop(0)
        return self.last


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


def test_ollama_client_aggregates_streamed_generate_chunks(monkeypatch, tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    client = OllamaClient(config)

    def fake_urlopen(req: request.Request, timeout):
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["stream"] is True
        return FakeStreamingResponse(
            [
                {"response": "hel", "done": False},
                {"response": "lo", "done": False},
                {"done": True},
            ]
        )

    monkeypatch.setattr("llm.ollama_client.request.urlopen", fake_urlopen)

    result = client.generate("say hello")

    assert result == "hello"


def test_ollama_client_treats_stream_heartbeats_as_progress_not_timeout(monkeypatch, tmp_path):
    config = AppConfig(workspace_root=str(tmp_path), llm_timeout=6)
    client = OllamaClient(config)

    def fake_urlopen(req: request.Request, timeout):
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["stream"] is True
        return FakeReadlineStreamingResponse(
            [
                socket.timeout("still generating"),
                (json.dumps({"response": "hel", "done": False}) + "\n").encode("utf-8"),
                socket.timeout("still generating"),
                (json.dumps({"response": "lo", "done": False}) + "\n").encode("utf-8"),
                (json.dumps({"done": True}) + "\n").encode("utf-8"),
            ]
        )

    monkeypatch.setattr("llm.ollama_client.request.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "llm.ollama_client.time.monotonic",
        FakeMonotonic([0.0, 4.0, 5.0, 9.0, 10.0]),
    )

    result = client.generate("say hello", timeout=6, total_timeout=20)

    assert result == "hello"
