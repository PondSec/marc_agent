from __future__ import annotations

import json
import socket
from urllib import request

import pytest

from config.settings import AppConfig
from llm.ollama_client import OllamaClient, OllamaGenerationError


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
    progress_events: list[dict] = []

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

    result = client.generate(
        "say hello",
        timeout=6,
        total_timeout=20,
        retries=0,
        progress_callback=progress_events.append,
    )

    assert result == "hello"
    assert [event["type"] for event in progress_events] == ["status", "chunk", "heartbeat", "chunk"]


def test_ollama_client_uses_longer_initial_response_timeout_for_slow_local_models(monkeypatch, tmp_path):
    captured: list[int] = []
    config = AppConfig(workspace_root=str(tmp_path), llm_timeout=25)
    client = OllamaClient(config)

    def fake_urlopen(req: request.Request, timeout):
        del req
        captured.append(timeout)
        return FakeResponse({"response": "ok"})

    monkeypatch.setattr("llm.ollama_client.request.urlopen", fake_urlopen)

    result = client.generate("slow first token", timeout=25, total_timeout=210, retries=0)

    assert result == "ok"
    assert captured[0] > 25


def test_ollama_client_raises_structured_inactivity_timeout_with_partial_progress(monkeypatch, tmp_path):
    config = AppConfig(workspace_root=str(tmp_path), llm_timeout=6, llm_request_retries=0)
    client = OllamaClient(config)

    def fake_urlopen(req: request.Request, timeout):
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["stream"] is True
        return FakeReadlineStreamingResponse(
            [
                (json.dumps({"response": "hel", "done": False}) + "\n").encode("utf-8"),
                socket.timeout("still generating"),
                socket.timeout("still generating"),
            ]
        )

    monkeypatch.setattr("llm.ollama_client.request.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "llm.ollama_client.time.monotonic",
        FakeMonotonic([0.0, 1.0, 2.0, 5.0, 9.5]),
    )

    with pytest.raises(OllamaGenerationError) as excinfo:
        client.generate("say hello", timeout=6, total_timeout=20, retries=0)

    error = excinfo.value
    assert error.reason == "inactivity_timeout"
    assert error.partial_text == "hel"
    assert error.characters == 3
    assert error.progress_seen is True


def test_ollama_client_classifies_pre_chunk_stream_wait_as_startup_timeout(monkeypatch, tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen3-coder:30b",
        llm_timeout=20,
        llm_request_retries=0,
    )
    client = OllamaClient(config)
    progress_events: list[dict] = []

    def fake_urlopen(req: request.Request, timeout):
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["stream"] is True
        return FakeReadlineStreamingResponse(
            [
                socket.timeout("still starting"),
                socket.timeout("still starting"),
                socket.timeout("still starting"),
                socket.timeout("still starting"),
                socket.timeout("still starting"),
                socket.timeout("still starting"),
            ]
        )

    monkeypatch.setattr("llm.ollama_client.request.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "llm.ollama_client.time.monotonic",
        FakeMonotonic([0.0, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0]),
    )

    with pytest.raises(OllamaGenerationError) as excinfo:
        client.generate(
            "generate html",
            timeout=20,
            total_timeout=120,
            retries=0,
            progress_callback=progress_events.append,
        )

    error = excinfo.value
    assert error.reason == "startup_timeout"
    assert error.progress_seen is False
    assert error.retryable is False
    assert progress_events[0]["stage"] == "request_started"
    assert any(event.get("phase") == "waiting_for_start" for event in progress_events if event.get("type") == "heartbeat")
    assert any(event.get("stage") == "startup_timeout_warning" for event in progress_events if event.get("type") == "status")
