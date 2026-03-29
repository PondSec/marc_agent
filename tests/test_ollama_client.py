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
        self.read_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def readline(self):
        self.read_calls += 1
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
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen2.5-coder:7b",
        llm_timeout=6,
    )
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
    assert [event["type"] for event in progress_events] == [
        "status",
        "status",
        "status",
        "status",
        "chunk",
        "heartbeat",
        "chunk",
        "status",
    ]
    assert progress_events[0]["stage"] == "request_started"
    assert progress_events[1]["stage"] == "response_headers_received"
    assert progress_events[2]["stage"] == "waiting_for_first_chunk"
    assert progress_events[3]["stage"] == "first_chunk_received"
    assert progress_events[-1]["stage"] == "response_completed"


def test_ollama_client_treats_timed_out_object_oserror_as_stream_wait(monkeypatch, tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen2.5-coder:14b",
        llm_timeout=30,
    )
    client = OllamaClient(config)
    progress_events: list[dict] = []

    def fake_urlopen(req: request.Request, timeout):
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["stream"] is True
        return FakeReadlineStreamingResponse(
            [
                (json.dumps({"response": "O", "done": False}) + "\n").encode("utf-8"),
                OSError("cannot read from timed out object"),
                (json.dumps({"response": "K", "done": False}) + "\n").encode("utf-8"),
                (json.dumps({"done": True}) + "\n").encode("utf-8"),
            ]
        )

    monkeypatch.setattr("llm.ollama_client.request.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "llm.ollama_client.time.monotonic",
        FakeMonotonic([0.0, 1.0, 7.0, 8.0, 12.0, 13.0]),
    )

    result = client.generate(
        "say ok",
        timeout=30,
        total_timeout=120,
        retries=0,
        progress_callback=progress_events.append,
    )

    assert result == "OK"
    assert [event.get("type") for event in progress_events if event.get("type") == "chunk"] == ["chunk", "chunk"]


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


def test_ollama_client_stops_streaming_json_once_object_is_complete(monkeypatch, tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    client = OllamaClient(config)
    response = FakeReadlineStreamingResponse(
        [
            (json.dumps({"response": "{\"ok\":", "done": False}) + "\n").encode("utf-8"),
            (json.dumps({"response": " true}", "done": False}) + "\n").encode("utf-8"),
            (json.dumps({"response": "   ", "done": False}) + "\n").encode("utf-8"),
        ]
    )

    def fake_urlopen(req: request.Request, timeout):
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["format"] == "json"
        return response

    monkeypatch.setattr("llm.ollama_client.request.urlopen", fake_urlopen)

    payload = client.generate_json("route this", retries=0)

    assert payload == {"ok": True}
    assert response.read_calls == 2


def test_ollama_client_raises_structured_inactivity_timeout_with_partial_progress(monkeypatch, tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen2.5-coder:7b",
        llm_timeout=6,
        llm_request_retries=0,
    )
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


def test_ollama_client_enforces_total_timeout_while_streaming_progress(monkeypatch, tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen2.5-coder:7b",
        llm_timeout=6,
        llm_request_retries=0,
    )
    client = OllamaClient(config)

    def fake_urlopen(req: request.Request, timeout):
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["stream"] is True
        return FakeReadlineStreamingResponse(
            [
                (json.dumps({"response": "hel", "done": False}) + "\n").encode("utf-8"),
                (json.dumps({"response": "lo", "done": False}) + "\n").encode("utf-8"),
                (json.dumps({"response": " world", "done": False}) + "\n").encode("utf-8"),
            ]
        )

    monkeypatch.setattr("llm.ollama_client.request.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "llm.ollama_client.time.monotonic",
        FakeMonotonic([0.0, 0.0, 4.0, 11.2]),
    )

    with pytest.raises(OllamaGenerationError) as excinfo:
        client.generate("say hello", timeout=6, total_timeout=10, retries=0)

    error = excinfo.value
    assert error.reason == "total_timeout"
    assert error.partial_text == "hello"
    assert error.characters == 5
    assert error.progress_seen is True


def test_ollama_client_classifies_pre_chunk_stream_wait_as_startup_timeout(monkeypatch, tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen2.5-coder:14b",
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
        FakeMonotonic([0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 132.0, 132.0]),
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
    assert error.model_name == "qwen2.5-coder:14b"
    assert error.backend_identifier == "ollama"
    assert error.startup_timeout_seconds == 130
    assert error.inactivity_timeout_seconds == 45
    assert error.total_timeout_seconds == 150
    assert error.first_output_received is False
    assert progress_events[0]["stage"] == "request_started"
    assert progress_events[0]["startup_timeout"] == 130
    assert progress_events[0]["inactivity_timeout"] == 45
    assert progress_events[0]["total_timeout"] == 150
    assert any(event.get("stage") == "waiting_for_first_chunk" for event in progress_events if event.get("type") == "status")
    assert any(event.get("phase") == "waiting_for_start" for event in progress_events if event.get("type") == "heartbeat")


def test_ollama_client_stops_model_after_pre_chunk_startup_timeout(monkeypatch, tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen2.5-coder:14b",
        llm_timeout=20,
        llm_request_retries=0,
    )
    client = OllamaClient(config)
    stopped: list[str] = []

    def fake_urlopen(req: request.Request, timeout):
        del req, timeout
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
        FakeMonotonic([0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 132.0, 132.0]),
    )
    monkeypatch.setattr(client, "_stop_running_model", stopped.append)

    with pytest.raises(OllamaGenerationError):
        client.generate(
            "generate html",
            timeout=20,
            total_timeout=120,
            retries=0,
        )

    assert stopped == ["qwen2.5-coder:14b"]


def test_ollama_client_waits_for_model_to_leave_ollama_ps_after_stop(monkeypatch, tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen2.5-coder:14b",
    )
    client = OllamaClient(config)
    calls: list[list[str]] = []
    slept: list[float] = []
    ps_outputs = iter(
        [
            "NAME ID SIZE PROCESSOR CONTEXT UNTIL\nqwen2.5-coder:14b abc 8.9 GB 100% CPU 2048 Stopping...\n",
            "NAME ID SIZE PROCESSOR CONTEXT UNTIL\nqwen2.5-coder:14b abc 8.9 GB 100% CPU 2048 Stopping...\n",
            "NAME ID SIZE PROCESSOR CONTEXT UNTIL\n",
        ]
    )

    class _Completed:
        def __init__(self, stdout: str = ""):
            self.stdout = stdout

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[:2] == ["ollama", "stop"]:
            return _Completed()
        if cmd[:2] == ["ollama", "ps"]:
            return _Completed(next(ps_outputs))
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("llm.ollama_client.subprocess.run", fake_run)
    monkeypatch.setattr("llm.ollama_client.time.monotonic", FakeMonotonic([0.0, 1.0, 2.0, 3.0, 4.0]))
    monkeypatch.setattr("llm.ollama_client.time.sleep", slept.append)

    client._stop_running_model("qwen2.5-coder:14b")

    assert calls == [
        ["ollama", "stop", "qwen2.5-coder:14b"],
        ["ollama", "ps"],
        ["ollama", "ps"],
        ["ollama", "ps"],
    ]
    assert slept == [1.0, 1.0]


def test_ollama_client_expands_large_model_time_budget(monkeypatch, tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen3-coder:30b",
        llm_timeout=25,
        llm_request_retries=0,
    )
    client = OllamaClient(config)
    captured: dict[str, object] = {}
    progress_events: list[dict] = []

    def fake_urlopen(req: request.Request, timeout):
        captured["timeout"] = timeout
        return FakeReadlineStreamingResponse(
            [
                (json.dumps({"response": "OK", "done": False}) + "\n").encode("utf-8"),
                (json.dumps({"response": "", "done": True}) + "\n").encode("utf-8"),
            ]
        )

    monkeypatch.setattr("llm.ollama_client.request.urlopen", fake_urlopen)

    result = client.generate(
        "Reply with OK only.",
        timeout=25,
        total_timeout=50,
        retries=0,
        progress_callback=progress_events.append,
    )

    assert result == "OK"
    assert captured["timeout"] == 480
    assert progress_events[0]["startup_timeout"] == 480
    assert progress_events[0]["inactivity_timeout"] == 240
    assert progress_events[0]["total_timeout"] == 1200


def test_ollama_client_gives_7b_models_more_startup_headroom(monkeypatch, tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen2.5-coder:7b",
        llm_timeout=45,
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
            ]
        )

    monkeypatch.setattr("llm.ollama_client.request.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "llm.ollama_client.time.monotonic",
        FakeMonotonic([0.0, 30.0, 60.0, 86.0, 86.0]),
    )

    with pytest.raises(OllamaGenerationError) as excinfo:
        client.generate(
            "compact repair prompt",
            timeout=45,
            total_timeout=120,
            retries=0,
            strict_timeouts=True,
            progress_callback=progress_events.append,
        )

    error = excinfo.value
    assert error.reason == "startup_timeout"
    assert error.startup_timeout_seconds == 85
    assert error.inactivity_timeout_seconds == 45
    assert error.total_timeout_seconds == 120
    assert progress_events[0]["startup_timeout"] == 85
    assert progress_events[0]["inactivity_timeout"] == 45
    assert progress_events[0]["total_timeout"] == 120


def test_ollama_client_strict_timeouts_preserve_compact_budget_for_14b_models(monkeypatch, tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen3:14b",
        llm_timeout=45,
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
            ]
        )

    monkeypatch.setattr("llm.ollama_client.request.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "llm.ollama_client.time.monotonic",
        FakeMonotonic([0.0, 12.0, 24.0, 39.0, 39.0]),
    )

    with pytest.raises(OllamaGenerationError) as excinfo:
        client.generate(
            "compact planning prompt",
            timeout=18,
            total_timeout=38,
            retries=0,
            strict_timeouts=True,
            progress_callback=progress_events.append,
        )

    error = excinfo.value
    assert error.reason == "startup_timeout"
    assert error.startup_timeout_seconds == 38
    assert error.inactivity_timeout_seconds == 18
    assert error.total_timeout_seconds == 38
    assert progress_events[0]["startup_timeout"] == 38
    assert progress_events[0]["inactivity_timeout"] == 18
    assert progress_events[0]["total_timeout"] == 38


def test_ollama_client_caps_num_ctx_for_large_models(monkeypatch, tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen3-coder:30b",
        llm_timeout=25,
        llm_request_retries=0,
    )
    client = OllamaClient(config)
    captured: dict[str, object] = {}

    def fake_urlopen(req: request.Request, timeout):
        del timeout
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeReadlineStreamingResponse(
            [
                (json.dumps({"response": "OK", "done": False}) + "\n").encode("utf-8"),
                (json.dumps({"response": "", "done": True}) + "\n").encode("utf-8"),
            ]
        )

    monkeypatch.setattr("llm.ollama_client.request.urlopen", fake_urlopen)

    result = client.generate(
        "Reply with OK only.",
        timeout=25,
        total_timeout=50,
        num_ctx=4096,
        retries=0,
    )

    assert result == "OK"
    assert captured["payload"]["options"]["num_ctx"] == 2048
