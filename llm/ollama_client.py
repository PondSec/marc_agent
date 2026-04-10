from __future__ import annotations

import json
import queue
import re
import socket
import subprocess
import time
from collections.abc import Callable
from threading import Event, Thread
from typing import Any, Iterator
from urllib import error, request

from config.settings import AppConfig


class OllamaClientError(RuntimeError):
    pass


class OllamaGenerationError(OllamaClientError):
    def __init__(
        self,
        message: str,
        *,
        reason: str,
        elapsed: float | None = None,
        idle_for: float | None = None,
        characters: int = 0,
        activity_count: int = 0,
        partial_text: str = "",
        retryable: bool = True,
        model_name: str | None = None,
        backend_identifier: str = "ollama",
        startup_timeout_seconds: int | None = None,
        inactivity_timeout_seconds: int | None = None,
        total_timeout_seconds: int | None = None,
        first_output_received: bool | None = None,
    ):
        super().__init__(message)
        self.reason = reason
        self.elapsed = elapsed
        self.idle_for = idle_for
        self.characters = max(int(characters), 0)
        self.activity_count = max(int(activity_count), 0)
        self.partial_text = str(partial_text or "")
        self.retryable = bool(retryable)
        self.model_name = str(model_name or "").strip() or None
        self.backend_identifier = str(backend_identifier or "ollama").strip() or "ollama"
        self.startup_timeout_seconds = (
            max(int(startup_timeout_seconds), 1)
            if startup_timeout_seconds is not None
            else None
        )
        self.inactivity_timeout_seconds = (
            max(int(inactivity_timeout_seconds), 1)
            if inactivity_timeout_seconds is not None
            else None
        )
        self.total_timeout_seconds = (
            max(int(total_timeout_seconds), 1)
            if total_timeout_seconds is not None
            else None
        )
        self.first_output_received = (
            bool(first_output_received)
            if first_output_received is not None
            else bool(self.partial_text) or self.characters > 0
        )

    @property
    def timed_out(self) -> bool:
        return self.reason in {"startup_timeout", "inactivity_timeout", "total_timeout"}

    @property
    def progress_seen(self) -> bool:
        return bool(self.partial_text) or self.characters > 0 or self.activity_count > 0

    @property
    def no_start_failure(self) -> bool:
        return self.reason == "startup_timeout" and not self.progress_seen and self.characters <= 0


class OllamaClient:
    JSON_NUM_PREDICT_LIMIT = 768

    def __init__(self, config: AppConfig):
        self.config = config

    def list_models(self) -> list[dict[str, Any]]:
        target = self.config.ollama_host.rstrip("/") + "/api/tags"
        req = request.Request(target, method="GET")
        timeout = min(self.config.llm_timeout, 5)

        try:
            with request.urlopen(req, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaClientError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise OllamaClientError(f"Could not reach Ollama at {target}: {exc}") from exc

        data = json.loads(raw)
        models = data.get("models", [])
        normalized: list[dict[str, Any]] = []
        for item in models:
            name = item.get("name") or item.get("model")
            if not name:
                continue
            details = item.get("details") or {}
            normalized.append(
                {
                    "name": str(name),
                    "size": item.get("size"),
                    "modified_at": item.get("modified_at"),
                    "family": details.get("family"),
                    "parameter_size": details.get("parameter_size"),
                }
            )
        return normalized

    def list_models_safe(self) -> list[dict[str, Any]]:
        try:
            return self.list_models()
        except OllamaClientError:
            return []

    def is_available(self) -> bool:
        try:
            self.list_models()
        except OllamaClientError:
            return False
        return True

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        expect_json: bool = False,
        model: str | None = None,
        retries: int | None = None,
        timeout: int | None = None,
        total_timeout: int | None = None,
        strict_timeouts: bool = False,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        num_ctx: int | None = None,
    ) -> str:
        effective_model = str(model or self.config.model_name)
        inactivity_timeout = max(int(timeout or self.config.llm_timeout), 1)
        if not strict_timeouts:
            inactivity_timeout = self._expanded_inactivity_timeout(
                model_name=effective_model,
                inactivity_timeout=inactivity_timeout,
            )
        overall_timeout = max(
            int(total_timeout or max(inactivity_timeout * 3, inactivity_timeout + 90)),
            inactivity_timeout,
        )
        if not strict_timeouts:
            overall_timeout = self._expanded_total_timeout(
                model_name=effective_model,
                inactivity_timeout=inactivity_timeout,
                total_timeout=overall_timeout,
            )
        startup_timeout = self._startup_stream_timeout(
            model_name=effective_model,
            inactivity_timeout=inactivity_timeout,
            total_timeout=overall_timeout,
        )
        effective_num_ctx = self._effective_num_ctx(
            model_name=effective_model,
            requested_num_ctx=num_ctx or self.config.ollama_num_ctx,
        )
        effective_retries = self.config.llm_request_retries if retries is None else max(int(retries), 0)
        payload: dict[str, Any] = {
            "model": effective_model,
            "prompt": prompt,
            "stream": True,
            "think": False,
            "options": {
                "temperature": self.config.ollama_temperature,
                "num_ctx": effective_num_ctx,
            },
        }
        if system:
            payload["system"] = system
        if expect_json:
            payload["format"] = "json"
            # Structured semantic calls should stay compact and finish. Without a
            # bounded generation budget, small local models can drift into long
            # invalid JSON streams that never close the object cleanly.
            payload["options"]["num_predict"] = self.JSON_NUM_PREDICT_LIMIT

        target = self.config.ollama_host.rstrip("/") + "/api/generate"
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            target,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        last_exception: Exception | None = None
        for attempt in range(effective_retries + 1):
            attempt_started_at = time.monotonic()
            if progress_callback is not None:
                progress_callback(
                    {
                        "type": "status",
                        "stage": "request_started",
                        "attempt": attempt + 1,
                        "model": effective_model,
                        "startup_timeout": startup_timeout,
                        "inactivity_timeout": inactivity_timeout,
                        "total_timeout": overall_timeout,
                    }
                )
            try:
                with request.urlopen(
                    req,
                    timeout=self._initial_response_timeout(startup_timeout=startup_timeout),
                ) as response:
                    headers_received_at = time.monotonic()
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "type": "status",
                                "stage": "response_headers_received",
                                "attempt": attempt + 1,
                                "model": effective_model,
                                "elapsed": round(headers_received_at - attempt_started_at, 3),
                                "time_to_header": round(headers_received_at - attempt_started_at, 3),
                                "startup_timeout": startup_timeout,
                                "inactivity_timeout": inactivity_timeout,
                                "total_timeout": overall_timeout,
                            }
                        )
                    return self._read_generate_response(
                        response,
                        model_name=effective_model,
                        request_started_at=attempt_started_at,
                        headers_received_at=headers_received_at,
                        startup_timeout=startup_timeout,
                        inactivity_timeout=inactivity_timeout,
                        total_timeout=overall_timeout,
                        expect_json=expect_json,
                        progress_callback=progress_callback,
                    )
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code in {429, 503, 504}:
                    raise OllamaGenerationError(
                        f"Ollama HTTP {exc.code}: {detail}",
                        reason="backend_overloaded",
                        elapsed=round(time.monotonic() - attempt_started_at, 1),
                        retryable=True,
                        model_name=effective_model,
                        backend_identifier="ollama",
                        startup_timeout_seconds=startup_timeout,
                        inactivity_timeout_seconds=inactivity_timeout,
                        total_timeout_seconds=overall_timeout,
                    ) from exc
                raise OllamaClientError(f"Ollama HTTP {exc.code}: {detail}") from exc
            except (error.URLError, TimeoutError, socket.timeout, OllamaClientError) as exc:
                normalized = self._normalize_generation_exception(
                    exc,
                    target=target,
                    started_at=attempt_started_at,
                    model_name=effective_model,
                    startup_timeout=startup_timeout,
                    inactivity_timeout=inactivity_timeout,
                    total_timeout=overall_timeout,
                )
                if isinstance(normalized, OllamaGenerationError) and not normalized.retryable:
                    raise normalized
                if isinstance(normalized, OllamaClientError) and not self._is_timeout_like(normalized):
                    raise normalized
                last_exception = normalized
                if attempt >= effective_retries:
                    break
                backoff = max(self.config.llm_retry_backoff_ms, 0) / 1000.0
                time.sleep(backoff * (attempt + 1))
                continue

        if isinstance(last_exception, error.URLError):
            raise OllamaClientError(f"Could not reach Ollama at {target}: {last_exception}") from last_exception
        if last_exception is not None:
            if isinstance(last_exception, OllamaClientError):
                raise last_exception
            raise OllamaClientError(str(last_exception)) from last_exception
        raise OllamaClientError(f"Unknown Ollama generation failure for model {effective_model}")

    def _stop_running_model(self, model_name: str | None) -> None:
        text = str(model_name or "").strip()
        if not text:
            return
        try:
            subprocess.run(
                ["ollama", "stop", text],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
        except Exception:  # noqa: BLE001
            pass
        self._wait_for_model_stop(text)

    def _wait_for_model_stop(
        self,
        model_name: str,
        *,
        timeout_seconds: float = 20.0,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        normalized = str(model_name or "").strip()
        if not normalized:
            return
        deadline = time.monotonic() + max(float(timeout_seconds), float(poll_interval_seconds))
        while time.monotonic() < deadline:
            running_models = self._running_model_names()
            if normalized not in running_models:
                return
            time.sleep(max(float(poll_interval_seconds), 0.1))

    @staticmethod
    def _running_model_names() -> set[str]:
        try:
            result = subprocess.run(
                ["ollama", "ps"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:  # noqa: BLE001
            return set()

        names: set[str] = set()
        for raw_line in str(result.stdout or "").splitlines():
            line = raw_line.strip()
            if not line or line.lower().startswith("name "):
                continue
            parts = line.split()
            if parts:
                names.add(parts[0])
        return names

    def _read_generate_response(
        self,
        response,
        *,
        model_name: str,
        request_started_at: float,
        headers_received_at: float,
        startup_timeout: int,
        inactivity_timeout: int,
        total_timeout: int,
        expect_json: bool = False,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        if not hasattr(response, "readline"):
            return self._read_generate_response_fallback(response)

        poll_timeout = max(2.0, min(float(inactivity_timeout) / 3.0, 12.0))
        self._set_socket_timeout(response, poll_timeout)
        line_queue, reader_stop, reader_thread = self._start_stream_reader(response)
        started_at = request_started_at
        last_activity_at = started_at
        last_heartbeat_at = headers_received_at
        streamed_parts: list[str] = []
        total_characters = 0
        activity_count = 0
        startup_warning_sent = False
        stream_timeout_promoted = False
        first_chunk_announced = False

        if progress_callback is not None:
            progress_callback(
                {
                    "type": "status",
                    "stage": "waiting_for_first_chunk",
                    "model": model_name,
                    "elapsed": round(headers_received_at - started_at, 3),
                    "time_to_header": round(headers_received_at - started_at, 3),
                    "startup_timeout": startup_timeout,
                    "inactivity_timeout": inactivity_timeout,
                    "total_timeout": total_timeout,
                }
            )

        try:
            while True:
                try:
                    raw_line = self._wait_for_stream_line(
                        line_queue,
                        response=response,
                        model_name=model_name,
                        started_at=started_at,
                        last_activity_at=last_activity_at,
                        last_heartbeat_at=last_heartbeat_at,
                        activity_count=activity_count,
                        streamed_parts=streamed_parts,
                        total_characters=total_characters,
                        startup_timeout=startup_timeout,
                        inactivity_timeout=inactivity_timeout,
                        total_timeout=total_timeout,
                        startup_warning_sent=startup_warning_sent,
                        progress_callback=progress_callback,
                    )
                except OllamaGenerationError:
                    self._abort_stream_response(response)
                    raise
                except Exception:
                    self._abort_stream_response(response)
                    raise

                now = time.monotonic()
                if progress_callback is not None:
                    idle_for = now - last_activity_at
                    elapsed = now - started_at
                    if activity_count <= 0 and not streamed_parts:
                        if (
                            not startup_warning_sent
                            and elapsed >= max(startup_timeout - 10.0, startup_timeout * 0.75)
                        ):
                            progress_callback(
                                {
                                    "type": "status",
                                    "stage": "startup_timeout_warning",
                                    "model": model_name,
                                    "elapsed": round(elapsed, 1),
                                    "startup_timeout": startup_timeout,
                                    "seconds_remaining": round(max(startup_timeout - elapsed, 0.0), 1),
                                }
                            )
                            startup_warning_sent = True
                    elif now - last_heartbeat_at >= min(10.0, inactivity_timeout):
                        progress_callback(
                            {
                                "type": "heartbeat",
                                "elapsed": round(elapsed, 1),
                                "idle_for": round(idle_for, 1),
                                "characters": total_characters,
                                "model": model_name,
                            }
                        )
                        last_heartbeat_at = now

                if not raw_line:
                    if activity_count <= 0 and not streamed_parts:
                        elapsed = round(time.monotonic() - started_at, 1)
                        raise OllamaGenerationError(
                            f"timed out waiting for the model to start streaming after {elapsed:.1f} seconds",
                            reason="startup_timeout",
                            elapsed=elapsed,
                            idle_for=elapsed,
                            characters=total_characters,
                            activity_count=activity_count,
                            partial_text="".join(streamed_parts),
                            retryable=True,
                            model_name=model_name,
                            backend_identifier="ollama",
                            startup_timeout_seconds=startup_timeout,
                            inactivity_timeout_seconds=inactivity_timeout,
                            total_timeout_seconds=total_timeout,
                            first_output_received=False,
                        )
                    break

                last_activity_at = time.monotonic()
                activity_count += 1
                if progress_callback is not None and not first_chunk_announced:
                    progress_callback(
                        {
                            "type": "status",
                            "stage": "first_chunk_received",
                            "model": model_name,
                            "elapsed": round(last_activity_at - started_at, 3),
                            "time_to_header": round(headers_received_at - started_at, 3),
                            "time_to_first_chunk": round(last_activity_at - started_at, 3),
                            "startup_timeout": startup_timeout,
                            "inactivity_timeout": inactivity_timeout,
                            "total_timeout": total_timeout,
                        }
                    )
                    first_chunk_announced = True
                if not stream_timeout_promoted:
                    self._set_socket_timeout(response, float(inactivity_timeout))
                    stream_timeout_promoted = True
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    continue
                if payload.get("error"):
                    raise OllamaGenerationError(
                        str(payload["error"]),
                        reason="provider_error",
                        elapsed=round(last_activity_at - started_at, 1),
                        characters=total_characters,
                        activity_count=activity_count,
                        partial_text="".join(streamed_parts),
                        retryable=False,
                        model_name=model_name,
                        backend_identifier="ollama",
                        startup_timeout_seconds=startup_timeout,
                        inactivity_timeout_seconds=inactivity_timeout,
                        total_timeout_seconds=total_timeout,
                        first_output_received=bool(streamed_parts),
                    )
                chunk = payload.get("response")
                if chunk:
                    piece = str(chunk)
                    streamed_parts.append(piece)
                    total_characters += len(piece)
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "type": "chunk",
                                "elapsed": round(last_activity_at - started_at, 1),
                                "characters": total_characters,
                                "model": model_name,
                            }
                        )
                    if expect_json and "}" in piece:
                        completed_json = self._try_coerce_json_object("".join(streamed_parts))
                        if completed_json is not None:
                            return json.dumps(completed_json, ensure_ascii=False)
                if payload.get("done") is True:
                    break
                elapsed = round(last_activity_at - started_at, 1)
                if elapsed >= total_timeout:
                    raise OllamaGenerationError(
                        f"timed out waiting for model completion after {elapsed:.1f} seconds",
                        reason="total_timeout",
                        elapsed=elapsed,
                        idle_for=0.0,
                        characters=total_characters,
                        activity_count=activity_count,
                        partial_text="".join(streamed_parts),
                        model_name=model_name,
                        backend_identifier="ollama",
                        startup_timeout_seconds=startup_timeout,
                        inactivity_timeout_seconds=inactivity_timeout,
                        total_timeout_seconds=total_timeout,
                        first_output_received=bool(streamed_parts),
                    )
        finally:
            reader_stop.set()
            self._abort_stream_response(response)
            reader_thread.join(timeout=1)

        if streamed_parts:
            if progress_callback is not None:
                finished_at = time.monotonic()
                progress_callback(
                    {
                        "type": "status",
                        "stage": "response_completed",
                        "model": model_name,
                        "elapsed": round(finished_at - started_at, 3),
                        "time_to_header": round(headers_received_at - started_at, 3),
                        "time_to_last_chunk": round(finished_at - started_at, 3),
                        "characters": total_characters,
                    }
                )
            return "".join(streamed_parts).strip()

        if not hasattr(response, "read"):
            elapsed = round(time.monotonic() - started_at, 1)
            raise OllamaGenerationError(
                f"timed out waiting for the model to start streaming after {elapsed:.1f} seconds",
                reason="startup_timeout",
                elapsed=elapsed,
                idle_for=elapsed,
                characters=total_characters,
                activity_count=activity_count,
                partial_text="".join(streamed_parts),
                retryable=True,
                model_name=model_name,
                backend_identifier="ollama",
                startup_timeout_seconds=startup_timeout,
                inactivity_timeout_seconds=inactivity_timeout,
                total_timeout_seconds=total_timeout,
                first_output_received=False,
            )

        raw = response.read().decode("utf-8")
        data = json.loads(raw)
        if "error" in data:
            raise OllamaClientError(str(data["error"]))
        return str(data.get("response", "")).strip()

    def _read_generate_response_fallback(self, response) -> str:
        streamed_parts: list[str] = []
        saw_stream = False
        try:
            for raw_line in response:
                saw_stream = True
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    continue
                if payload.get("error"):
                    raise OllamaClientError(str(payload["error"]))
                chunk = payload.get("response")
                if chunk:
                    streamed_parts.append(str(chunk))
        except TypeError:
            saw_stream = False

        if saw_stream:
            return "".join(streamed_parts).strip()

        raw = response.read().decode("utf-8")
        data = json.loads(raw)
        if "error" in data:
            raise OllamaClientError(str(data["error"]))
        return str(data.get("response", "")).strip()

    def generate_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        retries: int | None = None,
        timeout: int | None = None,
        total_timeout: int | None = None,
        strict_timeouts: bool = False,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        num_ctx: int | None = None,
    ) -> dict[str, Any]:
        text = self.generate(
            prompt,
            system=system,
            expect_json=True,
            model=model,
            retries=retries,
            timeout=timeout,
            total_timeout=total_timeout,
            strict_timeouts=strict_timeouts,
            progress_callback=progress_callback,
            num_ctx=num_ctx,
        )
        return self._coerce_json_object(text)

    @staticmethod
    def _coerce_json_object(text: str) -> dict[str, Any]:
        candidates = [text.strip()]
        if "```json" in text:
            for block in text.split("```json")[1:]:
                candidates.append(block.split("```", 1)[0].strip())
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            candidates.append(text[start:end].strip())

        for candidate in candidates:
            if not candidate:
                continue
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        raise OllamaClientError(f"Model did not return valid JSON: {text[:500]}")

    @staticmethod
    def _try_coerce_json_object(text: str) -> dict[str, Any] | None:
        try:
            return OllamaClient._coerce_json_object(text)
        except OllamaClientError:
            return None

    def pull_model_events(self, model_name: str) -> Iterator[dict[str, Any]]:
        payload = {
            "name": model_name,
            "stream": True,
        }
        target = self.config.ollama_host.rstrip("/") + "/api/pull"
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            target,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=max(self.config.shell_timeout, 900)) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        yield payload
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaClientError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise OllamaClientError(f"Could not reach Ollama at {target}: {exc}") from exc

    @staticmethod
    def _is_timeout_like(exc: Exception) -> bool:
        if isinstance(exc, OllamaGenerationError):
            return exc.timed_out
        if isinstance(exc, (TimeoutError, socket.timeout)):
            return True
        return "timed out" in str(exc).lower() or "timeout" in str(exc).lower()

    @staticmethod
    def _start_stream_reader(response) -> tuple[queue.Queue[tuple[str, Any]], Event, Thread]:
        line_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        stop_event = Event()

        def _reader() -> None:
            try:
                while not stop_event.is_set():
                    raw_line = response.readline()
                    line_queue.put(("line", raw_line))
                    if not raw_line:
                        return
            except Exception as exc:  # noqa: BLE001
                line_queue.put(("error", exc))

        thread = Thread(target=_reader, daemon=True)
        thread.start()
        return line_queue, stop_event, thread

    def _wait_for_stream_line(
        self,
        line_queue: queue.Queue[tuple[str, Any]],
        *,
        response,
        model_name: str,
        started_at: float,
        last_activity_at: float,
        last_heartbeat_at: float,
        activity_count: int,
        streamed_parts: list[str],
        total_characters: int,
        startup_timeout: int,
        inactivity_timeout: int,
        total_timeout: int,
        startup_warning_sent: bool,
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> bytes:
        poll_timeout = max(2.0, min(float(inactivity_timeout) / 3.0, 12.0))
        while True:
            try:
                kind, payload = line_queue.get(timeout=poll_timeout)
            except queue.Empty:
                now = time.monotonic()
                idle_for = now - last_activity_at
                elapsed = now - started_at
                if activity_count <= 0 and not streamed_parts:
                    if (
                        progress_callback is not None
                        and not startup_warning_sent
                        and elapsed >= max(startup_timeout - 10.0, startup_timeout * 0.75)
                    ):
                        progress_callback(
                            {
                                "type": "status",
                                "stage": "startup_timeout_warning",
                                "model": model_name,
                                "elapsed": round(elapsed, 1),
                                "startup_timeout": startup_timeout,
                                "seconds_remaining": round(max(startup_timeout - elapsed, 0.0), 1),
                            }
                        )
                    if progress_callback is not None and now - last_heartbeat_at >= min(10.0, max(startup_timeout / 4.0, 4.0)):
                        progress_callback(
                            {
                                "type": "heartbeat",
                                "phase": "waiting_for_start",
                                "model": model_name,
                                "elapsed": round(elapsed, 1),
                                "idle_for": round(idle_for, 1),
                                "characters": total_characters,
                                "startup_timeout": startup_timeout,
                                "seconds_remaining": round(max(startup_timeout - elapsed, 0.0), 1),
                            }
                        )
                    if elapsed >= startup_timeout:
                        self._abort_stream_response(response)
                        raise OllamaGenerationError(
                            f"timed out waiting for the model to start streaming after {elapsed:.1f} seconds",
                            reason="startup_timeout",
                            elapsed=round(elapsed, 1),
                            idle_for=round(idle_for, 1),
                            characters=total_characters,
                            activity_count=activity_count,
                            partial_text="".join(streamed_parts),
                            retryable=True,
                            model_name=model_name,
                            backend_identifier="ollama",
                            startup_timeout_seconds=startup_timeout,
                            inactivity_timeout_seconds=inactivity_timeout,
                            total_timeout_seconds=total_timeout,
                            first_output_received=False,
                        )
                    continue
                if idle_for >= inactivity_timeout:
                    self._abort_stream_response(response)
                    raise OllamaGenerationError(
                        f"timed out waiting for model progress after {idle_for:.1f} seconds",
                        reason="inactivity_timeout",
                        elapsed=round(elapsed, 1),
                        idle_for=round(idle_for, 1),
                        characters=total_characters,
                        activity_count=activity_count,
                        partial_text="".join(streamed_parts),
                        model_name=model_name,
                        backend_identifier="ollama",
                        startup_timeout_seconds=startup_timeout,
                        inactivity_timeout_seconds=inactivity_timeout,
                        total_timeout_seconds=total_timeout,
                        first_output_received=bool(streamed_parts),
                    )
                if elapsed >= total_timeout:
                    self._abort_stream_response(response)
                    raise OllamaGenerationError(
                        f"timed out waiting for model completion after {elapsed:.1f} seconds",
                        reason="total_timeout",
                        elapsed=round(elapsed, 1),
                        idle_for=round(idle_for, 1),
                        characters=total_characters,
                        activity_count=activity_count,
                        partial_text="".join(streamed_parts),
                        model_name=model_name,
                        backend_identifier="ollama",
                        startup_timeout_seconds=startup_timeout,
                        inactivity_timeout_seconds=inactivity_timeout,
                        total_timeout_seconds=total_timeout,
                        first_output_received=bool(streamed_parts),
                    )
                continue
            if kind == "error":
                exc = payload
                if self._is_timeout_like(exc):
                    continue
                raise exc
            return payload

    @staticmethod
    def _abort_stream_response(response) -> None:
        try:
            response.close()
        except Exception:  # noqa: BLE001
            pass
        target = getattr(response, "fp", None)
        target = getattr(target, "raw", target)
        sock = getattr(target, "_sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    @staticmethod
    def _set_socket_timeout(response, timeout_seconds: float) -> None:
        target = getattr(response, "fp", None)
        target = getattr(target, "raw", target)
        sock = getattr(target, "_sock", None)
        if sock is None:
            return
        try:
            sock.settimeout(timeout_seconds)
        except OSError:
            return

    @staticmethod
    def _initial_response_timeout(
        *,
        startup_timeout: int,
    ) -> int:
        return max(int(startup_timeout), 1)

    @staticmethod
    def _startup_stream_timeout(
        *,
        model_name: str,
        inactivity_timeout: int,
        total_timeout: int,
    ) -> int:
        parameter_hint = OllamaClient._parameter_size_hint(model_name)
        extra_buffer = 20
        minimum_floor = 30
        startup_share_numerator = 0
        startup_share_denominator = 1
        if parameter_hint >= 24:
            extra_buffer = 60
            minimum_floor = 480
        elif parameter_hint >= 14:
            extra_buffer = 40
            minimum_floor = 130
        elif parameter_hint >= 7:
            extra_buffer = 40
            minimum_floor = 100
            startup_share_numerator = 2
            startup_share_denominator = 3
        elif parameter_hint > 0:
            extra_buffer = 30
            minimum_floor = 60
            startup_share_numerator = 3
            startup_share_denominator = 5
        shared_floor = 0
        if startup_share_numerator > 0:
            # Smaller local models often spend most of a long recovery budget before
            # the first token arrives, so let startup consume a larger share there.
            shared_floor = max(
                (int(total_timeout) * startup_share_numerator) // startup_share_denominator,
                int(inactivity_timeout),
            )
        return int(
            min(
                total_timeout,
                max(inactivity_timeout + extra_buffer, minimum_floor, shared_floor),
            )
        )

    @staticmethod
    def _expanded_total_timeout(
        *,
        model_name: str,
        inactivity_timeout: int,
        total_timeout: int,
    ) -> int:
        parameter_hint = OllamaClient._parameter_size_hint(model_name)
        adjusted_total = max(int(total_timeout), int(inactivity_timeout), 1)
        if parameter_hint >= 24:
            adjusted_total = max(adjusted_total, 1200)
        elif parameter_hint >= 14:
            adjusted_total = max(adjusted_total, 150)
        elif parameter_hint >= 7:
            adjusted_total = max(adjusted_total, 120)
        return adjusted_total

    @staticmethod
    def _expanded_inactivity_timeout(
        *,
        model_name: str,
        inactivity_timeout: int,
    ) -> int:
        parameter_hint = OllamaClient._parameter_size_hint(model_name)
        adjusted_timeout = max(int(inactivity_timeout), 1)
        if parameter_hint >= 24:
            adjusted_timeout = max(adjusted_timeout, 240)
        elif parameter_hint >= 14:
            adjusted_timeout = max(adjusted_timeout, 45)
        return adjusted_timeout

    @staticmethod
    def _effective_num_ctx(
        *,
        model_name: str,
        requested_num_ctx: int,
    ) -> int:
        parameter_hint = OllamaClient._parameter_size_hint(model_name)
        effective_num_ctx = max(int(requested_num_ctx), 256)
        if parameter_hint >= 24:
            effective_num_ctx = min(effective_num_ctx, 2048)
        return effective_num_ctx

    @staticmethod
    def _parameter_size_hint(model_name: str) -> float:
        match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", str(model_name or "").lower())
        if not match:
            return 0.0
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0

    def _normalize_generation_exception(
        self,
        exc: Exception,
        *,
        target: str,
        started_at: float,
        model_name: str,
        startup_timeout: int,
        inactivity_timeout: int,
        total_timeout: int,
    ) -> Exception:
        if isinstance(exc, OllamaGenerationError):
            return exc
        if isinstance(exc, error.URLError):
            if self._is_timeout_like(exc):
                elapsed = round(time.monotonic() - started_at, 1)
                return OllamaGenerationError(
                    f"timed out waiting for the model to start streaming after {elapsed:.1f} seconds",
                    reason="startup_timeout",
                    elapsed=elapsed,
                    retryable=True,
                    model_name=model_name,
                    backend_identifier="ollama",
                    startup_timeout_seconds=startup_timeout,
                    inactivity_timeout_seconds=inactivity_timeout,
                    total_timeout_seconds=total_timeout,
                    first_output_received=False,
                )
            elapsed = round(time.monotonic() - started_at, 1)
            return OllamaGenerationError(
                f"Could not reach Ollama at {target}: {exc}",
                reason="backend_unavailable",
                elapsed=elapsed,
                retryable=True,
                model_name=model_name,
                backend_identifier="ollama",
                startup_timeout_seconds=startup_timeout,
                inactivity_timeout_seconds=inactivity_timeout,
                total_timeout_seconds=total_timeout,
                first_output_received=False,
            )
        if isinstance(exc, (TimeoutError, socket.timeout)):
            elapsed = round(time.monotonic() - started_at, 1)
            return OllamaGenerationError(
                f"timed out waiting for the model to start streaming after {elapsed:.1f} seconds",
                reason="startup_timeout",
                elapsed=elapsed,
                retryable=True,
                model_name=model_name,
                backend_identifier="ollama",
                startup_timeout_seconds=startup_timeout,
                inactivity_timeout_seconds=inactivity_timeout,
                total_timeout_seconds=total_timeout,
                first_output_received=False,
            )
        if isinstance(exc, OllamaClientError) and self._is_timeout_like(exc):
            elapsed = round(time.monotonic() - started_at, 1)
            return OllamaGenerationError(
                str(exc),
                reason="startup_timeout",
                elapsed=elapsed,
                retryable=True,
                model_name=model_name,
                backend_identifier="ollama",
                startup_timeout_seconds=startup_timeout,
                inactivity_timeout_seconds=inactivity_timeout,
                total_timeout_seconds=total_timeout,
                first_output_received=False,
            )
        return exc
