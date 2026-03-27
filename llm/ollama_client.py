from __future__ import annotations

import json
import socket
import time
from collections.abc import Callable
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
    ):
        super().__init__(message)
        self.reason = reason
        self.elapsed = elapsed
        self.idle_for = idle_for
        self.characters = max(int(characters), 0)
        self.activity_count = max(int(activity_count), 0)
        self.partial_text = str(partial_text or "")
        self.retryable = bool(retryable)

    @property
    def timed_out(self) -> bool:
        return self.reason in {"startup_timeout", "inactivity_timeout", "total_timeout"}

    @property
    def progress_seen(self) -> bool:
        return bool(self.partial_text) or self.characters > 0 or self.activity_count > 0


class OllamaClient:
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
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        num_ctx: int | None = None,
    ) -> str:
        effective_model = str(model or self.config.model_name)
        inactivity_timeout = max(int(timeout or self.config.llm_timeout), 1)
        overall_timeout = max(int(total_timeout or max(inactivity_timeout * 3, inactivity_timeout + 90)), inactivity_timeout)
        effective_retries = self.config.llm_request_retries if retries is None else max(int(retries), 0)
        payload: dict[str, Any] = {
            "model": effective_model,
            "prompt": prompt,
            "stream": True,
            "think": False,
            "options": {
                "temperature": self.config.ollama_temperature,
                "num_ctx": num_ctx or self.config.ollama_num_ctx,
            },
        }
        if system:
            payload["system"] = system
        if expect_json:
            payload["format"] = "json"

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
            try:
                with request.urlopen(
                    req,
                    timeout=self._initial_response_timeout(
                        inactivity_timeout=inactivity_timeout,
                        total_timeout=overall_timeout,
                    ),
                ) as response:
                    return self._read_generate_response(
                        response,
                        inactivity_timeout=inactivity_timeout,
                        total_timeout=overall_timeout,
                        progress_callback=progress_callback,
                    )
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise OllamaClientError(f"Ollama HTTP {exc.code}: {detail}") from exc
            except (error.URLError, TimeoutError, socket.timeout, OllamaClientError) as exc:
                normalized = self._normalize_generation_exception(
                    exc,
                    target=target,
                    started_at=attempt_started_at,
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

    def _read_generate_response(
        self,
        response,
        *,
        inactivity_timeout: int,
        total_timeout: int,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        if not hasattr(response, "readline"):
            return self._read_generate_response_fallback(response)

        poll_timeout = max(2.0, min(float(inactivity_timeout) / 3.0, 12.0))
        self._set_socket_timeout(response, poll_timeout)
        started_at = time.monotonic()
        last_activity_at = started_at
        last_heartbeat_at = started_at
        streamed_parts: list[str] = []
        total_characters = 0
        activity_count = 0

        while True:
            try:
                raw_line = response.readline()
            except socket.timeout as exc:
                now = time.monotonic()
                idle_for = now - last_activity_at
                if idle_for >= inactivity_timeout:
                    raise OllamaGenerationError(
                        f"timed out waiting for model progress after {idle_for:.1f} seconds",
                        reason="inactivity_timeout",
                        elapsed=round(now - started_at, 1),
                        idle_for=round(idle_for, 1),
                        characters=total_characters,
                        activity_count=activity_count,
                        partial_text="".join(streamed_parts),
                    ) from exc
                if now - started_at >= total_timeout:
                    raise OllamaGenerationError(
                        f"timed out waiting for model completion after {now - started_at:.1f} seconds",
                        reason="total_timeout",
                        elapsed=round(now - started_at, 1),
                        idle_for=round(idle_for, 1),
                        characters=total_characters,
                        activity_count=activity_count,
                        partial_text="".join(streamed_parts),
                    ) from exc
                if progress_callback is not None and now - last_heartbeat_at >= min(10.0, inactivity_timeout):
                    progress_callback(
                        {
                            "type": "heartbeat",
                            "elapsed": round(now - started_at, 1),
                            "idle_for": round(idle_for, 1),
                            "characters": total_characters,
                        }
                    )
                    last_heartbeat_at = now
                continue

            if not raw_line:
                break

            last_activity_at = time.monotonic()
            activity_count += 1
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
                        }
                    )
            if payload.get("done") is True:
                break

        if streamed_parts:
            return "".join(streamed_parts).strip()

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
        return "timed out" in str(exc).lower() or "timeout" in str(exc).lower()

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
        inactivity_timeout: int,
        total_timeout: int,
    ) -> int:
        # First token latency on slow local models is often much longer than the
        # steady-state inactivity window once streaming has started.
        return max(
            inactivity_timeout,
            min(total_timeout, max(inactivity_timeout + 20, 30)),
        )

    def _normalize_generation_exception(
        self,
        exc: Exception,
        *,
        target: str,
        started_at: float,
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
                )
            return OllamaClientError(f"Could not reach Ollama at {target}: {exc}")
        if isinstance(exc, (TimeoutError, socket.timeout)):
            elapsed = round(time.monotonic() - started_at, 1)
            return OllamaGenerationError(
                f"timed out waiting for the model to start streaming after {elapsed:.1f} seconds",
                reason="startup_timeout",
                elapsed=elapsed,
            )
        if isinstance(exc, OllamaClientError) and self._is_timeout_like(exc):
            elapsed = round(time.monotonic() - started_at, 1)
            return OllamaGenerationError(
                str(exc),
                reason="startup_timeout",
                elapsed=elapsed,
            )
        return exc
