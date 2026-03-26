from __future__ import annotations

import json
from typing import Any, Iterator
from urllib import error, request

from config.settings import AppConfig


class OllamaClientError(RuntimeError):
    pass


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

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        expect_json: bool = False,
        timeout: int | None = None,
        num_ctx: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
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

        try:
            with request.urlopen(req, timeout=timeout or self.config.llm_timeout) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaClientError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise OllamaClientError(f"Could not reach Ollama at {target}: {exc}") from exc

        data = json.loads(raw)
        if "error" in data:
            raise OllamaClientError(str(data["error"]))
        return str(data.get("response", "")).strip()

    def generate_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        timeout: int | None = None,
        num_ctx: int | None = None,
    ) -> dict[str, Any]:
        text = self.generate(
            prompt,
            system=system,
            expect_json=True,
            timeout=timeout,
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
