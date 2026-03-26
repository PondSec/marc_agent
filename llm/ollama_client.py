from __future__ import annotations

import json
from typing import Any
from urllib import error, request

from config.settings import AppConfig


class OllamaClientError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, config: AppConfig):
        self.config = config

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        expect_json: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.ollama_temperature,
                "num_ctx": self.config.ollama_num_ctx,
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
            with request.urlopen(req, timeout=self.config.shell_timeout) as response:
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

    def generate_json(self, prompt: str, *, system: str | None = None) -> dict[str, Any]:
        text = self.generate(prompt, system=system, expect_json=True)
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
