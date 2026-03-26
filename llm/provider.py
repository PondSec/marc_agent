from __future__ import annotations

from typing import Any, Protocol


class LLMProvider(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        expect_json: bool = False,
        timeout: int | None = None,
        num_ctx: int | None = None,
    ) -> str: ...

    def generate_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        timeout: int | None = None,
        num_ctx: int | None = None,
    ) -> dict[str, Any]: ...
