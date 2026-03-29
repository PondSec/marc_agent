from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol


ProgressCallback = Callable[[dict[str, Any]], None]


class LLMProvider(Protocol):
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
        progress_callback: ProgressCallback | None = None,
        num_ctx: int | None = None,
    ) -> str: ...

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
        progress_callback: ProgressCallback | None = None,
        num_ctx: int | None = None,
    ) -> dict[str, Any]: ...
