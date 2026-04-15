from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Callable


class AgentLogger:
    def __init__(
        self,
        log_dir: Path,
        session_id: str,
        verbose: bool = False,
        *,
        activity_heartbeat: Callable[[], None] | None = None,
        activity_heartbeat_interval_seconds: float = 15.0,
    ):
        self.verbose = verbose
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"{session_id}.jsonl"
        self.activity_heartbeat = activity_heartbeat
        self.activity_heartbeat_interval_seconds = max(
            float(activity_heartbeat_interval_seconds),
            0.0,
        )
        self._last_activity_heartbeat_at = monotonic()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _maybe_heartbeat_activity(self) -> None:
        if self.activity_heartbeat is None:
            return
        now = monotonic()
        if (now - self._last_activity_heartbeat_at) < self.activity_heartbeat_interval_seconds:
            return
        self._last_activity_heartbeat_at = now
        try:
            self.activity_heartbeat()
        except Exception:
            return

    def log_event(self, event: str, **payload: Any) -> None:
        record = {"timestamp": self._timestamp(), "event": event, "payload": payload}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._maybe_heartbeat_activity()
        if self.verbose:
            print(f"[{event}] {json.dumps(payload, ensure_ascii=False, default=str)[:500]}")
