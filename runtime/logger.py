from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AgentLogger:
    def __init__(self, log_dir: Path, session_id: str, verbose: bool = False):
        self.verbose = verbose
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"{session_id}.jsonl"

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def log_event(self, event: str, **payload: Any) -> None:
        record = {"timestamp": self._timestamp(), "event": event, "payload": payload}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.verbose:
            print(f"[{event}] {json.dumps(payload, ensure_ascii=False)[:500]}")
