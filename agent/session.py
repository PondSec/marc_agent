from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from agent.models import SessionState


class SessionStore:
    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.last_session_file = self.session_dir / "last_session.txt"

    def save(self, session: SessionState) -> None:
        target = self.session_dir / f"{session.id}.json"
        temp_target = self.session_dir / f".{session.id}.{uuid4().hex}.tmp"
        temp_target.write_text(session.model_dump_json(indent=2), encoding="utf-8")
        temp_target.replace(target)
        self.last_session_file.write_text(session.id, encoding="utf-8")

    def load(self, session_id: str | None) -> SessionState | None:
        if not session_id:
            return None
        target = self.session_dir / f"{session_id}.json"
        if not target.exists():
            return None
        return SessionState.model_validate_json(target.read_text(encoding="utf-8"))

    def load_last(self) -> SessionState | None:
        if not self.last_session_file.exists():
            return None
        session_id = self.last_session_file.read_text(encoding="utf-8").strip()
        return self.load(session_id)

    def delete(self, session_id: str) -> bool:
        target = self.session_dir / f"{session_id}.json"
        if not target.exists():
            return False
        target.unlink()
        self._refresh_last_session(session_id)
        return True

    def list_sessions(self, limit: int = 100) -> list[SessionState]:
        session_files = sorted(
            self.session_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        sessions: list[SessionState] = []
        for path in session_files[:limit]:
            sessions.append(
                SessionState.model_validate_json(path.read_text(encoding="utf-8"))
            )
        return sessions

    def _refresh_last_session(self, deleted_session_id: str) -> None:
        if not self.last_session_file.exists():
            return
        last_session_id = self.last_session_file.read_text(encoding="utf-8").strip()
        if last_session_id != deleted_session_id:
            return

        remaining = self.list_sessions(limit=1)
        if remaining:
            self.last_session_file.write_text(remaining[0].id, encoding="utf-8")
            return

        self.last_session_file.unlink(missing_ok=True)
