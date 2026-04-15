from __future__ import annotations

from agent.models import SessionState
from agent.session import SessionStore
import agent.models as models_module
import runtime.logger as logger_module


def test_agent_logger_activity_heartbeat_is_throttled(tmp_path, monkeypatch):
    heartbeat_calls: list[str] = []
    ticks = iter([0.0, 10.0, 30.0, 76.0])
    monkeypatch.setattr(logger_module, "monotonic", lambda: next(ticks))

    logger = logger_module.AgentLogger(
        tmp_path,
        "heartbeat-throttle",
        activity_heartbeat=lambda: heartbeat_calls.append("beat"),
        activity_heartbeat_interval_seconds=45,
    )

    logger.log_event("content_generation_progress", stage="first")
    logger.log_event("content_generation_progress", stage="second")
    logger.log_event("content_generation_progress", stage="third")

    assert heartbeat_calls == ["beat"]


def test_agent_logger_activity_heartbeat_persists_session_progress(tmp_path, monkeypatch):
    session_store = SessionStore(tmp_path / "sessions")
    session = SessionState(
        task="Build the website",
        status="running",
        workspace_root=str(tmp_path),
    )
    session.updated_at = "2026-04-15T09:00:00+00:00"
    session_store.save(session)

    ticks = iter([0.0, 20.0])
    monkeypatch.setattr(logger_module, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(models_module, "utc_now", lambda: "2026-04-15T09:00:20+00:00")

    def persist_activity() -> None:
        session.touch()
        session_store.save(session)

    logger = logger_module.AgentLogger(
        tmp_path / "logs",
        session.id,
        activity_heartbeat=persist_activity,
        activity_heartbeat_interval_seconds=15,
    )

    logger.log_event("content_generation_progress", stage="streaming")

    reloaded = session_store.load(session.id)

    assert reloaded is not None
    assert reloaded.updated_at == "2026-04-15T09:00:20+00:00"
