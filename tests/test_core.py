from __future__ import annotations

from agent.core import AgentCore
from agent.models import FileChangeRecord, SessionState
from config.settings import AppConfig


def test_core_marks_unvalidated_changed_files_as_partial(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path))
    config.ensure_state_dirs()
    core = AgentCore(config)
    session = SessionState(
        task="Implementiere etwas",
        workspace_root=str(tmp_path),
        validation_status="not_run",
    )
    session.changed_files.append(FileChangeRecord(path="game.py", operation="create"))

    status = core._resolve_final_status(session, final_action=True)
    stop_reason = core._derive_stop_reason(session)

    assert status == "partial"
    assert stop_reason == "validation_missing"
