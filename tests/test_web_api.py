from __future__ import annotations

import time

from fastapi.testclient import TestClient

from config.settings import AppConfig
from server.app import create_app


def test_web_root_serves_gui(tmp_path):
    config = AppConfig(workspace_root=str(tmp_path), ollama_host="http://127.0.0.1:9")
    config.ensure_state_dirs()
    app = create_app(config)
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Browserkonsole" in response.text


def test_start_task_and_fetch_session_via_api(tmp_path):
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")

    config = AppConfig(
        workspace_root=str(tmp_path),
        ollama_host="http://127.0.0.1:9",
        max_iterations=1,
        shell_timeout=1,
    )
    config.ensure_state_dirs()
    app = create_app(config)
    client = TestClient(app)

    create_response = client.post(
        "/api/tasks",
        json={"prompt": "Lies die README und fasse das Repo zusammen", "dry_run": True},
    )

    assert create_response.status_code == 202
    session_id = create_response.json()["id"]

    deadline = time.time() + 5
    session_payload = None
    while time.time() < deadline:
        session_response = client.get(f"/api/sessions/{session_id}")
        assert session_response.status_code == 200
        session_payload = session_response.json()
        if session_payload["status"] in {"completed", "partial", "failed"}:
            break
        time.sleep(0.1)

    assert session_payload is not None
    assert session_payload["id"] == session_id
    assert session_payload["status"] in {"completed", "partial"}

    logs_response = client.get(f"/api/sessions/{session_id}/logs")
    assert logs_response.status_code == 200
    assert len(logs_response.json()) >= 1

    sessions_response = client.get("/api/sessions")
    assert sessions_response.status_code == 200
    assert any(item["id"] == session_id for item in sessions_response.json())
