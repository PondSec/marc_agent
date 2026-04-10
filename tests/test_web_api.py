from __future__ import annotations

import io
import json
import os
import sqlite3
import time
import zipfile
from dataclasses import dataclass
from threading import Event
from unittest.mock import patch

from fastapi.testclient import TestClient

from agent.models import FileChangeRecord, SessionState, ToolCallRecord
from config.settings import AppConfig
from server.app import _allowed_model_candidates, _with_available_models, create_app

TEST_ORIGIN = "https://testserver"
TEST_LOCAL_ORIGIN = "http://127.0.0.1:8000"
TEST_AUTH_SECRET = "test-auth-secret-please-change"
TEST_ADMIN_EMAIL = "operator@example.com"
TEST_ADMIN_PASSWORD = "VeryStrongPassword!2026"


def build_test_config(root, **overrides) -> AppConfig:
    return AppConfig(
        workspace_root=str(root),
        ollama_host="http://127.0.0.1:9",
        auth_secret_key=TEST_AUTH_SECRET,
        auth_initial_admin_email=TEST_ADMIN_EMAIL,
        auth_initial_admin_password=TEST_ADMIN_PASSWORD,
        auth_cookie_secure=True,
        **overrides,
    )


def build_incomplete_setup_config(root, **overrides) -> AppConfig:
    values = {
        "workspace_root": str(root),
        "ollama_host": "http://127.0.0.1:9",
        "auth_secret_key": None,
        "auth_initial_admin_email": None,
        "auth_initial_admin_password": None,
        "auth_cookie_secure": False,
    }
    values.update(overrides)
    return AppConfig(**values)


def current_csrf_token(client: TestClient) -> str:
    return client.cookies.get("__Host-marc_csrf") or client.cookies.get("marc_csrf") or ""


def apply_csrf_headers(client: TestClient) -> None:
    client.headers.update(
        {
            "Origin": TEST_ORIGIN,
            "X-CSRF-Token": current_csrf_token(client),
        }
    )


def authenticate_client(client: TestClient) -> None:
    session_response = client.get("/api/auth/session")
    assert session_response.status_code == 200
    apply_csrf_headers(client)
    login_response = client.post(
        "/api/auth/login",
        json={"email": TEST_ADMIN_EMAIL, "password": TEST_ADMIN_PASSWORD},
    )
    assert login_response.status_code == 200
    apply_csrf_headers(client)


def build_test_client(app, *, authenticate: bool = True) -> TestClient:
    client = TestClient(app, base_url=TEST_ORIGIN)
    if authenticate:
        authenticate_client(client)
    return client


def login_as(client: TestClient, *, email: str, password: str) -> None:
    session_response = client.get("/api/auth/session")
    assert session_response.status_code == 200
    apply_csrf_headers(client)
    login_response = client.post(
        "/api/auth/login",
        json={"email": email, "password": password},
    )
    assert login_response.status_code == 200
    apply_csrf_headers(client)


def complete_setup_flow(
    client: TestClient,
    workspace_root,
    *,
    admin_email: str = TEST_ADMIN_EMAIL,
    admin_password: str = TEST_ADMIN_PASSWORD,
    admin_display_name: str = "Setup Admin",
) -> dict:
    setup_status = client.get("/api/setup/status")
    assert setup_status.status_code == 200
    apply_csrf_headers(client)

    response = client.post(
        "/api/setup/complete",
        json={
            "admin_display_name": admin_display_name,
            "admin_email": admin_email,
            "admin_password": admin_password,
            "admin_password_confirm": admin_password,
            "initial_workspace_name": "Demo Project",
            "initial_workspace_path": str(workspace_root),
            "ollama_host": "http://127.0.0.1:11434",
            "model_name": "qwen3-coder:30b",
            "router_model_name": "qwen2.5-coder:14b",
            "access_mode": "approval",
            "auth_cookie_secure": True,
            "public_base_url": "https://testserver",
        },
    )

    assert response.status_code == 200
    return response.json()


def create_test_workspace(client: TestClient, root, name: str = "Workspace A") -> dict:
    workspace_root = root / name.lower().replace(" ", "-")
    workspace_root.mkdir()
    response = client.post(
        "/api/workspaces",
        json={"name": name, "path": str(workspace_root)},
    )
    assert response.status_code == 201
    return response.json()


def test_web_root_serves_gui(tmp_path):
    config = build_test_config(tmp_path)
    config.ensure_state_dirs()
    app = create_app(config)
    client = build_test_client(app, authenticate=False)

    response = client.get("/")

    assert response.status_code == 200
    assert "M.A.R.C A2" in response.text


def test_workspaces_api_does_not_auto_add_base_workspace(tmp_path):
    config = build_test_config(tmp_path)
    config.ensure_state_dirs()
    app = create_app(config)
    client = build_test_client(app)

    response = client.get("/api/workspaces")

    assert response.status_code == 200
    assert response.json() == []


def test_first_run_setup_status_is_available_when_config_is_incomplete(tmp_path):
    config = build_incomplete_setup_config(tmp_path)
    app = create_app(config)
    client = TestClient(app, base_url=TEST_ORIGIN)

    response = client.get("/api/setup/status")
    blocked_response = client.get("/api/workspaces")

    assert response.status_code == 200
    payload = response.json()
    assert payload["required"] is True
    assert payload["reason"] == "missing_auth_secret_key"
    assert payload["defaults"]["initial_workspace_path"] == str(tmp_path.resolve())
    assert payload["env_path"] == str((tmp_path / ".env").resolve())
    assert current_csrf_token(client)
    assert blocked_response.status_code == 503


def test_setup_status_stays_available_when_auth_db_cannot_switch_to_wal(tmp_path):
    state_root = tmp_path / ".marc_a1"
    state_root.mkdir()
    for name in ("sessions", "logs", "memory", "helpers", "reports", "active_sessions"):
        (state_root / name).mkdir()
    (state_root / "workspaces.json").write_text('{"workspaces": []}', encoding="utf-8")
    from server.auth_store import AuthStore

    store = AuthStore(state_root / "auth.db")
    store.create_user(
        user_id="admin-1",
        email=TEST_ADMIN_EMAIL,
        display_name="Operator",
        password_hash="hash",
        role="admin",
        totp_secret=None,
        now="2026-04-04T00:00:00+00:00",
    )

    os.chmod(state_root, 0o555)
    try:
        config = AppConfig(
            workspace_root=str(tmp_path),
            ollama_host="http://127.0.0.1:9",
            auth_secret_key=TEST_AUTH_SECRET,
            auth_initial_admin_email=None,
            auth_initial_admin_password=None,
            auth_cookie_secure=True,
        )
        app = create_app(config)
        client = TestClient(app, base_url=TEST_ORIGIN)

        response = client.get("/api/setup/status")

        assert response.status_code == 200
        payload = response.json()
        assert payload["required"] is False
        assert payload["defaults"]["initial_workspace_path"] == str(tmp_path.resolve())
    finally:
        os.chmod(state_root, 0o755)


def test_setup_completion_creates_env_admin_and_workspace(tmp_path):
    config = build_incomplete_setup_config(tmp_path)
    app = create_app(config)
    client = TestClient(app, base_url=TEST_ORIGIN)

    setup_status = client.get("/api/setup/status")
    assert setup_status.status_code == 200
    apply_csrf_headers(client)

    workspace_root = tmp_path / "demo-project"
    response = client.post(
        "/api/setup/complete",
        json={
            "admin_display_name": "Setup Admin",
            "admin_email": TEST_ADMIN_EMAIL,
            "admin_password": TEST_ADMIN_PASSWORD,
            "admin_password_confirm": TEST_ADMIN_PASSWORD,
            "initial_workspace_name": "Demo Project",
            "initial_workspace_path": str(workspace_root),
            "ollama_host": "http://127.0.0.1:11434",
            "model_name": "qwen3-coder:30b",
            "router_model_name": "qwen2.5-coder:14b",
            "access_mode": "approval",
            "auth_cookie_secure": True,
            "public_base_url": "https://testserver",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["auth"]["authenticated"] is True
    assert payload["workspace"]["name"] == "Demo Project"
    assert payload["workspace"]["path"] == str(workspace_root.resolve())
    assert workspace_root.exists()

    env_path = tmp_path / ".env"
    env_text = env_path.read_text(encoding="utf-8")
    assert "AUTH_SECRET_KEY=" in env_text
    assert "MODEL_NAME=qwen3-coder:30b" in env_text
    assert "ROUTER_MODEL_NAME=qwen2.5-coder:14b" in env_text
    assert TEST_ADMIN_PASSWORD not in env_text

    session_response = client.get("/api/auth/session")
    assert session_response.status_code == 200
    assert session_response.json()["authenticated"] is True

    workspaces_response = client.get("/api/workspaces")
    assert workspaces_response.status_code == 200
    assert workspaces_response.json()[0]["name"] == "Demo Project"


def test_setup_completion_works_on_local_http_when_base_config_prefers_secure_cookies(tmp_path):
    config = build_incomplete_setup_config(tmp_path, auth_cookie_secure=True)
    app = create_app(config)
    client = TestClient(app, base_url=TEST_LOCAL_ORIGIN)

    setup_status = client.get("/api/setup/status")
    assert setup_status.status_code == 200
    client.headers.update(
        {
            "Origin": TEST_LOCAL_ORIGIN,
            "X-CSRF-Token": current_csrf_token(client),
        }
    )

    workspace_root = tmp_path / "demo-project"
    response = client.post(
        "/api/setup/complete",
        json={
            "admin_display_name": "Setup Admin",
            "admin_email": TEST_ADMIN_EMAIL,
            "admin_password": TEST_ADMIN_PASSWORD,
            "admin_password_confirm": TEST_ADMIN_PASSWORD,
            "initial_workspace_name": "Demo Project",
            "initial_workspace_path": str(workspace_root),
            "ollama_host": "http://127.0.0.1:11434",
            "model_name": "qwen3-coder:30b",
            "router_model_name": "qwen2.5-coder:14b",
            "access_mode": "approval",
            "auth_cookie_secure": False,
            "public_base_url": None,
        },
    )

    assert response.status_code == 200
    assert response.json()["workspace"]["name"] == "Demo Project"


def test_deleted_env_reactivates_setup_and_blocks_console(tmp_path):
    config = build_incomplete_setup_config(tmp_path)
    app = create_app(config)
    client = TestClient(app, base_url=TEST_ORIGIN)

    workspace_root = tmp_path / "demo-project"
    complete_setup_flow(client, workspace_root)

    env_path = tmp_path / ".env"
    env_path.unlink()

    status_response = client.get("/api/setup/status")
    blocked_response = client.get("/api/workspaces")

    assert status_response.status_code == 200
    payload = status_response.json()
    assert payload["required"] is True
    assert payload["reason"] == "missing_runtime_env"
    assert payload["has_env_file"] is False
    assert blocked_response.status_code == 503


def test_setup_completion_recovers_admin_access_after_env_was_deleted(tmp_path):
    config = build_incomplete_setup_config(tmp_path)
    app = create_app(config)
    client = TestClient(app, base_url=TEST_ORIGIN)

    workspace_root = tmp_path / "demo-project"
    complete_setup_flow(client, workspace_root)

    env_path = tmp_path / ".env"
    env_path.unlink()

    status_response = client.get("/api/setup/status")
    assert status_response.status_code == 200
    assert status_response.json()["required"] is True
    apply_csrf_headers(client)

    recovered_password = "RecoveredVeryStrongPassword!2026"
    response = client.post(
        "/api/setup/complete",
        json={
            "admin_display_name": "Recovered Admin",
            "admin_email": TEST_ADMIN_EMAIL,
            "admin_password": recovered_password,
            "admin_password_confirm": recovered_password,
            "initial_workspace_name": "Demo Project",
            "initial_workspace_path": str(workspace_root),
            "ollama_host": "http://127.0.0.1:11434",
            "model_name": "qwen3-coder:30b",
            "router_model_name": "qwen2.5-coder:14b",
            "access_mode": "approval",
            "auth_cookie_secure": True,
            "public_base_url": "https://testserver",
        },
    )

    assert response.status_code == 200
    assert response.json()["auth"]["authenticated"] is True
    assert env_path.exists()
    assert recovered_password not in env_path.read_text(encoding="utf-8")

    old_password_client = TestClient(app, base_url=TEST_ORIGIN)
    old_password_session = old_password_client.get("/api/auth/session")
    assert old_password_session.status_code == 200
    apply_csrf_headers(old_password_client)
    old_password_login = old_password_client.post(
        "/api/auth/login",
        json={"email": TEST_ADMIN_EMAIL, "password": TEST_ADMIN_PASSWORD},
    )
    assert old_password_login.status_code == 401

    recovered_client = TestClient(app, base_url=TEST_ORIGIN)
    recovered_session = recovered_client.get("/api/auth/session")
    assert recovered_session.status_code == 200
    apply_csrf_headers(recovered_client)
    recovered_login = recovered_client.post(
        "/api/auth/login",
        json={"email": TEST_ADMIN_EMAIL, "password": recovered_password},
    )
    assert recovered_login.status_code == 200
    assert recovered_login.json()["authenticated"] is True


def test_new_chat_requires_explicit_workspace_selection(tmp_path):
    config = build_test_config(tmp_path)
    config.ensure_state_dirs()
    app = create_app(config)
    client = build_test_client(app)

    response = client.post("/api/tasks", json={"prompt": "ohne workspace starten"})

    assert response.status_code == 400
    assert "workspace" in response.json()["detail"].lower()


def test_start_task_and_fetch_session_via_api(tmp_path):
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")

    config = build_test_config(tmp_path, max_iterations=1, shell_timeout=1)
    config.ensure_state_dirs()
    app = create_app(config)
    client = build_test_client(app)
    workspace = create_test_workspace(client, tmp_path)

    create_response = client.post(
        "/api/tasks",
        json={
            "prompt": "Lies die README und fasse das Repo zusammen",
            "dry_run": True,
            "workspace_id": workspace["id"],
        },
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
    assert session_payload["access_mode"] == config.access_mode
    assert session_payload["archived"] is False
    assert "validation_status" in session_payload
    assert "current_phase" in session_payload
    assert "workflow_stage" in session_payload
    assert session_payload["final_response"]
    assert "Status=" not in session_payload["final_response"]
    assert "report" in session_payload
    assert session_payload["report"]["summary"]
    assert session_payload["report"]["report_path"]

    archive_response = client.patch(f"/api/sessions/{session_id}", json={"archived": True})
    assert archive_response.status_code == 200
    assert archive_response.json()["archived"] is True
    assert archive_response.json()["archived_at"]

    logs_response = client.get(f"/api/sessions/{session_id}/logs")
    assert logs_response.status_code == 200
    assert len(logs_response.json()) >= 1

    sessions_response = client.get("/api/sessions")
    assert sessions_response.status_code == 200
    assert any(
        item["id"] == session_id and item["archived"] is True
        for item in sessions_response.json()
    )


def test_config_exposes_preferred_and_installed_ollama_models(tmp_path):
    config = build_test_config(tmp_path)
    config.ensure_state_dirs()
    catalog = {
        "installed_models": [
            {
                "name": "qwen2.5-coder:14b",
                "size": 8988124298,
                "modified_at": "2026-03-20T18:09:35+01:00",
                "family": "qwen2",
                "parameter_size": "14.8B",
            }
        ],
        "recommended_models": [
            {
                "name": "qwen3-coder:30b",
                "label": "Qwen3 Coder 30B",
                "summary": "Coding default.",
                "installed": False,
                "status": "missing",
                "progress": None,
                "completed_bytes": None,
                "total_bytes": None,
                "message": "Noch nicht installiert",
                "error": None,
                "updated_at": None,
            }
        ],
    }

    with patch(
        "server.app._fetch_installed_ollama_models",
        return_value=[
            {
                "name": "qwen2.5-coder:14b",
                "size": 8988124298,
                "modified_at": "2026-03-20T18:09:35+01:00",
                "family": "qwen2",
                "parameter_size": "14.8B",
            }
        ],
    ), patch("server.app.ModelManager.catalog", return_value=catalog):
        app = create_app(config)
        client = build_test_client(app)

        response = client.get("/api/config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["preferred_model_name"] == "qwen2.5-coder:14b"
    assert payload["router_preferred_model_name"] == "qwen2.5-coder:14b"
    assert payload["model_candidates"] == ["qwen2.5-coder:14b"]
    assert payload["installed_ollama_models"][0]["name"] == "qwen2.5-coder:14b"
    assert payload["recommended_models"][0]["name"] == "qwen3-coder:30b"


def test_available_models_pick_fast_router_model_when_possible(tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen3-coder:30b",
        router_model_name="qwen2.5-coder:14b",
    )

    with patch(
        "server.app._fetch_installed_ollama_models",
        return_value=[
            {"name": "qwen3-coder:30b"},
            {"name": "qwen2.5-coder:14b"},
        ],
    ):
        updated = _with_available_models(config)

    assert updated.model_name == "qwen3-coder:30b"
    assert updated.router_model_name == "qwen2.5-coder:14b"
    assert updated.model_candidates == ("qwen3-coder:30b", "qwen2.5-coder:14b")


def test_available_models_do_not_expand_semantic_candidates_from_installed_inventory(tmp_path):
    config = AppConfig(
        workspace_root=str(tmp_path),
        model_name="qwen2.5-coder:7b",
        router_model_name="qwen2.5-coder:7b",
    )

    with patch(
        "server.app._fetch_installed_ollama_models",
        return_value=[
            {"name": "qwen2.5-coder:7b"},
            {"name": "qwen3:8b"},
            {"name": "qwen2.5-coder:14b"},
        ],
    ):
        updated = _with_available_models(config)

    assert updated.model_name == "qwen2.5-coder:7b"
    assert updated.router_model_name == "qwen2.5-coder:7b"
    assert updated.model_candidates == ("qwen2.5-coder:7b",)


def test_task_run_refreshes_stale_model_config_from_live_inventory(tmp_path):
    config = build_test_config(
        tmp_path,
        model_name="qwen2.5-coder:14b",
        router_model_name="qwen2.5-coder:14b",
    )
    config.ensure_state_dirs()

    with patch("server.app._fetch_installed_ollama_models", return_value=[]):
        app = create_app(config)

    client = build_test_client(app)
    workspace = create_test_workspace(client, tmp_path)
    observed: dict[str, str | None] = {}
    started = Event()

    def fake_run_task(self, task, session=None, *, should_stop=None):
        observed["model_name"] = self.config.model_name
        observed["router_model_name"] = self.config.router_model_name
        session.status = "completed"
        session.final_response = "Fertig"
        session.touch()
        self.session_store.save(session)
        started.set()
        return session

    with patch(
        "llm.model_selection.fetch_installed_ollama_models",
        return_value=[
            {"name": "qwen2.5-coder:7b"},
            {"name": "qwen3:8b"},
            {"name": "qwen3:14b"},
        ],
    ), patch("server.task_manager.AgentCore.run_task", new=fake_run_task):
        response = client.post(
            "/api/tasks",
            json={
                "prompt": "starte einen lauf",
                "dry_run": True,
                "workspace_id": workspace["id"],
            },
        )
        assert started.wait(timeout=2)

    assert response.status_code == 202
    assert observed["model_name"] == "qwen2.5-coder:7b"
    assert observed["router_model_name"] == "qwen2.5-coder:7b"


def test_allowed_model_candidates_falls_back_when_config_has_no_model_candidates_attr():
    class LegacyConfig:
        pass

    candidates = _allowed_model_candidates(
        LegacyConfig(),
        primary_model="qwen2.5-coder:7b",
        router_model="qwen2.5-coder:7b",
        installed_names=["qwen2.5-coder:7b"],
    )

    assert candidates == ["qwen2.5-coder:7b"]


def test_with_available_models_supports_legacy_config_without_model_candidates():
    @dataclass(slots=True)
    class LegacyConfig:
        ollama_host: str = "http://127.0.0.1:11434"
        model_name: str = "qwen2.5-coder:7b"
        router_model_name: str | None = "qwen2.5-coder:7b"

    config = LegacyConfig()

    with patch(
        "server.app._fetch_installed_ollama_models",
        return_value=[{"name": "qwen2.5-coder:7b"}],
    ):
        updated = _with_available_models(config)

    assert updated.model_name == "qwen2.5-coder:7b"
    assert updated.router_model_name == "qwen2.5-coder:7b"
    assert not hasattr(updated, "model_candidates")


def test_config_api_reports_allowed_semantic_candidates_not_entire_inventory(tmp_path):
    config = build_test_config(
        tmp_path,
        model_name="qwen2.5-coder:7b",
        router_model_name="qwen2.5-coder:7b",
    )
    config.ensure_state_dirs()
    catalog = {
        "installed_models": [
            {"name": "qwen2.5-coder:7b"},
            {"name": "qwen3:8b"},
            {"name": "qwen2.5-coder:14b"},
        ],
        "recommended_models": [],
    }

    with patch(
        "server.app._fetch_installed_ollama_models",
        return_value=catalog["installed_models"],
    ), patch("server.app.ModelManager.catalog", return_value=catalog):
        app = create_app(config)
        client = build_test_client(app)
        response = client.get("/api/config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["preferred_model_name"] == "qwen2.5-coder:7b"
    assert payload["model_candidates"] == ["qwen2.5-coder:7b"]
    assert [item["name"] for item in payload["installed_ollama_models"]] == [
        "qwen2.5-coder:7b",
        "qwen3:8b",
        "qwen2.5-coder:14b",
    ]


def test_models_api_and_ensure_endpoint_expose_recommended_download_status(tmp_path):
    config = build_test_config(tmp_path)
    config.ensure_state_dirs()
    catalog = {
        "installed_models": [
            {
                "name": "qwen2.5-coder:14b",
                "size": 8988124298,
                "modified_at": "2026-03-20T18:09:35+01:00",
                "family": "qwen2",
                "parameter_size": "14.8B",
            }
        ],
        "recommended_models": [
            {
                "name": "qwen3-coder:30b",
                "label": "Qwen3 Coder 30B",
                "summary": "Coding default.",
                "installed": False,
                "status": "pulling",
                "progress": 0.42,
                "completed_bytes": 42,
                "total_bytes": 100,
                "message": "pulling manifest",
                "error": None,
                "updated_at": "2026-03-26T18:20:00+00:00",
            }
        ],
    }

    with patch("server.app.ModelManager.catalog", return_value=catalog), patch(
        "server.app.ModelManager.ensure_recommended",
        return_value=catalog,
    ) as ensure_mock:
        app = create_app(config)
        client = build_test_client(app)

        response = client.get("/api/models")
        ensure_response = client.post("/api/models/ensure-recommended")

    assert response.status_code == 200
    assert response.json()["recommended_models"][0]["status"] == "pulling"
    assert ensure_response.status_code == 202
    assert ensure_response.json()["recommended_models"][0]["progress"] == 0.42
    ensure_mock.assert_called_once()


def test_workspace_crud_and_follow_up_messages_stay_in_same_chat(tmp_path):
    config = build_test_config(tmp_path, max_iterations=1, shell_timeout=1)
    config.ensure_state_dirs()
    app = create_app(config)
    client = build_test_client(app)

    workspace = create_test_workspace(client, tmp_path)
    workspace_root = tmp_path / "workspace-a"
    assert workspace["name"] == "Workspace A"

    def fake_run_task(self, task, session=None, *, should_stop=None):
        del should_stop
        assert session is not None
        if task == "hallo wer bist du?":
            session.final_response = "Ich bin dein lokaler Coding-Agent fuer diesen Workspace."
        else:
            session.final_response = "Ich kann hier Code analysieren, aendern, debuggen und validieren."
        session.status = "completed"
        session.append_message("assistant", session.final_response)
        session.touch()
        self.session_store.save(session)
        (self.config.log_dir_path / f"{session.id}.jsonl").write_text(
            '{"event":"task_finished"}\n',
            encoding="utf-8",
        )
        return session

    rename_workspace_response = client.patch(
        f"/api/workspaces/{workspace['id']}",
        json={"name": "Workspace Renamed", "path": str(workspace_root)},
    )
    assert rename_workspace_response.status_code == 200
    assert rename_workspace_response.json()["name"] == "Workspace Renamed"

    with patch("server.task_manager.AgentCore.run_task", new=fake_run_task):
        first_task_response = client.post(
            "/api/tasks",
            json={"prompt": "hallo wer bist du?", "workspace_id": workspace["id"]},
        )
        assert first_task_response.status_code == 202
        session_id = first_task_response.json()["id"]

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
        assert session_payload["title"] == "hallo wer bist du?"
        assert session_payload["status"] == "completed"
        assert session_payload["workspace_id"] == workspace["id"]
        assert session_payload["workspace_label"] == "Workspace Renamed"
        assert "Coding-Agent" in session_payload["final_response"]
        assert len(session_payload["messages"]) >= 2
        assert (
            len(
                [
                    message
                    for message in session_payload["messages"]
                    if message["role"] == "user" and message["content"] == "hallo wer bist du?"
                ]
            )
            == 1
        )
        first_message_count = len(session_payload["messages"])
        assert (config.log_dir_path / f"{session_id}.jsonl").exists()
        assert not (workspace_root / config.state_dir_name / "logs" / f"{session_id}.jsonl").exists()

        follow_up_response = client.post(
            "/api/tasks",
            json={"prompt": "und was kannst du hier machen?", "session_id": session_id},
        )
        assert follow_up_response.status_code == 202

        deadline = time.time() + 5
        follow_up_payload = None
        while time.time() < deadline:
            session_response = client.get(f"/api/sessions/{session_id}")
            assert session_response.status_code == 200
            follow_up_payload = session_response.json()
            if follow_up_payload["status"] in {"completed", "partial", "failed"}:
                break
            time.sleep(0.1)

        assert follow_up_payload is not None
        assert follow_up_payload["id"] == session_id
        assert follow_up_payload["title"] == "hallo wer bist du?"
        assert follow_up_payload["status"] == "completed"
    assert len(follow_up_payload["messages"]) >= first_message_count + 2
    assert (
        len(
            [
                message
                for message in follow_up_payload["messages"]
                if message["role"] == "user" and message["content"] == "hallo wer bist du?"
            ]
        )
        == 1
    )
    assert (
        len(
            [
                message
                for message in follow_up_payload["messages"]
                if message["role"] == "user"
                and message["content"] == "und was kannst du hier machen?"
            ]
        )
        == 1
    )


def test_stop_requested_updates_running_session(tmp_path):
    config = build_test_config(tmp_path, max_iterations=2, shell_timeout=1)
    config.ensure_state_dirs()
    app = create_app(config)
    client = build_test_client(app)

    started = Event()

    def fake_run_task(self, task, session=None, *, should_stop=None):
        assert session is not None
        session.status = "running"
        session.touch()
        self.session_store.save(session)
        started.set()

        deadline = time.time() + 2
        while time.time() < deadline:
            if should_stop and should_stop():
                session.stop_requested = True
                session.stop_reason = "user_cancelled"
                session.status = "partial"
                session.final_response = "Gestoppt"
                session.touch()
                self.session_store.save(session)
                return session
            time.sleep(0.02)

        session.status = "completed"
        session.final_response = "Fertig"
        session.touch()
        self.session_store.save(session)
        return session

    with patch("server.task_manager.AgentCore.run_task", new=fake_run_task):
        workspace = create_test_workspace(client, tmp_path)
        create_response = client.post(
            "/api/tasks",
            json={
                "prompt": "starte einen langen lauf",
                "dry_run": True,
                "workspace_id": workspace["id"],
            },
        )

        assert create_response.status_code == 202
        session_id = create_response.json()["id"]
        assert started.wait(timeout=2)

        stop_response = client.patch(
            f"/api/sessions/{session_id}",
            json={"stop_requested": True},
        )
        assert stop_response.status_code == 200
        assert stop_response.json()["stop_requested"] is True

        deadline = time.time() + 5
        session_payload = None
        while time.time() < deadline:
            session_response = client.get(f"/api/sessions/{session_id}")
            assert session_response.status_code == 200
            session_payload = session_response.json()
            if session_payload["status"] in {"completed", "partial", "failed"}:
                break
            time.sleep(0.05)

        assert session_payload is not None
        assert session_payload["status"] == "partial"
        assert session_payload["stop_requested"] is True
        assert session_payload["stop_reason"] == "user_cancelled"

        logs_response = client.get(f"/api/sessions/{session_id}/logs")
        assert logs_response.status_code == 200
        assert any(item["event"] == "task_stop_requested" for item in logs_response.json())


def test_second_task_requires_queue_when_another_run_is_active(tmp_path):
    config = build_test_config(tmp_path, max_iterations=2, shell_timeout=1)
    config.ensure_state_dirs()
    app = create_app(config)
    client = build_test_client(app)

    started = Event()
    release = Event()

    def fake_run_task(self, task, session=None, *, should_stop=None):
        assert session is not None
        session.status = "running"
        session.touch()
        self.session_store.save(session)

        if task == "erster lauf":
            started.set()
            assert release.wait(timeout=2)

        session.status = "completed"
        session.final_response = f"Fertig: {task}"
        session.touch()
        self.session_store.save(session)
        return session

    with patch("server.task_manager.AgentCore.run_task", new=fake_run_task):
        workspace = create_test_workspace(client, tmp_path)
        first_response = client.post(
            "/api/tasks",
            json={
                "prompt": "erster lauf",
                "dry_run": True,
                "workspace_id": workspace["id"],
            },
        )

        assert first_response.status_code == 202
        first_session_id = first_response.json()["id"]
        assert started.wait(timeout=2)

        second_response = client.post(
            "/api/tasks",
            json={
                "prompt": "zweiter lauf",
                "dry_run": True,
                "workspace_id": workspace["id"],
            },
        )

        assert second_response.status_code == 409
        assert "queue the next run" in second_response.json()["detail"].lower()

        sessions_response = client.get("/api/sessions")
        assert sessions_response.status_code == 200
        assert all(session["task"] != "zweiter lauf" for session in sessions_response.json())

        release.set()

        deadline = time.time() + 5
        first_payload = None
        while time.time() < deadline:
            session_response = client.get(f"/api/sessions/{first_session_id}")
            assert session_response.status_code == 200
            first_payload = session_response.json()
            if first_payload["status"] == "completed":
                break
            time.sleep(0.05)

        assert first_payload is not None
        assert first_payload["status"] == "completed"


def test_queued_task_starts_after_active_run_finishes(tmp_path):
    config = build_test_config(tmp_path, max_iterations=2, shell_timeout=1)
    config.ensure_state_dirs()
    app = create_app(config)
    client = build_test_client(app)

    first_started = Event()
    second_started = Event()
    release_first = Event()
    started_order: list[str] = []

    def fake_run_task(self, task, session=None, *, should_stop=None):
        assert session is not None
        started_order.append(task)
        session.status = "running"
        session.touch()
        self.session_store.save(session)

        if task == "erster lauf":
            first_started.set()
            assert release_first.wait(timeout=2)
        elif task == "zweiter lauf":
            second_started.set()

        session.status = "completed"
        session.final_response = f"Fertig: {task}"
        session.touch()
        self.session_store.save(session)
        return session

    with patch("server.task_manager.AgentCore.run_task", new=fake_run_task):
        workspace = create_test_workspace(client, tmp_path)
        first_response = client.post(
            "/api/tasks",
            json={
                "prompt": "erster lauf",
                "dry_run": True,
                "workspace_id": workspace["id"],
            },
        )

        assert first_response.status_code == 202
        assert first_started.wait(timeout=2)

        second_response = client.post(
            "/api/tasks",
            json={
                "prompt": "zweiter lauf",
                "dry_run": True,
                "workspace_id": workspace["id"],
                "enqueue_if_busy": True,
            },
        )

        assert second_response.status_code == 202
        second_payload = second_response.json()
        second_session_id = second_payload["id"]
        assert second_payload["status"] == "queued"
        assert not second_started.wait(timeout=0.2)

        queued_session_response = client.get(f"/api/sessions/{second_session_id}")
        assert queued_session_response.status_code == 200
        assert queued_session_response.json()["status"] == "queued"

        release_first.set()
        assert second_started.wait(timeout=2)

        deadline = time.time() + 5
        completed_payload = None
        while time.time() < deadline:
            session_response = client.get(f"/api/sessions/{second_session_id}")
            assert session_response.status_code == 200
            completed_payload = session_response.json()
            if completed_payload["status"] == "completed":
                break
            time.sleep(0.05)

        assert completed_payload is not None
        assert completed_payload["status"] == "completed"
        assert started_order[:2] == ["erster lauf", "zweiter lauf"]


def test_follow_up_task_clears_old_execution_state_but_keeps_last_paths(tmp_path):
    config = build_test_config(tmp_path, max_iterations=1, shell_timeout=1)
    config.ensure_state_dirs()
    app = create_app(config)
    client = build_test_client(app)

    observed_states: list[dict] = []

    def fake_run_task(self, task, session=None, *, should_stop=None):
        assert session is not None
        observed_states.append(
            {
                "task": task,
                "candidate_files": list(session.candidate_files),
                "changed_files": [item.path for item in session.changed_files],
                "tool_calls": [item.tool_name for item in session.tool_calls],
                "follow_up_context": (
                    None
                    if session.follow_up_context is None
                    else {
                        "previous_task": session.follow_up_context.previous_task,
                        "target_paths": list(session.follow_up_context.target_paths),
                    }
                ),
            }
        )
        if task == "erstelle etwas":
            session.changed_files = [FileChangeRecord(path="demo.py", operation="create")]
            session.tool_calls = [
                ToolCallRecord(
                    iteration=1,
                    tool_name="create_file",
                    tool_args={"path": "demo.py"},
                    success=True,
                    summary="Created demo.py.",
                )
            ]
            session.final_response = "Erstellt"
        else:
            session.final_response = "Aktualisiert"
        session.status = "completed"
        session.touch()
        self.session_store.save(session)
        return session

    with patch("server.task_manager.AgentCore.run_task", new=fake_run_task):
        workspace = create_test_workspace(client, tmp_path)

        first_response = client.post(
            "/api/tasks",
            json={"prompt": "erstelle etwas", "workspace_id": workspace["id"]},
        )
        assert first_response.status_code == 202
        session_id = first_response.json()["id"]

        deadline = time.time() + 5
        while time.time() < deadline:
            session_payload = client.get(f"/api/sessions/{session_id}").json()
            if session_payload["status"] in {"completed", "partial", "failed"}:
                break
            time.sleep(0.1)

        follow_up_response = client.post(
            "/api/tasks",
            json={"prompt": "mach es dunkler", "session_id": session_id},
        )
        assert follow_up_response.status_code == 202

        deadline = time.time() + 5
        while time.time() < deadline:
            session_payload = client.get(f"/api/sessions/{session_id}").json()
            if session_payload["status"] in {"completed", "partial", "failed"}:
                break
            time.sleep(0.1)

    assert observed_states[0]["task"] == "erstelle etwas"
    assert observed_states[1]["task"] == "mach es dunkler"
    assert observed_states[1]["changed_files"] == []
    assert observed_states[1]["tool_calls"] == []
    assert observed_states[1]["candidate_files"] == ["demo.py"]
    assert observed_states[1]["follow_up_context"] == {
        "previous_task": "erstelle etwas",
        "target_paths": ["demo.py"],
    }


def test_stale_failed_session_does_not_turn_completed_on_read(tmp_path):
    config = build_test_config(tmp_path, llm_timeout=1)
    config.ensure_state_dirs()
    app = create_app(config)
    client = build_test_client(app)

    session = SessionState(
        task="lauf ist haengen geblieben",
        workspace_root=str(tmp_path),
        status="failed",
        workflow_stage="blocked",
        validation_status="not_run",
        stop_reason="stale_session",
        final_response="Der vorherige Lauf wurde nicht sauber abgeschlossen. Bitte starte den Chat erneut.",
        last_error="Session thread is no longer active.",
    )
    session.updated_at = "2026-03-29T10:00:00+00:00"
    app.state.task_manager.session_store.save(session)

    response = client.get(f"/api/sessions/{session.id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "failed"
    assert payload["stop_reason"] == "stale_session"
    assert payload["final_response"].startswith("Der vorherige Lauf wurde nicht sauber abgeschlossen")


def test_running_session_with_fresh_active_lease_stays_running_on_read(tmp_path):
    config = build_test_config(tmp_path, llm_timeout=1)
    config.ensure_state_dirs()
    app = create_app(config)
    client = build_test_client(app)

    session = SessionState(
        task="lauf laeuft auf anderem worker",
        workspace_root=str(tmp_path),
        status="running",
        workflow_stage="plan",
        validation_status="not_run",
    )
    session.updated_at = "2026-03-29T10:00:00+00:00"
    app.state.task_manager.session_store.save(session)
    app.state.task_manager._touch_active_session_lease(session.id)

    response = client.get(f"/api/sessions/{session.id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "running"
    assert payload["stop_reason"] is None
    assert payload["final_response"] is None
    assert session.id in app.state.task_manager.active_sessions()


def test_stale_active_lease_does_not_block_stale_session_detection(tmp_path):
    config = build_test_config(tmp_path, llm_timeout=1)
    config.ensure_state_dirs()
    app = create_app(config)
    client = build_test_client(app)

    session = SessionState(
        task="lauf ist wirklich haengen geblieben",
        workspace_root=str(tmp_path),
        status="running",
        workflow_stage="act",
        validation_status="not_run",
    )
    session.updated_at = "2026-03-29T10:00:00+00:00"
    app.state.task_manager.session_store.save(session)

    lease_path = app.state.task_manager._active_session_lease_path(session.id)
    lease_path.touch()
    stale_timestamp = time.time() - (app.state.task_manager._stale_session_threshold_seconds() + 10)
    os.utime(lease_path, (stale_timestamp, stale_timestamp))

    response = client.get(f"/api/sessions/{session.id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "failed"
    assert payload["stop_reason"] == "stale_session"
    assert not lease_path.exists()


def test_delete_session_removes_chat_metadata_and_logs(tmp_path):
    config = build_test_config(tmp_path, max_iterations=1, shell_timeout=1)
    config.ensure_state_dirs()
    app = create_app(config)
    client = build_test_client(app)
    workspace = create_test_workspace(client, tmp_path)

    create_response = client.post(
        "/api/tasks",
        json={"prompt": "Sag kurz hallo", "workspace_id": workspace["id"]},
    )
    assert create_response.status_code == 202
    session_id = create_response.json()["id"]

    deadline = time.time() + 5
    while time.time() < deadline:
        session_response = client.get(f"/api/sessions/{session_id}")
        assert session_response.status_code == 200
        if session_response.json()["status"] in {"completed", "partial", "failed"}:
            break
        time.sleep(0.1)

    assert (config.session_dir_path / f"{session_id}.json").exists()
    assert (config.log_dir_path / f"{session_id}.jsonl").exists()
    assert (config.report_dir_path / f"{session_id}.json").exists()

    delete_response = client.delete(f"/api/sessions/{session_id}")
    assert delete_response.status_code == 204

    assert not (config.session_dir_path / f"{session_id}.json").exists()
    assert not (config.log_dir_path / f"{session_id}.jsonl").exists()
    assert not (config.report_dir_path / f"{session_id}.json").exists()

    session_response = client.get(f"/api/sessions/{session_id}")
    assert session_response.status_code == 404

    logs_response = client.get(f"/api/sessions/{session_id}/logs")
    assert logs_response.status_code == 404

    sessions_response = client.get("/api/sessions")
    assert sessions_response.status_code == 200
    assert all(item["id"] != session_id for item in sessions_response.json())


def test_delete_workspace_removes_workspace_and_associated_chats(tmp_path):
    config = build_test_config(tmp_path, max_iterations=1, shell_timeout=1)
    config.ensure_state_dirs()
    app = create_app(config)
    client = build_test_client(app)
    workspace = create_test_workspace(client, tmp_path)

    create_response = client.post(
        "/api/tasks",
        json={"prompt": "Sag kurz hallo", "workspace_id": workspace["id"]},
    )
    assert create_response.status_code == 202
    session_id = create_response.json()["id"]

    deadline = time.time() + 5
    while time.time() < deadline:
        session_response = client.get(f"/api/sessions/{session_id}")
        assert session_response.status_code == 200
        if session_response.json()["status"] in {"completed", "partial", "failed"}:
            break
        time.sleep(0.1)

    delete_response = client.delete(f"/api/workspaces/{workspace['id']}")
    assert delete_response.status_code == 204

    workspaces_response = client.get("/api/workspaces")
    assert workspaces_response.status_code == 200
    assert workspaces_response.json() == []

    session_response = client.get(f"/api/sessions/{session_id}")
    assert session_response.status_code == 404

    sessions_response = client.get("/api/sessions")
    assert sessions_response.status_code == 200
    assert sessions_response.json() == []


def test_clear_workspace_contents_removes_files_but_keeps_workspace(tmp_path):
    config = build_test_config(tmp_path)
    config.ensure_state_dirs()
    app = create_app(config)
    client = build_test_client(app)
    workspace = create_test_workspace(client, tmp_path)
    workspace_root = tmp_path / "workspace-a"

    nested_dir = workspace_root / "src"
    nested_dir.mkdir()
    (nested_dir / "main.py").write_text("print('ready')\n", encoding="utf-8")
    (workspace_root / "README.md").write_text("# Demo\n", encoding="utf-8")

    response = client.post(f"/api/workspaces/{workspace['id']}/clear")

    assert response.status_code == 204
    workspaces_response = client.get("/api/workspaces")
    assert workspaces_response.status_code == 200
    assert workspaces_response.json()[0]["id"] == workspace["id"]
    assert workspace_root.exists()
    assert list(workspace_root.iterdir()) == []


def test_clear_workspace_contents_preserves_internal_state_dir(tmp_path):
    config = build_test_config(tmp_path)
    config.ensure_state_dirs()
    app = create_app(config)
    client = build_test_client(app)
    workspace = create_test_workspace(client, tmp_path)
    workspace_root = tmp_path / "workspace-a"

    nested_dir = workspace_root / "src"
    nested_dir.mkdir()
    (nested_dir / "main.py").write_text("print('ready')\n", encoding="utf-8")
    state_dir = workspace_root / config.state_dir_name
    (state_dir / "sessions").mkdir(parents=True)
    (state_dir / "sessions" / "keep.json").write_text("{}", encoding="utf-8")

    response = client.post(f"/api/workspaces/{workspace['id']}/clear")

    assert response.status_code == 204
    assert workspace_root.exists()
    assert sorted(item.name for item in workspace_root.iterdir()) == [config.state_dir_name]
    assert (state_dir / "sessions" / "keep.json").exists()


def test_workspaces_are_recovered_from_workspace_root_when_store_is_empty(tmp_path):
    alpha = tmp_path / "alpha-project"
    beta = tmp_path / "beta-project"
    alpha.mkdir()
    beta.mkdir()
    (alpha / "README.md").write_text("# Alpha\n", encoding="utf-8")
    (beta / "main.py").write_text("print('beta')\n", encoding="utf-8")

    config = build_test_config(tmp_path)
    config.ensure_state_dirs()
    (config.state_root / "workspaces.json").write_text('{"workspaces": []}', encoding="utf-8")

    app = create_app(config)
    client = build_test_client(app)

    response = client.get("/api/workspaces")

    assert response.status_code == 200
    payload = response.json()
    assert [item["name"] for item in payload] == ["alpha-project", "beta-project"]
    assert {item["path"] for item in payload} == {
        str(alpha.resolve()),
        str(beta.resolve()),
    }

    persisted = json.loads((config.state_root / "workspaces.json").read_text(encoding="utf-8"))
    assert len(persisted["workspaces"]) == 2


def test_admin_terminal_session_executes_command(tmp_path):
    config = build_test_config(tmp_path)
    config.ensure_state_dirs()
    app = create_app(config)
    client = build_test_client(app)
    workspace = create_test_workspace(client, tmp_path)

    create_response = client.post("/api/admin/terminal/sessions", json={"workspace_id": workspace["id"]})

    assert create_response.status_code == 201
    payload = create_response.json()
    session_id = payload["id"]
    cursor = payload["cursor"]

    write_response = client.post(
        f"/api/admin/terminal/sessions/{session_id}/input",
        json={"data": "printf 'terminal-ok\\n'\n"},
    )
    assert write_response.status_code == 200

    collected = payload.get("output", "")
    deadline = time.time() + 5
    while time.time() < deadline and "terminal-ok" not in collected:
        read_response = client.get(f"/api/admin/terminal/sessions/{session_id}?cursor={cursor}")
        assert read_response.status_code == 200
        read_payload = read_response.json()
        cursor = read_payload["cursor"]
        collected += read_payload.get("output", "")
        if "terminal-ok" in collected:
            break
        time.sleep(0.1)

    assert "terminal-ok" in collected
    close_response = client.delete(f"/api/admin/terminal/sessions/{session_id}")
    assert close_response.status_code == 204


def test_admin_terminal_requires_admin_role(tmp_path):
    config = build_test_config(tmp_path)
    config.ensure_state_dirs()
    app = create_app(config)
    client = build_test_client(app)

    with sqlite3.connect(config.auth_db_path) as connection:
        connection.execute("UPDATE users SET role = 'member' WHERE email = ?", (TEST_ADMIN_EMAIL,))
        connection.commit()

    response = client.post("/api/admin/terminal/sessions", json={})

    assert response.status_code == 403


def test_admin_terminal_session_is_bound_to_the_creator(tmp_path):
    config = build_test_config(tmp_path)
    config.ensure_state_dirs()
    app = create_app(config)
    client = build_test_client(app)

    auth_service = app.state.auth_service
    assert auth_service is not None
    other_email = "second-admin@example.com"
    other_password = "AnotherStrongPassword!2026"
    now = auth_service.now().isoformat()
    auth_service.store.create_user(
        user_id="second-admin-user",
        email=other_email,
        display_name="Second Admin",
        password_hash=auth_service.password_hasher.hash(other_password),
        role="admin",
        totp_secret=None,
        now=now,
    )

    create_response = client.post("/api/admin/terminal/sessions", json={})
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    other_client = TestClient(app, base_url=TEST_ORIGIN)
    login_as(other_client, email=other_email, password=other_password)

    read_response = other_client.get(f"/api/admin/terminal/sessions/{session_id}?cursor=0")
    assert read_response.status_code == 403


def test_session_handoff_download_contains_changed_files_and_metadata(tmp_path):
    config = build_test_config(tmp_path)
    config.ensure_state_dirs()
    app = create_app(config)
    client = build_test_client(app)
    workspace = create_test_workspace(client, tmp_path)
    workspace_root = tmp_path / "workspace-a"

    source_dir = workspace_root / "src"
    source_dir.mkdir()
    (source_dir / "game.py").write_text("print('ahoy')\n", encoding="utf-8")

    session = SessionState(
        task="baue ein spiel",
        title="Battleship",
        status="completed",
        workspace_root=str(workspace_root),
        workspace_id=workspace["id"],
        workspace_label=workspace["name"],
        final_response="Fertig",
    )
    session.changed_files = [
        FileChangeRecord(path="src/game.py", operation="modify"),
        FileChangeRecord(path="missing.txt", operation="delete"),
    ]
    session.touch()
    app.state.task_manager.session_store.save(session)
    (config.log_dir_path / f"{session.id}.jsonl").write_text(
        '{"event":"task_finished"}\n',
        encoding="utf-8",
    )
    (config.report_dir_path / f"{session.id}.json").write_text(
        json.dumps({"summary": "ok"}, ensure_ascii=False),
        encoding="utf-8",
    )

    response = client.get(f"/api/sessions/{session.id}/download")

    assert response.status_code == 200
    assert "attachment" in response.headers.get("content-disposition", "")

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = set(archive.namelist())
        assert "src/game.py" in names
        assert "_marc_export/manifest.json" in names
        assert "_marc_export/session.json" in names
        assert "_marc_export/report.json" in names
        assert "_marc_export/logs.jsonl" in names

        manifest = json.loads(archive.read("_marc_export/manifest.json"))
        assert manifest["bundle_type"] == "session_handoff"
        assert any(item["path"] == "src/game.py" and item["included"] for item in manifest["changes"])
        assert any(
            item["path"] == "missing.txt" and item.get("reason") == "deleted"
            for item in manifest["changes"]
        )


def test_workspace_export_download_includes_workspace_files_without_ignored_dirs(tmp_path):
    config = build_test_config(tmp_path)
    config.ensure_state_dirs()
    app = create_app(config)
    client = build_test_client(app)
    workspace = create_test_workspace(client, tmp_path)
    workspace_root = tmp_path / "workspace-a"

    (workspace_root / "index.html").write_text("<h1>Preview</h1>\n", encoding="utf-8")
    assets_dir = workspace_root / "assets"
    assets_dir.mkdir()
    (assets_dir / "app.js").write_text("console.log('ready');\n", encoding="utf-8")
    ignored_dir = workspace_root / "node_modules" / "pkg"
    ignored_dir.mkdir(parents=True)
    (ignored_dir / "index.js").write_text("ignored\n", encoding="utf-8")

    response = client.get(f"/api/workspaces/{workspace['id']}/download")

    assert response.status_code == 200

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = set(archive.namelist())
        assert "index.html" in names
        assert "assets/app.js" in names
        assert "_marc_export/manifest.json" in names
        assert "node_modules/pkg/index.js" not in names


def test_workspace_preview_redirects_to_static_entry_and_serves_assets(tmp_path):
    config = build_test_config(tmp_path)
    config.ensure_state_dirs()
    app = create_app(config)
    client = build_test_client(app)
    workspace = create_test_workspace(client, tmp_path)
    workspace_root = tmp_path / "workspace-a"

    (workspace_root / "index.html").write_text(
        "<!doctype html><html><head><link rel='stylesheet' href='style.css'></head><body><h1>Cloud Preview</h1></body></html>",
        encoding="utf-8",
    )
    (workspace_root / "style.css").write_text("body { color: red; }\n", encoding="utf-8")

    redirect_response = client.get(f"/preview/workspaces/{workspace['id']}", follow_redirects=False)
    assert redirect_response.status_code == 307
    location = redirect_response.headers["location"]
    assert location.endswith("/preview/workspaces/" + workspace["id"] + "/index.html")

    html_response = client.get(location)
    css_response = client.get(f"/preview/workspaces/{workspace['id']}/style.css")

    assert html_response.status_code == 200
    assert "Cloud Preview" in html_response.text
    assert css_response.status_code == 200
    assert "color: red" in css_response.text


def test_workspace_preview_runs_python_entrypoint_when_no_static_site_exists(tmp_path):
    config = build_test_config(tmp_path)
    config.ensure_state_dirs()
    app = create_app(config)
    client = build_test_client(app)
    workspace = create_test_workspace(client, tmp_path)
    workspace_root = tmp_path / "workspace-a"

    (workspace_root / "main.py").write_text("print('Hallo aus der Cloud')\n", encoding="utf-8")

    response = client.get(f"/preview/workspaces/{workspace['id']}")

    assert response.status_code == 200
    assert "Python-Preview" in response.text
    assert "Hallo aus der Cloud" in response.text
