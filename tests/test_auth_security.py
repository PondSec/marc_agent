from __future__ import annotations

import time

import pyotp
import pytest
from fastapi.testclient import TestClient

from config.settings import AppConfig
from server.app import create_app
from server.auth_service import AuthService, PasswordPolicyError

TEST_ORIGIN = "https://testserver"
TEST_AUTH_SECRET = "test-auth-secret-please-change"
TEST_ADMIN_EMAIL = "operator@example.com"
TEST_ADMIN_PASSWORD = "VeryStrongPassword!2026"
TEST_TOTP_SECRET = "JBSWY3DPEHPK3PXP"


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


def current_csrf_token(client: TestClient) -> str:
    return client.cookies.get("__Host-marc_csrf") or client.cookies.get("marc_csrf") or ""


def apply_csrf_headers(client: TestClient) -> None:
    client.headers.update(
        {
            "Origin": TEST_ORIGIN,
            "X-CSRF-Token": current_csrf_token(client),
        }
    )


def build_test_client(app) -> TestClient:
    return TestClient(app, base_url=TEST_ORIGIN)


def bootstrap_csrf(client: TestClient) -> None:
    response = client.get("/api/auth/session")
    assert response.status_code == 200
    apply_csrf_headers(client)


def login(client: TestClient, *, email: str = TEST_ADMIN_EMAIL, password: str = TEST_ADMIN_PASSWORD, totp_code: str | None = None):
    bootstrap_csrf(client)
    payload = {"email": email, "password": password}
    if totp_code is not None:
        payload["totp_code"] = totp_code
    response = client.post("/api/auth/login", json=payload)
    if response.status_code == 200:
        apply_csrf_headers(client)
    return response


def test_password_policy_blocks_weak_passwords(tmp_path):
    service = AuthService(build_test_config(tmp_path))

    with pytest.raises(PasswordPolicyError):
        service.validate_password_policy("password123", email=TEST_ADMIN_EMAIL)

    service.validate_password_policy(TEST_ADMIN_PASSWORD, email=TEST_ADMIN_EMAIL)


def test_sanitize_return_path_blocks_open_redirect_targets(tmp_path):
    service = AuthService(build_test_config(tmp_path))

    assert service.sanitize_return_path("https://evil.example") == "/"
    assert service.sanitize_return_path("//evil.example") == "/"
    assert service.sanitize_return_path("/safe/path") == "/safe/path"


def test_protected_api_requires_authentication(tmp_path):
    config = build_test_config(tmp_path)
    app = create_app(config)
    client = build_test_client(app)

    response = client.get("/api/workspaces")

    assert response.status_code == 401
    assert "anmelden" in response.json()["detail"].lower()


def test_login_sets_cookie_flags_and_security_headers(tmp_path):
    config = build_test_config(tmp_path)
    app = create_app(config)
    client = build_test_client(app)

    root_response = client.get("/")
    login_response = login(client)

    assert root_response.status_code == 200
    assert root_response.headers["content-security-policy"]
    assert root_response.headers["x-frame-options"] == "DENY"
    assert root_response.headers["x-content-type-options"] == "nosniff"
    assert root_response.headers["referrer-policy"] == "no-referrer"
    assert "camera=()" in root_response.headers["permissions-policy"]
    assert "max-age=" in root_response.headers["strict-transport-security"]

    cookies = login_response.headers.get_list("set-cookie")
    session_cookie = next(item for item in cookies if "__Host-marc_session=" in item)
    csrf_cookie = next(item for item in cookies if "__Host-marc_csrf=" in item)
    assert "HttpOnly" in session_cookie
    assert "Secure" in session_cookie
    assert "samesite=lax" in session_cookie.lower()
    assert "Secure" in csrf_cookie
    assert "samesite=strict" in csrf_cookie.lower()
    assert "HttpOnly" not in csrf_cookie


def test_login_prevents_account_enumeration(tmp_path):
    config = build_test_config(tmp_path)
    app = create_app(config)
    client = build_test_client(app)

    unknown_response = login(client, email="unknown@example.com", password="WrongPassword!2026")
    wrong_password_response = login(client, email=TEST_ADMIN_EMAIL, password="WrongPassword!2026")

    assert unknown_response.status_code == 401
    assert wrong_password_response.status_code == 401
    assert unknown_response.json()["detail"] == wrong_password_response.json()["detail"]


def test_login_rate_limits_repeated_failures(tmp_path):
    config = build_test_config(tmp_path)
    app = create_app(config)
    client = build_test_client(app)

    statuses = []
    for _ in range(3):
        response = login(client, email=TEST_ADMIN_EMAIL, password="WrongPassword!2026")
        statuses.append(response.status_code)

    locked_response = client.post(
        "/api/auth/login",
        json={"email": TEST_ADMIN_EMAIL, "password": "WrongPassword!2026"},
    )

    assert statuses[:2] == [401, 401]
    assert statuses[2] == 429
    assert locked_response.status_code == 429
    assert int(locked_response.headers["retry-after"]) > 0


def test_login_requires_valid_totp_when_enabled(tmp_path):
    config = build_test_config(tmp_path, auth_initial_admin_totp_secret=TEST_TOTP_SECRET)
    app = create_app(config)
    client = build_test_client(app)

    missing_code_response = login(client)
    valid_code = pyotp.TOTP(TEST_TOTP_SECRET).now()
    success_response = login(client, totp_code=valid_code)

    assert missing_code_response.status_code == 401
    assert success_response.status_code == 200
    assert success_response.json()["user"]["mfa_enabled"] is True


def test_missing_csrf_or_wrong_origin_is_rejected(tmp_path):
    config = build_test_config(tmp_path)
    app = create_app(config)
    client = build_test_client(app)
    assert login(client).status_code == 200

    response_without_token = client.post(
        "/api/workspaces",
        json={"name": "A", "path": str(tmp_path / "a")},
        headers={"Origin": TEST_ORIGIN, "X-CSRF-Token": ""},
    )
    response_wrong_origin = client.post(
        "/api/workspaces",
        json={"name": "B", "path": str(tmp_path / "b")},
        headers={"Origin": "https://evil.example", "X-CSRF-Token": current_csrf_token(client)},
    )

    assert response_without_token.status_code == 403
    assert response_wrong_origin.status_code == 403


def test_injection_style_login_payload_does_not_bypass_auth(tmp_path):
    config = build_test_config(tmp_path)
    app = create_app(config)
    client = build_test_client(app)

    response = login(client, email="' OR 1=1 --@example.com", password="' OR 1=1 --")

    assert response.status_code == 401


def test_logout_invalidates_session(tmp_path):
    config = build_test_config(tmp_path)
    app = create_app(config)
    client = build_test_client(app)

    assert login(client).status_code == 200
    logout_response = client.post("/api/auth/logout")
    access_response = client.get("/api/workspaces")

    assert logout_response.status_code == 204
    assert access_response.status_code == 401


def test_idle_timeout_expires_session(tmp_path):
    config = build_test_config(
        tmp_path,
        auth_session_idle_seconds=1,
        auth_session_absolute_seconds=120,
        auth_session_touch_interval_seconds=10,
    )
    app = create_app(config)
    client = build_test_client(app)

    assert login(client).status_code == 200
    time.sleep(1.2)
    response = client.get("/api/workspaces")

    assert response.status_code == 401
