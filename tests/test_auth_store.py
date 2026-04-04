from __future__ import annotations

import os
from datetime import datetime, timezone

from server.auth_store import AuthStore


def test_count_users_reads_existing_auth_db_when_wal_switch_is_readonly(tmp_path):
    store = AuthStore(tmp_path / "auth.db")
    now = datetime.now(timezone.utc).isoformat()
    store.create_user(
        user_id="user-1",
        email="operator@example.com",
        display_name="Operator",
        password_hash="hash",
        role="admin",
        totp_secret=None,
        now=now,
    )

    os.chmod(tmp_path, 0o555)
    try:
        assert store.count_users() == 1
        user = store.get_user_by_email("operator@example.com")
        assert user is not None
        assert user.email == "operator@example.com"
    finally:
        os.chmod(tmp_path, 0o755)
