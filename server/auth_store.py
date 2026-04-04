from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    totp_secret TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    password_changed_at TEXT NOT NULL,
    last_login_at TEXT,
    previous_login_at TEXT
)
"""

SESSIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    csrf_token TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    idle_expires_at TEXT NOT NULL,
    rotated_from_session_id TEXT,
    revoked_at TEXT,
    ip_address TEXT,
    user_agent TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
)
"""

RATE_LIMITS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS rate_limits (
    key TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    identifier TEXT NOT NULL,
    ip_address TEXT,
    failure_count INTEGER NOT NULL,
    first_failure_at TEXT NOT NULL,
    last_failure_at TEXT NOT NULL,
    locked_until TEXT
)
"""

AUTH_EVENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS auth_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    event_type TEXT NOT NULL,
    outcome TEXT NOT NULL,
    email TEXT,
    user_id TEXT,
    ip_address TEXT,
    details TEXT NOT NULL
)
"""

AUTH_INDEX_STATEMENTS: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_expiry ON sessions(expires_at, idle_expires_at, revoked_at)",
    "CREATE INDEX IF NOT EXISTS idx_rate_limits_last_failure ON rate_limits(last_failure_at)",
    "CREATE INDEX IF NOT EXISTS idx_auth_events_created_at ON auth_events(created_at)",
)

AUTH_MIGRATION_COLUMNS: dict[str, dict[str, str]] = {
    "users": {
        "previous_login_at": "TEXT",
    },
    "sessions": {
        "rotated_from_session_id": "TEXT",
    },
}


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


@dataclass(slots=True)
class AuthUserRecord:
    id: str
    email: str
    display_name: str
    password_hash: str
    role: str
    is_active: bool
    totp_secret: str | None
    created_at: str
    updated_at: str
    password_changed_at: str
    last_login_at: str | None
    previous_login_at: str | None


@dataclass(slots=True)
class AuthSessionRecord:
    id: str
    user_id: str
    csrf_token: str
    created_at: str
    last_seen_at: str
    expires_at: str
    idle_expires_at: str
    rotated_from_session_id: str | None
    revoked_at: str | None
    ip_address: str | None
    user_agent: str | None


@dataclass(slots=True)
class RateLimitRecord:
    key: str
    scope: str
    identifier: str
    ip_address: str | None
    failure_count: int
    first_failure_at: str
    last_failure_at: str
    locked_until: str | None


class AuthStore:
    def __init__(self, target: Path):
        self.target = target
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def count_users(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM users").fetchone()
        return int(row["count"]) if row else 0

    def get_user_by_email(self, email: str) -> AuthUserRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    email,
                    display_name,
                    password_hash,
                    role,
                    is_active,
                    totp_secret,
                    created_at,
                    updated_at,
                    password_changed_at,
                    last_login_at,
                    previous_login_at
                FROM users
                WHERE email = ?
                """,
                (email,),
            ).fetchone()
        return self._user_from_row(row)

    def get_user_by_id(self, user_id: str) -> AuthUserRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    email,
                    display_name,
                    password_hash,
                    role,
                    is_active,
                    totp_secret,
                    created_at,
                    updated_at,
                    password_changed_at,
                    last_login_at,
                    previous_login_at
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()
        return self._user_from_row(row)

    def create_user(
        self,
        *,
        user_id: str,
        email: str,
        display_name: str,
        password_hash: str,
        role: str,
        totp_secret: str | None,
        now: str,
    ) -> AuthUserRecord:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO users (
                    id,
                    email,
                    display_name,
                    password_hash,
                    role,
                    is_active,
                    totp_secret,
                    created_at,
                    updated_at,
                    password_changed_at,
                    last_login_at,
                    previous_login_at
                )
                VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    user_id,
                    email,
                    display_name,
                    password_hash,
                    role,
                    totp_secret,
                    now,
                    now,
                    now,
                ),
            )
        return self.get_user_by_id(user_id)

    def update_password_hash(
        self,
        *,
        user_id: str,
        password_hash: str,
        changed_at: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE users
                SET password_hash = ?,
                    updated_at = ?,
                    password_changed_at = ?
                WHERE id = ?
                """,
                (password_hash, changed_at, changed_at, user_id),
            )

    def recover_admin_access(
        self,
        *,
        user_id: str,
        display_name: str,
        password_hash: str,
        changed_at: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE users
                SET display_name = ?,
                    password_hash = ?,
                    role = 'admin',
                    is_active = 1,
                    totp_secret = ?,
                    updated_at = ?,
                    password_changed_at = ?
                WHERE id = ?
                """,
                (
                    display_name,
                    password_hash,
                    None,
                    changed_at,
                    changed_at,
                    user_id,
                ),
            )

    def record_successful_login(self, *, user_id: str, logged_in_at: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE users
                SET previous_login_at = last_login_at,
                    last_login_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (logged_in_at, logged_in_at, user_id),
            )

    def create_session(
        self,
        *,
        session_id: str,
        user_id: str,
        csrf_token: str,
        created_at: str,
        last_seen_at: str,
        expires_at: str,
        idle_expires_at: str,
        ip_address: str | None,
        user_agent: str | None,
        rotated_from_session_id: str | None = None,
    ) -> AuthSessionRecord:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    id,
                    user_id,
                    csrf_token,
                    created_at,
                    last_seen_at,
                    expires_at,
                    idle_expires_at,
                    rotated_from_session_id,
                    revoked_at,
                    ip_address,
                    user_agent
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    session_id,
                    user_id,
                    csrf_token,
                    created_at,
                    last_seen_at,
                    expires_at,
                    idle_expires_at,
                    rotated_from_session_id,
                    ip_address,
                    user_agent,
                ),
            )
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> AuthSessionRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    user_id,
                    csrf_token,
                    created_at,
                    last_seen_at,
                    expires_at,
                    idle_expires_at,
                    rotated_from_session_id,
                    revoked_at,
                    ip_address,
                    user_agent
                FROM sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
        return self._session_from_row(row)

    def touch_session(
        self,
        *,
        session_id: str,
        last_seen_at: str,
        idle_expires_at: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE sessions
                SET last_seen_at = ?,
                    idle_expires_at = ?
                WHERE id = ? AND revoked_at IS NULL
                """,
                (last_seen_at, idle_expires_at, session_id),
            )

    def revoke_session(self, session_id: str, revoked_at: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE sessions
                SET revoked_at = COALESCE(revoked_at, ?)
                WHERE id = ?
                """,
                (revoked_at, session_id),
            )
        return cursor.rowcount > 0

    def revoke_all_sessions(self, revoked_at: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE sessions
                SET revoked_at = COALESCE(revoked_at, ?)
                WHERE revoked_at IS NULL
                """,
                (revoked_at,),
            )
        return int(cursor.rowcount or 0)

    def purge_expired_sessions(self, now: datetime) -> None:
        now_iso = now.astimezone(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM sessions
                WHERE revoked_at IS NOT NULL
                   OR expires_at <= ?
                   OR idle_expires_at <= ?
                """,
                (now_iso, now_iso),
            )

    def get_rate_limits(self, keys: list[str]) -> dict[str, RateLimitRecord]:
        if not keys:
            return {}
        placeholders = ", ".join("?" for _ in keys)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    key,
                    scope,
                    identifier,
                    ip_address,
                    failure_count,
                    first_failure_at,
                    last_failure_at,
                    locked_until
                FROM rate_limits
                WHERE key IN ({placeholders})
                """,
                tuple(keys),
            ).fetchall()
        return {
            str(row["key"]): self._rate_limit_from_row(row)
            for row in rows
        }

    def upsert_rate_limit(self, record: RateLimitRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO rate_limits (
                    key,
                    scope,
                    identifier,
                    ip_address,
                    failure_count,
                    first_failure_at,
                    last_failure_at,
                    locked_until
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    scope = excluded.scope,
                    identifier = excluded.identifier,
                    ip_address = excluded.ip_address,
                    failure_count = excluded.failure_count,
                    first_failure_at = excluded.first_failure_at,
                    last_failure_at = excluded.last_failure_at,
                    locked_until = excluded.locked_until
                """,
                (
                    record.key,
                    record.scope,
                    record.identifier,
                    record.ip_address,
                    record.failure_count,
                    record.first_failure_at,
                    record.last_failure_at,
                    record.locked_until,
                ),
            )

    def clear_rate_limits(self, keys: list[str]) -> None:
        if not keys:
            return
        placeholders = ", ".join("?" for _ in keys)
        with self._connect() as connection:
            connection.execute(
                f"DELETE FROM rate_limits WHERE key IN ({placeholders})",
                tuple(keys),
            )

    def purge_stale_rate_limits(self, cutoff: datetime) -> None:
        cutoff_iso = cutoff.astimezone(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM rate_limits
                WHERE last_failure_at <= ?
                """,
                (cutoff_iso,),
            )

    def record_auth_event(
        self,
        *,
        event_type: str,
        outcome: str,
        occurred_at: str,
        email: str | None = None,
        user_id: str | None = None,
        ip_address: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO auth_events (
                    created_at,
                    event_type,
                    outcome,
                    email,
                    user_id,
                    ip_address,
                    details
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    occurred_at,
                    event_type,
                    outcome,
                    email,
                    user_id,
                    ip_address,
                    json.dumps(details or {}, ensure_ascii=False),
                ),
            )

    def _connect(self) -> sqlite3.Connection:
        self.target.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.target, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        self._configure_connection(connection)
        self._ensure_schema(connection)
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            # Opening a connection is enough because _connect() now enforces
            # the current schema idempotently on every reconnect.
            connection.execute("SELECT 1")

    def _configure_connection(self, connection: sqlite3.Connection) -> None:
        try:
            connection.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError:
            # Some environments briefly expose the DB on filesystems where WAL
            # cannot be enabled. Reads should still keep working on the default mode.
            pass
        try:
            connection.execute("PRAGMA synchronous = NORMAL")
        except sqlite3.OperationalError:
            pass

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(USERS_TABLE_SQL)
        connection.execute(SESSIONS_TABLE_SQL)
        connection.execute(RATE_LIMITS_TABLE_SQL)
        connection.execute(AUTH_EVENTS_TABLE_SQL)
        for statement in AUTH_INDEX_STATEMENTS:
            connection.execute(statement)
        for table_name, columns in AUTH_MIGRATION_COLUMNS.items():
            for column_name, column_type in columns.items():
                self._ensure_column(connection, table_name, column_name, column_type)

    def _ensure_column(
        self,
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_type: str,
    ) -> None:
        rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        known_columns = {str(row["name"]) for row in rows}
        if column_name in known_columns:
            return
        connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
        )

    @staticmethod
    def _user_from_row(row: sqlite3.Row | None) -> AuthUserRecord | None:
        if row is None:
            return None
        return AuthUserRecord(
            id=str(row["id"]),
            email=str(row["email"]),
            display_name=str(row["display_name"]),
            password_hash=str(row["password_hash"]),
            role=str(row["role"]),
            is_active=bool(row["is_active"]),
            totp_secret=str(row["totp_secret"]) if row["totp_secret"] else None,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            password_changed_at=str(row["password_changed_at"]),
            last_login_at=str(row["last_login_at"]) if row["last_login_at"] else None,
            previous_login_at=str(row["previous_login_at"]) if row["previous_login_at"] else None,
        )

    @staticmethod
    def _session_from_row(row: sqlite3.Row | None) -> AuthSessionRecord | None:
        if row is None:
            return None
        return AuthSessionRecord(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            csrf_token=str(row["csrf_token"]),
            created_at=str(row["created_at"]),
            last_seen_at=str(row["last_seen_at"]),
            expires_at=str(row["expires_at"]),
            idle_expires_at=str(row["idle_expires_at"]),
            rotated_from_session_id=str(row["rotated_from_session_id"])
            if row["rotated_from_session_id"]
            else None,
            revoked_at=str(row["revoked_at"]) if row["revoked_at"] else None,
            ip_address=str(row["ip_address"]) if row["ip_address"] else None,
            user_agent=str(row["user_agent"]) if row["user_agent"] else None,
        )

    @staticmethod
    def _rate_limit_from_row(row: sqlite3.Row | None) -> RateLimitRecord | None:
        if row is None:
            return None
        return RateLimitRecord(
            key=str(row["key"]),
            scope=str(row["scope"]),
            identifier=str(row["identifier"]),
            ip_address=str(row["ip_address"]) if row["ip_address"] else None,
            failure_count=int(row["failure_count"]),
            first_failure_at=str(row["first_failure_at"]),
            last_failure_at=str(row["last_failure_at"]),
            locked_until=str(row["locked_until"]) if row["locked_until"] else None,
        )
