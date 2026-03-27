from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import pyotp
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from argon2.low_level import Type
from fastapi import HTTPException, Request, Response, status
from itsdangerous import BadSignature, BadTimeSignature, URLSafeSerializer, URLSafeTimedSerializer
from zxcvbn import zxcvbn

from server.auth_schemas import (
    AuthSessionMetadata,
    AuthStatusResponse,
    AuthenticatedUserResponse,
    LoginRequest,
    PasswordPolicyResponse,
)
from server.auth_store import (
    AuthSessionRecord,
    AuthStore,
    AuthUserRecord,
    RateLimitRecord,
    parse_timestamp,
)


SAFE_HTTP_METHODS = {"GET", "HEAD", "OPTIONS"}
COMMON_PASSWORDS = {
    "123456",
    "12345678",
    "123456789",
    "password",
    "password123",
    "qwerty",
    "qwerty123",
    "letmein",
    "admin",
    "admin123",
    "changeme",
}
LOCK_SCHEDULE_SECONDS = {
    1: 0,
    2: 0,
    3: 5,
    4: 15,
    5: 60,
    6: 300,
    7: 900,
    8: 1800,
}


class AuthBootstrapError(RuntimeError):
    pass


class PasswordPolicyError(ValueError):
    pass


@dataclass(slots=True)
class AuthContext:
    user: AuthUserRecord
    session: AuthSessionRecord


class AuthService:
    def __init__(self, config):
        self.config = config
        self.store = AuthStore(self.config.auth_db_path)
        self.password_hasher = PasswordHasher(
            time_cost=3,
            memory_cost=65536,
            parallelism=4,
            hash_len=32,
            salt_len=16,
            type=Type.ID,
        )
        if not self.config.auth_secret_key:
            raise AuthBootstrapError(
                "AUTH_SECRET_KEY is required when authentication is enabled."
            )
        self._csrf_serializer = URLSafeSerializer(
            self.config.auth_secret_key,
            salt="marc-auth-csrf",
        )
        self._password_reset_serializer = URLSafeTimedSerializer(
            self.config.auth_secret_key,
            salt="marc-auth-password-reset",
        )
        self._fake_password_hash = self.password_hasher.hash(
            "marc-a1-password-verification-padding"
        )

    @property
    def session_cookie_name(self) -> str:
        return "__Host-marc_session" if self.config.auth_cookie_secure else "marc_session"

    @property
    def csrf_cookie_name(self) -> str:
        return "__Host-marc_csrf" if self.config.auth_cookie_secure else "marc_csrf"

    @property
    def csrf_header_name(self) -> str:
        return "X-CSRF-Token"

    def bootstrap_initial_admin(self) -> None:
        if not self.config.auth_enabled:
            return
        if self.store.count_users() > 0:
            return
        if not self.config.auth_initial_admin_email or not self.config.auth_initial_admin_password:
            raise AuthBootstrapError(
                "Authentication is enabled but no initial admin credentials are configured."
            )

        email = self.normalize_email(self.config.auth_initial_admin_email)
        password = self.config.auth_initial_admin_password
        self.validate_password_policy(
            password,
            email=email,
            display_name=self.config.auth_initial_admin_name,
        )
        totp_secret = self.normalize_totp_secret(self.config.auth_initial_admin_totp_secret)
        now = self.now().isoformat()
        user = self.store.create_user(
            user_id=uuid4().hex,
            email=email,
            display_name=self.config.auth_initial_admin_name,
            password_hash=self.password_hasher.hash(password),
            role="admin",
            totp_secret=totp_secret,
            now=now,
        )
        self.store.record_auth_event(
            event_type="bootstrap_admin",
            outcome="created",
            occurred_at=now,
            email=user.email,
            user_id=user.id,
            details={"mfa_enabled": bool(user.totp_secret)},
        )

    def create_initial_admin(
        self,
        *,
        email: str,
        password: str,
        display_name: str | None = None,
        totp_secret: str | None = None,
    ) -> AuthUserRecord:
        if not self.config.auth_enabled:
            raise AuthBootstrapError("Authentication is disabled.")
        if self.store.count_users() > 0:
            raise AuthBootstrapError("An initial admin already exists.")

        normalized_email = self.normalize_email(email)
        display = str(display_name or self.config.auth_initial_admin_name).strip() or self.config.auth_initial_admin_name
        self.validate_password_policy(
            password,
            email=normalized_email,
            display_name=display,
        )
        normalized_totp_secret = self.normalize_totp_secret(totp_secret)
        now = self.now().isoformat()
        user = self.store.create_user(
            user_id=uuid4().hex,
            email=normalized_email,
            display_name=display,
            password_hash=self.password_hasher.hash(password),
            role="admin",
            totp_secret=normalized_totp_secret,
            now=now,
        )
        self.store.record_auth_event(
            event_type="bootstrap_admin",
            outcome="created",
            occurred_at=now,
            email=user.email,
            user_id=user.id,
            details={"mfa_enabled": bool(user.totp_secret), "source": "setup_assistant"},
        )
        return user

    def build_auth_status(
        self,
        context: AuthContext | None = None,
    ) -> AuthStatusResponse:
        if context is None:
            return AuthStatusResponse(
                authenticated=False,
                password_policy=self.password_policy(),
            )
        return AuthStatusResponse(
            authenticated=True,
            user=AuthenticatedUserResponse(
                email=context.user.email,
                display_name=context.user.display_name,
                role=context.user.role,
                mfa_enabled=bool(context.user.totp_secret),
                last_login_at=context.user.previous_login_at,
                current_login_at=context.session.created_at,
            ),
            session=AuthSessionMetadata(
                expires_at=context.session.expires_at,
                idle_expires_at=context.session.idle_expires_at,
                idle_timeout_seconds=self.config.auth_session_idle_seconds,
                absolute_timeout_seconds=self.config.auth_session_absolute_seconds,
            ),
            password_policy=self.password_policy(),
        )

    def password_policy(self) -> PasswordPolicyResponse:
        return PasswordPolicyResponse(
            min_length=self.config.auth_min_password_length,
            blocked_password_examples=["password", "12345678", "changeme"],
        )

    def validate_password_policy(
        self,
        password: str,
        *,
        email: str = "",
        display_name: str = "",
    ) -> None:
        violations: list[str] = []
        if len(password) < self.config.auth_min_password_length:
            violations.append(
                f"Passwoerter muessen mindestens {self.config.auth_min_password_length} Zeichen lang sein."
            )

        lowered = password.casefold().strip()
        if lowered in COMMON_PASSWORDS:
            violations.append("Dieses Passwort ist zu haeufig und wird blockiert.")

        user_inputs = [item for item in {email, display_name, email.split("@")[0]} if item]
        try:
            score = int(zxcvbn(password, user_inputs=user_inputs).get("score", 0))
        except Exception:
            score = 0
        if score < 3:
            violations.append("Dieses Passwort ist zu leicht zu erraten.")

        if violations:
            raise PasswordPolicyError(" ".join(violations))

    def normalize_email(self, value: str) -> str:
        return value.strip().casefold()

    def normalize_totp_secret(self, secret: str | None) -> str | None:
        if not secret:
            return None
        cleaned = re.sub(r"[^A-Z2-7]", "", secret.strip().upper())
        try:
            pyotp.TOTP(cleaned).now()
        except Exception as exc:
            raise AuthBootstrapError("AUTH_INITIAL_ADMIN_TOTP_SECRET is invalid.") from exc
        return cleaned

    def current_user(self, request: Request, response: Response) -> AuthContext:
        context = self.resolve_session(request, response)
        if context is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Sitzung abgelaufen. Bitte erneut anmelden.",
            )
        return context

    def resolve_session(
        self,
        request: Request,
        response: Response | None = None,
    ) -> AuthContext | None:
        if not self.config.auth_enabled:
            return None

        session_id = request.cookies.get(self.session_cookie_name)
        if not session_id:
            return None

        session = self.store.get_session(session_id)
        if session is None:
            if response is not None:
                self.clear_auth_cookies(response)
            return None

        now = self.now()
        if self._session_is_invalid(session, now):
            self.store.revoke_session(session.id, now.isoformat())
            if response is not None:
                self.clear_auth_cookies(response)
            return None

        user = self.store.get_user_by_id(session.user_id)
        if user is None or not user.is_active:
            self.store.revoke_session(session.id, now.isoformat())
            if response is not None:
                self.clear_auth_cookies(response)
            return None

        last_seen_at = parse_timestamp(session.last_seen_at) or now
        if (now - last_seen_at).total_seconds() >= self.config.auth_session_touch_interval_seconds:
            idle_expires_at = now + timedelta(seconds=self.config.auth_session_idle_seconds)
            self.store.touch_session(
                session_id=session.id,
                last_seen_at=now.isoformat(),
                idle_expires_at=idle_expires_at.isoformat(),
            )
            session = self.store.get_session(session.id) or session

        return AuthContext(user=user, session=session)

    def ensure_csrf_cookie(
        self,
        request: Request,
        response: Response,
        context: AuthContext | None = None,
    ) -> None:
        cookie_value = request.cookies.get(self.csrf_cookie_name)
        if context is not None:
            if cookie_value != context.session.csrf_token:
                self._set_cookie(
                    response,
                    key=self.csrf_cookie_name,
                    value=context.session.csrf_token,
                    httponly=False,
                    samesite="strict",
                    max_age=self.config.auth_session_absolute_seconds,
                )
            return

        if cookie_value and self._is_valid_anonymous_csrf(cookie_value):
            return

        token = self._issue_anonymous_csrf_token()
        self._set_cookie(
            response,
            key=self.csrf_cookie_name,
            value=token,
            httponly=False,
            samesite="strict",
            max_age=3600,
        )

    def require_csrf(self, request: Request, response: Response) -> None:
        if request.method.upper() in SAFE_HTTP_METHODS:
            return
        self._validate_origin(request)

        cookie_token = request.cookies.get(self.csrf_cookie_name)
        header_token = request.headers.get(self.csrf_header_name)
        if not cookie_token or not header_token or cookie_token != header_token:
            self._record_csrf_rejection(request, reason="missing_or_mismatch")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Anfrage konnte nicht verifiziert werden.",
            )

        context = self.resolve_session(request, response)
        if context is not None:
            if context.session.csrf_token != header_token:
                self._record_csrf_rejection(request, context=context, reason="session_token_mismatch")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Anfrage konnte nicht verifiziert werden.",
                )
            return

        if not self._is_valid_anonymous_csrf(cookie_token):
            self._record_csrf_rejection(request, reason="anonymous_token_invalid")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Anfrage konnte nicht verifiziert werden.",
            )

    def login(
        self,
        request: Request,
        response: Response,
        payload: LoginRequest,
    ) -> AuthStatusResponse:
        now = self.now()
        email = self.normalize_email(payload.email)
        client_ip = self.client_ip(request)
        user_agent = self.user_agent(request)

        retry_after = self.current_retry_after(email=email, ip_address=client_ip, now=now)
        if retry_after > 0:
            self._record_login_event(
                occurred_at=now.isoformat(),
                outcome="rate_limited",
                email=email,
                ip_address=client_ip,
                details={"retry_after": retry_after},
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Anmeldung voruebergehend blockiert. Bitte kurz warten und erneut versuchen.",
                headers={"Retry-After": str(retry_after)},
            )

        existing_cookie_session = request.cookies.get(self.session_cookie_name)
        user = self.store.get_user_by_email(email)
        authenticated_user = self._verify_login_candidate(user=user, payload=payload)
        if authenticated_user is None:
            self.record_failed_attempt(email=email, ip_address=client_ip, now=now)
            retry_after = self.current_retry_after(email=email, ip_address=client_ip, now=now)
            self._record_login_event(
                occurred_at=now.isoformat(),
                outcome="failed",
                email=email,
                user_id=user.id if user else None,
                ip_address=client_ip,
                details={"retry_after": retry_after},
            )
            if retry_after > 0:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Anmeldung voruebergehend blockiert. Bitte kurz warten und erneut versuchen.",
                    headers={"Retry-After": str(retry_after)},
                )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Anmeldung fehlgeschlagen. Bitte Zugangsdaten und Sicherheitscode pruefen.",
            )

        if existing_cookie_session:
            self.store.revoke_session(existing_cookie_session, now.isoformat())

        self.clear_rate_limits(email=email, ip_address=client_ip)
        self.store.record_successful_login(
            user_id=authenticated_user.id,
            logged_in_at=now.isoformat(),
        )
        reloaded_user = self.store.get_user_by_id(authenticated_user.id) or authenticated_user
        session = self.store.create_session(
            session_id=uuid4().hex,
            user_id=authenticated_user.id,
            csrf_token=secrets.token_urlsafe(32),
            created_at=now.isoformat(),
            last_seen_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=self.config.auth_session_absolute_seconds)).isoformat(),
            idle_expires_at=(now + timedelta(seconds=self.config.auth_session_idle_seconds)).isoformat(),
            ip_address=client_ip,
            user_agent=user_agent,
            rotated_from_session_id=existing_cookie_session,
        )
        self._set_session_cookies(response, session)
        self._record_login_event(
            occurred_at=now.isoformat(),
            outcome="succeeded",
            email=reloaded_user.email,
            user_id=reloaded_user.id,
            ip_address=client_ip,
            details={"mfa_enabled": bool(reloaded_user.totp_secret)},
        )
        return self.build_auth_status(AuthContext(user=reloaded_user, session=session))

    def logout(self, request: Request, response: Response) -> None:
        now = self.now().isoformat()
        session_id = request.cookies.get(self.session_cookie_name)
        context = self.resolve_session(request, response)
        if session_id:
            self.store.revoke_session(session_id, now)
        if context is not None:
            self._record_login_event(
                occurred_at=now,
                outcome="logged_out",
                email=context.user.email,
                user_id=context.user.id,
                ip_address=self.client_ip(request),
                details={},
            )
        self.clear_auth_cookies(response)

    def issue_password_reset_token(self, user: AuthUserRecord) -> str:
        return self._password_reset_serializer.dumps({"user_id": user.id, "email": user.email})

    def verify_password_reset_token(self, token: str, *, max_age_seconds: int = 900) -> AuthUserRecord | None:
        try:
            payload = self._password_reset_serializer.loads(token, max_age=max_age_seconds)
        except (BadSignature, BadTimeSignature):
            return None
        user_id = str(payload.get("user_id") or "").strip()
        email = self.normalize_email(str(payload.get("email") or ""))
        if not user_id or not email:
            return None
        user = self.store.get_user_by_id(user_id)
        if user is None or user.email != email:
            return None
        return user

    def sanitize_return_path(self, target: str | None) -> str:
        candidate = str(target or "").strip()
        if not candidate.startswith("/"):
            return "/"
        if candidate.startswith("//"):
            return "/"
        if any(ord(char) < 32 for char in candidate):
            return "/"
        return candidate

    def clear_auth_cookies(self, response: Response) -> None:
        response.delete_cookie(self.session_cookie_name, path="/")
        response.delete_cookie(self.csrf_cookie_name, path="/")

    def current_retry_after(
        self,
        *,
        email: str,
        ip_address: str,
        now: datetime,
    ) -> int:
        locked_until = self._locked_until(email=email, ip_address=ip_address)
        if locked_until is None:
            return 0
        delta = int((locked_until - now).total_seconds())
        return max(0, delta)

    def record_failed_attempt(
        self,
        *,
        email: str,
        ip_address: str,
        now: datetime,
    ) -> None:
        for scope, key in self._rate_limit_keys(email=email, ip_address=ip_address).items():
            existing = self.store.get_rate_limits([key]).get(key)
            if existing is None or self._rate_limit_is_stale(existing, now):
                failure_count = 0
                first_failure_at = now.isoformat()
            else:
                failure_count = existing.failure_count
                first_failure_at = existing.first_failure_at
            failure_count += 1
            lock_seconds = LOCK_SCHEDULE_SECONDS.get(
                failure_count,
                LOCK_SCHEDULE_SECONDS[max(LOCK_SCHEDULE_SECONDS)],
            )
            locked_until = (
                (now + timedelta(seconds=lock_seconds)).isoformat() if lock_seconds > 0 else None
            )
            self.store.upsert_rate_limit(
                RateLimitRecord(
                    key=key,
                    scope=scope,
                    identifier=email if scope != "ip" else ip_address,
                    ip_address=ip_address,
                    failure_count=failure_count,
                    first_failure_at=first_failure_at,
                    last_failure_at=now.isoformat(),
                    locked_until=locked_until,
                )
            )
        self.store.purge_stale_rate_limits(now - timedelta(hours=24))

    def clear_rate_limits(self, *, email: str, ip_address: str) -> None:
        self.store.clear_rate_limits(list(self._rate_limit_keys(email=email, ip_address=ip_address).values()))

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def client_ip(self, request: Request) -> str:
        return str(request.client.host if request.client else "unknown")

    def user_agent(self, request: Request) -> str | None:
        value = str(request.headers.get("user-agent") or "").strip()
        return value or None

    def _session_is_invalid(self, session: AuthSessionRecord, now: datetime) -> bool:
        if session.revoked_at:
            return True
        expires_at = parse_timestamp(session.expires_at)
        idle_expires_at = parse_timestamp(session.idle_expires_at)
        if expires_at and expires_at <= now:
            return True
        if idle_expires_at and idle_expires_at <= now:
            return True
        return False

    def _verify_login_candidate(
        self,
        *,
        user: AuthUserRecord | None,
        payload: LoginRequest,
    ) -> AuthUserRecord | None:
        if user is None:
            self._verify_placeholder_password(payload.password)
            return None

        try:
            verified = self.password_hasher.verify(user.password_hash, payload.password)
        except VerifyMismatchError:
            return None
        except (VerificationError, InvalidHashError):
            return None

        if not verified:
            return None

        if user.totp_secret:
            totp_code = payload.totp_code or ""
            if not pyotp.TOTP(user.totp_secret).verify(totp_code, valid_window=1):
                return None

        if self.password_hasher.check_needs_rehash(user.password_hash):
            self.store.update_password_hash(
                user_id=user.id,
                password_hash=self.password_hasher.hash(payload.password),
                changed_at=self.now().isoformat(),
            )
            return self.store.get_user_by_id(user.id) or user

        return user

    def _verify_placeholder_password(self, password: str) -> None:
        try:
            self.password_hasher.verify(self._fake_password_hash, password)
        except Exception:
            return

    def _issue_anonymous_csrf_token(self) -> str:
        return self._csrf_serializer.dumps({"nonce": secrets.token_urlsafe(24)})

    def _is_valid_anonymous_csrf(self, token: str) -> bool:
        try:
            payload = self._csrf_serializer.loads(token)
        except BadSignature:
            return False
        return bool(payload.get("nonce"))

    def _set_session_cookies(self, response: Response, session: AuthSessionRecord) -> None:
        self._set_cookie(
            response,
            key=self.session_cookie_name,
            value=session.id,
            httponly=True,
            samesite="lax",
            max_age=self.config.auth_session_absolute_seconds,
        )
        self._set_cookie(
            response,
            key=self.csrf_cookie_name,
            value=session.csrf_token,
            httponly=False,
            samesite="strict",
            max_age=self.config.auth_session_absolute_seconds,
        )

    def _set_cookie(
        self,
        response: Response,
        *,
        key: str,
        value: str,
        httponly: bool,
        samesite: str,
        max_age: int,
    ) -> None:
        response.set_cookie(
            key=key,
            value=value,
            httponly=httponly,
            secure=self.config.auth_cookie_secure,
            samesite=samesite,
            path="/",
            max_age=max_age,
        )

    def _locked_until(self, *, email: str, ip_address: str) -> datetime | None:
        records = self.store.get_rate_limits(
            list(self._rate_limit_keys(email=email, ip_address=ip_address).values())
        )
        locked_until: datetime | None = None
        for record in records.values():
            candidate = parse_timestamp(record.locked_until)
            if candidate is None:
                continue
            if locked_until is None or candidate > locked_until:
                locked_until = candidate
        return locked_until

    def _rate_limit_keys(self, *, email: str, ip_address: str) -> dict[str, str]:
        return {
            "account": f"account:{email}",
            "ip": f"ip:{ip_address}",
            "pair": f"pair:{email}:{ip_address}",
        }

    def _rate_limit_is_stale(self, record: RateLimitRecord, now: datetime) -> bool:
        last_failure_at = parse_timestamp(record.last_failure_at)
        if last_failure_at is None:
            return True
        return (now - last_failure_at) > timedelta(hours=24)

    def _validate_origin(self, request: Request) -> None:
        origin = str(request.headers.get("origin") or "").strip()
        referer = str(request.headers.get("referer") or "").strip()
        allowed = {
            self._origin_for_request(request),
        }
        if self.config.public_base_url:
            allowed.add(self._normalize_origin(self.config.public_base_url))

        if origin:
            if self._normalize_origin(origin) in allowed:
                return
        elif referer:
            if self._normalize_origin(referer) in allowed:
                return

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Anfrage konnte nicht verifiziert werden.",
        )

    def _origin_for_request(self, request: Request) -> str:
        return self._normalize_origin(str(request.base_url))

    def _normalize_origin(self, value: str) -> str:
        candidate = value.strip()
        if "://" not in candidate:
            return candidate.rstrip("/")
        prefix, _, rest = candidate.partition("://")
        host = rest.split("/", 1)[0]
        return f"{prefix.lower()}://{host.lower()}"

    def _record_csrf_rejection(
        self,
        request: Request,
        *,
        reason: str,
        context: AuthContext | None = None,
    ) -> None:
        now = self.now().isoformat()
        self.store.record_auth_event(
            event_type="csrf",
            outcome="rejected",
            occurred_at=now,
            email=context.user.email if context else None,
            user_id=context.user.id if context else None,
            ip_address=self.client_ip(request),
            details={"reason": reason},
        )

    def _record_login_event(
        self,
        *,
        occurred_at: str,
        outcome: str,
        email: str | None,
        ip_address: str | None,
        user_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.store.record_auth_event(
            event_type="login",
            outcome=outcome,
            occurred_at=occurred_at,
            email=email,
            user_id=user_id,
            ip_address=ip_address,
            details=details,
        )
