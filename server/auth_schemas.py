from __future__ import annotations

from pydantic import Field, field_validator

from server.schemas import StrictModel


class PasswordPolicyResponse(StrictModel):
    min_length: int
    max_length: int = 1024
    requires_totp_when_enabled: bool = True
    blocked_password_examples: list[str] = Field(default_factory=list)


class AuthenticatedUserResponse(StrictModel):
    email: str
    display_name: str
    role: str
    mfa_enabled: bool
    last_login_at: str | None = None
    current_login_at: str | None = None


class AuthSessionMetadata(StrictModel):
    expires_at: str
    idle_expires_at: str
    idle_timeout_seconds: int
    absolute_timeout_seconds: int


class AuthStatusResponse(StrictModel):
    authenticated: bool
    user: AuthenticatedUserResponse | None = None
    session: AuthSessionMetadata | None = None
    csrf_header_name: str = "X-CSRF-Token"
    mfa_totp_supported: bool = True
    password_policy: PasswordPolicyResponse


class LoginRequest(StrictModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=1, max_length=1024)
    totp_code: str | None = Field(default=None, min_length=6, max_length=12)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = value.strip()
        if "@" not in normalized or normalized.startswith("@") or normalized.endswith("@"):
            raise ValueError("Bitte eine gueltige E-Mail-Adresse angeben.")
        return normalized

    @field_validator("totp_code")
    @classmethod
    def normalize_totp_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = "".join(char for char in value.strip() if char.isdigit())
        if not cleaned:
            return None
        if len(cleaned) not in {6, 8}:
            raise ValueError("Der Sicherheitscode ist ungueltig.")
        return cleaned
