from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from server.auth_schemas import AuthStatusResponse, PasswordPolicyResponse
from server.schemas import ModelInventory, RecommendedModelStatus, StrictModel, WorkspaceRecord


class SetupDefaultsResponse(StrictModel):
    ollama_host: str
    model_name: str
    router_model_name: str | None = None
    access_mode: Literal["safe", "approval", "full"]
    auth_cookie_secure: bool
    public_base_url: str | None = None
    initial_workspace_name: str
    initial_workspace_path: str


class SetupStatusResponse(StrictModel):
    required: bool
    reason: str | None = None
    has_env_file: bool
    env_path: str
    password_policy: PasswordPolicyResponse
    defaults: SetupDefaultsResponse
    installed_ollama_models: list[ModelInventory] = Field(default_factory=list)
    recommended_models: list[RecommendedModelStatus] = Field(default_factory=list)


class SetupCompleteRequest(StrictModel):
    admin_display_name: str = Field(..., min_length=1, max_length=120)
    admin_email: str = Field(..., min_length=3, max_length=254)
    admin_password: str = Field(..., min_length=1, max_length=1024)
    admin_password_confirm: str = Field(..., min_length=1, max_length=1024)
    initial_workspace_name: str = Field(..., min_length=1, max_length=120)
    initial_workspace_path: str = Field(..., min_length=1)
    ollama_host: str = Field(..., min_length=1, max_length=512)
    model_name: str = Field(..., min_length=1, max_length=128)
    router_model_name: str | None = Field(default=None, max_length=128)
    access_mode: Literal["safe", "approval", "full"] = "approval"
    auth_cookie_secure: bool = False
    public_base_url: str | None = Field(default=None, max_length=512)

    @model_validator(mode="after")
    def validate_payload(self) -> "SetupCompleteRequest":
        self.admin_display_name = self.admin_display_name.strip()
        self.admin_email = self.admin_email.strip()
        self.initial_workspace_name = self.initial_workspace_name.strip()
        self.initial_workspace_path = self.initial_workspace_path.strip()
        self.ollama_host = self.ollama_host.strip()
        self.model_name = self.model_name.strip()
        self.router_model_name = (self.router_model_name or "").strip() or None
        self.public_base_url = (self.public_base_url or "").strip() or None
        if self.admin_password != self.admin_password_confirm:
            raise ValueError("Die Passwoerter stimmen nicht ueberein.")
        if "@" not in self.admin_email or self.admin_email.startswith("@") or self.admin_email.endswith("@"):
            raise ValueError("Bitte eine gueltige E-Mail-Adresse angeben.")
        if not self.initial_workspace_path:
            raise ValueError("Bitte einen gueltigen Projektpfad angeben.")
        return self


class SetupCompleteResponse(StrictModel):
    auth: AuthStatusResponse
    workspace: WorkspaceRecord
    env_path: str
