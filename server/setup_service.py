from __future__ import annotations

import secrets
from collections import OrderedDict
from pathlib import Path

from config.settings import AppConfig
from server.auth_schemas import PasswordPolicyResponse
from server.model_manager import RECOMMENDED_MODELS
from server.setup_schemas import SetupDefaultsResponse, SetupStatusResponse


MANAGED_ENV_KEYS: tuple[str, ...] = (
    "AUTH_ENABLED",
    "AUTH_SECRET_KEY",
    "AUTH_COOKIE_SECURE",
    "OLLAMA_HOST",
    "MODEL_NAME",
    "ROUTER_MODEL_NAME",
    "ACCESS_MODE",
    "PUBLIC_BASE_URL",
)


class SetupService:
    def __init__(self, env_path: Path):
        self.env_path = env_path

    def has_env_file(self) -> bool:
        return self.env_path.exists()

    def generate_secret_key(self) -> str:
        return secrets.token_urlsafe(48)

    def write_runtime_env(
        self,
        *,
        config: AppConfig,
        secret_key: str,
    ) -> None:
        values: OrderedDict[str, str] = OrderedDict(
            [
                ("AUTH_ENABLED", "1" if config.auth_enabled else "0"),
                ("AUTH_SECRET_KEY", secret_key),
                ("AUTH_COOKIE_SECURE", "1" if config.auth_cookie_secure else "0"),
                ("OLLAMA_HOST", config.ollama_host),
                ("MODEL_NAME", config.model_name),
                ("ROUTER_MODEL_NAME", config.router_model_name or config.model_name),
                ("ACCESS_MODE", config.access_mode),
            ]
        )
        if config.public_base_url:
            values["PUBLIC_BASE_URL"] = config.public_base_url
        else:
            values["PUBLIC_BASE_URL"] = ""

        self._merge_env_values(values)

    def build_status(
        self,
        *,
        config: AppConfig,
        password_policy: PasswordPolicyResponse,
        setup_required: bool,
        setup_reason: str | None,
        installed_models: list[dict],
        recommended_models: list[dict],
    ) -> SetupStatusResponse:
        workspace_path = config.workspace_path
        workspace_name = workspace_path.name or "Projekt"
        secure_default = config.auth_cookie_secure
        if not self.has_env_file() and not config.public_base_url:
            secure_default = False

        return SetupStatusResponse(
            required=setup_required,
            reason=setup_reason,
            has_env_file=self.has_env_file(),
            env_path=str(self.env_path),
            password_policy=password_policy,
            defaults=SetupDefaultsResponse(
                ollama_host=config.ollama_host,
                model_name=config.model_name,
                router_model_name=config.router_model_name or config.model_name,
                access_mode=config.access_mode,
                auth_cookie_secure=secure_default,
                public_base_url=config.public_base_url,
                initial_workspace_name=workspace_name,
                initial_workspace_path=str(workspace_path),
            ),
            installed_ollama_models=installed_models,
            recommended_models=recommended_models,
        )

    def model_candidates(
        self,
        *,
        preferred_model: str,
        router_model_name: str | None,
        installed_models: list[dict],
    ) -> list[str]:
        candidates = [preferred_model, router_model_name]
        candidates.extend(item.get("name") for item in installed_models)
        candidates.extend(item.get("name") for item in RECOMMENDED_MODELS)
        deduped: list[str] = []
        for item in candidates:
            value = str(item or "").strip()
            if value and value not in deduped:
                deduped.append(value)
        return deduped

    def _merge_env_values(self, updates: OrderedDict[str, str]) -> None:
        existing_lines = (
            self.env_path.read_text(encoding="utf-8").splitlines()
            if self.env_path.exists()
            else []
        )
        rendered_updates = {
            key: self._render_env_assignment(key, value)
            for key, value in updates.items()
            if key in MANAGED_ENV_KEYS
        }

        output_lines: list[str] = []
        seen_keys: set[str] = set()
        for line in existing_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in line:
                output_lines.append(line)
                continue
            key = line.split("=", 1)[0].strip()
            if key in rendered_updates:
                if key in seen_keys:
                    continue
                output_lines.append(rendered_updates[key])
                seen_keys.add(key)
                continue
            output_lines.append(line)

        missing = [key for key in rendered_updates if key not in seen_keys]
        if not existing_lines:
            output_lines.extend(
                [
                    "# M.A.R.C A1 runtime configuration",
                    "# Erstellt ueber den Web-Setup-Assistenten",
                    "",
                ]
            )
        if missing:
            if output_lines and output_lines[-1].strip():
                output_lines.append("")
            for key in missing:
                output_lines.append(rendered_updates[key])

        self.env_path.parent.mkdir(parents=True, exist_ok=True)
        self.env_path.write_text("\n".join(output_lines).rstrip() + "\n", encoding="utf-8")

    def _render_env_assignment(self, key: str, value: str) -> str:
        normalized = str(value or "")
        if normalized == "":
            return f"{key}="
        if any(char.isspace() for char in normalized) or "#" in normalized:
            escaped = normalized.replace("\\", "\\\\").replace('"', '\\"')
            return f'{key}="{escaped}"'
        return f"{key}={normalized}"
