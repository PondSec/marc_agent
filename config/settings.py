from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any


AGENT_NAME = "M.A.R.C A1"
AGENT_FULL_NAME = "Modular Autonomous Runtime Core - Agent 1"
AGENT_TAGLINE = "Local autonomous coding runtime for real implementation work"


class AccessMode(str, Enum):
    SAFE = "safe"
    APPROVAL = "approval"
    FULL = "full"


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _parse_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _parse_csv_list(value: Any, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return tuple(items) or default
    text = str(value).strip()
    if not text:
        return default
    return tuple(part.strip() for part in text.split(",") if part.strip()) or default


def _load_dotenv(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip().strip("'").strip('"')
    return data


def _load_json_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _app_source_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_source_root() -> Path:
    cwd = Path.cwd()
    if (cwd / ".env").exists() or (cwd / "config" / "agent.json").exists():
        return cwd
    return _app_source_root()


def _source_root_for_config_path(config_file: Path | None) -> Path:
    if config_file is None:
        return _default_source_root()
    parent = config_file.parent
    if parent.name == "config":
        return parent.parent
    return parent


def _normalize_access_mode(
    value: Any,
    *,
    read_only: bool = False,
    approval_mode: bool = False,
    default: AccessMode = AccessMode.APPROVAL,
) -> AccessMode:
    if isinstance(value, AccessMode):
        return value
    if value is not None:
        normalized = str(value).strip().lower()
        if normalized in {item.value for item in AccessMode}:
            return AccessMode(normalized)
    if read_only:
        return AccessMode.SAFE
    if approval_mode:
        return AccessMode.APPROVAL
    return default


@dataclass(slots=True)
class AppConfig:
    ollama_host: str = "http://127.0.0.1:11434"
    model_name: str = "qwen3:14b"
    router_model_name: str | None = "qwen3:8b"
    workspace_root: str = "."
    state_root_override: str | None = None
    access_mode: str = AccessMode.APPROVAL.value
    max_iterations: int = 18
    max_tool_calls: int = 32
    max_repair_attempts: int = 3
    shell_timeout: int = 120
    llm_timeout: int = 25
    router_timeout: int = 35
    llm_request_retries: int = 2
    llm_retry_backoff_ms: int = 800
    router_retries: int = 2
    approval_mode: bool = True
    read_only: bool = False
    verbose: bool = False
    allow_network: bool = False
    allow_dangerous_commands: bool = False
    dry_run: bool = False
    diff_preview_mode: str = "unified"
    max_read_chars: int = 14_000
    max_search_results: int = 200
    max_files_in_context: int = 80
    state_dir_name: str = ".marc_a1"
    ollama_num_ctx: int = 8_192
    router_num_ctx: int = 2_048
    ollama_temperature: float = 0.1
    auto_install_recommended_models: bool = True
    warmup_models_on_startup: bool = True
    warmup_timeout: int = 45
    auth_enabled: bool = True
    auth_secret_key: str | None = None
    auth_initial_admin_email: str | None = None
    auth_initial_admin_password: str | None = None
    auth_initial_admin_name: str = "Administrator"
    auth_initial_admin_totp_secret: str | None = None
    auth_cookie_secure: bool = True
    auth_session_idle_seconds: int = 1800
    auth_session_absolute_seconds: int = 43200
    auth_session_touch_interval_seconds: int = 60
    auth_min_password_length: int = 14
    security_allowed_hosts: tuple[str, ...] = (
        "127.0.0.1",
        "localhost",
        "::1",
        "testserver",
    )
    cors_allowed_origins: tuple[str, ...] = ()
    hsts_enabled: bool = True
    public_base_url: str | None = None

    @classmethod
    def from_sources(
        cls,
        workspace_override: str | None = None,
        config_path: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> "AppConfig":
        defaults = cls()
        config_file = Path(config_path).expanduser().resolve() if config_path else None
        source_root = _source_root_for_config_path(config_file)
        default_config = source_root / "config" / "agent.json"
        raw_json = _load_json_config(config_file or default_config)
        dotenv_values = _load_dotenv(source_root / ".env")
        env = os.environ

        def pick(name: str, fallback: Any) -> Any:
            if name in env:
                return env[name]
            if name in dotenv_values:
                return dotenv_values[name]
            return raw_json.get(name.lower(), raw_json.get(name, fallback))

        access_mode = _normalize_access_mode(
            pick("ACCESS_MODE", defaults.access_mode),
            read_only=_parse_bool(pick("READ_ONLY", defaults.read_only), defaults.read_only),
            approval_mode=_parse_bool(
                pick("APPROVAL_MODE", defaults.approval_mode),
                defaults.approval_mode,
            ),
            default=AccessMode(defaults.access_mode),
        )

        config = cls(
            ollama_host=str(pick("OLLAMA_HOST", defaults.ollama_host)),
            model_name=str(pick("MODEL_NAME", defaults.model_name)),
            router_model_name=pick("ROUTER_MODEL_NAME", defaults.router_model_name),
            workspace_root=str(pick("WORKSPACE_ROOT", defaults.workspace_root)),
            state_root_override=str(
                pick("STATE_ROOT_OVERRIDE", defaults.state_root_override) or ""
            ).strip()
            or None,
            access_mode=access_mode.value,
            max_iterations=_parse_int(
                pick("MAX_ITERATIONS", defaults.max_iterations),
                defaults.max_iterations,
            ),
            max_tool_calls=_parse_int(
                pick("MAX_TOOL_CALLS", defaults.max_tool_calls),
                defaults.max_tool_calls,
            ),
            max_repair_attempts=_parse_int(
                pick("MAX_REPAIR_ATTEMPTS", defaults.max_repair_attempts),
                defaults.max_repair_attempts,
            ),
            shell_timeout=_parse_int(
                pick("SHELL_TIMEOUT", defaults.shell_timeout),
                defaults.shell_timeout,
            ),
            llm_timeout=_parse_int(
                pick("LLM_TIMEOUT", defaults.llm_timeout),
                defaults.llm_timeout,
            ),
            router_timeout=_parse_int(
                pick("ROUTER_TIMEOUT", defaults.router_timeout),
                defaults.router_timeout,
            ),
            llm_request_retries=_parse_int(
                pick("LLM_REQUEST_RETRIES", defaults.llm_request_retries),
                defaults.llm_request_retries,
            ),
            llm_retry_backoff_ms=_parse_int(
                pick("LLM_RETRY_BACKOFF_MS", defaults.llm_retry_backoff_ms),
                defaults.llm_retry_backoff_ms,
            ),
            router_retries=_parse_int(
                pick("ROUTER_RETRIES", defaults.router_retries),
                defaults.router_retries,
            ),
            approval_mode=access_mode == AccessMode.APPROVAL,
            read_only=access_mode == AccessMode.SAFE,
            verbose=_parse_bool(pick("VERBOSE", defaults.verbose), defaults.verbose),
            allow_network=_parse_bool(
                pick("ALLOW_NETWORK", defaults.allow_network),
                defaults.allow_network,
            ),
            allow_dangerous_commands=_parse_bool(
                pick("ALLOW_DANGEROUS_COMMANDS", defaults.allow_dangerous_commands),
                defaults.allow_dangerous_commands,
            ),
            dry_run=_parse_bool(pick("DRY_RUN", defaults.dry_run), defaults.dry_run),
            diff_preview_mode=str(pick("DIFF_PREVIEW_MODE", defaults.diff_preview_mode)),
            max_read_chars=_parse_int(
                pick("MAX_READ_CHARS", defaults.max_read_chars),
                defaults.max_read_chars,
            ),
            max_search_results=_parse_int(
                pick("MAX_SEARCH_RESULTS", defaults.max_search_results),
                defaults.max_search_results,
            ),
            max_files_in_context=_parse_int(
                pick("MAX_FILES_IN_CONTEXT", defaults.max_files_in_context),
                defaults.max_files_in_context,
            ),
            state_dir_name=str(pick("STATE_DIR_NAME", defaults.state_dir_name)),
            ollama_num_ctx=_parse_int(
                pick("OLLAMA_NUM_CTX", defaults.ollama_num_ctx),
                defaults.ollama_num_ctx,
            ),
            router_num_ctx=_parse_int(
                pick("ROUTER_NUM_CTX", defaults.router_num_ctx),
                defaults.router_num_ctx,
            ),
            ollama_temperature=_parse_float(
                pick("OLLAMA_TEMPERATURE", defaults.ollama_temperature),
                defaults.ollama_temperature,
            ),
            auto_install_recommended_models=_parse_bool(
                pick(
                    "AUTO_INSTALL_RECOMMENDED_MODELS",
                    defaults.auto_install_recommended_models,
                ),
                defaults.auto_install_recommended_models,
            ),
            warmup_models_on_startup=_parse_bool(
                pick("WARMUP_MODELS_ON_STARTUP", defaults.warmup_models_on_startup),
                defaults.warmup_models_on_startup,
            ),
            warmup_timeout=_parse_int(
                pick("WARMUP_TIMEOUT", defaults.warmup_timeout),
                defaults.warmup_timeout,
            ),
            auth_enabled=_parse_bool(
                pick("AUTH_ENABLED", defaults.auth_enabled),
                defaults.auth_enabled,
            ),
            auth_secret_key=str(pick("AUTH_SECRET_KEY", defaults.auth_secret_key) or "").strip()
            or None,
            auth_initial_admin_email=str(
                pick("AUTH_INITIAL_ADMIN_EMAIL", defaults.auth_initial_admin_email) or ""
            ).strip()
            or None,
            auth_initial_admin_password=str(
                pick("AUTH_INITIAL_ADMIN_PASSWORD", defaults.auth_initial_admin_password) or ""
            ).strip()
            or None,
            auth_initial_admin_name=str(
                pick("AUTH_INITIAL_ADMIN_NAME", defaults.auth_initial_admin_name)
            ).strip()
            or defaults.auth_initial_admin_name,
            auth_initial_admin_totp_secret=str(
                pick("AUTH_INITIAL_ADMIN_TOTP_SECRET", defaults.auth_initial_admin_totp_secret)
                or ""
            ).strip()
            or None,
            auth_cookie_secure=_parse_bool(
                pick("AUTH_COOKIE_SECURE", defaults.auth_cookie_secure),
                defaults.auth_cookie_secure,
            ),
            auth_session_idle_seconds=_parse_int(
                pick("AUTH_SESSION_IDLE_SECONDS", defaults.auth_session_idle_seconds),
                defaults.auth_session_idle_seconds,
            ),
            auth_session_absolute_seconds=_parse_int(
                pick("AUTH_SESSION_ABSOLUTE_SECONDS", defaults.auth_session_absolute_seconds),
                defaults.auth_session_absolute_seconds,
            ),
            auth_session_touch_interval_seconds=_parse_int(
                pick(
                    "AUTH_SESSION_TOUCH_INTERVAL_SECONDS",
                    defaults.auth_session_touch_interval_seconds,
                ),
                defaults.auth_session_touch_interval_seconds,
            ),
            auth_min_password_length=_parse_int(
                pick("AUTH_MIN_PASSWORD_LENGTH", defaults.auth_min_password_length),
                defaults.auth_min_password_length,
            ),
            security_allowed_hosts=_parse_csv_list(
                pick(
                    "SECURITY_ALLOWED_HOSTS",
                    pick("ALLOWED_HOSTS", defaults.security_allowed_hosts),
                ),
                defaults.security_allowed_hosts,
            ),
            cors_allowed_origins=_parse_csv_list(
                pick("CORS_ALLOWED_ORIGINS", defaults.cors_allowed_origins),
                defaults.cors_allowed_origins,
            ),
            hsts_enabled=_parse_bool(
                pick("HSTS_ENABLED", defaults.hsts_enabled),
                defaults.hsts_enabled,
            ),
            public_base_url=str(pick("PUBLIC_BASE_URL", defaults.public_base_url) or "").strip()
            or None,
        ).normalized()

        if workspace_override:
            config = replace(config, workspace_root=workspace_override)
        if overrides:
            candidate = config
            for key, value in overrides.items():
                if value is None:
                    continue
                candidate = replace(candidate, **{key: value})
            config = candidate.normalized()

        workspace_path = Path(config.workspace_root).expanduser().resolve()
        return replace(config, workspace_root=str(workspace_path))

    def normalized(self) -> "AppConfig":
        mode = _normalize_access_mode(
            self.access_mode,
            read_only=self.read_only,
            approval_mode=self.approval_mode,
            default=AccessMode.APPROVAL,
        )
        return replace(
            self,
            access_mode=mode.value,
            read_only=mode == AccessMode.SAFE,
            approval_mode=mode == AccessMode.APPROVAL,
        )

    @property
    def workspace_path(self) -> Path:
        return Path(self.workspace_root).expanduser().resolve()

    @property
    def state_root(self) -> Path:
        if self.state_root_override:
            return Path(self.state_root_override).expanduser().resolve()
        return self.workspace_path / self.state_dir_name

    @property
    def session_dir_path(self) -> Path:
        return self.state_root / "sessions"

    @property
    def log_dir_path(self) -> Path:
        return self.state_root / "logs"

    @property
    def memory_dir_path(self) -> Path:
        return self.state_root / "memory"

    @property
    def helper_dir_path(self) -> Path:
        return self.state_root / "helpers"

    @property
    def report_dir_path(self) -> Path:
        return self.state_root / "reports"

    @property
    def auth_db_path(self) -> Path:
        return self.state_root / "auth.db"

    @property
    def full_access(self) -> bool:
        return self.access_mode == AccessMode.FULL.value

    def ensure_state_dirs(self) -> None:
        self.session_dir_path.mkdir(parents=True, exist_ok=True)
        self.log_dir_path.mkdir(parents=True, exist_ok=True)
        self.memory_dir_path.mkdir(parents=True, exist_ok=True)
        self.helper_dir_path.mkdir(parents=True, exist_ok=True)
        self.report_dir_path.mkdir(parents=True, exist_ok=True)

    def to_public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for secret_key in (
            "auth_secret_key",
            "auth_initial_admin_password",
            "auth_initial_admin_totp_secret",
        ):
            data.pop(secret_key, None)
        data["workspace_root"] = str(self.workspace_path)
        data["state_root"] = str(self.state_root)
        data["session_dir"] = str(self.session_dir_path)
        data["log_dir"] = str(self.log_dir_path)
        data["memory_dir"] = str(self.memory_dir_path)
        data["helper_dir"] = str(self.helper_dir_path)
        data["report_dir"] = str(self.report_dir_path)
        data["full_access"] = self.full_access
        data["path_scope"] = "system" if self.full_access else "workspace"
        data["branding"] = {
            "name": AGENT_NAME,
            "full_name": AGENT_FULL_NAME,
            "tagline": AGENT_TAGLINE,
        }
        data["access_modes"] = [item.value for item in AccessMode]
        data["model_candidates"] = [self.model_name]
        data["router_preferred_model_name"] = self.router_model_name
        data["agent_profiles"] = [
            {
                "id": "a2",
                "label": "MARC A2 Memory",
                "description": "Mehrschichtiges Memory fuer Cross-Run-Recall, Repair-Muster und Repo-Kontext.",
            },
            {
                "id": "a1",
                "label": "MARC A1 Classic",
                "description": "Leichter Legacy-Pfad ohne das neue Layered-Memory-System.",
            },
        ]
        data["execution_profiles"] = [
            {
                "id": "fast",
                "label": "Schnell",
                "description": "Kuerzerer Lauf mit weniger Iterationen.",
            },
            {
                "id": "balanced",
                "label": "Ausgewogen",
                "description": "Standardprofil fuer die meisten Aufgaben.",
            },
            {
                "id": "deep",
                "label": "Tief",
                "description": "Mehr Iterationen und groessere Verifikationstiefe.",
            },
        ]
        data["capabilities"] = {
            "session_archiving": True,
            "mfa_totp": True,
        }
        return data
