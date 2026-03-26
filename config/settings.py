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
    model_name: str = "qwen3-coder:30b"
    workspace_root: str = "."
    access_mode: str = AccessMode.APPROVAL.value
    max_iterations: int = 18
    max_tool_calls: int = 32
    max_repair_attempts: int = 3
    shell_timeout: int = 120
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
    ollama_num_ctx: int = 32_768
    ollama_temperature: float = 0.1

    @classmethod
    def from_sources(
        cls,
        workspace_override: str | None = None,
        config_path: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> "AppConfig":
        defaults = cls()
        cwd = Path.cwd()
        config_file = Path(config_path).expanduser().resolve() if config_path else None
        default_config = cwd / "config" / "agent.json"
        raw_json = _load_json_config(config_file or default_config)
        dotenv_values = _load_dotenv(cwd / ".env")
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
            workspace_root=str(pick("WORKSPACE_ROOT", defaults.workspace_root)),
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
            ollama_temperature=_parse_float(
                pick("OLLAMA_TEMPERATURE", defaults.ollama_temperature),
                defaults.ollama_temperature,
            ),
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
    def full_access(self) -> bool:
        return self.access_mode == AccessMode.FULL.value

    def ensure_state_dirs(self) -> None:
        self.session_dir_path.mkdir(parents=True, exist_ok=True)
        self.log_dir_path.mkdir(parents=True, exist_ok=True)
        self.memory_dir_path.mkdir(parents=True, exist_ok=True)
        self.helper_dir_path.mkdir(parents=True, exist_ok=True)

    def to_public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["workspace_root"] = str(self.workspace_path)
        data["state_root"] = str(self.state_root)
        data["session_dir"] = str(self.session_dir_path)
        data["log_dir"] = str(self.log_dir_path)
        data["memory_dir"] = str(self.memory_dir_path)
        data["helper_dir"] = str(self.helper_dir_path)
        data["full_access"] = self.full_access
        data["path_scope"] = "system" if self.full_access else "workspace"
        data["branding"] = {
            "name": AGENT_NAME,
            "full_name": AGENT_FULL_NAME,
            "tagline": AGENT_TAGLINE,
        }
        data["access_modes"] = [item.value for item in AccessMode]
        return data
