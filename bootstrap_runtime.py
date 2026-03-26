from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


RUNTIME_MODULES = ("fastapi", "httpx", "pydantic", "uvicorn")


def ensure_runtime_dependencies(
    *,
    requirements_file: str = "requirements-runtime.txt",
    modules: tuple[str, ...] = RUNTIME_MODULES,
) -> None:
    if _modules_available(modules):
        return

    requirements_path = Path(__file__).resolve().parent / requirements_file
    if not requirements_path.exists():
        raise RuntimeError(f"Missing requirements file: {requirements_path}")

    print("[M.A.R.C A1] Fehlende Python-Abhaengigkeiten erkannt.")
    print(f"[M.A.R.C A1] Installiere Runtime-Pakete aus {requirements_path.name} ...")

    _ensure_pip()
    command = build_pip_install_command(requirements_path)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "Automatische Installation der Runtime-Abhaengigkeiten ist fehlgeschlagen."
        )

    if not _modules_available(modules):
        raise RuntimeError(
            "Die Runtime-Abhaengigkeiten wurden installiert, sind aber noch nicht importierbar."
        )


def build_pip_install_command(requirements_path: Path) -> list[str]:
    command = [sys.executable, "-m", "pip", "install"]
    if _should_use_user_site():
        command.append("--user")
    extra_args = os.environ.get("MARC_A1_PIP_EXTRA_ARGS", "").strip()
    if extra_args:
        command.extend(extra_args.split())
    command.extend(["-r", str(requirements_path)])
    return command


def _modules_available(modules: tuple[str, ...]) -> bool:
    return all(importlib.util.find_spec(module) is not None for module in modules)


def _ensure_pip() -> None:
    try:
        import pip  # noqa: F401
    except ImportError:
        import ensurepip

        ensurepip.bootstrap(upgrade=True)


def _should_use_user_site() -> bool:
    scope = os.environ.get("MARC_A1_PIP_SCOPE", "auto").strip().lower()
    if scope == "user":
        return True
    if scope == "global":
        return False
    return sys.prefix == sys.base_prefix
