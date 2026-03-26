from __future__ import annotations

import importlib
import importlib.util
import json
import os
import site
import subprocess
import sys
import sysconfig
from pathlib import Path


RUNTIME_MODULES = ("fastapi", "httpx", "pydantic", "uvicorn")
EXTERNALLY_MANAGED_MARKERS = (
    "externally-managed-environment",
    "externally managed",
    "break-system-packages",
)
RUNTIME_VENV_DIRNAME = "runtime-venv"


def ensure_runtime_dependencies(
    *,
    requirements_file: str = "requirements-runtime.txt",
    modules: tuple[str, ...] = RUNTIME_MODULES,
) -> None:
    if _modules_available(modules):
        return

    project_root = Path(__file__).resolve().parent
    existing_runtime_venv = runtime_venv_path(project_root)
    existing_runtime_python = runtime_python_path(existing_runtime_venv)
    if existing_runtime_python.exists() and _modules_available_in_interpreter(
        existing_runtime_python,
        modules,
    ):
        _activate_runtime_venv(existing_runtime_venv)
        if _modules_available(modules):
            return

    requirements_path = project_root / requirements_file
    if not requirements_path.exists():
        raise RuntimeError(f"Missing requirements file: {requirements_path}")

    print("[M.A.R.C A1] Fehlende Python-Abhaengigkeiten erkannt.")
    print(f"[M.A.R.C A1] Installiere Runtime-Pakete aus {requirements_path.name} ...")

    _ensure_pip()
    current_command = build_pip_install_command(requirements_path)
    current_install = _run_command(current_command, capture_output=True)
    if current_install.returncode == 0:
        _refresh_current_import_paths()
        if _modules_available(modules):
            return
        print(
            "[M.A.R.C A1] Pakete wurden installiert, sind im aktuellen Interpreter aber noch nicht sichtbar."
        )
    else:
        _print_command_output(current_install)
        if _is_externally_managed_error(current_install):
            print(
                "[M.A.R.C A1] Extern verwaltete Python-Umgebung erkannt. "
                "Wechsle automatisch auf ein lokales Runtime-Venv."
            )
        else:
            print(
                "[M.A.R.C A1] Direkte Installation fehlgeschlagen. "
                "Versuche stattdessen ein lokales Runtime-Venv."
            )

    runtime_venv_dir = existing_runtime_venv
    runtime_python = _ensure_runtime_venv(runtime_venv_dir)
    if not _modules_available_in_interpreter(runtime_python, modules):
        print(
            "[M.A.R.C A1] Installiere Runtime-Pakete in "
            f"{_display_path_for_log(runtime_venv_dir, project_root)} ..."
        )
        venv_install = _run_command(
            build_pip_install_command(
                requirements_path,
                python_executable=runtime_python,
                scope="global",
            ),
            capture_output=True,
        )
        if venv_install.returncode != 0:
            _print_command_output(venv_install)
            raise RuntimeError(
                "Automatische Installation der Runtime-Abhaengigkeiten ist fehlgeschlagen."
            )

    _activate_runtime_venv(runtime_venv_dir)
    if not _modules_available(modules):
        raise RuntimeError(
            "Die Runtime-Abhaengigkeiten wurden installiert, sind aber noch nicht importierbar."
        )


def build_pip_install_command(
    requirements_path: Path,
    *,
    python_executable: str | Path | None = None,
    scope: str | None = None,
) -> list[str]:
    python = str(python_executable or sys.executable)
    command = [python, "-m", "pip", "install"]
    if _should_use_user_site(scope=scope, python_executable=python):
        command.append("--user")
    extra_args = os.environ.get("MARC_A1_PIP_EXTRA_ARGS", "").strip()
    if extra_args:
        command.extend(extra_args.split())
    command.extend(["-r", str(requirements_path)])
    return command


def runtime_venv_path(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parent
    return root / ".marc_a1" / RUNTIME_VENV_DIRNAME


def runtime_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _display_path_for_log(path: Path, project_root: Path | None = None) -> str:
    resolved = path.expanduser().resolve(strict=False)
    if project_root is not None:
        base = project_root.expanduser().resolve(strict=False)
        try:
            return resolved.relative_to(base).as_posix()
        except ValueError:
            pass
    return resolved.as_posix()


def _modules_available(modules: tuple[str, ...]) -> bool:
    return all(importlib.util.find_spec(module) is not None for module in modules)


def _modules_available_in_interpreter(
    python_executable: str | Path,
    modules: tuple[str, ...],
) -> bool:
    code = (
        "import importlib.util, json; "
        f"mods = {json.dumps(list(modules))}; "
        "print(json.dumps(all(importlib.util.find_spec(m) is not None for m in mods)))"
    )
    completed = _run_command(
        [str(python_executable), "-c", code],
        capture_output=True,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def _ensure_pip() -> None:
    try:
        import pip  # noqa: F401
    except ImportError:
        import ensurepip

        ensurepip.bootstrap(upgrade=True)


def _should_use_user_site(
    *,
    scope: str | None = None,
    python_executable: str | Path | None = None,
) -> bool:
    selected_scope = (scope or os.environ.get("MARC_A1_PIP_SCOPE", "auto")).strip().lower()
    if selected_scope == "user":
        return True
    if selected_scope == "global":
        return False
    if python_executable and str(python_executable) != sys.executable:
        return False
    return sys.prefix == getattr(sys, "base_prefix", sys.prefix)


def _run_command(
    command: list[str],
    *,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        capture_output=capture_output,
        check=False,
    )


def _print_command_output(completed: subprocess.CompletedProcess[str]) -> None:
    output = "\n".join(
        part.strip()
        for part in (completed.stdout or "", completed.stderr or "")
        if part and part.strip()
    )
    if output:
        print(output)


def _is_externally_managed_error(completed: subprocess.CompletedProcess[str]) -> bool:
    text = "\n".join(
        part.lower()
        for part in (completed.stdout or "", completed.stderr or "")
        if part
    )
    return any(marker in text for marker in EXTERNALLY_MANAGED_MARKERS)


def _ensure_runtime_venv(venv_dir: Path) -> Path:
    python_path = runtime_python_path(venv_dir)
    if python_path.exists():
        return python_path

    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[M.A.R.C A1] Erzeuge lokales Runtime-Venv unter {venv_dir.as_posix()} ..."
    )
    completed = _run_command([sys.executable, "-m", "venv", str(venv_dir)], capture_output=True)
    if completed.returncode != 0:
        _print_command_output(completed)
        raise RuntimeError("Lokales Runtime-Venv konnte nicht erstellt werden.")
    if not python_path.exists():
        raise RuntimeError("Lokales Runtime-Venv wurde erstellt, aber Python fehlt darin.")
    return python_path


def _activate_runtime_venv(venv_dir: Path) -> None:
    python_path = runtime_python_path(venv_dir)
    site_paths = _site_packages_for_interpreter(python_path)
    for site_path in reversed(site_paths):
        if site_path not in sys.path:
            sys.path.insert(0, site_path)
            site.addsitedir(site_path)

    bin_dir = str(python_path.parent)
    os.environ["VIRTUAL_ENV"] = str(venv_dir)
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    os.environ["MARC_A1_RUNTIME_VENV"] = str(venv_dir)
    importlib.invalidate_caches()


def _site_packages_for_interpreter(python_executable: str | Path) -> list[str]:
    code = (
        "import json, sysconfig; "
        "paths = sysconfig.get_paths(); "
        "items = [paths.get('purelib'), paths.get('platlib')]; "
        "print(json.dumps([item for item in items if item]))"
    )
    completed = _run_command(
        [str(python_executable), "-c", code],
        capture_output=True,
    )
    if completed.returncode != 0:
        _print_command_output(completed)
        raise RuntimeError("Site-packages des Runtime-Venv konnten nicht ermittelt werden.")
    paths = json.loads(completed.stdout.strip() or "[]")
    unique_paths: list[str] = []
    for path in paths:
        if path and path not in unique_paths:
            unique_paths.append(path)
    return unique_paths


def _refresh_current_import_paths() -> None:
    importlib.invalidate_caches()
    if _should_use_user_site() and site.ENABLE_USER_SITE:
        user_site = site.getusersitepackages()
        if user_site and user_site not in sys.path:
            site.addsitedir(user_site)
            sys.path.insert(0, user_site)
    current_site = sysconfig.get_paths().get("purelib")
    if current_site and current_site not in sys.path:
        site.addsitedir(current_site)
        sys.path.insert(0, current_site)
