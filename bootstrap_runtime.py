from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import shutil
import shlex
import site
import subprocess
import sys
import sysconfig
import time
from contextlib import suppress
from importlib import metadata as importlib_metadata
from pathlib import Path
from urllib import error, request


RUNTIME_MODULES = ("fastapi", "httpx", "pydantic", "uvicorn")
EXTERNALLY_MANAGED_MARKERS = (
    "externally-managed-environment",
    "externally managed",
    "break-system-packages",
)
RUNTIME_VENV_DIRNAME = "runtime-venv"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_WINDOWS_INSTALL_SCRIPT = "irm https://ollama.com/install.ps1 | iex"
OLLAMA_STARTUP_TIMEOUT_SECONDS = 90


def ensure_runtime_dependencies(
    *,
    requirements_file: str = "requirements-runtime.txt",
    modules: tuple[str, ...] = RUNTIME_MODULES,
) -> None:
    project_root = Path(__file__).resolve().parent
    requirements_path = project_root / requirements_file
    if not requirements_path.exists():
        raise RuntimeError(f"Missing requirements file: {requirements_path}")

    if _runtime_requirements_satisfied(requirements_path, modules):
        return

    existing_runtime_venv = runtime_venv_path(project_root)
    existing_runtime_python = runtime_python_path(existing_runtime_venv)
    if existing_runtime_python.exists() and _runtime_requirements_satisfied_in_interpreter(
        existing_runtime_python,
        requirements_path,
        modules,
    ):
        _activate_runtime_venv(existing_runtime_venv)
        if _runtime_requirements_satisfied(requirements_path, modules):
            return

    missing_requirements = _unsatisfied_runtime_requirements(requirements_path)
    print(
        "[M.A.R.C A1] Fehlende oder unpassende Python-Abhaengigkeiten erkannt: "
        f"{_summarize_requirement_list(missing_requirements)}"
    )
    print(f"[M.A.R.C A1] Installiere Runtime-Pakete aus {requirements_path.name} ...")

    _ensure_pip()
    current_command = build_pip_install_command(requirements_path)
    current_install = _run_command(current_command, capture_output=True)
    if current_install.returncode == 0:
        _refresh_current_import_paths()
        if _runtime_requirements_satisfied(requirements_path, modules):
            return
        print(
            "[M.A.R.C A1] Pakete wurden geprueft oder installiert, "
            "sind im aktuellen Interpreter aber noch nicht vollstaendig sichtbar."
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
    if not _runtime_requirements_satisfied_in_interpreter(
        runtime_python,
        requirements_path,
        modules,
    ):
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
    if not _runtime_requirements_satisfied(requirements_path, modules):
        raise RuntimeError(
            "Die Runtime-Abhaengigkeiten wurden installiert, sind aber noch nicht importierbar."
        )


def ensure_ollama_runtime(ollama_host: str | None = None) -> None:
    host = (ollama_host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).strip()
    if _ollama_api_ready(host):
        return

    ollama_binary = _find_ollama_binary()
    if ollama_binary is None and _should_auto_install_ollama():
        print("[M.A.R.C A1] Ollama wurde nicht gefunden. Starte die Windows-Installation ...")
        _install_ollama_windows()
        ollama_binary = _find_ollama_binary()

    if ollama_binary is None:
        raise RuntimeError(_missing_ollama_message(host))

    if _ollama_api_ready(host):
        return

    if not _is_local_ollama_host(host):
        raise RuntimeError(
            "OLLAMA_HOST ist nicht erreichbar und zeigt nicht auf eine lokale Adresse. "
            f"Pruefe den Server unter {host}."
        )

    if _should_auto_start_ollama():
        print("[M.A.R.C A1] Starte lokalen Ollama-Dienst ...")
        _start_ollama_server(ollama_binary)

    if _wait_for_ollama_api(host, timeout_seconds=OLLAMA_STARTUP_TIMEOUT_SECONDS):
        return

    raise RuntimeError(
        "Ollama wurde gefunden, aber die API ist nicht erreichbar. "
        f"Pruefe den Dienst unter {host} oder starte 'ollama serve' manuell."
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


def _runtime_requirements_satisfied(
    requirements_path: Path,
    modules: tuple[str, ...],
) -> bool:
    return not _unsatisfied_runtime_requirements(requirements_path) and _modules_available(
        modules
    )


def _runtime_requirements_satisfied_in_interpreter(
    python_executable: str | Path,
    requirements_path: Path,
    modules: tuple[str, ...],
) -> bool:
    code = """
import importlib.util
import json
import re
import shlex
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

with_packaging = False
Requirement = None
with_version = None
try:
    from packaging.requirements import Requirement as _Requirement
    from packaging.version import Version as _Version
    Requirement = _Requirement
    with_version = _Version
    with_packaging = True
except Exception:
    try:
        from pip._vendor.packaging.requirements import Requirement as _Requirement
        from pip._vendor.packaging.version import Version as _Version
        Requirement = _Requirement
        with_version = _Version
        with_packaging = True
    except Exception:
        with_packaging = False

NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+")


def normalize_name(value):
    return re.sub(r"[-_.]+", "-", value).lower()


def strip_comment(line):
    if " #" in line:
        return line.split(" #", 1)[0].rstrip()
    return line


def read_specs(path, seen=None):
    seen = seen or set()
    resolved = path.resolve()
    if resolved in seen:
        return []
    seen.add(resolved)
    specs = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = strip_comment(raw_line.strip())
        if not line or line.startswith("#"):
            continue
        parts = shlex.split(line, comments=False, posix=True)
        if not parts:
            continue
        head = parts[0]
        if head in {"-r", "--requirement"} and len(parts) >= 2:
            specs.extend(read_specs((path.parent / parts[1]).resolve(), seen))
            continue
        if head.startswith("-r") and len(head) > 2:
            specs.extend(read_specs((path.parent / head[2:]).resolve(), seen))
            continue
        if head.startswith("--requirement="):
            specs.extend(read_specs((path.parent / head.split("=", 1)[1]).resolve(), seen))
            continue
        if head.startswith("-"):
            continue
        specs.append(line)
    return specs


def requirement_satisfied(spec):
    if with_packaging:
        requirement = Requirement(spec)
        if requirement.marker and not requirement.marker.evaluate():
            return True
        try:
            installed_version = importlib_metadata.version(requirement.name)
        except importlib_metadata.PackageNotFoundError:
            return False
        if requirement.specifier:
            return requirement.specifier.contains(installed_version, prereleases=True)
        return True

    match = NAME_PATTERN.match(spec)
    if not match:
        return False
    name = normalize_name(match.group(0))
    try:
        importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return False
    return True


requirements_path = Path(sys.argv[1])
modules = json.loads(sys.argv[2])
ok = all(importlib.util.find_spec(module) is not None for module in modules)
if ok:
    for requirement_spec in read_specs(requirements_path):
        if not requirement_satisfied(requirement_spec):
            ok = False
            break
print(json.dumps(ok))
""".strip()
    completed = _run_command(
        [str(python_executable), "-c", code, str(requirements_path), json.dumps(list(modules))],
        capture_output=True,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "true"


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


def _unsatisfied_runtime_requirements(requirements_path: Path) -> list[str]:
    unsatisfied: list[str] = []
    for requirement_spec in _read_requirement_specs(requirements_path):
        if not _requirement_spec_satisfied(requirement_spec):
            unsatisfied.append(requirement_spec)
    return unsatisfied


def _summarize_requirement_list(requirements: list[str]) -> str:
    if not requirements:
        return "Importpfade oder Laufzeitmodule sind unvollstaendig"
    preview = ", ".join(requirements[:4])
    if len(requirements) > 4:
        preview += ", ..."
    return preview


def _read_requirement_specs(
    requirements_path: Path,
    seen_paths: set[Path] | None = None,
) -> list[str]:
    resolved_path = requirements_path.resolve()
    seen = seen_paths or set()
    if resolved_path in seen:
        return []
    seen.add(resolved_path)

    specs: list[str] = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = _strip_requirement_comment(raw_line.strip())
        if not line or line.startswith("#"):
            continue
        parts = shlex.split(line, comments=False, posix=True)
        if not parts:
            continue
        head = parts[0]
        if head in {"-r", "--requirement"} and len(parts) >= 2:
            nested = (requirements_path.parent / parts[1]).resolve()
            specs.extend(_read_requirement_specs(nested, seen))
            continue
        if head.startswith("-r") and len(head) > 2:
            nested = (requirements_path.parent / head[2:]).resolve()
            specs.extend(_read_requirement_specs(nested, seen))
            continue
        if head.startswith("--requirement="):
            nested = (requirements_path.parent / head.split("=", 1)[1]).resolve()
            specs.extend(_read_requirement_specs(nested, seen))
            continue
        if head.startswith("-"):
            continue
        specs.append(line)
    return specs


def _strip_requirement_comment(line: str) -> str:
    if " #" in line:
        return line.split(" #", 1)[0].rstrip()
    return line


def _requirement_spec_satisfied(requirement_spec: str) -> bool:
    requirement = _parse_requirement_spec(requirement_spec)
    if requirement is None:
        return False
    if requirement.get("marker_skipped"):
        return True

    try:
        installed_version = importlib_metadata.version(requirement["name"])
    except importlib_metadata.PackageNotFoundError:
        return False

    specifier = requirement.get("specifier")
    if specifier is None:
        return True
    if "packaging_requirement" in requirement:
        return specifier.contains(installed_version, prereleases=True)
    return True


def _parse_requirement_spec(requirement_spec: str) -> dict[str, object] | None:
    with suppress(Exception):
        from packaging.requirements import Requirement

        parsed = Requirement(requirement_spec)
        if parsed.marker and not parsed.marker.evaluate():
            return {"name": parsed.name, "marker_skipped": True}
        return {
            "name": parsed.name,
            "specifier": parsed.specifier,
            "packaging_requirement": parsed,
        }

    with suppress(Exception):
        from pip._vendor.packaging.requirements import Requirement

        parsed = Requirement(requirement_spec)
        if parsed.marker and not parsed.marker.evaluate():
            return {"name": parsed.name, "marker_skipped": True}
        return {
            "name": parsed.name,
            "specifier": parsed.specifier,
            "packaging_requirement": parsed,
        }

    match = re.match(r"^(?P<name>[A-Za-z0-9_.-]+)", requirement_spec)
    if not match:
        return None
    return {"name": match.group("name"), "specifier": None}


def _ensure_pip() -> None:
    try:
        import pip  # noqa: F401
    except ImportError:
        import ensurepip

        ensurepip.bootstrap(upgrade=True)


def _find_ollama_binary() -> Path | None:
    command = shutil.which("ollama")
    if command:
        return Path(command)

    if os.name != "nt":
        return None

    for candidate in _windows_ollama_candidate_paths():
        if candidate.exists():
            _prepend_command_path(candidate.parent)
            return candidate
    return None


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


def _ollama_api_ready(host: str) -> bool:
    target = host.rstrip("/") + "/api/tags"
    req = request.Request(target, method="GET")
    try:
        with request.urlopen(req, timeout=2) as response:
            return 200 <= getattr(response, "status", 200) < 500
    except (error.URLError, TimeoutError, ValueError):
        return False


def _wait_for_ollama_api(host: str, *, timeout_seconds: int) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _ollama_api_ready(host):
            return True
        time.sleep(1)
    return False


def _install_ollama_windows() -> None:
    if os.name != "nt":
        raise RuntimeError("Automatische Ollama-Installation ist nur fuer Windows hinterlegt.")

    completed = _run_command(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            OLLAMA_WINDOWS_INSTALL_SCRIPT,
        ],
        capture_output=True,
    )
    if completed.returncode != 0:
        _print_command_output(completed)
        raise RuntimeError(
            "Die automatische Ollama-Installation ist fehlgeschlagen. "
            f"Fuehre in PowerShell aus: {OLLAMA_WINDOWS_INSTALL_SCRIPT}"
        )

    with suppress(Exception):
        for candidate in _windows_ollama_candidate_paths():
            if candidate.exists():
                _prepend_command_path(candidate.parent)
                break


def _start_ollama_server(ollama_binary: Path) -> None:
    kwargs: dict[str, object] = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
        "close_fds": os.name != "nt",
    }
    if os.name == "nt":
        creationflags = 0
        for flag_name in ("DETACHED_PROCESS", "CREATE_NEW_PROCESS_GROUP", "CREATE_NO_WINDOW"):
            creationflags |= int(getattr(subprocess, flag_name, 0))
        kwargs["creationflags"] = creationflags
    else:
        kwargs["start_new_session"] = True

    subprocess.Popen([str(ollama_binary), "serve"], **kwargs)


def _windows_ollama_candidate_paths() -> list[Path]:
    candidates: list[Path] = []
    for env_name in ("LOCALAPPDATA", "ProgramFiles", "ProgramW6432"):
        base = os.environ.get(env_name, "").strip()
        if not base:
            continue
        candidates.append(Path(base) / "Programs" / "Ollama" / "ollama.exe")
        candidates.append(Path(base) / "Ollama" / "ollama.exe")
    seen: set[str] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        key = candidate.as_posix().lower()
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)
    return unique_candidates


def _prepend_command_path(path: Path) -> None:
    value = str(path)
    current_path = os.environ.get("PATH", "")
    parts = current_path.split(os.pathsep) if current_path else []
    if value not in parts:
        os.environ["PATH"] = value + (os.pathsep + current_path if current_path else "")


def _should_auto_install_ollama() -> bool:
    if os.name != "nt":
        return False
    return os.environ.get("MARC_A1_AUTO_INSTALL_OLLAMA", "1").strip().lower() not in {"0", "false", "no", "off"}


def _should_auto_start_ollama() -> bool:
    return os.environ.get("MARC_A1_AUTO_START_OLLAMA", "1").strip().lower() not in {"0", "false", "no", "off"}


def _is_local_ollama_host(host: str) -> bool:
    normalized = host.strip().lower()
    return normalized.startswith("http://127.0.0.1") or normalized.startswith("http://localhost") or normalized.startswith("http://0.0.0.0")


def _missing_ollama_message(host: str) -> str:
    if os.name == "nt":
        return (
            "Ollama ist nicht installiert oder nicht im PATH. "
            "Die App versucht es auf Windows automatisch ueber den offiziellen Installer. "
            f"Wenn das fehlschlaegt, fuehre in PowerShell aus: {OLLAMA_WINDOWS_INSTALL_SCRIPT}. "
            f"Danach muss die API unter {host} erreichbar sein."
        )
    return (
        "Ollama ist nicht installiert oder nicht im PATH. "
        "Installiere Ollama und starte danach 'ollama serve'. "
        f"Erwartete API-Adresse: {host}."
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
