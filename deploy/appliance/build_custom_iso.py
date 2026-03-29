#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
APPLIANCE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = APPLIANCE_DIR / "templates"
PRESEED_ARGS = "auto=true priority=critical file=/cdrom/preseed.cfg"
BOOT_CONFIGS = (
    "isolinux/gtk.cfg",
    "isolinux/txt.cfg",
    "isolinux/adgtk.cfg",
    "isolinux/adtxt.cfg",
    "isolinux/drkgtk.cfg",
    "isolinux/drk.cfg",
    "isolinux/addrkgtk.cfg",
    "isolinux/addrk.cfg",
    "boot/grub/grub.cfg",
)
BUNDLE_EXCLUDE_PARTS = {
    ".git",
    ".marc_a1",
    ".pytest_cache",
    ".tmp_iso_work",
    "__pycache__",
    "deploy",
    "docs",
    "ollama_usb_staging",
    "tests",
}
BUNDLE_EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
BUNDLE_EXCLUDE_NAMES = {".DS_Store"}


def run(command: list[str], *, cwd: Path | None = None, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def git_output(args: list[str]) -> str:
    result = run(["git", *args], cwd=REPO_ROOT, capture_output=True)
    return result.stdout.strip()


def render_template(path: Path, replacements: dict[str, str]) -> str:
    text = path.read_text(encoding="utf-8")
    for key, value in replacements.items():
        text = text.replace(f"@@{key}@@", value)
    return text


def should_bundle(relative_path: Path) -> bool:
    if not relative_path.parts:
        return False
    if any(part in BUNDLE_EXCLUDE_PARTS for part in relative_path.parts):
        return False
    if relative_path.name in BUNDLE_EXCLUDE_NAMES:
        return False
    if relative_path.suffix in BUNDLE_EXCLUDE_SUFFIXES:
        return False
    return True


def create_bundle(bundle_path: Path) -> None:
    with tarfile.open(bundle_path, "w:gz") as archive:
        for path in sorted(REPO_ROOT.rglob("*")):
            relative_path = path.relative_to(REPO_ROOT)
            if not should_bundle(relative_path):
                continue
            arcname = Path("marc-a1") / relative_path
            archive.add(path, arcname=str(arcname), recursive=False)


def extract_iso(base_iso: Path, output_dir: Path) -> None:
    run(
        [
            "xorriso",
            "-osirrox",
            "on",
            "-indev",
            str(base_iso),
            "-extract",
            "/",
            str(output_dir),
        ]
    )
    for path in output_dir.rglob("*"):
        current_mode = path.stat().st_mode
        if path.is_dir():
            path.chmod(current_mode | 0o700)
        else:
            path.chmod(current_mode | 0o600)


def patch_boot_config(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    patched: list[str] = []
    for line in lines:
        if "/install.amd/vmlinuz" in line and "---" in line:
            if "rescue/enable=true" in line or "priority=low" in line:
                patched.append(line)
                continue
            if "file=/cdrom/preseed.cfg" in line:
                patched.append(line)
                continue
            injection = "file=/cdrom/preseed.cfg" if "auto=true" in line else PRESEED_ARGS
            patched.append(line.replace(" ---", f" {injection} ---"))
            continue
        patched.append(line)

    text = "\n".join(patched) + "\n"
    if path.as_posix().endswith("isolinux/isolinux.cfg"):
        text = text.replace("timeout 0", "timeout 50")
    if path.as_posix().endswith("boot/grub/grub.cfg") and "set timeout=" not in text:
        text = "set timeout=5\n" + text
    path.write_text(text, encoding="utf-8")


def update_md5sums(iso_root: Path) -> None:
    entries: list[str] = []
    for path in sorted(iso_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(iso_root).as_posix()
        if rel in {"md5sum.txt", "isolinux/boot.cat"}:
            continue
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        entries.append(f"{digest}  ./{rel}")
    (iso_root / "md5sum.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")


def boot_arguments(base_iso: Path) -> list[str]:
    result = run(
        [
            "xorriso",
            "-indev",
            str(base_iso),
            "-report_el_torito",
            "as_mkisofs",
        ],
        capture_output=True,
    )
    args: list[str] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line.startswith("-"):
            continue
        args.extend(shlex.split(line))
    return args


def build_iso(iso_root: Path, base_iso: Path, output_iso: Path) -> None:
    output_iso.parent.mkdir(parents=True, exist_ok=True)
    args = boot_arguments(base_iso)
    command = [
        "xorriso",
        "-as",
        "mkisofs",
        "-r",
        "-J",
        "-joliet-long",
        "-cache-inodes",
        "-o",
        str(output_iso),
        *args,
        str(iso_root),
    ]
    run(command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a MARC A1 appliance Debian ISO.")
    parser.add_argument("--base-iso", required=True, type=Path)
    parser.add_argument("--output-iso", required=True, type=Path)
    parser.add_argument("--hostname", default="marc-a1")
    parser.add_argument("--install-username", default="marc")
    parser.add_argument("--full-name", default="MARC Operator")
    parser.add_argument("--timezone", default="Europe/Berlin")
    parser.add_argument("--locale", default="en_US.UTF-8")
    parser.add_argument("--keymap", default="de")
    parser.add_argument("--model-name", default="qwen2.5-coder:14b")
    parser.add_argument("--router-model-name", default="qwen2.5-coder:14b")
    parser.add_argument("--access-mode", default="full", choices=["safe", "approval", "full"])
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--state-root-override", default="/var/lib/marc-a1/state")
    parser.add_argument("--ollama-context-length", default="8192")
    parser.add_argument("--ollama-num-ctx", default="8192")
    parser.add_argument("--router-num-ctx", default="2048")
    parser.add_argument("--ollama-temperature", default="0.05")
    parser.add_argument("--ollama-keep-alive", default="24h")
    parser.add_argument("--ollama-models", default="/var/lib/ollama-models")
    parser.add_argument("--ollama-max-loaded-models", default="1")
    parser.add_argument("--ollama-num-parallel", default="1")
    parser.add_argument("--ollama-max-queue", default="64")
    parser.add_argument("--llm-timeout", default="45")
    parser.add_argument("--router-timeout", default="45")
    parser.add_argument("--repo-url", default=None)
    parser.add_argument("--repo-branch", default="main")
    parser.add_argument("--auto-update-interval-minutes", default="15")
    parser.add_argument("--static-ip", default="192.168.30.33")
    parser.add_argument("--static-prefix-length", default="24")
    parser.add_argument("--static-netmask", default="255.255.255.0")
    parser.add_argument("--static-gateway", default="192.168.30.1")
    parser.add_argument("--static-dns", default="8.8.8.8")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_iso = args.base_iso.expanduser().resolve()
    output_iso = args.output_iso.expanduser().resolve()
    if not base_iso.exists():
        raise FileNotFoundError(f"Base ISO not found: {base_iso}")
    if shutil.which("xorriso") is None:
        raise RuntimeError("xorriso is required to build the appliance ISO.")

    repo_url = args.repo_url or git_output(["remote", "get-url", "origin"])
    if not repo_url:
        raise RuntimeError("A git origin URL is required for autonomous updates.")

    workspace_root = args.workspace_root or f"/home/{args.install_username}/workspace"
    replacements = {
        "HOSTNAME": args.hostname,
        "INSTALL_USERNAME": args.install_username,
        "FULL_NAME": args.full_name,
        "TIMEZONE": args.timezone,
        "LOCALE": args.locale,
        "KEYMAP": args.keymap,
        "MODEL_NAME": args.model_name,
        "ROUTER_MODEL_NAME": args.router_model_name,
        "ACCESS_MODE": args.access_mode,
        "WORKSPACE_ROOT": workspace_root,
        "STATE_ROOT_OVERRIDE": args.state_root_override,
        "OLLAMA_CONTEXT_LENGTH": args.ollama_context_length,
        "OLLAMA_NUM_CTX": args.ollama_num_ctx,
        "ROUTER_NUM_CTX": args.router_num_ctx,
        "OLLAMA_TEMPERATURE": args.ollama_temperature,
        "OLLAMA_KEEP_ALIVE": args.ollama_keep_alive,
        "OLLAMA_MODELS": args.ollama_models,
        "OLLAMA_MAX_LOADED_MODELS": args.ollama_max_loaded_models,
        "OLLAMA_NUM_PARALLEL": args.ollama_num_parallel,
        "OLLAMA_MAX_QUEUE": args.ollama_max_queue,
        "LLM_TIMEOUT": args.llm_timeout,
        "ROUTER_TIMEOUT": args.router_timeout,
        "REPO_URL": repo_url,
        "REPO_BRANCH": args.repo_branch,
        "AUTO_UPDATE_INTERVAL_MINUTES": args.auto_update_interval_minutes,
        "STATIC_IP": args.static_ip,
        "STATIC_PREFIX_LENGTH": args.static_prefix_length,
        "STATIC_NETMASK": args.static_netmask,
        "STATIC_GATEWAY": args.static_gateway,
        "STATIC_DNS": args.static_dns,
    }

    with tempfile.TemporaryDirectory(prefix="marc-a1-appliance-") as temp_dir:
        temp_path = Path(temp_dir)
        iso_root = temp_path / "iso-root"
        appliance_payload = iso_root / "appliance"
        extract_iso(base_iso, iso_root)
        appliance_payload.mkdir(parents=True, exist_ok=True)

        create_bundle(appliance_payload / "marc-a1-bundle.tar.gz")
        (iso_root / "preseed.cfg").write_text(
            render_template(TEMPLATES_DIR / "preseed.cfg.in", replacements),
            encoding="utf-8",
        )
        (appliance_payload / "appliance.env").write_text(
            render_template(TEMPLATES_DIR / "appliance.env.in", replacements),
            encoding="utf-8",
        )
        (appliance_payload / "marc-a1-firstboot.sh").write_text(
            render_template(TEMPLATES_DIR / "marc-a1-firstboot.sh.in", replacements),
            encoding="utf-8",
        )
        os.chmod(appliance_payload / "marc-a1-firstboot.sh", 0o755)
        (appliance_payload / "marc-a1-firstboot.service").write_text(
            (TEMPLATES_DIR / "marc-a1-firstboot.service").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (appliance_payload / "marc-a1.service").write_text(
            render_template(TEMPLATES_DIR / "marc-a1.service.in", replacements),
            encoding="utf-8",
        )
        (appliance_payload / "ollama-override.conf").write_text(
            render_template(TEMPLATES_DIR / "ollama-override.conf.in", replacements),
            encoding="utf-8",
        )
        (appliance_payload / "99-marc-a1.conf").write_text(
            (TEMPLATES_DIR / "99-marc-a1.conf").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (appliance_payload / "marc-a1-performance-tune").write_text(
            render_template(TEMPLATES_DIR / "marc-a1-performance-tune.in", replacements),
            encoding="utf-8",
        )
        os.chmod(appliance_payload / "marc-a1-performance-tune", 0o755)
        (appliance_payload / "marc-a1-performance-tune.service").write_text(
            (TEMPLATES_DIR / "marc-a1-performance-tune.service").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (appliance_payload / "marc-a1-network-fallback").write_text(
            render_template(TEMPLATES_DIR / "marc-a1-network-fallback.in", replacements),
            encoding="utf-8",
        )
        os.chmod(appliance_payload / "marc-a1-network-fallback", 0o755)
        (appliance_payload / "marc-a1-network-fallback.service").write_text(
            (TEMPLATES_DIR / "marc-a1-network-fallback.service").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (appliance_payload / "updater.env").write_text(
            render_template(TEMPLATES_DIR / "updater.env.in", replacements),
            encoding="utf-8",
        )
        (appliance_payload / "marc-a1-updater").write_text(
            render_template(TEMPLATES_DIR / "marc-a1-updater.in", replacements),
            encoding="utf-8",
        )
        os.chmod(appliance_payload / "marc-a1-updater", 0o755)
        (appliance_payload / "marc-a1-updater.service").write_text(
            (TEMPLATES_DIR / "marc-a1-updater.service").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (appliance_payload / "marc-a1-updater.timer").write_text(
            render_template(TEMPLATES_DIR / "marc-a1-updater.timer.in", replacements),
            encoding="utf-8",
        )

        for relative in BOOT_CONFIGS:
            patch_boot_config(iso_root / relative)
        patch_boot_config(iso_root / "isolinux/isolinux.cfg")

        update_md5sums(iso_root)
        build_iso(iso_root, base_iso, output_iso)

    print(output_iso)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
