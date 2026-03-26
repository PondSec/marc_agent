from __future__ import annotations

import argparse
import sys

from bootstrap_runtime import ensure_ollama_runtime, ensure_runtime_dependencies


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start the M.A.R.C A1 local web console."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file. Defaults to config/agent.json if present.",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for the local web server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the local web server.",
    )
    parser.add_argument(
        "--access-mode",
        choices=["safe", "approval", "full"],
        default=None,
        help="Runtime access mode: safe, approval, or full.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose agent logging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and validate actions without mutating files or executing commands.",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Legacy alias for --access-mode safe.",
    )
    parser.add_argument(
        "--approval-mode",
        action="store_true",
        help="Legacy alias for --access-mode approval.",
    )
    return parser


def make_config(args: argparse.Namespace):
    from config.settings import AccessMode, AppConfig

    access_mode = args.access_mode
    if access_mode is None and args.read_only:
        access_mode = AccessMode.SAFE.value
    if access_mode is None and args.approval_mode:
        access_mode = AccessMode.APPROVAL.value

    overrides = {
        "verbose": args.verbose if args.verbose else None,
        "dry_run": args.dry_run if args.dry_run else None,
        "access_mode": access_mode,
    }
    return AppConfig.from_sources(
        workspace_override=args.workspace,
        config_path=args.config,
        overrides=overrides,
    )


def main() -> int:
    ensure_runtime_dependencies()

    import uvicorn

    from server.app import create_app

    parser = build_parser()
    args = parser.parse_args()
    config = make_config(args)
    ensure_ollama_runtime(config.ollama_host)
    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
