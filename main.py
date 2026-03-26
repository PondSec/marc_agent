from __future__ import annotations

import argparse
import sys

import uvicorn

from config.settings import AppConfig
from server.app import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start the local Codex-style web GUI."
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
        "--verbose",
        action="store_true",
        help="Enable verbose agent logging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Start the server with dry-run mode enabled by default.",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Start the server with read-only mode enabled by default.",
    )
    parser.add_argument(
        "--approval-mode",
        action="store_true",
        help="Start the server with approval mode enabled by default.",
    )
    return parser


def make_config(args: argparse.Namespace) -> AppConfig:
    overrides = {
        "verbose": args.verbose,
        "dry_run": args.dry_run,
        "read_only": args.read_only,
        "approval_mode": args.approval_mode,
    }
    return AppConfig.from_sources(
        workspace_override=args.workspace,
        config_path=args.config,
        overrides=overrides,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = make_config(args)
    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
