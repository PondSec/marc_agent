from __future__ import annotations

import argparse
import json
import sys

from agent.core import AgentCore
from agent.session import SessionStore
from config.settings import AppConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optional CLI for the local Codex-style coding agent."
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
        "--verbose",
        action="store_true",
        help="Enable verbose console logging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and validate actions without mutating files or executing commands.",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Disable all mutating tools.",
    )
    parser.add_argument(
        "--approval-mode",
        action="store_true",
        help="Require approval for medium/high-risk shell commands and git mutations.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    task_parser = subparsers.add_parser("task", help="Run a task-oriented agent loop.")
    task_parser.add_argument("prompt", type=str, help="Coding task to execute.")

    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat loop.")
    chat_parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Resume an existing session by ID.",
    )

    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect and summarize the workspace."
    )
    inspect_parser.add_argument(
        "--focus",
        type=str,
        default=None,
        help="Optional focus area, such as auth, tests, or billing.",
    )

    diff_parser = subparsers.add_parser(
        "diff", help="Show tracked diffs from the last or selected session."
    )
    diff_parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Show diffs for a specific session.",
    )

    config_parser = subparsers.add_parser("config", help="Inspect runtime configuration.")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)
    config_subparsers.add_parser("show", help="Print the resolved configuration.")

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


def print_session_result(session) -> None:
    print(f"Session: {session.id}")
    print(f"Task: {session.task}")
    print(f"Status: {session.status}")
    print(f"Iterations: {session.iterations}")
    if session.plan:
        print("\nPlan:")
        for item in session.plan:
            print(f"- [{item.status}] {item.step}")
    if session.changed_files:
        print("\nChanged files:")
        for change in session.changed_files:
            print(f"- {change.path} ({change.operation})")
    if session.executed_commands:
        print("\nExecuted commands:")
        for command in session.executed_commands:
            print(f"- {command}")
    if session.final_response:
        print("\nResult:")
        print(session.final_response)


def run_task_command(config: AppConfig, prompt: str) -> int:
    agent = AgentCore(config)
    session = agent.run_task(prompt)
    print_session_result(session)
    return 0 if session.status in {"completed", "partial"} else 1


def run_inspect_command(config: AppConfig, focus: str | None) -> int:
    agent = AgentCore(config)
    snapshot_text = agent.inspect_workspace(focus=focus)
    print(snapshot_text)
    return 0


def run_diff_command(config: AppConfig, session_id: str | None) -> int:
    store = SessionStore(config.session_dir_path)
    session = store.load(session_id) if session_id else store.load_last()
    if session is None:
        print("No session data found.")
        return 1
    if not session.changed_files:
        print(f"No tracked file diffs found for session {session.id}.")
        return 0
    print(f"Session: {session.id}\n")
    for change in session.changed_files:
        print(f"## {change.path} ({change.operation})")
        print(change.diff or "(no diff available)")
        print()
    return 0


def run_chat_command(config: AppConfig, session_id: str | None) -> int:
    agent = AgentCore(config)
    store = SessionStore(config.session_dir_path)
    session = store.load(session_id) if session_id else None
    print("Interactive chat. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            prompt = input("\nagent> ").strip()
        except EOFError:
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break
        session = agent.run_task(prompt, session=session)
        print_session_result(session)
    return 0


def run_config_show(config: AppConfig) -> int:
    print(json.dumps(config.to_public_dict(), indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = make_config(args)

    command = args.command
    if command == "task":
        return run_task_command(config, args.prompt)
    if command == "inspect":
        return run_inspect_command(config, args.focus)
    if command == "chat":
        return run_chat_command(config, args.session_id)
    if command == "diff":
        return run_diff_command(config, args.session_id)
    if command == "config" and args.config_command == "show":
        return run_config_show(config)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
