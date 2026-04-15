from __future__ import annotations

import argparse
import json
import sys

from bootstrap_runtime import ensure_ollama_runtime, ensure_runtime_dependencies


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optional CLI for the MARC A2 coding runtime."
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
        "--access-mode",
        choices=["safe", "approval", "full"],
        default=None,
        help="Runtime access mode: safe, approval, or full.",
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
        help="Legacy alias for --access-mode safe.",
    )
    parser.add_argument(
        "--approval-mode",
        action="store_true",
        help="Legacy alias for --access-mode approval.",
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


def print_session_result(session) -> None:
    print(f"Session: {session.id}")
    print(f"Task: {session.task}")
    print(f"Status: {session.status}")
    print(f"Phase: {session.current_phase}")
    print(f"Workflow stage: {session.workflow_stage}")
    print(f"Access mode: {session.access_mode}")
    print(f"Validation: {session.validation_status}")
    print(f"Iterations: {session.iterations}")
    if session.stop_reason:
        print(f"Stop reason: {session.stop_reason}")
    if session.plan_summary:
        print(f"\nPlan summary:\n{session.plan_summary}")
    if session.plan:
        print("\nPlan:")
        for item in session.plan:
            print(f"- [{item.status}] {item.step}")
    if session.candidate_files:
        print("\nCandidate files:")
        for path in session.candidate_files[:10]:
            print(f"- {path}")
    if session.verification_commands:
        print("\nVerification commands:")
        for command in session.verification_commands:
            print(f"- {command}")
    if session.validation_runs:
        print("\nValidation runs:")
        for run in session.validation_runs[-8:]:
            print(f"- [{run.status}] {run.kind or 'check'} :: {run.command}")
    if session.helper_artifacts:
        print("\nHelper artifacts:")
        for path in session.helper_artifacts:
            print(f"- {path}")
    if session.changed_files:
        print("\nChanged files:")
        for change in session.changed_files:
            print(f"- {change.path} ({change.operation})")
    if session.executed_commands:
        print("\nExecuted commands:")
        for command in session.executed_commands:
            print(f"- {command}")
    if session.blockers:
        print("\nBlockers:")
        for blocker in session.blockers:
            print(f"- {blocker}")
    if session.diagnostics:
        print("\nDiagnostics:")
        for diagnostic in session.diagnostics[-8:]:
            print(f"- [{diagnostic.category}] {diagnostic.summary}")
    if session.report and session.report.report_path:
        print(f"\nReport path:\n{session.report.report_path}")
    if session.final_response:
        print("\nResult:")
        print(session.final_response)


def run_task_command(config, prompt: str) -> int:
    from agent.core import AgentCore

    agent = AgentCore(config)
    session = agent.run_task(prompt)
    print_session_result(session)
    return 0 if session.status in {"completed", "partial"} else 1


def run_inspect_command(config, focus: str | None) -> int:
    from agent.core import AgentCore

    agent = AgentCore(config)
    snapshot_text = agent.inspect_workspace(focus=focus)
    print(snapshot_text)
    return 0


def run_diff_command(config, session_id: str | None) -> int:
    from agent.session import SessionStore

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


def run_chat_command(config, session_id: str | None) -> int:
    from agent.core import AgentCore
    from agent.session import SessionStore

    agent = AgentCore(config)
    store = SessionStore(config.session_dir_path)
    session = store.load(session_id) if session_id else None
    print("Interactive chat. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            prompt = input("\nmarc> ").strip()
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


def run_config_show(config) -> int:
    print(json.dumps(config.to_public_dict(), indent=2, sort_keys=True))
    return 0


def main() -> int:
    ensure_runtime_dependencies()

    parser = build_parser()
    args = parser.parse_args()
    config = make_config(args)

    command = args.command
    if command == "task":
        ensure_ollama_runtime(config.ollama_host)
        return run_task_command(config, args.prompt)
    if command == "inspect":
        return run_inspect_command(config, args.focus)
    if command == "chat":
        ensure_ollama_runtime(config.ollama_host)
        return run_chat_command(config, args.session_id)
    if command == "diff":
        return run_diff_command(config, args.session_id)
    if command == "config" and args.config_command == "show":
        return run_config_show(config)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
