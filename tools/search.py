from __future__ import annotations

import fnmatch
import re

from agent.memory import RepoMemoryStore
from config.settings import AppConfig
from llm.schemas import InspectWorkspaceArgs, ListFilesArgs, SearchInFilesArgs
from runtime.workspace import WorkspaceManager


class SearchTools:
    def __init__(
        self,
        config: AppConfig,
        workspace: WorkspaceManager,
        memory: RepoMemoryStore,
    ):
        self.config = config
        self.workspace = workspace
        self.memory = memory

    def inspect_workspace(self, args: InspectWorkspaceArgs) -> dict:
        snapshot = self.memory.build_snapshot(args.focus)
        return {
            "success": True,
            "message": "Workspace inspected successfully.",
            "snapshot": snapshot.model_dump(),
        }

    def list_files(self, args: ListFilesArgs) -> dict:
        files = self.workspace.iter_files(
            args.path,
            recursive=args.recursive,
            glob_pattern=args.glob,
            max_results=args.max_results,
        )
        return {
            "success": True,
            "message": f"Listed {len(files)} file(s).",
            "files": [self.workspace.display_path(path) for path in files],
        }

    def search_in_files(self, args: SearchInFilesArgs) -> dict:
        files = self.workspace.iter_files(
            args.path,
            recursive=True,
            glob_pattern=args.glob,
            max_results=10_000,
        )
        flags = 0 if args.case_sensitive else re.IGNORECASE
        pattern = re.compile(args.query if args.regex else re.escape(args.query), flags)
        matches: list[dict] = []
        for file_path in files:
            if len(matches) >= min(args.max_results, self.config.max_search_results):
                break
            rel = self.workspace.display_path(file_path)
            if args.glob and not fnmatch.fnmatch(rel, args.glob):
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for idx, line in enumerate(content.splitlines(), start=1):
                if pattern.search(line):
                    matches.append({"path": rel, "line_number": idx, "line": line.strip()})
                    if len(matches) >= min(args.max_results, self.config.max_search_results):
                        break
        return {
            "success": True,
            "message": f"Found {len(matches)} match(es).",
            "matches": matches,
        }
