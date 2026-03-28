from __future__ import annotations

from config.settings import AppConfig
from llm.schemas import (
    AppendFileArgs,
    CreateFileArgs,
    DeleteFileArgs,
    PatchFileArgs,
    ReadFileArgs,
    ReplaceInFileArgs,
    ShowDiffArgs,
    WriteFileArgs,
)
from runtime.workspace import WorkspaceManager
from tools.difftools import create_unified_diff
from tools.safety import SafetyManager


class FileSystemTools:
    def __init__(
        self,
        config: AppConfig,
        workspace: WorkspaceManager,
        safety: SafetyManager,
    ):
        self.config = config
        self.workspace = workspace
        self.safety = safety

    def read_file(self, args: ReadFileArgs) -> dict:
        target = self.safety.ensure_read_allowed(args.path)
        text = target.read_text(encoding="utf-8")
        lines = text.splitlines()
        start = args.start_line - 1 if args.start_line else 0
        end = args.end_line if args.end_line else len(lines)
        snippet = "\n".join(lines[start:end])
        snippet = snippet[: self.config.max_read_chars]
        return {
            "success": True,
            "message": f"Read {self.workspace.relative_path(target)}.",
            "path": self.workspace.relative_path(target),
            "content": snippet,
            "total_lines": len(lines),
        }

    def write_file(self, args: WriteFileArgs) -> dict:
        target = self.safety.ensure_write_allowed(args.path)
        old = target.read_text(encoding="utf-8") if target.exists() else ""
        if self.config.dry_run:
            new = args.content
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(args.content, encoding="utf-8")
            new = target.read_text(encoding="utf-8")
        rel = self.workspace.relative_path(target)
        diff = create_unified_diff(old, new, rel)
        return {
            "success": True,
            "message": f"Wrote {rel}.",
            "changed_files": [{"path": rel, "operation": "write", "diff": diff}],
            "diff": diff,
        }

    def append_file(self, args: AppendFileArgs) -> dict:
        target = self.safety.ensure_write_allowed(args.path)
        old = target.read_text(encoding="utf-8") if target.exists() else ""
        new = old + args.content
        if not self.config.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(new, encoding="utf-8")
        rel = self.workspace.relative_path(target)
        diff = create_unified_diff(old, new, rel)
        return {
            "success": True,
            "message": f"Appended to {rel}.",
            "changed_files": [{"path": rel, "operation": "append", "diff": diff}],
            "diff": diff,
        }

    def create_file(self, args: CreateFileArgs) -> dict:
        target = self.safety.ensure_write_allowed(args.path)
        if target.exists() and not args.overwrite:
            return {
                "success": False,
                "message": f"File already exists: {self.workspace.relative_path(target)}",
            }
        old = target.read_text(encoding="utf-8") if target.exists() else ""
        if not self.config.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(args.content, encoding="utf-8")
        rel = self.workspace.relative_path(target)
        diff = create_unified_diff(old, args.content, rel)
        return {
            "success": True,
            "message": f"Created {rel}.",
            "changed_files": [{"path": rel, "operation": "create", "diff": diff}],
            "diff": diff,
        }

    def delete_file(self, args: DeleteFileArgs) -> dict:
        target = self.safety.ensure_write_allowed(args.path)
        if not target.exists():
            return {"success": False, "message": f"File not found: {args.path}"}
        if target.is_dir():
            return {"success": False, "message": "Directory deletion is intentionally blocked."}
        old = target.read_text(encoding="utf-8")
        if not self.config.dry_run:
            target.unlink()
        rel = self.workspace.relative_path(target)
        diff = create_unified_diff(old, "", rel)
        return {
            "success": True,
            "message": f"Deleted {rel}.",
            "changed_files": [{"path": rel, "operation": "delete", "diff": diff}],
            "diff": diff,
        }

    def replace_in_file(self, args: ReplaceInFileArgs) -> dict:
        target = self.safety.ensure_write_allowed(args.path)
        old = target.read_text(encoding="utf-8")
        occurrences = old.count(args.find)
        if occurrences == 0:
            return {"success": False, "message": "Search text was not found in file."}
        new = old.replace(args.find, args.replace, args.count) if args.count else old.replace(args.find, args.replace)
        if not self.config.dry_run:
            target.write_text(new, encoding="utf-8")
        rel = self.workspace.relative_path(target)
        diff = create_unified_diff(old, new, rel)
        replaced = min(occurrences, args.count) if args.count else occurrences
        return {
            "success": True,
            "message": f"Replaced {replaced} occurrence(s) in {rel}.",
            "changed_files": [{"path": rel, "operation": "replace", "diff": diff}],
            "diff": diff,
        }

    def patch_file(self, args: PatchFileArgs) -> dict:
        target = self.safety.ensure_write_allowed(args.path)
        old = target.read_text(encoding="utf-8")
        new = old
        for patch in args.patches:
            occurrences = new.count(patch.old)
            if occurrences != patch.expected_count:
                return {
                    "success": False,
                    "message": (
                        f"Patch mismatch for {args.path}: expected {patch.expected_count} "
                        f"occurrence(s), found {occurrences}."
                    ),
                }
            new = new.replace(patch.old, patch.new)
        if not self.config.dry_run:
            target.write_text(new, encoding="utf-8")
        rel = self.workspace.relative_path(target)
        diff = create_unified_diff(old, new, rel)
        return {
            "success": True,
            "message": f"Patched {rel}.",
            "changed_files": [{"path": rel, "operation": "patch", "diff": diff}],
            "diff": diff,
        }

    def show_diff(self, args: ShowDiffArgs) -> dict:
        target = self.safety.ensure_read_allowed(args.path)
        old = target.read_text(encoding="utf-8") if target.exists() else ""
        rel = self.workspace.relative_path(target)
        diff = create_unified_diff(old, args.new_content, rel)
        return {
            "success": True,
            "message": f"Generated diff preview for {rel}.",
            "diff": diff,
        }
