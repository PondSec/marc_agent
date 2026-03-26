from __future__ import annotations

import fnmatch
from pathlib import Path


DEFAULT_IGNORES = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".venv",
    "venv",
    ".local_codex_agent",
    ".DS_Store",
}


class WorkspaceError(RuntimeError):
    pass


class WorkspaceManager:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def relative_path(self, path: str | Path) -> str:
        resolved = self.resolve_path(path)
        return resolved.relative_to(self.root).as_posix()

    def resolve_path(self, path: str | Path) -> Path:
        target = Path(path)
        if not target.is_absolute():
            target = (self.root / target).resolve()
        else:
            target = target.expanduser().resolve()
        if not self.is_within_root(target):
            raise WorkspaceError(f"Path escapes workspace root: {path}")
        return target

    def resolve_directory(self, path: str | Path) -> Path:
        resolved = self.resolve_path(path)
        if resolved.exists() and not resolved.is_dir():
            raise WorkspaceError(f"Expected directory, got file: {path}")
        return resolved

    def is_within_root(self, path: str | Path) -> bool:
        candidate = Path(path).expanduser().resolve()
        try:
            candidate.relative_to(self.root)
            return True
        except ValueError:
            return False

    def should_ignore(self, relative_path: str) -> bool:
        parts = Path(relative_path).parts
        if any(part in DEFAULT_IGNORES for part in parts):
            return True
        return False

    def iter_files(
        self,
        start: str | Path = ".",
        *,
        recursive: bool = True,
        glob_pattern: str | None = None,
        max_results: int = 5_000,
    ) -> list[Path]:
        start_dir = self.resolve_directory(start)
        results: list[Path] = []
        iterator = start_dir.rglob("*") if recursive else start_dir.glob("*")
        for candidate in iterator:
            if len(results) >= max_results:
                break
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(self.root).as_posix()
            if self.should_ignore(relative):
                continue
            if glob_pattern and not fnmatch.fnmatch(relative, glob_pattern):
                continue
            results.append(candidate)
        return sorted(results, key=lambda path: path.relative_to(self.root).as_posix())

    def read_text(self, path: str | Path) -> str:
        target = self.resolve_path(path)
        return target.read_text(encoding="utf-8")
