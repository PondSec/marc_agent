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
    ".marc_a1",
    ".DS_Store",
}


class WorkspaceError(RuntimeError):
    pass


class WorkspaceManager:
    def __init__(self, root: str | Path, *, allow_outside_root: bool = False):
        self.root = Path(root).expanduser().resolve()
        self.allow_outside_root = allow_outside_root
        self.root.mkdir(parents=True, exist_ok=True)

    def _workspace_display_path(self, path: str | Path) -> str | None:
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = self.root / target
        try:
            return target.absolute().relative_to(self.root).as_posix()
        except ValueError:
            return None

    def relative_path(self, path: str | Path) -> str:
        resolved = self.resolve_path(path)
        return self.display_path(resolved)

    def display_path(self, path: str | Path) -> str:
        resolved = Path(path).expanduser().resolve(strict=False)
        if self.is_within_root(resolved):
            return resolved.relative_to(self.root).as_posix()
        return str(resolved)

    def resolve_path(self, path: str | Path) -> Path:
        target = Path(path)
        if not target.is_absolute():
            target = (self.root / target).resolve(strict=False)
        else:
            target = target.expanduser().resolve(strict=False)
        if not self.allow_outside_root and not self.is_within_root(target):
            raise WorkspaceError(f"Path escapes workspace root: {path}")
        return target

    def resolve_directory(self, path: str | Path) -> Path:
        resolved = self.resolve_path(path)
        if resolved.exists() and not resolved.is_dir():
            raise WorkspaceError(f"Expected directory, got file: {path}")
        return resolved

    def is_within_root(self, path: str | Path) -> bool:
        candidate = Path(path).expanduser().resolve(strict=False)
        try:
            candidate.relative_to(self.root)
            return True
        except ValueError:
            return False

    def should_ignore(self, display_path: str) -> bool:
        parts = Path(display_path).parts
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
        results: list[tuple[str, Path]] = []
        iterator = start_dir.rglob("*") if recursive else start_dir.glob("*")
        for candidate in iterator:
            if len(results) >= max_results:
                break
            if not candidate.is_file():
                continue
            display = self._workspace_display_path(candidate) or self.display_path(candidate)
            if self.should_ignore(display):
                continue
            if glob_pattern and not fnmatch.fnmatch(display, glob_pattern):
                continue
            resolved = candidate.resolve(strict=False)
            if not self.allow_outside_root and candidate.is_symlink() and not self.is_within_root(resolved):
                continue
            results.append((display, candidate.absolute()))
        return [path for _, path in sorted(results, key=lambda item: item[0])]

    def read_text(self, path: str | Path) -> str:
        target = self.resolve_path(path)
        return target.read_text(encoding="utf-8")
