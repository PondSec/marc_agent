from __future__ import annotations

from collections import Counter
from pathlib import Path

from agent.models import WorkspaceSnapshot
from config.settings import AppConfig
from runtime.workspace import WorkspaceManager


LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".json": "json",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".sh": "shell",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
}

FOCUS_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "eine",
    "einen",
    "oder",
    "und",
    "fuer",
    "mit",
    "der",
    "die",
    "das",
    "den",
    "dem",
    "des",
    "zum",
    "zur",
    "task",
    "projekt",
}


class RepoMemoryStore:
    def __init__(self, config: AppConfig, workspace: WorkspaceManager):
        self.config = config
        self.workspace = workspace
        self.snapshot_path = config.memory_dir_path / "repo_snapshot.json"

    def build_snapshot(self, focus: str | None = None) -> WorkspaceSnapshot:
        files = self.workspace.iter_files(max_results=20_000)
        language_counts: Counter[str] = Counter()
        directory_counts: Counter[str] = Counter()
        scored_files: list[tuple[int, str]] = []

        for path in files:
            rel = path.relative_to(self.workspace.root).as_posix()
            language = LANGUAGE_MAP.get(path.suffix.lower(), "other")
            language_counts[language] += 1
            top_dir = rel.split("/", 1)[0] if "/" in rel else "."
            directory_counts[top_dir] += 1
            scored_files.append((self._score_file(rel, focus), rel))

        important_files = [
            rel for score, rel in sorted(scored_files, reverse=True) if score > 0
        ][: self.config.max_files_in_context]
        file_briefs = {
            rel: self._brief_for_file(rel)
            for rel in important_files[: min(12, len(important_files))]
        }
        likely_commands = self._detect_commands([rel for _, rel in scored_files])
        summary = self._build_summary(
            file_count=len(files),
            language_counts=language_counts,
            important_files=important_files,
            likely_commands=likely_commands,
            focus=focus,
        )

        snapshot = WorkspaceSnapshot(
            root=str(self.workspace.root),
            file_count=len(files),
            language_counts=dict(language_counts),
            top_directories=[
                name for name, _ in directory_counts.most_common(8) if name and name != "."
            ],
            important_files=important_files,
            file_briefs=file_briefs,
            likely_commands=likely_commands,
            repo_summary=summary,
        )
        self.snapshot_path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
        return snapshot

    def render_snapshot(self, snapshot: WorkspaceSnapshot) -> str:
        lines = [
            f"Workspace: {snapshot.root}",
            f"Files: {snapshot.file_count}",
            (
                "Languages: "
                + (
                    ", ".join(
                        f"{key}={value}" for key, value in snapshot.language_counts.items()
                    )
                    or "none detected"
                )
            ),
            f"Top directories: {', '.join(snapshot.top_directories) or '(flat workspace)'}",
            "",
            "Important files:",
        ]
        for path in snapshot.important_files[:12]:
            brief = snapshot.file_briefs.get(path, "")
            lines.append(f"- {path}: {brief}")
        if snapshot.likely_commands:
            lines.extend(["", "Likely validation commands:"])
            for command in snapshot.likely_commands:
                lines.append(f"- {command}")
        lines.extend(["", "Summary:", snapshot.repo_summary])
        return "\n".join(lines)

    def _brief_for_file(self, relative_path: str) -> str:
        try:
            text = self.workspace.read_text(relative_path)
        except Exception:
            return "Binary or unreadable file."
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return "Empty file."
        excerpt = " ".join(lines[:3])[:220]
        return excerpt

    def _score_file(self, relative_path: str, focus: str | None) -> int:
        name = Path(relative_path).name.lower()
        rel = relative_path.lower()
        score = 0
        if name.startswith("readme"):
            score += 120
        if name in {
            "pyproject.toml",
            "package.json",
            "requirements.txt",
            "cargo.toml",
            "go.mod",
            "makefile",
            "dockerfile",
            ".env.example",
        }:
            score += 110
        if any(
            token in rel
            for token in (
                "config",
                "settings",
                "routes",
                "controllers",
                "services",
                "middleware",
                "models",
                "schema",
                "server/",
                "agent/",
                "runtime/",
                "tools/",
                "src/",
            )
        ):
            score += 60
        if "/tests/" in f"/{rel}" or name.startswith("test_") or name.endswith("_test.py"):
            score += 80
        if rel.endswith((".md", ".rst")):
            score += 30
        for term in self._focus_terms(focus):
            if term in rel:
                score += 75
        return score

    def _detect_commands(self, files: list[str]) -> list[str]:
        commands: list[str] = []
        lower = {path.lower() for path in files}
        if {"pyproject.toml", "requirements.txt", "pytest.ini", "tox.ini"} & lower or any(
            path.endswith(".py") for path in lower
        ):
            commands.extend(["pytest -q", "python -m pytest"])
        if "ruff.toml" in lower or ".ruff.toml" in lower:
            commands.append("ruff check .")
        if "mypy.ini" in lower or "pyrightconfig.json" in lower:
            commands.append("mypy .")
        if "package.json" in lower:
            commands.extend(["npm test", "npm run lint", "npm run build"])
        if "pnpm-lock.yaml" in lower:
            commands.extend(["pnpm test", "pnpm lint", "pnpm build"])
        if "yarn.lock" in lower:
            commands.extend(["yarn test", "yarn lint", "yarn build"])
        if "cargo.toml" in lower:
            commands.extend(["cargo test", "cargo check"])
        if "go.mod" in lower:
            commands.append("go test ./...")
        return list(dict.fromkeys(commands))

    def _build_summary(
        self,
        *,
        file_count: int,
        language_counts: Counter[str],
        important_files: list[str],
        likely_commands: list[str],
        focus: str | None,
    ) -> str:
        dominant = ", ".join(
            f"{lang} ({count})" for lang, count in language_counts.most_common(4)
        ) or "no dominant language detected"
        summary = (
            f"The repository contains {file_count} tracked files in the current scan. "
            f"Dominant file types: {dominant}. "
            f"Prioritize {', '.join(important_files[:6]) or 'manifest and README files'} before broad reads. "
            f"Suggested validation commands: {', '.join(likely_commands) or 'no obvious test command detected'}."
        )
        if focus:
            summary += f" Current focus hint: {focus}."
        return summary

    def _focus_terms(self, focus: str | None) -> list[str]:
        if not focus:
            return []
        tokens = []
        for raw in focus.lower().replace("/", " ").replace("-", " ").split():
            token = raw.strip(".,:;()[]{}!?\"'")
            if len(token) < 3 or token in FOCUS_STOPWORDS:
                continue
            if token not in tokens:
                tokens.append(token)
        return tokens[:6]
