from __future__ import annotations

import json
import re
import tomllib
from collections import Counter
from pathlib import Path

from agent.models import FileInsight, ValidationCommand, WorkspaceSnapshot
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
    "agent",
    "coding",
}

MANIFEST_FILES = {
    "readme.md",
    "pyproject.toml",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-runtime.txt",
    "cargo.toml",
    "go.mod",
    "makefile",
    "dockerfile",
    "tox.ini",
    "pytest.ini",
    ".env.example",
}

CONFIG_HINTS = (
    "config",
    "settings",
    "schema",
    "middleware",
    "routes",
    "router",
    "service",
    "model",
    "server/",
    "agent/",
    "runtime/",
    "tools/",
)

ENTRYPOINT_NAMES = {
    "main.py",
    "cli.py",
    "app.py",
    "server.py",
    "manage.py",
    "index.ts",
    "index.js",
}

SCRIPT_KIND_MAP = {
    "test": ("test", 10),
    "lint": ("lint", 20),
    "typecheck": ("typecheck", 30),
    "check": ("check", 35),
    "build": ("build", 40),
    "verify": ("check", 45),
    "dev": ("dev", 70),
    "start": ("dev", 75),
    "release": ("release", 90),
    "deploy": ("deploy", 100),
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
        insights: list[FileInsight] = []
        manifests: list[str] = []
        configs: list[str] = []
        test_files: list[str] = []
        build_files: list[str] = []
        deploy_files: list[str] = []
        entrypoints: list[str] = []
        focus_terms = self._focus_terms(focus)

        for path in files:
            rel = path.relative_to(self.workspace.root).as_posix()
            language = LANGUAGE_MAP.get(path.suffix.lower(), "other")
            language_counts[language] += 1
            top_dir = rel.split("/", 1)[0] if "/" in rel else "."
            directory_counts[top_dir] += 1

            insight = self._insight_for_file(rel, focus_terms)
            insights.append(insight)

            lowered = rel.lower()
            name = Path(rel).name.lower()
            if name in MANIFEST_FILES:
                manifests.append(rel)
            if any(token in lowered for token in CONFIG_HINTS):
                configs.append(rel)
            if self._is_test_file(rel):
                test_files.append(rel)
            if self._is_build_file(rel):
                build_files.append(rel)
            if self._is_deploy_file(rel):
                deploy_files.append(rel)
            if name in ENTRYPOINT_NAMES:
                entrypoints.append(rel)

        insights.sort(key=lambda item: (-item.score, item.path))
        important_files = [item.path for item in insights[: self.config.max_files_in_context]]
        focus_files = [item.path for item in insights if any(term in item.path.lower() for term in focus_terms)]
        file_briefs = {
            item.path: self._brief_for_file(item.path)
            for item in insights[: min(16, len(insights))]
        }
        for item in insights[: min(16, len(insights))]:
            item.summary = file_briefs.get(item.path)

        validation_commands, workflow_commands = self._detect_commands(
            manifests=manifests,
            build_files=build_files,
            deploy_files=deploy_files,
        )
        likely_commands = [item.command for item in validation_commands]
        repo_map = self._build_repo_map(directory_counts, manifests, entrypoints)
        project_labels = self._project_labels(
            language_counts=language_counts,
            manifests=manifests,
            build_files=build_files,
            deploy_files=deploy_files,
        )
        summary = self._build_summary(
            file_count=len(files),
            language_counts=language_counts,
            important_files=important_files,
            likely_commands=likely_commands,
            validation_commands=validation_commands,
            project_labels=project_labels,
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
            focus_files=focus_files[:15],
            file_briefs=file_briefs,
            file_insights=insights[:24],
            manifests=manifests[:20],
            configs=configs[:24],
            test_files=test_files[:30],
            build_files=build_files[:20],
            deploy_files=deploy_files[:20],
            entrypoints=entrypoints[:20],
            repo_map=repo_map,
            project_labels=project_labels,
            likely_commands=likely_commands,
            validation_commands=validation_commands,
            workflow_commands=workflow_commands,
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
            f"Project labels: {', '.join(snapshot.project_labels) or 'none'}",
            "",
            "Important files:",
        ]
        for path in snapshot.important_files[:12]:
            brief = snapshot.file_briefs.get(path, "")
            lines.append(f"- {path}: {brief}")
        if snapshot.focus_files:
            lines.extend(["", "Focus-matched files:"])
            for path in snapshot.focus_files[:8]:
                lines.append(f"- {path}")
        if snapshot.validation_commands:
            lines.extend(["", "Validation plan:"])
            for command in snapshot.validation_commands[:6]:
                lines.append(
                    f"- [{command.kind}] {command.command} ({command.source})"
                )
        if snapshot.workflow_commands:
            lines.extend(["", "Workflow awareness:"])
            for command in snapshot.workflow_commands[:6]:
                lines.append(f"- [{command.kind}] {command.command} ({command.source})")
        if snapshot.repo_map:
            lines.extend(["", "Repo map:"])
            for line in snapshot.repo_map[:8]:
                lines.append(f"- {line}")
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

        symbol_patterns = (
            r"^(class|def)\s+[A-Za-z_][\w]*",
            r"^(export\s+)?(async\s+)?function\s+[A-Za-z_][\w]*",
            r"^(const|let|var)\s+[A-Za-z_][\w]*\s*=",
            r"^\[[^\]]+\]$",
        )
        symbols: list[str] = []
        for line in lines[:40]:
            if any(re.match(pattern, line) for pattern in symbol_patterns):
                symbols.append(line[:120])
            if len(symbols) >= 3:
                break
        if symbols:
            return " | ".join(symbols)[:220]
        return " ".join(lines[:3])[:220]

    def _insight_for_file(self, relative_path: str, focus_terms: list[str]) -> FileInsight:
        name = Path(relative_path).name.lower()
        rel = relative_path.lower()
        score = 0
        reasons: list[str] = []
        category = "source"

        if name.startswith("readme"):
            score += 140
            category = "manifest"
            reasons.append("README often describes architecture and usage.")
        if name in MANIFEST_FILES:
            score += 130
            category = "manifest"
            reasons.append("Manifest or root config file.")
        if any(token in rel for token in CONFIG_HINTS):
            score += 70
            category = "config"
            reasons.append("Likely architecture or dependency wiring file.")
        if self._is_test_file(relative_path):
            score += 85
            category = "test"
            reasons.append("Relevant test coverage or validation target.")
        if self._is_build_file(relative_path):
            score += 55
            category = "build"
            reasons.append("Build, packaging, or CI awareness file.")
        if self._is_deploy_file(relative_path):
            score += 45
            category = "deploy"
            reasons.append("Release or deployment workflow signal.")
        if Path(relative_path).name.lower() in ENTRYPOINT_NAMES:
            score += 65
            category = "entrypoint"
            reasons.append("Likely runtime entrypoint.")
        if rel.endswith((".md", ".rst")):
            score += 25
            reasons.append("Documentation or task context.")
        for term in focus_terms:
            if term in rel:
                score += 90
                reasons.append(f"Matches focus term '{term}'.")
        if "/src/" in f"/{rel}" or rel.startswith("src/"):
            score += 20
        if score == 0:
            score = 10
            reasons.append("General repository context.")
        return FileInsight(
            path=relative_path,
            category=category,
            score=score,
            reasons=reasons[:4],
        )

    def _detect_commands(
        self,
        *,
        manifests: list[str],
        build_files: list[str],
        deploy_files: list[str],
    ) -> tuple[list[ValidationCommand], list[ValidationCommand]]:
        validation: list[ValidationCommand] = []
        workflow: list[ValidationCommand] = []
        package_manager = self._detect_package_manager(manifests)

        for manifest in manifests:
            name = Path(manifest).name.lower()
            if name == "package.json":
                self._collect_package_json_commands(manifest, package_manager, validation, workflow)
            elif name == "pyproject.toml":
                self._collect_pyproject_commands(manifest, validation)
            elif name.startswith("requirements"):
                self._collect_requirements_commands(manifest, validation)
            elif name == "makefile":
                self._collect_makefile_commands(manifest, validation, workflow)
            elif name in {"tox.ini", "pytest.ini"}:
                self._add_command(
                    validation,
                    ValidationCommand(
                        command="python -m pytest",
                        kind="test",
                        source=name,
                        priority=10,
                        reason="Python test configuration detected.",
                    ),
                )

        if not validation:
            lower = {item.lower() for item in manifests}
            if {"requirements.txt", "requirements-runtime.txt", "pyproject.toml"} & lower:
                self._add_command(
                    validation,
                    ValidationCommand(
                        command="python -m pytest",
                        kind="test",
                        source="python-heuristic",
                        priority=10,
                        reason="Python repository detected.",
                    ),
                )

        build_signal = any(
            token in item.lower()
            for item in build_files
            for token in ("docker", ".github/workflows", "build", "vite.config", "webpack")
        )
        if build_signal and not any(item.kind == "build" for item in workflow):
            self._add_command(
                workflow,
                ValidationCommand(
                    command="review build automation before executing external build steps",
                    kind="build",
                    source="build-files",
                    priority=80,
                    reason="Docker or CI build files detected.",
                    required=False,
                ),
            )
        if deploy_files and not any(item.kind == "deploy" for item in workflow):
            self._add_command(
                workflow,
                ValidationCommand(
                    command="review release or deploy scripts before executing",
                    kind="deploy",
                    source="deploy-files",
                    priority=110,
                    reason="Release/deploy files detected.",
                    required=False,
                ),
            )

        validation.sort(key=lambda item: (item.priority, item.command))
        workflow.sort(key=lambda item: (item.priority, item.command))
        return validation[:8], workflow[:8]

    def _collect_package_json_commands(
        self,
        relative_path: str,
        package_manager: str,
        validation: list[ValidationCommand],
        workflow: list[ValidationCommand],
    ) -> None:
        target = self.workspace.resolve_path(relative_path)
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return

        scripts = payload.get("scripts", {})
        for script_name in sorted(scripts):
            lowered = script_name.lower()
            if lowered not in SCRIPT_KIND_MAP:
                continue
            kind, priority = SCRIPT_KIND_MAP[lowered]
            command = self._package_script_command(package_manager, script_name)
            spec = ValidationCommand(
                command=command,
                kind=kind,
                source="package.json",
                priority=priority,
                reason=f"package.json script '{script_name}' detected.",
                required=kind in {"test", "lint", "typecheck", "build", "check"},
            )
            target_list = validation if kind in {"test", "lint", "typecheck", "build", "check"} else workflow
            self._add_command(target_list, spec)

    def _collect_pyproject_commands(
        self,
        relative_path: str,
        validation: list[ValidationCommand],
    ) -> None:
        target = self.workspace.resolve_path(relative_path)
        try:
            payload = tomllib.loads(target.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError:
            return

        tool_section = payload.get("tool", {})
        project = payload.get("project", {})
        dependencies = [
            str(item).lower()
            for item in project.get("dependencies", [])
            if isinstance(item, str)
        ]

        if "pytest" in tool_section or any("pytest" in item for item in dependencies):
            self._add_command(
                validation,
                ValidationCommand(
                    command="python -m pytest",
                    kind="test",
                    source="pyproject.toml",
                    priority=10,
                    reason="pytest configuration detected in pyproject.",
                ),
            )
        if "ruff" in tool_section:
            self._add_command(
                validation,
                ValidationCommand(
                    command="ruff check .",
                    kind="lint",
                    source="pyproject.toml",
                    priority=20,
                    reason="ruff configuration detected in pyproject.",
                ),
            )
        if "mypy" in tool_section:
            self._add_command(
                validation,
                ValidationCommand(
                    command="mypy .",
                    kind="typecheck",
                    source="pyproject.toml",
                    priority=30,
                    reason="mypy configuration detected in pyproject.",
                ),
            )

    def _collect_requirements_commands(
        self,
        relative_path: str,
        validation: list[ValidationCommand],
    ) -> None:
        target = self.workspace.resolve_path(relative_path)
        content = target.read_text(encoding="utf-8").lower()
        if "pytest" in content:
            self._add_command(
                validation,
                ValidationCommand(
                    command="python -m pytest",
                    kind="test",
                    source=Path(relative_path).name,
                    priority=10,
                    reason="pytest dependency detected in requirements.",
                ),
            )
        if "ruff" in content:
            self._add_command(
                validation,
                ValidationCommand(
                    command="ruff check .",
                    kind="lint",
                    source=Path(relative_path).name,
                    priority=20,
                    reason="ruff dependency detected in requirements.",
                ),
            )
        if "mypy" in content or "pyright" in content:
            command = "pyright" if "pyright" in content else "mypy ."
            self._add_command(
                validation,
                ValidationCommand(
                    command=command,
                    kind="typecheck",
                    source=Path(relative_path).name,
                    priority=30,
                    reason="type checking dependency detected in requirements.",
                ),
            )

    def _collect_makefile_commands(
        self,
        relative_path: str,
        validation: list[ValidationCommand],
        workflow: list[ValidationCommand],
    ) -> None:
        target = self.workspace.resolve_path(relative_path)
        content = target.read_text(encoding="utf-8")
        targets = {
            match.group("target")
            for match in re.finditer(r"^(?P<target>[A-Za-z0-9_.-]+):", content, re.MULTILINE)
        }
        for target_name in sorted(targets):
            lowered = target_name.lower()
            if lowered not in SCRIPT_KIND_MAP:
                continue
            kind, priority = SCRIPT_KIND_MAP[lowered]
            spec = ValidationCommand(
                command=f"make {target_name}",
                kind=kind,
                source="Makefile",
                priority=priority,
                reason=f"Makefile target '{target_name}' detected.",
                required=kind in {"test", "lint", "typecheck", "build", "check"},
            )
            target_list = validation if spec.required else workflow
            self._add_command(target_list, spec)

    def _package_script_command(self, package_manager: str, script_name: str) -> str:
        if package_manager == "pnpm":
            return f"pnpm {script_name}" if script_name in {"test", "build"} else f"pnpm run {script_name}"
        if package_manager == "yarn":
            return f"yarn {script_name}"
        return f"npm run {script_name}"

    def _detect_package_manager(self, manifests: list[str]) -> str:
        lowered = {item.lower() for item in manifests}
        if "pnpm-lock.yaml" in lowered:
            return "pnpm"
        if "yarn.lock" in lowered:
            return "yarn"
        return "npm"

    def _build_repo_map(
        self,
        directory_counts: Counter[str],
        manifests: list[str],
        entrypoints: list[str],
    ) -> list[str]:
        lines = [
            f"{name}/ ({count} files)"
            for name, count in directory_counts.most_common(6)
            if name and name != "."
        ]
        if manifests:
            lines.append("manifests: " + ", ".join(manifests[:6]))
        if entrypoints:
            lines.append("entrypoints: " + ", ".join(entrypoints[:6]))
        return lines[:10]

    def _project_labels(
        self,
        *,
        language_counts: Counter[str],
        manifests: list[str],
        build_files: list[str],
        deploy_files: list[str],
    ) -> list[str]:
        labels: list[str] = []
        dominant_languages = [lang for lang, count in language_counts.most_common(3) if count > 0]
        for language in dominant_languages:
            if language != "other":
                labels.append(language)
        lower = {item.lower() for item in manifests}
        if "package.json" in lower:
            labels.append("node")
        if {"pyproject.toml", "requirements.txt", "requirements-runtime.txt"} & lower:
            labels.append("python-runtime")
        if any(".github/workflows" in item for item in build_files):
            labels.append("ci")
        if build_files:
            labels.append("build-aware")
        if deploy_files:
            labels.append("deploy-aware")
        return list(dict.fromkeys(labels))

    def _build_summary(
        self,
        *,
        file_count: int,
        language_counts: Counter[str],
        important_files: list[str],
        likely_commands: list[str],
        validation_commands: list[ValidationCommand],
        project_labels: list[str],
        focus: str | None,
    ) -> str:
        dominant = ", ".join(
            f"{lang} ({count})" for lang, count in language_counts.most_common(4)
        ) or "no dominant language detected"
        checks = ", ".join(
            f"{item.kind}:{item.command}" for item in validation_commands[:4]
        ) or ", ".join(likely_commands) or "no obvious validation command detected"
        summary = (
            f"The repository contains {file_count} scanned files. "
            f"Dominant file types: {dominant}. "
            f"Project labels: {', '.join(project_labels) or 'general repository'}. "
            f"Inspect {', '.join(important_files[:6]) or 'manifest and README files'} before broad edits. "
            f"Preferred validation sequence: {checks}."
        )
        if focus:
            summary += f" Current focus hint: {focus}."
        return summary

    def _add_command(
        self,
        bucket: list[ValidationCommand],
        command: ValidationCommand,
    ) -> None:
        if any(item.command == command.command for item in bucket):
            return
        bucket.append(command)

    def _is_test_file(self, relative_path: str) -> bool:
        rel = relative_path.lower()
        name = Path(relative_path).name.lower()
        return (
            "/tests/" in f"/{rel}"
            or name.startswith("test_")
            or name.endswith("_test.py")
            or name.endswith(".spec.ts")
            or name.endswith(".spec.js")
        )

    def _is_build_file(self, relative_path: str) -> bool:
        rel = relative_path.lower()
        name = Path(relative_path).name.lower()
        return any(
            token in rel
            for token in (
                "docker",
                ".github/workflows",
                "build",
                "vite.config",
                "webpack",
                "tsconfig",
            )
        ) or name in {"makefile", "package.json", "pyproject.toml"}

    def _is_deploy_file(self, relative_path: str) -> bool:
        rel = relative_path.lower()
        return any(
            token in rel
            for token in (
                "deploy",
                "release",
                "helm",
                "terraform",
                "k8s",
                "docker-compose",
            )
        )

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
        return tokens[:8]
