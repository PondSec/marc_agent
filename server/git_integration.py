from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from server.schemas import (
    GitRepositoryCandidate,
    GitSourceInspectionResponse,
    WorkspaceGitStatus,
    WorkspaceRecord,
)


class GitIntegrationError(RuntimeError):
    pass


_DISCOVERY_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}
_MAX_DISCOVERY_DEPTH = 4
_REMOTE_SOURCE_RE = re.compile(r"^(?:[a-z][a-z0-9+.-]*://|git@)", flags=re.IGNORECASE)
_REMOTE_LIKE_RE = re.compile(r"^[\w.-]+@[\w.-]+:.*$")


@dataclass(frozen=True, slots=True)
class GitSyncResult:
    sync_source: str | None
    remote_name: str | None
    remote_url: str | None
    branch: str | None


@dataclass(frozen=True, slots=True)
class _RepoSnapshot:
    path: str
    remote_name: str | None
    remote_url: str | None
    current_branch: str | None
    default_branch: str | None
    local_branches: list[str]
    remote_branches: list[str]
    has_uncommitted_changes: bool
    ahead_by: int
    behind_by: int


class GitWorkspaceService:
    def __init__(self, workspace_root: Path, *, state_dir_name: str):
        self.workspace_root = workspace_root.expanduser().resolve()
        self.state_dir_name = state_dir_name

    def discover_repositories(self, *, workspace_paths: Iterable[str] = ()) -> list[GitRepositoryCandidate]:
        candidates: list[GitRepositoryCandidate] = []
        seen: set[str] = set()

        def add_repo(path: Path) -> None:
            resolved = str(path.expanduser().resolve())
            if resolved in seen or not self._is_repo(path):
                return
            seen.add(resolved)
            snapshot = self._inspect_repo(path)
            candidates.append(
                GitRepositoryCandidate(
                    name=Path(resolved).name or resolved,
                    path=resolved,
                    remote_url=snapshot.remote_url,
                    current_branch=snapshot.current_branch,
                    local_branches=snapshot.local_branches,
                    remote_branches=snapshot.remote_branches,
                    has_uncommitted_changes=snapshot.has_uncommitted_changes,
                )
            )

        for raw_path in workspace_paths:
            if raw_path:
                add_repo(Path(raw_path))

        if self.workspace_root.exists():
            for current, dirnames, _ in os.walk(self.workspace_root):
                current_path = Path(current)
                try:
                    relative = current_path.relative_to(self.workspace_root)
                    depth = len(relative.parts)
                except ValueError:
                    depth = _MAX_DISCOVERY_DEPTH + 1
                dirnames[:] = [
                    item
                    for item in dirnames
                    if item not in _DISCOVERY_SKIP_DIRS and item != self.state_dir_name
                ]
                if ".git" in os.listdir(current_path):
                    add_repo(current_path)
                    dirnames[:] = []
                    continue
                if depth >= _MAX_DISCOVERY_DEPTH:
                    dirnames[:] = []
        return sorted(candidates, key=lambda item: (item.name.lower(), item.path.lower()))

    def inspect_source(self, source: str) -> GitSourceInspectionResponse:
        text = str(source or "").strip()
        if not text:
            raise GitIntegrationError("Provide a git repository URL or a local repository path.")
        local_path = Path(text).expanduser()
        if local_path.exists():
            resolved = local_path.resolve()
            if not self._is_repo(resolved):
                raise GitIntegrationError("The selected local path is not a git repository.")
            snapshot = self._inspect_repo(resolved)
            return GitSourceInspectionResponse(
                source=text,
                source_kind="local_path",
                resolved_path=snapshot.path,
                remote_url=snapshot.remote_url,
                current_branch=snapshot.current_branch,
                default_branch=snapshot.default_branch,
                local_branches=snapshot.local_branches,
                remote_branches=snapshot.remote_branches,
                has_uncommitted_changes=snapshot.has_uncommitted_changes,
            )
        if not self._looks_like_remote_source(text):
            raise GitIntegrationError("The git source must be a reachable repository URL or an existing local repository path.")
        remote_info = self._inspect_remote_source(text)
        return GitSourceInspectionResponse(
            source=text,
            source_kind="remote_url",
            resolved_path=None,
            remote_url=text,
            current_branch=None,
            default_branch=remote_info["default_branch"],
            local_branches=[],
            remote_branches=remote_info["branches"],
            has_uncommitted_changes=False,
        )

    def workspace_status(self, workspace: WorkspaceRecord) -> WorkspaceGitStatus:
        path = Path(workspace.path).expanduser().resolve()
        if not self._is_repo(path):
            return WorkspaceGitStatus(
                workspace_id=workspace.id,
                workspace_path=str(path),
                is_repo=False,
                configured_source=workspace.git_sync_source,
                remote_name=workspace.git_remote_name,
                configured_branch=workspace.git_branch,
                last_synced_at=workspace.last_git_sync_at,
            )
        snapshot = self._inspect_repo(path, preferred_remote=workspace.git_remote_name)
        return WorkspaceGitStatus(
            workspace_id=workspace.id,
            workspace_path=snapshot.path,
            is_repo=True,
            configured_source=workspace.git_sync_source,
            remote_name=snapshot.remote_name,
            remote_url=snapshot.remote_url,
            current_branch=snapshot.current_branch,
            configured_branch=workspace.git_branch,
            default_branch=snapshot.default_branch,
            local_branches=snapshot.local_branches,
            remote_branches=snapshot.remote_branches,
            has_uncommitted_changes=snapshot.has_uncommitted_changes,
            ahead_by=snapshot.ahead_by,
            behind_by=snapshot.behind_by,
            last_synced_at=workspace.last_git_sync_at,
        )

    def sync_workspace(
        self,
        workspace: WorkspaceRecord,
        *,
        sync_source: str | None = None,
        branch: str | None = None,
        remote_name: str | None = None,
    ) -> GitSyncResult:
        target = Path(workspace.path).expanduser().resolve()
        configured_source = str(sync_source or workspace.git_sync_source or "").strip() or None
        configured_branch = str(branch or workspace.git_branch or "").strip() or None
        configured_remote = str(remote_name or workspace.git_remote_name or "origin").strip() or "origin"

        if self._is_repo(target):
            return self._sync_existing_repo(
                target,
                sync_source=configured_source,
                branch=configured_branch,
                remote_name=configured_remote,
            )

        if configured_source is None:
            raise WorkspaceOperationError(
                "No git repository source is configured for this project. Choose a repository URL or an existing local repository path first."
            )

        local_source = Path(configured_source).expanduser()
        if local_source.exists() and local_source.resolve() == target:
            if not self._is_repo(target):
                raise GitIntegrationError("The selected project path is not a git repository yet.")
            return self._sync_existing_repo(
                target,
                sync_source=str(local_source.resolve()),
                branch=configured_branch,
                remote_name=configured_remote,
            )

        if target.exists() and any(target.iterdir()):
            raise GitIntegrationError(
                "The target project folder is not empty. Choose an existing repository path or an empty target folder before cloning."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        clone_command = ["clone", "--origin", configured_remote]
        if configured_branch:
            clone_command.extend(["--branch", configured_branch, "--single-branch"])
        clone_command.extend([configured_source, str(target)])
        self._run_git(clone_command, cwd=target.parent)
        snapshot = self._inspect_repo(target, preferred_remote=configured_remote)
        resolved_branch = configured_branch or snapshot.current_branch or snapshot.default_branch
        if resolved_branch and snapshot.current_branch != resolved_branch:
            self._checkout_branch(target, resolved_branch, configured_remote)
            snapshot = self._inspect_repo(target, preferred_remote=configured_remote)
        return GitSyncResult(
            sync_source=configured_source,
            remote_name=snapshot.remote_name or configured_remote,
            remote_url=snapshot.remote_url,
            branch=resolved_branch or snapshot.current_branch,
        )

    def _sync_existing_repo(
        self,
        target: Path,
        *,
        sync_source: str | None,
        branch: str | None,
        remote_name: str,
    ) -> GitSyncResult:
        snapshot = self._inspect_repo(target, preferred_remote=remote_name)
        effective_remote = snapshot.remote_name or remote_name

        if sync_source and not Path(sync_source).expanduser().exists():
            if snapshot.remote_name:
                current_url = snapshot.remote_url
                if current_url != sync_source:
                    self._run_git(["remote", "set-url", effective_remote, sync_source], cwd=target)
            else:
                self._run_git(["remote", "add", effective_remote, sync_source], cwd=target)
            snapshot = self._inspect_repo(target, preferred_remote=effective_remote)

        target_branch = branch or snapshot.current_branch or snapshot.default_branch
        if snapshot.has_uncommitted_changes:
            raise GitIntegrationError(
                "The repository has uncommitted changes. Commit or stash them before switching branches or pulling updates from the web UI."
            )

        if snapshot.remote_name:
            self._run_git(["fetch", "--prune", snapshot.remote_name], cwd=target)
            snapshot = self._inspect_repo(target, preferred_remote=snapshot.remote_name)

        if target_branch:
            self._checkout_branch(target, target_branch, snapshot.remote_name or effective_remote)
            snapshot = self._inspect_repo(target, preferred_remote=snapshot.remote_name or effective_remote)

        if snapshot.remote_name and snapshot.current_branch and snapshot.current_branch in snapshot.remote_branches:
            self._run_git(
                ["pull", "--ff-only", snapshot.remote_name, snapshot.current_branch],
                cwd=target,
            )
            snapshot = self._inspect_repo(target, preferred_remote=snapshot.remote_name)

        return GitSyncResult(
            sync_source=sync_source,
            remote_name=snapshot.remote_name or effective_remote,
            remote_url=snapshot.remote_url,
            branch=snapshot.current_branch or target_branch,
        )

    def _checkout_branch(self, target: Path, branch: str, remote_name: str | None) -> None:
        snapshot = self._inspect_repo(target, preferred_remote=remote_name)
        if snapshot.current_branch == branch:
            return
        if branch in snapshot.local_branches:
            self._run_git(["checkout", branch], cwd=target)
            return
        if remote_name and branch in snapshot.remote_branches:
            self._run_git(["checkout", "-b", branch, "--track", f"{remote_name}/{branch}"], cwd=target)
            return
        raise GitIntegrationError(f"The branch '{branch}' is not available in the selected repository.")

    def _inspect_repo(self, path: Path, *, preferred_remote: str | None = None) -> _RepoSnapshot:
        resolved = str(path.expanduser().resolve())
        if not self._is_repo(path):
            raise GitIntegrationError(f"{resolved} is not a git repository.")

        status_output = self._run_git(["status", "--porcelain=2", "--branch"], cwd=path)
        local_branches = self._lines_from_git(["for-each-ref", "refs/heads", "--format=%(refname:short)"], cwd=path)
        remotes = self._lines_from_git(["remote"], cwd=path)
        remote_name = preferred_remote if preferred_remote in remotes else (remotes[0] if remotes else None)
        remote_url = None
        remote_branches: list[str] = []
        default_branch = None
        if remote_name:
            remote_url = self._run_git(["remote", "get-url", remote_name], cwd=path).strip() or None
            remote_branches = [
                item
                for item in self._lines_from_git(
                    ["for-each-ref", f"refs/remotes/{remote_name}", "--format=%(refname:lstrip=3)"],
                    cwd=path,
                )
                if item and item != "HEAD"
            ]
            default_branch = self._remote_default_branch(path, remote_name)

        current_branch = None
        ahead_by = 0
        behind_by = 0
        has_uncommitted_changes = False
        for raw_line in status_output.splitlines():
            line = raw_line.strip()
            if line.startswith("# branch.head "):
                candidate = line.removeprefix("# branch.head ").strip()
                current_branch = None if candidate == "(detached)" else candidate
            elif line.startswith("# branch.ab "):
                match = re.search(r"\+(\d+)\s+\-(\d+)", line)
                if match is not None:
                    ahead_by = int(match.group(1))
                    behind_by = int(match.group(2))
            elif line and not line.startswith("#"):
                has_uncommitted_changes = True
        if current_branch is None:
            detached = self._run_git(["rev-parse", "--short", "HEAD"], cwd=path).strip()
            current_branch = detached or None

        return _RepoSnapshot(
            path=resolved,
            remote_name=remote_name,
            remote_url=remote_url,
            current_branch=current_branch,
            default_branch=default_branch,
            local_branches=local_branches,
            remote_branches=remote_branches,
            has_uncommitted_changes=has_uncommitted_changes,
            ahead_by=ahead_by,
            behind_by=behind_by,
        )

    def _inspect_remote_source(self, source: str) -> dict[str, object]:
        output = self._run_git(
            ["ls-remote", "--symref", "--heads", source, "HEAD"],
            cwd=self.workspace_root if self.workspace_root.exists() else Path.cwd(),
        )
        branches: list[str] = []
        default_branch = None
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("ref:"):
                match = re.match(r"ref:\s+refs/heads/([^\s]+)\s+HEAD", line)
                if match is not None:
                    default_branch = match.group(1)
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            ref = parts[1].strip()
            prefix = "refs/heads/"
            if not ref.startswith(prefix):
                continue
            branch = ref[len(prefix) :]
            if branch and branch not in branches:
                branches.append(branch)
        if default_branch is None and branches:
            default_branch = branches[0]
        return {"branches": branches, "default_branch": default_branch}

    def _remote_default_branch(self, path: Path, remote_name: str) -> str | None:
        try:
            output = self._run_git(["symbolic-ref", f"refs/remotes/{remote_name}/HEAD"], cwd=path).strip()
        except GitIntegrationError:
            return None
        prefix = f"refs/remotes/{remote_name}/"
        if output.startswith(prefix):
            return output[len(prefix) :]
        return None

    def _lines_from_git(self, args: list[str], *, cwd: Path) -> list[str]:
        output = self._run_git(args, cwd=cwd)
        return [line.strip() for line in output.splitlines() if line.strip()]

    def _run_git(self, args: list[str], *, cwd: Path) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            check=False,
            timeout=90,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or "git command failed"
            raise GitIntegrationError(message)
        return completed.stdout

    def _is_repo(self, path: Path) -> bool:
        if not path.exists():
            return False
        completed = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(path.expanduser().resolve()),
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        return completed.returncode == 0 and completed.stdout.strip() == "true"

    def _looks_like_remote_source(self, source: str) -> bool:
        text = str(source or "").strip()
        return bool(_REMOTE_SOURCE_RE.match(text) or _REMOTE_LIKE_RE.match(text))
