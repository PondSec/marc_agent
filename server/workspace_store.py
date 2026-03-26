from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from agent.models import utc_now
from server.schemas import WorkspaceRecord


class WorkspaceStore:
    def __init__(self, target: Path):
        self.target = target
        self.target.parent.mkdir(parents=True, exist_ok=True)

    def list_workspaces(self) -> list[WorkspaceRecord]:
        payload = self._read_payload()
        return [
            WorkspaceRecord.model_validate(item)
            for item in payload.get("workspaces", [])
        ]

    def get(self, workspace_id: str | None) -> WorkspaceRecord | None:
        if not workspace_id:
            return None
        for workspace in self.list_workspaces():
            if workspace.id == workspace_id:
                return workspace
        return None

    def create(self, name: str, path: str) -> WorkspaceRecord:
        workspaces = self.list_workspaces()
        resolved_path = str(Path(path).expanduser().resolve())
        existing = next((item for item in workspaces if item.path == resolved_path), None)
        if existing is not None:
            updated = existing.model_copy(
                update={
                    "name": name.strip(),
                    "updated_at": utc_now(),
                }
            )
            return self._replace(updated, workspaces)

        now = utc_now()
        workspace = WorkspaceRecord(
            id=uuid4().hex[:12],
            name=name.strip(),
            path=resolved_path,
            created_at=now,
            updated_at=now,
        )
        workspaces.append(workspace)
        self._write_payload(workspaces)
        return workspace

    def update(
        self,
        workspace_id: str,
        *,
        name: str | None = None,
        path: str | None = None,
    ) -> WorkspaceRecord | None:
        workspaces = self.list_workspaces()
        existing = next((item for item in workspaces if item.id == workspace_id), None)
        if existing is None:
            return None

        update_data = {"updated_at": utc_now()}
        if name is not None:
            update_data["name"] = name.strip()
        if path is not None:
            update_data["path"] = str(Path(path).expanduser().resolve())

        updated = existing.model_copy(update=update_data)
        return self._replace(updated, workspaces)

    def ensure_default_workspace(self, path: str, name: str) -> WorkspaceRecord:
        resolved = str(Path(path).expanduser().resolve())
        existing = next(
            (item for item in self.list_workspaces() if item.path == resolved),
            None,
        )
        if existing is not None:
            return existing
        return self.create(name, resolved)

    def _replace(
        self,
        updated: WorkspaceRecord,
        workspaces: list[WorkspaceRecord],
    ) -> WorkspaceRecord:
        next_items = [updated if item.id == updated.id else item for item in workspaces]
        self._write_payload(next_items)
        return updated

    def _read_payload(self) -> dict:
        if not self.target.exists():
            return {"workspaces": []}
        return json.loads(self.target.read_text(encoding="utf-8"))

    def _write_payload(self, workspaces: list[WorkspaceRecord]) -> None:
        payload = {"workspaces": [item.model_dump() for item in workspaces]}
        self.target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
