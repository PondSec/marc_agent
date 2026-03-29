from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from threading import Lock, Thread
from typing import Any

from config.settings import AppConfig
from llm.ollama_client import OllamaClient


RECOMMENDED_MODELS: tuple[dict[str, str], ...] = (
    {
        "name": "qwen3:14b",
        "label": "Qwen3 14B",
        "summary": "Standard fuer Coding-Arbeit, Refactors, Debugging und agentische Umsetzungen.",
    },
    {
        "name": "qwen3:8b",
        "label": "Qwen3 8B",
        "summary": "Schneller Pfad fuer Routing, kurze Shell-Aufgaben und kleine praezise Edits.",
    },
)

ACTIVE_INSTALL_STATES = {"queued", "pulling", "verifying"}


@dataclass(slots=True)
class ModelInstallJob:
    name: str
    status: str = "queued"
    message: str | None = None
    progress: float | None = None
    completed_bytes: int | None = None
    total_bytes: int | None = None
    error: str | None = None
    started_at: str | None = None
    updated_at: str | None = None


class ModelManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = OllamaClient(config)
        self._lock = Lock()
        self._jobs: dict[str, ModelInstallJob] = {}
        self._queue: list[str] = []
        self._worker: Thread | None = None

    def catalog(self) -> dict[str, Any]:
        installed_models = self.client.list_models_safe()
        installed_by_name = {
            str(item["name"]): item
            for item in installed_models
            if item.get("name")
        }
        with self._lock:
            jobs = {
                name: replace(job)
                for name, job in self._jobs.items()
            }
            queued_names = set(self._queue)

        recommended_models: list[dict[str, Any]] = []
        for spec in RECOMMENDED_MODELS:
            name = spec["name"]
            job = jobs.get(name)
            installed = name in installed_by_name
            status = "installed" if installed else "missing"
            message = "Bereit" if installed else "Noch nicht installiert"
            progress = 1.0 if installed else None
            completed_bytes = None
            total_bytes = None
            error = None
            updated_at = None

            if job and not installed:
                status = job.status
                message = job.message or _default_status_message(job.status)
                progress = job.progress
                completed_bytes = job.completed_bytes
                total_bytes = job.total_bytes
                error = job.error
                updated_at = job.updated_at
            elif job and installed and job.status == "failed":
                status = "installed"
                message = "Bereit"
                progress = 1.0

            if name in queued_names and status == "missing":
                status = "queued"
                message = "Wartet auf Download"

            recommended_models.append(
                {
                    "name": name,
                    "label": spec["label"],
                    "summary": spec["summary"],
                    "installed": installed,
                    "status": status,
                    "progress": progress,
                    "completed_bytes": completed_bytes,
                    "total_bytes": total_bytes,
                    "message": message,
                    "error": error,
                    "updated_at": updated_at,
                }
            )

        return {
            "installed_models": installed_models,
            "recommended_models": recommended_models,
        }

    def ensure_recommended(self) -> dict[str, Any]:
        if not self.client.is_available():
            return self.catalog()
        if not self.config.auto_install_recommended_models:
            return self.catalog()

        installed_names = {
            str(item.get("name"))
            for item in self.client.list_models_safe()
            if item.get("name")
        }

        with self._lock:
            for spec in RECOMMENDED_MODELS:
                name = spec["name"]
                if name in installed_names:
                    continue

                job = self._jobs.get(name)
                if job and job.status in ACTIVE_INSTALL_STATES:
                    continue
                if name in self._queue:
                    continue

                now = _utc_now()
                self._jobs[name] = ModelInstallJob(
                    name=name,
                    status="queued",
                    message="Wartet auf Download",
                    started_at=job.started_at if job and job.started_at else now,
                    updated_at=now,
                )
                self._queue.append(name)

            if self._queue and not self._worker_running_locked():
                self._worker = Thread(target=self._run_queue, daemon=True)
                self._worker.start()

        return self.catalog()

    def warmup_preferred_models(self) -> None:
        installed_names = {
            str(item.get("name"))
            for item in self.client.list_models_safe()
            if item.get("name")
        }
        # Warm the heavier content model first so the lightweight router model
        # remains loaded for the first interactive planning step after startup.
        warmup_targets: list[tuple[str, int]] = []
        for name, num_ctx in self._warmup_targets():
            if name not in installed_names:
                continue
            try:
                self.client.generate(
                    "Reply with OK only.",
                    model=name,
                    timeout=self.config.warmup_timeout,
                    num_ctx=num_ctx,
                    retries=0,
                )
            except Exception:
                continue

    def _warmup_targets(self) -> list[tuple[str, int]]:
        targets: list[tuple[str, int]] = []
        for name, num_ctx in (
            (self.config.model_name, max(int(self.config.ollama_num_ctx), 256)),
            (self.config.router_model_name, max(int(self.config.router_num_ctx), 256)),
        ):
            normalized = str(name or "").strip()
            if not normalized:
                continue
            if (normalized, num_ctx) in targets:
                continue
            targets.append((normalized, num_ctx))
        return targets

    def _run_queue(self) -> None:
        while True:
            with self._lock:
                if not self._queue:
                    self._worker = None
                    return
                name = self._queue.pop(0)
                job = self._jobs.get(name) or ModelInstallJob(name=name)
                now = _utc_now()
                job.status = "queued"
                job.message = "Download wird vorbereitet"
                job.started_at = job.started_at or now
                job.updated_at = now
                self._jobs[name] = job

            try:
                for payload in self.client.pull_model_events(name):
                    self._apply_pull_event(name, payload)
                installed_names = {
                    str(item.get("name"))
                    for item in self.client.list_models_safe()
                    if item.get("name")
                }
                with self._lock:
                    job = self._jobs.get(name) or ModelInstallJob(name=name)
                    if name in installed_names:
                        job.status = "installed"
                        job.message = "Installiert"
                        job.progress = 1.0
                        job.error = None
                    elif job.status != "failed":
                        job.status = "failed"
                        job.message = "Modell konnte nicht bestaetigt werden"
                        job.error = "Ollama meldet das Modell nach dem Download nicht als installiert."
                    job.updated_at = _utc_now()
                    self._jobs[name] = job
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    job = self._jobs.get(name) or ModelInstallJob(name=name)
                    job.status = "failed"
                    job.message = "Download fehlgeschlagen"
                    job.error = str(exc)
                    job.updated_at = _utc_now()
                    self._jobs[name] = job

    def _apply_pull_event(self, name: str, payload: dict[str, Any]) -> None:
        raw_status = str(payload.get("status") or "").strip()
        total_bytes = _coerce_int(payload.get("total"))
        completed_bytes = _coerce_int(payload.get("completed"))
        progress = None
        if total_bytes and completed_bytes is not None and total_bytes > 0:
            progress = max(0.0, min(1.0, completed_bytes / total_bytes))

        with self._lock:
            job = self._jobs.get(name) or ModelInstallJob(name=name)
            job.status = _map_pull_status(raw_status, progress)
            job.message = raw_status or _default_status_message(job.status)
            job.progress = progress if progress is not None else job.progress
            job.completed_bytes = completed_bytes
            job.total_bytes = total_bytes
            job.error = None
            job.started_at = job.started_at or _utc_now()
            job.updated_at = _utc_now()
            self._jobs[name] = job

    def _worker_running_locked(self) -> bool:
        return self._worker is not None and self._worker.is_alive()


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _default_status_message(status: str) -> str:
    if status == "installed":
        return "Installiert"
    if status == "queued":
        return "Wartet auf Download"
    if status == "verifying":
        return "Prueft heruntergeladene Daten"
    if status == "pulling":
        return "Laedt Modell herunter"
    if status == "failed":
        return "Download fehlgeschlagen"
    return "Unbekannter Status"


def _map_pull_status(raw_status: str, progress: float | None) -> str:
    lowered = raw_status.lower()
    if "success" in lowered:
        return "installed"
    if "verifying" in lowered or "writing manifest" in lowered or "removing any unused layers" in lowered:
        return "verifying"
    if progress is not None or "pull" in lowered or "download" in lowered or "manifest" in lowered:
        return "pulling"
    return "queued"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unique_model_names(names: list[str | None]) -> list[str]:
    unique: list[str] = []
    for name in names:
        value = str(name or "").strip()
        if not value or value in unique:
            continue
        unique.append(value)
    return unique
