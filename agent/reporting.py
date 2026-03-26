from __future__ import annotations

from pathlib import Path

from agent.models import SessionReport, SessionState
from config.settings import AppConfig


class SessionReporter:
    def __init__(self, config: AppConfig):
        self.config = config

    def build_report(self, session: SessionState) -> SessionReport:
        summary = self._summary(session)
        report = SessionReport(
            summary=summary,
            status=session.status,
            stop_reason=session.stop_reason,
            changed_files=[item.path for item in session.changed_files],
            commands=session.executed_commands[-20:],
            validation=session.validation_runs[-12:],
            diagnostics=session.diagnostics[-12:],
            blockers=session.blockers[-10:],
            helper_artifacts=session.helper_artifacts[-10:],
            notes=session.notes[-20:],
        )
        target = self.config.report_dir_path / f"{session.id}.json"
        final_report = report.model_copy(update={"report_path": str(target)})
        self._write_report(target, final_report)
        return final_report

    def render_final_response(
        self,
        session: SessionState,
        *,
        draft_response: str | None = None,
    ) -> str:
        report = session.report or self.build_report(session)
        lead = self._lead_message(session, draft_response)
        details: list[str] = []
        inspected_files: list[str] = []

        if report.changed_files:
            details.append(f"Geaendert: {self._join_items(report.changed_files, limit=4)}.")
            details.append(self._validation_sentence(session, report))
        else:
            inspected_files = self._inspected_files(session)
            if inspected_files:
                details.append(
                    f"Angesehen habe ich vor allem {self._join_items(inspected_files, limit=4)}."
                )

        blocker_text = self._join_items(report.blockers, limit=2, fallback="")
        diagnostic_text = self._join_items(
            [item.summary for item in report.diagnostics],
            limit=2,
            fallback="",
        )

        if blocker_text:
            details.append(f"Blocker: {blocker_text}.")
        elif session.validation_status == "failed":
            details.append("Validierung ist fehlgeschlagen und braucht noch einen Repair-Schritt.")
        elif diagnostic_text and session.status != "completed":
            details.append(f"Offener Hinweis: {diagnostic_text}.")

        if self._looks_like_greeting(session.task) and not report.changed_files and not inspected_files:
            details.append(
                "Sag mir einfach, was ich im Projekt analysieren, aendern oder beantworten soll."
            )

        detail_text = " ".join(part for part in details if part)
        if detail_text:
            return f"{lead}\n\n{detail_text}"
        return lead

    def _write_report(self, target: Path, report: SessionReport) -> None:
        target.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    def _summary(self, session: SessionState) -> str:
        if session.status == "completed" and session.validation_status == "passed":
            return "Task completed with validated changes."
        if session.status == "partial" and session.changed_files:
            return "Task ended with changes that still need follow-up or stronger validation."
        if session.status == "partial":
            return "Task stopped with blockers before a clean completion."
        if session.status == "failed":
            return "Task failed before reaching a stable result."
        return "Task finished without a decisive implementation outcome."

    def _lead_message(self, session: SessionState, draft_response: str | None) -> str:
        cleaned = self._clean_text(draft_response)
        if cleaned and not self._looks_like_machine_summary(cleaned):
            return cleaned

        if self._looks_like_greeting(session.task):
            return (
                "Hallo. Ich bin bereit."
                "\n\n"
                "Wenn du magst, kann ich den Code analysieren, einen Fehler suchen oder eine "
                "Aenderung umsetzen."
            )

        if session.status == "completed" and session.changed_files:
            return "Ich habe die Aufgabe umgesetzt."

        if session.status == "completed":
            return "Ich habe die Anfrage bearbeitet, aber keinen Code geaendert."

        if session.blockers:
            return "Ich bin auf einen Blocker gestossen und konnte die Aufgabe noch nicht sauber abschliessen."

        if session.status == "failed":
            return "Ich konnte in diesem Lauf noch kein belastbares Ergebnis liefern."

        return "Ich habe den Workspace untersucht, aber noch kein sauberes Abschlussergebnis erreicht."

    def _validation_sentence(self, session: SessionState, report: SessionReport) -> str:
        if session.validation_status == "passed":
            if report.validation:
                latest = report.validation[-1]
                return f"Validierung: bestanden ({latest.command})."
            return "Validierung: bestanden."

        if session.validation_status == "failed":
            if report.validation:
                latest = report.validation[-1]
                return f"Validierung: fehlgeschlagen ({latest.command})."
            return "Validierung: fehlgeschlagen."

        if session.validation_status == "blocked":
            return "Validierung: blockiert."

        if report.validation:
            latest = report.validation[-1]
            return f"Validierung: zuletzt {latest.status} ({latest.command})."

        return "Validierung: noch nicht gelaufen."

    def _inspected_files(self, session: SessionState) -> list[str]:
        files: list[str] = []
        for call in session.tool_calls:
            if call.tool_name != "read_file":
                continue
            path = str(call.tool_args.get("path") or "").strip()
            if path and path not in files:
                files.append(path)
        return files

    def _clean_text(self, text: str | None) -> str:
        return str(text or "").strip()

    def _looks_like_machine_summary(self, text: str) -> bool:
        lowered = text.lower()
        telemetry_markers = (
            "workflow_stage=",
            "changed files:",
            "validation:",
            "commands:",
            "diagnostics:",
            "blockers:",
            "task finished without",
        )
        if "status=" in lowered:
            return True
        return sum(marker in lowered for marker in telemetry_markers) >= 2

    def _looks_like_greeting(self, task: str) -> bool:
        normalized = " ".join(str(task or "").lower().split()).strip("!?., ")
        if not normalized:
            return False
        greetings = {
            "hallo",
            "hello",
            "hi",
            "hey",
            "moin",
            "servus",
            "guten morgen",
            "guten tag",
            "guten abend",
        }
        return normalized in greetings

    def _join_items(
        self,
        items: list[str],
        *,
        limit: int = 3,
        fallback: str = "nichts",
    ) -> str:
        visible = [item for item in items if item][:limit]
        if not visible:
            return fallback
        return ", ".join(visible)
