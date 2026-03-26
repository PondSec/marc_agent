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

    def render_final_response(self, session: SessionState) -> str:
        report = session.report or self.build_report(session)
        changed = ", ".join(report.changed_files) or "none"
        commands = ", ".join(report.commands[-6:]) or "none"
        validation = (
            ", ".join(
                f"{item.kind or 'check'}:{item.status}:{item.command}"
                for item in report.validation[-4:]
            )
            or "none"
        )
        blockers = " | ".join(report.blockers[-3:]) or "none"
        diagnostics = " | ".join(item.summary for item in report.diagnostics[-3:]) or "none"
        return (
            f"{report.summary} Changed files: {changed}. Commands: {commands}. "
            f"Validation: {validation}. Blockers: {blockers}. Diagnostics: {diagnostics}."
        )

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
