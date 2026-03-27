from __future__ import annotations

from pathlib import Path

from agent.models import SessionReport, SessionState
from config.settings import AppConfig


class SessionReporter:
    def __init__(self, config: AppConfig):
        self.config = config

    def build_report(self, session: SessionState) -> SessionReport:
        self._require_task_state(session)
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
        self._require_task_state(session)
        language = self._session_language(session)
        report = session.report or self.build_report(session)
        lead = self._lead_message(session, draft_response)
        details: list[str] = []
        inspected_files: list[str] = []

        if report.changed_files:
            details.append(
                self._localized_text(
                    language,
                    de=f"Geaendert: {self._join_items(report.changed_files, limit=4)}.",
                    en=f"Changed: {self._join_items(report.changed_files, limit=4)}.",
                )
            )
            details.append(self._validation_sentence(session, report))
        else:
            inspected_files = self._inspected_files(session)
            if inspected_files:
                details.append(
                    self._localized_text(
                        language,
                        de=f"Angesehen habe ich vor allem {self._join_items(inspected_files, limit=4)}.",
                        en=f"I mainly inspected {self._join_items(inspected_files, limit=4)}.",
                    )
                )

        blocker_text = self._join_items(report.blockers, limit=2, fallback="")
        diagnostic_text = self._join_items(
            [item.summary for item in report.diagnostics],
            limit=2,
            fallback="",
        )

        if blocker_text:
            details.append(self._localized_text(language, de=f"Blocker: {blocker_text}.", en=f"Blocker: {blocker_text}."))
        elif session.validation_status == "failed":
            details.append(
                self._localized_text(
                    language,
                    de="Validierung ist fehlgeschlagen und braucht noch einen Repair-Schritt.",
                    en="Validation failed and still needs a repair step.",
                )
            )
        elif diagnostic_text and session.status != "completed":
            details.append(
                self._localized_text(
                    language,
                    de=f"Offener Hinweis: {diagnostic_text}.",
                    en=f"Open note: {diagnostic_text}.",
                )
            )

        if self._is_intro_conversation(session) and not report.changed_files and not inspected_files:
            details.append(
                self._localized_text(
                    language,
                    de="Sag mir einfach, was ich im Projekt analysieren, aendern oder beantworten soll.",
                    en="Tell me what you want me to analyze, change, or answer in this project.",
                )
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
        language = self._session_language(session)

        if self._is_intro_conversation(session):
            return self._localized_text(
                language,
                de=(
                    "Hallo. Ich bin bereit."
                    "\n\n"
                    "Wenn du magst, kann ich den Code analysieren, einen Fehler suchen oder eine "
                    "Aenderung umsetzen."
                ),
                en=(
                    "Hello. I am ready."
                    "\n\n"
                    "I can analyze the code, investigate a bug, or implement a change."
                ),
            )

        if session.status == "completed" and session.changed_files:
            return self._localized_text(
                language,
                de="Ich habe die Aufgabe umgesetzt.",
                en="I implemented the task.",
            )

        if session.status == "completed":
            return self._localized_text(
                language,
                de="Ich habe die Anfrage bearbeitet, aber keinen Code geaendert.",
                en="I handled the request, but I did not change any code.",
            )

        if session.status == "partial" and session.changed_files:
            return self._localized_text(
                language,
                de="Ich habe Aenderungen umgesetzt, aber ich kann noch keinen sauber validierten Abschluss melden.",
                en="I made changes, but I cannot claim a clean validated completion yet.",
            )

        if session.blockers:
            return self._localized_text(
                language,
                de="Ich bin auf einen Blocker gestossen und konnte die Aufgabe noch nicht sauber abschliessen.",
                en="I hit a blocker and could not finish the task cleanly yet.",
            )

        if session.status == "failed":
            return self._localized_text(
                language,
                de="Ich konnte in diesem Lauf noch kein belastbares Ergebnis liefern.",
                en="I could not produce a reliable result in this run.",
            )

        return self._localized_text(
            language,
            de="Ich habe den Workspace untersucht, aber noch kein sauberes Abschlussergebnis erreicht.",
            en="I inspected the workspace, but I have not reached a clean final outcome yet.",
        )

    def _validation_sentence(self, session: SessionState, report: SessionReport) -> str:
        language = self._session_language(session)
        if session.stop_reason == "functional_validation_missing":
            return self._localized_text(
                language,
                de="Validierung: nur statische Checks wurden bestaetigt, ein funktionaler Repro- oder Smoke-Test fehlt noch.",
                en="Validation: only static checks were confirmed, and a functional reproduction or smoke test is still missing.",
            )
        if session.validation_status == "passed":
            if report.validation:
                latest = report.validation[-1]
                return self._localized_text(
                    language,
                    de=f"Validierung: bestanden ({latest.command}).",
                    en=f"Validation: passed ({latest.command}).",
                )
            return self._localized_text(language, de="Validierung: bestanden.", en="Validation: passed.")

        if session.validation_status == "failed":
            if report.validation:
                latest = report.validation[-1]
                return self._localized_text(
                    language,
                    de=f"Validierung: fehlgeschlagen ({latest.command}).",
                    en=f"Validation: failed ({latest.command}).",
                )
            return self._localized_text(language, de="Validierung: fehlgeschlagen.", en="Validation: failed.")

        if session.validation_status == "blocked":
            return self._localized_text(language, de="Validierung: blockiert.", en="Validation: blocked.")

        if report.validation:
            latest = report.validation[-1]
            return self._localized_text(
                language,
                de=f"Validierung: zuletzt {latest.status} ({latest.command}).",
                en=f"Validation: last status {latest.status} ({latest.command}).",
            )

        if report.changed_files:
            return self._localized_text(
                language,
                de="Validierung: kein sinnvoller Check wurde in diesem Lauf bestaetigt.",
                en="Validation: no meaningful check was confirmed in this run.",
            )
        return self._localized_text(
            language,
            de="Validierung: noch nicht gelaufen.",
            en="Validation: not run yet.",
        )

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

    def _is_intro_conversation(self, session: SessionState) -> bool:
        route = session.router_result
        if route is not None:
            first_action = route.action_plan[0].action.value if route.action_plan else ""
            if (
                route.intent.value == "explain"
                and not route.repo_context_needed
                and first_action == "respond_directly"
            ):
                return True
        task_state = session.task_state
        if task_state is not None and task_state.next_action == "explain" and not task_state.target_artifacts:
            return True
        return False

    def _require_task_state(self, session: SessionState) -> None:
        if session.task_state is None:
            raise RuntimeError(
                "SessionReporter requires session.task_state. Reporting must run on committed task state."
            )

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

    def _session_language(self, session: SessionState) -> str:
        task_state = session.task_state
        if task_state is not None and task_state.latest_user_turn:
            return self._language_for_text(task_state.latest_user_turn)
        return self._language_for_text(session.task)

    def _language_for_text(self, text: str | None) -> str:
        normalized = str(text or "").lower()
        german_markers = (
            " lies ",
            " fasse ",
            " ich ",
            " bitte ",
            " mach",
            " bau",
            " aenderung",
            " pruef",
            "prüf",
            "fehler",
            "datei",
            "sicher",
            "kannst",
            "moechte",
            "möchte",
            "jetzt",
            "dazu",
            "nur ",
            " zusammen",
        )
        padded = f" {normalized} "
        if any(marker in padded for marker in german_markers) or any(char in normalized for char in "äöüß"):
            return "de"
        return "en"

    def _localized_text(self, language: str, *, de: str, en: str) -> str:
        return de if language == "de" else en
