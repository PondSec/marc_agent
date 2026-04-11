from __future__ import annotations

from pathlib import Path

from agent.models import SessionReport, SessionState
from agent.verification import ValidationPlanner
from config.settings import AppConfig


class SessionReporter:
    def __init__(self, config: AppConfig):
        self.config = config
        self.validation_planner = ValidationPlanner()

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
            runtime_executions=session.runtime_executions[-12:],
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
            degraded_note = self._degraded_execution_note(session)
            if degraded_note:
                details.append(degraded_note)
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
        repair_summary = (
            str(session.active_repair_context.failure_summary or "").strip()
            if session.active_repair_context is not None
            else ""
        )

        if blocker_text:
            details.append(self._localized_text(language, de=f"Blocker: {blocker_text}.", en=f"Blocker: {blocker_text}."))
            startup_note = self._startup_failure_note(session)
            if startup_note:
                details.append(startup_note)
        elif session.stop_reason in {"no_effective_change", "repair_blocked_after_validation_failure"} and repair_summary:
            details.append(
                self._localized_text(
                    language,
                    de=f"Repair-Blocker: Aus der Validierungs-Evidenz liess sich noch keine wirksame, nicht-aequivalente Aenderung ableiten ({repair_summary}).",
                    en=f"Repair blocker: the validation evidence still did not support a substantive, non-equivalent fix ({repair_summary}).",
                )
            )
        elif session.validation_status == "bootstrap_failed":
            details.append(
                self._localized_text(
                    language,
                    de="Validierung scheitert aktuell im Bootstrap- oder Discovery-Pfad und ist noch nicht im normalen Repair-Fall.",
                    en="Validation is currently failing in the bootstrap or discovery path and is not yet in a normal repair case.",
                )
            )
        elif session.validation_status == "bootstrap_reset_required":
            details.append(
                self._localized_text(
                    language,
                    de="Der Bootstrap-Pfad braucht vor weiteren Retries einen Reset oder menschliche Entscheidung.",
                    en="The bootstrap path needs a reset or human intervention before more retries.",
                )
            )
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
        if self._latest_runtime_execution(session, operation_name="content_generation", final_state="degraded_success") is not None:
            return "Task completed with a degraded but honest runtime recovery path."
        if session.status == "partial" and session.changed_files:
            return "Task ended with changes that still need follow-up or stronger validation."
        if session.status == "partial":
            return "Task stopped with blockers before a clean completion."
        if session.status == "failed":
            return "Task failed before reaching a stable result."
        return "Task finished without a decisive implementation outcome."

    def _lead_message(self, session: SessionState, draft_response: str | None) -> str:
        cleaned = self._clean_text(draft_response)
        if cleaned and not self._looks_like_machine_summary(cleaned) and not self._conflicts_with_session(cleaned, session):
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

        if session.stop_reason == "model_start_failed":
            return self._localized_text(
                language,
                de="Ich konnte die Generierung in diesem Lauf nicht einmal sauber starten.",
                en="I could not cleanly start generation in this run.",
            )

        if session.stop_reason == "repair_generation_failed":
            return self._localized_text(
                language,
                de="Ich konnte den Repair-Inhalt in diesem Lauf nicht stabil erzeugen.",
                en="I could not generate stable repair content in this run.",
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

    def _degraded_execution_note(self, session: SessionState) -> str | None:
        execution = self._latest_runtime_execution(session, operation_name="content_generation", final_state="degraded_success")
        if execution is None:
            return None
        strategy = str(execution.get("recovery_strategy") or "").strip()
        language = self._session_language(session)
        if strategy == "starter_scaffold":
            return self._localized_text(
                language,
                de="Ausfuehrung: Ich habe nur ein minimales Starter-Grundgeruest erzeugt, keine vollstaendige frei generierte Implementierung.",
                en="Execution: I only produced a minimal starter scaffold, not a full free-form generated implementation.",
            )
        if strategy == "deterministic_template":
            return self._localized_text(
                language,
                de="Ausfuehrung: Ich habe einen deterministischen Vorlagen-Fallback genutzt, keine vollstaendige frei generierte Implementierung.",
                en="Execution: I used a deterministic template fallback, not a full free-form generated implementation.",
            )
        if str(execution.get("capability_tier") or "").strip() == "tier_c":
            return self._localized_text(
                language,
                de="Ausfuehrung: Ich habe nur eine minimal tragfaehige Generation erzeugt.",
                en="Execution: I produced only a minimally viable generation.",
            )
        return None

    def _startup_failure_note(self, session: SessionState) -> str | None:
        if session.stop_reason != "model_start_failed":
            return None
        execution = self._latest_runtime_execution(session, operation_name="content_generation")
        if execution is None:
            return None
        language = self._session_language(session)
        return self._localized_text(
            language,
            de="Verstehen und Planung waren abgeschlossen, aber die eigentliche Generierung lieferte vor dem Start-Timeout keinen ersten Output.",
            en="Understanding and planning completed, but the actual generation produced no first output before the startup timeout.",
        )

    def _latest_runtime_execution(
        self,
        session: SessionState,
        *,
        operation_name: str | None = None,
        final_state: str | None = None,
    ) -> dict | None:
        for item in reversed(session.runtime_executions):
            if operation_name and str(item.get("operation_name") or "").strip() != operation_name:
                continue
            if final_state and str(item.get("final_state") or "").strip() != final_state:
                continue
            return item
        return None

    def _validation_sentence(self, session: SessionState, report: SessionReport) -> str:
        language = self._session_language(session)
        strongest_scope = self.validation_planner.strongest_passed_scope(
            session,
            current_generation_only=False,
        )
        strongest_run = self._latest_run_for_scope(report, strongest_scope)
        if session.stop_reason == "functional_validation_missing":
            if strongest_scope == "structural":
                return self._localized_text(
                    language,
                    de="Validierung: strukturelle Web-Checks wurden bestaetigt, aber ein funktionaler Browser- oder Runtime-Smoke-Test fehlt noch.",
                    en="Validation: structural web checks were confirmed, but a functional browser or runtime smoke test is still missing.",
                )
            return self._localized_text(
                language,
                de="Validierung: nur statische Checks wurden bestaetigt, ein funktionaler Repro- oder Smoke-Test fehlt noch.",
                en="Validation: only static checks were confirmed, and a functional reproduction or smoke test is still missing.",
            )
        if session.stop_reason == "requirements_review_missing":
            return self._localized_text(
                language,
                de="Validierung: Projekt-Checks koennen vorliegen, aber die allgemeine Anforderungspruefung ueber die geaenderten Artefakte ist noch nicht abgeschlossen.",
                en="Validation: project checks may exist, but the general requirements review across the changed artifacts is not complete yet.",
            )
        if session.validation_status == "passed":
            if report.validation:
                if strongest_scope == "runtime":
                    return self._localized_text(
                        language,
                        de=f"Validierung: funktionaler Check bestanden ({(strongest_run or report.validation[-1]).command}).",
                        en=f"Validation: functional check passed ({(strongest_run or report.validation[-1]).command}).",
                    )
                if strongest_scope == "structural":
                    return self._localized_text(
                        language,
                        de=f"Validierung: strukturelle Web-Checks bestanden ({(strongest_run or report.validation[-1]).command}); funktionaler Smoke-Test fehlt noch.",
                        en=f"Validation: structural web checks passed ({(strongest_run or report.validation[-1]).command}); a functional smoke test is still missing.",
                    )
                if strongest_scope == "semantic":
                    return self._localized_text(
                        language,
                        de=f"Validierung: allgemeine Anforderungspruefung bestanden ({(strongest_run or report.validation[-1]).command}).",
                        en=f"Validation: general requirements review passed ({(strongest_run or report.validation[-1]).command}).",
                    )
                if strongest_scope == "static":
                    return self._localized_text(
                        language,
                        de=f"Validierung: statische Checks bestanden ({(strongest_run or report.validation[-1]).command}).",
                        en=f"Validation: static checks passed ({(strongest_run or report.validation[-1]).command}).",
                    )
                if strongest_scope == "syntax":
                    return self._localized_text(
                        language,
                        de=f"Validierung: Syntax-Checks bestanden ({(strongest_run or report.validation[-1]).command}).",
                        en=f"Validation: syntax checks passed ({(strongest_run or report.validation[-1]).command}).",
                    )
                latest = report.validation[-1]
                return self._localized_text(
                    language,
                    de=f"Validierung: bestanden ({latest.command}).",
                    en=f"Validation: passed ({latest.command}).",
                )
            return self._localized_text(language, de="Validierung: bestanden.", en="Validation: passed.")

        if session.validation_status == "bootstrap_failed":
            if report.validation:
                latest = report.validation[-1]
                return self._localized_text(
                    language,
                    de=f"Validierung: Bootstrap-/Discovery-Fehler ({latest.command}).",
                    en=f"Validation: bootstrap/discovery failure ({latest.command}).",
                )
            return self._localized_text(
                language,
                de="Validierung: Bootstrap-/Discovery-Fehler.",
                en="Validation: bootstrap/discovery failure.",
            )

        if session.validation_status == "bootstrap_reset_required":
            if report.validation:
                latest = report.validation[-1]
                return self._localized_text(
                    language,
                    de=f"Validierung: bootstrap_reset_required ({latest.command}).",
                    en=f"Validation: bootstrap_reset_required ({latest.command}).",
                )
            return self._localized_text(
                language,
                de="Validierung: bootstrap_reset_required.",
                en="Validation: bootstrap_reset_required.",
            )

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

    def _conflicts_with_session(self, text: str, session: SessionState) -> bool:
        lowered = str(text or "").lower()
        validation_claim_markers = ("validiert", "validated", "validation: passed", "clean validated")
        if session.stop_reason == "functional_validation_missing" and any(marker in lowered for marker in validation_claim_markers):
            return True
        if session.stop_reason == "requirements_review_missing" and any(marker in lowered for marker in validation_claim_markers):
            return True
        if session.status == "partial" and any(marker in lowered for marker in validation_claim_markers):
            return True
        return False

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
        if route is None:
            return False
        first_action = route.action_plan[0].action.value if route.action_plan else ""
        return (
            route.intent.value == "explain"
            and not route.repo_context_needed
            and first_action == "respond_directly"
            and bool(route.direct_response)
        )

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

    def _latest_run_for_scope(self, report: SessionReport, scope: str | None):
        if scope is None:
            return None
        for item in reversed(report.validation):
            if item.status == "passed" and item.verification_scope == scope:
                return item
        return None

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
