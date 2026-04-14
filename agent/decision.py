from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from agent.models import SessionState, WorkspaceSnapshot
from agent.semantic_defaults import (
    classify_conversation_request,
    infer_artifact_name_hint,
    infer_requested_extension,
    infer_scope_tokens,
)
from agent.task_state import TaskState
from agent.task_schema import TaskArtifact, TaskUnderstanding
from llm.schemas import RouteActionName, RouteIntent, RouterOutput
from runtime.logger import AgentLogger

_EXPLICIT_PATH_RE = re.compile(
    r"([\w./-]+\.(?:py|js|ts|tsx|jsx|json|md|txt|html|css|sh|toml|ya?ml|go|rs|java|kt|rb|ini|cfg|conf|env|log|sql|xml|svg|csv))",
    flags=re.IGNORECASE,
)


class ExecutionDecisionPolicy:
    """Maps committed task state into an executable route."""

    HIGH_CONFIDENCE = 0.78
    MEDIUM_CONFIDENCE = 0.5

    def __init__(self, *, logger: AgentLogger | None = None):
        self.logger = logger

    def build_route(
        self,
        task_state: TaskState,
        *,
        snapshot: WorkspaceSnapshot | None = None,
        session: SessionState | None = None,
    ) -> RouterOutput:
        return self._build_route_from_understanding(
            task_state.to_task_understanding(),
            snapshot=snapshot,
            session=session,
            task_state=task_state,
        )

    def _build_route_from_understanding(
        self,
        understanding: TaskUnderstanding,
        *,
        snapshot: WorkspaceSnapshot | None = None,
        session: SessionState | None = None,
        task_state: TaskState | None = None,
    ) -> RouterOutput:
        language = self._request_language(understanding, task_state=task_state)
        intent = self._route_intent(understanding, task_state=task_state)
        target_paths = self._target_paths(understanding, session=session, intent=intent)
        target_name = self._target_name(understanding, target_paths)
        search_terms = self._search_terms(understanding, target_paths, target_name)
        relevant_extensions = self._relevant_extensions(understanding, target_paths)

        if self._should_clarify(
            understanding,
            intent=intent,
            target_paths=target_paths,
            target_name=target_name,
        ):
            route = RouterOutput(
                user_goal=understanding.interpreted_goal,
                intent=RouteIntent.UNKNOWN,
                entities={
                    "target_type": "file" if target_paths else ("artifact" if target_name else None),
                    "target_name": target_name,
                    "target_paths": target_paths,
                    "attributes": understanding.subgoals[:6],
                    "constraints": self._route_constraints(understanding, task_state=task_state),
                },
                requested_outcome=self._localized_text(
                    language,
                    de="Fehlendes Ziel, fehlenden Scope oder Risikofaktoren klaeren, bevor ein Schritt ausgefuehrt wird.",
                    en="Clarify the missing target, scope, or risk before taking action.",
                ),
                action_plan=[
                    {
                        "step": 1,
                        "action": RouteActionName.ASK_CLARIFICATION,
                        "reason": self._clarification_reason(
                            understanding,
                            intent=intent,
                            target_paths=target_paths,
                            target_name=target_name,
                            language=language,
                        ),
                    }
                ],
                needs_clarification=True,
                clarification_questions=self._clarification_questions(understanding),
                confidence=max(min(understanding.confidence, 0.49), 0.05),
                safe_to_execute=False,
                repo_context_needed=False,
                search_terms=search_terms,
                relevant_extensions=relevant_extensions,
                direct_response=None,
            )
            self._log(
                "execution_decision",
                original_request=understanding.original_request,
                interpreted_goal=understanding.interpreted_goal,
                chosen_intent=route.intent,
                chosen_action=route.action_plan[0].action.value,
                confidence=route.confidence,
                clarification_reason=route.action_plan[0].reason,
            )
            return route

        route = RouterOutput(
            user_goal=understanding.interpreted_goal,
            intent=intent,
            entities={
                "target_type": "file" if target_paths else ("artifact" if target_name else None),
                "target_name": target_name,
                "target_paths": target_paths,
                "attributes": understanding.subgoals[:6],
                "constraints": self._route_constraints(understanding, task_state=task_state),
            },
            requested_outcome=self._requested_outcome(understanding, task_state=task_state, language=language),
            action_plan=self._action_plan(
                understanding,
                intent=intent,
                target_paths=target_paths,
                snapshot=snapshot,
                task_state=task_state,
                language=language,
            ),
            needs_clarification=False,
            clarification_questions=[],
            confidence=max(understanding.confidence, self.MEDIUM_CONFIDENCE),
            safe_to_execute=self._safe_to_execute(
                understanding,
                intent=intent,
                target_paths=target_paths,
                target_name=target_name,
            ),
            repo_context_needed=self._repo_context_needed(intent),
            search_terms=search_terms,
            relevant_extensions=relevant_extensions,
            direct_response=None,
        )
        self._log(
            "execution_decision",
            original_request=understanding.original_request,
            interpreted_goal=understanding.interpreted_goal,
            chosen_intent=route.intent,
            chosen_action=route.action_plan[0].action.value,
            confidence=route.confidence,
            clarification_reason=None,
        )
        return route

    def _route_intent(
        self,
        understanding: TaskUnderstanding,
        *,
        task_state: TaskState | None = None,
    ) -> RouteIntent:
        strategy = task_state.execution_strategy if task_state is not None else None
        current_user_intent = task_state.current_user_intent if task_state is not None else None
        goal_relation = task_state.goal_relation if task_state is not None else None
        mode = understanding.recommended_mode
        intent = understanding.intent_category
        has_named_targets = any(
            str(artifact.path or artifact.name or "").strip()
            for artifact in understanding.target_artifacts
        )
        bootstrap_create = (
            goal_relation == "new_task"
            and mode == "create"
            and current_user_intent in {"implement", "repair"}
            and task_state is not None
            and not task_state.evidence
            and not task_state.supplied_evidence
        )

        if bootstrap_create:
            return RouteIntent.CREATE
        if strategy == "debug_repair":
            return RouteIntent.DEBUG
        if strategy in {"refactor", "hardening", "rollback_correction"}:
            return RouteIntent.UPDATE
        if strategy == "feature_implementation":
            return RouteIntent.CREATE if mode == "create" or intent == "build" else RouteIntent.UPDATE
        if current_user_intent == "repair" or goal_relation == "report_problem":
            return RouteIntent.DEBUG
        if current_user_intent == "correct" or goal_relation in {"correct", "scope_change", "rollback_request"}:
            return RouteIntent.UPDATE
        if current_user_intent == "implement":
            if mode == "create" or intent == "build" or not has_named_targets:
                return RouteIntent.CREATE
            return RouteIntent.UPDATE
        if strategy == "validation_inspection":
            if mode == "test" or intent == "test":
                return RouteIntent.DEBUG
            if mode == "search" or intent == "search":
                return RouteIntent.SEARCH
            if mode == "plan" or intent == "plan":
                return RouteIntent.PLAN
            if mode == "inspect":
                return RouteIntent.INSPECT
            if mode == "explain" or intent in {"explain", "analyze"}:
                return RouteIntent.EXPLAIN
        if mode == "create" or intent == "build":
            return RouteIntent.CREATE
        if mode in {"modify", "refactor"} or intent in {"modify", "refactor", "configure"}:
            return RouteIntent.UPDATE
        if mode in {"debug", "test"} or intent in {"debug", "test"}:
            return RouteIntent.DEBUG
        if mode == "search" or intent == "search":
            return RouteIntent.SEARCH
        if mode == "plan" or intent == "plan":
            return RouteIntent.PLAN
        if mode == "inspect":
            return RouteIntent.INSPECT
        if mode == "explain" or intent in {"explain", "analyze"}:
            return RouteIntent.EXPLAIN
        return RouteIntent.UNKNOWN

    def _requested_outcome(
        self,
        understanding: TaskUnderstanding,
        *,
        task_state: TaskState | None = None,
        language: str | None = None,
    ) -> str:
        language = language or self._request_language(understanding, task_state=task_state)
        strategy = task_state.execution_strategy if task_state is not None else None
        if strategy == "rollback_correction":
            return self._localized_text(
                language,
                de="Den aktiven Scope oder die vorherige Aenderung mit dem kleinsten noetigen Update korrigieren und danach die neue Grenze pruefen.",
                en="Correct the active scope or prior change with the smallest necessary update, then verify the new boundary.",
            )
        if strategy == "hardening":
            return self._localized_text(
                language,
                de="Das gezielte Verhalten mit minimalem Scope haerten und pruefen, dass das gewollte Verhalten weiter funktioniert.",
                en="Harden the targeted behavior with minimal scope and verify that intended behavior still works.",
            )
        if understanding.intent_category == "debug":
            return self._localized_text(
                language,
                de="Das Problem anhand der aktuellen Evidenz diagnostizieren und danach den kleinsten sicheren Fix anwenden.",
                en="Diagnose the issue against current evidence, then apply the smallest safe fix.",
            )
        if understanding.intent_category == "refactor":
            return self._localized_text(
                language,
                de="Struktur und Wartbarkeit verbessern, ohne das Verhalten zu veraendern.",
                en="Improve structure and maintainability while preserving behavior.",
            )
        if understanding.intent_category == "test":
            return self._localized_text(
                language,
                de="Die aktuelle Implementierung pruefen und relevante Fehler klar sichtbar machen.",
                en="Verify the current implementation and surface the relevant failures clearly.",
            )
        return understanding.interpreted_goal

    def _action_plan(
        self,
        understanding: TaskUnderstanding,
        *,
        intent: RouteIntent,
        target_paths: list[str],
        snapshot: WorkspaceSnapshot | None,
        task_state: TaskState | None = None,
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        language = language or self._request_language(understanding, task_state=task_state)
        strategy = task_state.execution_strategy if task_state is not None else None
        actions: list[tuple[RouteActionName, str]] = []

        if intent == RouteIntent.PLAN:
            actions = [
                (
                    RouteActionName.PLAN_WORK,
                    self._localized_text(
                        language,
                        de="Der Nutzer will vor der Ausfuehrung zuerst einen konkreten Plan.",
                        en="The user primarily wants a concrete plan before execution.",
                    ),
                ),
            ]
        elif intent == RouteIntent.SEARCH:
            actions = [
                (
                    RouteActionName.SEARCH_WORKSPACE,
                    self._localized_text(
                        language,
                        de="Das Ziel soll zuerst semantisch im Workspace gefunden werden.",
                        en="The target should be located semantically in the workspace first.",
                    ),
                ),
                (
                    RouteActionName.READ_RELEVANT_FILES,
                    self._localized_text(
                        language,
                        de="Die staerksten Treffer erst lesen, bevor ich antworte.",
                        en="Inspect the strongest search hits before answering.",
                    ),
                ),
                (
                    RouteActionName.SUMMARIZE_RESULT,
                    self._localized_text(
                        language,
                        de="Die gefundene Implementierung oder die Erkenntnisse knapp zurueckgeben.",
                        en="Return the located implementation or findings.",
                    ),
                ),
            ]
        elif intent == RouteIntent.INSPECT:
            actions = [
                (
                    RouteActionName.INSPECT_WORKSPACE,
                    self._localized_text(
                        language,
                        de="Gezielten Repository-Kontext sammeln, bevor ich antworte.",
                        en="Collect focused repository context before answering.",
                    ),
                ),
                (
                    RouteActionName.READ_RELEVANT_FILES,
                    self._localized_text(
                        language,
                        de="Die relevantesten Artefakte lesen, bevor ich zusammenfasse.",
                        en="Inspect the most relevant artifacts before summarizing.",
                    ),
                ),
                (
                    RouteActionName.SUMMARIZE_RESULT,
                    self._localized_text(
                        language,
                        de="Die Erkenntnisse knapp und verstaendlich zusammenfassen.",
                        en="Return a concise explanation of the findings.",
                    ),
                ),
            ]
        elif intent == RouteIntent.EXPLAIN:
            if target_paths:
                actions = [
                    (
                        RouteActionName.READ_RELEVANT_FILES,
                        self._localized_text(
                            language,
                            de="Die referenzierten Artefakte erst lesen, bevor ich sie erklaere.",
                            en="Read the referenced artifacts before explaining them.",
                        ),
                    ),
                    (
                        RouteActionName.SUMMARIZE_RESULT,
                        self._localized_text(
                            language,
                            de="Das Ergebnis in nutzerfreundlicher Sprache erklaeren.",
                            en="Explain the result in user-facing language.",
                        ),
                    ),
                ]
            else:
                actions = [
                    (
                        RouteActionName.RESPOND_DIRECTLY,
                        self._localized_text(
                            language,
                            de="Die Anfrage kann direkt aus dem interpretierten Ziel beantwortet werden.",
                            en="The request can be answered directly from the interpreted goal.",
                        ),
                    ),
                ]
        elif intent == RouteIntent.CREATE:
            if snapshot is not None and snapshot.file_count > 0:
                actions.append(
                    (
                        RouteActionName.INSPECT_WORKSPACE,
                        self._localized_text(
                            language,
                            de="Bestehende Projektkonventionen lesen, bevor neuer Code entsteht.",
                            en="Inspect existing project conventions before creating new code.",
                        ),
                    )
                )
            actions.extend(
                [
                    (
                        RouteActionName.CREATE_ARTIFACT,
                        self._localized_text(
                            language,
                            de="Das angefragte Artefakt im kleinsten sinnvollen Scope erstellen.",
                            en="Create the requested artifact in the smallest sensible scope.",
                        ),
                    ),
                    (
                        RouteActionName.RUN_VALIDATION,
                        self._validation_reason(
                            task_state,
                            fallback=self._localized_text(
                                language,
                                de="Die neue Implementierung gegen den Projekt-Command-Plan validieren.",
                                en="Validate the new implementation against the project command plan.",
                            ),
                            language=language,
                        ),
                    ),
                    (
                        RouteActionName.SUMMARIZE_RESULT,
                        self._localized_text(
                            language,
                            de="Berichten, was erstellt wurde und wie es geprueft wurde.",
                            en="Report what was created and how it was verified.",
                        ),
                    ),
                ]
            )
        elif intent == RouteIntent.UPDATE:
            if target_paths:
                actions.append(
                    (
                        RouteActionName.READ_RELEVANT_FILES,
                        self._pre_edit_reason(
                            strategy,
                            fallback=self._localized_text(
                                language,
                                de="Die aktive Implementierung vor der Aenderung erst lesen.",
                                en="Inspect the active implementation before editing it.",
                            ),
                            language=language,
                        ),
                    )
                )
            else:
                actions.extend(
                    [
                        (
                            RouteActionName.SEARCH_WORKSPACE,
                            self._localized_text(
                                language,
                                de="Vor der Aenderung den relevantesten Implementierungsbereich finden.",
                                en="Find the most relevant implementation area before editing.",
                            ),
                        ),
                        (
                            RouteActionName.READ_RELEVANT_FILES,
                            self._pre_edit_reason(
                                strategy,
                                fallback=self._localized_text(
                                    language,
                                    de="Die wahrscheinlich relevanten Ziele lesen, bevor ich sie aendere.",
                                    en="Inspect the likely targets before changing them.",
                                ),
                                language=language,
                            ),
                        ),
                    ]
                )
            actions.extend(
                [
                    (
                        RouteActionName.UPDATE_ARTIFACT,
                        self._update_reason(
                            strategy,
                            fallback=self._localized_text(
                                language,
                                de="Die fokussierte Aenderung aus dem interpretierten Ziel anwenden.",
                                en="Apply the focused change implied by the interpreted goal.",
                            ),
                            language=language,
                        ),
                    ),
                    (
                        RouteActionName.RUN_VALIDATION,
                        self._validation_reason(
                            task_state,
                            fallback=self._localized_text(
                                language,
                                de="Das aktualisierte Verhalten nach der Aenderung validieren.",
                                en="Validate the updated behavior after the change.",
                            ),
                            language=language,
                        ),
                    ),
                    (
                        RouteActionName.SUMMARIZE_RESULT,
                        self._localized_text(
                            language,
                            de="Aenderung, Annahmen und Validierungsergebnis zusammenfassen.",
                            en="Report the change, assumptions, and verification result.",
                        ),
                    ),
                ]
            )
        elif intent == RouteIntent.DEBUG:
            if target_paths:
                actions.append(
                    (
                        RouteActionName.READ_RELEVANT_FILES,
                        self._localized_text(
                            language,
                            de="Das aktive Artefakt lesen, bevor ich den Fehler reproduziere oder etwas aendere.",
                            en="Inspect the active artifact before reproducing the failure or editing it.",
                        ),
                    )
                )
            actions.extend(
                [
                    (
                        RouteActionName.DIAGNOSE_ISSUE,
                        self._localized_text(
                            language,
                            de="Nutzerbericht und verfuegbare Evidenz erst in konkrete Diagnostik uebersetzen, bevor ich etwas fixe.",
                            en="Translate the user report and available evidence into concrete diagnostics before fixing anything.",
                        ),
                    ),
                ]
            )
            if understanding.intent_category != "test":
                actions.append(
                    (
                        RouteActionName.UPDATE_ARTIFACT,
                        self._localized_text(
                            language,
                            de="Wenn die Diagnostik die Ursache bestaetigt, den kleinsten sicheren Fix anwenden.",
                            en="If diagnostics confirm the cause, apply the smallest safe fix.",
                        ),
                    )
                )
            actions.extend(
                [
                    (
                        RouteActionName.RUN_VALIDATION,
                        self._validation_reason(
                            task_state,
                            fallback=self._localized_text(
                                language,
                                de="Nach Diagnose oder Reparatur die relevanteste Validierung erneut ausfuehren.",
                                en="Rerun the most relevant validation after diagnosis or repair.",
                            ),
                            language=language,
                        ),
                    ),
                    (
                        RouteActionName.SUMMARIZE_RESULT,
                        self._localized_text(
                            language,
                            de="Diagnose, Aenderung und Validierungsergebnis berichten.",
                            en="Report the diagnosis, change, and validation outcome.",
                        ),
                    ),
                ]
            )
        else:
            actions = [
                (
                    RouteActionName.INSPECT_WORKSPACE,
                    self._localized_text(
                        language,
                        de="Repository-Kontext sammeln, bevor ich den konkreten naechsten Schritt waehle.",
                        en="Gather repository context before choosing a concrete next step.",
                    ),
                ),
                (
                    RouteActionName.SUMMARIZE_RESULT,
                    self._localized_text(
                        language,
                        de="Die Erkenntnisse und verbleibende Unklarheiten zurueckgeben.",
                        en="Return the findings and any remaining ambiguity.",
                    ),
                ),
            ]

        return [
            {"step": index, "action": action, "reason": reason}
            for index, (action, reason) in enumerate(actions, start=1)
        ]

    def _route_constraints(
        self,
        understanding: TaskUnderstanding,
        *,
        task_state: TaskState | None = None,
    ) -> list[str]:
        constraints = list(understanding.constraints[:6])
        if task_state is None:
            return constraints
        for item in [
            f"Execution strategy: {task_state.execution_strategy}" if task_state.execution_strategy else "",
            f"Next best action: {task_state.next_best_action}" if task_state.next_best_action else "",
        ]:
            text = str(item or "").strip()
            if text and text not in constraints:
                constraints.append(text)
        return constraints[:6]

    def _pre_edit_reason(
        self,
        strategy: str | None,
        *,
        fallback: str,
        language: str | None = None,
    ) -> str:
        language = language or "en"
        if strategy == "rollback_correction":
            return self._localized_text(
                language,
                de="Die zuvor geaenderte Stelle erst ansehen, bevor ich sie eingrenze oder korrigiere.",
                en="Inspect the previously changed surface before narrowing or correcting it.",
            )
        if strategy == "refactor":
            return self._localized_text(
                language,
                de="Die aktive Implementierung zuerst ansehen, damit das Verhalten waehrend des Refactors stabil bleibt.",
                en="Inspect the active implementation first so behavior stays stable during the refactor.",
            )
        if strategy == "hardening":
            return self._localized_text(
                language,
                de="Die aktive Implementierung ansehen, bevor ich Sicherheit oder Zuverlaessigkeit verschaerfe.",
                en="Inspect the active implementation before tightening safety or reliability.",
            )
        return fallback

    def _update_reason(
        self,
        strategy: str | None,
        *,
        fallback: str,
        language: str | None = None,
    ) -> str:
        language = language or "en"
        if strategy == "rollback_correction":
            return self._localized_text(
                language,
                de="Nur den noetigen Teil der vorherigen Interpretation oder Aenderung zuruecknehmen oder eingrenzen.",
                en="Revert or narrow only the necessary part of the previous interpretation or change.",
            )
        if strategy == "refactor":
            return self._localized_text(
                language,
                de="Einen inkrementellen Refactor anwenden und dabei das Verhalten erhalten.",
                en="Apply an incremental refactor while preserving behavior.",
            )
        if strategy == "hardening":
            return self._localized_text(
                language,
                de="Die wirksamste Haertungs-Aenderung ohne breite Umbauten anwenden.",
                en="Apply the highest-value hardening change without broad rewrites.",
            )
        return fallback

    def _validation_reason(
        self,
        task_state: TaskState | None,
        *,
        fallback: str,
        language: str | None = None,
    ) -> str:
        language = language or "en"
        if task_state is None or not task_state.verification_target:
            return fallback
        return self._localized_text(
            language,
            de=f"Gegen das festgelegte Ziel verifizieren: {task_state.verification_target}",
            en=f"Verify against the committed target: {task_state.verification_target}",
        )

    def _should_clarify(
        self,
        understanding: TaskUnderstanding,
        *,
        intent: RouteIntent,
        target_paths: list[str],
        target_name: str | None,
    ) -> bool:
        if understanding.needs_clarification or understanding.recommended_mode == "clarify":
            return True
        if understanding.confidence < 0.4:
            return True
        if understanding.risk_level == "high" and understanding.confidence < self.HIGH_CONFIDENCE:
            return True
        if intent == RouteIntent.UNKNOWN:
            return True
        if understanding.missing_info and understanding.confidence < self.MEDIUM_CONFIDENCE:
            return True
        if intent == RouteIntent.CREATE:
            return not self._can_default_create_safely(
                understanding,
                target_paths=target_paths,
                target_name=target_name,
            )
        if intent in {RouteIntent.UPDATE, RouteIntent.DEBUG} and not (
            target_paths or understanding.target_artifacts or understanding.supplied_evidence
        ):
            return understanding.confidence < self.HIGH_CONFIDENCE
        return False

    def _safe_to_execute(
        self,
        understanding: TaskUnderstanding,
        *,
        intent: RouteIntent,
        target_paths: list[str],
        target_name: str | None,
    ) -> bool:
        if intent == RouteIntent.UNKNOWN:
            return False
        if understanding.risk_level == "high" and understanding.confidence < self.HIGH_CONFIDENCE:
            return False
        if understanding.confidence < self.MEDIUM_CONFIDENCE:
            return False
        if intent == RouteIntent.CREATE:
            return self._can_default_create_safely(
                understanding,
                target_paths=target_paths,
                target_name=target_name,
            )
        if intent in {RouteIntent.UPDATE, RouteIntent.DEBUG} and not (
            target_paths or understanding.target_artifacts or understanding.supplied_evidence
        ):
            return False
        return True

    def _repo_context_needed(self, intent: RouteIntent) -> bool:
        return intent not in {RouteIntent.EXPLAIN, RouteIntent.PLAN}

    def _target_paths(
        self,
        understanding: TaskUnderstanding,
        *,
        session: SessionState | None,
        intent: RouteIntent,
    ) -> list[str]:
        if classify_conversation_request(understanding.original_request) is not None:
            return []
        candidate_artifacts = [
            item
            for item in understanding.target_artifacts
            if item.path
            and (
                intent != RouteIntent.CREATE
                or item.role in {"primary_target", "validation_target", "supporting_context"}
            )
        ]
        candidates: list[str] = [str(item.path) for item in candidate_artifacts if item.path]
        if intent == RouteIntent.CREATE:
            explicit_request_paths = self._explicit_request_paths(understanding.original_request)
            if explicit_request_paths:
                candidates = self._merge_create_candidates_with_request_paths(
                    candidate_artifacts,
                    explicit_request_paths,
                )
        if candidates:
            return self._unique_paths(candidates)[:8]
        if intent == RouteIntent.CREATE:
            return []
        if session is not None:
            candidates.extend(session.candidate_files[:8])
            if session.follow_up_context is not None:
                candidates.extend(session.follow_up_context.target_paths[:8])
                candidates.extend(session.follow_up_context.changed_files[:8])
                candidates.extend(session.follow_up_context.read_files[:8])
        return self._unique_paths(candidates)[:8]

    def _explicit_request_paths(self, request: str) -> list[str]:
        paths: list[str] = []
        for match in _EXPLICIT_PATH_RE.finditer(str(request or "")):
            candidate = str(match.group(1) or "").lstrip("./")
            if candidate and candidate not in paths:
                paths.append(candidate)
        return paths[:8]

    def _merge_create_candidates_with_request_paths(
        self,
        candidate_artifacts: list[TaskArtifact],
        explicit_request_paths: list[str],
    ) -> list[str]:
        explicit_by_key = {
            self._path_merge_key(candidate): candidate
            for candidate in explicit_request_paths
            if candidate
        }
        merged: list[str] = []
        seen: set[str] = set()
        insert_after_primary = 0
        for artifact in candidate_artifacts:
            candidate = str(artifact.path or "").strip()
            if not candidate:
                continue
            candidate_key = self._path_merge_key(candidate)
            if candidate_key in seen:
                continue
            merged.append(explicit_by_key.pop(candidate_key, candidate))
            seen.add(candidate_key)
            if artifact.role == "primary_target":
                insert_after_primary = len(merged)
        for candidate in explicit_request_paths:
            candidate_key = self._path_merge_key(candidate)
            if not candidate or candidate_key in seen:
                continue
            merged.insert(insert_after_primary, candidate)
            seen.add(candidate_key)
            insert_after_primary += 1
        return merged

    def _path_merge_key(self, value: str | None) -> str:
        return str(value or "").strip().replace("\\", "/").lower()

    def _target_name(
        self,
        understanding: TaskUnderstanding,
        target_paths: list[str],
    ) -> str | None:
        if target_paths:
            return target_paths[0]
        create_intent = self._route_intent(understanding) == RouteIntent.CREATE
        prioritized_candidates = self._prioritized_target_name_candidates(
            understanding.target_artifacts,
            primary_only=create_intent,
        )
        if prioritized_candidates:
            return prioritized_candidates[0]
        if self._route_intent(understanding) == RouteIntent.CREATE:
            return infer_artifact_name_hint(
                understanding.original_request,
                understanding.interpreted_goal,
            )
        return None

    def _prioritized_target_name_candidates(
        self,
        artifacts: list[TaskArtifact],
        *,
        primary_only: bool,
    ) -> list[str]:
        scored: list[tuple[tuple[int, int, int, float, int], str]] = []
        seen: set[str] = set()
        for artifact_index, artifact in enumerate(artifacts):
            if primary_only and artifact.role != "primary_target":
                continue
            for candidate_index, candidate in enumerate((artifact.path, artifact.name)):
                text = str(candidate or "").strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                score = (
                    0 if self._looks_like_explicit_target_name(text) else 1,
                    candidate_index,
                    0 if str(artifact.kind or "").strip().lower() == "file" else 1,
                    -float(artifact.confidence or 0.0),
                    artifact_index,
                )
                scored.append((score, text))
        scored.sort(key=lambda item: item[0])
        return [text for _, text in scored]

    def _looks_like_explicit_target_name(self, value: str) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        path = Path(text)
        if path.suffix and path.name != path.suffix:
            return True
        return "/" in text or "\\" in text

    def _search_terms(
        self,
        understanding: TaskUnderstanding,
        target_paths: list[str],
        target_name: str | None,
    ) -> list[str]:
        terms: list[str] = []
        if target_name:
            terms.append(target_name)
        for artifact in understanding.target_artifacts:
            if artifact.name:
                terms.append(artifact.name)
            if artifact.path:
                terms.append(artifact.path)
        terms.append(understanding.interpreted_goal)
        terms.extend(understanding.subgoals[:3])
        terms.extend(target_paths[:2])
        return self._unique_strings(terms)[:6]

    def _relevant_extensions(
        self,
        understanding: TaskUnderstanding,
        target_paths: list[str],
    ) -> list[str]:
        extensions: list[str] = []
        for artifact in understanding.target_artifacts:
            extensions.extend(self._artifact_extensions(artifact))
        for path in target_paths:
            suffix = Path(path).suffix.strip()
            if suffix and suffix not in extensions:
                extensions.append(suffix)
        inferred = infer_requested_extension(
            understanding.original_request,
            understanding.interpreted_goal,
            *understanding.constraints,
            *understanding.relevant_context,
        )
        if inferred and inferred not in extensions:
            extensions.append(inferred)
        return extensions[:6]

    def _artifact_extensions(self, artifact: TaskArtifact) -> list[str]:
        values: list[str] = []
        if artifact.path:
            suffix = Path(artifact.path).suffix.strip()
            if suffix:
                values.append(suffix)
        if artifact.kind and artifact.kind.startswith(".") and artifact.kind not in values:
            values.append(artifact.kind)
        return values

    def _can_default_create_safely(
        self,
        understanding: TaskUnderstanding,
        *,
        target_paths: list[str],
        target_name: str | None,
    ) -> bool:
        if target_paths or understanding.target_artifacts or understanding.supplied_evidence:
            return understanding.confidence >= self.MEDIUM_CONFIDENCE
        if understanding.risk_level == "high" or understanding.confidence < self.MEDIUM_CONFIDENCE:
            return False
        inferred_extension = infer_requested_extension(
            understanding.original_request,
            understanding.interpreted_goal,
            *understanding.constraints,
            *understanding.relevant_context,
        )
        scope_tokens = infer_scope_tokens(
            understanding.original_request,
            understanding.interpreted_goal,
        )
        if target_name:
            return True
        if inferred_extension and scope_tokens:
            return True
        return len(scope_tokens) >= 2

    def _clarification_reason(
        self,
        understanding: TaskUnderstanding,
        *,
        intent: RouteIntent,
        target_paths: list[str],
        target_name: str | None,
        language: str | None = None,
    ) -> str:
        language = language or self._language_for_text(understanding.original_request)
        if understanding.risk_level == "high":
            return self._localized_text(
                language,
                de="Die Anfrage traegt ein hohes Ausfuehrungsrisiko, ohne dass das Ziel sicher genug ist.",
                en="The request carries high execution risk without enough target certainty.",
            )
        if intent == RouteIntent.CREATE and not self._can_default_create_safely(
            understanding,
            target_paths=target_paths,
            target_name=target_name,
        ):
            return self._localized_text(
                language,
                de="Der Anfrage fehlen noch genug Implementierungsdetails, um ein sinnvolles Default-Artefakt sicher zu waehlen.",
                en="The request still lacks enough implementation detail to choose a sensible default artifact safely.",
            )
        if intent in {RouteIntent.UPDATE, RouteIntent.DEBUG} and not target_paths:
            return self._localized_text(
                language,
                de="Die wahrscheinliche Aufgabenart ist klar, aber das konkrete Zielartefakt ist noch zu unsicher.",
                en="The likely task type is clear, but the concrete target artifact is still too uncertain.",
            )
        if understanding.missing_info:
            return self._localized_text(
                language,
                de="Fuer einen sicheren naechsten Schritt fehlen noch kritische Scope-Informationen.",
                en="Critical scope information is still missing for a safe next step.",
            )
        return self._localized_text(
            language,
            de="Das Aufgabenverstaendnis ist fuer eine sichere Ausfuehrung noch zu mehrdeutig.",
            en="The task understanding is still too ambiguous for safe execution.",
        )

    def _clarification_questions(self, understanding: TaskUnderstanding) -> list[str]:
        if understanding.clarification_questions:
            return understanding.clarification_questions[:3]
        if understanding.missing_info:
            return [understanding.missing_info[0]]
        return ["Welchen konkreten Bereich oder welches Artefakt soll ich als naechstes bearbeiten?"]

    def _unique_paths(self, values: list[str | None]) -> list[str]:
        unique: list[str] = []
        for raw in values:
            text = str(raw or "").strip()
            if not text or text in unique:
                continue
            unique.append(text)
        return unique

    def _unique_strings(self, values: list[str | None]) -> list[str]:
        unique: list[str] = []
        for raw in values:
            text = str(raw or "").strip()
            if not text or text in unique:
                continue
            unique.append(text)
        return unique

    def _request_language(
        self,
        understanding: TaskUnderstanding,
        *,
        task_state: TaskState | None = None,
    ) -> str:
        if task_state is not None and str(task_state.latest_user_turn or "").strip():
            return self._language_for_text(task_state.latest_user_turn)
        return self._language_for_text(understanding.original_request)

    def _language_for_text(self, text: str | None) -> str:
        normalized = str(text or "").lower()
        german_markers = (
            " bitte ",
            " ich ",
            " pruef",
            "prüf",
            " analys",
            " erklaer",
            "erklär",
            " datei",
            " fehler",
            " repository",
            " repo ",
            " mach",
            " bau",
            " jetzt",
        )
        padded = f" {normalized} "
        if any(marker in padded for marker in german_markers) or any(char in normalized for char in "äöüß"):
            return "de"
        return "en"

    def _localized_text(self, language: str, *, de: str, en: str) -> str:
        return de if language == "de" else en

    def _log(self, event: str, **payload: Any) -> None:
        if self.logger is None:
            return
        self.logger.log_event(event, **payload)
