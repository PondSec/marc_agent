from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.memory import ENTRYPOINT_NAMES, MANIFEST_FILES, RepoMemoryStore
from agent.models import (
    ConversationMemoryEntry,
    EpisodicMemoryEntry,
    FailureMemoryEntry,
    MemoryProvenance,
    MemoryRetrievalResult,
    MemorySummary,
    ProjectMemoryEntry,
    RememberedFact,
    RetrievedMemoryItem,
    RetrievalRequest,
    SessionState,
    WorkingMemoryEntry,
    utc_now,
)
from agent.semantic_defaults import classify_conversation_request, normalize_text, prioritized_focus_terms
from config.settings import AppConfig
from runtime.workspace import WorkspaceManager


PERSISTENT_MEMORY_TYPES = ("episodic", "project", "failure", "conversation")
ENTRY_MODEL_BY_TYPE = {
    "episodic": EpisodicMemoryEntry,
    "project": ProjectMemoryEntry,
    "failure": FailureMemoryEntry,
    "conversation": ConversationMemoryEntry,
}
RETENTION_DAYS = {
    "working": 0,
    "episodic": 120,
    "project": 365,
    "failure": 180,
    "conversation": 45,
}
TYPE_USE_CASE_BONUS = {
    "task_continuation": {"episodic": 0.08, "project": 0.12, "failure": 0.04, "conversation": 0.04},
    "repair_assistance": {"failure": 0.22, "episodic": 0.08, "project": 0.06, "conversation": 0.0},
    "similar_task_lookup": {"episodic": 0.18, "project": 0.06, "failure": 0.06, "conversation": 0.04},
    "project_context": {"project": 0.24, "episodic": 0.04, "failure": 0.02, "conversation": 0.02},
    "user_recall": {"episodic": 0.16, "conversation": 0.14, "project": 0.06, "failure": 0.02},
}

_QUESTION_WORD_TOKENS = {"how", "wann", "warum", "was", "what", "when", "wer", "where", "wie", "wieso", "wo", "who", "why"}
_FIRST_PERSON_TOKENS = {"i", "ich", "me", "mein", "meine", "mich", "mir", "my"}
_SECOND_PERSON_TOKENS = {"dein", "deine", "dich", "dir", "du", "you", "your"}
_IDENTITY_QUERY_TOKENS = {"bin", "heisse", "heisst", "identity", "identitaet", "name", "person", "profil", "profile", "who", "wer"}
_PROFILE_QUERY_PHRASES = ("about me", "ueber mich", "uber mich")
_PERSONAL_QUERY_STOPWORDS = {
    *_QUESTION_WORD_TOKENS,
    *_FIRST_PERSON_TOKENS,
    *_SECOND_PERSON_TOKENS,
    "am",
    "bin",
    "bist",
    "bist",
    "das",
    "denn",
    "der",
    "die",
    "du",
    "ein",
    "eine",
    "einem",
    "einer",
    "erinnern",
    "erinnere",
    "from",
    "fuer",
    "für",
    "hab",
    "habe",
    "haben",
    "heisse",
    "heisst",
    "identity",
    "identitaet",
    "ist",
    "mein",
    "meine",
    "mich",
    "mir",
    "my",
    "noch",
    "profil",
    "profile",
    "remember",
    "seid",
    "sind",
    "ueber",
    "uber",
    "von",
    "weiss",
    "weisst",
    "weißt",
    "werde",
    "wirst",
}
_PERSONAL_RECALL_EXCLUDE_TOKENS = {
    "api",
    "app",
    "auth",
    "backend",
    "bug",
    "build",
    "cli",
    "code",
    "config",
    "css",
    "datei",
    "diff",
    "feature",
    "file",
    "frontend",
    "html",
    "index",
    "js",
    "modul",
    "module",
    "projekt",
    "python",
    "repo",
    "route",
    "script",
    "server",
    "stacktrace",
    "test",
    "ui",
    "workspace",
}
_HISTORY_TEMPORAL_TOKENS = {"already", "bereits", "bisher", "earlier", "frueher", "history", "last", "letzte", "letzten", "letzter", "mal", "previous", "schon", "status", "stand", "vorher"}
_HISTORY_CONTINUITY_TOKENS = {"again", "gebaut", "did", "done", "gemacht", "have", "haben", "implemented", "schon", "we", "wir"}
_MEMORY_DIRECTIVE_PATTERNS = (
    re.compile(r"\b(?:merk(?:e)?\s+dir|speicher(?:e)?|behalte?)\b(?:\s+bitte|\s+mal)?(?:\s+dass)?\s*(?P<fact>.+)", flags=re.IGNORECASE),
    re.compile(r"\bremember\b(?:\s+this|\s+that)?\s*(?P<fact>.+)", flags=re.IGNORECASE),
)
_USER_NAME_PATTERNS = (
    re.compile(r"\b(?:ich\s+heisse|mein\s+name\s+ist|my\s+name\s+is)\s+(?P<value>[^,.!?;\n]+)", flags=re.IGNORECASE),
    re.compile(r"\b(?:dass\s+)?ich\s+(?P<value>[^,.!?;\n]+?)\s+heisse\b", flags=re.IGNORECASE),
    re.compile(r"\b(?:i\s+am|i'm)\s+(?P<value>[^,.!?;\n]+)", flags=re.IGNORECASE),
)


class AgentMemoryStore(RepoMemoryStore):
    def __init__(self, config: AppConfig, workspace: WorkspaceManager):
        super().__init__(config, workspace)
        self.layered_root = self.config.memory_dir_path / "layered"
        self.entries_root = self.layered_root / "entries"
        self.index_path = self.layered_root / "index.json"
        self.project_id = self._resolve_project_id()
        self._ensure_layout()
        self._index = self._load_index()
        self._prune_expired_entries()

    def refresh_session_memory(self, task: str, session: SessionState) -> None:
        resolved_project_id = self._resolve_project_id(session.workspace_snapshot)
        self.project_id = resolved_project_id
        session.project_id = resolved_project_id
        session.working_memory = self.build_working_memory(session)
        request = self.build_retrieval_request(task, session)
        current = session.memory_context
        if current is not None and self._retrieval_request_signature(current.request) == self._retrieval_request_signature(request):
            return
        repo_files, repo_symbols, repo_hints = self._repo_map_signal_bundle(
            session.workspace_snapshot,
            request,
        )
        if not self._should_retrieve_persistent_memory(request):
            summary = "No relevant persistent memory selected."
            if repo_hints:
                summary = self._render_repo_hint_summary(repo_hints, request.summary_budget_chars)
            session.memory_context = MemoryRetrievalResult(
                request=request,
                summary=summary,
                suggested_files=repo_files[:8],
                suggested_symbols=repo_symbols[:8],
                repo_map_hints=repo_hints[:6],
            )
            return
        result = self.retrieve(request)
        merged_files = self._unique_strings([*repo_files, *result.suggested_files])[:8]
        merged_symbols = self._unique_strings([*repo_symbols, *result.suggested_symbols])[:8]
        merged_hints = self._unique_strings([*repo_hints, *result.repo_map_hints])[:6]
        summary = result.summary
        if merged_hints:
            summary = self._merge_repo_hints_into_summary(summary, merged_hints, request.summary_budget_chars)
        session.memory_context = result.model_copy(
            update={
                "summary": summary,
                "suggested_files": merged_files,
                "suggested_symbols": merged_symbols,
                "repo_map_hints": merged_hints,
            }
        )

    def persist_session_memory(self, session: SessionState) -> None:
        resolved_project_id = self._resolve_project_id(session.workspace_snapshot)
        self.project_id = resolved_project_id
        session.project_id = resolved_project_id
        project_entry = self._build_project_entry(session)
        if project_entry is not None:
            self.upsert_entry(project_entry)
        episodic_entry = self._build_episodic_entry(session)
        if episodic_entry is not None:
            self.upsert_entry(episodic_entry)
        for entry in self._build_failure_entries(session):
            self.upsert_entry(entry)
        conversation_entry = self._build_conversation_entry(session)
        if conversation_entry is not None:
            self.upsert_entry(conversation_entry)

    def list_entries(
        self,
        memory_type: str | None = None,
        *,
        project_id: str | None = None,
    ) -> list[EpisodicMemoryEntry | ProjectMemoryEntry | FailureMemoryEntry | ConversationMemoryEntry]:
        if memory_type is not None and memory_type not in ENTRY_MODEL_BY_TYPE:
            return []
        candidate_ids: list[str]
        if project_id:
            candidate_ids = list(self._index.get("by_project", {}).get(project_id, []))
        else:
            candidate_ids = list(self._index.get("entries", {}).keys())
        entries: list[EpisodicMemoryEntry | ProjectMemoryEntry | FailureMemoryEntry | ConversationMemoryEntry] = []
        for entry_id in candidate_ids:
            metadata = self._index["entries"].get(entry_id, {})
            if memory_type is not None and metadata.get("memory_type") != memory_type:
                continue
            entry = self._load_entry(entry_id)
            if entry is None:
                continue
            entries.append(entry)
        entries.sort(key=lambda item: getattr(item, "updated_at", ""), reverse=True)
        return entries

    def build_working_memory(self, session: SessionState) -> WorkingMemoryEntry:
        task_state = session.task_state
        understanding = session.task_understanding
        route = session.router_result
        repair_context = session.active_repair_context
        repair_brief = getattr(repair_context, "repair_brief", None)
        current_goal = (
            str(getattr(task_state, "active_goal", "") or "").strip()
            or str(getattr(task_state, "root_goal", "") or "").strip()
            or str(getattr(understanding, "interpreted_goal", "") or "").strip()
            or str(session.task or "").strip()
        )
        current_subtask = (
            next((item.step for item in session.plan if item.status == "in_progress"), None)
            or str(getattr(task_state, "next_action", "") or "").strip()
            or str(getattr(task_state, "next_best_action", "") or "").strip()
            or None
        )
        primary_target = next(
            (
                str(getattr(item, "path", "") or "").strip()
                for item in (getattr(task_state, "target_artifacts", []) or [])
                if str(getattr(item, "path", "") or "").strip()
            ),
            None,
        )
        if not primary_target and session.candidate_files:
            primary_target = session.candidate_files[0]
        verification_target = str(getattr(task_state, "verification_target", "") or "").strip() or None
        active_constraints = self._unique_strings(
            [
                *(getattr(task_state, "constraints", []) or []),
                *(getattr(understanding, "constraints", []) or []),
                *(getattr(route.entities, "constraints", []) if route is not None else []),
            ]
        )[:6]
        recent_attempts = [
            f"{item.tool_name}: {self._trim_text(item.summary, 120)}"
            for item in session.tool_calls[-4:]
        ]
        recent_successes = [
            self._trim_text(item.summary, 120)
            for item in session.tool_calls
            if item.success
        ][-4:]
        recent_failures = [
            self._trim_text(item.summary, 120)
            for item in session.diagnostics[-4:]
        ]
        repo_files, repo_symbols, _ = self._repo_map_signal_bundle(
            session.workspace_snapshot,
            RetrievalRequest(
                query=str(session.task or "").strip(),
                use_case=self._infer_use_case(str(session.task or ""), session),
                project_id=session.project_id or self.project_id,
                workspace_root=session.workspace_root,
                session_id=session.id,
                target_paths=list(session.candidate_files[:8]),
                error_terms=self._request_error_terms(session),
                include_types=["project"],
                max_hits=0,
                max_per_type=0,
                summary_budget_chars=240,
            ),
        )
        relevant_files = self._unique_strings(
            [
                *[item.path for item in session.changed_files[-8:]],
                *(getattr(repair_context, "artifact_paths", []) if repair_context is not None else []),
                *(getattr(repair_context, "file_hints", []) if repair_context is not None else []),
                *repo_files[:6],
                *session.candidate_files[:12],
            ]
        )[:8]
        relevant_symbols = self._unique_strings(
            [
                *list(getattr(repair_brief, "implicated_symbols", []) or []),
                *repo_symbols[:6],
            ]
        )[:6]
        last_effective_strategy = next(
            (
                item.strategy
                for item in reversed(session.repair_history)
                if item.result == "mutation_planned"
            ),
            None,
        )
        last_ineffective_strategy = next(
            (
                item.strategy
                for item in reversed(session.repair_history)
                if item.result in {"no_effective_change", "generation_failed", "blocked"}
            ),
            None,
        )
        active_failure_signature = (
            str(getattr(repair_brief, "failure_signature", "") or "").strip()
            or str(getattr(repair_context, "failure_summary", "") or "").strip()
            or None
        )
        compact_state_summary = self._trim_text(
            " | ".join(
                part
                for part in [
                    current_goal,
                    f"current_subtask={current_subtask}" if current_subtask else "",
                    f"primary_target={primary_target}" if primary_target else "",
                    f"verification_target={verification_target}" if verification_target else "",
                    f"failure={active_failure_signature}" if active_failure_signature else "",
                ]
                if part
            ),
            340,
        )
        return WorkingMemoryEntry(
            project_id=session.project_id or self.project_id,
            workspace_root=session.workspace_root,
            session_id=session.id,
            summary=MemorySummary(
                title="Active run context",
                summary=compact_state_summary or self._trim_text(session.task, 220),
                key_points=[self._trim_text(item, 120) for item in recent_failures[:2] + recent_successes[:2]],
                why_relevant="Continuity for the active task without replaying raw history.",
            ),
            provenance=MemoryProvenance(
                source_type="session",
                session_id=session.id,
                project_id=session.project_id or self.project_id,
                workspace_root=session.workspace_root,
                detail="Derived from current task state, diagnostics, and recent tool activity.",
                file_paths=relevant_files[:6],
            ),
            tags=self._unique_strings(
                [
                    str(getattr(task_state, "current_user_intent", "") or "").strip(),
                    str(getattr(task_state, "execution_strategy", "") or "").strip(),
                    str(getattr(route, "intent", "") or "").strip(),
                ]
            ),
            file_paths=relevant_files,
            symbol_names=relevant_symbols,
            retention="transient",
            ttl_days=0,
            dedupe_key=f"working:{session.id}",
            importance=0.95,
            confidence=max(float(getattr(task_state, "confidence", 0.0) or 0.0), 0.5),
            current_goal=current_goal or None,
            current_subtask=current_subtask,
            primary_target=primary_target,
            verification_target=verification_target,
            request_excerpt=str(getattr(task_state, "request_excerpt", "") or "").strip() or None,
            request_requirements=list(getattr(task_state, "request_requirements", [])[:6] if task_state is not None else []),
            request_chunks=list(getattr(task_state, "request_chunks", [])[:4] if task_state is not None else []),
            request_digest=getattr(task_state, "request_digest", None),
            active_constraints=active_constraints,
            active_failure_signature=active_failure_signature,
            recent_attempts=[self._trim_text(item, 140) for item in recent_attempts],
            recent_successes=recent_successes,
            recent_failures=recent_failures,
            relevant_files=relevant_files,
            relevant_symbols=relevant_symbols,
            last_effective_strategy=last_effective_strategy,
            last_ineffective_strategy=last_ineffective_strategy,
            compact_state_summary=compact_state_summary,
        )

    def build_retrieval_request(self, task: str, session: SessionState) -> RetrievalRequest:
        working = session.working_memory or self.build_working_memory(session)
        use_case = self._infer_use_case(task, session)
        recall_subject, recall_attributes = self._personal_recall_focus(task)
        fresh_task_bootstrap = self._is_fresh_task_bootstrap(session)
        include_types = ["episodic", "project", "failure", "conversation"]
        if fresh_task_bootstrap and use_case != "user_recall":
            include_types = ["project"]
        elif use_case == "repair_assistance":
            include_types = ["failure", "episodic", "project", "conversation"]
        elif use_case == "project_context":
            include_types = ["project", "episodic", "failure", "conversation"]
        elif use_case == "user_recall":
            include_types = ["conversation"] if recall_subject else ["episodic", "conversation", "project", "failure"]
        target_paths = self._unique_strings(
            [
                *(
                    list(getattr(getattr(working, "request_digest", None), "explicit_paths", []) or [])
                    if working is not None
                    else []
                ),
                *(working.relevant_files if working is not None else []),
                *session.candidate_files[:8],
                *[item.path for item in session.changed_files[-6:]],
            ]
        )[:8]
        return RetrievalRequest(
            query=str(task or "").strip(),
            use_case=use_case,
            project_id=session.project_id or self.project_id,
            workspace_root=session.workspace_root,
            session_id=session.id,
            recall_subject=recall_subject,
            recall_attributes=list(recall_attributes),
            target_paths=target_paths,
            symbol_names=self._unique_strings(
                [
                    *(list(getattr(getattr(working, "request_digest", None), "explicit_symbols", []) or []) if working is not None else []),
                    *(working.relevant_symbols[:6] if working is not None else []),
                ]
            )[:6],
            error_terms=self._request_error_terms(session),
            failure_signature=working.active_failure_signature if working is not None else None,
            current_goal=working.current_goal if working is not None else None,
            current_subtask=working.current_subtask if working is not None else None,
            changed_files=[item.path for item in session.changed_files[-6:]],
            include_types=include_types,
            max_hits=3 if recall_subject else 4 if fresh_task_bootstrap else 6,
            max_per_type=3 if recall_subject else 1 if fresh_task_bootstrap else 2,
            summary_budget_chars=420 if recall_subject else 520 if fresh_task_bootstrap else 900,
            allow_cross_project=use_case == "user_recall",
        )

    def retrieve(self, request: RetrievalRequest) -> MemoryRetrievalResult:
        started = time.perf_counter()
        project_id = request.project_id or self.project_id
        candidate_ids = self._candidate_ids_for_request(request)
        scored: list[RetrievedMemoryItem] = []
        for entry_id in candidate_ids:
            metadata = self._index["entries"].get(entry_id)
            if not metadata:
                continue
            memory_type = str(metadata.get("memory_type") or "")
            if memory_type not in request.include_types:
                continue
            if not request.allow_cross_project and project_id and metadata.get("project_id") != project_id:
                continue
            item = self._score_entry(request, metadata)
            if item is not None:
                scored.append(item)
        scored.sort(key=lambda item: (item.score, item.exact_entity_relevance, item.project_relevance), reverse=True)
        selected = self._select_with_budgets(scored, request)
        recall_brief = self._render_user_recall_brief(selected, request) if request.use_case == "user_recall" else ""
        summary = recall_brief or self._render_result_summary(selected, request.summary_budget_chars)
        suggested_files = self._unique_strings(
            [path for item in selected for path in item.file_paths]
        )[:8]
        suggested_symbols = self._unique_strings(
            [name for item in selected for name in item.symbol_names]
        )[:8]
        related_sessions = self._unique_strings(
            [str(item.session_id or "").strip() for item in selected if str(item.session_id or "").strip()]
        )[:6]
        related_projects = self._unique_strings(
            [str(item.project_id or "").strip() for item in selected if str(item.project_id or "").strip()]
        )[:4]
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        selected_count = max(len(selected), 1)
        stale_count = sum(1 for item in selected if item.is_stale)
        useful_count = sum(
            1
            for item in selected
            if item.score >= 0.52 or item.exact_entity_relevance >= 0.45 or item.project_relevance >= 0.95
        )
        self._mark_entries_accessed(selected)
        return MemoryRetrievalResult(
            request=request,
            selected=selected,
            summary=summary,
            recall_brief=recall_brief,
            suggested_files=suggested_files,
            suggested_symbols=suggested_symbols,
            repo_map_hints=[],
            related_sessions=related_sessions,
            related_projects=related_projects,
            total_candidates=len(candidate_ids),
            total_hits=len(scored),
            hit_count_by_type=self._hit_count_by_type(selected),
            latency_ms=latency_ms,
            prompt_char_cost=len(summary),
            duplicate_rate=round(sum(1 for item in selected if item.duplicate_count > 0) / selected_count, 3),
            stale_recall_rate=round(stale_count / selected_count, 3),
            useful_recall_rate=round(useful_count / selected_count, 3),
        )

    def upsert_entry(
        self,
        entry: EpisodicMemoryEntry | ProjectMemoryEntry | FailureMemoryEntry | ConversationMemoryEntry,
    ) -> EpisodicMemoryEntry | ProjectMemoryEntry | FailureMemoryEntry | ConversationMemoryEntry:
        existing_id = None
        dedupe_key = str(entry.dedupe_key or "").strip()
        if dedupe_key:
            existing_id = self._index.get("by_dedupe", {}).get(dedupe_key)
        if existing_id:
            existing = self._load_entry(existing_id)
            if existing is not None:
                entry = self._merge_entries(existing, entry)
        self._write_entry(entry)
        return entry

    def _build_episodic_entry(self, session: SessionState) -> EpisodicMemoryEntry | None:
        if not str(session.task or "").strip():
            return None
        changed_files = [item.path for item in session.changed_files[-10:]]
        what_worked = self._unique_strings(
            [
                *[
                    self._trim_text(item.summary or item.command, 160)
                    for item in session.validation_runs[-8:]
                    if item.status == "passed"
                ],
                *[
                    item.reason
                    for item in session.repair_history[-8:]
                    if item.result == "mutation_planned"
                ],
            ]
        )[:4]
        what_failed = self._unique_strings(
            [
                *[self._trim_text(item.summary, 160) for item in session.diagnostics[-8:]],
                *[
                    item.reason
                    for item in session.repair_history[-8:]
                    if item.result in {"no_effective_change", "generation_failed", "blocked"}
                ],
            ]
        )[:5]
        failure_signatures = self._unique_strings(
            [
                *[
                    str(item.failure_signature or "").strip()
                    for item in session.repair_history[-8:]
                    if str(item.failure_signature or "").strip()
                ],
                str(
                    getattr(getattr(session.active_repair_context, "repair_brief", None), "failure_signature", "")
                    or ""
                ).strip(),
            ]
        )
        problem_type = (
            str(getattr(session.task_state, "current_user_intent", "") or "").strip()
            or str(getattr(session.router_result, "intent", "") or "").strip()
            or "unknown"
        )
        final_outcome = (
            str(session.report.summary if session.report is not None else "").strip()
            or self._trim_text(session.final_response or "", 220)
            or self._trim_text(session.task, 220)
        )
        result = "blocked" if session.blockers else session.status
        if result in {"queued", "running"}:
            result = "partial"
        return EpisodicMemoryEntry(
            project_id=session.project_id or self.project_id,
            workspace_root=session.workspace_root,
            session_id=session.id,
            summary=MemorySummary(
                title=self._trim_text(getattr(session.task_state, "active_goal", "") or session.task, 90),
                summary=self._trim_text(final_outcome, 220),
                key_points=[*what_worked[:2], *what_failed[:2]],
                why_relevant="Past session outcome that may help with similar work or recall questions.",
            ),
            provenance=MemoryProvenance(
                source_type="report",
                session_id=session.id,
                project_id=session.project_id or self.project_id,
                workspace_root=session.workspace_root,
                detail="Derived from the final session outcome, repair history, and validation evidence.",
                file_paths=changed_files[:6],
            ),
            tags=self._unique_strings(
                [
                    problem_type,
                    session.status,
                    session.validation_status,
                    str(getattr(session.task_state, "execution_strategy", "") or "").strip(),
                ]
            ),
            file_paths=changed_files,
            symbol_names=list(session.working_memory.relevant_symbols[:6] if session.working_memory else []),
            retention="medium",
            ttl_days=RETENTION_DAYS["episodic"],
            importance=0.8 if session.status == "completed" else 0.62,
            confidence=max(float(getattr(session.task_state, "confidence", 0.0) or 0.0), 0.45),
            dedupe_key=self._episodic_dedupe_key(
                session.project_id or self.project_id,
                problem_type or "unknown",
                final_outcome,
                changed_files,
                failure_signatures,
            ),
            problem_type=problem_type or None,
            strategy_used=self._unique_strings(
                [
                    str(getattr(session.task_state, "execution_strategy", "") or "").strip(),
                    *[item.strategy for item in session.repair_history[-6:]],
                ]
            )[:4],
            result=result,
            what_worked=what_worked,
            what_failed=what_failed,
            important_constraints=self._unique_strings(
                [
                    *(getattr(session.task_state, "constraints", []) or []),
                    *(getattr(session.task_understanding, "constraints", []) or []),
                ]
            )[:6],
            final_outcome=final_outcome,
            changed_files=changed_files,
            failure_signatures=failure_signatures,
        )

    def _build_project_entry(self, session: SessionState) -> ProjectMemoryEntry | None:
        snapshot = session.workspace_snapshot
        if snapshot is None:
            return None
        file_briefs = list(snapshot.file_briefs.items())[:8]
        module_roles = {path: self._trim_text(brief, 160) for path, brief in file_briefs}
        workflow_hints = self._unique_strings(
            [
                *snapshot.likely_commands[:4],
                *[item.command for item in snapshot.validation_commands[:4]],
            ]
        )[:6]
        common_file_relationships = [
            f"{test_path} -> {source_path}"
            for test_path, source_path in self._infer_test_mappings(snapshot.test_files, snapshot.important_files)
        ]
        return ProjectMemoryEntry(
            project_id=session.project_id or self.project_id,
            workspace_root=session.workspace_root,
            session_id=session.id,
            summary=MemorySummary(
                title=self._trim_text(session.workspace_label or Path(session.workspace_root).name or "Project", 90),
                summary=self._trim_text(snapshot.repo_summary, 220),
                key_points=[self._trim_text(item, 140) for item in snapshot.repo_map[:3]],
                why_relevant="Stable repository map for future tasks in the same project.",
            ),
            provenance=MemoryProvenance(
                source_type="workspace_snapshot",
                session_id=session.id,
                project_id=session.project_id or self.project_id,
                workspace_root=session.workspace_root,
                detail="Derived from repository inspection and recent workspace snapshots.",
                file_paths=snapshot.important_files[:6],
            ),
            tags=self._unique_strings(snapshot.project_labels[:8]),
            file_paths=snapshot.important_files[:10],
            symbol_names=self._unique_strings(
                [
                    symbol
                    for symbols in list(snapshot.symbol_index.values())[:10]
                    for symbol in list(symbols or [])[:6]
                ]
            )[:10],
            retention="long",
            ttl_days=RETENTION_DAYS["project"],
            importance=0.88,
            confidence=0.7,
            dedupe_key=f"project:{session.project_id or self.project_id}",
            repo_summary=snapshot.repo_summary,
            module_roles=module_roles,
            directory_map=list(snapshot.repo_map[:10]),
            entrypoints=list(snapshot.entrypoints[:8]),
            service_files=list(snapshot.service_files[:8]),
            import_hotspots=list(snapshot.import_hotspots[:8]),
            common_file_relationships=common_file_relationships[:8],
            test_mappings=[f"{left} -> {right}" for left, right in self._infer_test_mappings(snapshot.test_files, snapshot.important_files)][:8],
            symbol_index={path: symbols[:6] for path, symbols in list(snapshot.symbol_index.items())[:10]},
            file_relationships={path: relations[:6] for path, relations in list(snapshot.file_relationships.items())[:10]},
            module_summaries={path: self._trim_text(summary, 180) for path, summary in list(snapshot.module_summaries.items())[:8]},
            architecture_notes=[self._trim_text(item, 180) for item in snapshot.repo_map[:8]],
            known_hotspots=list(snapshot.important_files[:8]),
            conventions=[self._trim_text(item, 120) for item in snapshot.project_labels[:6]],
            workflow_hints=workflow_hints,
            co_change_hints=self._project_co_change_hints(session.project_id or self.project_id),
            subsystem_summaries={
                name: self._trim_text(summary, 180)
                for name, summary in (
                    list(getattr(snapshot, "subsystem_summaries", {}).items())[:6]
                    or [(path, brief) for path, brief in file_briefs[:6]]
                )
            },
        )

    def _build_failure_entries(self, session: SessionState) -> list[FailureMemoryEntry]:
        grouped: dict[str, dict[str, Any]] = {}
        active_context = session.active_repair_context
        active_signature = self._failure_signature_from_context(active_context)
        if active_signature:
            grouped[active_signature] = {
                "context": active_context,
                "attempts": [],
            }
        for attempt in session.repair_history:
            failure_signature = str(attempt.failure_signature or attempt.evidence_signature or "").strip()
            if not failure_signature:
                continue
            bucket = grouped.setdefault(
                failure_signature,
                {
                    "context": None,
                    "attempts": [],
                },
            )
            bucket["attempts"].append(attempt)
        entries: list[FailureMemoryEntry] = []
        for failure_signature, payload in grouped.items():
            entry = self._build_failure_entry(
                session,
                failure_signature=failure_signature,
                repair_context=payload.get("context"),
                tried=list(payload.get("attempts", [])),
            )
            if entry is not None:
                entries.append(entry)
        return entries

    def _build_conversation_entry(self, session: SessionState) -> ConversationMemoryEntry | None:
        if not session.messages and not str(session.task or "").strip():
            return None
        remembered_facts = self._extract_remembered_facts(session)
        request_summary = self._trim_text(
            str(getattr(session.task_state, "root_goal", "") or "").strip() or session.task,
            180,
        )
        delivered_summary = self._trim_text(
            str(session.final_response or "").strip()
            or str(session.report.summary if session.report is not None else "").strip(),
            220,
        )
        implemented_features = self._unique_strings(
            [
                str(getattr(session.task_state, "active_goal", "") or "").strip(),
                *(session.completion_criteria[:3] if session.status == "completed" else []),
            ]
        )[:4]
        base_tags = self._unique_strings(
            [
                str(getattr(session.task_state, "goal_relation", "") or "").strip(),
                str(getattr(session.task_state, "current_user_intent", "") or "").strip(),
                session.status,
                *[f"recall_subject:{item.subject}" for item in remembered_facts],
                *[f"recall_attribute:{item.attribute}" for item in remembered_facts],
            ]
        )
        dedupe_seed = " ".join(item.summary for item in remembered_facts) if remembered_facts else request_summary
        dedupe_key = (
            f"conversation:remembered:{self._hash_text(dedupe_seed)}"
            if remembered_facts
            else f"conversation:{session.project_id or self.project_id}:{self._hash_text(request_summary)}"
        )
        retention = "long" if remembered_facts else "short"
        ttl_days = 365 if remembered_facts else RETENTION_DAYS["conversation"]
        importance = 0.82 if remembered_facts else 0.58
        return ConversationMemoryEntry(
            project_id=session.project_id or self.project_id,
            workspace_root=session.workspace_root,
            session_id=session.id,
            summary=MemorySummary(
                title=request_summary or "Conversation milestone",
                summary=delivered_summary or request_summary,
                key_points=[self._trim_text(item.content, 140) for item in session.messages[-3:]],
                why_relevant="Compact conversation recall for later user-facing reminder questions.",
            ),
            provenance=MemoryProvenance(
                source_type="conversation",
                session_id=session.id,
                project_id=session.project_id or self.project_id,
                workspace_root=session.workspace_root,
                detail="Derived from the user request, assistant outcome, and recent conversation context.",
            ),
            tags=base_tags,
            file_paths=[item.path for item in session.changed_files[-6:]],
            symbol_names=list(session.working_memory.relevant_symbols[:4] if session.working_memory else []),
            retention=retention,
            ttl_days=ttl_days,
            importance=importance,
            confidence=max(float(getattr(session.task_state, "confidence", 0.0) or 0.0), 0.42),
            dedupe_key=dedupe_key,
            request_summary=request_summary,
            delivered_summary=delivered_summary or None,
            projects_touched=[session.project_id or self.project_id],
            decision_notes=[self._trim_text(item, 160) for item in session.notes[-4:]],
            implemented_features=implemented_features,
            referenced_sessions=[session.id],
            remembered_facts=remembered_facts,
        )

    def _build_failure_entry(
        self,
        session: SessionState,
        *,
        failure_signature: str,
        repair_context: Any | None,
        tried: list[Any],
    ) -> FailureMemoryEntry | None:
        repair_brief = getattr(repair_context, "repair_brief", None)
        chosen_targets = self._unique_strings(
            [
                *(getattr(repair_context, "artifact_paths", []) if repair_context is not None else []),
                *(getattr(repair_context, "file_hints", []) if repair_context is not None else []),
                *[str(item.artifact_path or "").strip() for item in tried],
            ]
        )[:8]
        review_rejection_reasons = self._unique_strings(
            [
                *[
                    self._trim_text(item.summary, 180)
                    for item in session.diagnostics[-10:]
                    if item.category in {"semantic_review", "repair", "validation"}
                ],
                *[self._trim_text(str(item.reason or ""), 180) for item in tried if item.result != "mutation_planned"],
                str(session.stop_reason or "").strip(),
            ]
        )[:6]
        title = self._trim_text(failure_signature or "Observed failure pattern", 90)
        if not title:
            return None
        failure_summary = (
            str(getattr(repair_context, "failure_summary", "") or "").strip()
            or self._trim_text(str(session.last_error or "").strip(), 220)
            or title
        )
        command = (
            str(getattr(repair_context, "command", "") or "").strip()
            or next(
                (
                    str(item.validation_command or "").strip()
                    for item in reversed(tried)
                    if str(item.validation_command or "").strip()
                ),
                "",
            )
            or None
        )
        verification_scope = str(getattr(repair_context, "verification_scope", "") or "").strip()
        return FailureMemoryEntry(
            project_id=session.project_id or self.project_id,
            workspace_root=session.workspace_root,
            session_id=session.id,
            summary=MemorySummary(
                title=title,
                summary=self._trim_text(failure_summary, 220),
                key_points=[
                    *[self._trim_text(item.reason, 140) for item in tried[-2:] if item.result != "mutation_planned"],
                    *[self._trim_text(item.reason, 140) for item in tried[-1:] if item.result == "mutation_planned"],
                ][:4],
                why_relevant="Failure-specific repair memory that should discourage repeated dead ends.",
            ),
            provenance=MemoryProvenance(
                source_type="repair" if tried else "validation",
                session_id=session.id,
                project_id=session.project_id or self.project_id,
                workspace_root=session.workspace_root,
                detail="Derived from validation failures, repair attempts, and diagnostic evidence across the session.",
                file_paths=chosen_targets[:6],
                command=command,
            ),
            tags=self._unique_strings(
                [
                    verification_scope,
                    str(getattr(session.task_state, "execution_strategy", "") or "").strip(),
                    "repair",
                ]
            ),
            file_paths=chosen_targets,
            symbol_names=self._unique_strings(
                list(getattr(repair_brief, "implicated_symbols", []) or [])
            )[:6],
            retention="medium",
            ttl_days=RETENTION_DAYS["failure"],
            importance=0.86,
            confidence=0.75,
            dedupe_key=self._failure_dedupe_key(session.project_id or self.project_id, failure_signature, chosen_targets[:2]),
            failure_signature=failure_signature or "unknown_failure",
            expected_semantics=list(getattr(repair_brief, "expected_semantics", []) or [])[:6],
            observed_semantics=list(getattr(repair_brief, "observed_semantics", []) or [])[:6],
            chosen_targets=chosen_targets,
            tried_strategies=self._unique_strings([item.strategy for item in tried])[:6],
            successful_repair_patterns=self._unique_strings(
                [item.strategy for item in tried if item.result == "mutation_planned"]
            )[:4],
            bad_retry_patterns=self._unique_strings(
                [item.strategy for item in tried if item.result in {"no_effective_change", "blocked", "generation_failed"}]
            )[:4],
            review_rejection_reasons=review_rejection_reasons,
            no_effective_change_count=sum(1 for item in tried if item.result == "no_effective_change"),
            last_result=str(tried[-1].result if tried else session.validation_status or session.status),
        )

    def _ensure_layout(self) -> None:
        self.entries_root.mkdir(parents=True, exist_ok=True)
        for memory_type in PERSISTENT_MEMORY_TYPES:
            (self.entries_root / memory_type).mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            index = self._empty_index()
            self._write_index(index)
            return index
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            payload = self._rebuild_index_from_disk()
            self._write_index(payload)
            return payload
        if not isinstance(payload, dict) or payload.get("version") != 1:
            payload = self._rebuild_index_from_disk()
            self._write_index(payload)
        return payload

    def _write_index(self, index: dict[str, Any]) -> None:
        temp = self.index_path.with_suffix(".tmp")
        temp.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.index_path)

    def _empty_index(self) -> dict[str, Any]:
        return {
            "version": 1,
            "entries": {},
            "by_project": {},
            "by_session": {},
            "by_failure": {},
            "by_file": {},
            "by_symbol": {},
            "by_tag": {},
            "by_term": {},
            "by_dedupe": {},
        }

    def _rebuild_index_from_disk(self) -> dict[str, Any]:
        index = self._empty_index()
        for memory_type in PERSISTENT_MEMORY_TYPES:
            folder = self.entries_root / memory_type
            if not folder.exists():
                continue
            for path in sorted(folder.glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                model = ENTRY_MODEL_BY_TYPE[memory_type]
                try:
                    entry = model.model_validate(payload)
                except Exception:
                    continue
                self._index_entry(index, entry)
        return index

    def _candidate_ids_for_request(self, request: RetrievalRequest) -> list[str]:
        buckets: list[str] = []
        if request.project_id:
            buckets.extend(self._index.get("by_project", {}).get(request.project_id, []))
        if request.session_id:
            buckets.extend(self._index.get("by_session", {}).get(request.session_id, []))
        if request.failure_signature:
            buckets.extend(self._index.get("by_failure", {}).get(request.failure_signature, []))
        for path in request.target_paths:
            buckets.extend(self._index.get("by_file", {}).get(path, []))
            buckets.extend(self._index.get("by_term", {}).get(Path(path).name.lower(), []))
            buckets.extend(self._index.get("by_term", {}).get(Path(path).stem.lower(), []))
        for symbol in request.symbol_names:
            normalized_symbol = str(symbol or "").strip()
            if not normalized_symbol:
                continue
            buckets.extend(self._index.get("by_symbol", {}).get(normalized_symbol, []))
            buckets.extend(self._index.get("by_symbol", {}).get(normalized_symbol.lower(), []))
        for term in request.error_terms:
            buckets.extend(self._index.get("by_term", {}).get(term, []))
        for term in self._query_terms(request):
            buckets.extend(self._index.get("by_term", {}).get(term, []))

        if not buckets:
            entries = list(self._index.get("entries", {}).items())
            entries.sort(key=lambda item: str(item[1].get("updated_at", "")), reverse=True)
            recent_ids = [
                entry_id
                for entry_id, metadata in entries
                if request.allow_cross_project or metadata.get("project_id") == request.project_id
            ]
            return recent_ids[:32]
        unique = self._unique_strings(buckets)
        unique.sort(
            key=lambda entry_id: str(self._index["entries"].get(entry_id, {}).get("updated_at", "")),
            reverse=True,
        )
        return unique[:48]

    def _should_retrieve_persistent_memory(self, request: RetrievalRequest) -> bool:
        if request.use_case in {"repair_assistance", "project_context", "user_recall"}:
            return True
        if request.failure_signature or request.target_paths or request.changed_files:
            return True
        if request.error_terms or request.symbol_names:
            return True
        project_entries = self._index.get("by_project", {}).get(str(request.project_id or "").strip(), [])
        return bool(project_entries)

    def _retrieval_request_signature(self, request: RetrievalRequest) -> str:
        payload = request.model_dump(mode="json")
        compact = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(compact.encode("utf-8")).hexdigest()[:16]

    def _score_entry(self, request: RetrievalRequest, metadata: dict[str, Any]) -> RetrievedMemoryItem | None:
        entry_id = str(metadata.get("id") or "")
        if not entry_id:
            return None
        memory_type = str(metadata.get("memory_type") or "")
        query_terms = self._query_terms(request)
        entry_terms = set(metadata.get("terms", []))
        similarity = self._jaccard_similarity(query_terms, entry_terms)
        personal_recall = bool(request.recall_subject)
        project_relevance = 0.0 if personal_recall else 1.0 if request.project_id and metadata.get("project_id") == request.project_id else 0.0
        exact_entity, entity_breakdown = self._exact_entity_relevance(request, metadata)
        failure_relevance = self._failure_relevance(request, metadata)
        recency, is_stale = self._recency_score(metadata)
        confidence = float(metadata.get("confidence", 0.0) or 0.0)
        importance = float(metadata.get("importance", 0.0) or 0.0)
        type_bonus = TYPE_USE_CASE_BONUS.get(request.use_case, {}).get(memory_type, 0.0)
        if personal_recall:
            score = (
                0.34 * similarity
                + 0.3 * exact_entity
                + 0.18 * recency
                + 0.08 * confidence
                + 0.06 * importance
                + type_bonus
            )
        else:
            score = (
                0.28 * similarity
                + 0.24 * project_relevance
                + 0.19 * exact_entity
                + 0.14 * recency
                + 0.09 * failure_relevance
                + 0.03 * confidence
                + 0.03 * importance
                + type_bonus
            )
        weak_cross_project_path_match = (
            not personal_recall
            and project_relevance <= 0.0
            and failure_relevance <= 0.0
            and similarity < 0.16
            and entity_breakdown.get("path", 0.0) > 0.0
            and entity_breakdown.get("symbol", 0.0) <= 0.0
            and entity_breakdown.get("generic_cross_project_path_only", False)
        )
        if weak_cross_project_path_match:
            score *= 0.3
        if is_stale:
            score *= 0.58
        if personal_recall and similarity < 0.08 and exact_entity < 0.22:
            return None
        if score < 0.12 and not project_relevance and not exact_entity and not failure_relevance:
            return None
        entry = self._load_entry(entry_id)
        if entry is None:
            return None
        reasons = []
        if project_relevance >= 0.99:
            reasons.append("same_project")
        if exact_entity >= 0.45:
            reasons.append("exact_entity_match")
        if failure_relevance >= 0.45:
            reasons.append("failure_match")
        if similarity >= 0.25:
            reasons.append("query_overlap")
        if recency >= 0.55:
            reasons.append("recent")
        return RetrievedMemoryItem(
            entry_id=entry_id,
            memory_type=memory_type,
            project_id=str(metadata.get("project_id") or "").strip() or None,
            session_id=str(metadata.get("session_id") or "").strip() or None,
            entry=entry,
            summary=MemorySummary.model_validate(metadata.get("summary", {})),
            provenance=MemoryProvenance.model_validate(metadata.get("provenance", {})),
            file_paths=list(metadata.get("file_paths", []))[:8],
            symbol_names=list(metadata.get("symbol_names", []))[:6],
            failure_signature=str(metadata.get("failure_signature") or "").strip() or None,
            score=round(score, 4),
            similarity=round(similarity, 4),
            recency=round(recency, 4),
            project_relevance=round(project_relevance, 4),
            failure_relevance=round(failure_relevance, 4),
            exact_entity_relevance=round(exact_entity, 4),
            confidence=confidence,
            duplicate_count=int(metadata.get("duplicate_count", 0) or 0),
            is_stale=is_stale,
            reasons=reasons,
        )

    def _select_with_budgets(
        self,
        scored: list[RetrievedMemoryItem],
        request: RetrievalRequest,
    ) -> list[RetrievedMemoryItem]:
        selected: list[RetrievedMemoryItem] = []
        per_type: defaultdict[str, int] = defaultdict(int)
        remaining_chars = max(int(request.summary_budget_chars or 0), 240)
        for item in scored:
            if len(selected) >= request.max_hits:
                break
            if per_type[item.memory_type] >= request.max_per_type:
                continue
            estimated_cost = len(item.summary.title) + len(item.summary.summary) + 48
            if selected and estimated_cost > remaining_chars:
                continue
            selected.append(item)
            per_type[item.memory_type] += 1
            remaining_chars -= estimated_cost
        return selected

    def _render_user_recall_brief(
        self,
        selected: list[RetrievedMemoryItem],
        request: RetrievalRequest,
    ) -> str:
        if not selected:
            return ""
        fact = self._best_remembered_fact(selected, request)
        if fact is not None:
            return self._render_remembered_fact_response(fact, request, request.summary_budget_chars)
        lines = ["Ich habe dazu relevante fruehere Arbeit gefunden:"]
        remaining = max(int(request.summary_budget_chars or 0), 240) - len(lines[0]) - 1
        for item in selected[:3]:
            session_ref = f"session={item.session_id}" if item.session_id else f"source={item.provenance.source_type}"
            line = self._trim_text(
                f"- {session_ref}: {item.summary.summary}",
                min(max(remaining, 80), 220),
            )
            if remaining <= 40:
                break
            lines.append(line)
            remaining -= len(line) + 1
        return "\n".join(lines)

    def _render_result_summary(
        self,
        selected: list[RetrievedMemoryItem],
        budget_chars: int,
    ) -> str:
        if not selected:
            return "No relevant persistent memory selected."
        lines: list[str] = []
        remaining = max(int(budget_chars or 0), 240)
        for item in selected:
            provenance = item.provenance
            source = []
            if provenance.session_id:
                source.append(f"session={provenance.session_id}")
            if provenance.source_type:
                source.append(f"source={provenance.source_type}")
            reason = ",".join(item.reasons[:3]) or "scored"
            line = self._trim_text(
                f"[{item.memory_type}] {item.summary.title}: {item.summary.summary} "
                f"(why={reason}; {' '.join(source)})",
                min(remaining, 240),
            )
            if len(line) > remaining and lines:
                break
            lines.append(line)
            remaining -= len(line) + 1
            if remaining <= 60:
                break
        return "\n".join(lines) if lines else "No relevant persistent memory selected."

    def _mark_entries_accessed(self, selected: list[RetrievedMemoryItem]) -> None:
        if not selected:
            return
        touched = False
        now = utc_now()
        for item in selected:
            metadata = self._index.get("entries", {}).get(item.entry_id)
            if metadata is not None:
                metadata["last_accessed_at"] = now
                touched = True
            entry = item.entry
            if entry is None:
                continue
            if getattr(entry, "last_accessed_at", None) == now:
                continue
            updated = entry.model_copy(update={"last_accessed_at": now})
            folder = self.entries_root / updated.memory_type
            path = folder / f"{updated.id}.json"
            if path.exists():
                path.write_text(updated.model_dump_json(indent=2), encoding="utf-8")
        if touched:
            self._write_index(self._index)

    def _write_entry(
        self,
        entry: EpisodicMemoryEntry | ProjectMemoryEntry | FailureMemoryEntry | ConversationMemoryEntry,
    ) -> None:
        folder = self.entries_root / entry.memory_type
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{entry.id}.json"
        temp = target.with_suffix(".tmp")
        payload = entry.model_dump(mode="json")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(target)
        self._index_entry(self._index, entry)
        self._write_index(self._index)

    def _index_entry(
        self,
        index: dict[str, Any],
        entry: EpisodicMemoryEntry | ProjectMemoryEntry | FailureMemoryEntry | ConversationMemoryEntry,
    ) -> None:
        metadata = self._entry_metadata(entry)
        entry_id = entry.id
        index["entries"][entry_id] = metadata
        if entry.project_id:
            self._index_list_add(index["by_project"], entry.project_id, entry_id)
        if entry.session_id:
            self._index_list_add(index["by_session"], entry.session_id, entry_id)
        if metadata.get("failure_signature"):
            self._index_list_add(index["by_failure"], metadata["failure_signature"], entry_id)
        for key in metadata.get("file_paths", []):
            self._index_list_add(index["by_file"], key, entry_id)
        for key in metadata.get("symbol_names", []):
            self._index_list_add(index["by_symbol"], key, entry_id)
            lowered = str(key or "").strip().lower()
            if lowered and lowered != key:
                self._index_list_add(index["by_symbol"], lowered, entry_id)
        for key in metadata.get("tags", []):
            self._index_list_add(index["by_tag"], key, entry_id)
        for key in metadata.get("terms", []):
            self._index_list_add(index["by_term"], key, entry_id)
        if entry.dedupe_key:
            index["by_dedupe"][entry.dedupe_key] = entry_id

    def _entry_metadata(
        self,
        entry: EpisodicMemoryEntry | ProjectMemoryEntry | FailureMemoryEntry | ConversationMemoryEntry,
    ) -> dict[str, Any]:
        failure_signature = None
        if isinstance(entry, FailureMemoryEntry):
            failure_signature = entry.failure_signature
        terms = self._entry_terms(entry)
        return {
            "id": entry.id,
            "memory_type": entry.memory_type,
            "project_id": entry.project_id,
            "workspace_root": entry.workspace_root,
            "session_id": entry.session_id,
            "summary": entry.summary.model_dump(mode="json"),
            "provenance": entry.provenance.model_dump(mode="json"),
            "tags": list(entry.tags[:12]),
            "file_paths": list(entry.file_paths[:12]),
            "symbol_names": list(entry.symbol_names[:10]),
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
            "last_accessed_at": entry.last_accessed_at,
            "importance": entry.importance,
            "confidence": entry.confidence,
            "dedupe_key": entry.dedupe_key,
            "duplicate_count": entry.duplicate_count,
            "retention": entry.retention,
            "ttl_days": entry.ttl_days,
            "failure_signature": failure_signature,
            "terms": terms,
        }

    def _entry_terms(
        self,
        entry: EpisodicMemoryEntry | ProjectMemoryEntry | FailureMemoryEntry | ConversationMemoryEntry,
    ) -> list[str]:
        chunks = [
            entry.summary.title,
            entry.summary.summary,
            *entry.summary.key_points,
            *entry.tags,
            *entry.file_paths,
            *entry.symbol_names,
        ]
        if isinstance(entry, EpisodicMemoryEntry):
            chunks.extend([entry.problem_type or "", *entry.strategy_used, *entry.failure_signatures, *entry.what_worked, *entry.what_failed])
        elif isinstance(entry, ProjectMemoryEntry):
            chunks.extend(
                [
                    entry.repo_summary,
                    *entry.directory_map,
                    *entry.entrypoints,
                    *entry.service_files,
                    *entry.import_hotspots,
                    *entry.workflow_hints,
                    *entry.known_hotspots,
                    *entry.co_change_hints,
                ]
            )
            for path, symbols in list(entry.symbol_index.items())[:10]:
                chunks.append(path)
                chunks.extend(symbols[:6])
        elif isinstance(entry, FailureMemoryEntry):
            chunks.extend([entry.failure_signature, *entry.tried_strategies, *entry.successful_repair_patterns, *entry.bad_retry_patterns, *entry.chosen_targets])
        elif isinstance(entry, ConversationMemoryEntry):
            chunks.extend(
                [
                    entry.request_summary,
                    entry.delivered_summary or "",
                    *entry.implemented_features,
                    *entry.decision_notes,
                    *[fact.summary for fact in entry.remembered_facts],
                    *[fact.value for fact in entry.remembered_facts],
                    *[fact.attribute for fact in entry.remembered_facts],
                    *[fact.subject for fact in entry.remembered_facts],
                ]
            )
        return self._terms_from_text(" ".join(chunk for chunk in chunks if chunk))

    def _load_entry(
        self,
        entry_id: str,
    ) -> EpisodicMemoryEntry | ProjectMemoryEntry | FailureMemoryEntry | ConversationMemoryEntry | None:
        metadata = self._index.get("entries", {}).get(entry_id)
        if not metadata:
            return None
        memory_type = str(metadata.get("memory_type") or "")
        model = ENTRY_MODEL_BY_TYPE.get(memory_type)
        if model is None:
            return None
        path = self.entries_root / memory_type / f"{entry_id}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return model.model_validate(payload)
        except Exception:
            return None

    def _merge_entries(
        self,
        existing: EpisodicMemoryEntry | ProjectMemoryEntry | FailureMemoryEntry | ConversationMemoryEntry,
        incoming: EpisodicMemoryEntry | ProjectMemoryEntry | FailureMemoryEntry | ConversationMemoryEntry,
    ) -> EpisodicMemoryEntry | ProjectMemoryEntry | FailureMemoryEntry | ConversationMemoryEntry:
        if type(existing) is not type(incoming):
            return incoming
        updated_at = utc_now()
        common_update = {
            "updated_at": updated_at,
            "duplicate_count": max(existing.duplicate_count, 0) + 1,
            "summary": MemorySummary(
                title=incoming.summary.title or existing.summary.title,
                summary=incoming.summary.summary or existing.summary.summary,
                key_points=self._unique_strings([*existing.summary.key_points, *incoming.summary.key_points])[:6],
                why_relevant=incoming.summary.why_relevant or existing.summary.why_relevant,
            ),
            "tags": self._unique_strings([*existing.tags, *incoming.tags])[:12],
            "file_paths": self._unique_strings([*existing.file_paths, *incoming.file_paths])[:12],
            "symbol_names": self._unique_strings([*existing.symbol_names, *incoming.symbol_names])[:10],
            "importance": max(existing.importance, incoming.importance),
            "confidence": max(existing.confidence, incoming.confidence),
        }
        if isinstance(existing, EpisodicMemoryEntry) and isinstance(incoming, EpisodicMemoryEntry):
            return existing.model_copy(
                update={
                    **common_update,
                    "strategy_used": self._unique_strings([*existing.strategy_used, *incoming.strategy_used])[:6],
                    "what_worked": self._unique_strings([*existing.what_worked, *incoming.what_worked])[:6],
                    "what_failed": self._unique_strings([*existing.what_failed, *incoming.what_failed])[:6],
                    "important_constraints": self._unique_strings([*existing.important_constraints, *incoming.important_constraints])[:8],
                    "changed_files": self._unique_strings([*existing.changed_files, *incoming.changed_files])[:12],
                    "failure_signatures": self._unique_strings([*existing.failure_signatures, *incoming.failure_signatures])[:8],
                    "final_outcome": incoming.final_outcome or existing.final_outcome,
                    "result": incoming.result if incoming.result != "completed" else existing.result,
                }
            )
        if isinstance(existing, ProjectMemoryEntry) and isinstance(incoming, ProjectMemoryEntry):
            merged_roles = dict(existing.module_roles)
            merged_roles.update(incoming.module_roles)
            merged_subsystems = dict(existing.subsystem_summaries)
            merged_subsystems.update(incoming.subsystem_summaries)
            merged_module_summaries = dict(existing.module_summaries)
            merged_module_summaries.update(incoming.module_summaries)
            return existing.model_copy(
                update={
                    **common_update,
                    "repo_summary": incoming.repo_summary or existing.repo_summary,
                    "module_roles": merged_roles,
                    "directory_map": self._unique_strings([*existing.directory_map, *incoming.directory_map])[:10],
                    "entrypoints": self._unique_strings([*existing.entrypoints, *incoming.entrypoints])[:10],
                    "service_files": self._unique_strings([*existing.service_files, *incoming.service_files])[:10],
                    "import_hotspots": self._unique_strings([*existing.import_hotspots, *incoming.import_hotspots])[:10],
                    "common_file_relationships": self._unique_strings([*existing.common_file_relationships, *incoming.common_file_relationships])[:10],
                    "test_mappings": self._unique_strings([*existing.test_mappings, *incoming.test_mappings])[:10],
                    "symbol_index": self._merge_symbol_indexes(existing.symbol_index, incoming.symbol_index),
                    "file_relationships": self._merge_symbol_indexes(existing.file_relationships, incoming.file_relationships),
                    "module_summaries": merged_module_summaries,
                    "architecture_notes": self._unique_strings([*existing.architecture_notes, *incoming.architecture_notes])[:10],
                    "known_hotspots": self._unique_strings([*existing.known_hotspots, *incoming.known_hotspots])[:10],
                    "conventions": self._unique_strings([*existing.conventions, *incoming.conventions])[:10],
                    "workflow_hints": self._unique_strings([*existing.workflow_hints, *incoming.workflow_hints])[:10],
                    "co_change_hints": self._unique_strings([*existing.co_change_hints, *incoming.co_change_hints])[:10],
                    "subsystem_summaries": merged_subsystems,
                }
            )
        if isinstance(existing, FailureMemoryEntry) and isinstance(incoming, FailureMemoryEntry):
            return existing.model_copy(
                update={
                    **common_update,
                    "expected_semantics": self._unique_strings([*existing.expected_semantics, *incoming.expected_semantics])[:8],
                    "observed_semantics": self._unique_strings([*existing.observed_semantics, *incoming.observed_semantics])[:8],
                    "chosen_targets": self._unique_strings([*existing.chosen_targets, *incoming.chosen_targets])[:10],
                    "tried_strategies": self._unique_strings([*existing.tried_strategies, *incoming.tried_strategies])[:8],
                    "successful_repair_patterns": self._unique_strings([*existing.successful_repair_patterns, *incoming.successful_repair_patterns])[:6],
                    "bad_retry_patterns": self._unique_strings([*existing.bad_retry_patterns, *incoming.bad_retry_patterns])[:6],
                    "review_rejection_reasons": self._unique_strings([*existing.review_rejection_reasons, *incoming.review_rejection_reasons])[:8],
                    "no_effective_change_count": existing.no_effective_change_count + incoming.no_effective_change_count,
                    "last_result": incoming.last_result or existing.last_result,
                }
            )
        if isinstance(existing, ConversationMemoryEntry) and isinstance(incoming, ConversationMemoryEntry):
            return existing.model_copy(
                update={
                    **common_update,
                    "delivered_summary": incoming.delivered_summary or existing.delivered_summary,
                    "projects_touched": self._unique_strings([*existing.projects_touched, *incoming.projects_touched])[:6],
                    "decision_notes": self._unique_strings([*existing.decision_notes, *incoming.decision_notes])[:8],
                    "implemented_features": self._unique_strings([*existing.implemented_features, *incoming.implemented_features])[:8],
                    "referenced_sessions": self._unique_strings([*existing.referenced_sessions, *incoming.referenced_sessions])[:8],
                    "remembered_facts": self._merge_remembered_facts(existing.remembered_facts, incoming.remembered_facts),
                }
            )
        return incoming

    def _infer_use_case(self, task: str, session: SessionState) -> str:
        lowered = normalize_text(task)
        if session.validation_status in {"failed", "bootstrap_failed", "bootstrap_reset_required"} or session.active_repair_context is not None:
            return "repair_assistance"
        if self._personal_recall_focus(task)[0] is not None or self._looks_like_history_recall_query(lowered):
            return "user_recall"
        if str(getattr(session.task_state, "goal_relation", "") or "").strip() in {"continue", "refine"}:
            return "task_continuation"
        if any(token in lowered for token in ("architecture", "repo", "projekt", "module", "subsystem")):
            return "project_context"
        return "similar_task_lookup"

    def _extract_remembered_facts(self, session: SessionState) -> list[RememberedFact]:
        facts: list[RememberedFact] = []
        task_state_facts = list(getattr(session.task_state, "remembered_facts", []) or [])
        for item in task_state_facts:
            attribute = self._clean_fact_value(str(getattr(item, "attribute", "") or ""))
            value = self._clean_fact_value(str(getattr(item, "value", "") or ""))
            if not attribute or not value:
                continue
            subject = str(getattr(item, "subject", "user") or "user").strip().lower()
            if subject not in {"user", "assistant"}:
                subject = "user"
            summary = self._clean_fact_value(str(getattr(item, "summary", "") or ""))
            facts.append(
                RememberedFact(
                    subject=subject,
                    attribute=attribute,
                    value=value,
                    summary=summary or f"{subject}:{attribute}={value}",
                )
            )
        for message in session.messages[-12:]:
            if message.role != "user":
                continue
            facts.extend(self._facts_from_user_text(message.content))
        return self._merge_remembered_facts([], facts)

    def _facts_from_user_text(self, text: str) -> list[RememberedFact]:
        raw = str(text or "").strip()
        if not raw:
            return []
        normalized = self._fact_text(raw)
        facts = self._named_facts_from_statement(normalized)
        directive_payload = self._memory_directive_payload(normalized)
        if directive_payload:
            directive_facts = self._named_facts_from_statement(directive_payload)
            if directive_facts:
                facts.extend(directive_facts)
            else:
                note = self._clean_fact_value(directive_payload)
                if note:
                    facts.append(
                        RememberedFact(
                            subject="user",
                            attribute="note",
                            value=note,
                            summary=f"User note: {note}.",
                        )
                    )
        return self._merge_remembered_facts([], facts)

    def _named_facts_from_statement(self, text: str) -> list[RememberedFact]:
        facts: list[RememberedFact] = []
        for pattern in _USER_NAME_PATTERNS:
            for match in pattern.finditer(text):
                value = self._clean_fact_value(str(match.group("value") or ""))
                if not self._looks_like_person_name(value):
                    continue
                facts.append(
                    RememberedFact(
                        subject="user",
                        attribute="name",
                        value=value,
                        summary=f"User name: {value}.",
                    )
                )
        return facts

    def _memory_directive_payload(self, text: str) -> str | None:
        for pattern in _MEMORY_DIRECTIVE_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            candidate = self._clean_fact_value(str(match.group("fact") or ""))
            if candidate:
                return candidate
        return None

    def _fact_text(self, text: str) -> str:
        normalized = str(text or "").replace("ß", "ss")
        return " ".join(normalized.split())

    def _clean_fact_value(self, value: str) -> str:
        return str(value or "").strip().strip("\"'`").rstrip(" .,:;!?")

    def _looks_like_person_name(self, value: str) -> bool:
        words = [item for item in re.split(r"[^A-Za-z0-9_-]+", str(value or "").strip()) if item]
        if not words or len(words) > 4:
            return False
        alpha_words = [item for item in words if any(ch.isalpha() for ch in item)]
        if not alpha_words:
            return False
        if len(alpha_words) >= 2:
            return all(item[0].isupper() for item in alpha_words if item)
        token = alpha_words[0]
        return len(token) >= 2 and token[0].isupper()

    def _merge_remembered_facts(
        self,
        existing: list[RememberedFact],
        incoming: list[RememberedFact],
    ) -> list[RememberedFact]:
        merged: list[RememberedFact] = []
        seen: set[tuple[str, str, str]] = set()
        for item in [*existing, *incoming]:
            key = (
                str(item.subject or "").strip().lower(),
                str(item.attribute or "").strip().lower(),
                str(item.value or "").strip().lower(),
            )
            if not all(key) or key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged[:8]

    def _personal_recall_focus(self, task: str) -> tuple[str | None, tuple[str, ...]]:
        text = str(task or "").strip()
        if not text:
            return None, ()
        normalized = normalize_text(text)
        tokens = self._memory_tokens(normalized)
        question_like = normalized.endswith("?") or bool(tokens and tokens[0] in _QUESTION_WORD_TOKENS)
        conversational = classify_conversation_request(text) is not None
        if not question_like and not conversational:
            return None, ()
        first_person = bool(set(tokens) & _FIRST_PERSON_TOKENS) or any(phrase in normalized for phrase in _PROFILE_QUERY_PHRASES)
        second_person = bool(set(tokens) & _SECOND_PERSON_TOKENS)
        if not first_person and not second_person:
            return None, ()
        identity_like = bool(set(tokens) & _IDENTITY_QUERY_TOKENS) or any(phrase in normalized for phrase in _PROFILE_QUERY_PHRASES)
        content_tokens = [
            token
            for token in tokens
            if len(token) >= 4 and token not in _PERSONAL_QUERY_STOPWORDS
        ]
        if not identity_like and any(token in _PERSONAL_RECALL_EXCLUDE_TOKENS for token in content_tokens):
            return None, ()
        if not identity_like and not content_tokens:
            return None, ()
        subject = "user" if first_person else "assistant"
        attributes: list[str] = []
        if identity_like:
            attributes.extend(["identity", "profile"])
            if "name" in tokens or "heisse" in tokens or "heisst" in tokens:
                attributes.insert(0, "name")
            elif subject == "user":
                attributes.append("name")
        for token in content_tokens:
            if token not in attributes:
                attributes.append(token)
        return subject, tuple(self._unique_strings(attributes))

    def _looks_like_history_recall_query(self, normalized: str) -> bool:
        tokens = self._memory_tokens(normalized)
        if not tokens:
            return False
        question_like = normalized.endswith("?") or tokens[0] in _QUESTION_WORD_TOKENS
        temporal = bool(set(tokens) & _HISTORY_TEMPORAL_TOKENS) or any(
            phrase in normalized for phrase in ("last time", "letzter stand", "previous run")
        )
        continuity = bool(set(tokens) & _HISTORY_CONTINUITY_TOKENS)
        return temporal and (question_like or continuity)

    def _memory_tokens(self, normalized: str) -> list[str]:
        return [token for token in re.split(r"[^a-z0-9_]+", normalized) if token]

    def _best_remembered_fact(
        self,
        selected: list[RetrievedMemoryItem],
        request: RetrievalRequest,
    ) -> RememberedFact | None:
        preferred_subject = str(request.recall_subject or "").strip().lower()
        preferred_attributes = {str(item or "").strip().lower() for item in request.recall_attributes if str(item or "").strip()}
        query_terms = self._query_terms(request)
        best_fact: RememberedFact | None = None
        best_score = -1.0
        for item in selected:
            entry = item.entry
            if not isinstance(entry, ConversationMemoryEntry):
                continue
            for fact in entry.remembered_facts:
                fact_subject = str(fact.subject or "").strip().lower()
                fact_attribute = str(fact.attribute or "").strip().lower()
                if preferred_subject and fact_subject != preferred_subject:
                    continue
                score = item.score
                if fact_attribute in preferred_attributes:
                    score += 0.6
                if preferred_attributes & {"identity", "profile"} and fact_attribute == "name":
                    score += 0.45
                fact_terms = set(self._terms_from_text(" ".join([fact.attribute, fact.summary, fact.value])))
                overlap = len(query_terms & fact_terms)
                if overlap:
                    score += min(0.45, 0.12 * overlap)
                if score > best_score:
                    best_fact = fact
                    best_score = score
        return best_fact

    def _render_remembered_fact_response(
        self,
        fact: RememberedFact,
        request: RetrievalRequest,
        budget_chars: int,
    ) -> str:
        preferred_attributes = {str(item or "").strip().lower() for item in request.recall_attributes if str(item or "").strip()}
        if fact.subject == "user" and (fact.attribute == "name" or preferred_attributes & {"identity", "name", "profile"}):
            return self._trim_text(f"Du bist {fact.value}.", max(int(budget_chars or 0), 60))
        if fact.subject == "assistant" and (fact.attribute == "name" or preferred_attributes & {"identity", "name", "profile"}):
            return self._trim_text(f"Ich bin {fact.value}.", max(int(budget_chars or 0), 60))
        if fact.subject == "user":
            return self._trim_text(f"Ich habe mir ueber dich gemerkt: {fact.value}.", max(int(budget_chars or 0), 80))
        return self._trim_text(f"Ich habe mir dazu gemerkt: {fact.value}.", max(int(budget_chars or 0), 80))

    def _is_fresh_task_bootstrap(self, session: SessionState) -> bool:
        if session.active_repair_context is not None:
            return False
        if session.validation_status in {"failed", "bootstrap_failed", "bootstrap_reset_required"}:
            return False
        if session.task_state is not None or session.task_understanding is not None or session.router_result is not None:
            return False
        if session.plan:
            return False
        if session.tool_calls or session.changed_files or session.diagnostics or session.validation_runs:
            return False
        if session.repair_history or session.executed_commands or session.notes:
            return False
        follow_up = session.follow_up_context
        if follow_up is None:
            return True
        return not any(
            [
                str(follow_up.previous_task or "").strip(),
                str(follow_up.previous_root_goal or "").strip(),
                str(follow_up.previous_active_goal or "").strip(),
                str(follow_up.previous_requested_outcome or "").strip(),
                str(follow_up.previous_interpreted_goal or "").strip(),
                str(follow_up.last_error or "").strip(),
                follow_up.previous_constraints,
                follow_up.previous_assumptions,
                follow_up.target_paths,
                follow_up.changed_files,
                follow_up.read_files,
                follow_up.recent_commands,
                follow_up.notes,
                follow_up.diagnostics,
                follow_up.validation_runs,
            ]
        )

    def _query_terms(
        self,
        request: RetrievalRequest,
        *,
        reference_terms: set[str] | None = None,
    ) -> set[str]:
        recall_chunks: list[str] = []
        if request.recall_subject:
            recall_chunks.append(request.recall_subject)
        recall_chunks.extend(request.recall_attributes)
        return set(
            prioritized_focus_terms(
                " ".join(
                    part
                    for part in [
                        request.query,
                        request.current_goal or "",
                        request.current_subtask or "",
                        " ".join(recall_chunks),
                        " ".join(request.target_paths),
                        " ".join(request.symbol_names),
                        " ".join(request.error_terms),
                        request.failure_signature or "",
                    ]
                    if part
                ),
                max_terms=32,
                reference_terms=reference_terms,
            )
        )

    def _exact_entity_relevance(self, request: RetrievalRequest, metadata: dict[str, Any]) -> tuple[float, dict[str, float | bool]]:
        target_paths = {item for item in request.target_paths if item}
        file_paths = set(metadata.get("file_paths", []))
        symbol_names = {str(item).strip().lower() for item in request.symbol_names if str(item).strip()}
        stored_symbols = {str(item).strip().lower() for item in metadata.get("symbol_names", []) if str(item).strip()}
        stored_tags = {str(item).strip().lower() for item in metadata.get("tags", []) if str(item).strip()}
        same_project = bool(request.project_id and metadata.get("project_id") == request.project_id)
        matches = 0.0
        total = 0.0
        path_match = 0.0
        symbol_match = 0.0
        recall_match = 0.0
        generic_cross_project_path_only = False
        if target_paths:
            total += 1.0
            matched_paths = target_paths & file_paths
            if matched_paths:
                if same_project:
                    path_match = 1.0
                else:
                    max_depth = max(len(Path(item).parts) for item in matched_paths)
                    if max_depth > 1:
                        path_match = 0.72
                    elif len(matched_paths) >= 3:
                        path_match = 0.38
                    elif len(matched_paths) >= 2:
                        path_match = 0.32
                    else:
                        path_match = 0.24
                    generic_cross_project_path_only = all(len(Path(item).parts) == 1 for item in matched_paths)
            else:
                matched_names = {Path(item).name for item in target_paths} & {Path(item).name for item in file_paths}
                if matched_names:
                    path_match = 0.65 if same_project else 0.18
                    generic_cross_project_path_only = not same_project
            matches += path_match
        if symbol_names:
            total += 1.0
            if symbol_names & stored_symbols:
                symbol_match = 1.0
                matches += symbol_match
        if request.recall_subject:
            total += 1.0
            subject_tag = f"recall_subject:{request.recall_subject}".lower()
            attribute_tags = {f"recall_attribute:{item}".lower() for item in request.recall_attributes if str(item).strip()}
            if subject_tag in stored_tags:
                recall_match += 0.7
                if attribute_tags & stored_tags:
                    recall_match += 0.3
                matches += recall_match
        return (
            matches / total if total else 0.0,
            {
                "path": path_match,
                "symbol": symbol_match,
                "recall": recall_match,
                "generic_cross_project_path_only": generic_cross_project_path_only,
            },
        )

    def _failure_relevance(self, request: RetrievalRequest, metadata: dict[str, Any]) -> float:
        failure_signature = str(request.failure_signature or "").strip()
        candidate = str(metadata.get("failure_signature") or "").strip()
        if not failure_signature or not candidate:
            return 0.0
        if failure_signature == candidate:
            return 1.0
        if failure_signature in candidate or candidate in failure_signature:
            return 0.7
        return 0.0

    def _request_error_terms(self, session: SessionState) -> list[str]:
        chunks: list[str] = []
        for item in session.diagnostics[-4:]:
            chunks.extend([str(item.summary or "").strip(), str(item.excerpt or "").strip()])
        repair_context = session.active_repair_context
        if repair_context is not None:
            chunks.extend(
                [
                    str(repair_context.failure_summary or "").strip(),
                    str(repair_context.summary or "").strip(),
                    str(repair_context.excerpt or "").strip(),
                ]
            )
        return self._terms_from_text(" ".join(chunk for chunk in chunks if chunk))[:16]

    def _repo_map_signal_bundle(
        self,
        snapshot: Any | None,
        request: RetrievalRequest,
    ) -> tuple[list[str], list[str], list[str]]:
        if snapshot is None:
            return [], [], []
        ranked: list[tuple[float, str]] = []
        exact_targets = {str(path or "").strip() for path in request.target_paths if str(path or "").strip()}
        exact_names = {Path(path).name.lower() for path in exact_targets}
        requested_symbols = {
            str(symbol or "").strip().lower()
            for symbol in request.symbol_names
            if str(symbol or "").strip()
        }
        symbol_index = getattr(snapshot, "symbol_index", {}) or {}
        file_relationships = getattr(snapshot, "file_relationships", {}) or {}
        candidate_paths = self._unique_strings(
            [
                *list(getattr(snapshot, "focus_files", []) or []),
                *list(getattr(snapshot, "important_files", []) or []),
                *list(getattr(snapshot, "entrypoints", []) or []),
                *list(getattr(snapshot, "service_files", []) or []),
                *list(getattr(snapshot, "import_hotspots", []) or []),
                *list(symbol_index.keys()),
            ]
        )
        query_terms = self._query_terms(
            request,
            reference_terms=self._candidate_reference_terms(candidate_paths, symbol_index),
        )
        if not query_terms and not request.target_paths and not request.symbol_names and not request.error_terms:
            return [], [], []
        path_terms_by_candidate = {
            path: set(self._terms_from_text(path))
            for path in candidate_paths
        }
        query_term_frequency: dict[str, int] = defaultdict(int)
        for terms in path_terms_by_candidate.values():
            for term in query_terms & terms:
                query_term_frequency[term] += 1
        for path in candidate_paths:
            path_terms = path_terms_by_candidate.get(path, set())
            score = 0.0
            if path in exact_targets:
                score += 3.2
            if Path(path).name.lower() in exact_names:
                score += 1.6
            matched_terms = query_terms & path_terms
            if matched_terms:
                specificity = sum(
                    1.0 / max(query_term_frequency.get(term, 1), 1)
                    for term in matched_terms
                )
                score += min(2.2, 1.2 * specificity)
            path_symbols = [str(item or "").strip() for item in symbol_index.get(path, []) if str(item or "").strip()]
            lowered_symbols = {item.lower() for item in path_symbols}
            if requested_symbols & lowered_symbols:
                score += 2.2
            symbol_term_matches = query_terms & lowered_symbols
            if symbol_term_matches:
                score += min(1.8, 0.7 * len(symbol_term_matches))
            if path in list(getattr(snapshot, "entrypoints", []) or []) and query_terms & {"main", "entry", "cli", "server"}:
                score += 0.35
            if path in list(getattr(snapshot, "service_files", []) or []) and query_terms & {"service", "route", "router", "handler", "api", "auth", "server"}:
                score += 0.4
            if path in list(getattr(snapshot, "import_hotspots", []) or []) and query_terms & {"import", "dependency", "module", "package"}:
                score += 0.4
            if score >= 0.6:
                ranked.append((score, path))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        suggested_files = [path for _, path in ranked[:8]]
        related_files = self._unique_strings(
            [
                relation
                for path in suggested_files[:3]
                for relation in list(file_relationships.get(path, []) or [])[:4]
            ]
        )[:4]
        suggested_files = self._unique_strings([*suggested_files, *related_files])[:8]
        suggested_symbols = self._unique_strings(
            [
                symbol
                for path in suggested_files[:4]
                for symbol in list(symbol_index.get(path, []) or [])[:6]
                if requested_symbols or str(symbol).lower() in query_terms
            ]
        )[:8]
        repo_hints = self._repo_map_hints(snapshot, suggested_files)
        return suggested_files, suggested_symbols, repo_hints

    def _repo_map_hints(self, snapshot: Any, suggested_files: list[str]) -> list[str]:
        hints: list[str] = []
        test_mappings = list(getattr(snapshot, "test_mappings", []) or [])
        import_hotspots = set(getattr(snapshot, "import_hotspots", []) or [])
        symbol_index = getattr(snapshot, "symbol_index", {}) or {}
        file_relationships = getattr(snapshot, "file_relationships", {}) or {}
        module_summaries = getattr(snapshot, "module_summaries", {}) or {}
        for path in suggested_files[:4]:
            for mapping in test_mappings:
                if path in mapping and mapping not in hints:
                    hints.append(mapping)
            if path in import_hotspots:
                hints.append(f"{path} is an import hotspot")
            symbols = list(symbol_index.get(path, []) or [])[:4]
            if symbols:
                hints.append(f"{path} symbols: {', '.join(symbols)}")
            related = list(file_relationships.get(path, []) or [])[:3]
            if related:
                hints.append(f"{path} related: {', '.join(related)}")
            summary = str(module_summaries.get(path) or "").strip()
            if summary:
                hints.append(f"{path} summary: {self._trim_text(summary, 160)}")
        return self._unique_strings(hints)[:6]

    def _render_repo_hint_summary(self, repo_hints: list[str], budget_chars: int) -> str:
        if not repo_hints:
            return "No relevant persistent memory selected."
        lines = ["Repo map hints:"]
        remaining = max(int(budget_chars or 0), 240) - len(lines[0]) - 1
        for hint in repo_hints[:4]:
            line = self._trim_text(f"- {hint}", min(max(remaining, 80), 180))
            lines.append(line)
            remaining -= len(line) + 1
            if remaining <= 40:
                break
        return "\n".join(lines)

    def _merge_repo_hints_into_summary(self, summary: str, repo_hints: list[str], budget_chars: int) -> str:
        repo_summary = self._render_repo_hint_summary(repo_hints, max(180, budget_chars // 2))
        if not summary or summary == "No relevant persistent memory selected.":
            return repo_summary
        merged = f"{summary}\n{repo_summary}"
        return self._trim_text(merged, max(int(budget_chars or 0), 240))

    def _recency_score(self, metadata: dict[str, Any]) -> tuple[float, bool]:
        timestamp = str(metadata.get("updated_at") or metadata.get("created_at") or "").strip()
        if not timestamp:
            return 0.0, False
        age_days = max(self._age_days(timestamp), 0.0)
        ttl_days = int(metadata.get("ttl_days") or 0)
        is_stale = ttl_days > 0 and age_days > ttl_days
        score = 1.0 / (1.0 + (age_days / 14.0))
        return score, is_stale

    def _hit_count_by_type(self, selected: list[RetrievedMemoryItem]) -> dict[str, int]:
        counts: defaultdict[str, int] = defaultdict(int)
        for item in selected:
            counts[item.memory_type] += 1
        return dict(counts)

    def _resolve_project_id(self, snapshot: Any | None = None) -> str:
        git_tokens = self._git_identity_tokens()
        if git_tokens:
            return self._project_id_from_tokens(git_tokens)
        snapshot_tokens = self._project_identity_tokens_from_snapshot(snapshot)
        if snapshot_tokens:
            return self._project_id_from_tokens(snapshot_tokens)
        workspace_tokens = self._workspace_identity_tokens()
        if self._workspace_identity_tokens_are_stable(workspace_tokens):
            return self._project_id_from_tokens(workspace_tokens)
        return self._project_id_for_root(self.workspace.root)

    def _project_id_from_tokens(self, identity_tokens: list[str]) -> str:
        if not identity_tokens:
            return self._project_id_for_root(self.workspace.root)
        slug = self._project_slug_from_tokens(identity_tokens) or "project"
        digest = hashlib.sha1("|".join(identity_tokens).encode("utf-8")).hexdigest()[:10]
        return f"{slug}-{digest}"

    def _git_identity_tokens(self) -> list[str]:
        git_path = self.workspace.root / ".git"
        if not git_path.exists():
            return []
        git_dir = git_path
        if git_path.is_file():
            try:
                raw = git_path.read_text(encoding="utf-8")
            except Exception:
                return []
            prefix = "gitdir:"
            for line in raw.splitlines():
                if not line.lower().startswith(prefix):
                    continue
                candidate = line[len(prefix) :].strip()
                git_dir = (self.workspace.root / candidate).resolve(strict=False)
                break
        config_path = git_dir / "config"
        if not config_path.exists():
            return []
        try:
            text = config_path.read_text(encoding="utf-8")
        except Exception:
            return []
        tokens: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line.lower().startswith("url ="):
                continue
            value = line.split("=", 1)[1].strip()
            normalized = value.lower().replace("\\", "/")
            if normalized.endswith(".git"):
                normalized = normalized[:-4]
            tokens.append(f"git:{normalized}")
        return self._unique_strings(tokens)

    def _project_identity_tokens_from_snapshot(self, snapshot: Any | None) -> list[str]:
        if snapshot is None:
            return []
        tokens = [
            *[f"manifest:{Path(path).name.lower()}" for path in list(getattr(snapshot, "manifests", []) or [])[:8]],
            *[f"entry:{Path(path).name.lower()}" for path in list(getattr(snapshot, "entrypoints", []) or [])[:8]],
            *[f"dir:{str(path).lower()}" for path in list(getattr(snapshot, "top_directories", []) or [])[:8]],
            *[f"test:{Path(path).stem.lower()}" for path in list(getattr(snapshot, "test_files", []) or [])[:8]],
        ]
        return self._unique_strings(tokens)

    def _workspace_identity_tokens(self) -> list[str]:
        files = self.workspace.iter_files(max_results=256)
        relative_paths: list[str] = []
        for path in files:
            try:
                relative_paths.append(path.relative_to(self.workspace.root).as_posix())
            except ValueError:
                continue
        if not relative_paths:
            return []
        top_dirs = sorted({path.split("/", 1)[0].lower() for path in relative_paths if "/" in path})[:8]
        manifests = [
            f"manifest:{Path(path).name.lower()}"
            for path in relative_paths
            if Path(path).name.lower() in MANIFEST_FILES
        ][:8]
        entrypoints = [
            f"entry:{Path(path).name.lower()}"
            for path in relative_paths
            if Path(path).name.lower() in ENTRYPOINT_NAMES
        ][:8]
        tests = [
            f"test:{Path(path).stem.lower()}"
            for path in relative_paths
            if self._looks_like_test_path(path)
        ][:8]
        tokens = [
            *manifests,
            *entrypoints,
            *[f"dir:{item}" for item in top_dirs],
            *tests,
        ]
        if len(tokens) < 3:
            tokens.extend(f"path:{item.lower()}" for item in relative_paths[:18])
        return self._unique_strings(tokens)

    def _workspace_identity_tokens_are_stable(self, tokens: list[str]) -> bool:
        if not tokens:
            return False
        structural_prefixes = ("manifest:", "entry:", "dir:", "test:")
        if any(str(token).startswith(structural_prefixes) for token in tokens):
            return True
        path_tokens = [
            str(token).split(":", 1)[1]
            for token in tokens
            if str(token).startswith("path:") and ":" in str(token)
        ]
        return any("/" in path for path in path_tokens)

    def _project_slug_from_tokens(self, tokens: list[str]) -> str:
        for token in tokens:
            normalized = str(token or "").strip().lower()
            if normalized.startswith("git:"):
                tail = normalized.rsplit("/", 1)[-1]
                candidate = re_sub_non_alnum(Path(tail).stem)
                if candidate:
                    return candidate
            candidate = normalized.split(":", 1)[-1]
            candidate = re_sub_non_alnum(Path(candidate).stem)
            if candidate and candidate not in {"readme", "project", "workspace", "src", "app", "tests", "test"}:
                return candidate
        return "project"

    def _project_id_for_root(self, root: Path) -> str:
        resolved = root.expanduser().resolve()
        slug = re_sub_non_alnum(resolved.name.lower()) or "workspace"
        digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:10]
        return f"{slug}-{digest}"

    def _episodic_dedupe_key(
        self,
        project_id: str,
        problem_type: str,
        final_outcome: str,
        changed_files: list[str],
        failure_signatures: list[str],
    ) -> str:
        normalized_targets = [Path(path).name.lower() for path in changed_files[:4]]
        normalized_failures = [str(item or "").strip().lower() for item in failure_signatures[:3]]
        normalized_summary_terms = self._terms_from_text(final_outcome)[:8]
        basis = "|".join(
            [
                project_id,
                problem_type,
                *normalized_targets,
                *normalized_failures,
                *normalized_summary_terms,
            ]
        )
        return f"episode:{self._hash_text(basis)}"

    def _failure_dedupe_key(self, project_id: str, failure_signature: str, targets: list[str]) -> str:
        basis = "|".join([project_id, failure_signature, *targets])
        return f"failure:{self._hash_text(basis)}"

    def _hash_text(self, text: str) -> str:
        return hashlib.sha1(str(text or "").encode("utf-8")).hexdigest()[:12]

    def _merge_symbol_indexes(
        self,
        left: dict[str, list[str]],
        right: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        merged: dict[str, list[str]] = {}
        for path, symbols in list(left.items()) + list(right.items()):
            existing = merged.get(path, [])
            merged[path] = self._unique_strings([*existing, *list(symbols or [])])[:8]
        return merged

    def _project_co_change_hints(self, project_id: str) -> list[str]:
        counts: defaultdict[tuple[str, str], int] = defaultdict(int)
        for entry in self.list_entries("episodic", project_id=project_id):
            changed = [
                str(path or "").strip()
                for path in list(getattr(entry, "changed_files", []) or [])[:8]
                if str(path or "").strip()
            ]
            for index, left in enumerate(changed):
                for right in changed[index + 1 :]:
                    pair = tuple(sorted((left, right)))
                    counts[pair] += 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [f"{left} <-> {right} ({count}x)" for (left, right), count in ranked[:6]]

    def _age_days(self, timestamp: str) -> float:
        try:
            parsed = datetime.fromisoformat(timestamp)
        except ValueError:
            return 0.0
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)
        return delta.total_seconds() / 86400.0

    def _failure_signature_from_context(self, repair_context: Any | None) -> str:
        if repair_context is None:
            return ""
        repair_brief = getattr(repair_context, "repair_brief", None)
        return (
            str(getattr(repair_brief, "failure_signature", "") or "").strip()
            or str(getattr(repair_context, "failure_summary", "") or "").strip()
            or str(getattr(repair_context, "evidence_signature", "") or "").strip()
        )

    def _looks_like_test_path(self, path: str) -> bool:
        lowered = str(path or "").lower()
        name = Path(lowered).name
        return "/tests/" in f"/{lowered}/" or name.startswith("test_") or name.endswith("_test.py")

    def _prune_expired_entries(self) -> None:
        removed = False
        for memory_type in PERSISTENT_MEMORY_TYPES:
            folder = self.entries_root / memory_type
            if not folder.exists():
                continue
            for path in list(folder.glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                ttl_days = int(payload.get("ttl_days") or 0)
                if ttl_days <= 0:
                    continue
                age_days = self._age_days(str(payload.get("updated_at") or payload.get("created_at") or ""))
                last_accessed_at = str(payload.get("last_accessed_at") or "").strip()
                last_accessed_age = self._age_days(last_accessed_at) if last_accessed_at else math.inf
                if memory_type == "project":
                    continue
                hard_limit = float(ttl_days if memory_type == "conversation" else ttl_days * 2)
                if age_days <= hard_limit:
                    continue
                if last_accessed_age <= max(ttl_days / 2, 7):
                    continue
                try:
                    path.unlink()
                    removed = True
                except Exception:
                    continue
        if removed:
            self._index = self._rebuild_index_from_disk()
            self._write_index(self._index)

    def _infer_test_mappings(self, test_files: list[str], important_files: list[str]) -> list[tuple[str, str]]:
        mappings: list[tuple[str, str]] = []
        source_by_stem = {Path(path).stem.replace("test_", ""): path for path in important_files}
        for test_path in test_files[:12]:
            stem = Path(test_path).stem.replace("test_", "").replace("_test", "")
            source = source_by_stem.get(stem)
            if source:
                mappings.append((test_path, source))
        return mappings

    def _jaccard_similarity(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        intersection = len(left & right)
        union = len(left | right)
        return intersection / union if union else 0.0

    def _terms_from_text(self, text: str) -> list[str]:
        return prioritized_focus_terms(text, max_terms=32)

    def _candidate_reference_terms(
        self,
        candidate_paths: list[str],
        symbol_index: dict[str, list[str]],
    ) -> set[str]:
        reference_terms: set[str] = set()
        for path in candidate_paths:
            lowered = str(path or "").strip().lower()
            if not lowered:
                continue
            stem = Path(lowered).stem
            if len(stem) >= 3:
                reference_terms.add(stem)
            for raw in re.split(r"[^a-z0-9_]+", lowered):
                token = raw.strip("_")
                if len(token) >= 3:
                    reference_terms.add(token)
            for symbol in list(symbol_index.get(path, []) or [])[:8]:
                for raw in re.split(r"[^a-z0-9_]+", str(symbol or "").strip().lower()):
                    token = raw.strip("_")
                    if len(token) >= 3:
                        reference_terms.add(token)
        return reference_terms

    def _index_list_add(self, bucket: dict[str, list[str]], key: str, value: str) -> None:
        if key not in bucket:
            bucket[key] = [value]
            return
        if value not in bucket[key]:
            bucket[key].append(value)

    def _unique_strings(self, values: list[str]) -> list[str]:
        unique: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if not text or text in unique:
                continue
            unique.append(text)
        return unique

    def _trim_text(self, text: str, limit: int) -> str:
        normalized = str(text or "").strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 1].rstrip() + "…"


def re_sub_non_alnum(value: str) -> str:
    cleaned = []
    previous_dash = False
    for char in str(value or ""):
        if char.isalnum():
            cleaned.append(char)
            previous_dash = False
            continue
        if previous_dash:
            continue
        cleaned.append("-")
        previous_dash = True
    return "".join(cleaned).strip("-")
