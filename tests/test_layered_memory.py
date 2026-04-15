from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent.layered_memory import AgentMemoryStore
from agent.models import (
    ConversationMemoryEntry,
    EpisodicMemoryEntry,
    FailureMemoryEntry,
    FileChangeRecord,
    MemoryProvenance,
    MemorySummary,
    RepairAttemptRecord,
    RepairBrief,
    SessionState,
    ValidationFailureEvidence,
)
from agent.planner import ESCALATED_REPAIR_STRATEGY, Planner, TARGETED_REPAIR_STRATEGY
from agent.prompts import generate_content_prompt, task_state_update_prompt
from agent.task_schema import TaskArtifact
from agent.task_state import TaskState
from config.settings import AppConfig
from llm.schemas import AgentActionType, RouteActionName, RouteActionStep, RouteEntities, RouteIntent, RouterOutput
from runtime.logger import AgentLogger
from runtime.workspace import WorkspaceManager


class DummyLLM:
    def __init__(self, config: AppConfig):
        self.config = config

    def generate_json(self, *args, **kwargs):
        raise RuntimeError("generate_json is not used in this test")

    def generate(self, *args, **kwargs):
        raise RuntimeError("generate is not used in this test")


def build_store(tmp_path: Path, *, state_root_override: Path | None = None) -> AgentMemoryStore:
    (tmp_path / "app").mkdir(parents=True)
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "app" / "main.py").write_text("def login(user):\n    return user\n", encoding="utf-8")
    (tmp_path / "app" / "auth.py").write_text("def check_role(role):\n    return role == 'admin'\n", encoding="utf-8")
    (tmp_path / "tests" / "test_main.py").write_text("def test_login():\n    assert True\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    config = AppConfig(
        workspace_root=str(tmp_path),
        state_root_override=str(state_root_override) if state_root_override is not None else None,
    )
    config.ensure_state_dirs()
    return AgentMemoryStore(config, WorkspaceManager(tmp_path))


def build_task_state(task: str, *, target: str = "app/auth.py") -> TaskState:
    return TaskState(
        latest_user_turn=task,
        root_goal="Harden the auth flow",
        active_goal="Update the auth flow safely",
        goal_relation="continue",
        output_expectation="Apply the requested change and validate it.",
        current_user_intent="implement",
        execution_strategy="feature_implementation",
        verification_target="python -m pytest",
        target_artifacts=[TaskArtifact(path=target, kind="file", role="primary_target", confidence=0.92)],
        constraints=["keep existing behavior stable", "avoid broad rewrites"],
        confidence=0.88,
        next_action="modify",
        next_best_action="modify",
    )


def build_route(*, target: str = "app/auth.py", requested_outcome: str = "Update the auth flow safely.") -> RouterOutput:
    return RouterOutput(
        user_goal="Harden the auth flow",
        intent=RouteIntent.UPDATE,
        entities=RouteEntities(
            target_type="file",
            target_name=Path(target).name,
            target_paths=[target],
            constraints=["keep existing behavior stable"],
        ),
        requested_outcome=requested_outcome,
        action_plan=[
            RouteActionStep(step=1, action=RouteActionName.READ_RELEVANT_FILES, reason="Inspect the target."),
            RouteActionStep(step=2, action=RouteActionName.UPDATE_ARTIFACT, reason="Apply the change."),
            RouteActionStep(step=3, action=RouteActionName.RUN_VALIDATION, reason="Validate the result."),
        ],
        needs_clarification=False,
        clarification_questions=[],
        confidence=0.91,
        safe_to_execute=True,
        repo_context_needed=True,
        search_terms=["auth", "role", "login"],
        relevant_extensions=[".py"],
        direct_response=None,
    )


def build_session(store: AgentMemoryStore, task: str = "Harden the auth flow") -> SessionState:
    session = SessionState(
        task=task,
        workspace_root=str(store.workspace.root),
        project_id=store.project_id,
        candidate_files=["app/auth.py", "tests/test_main.py"],
    )
    session.task_state = build_task_state(task)
    session.router_result = build_route()
    session.workspace_snapshot = store.build_snapshot(task)
    return session


def build_fresh_start_session(store: AgentMemoryStore, task: str = "Harden the auth flow") -> SessionState:
    session = SessionState(
        task=task,
        workspace_root=str(store.workspace.root),
        project_id=store.project_id,
        candidate_files=["app/auth.py", "tests/test_main.py"],
    )
    session.workspace_snapshot = store.build_snapshot(task)
    return session


def build_failure_context(signature: str = "role-cache-mismatch", *, target: str = "app/auth.py") -> ValidationFailureEvidence:
    return ValidationFailureEvidence(
        command="python -m pytest",
        verification_scope="runtime",
        status="failed",
        artifact_paths=[target],
        summary="Role cache update does not propagate to the auth check.",
        excerpt="AssertionError: expected admin role update to be visible after cache refresh",
        failure_summary="role cache update is not visible to the auth check",
        file_hints=[target, "tests/test_main.py"],
        action_hints=["repair the role cache propagation"],
        repair_requirements=["preserve existing login behavior"],
        evidence_signature=f"sig-{signature}",
        repair_brief=RepairBrief(
            failure_signature=signature,
            primary_target=target,
            locked_target=target,
            expected_semantics=["role updates become visible immediately"],
            observed_semantics=["old role stays cached"],
            implicated_symbols=["check_role", "role_cache"],
            implicated_region_hint="auth cache handling",
        ),
    )


def make_episode(
    store: AgentMemoryStore,
    *,
    session_id: str,
    title: str,
    summary: str,
    file_paths: list[str],
    project_id: str | None = None,
    updated_at: str | None = None,
) -> EpisodicMemoryEntry:
    entry = EpisodicMemoryEntry(
        project_id=project_id or store.project_id,
        workspace_root=str(store.workspace.root),
        session_id=session_id,
        summary=MemorySummary(
            title=title,
            summary=summary,
            key_points=["small targeted fix", "validation passed"],
            why_relevant="Similar historical task.",
        ),
        provenance=MemoryProvenance(
            source_type="session",
            session_id=session_id,
            project_id=project_id or store.project_id,
            workspace_root=str(store.workspace.root),
        ),
        file_paths=file_paths,
        tags=["repair", "auth"],
        problem_type="repair",
        strategy_used=["validation_targeted"],
        result="completed",
        what_worked=["target the runtime path before editing tests"],
        what_failed=["editing the test first"],
        final_outcome=summary,
        changed_files=file_paths,
        failure_signatures=["role-cache-mismatch"],
        dedupe_key=f"episode:{project_id or store.project_id}:{session_id}",
    )
    if updated_at is not None:
        entry = entry.model_copy(update={"updated_at": updated_at, "created_at": updated_at})
    return entry


def make_failure_entry(
    store: AgentMemoryStore,
    *,
    signature: str = "role-cache-mismatch",
    project_id: str | None = None,
    bad_patterns: list[str] | None = None,
    good_patterns: list[str] | None = None,
    updated_at: str | None = None,
) -> FailureMemoryEntry:
    entry = FailureMemoryEntry(
        project_id=project_id or store.project_id,
        workspace_root=str(store.workspace.root),
        session_id="failure-session",
        summary=MemorySummary(
            title="Role cache mismatch",
            summary="Runtime auth checks still see the old cached role after updates.",
            key_points=["runtime failure", "auth cache"],
            why_relevant="Recurring repair pattern.",
        ),
        provenance=MemoryProvenance(
            source_type="repair",
            session_id="failure-session",
            project_id=project_id or store.project_id,
            workspace_root=str(store.workspace.root),
            command="python -m pytest",
        ),
        file_paths=["app/auth.py", "tests/test_main.py"],
        tags=["repair", "runtime"],
        dedupe_key=f"failure:{project_id or store.project_id}:{signature}",
        failure_signature=signature,
        expected_semantics=["role updates are visible immediately"],
        observed_semantics=["old role remains cached"],
        chosen_targets=["app/auth.py"],
        tried_strategies=["validation_targeted", ESCALATED_REPAIR_STRATEGY],
        successful_repair_patterns=list(
            [ESCALATED_REPAIR_STRATEGY] if good_patterns is None else good_patterns
        ),
        bad_retry_patterns=list(["validation_targeted"] if bad_patterns is None else bad_patterns),
        review_rejection_reasons=["test-only change did not fix runtime path"],
        no_effective_change_count=1,
        last_result="mutation_planned",
    )
    if updated_at is not None:
        entry = entry.model_copy(update={"updated_at": updated_at, "created_at": updated_at})
    return entry


def make_conversation_entry(store: AgentMemoryStore, *, session_id: str, summary: str) -> ConversationMemoryEntry:
    return ConversationMemoryEntry(
        project_id=store.project_id,
        workspace_root=str(store.workspace.root),
        session_id=session_id,
        summary=MemorySummary(
            title="Auth recall",
            summary=summary,
            key_points=["user asked for recall"],
            why_relevant="User-facing reminder context.",
        ),
        provenance=MemoryProvenance(
            source_type="conversation",
            session_id=session_id,
            project_id=store.project_id,
            workspace_root=str(store.workspace.root),
        ),
        file_paths=["app/auth.py"],
        tags=["recall"],
        dedupe_key=f"conversation:{store.project_id}:{session_id}",
        request_summary="What was our last auth status?",
        delivered_summary=summary,
        projects_touched=[store.project_id],
        decision_notes=["Kept the recall concise."],
        implemented_features=["auth hardening"],
        referenced_sessions=[session_id],
    )


def test_working_memory_stays_relevant_and_compact_for_active_run(tmp_path):
    store = build_store(tmp_path)
    session = build_session(store)
    session.candidate_files = [f"app/file_{index}.py" for index in range(20)] + ["app/auth.py"]
    session.changed_files = [FileChangeRecord(path="app/auth.py", operation="update")]
    session.active_repair_context = build_failure_context()
    session.notes = [f"note {index}" for index in range(10)]
    session.tool_calls = []

    working = store.build_working_memory(session)

    assert working.current_goal == "Update the auth flow safely"
    assert working.primary_target == "app/auth.py"
    assert len(working.relevant_files) <= 8
    assert len(working.active_constraints) <= 6
    assert len(working.compact_state_summary) <= 340
    assert "app/auth.py" in working.relevant_files


def test_new_session_does_not_inherit_transient_working_ballast(tmp_path):
    store = build_store(tmp_path)
    previous = build_session(store, "Fix auth bug")
    previous.active_repair_context = build_failure_context()
    previous.changed_files = [FileChangeRecord(path="app/auth.py", operation="update")]
    previous.notes = ["first note", "second note"]
    store.refresh_session_memory(previous.task, previous)
    store.persist_session_memory(previous)

    fresh = build_session(store, "Check the auth flow again")
    store.refresh_session_memory(fresh.task, fresh)

    assert fresh.working_memory is not None
    assert fresh.working_memory.recent_attempts == []
    assert fresh.working_memory.recent_failures == []
    assert all(item.memory_type != "working" for item in fresh.memory_context.selected)


def test_fresh_semantic_start_limits_persistent_retrieval_to_project_memory(tmp_path):
    store = build_store(tmp_path)
    seeded = build_session(store, "Fix auth bug")
    store.refresh_session_memory(seeded.task, seeded)
    store.persist_session_memory(seeded)
    store.upsert_entry(
        make_conversation_entry(
            store,
            session_id="recall-chat",
            summary="Last time we only needed to rerun the validation command.",
        )
    )

    fresh = build_fresh_start_session(store, "Add audit logging to the auth flow")
    store.refresh_session_memory(fresh.task, fresh)

    assert fresh.memory_context is not None
    assert fresh.memory_context.request.include_types == ["project"]
    assert all(item.memory_type == "project" for item in fresh.memory_context.selected)


def test_fresh_semantic_start_prompt_excludes_prior_conversation_recall(tmp_path):
    store = build_store(tmp_path)
    seeded = build_session(store, "Fix auth bug")
    store.refresh_session_memory(seeded.task, seeded)
    store.persist_session_memory(seeded)
    store.upsert_entry(
        make_conversation_entry(
            store,
            session_id="recall-chat",
            summary="Last time we only needed to rerun the validation command.",
        )
    )

    fresh = build_fresh_start_session(store, "Add audit logging to the auth flow")
    store.refresh_session_memory(fresh.task, fresh)
    prompt = task_state_update_prompt(fresh.task, snapshot=fresh.workspace_snapshot, session=fresh, mode="compact")

    assert "Memory context:" in prompt
    assert "Last time we only needed to rerun the validation command." not in prompt
    assert '"repo_map_hints"' in prompt or '"suggested_files"' in prompt


def test_episodic_memory_recalls_similar_prior_case(tmp_path):
    store = build_store(tmp_path)
    store.upsert_entry(
        make_episode(
            store,
            session_id="episode-1",
            title="Inventory role cache fix",
            summary="We fixed the inventory role cache by updating the runtime auth path instead of patching the test.",
            file_paths=["app/auth.py", "tests/test_main.py"],
        )
    )

    result = store.retrieve(
        store.build_retrieval_request(
            "Haben wir schon mal das Inventory Rollenproblem im Auth-Flow gefixt?",
            build_session(store),
        )
    )

    assert result.selected
    assert result.selected[0].memory_type == "episodic"
    assert result.selected[0].session_id == "episode-1"


def test_project_memory_prefers_same_project_context_only(tmp_path):
    store = build_store(tmp_path)
    session = build_session(store)
    store.persist_session_memory(session)
    other_entry = make_episode(
        store,
        session_id="other-project-session",
        title="Unrelated billing fix",
        summary="Billing permissions cleanup in another project.",
        file_paths=["app/billing.py"],
        project_id="other-project",
    )
    store.upsert_entry(other_entry)

    request = store.build_retrieval_request("What are the important auth files in this project?", session)
    request = request.model_copy(update={"use_case": "project_context", "allow_cross_project": False})
    result = store.retrieve(request)

    assert result.selected
    assert all(item.project_id == store.project_id for item in result.selected)
    assert any(item.memory_type == "project" for item in result.selected)


def test_failure_memory_guides_repair_strategy_order(tmp_path):
    store = build_store(tmp_path)
    store.upsert_entry(make_failure_entry(store))
    session = build_session(store, "Fix the auth cache mismatch")
    session.active_repair_context = build_failure_context()
    store.refresh_session_memory(session.task, session)
    planner = Planner(DummyLLM(AppConfig(workspace_root=str(tmp_path))), "tools", logger=AgentLogger(store.config.log_dir_path, "memtest"))

    strategies = planner._repair_generation_strategies(session, session.active_repair_context, "app/auth.py")

    assert strategies == [ESCALATED_REPAIR_STRATEGY]


def test_failure_memory_does_not_exhaust_fresh_session_before_local_repair_attempt(tmp_path):
    store = build_store(tmp_path)
    store.upsert_entry(
        make_failure_entry(
            store,
            bad_patterns=[TARGETED_REPAIR_STRATEGY, ESCALATED_REPAIR_STRATEGY],
            good_patterns=[],
        )
    )
    session = build_session(store, "Fix the auth cache mismatch")
    session.active_repair_context = build_failure_context()
    store.refresh_session_memory(session.task, session)
    planner = Planner(
        DummyLLM(AppConfig(workspace_root=str(tmp_path))),
        "tools",
        logger=AgentLogger(store.config.log_dir_path, "memtest-fresh-repair"),
    )

    strategies = planner._repair_generation_strategies(session, session.active_repair_context, "app/auth.py")

    assert strategies == [TARGETED_REPAIR_STRATEGY, ESCALATED_REPAIR_STRATEGY]


def test_memory_retrieval_respects_hit_and_summary_budgets(tmp_path):
    store = build_store(tmp_path)
    for index in range(8):
        store.upsert_entry(
            make_episode(
                store,
                session_id=f"episode-{index}",
                title=f"Auth fix {index}",
                summary=f"Historical auth fix {index} with compact context.",
                file_paths=["app/auth.py"],
            )
        )

    request = store.build_retrieval_request("auth fix", build_session(store))
    request = request.model_copy(update={"max_hits": 3, "summary_budget_chars": 260})
    result = store.retrieve(request)

    assert len(result.selected) <= 3
    assert len(result.summary) <= 260
    assert result.prompt_char_cost <= 260


def test_duplicate_memory_entries_merge_instead_of_accumulating(tmp_path):
    store = build_store(tmp_path)
    store.upsert_entry(make_failure_entry(store, bad_patterns=["validation_targeted"]))
    store.upsert_entry(make_failure_entry(store, bad_patterns=["validation_targeted", "another_pattern"]))

    failures = store.list_entries("failure")

    assert len(failures) == 1
    assert failures[0].duplicate_count >= 1
    assert "another_pattern" in failures[0].bad_retry_patterns


def test_user_recall_query_returns_concise_grounded_summary(tmp_path):
    store = build_store(tmp_path)
    store.upsert_entry(
        make_episode(
            store,
            session_id="recall-episode",
            title="Previous auth hardening",
            summary="We stabilized the auth cache and revalidated with pytest.",
            file_paths=["app/auth.py"],
        )
    )
    store.upsert_entry(make_conversation_entry(store, session_id="recall-chat", summary="Last time we fixed the auth cache drift."))
    request = store.build_retrieval_request("Haben wir das nicht schon mal gemacht?", build_session(store))
    request = request.model_copy(update={"use_case": "user_recall"})

    result = store.retrieve(request)

    assert len(result.summary) <= 900
    assert "session=" in result.summary
    assert result.related_sessions


def test_old_irrelevant_memory_does_not_displace_current_project_signal(tmp_path):
    store = build_store(tmp_path)
    old_timestamp = (datetime.now(timezone.utc) - timedelta(days=800)).isoformat()
    store.upsert_entry(
        make_episode(
            store,
            session_id="very-old-other",
            title="Ancient auth fix",
            summary="Old auth fix from another project with similar words.",
            file_paths=["app/auth.py"],
            project_id="other-project",
            updated_at=old_timestamp,
        )
    )
    store.persist_session_memory(build_session(store))

    request = store.build_retrieval_request("auth fix app/auth.py", build_session(store))
    request = request.model_copy(update={"allow_cross_project": True, "use_case": "similar_task_lookup"})
    result = store.retrieve(request)

    assert result.selected
    assert result.selected[0].project_id == store.project_id


def test_prompt_memory_context_stays_bounded(tmp_path):
    store = build_store(tmp_path)
    session = build_session(store, "Refactor the auth flow")
    long_summary = "bounded-context " * 200 + "tail-marker-should-not-appear"
    store.upsert_entry(
        make_episode(
            store,
            session_id="long-memory",
            title="Very long historical note",
            summary=long_summary,
            file_paths=["app/auth.py"],
        )
    )
    store.refresh_session_memory(session.task, session)
    prompt = generate_content_prompt(
        build_route(),
        session,
        path="app/auth.py",
        current_content="def check_role(role):\n    return role == 'admin'\n",
    )

    assert "Memory context:" in prompt
    assert "tail-marker-should-not-appear" not in prompt
    assert len(prompt) < 9000


def test_retrieval_prefers_exact_project_and_entity_matches(tmp_path):
    store = build_store(tmp_path)
    store.upsert_entry(
        make_episode(
            store,
            session_id="exact-match",
            title="Auth file fix",
            summary="We fixed app/auth.py directly.",
            file_paths=["app/auth.py"],
        )
    )
    store.upsert_entry(
        make_episode(
            store,
            session_id="fuzzy-match",
            title="Other auth fix",
            summary="We changed a related auth area elsewhere.",
            file_paths=["app/main.py"],
        )
    )

    request = store.build_retrieval_request("fix auth", build_session(store))
    request = request.model_copy(update={"target_paths": ["app/auth.py"]})
    result = store.retrieve(request)

    assert result.selected
    assert result.selected[0].session_id == "exact-match"


def test_cross_session_recall_finds_previous_work_for_have_we_done_this(tmp_path):
    store = build_store(tmp_path)
    previous = build_session(store, "Make the auth flow more robust")
    previous.changed_files = [FileChangeRecord(path="app/auth.py", operation="update")]
    previous.final_response = "I hardened the auth flow and validated it."
    previous.status = "completed"
    previous.validation_status = "passed"
    store.refresh_session_memory(previous.task, previous)
    store.persist_session_memory(previous)

    current = build_session(store, "Haben wir das schon mal beim Auth-Flow gemacht?")
    store.refresh_session_memory(current.task, current)

    assert current.memory_context.related_sessions
    assert previous.id in current.memory_context.related_sessions


def test_failure_memory_merges_known_bad_retry_patterns(tmp_path):
    store = build_store(tmp_path)
    store.upsert_entry(make_failure_entry(store, bad_patterns=["validation_targeted"]))
    store.upsert_entry(make_failure_entry(store, bad_patterns=["validation_targeted", "retry_without_new_evidence"]))

    failure = store.list_entries("failure")[0]

    assert "retry_without_new_evidence" in failure.bad_retry_patterns
    assert failure.no_effective_change_count >= 2


def test_project_memory_exposes_relevant_repo_hints_without_noise(tmp_path):
    store = build_store(tmp_path)
    session = build_session(store, "Inspect the auth architecture")
    store.persist_session_memory(session)

    project_entries = store.list_entries("project")

    assert len(project_entries) == 1
    assert len(project_entries[0].workflow_hints) <= 6


def test_repo_map_signal_bundle_prefers_symbol_and_test_mappings(tmp_path):
    store = build_store(tmp_path)
    session = build_session(store, "Trace check_role behavior in the auth tests")
    session.workspace_snapshot = session.workspace_snapshot.model_copy(
        update={
            "test_mappings": ["tests/test_main.py -> app/main.py"],
            "symbol_index": {
                "app/auth.py": ["check_role"],
                "app/main.py": ["login"],
            },
            "service_files": ["app/auth.py"],
            "import_hotspots": ["app/main.py"],
        }
    )

    store.refresh_session_memory(session.task, session)

    assert session.memory_context is not None
    assert "app/auth.py" in session.memory_context.suggested_files
    assert "check_role" in session.memory_context.suggested_symbols
    assert any("tests/test_main.py -> app/main.py" in item for item in session.memory_context.repo_map_hints)


def test_project_memory_tracks_symbol_index_import_hotspots_and_co_change_hints(tmp_path):
    store = build_store(tmp_path)
    previous = build_session(store, "Fix auth flow")
    previous.changed_files = [
        FileChangeRecord(path="app/auth.py", operation="update"),
        FileChangeRecord(path="tests/test_main.py", operation="update"),
    ]
    previous.final_response = "Updated auth flow and tests."
    previous.status = "completed"
    previous.validation_status = "passed"
    store.refresh_session_memory(previous.task, previous)
    store.persist_session_memory(previous)

    session = build_session(store, "Inspect the auth architecture")
    session.workspace_snapshot = session.workspace_snapshot.model_copy(
        update={
            "symbol_index": {"app/auth.py": ["check_role"]},
            "import_hotspots": ["app/main.py"],
            "service_files": ["app/auth.py"],
        }
    )
    store.persist_session_memory(session)

    project_entry = store.list_entries("project")[0]

    assert project_entry.symbol_index["app/auth.py"] == ["check_role"]
    assert project_entry.import_hotspots == ["app/main.py"]
    assert project_entry.service_files == ["app/auth.py"]
    assert any("app/auth.py <-> tests/test_main.py" in item for item in project_entry.co_change_hints)
    assert "app/auth.py" in project_entry.known_hotspots


def test_project_identity_stays_stable_across_fresh_workspace_copies(tmp_path):
    store_a = build_store(tmp_path / "copy_a")
    store_b = build_store(tmp_path / "copy_b")

    assert store_a.project_id == store_b.project_id


def test_project_identity_does_not_drift_after_snapshot_refresh(tmp_path):
    store = build_store(tmp_path)
    original_project_id = store.project_id
    session = build_session(store, "Harden the auth flow")

    store.refresh_session_memory(session.task, session)

    assert store.project_id == original_project_id
    assert session.project_id == original_project_id


def test_failure_memory_persists_multiple_failure_signatures_from_same_session(tmp_path):
    store = build_store(tmp_path)
    session = build_session(store, "Repair the auth flow and cache propagation")
    session.active_repair_context = build_failure_context("role-cache-mismatch", target="app/auth.py")
    session.repair_history.extend(
        [
            RepairAttemptRecord(
                artifact_path="app/auth.py",
                validation_command="python -m pytest",
                verification_scope="runtime",
                strategy="validation_targeted",
                result="no_effective_change",
                reason="The first retry kept the stale cache behavior.",
                failure_signature="role-cache-mismatch",
            ),
            RepairAttemptRecord(
                artifact_path="app/main.py",
                validation_command="python -m pytest",
                verification_scope="runtime",
                strategy="validation_escalated",
                result="blocked",
                reason="The CLI bootstrap still fails before the auth module is reached.",
                failure_signature="cli-bootstrap-mismatch",
            ),
        ]
    )
    store.refresh_session_memory(session.task, session)
    store.persist_session_memory(session)

    failures = store.list_entries("failure")
    signatures = {item.failure_signature for item in failures}

    assert "role-cache-mismatch" in signatures
    assert "cli-bootstrap-mismatch" in signatures


def test_refresh_session_memory_reuses_cached_retrieval_when_request_is_unchanged(tmp_path, monkeypatch):
    store = build_store(tmp_path)
    session = build_session(store, "Harden the auth flow")
    original_retrieve = store.retrieve
    calls = {"count": 0}

    def counting_retrieve(request):
        calls["count"] += 1
        return original_retrieve(request)

    monkeypatch.setattr(store, "retrieve", counting_retrieve)

    store.refresh_session_memory(session.task, session)
    store.refresh_session_memory(session.task, session)

    assert calls["count"] == 1


def test_retrieve_updates_last_accessed_at_for_selected_entries(tmp_path):
    store = build_store(tmp_path)
    store.upsert_entry(
        make_episode(
            store,
            session_id="accessed-episode",
            title="Auth cache access record",
            summary="We previously fixed the auth cache visibility bug.",
            file_paths=["app/auth.py"],
        )
    )

    result = store.retrieve(store.build_retrieval_request("auth cache access record", build_session(store)))
    refreshed = store.list_entries("episodic")[0]

    assert result.selected
    assert refreshed.last_accessed_at is not None


def test_episodic_memory_deduplicates_repeated_similar_sessions(tmp_path):
    store = build_store(tmp_path)
    for response in (
        "I hardened the auth flow and revalidated it.",
        "I hardened the auth flow and revalidated it.",
    ):
        session = build_session(store, "Harden the auth flow")
        session.changed_files = [FileChangeRecord(path="app/auth.py", operation="update")]
        session.final_response = response
        session.status = "completed"
        session.validation_status = "passed"
        store.refresh_session_memory(session.task, session)
        store.persist_session_memory(session)

    episodes = store.list_entries("episodic")

    assert len(episodes) == 1
    assert episodes[0].duplicate_count >= 1


def test_planner_candidate_paths_include_memory_guided_files(tmp_path):
    store = build_store(tmp_path)
    store.upsert_entry(
        make_episode(
            store,
            session_id="readme-episode",
            title="Auth flow + README follow-up",
            summary="We updated app/auth.py and README.md together.",
            file_paths=["app/auth.py", "README.md"],
        )
    )
    session = build_session(store, "Harden the auth flow and keep docs aligned")
    store.refresh_session_memory(session.task, session)
    planner = Planner(DummyLLM(AppConfig(workspace_root=str(tmp_path))), "tools", logger=AgentLogger(store.config.log_dir_path, "memory-guided"))

    candidates = planner._candidate_paths(session.router_result, session)

    assert "README.md" in candidates


def test_planner_answers_user_recall_queries_from_memory(tmp_path):
    store = build_store(tmp_path)
    store.upsert_entry(
        make_episode(
            store,
            session_id="recall-session",
            title="Previous auth hardening",
            summary="We stabilized the auth cache and validated with pytest.",
            file_paths=["app/auth.py"],
        )
    )
    session = build_session(store, "Haben wir das schon mal beim Auth-Flow gemacht?")
    session.router_result = RouterOutput(
        user_goal="Recall previous auth work",
        intent=RouteIntent.EXPLAIN,
        entities=RouteEntities(
            target_type=None,
            target_name=None,
            target_paths=[],
            attributes=[],
            constraints=[],
        ),
        requested_outcome="Summarize previous auth work.",
        action_plan=[
            RouteActionStep(step=1, action=RouteActionName.RESPOND_DIRECTLY, reason="Answer from memory."),
        ],
        needs_clarification=False,
        clarification_questions=[],
        confidence=0.92,
        safe_to_execute=True,
        repo_context_needed=False,
        search_terms=["auth", "recall"],
        relevant_extensions=[],
        direct_response=None,
    )
    store.refresh_session_memory(session.task, session)
    planner = Planner(DummyLLM(AppConfig(workspace_root=str(tmp_path))), "tools", logger=AgentLogger(store.config.log_dir_path, "memory-recall"))

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.FINAL
    assert "relevante fruehere Arbeit" in str(decision.final_response or "")


def test_cross_project_personal_name_recall_round_trips_from_shared_memory(tmp_path):
    shared_state = tmp_path / ".shared_state"
    store_a = build_store(tmp_path / "workspace_a", state_root_override=shared_state)

    remembered = SessionState(
        task="Bitte merk dir, dass ich Joshua Pond heisse.",
        workspace_root=str(store_a.workspace.root),
        project_id=store_a.project_id,
        status="completed",
        final_response="Ich merke mir das fuer spaetere Rueckfragen.",
    )
    remembered.append_message("user", remembered.task)
    remembered.append_message("assistant", remembered.final_response or "")
    remembered.workspace_snapshot = store_a.build_snapshot(remembered.task)
    store_a.persist_session_memory(remembered)
    store_b = build_store(tmp_path / "workspace_b", state_root_override=shared_state)

    recall = SessionState(
        task="Wer bin ich?",
        workspace_root=str(store_b.workspace.root),
        project_id=store_b.project_id,
    )
    recall.workspace_snapshot = store_b.build_snapshot(recall.task)

    request = store_b.build_retrieval_request(recall.task, recall)
    result = store_b.retrieve(request)

    assert request.use_case == "user_recall"
    assert request.recall_subject == "user"
    assert "name" in request.recall_attributes
    assert "Joshua Pond" in result.recall_brief


def test_repeated_personal_fact_deduplicates_across_projects(tmp_path):
    shared_state = tmp_path / ".shared_state"
    store_a = build_store(tmp_path / "workspace_a", state_root_override=shared_state)

    first = SessionState(
        task="Mein Name ist Joshua Pond.",
        workspace_root=str(store_a.workspace.root),
        project_id=store_a.project_id,
        status="completed",
        final_response="Verstanden.",
    )
    first.append_message("user", first.task)
    first.append_message("assistant", first.final_response or "")
    first.workspace_snapshot = store_a.build_snapshot(first.task)
    store_a.persist_session_memory(first)
    store_b = build_store(tmp_path / "workspace_b", state_root_override=shared_state)
    second = SessionState(
        task="Mein Name ist Joshua Pond.",
        workspace_root=str(store_b.workspace.root),
        project_id=store_b.project_id,
        status="completed",
        final_response="Verstanden.",
    )
    second.append_message("user", second.task)
    second.append_message("assistant", second.final_response or "")
    second.workspace_snapshot = store_b.build_snapshot(second.task)
    store_b.persist_session_memory(second)

    remembered_entries = [
        item
        for item in store_b.list_entries("conversation")
        if any(fact.attribute == "name" and fact.value == "Joshua Pond" for fact in item.remembered_facts)
    ]

    assert len(remembered_entries) == 1
    assert set(remembered_entries[0].projects_touched) == {store_a.project_id, store_b.project_id}


def test_planner_answers_personal_identity_questions_from_memory(tmp_path):
    shared_state = tmp_path / ".shared_state"
    store_a = build_store(tmp_path / "workspace_a", state_root_override=shared_state)

    remembered = SessionState(
        task="Ich heisse Joshua Pond.",
        workspace_root=str(store_a.workspace.root),
        project_id=store_a.project_id,
        status="completed",
        final_response="Verstanden.",
    )
    remembered.append_message("user", remembered.task)
    remembered.append_message("assistant", remembered.final_response or "")
    remembered.workspace_snapshot = store_a.build_snapshot(remembered.task)
    store_a.persist_session_memory(remembered)
    store_b = build_store(tmp_path / "workspace_b", state_root_override=shared_state)

    session = SessionState(
        task="Wer bin ich?",
        workspace_root=str(store_b.workspace.root),
        project_id=store_b.project_id,
    )
    session.workspace_snapshot = store_b.build_snapshot(session.task)
    session.task_state = TaskState(
        latest_user_turn=session.task,
        root_goal=session.task,
        active_goal="Answer the identity question from persistent memory.",
        goal_relation="new_task",
        output_expectation="Return the remembered identity directly and concisely.",
        current_user_intent="explain",
        execution_strategy="validation_inspection",
        confidence=0.88,
        next_action="explain",
        next_best_action="explain",
    )
    session.router_result = RouterOutput(
        user_goal="Recall the user's identity",
        intent=RouteIntent.EXPLAIN,
        entities=RouteEntities(
            target_type=None,
            target_name=None,
            target_paths=[],
            attributes=[],
            constraints=[],
        ),
        requested_outcome="Answer the identity question from memory.",
        action_plan=[
            RouteActionStep(step=1, action=RouteActionName.RESPOND_DIRECTLY, reason="Answer from memory."),
        ],
        needs_clarification=False,
        clarification_questions=[],
        confidence=0.92,
        safe_to_execute=True,
        repo_context_needed=False,
        search_terms=["identity"],
        relevant_extensions=[],
        direct_response=None,
    )
    store_b.refresh_session_memory(session.task, session)
    planner = Planner(DummyLLM(AppConfig(workspace_root=str(tmp_path))), "tools", logger=AgentLogger(store_b.config.log_dir_path, "memory-personal-recall"))

    decision = planner.decide_next_action(session.task, session)

    assert decision.action_type == AgentActionType.FINAL
    assert "Joshua Pond" in str(decision.final_response or "")
