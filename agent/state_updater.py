from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.prompts import task_state_system_prompt, task_state_update_prompt
from agent.semantic_defaults import (
    extract_scope_constraints,
    has_follow_up_reference,
    infer_artifact_name_hint,
    infer_requested_extension,
    is_structural_follow_up_request,
    is_clear_low_risk_build_request,
    looks_like_additive_request,
    looks_like_correction_request,
    looks_like_hardening_request,
    looks_like_problem_report,
    looks_like_scope_narrowing_request,
    looks_like_validation_request,
)
from agent.task_schema import TaskArtifact
from agent.task_state import EvidenceItem, TaskState
from llm.provider import LLMProvider
from runtime.logger import AgentLogger


class TaskStateUpdater:
    """Updates the central working task state from the latest turn and prior session context."""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        logger: AgentLogger | None = None,
        model_name: str | None = None,
        timeout: int = 20,
        num_ctx: int = 4096,
    ):
        self.llm = llm
        self.logger = logger
        self.model_name = model_name
        self.timeout = timeout
        self.num_ctx = num_ctx

    def update_task_state(
        self,
        user_input: str,
        *,
        snapshot=None,
        session=None,
    ) -> TaskState:
        payload: dict[str, Any] | None = None
        try:
            payload = self.llm.generate_json(
                task_state_update_prompt(user_input, snapshot=snapshot, session=session),
                system=task_state_system_prompt(),
                model=self.model_name,
                retries=0,
                timeout=self.timeout,
                num_ctx=self.num_ctx,
            )
            state = TaskState.model_validate(payload)
            self._log("task_state_updated", task_state=state.model_dump())
            return state
        except Exception as exc:
            self._log("task_state_fallback", error=str(exc), payload=payload or {})
            state = self._fallback_state(user_input, snapshot=snapshot, session=session)
            self._log("task_state_updated", task_state=state.model_dump(), source="fallback")
            return state

    def _fallback_state(self, user_input: str, *, snapshot=None, session=None) -> TaskState:
        request = str(user_input or "").strip() or "Unclear request"
        previous_root = ""
        previous_goal = ""
        previous_next_action = ""
        output_expectation = ""
        open_problem = ""
        verification_target = ""
        if session is not None and getattr(session, "task_state", None) is not None:
            previous_root = str(session.task_state.root_goal or "").strip()
            previous_goal = str(session.task_state.active_goal or "").strip()
            previous_next_action = str(session.task_state.next_action or "").strip()
            output_expectation = str(session.task_state.output_expectation or "").strip()
            open_problem = str(session.task_state.open_problem or "").strip()
            verification_target = str(session.task_state.verification_target or "").strip()
        if session is not None and getattr(session, "follow_up_context", None) is not None:
            follow_up = session.follow_up_context
            previous_root = previous_root or str(follow_up.previous_root_goal or "").strip()
            previous_goal = previous_goal or str(
                follow_up.previous_active_goal
                or follow_up.previous_interpreted_goal
                or follow_up.previous_task
                or ""
            ).strip()
            previous_next_action = previous_next_action or self._normalize_next_action(
                follow_up.previous_next_action or follow_up.previous_recommended_mode
            )
            output_expectation = output_expectation or str(follow_up.previous_requested_outcome or "").strip()

        target_artifacts = self._collect_target_artifacts(session)
        evidence = self._collect_evidence(session)
        evidence_paths = self._unique_strings(
            [item.artifact_path for item in evidence if item.artifact_path]
        )
        artifact_paths = self._unique_strings([item.path for item in target_artifacts if item.path])
        has_failure_evidence = any(
            item.kind in {"diagnostic", "test", "log"}
            for item in evidence
        )
        ambiguous_targets = len(artifact_paths) > 1 and not evidence_paths
        has_active_context = bool(previous_root or previous_goal or target_artifacts or evidence)
        is_follow_up = has_active_context and is_structural_follow_up_request(
            request,
            previous_goal=previous_root or previous_goal,
            artifact_paths=artifact_paths,
        )
        clear_create = is_clear_low_risk_build_request(request)

        if clear_create and not is_follow_up:
            create_state = self._initial_create_state(request, snapshot=snapshot)
            if create_state is not None:
                return create_state

        if is_follow_up and self._should_create_follow_up_artifact(request):
            return self._follow_up_create_state(
                request,
                previous_root=previous_root or previous_goal or request,
                previous_goal=previous_goal or previous_root or request,
                output_expectation=output_expectation,
                verification_target=verification_target,
                existing_artifacts=target_artifacts,
            )

        if not has_active_context:
            return TaskState(
                latest_user_turn=request,
                root_goal=request,
                active_goal=request,
                goal_relation="unknown",
                output_expectation="Clarify the active target and decide the safest next step.",
                open_problem=None,
                verification_target=None,
                target_artifacts=[],
                evidence=[],
                relevant_context=["No reliable prior task context is available."],
                constraints=[],
                assumptions=[],
                missing_info=["The exact target is still ambiguous."],
                ambiguity_level="high",
                risk_level="medium",
                confidence=0.3,
                next_action="clarify",
                execution_outline=[
                    "Ask for the concrete target artifact or area.",
                ],
                needs_clarification=True,
                clarification_questions=["Welchen konkreten Bereich oder welches Artefakt soll ich als naechstes bearbeiten?"],
            )

        root_goal = previous_root or request
        relation = "continue" if previous_root else "new_task"
        assumptions: list[str] = []
        relevant_context: list[str] = []
        constraints: list[str] = []
        missing_info: list[str] = []
        next_action = "inspect"
        ambiguity_level = "medium"
        confidence = 0.55
        risk_level = "medium"
        current_output_expectation = output_expectation or ""
        current_open_problem = open_problem or None
        current_verification_target = verification_target or None
        execution_outline: list[str] = []

        if previous_root:
            assumptions.append("Continue from the existing root goal unless the user clearly changed topic.")
            relevant_context.append(f"Existing root goal: {previous_root}")
        if output_expectation:
            relevant_context.append(f"Previous expected outcome: {output_expectation}")

        if has_failure_evidence or (has_active_context and looks_like_problem_report(request)):
            relation = "report_problem" if previous_root else relation
            next_action = "debug"
            active_goal = (
                f"Diagnose and resolve the currently open issue affecting {previous_root}."
                if previous_root
                else "Diagnose the currently open issue from the available evidence."
            )
            current_output_expectation = "Diagnose the current failure, apply the smallest safe fix, and confirm the repaired path."
            current_open_problem = self._summarize_open_problem(evidence, fallback=open_problem or request)
            current_verification_target = current_verification_target or "Reproduce the failing path, apply the smallest safe fix, and rerun the relevant validation."
            execution_outline = [
                "Inspect the artifact most strongly implicated by existing evidence.",
                "Reproduce or validate the failing path with the most relevant command.",
                "Fix the highest-evidence cause and rerun verification.",
            ]
            confidence = 0.68 if len(evidence_paths) <= 1 and artifact_paths else 0.58
            ambiguity_level = "low" if len(evidence_paths) == 1 else "medium"
        elif has_active_context and looks_like_scope_narrowing_request(request):
            scope_constraints = extract_scope_constraints(request)
            relation = "scope_change"
            constraints.extend(scope_constraints)
            target_artifacts = self._filter_artifacts_for_scope(target_artifacts, scope_constraints) or target_artifacts
            active_goal = self._scope_constrained_goal(previous_goal or previous_root or request, scope_constraints)
            current_output_expectation = self._scope_output_expectation(scope_constraints)
            next_action = "modify"
            current_verification_target = current_verification_target or self._scope_verification_target(scope_constraints)
            execution_outline = [
                "Inspect the active implementation and identify only the artifacts that match the narrowed scope.",
                "Apply the requested change only inside the narrowed scope.",
                "Verify that only the constrained surface was affected.",
            ]
            confidence = 0.78 if len(target_artifacts) == 1 else 0.68
            ambiguity_level = "low" if len(target_artifacts) <= 1 else "medium"
        elif has_active_context and looks_like_hardening_request(request):
            relation = "refine"
            active_goal = f"Harden the active implementation around {previous_goal or previous_root or 'the current task'} without broad rewrites."
            current_output_expectation = "A safer and more robust version of the active implementation without widening scope."
            next_action = "modify"
            current_verification_target = current_verification_target or "Apply the highest-value hardening change and verify the intended behavior still works."
            execution_outline = [
                "Inspect the active implementation and identify the highest-risk weak spot.",
                "Apply a focused hardening change without broad rewrites.",
                "Verify that the intended behavior still works after the hardening update.",
            ]
            confidence = 0.76 if len(artifact_paths) <= 1 else 0.66
            ambiguity_level = "low" if len(artifact_paths) <= 1 else "medium"
        elif has_active_context and looks_like_validation_request(request):
            relation = "validation_request"
            active_goal = previous_goal or previous_root or request
            current_output_expectation = "A concrete validation result for the active implementation, including any failure details or limitations."
            next_action = "test"
            current_verification_target = current_verification_target or "Run the most relevant validation for the active task and report the result honestly."
            execution_outline = [
                "Select the strongest available validation command for the active implementation.",
                "Run the validation and capture any concrete failure output.",
                "Report whether validation passed, failed, or could not be run.",
            ]
            confidence = 0.72
            ambiguity_level = "low" if len(artifact_paths) <= 1 else "medium"
        elif has_active_context and looks_like_correction_request(request) and not ambiguous_targets:
            relation = "correct"
            active_goal = previous_goal or previous_root or request
            current_output_expectation = "A corrected implementation scope that matches the latest user instruction."
            next_action = "modify"
            current_open_problem = current_open_problem or "The current scope likely needs correction against the previous task."
            current_verification_target = current_verification_target or "Apply the narrowest correction and verify the corrected boundary."
            execution_outline = [
                "Inspect the active implementation and the previous target boundary.",
                "Apply the narrowest correction that matches the updated instruction.",
                "Verify that the corrected scope now matches the user's intent.",
            ]
            confidence = 0.7
            ambiguity_level = "low" if len(artifact_paths) <= 1 else "medium"
        elif ambiguous_targets:
            active_goal = previous_goal or request
            next_action = "clarify"
            missing_info.append("There are multiple active artifacts and the current target cannot be inferred safely.")
            execution_outline = [
                "Ask which active artifact should be changed next.",
            ]
            confidence = 0.34
            ambiguity_level = "high"
        elif target_artifacts:
            active_goal = previous_goal or request
            next_action = previous_next_action if previous_next_action in {
                "inspect",
                "search",
                "create",
                "modify",
                "debug",
                "test",
                "explain",
                "plan",
            } else "inspect"
            current_verification_target = current_verification_target or "Keep the change aligned with the active goal and verify the most relevant path afterwards."
            execution_outline = [
                "Inspect the active artifact and continue from the current task state.",
                "Perform the next smallest safe step and verify it.",
            ]
            confidence = 0.68 if has_follow_up_reference(request) and len(artifact_paths) == 1 else 0.62 if len(artifact_paths) == 1 else 0.48
            ambiguity_level = "low" if len(artifact_paths) == 1 else "medium"
        else:
            active_goal = previous_goal or request
            next_action = "search"
            missing_info.append("The target artifact is not explicit in the current state.")
            execution_outline = [
                "Search the workspace for the most relevant active implementation area.",
                "Inspect the top candidate before acting.",
            ]
            confidence = 0.46

        needs_clarification = next_action == "clarify" or confidence < 0.4
        clarification_questions = (
            ["Welche der aktuell aktiven Dateien oder Bereiche meinst du genau?"]
            if ambiguous_targets
            else ["Welchen konkreten Bereich oder welches Artefakt soll ich als naechstes bearbeiten?"]
            if needs_clarification
            else []
        )

        return TaskState(
            latest_user_turn=request,
            root_goal=root_goal,
            active_goal=active_goal,
            goal_relation=relation,
            output_expectation=current_output_expectation or (
                "Diagnose the open issue using existing evidence and apply the smallest safe next step."
                if has_failure_evidence
                else "Continue the active task from the current workspace state."
            ),
            open_problem=current_open_problem,
            verification_target=current_verification_target,
            target_artifacts=target_artifacts[:6],
            evidence=evidence[:6],
            relevant_context=relevant_context[:8],
            constraints=constraints[:8],
            assumptions=assumptions[:8],
            missing_info=missing_info[:6],
            ambiguity_level=ambiguity_level,
            risk_level=risk_level,
            confidence=confidence,
            next_action=next_action,
            execution_outline=execution_outline[:6],
            needs_clarification=needs_clarification,
            clarification_questions=clarification_questions[:3],
        )

    def _initial_create_state(self, request: str, *, snapshot=None) -> TaskState | None:
        del snapshot
        if not is_clear_low_risk_build_request(request):
            return None
        extension = infer_requested_extension(request)
        artifact_hint = self._artifact_name_hint(request, extension=extension)
        target_artifacts: list[TaskArtifact] = []
        if artifact_hint is not None:
            target_artifacts.append(
                TaskArtifact(
                    path=None,
                    name=artifact_hint,
                    kind=extension or "artifact",
                    role="primary_target",
                    confidence=0.62,
                )
            )
        relevant_context = [
            "This is a new implementation request and a conventional default artifact can be chosen safely.",
        ]
        assumptions = [
            "A small runnable implementation is the right default unless the user asks for a larger project.",
        ]
        if extension is not None:
            assumptions.append(f"The implementation medium implies a conventional {extension} entry artifact.")
        return TaskState(
            latest_user_turn=request,
            root_goal=request,
            active_goal=request,
            goal_relation="new_task",
            output_expectation="Create a small runnable implementation with a conventional default artifact and minimal scope.",
            open_problem=None,
            verification_target="Create the initial implementation and run the most relevant validation or entry command.",
            target_artifacts=target_artifacts,
            evidence=[],
            relevant_context=relevant_context,
            constraints=[],
            assumptions=assumptions,
            missing_info=[],
            ambiguity_level="low",
            risk_level="low",
            confidence=0.74 if extension is not None else 0.68,
            next_action="create",
            execution_outline=[
                "Choose a conventional default artifact or small scaffold for the requested implementation.",
                "Implement the smallest runnable version that satisfies the request.",
                "Validate the created artifact with the most relevant project command if available.",
            ],
            needs_clarification=False,
            clarification_questions=[],
        )

    def _follow_up_create_state(
        self,
        request: str,
        *,
        previous_root: str,
        previous_goal: str,
        output_expectation: str,
        verification_target: str,
        existing_artifacts: list[TaskArtifact],
    ) -> TaskState:
        extension = infer_requested_extension(request)
        artifact_hint = self._artifact_name_hint(
            request,
            extension=extension,
            existing_artifacts=existing_artifacts,
        )
        primary_targets: list[TaskArtifact] = []
        if artifact_hint is not None:
            primary_targets.append(
                TaskArtifact(
                    path=None,
                    name=artifact_hint,
                    kind=extension or "artifact",
                    role="primary_target",
                    confidence=0.74,
                )
            )
        return TaskState(
            latest_user_turn=request,
            root_goal=previous_root,
            active_goal=f"Extend the active task by adding the requested artifact or surface around {previous_goal}.",
            goal_relation="refine",
            output_expectation=output_expectation or "Extend the active implementation with the requested additional artifact using conventional defaults.",
            open_problem=None,
            verification_target=verification_target or "Create the additional artifact, connect it to the active implementation, and run the most relevant validation afterwards.",
            target_artifacts=primary_targets,
            active_artifacts=existing_artifacts[:6],
            evidence=[],
            relevant_context=[
                f"Previous root goal: {previous_root}",
                "The latest request extends the current task rather than replacing it.",
            ],
            constraints=[],
            assumptions=[
                "The new artifact should stay aligned with the current implementation and recently changed files.",
                "A conventional default filename is acceptable unless the workspace strongly suggests another entrypoint.",
            ],
            missing_info=[],
            ambiguity_level="low" if len(existing_artifacts) <= 2 else "medium",
            risk_level="low",
            confidence=0.78 if extension is not None else 0.72,
            next_action="create",
            execution_outline=[
                "Inspect the active implementation and nearby artifacts for integration points.",
                "Create the additional artifact with the smallest runnable scope that satisfies the follow-up request.",
                "Verify that the new artifact is wired consistently with the active task afterwards.",
            ],
            needs_clarification=False,
            clarification_questions=[],
        )

    def _artifact_name_hint(
        self,
        request: str,
        *,
        extension: str | None,
        existing_artifacts: list[TaskArtifact] | None = None,
    ) -> str | None:
        if has_follow_up_reference(request) and extension in {".html", ".css"} and existing_artifacts:
            for artifact in existing_artifacts:
                candidate = Path(str(artifact.path or artifact.name or "")).stem.strip()
                if candidate:
                    return candidate
        artifact_hint = infer_artifact_name_hint(request)
        if artifact_hint:
            return artifact_hint
        if extension in {".html", ".css"} and existing_artifacts:
            for artifact in existing_artifacts:
                candidate = Path(str(artifact.path or artifact.name or "")).stem.strip()
                if candidate:
                    return candidate
        defaults = {
            ".html": "index",
            ".css": "styles",
            ".js": "app",
            ".py": "main",
        }
        return defaults.get(extension or "")

    def _normalize_next_action(self, value: str | None) -> str:
        mapping = {
            "analyze": "inspect",
            "build": "create",
            "create": "create",
            "debug": "debug",
            "explain": "explain",
            "inspect": "inspect",
            "modify": "modify",
            "plan": "plan",
            "refactor": "modify",
            "search": "search",
            "test": "test",
        }
        normalized = str(value or "").strip().lower()
        return mapping.get(normalized, "")

    def _should_create_follow_up_artifact(self, request: str) -> bool:
        return is_clear_low_risk_build_request(request) or looks_like_additive_request(request)

    def _filter_artifacts_for_scope(
        self,
        artifacts: list[TaskArtifact],
        scope_constraints: list[str],
    ) -> list[TaskArtifact]:
        if not artifacts or not scope_constraints:
            return artifacts
        backend_only = "Backend only." in scope_constraints
        frontend_only = "Frontend only." in scope_constraints
        filtered: list[TaskArtifact] = []
        for artifact in artifacts:
            candidate = str(artifact.path or artifact.name or "").lower()
            if backend_only and any(token in candidate for token in ("backend", "api", "server")):
                filtered.append(artifact)
            elif frontend_only and any(token in candidate for token in ("frontend", "ui", "client", "web")):
                filtered.append(artifact)
        return filtered or artifacts

    def _scope_verification_target(self, scope_constraints: list[str]) -> str:
        if "Backend only." in scope_constraints:
            return "Only backend artifacts should be updated and verified."
        if "Frontend only." in scope_constraints:
            return "Only frontend artifacts should be updated and verified."
        return "Only artifacts inside the narrowed scope should be updated and verified."

    def _scope_constrained_goal(self, previous_goal: str, scope_constraints: list[str]) -> str:
        if "Backend only." in scope_constraints:
            return f"Continue {previous_goal} only in the backend scope."
        if "Frontend only." in scope_constraints:
            return f"Continue {previous_goal} only in the frontend scope."
        return previous_goal

    def _scope_output_expectation(self, scope_constraints: list[str]) -> str:
        if "Backend only." in scope_constraints:
            return "A backend-scoped change that leaves frontend surfaces untouched."
        if "Frontend only." in scope_constraints:
            return "A frontend-scoped change that leaves backend surfaces untouched."
        return "A narrowed-scope change aligned with the latest user instruction."

    def _collect_target_artifacts(self, session) -> list[TaskArtifact]:
        artifacts: list[TaskArtifact] = []
        seen: set[str] = set()

        def add(path: str | None, *, role: str, confidence: float) -> None:
            text = str(path or "").strip()
            if not text or text in seen:
                return
            seen.add(text)
            artifacts.append(
                TaskArtifact(
                    path=text,
                    name=text.split("/")[-1],
                    kind="file",
                    role=role,
                    confidence=confidence,
                )
            )

        if session is None:
            return artifacts
        if getattr(session, "task_state", None) is not None:
            for artifact in session.task_state.target_artifacts[:6]:
                add(artifact.path or artifact.name, role=artifact.role or "active_context", confidence=max(artifact.confidence, 0.7))
        if getattr(session, "follow_up_context", None) is not None:
            for path in session.follow_up_context.target_paths[:6]:
                add(path, role="active_context", confidence=0.62)
            for path in session.follow_up_context.changed_files[:4]:
                add(path, role="active_context", confidence=0.6)
            for path in session.follow_up_context.read_files[:4]:
                add(path, role="supporting_context", confidence=0.55)
        for path in getattr(session, "candidate_files", [])[:6]:
            add(path, role="supporting_context", confidence=0.5)
        return artifacts

    def _collect_evidence(self, session) -> list[EvidenceItem]:
        if session is None:
            return []
        evidence: list[EvidenceItem] = []
        seen: set[tuple[str, str, str]] = set()

        def append(item: EvidenceItem) -> None:
            key = (item.kind, item.summary, item.artifact_path or "")
            if key in seen:
                return
            seen.add(key)
            evidence.append(item)

        if getattr(session, "task_state", None) is not None:
            for item in session.task_state.evidence[:6]:
                append(item)

        follow_up = getattr(session, "follow_up_context", None)
        if follow_up is not None:
            for item in follow_up.diagnostics[-6:]:
                append(
                    EvidenceItem(
                        kind="diagnostic",
                        summary=item.summary,
                        source=item.source,
                        artifact_path=item.file_hints[0] if item.file_hints else None,
                        confidence=0.82,
                    )
                )
            for run in follow_up.validation_runs[-4:]:
                if run.status not in {"failed", "timeout", "blocked"}:
                    continue
                append(
                    EvidenceItem(
                        kind="test",
                        summary=run.excerpt or run.summary or f"{run.command} failed",
                        source=run.command,
                        artifact_path=None,
                        confidence=0.78,
                    )
                )
            for command in follow_up.recent_commands[-3:]:
                append(
                    EvidenceItem(
                        kind="command",
                        summary=f"Recent command: {command}",
                        source=command,
                        confidence=0.45,
                    )
                )
            if follow_up.last_error:
                append(
                    EvidenceItem(
                        kind="diagnostic",
                        summary=follow_up.last_error,
                        source="follow_up_context",
                        confidence=0.8,
                    )
                )
        return evidence[:8]

    def _summarize_open_problem(self, evidence: list[EvidenceItem], *, fallback: str) -> str:
        for item in evidence:
            if item.summary:
                return item.summary
        return str(fallback or "").strip()

    def _unique_strings(self, values: list[str | None]) -> list[str]:
        unique: list[str] = []
        for raw in values:
            text = str(raw or "").strip()
            if not text or text in unique:
                continue
            unique.append(text)
        return unique

    def _log(self, event: str, **payload: Any) -> None:
        if self.logger is None:
            return
        self.logger.log_event(event, **payload)
