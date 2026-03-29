from __future__ import annotations

import ast
import json
from pathlib import Path
import re

from agent.models import ProposedUpdateReview, SessionState, ValidationFailureEvidence, WorkspaceSnapshot
from agent.task_state import TaskState
from agent.task_schema import TaskUnderstanding
from config.settings import AGENT_FULL_NAME, AGENT_NAME
from llm.schemas import RouteActionName, RouterOutput


REPAIR_BLOCKED_SENTINEL = "__REPAIR_BLOCKED__"


def system_prompt() -> str:
    return (
        f"You are {AGENT_NAME} ({AGENT_FULL_NAME}), a local autonomous coding agent. "
        "Use the validated router output as the source of truth. "
        "Never select tools directly from the raw user prompt. "
        "Treat follow-up messages as continuing the current task unless the user clearly changes topic. "
        "For vague bug reports or comments like 'that is broken' or 'the terminal looks buggy', reconstruct the active task state, inspect available evidence, diagnose before editing, and ask only targeted follow-up questions when evidence is still missing. "
        "Inspect before editing, prefer the smallest sufficient change, and respect access mode."
    )


def router_system_prompt() -> str:
    actions = ", ".join(action.value for action in RouteActionName)
    return (
        f"You are {AGENT_NAME} ({AGENT_FULL_NAME}), a goal-oriented routing model. "
        "Return valid JSON only. "
        "Infer the user's true intent semantically, not from keyword matching. "
        "Be resilient to paraphrases, slang, typos, indirect wishes, and mixed German/English phrasing. "
        "Treat the current user turn as the primary evidence source. "
        "Resolve deictic follow-ups like 'that', 'there', 'the error', 'hier', 'da', and 'das' against the current task state and follow-up context only when the turn actually points back to the same task. "
        "Focus on the user's end goal, the minimum safe action plan, missing information, and whether execution is safe. "
        f"Allowed action names are: {actions}. "
        "If information is missing or the request is risky, ask one to three precise clarification questions instead of guessing."
    )


def task_understanding_system_prompt() -> str:
    return (
        f"You are {AGENT_NAME} ({AGENT_FULL_NAME}), an intent interpretation and goal extraction model "
        "for a local coding agent. "
        "Return valid JSON only. "
        "Infer the user's real objective semantically, not from literal verbs or keyword routing. "
        "Treat the current turn as primary evidence, and use previous context only when it clearly belongs to the same active task and does not conflict. "
        "Resolve references like 'that', 'there', 'the bug', 'das', 'da', 'hier', and 'backend' against the current task state only when the turn genuinely points back to it. "
        "Do not invent secondary specializations such as hardening or refactor modes unless the request or evidence supports them directly. "
        "Prefer reasonable assumptions when the likely intent is strong. "
        "Ask for clarification only when the risk of acting on the wrong target is materially high. "
        "Produce a normalized task object that captures the goal, artifacts, assumptions, ambiguity, missing info, plan, and confidence."
    )


def task_state_system_prompt() -> str:
    return (
        f"You are {AGENT_NAME} ({AGENT_FULL_NAME}), the task-state update model for a local coding agent. "
        "Return valid JSON only. "
        "Your job is to update the agent's working task state from the latest user turn, prior task state, thread context, artifacts, diffs, diagnostics, and terminal evidence. "
        "Do not route by keywords. "
        "Treat the latest user turn as the primary evidence source, and use earlier context only when it clearly belongs to the same task and does not conflict. "
        "Determine whether the user is continuing, refining, correcting, constraining, or replacing the current task. "
        "Bind references like 'that', 'this part', 'the bug', 'das', 'hier', and 'backend' to concrete artifacts whenever possible. "
        "Keep primary semantics and secondary semantics separate: goal relation, active goal, open problem, artifacts, and next_best_action come first; secondary execution strategy should stay null unless the turn or evidence supports it directly. "
        "If uncertainty remains material, preserve that uncertainty instead of inventing a specialization."
    )


def semantic_change_review_system_prompt() -> str:
    return (
        f"You are {AGENT_NAME} ({AGENT_FULL_NAME}), the final semantic change-review model for a local coding agent. "
        "Return valid JSON only. "
        "Review whether the current changed artifacts really satisfy the user's explicit request and constraints. "
        "Reason semantically across languages and frameworks, including HTML/CSS/JavaScript, Python, C/C++, Java, GDScript, config files, and tests. "
        "Treat cross-file mismatches as failures when names, ids, symbols, imports, selectors, event hooks, config keys, paths, or interfaces no longer line up. "
        "Fail the review when explicitly requested behavior is still missing, contradicted, or left dangling in the changed implementation, even if syntax or structural checks passed. "
        "Fail the review when a narrow update removed or rewrote unrelated existing behavior without evidence that the task required it. "
        "Treat unexplained loss of existing CLI options, imports, config handling, startup code, routes, public interfaces, or docs as a likely regression unless the request clearly called for that removal. "
        "Treat unrequested additive scope growth as a likely failure too: new sections, examples, commands, tests, or explanatory guidance are not harmless supporting edits unless the request or visible code evidence clearly requires them. "
        "When the request includes an explicit command example, path placeholder, argument order, code snippet, or literal value to preserve or update, deviations from that concrete example are likely failures unless visible code or validation evidence clearly requires the change. "
        "Do not fail for speculative improvements, style preferences, or unrequested polish."
    )


def proposed_update_review_system_prompt() -> str:
    return (
        f"You are {AGENT_NAME} ({AGENT_FULL_NAME}), the pre-write update review model for a local coding agent. "
        "Return valid JSON only. "
        "Review whether a proposed update to one existing file is safe to write. "
        "Approve only when the proposal is aligned with the explicit request and preserves unrelated existing behavior. "
        "Use semantic reasoning across languages and frameworks, not hardcoded task vocabularies. "
        "When the explicit request requires changing current behavior in the target file, treat that requested behavior change as in scope instead of calling it broadening solely because the current implementation behaves differently. "
        "Fail when the proposal broadens scope without evidence, removes working behavior the request did not ask to remove, or changes names, config keys, selectors, imports, commands, or interfaces without keeping dependent references coherent. "
        "Treat additive scope growth as broadening too: new sections, examples, commands, tests, or explanatory prose are not harmless supporting edits unless the request or visible evidence clearly requires them. "
        "When the request includes an explicit command example, path placeholder, argument order, code snippet, or literal value to preserve or update, deviations from that concrete example are failures unless visible code or validation evidence clearly requires the change. "
        "Treat documentation command examples as illustrative usage, not as evidence that an option is required or has a default value, unless the text explicitly states that behavior or the code evidence enforces it. "
        "Do not fail for harmless formatting shifts or clearly necessary supporting edits."
    )


def task_state_update_prompt(
    task: str,
    snapshot: WorkspaceSnapshot | None,
    session: SessionState | None = None,
    *,
    mode: str = "full",
) -> str:
    schema_shape = {
        "latest_user_turn": "string",
        "root_goal": "string",
        "active_goal": "string",
        "goal_relation": "new_task | continue | refine | correct | report_problem | scope_change | validation_request | rollback_request | approval | rejection | update_constraints | clarify | unknown",
        "output_expectation": "string",
        "current_user_intent": "repair | implement | refactor | harden | validate | inspect | explain | search | plan | correct | unknown | null (optional)",
        "execution_strategy": "debug_repair | feature_implementation | refactor | hardening | validation_inspection | rollback_correction | null (optional)",
        "open_problem": "string | null (optional)",
        "verification_target": "string | null (optional)",
        "target_artifacts": [
            {
                "path": "string | null",
                "name": "string | null",
                "kind": "file | module | feature | flow | command | test | doc | service | null",
                "role": "primary_target | supporting_context | validation_target | active_context | null",
                "confidence": 0.0,
            }
        ],
        "active_artifacts": "optional list[artifact]",
        "evidence": "optional list[evidence]",
        "supplied_evidence": "optional list[string]",
        "relevant_context": "optional list[string]",
        "constraints": "optional list[string]",
        "assumptions": "optional list[string]",
        "missing_info": "optional list[string]",
        "ambiguity_level": "low | medium | high",
        "risk_level": "low | medium | high",
        "confidence": 0.0,
        "next_action": "inspect | search | create | modify | debug | test | explain | plan | clarify",
        "next_best_action": "inspect | search | create | modify | debug | test | explain | plan | clarify | null (optional)",
        "execution_outline": "optional list[string]",
        "needs_clarification": "boolean (optional, default false)",
        "clarification_questions": "optional list[string]",
    }
    compact = mode != "full"
    lines = [
        "Update the central task state for this turn.",
        f"Latest user request: {_trim_text(task, 900 if not compact else 500)}",
        f"Follow-up context: {json.dumps(_compact_follow_up_context(session), ensure_ascii=False)}",
        f"Previous task state: {json.dumps(_compact_task_state(session.task_state if session is not None else None), ensure_ascii=False)}",
        f"Workspace context: {json.dumps(_compact_workspace_snapshot(snapshot, detail='router' if not compact else 'decision'), ensure_ascii=False)}",
        "State update rules:",
        "- Current turn first: use previous context only when it clearly belongs to the same task and does not conflict.",
        "- First decide whether this turn continues or changes the active task.",
        "- current_user_intent should reflect what the user is trying to do at this phase of the task.",
        "- execution_strategy is secondary semantics: leave it null if the request only supports a generic next action.",
        "- Extract concrete evidence from diagnostics, terminal errors, changed files, and referenced artifacts.",
        "- For bugs or regressions, prefer inspect/debug/test before modify.",
        "- For scope corrections or rollbacks, narrow or revert only the necessary part of the prior work.",
        "- Update constraints and assumptions explicitly.",
        "- Keep root_goal stable across refinements unless the user clearly starts a new task.",
        "- If a compatible active artifact already exists and the user is extending its behavior, prefer modify over create unless the user clearly asks for a distinct new artifact or file surface.",
        "- next_action and next_best_action should be the single best next move, not a full route tree.",
        "- Preserve the execution order: inspect current state and active artifacts, gather evidence, act in the smallest sensible step, then verify against verification_target.",
        "- Ask clarification only if acting now would likely hit the wrong artifact or cause destructive behavior.",
        "- Be terse: use short phrases instead of full sentences and avoid repeating the user request verbatim across multiple fields.",
        "- Keep root_goal, active_goal, output_expectation, and verification_target concise, ideally under 18 words each.",
        "- Omit optional keys when they would only be null, empty, or duplicate another field.",
        "- Keep lists to at most 3 concise items unless the user explicitly named more artifacts.",
        f"Return JSON only with this structure: {json.dumps(schema_shape, ensure_ascii=False)}",
    ]
    if not compact:
        lines.insert(2, f"Recent conversation: {_format_objects(_compact_recent_messages(session))}")
        lines.insert(3, f"Recent tool calls: {_format_objects(_compact_recent_calls(session))}")
        lines.insert(6, f"Previous task understanding: {json.dumps(_compact_task_understanding(session.task_understanding if session is not None else None), ensure_ascii=False)}")
        lines.insert(7, f"Recent diagnostics: {_format_objects(_compact_recent_diagnostics(session))}")
    return "\n".join(lines)


def task_understanding_prompt(
    task: str,
    snapshot: WorkspaceSnapshot | None,
    session: SessionState | None = None,
    *,
    mode: str = "full",
) -> str:
    workspace = _compact_workspace_snapshot(snapshot, detail="router")
    prior_understanding = _compact_task_understanding(
        session.task_understanding if session is not None else None
    )
    schema_shape = {
        "original_request": "string",
        "interpreted_goal": "string",
        "intent_category": "build | modify | debug | refactor | explain | analyze | test | search | configure | plan | unknown",
        "conversation_relation": "new_task | same_task_follow_up | refinement | problem_report | constraint_update | clarification | correction | unknown",
        "subgoals": ["string"],
        "target_artifacts": [
            {
                "path": "string | null",
                "name": "string | null",
                "kind": "file | module | feature | flow | command | test | doc | service | null",
                "role": "primary_target | supporting_context | validation_target | active_context | null",
                "confidence": 0.0,
            }
        ],
        "relevant_context": ["string"],
        "constraints": ["string"],
        "missing_info": ["string"],
        "assumptions": ["string"],
        "user_observations": ["string"],
        "supplied_evidence": ["string"],
        "ambiguity_level": "low | medium | high",
        "risk_level": "low | medium | high",
        "confidence": 0.0,
        "recommended_mode": "clarify | inspect | search | create | modify | debug | refactor | test | explain | plan",
        "execution_plan": [
            {
                "step": 1,
                "summary": "string",
                "action_hint": "inspect | search | create | modify | debug | test | explain | plan | null",
                "requires_tools": True,
            }
        ],
        "needs_clarification": False,
        "clarification_questions": ["string"],
    }
    compact = mode != "full"
    lines = [
        "Normalize the user's latest request into a task understanding object.",
        f"Latest user request: {_trim_text(task, 900 if not compact else 500)}",
        f"Follow-up context: {json.dumps(_compact_follow_up_context(session), ensure_ascii=False)}",
        f"Current task state: {json.dumps(_compact_task_state(session.task_state if session is not None else None), ensure_ascii=False)}",
        f"Workspace context: {json.dumps(workspace, ensure_ascii=False)}",
        "Interpretation rules:",
        "- Extract the user's actual end goal, not just the surface wording.",
        "- Current turn first: preserve continuity only when the latest turn clearly belongs to the same task and does not conflict.",
        "- Identify likely target artifacts even when the user names them indirectly.",
        "- Separate explicit constraints from assumptions.",
        "- If the user reports a vague problem, treat their message as an observation that should trigger diagnosis rather than blind fixing.",
        "- Keep secondary specializations conservative: do not invent hardening, refactor, or repair submodes unless the request or evidence supports them directly.",
        "- Use low ambiguity only when the target and outcome are both materially clear.",
        "- Use low confidence or needs_clarification only when acting now would likely hit the wrong target or be destructive.",
        "- Prefer short, concrete execution steps that can guide planning.",
        f"Return JSON only with this structure: {json.dumps(schema_shape, ensure_ascii=False)}",
    ]
    if not compact:
        lines.insert(2, f"Recent conversation: {_format_objects(_compact_recent_messages(session))}")
        lines.insert(3, f"Recent tool calls: {_format_objects(_compact_recent_calls(session))}")
        lines.insert(5, f"Previous task understanding: {json.dumps(prior_understanding, ensure_ascii=False)}")
    return "\n".join(lines)


def router_prompt(
    task: str,
    snapshot: WorkspaceSnapshot | None,
    session: SessionState | None = None,
    *,
    mode: str = "full",
) -> str:
    workspace = _compact_workspace_snapshot(snapshot, detail="router")
    recent_messages = _compact_recent_messages(session)
    recent_calls = _compact_recent_calls(session)
    follow_up_context = _compact_follow_up_context(session)
    recent_diagnostics = _compact_recent_diagnostics(session)
    changed_files = [item.path for item in session.changed_files[-6:]] if session else []
    action_catalog = [
        {
            "action": RouteActionName.RESPOND_DIRECTLY.value,
            "when": "Use when the request can be answered without any tool call.",
        },
        {
            "action": RouteActionName.ASK_CLARIFICATION.value,
            "when": "Use when required parameters, scope, or risk decisions are missing.",
        },
        {
            "action": RouteActionName.INSPECT_WORKSPACE.value,
            "when": "Use when repository structure or context must be gathered first.",
        },
        {
            "action": RouteActionName.SEARCH_WORKSPACE.value,
            "when": "Use when the target must be located semantically in the codebase.",
        },
        {
            "action": RouteActionName.READ_RELEVANT_FILES.value,
            "when": "Use when specific files should be inspected before replying or changing code.",
        },
        {
            "action": RouteActionName.DIAGNOSE_ISSUE.value,
            "when": "Use when a vague failure, bug report, terminal issue, or broken follow-up must be investigated before editing.",
        },
        {
            "action": RouteActionName.CREATE_ARTIFACT.value,
            "when": "Use when the user wants something new to be added.",
        },
        {
            "action": RouteActionName.UPDATE_ARTIFACT.value,
            "when": "Use when existing code or content should be changed.",
        },
        {
            "action": RouteActionName.DELETE_ARTIFACT.value,
            "when": "Use when something should be removed.",
        },
        {
            "action": RouteActionName.PLAN_WORK.value,
            "when": "Use when the user primarily wants a plan or implementation strategy.",
        },
        {
            "action": RouteActionName.RUN_VALIDATION.value,
            "when": "Use when execution should include tests, linting, or another verification step.",
        },
        {
            "action": RouteActionName.SUMMARIZE_RESULT.value,
            "when": "Use as the final step after inspection or mutation work.",
        },
    ]
    schema_shape = {
        "user_goal": "string",
        "intent": "inspect | create | update | debug | delete | search | explain | plan | unknown",
        "entities": {
            "target_type": "string | null",
            "target_name": "string | null",
            "target_paths": ["string"],
            "attributes": ["string"],
            "constraints": ["string"],
        },
        "requested_outcome": "string",
        "action_plan": [
            {
                "step": 1,
                "action": "one_of_the_allowed_action_names",
                "reason": "string",
            }
        ],
        "needs_clarification": True,
        "clarification_questions": ["string"],
        "confidence": 0.0,
        "safe_to_execute": False,
        "repo_context_needed": True,
        "search_terms": ["string"],
        "relevant_extensions": [".py"],
        "direct_response": "string | null",
    }
    compact = mode != "full"
    lines = [
        "Interpret the user's latest request.",
        f"User request: {_trim_text(task, 600 if not compact else 400)}",
        f"Follow-up context: {json.dumps(follow_up_context, ensure_ascii=False)}",
        f"Recently changed files: {_format_list(changed_files)}",
        f"Workspace context: {json.dumps(workspace, ensure_ascii=False)}",
        f"Allowed actions: {json.dumps(action_catalog, ensure_ascii=False)}",
        "Routing rules:",
        "- Infer intent from meaning, not literal verbs.",
        "- Extract the user's end goal and likely target entities.",
        "- Current turn first: treat previous context as supporting evidence, not as the controlling source.",
        "- Treat vague follow-ups like 'da', 'das', 'hier', 'der Fehler', 'kaputt', 'buggy', 'komisch', and 'geht nicht' as references to the current task only when the latest turn genuinely points back to it.",
        "- When the user reports a vague bug or terminal issue and follow-up context exists, prefer a debug route with diagnosis before any update.",
        "- Map bug reports to evidence-seeking actions: inspect the active artifact, inspect diagnostics, rerun the most relevant validation or terminal command, then update only if the evidence supports it.",
        "- Detect when the request is ambiguous, unsafe, or missing a required parameter.",
        "- Prefer the smallest practical action plan.",
        "- Use intent=unknown when the request cannot be interpreted safely enough.",
        "- For direct conversational answers, set direct_response and use respond_directly.",
        "- For ambiguous requests, set needs_clarification=true, safe_to_execute=false, and add 1-3 precise questions tied to the missing artifact, symptom, or evidence.",
        "- For executable requests, set needs_clarification=false and safe_to_execute=true only when enough information is present.",
        f"Return JSON only with this structure: {json.dumps(schema_shape, ensure_ascii=False)}",
    ]
    if not compact:
        lines.insert(2, f"Recent conversation: {_format_objects(recent_messages)}")
        lines.insert(3, f"Recent tool calls: {_format_objects(recent_calls)}")
        lines.insert(6, f"Recent diagnostics: {_format_objects(recent_diagnostics)}")
    return "\n".join(lines)


def router_repair_prompt(
    invalid_payload: dict[str, object],
    errors: list[dict[str, object]],
) -> str:
    return "\n".join(
        [
            "Repair the invalid router JSON output and return valid JSON only.",
            f"Invalid payload: {json.dumps(invalid_payload, ensure_ascii=False, default=str)}",
            f"Validation errors: {json.dumps(errors, ensure_ascii=False, default=str)}",
            "Do not add prose. Keep the same user intent and fix only the schema or safety issues.",
        ]
    )


def choose_path_prompt(route: RouterOutput, session: SessionState) -> str:
    snapshot = _compact_workspace_snapshot(session.workspace_snapshot, detail="decision")
    return "\n".join(
        [
            "Choose one relative workspace file path for a new implementation.",
            f"Route: {json.dumps(_compact_route(route), ensure_ascii=False)}",
            f"Workspace context: {json.dumps(snapshot, ensure_ascii=False)}",
            f"Already read files: {_format_list(_read_paths(session))}",
            "Return only the path, with no explanation or markdown.",
        ]
    )


def generate_content_prompt(
    route: RouterOutput,
    session: SessionState,
    *,
    path: str,
    current_content: str | None = None,
    repair_context: ValidationFailureEvidence | None = None,
    repair_strategy: str | None = None,
    review_feedback: ProposedUpdateReview | None = None,
    mode: str = "full",
) -> str:
    if mode != "full":
        file_focus = _artifact_scoped_focus(route, session, path, current_content=current_content)
        explicit_constraints = _explicit_generation_constraints(route, session)
        related_targets = [item for item in route.entities.target_paths if item and item != path][:4]
        related_context = _related_file_context(session, path)
        if current_content is None:
            sections = [
                "Produce the full file content for exactly one file.",
                f"Latest user request: {_trim_text(session.task, 360)}",
                f"Target path: {path}",
            ]
            if explicit_constraints != "none":
                sections.append(f"Explicit constraints: {explicit_constraints}")
            sections.append(f"File-scoped focus: {json.dumps(file_focus, ensure_ascii=False)}")
            if related_targets:
                sections.append(
                    f"Out-of-scope companion files for this step: {_format_list(related_targets)}. They will be handled separately."
                )
            if related_context != "none":
                sections.append(f"Related file hints: {related_context}")
            if repair_context is not None:
                sections.extend(
                    [
                        f"Validation-guided repair context: {json.dumps(_compact_repair_context(repair_context), ensure_ascii=False)}",
                        _repair_rules(repair_strategy),
                    ]
                )
                fixture_hints = _runtime_support_file_prompt_hints(
                    path=path,
                    current_content="",
                    repair_context=repair_context,
                )
                if fixture_hints:
                    sections.append(
                        "Runtime support file hints: "
                        + " ".join(_trim_text(item, 180) for item in fixture_hints[:4])
                    )
            sections.append(_single_file_boundary_instruction(path, route.entities.target_paths))
            sections.append("Create the file from scratch. Return the full new file content only.")
            sections.append("Do not add markdown fences or explanations.")
            return "\n\n".join(sections)

        if repair_context is not None:
            return _compact_repair_update_prompt(
                route,
                session,
                path=path,
                current_content=current_content,
                file_focus=file_focus,
                explicit_constraints=explicit_constraints,
                repair_context=repair_context,
                repair_strategy=repair_strategy,
                review_feedback=review_feedback,
            )

        sections = [
            "Produce the full file content for exactly one file.",
            f"Latest user request: {_trim_text(session.task, 420)}",
            f"User goal: {_trim_text(route.user_goal, 240)}",
            f"Requested outcome: {_trim_text(route.requested_outcome, 240)}",
            f"Explicit constraints: {explicit_constraints}",
            f"Task focus: {json.dumps(_compact_generation_focus(route, session, path), ensure_ascii=False)}",
            f"File-scoped focus: {json.dumps(file_focus, ensure_ascii=False)}",
            f"Related file hints: {related_context}",
        ]
        if repair_context is not None:
            sections.extend(
                [
                    f"Validation-guided repair context: {json.dumps(_compact_repair_context(repair_context), ensure_ascii=False)}",
                    _repair_rules(repair_strategy),
                ]
            )
            fixture_hints = _runtime_support_file_prompt_hints(
                path=path,
                current_content=current_content or "",
                repair_context=repair_context,
            )
            if fixture_hints:
                sections.append(
                    "Runtime support file hints: "
                    + " ".join(_trim_text(item, 180) for item in fixture_hints[:4])
                )
        if current_content is not None:
            sections.extend(
                [
                    _update_rules(),
                ]
            )
            if review_feedback is not None:
                sections.extend(
                    [
                        f"Self-review feedback on the previous proposal: {json.dumps(_compact_proposed_update_review(review_feedback), ensure_ascii=False)}",
                        "Use that feedback to make a smaller, safer update while preserving unrelated existing behavior.",
                    ]
                )
            sections.extend(
                [
                    "Current file content:",
                    current_content,
                    _single_file_boundary_instruction(path, route.entities.target_paths),
                    "Update this file to satisfy the request. Return the full updated file content only.",
                ]
            )
        else:
            sections.append(_single_file_boundary_instruction(path, route.entities.target_paths))
            sections.append("Create the file from scratch. Return the full new file content only.")
        sections.append("Do not add markdown fences or explanations.")
        return "\n\n".join(sections)

    sections = [
        "Produce the full file content for the requested task.",
        f"Latest user request: {_trim_text(session.task, 700)}",
        f"Validated route: {json.dumps(_compact_route(route), ensure_ascii=False)}",
        f"Task state: {json.dumps(_compact_task_state(session.task_state), ensure_ascii=False)}",
        f"Task understanding: {json.dumps(_compact_task_understanding(session.task_understanding), ensure_ascii=False)}",
        f"Target path: {path}",
        f"Explicit constraints: {_explicit_generation_constraints(route, session)}",
        f"File-scoped focus: {json.dumps(_artifact_scoped_focus(route, session, path, current_content=current_content), ensure_ascii=False)}",
        f"Workspace context: {json.dumps(_compact_workspace_snapshot(session.workspace_snapshot, detail='decision'), ensure_ascii=False)}",
        f"Inspected context: {_inspected_context(session)}",
        f"Diagnostic context: {_diagnostic_context(session)}",
        f"Follow-up context: {json.dumps(_compact_follow_up_context(session), ensure_ascii=False)}",
        _single_file_boundary_instruction(path, route.entities.target_paths),
    ]
    if repair_context is not None:
        sections.extend(
            [
                f"Validation-guided repair context: {json.dumps(_compact_repair_context(repair_context), ensure_ascii=False)}",
                _repair_rules(repair_strategy),
            ]
        )
        fixture_hints = _runtime_support_file_prompt_hints(
            path=path,
            current_content=current_content or "",
            repair_context=repair_context,
        )
        if fixture_hints:
            sections.append(
                "Runtime support file hints: "
                + " ".join(_trim_text(item, 180) for item in fixture_hints[:4])
            )
    if current_content is not None:
        sections.extend(
            [
                _update_rules(),
            ]
        )
        if review_feedback is not None:
            sections.extend(
                [
                    f"Self-review feedback on the previous proposal: {json.dumps(_compact_proposed_update_review(review_feedback), ensure_ascii=False)}",
                    "Use that feedback to make a smaller, safer update while preserving unrelated existing behavior.",
                ]
            )
        sections.extend(
            [
                "Current file content:",
                current_content,
                "Return the full updated file content only. No markdown fences. No explanation.",
            ]
        )
    else:
        sections.append("Return the full new file content only. No markdown fences. No explanation.")
    return "\n\n".join(sections)


def generate_content_retry_prompt(
    route: RouterOutput,
    session: SessionState | None = None,
    *,
    path: str,
    current_content: str | None = None,
    repair_context: ValidationFailureEvidence | None = None,
    repair_strategy: str | None = None,
    review_feedback: ProposedUpdateReview | None = None,
    mode: str = "full",
) -> str:
    if mode != "full":
        task_focus = (
            _compact_generation_focus(route, session, path)
            if session is not None
            else {
                "target_path": path,
                "active_goal": _trim_text(route.requested_outcome, 180),
                "output_expectation": _trim_text(route.requested_outcome, 180),
                "verification_target": "",
                "constraints": route.entities.constraints[:4],
                "related_targets": [item for item in route.entities.target_paths if item and item != path][:6],
            }
        )
        if current_content is not None and repair_context is not None and session is not None:
            if review_feedback is not None:
                return _compact_repair_retry_prompt(
                    route,
                    session,
                    path=path,
                    current_content=current_content,
                    repair_context=repair_context,
                    repair_strategy=repair_strategy,
                    review_feedback=review_feedback,
                )
            return _compact_repair_update_prompt(
                route,
                session,
                path=path,
                current_content=current_content,
                file_focus=_artifact_scoped_focus(route, session, path, current_content=current_content),
                explicit_constraints=_explicit_generation_constraints(route, session),
                repair_context=repair_context,
                repair_strategy=repair_strategy,
                review_feedback=review_feedback,
            )
        sections = [
            "Produce the full file content for exactly one file.",
            f"Latest user request: {_trim_text(session.task if session is not None else route.requested_outcome, 420)}",
            f"User goal: {_trim_text(route.user_goal, 240)}",
            f"Requested outcome: {_trim_text(route.requested_outcome, 240)}",
            f"Explicit constraints: {_explicit_generation_constraints(route, session)}",
            f"Task focus: {json.dumps(task_focus, ensure_ascii=False)}",
            f"File-scoped focus: {json.dumps(_artifact_scoped_focus(route, session, path, current_content=current_content), ensure_ascii=False)}",
            _single_file_boundary_instruction(path, route.entities.target_paths),
        ]
        if session is not None:
            sections.append(f"Related file hints: {_related_file_context(session, path)}")
        if repair_context is not None:
            sections.extend(
                [
                    f"Validation-guided repair context: {json.dumps(_compact_repair_context(repair_context), ensure_ascii=False)}",
                    _repair_rules(repair_strategy),
                ]
            )
        if review_feedback is not None:
            sections.extend(
                [
                    f"Self-review feedback on the previous proposal: {json.dumps(_compact_proposed_update_review(review_feedback), ensure_ascii=False)}",
                    "Address that feedback directly in the next full file draft.",
                ]
            )
        if current_content is not None:
            sections.append(_update_rules())
            sections.extend(
                [
                    "Current file content:",
                    current_content,
                    "Update this file to satisfy the request. Return the full updated file content only.",
                ]
            )
        else:
            sections.append("Create the file from scratch. Return the full new file content only.")
        sections.append("Do not add markdown fences or explanations.")
        return "\n\n".join(sections)

    sections = [
        "Produce the full file content for exactly one file.",
        f"Latest user request: {_trim_text(session.task if session is not None else route.requested_outcome, 700)}",
        f"User goal: {_trim_text(route.user_goal, 240)}",
        f"Requested outcome: {_trim_text(route.requested_outcome, 240)}",
        f"Explicit constraints: {_explicit_generation_constraints(route, session)}",
        f"Task state: {json.dumps(_compact_task_state(session.task_state if session is not None else None), ensure_ascii=False)}",
        f"Task understanding: {json.dumps(_compact_task_understanding(session.task_understanding if session is not None else None), ensure_ascii=False)}",
        f"Target path: {path}",
        f"File-scoped focus: {json.dumps(_artifact_scoped_focus(route, session, path, current_content=current_content), ensure_ascii=False)}",
        f"Search hints: {_format_list(route.search_terms[:6])}",
        _single_file_boundary_instruction(path, route.entities.target_paths),
    ]
    if session is not None:
        sections.extend(
            [
                f"Diagnostic context: {_diagnostic_context(session)}",
                f"Follow-up context: {json.dumps(_compact_follow_up_context(session), ensure_ascii=False)}",
            ]
        )
    if repair_context is not None:
        sections.extend(
            [
                f"Validation-guided repair context: {json.dumps(_compact_repair_context(repair_context), ensure_ascii=False)}",
                _repair_rules(repair_strategy),
            ]
        )
    if review_feedback is not None:
        sections.extend(
            [
                f"Self-review feedback on the previous proposal: {json.dumps(_compact_proposed_update_review(review_feedback), ensure_ascii=False)}",
                "Address that feedback directly in the next full file draft.",
            ]
        )
    if current_content is not None:
        sections.extend(
            [
                _update_rules(),
            ]
        )
        sections.extend(
            [
                "Current file content:",
                current_content,
                "Update this file to satisfy the request. Return the full updated file content only.",
            ]
        )
    else:
        sections.append("Create the file from scratch. Return the full new file content only.")
    sections.append("Do not add markdown fences or explanations.")
    return "\n\n".join(sections)


def generate_content_continuation_prompt(
    route: RouterOutput,
    session: SessionState | None = None,
    *,
    path: str,
    partial_content: str,
    current_content: str | None = None,
    repair_context: ValidationFailureEvidence | None = None,
    repair_strategy: str | None = None,
    review_feedback: ProposedUpdateReview | None = None,
) -> str:
    sections = [
        "Finish the full file content for exactly one file.",
        "A previous slow generation already produced a partial draft. Use that progress instead of starting over blindly.",
        f"Latest user request: {_trim_text(session.task if session is not None else route.requested_outcome, 700)}",
        f"User goal: {_trim_text(route.user_goal, 240)}",
        f"Requested outcome: {_trim_text(route.requested_outcome, 240)}",
        f"Task state: {json.dumps(_compact_task_state(session.task_state if session is not None else None), ensure_ascii=False)}",
        f"Task understanding: {json.dumps(_compact_task_understanding(session.task_understanding if session is not None else None), ensure_ascii=False)}",
        f"Target path: {path}",
        f"Search hints: {_format_list(route.search_terms[:6])}",
        _single_file_boundary_instruction(path, route.entities.target_paths),
    ]
    if session is not None:
        sections.extend(
            [
                f"Diagnostic context: {_diagnostic_context(session)}",
                f"Follow-up context: {json.dumps(_compact_follow_up_context(session), ensure_ascii=False)}",
            ]
        )
    if repair_context is not None:
        sections.extend(
            [
                f"Validation-guided repair context: {json.dumps(_compact_repair_context(repair_context), ensure_ascii=False)}",
                _repair_rules(repair_strategy),
            ]
        )
    if current_content is not None:
        sections.extend(
            [
                _update_rules(),
            ]
        )
        if review_feedback is not None:
            sections.extend(
                [
                    f"Self-review feedback on the previous proposal: {json.dumps(_compact_proposed_update_review(review_feedback), ensure_ascii=False)}",
                    "Use the feedback to preserve unrelated existing behavior while completing the update.",
                ]
            )
        sections.extend(
            [
                "Current file content:",
                current_content,
            ]
        )
    sections.extend(
        [
            "Partial draft from the previous attempt:",
            partial_content,
            "Return the complete final file content only. Preserve correct parts of the partial draft, finish the missing parts, and do not add markdown fences or explanations.",
        ]
    )
    return "\n\n".join(sections)


def _single_file_boundary_instruction(path: str, target_paths: list[str] | None) -> str:
    target = str(path or "").strip() or "this file"
    related_targets = [
        str(item or "").strip()
        for item in (target_paths or [])
        if str(item or "").strip() and str(item or "").strip() != target
    ]
    if related_targets:
        return (
            f"Only write {target}. The other requested files {_format_list(related_targets[:4])} are out of scope for this output. "
            "Do not include their content, filenames, headings, tests, README text, or multi-file sections here."
        )
    return (
        f"Only write {target}. Do not include content for any second file, extra headings, or multi-file output."
    )


def final_response_prompt(route: RouterOutput, session: SessionState) -> str:
    recent_notes = session.notes[-12:]
    recent_calls = _compact_recent_calls(session)
    report_context = {
        "route": _compact_route(route),
        "task_state": _compact_task_state(session.task_state),
        "task_understanding": _compact_task_understanding(session.task_understanding),
        "changed_files": [item.path for item in session.changed_files[-8:]],
        "validation_status": session.validation_status,
        "recent_tool_calls": recent_calls,
        "recent_diagnostics": _compact_recent_diagnostics(session),
        "follow_up_context": _compact_follow_up_context(session),
        "notes": recent_notes,
        "blockers": session.blockers[-6:],
    }
    return "\n".join(
        [
            "Write a concise user-facing response for the completed or blocked task.",
            f"Context: {json.dumps(report_context, ensure_ascii=False)}",
            "Mention the main outcome, changed files or inspected files when relevant, and any blocker or validation status.",
            "Do not emit JSON.",
        ]
    )


def semantic_change_review_prompt(
    route: RouterOutput,
    session: SessionState,
    *,
    artifacts: list[dict[str, object]],
    mode: str = "full",
) -> str:
    compact = mode != "full"
    schema_shape = {
        "requirements_satisfied": True,
        "summary": "string",
        "confidence": 0.0,
        "missing_requirements": ["string"],
        "suspicious_issues": ["string"],
        "file_hints": ["string"],
        "repair_hints": ["string"],
    }
    validation_context = [
        {
            "command": _trim_text(item.command, 140 if not compact else 100),
            "scope": item.verification_scope,
            "status": item.status,
            "summary": _trim_text(item.summary or "", 160 if not compact else 100),
            "excerpt": _trim_text(item.excerpt or "", 220 if not compact else 120),
        }
        for item in session.validation_runs[-6:]
    ]
    review_context = {
        "route": _compact_route(route),
        "task_state": _compact_task_state(session.task_state),
        "changed_files": [item.path for item in session.changed_files[-8:]],
        "artifacts": artifacts[:6],
        "validation": validation_context,
    }
    if not compact:
        review_context["task_understanding"] = _compact_task_understanding(session.task_understanding)
        review_context["diagnostics"] = _compact_recent_diagnostics(session)
        review_context["follow_up_context"] = _compact_follow_up_context(session)
    return "\n".join(
        [
            "Review the changed implementation before declaring the task complete.",
            f"Context: {json.dumps(review_context, ensure_ascii=False)}",
            "Review rules:",
            "- Compare the explicit requested outcome, constraints, and verification target against the current changed artifacts.",
            "- Use semantic reasoning, not hardcoded feature vocabularies tied to one stack or one task.",
            "- Fail when requested behavior is still missing, when cross-file references no longer line up, or when the code contradicts the intended behavior.",
            "- Catch obvious inconsistencies such as renamed ids or symbols, broken imports or selectors, config written but never consumed, UI hooks targeting the wrong element, or APIs changed in one file but not another.",
            "- Fail when an otherwise narrow update removed or rewrote unrelated existing behavior without evidence that the task required that broader change.",
            "- Fail when an otherwise narrow update adds unrequested new sections, paragraphs, examples, commands, tests, or guidance that the visible evidence does not require.",
            "- Treat explicit literal examples from the request as hard constraints when they are clearly part of the requested outcome; wrong placeholders, argument order, command shape, or snippet content are failures unless visible evidence contradicts them.",
            "- Base findings only on evidence present in the request, artifacts, diagnostics, and validation context.",
            "- Do not fail for optional refactors, broader hardening ideas, or style-only concerns.",
            f"Return JSON only with this structure: {json.dumps(schema_shape, ensure_ascii=False)}",
        ]
    )


def proposed_update_review_prompt(
    route: RouterOutput,
    session: SessionState,
    *,
    path: str,
    supporting_artifact_context: str,
    current_excerpt: str,
    proposed_excerpt: str,
    diff_excerpt: str,
    mode: str = "full",
) -> str:
    compact = mode != "full"
    changed_paths: list[str] = []
    for item in session.changed_files:
        candidate = str(item.path or "").strip()
        if candidate and candidate not in changed_paths:
            changed_paths.append(candidate)

    explicit_target_paths: list[str] = []
    task_state = session.task_state
    if task_state is not None:
        for artifact in task_state.target_artifacts:
            candidate = str(artifact.path or "").strip()
            if candidate and candidate not in explicit_target_paths:
                explicit_target_paths.append(candidate)
    for candidate in route.entities.target_paths:
        normalized = str(candidate or "").strip()
        if normalized and normalized not in explicit_target_paths:
            explicit_target_paths.append(normalized)

    pending_target_paths = [
        candidate
        for candidate in explicit_target_paths
        if candidate != path and candidate not in changed_paths
    ]
    schema_shape = {
        "safe_to_write": True,
        "summary": "string",
        "confidence": 0.0,
        "blocking_issues": ["string"],
        "preservation_risks": ["string"],
        "repair_hints": ["string"],
    }
    review_context = {
        "route": _compact_route(route),
        "task_state": _compact_task_state(session.task_state),
        "target_path": path,
        "path_scope": _artifact_scoped_focus(route, session, path, current_content=current_excerpt),
        "already_changed_artifacts": changed_paths[:6],
        "pending_target_artifacts": pending_target_paths[:6],
        "current_write_is_partial_step": bool(pending_target_paths),
    }
    if not compact:
        review_context["task_understanding"] = _compact_task_understanding(session.task_understanding)
        review_context["diagnostics"] = _compact_recent_diagnostics(session)
        review_context["follow_up_context"] = _compact_follow_up_context(session)
    return "\n".join(
        [
            "Review the proposed file update before it is written to disk.",
            f"Latest user request: {_trim_text(session.task, 700 if not compact else 420)}",
            f"Context: {json.dumps(review_context, ensure_ascii=False)}",
            "Review rules:",
            "- Judge whether writing this file now is safe as one step in the task, not whether the entire multi-file task is already finished.",
            "- Approve only when the proposal is tightly aligned with the explicit request and keeps unrelated existing behavior intact.",
            "- Do not treat an explicitly requested behavior change for this file as scope broadening just because the current implementation behaves differently.",
            "- Fail when the proposal broadens scope without evidence, or removes working behavior, imports, config handling, startup code, public interfaces, commands, docs, or tests that the request did not ask to remove.",
            "- Fail when a narrow request adds unrequested new sections, explanatory prose, examples, commands, tests, or guidance unless the visible evidence clearly requires that extra content.",
            "- Treat explicit literal examples from the request as hard constraints when they are clearly part of the requested outcome; wrong placeholders, argument order, command shape, or snippet content are failures unless visible evidence contradicts them.",
            "- Use path_scope.current_write_requirements as the hard requirements for this file when that list is non-empty.",
            "- Treat path_scope.other_pending_requirements as non-blocking for this file unless the proposal directly contradicts them or falsely claims to complete them here.",
            "- Fail when names, config keys, selectors, imports, commands, or interfaces change without the visible evidence showing the required dependent updates.",
            "- Treat changed supporting artifacts as completed evidence and unchanged recent context as reference only.",
            "- Do not block solely because other explicit target artifacts are still pending in the same task when the current proposal is a coherent partial step and remains consistent with the visible evidence.",
            "- Treat documentation command examples as illustrative usage unless the surrounding text explicitly claims they are required defaults or mandatory arguments.",
            "- Use semantic reasoning, not hardcoded feature vocabularies tied to one stack or one task.",
            "- Base findings only on the request, current excerpt, proposed excerpt, diff excerpt, supporting artifact evidence, diagnostics, and follow-up context.",
            "- Do not fail for harmless formatting shifts unless they conceal a behavioral regression.",
            "Supporting artifact evidence:",
            supporting_artifact_context,
            "Current file excerpt:",
            current_excerpt,
            "Proposed file excerpt:",
            proposed_excerpt,
            "Unified diff excerpt:",
            diff_excerpt,
            f"Return JSON only with this structure: {json.dumps(schema_shape, ensure_ascii=False)}",
        ]
    )


def planning_prompt(
    route: RouterOutput,
    snapshot: WorkspaceSnapshot | None,
) -> str:
    return planning_prompt_with_route(route, snapshot)


def planning_prompt_with_route(
    route: RouterOutput,
    snapshot: WorkspaceSnapshot | None,
) -> str:
    compact_snapshot = _compact_workspace_snapshot(snapshot, detail="decision")
    return "\n".join(
        [
            "Summarize the validated router plan into practical execution steps.",
            f"Route: {json.dumps(_compact_route(route), ensure_ascii=False)}",
            f"Workspace context: {json.dumps(compact_snapshot, ensure_ascii=False)}",
        ]
    )


def _compact_recent_messages(session: SessionState | None) -> list[dict[str, str]]:
    if session is None:
        return []
    return [
        {
            "role": item.role,
            "content": _trim_text(item.content, 220),
        }
        for item in session.messages[-6:]
    ]


def _compact_recent_calls(session: SessionState | None) -> list[dict[str, object]]:
    if session is None:
        return []
    return [
        {
            "tool": item.tool_name,
            "success": item.success,
            "summary": _trim_text(item.summary, 120),
        }
        for item in session.tool_calls[-6:]
    ]


def _compact_follow_up_context(session: SessionState | None) -> dict[str, object]:
    if session is None or session.follow_up_context is None:
        return {}
    follow_up = session.follow_up_context
    return {
        "previous_task": _trim_text(follow_up.previous_task or "", 220),
        "previous_root_goal": _trim_text(follow_up.previous_root_goal or "", 220),
        "previous_active_goal": _trim_text(follow_up.previous_active_goal or "", 220),
        "previous_next_action": follow_up.previous_next_action,
        "previous_intent": follow_up.previous_intent,
        "previous_requested_outcome": _trim_text(follow_up.previous_requested_outcome or "", 220),
        "previous_final_response": _trim_text(follow_up.previous_final_response or "", 220),
        "previous_interpreted_goal": _trim_text(follow_up.previous_interpreted_goal or "", 220),
        "previous_recommended_mode": follow_up.previous_recommended_mode,
        "previous_confidence": follow_up.previous_confidence,
        "previous_assumptions": [_trim_text(item, 140) for item in follow_up.previous_assumptions[:6]],
        "previous_constraints": [_trim_text(item, 140) for item in follow_up.previous_constraints[:6]],
        "target_paths": follow_up.target_paths[:8],
        "changed_files": follow_up.changed_files[:8],
        "read_files": follow_up.read_files[:8],
        "recent_commands": follow_up.recent_commands[:6],
        "last_error": _trim_text(follow_up.last_error or "", 240),
        "notes": [_trim_text(item, 140) for item in follow_up.notes[-6:]],
    }


def _compact_recent_diagnostics(session: SessionState | None) -> list[dict[str, object]]:
    return [
        {
            "category": item.category,
            "severity": item.severity,
            "summary": _trim_text(item.summary, 180),
            "files": item.file_hints[:4],
            "command": _trim_text(item.command or "", 120),
        }
        for item in _recent_diagnostics(session)
    ]


def _recent_diagnostics(session: SessionState | None):
    if session is None:
        return []
    diagnostics = list(session.diagnostics[-6:])
    if session.follow_up_context is not None:
        diagnostics.extend(session.follow_up_context.diagnostics[-6:])
    return diagnostics[-6:]


def _compact_workspace_snapshot(
    snapshot: WorkspaceSnapshot | None,
    *,
    detail: str = "standard",
) -> dict[str, object]:
    if snapshot is None:
        return {}
    important_limit = 10 if detail == "router" else 6
    focus_limit = 8 if detail == "router" else 4
    payload: dict[str, object] = {
        "project_labels": snapshot.project_labels[:8],
        "top_directories": snapshot.top_directories[:8],
        "manifests": snapshot.manifests[:6],
        "entrypoints": snapshot.entrypoints[:6],
        "focus_files": snapshot.focus_files[:focus_limit],
        "important_files": snapshot.important_files[:important_limit],
        "repo_summary": _trim_text(snapshot.repo_summary, 320),
        "likely_commands": snapshot.likely_commands[:4],
    }
    if detail != "router":
        payload["file_briefs"] = {
            path: _trim_text(snapshot.file_briefs.get(path, ""), 120)
            for path in snapshot.important_files[:4]
            if snapshot.file_briefs.get(path)
        }
    return payload


def _compact_route(route: RouterOutput | None) -> dict[str, object]:
    if route is None:
        return {}
    return {
        "user_goal": route.user_goal,
        "intent": route.intent,
        "entities": route.entities.model_dump(),
        "requested_outcome": route.requested_outcome,
        "action_plan": [item.model_dump() for item in route.action_plan],
        "needs_clarification": route.needs_clarification,
        "clarification_questions": route.clarification_questions,
        "confidence": route.confidence,
        "safe_to_execute": route.safe_to_execute,
        "repo_context_needed": route.repo_context_needed,
        "search_terms": route.search_terms,
        "relevant_extensions": route.relevant_extensions,
        "direct_response": route.direct_response,
    }


def _compact_task_understanding(task: TaskUnderstanding | None) -> dict[str, object]:
    if task is None:
        return {}
    return {
        "interpreted_goal": _trim_text(task.interpreted_goal, 220),
        "intent_category": task.intent_category,
        "conversation_relation": task.conversation_relation,
        "recommended_mode": task.recommended_mode,
        "confidence": task.confidence,
        "ambiguity_level": task.ambiguity_level,
        "risk_level": task.risk_level,
        "target_artifacts": [item.model_dump() for item in task.target_artifacts[:6]],
        "constraints": task.constraints[:6],
        "assumptions": task.assumptions[:6],
        "missing_info": task.missing_info[:4],
        "execution_plan": [item.model_dump() for item in task.execution_plan[:5]],
    }


def _compact_task_state(state: TaskState | None) -> dict[str, object]:
    if state is None:
        return {}
    return {
        "root_goal": _trim_text(state.root_goal, 220),
        "active_goal": _trim_text(state.active_goal, 220),
        "goal_relation": state.goal_relation,
        "output_expectation": _trim_text(state.output_expectation, 220),
        "current_user_intent": state.current_user_intent,
        "execution_strategy": state.execution_strategy,
        "open_problem": _trim_text(state.open_problem or "", 220),
        "verification_target": _trim_text(state.verification_target or "", 220),
        "next_action": state.next_action,
        "next_best_action": state.next_best_action,
        "confidence": state.confidence,
        "ambiguity_level": state.ambiguity_level,
        "risk_level": state.risk_level,
        "target_artifacts": [item.model_dump() for item in state.target_artifacts[:6]],
        "active_artifacts": [item.model_dump() for item in state.active_artifacts[:6]],
        "evidence": [item.model_dump() for item in state.evidence[:6]],
        "supplied_evidence": state.supplied_evidence[:6],
        "constraints": state.constraints[:6],
        "assumptions": state.assumptions[:6],
        "missing_info": state.missing_info[:4],
        "execution_outline": state.execution_outline[:5],
    }


def _compact_repair_context(context: ValidationFailureEvidence) -> dict[str, object]:
    prefer_balanced_failure = context.verification_scope == "runtime"
    payload = {
        "command": _trim_text(context.command, 140),
        "verification_scope": context.verification_scope,
        "artifact_paths": context.artifact_paths[:4],
        "failure_summary": _trim_balanced_text(context.failure_summary or context.summary, 220)
        if prefer_balanced_failure
        else _trim_text(context.failure_summary or context.summary, 180),
        "excerpt": _trim_balanced_text(context.excerpt or "", 260)
        if prefer_balanced_failure
        else _trim_text(context.excerpt or "", 180),
        "failure_focus": _runtime_failure_focus_lines(
            "\n".join(
                part
                for part in [
                    str(context.failure_summary or "").strip(),
                    str(context.excerpt or "").strip(),
                    str(context.summary or "").strip(),
                ]
                if part
            ),
            limit=6,
        )
        if prefer_balanced_failure
        else [],
        "missing_features": context.missing_features[:6],
        "file_hints": context.file_hints[:5],
        "line_hints": context.line_hints[:6],
        "action_hints": [_trim_text(item, 120) for item in context.action_hints[:3]],
        "repair_requirements": [_trim_text(item, 140) for item in context.repair_requirements[:4]],
    }
    brief = _compact_repair_brief(context, target_path=None)
    if brief:
        payload["repair_brief"] = brief
    return payload


def _compact_repair_brief(
    context: ValidationFailureEvidence,
    *,
    target_path: str | None,
) -> dict[str, object]:
    brief = context.repair_brief
    if brief is None:
        return {}
    target = str(target_path or "").strip()

    def _choose_paths(items: list[str], *, limit: int) -> list[str]:
        selected: list[str] = []
        normalized_target = target.lower()
        for candidate in items:
            path = str(candidate or "").strip()
            if not path:
                continue
            if normalized_target and _artifact_matches_path(target, path, path):
                if path not in selected:
                    selected.append(path)
        for candidate in items:
            path = str(candidate or "").strip()
            if path and path not in selected:
                selected.append(path)
        return selected[:limit]

    payload: dict[str, object] = {}
    if str(brief.failure_type or "").strip():
        payload["failure_type"] = str(brief.failure_type or "").strip()
    if str(brief.primary_target or "").strip():
        payload["primary_target"] = str(brief.primary_target or "").strip()
    if str(brief.locked_target or "").strip():
        payload["locked_target"] = str(brief.locked_target or "").strip()
    if brief.expected_semantics:
        payload["expected_semantics"] = [_trim_text(item, 140) for item in brief.expected_semantics[:3]]
    if brief.observed_semantics:
        payload["observed_semantics"] = [_trim_text(item, 140) for item in brief.observed_semantics[:3]]
    if brief.implicated_symbols:
        payload["implicated_symbols"] = [_trim_text(item, 60) for item in brief.implicated_symbols[:5]]
    if str(brief.implicated_region_hint or "").strip():
        payload["implicated_region_hint"] = _trim_text(str(brief.implicated_region_hint or "").strip(), 120)
    if brief.repair_constraints:
        payload["repair_constraints"] = [_trim_text(item, 120) for item in brief.repair_constraints[:3]]
    if brief.recent_failed_attempts:
        payload["recent_failed_attempts"] = [
            {
                "target": _trim_text(str(item.target or "").strip(), 80) if str(item.target or "").strip() else None,
                "strategy": _trim_text(str(item.strategy or "").strip(), 48),
                "result": _trim_text(str(item.result or "").strip(), 48),
                "reason": _trim_text(str(item.reason or "").strip(), 100) if str(item.reason or "").strip() else None,
            }
            for item in brief.recent_failed_attempts[:3]
        ]
    allowed_files = _choose_paths(brief.allowed_files, limit=4)
    if allowed_files:
        payload["allowed_files"] = allowed_files
    forbidden_files = _choose_paths(brief.forbidden_files, limit=4)
    if forbidden_files:
        payload["forbidden_files"] = forbidden_files
    return payload


def _targeted_compact_repair_context(
    context: ValidationFailureEvidence,
    *,
    target_path: str,
) -> dict[str, object]:
    normalized_target = str(target_path or "").strip()
    target_markers = _artifact_scope_markers(normalized_target)
    target_tokens = {
        normalized_target.lower(),
        Path(normalized_target).name.lower(),
        Path(normalized_target).stem.lower(),
    }
    other_markers: set[str] = set()
    other_tokens: set[str] = set()
    for candidate in [*context.artifact_paths, *context.file_hints]:
        path = str(candidate or "").strip()
        if not path or _artifact_matches_path(normalized_target, path, path):
            continue
        other_markers.update(_artifact_scope_markers(path))
        other_tokens.update(
            {
                path.lower(),
                Path(path).name.lower(),
                Path(path).stem.lower(),
            }
        )

    def _select_repair_items(items: list[str], *, limit: int, trim: int) -> list[str]:
        targeted: list[str] = []
        general: list[str] = []
        for raw in items:
            text = str(raw or "").strip()
            if not text:
                continue
            lowered = text.lower()
            if any(token and token in lowered for token in target_tokens):
                if text not in targeted:
                    targeted.append(text)
                continue
            if any(token and token in lowered for token in other_tokens):
                continue
            current_score = _scope_relevance_score(text, target_markers)
            other_score = _scope_relevance_score(text, other_markers)
            if current_score > 0 and current_score >= other_score:
                if text not in targeted:
                    targeted.append(text)
                continue
            if other_score > current_score:
                continue
            if text not in general:
                general.append(text)
        chosen = targeted + [item for item in general if item not in targeted]
        if not chosen:
            chosen = [str(item or "").strip() for item in items if str(item or "").strip()]
        return [_trim_text(item, trim) for item in chosen[:limit]]

    def _select_paths(items: list[str], *, limit: int) -> list[str]:
        targeted: list[str] = []
        related: list[str] = []
        for raw in items:
            path = str(raw or "").strip()
            if not path:
                continue
            if _artifact_matches_path(normalized_target, path, path):
                if path not in targeted:
                    targeted.append(path)
                continue
            if path not in related:
                related.append(path)
        combined = targeted + [item for item in related if item not in targeted]
        if context.verification_scope == "runtime":
            runtime_relevant = [
                item
                for item in combined
                if Path(item).suffix.lower()
                in {".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt", ".gd"}
                or item.lower().startswith("tests/")
                or "/tests/" in f"/{item.lower()}"
            ]
            if runtime_relevant:
                combined = runtime_relevant + [item for item in combined if item not in runtime_relevant]
        return combined[:limit]

    compact = _compact_repair_context(context)
    compact["target_path"] = normalized_target
    artifact_limit = 3 if context.verification_scope == "runtime" else 4
    file_hint_limit = 4 if context.verification_scope == "runtime" else 5
    compact["artifact_paths"] = _select_paths(context.artifact_paths, limit=artifact_limit)
    compact["file_hints"] = _select_paths(context.file_hints, limit=file_hint_limit)
    compact["repair_requirements"] = _select_repair_items(
        context.repair_requirements,
        limit=4,
        trim=140,
    )
    compact["action_hints"] = _select_repair_items(
        context.action_hints,
        limit=3,
        trim=120,
    )
    if context.verification_scope == "runtime":
        compact["failure_summary"] = _trim_balanced_text(context.failure_summary or context.summary, 160)
        compact["excerpt"] = _trim_balanced_text(context.excerpt or "", 180)
        compact["failure_focus"] = _targeted_runtime_failure_focus_lines(
            "\n".join(
                part
                for part in [
                    str(context.excerpt or "").strip(),
                    str(context.failure_summary or "").strip(),
                    str(context.summary or "").strip(),
                ]
                if part
            ),
            target_path=normalized_target,
            other_paths=[path for path in [*context.artifact_paths, *context.file_hints] if path and not _artifact_matches_path(normalized_target, path, path)],
            limit=6,
        )
    brief = _compact_repair_brief(context, target_path=normalized_target)
    if brief:
        compact["repair_brief"] = brief
    return compact


def _compact_proposed_update_review(review: ProposedUpdateReview) -> dict[str, object]:
    return {
        "safe_to_write": review.safe_to_write,
        "summary": _trim_text(review.summary, 220),
        "confidence": review.confidence,
        "blocking_issues": [_trim_text(item, 180) for item in review.blocking_issues[:4]],
        "preservation_risks": [_trim_text(item, 180) for item in review.preservation_risks[:4]],
        "repair_hints": [_trim_text(item, 180) for item in review.repair_hints[:4]],
    }


def _compact_generation_focus(
    route: RouterOutput,
    session: SessionState,
    path: str,
) -> dict[str, object]:
    related_targets = [item for item in route.entities.target_paths if item and item != path][:6]
    task_state = session.task_state
    return {
        "target_path": path,
        "active_goal": _trim_text(task_state.active_goal if task_state is not None else route.requested_outcome, 180),
        "output_expectation": _trim_text(
            task_state.output_expectation if task_state is not None else route.requested_outcome,
            180,
        ),
        "verification_target": _trim_text(task_state.verification_target or "", 200) if task_state is not None else "",
        "constraints": (task_state.constraints[:4] if task_state is not None else []) or route.entities.constraints[:4],
        "related_targets": related_targets,
    }


def _compact_repair_file_focus(
    file_focus: dict[str, object],
    *,
    target_path: str | None = None,
) -> dict[str, object]:
    normalized_target = str(target_path or file_focus.get("target_path") or "").strip()
    target_tokens = {
        normalized_target.lower(),
        Path(normalized_target).name.lower(),
        Path(normalized_target).stem.lower(),
    }
    filtered_requirements: list[str] = []
    for raw in file_focus.get("current_write_requirements", [])[:6]:
        text = str(raw or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if any(token and token in lowered for token in target_tokens):
            filtered_requirements.append(_trim_text(text, 140))
            continue
        if re.search(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+|\b[A-Za-z0-9_.-]+\.py\b", text):
            continue
        filtered_requirements.append(_trim_text(text, 140))
    return {
        "target_path": file_focus.get("target_path"),
        "artifact_role": file_focus.get("artifact_role"),
        "literal_constraints": file_focus.get("literal_constraints", [])[:4],
        "current_write_requirements": filtered_requirements[:4],
    }


def _compact_repair_update_prompt(
    route: RouterOutput,
    session: SessionState,
    *,
    path: str,
    current_content: str,
    file_focus: dict[str, object],
    explicit_constraints: str,
    repair_context: ValidationFailureEvidence,
    repair_strategy: str | None,
    review_feedback: ProposedUpdateReview | None,
) -> str:
    targeted_context = _targeted_compact_repair_context(repair_context, target_path=path)
    failure_focus = [
        _trim_text(str(item or "").strip(), 160)
        for item in targeted_context.get("failure_focus", [])
        if str(item or "").strip()
    ]
    validation_summary_parts: list[str] = []
    for candidate in (
        str(repair_context.failure_summary or "").strip(),
        str(repair_context.summary or "").strip(),
        str(repair_context.excerpt or "").strip(),
    ):
        if not candidate:
            continue
        if candidate in validation_summary_parts:
            continue
        validation_summary_parts.append(candidate)
    validation_context_parts = [
        f"command={_trim_text(repair_context.command, 120)}" if str(repair_context.command or "").strip() else "",
        f"scope={repair_context.verification_scope}",
        (
            "summary="
            + _trim_text(" | ".join(validation_summary_parts), 220)
        ),
    ]
    sections = [
        "Produce the full file content for exactly one file.",
        f"Target path: {path}",
        f"Latest user request: {_trim_text(session.task, 220)}",
        (
            "Repair objective: make the failed "
            f"{repair_context.verification_scope} validation pass while preserving the original requested behavior."
        ),
        "Validation-guided repair context: "
        + " ".join(part for part in validation_context_parts if part),
    ]
    compact_focus = _compact_repair_file_focus(file_focus, target_path=path)
    repair_brief = targeted_context.get("repair_brief") or {}
    if compact_focus.get("current_write_requirements"):
        sections.append(
            "Repair-scoped requirements: "
            + "; ".join(
                _trim_text(str(item or "").strip(), 140)
                for item in compact_focus.get("current_write_requirements", [])[:3]
                if str(item or "").strip()
            )
        )
    if repair_brief.get("locked_target") or repair_brief.get("primary_target"):
        target_summary = repair_brief.get("locked_target") or repair_brief.get("primary_target")
        sections.append(f"Primary repair target: {target_summary}")
    if repair_brief.get("expected_semantics"):
        sections.append(
            "Expected semantics: "
            + " | ".join(_trim_text(str(item or "").strip(), 160) for item in repair_brief.get("expected_semantics", [])[:3])
        )
    if repair_brief.get("observed_semantics"):
        sections.append(
            "Observed semantics: "
            + " | ".join(_trim_text(str(item or "").strip(), 160) for item in repair_brief.get("observed_semantics", [])[:3])
        )
    semantic_deltas = _repair_semantic_delta_lines(repair_context)
    if semantic_deltas:
        sections.append(
            "Minimal semantic delta: "
            + " | ".join(_trim_text(item, 180) for item in semantic_deltas[:2])
        )
    if failure_focus:
        sections.append("Failure focus: " + " | ".join(failure_focus[:3]))
    if repair_brief.get("implicated_region_hint") or repair_brief.get("implicated_symbols"):
        region_parts: list[str] = []
        if repair_brief.get("implicated_region_hint"):
            region_parts.append(f"region={repair_brief['implicated_region_hint']}")
        if repair_brief.get("implicated_symbols"):
            region_parts.append(
                "symbols=" + ", ".join(_trim_text(str(item or "").strip(), 40) for item in repair_brief.get("implicated_symbols", [])[:4])
            )
        if region_parts:
            sections.append("Repair focus: " + " ".join(region_parts))
    support_excerpt_limit = 500 if repair_context.verification_scope == "runtime" else 180
    support_max_files = 2 if repair_context.verification_scope == "runtime" else 1
    related_context = _repair_related_file_context(
        session,
        target_path=path,
        repair_context=repair_context,
        excerpt_limit=support_excerpt_limit,
        max_files=support_max_files,
    )
    if related_context != "none":
        sections.append(f"Supporting file hints: {related_context}")
    runtime_hints = _targeted_runtime_prompt_hints(
        path=path,
        current_content=current_content,
        supporting_context=related_context,
        targeted_context=targeted_context,
    )
    mutation_anchors = _mandatory_mutation_anchors(
        path=path,
        current_content=current_content,
        repair_context=repair_context,
        review_feedback=review_feedback,
    )
    sections.extend(
        [
            _repair_rules(repair_strategy),
            "Keep the update narrow and preserve unrelated existing behavior, imports, and interfaces.",
        ]
    )
    if repair_brief.get("repair_constraints"):
        sections.append(
            "Repair constraints: "
            + " | ".join(_trim_text(str(item or "").strip(), 140) for item in repair_brief.get("repair_constraints", [])[:3])
        )
    if repair_brief.get("allowed_files"):
        sections.append(
            "Allowed repair files: "
            + ", ".join(_trim_text(str(item or "").strip(), 80) for item in repair_brief.get("allowed_files", [])[:4])
        )
    if repair_brief.get("forbidden_files"):
        sections.append(
            "Avoid drifting into other files without strong new evidence: "
            + ", ".join(_trim_text(str(item or "").strip(), 80) for item in repair_brief.get("forbidden_files", [])[:4])
        )
    if repair_brief.get("recent_failed_attempts"):
        sections.append(
            "Recent failed repair attempts: "
            + " | ".join(
                _trim_text(
                    " ".join(
                        part
                        for part in [
                            str(item.get("target") or "").strip(),
                            str(item.get("strategy") or "").strip(),
                            str(item.get("result") or "").strip(),
                            str(item.get("reason") or "").strip(),
                        ]
                        if part
                    ),
                    160,
                )
                for item in repair_brief.get("recent_failed_attempts", [])[:3]
            )
        )
    if runtime_hints:
        sections.append(
            "Targeted runtime hints: "
            + " ".join(_trim_text(item, 220) for item in runtime_hints[:6])
        )
    fixture_hints = _runtime_support_file_prompt_hints(
        path=path,
        current_content=current_content,
        repair_context=repair_context,
    )
    if fixture_hints:
        sections.append(
            "Runtime support file hints: "
            + " ".join(_trim_text(item, 220) for item in fixture_hints[:4])
        )
    if mutation_anchors:
        sections.append(_mandatory_mutation_rules(mutation_anchors))
    if review_feedback is not None:
        sections.extend(
            [
                f"Self-review feedback on the previous proposal: {json.dumps(_compact_proposed_update_review(review_feedback), ensure_ascii=False)}",
                _direct_review_corrections(review_feedback),
                "Use that feedback to make a smaller, safer update while preserving unrelated existing behavior.",
            ]
        )
    sections.extend(
        [
            "Current file content:",
            current_content,
            "Update this file to satisfy the request. Return the full updated file content only.",
            "Do not add markdown fences or explanations.",
        ]
    )
    return "\n\n".join(sections)


def _compact_repair_retry_prompt(
    route: RouterOutput,
    session: SessionState,
    *,
    path: str,
    current_content: str,
    repair_context: ValidationFailureEvidence,
    repair_strategy: str | None,
    review_feedback: ProposedUpdateReview,
) -> str:
    support_max_files = 2 if repair_context.verification_scope == "runtime" else 1
    related_context = _repair_related_file_context(
        session,
        target_path=path,
        repair_context=repair_context,
        excerpt_limit=320,
        max_files=support_max_files,
    )
    targeted_context = _targeted_compact_repair_context(repair_context, target_path=path)
    runtime_hints = _targeted_runtime_prompt_hints(
        path=path,
        current_content=current_content,
        supporting_context=related_context,
        targeted_context=targeted_context,
    )
    mutation_anchors = _mandatory_mutation_anchors(
        path=path,
        current_content=current_content,
        repair_context=repair_context,
        review_feedback=review_feedback,
    )

    sections = [
        "Produce the full file content for exactly one file.",
        f"Target path: {path}",
        f"Latest user request: {_trim_text(session.task, 220)}",
        (
            "Repair objective: make the failed "
            f"{repair_context.verification_scope} validation pass while preserving the original requested behavior."
        ),
        f"Failed command: {_trim_text(repair_context.command, 120)}",
        f"Failure summary: {_trim_text(repair_context.failure_summary or repair_context.summary, 180)}",
        _repair_rules(repair_strategy),
        "Keep the update narrow and preserve unrelated existing behavior.",
    ]
    repair_brief = targeted_context.get("repair_brief") or {}
    if repair_brief.get("locked_target") or repair_brief.get("primary_target"):
        sections.append(
            "Primary repair target: "
            + str(repair_brief.get("locked_target") or repair_brief.get("primary_target") or "").strip()
        )
    if repair_brief.get("expected_semantics"):
        sections.append(
            "Expected semantics: "
            + " | ".join(_trim_text(str(item or "").strip(), 140) for item in repair_brief.get("expected_semantics", [])[:3])
        )
    if repair_brief.get("observed_semantics"):
        sections.append(
            "Observed semantics: "
            + " | ".join(_trim_text(str(item or "").strip(), 140) for item in repair_brief.get("observed_semantics", [])[:3])
        )
    semantic_deltas = _repair_semantic_delta_lines(repair_context)
    if semantic_deltas:
        sections.append(
            "Minimal semantic delta: "
            + " | ".join(_trim_text(item, 160) for item in semantic_deltas[:2])
        )
    if related_context != "none":
        sections.append(f"Supporting file hints: {related_context}")
    if repair_brief.get("allowed_files"):
        sections.append(
            "Allowed repair files: "
            + ", ".join(_trim_text(str(item or "").strip(), 80) for item in repair_brief.get("allowed_files", [])[:4])
        )
    if repair_brief.get("forbidden_files"):
        sections.append(
            "Avoid drifting into other files without strong new evidence: "
            + ", ".join(_trim_text(str(item or "").strip(), 80) for item in repair_brief.get("forbidden_files", [])[:4])
        )
    if runtime_hints:
        sections.append(
            "Targeted runtime hints: "
            + " ".join(_trim_text(item, 180) for item in runtime_hints[:6])
        )
    fixture_hints = _runtime_support_file_prompt_hints(
        path=path,
        current_content=current_content,
        repair_context=repair_context,
    )
    if fixture_hints:
        sections.append(
            "Runtime support file hints: "
            + " ".join(_trim_text(item, 180) for item in fixture_hints[:4])
        )
    if mutation_anchors:
        sections.append(_mandatory_mutation_rules(mutation_anchors))
    sections.extend(
        [
            _direct_review_corrections(review_feedback),
            "Current file content:",
            current_content,
            "Update this file to satisfy the request. Return the full updated file content only.",
            "Do not add markdown fences or explanations.",
        ]
    )
    return "\n\n".join(sections)


def _mandatory_mutation_rules(anchors: list[str]) -> str:
    lines = [
        "Mandatory mutation anchors:",
        "- At least one listed current target line or behavior anchor must change in the returned file.",
        "- If every listed anchor remains identical, the repair is incomplete.",
    ]
    for anchor in anchors[:3]:
        lines.append(f"- {anchor}")
    return "\n".join(lines)


def _direct_review_corrections(review: ProposedUpdateReview) -> str:
    lines = ["Required corrections from the rejected draft:"]
    for issue in review.blocking_issues[:2]:
        text = _trim_text(issue, 220)
        if text:
            lines.append(f"- Blocking issue: {text}")
    for hint in review.repair_hints[:2]:
        text = _trim_text(hint, 220)
        if text:
            lines.append(f"- Repair direction: {text}")
    combined = " ".join(review.blocking_issues + review.repair_hints).lower()
    if "sys.argv" in combined:
        lines.append("- If the updated file references sys.argv, add import sys before using it.")
    if "sys.argv[1:]" in combined or "sys.argv[2:]" in combined or "launcher prefix" in combined:
        lines.append(
            "- Do not use sys.argv[1:] or sys.argv[2:] here; argparse should receive only the trailing CLI arguments after the python -m module name."
        )
        lines.append(
            "- For a runtime argv shaped like ['python', '-m', '<module>', ...], sys.argv[3:] is the equivalent slice of only the trailing CLI arguments."
        )
    return "\n".join(lines)


def _repair_semantic_delta_lines(
    repair_context: ValidationFailureEvidence,
    *,
    limit: int = 2,
) -> list[str]:
    brief = getattr(repair_context, "repair_brief", None)
    if brief is None:
        return []

    expected_items = [
        _repair_semantic_value_text(item)
        for item in getattr(brief, "expected_semantics", [])
        if _repair_semantic_value_text(item)
    ]
    observed_items = [
        _repair_semantic_value_text(item)
        for item in getattr(brief, "observed_semantics", [])
        if _repair_semantic_value_text(item)
    ]
    if not expected_items or not observed_items:
        return []

    deltas: list[str] = []
    for observed_value, expected_value in zip(observed_items, expected_items):
        compared_observed, compared_expected = _first_mismatching_semantic_line_pair(
            observed_value,
            expected_value,
        )
        delta = _render_semantic_delta(compared_observed, compared_expected)
        if not delta or delta in deltas:
            continue
        deltas.append(delta)
        if len(deltas) >= limit:
            break
    return deltas


def _repair_semantic_value_text(raw: object) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    prefixes = (
        "Validation should produce:",
        "Validation currently produces:",
        "Required validation features must be present:",
    )
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def _first_mismatching_semantic_line_pair(observed_value: str, expected_value: str) -> tuple[str, str]:
    observed = str(observed_value or "")
    expected = str(expected_value or "")
    observed_lines = [line for line in observed.splitlines() if line.strip()]
    expected_lines = [line for line in expected.splitlines() if line.strip()]
    paired = zip(observed_lines, expected_lines)
    for observed_line, expected_line in paired:
        if observed_line != expected_line:
            return observed_line, expected_line
    if observed_lines and expected_lines:
        return observed_lines[0], expected_lines[0]
    return observed, expected


def _render_semantic_delta(observed_value: str, expected_value: str) -> str | None:
    observed = str(observed_value or "")
    expected = str(expected_value or "")
    if not observed or not expected or observed == expected:
        return None

    prefix_length = 0
    max_prefix = min(len(observed), len(expected))
    while prefix_length < max_prefix and observed[prefix_length] == expected[prefix_length]:
        prefix_length += 1
    observed_remainder = observed[prefix_length:]
    expected_remainder = expected[prefix_length:]

    suffix_length = 0
    max_suffix = min(len(observed_remainder), len(expected_remainder))
    while suffix_length < max_suffix and observed_remainder[-(suffix_length + 1)] == expected_remainder[-(suffix_length + 1)]:
        suffix_length += 1

    if suffix_length:
        observed_middle = observed_remainder[:-suffix_length]
        expected_middle = expected_remainder[:-suffix_length]
        shared_suffix = observed_remainder[-suffix_length:]
    else:
        observed_middle = observed_remainder
        expected_middle = expected_remainder
        shared_suffix = ""
    shared_prefix = observed[:prefix_length]

    context_parts: list[str] = []
    if shared_prefix.strip():
        context_parts.append(f"shared prefix '{_trim_repair_delta_context(shared_prefix, keep_tail=True)}'")
    if shared_suffix.strip():
        context_parts.append(f"shared suffix '{_trim_repair_delta_context(shared_suffix, keep_tail=False)}'")

    action = ""
    if observed_middle and expected_middle:
        action = (
            f"Replace observed-only text '{_trim_repair_delta_value(observed_middle)}' "
            f"with expected text '{_trim_repair_delta_value(expected_middle)}'"
        )
    elif observed_middle:
        action = f"Remove observed-only text '{_trim_repair_delta_value(observed_middle)}'"
    elif expected_middle:
        action = f"Insert expected-only text '{_trim_repair_delta_value(expected_middle)}'"
    else:
        return None

    if context_parts:
        return action + " between " + " and ".join(context_parts) + "."
    return action + "."


def _trim_repair_delta_context(text: str, *, keep_tail: bool, limit: int = 40) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    if keep_tail:
        return "…" + normalized[-(limit - 1) :]
    return normalized[: limit - 1] + "…"


def _trim_repair_delta_value(text: str, limit: int = 48) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"


def _repair_required_literal_anchors(
    *,
    path: str,
    current_content: str,
    repair_context: ValidationFailureEvidence,
    limit: int = 3,
) -> list[str]:
    focus_text = "\n".join(
        part
        for part in [
            str(repair_context.excerpt or "").strip(),
            str(repair_context.failure_summary or "").strip(),
            str(repair_context.summary or "").strip(),
        ]
        if part
    )
    scoped_lines = _targeted_runtime_failure_focus_lines(
        focus_text,
        target_path=path,
        other_paths=_repair_other_paths(repair_context, target_path=path),
        limit=8,
    )
    requirement_lines = list(scoped_lines)
    normalized_path = str(path or "").strip().lower()
    path_obj = Path(normalized_path)
    target_markers = {
        marker
        for marker in _artifact_scope_markers(path)
        if marker
        and (
            marker == normalized_path
            or marker == path_obj.name
            or marker == path_obj.stem
            or marker not in _GENERIC_ARTIFACT_MARKERS
        )
    }
    target_tokens = {
        token
        for token in [normalized_path, path_obj.name.lower(), path_obj.stem.lower()]
        if token and token not in _GENERIC_ARTIFACT_MARKERS
    }
    for raw in focus_text.splitlines():
        line = str(raw or "").strip()
        lowered = line.lower()
        references_target = any(marker and marker in lowered for marker in target_markers) or any(
            token and token in lowered for token in target_tokens
        )
        if line and references_target and line not in requirement_lines:
            requirement_lines.append(line)
    requirements = _literal_validation_requirements(requirement_lines)
    if not requirements:
        return []

    anchors: list[str] = []
    seen: set[tuple[str, int]] = set()
    for literal, minimum_count in requirements:
        key = (literal, minimum_count)
        if key in seen:
            continue
        seen.add(key)
        if not literal:
            continue
        exact_count = current_content.count(literal)
        if exact_count >= minimum_count:
            continue
        near_match = _literal_near_match_in_content(literal, current_content)
        if near_match and near_match != literal:
            if minimum_count > 1:
                anchors.append(
                    f"Replace near-match literal {near_match!r} with the exact required literal {literal!r} and ensure it appears at least {minimum_count} times in {path}."
                )
            else:
                anchors.append(
                    f"Replace near-match literal {near_match!r} with the exact required literal {literal!r} in {path}."
                )
        elif minimum_count > 1:
            anchors.append(
                f"Ensure the exact required literal {literal!r} appears at least {minimum_count} times in {path}."
            )
        else:
            anchors.append(
                f"Ensure the exact required literal {literal!r} appears verbatim in {path}."
            )
        if len(anchors) >= limit:
            break
    return anchors


def _literal_validation_requirements(lines: list[str]) -> list[tuple[str, int]]:
    requirements: list[tuple[str, int]] = []
    count_pattern = re.compile(
        r"assert(?:GreaterEqual|Equal)\([^\\n]*?count\(\s*(?P<literal>'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")\s*\)\s*,\s*(?P<count>\d+)",
    )
    contains_pattern = re.compile(
        r"assertIn\(\s*(?P<literal>'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")\s*,",
    )
    for raw in lines:
        stripped = str(raw or "").strip()
        if not stripped:
            continue
        count_match = count_pattern.search(stripped)
        if count_match is not None:
            literal = _literal_validation_token(count_match.group("literal"))
            if literal:
                try:
                    minimum_count = max(int(count_match.group("count") or 0), 1)
                except ValueError:
                    minimum_count = 1
                requirements.append((literal, minimum_count))
            continue
        contains_match = contains_pattern.search(stripped)
        if contains_match is not None:
            literal = _literal_validation_token(contains_match.group("literal"))
            if literal:
                requirements.append((literal, 1))
    return requirements


def _literal_validation_token(raw: str | None) -> str | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        value = text
    normalized = str(value or "")
    return normalized[:160] if normalized else None


def _literal_near_match_in_content(literal: str, current_content: str) -> str | None:
    text = str(literal or "")
    content = str(current_content or "")
    if not text or not content:
        return None
    pattern = re.escape(text)
    pattern = pattern.replace("\\'", "['\"]").replace('\\"', "['\"]")
    pattern = pattern.replace("'", "['\"]").replace('"', "['\"]")
    match = re.search(pattern, content)
    if match is None:
        return None
    candidate = str(match.group(0) or "")
    return candidate or None


def _mandatory_mutation_anchors(
    *,
    path: str,
    current_content: str,
    repair_context: ValidationFailureEvidence,
    review_feedback: ProposedUpdateReview | None,
) -> list[str]:
    anchors: list[str] = []

    line_excerpt = ""
    focused_line_hints = _repair_target_line_hints(
        path=path,
        current_content=current_content,
        repair_context=repair_context,
    )
    if focused_line_hints:
        line_excerpt = _line_focused_excerpt(
            current_content,
            line_hints=focused_line_hints,
            limit=220,
            before_radius=0,
            after_radius=0,
        )
    if line_excerpt:
        anchors.append(f"Change the implicated current lines in {path}:\n{line_excerpt}")
    elif repair_context.verification_scope == "runtime":
        focus_lines = [
            line
            for line in _targeted_runtime_failure_focus_lines(
                "\n".join(
                    part
                    for part in [
                        str(repair_context.excerpt or "").strip(),
                        str(repair_context.failure_summary or "").strip(),
                        str(repair_context.summary or "").strip(),
                    ]
                    if part
                ),
                target_path=path,
                other_paths=_repair_other_paths(repair_context, target_path=path),
                limit=4,
            )
            if line and not line.startswith("File ") and not line.startswith("Traceback")
        ]
        if focus_lines:
            anchors.append(
                "Resolve the failure focus tied to this file: "
                + " | ".join(_trim_text(item, 140) for item in focus_lines[:2])
            )

    for literal_anchor in _repair_required_literal_anchors(
        path=path,
        current_content=current_content,
        repair_context=repair_context,
    ):
        anchors.append(literal_anchor)

    for delta in _repair_semantic_delta_lines(repair_context, limit=2):
        anchors.append(
            "Apply this exact semantic delta in the behavior produced by this file: "
            + _trim_text(delta, 220)
        )

    if review_feedback is not None:
        if review_feedback.blocking_issues:
            anchors.append(
                "Previous proposal was rejected because: "
                + _trim_text(review_feedback.blocking_issues[0], 220)
            )
        if review_feedback.repair_hints:
            anchors.append(
                "Required repair direction: "
                + _trim_text(review_feedback.repair_hints[0], 220)
            )

    return anchors[:3]


def _repair_target_line_hints(
    *,
    path: str,
    current_content: str,
    repair_context: ValidationFailureEvidence,
) -> list[int]:
    hints: list[int] = []
    brief = getattr(repair_context, "repair_brief", None)
    region_hint = str(getattr(brief, "implicated_region_hint", "") or "").strip()
    prefix = f"{path}:line "
    if region_hint.startswith(prefix):
        suffix = region_hint[len(prefix) :].strip()
        try:
            line = int(suffix)
        except ValueError:
            line = 0
        if line > 0:
            hints.append(line)
    if hints:
        return hints
    supporting_hints = _supporting_file_line_hints(
        current_content,
        repair_context=repair_context,
        target_path=path,
        limit=4,
        allow_target_token_fallback=False,
    )
    if supporting_hints:
        return supporting_hints
    return _semantic_output_anchor_line_hints(
        current_content,
        repair_context=repair_context,
        limit=4,
    )


def _semantic_output_anchor_line_hints(
    text: str,
    *,
    repair_context: ValidationFailureEvidence,
    limit: int = 4,
) -> list[int]:
    semantic_deltas = _repair_semantic_delta_lines(repair_context, limit=1)
    if not semantic_deltas:
        return []

    lines = [str(line or "").rstrip() for line in str(text or "").splitlines()]
    if not lines:
        return []

    brief = getattr(repair_context, "repair_brief", None)
    symbol_tokens: set[str] = set()
    if brief is not None:
        for symbol in getattr(brief, "implicated_symbols", []) or []:
            normalized = str(symbol or "").strip().lower()
            if not normalized or normalized.startswith("test_"):
                continue
            for token in re.split(r"[^a-z0-9_]+", normalized):
                token = token.strip()
                if len(token) >= 4:
                    symbol_tokens.add(token)

    scored: list[tuple[int, int]] = []
    for index, line in enumerate(lines, start=1):
        lowered = line.lower()
        if not lowered.strip():
            continue
        score = 0
        if re.search(r"""(^|\W)f["']""", line):
            score += 5
        if ".format(" in lowered:
            score += 4
        if re.search(r"\b(print|return)\b", lowered) or any(
            marker in lowered for marker in (".write(", ".append(", "textcontent", "innerhtml", "innertext")
        ):
            score += 3
        if "=" in line:
            score += 1
        if any(token and token in lowered for token in symbol_tokens):
            score += 2
        if score > 0:
            scored.append((score, index))

    if not scored:
        return []

    best_line = max(scored, key=lambda item: (item[0], -item[1]))[1]
    hints: list[int] = []
    previous_index = best_line - 1
    if previous_index >= 1:
        previous_line = lines[previous_index - 1].strip().lower()
        if previous_line.startswith(("if ", "elif ", "else", "for ", "while ", "with ", "case ")):
            hints.append(previous_index)
    hints.append(best_line)
    return hints[:limit]


def _targeted_runtime_prompt_hints(
    *,
    path: str,
    current_content: str,
    supporting_context: str,
    targeted_context: dict[str, object],
) -> list[str]:
    normalized_path = str(path or "").strip().lower()
    if not normalized_path.endswith(".py"):
        return []

    lowered_current = str(current_content or "").lower()
    if not any(token in lowered_current for token in ("parse_args(", "parse_known_args(", "argparse.argumentparser")):
        return []

    lowered_support = str(supporting_context or "").lower()
    focus_text = "\n".join(
        str(item or "")
        for item in targeted_context.get("failure_focus", [])
        if str(item or "").strip()
    ).lower()
    failure_text = "\n".join(
        [
            str(targeted_context.get("failure_summary") or "").strip(),
            str(targeted_context.get("excerpt") or "").strip(),
            focus_text,
            "\n".join(
                str(item or "").strip()
                for item in targeted_context.get("file_hints", [])
                if str(item or "").strip()
            ),
        ]
    ).strip()
    lowered_failure = failure_text.lower()

    hints: list[str] = []
    patched_runtime_argv = "__main__.sys.argv" in lowered_support or "__main__.sys.argv" in lowered_failure
    python_m_launcher = (
        "'-m'" in supporting_context
        or "\"-m\"" in failure_text
        or "unrecognized arguments: -m" in lowered_failure
        or "python -m" in lowered_failure
    )
    direct_main_invocation = "__main__.main()" in lowered_support or "__main__.main()" in lowered_failure
    if (patched_runtime_argv and python_m_launcher) or (
        direct_main_invocation and python_m_launcher
    ):
        hints.append(
            "The failing tests patch __main__.sys.argv with launcher tokens like ['python', '-m', '<module>', ...]. Do not pass 'python', '-m', or the module name into argparse; only the trailing CLI arguments should reach the parser."
        )
        hints.append(
            "For the patched argv shown by the tests, argparse should see ['Ada'] in the named case and [] in the default-name case."
        )
        hints.append(
            "If the repair reads sys.argv, add import sys before using it."
        )
        hints.append(
            "Do not use sys.argv[1:] or sys.argv[2:] here. For a python -m style launcher like ['python', '-m', '<module>', ...], argparse should receive only the trailing CLI arguments after the module name."
        )
        hints.append(
            "Keep main() callable without positional arguments. An optional argv=None parameter is acceptable when the function must derive CLI arguments from the patched runtime argv."
        )
        hints.append(
            "When no explicit argv was passed, derive it from the runtime argv and strip a leading python -m launcher prefix before calling argparse."
        )
    if "parse_known_args(" in lowered_current and (
        "hello, ada!" in focus_text or "hello, world!" in focus_text
    ):
        hints.append(
            "Do not just ignore unknown flags if the module name would still become the positional argument; keep the explicit-name greeting and the default greeting both correct."
        )
    return hints[:6]


def _runtime_support_file_prompt_hints(
    *,
    path: str,
    current_content: str,
    repair_context: ValidationFailureEvidence,
) -> list[str]:
    if repair_context.verification_scope != "runtime":
        return []
    normalized_path = str(path or "").strip()
    if not normalized_path:
        return []
    suffix = Path(normalized_path).suffix.lower()
    if suffix in {".py", ".pyi", ".md", ".markdown", ".rst"}:
        return []

    support_markers = {
        "data",
        "fixture",
        "fixtures",
        "sample",
        "samples",
        "sample-data",
        "sample_data",
        "test",
        "test-data",
        "test_data",
        "testdata",
        "tests",
    }
    lowered_path = normalized_path.lower()
    path_parts = {part.lower() for part in Path(normalized_path).parts}
    failure_text = "\n".join(
        part
        for part in [
            str(repair_context.excerpt or "").strip(),
            str(repair_context.failure_summary or "").strip(),
            str(repair_context.summary or "").strip(),
        ]
        if part
    ).lower()
    if not path_parts.intersection(support_markers) and lowered_path not in failure_text:
        return []

    hints = [
        "This file is runtime input or fixture data, not source code or documentation.",
        "Write only the minimal raw content needed for the failing validation to pass.",
        "Do not add explanatory prose, headings, comments, or extra sample text unless the failure evidence explicitly requires them.",
    ]
    if "assertionerror" in failure_text or "self.assertequal" in failure_text:
        hints.append(
            "If the failure shows exact expected values, prefer fixture contents that produce only that expected result and no extra tokens or records."
        )
    lowered_current = str(current_content or "").lower()
    if lowered_current and any(
        marker in lowered_current
        for marker in ("sample text", "test data file", "functionality", "placeholder", "this is")
    ):
        hints.append(
            "Replace placeholder or descriptive sample prose with concrete fixture data instead of mixing explanation into the file."
        )
    return hints


def _artifact_scoped_focus(
    route: RouterOutput,
    session: SessionState | None,
    path: str,
    *,
    current_content: str | None = None,
) -> dict[str, object]:
    task_state = session.task_state if session is not None else None
    current_markers = _artifact_scope_markers(path)
    current_artifact_kind = ""
    current_artifact_role = ""
    current_role_priority = -1
    explicit_targets = [str(item or "").strip() for item in route.entities.target_paths if str(item or "").strip()]
    is_explicit_target = any(_artifact_matches_path(path, candidate, candidate) for candidate in explicit_targets)
    other_markers: set[str] = set()

    if task_state is not None:
        for artifact in task_state.target_artifacts:
            artifact_path = str(artifact.path or "").strip()
            artifact_name = str(artifact.name or "").strip()
            if _artifact_matches_path(path, artifact_path, artifact_name):
                if artifact_path:
                    current_markers.update(_artifact_scope_markers(artifact_path))
                if artifact_name:
                    current_markers.update(_artifact_scope_markers(artifact_name))
                artifact_role = str(artifact.role or "").strip()
                artifact_priority = _artifact_role_priority(artifact_role)
                if artifact_priority >= current_role_priority:
                    current_role_priority = artifact_priority
                    current_artifact_kind = str(artifact.kind or "").strip()
                    current_artifact_role = artifact_role
            else:
                if artifact_path:
                    other_markers.update(_artifact_scope_markers(artifact_path))
                if artifact_name:
                    other_markers.update(_artifact_scope_markers(artifact_name))

    for candidate in explicit_targets:
        if _artifact_matches_path(path, candidate, candidate):
            current_markers.update(_artifact_scope_markers(candidate))
        else:
            other_markers.update(_artifact_scope_markers(candidate))

    if current_artifact_kind:
        current_markers.add(current_artifact_kind.lower())
    if current_artifact_role:
        current_markers.add(current_artifact_role.lower())

    constraints = _generation_relevant_constraints(route, session, limit=8)

    current_requirements: list[str] = []
    other_requirements: list[str] = []
    general_constraints: list[str] = []
    for constraint in constraints:
        current_score = _scope_relevance_score(constraint, current_markers)
        other_score = _scope_relevance_score(constraint, other_markers)
        if current_score > 0 and current_score >= other_score:
            current_requirements.append(constraint)
            continue
        if other_score > current_score:
            other_requirements.append(constraint)
            continue
        general_constraints.append(constraint)

    for requirement_sentence in _derived_requirement_sentences(route, session):
        current_score = _scope_relevance_score(requirement_sentence, current_markers)
        other_score = _scope_relevance_score(requirement_sentence, other_markers)
        if current_score > 0 and current_score >= other_score:
            for requirement in _split_requirement_clauses(requirement_sentence):
                if requirement not in current_requirements:
                    current_requirements.append(requirement)
            continue
        if other_score > current_score:
            for requirement in _split_requirement_clauses(requirement_sentence):
                if requirement not in other_requirements:
                    other_requirements.append(requirement)
            continue
        for requirement in _split_requirement_clauses(requirement_sentence):
            if requirement not in general_constraints:
                general_constraints.append(requirement)

    if _artifact_is_validation_target(path, current_artifact_kind, current_artifact_role):
        for requirement in _validation_target_behavior_requirements(route, session):
            if requirement and requirement not in current_requirements:
                current_requirements.append(requirement)

    literal_constraints = _target_literal_constraints(
        route,
        session,
        path=path,
        current_content=current_content,
        current_markers=current_markers,
        other_markers=other_markers,
    )
    for candidate in literal_constraints:
        if candidate not in current_requirements:
            current_requirements.append(candidate)

    if not current_requirements:
        if len(explicit_targets) <= 1 or current_artifact_role == "primary_target" or is_explicit_target:
            fallback = _trim_text(
                (
                    task_state.output_expectation
                    if task_state is not None and task_state.output_expectation
                    else route.requested_outcome
                ),
                220,
            )
            if fallback:
                current_requirements.append(fallback)

    return {
        "target_path": path,
        "artifact_kind": current_artifact_kind,
        "artifact_role": current_artifact_role,
        "literal_constraints": literal_constraints[:4],
        "current_write_requirements": current_requirements[:4],
        "other_pending_requirements": other_requirements[:4],
        "general_constraints": general_constraints[:4],
    }


def _explicit_generation_constraints(
    route: RouterOutput,
    session: SessionState | None,
) -> str:
    items = _generation_relevant_constraints(route, session, limit=6)
    request_text = _request_text_for_literals(route, session)
    for candidate in _request_literal_candidates(request_text):
        text = _trim_text(candidate, 220)
        if text and text not in items:
            items.append(text)
    return _format_list(items[:6])


def _request_text_for_literals(route: RouterOutput, session: SessionState | None) -> str:
    task_state = session.task_state if session is not None else None
    parts = [
        task_state.latest_user_turn if task_state is not None else "",
        route.user_goal,
        route.requested_outcome,
    ]
    return "\n".join(str(part or "").strip() for part in parts if str(part or "").strip())


def _request_literal_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    normalized = str(text or "")
    if not normalized:
        return candidates

    quoted_pattern = re.compile(r"(?P<quote>`|'|\")(?P<value>[^`\"'\n]{4,160}?)(?P=quote)")
    command_pattern = re.compile(
        r"(?:--[a-z0-9][\w-]*(?:\s+(?!und\b|and\b|then\b|dann\b|wieder\b)[^\s,;]+)?\s*){2,8}",
        re.IGNORECASE,
    )
    call_pattern = re.compile(r"[A-Za-z_][\w.]{1,80}\([^()\n]{1,180}\)")

    for match in quoted_pattern.finditer(normalized):
        candidate = " ".join(str(match.group("value") or "").split()).strip()
        if len(candidate) < 4 or candidate in candidates:
            continue
        candidates.append(candidate)

    for pattern in (command_pattern, call_pattern):
        for match in pattern.finditer(normalized):
            candidate = " ".join(str(match.group(0) or "").split()).strip()
            candidate = re.sub(r"\s+(und|and|then|dann|wieder)$", "", candidate, flags=re.IGNORECASE).strip()
            if len(candidate) < 8 or candidate in candidates:
                continue
            candidates.append(candidate)
    return candidates[:8]


def _target_literal_constraints(
    route: RouterOutput,
    session: SessionState | None,
    *,
    path: str,
    current_content: str | None = None,
    current_markers: set[str] | None = None,
    other_markers: set[str] | None = None,
) -> list[str]:
    reference = str(current_content or "").strip()
    if not reference and session is not None:
        reference = _last_read_excerpt(session, path)
    if not reference:
        return []

    current_tokens = _literal_reference_tokens(reference)
    if not current_tokens:
        return []

    literals: list[str] = []
    request_text = _request_text_for_literals(route, session)
    other_paths = _literal_scope_other_artifacts(route, session, path)
    for candidate in _request_literal_candidates(request_text):
        if not _literal_matches_reference(candidate, current_tokens):
            continue
        if _python_call_literal_needs_path_scope(candidate, path) and not _call_literal_scopes_to_current_artifact(
            candidate,
            request_text=request_text,
            current_path=path,
            other_paths=other_paths,
            reference=reference,
        ):
            continue
        scoped_candidate = _scoped_literal_constraint(candidate, path)
        if scoped_candidate and scoped_candidate not in literals:
            literals.append(scoped_candidate)
    return literals[:4]


def _scoped_literal_constraint(candidate: str, path: str) -> str | None:
    normalized = str(candidate or "").strip()
    if not normalized:
        return None
    normalized_path = str(path or "").strip().replace("\\", "/")
    suffix = Path(normalized_path).suffix.lower()
    if "(" in normalized and ")" in normalized:
        if suffix in {".md", ".markdown", ".rst", ".txt"} or _related_context_is_test_like(normalized_path):
            callee = str(normalized.split("(", 1)[0] or "").strip()
            if callee:
                return callee
    return normalized


def _python_call_literal_needs_path_scope(candidate: str, path: str) -> bool:
    normalized = str(candidate or "").strip()
    if "(" not in normalized or ")" not in normalized:
        return False
    normalized_path = str(path or "").strip().replace("\\", "/")
    suffix = Path(normalized_path).suffix.lower()
    if suffix not in {".py", ".pyi"}:
        return False
    return not _related_context_is_test_like(normalized_path)


def _call_literal_scopes_to_current_artifact(
    candidate: str,
    *,
    request_text: str,
    current_path: str,
    other_paths: list[str],
    reference: str,
) -> bool:
    if _python_reference_defines_literal_callee(reference, candidate):
        return True

    segments = _request_segments_for_literal_candidate(request_text, candidate)
    if not segments:
        return False

    for segment in segments:
        current_score = _artifact_specific_relevance_score(segment, current_path)
        other_score = max((_artifact_specific_relevance_score(segment, other) for other in other_paths), default=0)
        if current_score > 0 and current_score >= other_score:
            return True
    return False


def _literal_scope_other_artifacts(
    route: RouterOutput,
    session: SessionState | None,
    current_path: str,
) -> list[str]:
    seen: set[str] = set()
    others: list[str] = []

    def add(candidate: str) -> None:
        normalized = str(candidate or "").strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        others.append(normalized)

    explicit_targets = [str(item or "").strip() for item in route.entities.target_paths if str(item or "").strip()]
    for candidate in explicit_targets:
        if not _artifact_matches_path(current_path, candidate, candidate):
            add(candidate)

    task_state = session.task_state if session is not None else None
    if task_state is None:
        return others

    for artifact in task_state.target_artifacts:
        artifact_path = str(artifact.path or "").strip()
        artifact_name = str(artifact.name or "").strip()
        if _artifact_matches_path(current_path, artifact_path, artifact_name):
            continue
        if artifact_path:
            add(artifact_path)
        elif artifact_name:
            add(artifact_name)
    return others


def _request_segments_for_literal_candidate(text: str, candidate: str) -> list[str]:
    normalized = str(text or "").strip()
    literal = str(candidate or "").strip()
    if not normalized or not literal:
        return []
    callee = str(literal.split("(", 1)[0] or "").strip()
    callee_pattern = (
        re.compile(rf"(?<![A-Za-z0-9_]){re.escape(callee)}(?![A-Za-z0-9_])")
        if callee
        else None
    )
    segments: list[str] = []
    for sentence in _requirement_sentences(normalized):
        clauses = _split_requirement_clauses(sentence)
        for segment in clauses or [sentence]:
            segment_text = str(segment or "").strip()
            if not segment_text:
                continue
            if literal in segment_text or (callee_pattern is not None and callee_pattern.search(segment_text)):
                if segment_text not in segments:
                    segments.append(segment_text)
    return segments


def _python_reference_defines_literal_callee(reference: str, candidate: str) -> bool:
    normalized = str(candidate or "").strip()
    if "(" not in normalized or ")" not in normalized:
        return False
    callee = str(normalized.split("(", 1)[0] or "").strip().split(".")[-1]
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", callee):
        return False
    try:
        tree = ast.parse(str(reference or ""))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == callee:
            return True
    return False


def _literal_matches_reference(candidate: str, reference_tokens: set[str]) -> bool:
    normalized = str(candidate or "").strip()
    if not normalized:
        return False
    if "--" in normalized:
        candidate_tokens = _literal_reference_tokens(normalized)
        return len(candidate_tokens & reference_tokens) >= 3
    identifier_tokens = _literal_identifier_tokens(normalized)
    return bool(identifier_tokens & reference_tokens)


def _literal_reference_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in re.split(r"[^a-z0-9]+", str(text or "").lower()):
        token = raw.strip()
        if len(token) >= 4:
            tokens.add(token)
    return tokens


def _literal_identifier_tokens(text: str) -> set[str]:
    if "(" not in text:
        return set()
    callee = str(text.split("(", 1)[0] or "").strip().lower()
    return {token for token in re.split(r"[^a-z0-9]+", callee) if len(token) >= 4}


def _artifact_role_priority(role: str | None) -> int:
    normalized = str(role or "").strip().lower()
    priorities = {
        "primary_target": 5,
        "validation_target": 4,
        "active_context": 3,
        "primary_context": 3,
        "supporting_context": 2,
    }
    return priorities.get(normalized, 1 if normalized else 0)


def _derived_requirement_sentences(
    route: RouterOutput,
    session: SessionState | None,
) -> list[str]:
    task_state = session.task_state if session is not None else None
    sentences: list[str] = []
    sources: list[str] = []
    if task_state is not None and task_state.output_expectation:
        sources.append(task_state.output_expectation)
    if task_state is not None and task_state.latest_user_turn:
        sources.append(task_state.latest_user_turn)
    if route.requested_outcome:
        sources.append(route.requested_outcome)

    for source in sources:
        for sentence in _requirement_sentences(source):
            if sentence not in sentences:
                sentences.append(sentence)
    return sentences[:8]


def _requirement_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+|;", normalized)
    sentences: list[str] = []
    for part in parts:
        cleaned = str(part or "").strip(" .")
        if len(cleaned) >= 8 and cleaned not in sentences:
            sentences.append(cleaned)
    return sentences


def _split_requirement_clauses(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip(" .")
    if not normalized:
        return []
    if re.search(r"[\"'`<>]", normalized):
        return [normalized]
    fragments = re.split(r",\s+|\s+(?:and|und)\s+", normalized)
    clauses: list[str] = []
    for fragment in fragments:
        cleaned = str(fragment or "").strip(" .")
        cleaned = re.sub(r"^(?:and|und|then|dann)\s+", "", cleaned, flags=re.IGNORECASE).strip(" .")
        if len(cleaned) < 8 or cleaned in clauses:
            continue
        clauses.append(cleaned)
    return clauses or [normalized]


def _last_read_excerpt(session: SessionState, path: str) -> str:
    target = str(path or "").strip()
    for item in reversed(session.tool_calls):
        if item.tool_name != "read_file":
            continue
        candidate = str(item.tool_args.get("path") or "").strip()
        if candidate == target:
            return str(item.output_excerpt or "").strip()
    return ""


def _generation_relevant_constraints(
    route: RouterOutput,
    session: SessionState | None,
    *,
    limit: int,
) -> list[str]:
    items: list[str] = []
    task_state = session.task_state if session is not None else None
    if task_state is not None:
        sources = list(task_state.constraints[:limit])
    else:
        sources = []
    sources.extend(route.entities.constraints[:limit])
    for candidate in sources:
        text = _trim_text(candidate, 220)
        if not text or text in items or _is_generation_metadata_constraint(text):
            continue
        items.append(text)
        if len(items) >= limit:
            break
    return items


def _is_generation_metadata_constraint(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    prefixes = (
        "verification target:",
        "execution strategy:",
        "next best action:",
        "current user intent:",
        "goal relation:",
    )
    return normalized.startswith(prefixes)


def _artifact_is_validation_target(path: str, artifact_kind: str, artifact_role: str) -> bool:
    normalized_path = str(path or "").strip().lower()
    normalized_kind = str(artifact_kind or "").strip().lower()
    normalized_role = str(artifact_role or "").strip().lower()
    if normalized_role == "validation_target":
        return True
    if normalized_kind == "test":
        return True
    return Path(normalized_path).name.startswith("test_")


def _validation_target_behavior_requirements(
    route: RouterOutput,
    session: SessionState | None,
) -> list[str]:
    prioritized: list[str] = []
    fallback: list[str] = []
    for requirement_sentence in _derived_requirement_sentences(route, session):
        lowered = str(requirement_sentence or "").strip().lower()
        target = fallback
        if _request_literal_candidates(requirement_sentence) or any(
            token in lowered
            for token in (
                " should ",
                " must ",
                "print",
                "output",
                "return",
                "when run",
                "default to",
            )
        ):
            target = prioritized
        for requirement in _split_requirement_clauses(requirement_sentence):
            if requirement and requirement not in target:
                target.append(requirement)
    return prioritized or fallback


def _artifact_matches_path(path: str, artifact_path: str, artifact_name: str) -> bool:
    normalized_path = str(path or "").strip().lower()
    if not normalized_path:
        return False
    normalized_artifact_path = str(artifact_path or "").strip().lower()
    normalized_artifact_name = str(artifact_name or "").strip().lower()
    target_name = Path(normalized_path).name
    return bool(
        normalized_artifact_path == normalized_path
        or normalized_artifact_name == normalized_path
        or (normalized_artifact_path and Path(normalized_artifact_path).name == target_name)
        or (normalized_artifact_name and normalized_artifact_name == target_name)
    )


def _artifact_scope_markers(path_or_name: str) -> set[str]:
    normalized = str(path_or_name or "").strip().lower()
    if not normalized:
        return set()
    markers: set[str] = {normalized}
    path_obj = Path(normalized)
    if path_obj.name:
        markers.add(path_obj.name)
    if path_obj.stem:
        markers.add(path_obj.stem)
    for part in path_obj.parts:
        part = str(part or "").strip().lower()
        if part:
            markers.add(part)
    expanded = set(markers)
    for marker in markers:
        for token in re.split(r"[^a-z0-9]+", marker):
            token = token.strip().lower()
            if len(token) >= 3:
                expanded.add(token)
    return expanded


_GENERIC_ARTIFACT_MARKERS = {
    "app",
    "apps",
    "asset",
    "assets",
    "component",
    "components",
    "config",
    "configs",
    "css",
    "dist",
    "docs",
    "file",
    "files",
    "fixture",
    "fixtures",
    "html",
    "include",
    "includes",
    "js",
    "json",
    "layout",
    "layouts",
    "markdown",
    "md",
    "module",
    "modules",
    "page",
    "pages",
    "partial",
    "partials",
    "public",
    "py",
    "pyi",
    "readme",
    "rst",
    "sample",
    "samples",
    "script",
    "scripts",
    "section",
    "sections",
    "site",
    "src",
    "static",
    "style",
    "styles",
    "test",
    "tests",
    "text",
    "toml",
    "tsx",
    "ts",
    "txt",
    "view",
    "views",
    "yaml",
    "yml",
}


def _artifact_specific_markers(path_or_name: str) -> set[str]:
    normalized = str(path_or_name or "").strip().lower()
    if not normalized:
        return set()

    path_obj = Path(normalized)
    specific: set[str] = set()
    exact_candidates = [normalized, path_obj.name, path_obj.stem]
    for candidate in exact_candidates:
        cleaned = str(candidate or "").strip().lower()
        if cleaned and cleaned not in _GENERIC_ARTIFACT_MARKERS:
            specific.add(cleaned)

    token_sources = [*path_obj.parts, path_obj.name, path_obj.stem]
    for source in token_sources:
        for token in re.split(r"[^a-z0-9]+", str(source or "").strip().lower()):
            token = token.strip()
            if len(token) < 3 or token in _GENERIC_ARTIFACT_MARKERS:
                continue
            specific.add(token)
    return specific


def _artifact_specific_relevance_score(text: str, path_or_name: str) -> int:
    normalized_text = str(text or "").strip().lower()
    normalized_path = str(path_or_name or "").strip().lower()
    if not normalized_text or not normalized_path:
        return 0

    path_obj = Path(normalized_path)
    score = 0
    if normalized_path in normalized_text:
        score += 10

    name = path_obj.name.lower()
    stem = path_obj.stem.lower()
    if name and name in normalized_text:
        score += 7
    elif stem and stem in normalized_text:
        score += 4

    if name and re.search(rf"['\"]{re.escape(name)}['\"]", normalized_text):
        score += 4
    elif normalized_path and re.search(rf"['\"]{re.escape(normalized_path)}['\"]", normalized_text):
        score += 4

    for marker in _artifact_specific_markers(normalized_path):
        if marker in {normalized_path, name, stem}:
            continue
        if marker in normalized_text:
            score += 3 if len(marker) >= 5 else 2
    return score


def _scope_relevance_score(text: str, markers: set[str]) -> int:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return 0
    score = 0
    for marker in markers:
        candidate = str(marker or "").strip().lower()
        if len(candidate) < 3:
            continue
        if candidate in normalized:
            score += 3 if "/" in candidate or "." in candidate else 1
    return score


def _repair_rules(repair_strategy: str | None) -> str:
    lines = [
        "Repair rules:",
        "- Use the failed validation evidence as a hard constraint for this update.",
        "- Make a concrete mutation that addresses the failed verification scope directly.",
        "- Do not return equivalent content or formatting-only changes.",
        f"- If the current evidence is still insufficient to derive a concrete fix, return exactly {REPAIR_BLOCKED_SENTINEL}.",
    ]
    if repair_strategy == "validation_escalated":
        lines.append(
            "- A previous repair attempt produced no effective change. Tighten the repair and change the file materially."
        )
    return "\n".join(lines)


def _update_rules() -> str:
    return "\n".join(
        [
            "Update rules:",
            "- When the request explicitly changes the target file's behavior, implement that requested behavior change while preserving unrelated surrounding behavior.",
            "- Preserve unrelated existing behavior, imports, exports, config handling, startup code, interfaces, and tests unless the request or evidence requires changing them.",
            "- Prefer the smallest coherent diff that satisfies the requested outcome.",
            "- Do not simplify, rewrite, or reorganize the file unless that broader change is necessary for the request.",
            "- If you change names, config keys, selectors, commands, or interfaces, keep all dependent references aligned.",
            "- When the current file already contains working behavior not mentioned by the request, keep it intact.",
            "- Return a structurally complete file: do not leave truncated blocks, unterminated strings, invalid JSON/TOML, or unclosed markdown code fences.",
        ]
    )


def _inspected_context(session: SessionState) -> str:
    sections: list[str] = []
    for item in session.tool_calls:
        if item.tool_name != "read_file":
            continue
        path = str(item.tool_args.get("path", "")).strip()
        excerpt = (item.output_excerpt or "").strip()
        if not path or not excerpt:
            continue
        sections.append(f"{path}:\n{excerpt[:1200]}")
        if len(sections) >= 3:
            break
    if not sections and session.workspace_snapshot:
        sections.append(session.workspace_snapshot.repo_summary[:1000])
    return "\n\n".join(sections) or "none"


def _related_file_context(
    session: SessionState,
    target_path: str,
    *,
    excerpt_limit: int = 400,
    max_files: int = 2,
) -> str:
    candidate_paths: list[str] = []
    latest_excerpts: dict[str, str] = {}
    for item in session.tool_calls:
        if item.tool_name not in {"read_file", "write_file", "create_file", "replace_file", "patch_file"}:
            continue
        path = str(item.tool_args.get("path", "")).strip()
        if not path or path == target_path or path in candidate_paths:
            continue
        candidate_paths.append(path)

    candidate_paths = _prioritize_related_context_paths(candidate_paths, target_path=target_path)

    for path in candidate_paths:
        excerpt = _workspace_file_excerpt(session, path)
        if excerpt:
            latest_excerpts[path] = excerpt

    for item in reversed(session.tool_calls):
        if item.tool_name not in {"read_file", "write_file", "create_file", "replace_file", "patch_file"}:
            continue
        path = str(item.tool_args.get("path", "")).strip()
        if not path or path == target_path or path in latest_excerpts:
            continue
        excerpt = _tool_record_path_excerpt(item)
        if not excerpt:
            continue
        latest_excerpts[path] = excerpt

    sections: list[str] = []
    for path in candidate_paths:
        excerpt = str(latest_excerpts.get(path) or "").strip()
        if not excerpt:
            continue
        if _related_context_is_test_like(path):
            focused_line_hints = _related_context_task_line_hints(
                excerpt,
                session=session,
                target_path=target_path,
            )
            if focused_line_hints:
                focused_excerpt = _line_focused_excerpt(
                    excerpt,
                    line_hints=focused_line_hints,
                    limit=excerpt_limit,
                    before_radius=1,
                    after_radius=0,
                )
            else:
                focused_excerpt = _trim_balanced_text(excerpt, excerpt_limit)
        else:
            focused_excerpt = excerpt[:excerpt_limit]
        sections.append(f"{path}:\n{focused_excerpt}")
        if len(sections) >= max_files:
            break
    return "\n\n".join(sections) or "none"


def _prioritize_related_context_paths(
    candidate_paths: list[str],
    *,
    target_path: str,
) -> list[str]:
    target_suffix = Path(str(target_path or "").strip()).suffix.lower()
    runtime_suffixes = {".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt", ".gd"}
    config_suffixes = {".json", ".toml", ".yaml", ".yml", ".ini", ".cfg"}
    doc_suffixes = {".md", ".markdown", ".rst", ".txt"}
    target_is_runtime = target_suffix in runtime_suffixes
    if not target_is_runtime:
        return candidate_paths

    def is_test_like(path: str) -> bool:
        return _related_context_is_test_like(path)

    def priority(path: str) -> tuple[int, int]:
        lowered = path.lower()
        suffix = Path(path).suffix.lower()
        if is_test_like(path):
            return (0, len(lowered))
        if suffix in runtime_suffixes:
            return (1, len(lowered))
        if suffix in config_suffixes:
            return (2, len(lowered))
        if suffix in doc_suffixes:
            return (3, len(lowered))
        return (4, len(lowered))

    return [
        path
        for _, _, path in sorted(
            ((priority(path), index, path) for index, path in enumerate(candidate_paths)),
            key=lambda item: (item[0], item[1]),
        )
    ]


def _related_context_is_test_like(path: str) -> bool:
    lowered = str(path or "").lower()
    name = Path(str(path or "")).name.lower()
    return lowered.startswith("tests/") or "/tests/" in f"/{lowered}" or name.startswith("test_")


def _related_context_task_line_hints(
    text: str,
    *,
    session: SessionState,
    target_path: str,
    limit: int = 6,
) -> list[int]:
    file_lines = [str(line or "").strip() for line in str(text or "").splitlines()]
    if not file_lines:
        return []

    token_sources = [
        str(getattr(session, "task", "") or "").strip(),
        str(getattr(getattr(session, "task_state", None), "active_goal", "") or "").strip(),
        str(getattr(getattr(session, "task_state", None), "output_expectation", "") or "").strip(),
    ]
    raw_tokens: list[str] = []
    for source in token_sources:
        if not source:
            continue
        raw_tokens.extend(re.findall(r"--[A-Za-z0-9_-]+", source))
        raw_tokens.extend(re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{5,}\b", source))

    stopwords = {
        "update",
        "implementation",
        "documented",
        "behavior",
        "existing",
        "finish",
        "passes",
        "requested",
        "change",
        "changes",
        "relevant",
        "validation",
        "workspace",
        "artifact",
        "artifacts",
        "requested_outcome",
    }
    target_tokens = {
        token.lower()
        for token in (
            str(target_path or "").strip(),
            Path(str(target_path or "")).name,
            Path(str(target_path or "")).stem,
        )
        if str(token or "").strip()
    }
    query_tokens: list[str] = []
    for token in raw_tokens:
        lowered = str(token or "").strip().lower()
        if not lowered or lowered in stopwords or lowered in target_tokens:
            continue
        if lowered not in query_tokens:
            query_tokens.append(lowered)

    if not query_tokens:
        return []

    matches: list[int] = []
    for index, line in enumerate(file_lines, start=1):
        lowered_line = line.lower()
        if not lowered_line:
            continue
        if not any(token in lowered_line for token in query_tokens):
            continue
        previous_index = index - 1
        if previous_index >= 1 and file_lines[previous_index - 1]:
            if previous_index not in matches:
                matches.append(previous_index)
            if len(matches) >= limit:
                return matches
        if index not in matches:
            matches.append(index)
        if len(matches) >= limit:
            return matches
    return matches


def _repair_related_file_context(
    session: SessionState,
    *,
    target_path: str,
    repair_context: ValidationFailureEvidence,
    excerpt_limit: int = 300,
    max_files: int = 2,
) -> str:
    candidate_paths = _repair_support_paths(repair_context, target_path=target_path)
    if not candidate_paths:
        return _related_file_context(
            session,
            target_path,
            excerpt_limit=excerpt_limit,
            max_files=max_files,
        )

    target_focus_lines: list[str] = []
    allowed_related_paths: set[str] = set()
    if repair_context.verification_scope == "runtime":
        target_focus_lines = _targeted_runtime_failure_focus_lines(
            "\n".join(
                part
                for part in [
                    str(repair_context.excerpt or "").strip(),
                    str(repair_context.failure_summary or "").strip(),
                    str(repair_context.summary or "").strip(),
                ]
                if part
            ),
            target_path=target_path,
            other_paths=_repair_other_paths(repair_context, target_path=target_path),
            limit=8,
        )
        brief = getattr(repair_context, "repair_brief", None)
        if brief is not None:
            allowed_related_paths = {
                str(item or "").strip()
                for item in getattr(brief, "allowed_files", []) or []
                if str(item or "").strip() and str(item or "").strip() != target_path
            }
    target_focus_text = "\n".join(target_focus_lines)

    latest_excerpts: dict[str, str] = {}
    for path in candidate_paths:
        excerpt = _workspace_file_excerpt(session, path)
        if excerpt:
            latest_excerpts[path] = excerpt

    for item in reversed(session.tool_calls):
        if item.tool_name not in {"read_file", "write_file", "create_file", "replace_file", "patch_file"}:
            continue
        path = str(item.tool_args.get("path") or "").strip()
        if not path or path == target_path or path in latest_excerpts:
            continue
        if path not in candidate_paths:
            continue
        excerpt = _tool_record_path_excerpt(item)
        if not excerpt:
            continue
        latest_excerpts[path] = excerpt

    sections: list[str] = []
    for path in candidate_paths:
        excerpt = latest_excerpts.get(path)
        if not excerpt:
            continue
        if (
            repair_context.verification_scope == "runtime"
            and target_focus_lines
            and not _related_context_is_test_like(path)
            and path not in allowed_related_paths
            and _artifact_specific_relevance_score(target_focus_text, path) <= 0
        ):
            continue
        normalized_excerpt = str(excerpt or "").strip()
        focused_line_hints = _supporting_file_line_hints(
            excerpt,
            repair_context=repair_context,
            target_path=target_path,
        )
        if focused_line_hints:
            focused_excerpt = _line_focused_excerpt(
                excerpt,
                line_hints=focused_line_hints,
                limit=excerpt_limit,
                before_radius=1,
                after_radius=0,
            )
            if (
                _related_context_is_test_like(path)
                and _path_was_created(session, path)
                and len(str(excerpt or "").splitlines()) <= 10
            ):
                header_excerpt = _line_focused_excerpt(
                    excerpt,
                    line_hints=[1, 2, 3],
                    limit=min(180, excerpt_limit),
                    before_radius=0,
                    after_radius=0,
                )
                if header_excerpt and header_excerpt not in focused_excerpt:
                    focused_excerpt = _trim_balanced_text(
                        header_excerpt + "\n...\n" + focused_excerpt,
                        excerpt_limit,
                    )
        elif len(normalized_excerpt) <= excerpt_limit or (
            len(normalized_excerpt.splitlines()) <= 12 and len(normalized_excerpt) <= excerpt_limit + 120
        ):
            focused_excerpt = normalized_excerpt
        else:
            if focused_line_hints:
                focused_excerpt = _line_focused_excerpt(
                    excerpt,
                    line_hints=focused_line_hints,
                    limit=excerpt_limit,
                    before_radius=1,
                    after_radius=0,
                )
            else:
                focused_excerpt = _line_focused_excerpt(
                    excerpt,
                    line_hints=repair_context.line_hints,
                    limit=excerpt_limit,
                )
        sections.append(f"{path}:\n{focused_excerpt}")
        if len(sections) >= max_files:
            break
    if sections:
        return "\n\n".join(sections)
    return _related_file_context(
        session,
        target_path,
        excerpt_limit=excerpt_limit,
        max_files=max_files,
    )


def _diagnostic_context(session: SessionState) -> str:
    sections: list[str] = []
    for item in _recent_diagnostics(session):
        command = f" command={item.command}" if item.command else ""
        files = f" files={','.join(item.file_hints[:4])}" if item.file_hints else ""
        excerpt = _trim_text(item.excerpt or "", 360)
        sections.append(
            f"[{item.severity}] {item.category}:{command}{files} summary={_trim_text(item.summary, 180)} excerpt={excerpt}"
        )
        if len(sections) >= 4:
            break
    if not sections and session.follow_up_context and session.follow_up_context.last_error:
        sections.append(_trim_text(session.follow_up_context.last_error, 360))
    return "\n".join(section for section in sections if section) or "none"


def _read_paths(session: SessionState) -> list[str]:
    return [
        str(item.tool_args.get("path") or "").strip()
        for item in session.tool_calls
        if item.tool_name == "read_file" and str(item.tool_args.get("path") or "").strip()
    ]


def _trim_text(text: str, limit: int) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _trim_balanced_text(text: str, limit: int) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= limit:
        return normalized
    marker = "\n...\n"
    if limit <= len(marker) + 20:
        return _trim_text(normalized, limit)
    head = max((limit - len(marker)) // 3, 40)
    tail = max(limit - len(marker) - head, 40)
    return normalized[:head].rstrip() + marker + normalized[-tail:].lstrip()


def _line_focused_excerpt(
    text: str,
    *,
    line_hints: list[int],
    limit: int,
    radius: int = 1,
    before_radius: int | None = None,
    after_radius: int | None = None,
) -> str:
    normalized = str(text or "").rstrip()
    if not normalized:
        return ""
    lines = normalized.splitlines()
    if not lines:
        return ""
    leading_radius = max(0, radius if before_radius is None else before_radius)
    trailing_radius = max(0, radius if after_radius is None else after_radius)
    hinted_lines = sorted(
        {
            hint
            for raw in line_hints
            for hint in [int(raw)]
            if hint >= 1 and hint <= len(lines)
        }
    )
    if not hinted_lines:
        return _trim_balanced_text(normalized, limit)

    selected_indexes: set[int] = set()
    for hint in hinted_lines[:4]:
        start = max(hint - 1 - leading_radius, 0)
        end = min(hint + trailing_radius, len(lines))
        selected_indexes.update(range(start, end))

    ordered_indexes = sorted(selected_indexes)
    if not ordered_indexes:
        return _trim_balanced_text(normalized, limit)

    def render(line_width: int | None = None) -> str:
        excerpt_lines: list[str] = []
        previous_index: int | None = None
        for index in ordered_indexes:
            if previous_index is not None and index != previous_index + 1:
                excerpt_lines.append("...")
            content = lines[index]
            if line_width is not None:
                content = _trim_excerpt_line(content, line_width)
            excerpt_lines.append(f"{index + 1}: {content}")
            previous_index = index
        return "\n".join(excerpt_lines)

    excerpt = render()
    if len(excerpt) <= limit:
        return excerpt

    gap_count = 0
    previous_index: int | None = None
    for index in ordered_indexes:
        if previous_index is not None and index != previous_index + 1:
            gap_count += 1
        previous_index = index
    line_count = len(ordered_indexes)
    line_width = min(180, max(48, (limit - (gap_count * 4) - (line_count * 6)) // max(line_count, 1)))
    while line_width >= 32:
        excerpt = render(line_width)
        if len(excerpt) <= limit:
            return excerpt
        line_width -= 8
    return _trim_balanced_text(render(32), limit)


def _supporting_file_line_hints(
    text: str,
    *,
    repair_context: ValidationFailureEvidence,
    target_path: str,
    limit: int = 4,
    allow_target_token_fallback: bool = True,
) -> list[int]:
    focus_lines = _targeted_runtime_failure_focus_lines(
        "\n".join(
            part
            for part in [
                str(repair_context.excerpt or "").strip(),
                str(repair_context.failure_summary or "").strip(),
                str(repair_context.summary or "").strip(),
            ]
            if part
        ),
        target_path=target_path,
        other_paths=_repair_other_paths(repair_context, target_path=target_path),
        limit=8,
    )
    file_lines = [str(line or "").strip() for line in str(text or "").splitlines()]
    if not file_lines or not focus_lines:
        return []

    normalized_file_lines = [_normalized_focus_text(line) for line in file_lines]
    matches: list[int] = []
    for anchor in focus_lines:
        normalized_anchor = _normalized_focus_text(anchor)
        if not normalized_anchor or normalized_anchor.startswith("file ") or normalized_anchor.startswith("traceback"):
            continue
        for index, line in enumerate(normalized_file_lines, start=1):
            if not line:
                continue
            if normalized_anchor in line or line in normalized_anchor:
                previous_index = index - 1
                if previous_index >= 1 and file_lines[previous_index - 1].strip():
                    if previous_index not in matches:
                        matches.append(previous_index)
                    if len(matches) >= limit:
                        return matches
                if index not in matches:
                    matches.append(index)
                if len(matches) >= limit:
                    return matches
        query_tokens = _focus_line_query_tokens(anchor)
        if not query_tokens:
            continue
        scored_indexes: list[tuple[int, int]] = []
        for index, line in enumerate(normalized_file_lines, start=1):
            if not line:
                continue
            score = sum(1 for token in query_tokens if token in line)
            if score > 0:
                scored_indexes.append((score, index))
        for _, index in sorted(scored_indexes, key=lambda item: (-item[0], item[1]))[:2]:
            previous_index = index - 1
            if previous_index >= 1 and file_lines[previous_index - 1].strip():
                if previous_index not in matches:
                    matches.append(previous_index)
                if len(matches) >= limit:
                    return matches
            if index not in matches:
                matches.append(index)
            if len(matches) >= limit:
                return matches
    if matches:
        return matches

    if not allow_target_token_fallback:
        return matches

    fallback_tokens = _supporting_target_query_tokens(target_path, repair_context=repair_context)
    if not fallback_tokens:
        return matches

    scored_indexes: list[tuple[int, int]] = []
    for index, line in enumerate(file_lines, start=1):
        lowered = normalized_file_lines[index - 1]
        if not lowered:
            continue
        score = sum(3 for token in fallback_tokens if token in lowered)
        if "patch(" in lowered or ".read(" in lowered or "assert" in lowered:
            score += 2
        if re.search(r"[A-Za-z_][A-Za-z0-9_.]*\([^)]*\)", line):
            score += 1
        if score > 0:
            scored_indexes.append((score, index))
    for _, index in sorted(scored_indexes, key=lambda item: (-item[0], item[1]))[:2]:
        previous_index = index - 1
        if previous_index >= 1 and file_lines[previous_index - 1].strip():
            if previous_index not in matches:
                matches.append(previous_index)
            if len(matches) >= limit:
                return matches
        if index not in matches:
            matches.append(index)
        if len(matches) >= limit:
            return matches
    return matches


def _focus_line_query_tokens(text: str) -> list[str]:
    tokens: list[str] = []

    def add(candidate: str, *, min_length: int = 4) -> None:
        normalized = _normalized_focus_text(candidate)
        if len(normalized) < min_length:
            return
        if normalized in {
            "assertin",
            "assertequal",
            "traceback",
            "file",
            "read",
            "html",
            "css",
            "line",
            "error",
            "assertionerror",
        }:
            return
        if normalized not in tokens:
            tokens.append(normalized)

    raw = str(text or "").strip()
    for candidate in re.findall(r"['\"]([^'\"]{3,})['\"]", raw):
        add(candidate, min_length=3)
    for candidate in re.findall(r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.[A-Za-z0-9_-]+", raw):
        add(candidate, min_length=3)
    for candidate in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{4,}\b", raw):
        add(candidate, min_length=5)
    return tokens[:6]


def _supporting_target_query_tokens(
    target_path: str,
    *,
    repair_context: ValidationFailureEvidence,
) -> list[str]:
    tokens: list[str] = []

    def add(candidate: str, *, min_length: int = 3) -> None:
        normalized = _normalized_focus_text(candidate)
        if len(normalized) < min_length:
            return
        if normalized not in tokens:
            tokens.append(normalized)

    for marker in _artifact_specific_markers(target_path):
        add(marker, min_length=3)

    brief = getattr(repair_context, "repair_brief", None)
    if brief is not None:
        for symbol in getattr(brief, "implicated_symbols", []) or []:
            add(str(symbol or ""), min_length=4)

    return tokens[:6]


def _workspace_file_excerpt(session: SessionState, path: str) -> str:
    workspace_root = Path(str(getattr(session, "workspace_root", "") or "").strip())
    relative_path = str(path or "").strip()
    if not relative_path or not str(workspace_root):
        return ""

    try:
        root = workspace_root.resolve()
        absolute = (workspace_root / relative_path).resolve()
        absolute.relative_to(root)
    except (OSError, ValueError):
        return ""

    if not absolute.exists() or not absolute.is_file():
        return ""

    try:
        if absolute.stat().st_size > 200_000:
            return ""
        content = absolute.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""

    if "\x00" in content:
        return ""
    return content


def _path_was_created(session: SessionState, path: str) -> bool:
    target = str(path or "").strip()
    if not target:
        return False
    for item in reversed(session.tool_calls):
        if item.tool_name != "create_file" or not item.success:
            continue
        if str(item.tool_args.get("path") or "").strip() == target:
            return True
    return False


def _normalized_focus_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _trim_excerpt_line(text: str, limit: int) -> str:
    content = str(text or "").rstrip()
    if len(content) <= limit:
        return content
    return content[: limit - 1].rstrip() + "…"


def _tool_record_path_excerpt(item) -> str:
    if item.tool_name in {"write_file", "create_file", "replace_file", "patch_file"}:
        content = str(item.tool_args.get("content") or "").strip()
        if content:
            return content
    excerpt = (item.output_excerpt or "").strip()
    if excerpt:
        return excerpt
    return str(item.tool_args.get("content") or "").strip()


def _runtime_failure_focus_lines(text: str, *, limit: int = 6) -> list[str]:
    lines = [str(line or "").rstrip() for line in str(text or "").splitlines()]
    if not lines:
        return []

    focus: list[str] = []

    def add(candidate: str) -> None:
        normalized = str(candidate or "").strip()
        if not normalized:
            return
        clipped = _trim_text(normalized, 180)
        if clipped not in focus:
            focus.append(clipped)

    for index, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("Traceback"):
            add(stripped)
            continue
        if re.match(r'File ".+", line \d+, in .+', stripped) or re.match(
            r"[\w./\\-]+\.py:\d+(?::\d+)?",
            stripped,
        ):
            add(stripped)
            if index + 1 < len(lines):
                next_line = lines[index + 1].strip()
                if (
                    next_line
                    and not next_line.startswith("File ")
                    and not next_line.startswith("Traceback")
                ):
                    add(next_line)
            continue
        if re.search(
            r"\b(?:AssertionError|TypeError|ValueError|RuntimeError|NameError|AttributeError|ImportError|ModuleNotFoundError|SyntaxError|IndexError|KeyError|FAIL|FAILED|ERROR)\b",
            stripped,
        ):
            add(stripped)
            for diff_line in _adjacent_assertion_diff_lines(lines, start_index=index):
                add(diff_line)
            continue
        if re.search(r"[A-Za-z_][A-Za-z0-9_.]*\([^)]*\)", stripped) and (
            "assert" in stripped.lower()
            or "[" in stripped
            or "." in stripped
        ):
            add(stripped)
        if len(focus) >= limit:
            break

    if focus:
        return focus[:limit]

    return [_trim_text(line.strip(), 180) for line in lines if line.strip()][:limit]


def _runtime_failure_blocks(lines: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            blocks.append(current)
            current = []

    for raw in lines:
        stripped = str(raw or "").rstrip()
        compact = stripped.strip()
        if not compact:
            flush()
            continue
        if re.fullmatch(r"[=\-_/]{5,}", compact):
            flush()
            continue
        if current and re.match(r"\b(?:FAIL|FAILED|ERROR):", compact):
            flush()
        current.append(stripped)

    flush()
    return blocks


def _targeted_runtime_failure_blocks(
    text: str,
    *,
    target_path: str,
    other_paths: list[str] | None = None,
) -> list[list[str]]:
    lines = [str(line or "").rstrip() for line in str(text or "").splitlines()]
    blocks = _runtime_failure_blocks(lines)
    if len(blocks) <= 1:
        return blocks

    target_specific: list[list[str]] = []
    neutral: list[list[str]] = []
    other_candidates = [str(item or "").strip() for item in (other_paths or []) if str(item or "").strip()]
    for block in blocks:
        block_text = "\n".join(block)
        target_score = _artifact_specific_relevance_score(block_text, target_path)
        other_score = max(
            (_artifact_specific_relevance_score(block_text, candidate) for candidate in other_candidates),
            default=0,
        )
        if target_score > 0 and target_score >= other_score:
            target_specific.append(block)
            continue
        if target_score == 0 and other_score == 0:
            neutral.append(block)

    if target_specific:
        return target_specific + neutral[:1]
    return blocks


def _targeted_runtime_failure_focus_lines(
    text: str,
    *,
    target_path: str,
    other_paths: list[str] | None = None,
    limit: int = 6,
) -> list[str]:
    original_lines = [str(line or "").rstrip() for line in str(text or "").splitlines()]
    original_blocks = _runtime_failure_blocks(original_lines)
    scoped_blocks = _targeted_runtime_failure_blocks(
        text,
        target_path=target_path,
        other_paths=other_paths,
    )
    block_scoped = bool(original_blocks) and scoped_blocks != original_blocks
    scoped_text = "\n".join(line for block in scoped_blocks for line in block)
    lines = [str(line or "").rstrip() for line in str(scoped_text or "").splitlines()]
    if not lines:
        return []

    target_markers = _artifact_scope_markers(target_path)
    target_tokens = {
        str(target_path or "").strip().lower(),
        Path(str(target_path or "").strip()).name.lower(),
        Path(str(target_path or "").strip()).stem.lower(),
    }
    other_markers: set[str] = set()
    other_tokens: set[str] = set()
    for candidate in other_paths or []:
        path = str(candidate or "").strip()
        if not path:
            continue
        other_markers.update(_artifact_scope_markers(path))
        other_tokens.update(
            {
                path.lower(),
                Path(path).name.lower(),
                Path(path).stem.lower(),
            }
        )
    has_target_frame = any(
        (
            re.match(r'File ".+", line \d+, in .+', stripped)
            or re.match(r"[\w./\\-]+\.py:\d+(?::\d+)?", stripped)
        )
        and any(marker and marker in stripped.lower() for marker in target_markers)
        for stripped in (line.strip() for line in lines)
        if stripped
    )
    scored: list[tuple[int, int, int, int, str]] = []
    seen: set[str] = set()

    def add(candidate: str, *, index: int, score: int, target_score: int, other_score: int) -> None:
        normalized = str(candidate or "").strip()
        if not normalized:
            return
        clipped = _trim_text(normalized, 180)
        if clipped in seen:
            return
        seen.add(clipped)
        scored.append((score, target_score, other_score, index, clipped))

    for index, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        score = 0
        target_score = _scope_relevance_score(stripped, target_markers)
        other_score = _scope_relevance_score(stripped, other_markers)
        if any(token and token in lowered for token in target_tokens):
            target_score += 4
        if any(token and token in lowered for token in other_tokens):
            other_score += 4
        if stripped.startswith("Traceback"):
            score += 2
        if re.search(
            r"\b(?:AssertionError|TypeError|ValueError|RuntimeError|NameError|AttributeError|ImportError|ModuleNotFoundError|SyntaxError|IndexError|KeyError|FAIL|FAILED|ERROR|SystemExit)\b",
            stripped,
        ):
            score += 6
        if re.match(r'File ".+", line \d+, in .+', stripped) or re.match(
            r"[\w./\\-]+\.py:\d+(?::\d+)?",
            stripped,
        ):
            score += 3
            target_frame = any(marker and marker in lowered for marker in target_markers)
            other_frame = any(marker and marker in lowered for marker in other_markers)
            if target_frame:
                score += 5
                target_score += 5
            if other_frame:
                other_score += 5
            add(stripped, index=index, score=score, target_score=target_score, other_score=other_score)
            if index + 1 < len(lines):
                next_line = lines[index + 1].strip()
                if next_line and not next_line.startswith("Traceback") and not next_line.startswith("File "):
                    next_score = 4
                    next_target_score = target_score
                    next_other_score = other_score
                    if re.search(r"[A-Za-z_][A-Za-z0-9_.]*\([^)]*\)", next_line):
                        next_score += 2
                    if target_frame:
                        next_score += 5
                        next_target_score += 3
                    elif has_target_frame and re.search(r"[A-Za-z_][A-Za-z0-9_.]*\([^)]*\)", next_line):
                        next_score += 3
                        next_target_score = max(next_target_score, next_other_score + 1, 2)
                    elif re.search(r"[A-Za-z_][A-Za-z0-9_.]*\([^)]*\)", next_line):
                        next_score += 2
                    add(
                        next_line,
                        index=index + 1,
                        score=next_score,
                        target_score=next_target_score,
                        other_score=next_other_score,
                    )
            continue
        if target_score > 0:
            score += min(target_score, 6)
        if re.search(r"[A-Za-z_][A-Za-z0-9_.]*\([^)]*\)", stripped):
            score += 3
        if "test_" in lowered or "mock" in lowered:
            score += 1
        if score > 0:
            add(stripped, index=index, score=score, target_score=target_score, other_score=other_score)
            if "assertionerror" in lowered:
                for offset, diff_line in enumerate(
                    _adjacent_assertion_diff_lines(lines, start_index=index),
                    start=1,
                ):
                    add(
                        diff_line,
                        index=index + offset,
                        score=max(score - 1, 1),
                        target_score=target_score,
                        other_score=other_score,
                    )

    if not scored:
        return _runtime_failure_focus_lines(text, limit=limit)

    if not block_scoped and any(item[1] > 0 for item in scored):
        targeted = [item for item in scored if item[1] > 0 and item[1] >= item[2]]
        if targeted:
            scored = targeted
        else:
            filtered = [item for item in scored if item[2] <= item[1]]
            if filtered:
                scored = filtered

    def kind_priority(line: str) -> int:
        lowered = str(line or "").lower()
        if "assertionerror" in lowered:
            return 0
        if str(line or "").startswith(("-", "+", "?")):
            return 1
        if re.search(
            r"\b(?:typeerror|valueerror|runtimeerror|nameerror|attributeerror|importerror|modulenotfounderror|syntaxerror|indexerror|keyerror|systemexit)\b",
            lowered,
        ):
            return 2
        if any(marker and marker in lowered for marker in target_markers):
            return 3
        if "assert" in lowered:
            return 5
        if lowered.startswith("file ") or lowered.startswith("traceback"):
            return 6
        return 4

    ranked = sorted(scored, key=lambda item: (kind_priority(item[4]), -item[0], -item[1], item[3]))
    selected = [line for _, _, _, _, line in ranked[:limit]]
    assertion_lines = [
        _trim_text(str(line or "").strip(), 180)
        for line in lines
        if str(line or "").strip()
        and (
            "assertionerror" in str(line or "").lower()
            or str(line or "").strip().startswith(("-", "+", "?"))
        )
    ]
    if assertion_lines and not any(
        "assertionerror" in str(line or "").lower() or str(line or "").startswith(("-", "+", "?"))
        for line in selected
    ):
        selected = assertion_lines[: min(len(assertion_lines), limit)] + [
            line for line in selected if line not in assertion_lines
        ]
    if not any(
        ("assert" in str(line or "").lower()) or re.search(r"[A-Za-z_][A-Za-z0-9_.]*\([^)]*\)", str(line or ""))
        for line in selected
        if not str(line or "").startswith(("File ", "Traceback"))
    ):
        for _, target_score, other_score, _, line in ranked:
            lowered = str(line or "").lower()
            if str(line or "").startswith(("File ", "Traceback")):
                continue
            if not (
                "assert" in lowered
                or re.search(r"[A-Za-z_][A-Za-z0-9_.]*\([^)]*\)", str(line or ""))
            ):
                continue
            if not block_scoped and target_score <= 0 and other_score > target_score:
                continue
            if line not in selected:
                if len(selected) >= limit:
                    selected[-1] = line
                else:
                    selected.append(line)
                break
    return selected[:limit]


def _repair_other_paths(
    repair_context: ValidationFailureEvidence,
    *,
    target_path: str,
) -> list[str]:
    target = str(target_path or "").strip()
    return [
        path
        for path in [str(item or "").strip() for item in [*repair_context.artifact_paths, *repair_context.file_hints]]
        if path and not _artifact_matches_path(target, path, path)
    ]


def _adjacent_assertion_diff_lines(
    lines: list[str],
    *,
    start_index: int,
    limit: int = 4,
) -> list[str]:
    collected: list[str] = []
    for raw in lines[start_index + 1 :]:
        stripped = str(raw or "").strip()
        if not stripped:
            break
        if stripped.startswith(("Traceback", "File ")):
            break
        if re.match(r"[=-]{5,}", stripped):
            break
        if re.search(r"\b(?:FAIL|FAILED|ERROR)\b", stripped):
            break
        if stripped.startswith(("-", "+", "?")):
            collected.append(stripped)
            if len(collected) >= limit:
                break
            continue
        if collected:
            break
        break
    return collected


def _repair_support_paths(
    repair_context: ValidationFailureEvidence,
    *,
    target_path: str,
) -> list[str]:
    target = str(target_path or "").strip()
    candidates = [
        str(item or "").strip()
        for item in [*repair_context.file_hints, *repair_context.artifact_paths]
        if str(item or "").strip() and str(item or "").strip() != target
    ]

    def rank(path: str) -> tuple[int, str]:
        lowered = path.lower()
        suffix = Path(path).suffix.lower()
        name = Path(path).name.lower()
        is_test = lowered.startswith("tests/") or "/tests/" in f"/{lowered}" or name.startswith("test_")
        if is_test:
            return (0, lowered)
        if suffix in {".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt", ".gd"}:
            return (1, lowered)
        if suffix in {".json", ".toml", ".yaml", ".yml", ".ini", ".cfg"}:
            return (2, lowered)
        if suffix in {".md", ".rst", ".txt"}:
            return (3, lowered)
        return (4, lowered)

    return [path for path in sorted(dict.fromkeys(candidates), key=rank)]


def _format_list(values: object) -> str:
    if not values:
        return "none"
    if isinstance(values, list):
        return ", ".join(str(value) for value in values if value) or "none"
    return str(values)


def _format_objects(values: list[object], formatter=None) -> str:
    if not values:
        return "none"
    if formatter is None:
        formatter = lambda item: item
    return json.dumps([formatter(item) for item in values], ensure_ascii=False)
