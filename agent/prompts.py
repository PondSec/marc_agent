from __future__ import annotations

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
        sections = [
            "Produce the full file content for exactly one file.",
            f"Latest user request: {_trim_text(session.task, 420)}",
            f"User goal: {_trim_text(route.user_goal, 240)}",
            f"Requested outcome: {_trim_text(route.requested_outcome, 240)}",
            f"Explicit constraints: {_explicit_generation_constraints(route, session)}",
            f"Task focus: {json.dumps(_compact_generation_focus(route, session, path), ensure_ascii=False)}",
            f"File-scoped focus: {json.dumps(_artifact_scoped_focus(route, session, path, current_content=current_content), ensure_ascii=False)}",
            f"Related file hints: {_related_file_context(session, path)}",
        ]
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
                        "Use that feedback to make a smaller, safer update while preserving unrelated existing behavior.",
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
    ]
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
        sections = [
            "Produce the full file content for exactly one file.",
            f"Latest user request: {_trim_text(session.task if session is not None else route.requested_outcome, 420)}",
            f"User goal: {_trim_text(route.user_goal, 240)}",
            f"Requested outcome: {_trim_text(route.requested_outcome, 240)}",
            f"Explicit constraints: {_explicit_generation_constraints(route, session)}",
            f"Task focus: {json.dumps(task_focus, ensure_ascii=False)}",
            f"File-scoped focus: {json.dumps(_artifact_scoped_focus(route, session, path, current_content=current_content), ensure_ascii=False)}",
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
        if current_content is not None:
            sections.append(_update_rules())
            if review_feedback is not None:
                sections.extend(
                    [
                        f"Self-review feedback on the previous proposal: {json.dumps(_compact_proposed_update_review(review_feedback), ensure_ascii=False)}",
                        "Address that feedback directly with a smaller, safer update that preserves unrelated existing behavior.",
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
                    "Address the review feedback directly and keep the update tightly scoped.",
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
    return {
        "command": _trim_text(context.command, 180),
        "verification_scope": context.verification_scope,
        "status": context.status,
        "artifact_paths": context.artifact_paths[:6],
        "summary": _trim_text(context.summary, 180),
        "failure_summary": _trim_text(context.failure_summary, 220),
        "excerpt": _trim_text(context.excerpt or "", 320),
        "expected_features": context.expected_features[:8],
        "missing_features": context.missing_features[:8],
        "file_hints": context.file_hints[:6],
        "line_hints": context.line_hints[:8],
        "action_hints": [_trim_text(item, 160) for item in context.action_hints[:4]],
        "repair_requirements": [_trim_text(item, 200) for item in context.repair_requirements[:6]],
        "evidence_signature": context.evidence_signature,
    }


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

    constraints: list[str] = []
    if task_state is not None:
        for candidate in task_state.constraints[:8]:
            text = _trim_text(candidate, 220)
            if text and text not in constraints:
                constraints.append(text)
    for candidate in route.entities.constraints[:8]:
        text = _trim_text(candidate, 220)
        if text and text not in constraints:
            constraints.append(text)

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

    literal_constraints = _target_literal_constraints(
        route,
        session,
        path=path,
        current_content=current_content,
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
    items: list[str] = []
    task_state = session.task_state if session is not None else None
    if task_state is not None:
        for candidate in task_state.constraints[:6]:
            text = _trim_text(candidate, 220)
            if text and text not in items:
                items.append(text)
    for candidate in route.entities.constraints[:6]:
        text = _trim_text(candidate, 220)
        if text and text not in items:
            items.append(text)
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

    command_pattern = re.compile(
        r"(?:--[a-z0-9][\w-]*(?:\s+(?!und\b|and\b|then\b|dann\b|wieder\b)[^\s,;]+)?\s*){2,8}",
        re.IGNORECASE,
    )
    call_pattern = re.compile(r"[A-Za-z_][\w.]{1,80}\([^()\n]{1,180}\)")

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
    for candidate in _request_literal_candidates(request_text):
        if _literal_matches_reference(candidate, current_tokens):
            literals.append(candidate)
    return literals[:4]


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


def _related_file_context(session: SessionState, target_path: str) -> str:
    sections: list[str] = []
    for item in session.tool_calls:
        if item.tool_name != "read_file":
            continue
        path = str(item.tool_args.get("path", "")).strip()
        if not path or path == target_path:
            continue
        excerpt = (item.output_excerpt or "").strip()
        if not excerpt:
            continue
        sections.append(f"{path}:\n{excerpt[:400]}")
        if len(sections) >= 2:
            break
    return "\n\n".join(sections) or "none"


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
