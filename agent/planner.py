from __future__ import annotations

from pathlib import Path
import re

from agent.models import SessionState, WorkspaceSnapshot
from agent.prompts import (
    decision_prompt,
    planning_prompt_with_analysis,
    system_prompt,
    task_analysis_prompt,
)
from llm.ollama_client import OllamaClient
from llm.schemas import AgentActionType, AgentDecision, PlanningResponse, TaskAnalysis, TaskIntent


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "ein",
    "eine",
    "einen",
    "und",
    "oder",
    "fuer",
    "mit",
    "der",
    "die",
    "das",
    "ist",
    "mein",
    "dein",
    "repo",
    "projekt",
    "agent",
    "coding",
    "hallo",
    "hello",
    "hi",
    "hey",
    "moin",
    "servus",
    "marc",
    "bitte",
    "kann",
    "kannst",
    "koenntest",
    "du",
    "mir",
    "mal",
    "doch",
    "einfach",
    "wo",
    "ich",
    "gegen",
    "fuer",
    "für",
}

LOW_SIGNAL_TERMS = {
    "code",
    "python",
    "javascript",
    "typescript",
    "computer",
    "ki",
    "ai",
    "spiel",
    "spiele",
    "programm",
    "projekt",
    "workspace",
}

SIGNAL_PHRASES = (
    ("tic tac toe", {"tic", "tac", "toe"}),
    ("tick tack toe", {"tick", "tack", "toe"}),
    ("tick-tack-toe", {"tick", "tack", "toe"}),
    ("tic-tac-toe", {"tic", "tac", "toe"}),
    ("todo app", {"todo", "app"}),
    ("to do app", {"to", "do", "app"}),
    ("rest api", {"rest", "api"}),
    ("tic tac", {"tic", "tac"}),
    ("tick tack", {"tick", "tack"}),
)

WRITE_TOOLS = {
    "write_file",
    "append_file",
    "create_file",
    "replace_in_file",
    "patch_file",
}

CREATE_TOKENS = {
    "schreib",
    "schreibe",
    "schreiben",
    "schreibst",
    "write",
    "create",
    "erstelle",
    "erstellen",
    "baue",
    "build",
    "implement",
    "implementiere",
    "generate",
    "erzeuge",
    "script",
    "skript",
    "component",
    "komponente",
    "calculator",
    "taschenrechner",
}

FIX_TOKENS = {
    "fix",
    "bug",
    "behebe",
    "reparier",
    "repair",
    "defekt",
    "kaputt",
    "error",
    "fehler",
    "broken",
}

MODIFY_TOKENS = {
    "aendere",
    "ändere",
    "update",
    "aktualisiere",
    "refactor",
    "umbau",
    "verbessere",
    "erweitere",
    "anpassen",
}

INSPECT_TOKENS = {
    "finde",
    "suche",
    "inspect",
    "pruef",
    "prüf",
    "show",
    "zeige",
    "read",
    "lies",
}

LANGUAGE_EXTENSIONS = {
    "python": ".py",
    "py": ".py",
    "javascript": ".js",
    "js": ".js",
    "typescript": ".ts",
    "ts": ".ts",
    "tsx": ".tsx",
    "jsx": ".jsx",
    "react": ".tsx",
    "html": ".html",
    "css": ".css",
    "json": ".json",
    "markdown": ".md",
    "md": ".md",
    "shell": ".sh",
    "bash": ".sh",
}


class Planner:
    def __init__(self, llm: OllamaClient, tool_manifest: str):
        self.llm = llm
        self.tool_manifest = tool_manifest

    def analyze_task(
        self,
        task: str,
        snapshot: WorkspaceSnapshot | None,
    ) -> TaskAnalysis:
        heuristic = self._fallback_task_analysis(task, snapshot)
        if heuristic.intent in {TaskIntent.REPLY, TaskIntent.CREATE, TaskIntent.FIX}:
            return heuristic
        try:
            data = self.llm.generate_json(
                task_analysis_prompt(task, snapshot),
                system=system_prompt(),
                timeout=self._llm_timeout(8),
                num_ctx=self._llm_num_ctx(4096),
            )
            analysis = TaskAnalysis.model_validate(data)
            return self._normalize_task_analysis(task, snapshot, analysis)
        except Exception:
            return heuristic

    def create_plan(
        self,
        task: str,
        snapshot: WorkspaceSnapshot,
        analysis: TaskAnalysis | None = None,
    ) -> PlanningResponse:
        return self._fallback_plan(task, snapshot, analysis)

    def decide_next_action(self, task: str, session: SessionState) -> AgentDecision:
        analysis = session.task_analysis or self.analyze_task(task, session.workspace_snapshot)
        if analysis.direct_response:
            return AgentDecision(
                thought_summary="The prompt is a direct conversational question and does not need repo work.",
                action_type=AgentActionType.FINAL,
                tool_name=None,
                tool_args={},
                expected_outcome="Reply directly in natural language.",
                final_response=analysis.direct_response,
            )
        deterministic = self._deterministic_next_decision(session, analysis)
        if deterministic is not None:
            return deterministic
        create_draft = self._draft_create_decision(session, analysis)
        if create_draft is not None:
            return create_draft
        try:
            data = self.llm.generate_json(
                decision_prompt(task, session, self.tool_manifest),
                system=system_prompt(),
                timeout=self._llm_timeout(8),
                num_ctx=self._llm_num_ctx(4096),
            )
            decision = AgentDecision.model_validate(data)
            return self._normalize_decision(session, analysis, decision)
        except Exception:
            return self._fallback_decision(session, analysis)

    def _fallback_plan(
        self,
        task: str,
        snapshot: WorkspaceSnapshot,
        analysis: TaskAnalysis | None = None,
    ) -> PlanningResponse:
        steps = [
            "Discover the repo shape, manifests, tests, scripts, and architecture boundaries.",
            "Plan the smallest file set that should be read or searched before editing.",
            "Act on the existing implementation points instead of creating parallel architecture.",
            "Verify with the project-aware validation plan, not just a single command.",
            "Repair from concrete diagnostics until validation passes or a blocker is clear.",
            "Report changed files, commands, diagnostics, diffs, and stop reason.",
        ]
        if analysis and analysis.intent == TaskIntent.CREATE:
            steps = [
                "Inspect the smallest manifest, README, or nearby example needed to match project conventions.",
                "Create or update only the files required for the requested deliverable.",
                "Run the most relevant validation or smoke test for the new code.",
                "Report changed files, commands, diagnostics, diffs, and remaining risks.",
            ]
        elif self._looks_like_analysis_task(task):
            steps = [
                "Discover the repository layout, entrypoints, and major subsystems.",
                "Read the highest-signal files for the requested analysis.",
                "Report strengths, weaknesses, and the highest-priority gaps.",
            ]
        if self._might_need_helper(task):
            steps.insert(
                min(4, len(steps)),
                "If direct inspection is insufficient, create a small helper script or parser to unblock the task.",
            )
        completion = [
            "Relevant files were inspected before editing.",
            "Any code changes were validated with the project-aware command plan or the blocker is explicit.",
            "Final output includes changed files, commands, diagnostics, and remaining risks.",
        ]
        return PlanningResponse(
            summary=(
                "Work from repository discovery into focused edits, multi-step verification, repair, and reporting."
            ),
            steps=steps,
            files_to_inspect=snapshot.focus_files[:6] or snapshot.important_files[:10],
            tests_to_run=[item.command for item in snapshot.validation_commands[:4]]
            or snapshot.likely_commands[:4],
            completion_criteria=completion,
        )

    def _fallback_decision(
        self,
        session: SessionState,
        analysis: TaskAnalysis | None = None,
    ) -> AgentDecision:
        analysis = analysis or session.task_analysis or self._fallback_task_analysis(
            session.task,
            session.workspace_snapshot,
        )
        tool_names = [item.tool_name for item in session.tool_calls]
        read_paths = {
            item.tool_args.get("path")
            for item in session.tool_calls
            if item.tool_name == "read_file"
        }
        searched_queries = {
            str(item.tool_args.get("query", "")).lower()
            for item in session.tool_calls
            if item.tool_name == "search_in_files"
        }
        snapshot = session.workspace_snapshot
        analysis_task = analysis.intent == TaskIntent.ANALYZE or self._looks_like_analysis_task(
            session.task
        )
        focus_terms = analysis.search_terms or self._focus_terms(session.task)
        diagnostic_files = [
            path for item in session.diagnostics[-6:] for path in item.file_hints
        ]

        if analysis.direct_response:
            return AgentDecision(
                thought_summary="A short conversational prompt does not need repository inspection.",
                action_type=AgentActionType.FINAL,
                tool_name=None,
                tool_args={},
                expected_outcome="Reply naturally and invite a concrete task.",
                final_response=analysis.direct_response,
            )

        if not tool_names:
            bootstrap_file = self._pick_bootstrap_file(session, analysis, read_paths)
            if bootstrap_file:
                return AgentDecision(
                    thought_summary="Read the closest manifest or example before creating new code.",
                    action_type=AgentActionType.CALL_TOOL,
                    tool_name="read_file",
                    tool_args={"path": bootstrap_file},
                    expected_outcome="Match existing project conventions before editing.",
                    final_response=None,
                )
            return AgentDecision(
                thought_summary="Need a repository map and validation plan before selecting files.",
                action_type=AgentActionType.CALL_TOOL,
                tool_name="inspect_workspace",
                tool_args={"focus": session.task},
                expected_outcome="Identify relevant files, workflows, and validation commands.",
                final_response=None,
            )

        if Path(session.workspace_root, ".git").exists() and "git_status" not in tool_names:
            return AgentDecision(
                thought_summary="Checking git status avoids blind edits on a dirty worktree.",
                action_type=AgentActionType.CALL_TOOL,
                tool_name="git_status",
                tool_args={},
                expected_outcome="See current repository status before further work.",
                final_response=None,
            )

        if analysis.intent == TaskIntent.CREATE:
            bootstrap_file = self._pick_bootstrap_file(session, analysis, read_paths)
            if bootstrap_file:
                return AgentDecision(
                    thought_summary="Read one more high-signal file before creating new code.",
                    action_type=AgentActionType.CALL_TOOL,
                    tool_name="read_file",
                    tool_args={"path": bootstrap_file},
                    expected_outcome="Clarify local conventions for the new deliverable.",
                    final_response=None,
                )

        if focus_terms and analysis.intent in {
            TaskIntent.ANALYZE,
            TaskIntent.INSPECT,
            TaskIntent.MODIFY,
            TaskIntent.FIX,
        }:
            for term in focus_terms:
                if term not in searched_queries:
                    return AgentDecision(
                        thought_summary="Search task-specific keywords before broad file reads.",
                        action_type=AgentActionType.CALL_TOOL,
                        tool_name="search_in_files",
                        tool_args={"query": term, "path": ".", "max_results": 30},
                        expected_outcome="Find candidate files related to the task.",
                        final_response=None,
                    )

        candidates = session.candidate_files or (snapshot.important_files if snapshot else [])
        for candidate in [*diagnostic_files, *candidates[:16]]:
            if candidate not in read_paths:
                return AgentDecision(
                    thought_summary="Read the highest-signal file instead of guessing.",
                    action_type=AgentActionType.CALL_TOOL,
                    tool_name="read_file",
                    tool_args={"path": candidate},
                    expected_outcome="Collect concrete implementation or failure context.",
                    final_response=None,
                )

        if session.changed_files and session.validation_status in {"not_run", "failed"}:
            command = self._pick_validation_command(session)
            if command:
                return AgentDecision(
                    thought_summary="Changed code must run through the remaining validation plan.",
                    action_type=AgentActionType.CALL_TOOL,
                    tool_name="run_tests",
                    tool_args={"command": command, "cwd": ".", "timeout": 120},
                    expected_outcome="Validate current changes or reproduce remaining failures.",
                    final_response=None,
                )

        if session.validation_status == "failed" and session.repair_attempts >= 1:
            if diagnostic_files:
                unread = next((path for path in diagnostic_files if path not in read_paths), None)
                if unread:
                    return AgentDecision(
                        thought_summary="Inspect the file hinted by the failing validation before giving up.",
                        action_type=AgentActionType.CALL_TOOL,
                        tool_name="read_file",
                        tool_args={"path": unread},
                        expected_outcome="Understand the failing area for a repair attempt.",
                        final_response=None,
                    )
            return AgentDecision(
                thought_summary="Validation is still failing and deterministic fallback cannot repair further.",
                action_type=AgentActionType.FINAL,
                tool_name=None,
                tool_args={},
                expected_outcome="Report the blocker and current repository state.",
                final_response=self.summarize_session(session),
            )

        if analysis_task and (session.tool_calls or session.notes):
            return AgentDecision(
                thought_summary="Enough inspection context exists for a concrete analysis summary.",
                action_type=AgentActionType.FINAL,
                tool_name=None,
                tool_args={},
                expected_outcome="Provide an architecture and gap analysis.",
                final_response=self.summarize_session(session),
            )

        return AgentDecision(
            thought_summary="No better deterministic next step is available.",
            action_type=AgentActionType.FINAL,
            tool_name=None,
            tool_args={},
            expected_outcome="Provide a concise task summary.",
            final_response=self.summarize_session(session),
        )

    def summarize_session(self, session: SessionState) -> str:
        direct_reply = session.task_analysis.direct_response if session.task_analysis else None
        if direct_reply is None:
            direct_reply = self._direct_reply_for_prompt(session.task)
        if direct_reply is not None:
            return direct_reply

        if session.changed_files and session.validation_status == "passed":
            return "Ich habe die Aufgabe umgesetzt und validiert."

        if session.changed_files and session.validation_status == "failed":
            return "Ich habe die Aufgabe weitgehend umgesetzt, aber die Validierung ist noch nicht sauber."

        if session.changed_files:
            return "Ich habe die Aufgabe umgesetzt."

        if session.status == "completed":
            return "Ich habe die Anfrage bearbeitet, aber keinen Code geaendert."

        if session.blockers:
            return "Ich bin auf einen Blocker gestossen und konnte die Aufgabe noch nicht sauber abschliessen."

        if session.validation_status == "failed":
            return "Ich habe Fortschritt gemacht, aber die Validierung ist noch nicht sauber durchgelaufen."

        return "Ich habe den Workspace untersucht, aber noch kein belastbares Abschlussergebnis erreicht."

    def _pick_validation_command(self, session: SessionState) -> str | None:
        passed = {
            run.command
            for run in session.validation_runs
            if run.edit_generation == session.edit_generation and run.status == "passed"
        }
        for item in session.validation_plan:
            if item.command not in passed:
                return item.command
        for command in session.verification_commands:
            if command and command not in passed:
                return command
        snapshot = session.workspace_snapshot
        if snapshot:
            for item in snapshot.validation_commands:
                if item.command not in passed:
                    return item.command
            for command in snapshot.likely_commands:
                if command and command not in passed:
                    return command
        return None

    def _looks_like_analysis_task(self, task: str) -> bool:
        lowered = task.lower()
        analysis_tokens = {
            "analyse",
            "analysiere",
            "bewerte",
            "review",
            "explain",
            "erklaer",
            "summarize",
            "zusammen",
            "understand",
        }
        action_tokens = {
            "fix",
            "behebe",
            "implement",
            "fuege",
            "refactor",
            "baue",
            "schreibe",
            "erweitere",
            "verbessere",
        }
        return any(token in lowered for token in analysis_tokens) and not any(
            token in lowered for token in action_tokens
        )

    def _might_need_helper(self, task: str) -> bool:
        lowered = task.lower()
        helper_tokens = {
            "log",
            "asset",
            "format",
            "parser",
            "convert",
            "analyse",
            "analyze",
            "build",
            "trace",
        }
        return any(token in lowered for token in helper_tokens)

    def _looks_like_greeting(self, task: str) -> bool:
        normalized = " ".join(task.lower().split()).strip("!?., ")
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

    def _direct_reply_for_prompt(self, task: str) -> str | None:
        normalized = " ".join(task.lower().split()).strip()
        if not normalized:
            return None
        if self._looks_like_greeting(task):
            return (
                "Hallo. Ich bin bereit."
                "\n\n"
                "Wenn du magst, kann ich den Code analysieren, einen Fehler suchen oder "
                "eine Aenderung umsetzen."
            )
        intro_fragments = (
            "wer bist du",
            "who are you",
            "was kannst du",
            "what can you do",
            "was machst du",
            "what do you do",
            "hilfe",
            "help",
        )
        if any(fragment in normalized for fragment in intro_fragments):
            return (
                "Ich bin M.A.R.C A1, dein lokaler Coding-Agent fuer den gewaehlten Workspace."
                "\n\n"
                "Ich kann Code analysieren, Aenderungen umsetzen, Fehler suchen, Tests anstossen "
                "und dir normale Rueckfragen zum Projekt beantworten."
            )
        return None

    def _focus_terms(self, task: str) -> list[str]:
        normalized = " ".join(task.lower().split())
        grouped_tokens: set[str] = set()
        tokens: list[str] = []

        for phrase, parts in SIGNAL_PHRASES:
            if phrase in normalized and phrase not in tokens:
                tokens.append(phrase.replace("-", " "))
                grouped_tokens.update(parts)

        for raw in task.lower().replace("/", " ").replace("-", " ").split():
            token = raw.strip(".,:;()[]{}!?\"'")
            if (
                len(token) < 3
                or token in STOPWORDS
                or token in LOW_SIGNAL_TERMS
                or token in CREATE_TOKENS
                or token in FIX_TOKENS
                or token in MODIFY_TOKENS
                or token in INSPECT_TOKENS
                or token in grouped_tokens
            ):
                continue
            if token not in tokens:
                tokens.append(token)
        return tokens[:4]

    def _normalize_decision(
        self,
        session: SessionState,
        analysis: TaskAnalysis,
        decision: AgentDecision,
    ) -> AgentDecision:
        read_paths = {
            item.tool_args.get("path")
            for item in session.tool_calls
            if item.tool_name == "read_file"
        }
        if decision.action_type != AgentActionType.CALL_TOOL:
            return decision
        if decision.tool_name != "search_in_files":
            return decision
        if analysis.intent != TaskIntent.CREATE:
            return decision
        query = str(decision.tool_args.get("query", "")).strip().lower()
        if query and query not in STOPWORDS and query not in LOW_SIGNAL_TERMS and len(query) >= 4:
            return decision
        bootstrap_file = self._pick_bootstrap_file(session, analysis, read_paths)
        if bootstrap_file is None:
            return decision
        return AgentDecision(
            thought_summary="Read a high-signal project file before creating new code.",
            action_type=AgentActionType.CALL_TOOL,
            tool_name="read_file",
            tool_args={"path": bootstrap_file},
            expected_outcome="Understand local conventions before creating files.",
            final_response=None,
        )

    def _deterministic_next_decision(
        self,
        session: SessionState,
        analysis: TaskAnalysis,
    ) -> AgentDecision | None:
        tool_names = [item.tool_name for item in session.tool_calls]
        read_paths = {
            item.tool_args.get("path")
            for item in session.tool_calls
            if item.tool_name == "read_file"
        }
        if not session.tool_calls:
            bootstrap_file = self._pick_bootstrap_file(session, analysis, read_paths)
            if bootstrap_file:
                return AgentDecision(
                    thought_summary="Read the closest manifest or example before creating new code.",
                    action_type=AgentActionType.CALL_TOOL,
                    tool_name="read_file",
                    tool_args={"path": bootstrap_file},
                    expected_outcome="Match existing project conventions before editing.",
                    final_response=None,
                )
            return AgentDecision(
                thought_summary="Need a repository map and validation plan before selecting files.",
                action_type=AgentActionType.CALL_TOOL,
                tool_name="inspect_workspace",
                tool_args={"focus": session.task},
                expected_outcome="Identify relevant files, workflows, and validation commands.",
                final_response=None,
            )

        if analysis.intent == TaskIntent.CREATE and len(read_paths) < 2:
            bootstrap_file = self._pick_bootstrap_file(session, analysis, read_paths)
            if bootstrap_file:
                return AgentDecision(
                    thought_summary="Read one more high-signal file before creating new code.",
                    action_type=AgentActionType.CALL_TOOL,
                    tool_name="read_file",
                    tool_args={"path": bootstrap_file},
                    expected_outcome="Clarify local conventions for the new deliverable.",
                    final_response=None,
                )
        if Path(session.workspace_root, ".git").exists() and "git_status" not in tool_names:
            return AgentDecision(
                thought_summary="Check git status before creating or editing files in a repository.",
                action_type=AgentActionType.CALL_TOOL,
                tool_name="git_status",
                tool_args={},
                expected_outcome="See current repository status before further work.",
                final_response=None,
            )
        return None

    def _draft_create_decision(
        self,
        session: SessionState,
        analysis: TaskAnalysis,
    ) -> AgentDecision | None:
        if analysis.intent != TaskIntent.CREATE:
            return None
        if session.changed_files:
            return None
        tool_names = [item.tool_name for item in session.tool_calls]
        if any(tool_name in WRITE_TOOLS for tool_name in tool_names):
            return None
        read_calls = [item for item in session.tool_calls if item.tool_name == "read_file"]
        if len(read_calls) < 1 and not self._can_create_without_reads(session, analysis):
            return None
        if Path(session.workspace_root, ".git").exists() and "git_status" not in tool_names:
            return None

        path = self._choose_create_path(session, analysis)
        if not path:
            return None
        content = self._generate_file_content(session, analysis, path)
        if not content:
            return None
        absolute_target = Path(session.workspace_root, path)
        tool_name = "write_file" if absolute_target.exists() else "create_file"
        tool_args = {"path": path, "content": content}
        if tool_name == "create_file":
            tool_args["overwrite"] = False
        return AgentDecision(
            thought_summary=f"Create the requested implementation in {path}.",
            action_type=AgentActionType.CALL_TOOL,
            tool_name=tool_name,
            tool_args=tool_args,
            expected_outcome="Add the requested implementation to the workspace.",
            final_response=None,
        )

    def _can_create_without_reads(
        self,
        session: SessionState,
        analysis: TaskAnalysis,
    ) -> bool:
        snapshot = session.workspace_snapshot
        if snapshot is None:
            return True
        if snapshot.file_count == 0:
            return True
        return self._pick_bootstrap_file(session, analysis, set()) is None

    def _fallback_task_analysis(
        self,
        task: str,
        snapshot: WorkspaceSnapshot | None,
    ) -> TaskAnalysis:
        direct_reply = self._direct_reply_for_prompt(task)
        if direct_reply is not None:
            return TaskAnalysis(
                summary="The user expects a direct conversational reply.",
                intent=TaskIntent.REPLY,
                requires_repo_context=False,
                should_create_files=False,
                deliverable="Natural language reply",
                search_terms=[],
                target_paths=[],
                relevant_extensions=[],
                direct_response=direct_reply,
            )

        intent = self._infer_intent(task)
        relevant_extensions = self._relevant_extensions(task)
        search_terms = self._focus_terms(task)
        target_paths = self._heuristic_target_paths(
            snapshot,
            intent=intent,
            relevant_extensions=relevant_extensions,
            search_terms=search_terms,
        )
        return TaskAnalysis(
            summary=self._analysis_summary(task, intent),
            intent=intent,
            requires_repo_context=intent != TaskIntent.REPLY,
            should_create_files=intent == TaskIntent.CREATE,
            deliverable=self._deliverable_hint(task, intent),
            search_terms=search_terms,
            target_paths=target_paths,
            relevant_extensions=relevant_extensions,
            direct_response=None,
        )

    def _normalize_task_analysis(
        self,
        task: str,
        snapshot: WorkspaceSnapshot | None,
        analysis: TaskAnalysis,
    ) -> TaskAnalysis:
        relevant_extensions = self._unique_terms(
            [*analysis.relevant_extensions, *self._relevant_extensions(task)]
        )[:4]
        search_terms = self._unique_terms(
            [*analysis.search_terms, *self._focus_terms(task)]
        )[:4]
        target_paths = self._normalize_target_paths(
            snapshot,
            analysis.target_paths,
            analysis.intent,
            relevant_extensions,
            search_terms,
        )
        direct_response = analysis.direct_response
        if analysis.intent == TaskIntent.REPLY and not direct_response:
            direct_response = self._direct_reply_for_prompt(task)
        return analysis.model_copy(
            update={
                "requires_repo_context": False
                if analysis.intent == TaskIntent.REPLY
                else analysis.requires_repo_context,
                "should_create_files": analysis.should_create_files
                or analysis.intent == TaskIntent.CREATE,
                "deliverable": analysis.deliverable or self._deliverable_hint(task, analysis.intent),
                "search_terms": search_terms,
                "target_paths": target_paths,
                "relevant_extensions": relevant_extensions,
                "direct_response": direct_response,
            }
        )

    def _infer_intent(self, task: str) -> TaskIntent:
        lowered = " ".join(task.lower().split())
        if self._looks_like_greeting(task) or self._direct_reply_for_prompt(task) is not None:
            return TaskIntent.REPLY
        if any(token in lowered for token in FIX_TOKENS):
            return TaskIntent.FIX
        if self._looks_like_analysis_task(task):
            return TaskIntent.ANALYZE
        if any(token in lowered for token in MODIFY_TOKENS):
            return TaskIntent.MODIFY
        if any(token in lowered for token in CREATE_TOKENS):
            if self._mentions_existing_path(task):
                return TaskIntent.MODIFY
            return TaskIntent.CREATE
        if any(token in lowered for token in INSPECT_TOKENS):
            return TaskIntent.INSPECT
        if "?" in task:
            return TaskIntent.ANALYZE
        return TaskIntent.INSPECT

    def _mentions_existing_path(self, task: str) -> bool:
        return bool(
            re.search(
                r"[\w./-]+\.(py|js|ts|tsx|jsx|json|md|html|css|sh|toml|ya?ml|go|rs|java|kt|rb)",
                task.lower(),
            )
        )

    def _relevant_extensions(self, task: str) -> list[str]:
        lowered = task.lower()
        return [
            extension
            for token, extension in LANGUAGE_EXTENSIONS.items()
            if token in lowered
        ][:4]

    def _analysis_summary(self, task: str, intent: TaskIntent) -> str:
        summaries = {
            TaskIntent.REPLY: "Answer the user directly without repository work.",
            TaskIntent.ANALYZE: "Inspect the repository and return an explanation or assessment.",
            TaskIntent.INSPECT: "Inspect the repository to find the relevant implementation area.",
            TaskIntent.MODIFY: "Change existing code in the repository to satisfy the request.",
            TaskIntent.CREATE: "Create a new implementation that matches the user's requested deliverable.",
            TaskIntent.FIX: "Diagnose the failure and repair the existing implementation.",
        }
        return summaries.get(intent, f"Handle the request: {task}")

    def _deliverable_hint(self, task: str, intent: TaskIntent) -> str | None:
        lowered = task.lower()
        if intent == TaskIntent.CREATE:
            if "script" in lowered or "skript" in lowered:
                return "New script"
            if "component" in lowered or "komponente" in lowered:
                return "New component"
            if "api" in lowered:
                return "New API implementation"
            if "calculator" in lowered or "taschenrechner" in lowered:
                return "Calculator implementation"
            return "New implementation"
        if intent == TaskIntent.FIX:
            return "Bug fix"
        if intent == TaskIntent.ANALYZE:
            return "Analysis summary"
        return None

    def _heuristic_target_paths(
        self,
        snapshot: WorkspaceSnapshot | None,
        *,
        intent: TaskIntent,
        relevant_extensions: list[str],
        search_terms: list[str],
    ) -> list[str]:
        if snapshot is None:
            return []
        candidates: list[str] = []
        if intent == TaskIntent.CREATE:
            candidates.extend(snapshot.manifests[:4])
            candidates.extend(snapshot.entrypoints[:4])
        candidates.extend(snapshot.focus_files[:8])
        if relevant_extensions:
            candidates.extend(
                path
                for path in snapshot.important_files[:20]
                if any(path.endswith(extension) for extension in relevant_extensions)
            )
        if search_terms:
            lowered_terms = tuple(search_terms)
            candidates.extend(
                path
                for path in snapshot.important_files[:20]
                if any(term in path.lower() for term in lowered_terms)
            )
        candidates.extend(snapshot.important_files[:8])
        return self._unique_terms(candidates)[:8]

    def _normalize_target_paths(
        self,
        snapshot: WorkspaceSnapshot | None,
        proposed_paths: list[str],
        intent: TaskIntent,
        relevant_extensions: list[str],
        search_terms: list[str],
    ) -> list[str]:
        snapshot_paths = set()
        if snapshot is not None:
            snapshot_paths.update(snapshot.important_files)
            snapshot_paths.update(snapshot.focus_files)
            snapshot_paths.update(snapshot.manifests)
            snapshot_paths.update(snapshot.entrypoints)
        normalized = [
            path
            for path in self._unique_terms(proposed_paths)
            if path in snapshot_paths or not snapshot_paths
        ]
        if normalized:
            return normalized[:8]
        return self._heuristic_target_paths(
            snapshot,
            intent=intent,
            relevant_extensions=relevant_extensions,
            search_terms=search_terms,
        )

    def _pick_bootstrap_file(
        self,
        session: SessionState,
        analysis: TaskAnalysis,
        read_paths: set[str],
    ) -> str | None:
        snapshot = session.workspace_snapshot
        if snapshot is None:
            return None
        candidates = self._normalize_target_paths(
            snapshot,
            analysis.target_paths,
            analysis.intent,
            analysis.relevant_extensions,
            analysis.search_terms,
        )
        if analysis.intent == TaskIntent.CREATE:
            candidates = self._unique_terms(
                [
                    *candidates,
                    *snapshot.manifests[:4],
                    *snapshot.entrypoints[:4],
                ]
            )
        for candidate in candidates:
            if candidate and candidate not in read_paths and Path(session.workspace_root, candidate).exists():
                return candidate
        return None

    def _choose_create_path(
        self,
        session: SessionState,
        analysis: TaskAnalysis,
    ) -> str | None:
        task = session.task
        preferred_extension = analysis.relevant_extensions[0] if analysis.relevant_extensions else ""
        prompt = "\n".join(
            [
                f"Task: {task}",
                f"Deliverable: {analysis.deliverable or 'new implementation'}",
                f"Project labels: {', '.join(session.workspace_snapshot.project_labels[:6]) if session.workspace_snapshot else 'none'}",
                f"Read files: {', '.join(item.tool_args.get('path', '') for item in session.tool_calls if item.tool_name == 'read_file')}",
                f"Preferred extension: {preferred_extension or 'choose the most sensible extension'}",
                "Choose one relative workspace file path for the new implementation.",
                "Prefer a simple path, no explanation, path only.",
            ]
        )
        try:
            response = self.llm.generate(
                prompt,
                timeout=self._llm_timeout(6),
                num_ctx=self._llm_num_ctx(2048),
            )
            path = self._sanitize_generated_path(response, preferred_extension)
            if path:
                return path
        except Exception:
            pass
        return self._default_create_path(task, preferred_extension or ".txt")

    def _generate_file_content(
        self,
        session: SessionState,
        analysis: TaskAnalysis,
        path: str,
    ) -> str | None:
        prompt = "\n\n".join(
            [
                "Create the complete file content for the requested task.",
                f"Task: {session.task}",
                f"Target file: {path}",
                f"Deliverable: {analysis.deliverable or 'new implementation'}",
                f"Project labels: {', '.join(session.workspace_snapshot.project_labels[:6]) if session.workspace_snapshot else 'none'}",
                self._inspected_context(session),
                "Requirements: single-file solution, minimal working implementation, keep the code concise, prefer simple terminal/CLI interaction unless the task explicitly asks for a web or GUI app.",
                "Return the file content only. Do not wrap the answer in markdown fences. Do not explain anything.",
            ]
        )
        try:
            content = self.llm.generate(
                prompt,
                timeout=max(self._llm_timeout(90), 90),
                num_ctx=self._llm_num_ctx(4096),
            )
            return self._strip_code_fences(content).strip()
        except Exception:
            return None

    def _inspected_context(self, session: SessionState) -> str:
        sections: list[str] = []
        for item in session.tool_calls:
            if item.tool_name != "read_file":
                continue
            path = str(item.tool_args.get("path", "")).strip()
            excerpt = (item.output_excerpt or "").strip()
            if not path or not excerpt:
                continue
            sections.append(f"Inspected {path}:\n{excerpt[:1200]}")
            if len(sections) >= 2:
                break
        if not sections and session.workspace_snapshot:
            sections.append(f"Workspace summary:\n{session.workspace_snapshot.repo_summary[:800]}")
        return "\n\n".join(sections)

    def _sanitize_generated_path(
        self,
        raw: str,
        preferred_extension: str,
    ) -> str | None:
        text = self._strip_code_fences(raw).strip().splitlines()[0].strip("`'\" ")
        if not text:
            return None
        text = text.lstrip("./")
        text = text.replace("\\", "/")
        text = re.sub(r"\s+", "_", text)
        if "/" in text:
            parts = [part for part in text.split("/") if part not in {"", ".", ".."}]
            text = "/".join(parts)
        if not text:
            return None
        if preferred_extension and not text.endswith(preferred_extension):
            if "." not in Path(text).name:
                text = f"{text}{preferred_extension}"
        return text

    def _default_create_path(self, task: str, preferred_extension: str) -> str:
        meaningful = [
            token
            for token in self._focus_terms(task)
            if token not in LOW_SIGNAL_TERMS
        ]
        base = meaningful[0] if meaningful else "generated_output"
        slug = re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_") or "generated_output"
        extension = preferred_extension if preferred_extension.startswith(".") else f".{preferred_extension}"
        return f"{slug}{extension}"

    def _strip_code_fences(self, text: str) -> str:
        cleaned = str(text or "").strip()
        if cleaned.startswith("```"):
            parts = cleaned.split("```")
            if len(parts) >= 3:
                cleaned = parts[1]
        return cleaned.removeprefix("python").removeprefix("py").strip()

    def _unique_terms(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        items: list[str] = []
        for value in values:
            normalized = str(value or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            items.append(normalized)
        return items

    def _llm_timeout(self, limit: int) -> int:
        config = getattr(self.llm, "config", None)
        if config is None:
            return limit
        return min(config.llm_timeout, limit)

    def _llm_num_ctx(self, limit: int) -> int:
        config = getattr(self.llm, "config", None)
        if config is None:
            return limit
        return min(config.ollama_num_ctx, limit)
