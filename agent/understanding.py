from __future__ import annotations

from typing import Any

from agent.prompts import task_understanding_prompt, task_understanding_system_prompt
from agent.task_schema import TaskArtifact, TaskPlanStep, TaskUnderstanding
from llm.provider import LLMProvider
from runtime.logger import AgentLogger


class TaskInterpreter:
    """Turns raw natural language into a normalized task object before execution planning."""

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

    def interpret(
        self,
        user_input: str,
        *,
        snapshot=None,
        session=None,
    ) -> TaskUnderstanding:
        payload: dict[str, Any] | None = None
        try:
            payload = self.llm.generate_json(
                task_understanding_prompt(user_input, snapshot=snapshot, session=session),
                system=task_understanding_system_prompt(),
                model=self.model_name,
                retries=0,
                timeout=self.timeout,
                num_ctx=self.num_ctx,
            )
            understanding = TaskUnderstanding.model_validate(payload)
            self._log("task_understanding", understanding=understanding.model_dump())
            return understanding
        except Exception as exc:
            self._log(
                "task_understanding_fallback",
                error=str(exc),
                payload=payload or {},
            )
            fallback = self._fallback_understanding(user_input, session=session)
            self._log("task_understanding", understanding=fallback.model_dump(), source="fallback")
            return fallback

    def _fallback_understanding(self, user_input: str, *, session=None) -> TaskUnderstanding:
        request = str(user_input or "").strip() or "Unclear request"
        recent_paths: list[str] = []
        if session is not None:
            if getattr(session, "follow_up_context", None) is not None:
                recent_paths.extend(session.follow_up_context.target_paths[:4])
            recent_paths.extend(session.candidate_files[:4])
        artifacts = [
            TaskArtifact(path=path, name=path.split("/")[-1], kind="file", role="active_context", confidence=0.55)
            for path in recent_paths[:4]
        ]
        plan = [
            TaskPlanStep(step=1, summary="Inspect the most relevant context before acting.", action_hint="inspect"),
            TaskPlanStep(step=2, summary="Choose the smallest safe implementation step.", action_hint="plan"),
        ]
        return TaskUnderstanding(
            original_request=request,
            interpreted_goal=request,
            intent_category="unknown",
            conversation_relation="same_task_follow_up" if session is not None else "unknown",
            subgoals=["Clarify the immediate task and inspect context."],
            target_artifacts=artifacts,
            relevant_context=["Recent session context is available."] if session is not None else [],
            constraints=[],
            missing_info=["The exact target is still ambiguous."],
            assumptions=["Continue from the current task unless the user clearly changed topic."],
            user_observations=[],
            supplied_evidence=[],
            ambiguity_level="high",
            risk_level="medium",
            confidence=0.35,
            recommended_mode="clarify",
            execution_plan=plan,
            needs_clarification=True,
            clarification_questions=["Welchen konkreten Bereich soll ich als naechstes bearbeiten?"],
        )

    def _log(self, event: str, **payload: Any) -> None:
        if self.logger is None:
            return
        self.logger.log_event(event, **payload)
