from __future__ import annotations

from agent.models import ToolRunResult
from llm.schemas import AgentDecision, AgentActionType
from runtime.tool_dispatcher import ToolDispatcher


class Executor:
    def __init__(self, dispatcher: ToolDispatcher):
        self.dispatcher = dispatcher

    def execute(self, decision: AgentDecision, iteration: int) -> ToolRunResult:
        if decision.action_type != AgentActionType.CALL_TOOL or not decision.tool_name:
            raise ValueError("Executor can only run tool decisions.")
        return self.dispatcher.dispatch(
            tool_name=decision.tool_name,
            raw_args=decision.tool_args,
            iteration=iteration,
        )
