from __future__ import annotations

from pydantic import ValidationError

from agent.models import FileChangeRecord, ToolRunResult
from runtime.logger import AgentLogger
from tools.registry import ToolRegistry


class ToolDispatcher:
    def __init__(self, registry: ToolRegistry, logger: AgentLogger):
        self.registry = registry
        self.logger = logger

    def dispatch(self, tool_name: str, raw_args: dict, iteration: int) -> ToolRunResult:
        spec = self.registry.get(tool_name)
        if spec is None:
            result = ToolRunResult(
                tool_name=tool_name,
                success=False,
                message=f"Unknown tool: {tool_name}",
            )
            self.logger.log_event("tool_error", iteration=iteration, tool=tool_name, message=result.message)
            return result

        try:
            parsed_args = spec.input_model.model_validate(raw_args or {})
        except ValidationError as exc:
            result = ToolRunResult(
                tool_name=tool_name,
                success=False,
                message="Tool argument validation failed.",
                data={"errors": exc.errors()},
            )
            self.logger.log_event(
                "tool_validation_error",
                iteration=iteration,
                tool=tool_name,
                errors=exc.errors(),
            )
            return result

        try:
            payload = spec.handler(parsed_args)
        except Exception as exc:
            result = ToolRunResult(
                tool_name=tool_name,
                success=False,
                message=f"Tool execution failed: {exc}",
            )
            self.logger.log_event(
                "tool_execution_error",
                iteration=iteration,
                tool=tool_name,
                error=str(exc),
            )
            return result

        changed_files = [
            FileChangeRecord.model_validate(item)
            for item in payload.get("changed_files", [])
        ]
        result = ToolRunResult(
            tool_name=tool_name,
            success=bool(payload.get("success", True)),
            message=str(payload.get("message", "")),
            data=payload,
            risk_level=payload.get("risk_level"),
            changed_files=changed_files,
        )
        self.logger.log_event(
            "tool_result",
            iteration=iteration,
            tool=tool_name,
            success=result.success,
            message=result.message,
            risk_level=result.risk_level,
        )
        return result
