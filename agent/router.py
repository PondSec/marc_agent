from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from agent.models import SessionState, WorkspaceSnapshot
from agent.prompts import router_prompt, router_repair_prompt, router_system_prompt
from llm.provider import LLMProvider
from llm.schemas import RouteActionName, RouteIntent, RouterOutput
from runtime.logger import AgentLogger


class IntentRouter:
    def __init__(
        self,
        llm: LLMProvider,
        *,
        logger: AgentLogger | None = None,
        timeout: int = 12,
        num_ctx: int = 4096,
    ):
        self.llm = llm
        self.logger = logger
        self.timeout = timeout
        self.num_ctx = num_ctx

    def interpret_user_request(
        self,
        user_input: str,
        snapshot: WorkspaceSnapshot | None,
        session: SessionState | None = None,
    ) -> RouterOutput:
        self._log("router_input", raw_user_input=user_input)
        payload: dict[str, Any] | None = None
        try:
            payload = self.llm.generate_json(
                router_prompt(user_input, snapshot, session=session),
                system=router_system_prompt(),
                timeout=self.timeout,
                num_ctx=self.num_ctx,
            )
            return self.validate_router_output(payload)
        except ValidationError as exc:
            self._log(
                "router_validation_failed",
                raw_router_output=payload or {},
                errors=exc.errors(),
            )
            repaired = self._repair_invalid_output(payload or {}, exc.errors())
            if repaired is not None:
                return repaired
        except Exception as exc:
            self._log(
                "router_error",
                raw_router_output=payload or {},
                error=str(exc),
            )
        fallback = self._fallback_route(user_input)
        self._log("router_fallback", router_result=fallback.model_dump())
        return fallback

    def validate_router_output(self, payload: dict[str, Any]) -> RouterOutput:
        route = RouterOutput.model_validate(payload)
        self._log(
            "router_validation_succeeded",
            validation_result="valid",
            router_result=route.model_dump(),
        )
        return route

    def _repair_invalid_output(
        self,
        invalid_payload: dict[str, Any],
        errors: list[dict[str, Any]],
    ) -> RouterOutput | None:
        try:
            repaired = self.llm.generate_json(
                router_repair_prompt(invalid_payload, errors),
                system=router_system_prompt(),
                timeout=max(self.timeout, 16),
                num_ctx=self.num_ctx,
            )
            route = self.validate_router_output(repaired)
            self._log("router_repair_succeeded", router_result=route.model_dump())
            return route
        except Exception as exc:
            self._log(
                "router_repair_failed",
                raw_router_output=invalid_payload,
                errors=errors,
                error=str(exc),
            )
            return None

    def _fallback_route(self, user_input: str) -> RouterOutput:
        trimmed = str(user_input or "").strip() or "Unklarer Nutzerwunsch"
        direct_response = self._fallback_direct_response(trimmed)
        if direct_response is not None:
            return RouterOutput(
                user_goal=trimmed,
                intent=RouteIntent.EXPLAIN,
                requested_outcome="Provide a short direct answer without repository work.",
                action_plan=[
                    {
                        "step": 1,
                        "action": RouteActionName.RESPOND_DIRECTLY,
                        "reason": "This is a simple conversational request that does not require tool execution.",
                    }
                ],
                needs_clarification=False,
                clarification_questions=[],
                confidence=0.3,
                safe_to_execute=True,
                repo_context_needed=False,
                search_terms=[],
                relevant_extensions=[],
                direct_response=direct_response,
            )
        return RouterOutput(
            user_goal=trimmed,
            intent=RouteIntent.UNKNOWN,
            requested_outcome="Clarify the user's exact target before doing any work.",
            action_plan=[
                {
                    "step": 1,
                    "action": RouteActionName.ASK_CLARIFICATION,
                    "reason": "The router could not derive a safe executable goal with enough confidence.",
                }
            ],
            needs_clarification=True,
            clarification_questions=[
                "Was genau soll ich fuer dich erreichen?",
                "Auf welches Objekt, welche Datei oder welchen Bereich soll ich mich konzentrieren?",
            ],
            confidence=0.0,
            safe_to_execute=False,
            repo_context_needed=False,
            search_terms=[],
            relevant_extensions=[],
            direct_response=None,
        )

    def _fallback_direct_response(self, user_input: str) -> str | None:
        normalized = " ".join(str(user_input or "").lower().split()).strip("!?., ")
        if not normalized:
            return None
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
        if normalized in greetings:
            return (
                "Hallo. Ich bin bereit.\n\n"
                "Wenn du magst, kann ich den Code analysieren, eine Aenderung planen oder etwas im Projekt umsetzen."
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
                "Ich bin dein lokaler Coding-Agent fuer diesen Workspace.\n\n"
                "Ich kann Code analysieren, Aenderungen planen und auf Basis des validierten Router-Outputs ausfuehren."
            )
        return None

    def _log(self, event: str, **payload: Any) -> None:
        if self.logger is None:
            return
        self.logger.log_event(event, **payload)
