from __future__ import annotations

import json

from agent.models import FileChangeRecord, FollowUpContext, SessionState
from agent.router import IntentRouter
from llm.schemas import RouteActionName, RouteIntent, router_output_schema
from runtime.logger import AgentLogger


class ScriptedLLM:
    def __init__(self, json_payloads=None, fail=False, fail_times: int = 0, fail_message: str = "llm unavailable"):
        self.json_payloads = list(json_payloads or [])
        self.fail = fail
        self.fail_times = fail_times
        self.fail_message = fail_message
        self.prompts: list[str] = []

    def generate_json(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError(self.fail_message)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError(self.fail_message)
        if not self.json_payloads:
            raise RuntimeError("No JSON payload configured")
        return self.json_payloads.pop(0)

    def generate(self, *args, **kwargs):
        raise RuntimeError("Text generation not configured for this test")


def make_payload(
    prompt: str,
    *,
    intent: str,
    action: str,
    safe_to_execute: bool = True,
    needs_clarification: bool = False,
    questions=None,
    direct_response=None,
):
    return {
        "user_goal": f"Handle: {prompt}",
        "intent": intent,
        "entities": {
            "target_type": "file" if "datei" in prompt.lower() or "file" in prompt.lower() else None,
            "target_name": None,
            "target_paths": [],
            "attributes": [],
            "constraints": [],
        },
        "requested_outcome": "Produce the intended result safely.",
        "action_plan": [
            {
                "step": 1,
                "action": action,
                "reason": "Follow the routed intent with the minimum safe first step.",
            }
        ],
        "needs_clarification": needs_clarification,
        "clarification_questions": questions or [],
        "confidence": 0.88 if safe_to_execute else 0.22,
        "safe_to_execute": safe_to_execute,
        "repo_context_needed": action not in {RouteActionName.RESPOND_DIRECTLY.value, RouteActionName.ASK_CLARIFICATION.value},
        "search_terms": [],
        "relevant_extensions": [],
        "direct_response": direct_response,
    }


def test_router_repairs_invalid_json_output(tmp_path):
    logger = AgentLogger(tmp_path, "router-repair")
    llm = ScriptedLLM(
        json_payloads=[
            {
                "user_goal": "Broken payload",
                "intent": "unknown",
                "entities": {
                    "target_type": None,
                    "target_name": None,
                    "target_paths": [],
                    "attributes": [],
                    "constraints": [],
                },
                "requested_outcome": "Need clarification.",
                "action_plan": [
                    {
                        "step": 1,
                        "action": "ask_clarification",
                        "reason": "Missing information.",
                    }
                ],
                "needs_clarification": True,
                "clarification_questions": ["Was genau meinst du?"],
                "confidence": 0.2,
                "safe_to_execute": True,
                "repo_context_needed": False,
                "search_terms": [],
                "relevant_extensions": [],
                "direct_response": None,
            },
            {
                "user_goal": "Clarify the ambiguous request.",
                "intent": "unknown",
                "entities": {
                    "target_type": None,
                    "target_name": None,
                    "target_paths": [],
                    "attributes": [],
                    "constraints": [],
                },
                "requested_outcome": "Ask for the missing target details.",
                "action_plan": [
                    {
                        "step": 1,
                        "action": "ask_clarification",
                        "reason": "The request is ambiguous.",
                    }
                ],
                "needs_clarification": True,
                "clarification_questions": ["Welche Datei oder welchen Bereich meinst du genau?"],
                "confidence": 0.2,
                "safe_to_execute": False,
                "repo_context_needed": False,
                "search_terms": [],
                "relevant_extensions": [],
                "direct_response": None,
            },
        ]
    )
    router = IntentRouter(llm, logger=logger)

    route = router.interpret_user_request("Mach das mal anders", None)

    assert route.intent == RouteIntent.UNKNOWN
    assert route.safe_to_execute is False
    assert route.needs_clarification is True
    assert "Welch" in route.clarification_questions[0]
    logs = (tmp_path / "router-repair.jsonl").read_text(encoding="utf-8")
    assert "router_validation_failed" in logs
    assert "router_repair_succeeded" in logs


def test_router_falls_back_to_unknown_if_provider_fails():
    router = IntentRouter(ScriptedLLM(fail=True))

    route = router.interpret_user_request("Kannst du das mal machen?", None)

    assert route.intent == RouteIntent.UNKNOWN
    assert route.needs_clarification is True
    assert route.safe_to_execute is False


def test_router_fallback_still_answers_simple_intro_questions():
    router = IntentRouter(ScriptedLLM(fail=True))

    route = router.interpret_user_request("hallo wer bist du?", None)

    assert route.intent == RouteIntent.EXPLAIN
    assert route.safe_to_execute is True
    assert "Coding-Agent" in (route.direct_response or "")


def test_router_fast_paths_clear_create_request_without_calling_llm():
    llm = ScriptedLLM(fail=True, fail_message="should not be called")
    router = IntentRouter(llm)

    route = router.interpret_user_request("Programmiere mir ein Tic Tac Toe in Python", None)

    assert route.intent == RouteIntent.CREATE
    assert route.safe_to_execute is True
    assert route.action_plan[0].action == RouteActionName.CREATE_ARTIFACT
    assert route.entities.target_name == "tic tac toe"
    assert llm.prompts == []


def test_router_recognizes_tictactoe_without_spaces():
    router = IntentRouter(ScriptedLLM(fail=True, fail_message="should not be called"))

    route = router.interpret_user_request("schreib ein python TicTacToe spiel", None)

    assert route.intent == RouteIntent.CREATE
    assert route.entities.target_name == "tic tac toe"


def test_router_treats_computer_opponent_follow_up_as_update():
    router = IntentRouter(ScriptedLLM(fail=True, fail_message="should not be called"))
    session = SessionState(task="vorheriger prompt", workspace_root=".")
    session.changed_files.append(FileChangeRecord(path="tic_tac_toe.py", operation="create"))

    route = router.interpret_user_request(
        "ich moechte gegen einen computer spielen statt mit 2 spielern",
        None,
        session=session,
    )

    assert route.intent == RouteIntent.UPDATE
    assert route.entities.target_paths == ["tic_tac_toe.py"]


def test_router_treats_generic_follow_up_as_update_without_new_artifact_request():
    router = IntentRouter(ScriptedLLM(fail=True, fail_message="should not be called"))
    session = SessionState(task="vorheriger prompt", workspace_root=".")
    session.changed_files.append(FileChangeRecord(path="dashboard.py", operation="create"))

    route = router.interpret_user_request(
        "mach die schrift groesser und die farben freundlicher",
        None,
        session=session,
    )

    assert route.intent == RouteIntent.UPDATE
    assert route.entities.target_paths == ["dashboard.py"]


def test_router_treats_vague_bug_follow_up_as_debug_route():
    router = IntentRouter(ScriptedLLM(fail=True, fail_message="should not be called"))
    session = SessionState(task="mach ein tic tac toe", workspace_root=".")
    session.follow_up_context = FollowUpContext(
        previous_task="mach ein tic tac toe",
        target_paths=["tic_tac_toe.py"],
        changed_files=["tic_tac_toe.py"],
        recent_commands=["python tic_tac_toe.py"],
        last_error="Traceback: invalid move handling",
    )

    route = router.interpret_user_request(
        "ah da ist ein fehler im terminal sieht alles buggy aus",
        None,
        session=session,
    )

    assert route.intent == RouteIntent.DEBUG
    assert route.entities.target_paths == ["tic_tac_toe.py"]
    assert any(step.action == RouteActionName.DIAGNOSE_ISSUE for step in route.action_plan)


def test_router_timeout_retry_can_recover_with_second_attempt():
    llm = ScriptedLLM(
        json_payloads=[
            make_payload(
                "bitte programmier mir ein Tic tac toe spiel in python",
                intent="create",
                action="create_artifact",
            )
        ],
        fail_times=1,
        fail_message="timed out",
    )
    router = IntentRouter(llm)

    route = router.interpret_user_request("bitte programmier mir ein Tic tac toe spiel in python", None)

    assert route.intent == RouteIntent.CREATE
    assert route.safe_to_execute is True
    assert route.action_plan[0].action == RouteActionName.CREATE_ARTIFACT


def test_router_timeout_fallback_classifies_clear_create_request_without_llm():
    router = IntentRouter(ScriptedLLM(fail=True, fail_message="timed out"))

    route = router.interpret_user_request(
        "ich moechte ein Tic Tac Toe spiel in python haben das ich spielen kann",
        None,
    )

    assert route.intent == RouteIntent.CREATE
    assert route.safe_to_execute is True
    assert route.needs_clarification is False
    assert route.action_plan[0].action == RouteActionName.CREATE_ARTIFACT
    assert ".py" in route.relevant_extensions


def test_router_schema_contains_core_contract_fields():
    schema = router_output_schema()

    assert "properties" in schema
    assert "intent" in schema["properties"]
    assert "action_plan" in schema["properties"]
    assert "confidence" in schema["properties"]
    assert "safe_to_execute" in schema["properties"]


def test_router_logs_raw_input_and_validation_result(tmp_path):
    logger = AgentLogger(tmp_path, "router-log")
    prompt = "Schau dir mal die Datei an"
    llm = ScriptedLLM(
        json_payloads=[
            make_payload(
                prompt,
                intent="inspect",
                action="read_relevant_files",
            )
        ]
    )
    router = IntentRouter(llm, logger=logger)

    router.interpret_user_request(prompt, None, session=SessionState(task=prompt, workspace_root=str(tmp_path)))

    records = (tmp_path / "router-log.jsonl").read_text(encoding="utf-8")
    assert "router_input" in records
    assert "router_validation_succeeded" in records


PROMPT_SUITE = [
    ("Schau dir mal die Datei an", "inspect", "read_relevant_files", False),
    ("Pruef bitte den Inhalt", "inspect", "read_relevant_files", False),
    ("Kannst du das inspizieren?", "inspect", "inspect_workspace", False),
    ("Was ist da drin?", "inspect", "read_relevant_files", False),
    ("Analysiere das mal", "inspect", "inspect_workspace", False),
    ("Leg bitte eine neue API-Route an", "create", "create_artifact", False),
    ("Bau mir was fuer den Import", "create", "create_artifact", False),
    ("Kannst du eine kleine Helper-Datei anlegen?", "create", "create_artifact", False),
    ("Mach aus dem Modul was Neues", "update", "update_artifact", False),
    ("Bitte passe den bestehenden Handler an", "update", "update_artifact", False),
    ("Korrigier das im Service", "update", "update_artifact", False),
    ("Loesch die alte Datei", "delete", "delete_artifact", False),
    ("Raeum den veralteten Kram raus", "delete", "delete_artifact", False),
    ("Kann die Komponente weg?", "delete", "delete_artifact", False),
    ("Wo steckt die Logik fuer die Auth?", "search", "search_workspace", False),
    ("Find mal die Stelle mit dem Token-Check", "search", "search_workspace", False),
    ("Such bitte nach der Config dafuer", "search", "search_workspace", False),
    ("Erklaer mir kurz, wie das hier zusammenhaengt", "explain", "summarize_result", False),
    ("Was macht der Agent da eigentlich?", "explain", "respond_directly", False),
    ("Kannst du mir das Konzept erklaeren?", "explain", "respond_directly", False),
    ("Mach mir erstmal einen Plan", "plan", "plan_work", False),
    ("Wie wuerdest du das umbauen?", "plan", "plan_work", False),
    ("Skizzier die naechsten Schritte", "plan", "plan_work", False),
    ("mach ma iwie besser", "unknown", "ask_clarification", True),
    ("pls updtae das dingens", "unknown", "ask_clarification", True),
    ("irgendwas mit der datei da", "unknown", "ask_clarification", True),
    ("??", "unknown", "ask_clarification", True),
]


def test_router_handles_diverse_prompt_suite():
    for prompt, expected_intent, expected_action, expects_clarification in PROMPT_SUITE:
        payload = make_payload(
            prompt,
            intent=expected_intent,
            action=expected_action,
            safe_to_execute=not expects_clarification,
            needs_clarification=expects_clarification,
            questions=["Was genau soll ich tun?"] if expects_clarification else [],
            direct_response="Kurz erklaert." if expected_action == "respond_directly" else None,
        )
        router = IntentRouter(ScriptedLLM(json_payloads=[payload]))

        route = router.interpret_user_request(prompt, None)

        assert route.intent.value == expected_intent
        assert route.action_plan[0].action.value == expected_action
        assert route.needs_clarification is expects_clarification


def test_router_prompt_mentions_semantic_goal_routing():
    prompt = "Kannst du das mal inspizieren?"
    llm = ScriptedLLM(
        json_payloads=[
            make_payload(
                prompt,
                intent="inspect",
                action="inspect_workspace",
            )
        ]
    )
    router = IntentRouter(llm)

    router.interpret_user_request(prompt, None)

    emitted_prompt = llm.prompts[0]
    assert "Infer intent from meaning" in emitted_prompt
    assert "Allowed actions" in emitted_prompt
    assert "intent=unknown" in emitted_prompt
    assert "direct_response" in emitted_prompt


def test_router_output_schema_is_json_serializable():
    schema_text = json.dumps(router_output_schema(), ensure_ascii=False)
    assert '"intent"' in schema_text
    assert '"action_plan"' in schema_text
