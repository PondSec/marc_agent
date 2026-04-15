from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Literal

from agent.local_nlp import classify_fallback_intent


_COMPOUND_REPLACEMENTS = (
    ("to do", "todo"),
    ("to-do", "todo"),
)

_CREATE_SIGNALS = (
    "bau",
    "build",
    "create",
    "erstell",
    "anleg",
    "generate",
    "implement",
    "add",
    "mach",
    "make",
    "program",
    "programmier",
    "fueg",
    "füge",
    "hinzu",
    "ergaenz",
    "ergänz",
    "schreib",
    "write",
    "i want",
    "ich moechte",
    "ich möchte",
    "ich will",
    "möchte",
    "want",
)

_NEED_CREATE_SIGNALS = (
    "bedarf",
    "benoetig",
    "benötig",
    "brauch",
    "need",
)

_IMPLEMENTATION_NOUNS = (
    "api",
    "app",
    "bot",
    "cli",
    "dashboard",
    "datei",
    "file",
    "game",
    "helper",
    "library",
    "modul",
    "module",
    "oberflaeche",
    "oberfläche",
    "page",
    "script",
    "service",
    "site",
    "spiel",
    "tool",
    "ui",
    "frontend",
    "component",
    "komponente",
    "webseite",
)

_EXTENSION_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (".py", ("python", "flask", "django", "fastapi", "pyside", "tkinter")),
    (".js", ("javascript", "node", "nodejs", "express")),
    (".ts", ("typescript",)),
    (".tsx", ("react", "next.js", "nextjs")),
    (".html", ("html",)),
    (".css", ("css",)),
    (".sh", ("bash", "shell", "zsh")),
    (".go", ("golang",)),
    (".rs", ("rust",)),
    (".rb", ("ruby",)),
    (".php", ("php",)),
    (".java", ("java",)),
)

_EXPLICIT_EXTENSION_PHRASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (".html", ("html datei", "html file", "html page", "als html")),
    (".css", ("css datei", "css file", "als css")),
    (".js", ("javascript datei", "javascript file", "js datei", "js file", "als javascript", "als js")),
    (".ts", ("typescript datei", "typescript file", "ts datei", "ts file", "als typescript", "als ts")),
    (".py", ("python datei", "python file", "als python")),
)

_GENERIC_MEDIUM_TOKENS = {
    "bash",
    "html",
    "javascript",
    "python",
    "shell",
    "typescript",
    "zsh",
}

_GENERIC_SCOPE_STOPWORDS = {
    "a",
    "an",
    "and",
    "bau",
    "bitte",
    "build",
    "create",
    "das",
    "dazu",
    "dem",
    "den",
    "der",
    "des",
    "die",
    "ein",
    "eine",
    "einen",
    "einfach",
    "einfaches",
    "erstell",
    "erstelle",
    "erstellen",
    "erstellt",
    "etwas",
    "fuer",
    "für",
    "game",
    "haben",
    "have",
    "hinzu",
    "ich",
    "implement",
    "in",
    "it",
    "jetzt",
    "javascript",
    "kannst",
    "klein",
    "kleine",
    "local",
    "mach",
    "mache",
    "machen",
    "make",
    "me",
    "mir",
    "mit",
    "moechte",
    "möchte",
    "noch",
    "new",
    "please",
    "program",
    "programmiere",
    "programmier",
    "python",
    "schreib",
    "schreibe",
    "script",
    "service",
    "spiel",
    "something",
    "thing",
    "tool",
    "und",
    "weiter",
    "want",
    "will",
    "need",
    "brauche",
    "write",
    "was",
    "zum",
}

_FOLLOW_UP_REFERENCE_TOKENS = (
    "again",
    "auch",
    "continue",
    "dabei",
    "daran",
    "dazu",
    "das",
    "dem",
    "den",
    "der",
    "die",
    "dies",
    "dieses",
    "fuer das",
    "für das",
    "hier",
    "hinzu",
    "jetzt",
    "more",
    "naechst",
    "nächst",
    "noch",
    "nun",
    "weiter",
)

_ADDITIVE_REQUEST_TOKENS = (
    "add",
    "append",
    "auch",
    "dazu",
    "ergaenz",
    "ergänz",
    "extend",
    "fueg",
    "füge",
    "hinzu",
    "include",
    "noch",
    "plus",
    "ui",
    "frontend",
    "css",
    "html",
)

_PROBLEM_REPORT_TOKENS = (
    "bug",
    "broken",
    "crash",
    "error",
    "exception",
    "fail",
    "failed",
    "failing",
    "fails",
    "fehler",
    "geht nicht",
    "kaputt",
    "not working",
    "problem",
    "stacktrace",
    "terminal",
    "traceback",
    "warning",
)

_NEGATED_PROBLEM_CONSTRAINT_PATTERNS = (
    re.compile(
        r"\b(?:kein|keine|keinen|keinem|keiner|ohne|no|not)\s+"
        r"(?:[a-z0-9_äöüß-]+\s+){0,3}"
        r"(?:bug|bugs|broken|crash(?:es)?|error(?:s)?|exception(?:s)?|fail(?:ed|ing|s)?|fehler|kaputt[a-z0-9_äöüß-]*|problem(?:e|s)?)\b"
    ),
)

_DEBUG_REQUEST_TOKENS = (
    "beheb",
    "debug",
    "fix",
    "reparier",
    "repair",
    "resolve",
)

_VALIDATION_REQUEST_TOKENS = (
    "check",
    "lint",
    "prüf",
    "pruef",
    "run test",
    "test",
    "validier",
    "verify",
)

_EXPLANATION_REQUEST_TOKENS = (
    "analysiere",
    "erklär",
    "erklaer",
    "explain",
    "review",
    "warum",
    "was macht",
    "wie funktioniert",
    "wieso",
    "zusammen",
)

_UPDATE_REQUEST_TOKENS = (
    "aender",
    "änder",
    "aktualis",
    "anpass",
    "change",
    "clean up",
    "cleanup",
    "modify",
    "optimi",
    "refactor",
    "rewrite",
    "umbau",
    "update",
    "verbesser",
)

_CORRECTION_REQUEST_TOKENS = (
    "anders",
    "eigentlich",
    "falsch",
    "ich meinte",
    "nein",
    "rather",
    "rollback",
    "rueckgaengig",
    "rückgängig",
    "statt",
)

_WEB_SURFACE_EXTENSIONS = {".html", ".css", ".js", ".jsx", ".ts", ".tsx"}

_HARDENING_REQUEST_TOKENS = (
    "absichern",
    "harden",
    "robust",
    "sicher",
    "sicherer",
    "sicherheit",
    "secure",
    "security",
    "safer",
)

_BACKEND_SCOPE_TOKENS = (
    "api only",
    "backend",
    "backend only",
    "nur backend",
    "nur im backend",
    "only backend",
    "server only",
)

_FRONTEND_SCOPE_TOKENS = (
    "client only",
    "frontend only",
    "nur frontend",
    "nur im frontend",
    "only frontend",
    "ui only",
)

_BACKEND_SCOPE_PATTERNS = (
    re.compile(r"\b(?:backend|server|api)\s+only\b"),
    re.compile(r"\bonly\s+(?:the\s+)?(?:backend|server|api)\b"),
    re.compile(r"\b(?:nur|just|lediglich|ausschliesslich|ausschließlich)\s+(?:im\s+|den\s+|das\s+)?(?:backend|server)\b"),
)

_FRONTEND_SCOPE_PATTERNS = (
    re.compile(r"\b(?:frontend|client|ui)\s+only\b"),
    re.compile(r"\bonly\s+(?:the\s+)?(?:frontend|client|ui)\b"),
    re.compile(r"\b(?:nur|just|lediglich|ausschliesslich|ausschließlich)\s+(?:im\s+|das\s+|die\s+)?(?:frontend|client|ui)\b"),
)

_QUESTION_WORD_TOKENS = (
    "how",
    "wann",
    "warum",
    "was",
    "what",
    "when",
    "wer",
    "where",
    "wie",
    "wieso",
    "wo",
    "who",
    "why",
)

_SECOND_PERSON_TOKENS = (
    "dich",
    "dir",
    "du",
    "dein",
    "deine",
    "you",
    "your",
)

_REPO_CONTEXT_TOKENS = (
    "auth",
    "branch",
    "code",
    "komponente",
    "commit",
    "component",
    "config",
    "datei",
    "diff",
    "endpoint",
    "file",
    "function",
    "handler",
    "klasse",
    "class",
    "logik",
    "logic",
    "method",
    "modul",
    "module",
    "project",
    "projekt",
    "repo",
    "repository",
    "route",
    "selector",
    "source",
    "test",
    "workspace",
)

_REPO_ACTION_TOKENS = (
    "analysier",
    "analysiere",
    "check",
    "erklaer",
    "erklär",
    "explain",
    "find",
    "inspect",
    "lese",
    "locate",
    "look through",
    "read",
    "review",
    "search",
    "show",
    "such",
    "summarize",
    "where is",
    "wo ist",
    "wo steckt",
    "zusammen",
)

_DEICTIC_TASK_OBJECT_TOKENS = (
    "das",
    "dies",
    "dieses",
    "es",
    "it",
    "that",
    "this",
)

_DIRECT_TASK_REQUEST_TOKENS = (
    "analys",
    "analyz",
    "build",
    "check",
    "create",
    "debug",
    "delete",
    "find",
    "fix",
    "inspect",
    "inspiz",
    "les",
    "locate",
    "modify",
    "read",
    "remove",
    "repair",
    "review",
    "run",
    "search",
    "show",
    "start",
    "such",
    "update",
)

_GENERIC_EXECUTION_TOKENS = (
    "do",
    "mach",
    "make",
    "tu",
)

_AMBIGUOUS_CONTEXT_REFERENCE_TOKENS = (
    "da",
    "darin",
    "drin",
    "here",
    "hier",
    "inside",
    "there",
)

_CODE_PATH_RE = re.compile(
    r"([\w./-]+\.(?:py|js|ts|tsx|jsx|json|md|txt|html|css|sh|toml|ya?ml|go|rs|java|kt|rb|ini|cfg|conf|env|log|sql|xml|svg|csv))",
    flags=re.IGNORECASE,
)


PrimarySemanticIntent = Literal["create", "debug", "explain", "update"]


@dataclass(frozen=True, slots=True)
class ObviousRequestClassification:
    intent: PrimarySemanticIntent
    confidence: float
    requested_extension: str | None = None
    artifact_name_hint: str | None = None


@dataclass(frozen=True, slots=True)
class ConversationClassification:
    confidence: float


def normalize_text(text: str) -> str:
    normalized = str(text or "").strip().lower()
    for source, target in _COMPOUND_REPLACEMENTS:
        normalized = normalized.replace(source, target)
    return " ".join(normalized.split())


def _text_tokens(normalized: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9_äöüß]+", normalized) if token]


def _contains_term(
    normalized: str,
    tokens: list[str],
    terms: tuple[str, ...],
    *,
    prefix: bool = False,
) -> bool:
    for term in terms:
        if " " in term:
            if term in normalized:
                return True
            continue
        if prefix:
            if any(token.startswith(term) for token in tokens):
                return True
            continue
        if term in tokens:
            return True
    return False


def _looks_like_actionable_request(normalized: str, tokens: list[str]) -> bool:
    if _contains_term(normalized, tokens, _DIRECT_TASK_REQUEST_TOKENS, prefix=True):
        return True
    has_deictic_object = _contains_term(normalized, tokens, _DEICTIC_TASK_OBJECT_TOKENS)
    has_generic_execution = _contains_term(normalized, tokens, _GENERIC_EXECUTION_TOKENS, prefix=True)
    return has_deictic_object and has_generic_execution


def _contains_implementation_noun(normalized: str) -> bool:
    tokens = _text_tokens(normalized)
    return _contains_term(normalized, tokens, _IMPLEMENTATION_NOUNS)


def _contains_create_request_signal(normalized: str, tokens: list[str]) -> bool:
    for term in _CREATE_SIGNALS:
        if " " in term:
            if term in normalized:
                return True
            continue
        if term == "implement":
            if "implement" in tokens or any(token.startswith("implementier") for token in tokens):
                return True
            continue
        if any(token.startswith(term) for token in tokens):
            return True
    return False


def infer_requested_extension(*texts: str | None) -> str | None:
    normalized = " ".join(normalize_text(text or "") for text in texts if text)
    for extension, phrases in _EXPLICIT_EXTENSION_PHRASES:
        if any(phrase in normalized for phrase in phrases):
            return extension
    matched_extensions: list[str] = []
    for extension, hints in _EXTENSION_HINTS:
        if any(hint in normalized for hint in hints):
            matched_extensions.append(extension)
    if ".html" in matched_extensions and any(item in matched_extensions for item in {".js", ".css"}):
        return ".html"
    return matched_extensions[0] if matched_extensions else None


def infer_scope_tokens(*texts: str | None) -> list[str]:
    normalized = " ".join(normalize_text(text or "") for text in texts if text)
    tokens: list[str] = []
    for raw in re.split(r"[^a-z0-9_äöüß]+", normalized):
        token = raw.strip("_")
        if not token or token in tokens:
            continue
        if token in _GENERIC_SCOPE_STOPWORDS:
            continue
        if token in _GENERIC_MEDIUM_TOKENS:
            continue
        tokens.append(token)
    return tokens


def infer_artifact_name_hint(*texts: str | None) -> str | None:
    normalized = " ".join(normalize_text(text or "") for text in texts if text)
    if not normalized:
        return None
    if infer_requested_extension(normalized) is None and not _contains_implementation_noun(normalized):
        return None
    tokens = infer_scope_tokens(normalized)
    if not tokens:
        return None
    return " ".join(tokens[:4])


def is_clear_low_risk_build_request(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    tokens = _text_tokens(normalized)
    has_create_signal = _contains_create_request_signal(normalized, tokens) or _contains_term(
        normalized,
        tokens,
        _NEED_CREATE_SIGNALS,
        prefix=True,
    )
    update_signal = _contains_term(
        normalized,
        tokens,
        _UPDATE_REQUEST_TOKENS,
        prefix=True,
    )
    if update_signal and _is_scope_limiter_change_guardrail(normalized):
        update_signal = False
    has_change_signal = _contains_term(normalized, tokens, _DEBUG_REQUEST_TOKENS, prefix=True) or update_signal
    has_delivery_signal = _contains_implementation_noun(normalized)
    has_medium_signal = infer_requested_extension(normalized) is not None
    has_named_scope = bool(infer_scope_tokens(normalized))
    return (
        has_create_signal
        and not has_change_signal
        and (has_delivery_signal or has_medium_signal)
        and has_named_scope
    )


def _is_scope_limiter_change_guardrail(normalized: str) -> bool:
    text = str(normalized or "").strip()
    if not text:
        return False
    return any(
        marker in text
        for marker in (
            "sonst nichts",
            "nichts sonst",
            "nothing else",
            "anything else",
            "any other",
            "do not change anything else",
            "don't change anything else",
        )
    )


def has_follow_up_reference(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    tokens = _text_tokens(normalized)
    score = 0
    if _contains_term(normalized, tokens, _FOLLOW_UP_REFERENCE_TOKENS):
        score += 1
    if _contains_term(
        normalized,
        tokens,
        ("mach weiter", "build on", "continue with", "als naechstes", "als nächstes"),
    ):
        score += 1
    if _contains_term(normalized, tokens, _ADDITIVE_REQUEST_TOKENS, prefix=True):
        score += 1
    return score >= 2


def looks_like_additive_request(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    tokens = _text_tokens(normalized)
    has_add_signal = _contains_term(normalized, tokens, _ADDITIVE_REQUEST_TOKENS, prefix=True)
    has_artifact_signal = _contains_implementation_noun(normalized) or infer_requested_extension(normalized) is not None
    return has_add_signal and has_artifact_signal


def looks_like_problem_report(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    scrubbed = normalized
    for pattern in _NEGATED_PROBLEM_CONSTRAINT_PATTERNS:
        scrubbed = pattern.sub(" ", scrubbed)
    return any(token in scrubbed for token in _PROBLEM_REPORT_TOKENS)


def looks_like_debug_request(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    tokens = _text_tokens(normalized)
    has_fix_signal = _contains_term(normalized, tokens, _DEBUG_REQUEST_TOKENS, prefix=True)
    has_problem_signal = looks_like_problem_report(text)
    return has_fix_signal or has_problem_signal


def classify_conversation_request(text: str) -> ConversationClassification | None:
    normalized = normalize_text(text)
    if not normalized:
        return None
    tokens = _text_tokens(normalized)
    if not tokens:
        return None
    if _CODE_PATH_RE.search(str(text or "")):
        return None
    if (
        is_clear_low_risk_build_request(text)
        or looks_like_debug_request(text)
        or looks_like_update_request(text)
        or looks_like_validation_request(text)
        or looks_like_correction_request(text)
        or looks_like_hardening_request(text)
        or looks_like_scope_narrowing_request(text)
    ):
        return None
    nlp_prediction = classify_fallback_intent(text)
    if nlp_prediction.intent in {"create", "debug", "inspect", "plan", "search", "update"} and nlp_prediction.confidence >= 0.38:
        return None
    has_repo_context = _contains_term(normalized, tokens, _REPO_CONTEXT_TOKENS)
    has_repo_action = _contains_term(normalized, tokens, _REPO_ACTION_TOKENS, prefix=True)
    question_like = normalized.endswith("?") or (tokens and tokens[0] in _QUESTION_WORD_TOKENS)
    addresses_agent = _contains_term(normalized, tokens, _SECOND_PERSON_TOKENS)
    actionable_request = _looks_like_actionable_request(normalized, tokens)
    ambiguous_context_question = (
        question_like
        and nlp_prediction.intent != "conversation"
        and _contains_term(normalized, tokens, _AMBIGUOUS_CONTEXT_REFERENCE_TOKENS)
    )
    repo_grounded_explanation = (
        any(token in normalized for token in _EXPLANATION_REQUEST_TOKENS)
        and (has_repo_context or _contains_implementation_noun(normalized))
    )
    if (
        repo_grounded_explanation
        or (has_repo_context and has_repo_action)
        or actionable_request
        or ambiguous_context_question
    ):
        return None
    if nlp_prediction.intent == "conversation" and nlp_prediction.confidence >= 0.34:
        return ConversationClassification(confidence=max(nlp_prediction.confidence, 0.46))
    if question_like and addresses_agent:
        return ConversationClassification(confidence=max(nlp_prediction.confidence, 0.48))
    if question_like and not has_repo_context and nlp_prediction.intent in {"conversation", "unknown", "explain"}:
        return ConversationClassification(confidence=max(nlp_prediction.confidence, 0.44))
    return None


def looks_like_explanation_request(text: str) -> bool:
    normalized = normalize_text(text)
    return bool(normalized) and any(token in normalized for token in _EXPLANATION_REQUEST_TOKENS)


def looks_like_update_request(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    tokens = _text_tokens(normalized)
    if _contains_term(normalized, tokens, _UPDATE_REQUEST_TOKENS, prefix=True):
        return True
    existing_surface_markers = ("aus dem", "bestehend", "current", "existing", "in place")
    if any(marker in normalized for marker in existing_surface_markers) and _contains_implementation_noun(normalized):
        return True
    return False


def looks_like_validation_request(text: str) -> bool:
    normalized = normalize_text(text)
    return bool(normalized) and any(token in normalized for token in _VALIDATION_REQUEST_TOKENS)


def looks_like_correction_request(text: str) -> bool:
    normalized = normalize_text(text)
    return bool(normalized) and any(token in normalized for token in _CORRECTION_REQUEST_TOKENS)


def looks_like_hardening_request(text: str) -> bool:
    normalized = normalize_text(text)
    return bool(normalized) and any(token in normalized for token in _HARDENING_REQUEST_TOKENS)


def classify_obvious_request(text: str) -> ObviousRequestClassification | None:
    normalized = normalize_text(text)
    if not normalized:
        return None
    requested_extension = infer_requested_extension(text)
    artifact_name_hint = infer_artifact_name_hint(text)
    if looks_like_explanation_request(text):
        return ObviousRequestClassification(
            intent="explain",
            confidence=0.78,
            requested_extension=requested_extension,
            artifact_name_hint=artifact_name_hint,
        )
    if looks_like_debug_request(text):
        return ObviousRequestClassification(
            intent="debug",
            confidence=0.8,
            requested_extension=requested_extension,
            artifact_name_hint=artifact_name_hint,
        )
    if looks_like_update_request(text):
        return ObviousRequestClassification(
            intent="update",
            confidence=0.74,
            requested_extension=requested_extension,
            artifact_name_hint=artifact_name_hint,
        )
    nlp_prediction = classify_fallback_intent(text)
    if nlp_prediction.intent in {"create", "debug", "explain", "update"} and nlp_prediction.confidence >= 0.5:
        return ObviousRequestClassification(
            intent=nlp_prediction.intent,
            confidence=max(nlp_prediction.confidence, 0.7),
            requested_extension=requested_extension,
            artifact_name_hint=artifact_name_hint,
        )
    if is_clear_low_risk_build_request(text):
        return ObviousRequestClassification(
            intent="create",
            confidence=0.84 if requested_extension is not None else 0.78,
            requested_extension=requested_extension,
            artifact_name_hint=artifact_name_hint,
        )
    return None


def looks_like_scope_narrowing_request(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    return _has_explicit_scope_constraint(normalized, _BACKEND_SCOPE_PATTERNS) or _has_explicit_scope_constraint(
        normalized,
        _FRONTEND_SCOPE_PATTERNS,
    )


def extract_scope_constraints(text: str) -> list[str]:
    normalized = normalize_text(text)
    constraints: list[str] = []
    if _has_explicit_scope_constraint(normalized, _BACKEND_SCOPE_PATTERNS):
        constraints.append("Backend only.")
    if _has_explicit_scope_constraint(normalized, _FRONTEND_SCOPE_PATTERNS):
        constraints.append("Frontend only.")
    return constraints[:2]


def _has_explicit_scope_constraint(normalized: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(normalized) for pattern in patterns)


def is_structural_follow_up_request(
    text: str,
    *,
    previous_goal: str | None = None,
    artifact_paths: list[str] | None = None,
) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False

    artifact_paths = [str(path or "").strip() for path in artifact_paths or [] if str(path or "").strip()]
    context_text = " ".join([previous_goal or "", *artifact_paths])
    request_tokens = set(infer_scope_tokens(text))
    context_tokens = set(infer_scope_tokens(context_text))
    shared_scope = bool(request_tokens & context_tokens)

    artifact_extensions = {
        Path(path).suffix.lower()
        for path in artifact_paths
        if Path(path).suffix
    }
    requested_extension = infer_requested_extension(text)
    medium_compatible = (
        requested_extension is None
        or requested_extension in artifact_extensions
        or ({requested_extension, *artifact_extensions} <= _WEB_SURFACE_EXTENSIONS)
    )

    tokens = _text_tokens(normalized)
    explicit_reference = _contains_term(normalized, tokens, _FOLLOW_UP_REFERENCE_TOKENS)

    follow_up_signals = 0
    if has_follow_up_reference(text):
        follow_up_signals += 2
    elif explicit_reference and (previous_goal or artifact_paths):
        follow_up_signals += 1
    if shared_scope:
        follow_up_signals += 1
    if medium_compatible and (previous_goal or artifact_paths):
        follow_up_signals += 1
    if looks_like_additive_request(text):
        follow_up_signals += 1

    if is_clear_low_risk_build_request(text) and not has_follow_up_reference(text) and not shared_scope:
        return False
    return follow_up_signals >= 2
