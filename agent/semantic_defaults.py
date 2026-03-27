from __future__ import annotations

import re
from pathlib import Path


_COMPOUND_REPLACEMENTS = (
    ("to do", "todo"),
    ("to-do", "todo"),
)

_CREATE_SIGNALS = (
    "bau",
    "build",
    "create",
    "erstell",
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
    "fehler",
    "geht nicht",
    "kaputt",
    "problem",
    "stacktrace",
    "terminal",
    "traceback",
    "warning",
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
    "frontend",
    "frontend only",
    "nur frontend",
    "nur im frontend",
    "only frontend",
    "ui only",
)


def normalize_text(text: str) -> str:
    normalized = str(text or "").strip().lower()
    for source, target in _COMPOUND_REPLACEMENTS:
        normalized = normalized.replace(source, target)
    return " ".join(normalized.split())


def infer_requested_extension(*texts: str | None) -> str | None:
    normalized = " ".join(normalize_text(text or "") for text in texts if text)
    for extension, hints in _EXTENSION_HINTS:
        if any(hint in normalized for hint in hints):
            return extension
    return None


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
    tokens = infer_scope_tokens(*texts)
    if not tokens:
        return None
    return " ".join(tokens[:4])


def is_clear_low_risk_build_request(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    has_create_signal = any(signal in normalized for signal in _CREATE_SIGNALS)
    has_delivery_signal = any(noun in normalized for noun in _IMPLEMENTATION_NOUNS)
    has_medium_signal = infer_requested_extension(normalized) is not None
    has_named_scope = bool(infer_scope_tokens(normalized))
    return has_create_signal and (has_delivery_signal or has_medium_signal) and has_named_scope


def has_follow_up_reference(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    score = 0
    if any(token in normalized for token in _FOLLOW_UP_REFERENCE_TOKENS):
        score += 1
    if any(token in normalized for token in ("mach weiter", "build on", "continue with", "als naechstes", "als nächstes")):
        score += 1
    if any(token in normalized for token in _ADDITIVE_REQUEST_TOKENS):
        score += 1
    return score >= 2


def looks_like_additive_request(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    has_add_signal = any(token in normalized for token in _ADDITIVE_REQUEST_TOKENS)
    has_artifact_signal = any(token in normalized for token in _IMPLEMENTATION_NOUNS) or infer_requested_extension(normalized) is not None
    return has_add_signal and has_artifact_signal


def looks_like_problem_report(text: str) -> bool:
    normalized = normalize_text(text)
    return bool(normalized) and any(token in normalized for token in _PROBLEM_REPORT_TOKENS)


def looks_like_validation_request(text: str) -> bool:
    normalized = normalize_text(text)
    return bool(normalized) and any(token in normalized for token in _VALIDATION_REQUEST_TOKENS)


def looks_like_correction_request(text: str) -> bool:
    normalized = normalize_text(text)
    return bool(normalized) and any(token in normalized for token in _CORRECTION_REQUEST_TOKENS)


def looks_like_hardening_request(text: str) -> bool:
    normalized = normalize_text(text)
    return bool(normalized) and any(token in normalized for token in _HARDENING_REQUEST_TOKENS)


def looks_like_scope_narrowing_request(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    narrowing_markers = ("nur", "only", "just", "lediglich", "ausschliesslich", "ausschließlich")
    has_narrowing = any(marker in normalized for marker in narrowing_markers)
    return has_narrowing and (
        any(token in normalized for token in _BACKEND_SCOPE_TOKENS)
        or any(token in normalized for token in _FRONTEND_SCOPE_TOKENS)
    )


def extract_scope_constraints(text: str) -> list[str]:
    normalized = normalize_text(text)
    constraints: list[str] = []
    if any(token in normalized for token in _BACKEND_SCOPE_TOKENS):
        constraints.append("Backend only.")
    if any(token in normalized for token in _FRONTEND_SCOPE_TOKENS):
        constraints.append("Frontend only.")
    return constraints[:2]


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

    follow_up_signals = 0
    if has_follow_up_reference(text):
        follow_up_signals += 2
    if shared_scope:
        follow_up_signals += 1
    if medium_compatible and (previous_goal or artifact_paths):
        follow_up_signals += 1
    if looks_like_additive_request(text):
        follow_up_signals += 1

    if is_clear_low_risk_build_request(text) and not has_follow_up_reference(text) and not shared_scope:
        return False
    return follow_up_signals >= 2
