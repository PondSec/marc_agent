from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from math import log, sqrt
import re
from typing import Literal


FallbackIntent = Literal["conversation", "create", "debug", "explain", "inspect", "plan", "search", "unknown", "update"]

_SEED_EXAMPLES: dict[FallbackIntent, tuple[str, ...]] = {
    "conversation": (
        "hallo wie geht es dir",
        "wer bist du",
        "wie heisst du",
        "was kannst du",
        "tell me about yourself",
        "what is a hamburger",
        "can you explain what a hamburger is",
        "weißt du was ein hamburger ist",
    ),
    "create": (
        "ich brauche ein snake spiel in html",
        "schreib mir ein asteroids spiel in html",
        "baue eine landing page fuer mein produkt",
        "erstelle ein kleines tool in python",
        "make a small todo app in javascript",
        "build a landing page with html and css",
        "create a simple rest api with login",
        "program a small game in python",
    ),
    "debug": (
        "fix den fehler in app/auth.py",
        "beheb den bug in der upload route",
        "warum crasht das hier",
        "debug this failing test",
        "fix the broken login flow",
        "the app crashes on startup please fix it",
        "diagnose why this command fails",
        "repair the bug in this file",
    ),
    "explain": (
        "erklaer mir warum das fehlschlaegt",
        "erklär mir wie das funktioniert",
        "erkläre mir die auth logik",
        "explain why this test is failing",
        "explain how this module works",
        "summarize what this code does",
        "why is the agent doing this",
        "review and explain this flow",
    ),
    "inspect": (
        "schau dir die datei an",
        "analysiere das repo",
        "inspect the current implementation",
        "look through the codebase",
        "check the file contents",
        "read the relevant module",
    ),
    "plan": (
        "mach mir einen plan",
        "wie wuerdest du das angehen",
        "wie würdest du das umbauen",
        "plan the implementation",
        "give me a roadmap for this change",
        "outline the next steps",
    ),
    "search": (
        "finde die stelle mit dem token check",
        "such die auth logik im repo",
        "wo ist der upload handler",
        "find where this is implemented",
        "search for the config loader",
        "locate the login component",
    ),
    "update": (
        "aendere die landing page",
        "passe die bestehende route an",
        "mach das modul sauberer",
        "update the existing handler",
        "modify the current component",
        "refactor this module",
        "change the current implementation",
        "adjust the file in place",
    ),
    "unknown": (
        "mach das besser",
        "irgendwas mit der datei da",
        "??",
        "hilfe",
        "do something with it",
    ),
}


@dataclass(frozen=True, slots=True)
class IntentPrediction:
    intent: FallbackIntent
    confidence: float
    similarity: float
    margin: float


def _normalize(text: str) -> str:
    lowered = str(text or "").strip().lower()
    replacements = (
        ("to do", "todo"),
        ("to-do", "todo"),
    )
    for source, target in replacements:
        lowered = lowered.replace(source, target)
    return " ".join(lowered.split())


def _word_tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9_äöüß]+", text) if token]


def _ngrams(text: str) -> Counter[str]:
    normalized = _normalize(text)
    features: Counter[str] = Counter()
    tokens = _word_tokens(normalized)
    for token in tokens:
        features[f"w:{token}"] += 1
    for left, right in zip(tokens, tokens[1:]):
        features[f"b:{left}_{right}"] += 1
    padded = f"  {normalized}  "
    for size in (3, 4, 5):
        for index in range(max(len(padded) - size + 1, 0)):
            features[f"c:{padded[index:index + size]}"] += 1
    return features


def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(weight * right.get(feature, 0.0) for feature, weight in left.items())
    left_norm = sqrt(sum(weight * weight for weight in left.values()))
    right_norm = sqrt(sum(weight * weight for weight in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


class _IntentVectorModel:
    def __init__(self, seed_examples: dict[FallbackIntent, tuple[str, ...]]):
        self.idf = self._build_idf(seed_examples)
        self.centroids = {
            intent: self._centroid(samples)
            for intent, samples in seed_examples.items()
        }

    def _build_idf(self, seed_examples: dict[FallbackIntent, tuple[str, ...]]) -> dict[str, float]:
        document_count = 0
        doc_frequency: defaultdict[str, int] = defaultdict(int)
        for samples in seed_examples.values():
            for sample in samples:
                document_count += 1
                for feature in _ngrams(sample):
                    doc_frequency[feature] += 1
        return {
            feature: log((1 + document_count) / (1 + frequency)) + 1.0
            for feature, frequency in doc_frequency.items()
        }

    def _vectorize(self, text: str) -> dict[str, float]:
        counts = _ngrams(text)
        if not counts:
            return {}
        total = float(sum(counts.values()))
        return {
            feature: (count / total) * self.idf.get(feature, 1.0)
            for feature, count in counts.items()
        }

    def _centroid(self, samples: tuple[str, ...]) -> dict[str, float]:
        accumulator: defaultdict[str, float] = defaultdict(float)
        for sample in samples:
            for feature, weight in self._vectorize(sample).items():
                accumulator[feature] += weight
        count = max(len(samples), 1)
        return {feature: weight / count for feature, weight in accumulator.items()}

    def predict(self, text: str) -> IntentPrediction:
        vector = self._vectorize(text)
        ranked = sorted(
            (
                (intent, _cosine_similarity(vector, centroid))
                for intent, centroid in self.centroids.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        top_intent, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = max(top_score - second_score, 0.0)
        confidence = min(max((top_score * 0.72) + (margin * 0.58), 0.0), 1.0)
        if top_score < 0.18 or confidence < 0.32:
            return IntentPrediction(intent="unknown", confidence=confidence, similarity=top_score, margin=margin)
        return IntentPrediction(intent=top_intent, confidence=confidence, similarity=top_score, margin=margin)


@lru_cache(maxsize=1)
def _model() -> _IntentVectorModel:
    return _IntentVectorModel(_SEED_EXAMPLES)


def classify_fallback_intent(text: str) -> IntentPrediction:
    return _model().predict(text)
