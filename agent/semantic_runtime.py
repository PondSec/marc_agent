from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Literal


SemanticResolution = Literal[
    "full_model",
    "reserve_model",
    "reduced_model",
    "minimal_inference",
    "blocked",
]


def semantic_model_candidates(preferred_model: str | None, config: object | None) -> list[str]:
    primary = str(preferred_model or "").strip()
    explicit_primary = bool(primary)
    discovered_pool = _configured_candidate_pool(config)
    configured_pool = _configured_model_pool(config)
    if not primary:
        for candidate in [*discovered_pool, *configured_pool]:
            primary = str(candidate or "").strip()
            if primary:
                break
    if not primary:
        return []
    if discovered_pool:
        return _rank_model_candidates(
            primary,
            discovered_pool,
            allow_larger_if_needed=True,
        )
    return _rank_model_candidates(
        primary,
        configured_pool,
        allow_larger_if_needed=not explicit_primary,
    )


def rank_semantic_model_candidates(
    primary_model: str | None,
    candidate_pool: Iterable[str] | None,
    *,
    allow_larger_if_needed: bool = True,
) -> list[str]:
    primary = str(primary_model or "").strip()
    pool = [str(item or "").strip() for item in candidate_pool or () if str(item or "").strip()]
    if not primary:
        if not pool:
            return []
        primary = pool[0]
    return _rank_model_candidates(
        primary,
        pool,
        allow_larger_if_needed=allow_larger_if_needed,
    )


def _configured_model_pool(config: object | None) -> list[str]:
    candidates: list[str] = []
    for raw in (
        getattr(config, "router_model_name", None) if config is not None else None,
        getattr(config, "model_name", None) if config is not None else None,
    ):
        text = str(raw or "").strip()
        if text and text not in candidates:
            candidates.append(text)
    return candidates


def _configured_candidate_pool(config: object | None) -> list[str]:
    raw_candidates = getattr(config, "model_candidates", ()) if config is not None else ()
    if isinstance(raw_candidates, str):
        raw_candidates = [raw_candidates]
    candidates: list[str] = []
    for raw in raw_candidates or ():
        text = str(raw or "").strip()
        if text and text not in candidates:
            candidates.append(text)
    return candidates


def _rank_model_candidates(
    primary: str,
    candidate_pool: list[str],
    *,
    allow_larger_if_needed: bool,
) -> list[str]:
    candidates = [primary]
    alternatives: list[str] = []
    for raw in candidate_pool:
        text = str(raw or "").strip()
        if text and text != primary and text not in alternatives:
            alternatives.append(text)

    primary_size = _estimated_model_size_billions(primary)
    primary_family = _model_family(primary)
    ranked_smaller_or_equal: list[tuple[int, float, float, str]] = []
    ranked_larger: list[tuple[int, float, float, str]] = []
    unknown_size: list[tuple[int, str]] = []
    for candidate in alternatives:
        candidate_size = _estimated_model_size_billions(candidate)
        family_penalty = 0 if primary_family and _model_family(candidate) == primary_family else 1
        if candidate_size is None:
            unknown_size.append((family_penalty, candidate))
            continue
        if primary_size is not None and candidate_size > primary_size:
            if not allow_larger_if_needed:
                continue
            ranked_larger.append((family_penalty, candidate_size - primary_size, candidate_size, candidate))
            continue
        distance = abs(candidate_size - primary_size) if primary_size is not None else 0.0
        ranked_smaller_or_equal.append((family_penalty, distance, -candidate_size, candidate))

    candidates.extend(candidate for _, _, _, candidate in sorted(ranked_smaller_or_equal))
    candidates.extend(candidate for _, _, _, candidate in sorted(ranked_larger))
    candidates.extend(candidate for _, candidate in sorted(unknown_size))
    return candidates


def _model_family(model_name: str) -> str:
    return str(model_name or "").strip().lower().partition(":")[0]


def _estimated_model_size_billions(model_name: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)b\b", str(model_name or "").lower())
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def semantic_resolution_from_attempt(
    *,
    capability_tier: str,
    prompt_variant: str,
    model_identifier: str | None,
    primary_model: str | None,
) -> SemanticResolution:
    if capability_tier == "tier_a" and prompt_variant == "full":
        return "full_model"
    if capability_tier == "tier_b" and prompt_variant == "full":
        return "reserve_model"
    if capability_tier in {"tier_b", "tier_c"}:
        return "reduced_model"
    if capability_tier == "tier_d":
        return "minimal_inference"
    if capability_tier == "tier_e":
        return "blocked"
    if prompt_variant in {"compact", "minimal", "repair"}:
        return "reduced_model"
    if model_identifier and primary_model and model_identifier != primary_model:
        return "reserve_model"
    return "full_model"


def availability_recovery_model(
    primary_model: str | None,
    candidate_models: str | Iterable[str] | None,
) -> str | None:
    primary = str(primary_model or "").strip()
    candidates = _normalized_recovery_candidates(candidate_models, primary)
    if not candidates:
        return None
    if not primary:
        return candidates[0]
    primary_size = _estimated_model_size_billions(primary)
    primary_family = _model_family(primary)
    ranked: list[tuple[tuple[int, int, float, float, int], str]] = []
    for index, candidate in enumerate(candidates):
        candidate_size = _estimated_model_size_billions(candidate)
        candidate_family = _model_family(candidate)
        family_penalty = 0 if primary_family and candidate_family == primary_family else 1
        if primary_size is None or candidate_size is None:
            ranked.append(((2, family_penalty, 0.0, 0.0, index), candidate))
            continue
        if candidate_size <= primary_size:
            ranked.append(
                (
                    (
                        0,
                        family_penalty,
                        abs(primary_size - candidate_size),
                        -candidate_size,
                        index,
                    ),
                    candidate,
                )
            )
            continue
        growth_ratio = candidate_size / primary_size if primary_size > 0 else float("inf")
        size_delta = candidate_size - primary_size
        if growth_ratio <= 1.25 and size_delta <= 2.0:
            ranked.append(((1, family_penalty, size_delta, candidate_size, index), candidate))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1]


def _normalized_recovery_candidates(
    candidate_models: str | Iterable[str] | None,
    primary_model: str,
) -> list[str]:
    if candidate_models is None:
        return []
    if isinstance(candidate_models, str):
        raw_candidates = [candidate_models]
    else:
        raw_candidates = list(candidate_models)
    normalized: list[str] = []
    for raw in raw_candidates:
        text = str(raw or "").strip()
        if not text or text == primary_model or text in normalized:
            continue
        normalized.append(text)
    return normalized


def secondary_semantics_limited(resolution: SemanticResolution) -> bool:
    return resolution in {"reduced_model", "minimal_inference", "blocked"}


def annotate_semantic_record(
    record: dict[str, Any],
    *,
    semantic_resolution: SemanticResolution,
) -> dict[str, Any]:
    record["semantic_resolution"] = semantic_resolution
    record["secondary_semantics_limited"] = secondary_semantics_limited(semantic_resolution)
    return record
