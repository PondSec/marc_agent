from __future__ import annotations

import re
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
    if not primary:
        for raw in (
            getattr(config, "router_model_name", None) if config is not None else None,
            getattr(config, "model_name", None) if config is not None else None,
        ):
            text = str(raw or "").strip()
            if text:
                primary = text
                break
    if not primary:
        return []

    candidates = [primary]
    alternatives: list[str] = []
    for raw in (
        getattr(config, "router_model_name", None) if config is not None else None,
        getattr(config, "model_name", None) if config is not None else None,
    ):
        text = str(raw or "").strip()
        if text and text != primary and text not in alternatives:
            alternatives.append(text)

    primary_size = _estimated_model_size_billions(primary)
    ranked: list[tuple[float, str]] = []
    unknown_size: list[str] = []
    for candidate in alternatives:
        candidate_size = _estimated_model_size_billions(candidate)
        if (
            primary_size is not None
            and candidate_size is not None
            and candidate_size > primary_size
        ):
            continue
        if candidate_size is None:
            unknown_size.append(candidate)
            continue
        ranked.append((candidate_size, candidate))

    candidates.extend(candidate for _, candidate in sorted(ranked, key=lambda item: item[0]))
    if primary_size is None:
        candidates.extend(unknown_size)
    return candidates


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
