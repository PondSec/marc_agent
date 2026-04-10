from __future__ import annotations

import re
from dataclasses import fields, is_dataclass, replace
from typing import Callable

from config.settings import AppConfig
from llm.ollama_client import OllamaClient


def fetch_installed_ollama_models(config: AppConfig) -> list[dict]:
    return OllamaClient(config).list_models_safe()


def with_available_models(
    config: AppConfig,
    *,
    fetch_installed_models: Callable[[AppConfig], list[dict]] | None = None,
) -> AppConfig:
    fetcher = fetch_installed_models or fetch_installed_ollama_models
    installed_models = fetcher(config)
    installed_names = [str(item.get("name")) for item in installed_models if item.get("name")]
    if not installed_names:
        router_model_name = config.router_model_name or config.model_name
        return replace_config_compatible(
            config,
            router_model_name=router_model_name,
            model_candidates=tuple(
                allowed_model_candidates(
                    config,
                    primary_model=str(config.model_name or "").strip(),
                    router_model=router_model_name,
                    installed_names=(),
                )
            ),
        )

    primary_model = select_primary_model(
        preferred_model=config.model_name,
        installed_names=installed_names,
    )
    router_model = select_router_model(
        preferred_router=config.router_model_name,
        installed_names=installed_names,
        primary_model=primary_model,
    )
    return replace_config_compatible(
        config,
        model_name=primary_model,
        router_model_name=router_model,
        model_candidates=tuple(
            allowed_model_candidates(
                config,
                primary_model=primary_model,
                router_model=router_model,
                installed_names=installed_names,
            )
        ),
    )


def replace_config_compatible(config: AppConfig, **changes: object) -> AppConfig:
    if not is_dataclass(config):
        return config
    allowed_fields = {field.name for field in fields(config)}
    compatible_changes = {key: value for key, value in changes.items() if key in allowed_fields}
    return replace(config, **compatible_changes)


def configured_model_candidates(config: object) -> tuple[str, ...]:
    raw_candidates = getattr(config, "model_candidates", ()) or ()
    return tuple(
        str(raw or "").strip()
        for raw in raw_candidates
        if str(raw or "").strip()
    )


def allowed_model_candidates(
    config: object,
    *,
    primary_model: str,
    router_model: str | None,
    installed_names: list[str] | tuple[str, ...],
) -> list[str]:
    installed_lookup = {str(name or "").strip() for name in installed_names if str(name or "").strip()}
    configured_candidates = list(configured_model_candidates(config))
    candidates: list[str] = []

    def add(name: str | None, *, require_install: bool = True) -> None:
        text = str(name or "").strip()
        if not text or text in candidates:
            return
        if require_install and installed_lookup and text not in installed_lookup:
            return
        candidates.append(text)

    add(primary_model, require_install=False)
    add(router_model)
    for candidate in configured_candidates:
        add(candidate)

    if candidates:
        return candidates
    if primary_model:
        return [primary_model]
    if router_model:
        return [router_model]
    return []


def select_primary_model(
    *,
    preferred_model: str,
    installed_names: list[str],
) -> str:
    preferred = str(preferred_model or "").strip()
    if not installed_names:
        return preferred
    if preferred and preferred in installed_names:
        return preferred

    candidates = list(installed_names)
    if "coder" in preferred.lower():
        coder_candidates = [name for name in candidates if "coder" in name.lower()]
        if coder_candidates:
            candidates = coder_candidates

    preferred_size = model_size_hint(preferred)
    if preferred_size is not None:
        not_larger: list[str] = []
        larger_candidates_seen = False
        for name in candidates:
            candidate_size = model_size_hint(name)
            if candidate_size is None:
                continue
            if candidate_size <= preferred_size:
                not_larger.append(name)
                continue
            larger_candidates_seen = True
        if not_larger:
            candidates = not_larger
        elif larger_candidates_seen and preferred:
            return preferred

    ranked = sorted(
        candidates,
        key=lambda name: primary_model_fallback_rank(
            preferred_model=preferred,
            candidate=name,
        ),
    )
    return ranked[0]


def primary_model_fallback_rank(
    *,
    preferred_model: str,
    candidate: str,
) -> tuple[int, float, int, tuple[int, str]]:
    preferred = str(preferred_model or "").strip().lower()
    name = str(candidate or "").strip().lower()
    preferred_base = preferred.partition(":")[0]
    candidate_base = name.partition(":")[0]
    preferred_size = model_size_hint(preferred_model)
    candidate_size = model_size_hint(candidate)

    base_penalty = 0 if preferred_base and candidate_base == preferred_base else 1

    if preferred_size is None or candidate_size is None:
        size_distance = 999.0
        larger_penalty = 0
    else:
        size_distance = abs(candidate_size - preferred_size)
        larger_penalty = 1 if candidate_size > preferred_size else 0

    return (
        base_penalty,
        size_distance,
        larger_penalty,
        model_preference_rank(candidate),
    )


def model_size_hint(name: str) -> float | None:
    lowered = str(name or "").strip().lower()
    match = re.search(r"(\d+(?:\.\d+)?)b\b", lowered)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def model_preference_rank(name: str) -> tuple[int, str]:
    lowered = name.lower()
    if "coder" in lowered and "qwen3" in lowered:
        return (0, lowered)
    if "coder" in lowered:
        return (1, lowered)
    if "qwen3" in lowered:
        return (2, lowered)
    return (3, lowered)


def select_router_model(
    *,
    preferred_router: str | None,
    installed_names: list[str],
    primary_model: str,
) -> str:
    if preferred_router and preferred_router in installed_names:
        return preferred_router
    candidates = list(installed_names)
    if not candidates:
        return preferred_router or primary_model

    primary_size = model_size_hint(primary_model)
    if primary_size is not None:
        not_larger = [
            name
            for name in candidates
            if (candidate_size := model_size_hint(name)) is not None and candidate_size <= primary_size
        ]
        if not_larger:
            candidates = not_larger
        else:
            return preferred_router or primary_model

    coder_candidates = [name for name in candidates if "coder" in name.lower()]
    if coder_candidates:
        candidates = coder_candidates

    coder_ranked = sorted(
        candidates,
        key=router_model_preference_rank,
    )
    if coder_ranked:
        return coder_ranked[0]
    return preferred_router or primary_model


def router_model_preference_rank(name: str) -> tuple[int, float, str]:
    lowered = name.lower()
    family_penalty = model_preference_rank(name)[0]
    size = model_size_hint(name)
    return (
        family_penalty,
        size if size is not None else 999.0,
        lowered,
    )
