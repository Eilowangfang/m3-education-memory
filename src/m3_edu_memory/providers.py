from __future__ import annotations

import json
from pathlib import Path

from .vlm import OpenAICompatibleVisionClient, VisionClient
from .image_preprocessing import ImagePreparationPolicy


def load_provider_profiles(path: str | Path) -> dict[str, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    profiles = payload.get("profiles") if isinstance(payload, dict) else None
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Provider config must contain a non-empty profiles object")
    for name, profile in profiles.items():
        if not isinstance(profile, dict):
            raise ValueError(f"Provider profile {name!r} must be an object")
        if "api_key" in profile:
            raise ValueError(
                f"Provider profile {name!r} contains api_key; use api_key_env instead"
            )
        for field in ("base_url", "model", "api_key_env"):
            if not isinstance(profile.get(field), str) or not profile[field].strip():
                raise ValueError(f"Provider profile {name!r} requires {field}")
        if profile.get("image_detail", "high") not in {"low", "high", "xhigh"}:
            raise ValueError(
                f"Provider profile {name!r} image_detail must be low, high, or xhigh"
            )
        if not isinstance(profile.get("request_options", {}), dict):
            raise ValueError(
                f"Provider profile {name!r} request_options must be an object"
            )
    return profiles


def create_vision_client(
    config_path: str | Path,
    *,
    profile_name: str,
) -> VisionClient:
    profiles = load_provider_profiles(config_path)
    if profile_name not in profiles:
        choices = ", ".join(sorted(profiles))
        raise KeyError(f"Unknown provider profile {profile_name!r}; available: {choices}")
    profile = profiles[profile_name]
    protocol = profile.get("protocol", "openai-compatible")
    if protocol != "openai-compatible":
        raise ValueError(f"Unsupported provider protocol: {protocol!r}")
    image_config = profile.get("image_preprocessing", {})
    if not isinstance(image_config, dict):
        raise ValueError("image_preprocessing must be an object")
    return OpenAICompatibleVisionClient(
        base_url=profile["base_url"],
        model=profile["model"],
        api_key_env=profile["api_key_env"],
        timeout=int(profile.get("timeout", 180)),
        use_json_mode=bool(profile.get("use_json_mode", True)),
        max_retries=int(profile.get("max_retries", 3)),
        provider=str(profile.get("provider", profile_name)),
        image_detail=str(profile.get("image_detail", "high")),
        request_options=dict(profile.get("request_options", {})),
        image_policy=ImagePreparationPolicy(
            enabled=bool(image_config.get("enabled", True)),
            max_long_edge=int(image_config.get("max_long_edge", 3072)),
            max_pixels=int(image_config.get("max_pixels", 4_014_080)),
            max_bytes=int(image_config.get("max_bytes", 4_000_000)),
            jpeg_quality=int(image_config.get("jpeg_quality", 92)),
            min_jpeg_quality=int(image_config.get("min_jpeg_quality", 72)),
        ),
    )
