from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

from m3_edu_memory.providers import load_provider_profiles


def test_profile(config_path: str, profile_name: str) -> dict:
    profile = load_provider_profiles(config_path)[profile_name]
    key = os.environ.get(profile["api_key_env"])
    if not key:
        raise RuntimeError(f"Environment variable {profile['api_key_env']} is missing")
    payload = json.dumps(
        {
            "model": profile["model"],
            "messages": [{"role": "user", "content": "Reply with OK only."}],
            "max_tokens": 8,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        profile["base_url"].rstrip("/") + "/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request, timeout=int(profile.get("timeout", 180))
        ) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{profile_name} returned HTTP {exc.code}: {detail[:1000]}"
        ) from exc
    return {
        "profile": profile_name,
        "configured_model": profile["model"],
        "response_model": result.get("model"),
        "status": "ok",
        "usage": result.get("usage", {}),
        "finish_reason": (result.get("choices") or [{}])[0].get("finish_reason"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/vlm_providers.json")
    parser.add_argument("profiles", nargs="+")
    args = parser.parse_args()
    print(
        json.dumps(
            [test_profile(args.config, profile) for profile in args.profiles],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
