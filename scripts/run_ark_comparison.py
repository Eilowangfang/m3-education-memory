from __future__ import annotations

import argparse
import json
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from m3_edu_memory.database import connect
from m3_edu_memory.providers import load_provider_profiles

from run_e2e_validation import run


DEFAULT_PROFILES = (
    "volcengine-seed-2.1-pro",
    "volcengine-seed-2.1-turbo",
)


def profile_args(args, profile_name: str, profile: dict) -> Namespace:
    prep = profile.get("image_preprocessing", {})
    return Namespace(
        source_db=args.source_db,
        source_attempt_id=args.source_attempt_id,
        output_root=args.output_root,
        base_url=profile["base_url"],
        model=profile["model"],
        api_key_env=profile.get("api_key_env", "ARK_API_KEY"),
        provider=profile.get("provider", "volcengine-ark"),
        run_prefix=profile_name,
        timeout=int(profile.get("timeout", 240)),
        image_detail=profile.get("image_detail", "high"),
        image_max_long_edge=int(prep.get("max_long_edge", 4096)),
        image_max_pixels=int(prep.get("max_pixels", 9_031_680)),
        image_max_bytes=int(prep.get("max_bytes", 8_000_000)),
        image_jpeg_quality=int(prep.get("jpeg_quality", 94)),
        image_min_jpeg_quality=int(prep.get("min_jpeg_quality", 80)),
        request_options=dict(profile.get("request_options", {})),
    )


def load_truth(db_path: str, attempt_id: str) -> dict | None:
    connection = connect(db_path)
    row = connection.execute(
        """SELECT has_error,orig_a,pert_a,pert_reasoning,reference_error_type
           FROM benchmark_truth WHERE attempt_id=?""",
        (attempt_id,),
    ).fetchone()
    connection.close()
    return None if row is None else dict(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-db", default="data/education_memory.db")
    parser.add_argument("--source-attempt-id", default="fermat-00000-00010-000027")
    parser.add_argument("--config", default="config/vlm_providers.json")
    parser.add_argument("--output-root", default="outputs/ark-comparison")
    parser.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES))
    args = parser.parse_args()

    profiles = load_provider_profiles(args.config)
    comparison_root = Path(args.output_root)
    comparison_root.mkdir(parents=True, exist_ok=True)
    results = []
    for profile_name in args.profiles:
        if profile_name not in profiles:
            raise KeyError(f"Unknown provider profile: {profile_name}")
        try:
            result = run(profile_args(args, profile_name, profiles[profile_name]))
            results.append({"profile": profile_name, "status": "passed", **result})
        except Exception as exc:
            results.append(
                {
                    "profile": profile_name,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_attempt_id": args.source_attempt_id,
        "ground_truth": load_truth(args.source_db, args.source_attempt_id),
        "results": results,
    }
    summary_path = comparison_root / (
        "comparison-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary["summary_path"] = str(summary_path.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(item["status"] == "passed" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
