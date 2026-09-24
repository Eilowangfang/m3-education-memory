from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROFILES = (
    ("volcengine-seed-2.1-turbo", "doubao-seed-2-1-turbo-260628"),
    ("volcengine-seed-2.1-pro", "doubao-seed-2-1-pro-260915"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def run_command(
    command: list[str], log, *, allowed_returncodes: tuple[int, ...] = (0,)
) -> int:
    log.write("\n$ " + subprocess.list2cmdline(command) + "\n")
    log.flush()
    completed = subprocess.run(
        command, stdout=log, stderr=subprocess.STDOUT, check=False
    )
    if completed.returncode not in allowed_returncodes:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/seed_eval_100_20260922.db")
    parser.add_argument("--suite", default="fermat-vlm-pilot-100")
    parser.add_argument("--config", default="config/vlm_providers.json")
    parser.add_argument("--output", default="outputs/seed-fixed-evaluation")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    if not os.environ.get("ARK_API_KEY"):
        raise RuntimeError("ARK_API_KEY is not configured in this process")

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "status.json"
    log_path = root / "evaluation.log"
    status = {
        "status": "running",
        "stage": "starting",
        "pid": os.getpid(),
        "database": str(Path(args.db).resolve()),
        "suite": args.suite,
        "profiles": [item[0] for item in PROFILES],
        "max_workers": args.max_workers,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "completed_profiles": [],
    }
    write_status(status_path, status)

    base = [sys.executable, "-u", "-m", "m3_edu_memory.cli"]
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            for profile, model in PROFILES:
                status["stage"] = f"diagnose:{profile}"
                status["updated_at"] = utc_now()
                write_status(status_path, status)
                for pass_number in (1, 2):
                    status["stage"] = f"diagnose:{profile}:pass-{pass_number}"
                    status["updated_at"] = utc_now()
                    write_status(status_path, status)
                    returncode = run_command(
                        base
                        + [
                            "diagnose",
                            "--db", args.db,
                            "--evaluation-suite", args.suite,
                            "--provider-config", args.config,
                            "--provider-profile", profile,
                            "--only-unprocessed",
                            "--max-workers", str(args.max_workers),
                            "--result-output", str(
                                root / f"{profile}-batch-pass-{pass_number}.json"
                            ),
                            "--audit-output-dir", str(root / profile / "audits"),
                        ],
                        log,
                        allowed_returncodes=(0, 1),
                    )
                    if returncode == 0:
                        break
                status["stage"] = f"evaluate:{profile}"
                status["updated_at"] = utc_now()
                write_status(status_path, status)
                run_command(
                    base
                    + [
                        "evaluate",
                        "--db", args.db,
                        "--model", model,
                        "--provider", "volcengine-ark",
                        "--suite", args.suite,
                        "--persist",
                    ],
                    log,
                )
                status["completed_profiles"].append(profile)
                status["updated_at"] = utc_now()
                write_status(status_path, status)

            status["stage"] = "compare"
            status["updated_at"] = utc_now()
            write_status(status_path, status)
            run_command(
                base
                + [
                    "compare-eval-suite",
                    "--db", args.db,
                    "--suite", args.suite,
                ],
                log,
            )
        status["status"] = "completed"
        status["stage"] = "completed"
        status["completed_at"] = utc_now()
        status["updated_at"] = status["completed_at"]
        write_status(status_path, status)
        return 0
    except Exception as exc:
        status["status"] = "failed"
        status["error_type"] = type(exc).__name__
        status["error"] = str(exc)
        status["updated_at"] = utc_now()
        write_status(status_path, status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
