from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from m3_edu_memory.database import connect, initialize
from m3_edu_memory.embeddings import create_embedding_client
from m3_edu_memory.memory_writer import rebuild_mastery_states
from m3_edu_memory.retrieval import build_memory_index
from m3_edu_memory.routing import load_routing_policy
from m3_edu_memory.verification import reverify_corrections
from m3_edu_memory.vlm import PROMPT_VERSION


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, payload: dict) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1 * (attempt + 1))


def read_progress(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def counts(db_path: str, policy_name: str, policy_version: str) -> dict:
    connection = sqlite3.connect(db_path)
    total = connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    routed = connection.execute(
        """SELECT COUNT(DISTINCT attempt_id) FROM routing_decisions
           WHERE policy_name=? AND policy_version=?
             AND materialized_run_id IS NOT NULL""",
        (policy_name, policy_version),
    ).fetchone()[0]
    connection.close()
    return {"attempts": total, "routed": routed, "remaining": total - routed}


def validate_media_runtime(db_path: str) -> None:
    """Fail before a large batch when Parquet-backed images cannot be read."""
    connection = sqlite3.connect(db_path)
    parquet_source = connection.execute(
        """SELECT 1 FROM attempts
           WHERE image_source_path LIKE 'parquet://%'
           LIMIT 1"""
    ).fetchone()
    connection.close()
    if parquet_source is None:
        return
    try:
        import pyarrow.parquet  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for Parquet-backed FERMAT images. "
            "Set PYTHONPATH to "
            "'vendor\\python;..\\dataset\\.python-libs;src' before starting "
            "the Seed main-memory batch."
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/seed_memory.db")
    parser.add_argument("--provider-config", default="config/vlm_providers.json")
    parser.add_argument("--routing-config", default="config/model_routing.json")
    parser.add_argument("--output", default="outputs/seed-main-memory")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument(
        "--embedding-config", default="config/embedding_providers.json"
    )
    parser.add_argument(
        "--embedding-profile", default="volcengine-doubao-embedding-vision"
    )
    args = parser.parse_args()
    if not os.environ.get("ARK_API_KEY"):
        raise RuntimeError("ARK_API_KEY is not configured in this process")
    validate_media_runtime(args.db)

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "status.json"
    log_path = root / "batch.log"
    policy = load_routing_policy(
        args.routing_config, provider_config=args.provider_config
    )
    status = {
        "status": "running",
        "stage": "starting",
        "pid": os.getpid(),
        "database": str(Path(args.db).resolve()),
        "policy_name": policy.policy_name,
        "policy_version": policy.policy_version,
        "model_alias": policy.model_alias,
        "max_workers": args.max_workers,
        "started_at": now(),
        "updated_at": now(),
        **counts(args.db, policy.policy_name, policy.policy_version),
    }
    write_status(status_path, status)
    base = [sys.executable, "-u", "-m", "m3_edu_memory.cli"]
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            for pass_number in range(1, args.max_passes + 1):
                current = counts(
                    args.db, policy.policy_name, policy.policy_version
                )
                status.update(current)
                if current["remaining"] == 0:
                    break
                status["stage"] = f"routing-pass-{pass_number}"
                status["updated_at"] = now()
                write_status(status_path, status)
                progress_path = root / f"routing-pass-{pass_number}.progress.json"
                progress_path.unlink(missing_ok=True)
                command = base + [
                    "diagnose-routed",
                    "--db", args.db,
                    "--provider-config", args.provider_config,
                    "--routing-config", args.routing_config,
                    "--only-unprocessed",
                    "--limit", str(current["remaining"]),
                    "--max-workers", str(args.max_workers),
                    "--defer-derived-refresh",
                    "--result-output", str(root / f"routing-pass-{pass_number}.json"),
                    "--progress-output", str(progress_path),
                ]
                log.write("\n$ " + subprocess.list2cmdline(command) + "\n")
                log.flush()
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT
                )
                while process.poll() is None:
                    current = counts(
                        args.db, policy.policy_name, policy.policy_version
                    )
                    status.update(current)
                    status["active_pass"] = pass_number
                    status["child_pid"] = process.pid
                    pass_progress = read_progress(progress_path)
                    if pass_progress is not None:
                        status["pass_progress"] = pass_progress
                    status["updated_at"] = now()
                    write_status(status_path, status)
                    time.sleep(15)
                status["last_pass_exit_code"] = process.returncode
                if process.returncode != 0:
                    status["status"] = "failed"
                    status["stage"] = "routing-pass-failed"
                    status["error"] = (
                        f"routing pass {pass_number} exited with code "
                        f"{process.returncode}; inspect batch.log"
                    )
                    status["updated_at"] = now()
                    write_status(status_path, status)
                    raise RuntimeError(status["error"])

            final_counts = counts(
                args.db, policy.policy_name, policy.policy_version
            )
            status.update(final_counts)
            status["stage"] = "reverify-corrections"
            status["updated_at"] = now()
            write_status(status_path, status)
            status["verification"] = reverify_corrections(
                args.db, model=policy.model_alias
            )
            status["stage"] = "rebuild-derived-memory"
            status["updated_at"] = now()
            write_status(status_path, status)
            connection = connect(args.db)
            initialize(connection)
            with connection:
                rebuild_mastery_states(
                    connection,
                    model=policy.model_alias,
                    prompt_version=PROMPT_VERSION,
                    change_reason="seed-main-batch-completed",
                )
            connection.close()
            embedding_client = create_embedding_client(
                args.embedding_config, profile_name=args.embedding_profile
            )
            index_result = build_memory_index(
                args.db,
                model=policy.model_alias,
                embedding_client=embedding_client,
            )
            status["index"] = index_result
            status["status"] = (
                "completed" if final_counts["remaining"] == 0 else "partial"
            )
            status["stage"] = status["status"]
            status["completed_at"] = now()
            status["updated_at"] = status["completed_at"]
            write_status(status_path, status)
            return 0 if status["status"] == "completed" else 1
    except Exception as exc:
        status["status"] = "failed"
        status["stage"] = "failed"
        status["error_type"] = type(exc).__name__
        status["error"] = str(exc)
        status["updated_at"] = now()
        write_status(status_path, status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
