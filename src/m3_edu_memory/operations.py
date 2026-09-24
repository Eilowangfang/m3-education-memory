from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .database import connect, initialize


def routing_batch_status(
    db_path: str | Path,
    *,
    routing_config: str | Path,
    recent_minutes: int = 30,
) -> dict:
    config = json.loads(Path(routing_config).read_text(encoding="utf-8"))
    policy_name = config["policy_name"]
    policy_version = config["policy_version"]
    model_alias = config["model_alias"]
    connection = connect(db_path)
    initialize(connection)
    total = connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    routed = connection.execute(
        """SELECT COUNT(DISTINCT attempt_id) FROM routing_decisions
           WHERE policy_name=? AND policy_version=?
             AND materialized_run_id IS NOT NULL""",
        (policy_name, policy_version),
    ).fetchone()[0]
    decisions = connection.execute(
        """SELECT COUNT(*) AS decisions,
                  SUM(CASE WHEN escalated=1 THEN 1 ELSE 0 END) AS escalated,
                  COUNT(DISTINCT CASE
                    WHEN materialized_run_id IS NULL AND NOT EXISTS(
                      SELECT 1 FROM routing_decisions successful
                      WHERE successful.attempt_id=routing_decisions.attempt_id
                        AND successful.policy_name=routing_decisions.policy_name
                        AND successful.policy_version=routing_decisions.policy_version
                        AND successful.materialized_run_id IS NOT NULL
                    ) THEN routing_decisions.attempt_id END) AS retryable
           FROM routing_decisions WHERE policy_name=? AND policy_version=?""",
        (policy_name, policy_version),
    ).fetchone()
    selected = connection.execute(
        """SELECT selected_model,COUNT(*) AS count FROM routing_decisions
           WHERE policy_name=? AND policy_version=?
             AND materialized_run_id IS NOT NULL
           GROUP BY selected_model ORDER BY selected_model""",
        (policy_name, policy_version),
    ).fetchall()
    ingestion = connection.execute(
        """SELECT status,COUNT(*) AS count FROM ingestion_jobs
           GROUP BY status ORDER BY status"""
    ).fetchall()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max(1, recent_minutes))
    recent_rows = connection.execute(
        """SELECT created_at FROM analysis_runs
           WHERE model=? AND status='completed'""",
        (model_alias,),
    ).fetchall()
    connection.close()
    recent_count = 0
    for row in recent_rows:
        try:
            created = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            if created >= cutoff:
                recent_count += 1
        except (TypeError, ValueError):
            continue
    rate = recent_count * 60.0 / max(1, recent_minutes)
    remaining = max(0, total - routed)
    eta_hours = remaining / rate if rate > 0 else None
    return {
        "policy_name": policy_name,
        "policy_version": policy_version,
        "model_alias": model_alias,
        "total_attempts": total,
        "materialized_attempts": routed,
        "remaining_attempts": remaining,
        "coverage": routed / total if total else 0.0,
        "routing_decisions": int(decisions["decisions"] or 0),
        "escalated_decisions": int(decisions["escalated"] or 0),
        "retryable_unmaterialized_decisions": int(decisions["retryable"] or 0),
        "selected_models": [dict(row) for row in selected],
        "ingestion_jobs": [dict(row) for row in ingestion],
        "recent_window_minutes": max(1, recent_minutes),
        "recent_completed": recent_count,
        "throughput_per_hour": round(rate, 2),
        "estimated_remaining_hours": round(eta_hours, 2) if eta_hours is not None else None,
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }
_FATAL_PROVIDER_MARKERS = (
    "accountoverdueerror",
    "insufficient_quota",
    "http 401",
    "http 403",
    "invalid api key",
    "invalid_api_key",
    "authentication failed",
    "unauthorized",
    "permission denied",
    "access denied",
)


def is_fatal_provider_error(error: BaseException | str) -> bool:
    """Return true when retrying other records cannot fix the provider error."""
    message = str(error).casefold()
    return any(marker in message for marker in _FATAL_PROVIDER_MARKERS)
