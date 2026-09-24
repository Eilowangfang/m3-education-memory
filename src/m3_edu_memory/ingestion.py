from __future__ import annotations

import base64
import hashlib
import io
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image

from .database import connect, initialize
from .diagnosis import diagnose_attempt
from .object_store import LocalObjectStore
from .vlm import VisionClient


MAX_IMAGE_BYTES = 20 * 1024 * 1024


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_timestamp(value: str | None) -> str:
    if not value:
        return _iso_now()
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("attempted_at must include a timezone")
    return parsed.isoformat()


def decode_image_base64(value: str) -> bytes:
    if value.startswith("data:"):
        _, _, value = value.partition(",")
    try:
        data = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise ValueError("image_base64 is not valid base64") from exc
    if not data:
        raise ValueError("image_base64 is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image exceeds the 20 MiB limit")
    return data


def enqueue_uploaded_attempt(
    db_path: str | Path,
    *,
    object_root: str | Path,
    image_bytes: bytes,
    question_text: str,
    domain_code: str,
    subdomain_code: str | None = None,
    grade: str = "c00",
    attempted_at: str | None = None,
    idempotency_key: str | None = None,
    requested_model: str | None = None,
    max_attempts: int = 3,
) -> dict:
    question_text = question_text.strip()
    domain_code = domain_code.strip()
    if not question_text:
        raise ValueError("question_text is required")
    if not domain_code:
        raise ValueError("domain_code is required")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("image exceeds the 20 MiB limit")
    if not 1 <= max_attempts <= 10:
        raise ValueError("max_attempts must be between 1 and 10")
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
    except Exception as exc:
        raise ValueError("uploaded bytes are not a readable image") from exc

    stored = LocalObjectStore(object_root).put_image(image_bytes)
    event_time = _validate_timestamp(attempted_at)
    key = idempotency_key or hashlib.sha256(
        f"{stored.sha256}\0{question_text}\0{event_time}".encode("utf-8")
    ).hexdigest()
    connection = connect(db_path)
    initialize(connection)
    existing = connection.execute(
        "SELECT * FROM ingestion_jobs WHERE idempotency_key=?", (key,)
    ).fetchone()
    if existing:
        connection.close()
        return {**dict(existing), "idempotent_replay": True}

    attempt_id = f"upload-{uuid.uuid4()}"
    job_id = str(uuid.uuid4())
    created_at = _iso_now()
    # Existing FERMAT-compatible columns remain populated for schema stability;
    # attempt_assets.event_time_source is the authoritative provenance field.
    try:
        with connection:
            connection.execute(
                "INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    attempt_id, "user-upload-v1", f"upload:{attempt_id}", 0,
                    attempt_id, None, stored.sha256, stored.uri, grade, event_time,
                    "simulated_from_grade", grade, "uploaded-event-time-v1",
                    domain_code, subdomain_code, question_text, None, None, None, 0,
                ),
            )
            connection.execute(
                """INSERT INTO attempt_assets(
                     attempt_id,source_type,object_path,mime_type,byte_size,width,height,
                     event_time_source,uploaded_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    attempt_id, "user_upload", str(stored.path), stored.mime_type,
                    stored.byte_size, width, height, "uploaded_or_provided", created_at,
                ),
            )
            connection.execute(
                """INSERT INTO memory_nodes(node_id,node_type,label,properties_json)
                   VALUES(?,?,?,?)""",
                (f"attempt:{attempt_id}", "Attempt", attempt_id, "{}"),
            )
            connection.execute(
                """INSERT INTO ingestion_jobs(
                     job_id,attempt_id,idempotency_key,requested_model,status,
                     max_attempts,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    job_id, attempt_id, key, requested_model, "queued",
                    max_attempts, created_at, created_at,
                ),
            )
    except sqlite3.IntegrityError:
        existing = connection.execute(
            "SELECT * FROM ingestion_jobs WHERE idempotency_key=?", (key,)
        ).fetchone()
        if existing:
            connection.close()
            return {**dict(existing), "idempotent_replay": True}
        connection.close()
        raise
    connection.close()
    return {
        "job_id": job_id,
        "attempt_id": attempt_id,
        "idempotency_key": key,
        "requested_model": requested_model,
        "status": "queued",
        "created_at": created_at,
        "updated_at": created_at,
        "idempotent_replay": False,
    }


def get_ingestion_job(db_path: str | Path, *, job_id: str) -> dict:
    connection = connect(db_path)
    initialize(connection)
    row = connection.execute(
        "SELECT * FROM ingestion_jobs WHERE job_id=?", (job_id,)
    ).fetchone()
    connection.close()
    if row is None:
        raise KeyError(f"Unknown job_id: {job_id}")
    return dict(row)


def get_attempt_processing_status(db_path: str | Path, *, attempt_id: str) -> dict:
    connection = connect(db_path)
    initialize(connection)
    row = connection.execute(
        """SELECT a.attempt_id,a.Time,a.domain_code,a.subdomain_code,a.orig_q,
                  a.image_source_path,j.job_id,j.status,j.requested_model,j.run_id,
                  j.error_message,j.attempt_count,j.max_attempts,j.next_attempt_at,
                  j.lease_expires_at,j.dead_lettered_at,
                  s.source_type,s.object_path,s.mime_type,s.byte_size,
                  s.width,s.height,s.event_time_source,s.uploaded_at
           FROM attempts a
           LEFT JOIN ingestion_jobs j USING(attempt_id)
           LEFT JOIN attempt_assets s USING(attempt_id)
           WHERE a.attempt_id=?""",
        (attempt_id,),
    ).fetchone()
    connection.close()
    if row is None:
        raise KeyError(f"Unknown attempt_id: {attempt_id}")
    result = dict(row)
    result["status"] = result["status"] or "ready"
    return result


def recover_expired_jobs(db_path: str | Path) -> dict:
    connection = connect(db_path)
    initialize(connection)
    now = _iso_now()
    with connection:
        retryable = connection.execute(
            """UPDATE ingestion_jobs
               SET status='queued',lease_owner=NULL,lease_expires_at=NULL,
                   next_attempt_at=?,error_message='worker lease expired',updated_at=?
               WHERE status='processing' AND lease_expires_at IS NOT NULL
                 AND lease_expires_at<=? AND attempt_count<max_attempts""",
            (now, now, now),
        ).rowcount
        dead = connection.execute(
            """UPDATE ingestion_jobs
               SET status='failed',lease_owner=NULL,lease_expires_at=NULL,
                   dead_lettered_at=?,error_message='worker lease expired; retries exhausted',
                   updated_at=?
               WHERE status='processing' AND lease_expires_at IS NOT NULL
                 AND lease_expires_at<=? AND attempt_count>=max_attempts""",
            (now, now, now),
        ).rowcount
    connection.close()
    return {"requeued": retryable, "dead_lettered": dead}


def retry_ingestion_job(db_path: str | Path, *, job_id: str) -> dict:
    connection = connect(db_path)
    initialize(connection)
    now = _iso_now()
    with connection:
        changed = connection.execute(
            """UPDATE ingestion_jobs
               SET status='queued',attempt_count=0,lease_owner=NULL,
                   lease_expires_at=NULL,next_attempt_at=NULL,dead_lettered_at=NULL,
                   error_message=NULL,updated_at=?
               WHERE job_id=? AND status='failed'""",
            (now, job_id),
        ).rowcount
    connection.close()
    if not changed:
        raise ValueError("only failed jobs can be retried")
    return get_ingestion_job(db_path, job_id=job_id)


def ingestion_metrics(db_path: str | Path) -> dict:
    connection = connect(db_path)
    initialize(connection)
    rows = connection.execute(
        "SELECT status,COUNT(*) AS count FROM ingestion_jobs GROUP BY status"
    ).fetchall()
    now = _iso_now()
    expired = connection.execute(
        """SELECT COUNT(*) FROM ingestion_jobs WHERE status='processing'
           AND lease_expires_at IS NOT NULL AND lease_expires_at<=?""",
        (now,),
    ).fetchone()[0]
    retried = connection.execute(
        "SELECT COUNT(*) FROM ingestion_jobs WHERE attempt_count>1"
    ).fetchone()[0]
    exhausted = connection.execute(
        "SELECT COUNT(*) FROM ingestion_jobs WHERE dead_lettered_at IS NOT NULL"
    ).fetchone()[0]
    connection.close()
    counts = {row["status"]: row["count"] for row in rows}
    return {
        "total": sum(counts.values()),
        "by_status": counts,
        "expired_leases": expired,
        "retried_jobs": retried,
        "dead_lettered_jobs": exhausted,
    }


def process_next_job(
    db_path: str | Path,
    *,
    client: VisionClient,
    worker_id: str | None = None,
    lease_seconds: int = 600,
    retry_base_seconds: int = 5,
) -> dict | None:
    if lease_seconds < 30:
        raise ValueError("lease_seconds must be at least 30")
    if retry_base_seconds < 0:
        raise ValueError("retry_base_seconds must be non-negative")
    recover_expired_jobs(db_path)
    connection = connect(db_path)
    initialize(connection)
    now = _iso_now()
    owner = worker_id or f"worker-{uuid.uuid4()}"
    lease_expires = (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat()
    with connection:
        job = connection.execute(
            """SELECT * FROM ingestion_jobs
               WHERE status='queued' AND (requested_model IS NULL OR requested_model=?)
                 AND (next_attempt_at IS NULL OR next_attempt_at<=?)
                 AND attempt_count<max_attempts
               ORDER BY created_at,job_id LIMIT 1""",
            (client.model, now),
        ).fetchone()
        if job is None:
            connection.close()
            return None
        claimed = connection.execute(
            """UPDATE ingestion_jobs
               SET status='processing',attempt_count=attempt_count+1,
                   lease_owner=?,lease_expires_at=?,next_attempt_at=NULL,updated_at=?
               WHERE job_id=? AND status='queued'""",
            (owner, lease_expires, now, job["job_id"]),
        ).rowcount
    connection.close()
    if not claimed:
        return None
    try:
        diagnosis = diagnose_attempt(
            db_path, attempt_id=job["attempt_id"], client=client
        )
        connection = connect(db_path)
        with connection:
            connection.execute(
                """UPDATE ingestion_jobs
                   SET status='completed',run_id=?,error_message=NULL,
                       lease_owner=NULL,lease_expires_at=NULL,updated_at=?
                   WHERE job_id=? AND lease_owner=?""",
                (diagnosis["run_id"], _iso_now(), job["job_id"], owner),
            )
        connection.close()
        return {"job_id": job["job_id"], "status": "completed", "diagnosis": diagnosis}
    except Exception as exc:
        connection = connect(db_path)
        attempt_number = int(job["attempt_count"]) + 1
        terminal = attempt_number >= int(job["max_attempts"])
        retry_at = (
            datetime.now(timezone.utc)
            + timedelta(seconds=retry_base_seconds * (2 ** max(0, attempt_number - 1)))
        ).isoformat()
        failed_at = _iso_now()
        with connection:
            if terminal:
                connection.execute(
                    """UPDATE ingestion_jobs
                       SET status='failed',error_message=?,dead_lettered_at=?,
                           lease_owner=NULL,lease_expires_at=NULL,updated_at=?
                       WHERE job_id=? AND lease_owner=?""",
                    (str(exc), failed_at, failed_at, job["job_id"], owner),
                )
            else:
                connection.execute(
                    """UPDATE ingestion_jobs
                       SET status='queued',error_message=?,next_attempt_at=?,
                           lease_owner=NULL,lease_expires_at=NULL,updated_at=?
                       WHERE job_id=? AND lease_owner=?""",
                    (str(exc), retry_at, failed_at, job["job_id"], owner),
                )
        connection.close()
        return {
            "job_id": job["job_id"],
            "status": "failed" if terminal else "retry_scheduled",
            "attempt_count": attempt_number,
            "max_attempts": int(job["max_attempts"]),
            "next_attempt_at": None if terminal else retry_at,
            "error": str(exc),
        }
