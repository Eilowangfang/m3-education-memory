from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT PRIMARY KEY,
    dataset_revision TEXT NOT NULL,
    shard TEXT NOT NULL,
    row_index INTEGER NOT NULL,
    custom_id TEXT,
    img_id INTEGER,
    image_sha256 TEXT NOT NULL,
    image_source_path TEXT NOT NULL,
    grade TEXT NOT NULL,
    Time TEXT NOT NULL,
    time_source TEXT NOT NULL CHECK (time_source = 'simulated_from_grade'),
    time_bucket TEXT NOT NULL,
    simulation_seed_version TEXT NOT NULL,
    domain_code TEXT NOT NULL,
    subdomain_code TEXT,
    orig_q TEXT NOT NULL,
    handwriting_style INTEGER,
    image_quality INTEGER,
    rotation TEXT,
    is_table INTEGER,
    UNIQUE(shard, row_index)
);

CREATE INDEX IF NOT EXISTS idx_attempts_domain_time
ON attempts(domain_code, Time);

CREATE INDEX IF NOT EXISTS idx_attempts_subdomain_time
ON attempts(subdomain_code, Time);

CREATE TABLE IF NOT EXISTS benchmark_truth (
    attempt_id TEXT PRIMARY KEY REFERENCES attempts(attempt_id) ON DELETE CASCADE,
    has_error INTEGER NOT NULL,
    orig_a TEXT NOT NULL,
    pert_a TEXT NOT NULL,
    pert_reasoning TEXT,
    reference_error_type TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_truth_error
ON benchmark_truth(has_error, reference_error_type);

CREATE TABLE IF NOT EXISTS memory_nodes (
    node_id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    label TEXT NOT NULL,
    properties_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_nodes_type ON memory_nodes(node_type);

CREATE TABLE IF NOT EXISTS memory_edges (
    source_id TEXT NOT NULL REFERENCES memory_nodes(node_id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    target_id TEXT NOT NULL REFERENCES memory_nodes(node_id) ON DELETE CASCADE,
    properties_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY(source_id, relation, target_id)
);

CREATE TABLE IF NOT EXISTS episodic_memories (
    memory_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    error_type TEXT NOT NULL,
    status TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(attempt_id, source)
);

CREATE TABLE IF NOT EXISTS semantic_memories (
    memory_id TEXT PRIMARY KEY,
    domain_code TEXT NOT NULL,
    subdomain_code TEXT,
    error_type TEXT NOT NULL,
    evidence_count INTEGER NOT NULL,
    content TEXT NOT NULL,
    evidence_attempt_ids_json TEXT NOT NULL,
    source TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(domain_code, subdomain_code, error_type, source)
);

CREATE TABLE IF NOT EXISTS analysis_runs (
    run_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    status TEXT NOT NULL,
    raw_response TEXT,
    parsed_json TEXT,
    error_message TEXT,
    input_image_bytes INTEGER,
    input_width INTEGER,
    input_height INTEGER,
    input_mime_type TEXT,
    preprocessing_json TEXT,
    latency_ms INTEGER,
    provider_usage_json TEXT,
    request_attempts INTEGER,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_analysis_attempt_created
ON analysis_runs(attempt_id, created_at);

CREATE TABLE IF NOT EXISTS diagnoses (
    diagnosis_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
    attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id) ON DELETE CASCADE,
    question_transcription TEXT NOT NULL,
    solution_transcription TEXT NOT NULL,
    has_error_pred INTEGER,
    error_type_pred TEXT NOT NULL,
    first_error_step INTEGER,
    error_explanation TEXT NOT NULL,
    knowledge_points_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    requires_review INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS diagnosis_steps (
    run_id TEXT NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL,
    transcription TEXT NOT NULL,
    normalized_latex TEXT,
    bbox_json TEXT,
    is_error INTEGER,
    confidence REAL NOT NULL,
    PRIMARY KEY(run_id, step_index)
);

CREATE TABLE IF NOT EXISTS diagnosis_reviews (
    review_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
    attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id) ON DELETE CASCADE,
    verdict TEXT NOT NULL CHECK (verdict IN ('confirmed','rejected','modified')),
    has_error_override INTEGER,
    error_type_override TEXT,
    first_error_step_override INTEGER,
    error_bbox_override_json TEXT,
    knowledge_points_override_json TEXT,
    error_explanation_override TEXT,
    reviewer TEXT,
    notes TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reviews_run_created
ON diagnosis_reviews(run_id, created_at DESC);

CREATE TABLE IF NOT EXISTS corrections (
    correction_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
    attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id) ON DELETE CASCADE,
    corrected_solution TEXT NOT NULL,
    corrected_steps_json TEXT NOT NULL,
    final_answer TEXT,
    verification_expression TEXT,
    verification_status TEXT NOT NULL,
    verification_method TEXT NOT NULL,
    verification_details_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    rendered_path TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_corrections_attempt
ON corrections(attempt_id, created_at);

CREATE TABLE IF NOT EXISTS correction_versions (
    version_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id) ON DELETE CASCADE,
    run_id TEXT REFERENCES analysis_runs(run_id) ON DELETE SET NULL,
    model TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('vlm','human','rule')),
    parent_version_id TEXT REFERENCES correction_versions(version_id) ON DELETE SET NULL,
    corrected_solution TEXT NOT NULL,
    corrected_steps_json TEXT NOT NULL,
    final_answer TEXT,
    verification_expression TEXT,
    verification_status TEXT NOT NULL,
    verification_method TEXT NOT NULL,
    verification_details_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('draft','active','superseded','rejected')),
    created_by TEXT NOT NULL,
    rendered_path TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_correction_versions_attempt_model_created
ON correction_versions(attempt_id, model, created_at DESC);

CREATE TABLE IF NOT EXISTS mastery_states (
    mastery_id TEXT PRIMARY KEY,
    domain_code TEXT NOT NULL,
    subdomain_code TEXT,
    knowledge_point TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    attempt_count INTEGER NOT NULL,
    error_count INTEGER NOT NULL,
    verified_correction_count INTEGER NOT NULL,
    weakness_score REAL NOT NULL,
    score_components_json TEXT NOT NULL,
    error_types_json TEXT NOT NULL,
    evidence_attempt_ids_json TEXT NOT NULL,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(domain_code, subdomain_code, knowledge_point, model, prompt_version)
);

CREATE INDEX IF NOT EXISTS idx_mastery_model_domain_score
ON mastery_states(model, domain_code, weakness_score DESC);

CREATE TABLE IF NOT EXISTS mastery_state_history (
    snapshot_id TEXT PRIMARY KEY,
    mastery_id TEXT NOT NULL,
    domain_code TEXT NOT NULL,
    subdomain_code TEXT,
    knowledge_point TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    attempt_count INTEGER NOT NULL,
    error_count INTEGER NOT NULL,
    verified_correction_count INTEGER NOT NULL,
    weakness_score REAL NOT NULL,
    score_components_json TEXT NOT NULL,
    error_types_json TEXT NOT NULL,
    evidence_attempt_ids_json TEXT NOT NULL,
    status TEXT NOT NULL,
    state_hash TEXT NOT NULL,
    change_reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mastery_history_model_point_created
ON mastery_state_history(model, knowledge_point, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_mastery_history_mastery_created
ON mastery_state_history(mastery_id, created_at DESC);

CREATE TABLE IF NOT EXISTS memory_documents (
    document_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id) ON DELETE CASCADE,
    run_id TEXT NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    domain_code TEXT NOT NULL,
    subdomain_code TEXT,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(attempt_id, model)
);

CREATE INDEX IF NOT EXISTS idx_memory_documents_model_domain
ON memory_documents(model, domain_code);

CREATE TABLE IF NOT EXISTS memory_embeddings (
    document_id TEXT NOT NULL REFERENCES memory_documents(document_id) ON DELETE CASCADE,
    embedding_model TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    vector_blob BLOB NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(document_id, embedding_model)
);

CREATE TABLE IF NOT EXISTS attempt_assets (
    attempt_id TEXT PRIMARY KEY REFERENCES attempts(attempt_id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,
    object_path TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    event_time_source TEXT NOT NULL,
    uploaded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL UNIQUE REFERENCES attempts(attempt_id) ON DELETE CASCADE,
    idempotency_key TEXT NOT NULL UNIQUE,
    requested_model TEXT,
    status TEXT NOT NULL CHECK (status IN ('queued','processing','completed','failed')),
    attempt_count INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    lease_owner TEXT,
    lease_expires_at TEXT,
    next_attempt_at TEXT,
    dead_lettered_at TEXT,
    run_id TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status_created
ON ingestion_jobs(status, created_at);

CREATE TABLE IF NOT EXISTS evaluation_suites (
    suite_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    selection_seed TEXT NOT NULL,
    attempt_ids_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    evaluation_run_id TEXT PRIMARY KEY,
    suite_id TEXT NOT NULL REFERENCES evaluation_suites(suite_id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    evaluated_attempt_count INTEGER NOT NULL,
    metrics_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_evaluation_runs_suite_created
ON evaluation_runs(suite_id, created_at DESC);

CREATE TABLE IF NOT EXISTS routing_decisions (
    routing_decision_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id) ON DELETE CASCADE,
    policy_name TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    initial_model TEXT NOT NULL,
    fallback_model TEXT NOT NULL,
    initial_run_id TEXT REFERENCES analysis_runs(run_id) ON DELETE SET NULL,
    fallback_run_id TEXT REFERENCES analysis_runs(run_id) ON DELETE SET NULL,
    selected_run_id TEXT NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
    materialized_run_id TEXT REFERENCES analysis_runs(run_id) ON DELETE SET NULL,
    selected_model TEXT NOT NULL,
    escalated INTEGER NOT NULL,
    reasons_json TEXT NOT NULL,
    features_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_routing_attempt_policy_created
ON routing_decisions(attempt_id, policy_name, policy_version, created_at DESC);

CREATE TABLE IF NOT EXISTS evaluation_annotations (
    annotation_id TEXT PRIMARY KEY,
    suite_id TEXT NOT NULL REFERENCES evaluation_suites(suite_id) ON DELETE CASCADE,
    attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id) ON DELETE CASCADE,
    parent_annotation_id TEXT REFERENCES evaluation_annotations(annotation_id) ON DELETE SET NULL,
    annotator TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('draft','approved','rejected')),
    has_error INTEGER,
    first_error_step INTEGER,
    error_bbox_json TEXT,
    knowledge_points_json TEXT NOT NULL,
    corrected_final_answer TEXT,
    correction_steps_json TEXT NOT NULL,
    notes TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_evaluation_annotations_suite_attempt_created
ON evaluation_annotations(suite_id, attempt_id, created_at DESC);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def initialize(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA)
    # Lightweight forward migrations for databases created by earlier
    # prototype versions. SQLite supports adding these nullable/defaulted
    # columns without rebuilding the table or disturbing foreign keys.
    job_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(ingestion_jobs)")
    }
    migrations = {
        "attempt_count": "INTEGER NOT NULL DEFAULT 0",
        "max_attempts": "INTEGER NOT NULL DEFAULT 3",
        "lease_owner": "TEXT",
        "lease_expires_at": "TEXT",
        "next_attempt_at": "TEXT",
        "dead_lettered_at": "TEXT",
    }
    for name, definition in migrations.items():
        if name not in job_columns:
            try:
                connection.execute(
                    f"ALTER TABLE ingestion_jobs ADD COLUMN {name} {definition}"
                )
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc).casefold():
                    raise
    review_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(diagnosis_reviews)")
    }
    review_migrations = {
        "first_error_step_override": "INTEGER",
        "error_bbox_override_json": "TEXT",
        "knowledge_points_override_json": "TEXT",
        "error_explanation_override": "TEXT",
        "reviewer": "TEXT",
    }
    for name, definition in review_migrations.items():
        if name not in review_columns:
            try:
                connection.execute(
                    f"ALTER TABLE diagnosis_reviews ADD COLUMN {name} {definition}"
                )
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc).casefold():
                    raise
    run_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(analysis_runs)")
    }
    run_migrations = {
        "input_image_bytes": "INTEGER",
        "input_width": "INTEGER",
        "input_height": "INTEGER",
        "input_mime_type": "TEXT",
        "preprocessing_json": "TEXT",
        "latency_ms": "INTEGER",
        "provider_usage_json": "TEXT",
        "request_attempts": "INTEGER",
    }
    for name, definition in run_migrations.items():
        if name not in run_columns:
            try:
                connection.execute(
                    f"ALTER TABLE analysis_runs ADD COLUMN {name} {definition}"
                )
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc).casefold():
                    raise
    routing_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(routing_decisions)")
    }
    if "materialized_run_id" not in routing_columns:
        try:
            connection.execute(
                "ALTER TABLE routing_decisions ADD COLUMN materialized_run_id TEXT"
            )
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).casefold():
                raise
    # Backfill version history for corrections created before the versioned
    # correction service existed. Repeated initialization remains idempotent.
    connection.execute(
        """INSERT OR IGNORE INTO correction_versions(
             version_id,attempt_id,run_id,model,source,parent_version_id,
             corrected_solution,corrected_steps_json,final_answer,
             verification_expression,verification_status,verification_method,
             verification_details_json,confidence,status,created_by,
             rendered_path,created_at)
           SELECT 'correction-version:' || c.run_id,c.attempt_id,c.run_id,r.model,
                  'vlm',NULL,c.corrected_solution,c.corrected_steps_json,
                  c.final_answer,c.verification_expression,c.verification_status,
                  c.verification_method,c.verification_details_json,c.confidence,
                  'active','migration',c.rendered_path,c.created_at
           FROM corrections c JOIN analysis_runs r ON r.run_id=c.run_id"""
    )
    # Mirror migrated versions into the memory graph. The joins make this safe
    # for partially imported databases whose attempt nodes are not present yet.
    connection.execute(
        """INSERT OR IGNORE INTO memory_nodes(node_id,node_type,label,properties_json)
           SELECT 'correction-version-node:' || version_id,'CorrectionVersion',
                  COALESCE(final_answer,'订正版本'),'{}'
           FROM correction_versions"""
    )
    connection.execute(
        """INSERT OR IGNORE INTO memory_edges(source_id,relation,target_id)
           SELECT a.node_id,'HAS_CORRECTION_VERSION',v.node_id
           FROM correction_versions c
           JOIN memory_nodes a ON a.node_id='attempt:' || c.attempt_id
           JOIN memory_nodes v
             ON v.node_id='correction-version-node:' || c.version_id"""
    )
    connection.execute(
        """INSERT OR IGNORE INTO memory_edges(source_id,relation,target_id)
           SELECT child.node_id,'REVISES',parent.node_id
           FROM correction_versions c
           JOIN memory_nodes child
             ON child.node_id='correction-version-node:' || c.version_id
           JOIN memory_nodes parent
             ON parent.node_id='correction-version-node:' || c.parent_version_id
           WHERE c.parent_version_id IS NOT NULL"""
    )
    # Seed one auditable snapshot for mastery rows created before history was
    # introduced. The hash matches the writer so an unchanged rebuild remains
    # idempotent.
    for row in connection.execute("SELECT * FROM mastery_states").fetchall():
        exists = connection.execute(
            "SELECT 1 FROM mastery_state_history WHERE mastery_id=? LIMIT 1",
            (row["mastery_id"],),
        ).fetchone()
        if exists:
            continue
        state_payload = {
            "attempt_count": row["attempt_count"],
            "error_count": row["error_count"],
            "verified_correction_count": row["verified_correction_count"],
            "weakness_score": row["weakness_score"],
            "score_components_json": row["score_components_json"],
            "error_types_json": row["error_types_json"],
            "evidence_attempt_ids_json": row["evidence_attempt_ids_json"],
            "status": row["status"],
        }
        state_hash = hashlib.sha256(
            json.dumps(state_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        connection.execute(
            """INSERT INTO mastery_state_history(
                 snapshot_id,mastery_id,domain_code,subdomain_code,
                 knowledge_point,model,prompt_version,attempt_count,error_count,
                 verified_correction_count,weakness_score,score_components_json,
                 error_types_json,evidence_attempt_ids_json,status,state_hash,
                 change_reason,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                f"mastery-snapshot:{uuid.uuid4().hex}", row["mastery_id"],
                row["domain_code"], row["subdomain_code"], row["knowledge_point"],
                row["model"], row["prompt_version"], row["attempt_count"],
                row["error_count"], row["verified_correction_count"],
                row["weakness_score"], row["score_components_json"],
                row["error_types_json"], row["evidence_attempt_ids_json"],
                row["status"], state_hash, "migration:existing_mastery_state",
                row["updated_at"] or datetime.now(timezone.utc).isoformat(),
            ),
        )
    connection.commit()
