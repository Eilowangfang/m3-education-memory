from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

from .database import connect, initialize
from .taxonomy import classify_reference_reason
from .time_simulation import SIMULATION_SEED_VERSION, simulated_time


DEFAULT_REVISION = "80ff9934c38615bb8d3a33c24252db02e21774f0"


@dataclass(frozen=True)
class ImportStats:
    attempts: int = 0
    errors: int = 0
    episodic_memories: int = 0


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to import FERMAT; install the project dependencies"
        ) from exc
    return pq


def parquet_files(dataset_dir: str | Path) -> list[Path]:
    root = Path(dataset_dir)
    candidates = sorted((root / "data").glob("*.parquet"))
    if not candidates:
        candidates = sorted(root.glob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet shards found under {root}")
    return candidates


def attempt_id_for(shard: Path, row_index: int) -> str:
    shard_token = shard.stem.replace("train-", "").replace("-of-", "-")
    return f"fermat-{shard_token}-{row_index:06d}"


def _node(connection, node_id: str, node_type: str, label: str, properties: dict | None = None):
    connection.execute(
        """INSERT INTO memory_nodes(node_id,node_type,label,properties_json)
           VALUES(?,?,?,?)
           ON CONFLICT(node_id) DO UPDATE SET
             node_type=excluded.node_type,
             label=excluded.label,
             properties_json=excluded.properties_json""",
        (node_id, node_type, label, json.dumps(properties or {}, ensure_ascii=False)),
    )


def _edge(connection, source: str, relation: str, target: str):
    connection.execute(
        """INSERT OR IGNORE INTO memory_edges(source_id,relation,target_id)
           VALUES(?,?,?)""",
        (source, relation, target),
    )


def import_fermat(
    *,
    dataset_dir: str | Path,
    db_path: str | Path,
    dataset_revision: str = DEFAULT_REVISION,
    simulation_year: int | None = None,
    simulation_anchor: date | str | None = None,
    timezone_name: str = "Asia/Shanghai",
    seed: str = "m3-education-memory",
    bootstrap_reference_memory: bool = False,
) -> ImportStats:
    pq = _require_pyarrow()
    connection = connect(db_path)
    initialize(connection)
    attempts = errors = episodic = 0

    with connection:
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES('dataset_revision',?)",
            (dataset_revision,),
        )
        effective_anchor = (
            date(simulation_year, 7, 31).isoformat()
            if simulation_year is not None and simulation_anchor is None
            else (
                simulation_anchor.isoformat()
                if isinstance(simulation_anchor, date)
                else str(simulation_anchor or date.today().isoformat())
            )
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES('simulation_anchor_date',?)",
            (effective_anchor,),
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES('simulation_seed_version',?)",
            (SIMULATION_SEED_VERSION,),
        )
        connection.execute("DELETE FROM metadata WHERE key='simulation_year'")
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES('timezone',?)",
            (timezone_name,),
        )

        for shard in parquet_files(dataset_dir):
            table = pq.read_table(shard)
            for row_index, row in enumerate(table.to_pylist()):
                image = row["image"] or {}
                image_bytes = image.get("bytes") or b""
                if not image_bytes:
                    raise ValueError(f"Missing image bytes in {shard.name} row {row_index}")
                attempt_id = attempt_id_for(shard, row_index)
                timestamp = simulated_time(
                    grade=row["grade"],
                    dataset_revision=dataset_revision,
                    shard=shard.name,
                    row_index=row_index,
                    anchor_date=effective_anchor,
                    year=simulation_year,
                    timezone=timezone_name,
                    seed=seed,
                )
                image_source = f"parquet://{shard.resolve().as_posix()}#row={row_index}"
                error_type = classify_reference_reason(
                    row.get("pert_reasoning"), bool(row["has_error"])
                )
                connection.execute(
                    """INSERT INTO attempts(
                         attempt_id,dataset_revision,shard,row_index,custom_id,img_id,
                         image_sha256,image_source_path,grade,Time,time_source,time_bucket,
                         simulation_seed_version,domain_code,subdomain_code,orig_q,
                         handwriting_style,image_quality,rotation,is_table)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(attempt_id) DO UPDATE SET
                         image_sha256=excluded.image_sha256,
                         image_source_path=excluded.image_source_path,
                         Time=excluded.Time""",
                    (
                        attempt_id, dataset_revision, shard.name, row_index,
                        row.get("new_custom_id"), row.get("img_id"),
                        hashlib.sha256(image_bytes).hexdigest(), image_source,
                        row["grade"], timestamp.isoformat(), "simulated_from_grade",
                        row["grade"], SIMULATION_SEED_VERSION, row["domain_code"],
                        row.get("subdomain_code"), row["orig_q"],
                        int(bool(row.get("handwriting_style"))),
                        int(bool(row.get("image_quality"))), row.get("rotation"),
                        int(bool(row.get("is_table"))),
                    ),
                )
                connection.execute(
                    """INSERT INTO benchmark_truth(
                         attempt_id,has_error,orig_a,pert_a,pert_reasoning,reference_error_type)
                       VALUES(?,?,?,?,?,?)
                       ON CONFLICT(attempt_id) DO UPDATE SET
                         has_error=excluded.has_error,
                         orig_a=excluded.orig_a,
                         pert_a=excluded.pert_a,
                         pert_reasoning=excluded.pert_reasoning,
                         reference_error_type=excluded.reference_error_type""",
                    (
                        attempt_id, int(bool(row["has_error"])), row["orig_a"],
                        row["pert_a"], row.get("pert_reasoning"), error_type,
                    ),
                )

                attempt_node = f"attempt:{attempt_id}"
                problem_node = f"problem:{row['img_id']}"
                domain_node = f"knowledge:domain:{row['domain_code']}"
                subdomain = row.get("subdomain_code") or "unknown"
                subdomain_node = f"knowledge:subdomain:{row['domain_code']}:{subdomain}"
                _node(connection, attempt_node, "Attempt", attempt_id, {"Time": timestamp.isoformat()})
                _node(connection, problem_node, "Problem", str(row["img_id"]), {"orig_q": row["orig_q"]})
                _node(connection, domain_node, "KnowledgePoint", row["domain_code"])
                _node(connection, subdomain_node, "KnowledgePoint", subdomain)
                _edge(connection, attempt_node, "ANSWERS", problem_node)
                _edge(connection, problem_node, "TESTS", subdomain_node)
                _edge(connection, subdomain_node, "IS_A", domain_node)

                if bootstrap_reference_memory:
                    memory_id = f"episode:{attempt_id}:benchmark"
                    status = "confirmed_reference" if row["has_error"] else "no_actual_error_reference"
                    content = (
                        f"At {timestamp.isoformat()} the attempt in {row['domain_code']}/"
                        f"{subdomain} was labeled {status}. "
                        f"Reference reason: {row.get('pert_reasoning') or 'not provided'}"
                    )
                    connection.execute(
                        """INSERT INTO episodic_memories(
                             memory_id,attempt_id,content,error_type,status,source,created_at)
                           VALUES(?,?,?,?,?,?,?)
                           ON CONFLICT(memory_id) DO UPDATE SET
                             content=excluded.content,error_type=excluded.error_type,
                             status=excluded.status""",
                        (
                            memory_id, attempt_id, content, error_type, status,
                            "benchmark_reference", datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                    _node(connection, memory_id, "ErrorEpisode", error_type, {"source": "benchmark_reference"})
                    _edge(connection, attempt_node, "HAS_EPISODE", memory_id)
                    _edge(connection, memory_id, "ABOUT", subdomain_node)
                    episodic += 1

                attempts += 1
                errors += int(bool(row["has_error"]))

        if bootstrap_reference_memory:
            _rebuild_semantic_memories(connection)

    connection.close()
    return ImportStats(attempts=attempts, errors=errors, episodic_memories=episodic)


def _rebuild_semantic_memories(connection) -> None:
    rows = connection.execute(
        """SELECT a.domain_code,a.subdomain_code,b.reference_error_type,
                  COUNT(*) AS evidence_count,
                  json_group_array(a.attempt_id) AS evidence_ids
           FROM attempts a JOIN benchmark_truth b USING(attempt_id)
           WHERE b.has_error=1
           GROUP BY a.domain_code,a.subdomain_code,b.reference_error_type"""
    ).fetchall()
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        subdomain = row["subdomain_code"] or "unknown"
        memory_id = f"semantic:{row['domain_code']}:{subdomain}:{row['reference_error_type']}:benchmark"
        content = (
            f"The simulated learner has {row['evidence_count']} benchmark-labeled "
            f"{row['reference_error_type']} errors in {row['domain_code']}/{subdomain}."
        )
        connection.execute(
            """INSERT INTO semantic_memories(
                 memory_id,domain_code,subdomain_code,error_type,evidence_count,
                 content,evidence_attempt_ids_json,source,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?)
               ON CONFLICT(memory_id) DO UPDATE SET
                 evidence_count=excluded.evidence_count,
                 content=excluded.content,
                 evidence_attempt_ids_json=excluded.evidence_attempt_ids_json,
                 updated_at=excluded.updated_at""",
            (
                memory_id, row["domain_code"], row["subdomain_code"],
                row["reference_error_type"], row["evidence_count"], content,
                row["evidence_ids"], "benchmark_reference", now,
            ),
        )
