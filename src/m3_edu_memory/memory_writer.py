from __future__ import annotations

import hashlib
import json
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone

from .knowledge import canonicalize_knowledge_points


def _id_token(value: str) -> str:
    return hashlib.sha256(value.casefold().encode("utf-8")).hexdigest()[:16]


def rebuild_mastery_states(
    connection,
    *,
    model: str,
    prompt_version: str,
    change_reason: str = "mastery_rebuild",
) -> int:
    """Aggregate latest VLM diagnoses only; benchmark truth is never joined."""
    rows = connection.execute(
        """WITH ranked AS (
             SELECT r.*,
                    ROW_NUMBER() OVER (
                      PARTITION BY r.attempt_id
                      ORDER BY r.created_at DESC,r.run_id DESC
                    ) AS row_number
             FROM analysis_runs r
             WHERE r.model=? AND r.prompt_version=? AND r.status='completed'
           ), review_ranked AS (
             SELECT v.*,
                    ROW_NUMBER() OVER (
                      PARTITION BY v.run_id ORDER BY v.created_at DESC,v.review_id DESC
                    ) AS row_number
             FROM diagnosis_reviews v
           ), correction_ranked AS (
             SELECT cv.*,
                    ROW_NUMBER() OVER (
                      PARTITION BY cv.attempt_id,cv.model
                      ORDER BY cv.created_at DESC,cv.version_id DESC
                    ) AS row_number
             FROM correction_versions cv WHERE cv.status='active'
           )
           SELECT a.attempt_id,a.domain_code,a.subdomain_code,a.Time,
                  COALESCE(v.has_error_override,d.has_error_pred) AS has_error_pred,
                  COALESCE(v.error_type_override,d.error_type_pred) AS error_type_pred,
                  COALESCE(v.knowledge_points_override_json,d.knowledge_points_json)
                    AS knowledge_points_json,
                  COALESCE(cv.verification_status,c.verification_status) AS verification_status
           FROM ranked r
           JOIN attempts a ON a.attempt_id=r.attempt_id
           JOIN diagnoses d ON d.run_id=r.run_id
           LEFT JOIN corrections c ON c.run_id=r.run_id
           LEFT JOIN review_ranked v ON v.run_id=r.run_id AND v.row_number=1
           LEFT JOIN correction_ranked cv
             ON cv.attempt_id=a.attempt_id AND cv.model=r.model AND cv.row_number=1
           WHERE r.row_number=1""",
        (model, prompt_version),
    ).fetchall()

    groups: dict[tuple[str, str | None, str], dict] = defaultdict(
        lambda: {
            "attempts": set(), "errors": [], "weighted_error": 0.0,
            "weight": 0.0, "verified": 0, "types": Counter(), "evidence": [],
        }
    )
    for row in rows:
        points = canonicalize_knowledge_points(
            json.loads(row["knowledge_points_json"]),
            domain_code=row["domain_code"],
            subdomain_code=row["subdomain_code"],
        )
        month_weight = int(row["Time"][5:7])
        for point in points:
            key = (row["domain_code"], row["subdomain_code"], point)
            group = groups[key]
            group["attempts"].add(row["attempt_id"])
            group["weight"] += month_weight
            if row["has_error_pred"] == 1:
                group["errors"].append(row["attempt_id"])
                group["evidence"].append(row["attempt_id"])
                group["weighted_error"] += month_weight
                group["types"][row["error_type_pred"]] += 1
                group["verified"] += int(row["verification_status"] == "verified")

    source = f"vlm:{model}:{prompt_version}"
    now = datetime.now(timezone.utc).isoformat()
    connection.execute(
        "DELETE FROM mastery_states WHERE model=? AND prompt_version=?",
        (model, prompt_version),
    )
    count = 0
    for (domain, subdomain, point), group in groups.items():
        attempt_count = len(group["attempts"])
        error_count = len(set(group["errors"]))
        error_rate = error_count / attempt_count if attempt_count else 0.0
        smoothed_error_rate = (error_count + 1) / (attempt_count + 2)
        recent_error_rate = (
            group["weighted_error"] / group["weight"] if group["weight"] else 0.0
        )
        repeat_factor = min(1.0, error_count / 3)
        correction_gap = (
            1 - group["verified"] / error_count if error_count else 0.0
        )
        components = {
            "error_rate": round(error_rate, 4),
            "smoothed_error_rate": round(smoothed_error_rate, 4),
            "recent_weighted_error_rate": round(recent_error_rate, 4),
            "repeat_factor": round(repeat_factor, 4),
            "correction_gap": round(correction_gap, 4),
        }
        score = round(
            100 * (
                0.4 * smoothed_error_rate
                + 0.25 * recent_error_rate
                + 0.2 * repeat_factor
                + 0.15 * correction_gap
            ),
            2,
        )
        # The same VLM knowledge-point label can legitimately occur in multiple
        # curriculum subdomains. Include the complete grouping key so parallel
        # rebuilds cannot try to insert two rows with the same primary key.
        subdomain_token = _id_token(subdomain or "unknown")
        mastery_id = (
            f"mastery:{model}:{prompt_version}:{domain}:"
            f"{subdomain_token}:{_id_token(point)}"
        )
        evidence = sorted(set(group["evidence"]))
        components_json = json.dumps(components, ensure_ascii=False, sort_keys=True)
        error_types_json = json.dumps(
            dict(group["types"]), ensure_ascii=False, sort_keys=True
        )
        evidence_json = json.dumps(evidence, ensure_ascii=False)
        state_payload = {
            "attempt_count": attempt_count,
            "error_count": error_count,
            "verified_correction_count": group["verified"],
            "weakness_score": score,
            "score_components_json": components_json,
            "error_types_json": error_types_json,
            "evidence_attempt_ids_json": evidence_json,
            "status": "hypothesis",
        }
        state_hash = hashlib.sha256(
            json.dumps(state_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        connection.execute(
            """INSERT INTO mastery_states(
                 mastery_id,domain_code,subdomain_code,knowledge_point,model,
                 prompt_version,attempt_count,error_count,verified_correction_count,
                 weakness_score,score_components_json,error_types_json,
                 evidence_attempt_ids_json,status,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                mastery_id, domain, subdomain, point, model, prompt_version,
                attempt_count, error_count, group["verified"], score,
                components_json, error_types_json, evidence_json, "hypothesis", now,
            ),
        )
        latest = connection.execute(
            """SELECT state_hash FROM mastery_state_history
               WHERE mastery_id=? ORDER BY created_at DESC,snapshot_id DESC LIMIT 1""",
            (mastery_id,),
        ).fetchone()
        if latest is None or latest["state_hash"] != state_hash:
            connection.execute(
                """INSERT INTO mastery_state_history(
                     snapshot_id,mastery_id,domain_code,subdomain_code,
                     knowledge_point,model,prompt_version,attempt_count,error_count,
                     verified_correction_count,weakness_score,score_components_json,
                     error_types_json,evidence_attempt_ids_json,status,state_hash,
                     change_reason,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    f"mastery-snapshot:{uuid.uuid4().hex}", mastery_id, domain,
                    subdomain, point, model, prompt_version, attempt_count,
                    error_count, group["verified"], score, components_json,
                    error_types_json, evidence_json, "hypothesis", state_hash,
                    change_reason, now,
                ),
            )
        knowledge_id = f"knowledge:vlm:{_id_token(point)}"
        connection.execute(
            """INSERT INTO memory_nodes(node_id,node_type,label,properties_json)
               VALUES(?,?,?,?)
               ON CONFLICT(node_id) DO UPDATE SET label=excluded.label""",
            (knowledge_id, "KnowledgePoint", point, "{}"),
        )
        connection.execute(
            """INSERT INTO memory_nodes(node_id,node_type,label,properties_json)
               VALUES(?,?,?,?)
               ON CONFLICT(node_id) DO UPDATE SET
                 label=excluded.label,properties_json=excluded.properties_json""",
            (
                mastery_id, "MasteryState", point,
                json.dumps({"source": source, "score": score}, ensure_ascii=False),
            ),
        )
        connection.execute(
            "INSERT OR IGNORE INTO memory_edges(source_id,relation,target_id) VALUES(?,?,?)",
            (mastery_id, "SUMMARIZES", knowledge_id),
        )
        connection.execute(
            "DELETE FROM memory_edges WHERE source_id=? AND relation='SUPPORTED_BY'",
            (mastery_id,),
        )
        for attempt_id in evidence:
            connection.execute(
                "INSERT OR IGNORE INTO memory_edges(source_id,relation,target_id) VALUES(?,?,?)",
                (mastery_id, "SUPPORTED_BY", f"attempt:{attempt_id}"),
            )
        count += 1
    return count
