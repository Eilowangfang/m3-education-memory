from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .database import connect, initialize
from .diagnosis import diagnose_attempt
from .memory_writer import rebuild_mastery_states
from .providers import create_vision_client, load_provider_profiles
from .vlm import PROMPT_VERSION


_ROUTING_WRITE_LOCK = threading.Lock()


@dataclass(frozen=True)
class RoutingPolicy:
    policy_name: str
    policy_version: str
    initial_profile: str
    fallback_profile: str
    initial_model: str
    fallback_model: str
    model_alias: str = "seed-education-router-v1"
    confidence_threshold: float = 0.9
    escalate_on_requires_review: bool = True
    escalate_on_uncertain: bool = True


def load_routing_policy(
    routing_config: str | Path,
    *,
    provider_config: str | Path,
) -> RoutingPolicy:
    payload = json.loads(Path(routing_config).read_text(encoding="utf-8"))
    profiles = load_provider_profiles(provider_config)
    initial_profile = str(payload["initial_profile"])
    fallback_profile = str(payload["fallback_profile"])
    if initial_profile not in profiles or fallback_profile not in profiles:
        raise ValueError("Routing policy references an unknown provider profile")
    threshold = float(payload.get("confidence_threshold", 0.9))
    if not 0 <= threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    return RoutingPolicy(
        policy_name=str(payload["policy_name"]),
        policy_version=str(payload["policy_version"]),
        initial_profile=initial_profile,
        fallback_profile=fallback_profile,
        initial_model=str(profiles[initial_profile]["model"]),
        fallback_model=str(profiles[fallback_profile]["model"]),
        model_alias=str(payload.get("model_alias", "seed-education-router-v1")),
        confidence_threshold=threshold,
        escalate_on_requires_review=bool(
            payload.get("escalate_on_requires_review", True)
        ),
        escalate_on_uncertain=bool(payload.get("escalate_on_uncertain", True)),
    )


def _latest_completed(connection, *, attempt_id: str, model: str):
    return connection.execute(
        """SELECT r.run_id,r.model,r.created_at,d.has_error_pred,
                  d.error_type_pred,d.confidence,d.requires_review,
                  c.verification_status,
                  EXISTS(
                    SELECT 1 FROM diagnosis_steps s
                    WHERE s.run_id=r.run_id AND s.bbox_json IS NOT NULL
                  ) AS has_bbox
           FROM analysis_runs r
           JOIN diagnoses d USING(run_id)
           LEFT JOIN corrections c USING(run_id)
           WHERE r.attempt_id=? AND r.model=? AND r.status='completed'
           ORDER BY r.created_at DESC,r.run_id DESC LIMIT 1""",
        (attempt_id, model),
    ).fetchone()


def decide_route(
    db_path: str | Path,
    *,
    attempt_id: str,
    policy: RoutingPolicy,
) -> dict:
    connection = connect(db_path)
    initialize(connection)
    initial = _latest_completed(
        connection, attempt_id=attempt_id, model=policy.initial_model
    )
    connection.close()
    if initial is None:
        return {
            "attempt_id": attempt_id,
            "escalate": True,
            "selected_model": policy.fallback_model,
            "initial_run_id": None,
            "reasons": ["initial_result_unavailable"],
            "features": {"initial_available": False},
        }

    features = {
        "initial_available": True,
        "confidence": float(initial["confidence"]),
        "requires_review": bool(initial["requires_review"]),
        "has_error": (
            None
            if initial["has_error_pred"] is None
            else bool(initial["has_error_pred"])
        ),
        "error_type": initial["error_type_pred"],
        "verification_status": initial["verification_status"],
        "has_bbox": bool(initial["has_bbox"]),
    }
    reasons: list[str] = []
    if policy.escalate_on_requires_review and features["requires_review"]:
        reasons.append("initial_requires_review")
    if features["confidence"] < policy.confidence_threshold:
        reasons.append("confidence_below_threshold")
    if policy.escalate_on_uncertain and (
        features["has_error"] is None or features["error_type"] == "uncertain"
    ):
        reasons.append("uncertain_diagnosis")
    escalate = bool(reasons)
    return {
        "attempt_id": attempt_id,
        "escalate": escalate,
        "selected_model": (
            policy.fallback_model if escalate else policy.initial_model
        ),
        "initial_run_id": initial["run_id"],
        "reasons": reasons or ["initial_result_accepted"],
        "features": features,
    }


def record_routing_decision(
    db_path: str | Path,
    *,
    attempt_id: str,
    policy: RoutingPolicy,
    decision: dict,
    selected_run_id: str,
    fallback_run_id: str | None = None,
) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    routing_decision_id = f"routing:{uuid.uuid4()}"
    connection = connect(db_path)
    initialize(connection)
    with connection:
        connection.execute(
            """INSERT INTO routing_decisions(
                 routing_decision_id,attempt_id,policy_name,policy_version,
                 initial_model,fallback_model,initial_run_id,fallback_run_id,
                 selected_run_id,materialized_run_id,selected_model,escalated,
                 reasons_json,features_json,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,NULL,?,?,?,?,?)""",
            (
                routing_decision_id, attempt_id, policy.policy_name,
                policy.policy_version, policy.initial_model, policy.fallback_model,
                decision.get("initial_run_id"), fallback_run_id, selected_run_id,
                decision["selected_model"], int(bool(decision["escalate"])),
                json.dumps(decision["reasons"], ensure_ascii=False),
                json.dumps(decision["features"], ensure_ascii=False), now,
            ),
        )
    connection.close()
    return {
        "routing_decision_id": routing_decision_id,
        "attempt_id": attempt_id,
        "selected_run_id": selected_run_id,
        "selected_model": decision["selected_model"],
        "escalated": bool(decision["escalate"]),
        "reasons": decision["reasons"],
        "created_at": now,
    }


def materialize_routing_view(
    db_path: str | Path,
    *,
    routing_decision_id: str,
    policy: RoutingPolicy,
    refresh_derived: bool = True,
) -> dict:
    connection = connect(db_path)
    initialize(connection)
    decision = connection.execute(
        "SELECT * FROM routing_decisions WHERE routing_decision_id=?",
        (routing_decision_id,),
    ).fetchone()
    if decision is None:
        connection.close()
        raise KeyError(f"Unknown routing decision: {routing_decision_id}")
    source = connection.execute(
        """SELECT r.*,d.question_transcription,d.solution_transcription,
                  d.has_error_pred,d.error_type_pred,d.first_error_step,
                  d.error_explanation,d.knowledge_points_json,d.confidence,
                  d.requires_review
           FROM analysis_runs r JOIN diagnoses d USING(run_id)
           WHERE r.run_id=? AND r.status='completed'""",
        (decision["selected_run_id"],),
    ).fetchone()
    if source is None:
        connection.close()
        raise RuntimeError("Selected routing source run is unavailable")
    source_correction = connection.execute(
        "SELECT * FROM corrections WHERE run_id=?", (source["run_id"],)
    ).fetchone()
    source_steps = connection.execute(
        "SELECT * FROM diagnosis_steps WHERE run_id=? ORDER BY step_index",
        (source["run_id"],),
    ).fetchall()
    route_run_id = f"routing-run:{routing_decision_id.removeprefix('routing:')}"
    # Keep the diagnosis prompt identity compatible with existing query and
    # mastery APIs; routing provenance is versioned in routing_decisions.
    prompt_version = PROMPT_VERSION
    created_at = datetime.now(timezone.utc).isoformat()
    with _ROUTING_WRITE_LOCK, connection:
        connection.execute(
            """INSERT INTO analysis_runs(
                 run_id,attempt_id,provider,model,prompt_version,status,
                 raw_response,parsed_json,error_message,input_image_bytes,
                 input_width,input_height,input_mime_type,preprocessing_json,
                 latency_ms,provider_usage_json,request_attempts,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                route_run_id, source["attempt_id"], "routing-policy",
                policy.model_alias, prompt_version, "completed",
                json.dumps(
                    {
                        "routing_decision_id": routing_decision_id,
                        "source_run_id": source["run_id"],
                        "source_model": source["model"],
                    },
                    ensure_ascii=False,
                ),
                source["parsed_json"], None, source["input_image_bytes"],
                source["input_width"], source["input_height"],
                source["input_mime_type"], source["preprocessing_json"],
                source["latency_ms"], source["provider_usage_json"],
                source["request_attempts"], created_at,
            ),
        )
        connection.execute(
            """INSERT INTO diagnoses(
                 diagnosis_id,run_id,attempt_id,question_transcription,
                 solution_transcription,has_error_pred,error_type_pred,
                 first_error_step,error_explanation,knowledge_points_json,
                 confidence,requires_review)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                f"diagnosis:{route_run_id}", route_run_id, source["attempt_id"],
                source["question_transcription"], source["solution_transcription"],
                source["has_error_pred"], source["error_type_pred"],
                source["first_error_step"], source["error_explanation"],
                source["knowledge_points_json"], source["confidence"],
                source["requires_review"],
            ),
        )
        for step in source_steps:
            connection.execute(
                """INSERT INTO diagnosis_steps(
                     run_id,step_index,transcription,normalized_latex,bbox_json,
                     is_error,confidence) VALUES(?,?,?,?,?,?,?)""",
                (
                    route_run_id, step["step_index"], step["transcription"],
                    step["normalized_latex"], step["bbox_json"], step["is_error"],
                    step["confidence"],
                ),
            )
        if source_correction is not None:
            correction_id = f"correction:{route_run_id}"
            connection.execute(
                """INSERT INTO corrections(
                     correction_id,run_id,attempt_id,corrected_solution,
                     corrected_steps_json,final_answer,verification_expression,
                     verification_status,verification_method,
                     verification_details_json,confidence,rendered_path,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    correction_id, route_run_id, source["attempt_id"],
                    source_correction["corrected_solution"],
                    source_correction["corrected_steps_json"],
                    source_correction["final_answer"],
                    source_correction["verification_expression"],
                    source_correction["verification_status"],
                    source_correction["verification_method"],
                    source_correction["verification_details_json"],
                    source_correction["confidence"],
                    source_correction["rendered_path"], created_at,
                ),
            )
            connection.execute(
                """UPDATE correction_versions SET status='superseded'
                   WHERE attempt_id=? AND model=? AND status='active'""",
                (source["attempt_id"], policy.model_alias),
            )
            connection.execute(
                """INSERT INTO correction_versions(
                     version_id,attempt_id,run_id,model,source,parent_version_id,
                     corrected_solution,corrected_steps_json,final_answer,
                     verification_expression,verification_status,
                     verification_method,verification_details_json,confidence,
                     status,created_by,rendered_path,created_at)
                   VALUES(?,?,?,?,?,NULL,?,?,?,?,?,?,?,?,?,'routing-policy',?,?)""",
                (
                    f"correction-version:{route_run_id}", source["attempt_id"],
                    route_run_id, policy.model_alias, "rule",
                    source_correction["corrected_solution"],
                    source_correction["corrected_steps_json"],
                    source_correction["final_answer"],
                    source_correction["verification_expression"],
                    source_correction["verification_status"],
                    source_correction["verification_method"],
                    source_correction["verification_details_json"],
                    source_correction["confidence"], "active",
                    source_correction["rendered_path"], created_at,
                ),
            )
        episode_source = f"routing:{policy.policy_name}:{policy.policy_version}"
        existing_episode = connection.execute(
            """SELECT memory_id FROM episodic_memories
               WHERE attempt_id=? AND source=?""",
            (source["attempt_id"], episode_source),
        ).fetchone()
        memory_id = (
            existing_episode["memory_id"]
            if existing_episode is not None
            else f"episode:{source['attempt_id']}:{route_run_id}"
        )
        content = (
            f"Routed diagnosis selected {source['model']} for attempt "
            f"{source['attempt_id']}: has_error={source['has_error_pred']}, "
            f"type={source['error_type_pred']}. {source['error_explanation']}"
        )
        connection.execute(
            """INSERT INTO episodic_memories(
                 memory_id,attempt_id,content,error_type,status,source,created_at)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(attempt_id,source) DO UPDATE SET
                 content=excluded.content,
                 error_type=excluded.error_type,
                 status=excluded.status,
                 created_at=excluded.created_at""",
            (
                memory_id, source["attempt_id"], content,
                source["error_type_pred"],
                "hypothesis_requires_review" if source["requires_review"] else "hypothesis",
                episode_source, created_at,
            ),
        )
        connection.execute(
            """INSERT INTO memory_nodes(node_id,node_type,label,properties_json)
               VALUES(?,?,?,?)
               ON CONFLICT(node_id) DO UPDATE SET
                 node_type=excluded.node_type,
                 label=excluded.label,
                 properties_json=excluded.properties_json""",
            (
                memory_id, "ErrorEpisode", source["error_type_pred"],
                json.dumps(
                    {
                        "routing_decision_id": routing_decision_id,
                        "source_run_id": source["run_id"],
                        "run_id": route_run_id,
                    },
                    ensure_ascii=False,
                ),
            ),
        )
        connection.execute(
            """INSERT OR IGNORE INTO memory_edges(
                 source_id,relation,target_id) VALUES(?,?,?)""",
            (f"attempt:{source['attempt_id']}", "HAS_ROUTED_DIAGNOSIS", memory_id),
        )
        connection.execute(
            "UPDATE routing_decisions SET materialized_run_id=? WHERE routing_decision_id=?",
            (route_run_id, routing_decision_id),
        )
        if refresh_derived:
            rebuild_mastery_states(
                connection,
                model=policy.model_alias,
                prompt_version=prompt_version,
                change_reason=f"routing:{routing_decision_id}",
            )
    connection.close()
    if refresh_derived:
        try:
            from .retrieval import index_memory_attempt

            index_update = index_memory_attempt(
                db_path, model=policy.model_alias, attempt_id=source["attempt_id"]
            )
        except Exception as exc:
            index_update = {"error": str(exc)}
    else:
        index_update = {"deferred": True}
    return {
        "routing_decision_id": routing_decision_id,
        "attempt_id": source["attempt_id"],
        "source_run_id": source["run_id"],
        "source_model": source["model"],
        "materialized_run_id": route_run_id,
        "model_alias": policy.model_alias,
        "index_update": index_update,
    }


def run_routed_diagnosis(
    db_path: str | Path,
    *,
    attempt_id: str,
    provider_config: str | Path,
    policy: RoutingPolicy,
    reuse_existing: bool = True,
    refresh_derived: bool = True,
) -> dict:
    connection = connect(db_path)
    initialize(connection)
    initial = _latest_completed(
        connection, attempt_id=attempt_id, model=policy.initial_model
    )
    connection.close()
    initial_result = None
    initial_error = None
    if initial is None or not reuse_existing:
        try:
            initial_result = diagnose_attempt(
                db_path,
                attempt_id=attempt_id,
                client=create_vision_client(
                    provider_config, profile_name=policy.initial_profile
                ),
                refresh_derived=refresh_derived,
            )
        except Exception as exc:
            initial_error = str(exc)

    decision = decide_route(db_path, attempt_id=attempt_id, policy=policy)
    if initial_error:
        decision["reasons"] = ["initial_call_failed"]
        decision["features"]["initial_error"] = initial_error
        decision["escalate"] = True
        decision["selected_model"] = policy.fallback_model

    fallback_result = None
    fallback_run_id = None
    if decision["escalate"]:
        connection = connect(db_path)
        fallback = _latest_completed(
            connection, attempt_id=attempt_id, model=policy.fallback_model
        )
        connection.close()
        if fallback is None or not reuse_existing:
            fallback_result = diagnose_attempt(
                db_path,
                attempt_id=attempt_id,
                client=create_vision_client(
                    provider_config, profile_name=policy.fallback_profile
                ),
                refresh_derived=refresh_derived,
            )
            fallback_run_id = fallback_result["run_id"]
        else:
            fallback_run_id = fallback["run_id"]

    selected_run_id = fallback_run_id or decision.get("initial_run_id")
    if selected_run_id is None and initial_result:
        selected_run_id = initial_result["run_id"]
    if selected_run_id is None:
        raise RuntimeError("Routing produced no completed diagnosis")
    recorded = record_routing_decision(
        db_path,
        attempt_id=attempt_id,
        policy=policy,
        decision=decision,
        selected_run_id=selected_run_id,
        fallback_run_id=fallback_run_id,
    )
    materialized = materialize_routing_view(
        db_path,
        routing_decision_id=recorded["routing_decision_id"],
        policy=policy,
        refresh_derived=refresh_derived,
    )
    return {
        **recorded,
        "materialized": materialized,
        "initial_result": initial_result,
        "fallback_result": fallback_result,
    }


def materialize_existing_routing(
    db_path: str | Path,
    *,
    attempt_id: str,
    policy: RoutingPolicy,
    refresh_derived: bool = True,
) -> dict:
    decision = decide_route(db_path, attempt_id=attempt_id, policy=policy)
    connection = connect(db_path)
    initialize(connection)
    selected = _latest_completed(
        connection, attempt_id=attempt_id, model=decision["selected_model"]
    )
    fallback_run_id = None
    if decision["escalate"] and selected is not None:
        fallback_run_id = selected["run_id"]
    if selected is None:
        selected = _latest_completed(
            connection, attempt_id=attempt_id, model=policy.initial_model
        )
        if selected is not None:
            decision["selected_model"] = policy.initial_model
            decision["reasons"].append("fallback_unavailable_used_initial")
    connection.close()
    if selected is None:
        raise RuntimeError(f"No completed model result for {attempt_id}")
    recorded = record_routing_decision(
        db_path,
        attempt_id=attempt_id,
        policy=policy,
        decision=decision,
        selected_run_id=selected["run_id"],
        fallback_run_id=fallback_run_id,
    )
    materialized = materialize_routing_view(
        db_path,
        routing_decision_id=recorded["routing_decision_id"],
        policy=policy,
        refresh_derived=refresh_derived,
    )
    return {**recorded, "materialized": materialized}


def evaluate_routing_policy(
    db_path: str | Path,
    *,
    suite: str,
    policy: RoutingPolicy,
) -> dict:
    connection = connect(db_path)
    initialize(connection)
    suite_row = connection.execute(
        "SELECT attempt_ids_json FROM evaluation_suites WHERE name=?", (suite,)
    ).fetchone()
    if suite_row is None:
        connection.close()
        raise KeyError(f"Unknown evaluation suite: {suite}")
    attempt_ids = json.loads(suite_row["attempt_ids_json"])
    truth = {
        row["attempt_id"]: bool(row["has_error"])
        for row in connection.execute(
            "SELECT attempt_id,has_error FROM benchmark_truth"
        )
        if row["attempt_id"] in set(attempt_ids)
    }
    connection.close()

    predictions: dict[str, bool] = {}
    escalated = pro_used = 0
    reasons: dict[str, int] = {}
    for attempt_id in attempt_ids:
        decision = decide_route(db_path, attempt_id=attempt_id, policy=policy)
        if decision["escalate"]:
            escalated += 1
        for reason in decision["reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
        connection = connect(db_path)
        selected = _latest_completed(
            connection,
            attempt_id=attempt_id,
            model=decision["selected_model"],
        )
        if selected is None and decision["escalate"]:
            selected = _latest_completed(
                connection, attempt_id=attempt_id, model=policy.initial_model
            )
        connection.close()
        if selected is not None and selected["has_error_pred"] is not None:
            predictions[attempt_id] = bool(selected["has_error_pred"])
            if selected["model"] == policy.fallback_model:
                pro_used += 1

    tp = tn = fp = fn = 0
    for attempt_id, prediction in predictions.items():
        actual = truth[attempt_id]
        if prediction and actual:
            tp += 1
        elif prediction and not actual:
            fp += 1
        elif not prediction and not actual:
            tn += 1
        else:
            fn += 1
    count = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "policy_name": policy.policy_name,
        "policy_version": policy.policy_version,
        "suite": suite,
        "suite_attempts": len(attempt_ids),
        "prediction_count": count,
        "coverage": count / len(attempt_ids) if attempt_ids else 0.0,
        "accuracy": (tp + tn) / count if count else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        ),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "escalation_rate": escalated / len(attempt_ids),
        "fallback_use_rate": pro_used / len(attempt_ids),
        "reasons": reasons,
    }


def select_unrouted_attempt_ids(
    db_path: str | Path,
    *,
    policy: RoutingPolicy,
    limit: int,
) -> list[str]:
    connection = connect(db_path)
    initialize(connection)
    rows = connection.execute(
        """SELECT a.attempt_id FROM attempts a
           WHERE NOT EXISTS(
             SELECT 1 FROM routing_decisions rd
             WHERE rd.attempt_id=a.attempt_id
               AND rd.policy_name=? AND rd.policy_version=?
               AND rd.materialized_run_id IS NOT NULL
           )
           ORDER BY a.Time,a.attempt_id LIMIT ?""",
        (policy.policy_name, policy.policy_version, limit),
    ).fetchall()
    connection.close()
    return [row["attempt_id"] for row in rows]
