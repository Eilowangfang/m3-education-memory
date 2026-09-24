from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from m3_edu_memory.control import build_memory_report
from m3_edu_memory.database import connect, initialize
from m3_edu_memory.embeddings import create_embedding_client
from m3_edu_memory.memory_writer import rebuild_mastery_states
from m3_edu_memory.operations import routing_batch_status
from m3_edu_memory.retrieval import build_memory_index, memory_index_stats
from m3_edu_memory.retrieval_evaluation import (
    evaluate_retrieval,
    load_retrieval_queries,
)
from m3_edu_memory.routing import load_routing_policy
from m3_edu_memory.verification import reverify_corrections
from m3_edu_memory.vlm import PROMPT_VERSION


QUERIES = {
    "calculus": "帮我整理过去七个月的微积分错题",
    "algebra": "帮我整理过去七个月的代数错题",
    "geometry": "帮我整理过去七个月的几何错题",
    "trigonometry": "帮我整理过去七个月的三角学错题",
    "probability": "帮我整理过去七个月的概率统计错题",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
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


def process_exists(pid: int | None) -> bool:
    if not pid:
        return False
    completed = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False,
    )
    return str(pid) in completed.stdout


def build_acceptance(
    *,
    coverage: dict,
    index: dict,
    embedding_model: str,
    retrieval: dict,
    reports: dict,
    expected_query_count: int,
) -> dict:
    total = int(coverage["total_attempts"])
    embedding_counts = {
        item["embedding_model"]: int(item["count"])
        for item in index.get("embeddings", [])
    }
    expected_reports = set(QUERIES)
    actual_reports = set(reports)
    report_results_present = all(
        int(item.get("matched", 0)) > 0 and int(item.get("returned", 0)) > 0
        for item in reports.values()
    )
    galleries_present = all(
        Path(str(item.get("gallery_html", ""))).is_file()
        for item in reports.values()
    )
    checks = {
        "full_memory_coverage": {
            "expected": total,
            "actual": int(coverage["materialized_attempts"]),
            "passed": (
                int(coverage["materialized_attempts"]) == total
                and int(coverage["remaining_attempts"]) == 0
            ),
        },
        "memory_documents": {
            "expected": total,
            "actual": int(index.get("documents", 0)),
            "passed": int(index.get("documents", 0)) == total,
        },
        "vision_embeddings": {
            "expected": total,
            "actual": embedding_counts.get(embedding_model, 0),
            "embedding_model": embedding_model,
            "passed": embedding_counts.get(embedding_model, 0) == total,
        },
        "retrieval_queries": {
            "expected": expected_query_count,
            "actual": int(retrieval.get("query_count", 0)),
            "evaluated": int(retrieval.get("evaluated_query_count", 0)),
            "passed": (
                int(retrieval.get("query_count", 0)) == expected_query_count
                and int(retrieval.get("evaluated_query_count", 0))
                == expected_query_count
            ),
        },
        "subject_reports": {
            "expected": sorted(expected_reports),
            "actual": sorted(actual_reports),
            "passed": actual_reports == expected_reports and report_results_present,
        },
        "report_galleries": {
            "expected": len(expected_reports),
            "actual": sum(
                Path(str(item.get("gallery_html", ""))).is_file()
                for item in reports.values()
            ),
            "passed": actual_reports == expected_reports and galleries_present,
        },
    }
    passed = all(item["passed"] for item in checks.values())
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "checks": checks,
    }


def wait_for_routing(args, final_status: Path) -> dict:
    while True:
        live = routing_batch_status(
            args.db,
            routing_config=args.routing_config,
            recent_minutes=60,
        )
        batch = {}
        if Path(args.batch_status).exists():
            try:
                batch = json.loads(Path(args.batch_status).read_text(encoding="utf-8"))
            except Exception:
                batch = {}
        write_json(final_status, {
            "status": "waiting",
            "stage": "routing",
            "updated_at": now(),
            "batch": batch,
            "coverage": live,
        })
        if live["remaining_attempts"] == 0 and batch.get("status") in {
            "completed", "failed", "partial"
        }:
            return live
        if (
            batch.get("status") == "running"
            and not process_exists(batch.get("pid"))
        ):
            raise RuntimeError(
                "Seed batch process stopped before routing coverage reached 100%"
            )
        time.sleep(args.poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/seed_memory.db")
    parser.add_argument("--routing-config", default="config/model_routing.json")
    parser.add_argument("--provider-config", default="config/vlm_providers.json")
    parser.add_argument(
        "--embedding-config", default="config/embedding_providers.json"
    )
    parser.add_argument(
        "--embedding-profile", default="volcengine-doubao-embedding-vision"
    )
    parser.add_argument(
        "--queries", default="config/retrieval_eval_queries.json"
    )
    parser.add_argument(
        "--batch-status", default="outputs/seed-main-memory/status.json"
    )
    parser.add_argument("--output", default="outputs/seed-main-memory/finalization")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    args.poll_seconds = max(10, min(args.poll_seconds, 300))

    root = Path(args.output)
    status_path = root / "status.json"
    policy = load_routing_policy(
        args.routing_config, provider_config=args.provider_config
    )
    try:
        coverage = wait_for_routing(args, status_path)
        write_json(status_path, {
            "status": "running", "stage": "reverify-corrections",
            "updated_at": now(), "coverage": coverage,
        })
        verification = reverify_corrections(args.db, model=policy.model_alias)

        write_json(status_path, {
            "status": "running", "stage": "rebuild-mastery",
            "updated_at": now(), "coverage": coverage,
            "verification": verification,
        })
        connection = connect(args.db)
        initialize(connection)
        with connection:
            mastery_count = rebuild_mastery_states(
                connection,
                model=policy.model_alias,
                prompt_version=PROMPT_VERSION,
                change_reason="seed-main-post-batch-finalization",
            )
        connection.close()

        embedding_client = create_embedding_client(
            args.embedding_config, profile_name=args.embedding_profile
        )
        index = memory_index_stats(args.db, model=policy.model_alias)
        expected = coverage["materialized_attempts"]
        embedding_counts = {
            item["embedding_model"]: item["count"] for item in index["embeddings"]
        }
        if (
            index["documents"] != expected
            or embedding_counts.get(embedding_client.model) != expected
        ):
            write_json(status_path, {
                "status": "running", "stage": "repair-index",
                "updated_at": now(), "coverage": coverage,
                "verification": verification, "index_before": index,
            })
            index = build_memory_index(
                args.db, model=policy.model_alias,
                embedding_client=embedding_client,
                max_workers=8,
            )

        write_json(status_path, {
            "status": "running", "stage": "retrieval-evaluation",
            "updated_at": now(), "coverage": coverage,
            "verification": verification, "index": index,
        })
        retrieval = evaluate_retrieval(
            args.db,
            model=policy.model_alias,
            queries_path=args.queries,
            k=10,
            embedding_client=embedding_client,
            vector_weight=1.0,
            max_workers=4,
        )
        write_json(root / "retrieval-evaluation.json", retrieval)

        reports = {}
        for name, request in QUERIES.items():
            report = build_memory_report(
                args.db,
                request=request,
                model=policy.model_alias,
                output_dir=root / "reports" / name,
                default_limit=8,
            )
            reports[name] = {
                "request": request,
                "matched": report["matched"],
                "returned": report["returned"],
                "gallery_html": report["gallery_html"],
            }

        write_json(status_path, {
            "status": "running", "stage": "acceptance",
            "updated_at": now(), "coverage": coverage,
            "verification": verification, "index": index,
            "retrieval_macro_metrics": retrieval["macro_metrics"],
            "reports": reports,
        })
        acceptance = build_acceptance(
            coverage=coverage,
            index=index,
            embedding_model=embedding_client.model,
            retrieval=retrieval,
            reports=reports,
            expected_query_count=len(load_retrieval_queries(args.queries)),
        )

        summary = {
            "status": "completed" if acceptance["passed"] else "failed",
            "stage": "completed" if acceptance["passed"] else "acceptance-failed",
            "completed_at": now(),
            "coverage": coverage,
            "verification": verification,
            "mastery_states": mastery_count,
            "index": index,
            "retrieval_macro_metrics": retrieval["macro_metrics"],
            "reports": reports,
            "acceptance": acceptance,
        }
        write_json(root / "summary.json", summary)
        write_json(status_path, summary)
        return 0 if acceptance["passed"] else 2
    except Exception as exc:
        write_json(status_path, {
            "status": "failed", "stage": "failed", "updated_at": now(),
            "error_type": type(exc).__name__, "error": str(exc),
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
