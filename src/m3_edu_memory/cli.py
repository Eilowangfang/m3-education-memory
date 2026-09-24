from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from .diagnosis import (
    diagnose_attempt,
    select_attempt_ids,
    select_stratified_attempt_ids,
)
from .database import connect, initialize
from .audit_render import render_audit_image
from .control import build_memory_report, plan_memory_query
from .evaluation import (
    compare_evaluation_runs,
    create_evaluation_suite,
    evaluate_model,
    get_evaluation_suite,
)
from .embeddings import create_embedding_client
from .exporter import export_query
from .importer import DEFAULT_REVISION, import_fermat
from .ingestion import process_next_job
from .operations import is_fatal_provider_error, routing_batch_status
from .query import MemoryQuery, query_attempts, query_vlm_memories
from .providers import create_vision_client
from .retrieval import build_memory_index, hybrid_search, memory_index_stats
from .memory_writer import rebuild_mastery_states
from .retrieval_evaluation import evaluate_retrieval
from .routing import (
    evaluate_routing_policy,
    load_routing_policy,
    materialize_existing_routing,
    run_routed_diagnosis,
    select_unrouted_attempt_ids,
)
from .stats import database_stats
from .time_rebase import rebase_simulated_times
from .verification import reverify_corrections
from .weakness import extract_vlm_weaknesses, extract_weaknesses
from .vlm import FixtureVisionClient, OpenAICompatibleVisionClient, PROMPT_VERSION


def _add_query_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", required=True)
    parser.add_argument("--domain")
    parser.add_argument("--subdomain")
    parser.add_argument("--errors-only", action="store_true")
    parser.add_argument("--time-from")
    parser.add_argument("--time-to")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)


def _memory_query(args) -> MemoryQuery:
    return MemoryQuery(
        domain_code=args.domain,
        subdomain_code=args.subdomain,
        errors_only=args.errors_only,
        time_from=args.time_from,
        time_to=args.time_to,
        limit=args.limit,
        offset=args.offset,
    )


def _configured_vision_client(args, *, command: str):
    if args.fixture_response:
        fixture = json.loads(Path(args.fixture_response).read_text(encoding="utf-8"))
        return FixtureVisionClient(fixture=fixture)
    if getattr(args, "provider_config", None) or getattr(args, "provider_profile", None):
        if not args.provider_config or not args.provider_profile:
            raise SystemExit("--provider-config and --provider-profile must be used together")
        return create_vision_client(
            args.provider_config, profile_name=args.provider_profile
        )
    if not args.base_url or not args.model:
        raise SystemExit(
            f"{command} requires a provider profile or both --base-url and --model"
        )
    return OpenAICompatibleVisionClient(
        base_url=args.base_url,
        model=args.model,
        api_key_env=args.api_key_env,
        timeout=args.timeout,
        use_json_mode=not args.no_json_mode,
        max_retries=args.max_retries,
    )


def _configured_embedding_client(args):
    config = getattr(args, "embedding_config", None)
    profile = getattr(args, "embedding_profile", None)
    if not config and not profile:
        return None
    if not config or not profile:
        raise SystemExit(
            "--embedding-config and --embedding-profile must be used together"
        )
    return create_embedding_client(config, profile_name=profile)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="m3-edu-memory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import-fermat")
    import_parser.add_argument("--dataset", required=True)
    import_parser.add_argument("--db", required=True)
    import_parser.add_argument("--revision", default=DEFAULT_REVISION)
    import_parser.add_argument("--simulation-year", type=int)
    import_parser.add_argument(
        "--simulation-anchor",
        help="Last day of the rolling seven-month simulation, in YYYY-MM-DD form.",
    )
    import_parser.add_argument("--timezone", default="Asia/Shanghai")
    import_parser.add_argument("--seed", default="m3-education-memory")
    import_parser.add_argument("--bootstrap-reference-memory", action="store_true")

    query_parser = subparsers.add_parser("query")
    _add_query_arguments(query_parser)

    export_parser = subparsers.add_parser("export")
    _add_query_arguments(export_parser)
    export_parser.add_argument("--output", required=True)

    stats_parser = subparsers.add_parser("stats")
    stats_parser.add_argument("--db", required=True)

    rebase_parser = subparsers.add_parser("rebase-times")
    rebase_parser.add_argument("--db", required=True)
    rebase_parser.add_argument("--anchor", required=True)
    rebase_parser.add_argument("--timezone", default="Asia/Shanghai")
    rebase_parser.add_argument("--seed", default="m3-education-memory")

    weakness_parser = subparsers.add_parser("weaknesses")
    weakness_parser.add_argument("--db", required=True)
    weakness_parser.add_argument("--domain")

    vlm_weakness_parser = subparsers.add_parser("weaknesses-vlm")
    vlm_weakness_parser.add_argument("--db", required=True)
    vlm_weakness_parser.add_argument("--model", required=True)
    vlm_weakness_parser.add_argument("--domain")

    diagnosis_parser = subparsers.add_parser("diagnose")
    diagnosis_parser.add_argument("--db", required=True)
    diagnosis_parser.add_argument("--attempt-id", action="append")
    diagnosis_parser.add_argument(
        "--evaluation-suite",
        help="Run exactly the attempt IDs stored in this evaluation suite.",
    )
    diagnosis_parser.add_argument("--domain")
    diagnosis_parser.add_argument("--limit", type=int, default=1)
    diagnosis_parser.add_argument("--stratified-pilot", action="store_true")
    diagnosis_parser.add_argument("--selection-seed", default="fermat-vlm-pilot-v1")
    diagnosis_parser.add_argument("--max-workers", type=int, default=1)
    diagnosis_parser.add_argument(
        "--result-output",
        help="Write the full batch result JSON here and print a compact summary.",
    )
    diagnosis_parser.add_argument("--base-url")
    diagnosis_parser.add_argument("--model")
    diagnosis_parser.add_argument("--provider-config")
    diagnosis_parser.add_argument("--provider-profile")
    diagnosis_parser.add_argument("--api-key-env", default="VLM_API_KEY")
    diagnosis_parser.add_argument("--timeout", type=int, default=180)
    diagnosis_parser.add_argument("--max-retries", type=int, default=3)
    diagnosis_parser.add_argument("--no-json-mode", action="store_true")
    diagnosis_parser.add_argument("--only-unprocessed", action="store_true")
    diagnosis_parser.add_argument(
        "--audit-output-dir",
        help="Render an audit PNG for each successful diagnosis with a correction.",
    )
    diagnosis_parser.add_argument(
        "--fixture-response",
        help="Local JSON fixture for integration testing; does not call a VLM.",
    )

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--db", required=True)
    evaluate_parser.add_argument("--model", required=True)
    evaluate_parser.add_argument("--provider")
    evaluate_parser.add_argument("--suite")
    evaluate_parser.add_argument("--persist", action="store_true")

    suite_parser = subparsers.add_parser("create-eval-suite")
    suite_parser.add_argument("--db", required=True)
    suite_parser.add_argument("--name", required=True)
    suite_parser.add_argument("--limit", type=int, default=100)
    suite_parser.add_argument("--seed", default="fermat-math-vlm-eval-v1")
    suite_parser.add_argument("--domain")
    suite_parser.add_argument("--description", default="")

    show_suite_parser = subparsers.add_parser("show-eval-suite")
    show_suite_parser.add_argument("--db", required=True)
    show_suite_parser.add_argument("--suite", required=True)

    compare_parser = subparsers.add_parser("compare-eval-suite")
    compare_parser.add_argument("--db", required=True)
    compare_parser.add_argument("--suite", required=True)

    routing_eval_parser = subparsers.add_parser("evaluate-routing")
    routing_eval_parser.add_argument("--db", required=True)
    routing_eval_parser.add_argument("--suite", required=True)
    routing_eval_parser.add_argument("--provider-config", required=True)
    routing_eval_parser.add_argument("--routing-config", required=True)

    routed_parser = subparsers.add_parser("diagnose-routed")
    routed_parser.add_argument("--db", required=True)
    routed_parser.add_argument("--provider-config", required=True)
    routed_parser.add_argument("--routing-config", required=True)
    routed_parser.add_argument("--attempt-id", action="append")
    routed_parser.add_argument("--evaluation-suite")
    routed_parser.add_argument("--limit", type=int, default=100)
    routed_parser.add_argument("--max-workers", type=int, default=1)
    routed_parser.add_argument("--only-unprocessed", action="store_true")
    routed_parser.add_argument("--result-output")
    routed_parser.add_argument(
        "--progress-output",
        help="Atomically updated JSON summary for monitoring a long routing batch.",
    )
    routed_parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Materialize routing from existing model runs without API calls.",
    )
    routed_parser.add_argument(
        "--defer-derived-refresh",
        action="store_true",
        help="Rebuild mastery and retrieval indexes once after a large batch.",
    )

    vlm_query_parser = subparsers.add_parser("query-vlm")
    vlm_query_parser.add_argument("--db", required=True)
    vlm_query_parser.add_argument("--model", required=True)
    vlm_query_parser.add_argument("--domain")
    vlm_query_parser.add_argument("--errors-only", action="store_true")
    vlm_query_parser.add_argument("--limit", type=int, default=100)
    vlm_query_parser.add_argument("--offset", type=int, default=0)

    audit_parser = subparsers.add_parser("render-audit")
    audit_parser.add_argument("--db", required=True)
    audit_parser.add_argument("--attempt-id", required=True)
    audit_parser.add_argument("--output", required=True)
    audit_parser.add_argument("--run-id")
    audit_parser.add_argument("--note")
    audit_parser.add_argument("--correction")

    answer_parser = subparsers.add_parser("answer")
    answer_parser.add_argument("--db", required=True)
    answer_parser.add_argument("--model", required=True)
    answer_request = answer_parser.add_mutually_exclusive_group(required=True)
    answer_request.add_argument("--request")
    answer_request.add_argument("--request-file", help="UTF-8 text file containing the user request.")
    answer_parser.add_argument("--output", required=True)
    answer_parser.add_argument("--default-limit", type=int, default=10)

    plan_parser = subparsers.add_parser("plan-query")
    plan_request = plan_parser.add_mutually_exclusive_group(required=True)
    plan_request.add_argument("--request")
    plan_request.add_argument("--request-file", help="UTF-8 text file containing the user request.")
    plan_parser.add_argument("--default-limit", type=int, default=10)

    index_parser = subparsers.add_parser("index-memory")
    index_parser.add_argument("--db", required=True)
    index_parser.add_argument("--model", required=True)
    index_parser.add_argument("--embedding-config")
    index_parser.add_argument("--embedding-profile")

    rebuild_parser = subparsers.add_parser("rebuild-derived-memory")
    rebuild_parser.add_argument("--db", required=True)
    rebuild_parser.add_argument("--model", required=True)
    rebuild_parser.add_argument("--prompt-version", default=PROMPT_VERSION)
    rebuild_parser.add_argument("--embedding-config")
    rebuild_parser.add_argument("--embedding-profile")
    rebuild_parser.add_argument("--change-reason", default="manual-derived-rebuild")
    rebuild_parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Rebuild normalized mastery states without replacing the existing vector index.",
    )

    reverify_parser = subparsers.add_parser("reverify-corrections")
    reverify_parser.add_argument("--db", required=True)
    reverify_parser.add_argument("--model")

    search_parser = subparsers.add_parser("search-memory")
    search_parser.add_argument("--db", required=True)
    search_parser.add_argument("--model", required=True)
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--limit", type=int, default=20)
    search_parser.add_argument("--embedding-config")
    search_parser.add_argument("--embedding-profile")
    search_parser.add_argument("--vector-weight", type=float, default=0.65)

    retrieval_eval_parser = subparsers.add_parser("evaluate-retrieval")
    retrieval_eval_parser.add_argument("--db", required=True)
    retrieval_eval_parser.add_argument("--model", required=True)
    retrieval_eval_parser.add_argument("--queries", required=True)
    retrieval_eval_parser.add_argument("--k", type=int, default=10)
    retrieval_eval_parser.add_argument("--vector-weight", type=float, default=0.65)
    retrieval_eval_parser.add_argument("--embedding-config")
    retrieval_eval_parser.add_argument("--embedding-profile")
    retrieval_eval_parser.add_argument("--output")
    retrieval_eval_parser.add_argument("--max-workers", type=int, default=1)

    index_stats_parser = subparsers.add_parser("index-stats")
    index_stats_parser.add_argument("--db", required=True)
    index_stats_parser.add_argument("--model", required=True)

    batch_status_parser = subparsers.add_parser("batch-status")
    batch_status_parser.add_argument("--db", required=True)
    batch_status_parser.add_argument("--routing-config", required=True)
    batch_status_parser.add_argument("--recent-minutes", type=int, default=30)
    batch_status_parser.add_argument("--output")

    worker_parser = subparsers.add_parser("process-jobs")
    worker_parser.add_argument("--db", required=True)
    worker_parser.add_argument("--model")
    worker_parser.add_argument("--base-url")
    worker_parser.add_argument("--provider-config")
    worker_parser.add_argument("--provider-profile")
    worker_parser.add_argument("--api-key-env", default="VLM_API_KEY")
    worker_parser.add_argument("--fixture-response")
    worker_parser.add_argument("--limit", type=int, default=1)
    worker_parser.add_argument("--timeout", type=int, default=180)
    worker_parser.add_argument("--max-retries", type=int, default=3)
    worker_parser.add_argument("--no-json-mode", action="store_true")
    worker_parser.add_argument("--worker-id")
    worker_parser.add_argument("--lease-seconds", type=int, default=600)
    worker_parser.add_argument("--retry-base-seconds", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    if args.command == "import-fermat":
        stats = import_fermat(
            dataset_dir=args.dataset,
            db_path=args.db,
            dataset_revision=args.revision,
            simulation_year=args.simulation_year,
            simulation_anchor=args.simulation_anchor,
            timezone_name=args.timezone,
            seed=args.seed,
            bootstrap_reference_memory=args.bootstrap_reference_memory,
        )
        print(json.dumps(stats.__dict__, ensure_ascii=False, indent=2))
        return 0
    if args.command == "query":
        print(json.dumps(query_attempts(args.db, _memory_query(args)), ensure_ascii=False, indent=2))
        return 0
    if args.command == "export":
        result = export_query(
            db_path=args.db,
            query=_memory_query(args),
            output_dir=args.output,
        )
        print(json.dumps({"total": result["total"], "returned": result["returned"], "output": args.output}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "stats":
        print(json.dumps(database_stats(args.db), ensure_ascii=False, indent=2))
        return 0
    if args.command == "rebase-times":
        print(json.dumps(
            rebase_simulated_times(
                args.db, anchor_date=args.anchor,
                timezone_name=args.timezone, seed=args.seed,
            ),
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    if args.command == "weaknesses":
        print(json.dumps(
            extract_weaknesses(args.db, domain_code=args.domain),
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    if args.command == "weaknesses-vlm":
        print(json.dumps(
            extract_vlm_weaknesses(
                args.db, model=args.model, domain_code=args.domain
            ),
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    if args.command == "diagnose":
        client = _configured_vision_client(args, command="diagnose")
        if args.attempt_id:
            attempt_ids = args.attempt_id
        elif args.evaluation_suite:
            attempt_ids = get_evaluation_suite(
                args.db, suite=args.evaluation_suite
            )["attempt_ids"]
            if args.only_unprocessed:
                connection = connect(args.db)
                attempt_ids = [
                    attempt_id for attempt_id in attempt_ids
                    if connection.execute(
                        """SELECT 1 FROM analysis_runs
                           WHERE attempt_id=? AND model=? AND status='completed'
                           LIMIT 1""",
                        (attempt_id, client.model),
                    ).fetchone() is None
                ]
                connection.close()
        elif args.stratified_pilot:
            attempt_ids = select_stratified_attempt_ids(
                args.db,
                limit=args.limit,
                seed=args.selection_seed,
                model=client.model if args.only_unprocessed else None,
            )
        else:
            attempt_ids = select_attempt_ids(
                args.db,
                domain_code=args.domain,
                limit=args.limit,
                only_unprocessed_model=client.model if args.only_unprocessed else None,
            )
        completed = []
        failed = []

        def process(attempt_id: str) -> tuple[str, dict | None, Exception | None]:
            try:
                result = diagnose_attempt(
                    args.db, attempt_id=attempt_id, client=client
                )
                if args.audit_output_dir and result.get("correction"):
                    try:
                        audit_path = Path(args.audit_output_dir) / f"{attempt_id}_audit.png"
                        result["audit_artifact"] = render_audit_image(
                            args.db,
                            attempt_id=attempt_id,
                            run_id=result["run_id"],
                            output_path=audit_path,
                        )
                    except Exception as audit_exc:
                        result["audit_error"] = str(audit_exc)
                return attempt_id, result, None
            except Exception as exc:
                return attempt_id, None, exc

        if args.max_workers < 1:
            raise SystemExit("--max-workers must be at least 1")
        if args.max_workers == 1:
            outcomes = [process(attempt_id) for attempt_id in attempt_ids]
        else:
            outcomes = []
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = {
                    executor.submit(process, attempt_id): attempt_id
                    for attempt_id in attempt_ids
                }
                for completed_count, future in enumerate(as_completed(futures), start=1):
                    outcomes.append(future.result())
                    print(
                        json.dumps(
                            {
                                "progress": completed_count,
                                "total": len(attempt_ids),
                                "attempt_id": futures[future],
                            }
                        ),
                        flush=True,
                    )
        for attempt_id, result, exc in outcomes:
            if exc is None:
                completed.append(result)
            else:
                failed.append({"attempt_id": attempt_id, "error": str(exc)})
        completed.sort(key=lambda item: item["attempt_id"])
        failed.sort(key=lambda item: item["attempt_id"])
        payload = {"model": client.model, "completed": completed, "failed": failed}
        if args.result_output:
            result_path = Path(args.result_output)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(
                {
                    "model": client.model,
                    "selected_count": len(attempt_ids),
                    "completed_count": len(completed),
                    "failed_count": len(failed),
                    "result_output": str(result_path.resolve()),
                },
                ensure_ascii=False,
                indent=2,
            ))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1 if failed else 0
    if args.command == "evaluate":
        print(json.dumps(
            evaluate_model(
                args.db, model=args.model, provider=args.provider,
                suite=args.suite, persist=args.persist
            ),
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    if args.command == "create-eval-suite":
        print(json.dumps(
            create_evaluation_suite(
                args.db,
                name=args.name,
                limit=args.limit,
                seed=args.seed,
                domain_code=args.domain,
                description=args.description,
            ),
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    if args.command == "show-eval-suite":
        print(json.dumps(
            get_evaluation_suite(args.db, suite=args.suite),
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    if args.command == "compare-eval-suite":
        print(json.dumps(
            compare_evaluation_runs(args.db, suite=args.suite),
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    if args.command == "evaluate-routing":
        policy = load_routing_policy(
            args.routing_config, provider_config=args.provider_config
        )
        print(json.dumps(
            evaluate_routing_policy(
                args.db, suite=args.suite, policy=policy
            ),
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    if args.command == "diagnose-routed":
        policy = load_routing_policy(
            args.routing_config, provider_config=args.provider_config
        )
        if args.attempt_id:
            attempt_ids = list(args.attempt_id)
        elif args.evaluation_suite:
            attempt_ids = get_evaluation_suite(
                args.db, suite=args.evaluation_suite
            )["attempt_ids"][:args.limit]
        elif args.only_unprocessed:
            attempt_ids = select_unrouted_attempt_ids(
                args.db, policy=policy, limit=args.limit
            )
        else:
            attempt_ids = select_attempt_ids(args.db, limit=args.limit)
        if args.max_workers < 1:
            raise SystemExit("--max-workers must be at least 1")

        def route(attempt_id: str):
            try:
                if args.existing_only:
                    result = materialize_existing_routing(
                        args.db, attempt_id=attempt_id, policy=policy,
                        refresh_derived=not args.defer_derived_refresh,
                    )
                else:
                    result = run_routed_diagnosis(
                        args.db,
                        attempt_id=attempt_id,
                        provider_config=args.provider_config,
                        policy=policy,
                        refresh_derived=not args.defer_derived_refresh,
                    )
                return attempt_id, result, None
            except Exception as exc:
                return attempt_id, None, exc

        fatal_error = None
        outcomes = []
        started = time.monotonic()
        progress_output = Path(args.progress_output) if args.progress_output else None

        def emit_progress(attempt_id: str | None, outcome, *, state: str = "running"):
            completed_count = sum(item[2] is None for item in outcomes)
            failed_count = len(outcomes) - completed_count
            elapsed_seconds = max(0.001, time.monotonic() - started)
            error = outcome[2] if outcome is not None else None
            payload = {
                "event": "routing_progress",
                "state": state,
                "processed_count": len(outcomes),
                "selected_count": len(attempt_ids),
                "completed_count": completed_count,
                "failed_count": failed_count,
                "unattempted_count": len(attempt_ids) - len(outcomes),
                "last_attempt_id": attempt_id,
                "last_outcome": (
                    "none" if outcome is None
                    else ("failed" if error is not None else "completed")
                ),
                "last_error": str(error)[:2000] if error is not None else None,
                "fatal_provider_error": bool(
                    error is not None and is_fatal_provider_error(error)
                ),
                "elapsed_seconds": round(elapsed_seconds, 3),
                "throughput_per_hour": round(
                    len(outcomes) * 3600.0 / elapsed_seconds, 2
                ),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            if progress_output is not None:
                progress_output.parent.mkdir(parents=True, exist_ok=True)
                temporary = progress_output.with_suffix(
                    progress_output.suffix + ".tmp"
                )
                temporary.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                for attempt in range(5):
                    try:
                        temporary.replace(progress_output)
                        break
                    except PermissionError:
                        if attempt == 4:
                            raise
                        time.sleep(0.1 * (attempt + 1))

        if args.max_workers == 1:
            for attempt_id in attempt_ids:
                outcome = route(attempt_id)
                outcomes.append(outcome)
                emit_progress(attempt_id, outcome)
                if outcome[2] is not None and is_fatal_provider_error(outcome[2]):
                    fatal_error = str(outcome[2])
                    break
        else:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                pending_ids = iter(attempt_ids)
                futures = {}
                for _ in range(min(args.max_workers, len(attempt_ids))):
                    attempt_id = next(pending_ids, None)
                    if attempt_id is not None:
                        futures[executor.submit(route, attempt_id)] = attempt_id
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        attempt_id = futures.pop(future)
                        outcome = future.result()
                        outcomes.append(outcome)
                        emit_progress(attempt_id, outcome)
                        if (
                            outcome[2] is not None
                            and is_fatal_provider_error(outcome[2])
                        ):
                            fatal_error = str(outcome[2])
                            break
                        next_attempt_id = next(pending_ids, None)
                        if next_attempt_id is not None:
                            futures[executor.submit(route, next_attempt_id)] = next_attempt_id
                    if fatal_error:
                        for future in futures:
                            future.cancel()
                        break
        emit_progress(
            outcomes[-1][0] if outcomes else None,
            outcomes[-1] if outcomes else None,
            state="aborted" if fatal_error else "completed",
        )
        completed = [result for _, result, exc in outcomes if exc is None]
        failed = [
            {"attempt_id": attempt_id, "error": str(exc)}
            for attempt_id, _, exc in outcomes if exc is not None
        ]
        payload = {
            "policy_name": policy.policy_name,
            "policy_version": policy.policy_version,
            "selected_count": len(attempt_ids),
            "completed_count": len(completed),
            "failed_count": len(failed),
            "aborted": fatal_error is not None,
            "fatal_error": fatal_error,
            "unattempted_count": len(attempt_ids) - len(outcomes),
            "escalated_count": sum(
                int(bool(item["escalated"])) for item in completed
            ),
            "completed": completed,
            "failed": failed,
        }
        if args.result_output:
            output = Path(args.result_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps({
                key: payload[key] for key in (
                    "policy_name", "policy_version", "selected_count",
                    "completed_count", "failed_count", "escalated_count",
                    "aborted", "fatal_error", "unattempted_count",
                )
            }, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1 if failed or fatal_error else 0
    if args.command == "query-vlm":
        print(json.dumps(
            query_vlm_memories(
                args.db,
                model=args.model,
                domain_code=args.domain,
                errors_only=args.errors_only,
                limit=args.limit,
                offset=args.offset,
            ),
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    if args.command == "render-audit":
        print(json.dumps(
            render_audit_image(
                args.db,
                attempt_id=args.attempt_id,
                output_path=args.output,
                run_id=args.run_id,
                note=args.note,
                correction=args.correction,
            ),
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    if args.command == "plan-query":
        request = (
            Path(args.request_file).read_text(encoding="utf-8").strip()
            if args.request_file else args.request
        )
        print(json.dumps(
            plan_memory_query(request, default_limit=args.default_limit).__dict__,
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    if args.command == "answer":
        request = (
            Path(args.request_file).read_text(encoding="utf-8").strip()
            if args.request_file else args.request
        )
        print(json.dumps(
            build_memory_report(
                args.db,
                request=request,
                model=args.model,
                output_dir=args.output,
                default_limit=args.default_limit,
            ),
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    if args.command == "index-memory":
        print(json.dumps(
            build_memory_index(
                args.db,
                model=args.model,
                embedding_client=_configured_embedding_client(args),
            ),
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    if args.command == "rebuild-derived-memory":
        connection = connect(args.db)
        initialize(connection)
        with connection:
            mastery_count = rebuild_mastery_states(
                connection,
                model=args.model,
                prompt_version=args.prompt_version,
                change_reason=args.change_reason,
            )
        connection.close()
        index = None if args.skip_index else build_memory_index(
            args.db,
            model=args.model,
            embedding_client=_configured_embedding_client(args),
        )
        print(json.dumps(
            {"model": args.model, "mastery_states": mastery_count, "index": index},
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    if args.command == "reverify-corrections":
        print(json.dumps(
            reverify_corrections(args.db, model=args.model),
            ensure_ascii=False, indent=2,
        ))
        return 0
    if args.command == "search-memory":
        print(json.dumps(
            hybrid_search(
                args.db, model=args.model, query=args.query, limit=args.limit,
                embedding_client=_configured_embedding_client(args),
                vector_weight=args.vector_weight,
            ),
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    if args.command == "evaluate-retrieval":
        result = evaluate_retrieval(
            args.db,
            model=args.model,
            queries_path=args.queries,
            k=args.k,
            embedding_client=_configured_embedding_client(args),
            vector_weight=args.vector_weight,
            max_workers=args.max_workers,
        )
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "index-stats":
        print(json.dumps(
            memory_index_stats(args.db, model=args.model),
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    if args.command == "batch-status":
        result = routing_batch_status(
            args.db,
            routing_config=args.routing_config,
            recent_minutes=args.recent_minutes,
        )
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "process-jobs":
        client = _configured_vision_client(args, command="process-jobs")
        processed = []
        for _ in range(max(1, args.limit)):
            result = process_next_job(
                args.db,
                client=client,
                worker_id=args.worker_id,
                lease_seconds=args.lease_seconds,
                retry_base_seconds=args.retry_base_seconds,
            )
            if result is None:
                break
            processed.append(result)
        print(json.dumps(
            {
                "model": client.model,
                "processed_count": len(processed),
                "completed_count": sum(item["status"] == "completed" for item in processed),
                "retry_scheduled_count": sum(item["status"] == "retry_scheduled" for item in processed),
                "failed_count": sum(item["status"] == "failed" for item in processed),
                "jobs": processed,
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 1 if any(item["status"] == "failed" for item in processed) else 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
