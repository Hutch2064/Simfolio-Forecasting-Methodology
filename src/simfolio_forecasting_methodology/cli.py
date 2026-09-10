"""Canonical-only command-line interface for the standalone OOS package."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .catalogue import (
    EXPECTED_CANONICAL_COUNT,
    EXPECTED_CELLS_PER_MODEL,
    EXPECTED_MEMBERSHIP_DIGEST,
    canonical_model,
    frontier_model_id,
    load_canonical_ledger,
    load_canonical_models,
)
from .data import write_canonical_data
from .experiment import build_experiment_plan, iter_smoke_tasks
from .models.registry import build_model, registration
from .protocol import CANONICAL_DENSE_PROTOCOL
from .runner import execute_model_checkpointed
from .results.retained import retained_score_report


CANONICAL_COMMAND = "canonical-175"


def _plan_payload() -> dict[str, Any]:
    protocol = asdict(CANONICAL_DENSE_PROTOCOL)
    return {
        "scope": "canonical_175_only",
        "membership_digest": EXPECTED_MEMBERSHIP_DIGEST,
        "model_count": EXPECTED_CANONICAL_COUNT,
        "model_ids": [row["public_model_id"] for row in load_canonical_models()],
        "protocol": protocol,
        "origin_tasks_per_model": CANONICAL_DENSE_PROTOCOL.total_origin_tasks,
        "scored_cells_per_model": EXPECTED_CELLS_PER_MODEL,
        "executor": {
            "task_identity": "SHA-256 of model, simulation, data, origin, and seed inputs",
            "checkpoint_resume": "manifest fingerprint must match exactly",
            "aggregation": "fixed denominator over every portfolio-horizon cell",
        },
    }


def _catalogue_payload() -> dict[str, Any]:
    ledger = load_canonical_ledger()
    return {
        "scope": ledger["scope"],
        "membership_digest": ledger["membership"]["membership_digest"],
        "model_count": ledger["membership"]["count"],
        "rows": [
            {
                "public_model_id": row["public_model_id"],
                "display_name": row["display_name"],
                "canonical_rank": row["canonical_rank"],
                "historical_rank": row["historical_rank"],
                "retained_score_token": row["historical_score"]["exact_empirical_crps"],
                "publication_score_token": row["score_precision"]["publication_token"],
                "verification_status": row["verification_status"],
            }
            for row in ledger["models"]
        ],
    }


def _coverage_payload() -> dict[str, Any]:
    ledger = load_canonical_ledger()
    models = ledger["models"]
    bool_fields = (
        "identity_recovered",
        "specification_recovered",
        "source_reference_verified",
        "implementation_available",
        "instantiation_validated",
        "forecast_smoke_tested",
        "source_parity_checked",
        "historical_score_verified",
    )
    counts = {field: sum(bool(row[field]) for row in models) for field in bool_fields}
    counts["full_statistical_specifications_confirmed"] = ledger["identity_policy"][
        "full_statistical_specifications_confirmed"
    ]
    counts["verified_protocol_fingerprints"] = sum(
        row["protocol_fingerprint"] is not None for row in models
    )
    counts["verified_dataset_fingerprints"] = sum(
        row["dataset_fingerprint"] is not None for row in models
    )
    counts["verified_panel_fingerprints"] = sum(
        row["panel_fingerprint"] is not None for row in models
    )
    counts["registered_factories"] = sum(
        row["implementation_factory"].get("callable", False) for row in models
    )
    return {
        "scope": "canonical_175_only",
        "model_count": len(models),
        "membership_digest": ledger["membership"]["membership_digest"],
        "counts": counts,
        "verification_status_counts": {
            status: sum(row["verification_status"] == status for row in models)
            for status in sorted({row["verification_status"] for row in models})
        },
        "retained_score_artifact_digest_status": ledger["score_evidence"][
            "retained_score_artifact"
        ]["digest_status"],
    }


def _add_execution_arguments(
    command: argparse.ArgumentParser, *, smoke_default: bool = False
) -> None:
    command.add_argument(
        "--plan", action="store_true", help="Print the canonical protocol plan only."
    )
    command.add_argument("--data", type=Path, default=Path(".simfolio-oos-data"))
    selection = command.add_mutually_exclusive_group()
    selection.add_argument("--model", default="", help="Run one exact canonical model ID.")
    selection.add_argument("--frontier", action="store_true", help="Select canonical rank one.")
    selection.add_argument("--all", action="store_true", help="Select all 175 canonical IDs.")
    command.add_argument("--smoke", action="store_true", help="Use bounded smoke tasks.")
    command.add_argument("--simulations", type=int, default=32 if smoke_default else 240)
    command.add_argument("--workers", type=int, default=1)
    command.add_argument(
        "--checkpoint",
        "--output",
        dest="checkpoint",
        type=Path,
        default=Path("results/canonical-175"),
        help="Directory containing per-model checkpoints and run summaries.",
    )
    command.add_argument(
        "--resume", action="store_true", help="Resume an identical checkpoint manifest."
    )
    command.add_argument("--task-count", type=int, default=2)
    command.add_argument("--horizon", type=int, default=63)
    command.add_argument("--json", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simfolio-oos",
        description="Inspect or execute the canonical-175 forecasting OOS study.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="Build public-source canonical research inputs.")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    build = data_sub.add_parser("build")
    build.add_argument("--cache", type=Path, default=Path(".simfolio-oos-data"))
    build.add_argument("--refresh", action="store_true")

    canonical = sub.add_parser(
        CANONICAL_COMMAND, help="Execute the canonical-175 task constructor."
    )
    _add_execution_arguments(canonical)

    smoke = sub.add_parser("smoke", help="Execute bounded canonical smoke tasks.")
    _add_execution_arguments(smoke, smoke_default=True)

    catalogue = sub.add_parser("catalogue", help="Inspect exact canonical-175 membership.")
    catalogue.add_argument("--json", action="store_true")
    coverage = sub.add_parser("coverage", help="Inspect canonical evidence coverage.")
    coverage.add_argument("--json", action="store_true")
    plan = sub.add_parser("plan", help="Inspect the canonical protocol and task plan.")
    plan.add_argument("--json", action="store_true")
    scores = sub.add_parser("scores", help="Inspect retained historical score evidence.")
    scores.add_argument("--model", default="", help="Inspect one exact canonical model ID.")
    scores.add_argument("--json", action="store_true")
    return parser


def _selected_models(args: argparse.Namespace, *, command: str) -> tuple[str, ...]:
    canonical_ids = tuple(row["public_model_id"] for row in load_canonical_models())
    if args.all:
        if command == "smoke":
            raise SystemExit("smoke accepts one model; remove --all")
        return canonical_ids
    if args.model:
        if args.model not in canonical_ids:
            raise SystemExit(f"unknown canonical model ID: {args.model}")
        return (args.model,)
    if args.frontier or args.smoke or command == "smoke" or command == CANONICAL_COMMAND:
        return (frontier_model_id(),)
    raise SystemExit("canonical execution requires --model, --frontier, --smoke, or --all")


def _safe_model_path(model_id: str) -> str:
    return model_id.replace("|", "__")


def _execute(args: argparse.Namespace, *, command: str, smoke: bool) -> int:
    if args.plan:
        payload = _plan_payload()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if int(args.simulations) < 1:
        raise SystemExit("--simulations must be positive")
    if int(args.workers) < 1:
        raise SystemExit("--workers must be positive")
    model_ids = _selected_models(args, command=command)

    # Preflight every requested factory before reading data or creating a
    # checkpoint.  A --all run therefore cannot silently shrink to available
    # models after one canonical ID is unavailable.
    models = [(model_id, build_model(model_id), registration(model_id)) for model_id in model_ids]
    plan = build_experiment_plan(
        args.data,
        portfolio_limit=1 if smoke else None,
        rolling_origins=1 if smoke else 48,
    )
    tasks = list(
        iter_smoke_tasks(plan, count=int(args.task_count), horizon=int(args.horizon))
        if smoke
        else plan.tasks
    )
    if not tasks:
        raise SystemExit("canonical task constructor returned no tasks")

    results: list[dict[str, Any]] = []
    for model_id, model, model_registration in models:
        checkpoint_dir = Path(args.checkpoint) / _safe_model_path(model_id)
        summary = execute_model_checkpointed(
            model,
            tasks,
            simulations=int(args.simulations),
            checkpoint_dir=checkpoint_dir,
            workers=int(args.workers),
            resume=bool(args.resume),
        )
        result = summary.to_dict()
        result.update(
            {
                "display_name": canonical_model(model_id)["display_name"],
                "implementation_fidelity": model_registration.fidelity,
                "smoke": bool(smoke),
            }
        )
        results.append(result)
        output = json.dumps(result, sort_keys=True)
        if args.json:
            print(output)
        else:
            print(
                f"{model_id}: status={summary.status} tasks={summary.completed_count}/"
                f"{summary.task_count} failed={summary.failed_count} score={summary.score}"
            )
    return 0 if all(item["status"] == "completed" for item in results) else 1


def _print_inspection(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if payload.get("result_kind") == "retained_score_evidence":
        artifact = payload["retained_score_artifact"]
        print(
            f"retained historical score evidence: {len(payload['rows'])} canonical rows; "
            f"new_execution=false; reproduced=false; digest_status={artifact['digest_status']}"
        )
        for row in payload["rows"]:
            print(
                f"{row['canonical_rank']:>3}  {row['retained_score_token']}  "
                f"{row['publication_score_token']}  {row['display_name']}"
            )
        return
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "data":
        manifest = write_canonical_data(args.cache, refresh=bool(args.refresh))
        print(manifest)
        return 0
    if args.command == "catalogue":
        _print_inspection(_catalogue_payload(), as_json=bool(args.json))
        return 0
    if args.command == "coverage":
        _print_inspection(_coverage_payload(), as_json=bool(args.json))
        return 0
    if args.command == "plan":
        _print_inspection(_plan_payload(), as_json=bool(args.json))
        return 0
    if args.command == "scores":
        model_id = args.model or None
        if model_id is not None and model_id not in {
            row["public_model_id"] for row in load_canonical_models()
        }:
            raise SystemExit(f"unknown canonical model ID: {model_id}")
        _print_inspection(retained_score_report(model_id), as_json=bool(args.json))
        return 0
    if args.command == "smoke":
        return _execute(args, command="smoke", smoke=True)
    if args.command == CANONICAL_COMMAND:
        return _execute(args, command=CANONICAL_COMMAND, smoke=bool(args.smoke))
    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
