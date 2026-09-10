"""Canonical-only command-line interface for the standalone OOS package."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .catalogue import (
    EXPECTED_CANONICAL_COUNT,
    EXPECTED_MEMBERSHIP_DIGEST,
    canonical_model,
    frontier_model_id,
    load_canonical_ledger,
    load_canonical_models,
)
from .experiment import build_experiment_plan, iter_smoke_tasks
from .models.registry import build_model, registration
from .protocol import CANONICAL_DENSE_PROTOCOL
from .results.retained import retained_score_report
from .runner import execute_model_checkpointed

CANONICAL_COMMAND = "canonical-175"


def _plan_payload() -> dict[str, Any]:
    try:
        from .experiment import build_experiment_schedule
    except ImportError as exc:
        raise RuntimeError(
            "canonical schedule API is unavailable; install the verified protocol integration"
        ) from exc
    schedule = build_experiment_schedule()
    schedule_tasks = []
    for task in schedule.tasks:
        descriptor = task.descriptor
        horizon_mask = descriptor.horizon_mask(len(schedule.common_dates))
        schedule_tasks.append(
            {
                "portfolio_id": task.portfolio_id,
                "portfolio_index": int(task.portfolio_index),
                "origin_label": str(descriptor.origin_label),
                "evaluation_split": str(descriptor.evaluation_split),
                "position": int(descriptor.position),
                "origin_date": str(descriptor.origin_date),
                "max_horizon": int(descriptor.max_horizon),
                "train_fraction": (
                    None
                    if descriptor.train_fraction is None
                    else float(descriptor.train_fraction)
                ),
                "horizon_mask_digest": hashlib.sha256(
                    bytes(int(value) for value in horizon_mask)
                ).hexdigest(),
            }
        )
    schedule_identity = {
        "dates": [str(date.date()) for date in schedule.common_dates],
        "portfolios": [spec.name for spec in schedule.portfolios],
        "tasks": schedule_tasks,
    }
    schedule_fingerprint = hashlib.sha256(
        json.dumps(schedule_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    protocol = asdict(CANONICAL_DENSE_PROTOCOL)
    return {
        "scope": "canonical_175_only",
        "membership_digest": EXPECTED_MEMBERSHIP_DIGEST,
        "model_count": EXPECTED_CANONICAL_COUNT,
        "model_ids": [row["public_model_id"] for row in load_canonical_models()],
        "protocol": protocol,
        "protocol_fingerprint": CANONICAL_DENSE_PROTOCOL.protocol_fingerprint,
        "dataset_fingerprint": load_canonical_ledger()["identity_policy"]["dataset_fingerprint"],
        "panel_fingerprint": CANONICAL_DENSE_PROTOCOL.panel_fingerprint,
        "origin_tasks_per_model": schedule.task_count,
        "scored_cells_per_model": schedule.cell_capacity,
        "schedule_fingerprint": schedule_fingerprint,
        "schedule_date_count": len(schedule.common_dates),
        "schedule_first_date": str(schedule.common_dates.min().date()),
        "schedule_last_date": str(schedule.common_dates.max().date()),
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
    command.add_argument("--data", type=Path, default=None)
    selection = command.add_mutually_exclusive_group()
    selection.add_argument("--model", default="", help="Run one exact canonical model ID.")
    selection.add_argument("--frontier", action="store_true", help="Select canonical rank one.")
    selection.add_argument("--all", action="store_true", help="Select all 175 canonical IDs.")
    command.add_argument("--smoke", action="store_true", help="Use bounded smoke tasks.")
    del smoke_default
    command.add_argument(
        "--simulations",
        type=int,
        default=None,
        help="Canonical runs require 240; smoke runs default to 32 and cap at 32.",
    )
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

    data = sub.add_parser(
        "data", help="Verify or prepare a caller-owned canonical data snapshot."
    )
    data_sub = data.add_subparsers(dest="data_command", required=True)
    verify = data_sub.add_parser("verify", help="Verify a frozen source snapshot against its identity.")
    verify.add_argument("--snapshot", type=Path, default=None)
    verify.add_argument("--rights-confirmed", action="store_true", default=None)
    verify.add_argument("--json", action="store_true")
    prepare = data_sub.add_parser(
        "prepare", help="Prepare the bundled frozen snapshot into an execution cache."
    )
    prepare.add_argument("--snapshot", type=Path, default=None)
    prepare.add_argument("--destination", type=Path, required=True)
    prepare.add_argument("--rights-confirmed", action="store_true", default=None)
    prepare.add_argument("--json", action="store_true")

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
    scores.add_argument(
        "--experiment",
        default="canonical-whitepaper",
        choices=("canonical-whitepaper", "canonical-dense-oos-2026-08-23"),
    )
    scores.add_argument("--json", action="store_true")
    validate = sub.add_parser("validate", help="Run bounded canonical release checks.")
    validate.add_argument("--quick", action="store_true", help="Validate ledger/protocol identity only.")
    validate.add_argument("--json", action="store_true")
    return parser


def _selected_models(args: argparse.Namespace, *, command: str) -> tuple[str, ...]:
    canonical_ids = tuple(row["public_model_id"] for row in load_canonical_models())
    if args.all:
        if command == "smoke" or args.smoke:
            raise SystemExit("smoke accepts one model; remove --all")
        return canonical_ids
    if args.model:
        if args.model not in canonical_ids:
            raise SystemExit(f"unknown canonical model ID: {args.model}")
        return (args.model,)
    if args.frontier or args.smoke or command == "smoke":
        return (frontier_model_id(),)
    if command == CANONICAL_COMMAND:
        return canonical_ids
    raise SystemExit("canonical execution requires --model, --frontier, --smoke, or --all")


def _safe_model_path(model_id: str) -> str:
    return model_id.replace("|", "__")


def _progress_callback(args: argparse.Namespace):
    def callback(event) -> None:
        payload = {"result_kind": "progress", **event.to_dict()}
        if args.json:
            print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)
        else:
            print(
                f"progress {event.processed_count}/{event.task_count} "
                f"status={event.status} task={event.task_id}",
                flush=True,
            )

    return callback


def _execute(args: argparse.Namespace, *, command: str, smoke: bool) -> int:
    if args.plan:
        payload = _plan_payload()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    simulations = int(args.simulations) if args.simulations is not None else (32 if smoke else 240)
    if simulations < 1:
        raise SystemExit("--simulations must be positive")
    if smoke and simulations > 32:
        raise SystemExit("smoke runs require --simulations <= 32")
    if not smoke and simulations != CANONICAL_DENSE_PROTOCOL.simulations_per_origin:
        raise SystemExit(
            "canonical runs require --simulations 240; use smoke for bounded noncanonical runs"
        )
    if int(args.workers) < 1:
        raise SystemExit("--workers must be positive")
    model_ids = _selected_models(args, command=command)

    # Preflight every requested factory before reading data or creating a
    # checkpoint.  A --all run therefore cannot silently shrink to available
    # models after one canonical ID is unavailable.
    try:
        models = [
            (model_id, build_model(model_id), registration(model_id))
            for model_id in model_ids
        ]
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    if smoke and args.data is None:
        from .smoke import fixture_smoke_tasks

        tasks = fixture_smoke_tasks(origins=int(args.task_count), horizon=int(args.horizon))
    else:
        try:
            cache = args.data or Path(".simfolio-oos-data")
            if args.data is None and not cache.exists():
                _data_function("prepare_canonical_data")(destination=cache)
            plan = build_experiment_plan(
                cache,
                portfolio_limit=1 if smoke else None,
                rolling_origins=1 if smoke else 48,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            raise SystemExit(f"canonical data/protocol unavailable: {exc}") from exc
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
            simulations=simulations,
            checkpoint_dir=checkpoint_dir,
            workers=int(args.workers),
            resume=bool(args.resume),
            progress_callback=_progress_callback(args),
            execution_variant="smoke_noncanonical" if smoke else "canonical",
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


def _data_function(name: str):
    try:
        from . import data

        function = getattr(data, name)
    except (ImportError, AttributeError) as exc:
        raise SystemExit(
            f"canonical data {name} API is unavailable; install the verified data integration"
        ) from exc
    return function


def _quick_validation_payload() -> dict[str, Any]:
    ledger = load_canonical_ledger()
    CANONICAL_DENSE_PROTOCOL.validate()
    executable_count = sum(
        registration(row["public_model_id"]).factory is not None
        for row in ledger["models"]
    )
    return {
        "scope": "canonical_175_only",
        "status": "passed",
        "checks": {
            "ledger": "passed",
            "protocol": "passed",
            "membership_digest": ledger["membership"]["membership_digest"],
            "model_count": ledger["membership"]["count"],
            "explicit_factory_count": executable_count,
            "implementation_release_gate": (
                "passed" if executable_count == EXPECTED_CANONICAL_COUNT
                else "incomplete_explicit_factory_coverage"
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "data":
        if args.data_command == "verify":
            result = _data_function("verify_canonical_snapshot")(
                args.snapshot, rights_confirmed=args.rights_confirmed
            )
            _print_inspection(dict(result), as_json=bool(args.json))
            return 0
        if args.data_command == "prepare":
            manifest = _data_function("prepare_canonical_data")(
                args.snapshot,
                args.destination,
                rights_confirmed=args.rights_confirmed,
            )
            payload = {"status": "prepared", "manifest": str(manifest)}
            _print_inspection(payload, as_json=bool(args.json))
            return 0
        raise SystemExit(f"unsupported data command: {args.data_command}")
    if args.command == "validate":
        if not args.quick:
            raise SystemExit("validate requires --quick; full forecast execution is a separate command")
        _print_inspection(_quick_validation_payload(), as_json=bool(args.json))
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
        payload = retained_score_report(model_id)
        payload["requested_experiment"] = args.experiment
        _print_inspection(payload, as_json=bool(args.json))
        return 0
    if args.command == "smoke":
        return _execute(args, command="smoke", smoke=True)
    if args.command == CANONICAL_COMMAND:
        return _execute(args, command=CANONICAL_COMMAND, smoke=bool(args.smoke))
    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
