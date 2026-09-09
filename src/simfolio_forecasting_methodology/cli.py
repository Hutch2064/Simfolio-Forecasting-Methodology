"""Command-line interface for the standalone OOS research package."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .data import write_canonical_data
from .experiment import build_experiment_plan, iter_smoke_tasks, run_model
from .harnesses import CANONICAL_HARNESS_ID, MASTER_HARNESS_ID, canonical_plan, master_plan
from .models.frontier import FRONTIER_MODEL_ID
from .models.registry import build_model, registration
from .naming import descriptive_name
from .runner import evaluate_model
from .scores import canonical_scores


def _plan_payload(harness_id: str) -> dict[str, object]:
    plan = canonical_plan() if harness_id == CANONICAL_HARNESS_ID else master_plan()
    return {
        "harness_id": plan.harness_id,
        "model_count": plan.model_count,
        "protocol": asdict(plan.protocol),
        "origin_tasks_per_model": plan.expected_origin_tasks,
        "model_ids": list(plan.model_ids),
    }


def _add_run_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--plan", action="store_true", help="Print protocol/catalogue only.")
    command.add_argument("--data", type=Path, default=Path(".simfolio-oos-data"))
    command.add_argument("--model", default="", help="Run one immutable model ID.")
    command.add_argument("--frontier", action="store_true", help="Run the Frontier model.")
    command.add_argument("--all", action="store_true", help="Run the entire harness catalogue.")
    command.add_argument("--smoke", action="store_true", help="Use two 63-day origin tasks.")
    command.add_argument("--simulations", type=int, default=240)
    command.add_argument("--output", type=Path, default=Path("outputs"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simfolio-oos",
        description="Reproduce or inspect the standalone forecasting OOS research.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("data", help="Build public-source canonical research inputs.")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    build = data_sub.add_parser("build")
    build.add_argument("--cache", type=Path, default=Path(".simfolio-oos-data"))
    build.add_argument("--refresh", action="store_true")

    for harness_id in (CANONICAL_HARNESS_ID, MASTER_HARNESS_ID):
        command = sub.add_parser(harness_id, help=f"Inspect or execute {harness_id}.")
        _add_run_arguments(command)

    catalogue = sub.add_parser("catalogue", help="List canonical retained results.")
    catalogue.add_argument("--json", action="store_true")
    scores = sub.add_parser("scores", help="Emit retained canonical score registry.")
    scores.add_argument("--json", action="store_true")
    return parser


def _selected_models(args: argparse.Namespace, harness_id: str) -> tuple[str, ...]:
    plan = canonical_plan() if harness_id == CANONICAL_HARNESS_ID else master_plan()
    if args.all:
        return plan.model_ids
    if args.frontier:
        if FRONTIER_MODEL_ID not in plan.model_ids and harness_id == MASTER_HARNESS_ID:
            return (FRONTIER_MODEL_ID,)
        return (FRONTIER_MODEL_ID,)
    if args.model:
        if args.model not in plan.model_ids:
            raise SystemExit(f"model ID is not in {harness_id}: {args.model}")
        return (args.model,)
    # Canonical defaults to the published winner; master defaults to a plan to
    # prevent an accidental multi-day 369-model execution.
    if harness_id == CANONICAL_HARNESS_ID:
        return (FRONTIER_MODEL_ID,)
    raise SystemExit("master-research requires --model, --frontier, --smoke, or --all")


def _execute(args: argparse.Namespace, harness_id: str) -> int:
    if args.plan:
        print(json.dumps(_plan_payload(harness_id), indent=2, sort_keys=True))
        return 0
    model_ids = _selected_models(args, harness_id)
    plan = build_experiment_plan(args.data, portfolio_limit=1 if args.smoke else None,
                                 rolling_origins=1 if args.smoke else 48)
    tasks = list(iter_smoke_tasks(plan)) if args.smoke else list(plan.tasks)
    outputs: list[dict[str, object]] = []
    for model_id in model_ids:
        model = build_model(model_id)
        if args.smoke:
            accumulator = evaluate_model(model, tasks, simulations=min(int(args.simulations), 32))
            result: dict[str, object] = {
                "model_id": model_id,
                "name": descriptive_name(model_id),
                "fidelity": registration(model_id).fidelity,
                "smoke": True,
                "exact_empirical_crps": accumulator.model_score(),
                "cells": accumulator.cell_count,
                "origin_tasks": len(tasks),
                "simulations": min(int(args.simulations), 32),
            }
        else:
            result = run_model(model, plan, simulations=int(args.simulations))
            result["name"] = descriptive_name(model_id)
            result["fidelity"] = registration(model_id).fidelity
        outputs.append(result)
        path = args.output / harness_id / f"{model_id.replace('|', '__')}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "data":
        manifest = write_canonical_data(args.cache, refresh=bool(args.refresh))
        print(manifest)
        return 0
    if args.command in {CANONICAL_HARNESS_ID, MASTER_HARNESS_ID}:
        return _execute(args, args.command)

    payload = [asdict(record) for record in canonical_scores()]
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for row in payload:
            print(
                f"{row['canonical_rank']:>3}  {row['exact_empirical_crps']:.9f}  "
                f"{row['model_name']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
