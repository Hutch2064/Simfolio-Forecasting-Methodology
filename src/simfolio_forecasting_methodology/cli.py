"""Command-line interface for the two reproducible OOS harnesses."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .catalog import load_canonical_175
from .harnesses import CANONICAL_HARNESS_ID, MASTER_HARNESS_ID, canonical_plan, master_plan
from .naming import descriptive_name


def _plan_payload(harness_id: str) -> dict[str, object]:
    plan = canonical_plan() if harness_id == CANONICAL_HARNESS_ID else master_plan()
    return {
        "harness_id": plan.harness_id,
        "model_count": plan.model_count,
        "protocol": asdict(plan.protocol),
        "origin_tasks_per_model": plan.expected_origin_tasks,
        "model_ids": list(plan.model_ids),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simfolio-oos",
        description="Run or inspect the standalone forecasting OOS research harnesses.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for harness_id in (CANONICAL_HARNESS_ID, MASTER_HARNESS_ID):
        command = sub.add_parser(harness_id, help=f"Inspect or run {harness_id}.")
        command.add_argument(
            "--plan",
            action="store_true",
            help="Print the exact protocol and catalogue without running forecasts.",
        )

    catalogue = sub.add_parser("catalogue", help="List canonical model identities and scores.")
    catalogue.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {CANONICAL_HARNESS_ID, MASTER_HARNESS_ID}:
        if not args.plan:
            raise SystemExit(
                "forecast execution requires the standalone model/data runtime; "
                "use --plan until that runtime is installed"
            )
        print(json.dumps(_plan_payload(args.command), indent=2, sort_keys=True))
        return 0

    rows = load_canonical_175()
    payload = [
        {
            "canonical_rank": row.canonical_rank,
            "model_id": row.model_id,
            "name": descriptive_name(row.model_id),
            "exact_empirical_crps": row.exact_empirical_crps,
            "cells": row.cells,
        }
        for row in rows
    ]
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for row in payload:
            print(
                f"{row['canonical_rank']:>3}  {row['exact_empirical_crps']:.9f}  "
                f"{row['name']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
