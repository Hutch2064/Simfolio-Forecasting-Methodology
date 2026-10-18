"""Enqueue the frozen full inventory as isolated-origin admissions finish.

No smoke loss is used for selection. Failures remain pending recovery, and a
removed/failed registered model is never silently re-added by this feeder.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from adapters import ROOT, model_specs
from campaign import add_model, atomic, read


def passed_ranks(evidence_root):
    passed = set()
    for folder in ("admission-legacy", "admission-modern", "admission-singleton-repair",
                   "admission-bayesian"):
        for path in (Path(evidence_root) / folder).glob("m*-0000.json"):
            record = read(path)
            if (record.get("cached_repeat_array_equal") is True
                    and record.get("simulations") == 240
                    and record.get("index") == 0
                    and record.get("loss_count") == 126):
                passed.add(record["canonical_rank"])
    return passed


def feed(campaign, evidence_root, rows):
    passed = passed_ranks(evidence_root)
    control = read(campaign / "control.json")
    for row in rows:
        rank = row["canonical_rank"]
        name = f"m{rank:03d}"
        if rank == 261 or rank not in passed or name in control["models"]:
            continue
        add_model(campaign, row["model_id"], name=name, adapter=row["adapter"],
                  environment=row["environment"], source_root=ROOT)
    control = read(campaign / "control.json")
    pending = [row["canonical_rank"] for row in rows
               if row["canonical_rank"] != 261
               and f"m{row['canonical_rank']:03d}" not in control["models"]]
    status = {"expected_fresh_models": 258, "registered": len(control["models"]),
              "pending_admission_or_recovery_ranks": pending,
              "saved_control_rank": 261, "score_used_for_pruning": False}
    atomic(campaign / "queue.json", status)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    rows = model_specs()
    if len(rows) != 259:
        raise ValueError("initial frozen included population must contain exactly 259 models")
    while True:
        status = feed(args.campaign.resolve(), args.evidence_root.resolve(), rows)
        print(json.dumps(status), flush=True)
        if args.once or not status["pending_admission_or_recovery_ranks"]:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
