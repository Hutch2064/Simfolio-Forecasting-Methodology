"""Publishable score index: full single52 rankings, separate historical gates."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from adapters import ROOT
from campaign import EXPECTED, atomic, digest, file_hash, read, verified_process


def make_report(campaign, *, root=ROOT):
    campaign, root = Path(campaign), Path(root)
    inventory_path = root / "docs/results/catalogue52/inventory.json"
    inventory = read(inventory_path)
    entries = {row["public_model_id"]: row for row in inventory["models"]}
    baseline = read(root / "docs/results/expanded132/incumbent-single52.json")
    naive = read(root / "docs/results/expanded132/whitepaper-naive/single52.json")
    saved_control = read(root / "docs/results/catalogue52/saved-control-verification.json")
    control_entry = inventory["models"][260]
    historical_baseline = float(control_entry["historical_80"]["ledger_score_token"])
    ranked = []
    progress = []
    control = read(campaign / "control.json")
    for name, desired in control["models"].items():
        output = campaign / "models" / name
        spec = read(output / "registration.json")
        current = read(output / "progress.json") if (output / "progress.json").exists() else {}
        state = read(output / "process.json") if (output / "process.json").exists() else None
        record = entries.get(spec["model_id"])
        status = {"execution_name": name, "model_id": spec["model_id"],
                  "desired": desired["desired"], "status": current.get("status", "queued"),
                  "completed_tasks": current.get("completed_tasks", 0), "required_tasks": 2652,
                  "live": bool(state and verified_process(state))}
        if (output / "error.json").exists():
            error = read(output / "error.json")
            status["error"] = {"type": error["error"], "details_in_raw_run_envelope": True}
        progress.append(status)
        result_path = output / "results.json"
        if not result_path.exists():
            continue
        result = read(result_path)
        manifest = read(output / "manifest.json")
        if (result.get("all_denominator_gates_passed") is not True
                or any(result.get(key) != value for key, value in EXPECTED.items())
                or result.get("simulations") != 240
                or result.get("manifest_sha256") != digest(manifest)
                or not math.isfinite(result["exact_empirical_crps"])
                or len(result.get("portfolio_scores", {})) != 52):
            raise ValueError(f"unqualified full single52 result: {name}")
        historical = None if record is None else record["historical_80"]
        history_valid = bool(historical and historical["evidence_status"] == "full_80_artifact_verified")
        score80 = (float(historical["ledger_score_token"])
                   if historical and historical["ledger_score_token"] is not None else None)
        archive_path = output / "archive-receipt.json"
        raw = {"archived_and_download_verified": False}
        if archive_path.exists():
            archive = read(archive_path)
            remote = archive["remote_receipt"]
            raw = {"archived_and_download_verified": archive["status"] == "archived",
                   "access": "authenticated_research_archive",
                   "archive_sha256": remote["archive"]["sha256"],
                   "remote_receipt_sha256": archive["remote_receipt_sha256"]}
        ranked.append({"canonical_rank": None if record is None else record["canonical_rank"],
            "model_id": spec["model_id"], "display_name": None if record is None else record["display_name"],
            "execution_name": name, "single52_crps": result["exact_empirical_crps"],
            "improvement_vs_m261_single52": 1 - result["exact_empirical_crps"] / baseline["exact_empirical_crps"],
            "improvement_vs_whitepaper_naive_single52": 1 - result["exact_empirical_crps"] / naive["exact_empirical_crps"],
            "historical80_crps": score80, "historical80_evidence_verified": history_valid,
            "historical80_improves_production": bool(history_valid and score80 < historical_baseline),
            "runtime": {"summed_worker_seconds": result["summed_worker_seconds"],
                        "scope": "concurrent_campaign_worker_wall_seconds",
                        "matched_production_runtime_gate": "not_yet_measured"},
            "production_eligibility": "pending_separate_historical80_and_matched_runtime_gates",
            "manifest_sha256": result["manifest_sha256"], "source_snapshot_sha256": spec["source_sha256"],
            "result_sha256": file_hash(result_path), "cells_sha256": result["cells_sha256"],
            "simulations": 240, **EXPECTED, "portfolio_scores": result["portfolio_scores"],
            "raw_evidence": raw})
    ranked.append({"canonical_rank": 261, "model_id": control_entry["public_model_id"],
        "display_name": control_entry["display_name"], "single52_crps": baseline["exact_empirical_crps"],
        "historical80_crps": historical_baseline, "historical80_evidence_verified": True,
        "role": "validated_saved_production_control", "cells_sha256": baseline["cells_sha256"],
        "saved_control_verification_sha256": digest(saved_control), "simulations": 240, **EXPECTED})
    ranked.sort(key=lambda row: (row["single52_crps"], row["model_id"]))
    return {"protocol": "catalogue52-terminal-log-empirical-crps-240-v1",
            "ranking_metric": "single52_crps_only_lower_is_better",
            "historical80_role": "separate_required_production_validation_gate_no_new80_runs",
            "naive_benchmark": {"model_id": "asset_level_naive_politis_white_stationary_bootstrap_rebalanced",
                                "single52_crps": naive["exact_empirical_crps"],
                                "cells_sha256": naive["cells_sha256"]},
            "inventory_sha256": file_hash(inventory_path), "fresh_models_required": 258,
            "fresh_models_registered": len({read(campaign / "models" / name / "registration.json")["model_id"]
                                             for name in control["models"]}),
            "executions_registered": len(control["models"]), "fresh_models_complete": len(ranked) - 1,
            "full_scores": ranked, "progress": progress,
            "exclusions": inventory["population"]["excluded_models"],
            "partial_results_ranked": False, "blended_score_used": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = make_report(args.campaign)
    atomic(args.output, report)
    print(json.dumps({"fresh_complete": report["fresh_models_complete"],
                      "registered": report["fresh_models_registered"],
                      "lowest_complete_single52_crps": report["full_scores"][0]["single52_crps"]}))
