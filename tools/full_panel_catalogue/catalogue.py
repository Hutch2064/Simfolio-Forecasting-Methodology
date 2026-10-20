"""Append-only full-panel evidence ledger; independent of active model controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def version_key(identity):
    """Execution seeds, display names and panel coverage do not create a model."""
    return digest({k: identity[k] for k in ("model_id", "implementation_digest")})


def validate(record):
    identity = record["identity"]
    if len(identity["implementation_digest"]) != 64 or not record["definition"]:
        raise ValueError("A source-bound model definition is required")
    coverage = record["coverage"]
    if coverage["successful_tasks"] + coverage["native_failed_tasks"] != coverage["expected_tasks"]:
        raise ValueError("Pilot or unfinished execution: the entire panel was not attempted")
    if coverage["expected_tasks"] <= 0 or coverage["missing_tasks"] != 0:
        raise ValueError("Full-panel coverage must have no unattempted tasks")
    if min(coverage[k] for k in ("successful_tasks", "native_failed_tasks")) < 0:
        raise ValueError("Invalid task counts")
    if record["comparison"]["matched_successful_tasks"] != coverage["successful_tasks"]:
        raise ValueError("Comparison must cover the identical successful task set")
    for profile in ("primary", "sensitivity"):
        score = record["scores"][profile]
        reference = record["comparison"]["public_scores"][profile]
        if not all(math.isfinite(x) and x > 0 for x in (score, reference)):
            raise ValueError("Both score profiles and matched Public scores are required")
    if record["runtime"]["forecast_and_score_worker_seconds"] < 0:
        raise ValueError("Invalid measured runtime")
    if not record["archives"] or not record["reproducibility"]:
        raise ValueError("Repository-linked complete evidence and recovery details are required")


def append(ledger, record):
    """Retain failed/non-winning methods; reruns append evidence without another model count."""
    validate(record)
    key = version_key(record["identity"])
    models = ledger.setdefault("models", {})
    model = models.setdefault(key, {"identity": record["identity"], "executions": []})
    for prior in model["executions"]:
        if prior["execution_fingerprint"] == record["execution_fingerprint"]:
            if prior != record:
                raise ValueError("Existing execution evidence changed; retain a separate correction record")
            return False
    model["executions"].append(record)
    ledger["unique_full_panel_models"] = len(models)
    ledger["full_panel_executions"] = sum(len(m["executions"]) for m in models.values())
    ledger["failure_free_models"] = sum(
        any(e["coverage"]["native_failed_tasks"] == 0 for e in m["executions"])
        for m in models.values())
    return True


def retirement_allowed(record, receipt):
    """A receipt must bind the exact archive and manifest, not just assert upload success."""
    verified = {r["archive_sha256"]: r for r in receipt.get("archives", [])}
    for archive in record["archives"]:
        check = verified.get(archive["archive_sha256"], {})
        if (check.get("manifest_sha256") != archive["manifest_sha256"]
                or not check.get("remote_download_sha256_verified")
                or not check.get("all_files_disk_restored_and_sha256_verified")):
            return False
    return bool(record["archives"])


def render(ledger):
    lines = ["# Expanded daily panel: full-panel model catalogue", "",
             f"**{ledger['unique_full_panel_models']} unique model versions; "
             f"{ledger['failure_free_models']} with a failure-free full-panel result.**", "",
             "This is an evidence inventory, not a shared-coverage leaderboard. "
             "Scores are lower-is-better. Public deltas use the exact matched successful tasks. "
             "The FAST comparison is incomplete/preliminary because of native fitting failures.", "",
             "| Model | Active | Successful / attempted | Native failures | Primary Q | Sensitivity Q | Worker seconds |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for model in ledger["models"].values():
        e = model["executions"][-1]
        c = e["coverage"]
        active = e['identity']['model_id'] in ledger.get('active_model_ids', [])
        lines.append(f"| {e['display_name']} | {'yes' if active else 'no'} | {c['successful_tasks']} / {c['expected_tasks']} | "
                     f"{c['native_failed_tasks']} | {e['scores']['primary']:.12f} | "
                     f"{e['scores']['sensitivity']:.12f} | "
                     f"{e['runtime']['forecast_and_score_worker_seconds']:.3f} |")
    lines += ["", "Worker seconds are summed per-forecast fitting/simulation/scoring measurements, "
              "not elapsed campaign time, website latency, or a controlled speed comparison.", "",
              "See [catalogue.json](catalogue.json) for exact values, model definitions, source versions, "
              "both profiles, matched Public scores, failures, runtime, execution fingerprints and archive links. "
              "[SCORING.md](SCORING.md) freezes the scoring contract; [RESTORE.md](RESTORE.md) explains recovery.", ""]
    if ledger.get('current_best_model_id'):
        best = next(m for m in ledger['models'].values() if m['identity']['model_id'] == ledger['current_best_model_id'])
        report = ledger.get('current_best_report', 'catalogue.json')
        lines += ["Current research baseline: **"+best['executions'][-1]['display_name']+"**. "
                  f"Its [full-panel evidence]({report}) includes "
                  "all class/family/horizon results and limitations. The accuracy-maximization search remains active.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--append", type=Path, help="One completed full-panel execution record")
    parser.add_argument("--render", type=Path)
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_text()) if args.ledger.exists() else {"schema": "full-panel-catalogue-v1"}
    if args.append:
        append(ledger, json.loads(args.append.read_text()))
        args.ledger.write_text(json.dumps(ledger, indent=2, allow_nan=False) + "\n")
    for model in ledger["models"].values():
        for record in model["executions"]:
            validate(record)
    if args.render:
        args.render.write_text(render(ledger))


if __name__ == "__main__":
    main()
