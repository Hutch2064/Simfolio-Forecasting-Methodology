"""Append-only full-panel evidence ledger; independent of active model controls."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

V11 = "daily_portfolio_panel_crps_v11_train_sd_sqrt_requested_h_equal_days_fixed_calendar_buckets"
WR = "model_testing_asset_joint_blocks_replacement_v6"

SINGLE_SCORE_PROTOCOLS = {
    V11,
    "daily_portfolio_panel_crps_v4_train_sd_sqrt_h_logcalendar_two_branch",
    "daily_portfolio_panel_crps_v9_raw_equal_calendar_buckets_fixed_planned_series_no_quarters",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def version_key(identity):
    """Execution seeds, display names and panel coverage do not create a model."""
    return digest({k: identity[k] for k in ("model_id", "implementation_digest")})


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


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
    single = record.get("score_protocol_id") in SINGLE_SCORE_PROTOCOLS
    expanded_v9 = record.get("score_protocol_id") == "daily_portfolio_panel_crps_v9_raw_equal_calendar_buckets_fixed_planned_series_no_quarters"
    expanded = expanded_v9 or record.get("score_protocol_id") == V11
    if expanded:
        counts = coverage["requests_by_final_paths"]
        if (any(type(n) is not int or n < 0 for n in counts.values())
                or sum(counts.values()) != coverage["expected_tasks"]
                or coverage["same_count_within_every_pair"] is not True):
            raise ValueError("V9/V11 requires complete matched mixed-path counts")
        if record["comparison"]["reference_model_id"] != "model_testing_asset_joint_blocks_replacement_v6":
            raise ValueError("V9/V11 retains its fixed WR reference")
        if not record["reproducibility"].get("posthoc_score_selection"):
            raise ValueError("V9/V11 requires retrospective selection disclosure")
    if record.get("score_protocol_id") == V11:
        provenance = record["reproducibility"]
        if (not is_sha256(provenance.get("scale_index_sha256"))
                or not isinstance(provenance.get("scale_index_receipt"), str)
                or not provenance["scale_index_receipt"].strip()):
            raise ValueError("V11 requires hash-bound normalization scale-index provenance")
    profiles = ("main",) if single else ("primary", "sensitivity")
    if single and (set(record["scores"]) != {"main"} or set(record["comparison"]["public_scores"]) != {"main"}):
        raise ValueError("Single-score records require exactly one main score and matched Public score")
    for profile in profiles:
        score = record["scores"][profile]
        reference = record["comparison"]["public_scores"][profile]
        if (not math.isfinite(score) or score < 0 or not math.isfinite(reference) or reference < 0
                or reference == 0 and record.get("score_protocol_id") != V11
                or not single and score == 0):
            raise ValueError("Declared score profiles and matched Public scores are required")
    worker_seconds = record["runtime"]["forecast_and_score_worker_seconds"]
    if worker_seconds is None:
        if not expanded or not record["runtime"].get("scope"):
            raise ValueError("Unmeasured runtime requires an explicit V9/V11 scope")
    elif not math.isfinite(worker_seconds) or worker_seconds < 0:
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
    if ledger.get("current_score_protocol_id") in SINGLE_SCORE_PROTOCOLS:
        return render_single_score(ledger)
    lines = ["# Expanded daily panel: full-panel model catalogue", "",
             (f"**{ledger['unique_full_panel_models']} unique model versions; "
             f"{ledger['failure_free_models']} with a failure-free full-panel result.**"), "",
             ("This is an evidence inventory, not a shared-coverage leaderboard. "
             "Scores are lower-is-better. Public deltas use the exact matched successful tasks. "
             "The FAST comparison is incomplete/preliminary because of native fitting failures."), "",
             "| Model | Active | Successful / attempted | Native failures | Primary Q | Sensitivity Q | Worker seconds |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for model in ledger["models"].values():
        e = model["executions"][-1]
        c = e["coverage"]
        active = e['identity']['model_id'] in ledger.get('active_model_ids', [])
        lines.append(f"| {e['display_name']} | {'yes' if active else 'no'} | {c['successful_tasks']} / {c['expected_tasks']} | "
                     f"{c['native_failed_tasks']} | {e['scores']['primary']:.12f} | "
                     f"{e['scores']['sensitivity']:.12f} | "
                     f"{format(e['runtime']['forecast_and_score_worker_seconds'], '.3f') if e['runtime']['forecast_and_score_worker_seconds'] is not None else 'not reconstructed'} |")
    lines += ["", ("Worker seconds are summed per-forecast fitting/simulation/scoring measurements, "
              "not elapsed campaign time, website latency, or a controlled speed comparison."), "",
              ("See [catalogue.json](catalogue.json) for exact values, model definitions, source versions, "
              "both profiles, matched Public scores, failures, runtime, execution fingerprints and archive links. "
              "[SCORING.md](SCORING.md) freezes the scoring contract; [RESTORE.md](RESTORE.md) explains recovery."), ""]
    if ledger.get('current_best_model_id'):
        best = next(m for m in ledger['models'].values() if m['identity']['model_id'] == ledger['current_best_model_id'])
        report = ledger.get('current_best_report', 'catalogue.json')
        lines += ["Current research baseline: **"+best['executions'][-1]['display_name']+"**. "
                  f"Its [full-panel evidence]({report}) includes "
                  "all class/family/horizon results and limitations. The accuracy-maximization search remains active.", ""]
    return "\n".join(lines)


def v11_rescored_view(ledger, model, panel):
    """A display-only rescore of preserved execution evidence; never append it."""
    result = model.get("current_rescored_result")
    status = ledger.get("current_benchmark_status", {})
    model_id = model["identity"]["model_id"]
    if result is None:
        if model_id in status.get("main_Q_by_model", {}):
            raise ValueError("Current V11 model is missing its explicit rescore identity")
        return None
    if (result.get("score_protocol_id") != V11 or result.get("rescore_only") is not True
            or result.get("full_forecast_execution_version") != "expanded_v9_mixed_paths_source_bound"
            or result.get("model_id", model_id) != model_id):
        raise ValueError("V11 rescore identity differs from preserved forecast execution")
    matching = [e for e in model["executions"]
                if e.get("version") == result["full_forecast_execution_version"]
                and e["coverage"]["panel_fingerprint"] == panel]
    if not matching:
        raise ValueError("V11 rescore has no matching source execution/panel")
    source = matching[-1]
    validate(source)
    n = result.get("matched_requests")
    if (source["identity"] != model["identity"] or n != source["coverage"]["successful_tasks"]
            or n != source["coverage"]["expected_tasks"] or n != status.get("matched_requests_per_model")
            or status.get("same_count_within_every_pair") is not True
            or source["coverage"]["requests_by_final_paths"] != status.get("requests_by_final_paths_per_model")):
        raise ValueError("V11 rescore identity/coverage/matched counts differ")
    q = result.get("main_Q")
    if (not isinstance(q, (int, float)) or not math.isfinite(q) or q < 0
            or status.get("score_protocol_id") != V11 or status.get("fixed_reference_model_id") != WR
            or q != status.get("main_Q_by_model", {}).get(model_id)
            or q != status.get("models", {}).get(model_id, {}).get("Q")
            or not is_sha256(result.get("source_rescore_sha256"))
            or result["source_rescore_sha256"] != status.get("full_saved_rescore_sha256")
            or not is_sha256(status.get("scale_index_sha256"))
            or not result.get("source_report") or result["source_report"] != status.get("source_report")):
        raise ValueError("V11 rescore score/hash/normalization provenance differs")
    view = copy.deepcopy(source)
    view["score_protocol_id"] = V11
    view["scores"] = {"main": q}
    view["comparison"]["public_scores"] = {"main": status["main_Q_by_model"][WR]}
    view["reproducibility"].update(scale_index_sha256=status["scale_index_sha256"],
        scale_index_receipt=str(Path(result["source_report"]).with_name("scale_index_receipt.json")),
        source_rescore_sha256=result["source_rescore_sha256"])
    validate(view)
    return view


def render_single_score(ledger):
    """Current benchmark only; historical definitions/results remain in JSON."""
    panel = ledger["current_panel_fingerprint"]
    protocol = ledger["current_score_protocol_id"]
    records = []
    for model in ledger["models"].values():
        matching = [e for e in model["executions"] if e.get("score_protocol_id") == protocol
                    and e["coverage"]["panel_fingerprint"] == panel]
        if matching:
            validate(matching[-1])
            records.append(matching[-1])
        elif protocol == V11:
            view = v11_rescored_view(ledger, model, panel)
            if view is not None:
                records.append(view)
    lines = ["# Versioned full-panel model catalogue", "",
        (f"**{ledger['unique_full_panel_models']} unique model versions across preserved benchmarks; "
        f"{ledger['full_panel_executions']} completed executions; "
        f"{len(records)} with completed attempts on the current benchmark.**"), "",
        ("The current benchmark has one main score. Historical normalizations and all older "
        "executions remain in catalogue.json; scores from different benchmarks are not compared."), ""]
    if protocol == V11:
        lines += ["V11 rescoring views reuse preserved forecast executions; they do not add executions or model versions.", ""]
    for failed, title in [(False, "Complete successful coverage"), (True, "Incomplete paired results: native forecast failures")]:
        selected = [e for e in records if bool(e['coverage']['native_failed_tasks']) == failed]
        if not selected:
            continue
        lines += ["## "+title, "", "This is an evidence inventory, not a mixed-coverage ranking.", "",
            "| Model | Successful / attempted | Native failures | Main Q | Worker seconds |",
            "|---|---:|---:|---:|---:|"]
        for e in selected:
            c = e['coverage']
            lines.append(f"| {e['display_name']} | {c['successful_tasks']} / {c['expected_tasks']} | "
                         f"{c['native_failed_tasks']} | {e['scores']['main']:.12f} | "
                         f"{format(e['runtime']['forecast_and_score_worker_seconds'], '.3f') if e['runtime']['forecast_and_score_worker_seconds'] is not None else 'not reconstructed'} |")
        lines.append("")
    lines += ["Worker seconds are recorded compute, not a controlled speed comparison or production latency.", "",
        ("[catalogue.json](catalogue.json) retains every historical model definition, version, benchmark, "
        "matched reference, failure, runtime and complete repository-linked recovery evidence. "
        "Identical models rerun on another panel add executions, not model counts."), ""]
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
