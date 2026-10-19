"""Validate a closed study packet and add it to two *new* catalogue sidecars.

This does not run numerical work, audit raw cells, archive, publish, or edit the
historical refinement52 catalogues. See README.md for the auditor contract.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys


SCHEMA = "unified-frontier-studies-v1"
DATA_SHA256 = "3825258f753ba0a4ec653c56403ef7e163e80ea89eb1c06856b6415160672a76"
PANELS = {"52": (52, 2652, 455832, 3904212),
          "80": (80, 4080, 701280, 6006480)}
ROLES = {"manifest", "reference_manifest", "completion", "results", "analysis", "cells", "audit"}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def artifact(root, ref):
    """Use portable evidence-root paths; disallow escapes and symlink aliases."""
    root = root.resolve()
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "bad artifact ref")
    raw = ref["path"]
    require(isinstance(raw, str) and raw and not Path(raw).is_absolute()
            and ".." not in Path(raw).parts, "unsafe artifact path")
    path = root / raw
    require(not any(part.is_symlink() for part in (path, *path.parents)),
            "symlink evidence is not accepted")
    require(path.is_file() and path.resolve().is_relative_to(root.resolve()),
            "artifact is missing or outside evidence root")
    require(re.fullmatch(r"[0-9a-f]{64}", str(ref["sha256"])) is not None
            and file_sha256(path) == ref["sha256"], "artifact hash mismatch: " + raw)
    return path


def finite_score(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value) and value >= 0


def ulp_distance(left, right):
    return abs(struct.unpack(">Q", struct.pack(">d", left))[0]
               - struct.unpack(">Q", struct.pack(">d", right))[0])


def validate_score_aggregation(audit, row, raw_scores, cell_count):
    """Permit only independently closed scalar reduction-order roundoff."""
    scores = row["scores"]
    require(isinstance(raw_scores, dict) and set(raw_scores) == set(scores)
            and all(finite_score(value) for value in raw_scores.values()),
            "raw score labels or values differ")
    if "raw_results_scores" in row:
        require(row["raw_results_scores"] == raw_scores, "raw score representation changed")
    receipt = audit.get("score_aggregation")
    if "score_aggregation" in row:
        require(row["score_aggregation"] == receipt,
                "public scalar aggregation proof differs from closed audit")
    if receipt is None:
        require(raw_scores == scores, "score receipts disagree without aggregation proof")
        return
    epsilon = sys.float_info.epsilon
    gamma = cell_count * epsilon / (1 - cell_count * epsilon)
    require(receipt.get("schema") == "unified-frontier-scalar-aggregation-v1"
            and receipt.get("method") == "float64_gamma_n_mean_absolute_cells"
            and receipt.get("dtype") == "float64" and receipt.get("canonical_order") == "C"
            and receipt.get("n") == cell_count and not isinstance(receipt.get("n"), bool)
            and receipt.get("epsilon") == epsilon and receipt.get("gamma_n") == gamma,
            "aggregation proof does not use the derived binary64 gamma_n bound")
    require(receipt.get("canonical_scores") == scores
            and receipt.get("raw_results_scores") == raw_scores
            and row.get("raw_results_scores") == raw_scores,
            "aggregation proof does not preserve raw and canonical scores")
    comparisons = receipt.get("per_label", {})
    require(set(comparisons) == set(scores) == set(raw_scores)
            and all(finite_score(value) for value in raw_scores.values()),
            "aggregation proof score labels or values differ")
    for label, canonical in scores.items():
        raw = raw_scores[label]
        detail = comparisons[label]
        mean_absolute = detail.get("mean_absolute_cells")
        require(finite_score(mean_absolute), "aggregation proof has invalid mean absolute cells")
        require(mean_absolute == canonical,
                "nonnegative CRPS mean absolute cells must equal canonical score")
        difference = abs(raw - canonical)
        bound = gamma * mean_absolute
        ulps = ulp_distance(raw, canonical)
        require(finite_score(detail.get("absolute_difference"))
                and finite_score(detail.get("error_bound"))
                and detail.get("absolute_difference") == difference
                and detail.get("error_bound") == bound
                and detail.get("ulp_difference") == ulps
                and not isinstance(detail.get("ulp_difference"), bool)
                and detail.get("within_bound") is True and difference <= bound,
                "aggregation discrepancy exceeds or misstates the derived roundoff bound: " + label)


def validate_panel(root, panel, row, variants, study_sources):
    require(row.get("status") in {"not_evaluated", "complete_scored_and_audited"},
            "panel status must distinguish unscored from completed evidence")
    if row["status"] == "not_evaluated":
        require(set(row) == {"status", "reason"} and row["reason"],
                "unscored panel cannot carry scores or full-panel claims")
        return []
    refs = row.get("evidence", {})
    require(set(refs) == ROLES, "completed panel needs all seven evidence roles")
    files = {role: artifact(root, ref) for role, ref in refs.items()}
    manifest, completion, results, analysis, audit = (
        read_json(files[role]) for role in ("manifest", "completion", "results", "analysis", "audit"))
    entities, origins, cells, contributions = PANELS[panel]
    identities = manifest.get("task_identities", {}).get(panel, [])
    reference = read_json(files["reference_manifest"])
    require(reference.get("data_manifest_sha256") == DATA_SHA256
            and reference.get("task_identities") == identities,
            "canonical reference task/data identities differ")
    require(len(identities) == origins
            and len({item.get("task_id") for item in identities}) == origins,
            "full panel task identities are incomplete or duplicated")
    require(all(isinstance(item.get("seed"), int) and not isinstance(item["seed"], bool)
                and item.get("origin_date") and item.get("training_digest")
                and item.get("realized_digest") and item.get("future_dates_digest")
                for item in identities), "task seeds/dates/input identities are missing")
    require(manifest.get("simulations") == 240, "canonical simulation count changed")
    source_refs = []
    for group in ("source_hashes", "snapshot_sources"):
        require(bool(manifest.get(group)), "manifest source closure is missing")
        for raw, expected in manifest[group].items():
            source = Path(raw)
            source = source if source.is_absolute() else root / source
            relative = source.absolute().relative_to(root.resolve()).as_posix()
            ref = {"path": relative, "sha256": expected}
            artifact(root, ref)
            source_refs.append(ref)
    require(completion.get("complete_origins", {}).get(panel) == origins
            and completion.get("required_origins", {}).get(panel) == origins
            and completion.get("failures", {}).get(panel) in ({}, [], 0),
            "completion is partial, failed, or lacks an explicit denominator")
    require(results.get("complete_origins") == origins
            and results.get("required_origins") == origins
            and results.get("complete_portfolios") == entities, "results are partial")
    scored = analysis.get(panel, {})
    expected = {"origins": origins, "portfolios": entities, "simulations": 240,
                "cells_per_model": cells, "origin_horizon_contributions": contributions}
    require(all(scored.get(key) == value for key, value in expected.items()),
            "analysis denominators are not canonical")
    scores = row.get("scores", {})
    labels = manifest.get("labels", {}).get(panel, [])
    require(set(scores) == set(labels) and variants <= set(labels)
            and all(finite_score(score) for score in scores.values()), "scores/labels differ")
    require(all(scored.get("models", {}).get(label, {}).get("CRPS") == score
                    for label, score in scores.items()), "score receipts disagree")
    controls = row.get("controls", {})
    require("M261_saved_control" in controls and len(controls) >= 2,
            "M261 and fast/reference control bindings are required")
    for label, ref in controls.items():
        require(label in scores, "control has no scored label")
        path = artifact(root, ref)
        require(manifest["source_hashes"].get(str(path.resolve()),
                manifest["source_hashes"].get(ref["path"])) == ref["sha256"],
                "control is not bound by the run manifest")
    # The independent auditor owns raw checkpoint reconstruction and counts
    # hashing. The catalogue helper verifies its closure, never substitutes for it.
    require(audit.get("schema") == "unified-frontier-publication-audit-v1"
            and audit.get("panel") == panel and audit.get("independent_from_runner") is True
            and audit.get("data_manifest_sha256") == DATA_SHA256
            and audit.get("complete_origins") == origins and audit.get("failures") == 0
            and audit.get("all_task_identities_match_reference") is True
            and audit.get("all_cells_and_counts_rebuilt_byte_exactly") is True
            and audit.get("all_saved_controls_byte_equal") is True
            and audit.get("all_bound_sources_and_controls_match") is True,
            "independent cell-rebuild audit is incomplete")
    require(audit.get("task_identities_sha256") == digest_json(identities),
            "audited task/seed identity digest differs")
    denominator = {**expected, "horizons": 8766,
                   "counts_sha256": audit.get("counts_sha256")}
    require(re.fullmatch(r"[0-9a-f]{64}", str(denominator["counts_sha256"])) is not None
            and audit.get("denominator_sha256") == digest_json(denominator),
            "denominator/counts digest differs")
    require(audit.get("artifact_sha256") == {
        role: refs[role]["sha256"] for role in ROLES - {"audit"}},
        "audit does not close all scored artifacts")
    validate_score_aggregation(audit, row, results.get("scores"), cells)
    auditor = audit.get("auditor_source", {})
    artifact(root, auditor)
    required_sources = audit.get("required_publication_source_refs")
    dependencies = audit.get("auditor_dependencies")
    require(isinstance(required_sources, list) and required_sources
            and isinstance(dependencies, list), "publication auditor source closure is missing")
    for ref in [*required_sources, *dependencies]:
        artifact(root, ref)
    require(auditor in required_sources, "required publication sources omit auditor_source")
    require(all(ref in required_sources for ref in dependencies),
            "required publication sources omit auditor dependency")
    require(all(ref in study_sources for ref in required_sources),
            "study.sources omits required publication source")
    runners = row.get("runner_sources", [])
    require(runners, "runner source bindings are required")
    for ref in runners:
        path = artifact(root, ref)
        require(manifest["source_hashes"].get(str(path.resolve()),
                manifest["source_hashes"].get(ref["path"])) == ref["sha256"],
                "runner is not bound by manifest")
    require(all(auditor["sha256"] != ref["sha256"] for ref in runners),
            "auditor must be separate from runner")
    return [*refs.values(), *controls.values(), *runners, *source_refs,
            *required_sources, *dependencies, auditor]


def validate_study(root, study):
    root = root.resolve()
    require(study.get("schema") == "unified-frontier-study-v1"
            and re.fullmatch(r"UF-[A-Za-z0-9._-]+", str(study.get("study_id", ""))),
            "study schema/immutable ID is missing")
    require(study.get("purpose") and study.get("methodology")
            and study.get("uniform_asset_law") is True, "unified methodology is missing")
    variants = study.get("variants", [])
    require(variants and len({v.get("label") for v in variants}) == len(variants)
            and all(v.get("research_id") and v.get("methodology") and v.get("rationale")
                    and v.get("decision") in {"retained", "rejected", "diagnostic"}
                    for v in variants), "variant/rejection ledger is incomplete")
    require(study.get("revisions") and all(re.fullmatch(r"[0-9a-f]{40}", str(sha))
            for sha in study["revisions"].values()), "full source revisions are required")
    refs = list(study.get("sources", []))
    require(refs, "methodology/source evidence is missing")
    for ref in refs:
        artifact(root, ref)
    require(set(study.get("panels", {})) == set(PANELS), "both panel statuses are required")
    for panel, row in study["panels"].items():
        labels = {v.get("panel_labels", {}).get(panel, v["label"]) for v in variants}
        refs.extend(validate_panel(root, panel, row, labels, study["sources"]))
    growth = study.get("growth_diagnostics", {})
    require(growth.get("status") in {"not_evaluated", "recorded"}
            and growth.get("scope") == "diagnostic_no_historical_return_floor",
            "growth diagnostics need an explicit non-floor scope")
    if growth["status"] == "recorded":
        require(growth.get("evidence"), "growth diagnostic evidence is missing")
        for ref in growth["evidence"]:
            artifact(root, ref)
            refs.append(ref)
    require(study.get("data_manifest_sha256") == DATA_SHA256, "canonical data identity differs")
    runtime = study.get("runtime_evidence", {})
    require(runtime.get("status") in {"not_measured", "representative_matched_compute"}
            and runtime.get("scope") and runtime.get("production_browser_ctv") is False,
            "runtime evidence must remain separate from browser CTV")
    if runtime["status"] == "representative_matched_compute":
        require(set(runtime.get("cases", {})) == {"single_asset", "multi_asset"}
                and runtime.get("fresh_and_cached_fit") is True
                and runtime.get("decomposition") == ["fitting", "dependence", "simulation", "accounting"],
                "representative runtime evidence lacks matched cases/decomposition")
        for ref in runtime.get("evidence", []):
            artifact(root, ref)
            refs.append(ref)
        require(runtime.get("evidence"), "runtime evidence is missing")
    validation = study.get("validation", {})
    require(validation.get("canonical_use") == "exploratory_reused_panels"
            and validation.get("seed_robustness", {}).get("status") in {"not_tested", "tested"}
            and validation.get("independent_post_selection", {}).get("status") in {"not_tested", "tested"}
            and validation.get("production") == "not_validated", "validation claims are conflated")
    for kind in ("seed_robustness", "independent_post_selection"):
        if validation[kind]["status"] == "tested":
            require(validation[kind].get("evidence"), "validation evidence is missing")
            for ref in validation[kind]["evidence"]:
                artifact(root, ref)
                refs.append(ref)
    archive_ref = study.get("archive_receipt", {})
    archive = read_json(artifact(root, archive_ref))
    require(archive.get("schema") == "unified-frontier-archive-closure-v1"
            and archive.get("status") == "verified"
            and archive.get("independent_download_and_member_verification") is True
            and archive.get("source_input_seed_environment_closure") is True
            and archive.get("study_id") == study["study_id"]
            and archive.get("remote_receipt_url") and archive.get("restore_command"),
            "verified remote archive closure is missing")
    members = archive.get("closed_artifacts", {})
    require(all(members.get(ref["path"]) == ref["sha256"] for ref in refs),
            "archive closure omits or changes study evidence")
    for ref in archive.get("verification_sources", []):
        artifact(root, ref)
    require(archive.get("verification_sources"), "archive verification receipts are missing")
    return copy.deepcopy(study)


def add_study(payload, study):
    require(payload.get("schema") == SCHEMA and isinstance(payload.get("studies"), list),
            "unexpected destination sidecar schema")
    result = copy.deepcopy(payload)
    existing = [row for row in result["studies"] if row.get("study_id") == study["study_id"]]
    require(len(existing) <= 1, "duplicate existing study ID")
    if existing:
        require(existing[0] == study, "immutable study ID conflicts; use a new study ID")
    else:
        result["studies"].append(copy.deepcopy(study))
    return result


def report(payload):
    lines = ["# Unified Frontier studies", "",
             "Research evidence only. Scores are separate by panel; compute timings are not browser CTV.", ""]
    for study in payload["studies"]:
        lines += ["## " + study["study_id"], "", study["purpose"], "",
                  "| Panel | Status | Canonical C-order empirical CRPS by label | Raw-result scalar differences |",
                  "| --- | --- | --- | --- |"]
        for panel, row in sorted(study["panels"].items()):
            values = "; ".join(f"{label}: {score:.17g}" for label, score in sorted(row.get("scores", {}).items()))
            differences = "; ".join(f"{label}: raw {raw:.17g}, {ulp_distance(raw, row['scores'][label])} ULP"
                                    for label, raw in sorted(row.get("raw_results_scores", {}).items())
                                    if raw != row["scores"][label])
            lines.append(f"| Full {panel} | {row['status']} | {values or 'Unscored'} | {differences or 'None'} |")
        lines += ["", "Runtime: " + study["runtime_evidence"]["status"] + ". "
                  + study["runtime_evidence"]["scope"], "", "Production: not validated.", "",
                  "Canonical panels: exploratory reused evidence. Seed robustness: "
                  + study["validation"]["seed_robustness"]["status"] + ". Independent post-selection: "
                  + study["validation"]["independent_post_selection"]["status"] + ".", ""]
        for variant in study["variants"]:
            lines += [f"- {variant['research_id']} ({variant['decision']}): {variant['rationale']}"]
        lines += ["", "Archive receipt: `" + study["archive_receipt"]["path"] + "`.", ""]
    return "\n".join(lines)


def prepare(root, packet, methodology_dir, master_dir):
    """Write reviewable copies only; directories must already exist and be distinct."""
    roots = [methodology_dir.resolve(), master_dir.resolve()]
    require(roots[0] != roots[1] and all(p.is_dir() for p in roots), "two destination directories required")
    study = validate_study(root.resolve(), read_json(packet))
    name = "unified_frontier_studies.json"
    paths = [folder / name for folder in roots]
    before = [read_json(path) if path.exists() else {"schema": SCHEMA, "studies": []} for path in paths]
    require(before[0] == before[1], "catalogue study sidecars differ before update")
    updated = add_study(before[0], study)
    content = json.dumps(updated, indent=2, sort_keys=True, allow_nan=False) + "\n"
    for path in paths:
        path.write_text(content, encoding="utf-8")
        path.with_suffix(".md").write_text(report(updated), encoding="utf-8")
    return updated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evidence-root", "study-packet", "methodology-dir", "master-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.evidence_root, args.study_packet, args.methodology_dir, args.master_dir)
