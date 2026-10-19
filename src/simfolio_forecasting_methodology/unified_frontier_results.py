"""Portable additive unified Frontier research evidence; raw archives remain private."""
import json
import math
import re
import struct
import sys
from importlib.resources import files
from pathlib import Path

DEFAULT_SOURCE = files(__package__).joinpath("resources/results/unified_frontier_studies.json")

PANELS = {'52': (52, 2652, 455832, 3904212), '80': (80, 4080, 701280, 6006480)}
EVIDENCE_ROLES = {'analysis', 'audit', 'cells', 'completion', 'manifest', 'reference_manifest', 'results'}

def require(condition, message):
    if not condition:
        raise ValueError(message)


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


def validate_unified_frontier_index(payload):
    """Check portable metadata without reading authenticated private raw evidence."""
    require(payload.get("schema") == "unified-frontier-studies-v1"
            and isinstance(payload.get("studies"), list) and payload["studies"], "invalid study index")
    ids = [row.get("study_id") for row in payload["studies"]]
    require(len(ids) == len(set(ids)), "study IDs must be unique")
    for study in payload["studies"]:
        require(isinstance(study.get("study_id"), str) and study["study_id"].startswith("UF-")
                and study.get("uniform_asset_law") is True, "missing unified study identity")
        require(set(study.get("panels", {})) == {"52", "80"} and "blended_score" not in study,
                "separate full52 and full80 panels are required")
        sources = study.get("sources", [])
        require(sources and all(isinstance(ref.get("path"), str) and not Path(ref["path"]).is_absolute()
                and ".." not in Path(ref["path"]).parts and re.fullmatch(r"[0-9a-f]{64}", str(ref.get("sha256")))
                for ref in sources), "source hash references are invalid")
        for panel, row in study["panels"].items():
            require(row.get("status") in {"not_evaluated", "complete_scored_and_audited"}, "invalid panel status")
            if row["status"] == "not_evaluated":
                require(set(row) == {"status", "reason"}, "unscored panels cannot carry scores")
                continue
            entities, origins, cells, contributions = PANELS[panel]
            expected = {"origins": origins, "portfolios": entities, "simulations": 240,
                        "cells_per_model": cells, "origin_horizon_contributions": contributions, "horizons": 8766}
            require(all(row.get("denominators", {}).get(key) == value for key, value in expected.items()),
                    "canonical panel denominators differ")
            scores = row.get("scores", {})
            require(scores and all(finite_score(value) for value in scores.values()), "invalid panel scores")
            labels = {variant.get("panel_labels", {}).get(panel, variant["label"]) for variant in study["variants"]}
            require(labels == set(scores) and "M261_saved_control" in scores, "variant/control scores missing")
            require(set(row.get("evidence", {})) == EVIDENCE_ROLES and all(
                re.fullmatch(r"[0-9a-f]{64}", str(ref.get("sha256"))) for ref in row["evidence"].values()),
                "complete panel evidence hashes missing")
            require("score_aggregation" in row, "canonical scalar aggregation proof missing")
            validate_score_aggregation({"score_aggregation": row["score_aggregation"]}, row,
                                       row.get("raw_results_scores"), cells)
        validation = study.get("validation", {})
        require(validation.get("canonical_use") == "exploratory_reused_panels"
                and validation.get("production") == "not_validated"
                and all(validation.get(kind, {}).get("status") in {"not_tested", "tested"}
                        for kind in ("seed_robustness", "independent_post_selection")),
                "validation evidence types must remain distinct")
        runtime = study.get("runtime_evidence", {})
        require(runtime.get("production_browser_ctv") is False and runtime.get("scope")
                and runtime.get("status") in {"not_measured", "representative_matched_compute"},
                "research compute is not browser CTV")
        archive = study.get("archive_summary", {})
        require(archive.get("status") == "verified_restored_scientific_smoke"
                and archive.get("private") is True and isinstance(archive.get("restored_oracles"), int)
                and not isinstance(archive.get("restored_oracles"), bool) and archive["restored_oracles"] > 0
                and archive.get("all_path_and_CRPS_oracles_byte_equal") is True
                and re.fullmatch(r"[0-9a-f]{64}", str(study.get("archive_receipt", {}).get("sha256"))),
                "verified private archive references missing")
        if study.get("selection", {}).get("joint_objective_met") is True:
            winners = [variant for variant in study["variants"] if variant["decision"] == "retained"
                       and all(row.get("status") == "complete_scored_and_audited"
                           and row["scores"][variant.get("panel_labels", {}).get(panel, variant["label"])]
                           < row["scores"]["M261_saved_control"] for panel, row in study["panels"].items())]
            require(winners, "joint objective cannot be claimed without one law winning both panels")


def load_unified_frontier_studies(path=None):
    """Load this additive research sidecar without changing existing catalogue ranks."""
    source = DEFAULT_SOURCE if path is None else Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    validate_unified_frontier_index(payload)
    return payload
