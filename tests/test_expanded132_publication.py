"""Verify the separately published equal-block protocol and retained evidence."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from simfolio_forecasting_methodology.origins import origin_schedule
from simfolio_forecasting_methodology.panel import generate_equal_class_history_panel

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "expanded132_publication_protocol", ROOT / "tools/expanded_panel/protocol.py"
)
protocol_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = protocol_module
spec.loader.exec_module(protocol_module)
block_cells = protocol_module.block_cells
combined_score = protocol_module.combined_score
OUT = ROOT / "docs/results/expanded132"
RESOURCES = ROOT / "src/simfolio_forecasting_methodology/resources"
REPORT = json.loads((OUT / "score-report.json").read_text())
PROTOCOL = yaml.safe_load((RESOURCES / "protocols/portfolio80_asset52_equal_weight_v1.yaml").read_text())


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def verified_file(reference):
    path = ROOT / reference["path"]
    assert digest(path.read_bytes()) == reference["sha256"]
    assert path.stat().st_size == reference["bytes"]
    return path


def test_protocol_membership_fingerprints_and_frozen_dates():
    protocol = dict(PROTOCOL)
    assert protocol.pop("protocol_fingerprint") == canonical(protocol) == REPORT["protocol_fingerprint"]
    verified_file(REPORT["protocol"])
    legacy_path = RESOURCES / "protocols/canonical_dense_oos.yaml"
    legacy = yaml.safe_load(legacy_path.read_text())
    assert digest(legacy_path.read_bytes()) == PROTOCOL["legacy_protocol"]["sha256"]
    assert legacy["identity"]["protocol_fingerprint"] == PROTOCOL["legacy_protocol"]["protocol_fingerprint"]
    tickers = sorted({ticker for p in generate_equal_class_history_panel() for ticker in p.tickers})
    assert tickers == PROTOCOL["blocks"]["single52"]["tickers"]
    assert len(tickers) == 52
    for key in ("normalized_return_matrix_sha256", "derived_asset_log_matrix_sha256",
                "derived_portfolio_log_matrix_sha256", "source_manifest_sha256", "calendar_sha256"):
        assert PROTOCOL["data"][key] == legacy.get("data", {}).get(key, legacy["identity"].get(key))
    dates = pd.read_csv(RESOURCES / "data/canonical_calendar.csv")["date"]
    assert (dates.iloc[0], dates.iloc[-1], len(dates)) == ("1979-12-31", "2026-05-13", 11687)
    schedule = [asdict(origin) for origin in origin_schedule(dates)]
    assert schedule == PROTOCOL["origins"]["schedule"]
    assert canonical(schedule) == PROTOCOL["origins"]["schedule_sha256"]
    assert len(schedule) == 51 and max(o["max_horizon"] for o in schedule) == 8766
    assert PROTOCOL["simulation"]["simulations_per_origin"] == 240
    assert sum(b["tasks"] for b in PROTOCOL["blocks"].values()) == 6732
    assert sum(b["cells"] for b in PROTOCOL["blocks"].values()) == 1157112


@pytest.mark.parametrize("name", REPORT["evidence"])
def test_results_and_full_manifest_files_are_sha_bound(name):
    evidence = REPORT["evidence"][name]
    result = json.loads(verified_file(evidence["result"]).read_text())
    if "result_selector" in evidence:
        result = result[evidence["result_selector"]]
        manifest = result["manifest"]
        assert result["audit"]["optimization_all_4080_vectors_byte_equal"] is True
        assert result["scores"][evidence["model_id"]]["exact_empirical_crps"] == evidence["score"]
        assert result["paired_cells_sha256"] == evidence["cells"]["sha256"]
    else:
        reference = evidence["manifest"]
        path = verified_file(reference)
        raw = gzip.decompress(path.read_bytes()) if reference["encoding"] == "gzip" else path.read_bytes()
        assert digest(raw) == reference["uncompressed_file_sha256"]
        manifest = json.loads(raw)
        manifest_digest = digest(json.dumps(manifest, sort_keys=True).encode())
        if reference.get("representation") == "portable_public_derivative":
            attestation = reference["derivation_attestation"]
            assert attestation["original_semantic_sha256"] == result["manifest_sha256"] == reference["canonical_sha256"]
            assert attestation["original_file_sha256"] == "75a7ae61daf3d1c767380a71787b4fbfb760af865a69f431485222a5987d9c1b"
            assert manifest_digest != result["manifest_sha256"]
            numerical = json.loads(json.dumps(manifest))
            normalization = attestation["normalization"]
            assert normalization["field_path"] == "environment.SIMFOLIO_RESEARCH_FIT_CACHE"
            assert numerical["environment"].pop("SIMFOLIO_RESEARCH_FIT_CACHE") == normalization["public_value"]
            assert canonical(numerical) == attestation["portable_numerical_identity_sha256"]
        else:
            assert manifest_digest == result["manifest_sha256"] == reference["canonical_sha256"]
        assert manifest["source_revision"] == evidence["source_binding"]["manifest_base_revision"]
        assert manifest["source_revision"] != evidence["source_binding"]["implementation_revision"]
        adapter = manifest["adapter"]
        assert digest((ROOT / adapter).read_bytes()) == manifest["source_hashes"][adapter]
        expected = PROTOCOL["blocks"][result["block"]]
        assert (result["portfolios"], result["tasks"], result["cells"]) == tuple(expected[k] for k in ("portfolios", "tasks", "cells"))
        assert result["all_denominator_gates_passed"] is True
        assert len(manifest["task_identities"]) == result["tasks"]
        assert manifest["indices"] == list(range(result["tasks"]))
        assert manifest["data_manifest_sha256"] == PROTOCOL["data"]["frozen_prepared_manifest_sha256"]
        assert result["exact_empirical_crps"] == evidence["score"]
    assert manifest["simulations"] == 240


def block_mean(name):
    evidence = REPORT["evidence"][name]
    with np.load(verified_file(evidence["cells"])) as saved:
        losses = saved["model_0"] if "model_0" in saved else saved["losses"]
        expected = PROTOCOL["blocks"][evidence["block"]]
        assert losses.size == expected["cells"]
        assert np.isfinite(losses).all() and (losses >= 0).all()
        assert len(set(saved["portfolio"])) == expected["portfolios"]
        if "counts" in saved:
            counts = np.zeros(8766, dtype=np.int64)
            for origin in PROTOCOL["origins"]["schedule"]:
                counts[:origin["max_horizon"]] += 1
            np.testing.assert_array_equal(saved["counts"], np.tile(counts, (expected["portfolios"], 1)))
        assert float(losses.mean()) == evidence["score"]
        return float(losses.mean())


def test_all_ranked_scores_recompute_from_both_complete_cell_blocks():
    assert [row["candidate"] for row in REPORT["aggregate_ranking"]] == [
        "pool", "ou-pool", "incumbent", "ewma"
    ]
    assert [row["rank"] for row in REPORT["aggregate_ranking"]] == [1, 2, 3, 4]
    scores = []
    for row in REPORT["aggregate_ranking"]:
        original, single = (block_mean(name) for name in row["evidence"])
        assert (original, single) == (row["original80"], row["single52"])
        assert combined_score(original, single) == row["equal_block_score"]
        assert row["equal_block_score"] != pytest.approx((80 * original + 52 * single) / 132)
        scores.append(row["equal_block_score"])
    assert scores == sorted(scores)
    for row in REPORT["partial_controls"].values():
        assert row["equal_block_score"] is None
    assert REPORT["garch"]["equal_block_score"] is None
    assert REPORT["raw_sbb"]["nonnegative_other_block_lower_bound"] > scores[0]
    audit = json.loads(verified_file(REPORT["independent_pool_audit"]).read_text())
    assert audit["passed"] and audit["equal_block_score"] == scores[0]
    assert audit["total_cells"] == 1157112 and audit["total_tasks"] == 6732
    assert REPORT["runtime_claim"] is False


def test_ou_pool_audits_bind_both_complete_blocks_and_accepted_fits():
    combined = json.loads(verified_file(REPORT["independent_ou_pool_audit"]).read_text())
    assert combined["all_full_block_audits_passed"] is True
    assert combined["equal_block_aggregate"] == REPORT["aggregate_ranking"][1]["equal_block_score"]
    assert combined["beat_best_pool"] is False
    tasks = cells = fits = 0
    for block in ("original80", "single52"):
        evidence = REPORT["evidence"]["ou-pool-" + block]
        audit = json.loads(verified_file(evidence["independent_audit"]).read_text())
        assert audit["all_checks_passed"] and audit["independent_reaggregation_exact"]
        assert audit["all_accepted_ou_optimizers_successful"] and audit["source_hash_guard_verified"]
        assert audit["actual_execution_git_sha"] == evidence["source_binding"]["implementation_revision"]
        assert audit["manifest_sha256"] == evidence["manifest"]["canonical_sha256"]
        assert audit["cells_sha256"] == evidence["cells"]["sha256"]
        assert audit["exact_empirical_crps"] == evidence["score"]
        assert audit["component_paths_each"] == 120 and audit["simulations"] == 240
        assert combined["blocks"][block]["same_targets_seeds_dates_horizons_counts"]
        tasks += audit["loss_vectors"]
        cells += audit["cells"]
        fits += audit["ou_diagnostic_records"]
    assert (tasks, cells, fits) == (6732, 1157112, 27132)


def test_pool_loss_decomposition_preserves_scores_inputs_and_horizon_coverage():
    reference = REPORT["pool_loss_decomposition"]
    receipt = json.loads(verified_file(reference["receipt"]).read_text())
    helper = verified_file(reference["helper"])
    assert digest(helper.read_bytes()) == receipt["helper_sha256"]
    for source in receipt["inputs"].values():
        assert digest((ROOT / source["path"]).read_bytes()) == source["sha256"]
    checks = receipt["canonical_score_checks"]
    for name, evidence in (("incumbent52", "incumbent-single52"),
                           ("incumbent80", "incumbent-original80"),
                           ("pool52", "pool-single52"), ("pool80", "pool-original80")):
        assert checks[name]["canonical_exact_match"] is True
        assert checks[name]["canonical_mean_recomputed"] == REPORT["evidence"][evidence]["score"]
    bands = receipt["horizon_bands"]
    assert bands[0]["days_inclusive"][0] == 1
    assert bands[-1]["days_inclusive"][1] == 8766
    for left, right in itertools.pairwise(bands):
        assert left["days_inclusive"][1] + 1 == right["days_inclusive"][0]
    total = sum(band["total_50_50_aggregate_gain_contribution"] for band in bands)
    assert total == pytest.approx(receipt["overall"]["equal_50_50_gain"], abs=2e-16)
    assets = receipt["asset_wins_losses_and_contributions"]
    assert (assets["pool_wins"], assets["pool_losses"], assets["ties"]) == (21, 31, 0)
    assert assets["SPYSIM"]["mean_crps_gain_incumbent_minus_pool"] < 0
    assert all(row["sign_flip"] is False for row in receipt["leave_one_out_sensitivity"].values())


@pytest.mark.parametrize("defect", ["missing", "short", "nan", "inf"])
def test_invalid_origin_losses_fail_closed(defect):
    from types import SimpleNamespace
    tasks = [SimpleNamespace(portfolio_id="single_test", horizon_days=3)]
    losses = [] if defect == "missing" else [np.ones(2 if defect == "short" else 3)]
    if defect in ("nan", "inf"):
        losses[0][0] = float(defect)
    with pytest.raises(ValueError):
        block_cells(tasks, losses)
    with pytest.raises(ValueError):
        combined_score(0.2, np.nan)


def test_garch_audit_and_bounded_runtime_attachments_are_sha_bound():
    audit = json.loads(verified_file(REPORT["garch"]["independent_audit"]).read_text())
    assert audit["passed"] and audit["independent_accumulator_all_cells_byte_equal"]
    assert (audit["tasks"], audit["cells"]) == (2652, 455832)
    assert audit["score"] == REPORT["garch"]["single52"]
    assert audit["source_revision"] == REPORT["evidence"]["garch-final-single52"]["source_binding"]["implementation_revision"]
    for reference in REPORT["garch"]["failed_attempt_receipts"]:
        assert "error" in json.loads(verified_file(reference).read_text())
    timing = json.loads(verified_file(REPORT["local_matched_runtime"]["receipt"]).read_text())
    assert timing["simulations"] == 240 and timing["threads"] == 1
    assert len(timing["records"]) == 48
    assert timing["source_revision"] == PROTOCOL["execution"]["tested_implementation_revision"]
    assert timing["data_manifest_sha256"] == PROTOCOL["data"]["frozen_prepared_manifest_sha256"]


@pytest.mark.parametrize("reference, original_sha", [
    (REPORT["local_matched_runtime"]["receipt"],
     "a59a0edd8040934e1ce7ef4088e5cf6036c7d3870b4034a3d15a1f1957aaa4db"),
    (REPORT["final_three_model_runtime"]["receipt"],
     "92beb69643e31f19a3ec816203036c333bf8f30cd8c447a0ac94dd92a40341e1"),
    (REPORT["independent_ou_pool_audit"],
     "2ccb97b844efedcd9e3b5ebd37596ff5e1b46791bd8c2d17709a71ac987e1b56"),
])
def test_portable_receipts_attest_path_only_derivation(reference, original_sha):
    from simfolio_forecasting_methodology.safety_scan import scan_text
    path = verified_file(reference)
    assert scan_text(path.read_text()) == []
    payload = json.loads(path.read_text())
    attestation = payload.pop("publication_derivation")
    assert attestation["original_file_sha256"] == original_sha
    assert canonical(payload) == attestation["normalized_payload_sha256"]
    assert attestation["normalized_json_pointers"]
    for pointer in attestation["normalized_json_pointers"]:
        parts = [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]
        parent = payload
        for part in parts[:-1]:
            parent = parent[int(part)] if isinstance(parent, list) else parent[part]
        key = int(parts[-1]) if isinstance(parent, list) else parts[-1]
        assert isinstance(parent[key], str) and not Path(parent[key]).is_absolute()
        parent[key] = "<campaign-relative-path>"
    assert canonical(payload) == attestation["path_elided_payload_sha256"]


def test_final_three_model_runtime_and_static_mean_pruning_receipts():
    runtime = json.loads(verified_file(REPORT["final_three_model_runtime"]["receipt"]).read_text())
    assert len(runtime["records"]) == 72
    assert runtime["simulations"] == 240 and runtime["threads"] == 1
    assert all(row["all_loss_hashes_match_previous_receipt"]
               and row["all_path_hashes_match_previous_receipt"]
               for row in runtime["prior_receipt_parity"])
    ablation = REPORT["static_mean_ablation"]
    assert block_mean("static-mean-single52") == ablation["single52"]
    audit = json.loads(verified_file(ablation["independent_audit"]).read_text())
    assert audit["passed"] and audit["independent_accumulator_all_cells_byte_equal"]
    assert (audit["tasks"], audit["cells"]) == (2652, 455832)
    evidence = REPORT["evidence"]["static-mean-single52"]
    assert audit["source_revision"] == evidence["source_binding"]["implementation_revision"]
    assert audit["manifest_semantic_sha256"] == evidence["manifest"]["canonical_sha256"]
    assert audit["cells_sha256"] == evidence["cells"]["sha256"]
    assert all(row["all_horizon_loss_replay_byte_equal"]
               and row["independent_pairwise_crps_max_abs_error"] < 2e-17
               for row in audit["replays"])
    assert ablation["equal_block_score"] is None
    assert 0.5 * ablation["single52"] == ablation["nonnegative_other_block_lower_bound"]
    assert ablation["nonnegative_other_block_lower_bound"] > REPORT["aggregate_ranking"][0]["equal_block_score"]


@pytest.mark.parametrize("key, folder_name, count", [
    ("long_cagr", "long-cagr", 224), ("long_cagr_ou", "long-cagr-ou", 112)
])
def test_all_prospective_case_archives_and_statistics_are_retained(key, folder_name, count):
    from scipy.special import logsumexp
    receipt = json.loads(verified_file(REPORT[key]["receipt"]).read_text())
    if "comparison" in REPORT[key]:
        verified_file(REPORT[key]["comparison"])
    verified_file(REPORT[key]["script"])
    assert receipt["passed"] and receipt["completed_cases"] == count
    assert receipt["independent_statistic_validation"]["passed"]
    folder = OUT / folder_name
    for name, expected in receipt["files"].items():
        path = folder / name
        assert digest(path.read_bytes()) == expected["sha256"]
        assert path.stat().st_size == expected["bytes"]
    records = sorted(folder.glob("*--*y.json"))
    assert len(records) == count
    for path in records:
        record = json.loads(path.read_text())
        archive = path.with_suffix(".npz")
        assert digest(archive.read_bytes()) == record["terminal_file_sha256"]
        with np.load(archive) as saved:
            terminal = saved["terminal_log_nav"]
        assert terminal.shape == (240,) and np.isfinite(terminal).all()
        horizon = record["horizon_days"]
        assert horizon in (12600, 25200)
        cagr = np.expm1(252 * terminal / horizon)
        assert np.median(cagr) == pytest.approx(record["median_path_cagr"], abs=2e-16)
        assert cagr.mean() == pytest.approx(record["mean_path_cagr"], abs=2e-16)
        nav_cagr = np.expm1(252 * (logsumexp(terminal) - np.log(240)) / horizon)
        assert nav_cagr == pytest.approx(record["cagr_of_mean_ending_nav"], abs=2e-16)


def test_final_screening_and_stopping_decision_are_bound_to_completed_evidence():
    screening = json.loads(verified_file(REPORT["static_mean_ablation"]["mixture_followup_screening"]).read_text())
    scores = screening["overall"]["canonical_rowwise_scores"]
    assert scores["static-mean-single52"] == REPORT["static_mean_ablation"]["single52"]
    comparisons = screening["overall"]["ticker_win_loss_counts"]
    assert comparisons["incumbent-single52"]["ticker_wins"] == 9
    assert comparisons["pool-single52"]["ticker_wins"] == 11
    assert screening["overall"]["ticker_level_signed_gain_correlations"]["pearson_static_gain_vs_pool_failure"] < 0
    assert REPORT["research_decision"]["best_validated_aggregate_model"] == REPORT["evidence"][REPORT["aggregate_ranking"][0]["evidence"][0]]["model_id"]
    assert REPORT["research_decision"]["no_production_change"] is True
