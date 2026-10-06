from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from simfolio_forecasting_methodology import cli
from simfolio_forecasting_methodology.candidate_registry import candidate_execution_record
from simfolio_forecasting_methodology.catalogue import load_canonical_ledger
from simfolio_forecasting_methodology.models.numerical import bdes_fastmap as bd
from simfolio_forecasting_methodology.results.verified_candidate_scores import (
    verified_candidate_score_report,
)

_CANDIDATE_ID = "experimental_filtered_innovation_moment_sv_fixed_mean"


def test_six_volatility_ablation_scores_bind_to_complete_paired_cells():
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads((root / "docs/results/rough-bayesian/volatility-frontier-ablations.json").read_text())
    evidence = root / receipt["paired_cells"]["path"]
    assert hashlib.sha256(evidence.read_bytes()).hexdigest() == receipt["paired_cells"]["sha256"]
    assert receipt["audit"]["passed"] is True
    assert receipt["audit"]["origin_vectors"] == 24480
    ledger = {row["public_model_id"]: row for row in load_canonical_ledger()["models"]}
    with np.load(evidence) as cells:
        assert len(cells["portfolio"]) == 701280
        assert len(np.unique(cells["portfolio"])) == 80
        assert float(cells["baseline"].mean()) == receipt["baseline_crps"]
        for candidate in receipt["candidates"]:
            model_id = candidate["model_id"]
            score = float(cells[candidate["cell_array"]].mean())
            assert np.isfinite(cells[candidate["cell_array"]]).all()
            assert score == receipt["scores"][model_id]["exact_empirical_crps"]
            assert str(score) == ledger[model_id]["historical_score"]["exact_empirical_crps"]
            assert ledger[model_id]["historical_score_verified"] is True


@pytest.mark.parametrize("filename,origins", [("spectral-rough-panels", 8160), ("gamma-supou-panel", 4080), ("reml-rough-panel", 4080), ("whittle-gls-rough-panel", 4080), ("whittle-mle-rough-panel", 4080), ("whittle-integrated-level-rough-panel", 4080), ("exact-covariance-whittle-rough-panel", 4080), ("full-hurst-domain-rough-panel", 4080), ("analytic-differenced-rough-panel", 4080), ("relaxed-hurst-differenced-rough-panel", 4080), ("full-hurst-differenced-rough-panel", 4080), ("consistent-noise-rough-panel", 4080), ("parametric-innovation-panels", 8160), ("gaussian-innovation-panel", 4080), ("joint-noise-sv-panel", 4080), ("continuous-quantile-panel", 4080), ("raw-return-laplace-panel", 4080), ("student-return-laplace-panel", 4080), ("student-implied-noise-panel", 4080), ("pathwise-multiscale-panel", 4080), ("joint-leverage-panel", 4080), ("student-full-hurst-panel", 4080), ("student-joint-rough-noise-panel", 4080), ("matched-student-innovations-panel", 4080), ("eb-multiscale-panel", 4080), ("untruncated-priors-panel", 4080)])
def test_spectral_scores_bind_to_complete_paired_cells(filename, origins):
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads((root / f"docs/results/rough-bayesian/{filename}.json").read_text())
    evidence = root / receipt["paired_cells"]["path"]
    assert hashlib.sha256(evidence.read_bytes()).hexdigest() == receipt["paired_cells"]["sha256"]
    assert receipt["audit"]["passed"] is True
    assert receipt["audit"]["origin_vectors"] == origins
    ledger = {row["public_model_id"]: row for row in load_canonical_ledger()["models"]}
    with np.load(evidence) as cells:
        assert len(cells["portfolio"]) == 701280
        assert len(np.unique(cells["portfolio"])) == 80
        for candidate in receipt["candidates"]:
            model_id = candidate["model_id"]
            score = float(cells[candidate["cell_array"]].mean())
            assert np.isfinite(cells[candidate["cell_array"]]).all()
            assert score == receipt["scores"][model_id]["exact_empirical_crps"]
            assert str(score) == ledger[model_id]["historical_score"]["exact_empirical_crps"]
            assert candidate["source_revision"] == ledger[model_id]["source_revision"]


def test_combined_candidate_ranking_preserves_ledger_and_evidence_origins():
    ledger = load_canonical_ledger()
    report = verified_candidate_score_report()
    artifact_path = (
        Path(__file__).resolve().parents[1]
        / "docs/results/combined-176-score-ranking.json"
    )
    artifact_text = artifact_path.read_text(encoding="utf-8")
    assert json.loads(artifact_text) == report
    assert all(path not in artifact_text for path in ("/Users/", "/private/", "/tmp/"))

    assert report["row_count"] == 227
    assert report["retained_historical_row_count"] == 175
    assert report["independently_audited_new_execution_row_count"] == 0
    assert report["validated_full_panel_row_count"] == 42
    assert report["full_176_model_reproduction"] is False
    assert report["canonical_membership_unchanged"] is True
    assert report["canonical_membership_digest"] == ledger["membership"]["membership_digest"]

    rows = report["rows"]
    assert [row["display_rank"] for row in rows] == list(range(1, 218)) + [None] * 10
    assert report["partial_panel_row_count"] == 4
    assert report["unscored_candidate_row_count"] == 6
    candidate = next(row for row in rows if row["public_model_id"] == _CANDIDATE_ID)
    assert candidate["public_model_id"] == _CANDIDATE_ID
    assert candidate["exact_empirical_crps"] == "0.25439860867855635"
    assert candidate["canonical_rank"] == 177
    assert candidate["is_new_execution"] is False
    assert candidate["score_evidence_origin"] == "validated_full_canonical_execution"

    historical_rows = {row["public_model_id"]: row for row in rows if not row["is_new_execution"]}
    assert len(historical_rows) == 227
    for model in ledger["models"]:
        row = historical_rows[model["public_model_id"]]
        assert row["canonical_rank"] == model["canonical_rank"]
        assert row["historical_rank"] == model["historical_rank"]
        assert row["exact_empirical_crps"] == model["historical_score"]["exact_empirical_crps"]
        assert row["verification_status"] == model["verification_status"]

    paired = report["paired_run_models"]
    assert [row["evidence_origin"] for row in paired] == [
        "newly_executed_in_filtered_candidate_run",
        "newly_executed_in_filtered_candidate_run",
        "reused_independently_audited_incumbent_reference",
    ]
    replay = report["full_candidate_replay"]
    assert replay["tasks"] == 4080
    assert replay["cells_per_model"] == 701280
    assert replay["simulations_per_origin"] == 240
    assert replay["every_loss_vector_byte_equal"] is True
    assert replay["exact_empirical_crps"] == candidate["exact_empirical_crps"]
    assert replay["validated_numerical_kernels"]["native_moment_function_ast_equal"] == {
        "predictive_state_moments": True,
        "moment_return_curves": True,
    }
    cross_runtime = replay["cross_runtime_marginal_parity"]
    assert cross_runtime["assets"] == 52
    assert cross_runtime["cases"] == 156
    assert cross_runtime["all_checked_fields_exact"] is True
    assert cross_runtime["empirical_quantile_node_values"] == 37440
    assert cross_runtime["empirical_quantile_nodes_exact"] is True
    assert replay["replay_driver"]["cli_qualification"].startswith(
        "The recorded CLI source hash predates"
    )
    assert report["selection_and_inference"]["candidate_search_family_size"] == 23
    assert report["selection_and_inference"]["selected_adaptively"] is True


def test_candidate_plan_selector_is_separate_and_keeps_canonical_task_counts(capsys):
    assert (
        cli.main(
            [
                "candidate",
                "--model",
                _CANDIDATE_ID,
                "--plan",
                "--json",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["scope"] == "verified_candidate_same_canonical_dense_protocol"
    assert plan["model_ids"] == [_CANDIDATE_ID]
    assert plan["origin_tasks_per_model"] == 4080
    assert plan["scored_cells_per_model"] == 701280
    assert plan["protocol"]["simulations_per_origin"] == 240

    record = candidate_execution_record(_CANDIDATE_ID)
    seed_specification = {
        "model_id": _CANDIDATE_ID,
        "filtered_innovations": "eps_x * exp(-0.5 * state_path)",
        "mean": "fixed_historical_mean",
        "predictive_distribution": "Gaussian_moment_matched",
        "copula_seed": [
            "copula_alternatives",
            bd.FRONTIER_DEPENDENCE_ID,
            "origin_date",
            "horizon",
            "simulations",
        ],
    }
    expected_fingerprint = hashlib.sha256(
        json.dumps(seed_specification, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    assert record["specification_fingerprint"]["value"] == expected_fingerprint


def test_public_combined_command_and_historical_score_command_keep_distinct_scopes(capsys):
    assert cli.main(["candidate-scores", "--json"]) == 0
    combined = json.loads(capsys.readouterr().out)
    assert combined["rows"][0]["public_model_id"] == "asset_map_student_implied_noise_pathwise_multiscale_untruncated_rough_priors"
    assert combined["rows"][0]["display_rank"] == 1
    assert combined["rows"][1]["canonical_rank"] == 221

    assert cli.main(["scores", "--json"]) == 0
    retained = json.loads(capsys.readouterr().out)
    assert retained["result_kind"] == "retained_score_evidence"
    assert retained["is_new_execution"] is False
    assert len(retained["rows"]) == 227
