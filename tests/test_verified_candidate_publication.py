from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simfolio_forecasting_methodology import cli
from simfolio_forecasting_methodology.candidate_registry import candidate_execution_record
from simfolio_forecasting_methodology.catalogue import load_canonical_ledger
from simfolio_forecasting_methodology.models.numerical import bdes_fastmap as bd
from simfolio_forecasting_methodology.results.verified_candidate_scores import (
    verified_candidate_score_report,
)

_CANDIDATE_ID = "experimental_filtered_innovation_moment_sv_fixed_mean"


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

    assert report["row_count"] == 183
    assert report["retained_historical_row_count"] == 175
    assert report["independently_audited_new_execution_row_count"] == 0
    assert report["validated_full_panel_row_count"] == 5
    assert report["full_176_model_reproduction"] is False
    assert report["canonical_membership_unchanged"] is True
    assert report["canonical_membership_digest"] == ledger["membership"]["membership_digest"]

    rows = report["rows"]
    assert [row["display_rank"] for row in rows] == list(range(1, 181)) + [None] * 3
    assert report["partial_panel_row_count"] == 3
    candidate = next(row for row in rows if row["public_model_id"] == _CANDIDATE_ID)
    assert candidate["public_model_id"] == _CANDIDATE_ID
    assert candidate["exact_empirical_crps"] == "0.25439860867855635"
    assert candidate["canonical_rank"] == 177
    assert candidate["is_new_execution"] is False
    assert candidate["score_evidence_origin"] == "validated_full_canonical_execution"

    historical_rows = {row["public_model_id"]: row for row in rows if not row["is_new_execution"]}
    assert len(historical_rows) == 183
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
    assert combined["rows"][0]["public_model_id"] == "asset_rough_volterra_sv_eight_factor"
    assert combined["rows"][0]["display_rank"] == 1
    assert combined["rows"][1]["canonical_rank"] == 176

    assert cli.main(["scores", "--json"]) == 0
    retained = json.loads(capsys.readouterr().out)
    assert retained["result_kind"] == "retained_score_evidence"
    assert retained["is_new_execution"] is False
    assert len(retained["rows"]) == 183
