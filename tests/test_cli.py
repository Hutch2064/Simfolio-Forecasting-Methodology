from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from simfolio_forecasting_methodology import cli, experiment


def test_obsolete_master_harness_command_is_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["master-research"])


def test_canonical_cli_inspection_is_ledger_scoped(capsys):
    assert cli.main(["coverage", "--json"]) == 0
    coverage = json.loads(capsys.readouterr().out)
    assert coverage["scope"] == "canonical_175_only"
    assert coverage["model_count"] == 175
    assert coverage["counts"]["full_statistical_specifications_confirmed"] == 159

    assert cli.main(["scores", "--json"]) == 0
    scores = json.loads(capsys.readouterr().out)
    assert scores["result_kind"] == "retained_score_evidence"
    assert scores["is_new_execution"] is False
    assert scores["reproduced"] is False
    assert len(scores["rows"]) == 175
    assert scores["retained_score_artifact"]["digest_status"] == "mismatch_observed_vs_declared"
    assert isinstance(scores["rows"][0]["retained_score_token"], str)


def test_cli_rejects_unknown_ids_without_substring_aliases():
    with pytest.raises(SystemExit, match="unknown canonical model ID"):
        cli.main(["scores", "--model", "frontier"])
    with pytest.raises(SystemExit, match="unknown canonical model ID"):
        cli.main(["canonical-175", "--model", "frontier"])


def test_canonical_execution_defaults_to_all_175_ids():
    args = cli.build_parser().parse_args(["canonical-175"])

    selected = cli._selected_models(args, command="canonical-175")

    assert len(selected) == 175
    assert selected == tuple(row["public_model_id"] for row in cli.load_canonical_models())


def test_canonical_and_smoke_simulation_contracts_fail_before_model_or_data_access():
    with pytest.raises(SystemExit, match="canonical runs require --simulations 240"):
        cli.main(["canonical-175", "--frontier", "--simulations", "32"])
    with pytest.raises(SystemExit, match="smoke runs require --simulations <= 32"):
        cli.main(["smoke", "--simulations", "33"])


def test_smoke_selection_cannot_expand_to_all_models():
    with pytest.raises(SystemExit, match="smoke accepts one model"):
        cli.main(["canonical-175", "--all", "--smoke"])
    with pytest.raises(SystemExit, match="smoke accepts one model"):
        cli.main(["smoke", "--all"])


def test_plan_fingerprint_is_derived_from_schedule_masks(monkeypatch):
    class Descriptor:
        origin_label = "rolling_01"
        evaluation_split = "rolling_origin"
        position = 1
        origin_date = "2020-01-02"
        max_horizon = 2
        train_fraction = None

        def horizon_mask(self, length):
            del length
            return [True, True]

    schedule = SimpleNamespace(
        common_dates=pd.date_range("2020-01-01", periods=4, freq="D"),
        portfolios=(SimpleNamespace(name="p1"),),
        tasks=(SimpleNamespace(portfolio_id="p1", portfolio_index=1, descriptor=Descriptor()),),
        task_count=1,
        cell_capacity=2,
    )
    monkeypatch.setattr(experiment, "build_experiment_schedule", lambda: schedule, raising=False)

    payload = cli._plan_payload()

    assert payload["origin_tasks_per_model"] == 1
    assert payload["scored_cells_per_model"] == 2
    assert len(payload["schedule_fingerprint"]) == 64
    assert len(payload["model_ids"]) == 175
