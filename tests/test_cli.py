from __future__ import annotations

import json

import pytest

from simfolio_forecasting_methodology import cli


def test_obsolete_master_harness_command_is_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["master-research"])


def test_canonical_cli_inspection_is_ledger_scoped(capsys):
    assert cli.main(["coverage", "--json"]) == 0
    coverage = json.loads(capsys.readouterr().out)
    assert coverage["scope"] == "canonical_175_only"
    assert coverage["model_count"] == 175
    assert coverage["counts"]["full_statistical_specifications_confirmed"] == 0

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
