"""Checks that a built distribution exposes the public offline contract."""

from __future__ import annotations

import os
from importlib import metadata, resources

import pytest

from simfolio_forecasting_methodology.catalogue import load_canonical_models
from simfolio_forecasting_methodology.data import verify_canonical_snapshot
from simfolio_forecasting_methodology.experiment import build_experiment_schedule
from simfolio_forecasting_methodology.models.registry import build_model


def test_installed_package_contains_the_offline_canonical_snapshot():
    if os.environ.get("SIMFOLIO_WHEEL_TEST") != "1":
        pytest.skip("the distribution metadata check runs in the clean-wheel job")
    snapshot = resources.files("simfolio_forecasting_methodology").joinpath(
        "resources", "data", "canonical_snapshot"
    )
    assert snapshot.is_dir()
    extras = metadata.metadata("simfolio-forecasting-methodology").get_all("Provides-Extra") or []
    assert "all-models" in extras

    verification = verify_canonical_snapshot()
    assert verification["series_count"] == 52
    assert verification["supporting_series"] == ["BNDSIM", "CASHX", "EFFRX", "KMLMSIM", "TIPSIM", "UUPSIM"]
    assert verification["factor_inputs"] == ["french_daily", "q5_daily"]
    assert verification["redistribution_status"] == "user_authorized_exact_snapshot"

    schedule = build_experiment_schedule()
    assert len(schedule.portfolios) == 80
    assert schedule.task_count == 4_080
    assert schedule.cell_capacity == 701_280


def test_every_registered_factory_instantiates_from_the_installed_package():
    rows = load_canonical_models()
    executable = [row for row in rows if row["implementation_factory"]["callable"]]
    assert len(rows) == 175
    assert len(executable) == 175

    for row in executable:
        model = build_model(row["public_model_id"])
        assert model.model_id == row["public_model_id"]
