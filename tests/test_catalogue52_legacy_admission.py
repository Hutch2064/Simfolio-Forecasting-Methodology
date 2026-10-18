"""Legacy admission verifies real factories without claiming numerical parity."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "catalogue52"))
import legacy_admission as admission
from legacy_admission import legacy_admission


def test_all_175_have_exact_source_factories():
    report = legacy_admission()
    assert report["scope"] == "legacy175_constructor_admission_only"
    assert report["count"] == 175
    assert len({row["model_id"] for row in report["models"]}) == 175
    assert all(row["factory_instantiated"] for row in report["models"])
    assert sum(report["implementation_counts"].values()) == 175
    assert all("legacy_adapter" not in row["implementation"] for row in report["models"])


def test_mcmc_adapter_preserves_training_context_and_reuses_initializer(monkeypatch):
    calls = []
    training, context, output = object(), object(), object()

    class Delegate:
        def simulate_daily_log_returns(self, received_training, received_context):
            assert received_training is training
            assert received_context is context
            return output

    def initialize():
        calls.append("initialized")
        return Delegate()

    monkeypatch.setattr(admission, "initialize_parameter_mcmc", initialize)
    model = admission.ParameterMCMC()
    assert model.simulate_daily_log_returns(training, context) is output
    assert model.simulate_daily_log_returns(training, context) is output
    assert calls == ["initialized"]


def test_listing_candidates_does_not_install_mcmc(monkeypatch):
    def forbidden():
        raise AssertionError("listing must not install numerical hooks")

    monkeypatch.setattr(admission, "initialize_parameter_mcmc", forbidden)
    admission.initialize(None, 1, 1)
    assert [model.model_id for model in admission.CANDIDATES] == [
        "sv_parameter_mcmc_twochain_sixteen_node_moment_mixture",
        "experimental_filtered_innovation_moment_sv_fixed_mean",
        "experimental_filtered_innovation_moment_sv_dlm",
        "experimental_gaussian_moment_matched_sv_empirical_fixed_mean",
    ]


def test_variant_uses_exact_recovered_flags_and_restores_fitter_on_failure(monkeypatch):
    import pytest

    from simfolio_forecasting_methodology.models.asset_level.filtered_innovation_moment_sv import (
        FilteredInnovationFixedMeanMomentSV,
    )
    from simfolio_forecasting_methodology.models.numerical import bdes_fastmap as bd

    calls = []

    def original(values, **flags):
        calls.append((values, flags))
        return flags

    def simulate(self, training, context):
        result = bd.fit_bdes_fastmap(training, filtered_innovations=True, fixed_mean=True)
        if context == "fail":
            raise ValueError("source failure")
        return result

    monkeypatch.setattr(bd, "fit_bdes_fastmap", original)
    monkeypatch.setattr(FilteredInnovationFixedMeanMomentSV, "simulate_daily_log_returns", simulate)
    for filtered, fixed in ((True, False), (False, True)):
        variant = admission.MomentSVVariant("variant", filtered, fixed)
        assert variant.simulate_daily_log_returns("training", None) == {
            "filtered_innovations": filtered, "fixed_mean": fixed}
        assert bd.fit_bdes_fastmap is original
        with pytest.raises(ValueError, match="source failure"):
            variant.simulate_daily_log_returns("training", "fail")
        assert bd.fit_bdes_fastmap is original
