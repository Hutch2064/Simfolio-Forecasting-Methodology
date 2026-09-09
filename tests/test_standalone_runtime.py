import numpy as np
import pandas as pd

from simfolio_forecasting_methodology.catalog import load_canonical_175
from simfolio_forecasting_methodology.harnesses import master_plan
from simfolio_forecasting_methodology.models.frontier import FRONTIER_MODEL_ID
from simfolio_forecasting_methodology.models.frontier_calendar import CalendarFrontierModel
from simfolio_forecasting_methodology.models.registry import build_model, registration
from simfolio_forecasting_methodology.runner import ForecastContext, PortfolioPolicy, TrainingData
from simfolio_forecasting_methodology.scores import canonical_scores


def _training() -> TrainingData:
    rng = np.random.default_rng(17)
    assets = rng.normal(0.0002, 0.01, size=(420, 3))
    portfolio = np.mean(assets, axis=1)
    return TrainingData(
        portfolio_log_returns=portfolio,
        asset_log_returns=assets,
        policy=PortfolioPolicy(("A", "B", "C"), (1 / 3, 1 / 3, 1 / 3), "monthly"),
    )


def test_frontier_standalone_runtime_is_deterministic_and_finite():
    model = CalendarFrontierModel()
    dates = pd.bdate_range("2026-01-02", periods=42).to_numpy()
    context = ForecastContext(FRONTIER_MODEL_ID, "p1", "o1", 42, 12, 1234, dates)
    first = model.simulate_daily_log_returns(_training(), context)
    second = model.simulate_daily_log_returns(_training(), context)
    assert first.shape == (12, 42)
    assert np.all(np.isfinite(first))
    assert np.array_equal(first, second)


def test_every_canonical_and_master_id_resolves_to_a_model():
    ids = set(row.model_id for row in load_canonical_175()) | set(master_plan().model_ids)
    for model_id in ids:
        model = build_model(model_id)
        assert model.model_id == model_id
        assert registration(model_id).fidelity


def test_retained_score_registry_is_complete_and_frontier_is_first():
    scores = canonical_scores()
    assert len(scores) == 175
    assert scores[0].model_id == FRONTIER_MODEL_ID
    assert scores[0].canonical_rank == 1
    assert scores[0].cells == 701_280
    assert scores[0].exact_empirical_crps > 0.0
