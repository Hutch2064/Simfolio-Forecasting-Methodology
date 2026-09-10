import numpy as np

from simfolio_forecasting_methodology.runner import (
    TERMINAL_FORECAST_SEMANTICS,
    OriginTask,
    TrainingData,
    evaluate_model,
    evaluate_origin_task,
)


class ZeroPathModel:
    model_id = "test_zero"

    def simulate_daily_log_returns(self, training, context):
        del training
        return np.zeros((context.simulations, context.horizon_days), dtype=float)


class TerminalEnsembleModel:
    model_id = "test_terminal"
    forecast_output_semantics = TERMINAL_FORECAST_SEMANTICS

    def simulate_terminal_log_returns(self, training, context):
        del training
        return np.tile(np.array([[0.2, 0.3]], dtype=float), (context.simulations, 1))


def test_origin_task_scores_every_daily_horizon():
    task = OriginTask(
        portfolio_id="p1",
        origin_label="o1",
        training=TrainingData(portfolio_log_returns=np.array([0.01, -0.02, 0.03])),
        realized_future_daily_log_returns=np.array([0.1, -0.05, 0.02]),
        seed=17,
    )
    losses = evaluate_origin_task(ZeroPathModel(), task, simulations=4)
    expected_terminal = np.array([0.1, 0.05, 0.07])
    assert np.allclose(losses, np.abs(expected_terminal), atol=1e-15, rtol=0.0)


def test_model_evaluation_streams_origins_into_cells():
    tasks = [
        OriginTask(
            portfolio_id="p1",
            origin_label="o1",
            training=TrainingData(portfolio_log_returns=np.array([0.0, 0.0])),
            realized_future_daily_log_returns=np.array([0.1, 0.1]),
            seed=1,
        ),
        OriginTask(
            portfolio_id="p1",
            origin_label="o2",
            training=TrainingData(portfolio_log_returns=np.array([0.0, 0.0])),
            realized_future_daily_log_returns=np.array([0.3]),
            seed=2,
        ),
    ]
    result = evaluate_model(ZeroPathModel(), tasks, simulations=3)
    means = result.cell_means()
    assert set(means) == {("p1", 1), ("p1", 2)}
    assert np.isclose(means[("p1", 1)], 0.2, atol=1e-15, rtol=0.0)
    assert np.isclose(means[("p1", 2)], 0.2, atol=1e-15, rtol=0.0)
    assert np.isclose(result.aggregate_score(), 0.2, atol=1e-15, rtol=0.0)


def test_terminal_ensemble_extension_does_not_reconstruct_daily_increments():
    task = OriginTask(
        portfolio_id="p1",
        origin_label="o1",
        training=TrainingData(portfolio_log_returns=np.array([0.0, 0.0])),
        realized_future_daily_log_returns=np.array([0.1, 0.1]),
        seed=3,
    )
    losses = evaluate_origin_task(TerminalEnsembleModel(), task, simulations=2)
    # Direct terminal samples [0.2, 0.3] score against terminal observations
    # [0.1, 0.2]. A daily cumsum would incorrectly produce [0.2, 0.5].
    assert np.allclose(losses, [0.1, 0.1], atol=1e-15, rtol=0.0)
