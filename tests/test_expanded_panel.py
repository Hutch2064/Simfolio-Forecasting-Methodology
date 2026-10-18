"""Independent checks for the new block weights and denominator reduction."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from simfolio_forecasting_methodology.evaluation import CellAccumulator

spec = importlib.util.spec_from_file_location(
    "expanded_panel_protocol", Path(__file__).parents[1] / "tools/expanded_panel/protocol.py"
)
protocol = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = protocol
spec.loader.exec_module(protocol)


def test_equal_block_weights_are_not_equal_portfolio_weights():
    assert protocol.combined_score(0.2, 0.4) == pytest.approx(0.3)
    assert protocol.combined_score(0.2, 0.4) != pytest.approx((80 * 0.2 + 52 * 0.4) / 132)


def test_block_reduction_matches_canonical_accumulator():
    tasks = [SimpleNamespace(portfolio_id=p, horizon_days=h)
             for p, h in [("a", 4), ("a", 2), ("b", 4), ("b", 3)]]
    losses = [np.arange(t.horizon_days, dtype=float) + i for i, t in enumerate(tasks)]
    canonical = CellAccumulator()
    for task, vector in zip(tasks, losses):
        canonical.add_vector(task.portfolio_id, vector)
    names, cells, counts = protocol.block_cells(tasks, losses)
    expected = np.array([[canonical.cell_means()[name, h] for h in range(1, 5)] for name in names])
    np.testing.assert_array_equal(cells, expected)
    np.testing.assert_array_equal(counts, [[2, 2, 1, 1], [2, 2, 2, 1]])


@pytest.mark.parametrize("defect", ["missing", "short", "nonfinite"])
def test_incomplete_or_invalid_losses_fail_closed(defect):
    tasks = [SimpleNamespace(portfolio_id="a", horizon_days=4)]
    losses = [np.ones(4)]
    if defect == "missing":
        losses = []
    elif defect == "short":
        losses = [np.ones(3)]
    else:
        losses[0][1] = np.nan
    with pytest.raises(ValueError):
        protocol.block_cells(tasks, losses)
