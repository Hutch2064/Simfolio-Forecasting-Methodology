"""Public, standalone evaluation protocol definitions.

No production service package is imported here.  The research catalogue and
the evaluation protocol are deliberately independent objects.
"""

from __future__ import annotations

from dataclasses import dataclass


ORIGIN_SELECTION_HORIZONS_DAYS = (
    21, 42, 63, 126, 189, 252, 378, 504, 756, 1008, 1260, 1512, 1764,
    2016, 2268, 2520, 3780, 5040, 7560,
)
TEMPORAL_SPLITS = (
    ("train_first_quarter_test_remaining", 0.25),
    ("train_first_half_test_remaining", 0.50),
    ("train_first_three_quarters_test_final_quarter", 0.75),
)


@dataclass(frozen=True)
class DenseDailyProtocol:
    common_start: str = "1979-12-31"
    common_end: str = "2026-05-13"
    portfolio_count: int = 80
    rolling_origins_per_portfolio: int = 48
    temporal_origins_per_portfolio: int = 3
    simulations_per_origin: int = 240
    panel_seed: int = 20260528
    origin_policy: str = "full_history_even"
    scored_cells_per_model: int = 701280
    bootstrap_reps: int = 1000

    @property
    def total_origin_tasks(self) -> int:
        return self.portfolio_count * (
            self.rolling_origins_per_portfolio + self.temporal_origins_per_portfolio
        )

    def validate(self) -> None:
        if self.total_origin_tasks != 4080:
            raise ValueError("dense protocol must contain exactly 4,080 origin tasks")
        if self.simulations_per_origin != 240:
            raise ValueError("dense protocol simulation count drifted")
        if self.origin_policy != "full_history_even":
            raise ValueError("dense protocol origin selection policy drifted")
        if self.scored_cells_per_model != 701280:
            raise ValueError("dense protocol cell count drifted")


CANONICAL_DENSE_PROTOCOL = DenseDailyProtocol()
CANONICAL_DENSE_PROTOCOL.validate()
