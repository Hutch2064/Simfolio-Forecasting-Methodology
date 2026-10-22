"""Frozen canonical dense daily OOS evaluation contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib import resources

from .origins import ORIGIN_SELECTION_HORIZONS_DAYS, TEMPORAL_SPLITS
from .panel import CANONICAL_PANEL_FINGERPRINT, CANONICAL_PANEL_ID

CANONICAL_PROTOCOL_ID = "canonical-whitepaper-dense-daily-v1"
CANONICAL_DATASET_ID = "simfolio-canonical-simulated-returns-2026-05-13"
CANONICAL_SOURCE_REVISION = "773bc1c325559e6bf57a567f1d8bf473a3427fbc"
CANONICAL_PROTOCOL_FINGERPRINT = "24bace549583831ccfa016ce62d3f604020fa7b2e0b36574cb697aa02baa4ba3"
CANONICAL_IDENTITY_RESOURCE_SHA256 = "742ad71ee40cc27a4a816d0ea161d742984d5c3449ad4567bcd971b196b1115e"


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
    selection_horizons_days: tuple[int, ...] = ORIGIN_SELECTION_HORIZONS_DAYS
    temporal_splits: tuple[tuple[str, float], ...] = TEMPORAL_SPLITS
    rolling_min_training_observations: int = 504
    rolling_horizon_rule: str = "daily_1_to_min_forward_and_floor_training_over_4"
    temporal_position_rule: str = "floor(n_times_train_fraction)-1_clipped_to_79_through_n-2"
    horizon_units: str = "trading_days"
    target: str = "terminal_log_return"
    score: str = "exact_empirical_crps"
    score_pairwise_denominator: str = "n_squared"
    aggregation: str = "mean_origins_within_portfolio_horizon_cells_then_equal_cells"
    failure_policy: str = "fail_closed_nonfinite_or_missing_cells"
    panel_id: str = CANONICAL_PANEL_ID
    panel_fingerprint: str = CANONICAL_PANEL_FINGERPRINT
    dataset_id: str = CANONICAL_DATASET_ID
    source_revision: str = CANONICAL_SOURCE_REVISION

    @property
    def total_origin_tasks(self) -> int:
        return self.portfolio_count * (
            self.rolling_origins_per_portfolio + self.temporal_origins_per_portfolio
        )

    def contract(self) -> dict[str, object]:
        """Return the JSON-safe immutable protocol identity payload."""
        return asdict(self)

    @property
    def protocol_fingerprint(self) -> str:
        payload = json.dumps(
            self.contract(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def validate(self) -> None:
        if self.total_origin_tasks != 4080:
            raise ValueError("dense protocol must contain exactly 4,080 origin tasks")
        if self.portfolio_count != 80:
            raise ValueError("dense protocol portfolio count drifted")
        if self.rolling_origins_per_portfolio != 48 or self.temporal_origins_per_portfolio != 3:
            raise ValueError("dense protocol origin counts drifted")
        if self.simulations_per_origin != 240:
            raise ValueError("dense protocol simulation count drifted")
        if self.panel_seed != 20260528:
            raise ValueError("dense protocol panel seed drifted")
        if self.origin_policy != "full_history_even":
            raise ValueError("dense protocol origin selection policy drifted")
        if self.scored_cells_per_model != 701280:
            raise ValueError("dense protocol cell count drifted")
        if self.selection_horizons_days != ORIGIN_SELECTION_HORIZONS_DAYS:
            raise ValueError("origin-selection horizon set drifted")
        if self.temporal_splits != TEMPORAL_SPLITS:
            raise ValueError("temporal holdout definitions drifted")
        if self.rolling_min_training_observations != 504:
            raise ValueError("rolling minimum training history drifted")
        if self.score_pairwise_denominator != "n_squared":
            raise ValueError("canonical CRPS pairwise denominator drifted")
        if self.failure_policy != "fail_closed_nonfinite_or_missing_cells":
            raise ValueError("canonical failure policy drifted")
        if self.panel_fingerprint != CANONICAL_PANEL_FINGERPRINT:
            raise ValueError("canonical panel identity drifted")
        if self.dataset_id != CANONICAL_DATASET_ID:
            raise ValueError("canonical dataset identity drifted")


def canonical_protocol_identity() -> dict[str, object]:
    """Load and verify the source-relative frozen protocol identity resource."""
    resource = resources.files("simfolio_forecasting_methodology").joinpath(
        "resources", "protocols", "canonical_protocol_identity.json"
    )
    with resource.open("rb") as handle:
        raw = handle.read()
    if hashlib.sha256(raw).hexdigest() != CANONICAL_IDENTITY_RESOURCE_SHA256:
        raise ValueError("canonical protocol identity resource hash drifted")
    identity = json.loads(raw.decode("utf-8"))
    if identity.get("protocol_id") != CANONICAL_PROTOCOL_ID:
        raise ValueError("canonical protocol identity id drifted")
    if identity.get("protocol_fingerprint") != CANONICAL_PROTOCOL_FINGERPRINT:
        raise ValueError("canonical protocol identity fingerprint drifted")
    def _json_shape(value: object) -> object:
        if isinstance(value, tuple):
            return [_json_shape(item) for item in value]
        if isinstance(value, list):
            return [_json_shape(item) for item in value]
        if isinstance(value, dict):
            return {key: _json_shape(item) for key, item in value.items()}
        return value

    if _json_shape(identity.get("contract")) != _json_shape(CANONICAL_DENSE_PROTOCOL.contract()):
        raise ValueError("canonical protocol contract differs from identity resource")
    return identity


CANONICAL_DENSE_PROTOCOL = DenseDailyProtocol()
CANONICAL_DENSE_PROTOCOL.validate()
