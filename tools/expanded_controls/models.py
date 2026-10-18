"""Select one unchanged canonical registry model for expanded-panel execution."""

import os

from simfolio_forecasting_methodology.models.registry import build_model


MODEL_IDS = (
    "naive_iid_historical_portfolio_bootstrap",
    "constant_mean_gaussian",
    "bayesian_sbb_overlay_ewma_absolute",
    "factor_ff6_ridge_residual_sbb_none",
)

selected = os.environ.get("SIMFOLIO_EXPANDED_CONTROL")
if selected not in MODEL_IDS:
    raise ValueError(f"SIMFOLIO_EXPANDED_CONTROL must select one of {MODEL_IDS!r}")

# Return the actual catalogue class: no forecast, seed, calendar, or terminal
# ensemble adapter can alter the registered implementation's law.
CANDIDATES = (build_model(selected),)
TIMINGS = {}
FIT_DIAGNOSTICS = {}


def initialize(cache_root=None, vine_threads=1, state_workers=1):
    """Accept the panel interface; these models own no persistent adapter cache."""


def clear_path_cache():
    """These exact registry implementations expose no adapter path cache."""
