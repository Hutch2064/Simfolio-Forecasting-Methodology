"""Unchanged fast MAP moment-SV predecessor, without any rough overlay."""
from simfolio_forecasting_methodology.models.asset_level.filtered_innovation_moment_sv import (
    FilteredInnovationFixedMeanMomentSV,
)
from runtime_setup import initialize as initialize_numeric

CANDIDATES = (FilteredInnovationFixedMeanMomentSV(),)
TIMINGS = {}
FIT_DIAGNOSTICS = {}


def initialize(cache_root=None, vine_threads=1, state_workers=1):
    initialize_numeric(True)


def clear_path_cache():
    pass
