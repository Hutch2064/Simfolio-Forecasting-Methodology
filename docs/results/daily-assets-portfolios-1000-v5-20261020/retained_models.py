"""Explicit1,000-path numerical execution versions of the retained laws."""
from pathlib import Path
import hashlib
import streaming
import bounded_public
import bounded_public_without_replacement
from simfolio_forecasting_methodology.french_panel.control_adapters import TestingNaiveControl


class RetainedModel:
    def __init__(self, model_id):
        self.model_id = model_id

    @property
    def dependency_identity(self):
        parents = {streaming.ADAPTIVE: TestingNaiveControl,
            streaming.REPLACEMENT: bounded_public.build_bounded_public,
            streaming.CYCLES: bounded_public_without_replacement.build_bounded_cycle_public,
            streaming.BEST: streaming.incumbent.build_candidate}
        return {'method': self.model_id, 'simulation_execution': streaming.source_identity(),
            'original_financial_law': parents[self.model_id]().dependency_identity,
            'financial_law_change': False, 'new_numerical_sampling_version': True,
            'forecast_source_sha256': hashlib.sha256((Path(__file__).parent/'forecast.py').read_bytes()).hexdigest(),
            'factory_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def iter_terminal_log_returns(self, training, context):
        training.validate()
        return streaming.terminal_chunks(self.model_id, training.portfolio_log_returns,
            context.horizon_days, context.simulations, context.seed)


def build_adaptive():
    return RetainedModel(streaming.ADAPTIVE)


def build_replacement():
    return RetainedModel(streaming.REPLACEMENT)


def build_cycles():
    return RetainedModel(streaming.CYCLES)


def build_best():
    return RetainedModel(streaming.BEST)
