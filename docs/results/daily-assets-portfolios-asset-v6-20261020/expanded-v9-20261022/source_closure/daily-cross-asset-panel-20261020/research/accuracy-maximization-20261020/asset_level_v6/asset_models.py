"""Source-bound asset-level reference factories; no portfolio-series proxy."""
from pathlib import Path
import json
import hashlib
from asset_paths import JointAssetBootstrap
from blocks import BlockLimits

OWN = Path(__file__).resolve().parent


class Reference(JointAssetBootstrap):
    def __init__(self, replacement):
        self.profile = json.loads((OWN/'profile.json').read_text())
        super().__init__(replacement=replacement, limits=BlockLimits(
            self.profile['minimum_block_observations'], self.profile['maximum_block_observations']))

    @property
    def dependency_identity(self):
        return {'method': self.model_id, 'forecast_level': 'joint_assets',
                'paths_per_forecast': 1000, 'replacement': self.replacement,
                'portfolio_accounting': 'inside each path, after shared asset row selection',
                'profile': self.profile,
                'source_hashes': {name: hashlib.sha256((OWN/name).read_bytes()).hexdigest()
                    for name in ['asset_models.py', 'asset_paths.py', 'blocks.py', 'asset_forecast.py', 'inputs.py']}}


def build_replacement():
    return Reference(True)


def build_without_replacement():
    return Reference(False)
