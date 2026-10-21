"""Stateless native factories: bind one historical calendar prefix per request."""
from pathlib import Path
import hashlib,json
import numpy as np
from candidates.asset_selector_coefficient_sweep.candidate import PathCriterionPool,identity
from candidates.asset_selector_online_pool.candidate import OnlinePool,online_identity
from candidates.asset_selector_universe.bank import BANK

class NativeSelector:
    def __init__(self,rule):
        if rule not in('path','terminal','online'):raise ValueError('Unknown frozen selector rule')
        self.rule=rule;self.config={'inner_paths':1000,'weight_rule':'SOLO_FTRL'}if rule=='online'else{'criterion':rule,'inner_paths':1000}
        self.model_id=online_identity(self.config)if rule=='online'else identity(self.config)
    @property
    def dependency_identity(self):
        root=Path(__file__).resolve().parents[2]
        folders=('asset_selector_native_adapter','asset_selector_coefficient_sweep','asset_selector_path_objective','asset_selector_online_pool')
        manifest=Path(__file__).parent/'numerical_sources.json';declared=json.loads(manifest.read_text())['source_files'];sources={}
        for relative,expected in declared.items():
            p=(root/relative).resolve()
            if root not in p.parents or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('Frozen selector parent source changed: '+relative)
            sources[relative]=expected
        files=[p for folder in folders for p in(root/'candidates'/folder).glob('*.py')]+[manifest]
        sources.update({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(files)})
        return {'method':self.model_id,'forecast_level':'joint_assets','config':self.config,'bank':BANK,'paths_per_forecast':1000,'training_context':'Only dates[:origin_position+1]; no target dates, future returns or prior benchmark scores','request_binding':'New independent dated model for every history/horizon; no persistent first-request choice','source_hashes':sources}
    def with_training_dates(self,dates):
        past=np.array(dates,dtype='datetime64[D]',copy=True)
        if past.ndim!=1 or len(past)<2 or np.isnat(past).any()or np.any(np.diff(past)<=np.timedelta64(0,'D')):raise ValueError('Strictly increasing historical calendar prefix required')
        past.flags.writeable=False
        return OnlinePool(self.config,past)if self.rule=='online'else PathCriterionPool(self.config,past)
    def terminal_chunks(self,*args,**kwargs):
        raise ValueError('Dated selector requires explicit past-calendar binding; no portfolio proxy or fallback')

def build_path():return NativeSelector('path')
def build_endpoint():return NativeSelector('terminal')
def build_online():return NativeSelector('online')
