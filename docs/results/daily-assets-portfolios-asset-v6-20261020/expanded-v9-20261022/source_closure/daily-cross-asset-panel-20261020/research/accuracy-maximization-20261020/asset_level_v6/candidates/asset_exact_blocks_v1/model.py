"""Existing asset-level baseline, with exact faster block descriptors."""
import numpy as np
from asset_models import Reference as OriginalReference
from asset_paths import advance
from .blocks import JointBlockStream

class Reference(OriginalReference):
    def terminal_chunks(self,history,weights,horizon,paths,seed,*,width=64,rebalance_every=1):
        x=np.asarray(history,dtype=np.float64);w=np.asarray(weights,dtype=np.float64)
        if (x.ndim!=2 or len(x)<2 or x.shape[1]!=len(w) or not np.isfinite(x).all() or not np.isfinite(w).all()
                or np.any(w<0) or not np.isclose(w.sum(),1.,rtol=0,atol=1e-14) or rebalance_every<0 or width<1):
            raise ValueError('Finite aligned assets, long-only unit weights and valid policy required')
        stream=JointBlockStream(len(x),horizon,paths,seed,replacement=self.replacement,limits=self.limits)
        target=np.full(len(w),-np.inf);target[w>0]=np.log(w[w>0])
        log_weights=np.tile(target,(paths,1));asset=np.zeros((paths,len(w)));portfolio=np.zeros(paths)
        for start in range(0,horizon,width):
            count=min(width,horizon-start)
            terminal=advance(x,stream.take(count),target,log_weights,asset,portfolio,start,rebalance_every)
            if not np.isfinite(terminal).all():raise ValueError('Nonfinite asset-level portfolio forecast')
            self.last_asset_log_paths=asset.copy();self.last_asset_weights=np.exp(log_weights)
            yield start,terminal

def build_replacement():return Reference(True)
def build_without_replacement():return Reference(False)
