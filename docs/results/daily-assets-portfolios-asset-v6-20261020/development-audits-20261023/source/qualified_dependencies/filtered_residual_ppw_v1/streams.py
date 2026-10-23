"""Uniform-marginal common residual rows, exactly aligned with empirical IID."""
import numpy as np
from candidates.asset_fast_selector.bank import seed

class RowStream:
    def __init__(self,n,horizon,paths,context_seed,kind,length):
        if min(n,horizon,paths)<1 or kind not in ('stationary','circular') or not np.isfinite(length) or not 1<=length<=n:raise ValueError('Finite feasible residual block law required')
        self.n,self.horizon,self.paths,self.kind=n,horizon,paths,kind
        self.length=float(length) if kind=='stationary' else int(np.floor(length+.5));self.position=0;self.current=None
        self.points=np.random.default_rng(seed('joint_FHS_asset_rows_v1',context_seed))
        self.restart=np.random.default_rng(seed('joint_filtered_residual_PPW_restart_v1',context_seed))
    def take(self,width):
        if width<1 or self.position+width>self.horizon:raise ValueError('Rows outside requested horizon')
        candidates=self.points.integers(0,self.n,size=(width,self.paths));out=np.empty((self.paths,width),dtype=np.int64)
        for t in range(width):
            position=self.position+t
            if position==0 or self.length==1:self.current=candidates[t].copy()
            elif self.kind=='circular':self.current=candidates[t].copy() if position%self.length==0 else (self.current+1)%self.n
            else:self.current=np.where(self.restart.random(self.paths)<1./self.length,candidates[t],(self.current+1)%self.n)
            out[:,t]=self.current
        self.position+=width;return out
