"""Request-local reuse of deterministic history-only fits across experiments."""
import time
import numpy as np
from candidates.asset_selector_menu_components.bank import prefix_fits as original_fit
from candidates.asset_selector_universe_precision.candidate import signature


class PreparedCache:
    def __init__(self,eligible,weights):
        self.eligible=tuple(eligible);self.weights=np.asarray(weights,dtype=float).copy();self.entries={};self.fit_cpu_seconds=0.;self.requests=0;self.actual_fits=0
    def get(self,history,requested,weights):
        if not np.array_equal(weights,self.weights):raise ValueError('Fitted portfolio weights changed')
        requested=tuple(requested)
        if not set(requested).issubset(self.eligible):raise ValueError('Request exceeds this fitted universe')
        self.requests+=1;key=signature(history)
        if key not in self.entries:
            started=time.process_time();prepared,invalid=original_fit(history,self.eligible,self.weights)
            self.fit_cpu_seconds+=time.process_time()-started;self.actual_fits+=1
            self.entries[key]=(prepared,invalid)
        prepared,invalid=self.entries[key]
        return prepared,{i:error for i,error in invalid.items()if i in requested}
    def numpy_bytes(self):
        seen=set()
        def size(value):
            if id(value)in seen:return 0
            seen.add(id(value))
            if isinstance(value,np.ndarray):return value.nbytes
            if isinstance(value,dict):return sum(size(v)for v in value.values())
            if isinstance(value,(list,tuple)):return sum(size(v)for v in value)
            return 0
        return size(self.entries)
