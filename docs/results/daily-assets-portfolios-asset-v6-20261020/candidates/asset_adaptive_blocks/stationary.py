"""Training-only corrected PPW mean length and memory-bounded joint indices."""
from pathlib import Path
import inspect,hashlib
import arch
from arch.bootstrap import optimal_block_length
import numpy as np

def selector_identity():
    source=Path(inspect.getsourcefile(optimal_block_length))
    return {'library':'arch','version':arch.__version__,'selector':'arch.bootstrap.optimal_block_length stationary',
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}

def block_length(history):
    x=np.asarray(history,dtype=float)
    if x.ndim!=2 or len(x)<16 or not np.isfinite(x).all():
        raise ValueError('Finite joint training history required')
    centered=x-x.mean(axis=0)
    features=np.column_stack((centered,centered*centered))
    active=np.any(features!=features[0],axis=0)
    if not active.any():raise ValueError('Nonconstant training features required')
    values=optimal_block_length(features[:,active])['stationary'].to_numpy(dtype=float)
    if not np.isfinite(values).all():raise ArithmeticError('Nonfinite automatic stationary length')
    return max(1.,min(float(len(x)),float(values.max())))

class StationaryStream:
    def __init__(self,n,horizon,paths,context_seed,length):
        if min(n,horizon,paths)<1 or not np.isfinite(length) or length<1:
            raise ValueError('Positive dimensions and finite mean block length required')
        children=np.random.SeedSequence(int(context_seed)).spawn(3)
        self.current=np.random.default_rng(children[0]).integers(0,n,paths)
        self.restart=np.random.default_rng(children[1]);self.points=np.random.default_rng(children[2])
        self.n,self.horizon,self.paths,self.position=n,horizon,paths,0
        self.probability=1./length

    def take(self,width):
        if width<1 or self.position+width>self.horizon:raise ValueError('Columns outside forecast')
        flags=self.restart.random((width,self.paths))<self.probability
        points=self.points.integers(0,self.n,size=(width,self.paths))
        out=np.empty((self.paths,width),dtype=np.int64)
        for t in range(width):
            if self.position+t:self.current=np.where(flags[t],points[t],(self.current+1)%self.n)
            out[:,t]=self.current
        self.position+=width
        return out
