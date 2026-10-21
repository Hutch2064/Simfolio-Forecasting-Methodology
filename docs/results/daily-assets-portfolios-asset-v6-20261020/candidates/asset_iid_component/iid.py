"""Memory-bounded joint IID rows; same random sequence across horizon chunks."""
import numpy as np

class JointIIDStream:
    def __init__(self,n,horizon,paths,context_seed,*,replacement=True,limits=None):
        if not replacement or min(n,horizon,paths)<1:
            raise ValueError('Positive IID resampling dimensions with replacement required')
        self.n,self.horizon,self.paths=n,horizon,paths
        self.rng=np.random.default_rng(context_seed)
        self.position=0

    def take(self,width):
        if width<1 or self.position+width>self.horizon:
            raise ValueError('Requested IID columns outside forecast')
        self.position+=width
        return self.rng.integers(0,self.n,size=(width,self.paths)).T.copy()
