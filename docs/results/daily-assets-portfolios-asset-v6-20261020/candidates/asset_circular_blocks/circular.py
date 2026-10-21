"""Uniform circular block starts: equal historical-row exposure without fitting."""
import numpy as np
from blocks import BlockLimits, JointBlockStream

class CircularBlockStream(JointBlockStream):
    def __init__(self,n,horizon,paths,seed,*,replacement=True,limits=BlockLimits()):
        if not replacement or paths<1:
            raise ValueError('This contrast uses circular blocks with replacement')
        lower,upper=limits.resolve(n,horizon)
        self.n=n;self.descriptors=[]
        for child in np.random.SeedSequence(int(seed)).spawn(paths):
            rng=np.random.default_rng(child)
            starts,lengths,remaining=[],[],horizon
            while remaining:
                length=int(rng.integers(lower,upper+1))
                start=int(rng.integers(0,n))
                copied=min(length,remaining)
                starts.append(start);lengths.append(copied);remaining-=copied
            self.descriptors.append((np.asarray(starts),np.asarray(lengths)))
        self.pointer=np.zeros(paths,dtype=np.int64)
        self.within=np.zeros(paths,dtype=np.int64)
        self.position,self.horizon=0,horizon

    def take(self,width):
        return super().take(width)%self.n
