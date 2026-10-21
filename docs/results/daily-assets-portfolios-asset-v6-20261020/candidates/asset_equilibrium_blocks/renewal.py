"""Equilibrium first residual for otherwise unchanged uniform circular blocks."""
import numpy as np
from blocks import BlockLimits,JointBlockStream

def initial_residual(rng,upper):
    # Two distinct cutpoints among0..upper give P(R=r)=2*(upper-r+1)/(upper*(upper+1)).
    first=int(rng.integers(0,upper+1));second=int(rng.integers(0,upper))
    if second>=first:second+=1
    return abs(first-second)

class EquilibriumBlockStream(JointBlockStream):
    def __init__(self,n,horizon,paths,seed,*,replacement=True,limits=BlockLimits()):
        lower,upper=limits.resolve(n,horizon)
        if not replacement or paths<1 or lower!=1:
            raise ValueError('Equilibrium contrast requires replacement and unit minimum block length')
        self.n=n;self.descriptors=[]
        for child in np.random.SeedSequence(int(seed)).spawn(paths):
            rng=np.random.default_rng(child)
            starts,lengths,remaining=[],[],horizon
            first=True
            while remaining:
                length=initial_residual(rng,upper) if first else int(rng.integers(1,upper+1))
                first=False
                start=int(rng.integers(0,n));copied=min(length,remaining)
                starts.append(start);lengths.append(copied);remaining-=copied
            self.descriptors.append((np.asarray(starts),np.asarray(lengths)))
        self.pointer=np.zeros(paths,dtype=np.int64);self.within=np.zeros(paths,dtype=np.int64)
        self.position,self.horizon=0,horizon

    def take(self,width):
        return super().take(width)%self.n
