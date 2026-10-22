"""Same integer RNG calls, partitions and row streams; fewer allocations."""
import numpy as np
from blocks import BlockLimits,JointBlockStream as OriginalStream

def partition(n,lower,upper,rng):
    if lower==upper:
        starts=np.arange(0,n,lower,dtype=np.int64)
        return starts,np.minimum(lower,n-starts)
    batches=[];consumed=0
    while consumed<n:
        count=max(8,int(2*(n-consumed)/((lower+upper)/2))+1)
        draws=rng.integers(lower,upper+1,count)
        cumulative=np.cumsum(draws)
        take=min(count,int(np.searchsorted(cumulative,n-consumed))+1)
        batch=draws[:take].copy()
        if cumulative[take-1]>=n-consumed:
            batch[-1]=n-consumed-int(cumulative[take-2] if take>1 else 0)
        batches.append(batch);consumed+=int(batch.sum())
    lengths=np.concatenate(batches)
    starts=np.concatenate(([0],np.cumsum(lengths[:-1])))
    return starts,lengths

class JointBlockStream(OriginalStream):
    def __init__(self,n,horizon,paths,seed,*,replacement,limits=BlockLimits()):
        if paths<1:raise ValueError('Positive simulation count required')
        lower,upper=limits.resolve(n,horizon)
        self.descriptors=[]
        # This branch consumes NO randomness in the original partition either.
        fixed=partition(n,lower,upper,None) if not replacement and lower==upper else None
        for child in np.random.SeedSequence(int(seed)).spawn(paths):
            rng=np.random.default_rng(child)
            starts,lengths,remaining=[],[],horizon
            while remaining:
                if replacement:
                    length=int(rng.integers(lower,upper+1))
                    start=int(rng.integers(0,n-length+1))
                    copied=min(length,remaining)
                    starts.append(start);lengths.append(copied);remaining-=copied
                else:
                    ps,pl=fixed if fixed is not None else partition(n,lower,upper,rng)
                    for index in rng.permutation(len(ps)):
                        copied=min(int(pl[index]),remaining)
                        starts.append(int(ps[index]));lengths.append(copied);remaining-=copied
                        if not remaining:break
            self.descriptors.append((np.asarray(starts),np.asarray(lengths)))
        self.pointer=np.zeros(paths,dtype=np.int64)
        self.within=np.zeros(paths,dtype=np.int64)
        self.position,self.horizon=0,horizon
