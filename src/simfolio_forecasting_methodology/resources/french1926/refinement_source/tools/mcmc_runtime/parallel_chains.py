"""Independent chain execution with unchanged seeds, order and budgets."""
import inspect
from concurrent.futures import ThreadPoolExecutor

import parameter_mcmc_tuned as pm


def make():
    def two(y,theta,seed):
        seeds=(seed,(seed+99173)%2**32)
        def one(s):return pm.chain(y,theta,1024,2048,s)
        if len(y)<8192:return one(seeds[0]),one(seeds[1])
        with ThreadPoolExecutor(max_workers=2) as pool:return tuple(pool.map(one,seeds))
    original=pm.parameter_fit
    source=inspect.getsource(original)
    old='''    first,accept=chain(y,theta,1024,2048,seed)
    second,accept_second=chain(y,theta,1024,2048,(seed+99173)%2**32)'''
    assert source.count(old)==1
    source=source.replace(old,'    (first,accept),(second,accept_second)=two(y,theta,seed)')
    namespace=dict(pm.__dict__);namespace['two']=two
    exec(compile(source,__file__,'exec'),namespace)  # noqa: S102 - compiles trusted repository source, never user input.
    return namespace['parameter_fit']
