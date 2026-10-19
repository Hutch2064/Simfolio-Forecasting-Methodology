"""Chronological standardized innovations from the unchanged asset fitter."""
import numpy as np


def aligned_pools(assets,refit):
    assets=np.asarray(assets,dtype=np.float64)
    columns=[]
    for x in assets.T:
        original,_=refit(x.tobytes())
        pool=np.asarray(original['innovation_pool'],dtype=np.float64)
        if pool.shape!=(len(assets),) or not np.isfinite(pool).all():
            raise ArithmeticError('copula requires full chronological aligned asset innovations')
        columns.append(pool)
    return np.column_stack(columns)
