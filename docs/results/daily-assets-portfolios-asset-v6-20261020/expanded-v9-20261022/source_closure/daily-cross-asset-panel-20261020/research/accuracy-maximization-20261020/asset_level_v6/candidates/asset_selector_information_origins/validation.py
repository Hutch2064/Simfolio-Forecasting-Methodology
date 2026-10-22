"""Uncapped information-spaced origin representatives and exact coverage counts."""
import numpy as np
from candidates.asset_selector_universe_precision.candidate import schedule as original
RULES=('all','doubling','half_doubling')


def schedule(rows,requested_horizon,rule):
    if rule not in RULES:raise ValueError('Known information-spacing rule required')
    eligible,full=original(rows,requested_horizon)
    if rule=='all':return eligible,[(stop,h,1)for stop,h in full],len(full)
    factor=2. if rule=='doubling'else np.sqrt(2.)
    kept=[0]
    for i,(stop,h)in enumerate(full[1:],1):
        if stop>=factor*full[kept[-1]][0]:kept.append(i)
    if kept[-1]!=len(full)-1:kept.append(len(full)-1)
    # Every available original origin contributes one representation count.
    # Nearest representative in the equally spaced original-origin sequence;
    # exact midpoint ties belong to the earlier representative.
    boundaries=np.array(kept[:-1])+np.array(kept[1:])
    owners=np.searchsorted(boundaries,2*np.arange(len(full)),side='left')
    counts=np.bincount(owners,minlength=len(kept))
    assert counts.sum()==len(full)and np.all(counts>0)
    return eligible,[(full[i][0],full[i][1],int(count))for i,count in zip(kept,counts)],len(full)
