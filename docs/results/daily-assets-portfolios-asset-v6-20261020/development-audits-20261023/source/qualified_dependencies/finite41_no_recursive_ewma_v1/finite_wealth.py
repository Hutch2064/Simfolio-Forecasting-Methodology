"""Explicit second fixed contrast; no new kernels or innovations."""
from . import candidate
CONFIG=dict(candidate.FINITE_CONFIG);ELIGIBLE=candidate.FINITE_ELIGIBLE;GroupWork=candidate.GroupWork;MenuPool=candidate.MenuPool
def identity(config=CONFIG):return candidate.identity(config)
def prepare(history,weights,horizon,training_dates,**kwargs):
    if'menu'in kwargs:raise ValueError('Finite41 module fixes its menu; no override')
    return candidate.prepare(history,weights,horizon,training_dates,menu='finite_wealth41',**kwargs)
