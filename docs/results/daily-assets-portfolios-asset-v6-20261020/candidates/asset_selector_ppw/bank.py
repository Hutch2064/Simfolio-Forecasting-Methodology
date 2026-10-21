"""Nested168-option menu; the original166options remain source-identical."""
from candidates.asset_selector_universe import bank as base
from candidates.asset_selector_universe_precision.candidate import schedule as base_schedule
from .blocks import fit_lengths,PPWJointBootstrap

BANK=[*base.BANK,{'kind':'PPW_blocks','sampling':'stationary','replacement':True},
      {'kind':'PPW_blocks','sampling':'circular','replacement':True}]
BASE_COMPONENT_COUNT=len(base.BANK)

def schedule(n,horizon):
    eligible,folds=base_schedule(n,horizon)
    return [*eligible,len(base.BANK),len(base.BANK)+1],folds

def prefix_fits(history,eligible,weights):
    original=[i for i in eligible if i<len(base.BANK)];prepared,invalid=base.prefix_fits(history,original)
    added=[i for i in eligible if i>=len(base.BANK)]
    if added:
        try:prepared['PPW']=fit_lengths(history,weights)
        except Exception as e:
            for i in added:invalid[i]={'type':type(e).__name__,'reason':str(e)}
    return prepared,invalid

def model(index,prepared):
    if index<len(base.BANK):return base.model(index,prepared)
    return PPWJointBootstrap(BANK[index]['sampling'],prepared['PPW'])
