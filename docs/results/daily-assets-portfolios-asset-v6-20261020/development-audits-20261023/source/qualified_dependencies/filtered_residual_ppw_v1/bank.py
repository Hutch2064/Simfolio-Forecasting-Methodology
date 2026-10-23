"""Only empirical-family residual-block variants; no new Gaussian/Student law."""
from candidates.asset_selector_menu_components.bank import BANK as CONTROL_BANK
from candidates.asset_feature_batch.fits import FAMILIES
BASE_IDS=[i for i,s in enumerate(CONTROL_BANK) if s['kind']=='filtered' and s['innovations']=='empirical']
ADDED=[{**CONTROL_BANK[i],'kind':'filtered','sampling':kind,'base_iid_index':i,'residual_block':True} for i in BASE_IDS for kind in ('stationary','circular')]
EGARCH_ADDED=[{'kind':'Gaussian_EGARCH','mean_mode':m,'mean_uncertainty':u} for m in ('sample','shrink','zero') for u in (False,True)]
BANK=tuple([*CONTROL_BANK,*ADDED,*EGARCH_ADDED])
NEW_IDS=tuple(range(170,278))
EGARCH_IDS=tuple(range(278,284))
assert len(BASE_IDS)==54 and len(NEW_IDS)==108 and len(CONTROL_BANK)==170
