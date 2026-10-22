"""Reuse the qualified group cache; accelerate only exact pure-state fitting."""
from candidates.asset_selector_endpoint_group_reuse_v1.fits_cache import GroupWork as ReferenceWork,menu
from . import fast_fits
from collections import Counter

class GroupWork(ReferenceWork):
    def __init__(self,*args,skip_unused_parent_calibration=False,**kwargs):
        super().__init__(*args,**kwargs);self.skip_unused_parent_calibration=bool(skip_unused_parent_calibration);self.family_alias_hits=Counter()

    def prefix_fits(self,history,eligible,weights):
        if self.skip_unused_parent_calibration and 0 in eligible:raise ValueError('Leg-only preparation supports parent legs166/167, not the learned parent mixture0')
        return super().prefix_fits(history,eligible,weights)

    def diagnostics(self):
        return {**super().diagnostics(),'skip_unused_parent_calibration':self.skip_unused_parent_calibration,'legacy_unused_calibration_failure_guard_removed':self.skip_unused_parent_calibration,'universal_old_invalid_mask_parity_claim':False,'family_alias_hits':dict(self.family_alias_hits)}
    def _base_prefix(self,history,eligible):
        # Same call order, exceptions, eligible-family and parent conditions as
        # asset_selector_universe.bank.prefix_fits; only pure-state reuse differs.
        base_bank=menu.base;n=len(history);prepared={'families':{}};invalid={}
        if all(base_bank.BANK[i]['kind']=='blocks' for i in eligible):return prepared,invalid
        try:base=self._get((n,'base',''),lambda:base_bank.fits.base_fit(history))
        except Exception as e:return prepared,{i:{'type':type(e).__name__,'reason':str(e)} for i in eligible if base_bank.BANK[i]['kind']!='blocks'}
        needed={base_bank.BANK[i]['family'] for i in eligible if base_bank.BANK[i]['kind']=='filtered'}|({'SV_RMS'} if 0 in eligible else set())
        family_states={};prefix_work={}
        def fitted_family(family):
            canonical={'EWMA_recursive':'EWMA_reverting','SV_path':'SV_RMS'}.get(family,family)
            if canonical in family_states:self.family_alias_hits[family]+=1
            else:family_states[canonical]=self._get((n,'family',canonical),lambda:fast_fits.pure_prepare(history,canonical,base,prefix_work=prefix_work))
            state=family_states[canonical]
            return state if family==canonical else {**state,'family':family}
        for family in sorted(needed):
            try:prepared['families'][family]=fitted_family(family)
            except Exception as e:
                for i in eligible:
                    if (base_bank.BANK[i]['kind']=='filtered' and base_bank.BANK[i]['family']==family) or (i==0 and family=='SV_RMS'):invalid[i]={'type':type(e).__name__,'reason':str(e)}
        if 0 in eligible and 0 not in invalid:
            parent_prepare=fast_fits.leg_parent_prepare if self.skip_unused_parent_calibration else base_bank.parent_prepare
            try:prepared['parent']=self._get((n,'parent',''),lambda:parent_prepare(history,'SV_RMS','EWMA',base,prepared['families']['SV_RMS'],**({'prefix_work':prefix_work} if self.skip_unused_parent_calibration else {})))
            except Exception as e:invalid[0]={'type':type(e).__name__,'reason':str(e)}
        return prepared,invalid
