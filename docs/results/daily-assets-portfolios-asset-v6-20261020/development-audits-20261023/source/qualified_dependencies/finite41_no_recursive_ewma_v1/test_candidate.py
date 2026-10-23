import copy,hashlib,concurrent.futures
from unittest.mock import patch
import numpy as np
import pytest
from . import candidate as c
class Work:
    skip_unused_parent_calibration=True
    def __init__(self,fail=False):self.calls=[];self.fail=fail
    def validate_request(self,*args):pass
    def prefix_fits(self,x,eligible,w):
        assert not set(eligible)&set(c.EXCLUDED);self.calls.append(tuple(eligible))
        return {'families':{}},({i:{'type':'fixture_failure','reason':'unsupported'}for i in eligible}if self.fail else{})
def fixture():return np.zeros((30,1)),np.ones(1),np.arange('2000-01-01','2000-01-31',dtype='datetime64[D]')
def stage(H,work=None,seed=0,menu='flat'):
    x,w,d=fixture();work=Work()if work is None else work;eligible=[0,1,4,148,150,162,165]
    def chunks(x,w,h,paths,context,valid,prepared,**kwargs):
        assert not set(valid)&set(c.EXCLUDED);yield 0,np.ones((len(valid),2,h))
    def coefficients(last,actual):return np.ones(len(last)),np.zeros((len(last),len(last)))
    def solver(A,B):
        assert len(A)==4;v=np.zeros(len(A));v[-1]=1.;return {'weights':v,'objective':1.}
    with patch.object(c,'schedule',return_value=(eligible,[(24,2,1)],1)),patch.object(c,'all_chunks',side_effect=chunks),patch.object(c,'coefficients',side_effect=coefficients):
        return c.prepare(x,w,H,d,fit_cache=work,solver=solver,seed_replication=seed,menu=menu),work
def test_exact149_universe_H1_and_normal_no_forbidden_stage_or_QP():
    assert c.EXCLUDED==tuple(range(148,166))and len(c.ELIGIBLE)==149
    for H in(1,21):
        b,work=stage(H);assert all(not set(ids)&set(c.EXCLUDED)for ids in work.calls);assert b['coefficients']['flat']['valid']==[1,4,166,167];assert np.all(b['weights_by_mode']['flat'][list(c.EXCLUDED)]==0)
        model=c.MenuPool(c.CONFIG,fixture()[2],b);assert model.model_id!=c.parent.identity(c.CONFIG)
def test_parent_state_is_untouched_and_concurrent_prototypes_isolated():
    bank=copy.deepcopy(c.parent.BANK);menus=copy.deepcopy(c.parent.MENUS);schedule=c.parent.schedule
    def chunks(x,w,h,paths,seed,valid,prepared,**kwargs):yield 0,np.ones((len(valid),2,h))
    def fitted(n):
        x,w,d=fixture();work=Work();b=c.prepare(x,w,21,d,fit_cache=work,solver=lambda A,B:{'weights':np.eye(1,len(A),0).ravel()});return work,b
    with patch.object(c,'schedule',return_value=([0,1,4,150],[(24,2,1)],1)),patch.object(c,'all_chunks',side_effect=chunks),patch.object(c,'coefficients',side_effect=lambda last,actual:(np.ones(len(last)),np.zeros((len(last),len(last))))):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:results=list(pool.map(fitted,range(2)))
    assert results[0][0]is not results[1][0]and results[0][1]['weights_by_mode']['flat']is not results[1][1]['weights_by_mode']['flat']
    assert c.parent.BANK==bank and c.parent.MENUS==menus and c.parent.schedule is schedule
def test_all_retained_fail_closed_without_QP_fallback():
    with pytest.raises(ArithmeticError,match='No valid retained'):stage(1,Work(fail=True))
def test_seed_formula_unchanged_and_source_bound_bundle_required():
    b,_=stage(21,seed=3);x,w,d=fixture();past=x[:24];expected=int.from_bytes(hashlib.sha256(f'horizon_pool_inner_v1:{c.parent.signature(past)}:2:2:replication=3'.encode()).digest()[:4],'little');assert b['trace'][0]['seed']==expected
    with pytest.raises(ValueError,match='source-bound'):c.MenuPool(c.CONFIG,d)
    other={**b,'implementation_sources':{}}
    with pytest.raises(ValueError,match='source-bound'):c.MenuPool(c.CONFIG,d,other)
def test_forbidden_final_weight_fails_not_replaced():
    b,_=stage(21);b['weights_by_mode']['flat'][150]=.1;model=c.MenuPool(c.CONFIG,fixture()[2],b)
    with pytest.raises(ValueError,match='no final substitution'):next(model.terminal_chunks(*fixture()[:2],21,240,1))
def test_finite41_semantic_rule_all_stages_and_cross_menu_rejection():
    from . import finite_wealth as f
    expected=tuple([1,2,3]+list(range(4,16))+list(range(22,34))+list(range(112,124))+[166,167]);assert c.FINITE_ELIGIBLE==expected
    for H in(1,21):
        b,work=stage(H,menu='finite_wealth41');assert all(set(ids)<=set(expected)for ids in work.calls)
        assert c.identity()!=f.identity()and len(f.ELIGIBLE)==41
        f.MenuPool(f.CONFIG,fixture()[2],b)
        with pytest.raises(ValueError,match='source-bound'):c.MenuPool(c.CONFIG,fixture()[2],b)
    with pytest.raises(ValueError,match='no override'):f.prepare(*fixture()[:2],1,fixture()[2],menu='flat')
def test_transitive_source_manifest_and_dependency_drift_namespace():
    before=c.identity();manifest=c.sources()
    for suffix in('/asset_feature_batch/kernels.py','/asset_feature_batch/fits.py','/asset_selector_batched_paths/kernels.py','/asset_selector_menu_components/bank.py','/asset_prior_best/fit.py'):
        assert any(name.endswith(suffix)for name in manifest)
    assert all(not name.startswith('/')for name in manifest)
    with patch.object(c,'sources',return_value={**manifest,'changed_dependency.py':'0'*64}):assert c.identity()!=before
def test_final_generator_delegation_is_retained_parent_kernel():
    b,_=stage(21);model=c.MenuPool(c.CONFIG,fixture()[2],b);expected=np.ones((240,2))
    def original(self,*args,**kwargs):self.last_selection={'parent_delegate':True};yield 0,expected
    with patch.object(c.parent.MenuPool,'terminal_chunks',original):
        got=list(model.terminal_chunks(*fixture()[:2],21,240,1))
    assert got[0][1]is expected and model.last_selection['research_config']==c.CONFIG
def test_bad_simplex_and_failed_current_fit_are_not_substituted():
    x,w,d=fixture()
    def chunks(x,w,h,paths,seed,valid,prepared,**kwargs):yield 0,np.ones((len(valid),2,h))
    class CurrentFailure(Work):
        def prefix_fits(self,x,eligible,w):
            full,failed=super().prefix_fits(x,eligible,w)
            return full,({eligible[0]:{'type':'current_failure','reason':'fixture'}}if len(x)==30 else failed)
    with patch.object(c,'schedule',return_value=([1,4],[(24,2,1)],1)),patch.object(c,'all_chunks',side_effect=chunks),patch.object(c,'coefficients',side_effect=lambda last,actual:(np.ones(len(last)),np.zeros((len(last),len(last))))):
        with pytest.raises(ArithmeticError,match='simplex'):c.prepare(x,w,21,d,fit_cache=Work(),solver=lambda A,B:{'weights':np.array([-.1,1.1])})
        with pytest.raises(ArithmeticError,match='no substitution'):c.prepare(x,w,21,d,fit_cache=CurrentFailure(),solver=lambda A,B:{'weights':np.array([1.,0.])})
