import copy,hashlib
from unittest.mock import patch
import numpy as np
import pytest
from . import candidate as c
ELIGIBLE=[0,1,4,8,10,14,22,26,112,116,148,152]
FOLDS=[(20,2,3),(24,4,5)]
def fixture():return np.arange(80,dtype=float).reshape(40,2)/1000,np.array([.4,.6]),np.arange('2000-01-01','2000-02-10',dtype='datetime64[D]')
class Work:
    skip_unused_parent_calibration=True
    def __init__(self,fail=False):self.calls=[];self.fail=fail
    def validate_request(self,*args):pass
    def prefix_fits(self,x,ids,w):
        self.calls.append((len(x),tuple(ids)))
        return {'families':{}},({i:{'type':'fixture','reason':'unsupported'}for i in ids}if self.fail else{})
def chunks(x,w,h,paths,seed,valid,prepared,**kwargs):yield 0,np.broadcast_to((len(x)+np.asarray(valid,dtype=float))[:,None,None],(len(valid),2,h)).copy()
def coefficients(last,actual):
    K=len(last);return last[:,0].copy(),np.add.outer(last[:,0],last[:,0])/20
def solver(A,B):return {'weights':np.eye(1,len(A),int(np.argmin(A))).ravel(),'objective':float(np.min(A))}
def stage(H=21,work=None,seed=0):
    work=Work()if work is None else work
    with patch.object(c,'schedule',return_value=(ELIGIBLE,FOLDS,8)),patch.object(c,'all_chunks',side_effect=chunks),patch.object(c,'coefficients',side_effect=coefficients):
        return c.prepare_all(*fixture()[:2],H,fixture()[2],fit_cache=work,solver=solver,seed_replication=seed),work
def test_exact_prespecified_counts_and_only_filtered_zero_removed():
    assert len(c.ZERO_IDS)==54 and [len(v)for v in c.MENUS.values()]==[167,113,41,29]
    assert set(c.MENUS['full113'])==set(c.MENUS['full167'])-set(c.ZERO_IDS)
    assert set(c.MENUS['finite29'])==set(c.MENUS['finite41'])-set(c.ZERO_IDS)
    assert all(c.BANK[i]['mean_mode']=='zero'and c.BANK[i]['kind']=='filtered'for i in c.ZERO_IDS)
    assert all(i in c.MENUS['full113']for i in(1,2,3,166,167))
def test_submatrix_fits_not_posthoc_weight_filter_and_shared_two_cubes():
    seen=[]
    def track(A,B):seen.append((A.copy(),B.copy()));return solver(A,B)
    work=Work()
    with patch.object(c,'schedule',return_value=(ELIGIBLE,FOLDS,8)),patch.object(c,'all_chunks',side_effect=chunks)as draw,patch.object(c,'coefficients',side_effect=coefficients)as coef:
        result=c.prepare_all(*fixture()[:2],21,fixture()[2],fit_cache=work,solver=track)
    assert draw.call_count==coef.call_count==2 and len(seen)==8 and len(work.calls)==3
    full=[i for i in ELIGIBLE if i!=0]+[166,167]
    for mode,(A,B)in zip(c.MODES,seen):
        menu,_=c.split_mode(mode);valid=[i for i in full if i in c.MENUS[menu]]
        assert result[mode]['coefficients']['flat']['valid']==valid and len(A)==len(valid)
        assert all(set(t['eligible_this_fold'])<=set(c.MENUS[menu])for t in result[mode]['trace'])
        assert np.all(result[mode]['weights_by_mode']['flat'][[i for i in range(len(c.BANK))if i not in valid]]==0)
    assert set(work.calls[-1][1])=={int(i)for b in result.values()for i in np.flatnonzero(b['weights_by_mode']['flat']>0)}
def test_current_control_byte_exact_frozen_prepare_coefficients_weights_counts():
    result,_=stage();work=Work();x,w,d=fixture()
    with patch.object(c.parent,'schedule',return_value=(ELIGIBLE,FOLDS,8)),patch.object(c.parent,'all_chunks',side_effect=chunks),patch.object(c.parent,'compiled_coefficients',side_effect=coefficients):
        old=c.parent.prepare(x,w,21,d,inner_paths=2,schedule_rule='half_doubling',fit_cache=work,solver=solver,skip_unused_parent_calibration=True)
    new=result['full167__current']
    for key in('A','B'):assert old['coefficients']['flat'][key].tobytes()==new['coefficients']['flat'][key].tobytes()
    assert old['weights_by_mode']['flat'].tobytes()==new['weights_by_mode']['flat'].tobytes()
    assert c.parent.energy.even_allocation(old['weights_by_mode']['flat'],240).tobytes()==c.parent.energy.even_allocation(new['weights_by_mode']['flat'],240).tobytes()
    assert [t['seed']for t in old['trace']]==[t['seed']for t in new['trace']]
def test_reference_weights_mass_applied_once_to_both_A_B():
    result,_=stage();full=result['full167__current']['coefficients']['flat']['valid'];K=len(full)
    raw=[coefficients(np.broadcast_to((stop+np.asarray(full,dtype=float))[:,None],(K,2)).copy(),0)for stop,h,m in FOLDS];factors=[1/np.sqrt(1+h/stop)for stop,h,m in FOLDS];mass=sum(m*f for(_,h,m),f in zip(FOLDS,factors))
    A=sum(m*(f*(a/np.sqrt(h)))for(_,h,m),(a,b),f in zip(FOLDS,raw,factors))/mass
    B=sum(m*(f*(b/np.sqrt(h)))for(_,h,m),(a,b),f in zip(FOLDS,raw,factors))/mass
    got=result['full167__mean_estimation_reference']['coefficients']['flat']
    np.testing.assert_array_equal(got['A'],A);np.testing.assert_array_equal(got['B'],B);assert got['represented_mass']==mass
    assert c.reliability('mean_estimation_reference',24,6041)<c.reliability('mean_estimation_reference',6065,6041)
def test_H1_same_menus_dates_seeds_and_no_parent_mutation():
    bank=copy.deepcopy(c.parent.BANK);menus=copy.deepcopy(c.parent.MENUS);fn=c.parent.schedule
    result,work=stage(1,seed=3);x,w,d=fixture()
    assert c.parent.BANK==bank and c.parent.MENUS==menus and c.parent.schedule is fn
    for mode,b in result.items():
        for t,(stop,h,m)in zip(b['trace'],FOLDS):
            assert t['training_last_date']==str(d[stop-1])and t['validation_last_known_date']==str(d[stop+h-1])
            expected=int.from_bytes(hashlib.sha256(f'horizon_pool_inner_v1:{c.parent.signature(x[:stop])}:{h}:2:replication=3'.encode()).digest()[:4],'little');assert t['seed']==expected
        assert b['diagnostics']['eligible_menu_size']==len(c.MENUS[c.split_mode(mode)[0]])
def test_all_historical_invalid_fails_closed():
    with pytest.raises(ArithmeticError,match='no fallback'):stage(work=Work(fail=True))
def test_selected_current_failure_isolated_to_affected_mode():
    class FailedCurrent(Work):
        def prefix_fits(self,x,ids,w):
            full,errors=super().prefix_fits(x,ids,w)
            return full,({148:{'type':'fixture','reason':'current'}}if len(x)==40 else errors)
    def select(A,B):
        v=np.zeros(len(A));v[9 if len(A)==13 else 0]=1.;return {'weights':v}
    with patch.object(c,'schedule',return_value=(ELIGIBLE,FOLDS,8)),patch.object(c,'all_chunks',side_effect=chunks),patch.object(c,'coefficients',side_effect=coefficients):result=c.prepare_all(*fixture()[:2],21,fixture()[2],fit_cache=FailedCurrent(),solver=select)
    failed=result['full167__current'];good=result['finite41__current']
    assert list(failed['unsupported_selected_current'])==[148]and failed['weights_by_mode']['flat'][148]==1.
    assert not good['unsupported_selected_current']and not good['unsupported_solver']
    with pytest.raises(ArithmeticError,match='no substitution'):c.build_model('full167__current',fixture()[2],failed)
    c.build_model('finite41__current',fixture()[2],good)
def test_one_solver_failure_and_bad_simplex_preserve_other_modes():
    for first in ('exception','simplex'):
        calls=[]
        def isolated(A,B):
            calls.append(len(A))
            if len(calls)==1:
                if first=='exception':raise ArithmeticError('fixture nonconvergence')
                return {'weights':np.zeros(len(A))}
            return solver(A,B)
        with patch.object(c,'schedule',return_value=(ELIGIBLE,FOLDS,8)),patch.object(c,'all_chunks',side_effect=chunks),patch.object(c,'coefficients',side_effect=coefficients):result=c.prepare_all(*fixture()[:2],21,fixture()[2],fit_cache=Work(),solver=isolated)
        assert len(calls)==8 and result['full167__current']['unsupported_solver']
        assert result['full167__current']['weights_by_mode']['flat']is None
        with pytest.raises(ArithmeticError,match='no substitution'):c.build_model('full167__current',fixture()[2],result['full167__current'])
        assert sum(not b['unsupported_solver']for b in result.values())==7
        for mode,b in result.items():
            if not b['unsupported_solver']:c.build_model(mode,fixture()[2],b)
def test_finite41_current_byte_exact_frozen_finite_adapter():
    from candidates.asset_selector_no_recursive_ewma_v1 import candidate as old
    result,_=stage();x,w,d=fixture()
    with patch.object(old,'schedule',return_value=(ELIGIBLE,FOLDS,8)),patch.object(old,'all_chunks',side_effect=chunks),patch.object(old,'coefficients',side_effect=coefficients):reference=old.prepare(x,w,21,d,fit_cache=Work(),solver=solver,menu='finite_wealth41')
    # The synthetic coefficient fixture must index by recipe identity, like real path rows.
    got=result['finite41__current']
    for key in ('A','B'):assert reference['coefficients']['flat'][key].tobytes()==got['coefficients']['flat'][key].tobytes()
    assert reference['weights_by_mode']['flat'].tobytes()==got['weights_by_mode']['flat'].tobytes()
    assert c.parent.energy.even_allocation(reference['weights_by_mode']['flat'],240).tobytes()==c.parent.energy.even_allocation(got['weights_by_mode']['flat'],240).tobytes()
def test_model_identity_cross_mode_dependency_and_final_delegate():
    result,_=stage();assert len({c.identity(mode)for mode in c.MODES})==8
    mode='full113__current';b=result[mode]
    with pytest.raises(ValueError,match='source-bound'):c.build_model('finite29__current',fixture()[2],b)
    with patch.object(c,'sources',return_value={}):
        with pytest.raises(ValueError,match='source-bound'):c.build_model(mode,fixture()[2],b)
    model=c.build_model(mode,fixture()[2],b);expected=np.ones((240,2))
    def delegate(self,*args,**kwargs):self.last_selection={};yield 0,expected
    with patch.object(c.parent.MenuPool,'terminal_chunks',delegate):got=list(model.terminal_chunks(*fixture()[:2],21,240,5))
    assert got[0][1]is expected and model.last_selection['factorial_mode']==mode
