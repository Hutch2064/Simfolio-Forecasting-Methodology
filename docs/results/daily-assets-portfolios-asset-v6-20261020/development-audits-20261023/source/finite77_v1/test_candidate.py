import copy
from unittest.mock import patch
import numpy as np
import pytest
from . import candidate as c
ELIGIBLE=[0,1,4,8,10,14,22,26,28,32,112,116,118,122,148,152]
FOLDS=[(20,2,3),(24,4,5)]
def fixture():return np.arange(80,dtype=float).reshape(40,2)/1000,np.array([.4,.6]),np.arange('2000-01-01','2000-02-10',dtype='datetime64[D]')
class Work:
    skip_unused_parent_calibration=True
    def __init__(self,length=2):self.calls=[];self.length=length
    def validate_request(self,*args):pass
    def prefix_fits(self,x,ids,w):
        assert set(ids)<=set(c.MENUS['finite77']);self.calls.append((len(x),tuple(ids)))
        deps={family:{'lengths':{'stationary':self.length,'circular':self.length}}for family in('constant','EWMA_reverting','SV_RMS')}
        return {'families':{},'residual_PPW':deps},{}
def chunks(x,w,h,paths,seed,valid,prepared,**kwargs):
    assert set(valid)<=set(c.MENUS['finite77']);yield 0,np.broadcast_to((len(x)+np.asarray(valid,dtype=float))[:,None,None],(len(valid),2,h)).copy()
def coefficients(last,actual):return last[:,0].copy(),np.add.outer(last[:,0],last[:,0])/20
def solver(A,B):return {'weights':np.eye(1,len(A),0).ravel(),'objective':float(A[0])}
def stage(H=21,work=None,solve=solver):
    work=Work()if work is None else work
    with patch.object(c,'schedule',return_value=(ELIGIBLE,FOLDS,8)),patch.object(c,'all_chunks',side_effect=chunks),patch.object(c,'coefficients',side_effect=coefficients):result=c.prepare_all(*fixture()[:2],H,fixture()[2],fit_cache=work,solver=solve)
    return result,work
def test_exact_finite77_semantics_and_existing_kernel_binding():
    assert len(c.CORE)==41 and len(c.DERIVATIVES)==36 and len(c.MENUS['finite77'])==77
    assert c.DERIVATIVES==tuple(list(range(170,194))+list(range(242,254)))
    for i in c.DERIVATIVES:
        s=c.BANK[i];assert s['family']in('constant','EWMA_reverting','SV_RMS')and s['innovations']=='empirical'and s['base_iid_index']in c.CORE
    assert c.all_chunks is c.ppw.all_chunks and c.GroupWork is c.ppw.GroupWork
    assert c.Pool.terminal_chunks is c.ppw.Pool.terminal_chunks
    assert not set(range(148,170))&set(c.DERIVATIVES)and not set(range(278,284))&set(c.MENUS['finite77'])
def test_H1_derivatives_never_reach_prefix_cube_or_QP_and_control_identical():
    seen=[]
    def track(A,B):seen.append(len(A));return solver(A,B)
    result,work=stage(1,solve=track)
    assert len(seen)==1 and all(not set(ids)&set(c.DERIVATIVES)for n,ids in work.calls)
    for key in('A','B'):assert result['finite41']['coefficients']['flat'][key].tobytes()==result['finite77']['coefficients']['flat'][key].tobytes()
    assert result['finite41']['weights_by_mode']['flat'].tobytes()==result['finite77']['weights_by_mode']['flat'].tobytes()
    assert result['finite77']['diagnostics']['H1_PPW_excluded_before_QP']
def test_finite41_frozen_control_bytes_same_shared_bank():
    from candidates.asset_selector_no_recursive_ewma_v1 import candidate as old
    result,_=stage();x,w,d=fixture()
    with patch.object(old,'schedule',return_value=(ELIGIBLE,FOLDS,8)),patch.object(old,'all_chunks',side_effect=chunks),patch.object(old,'coefficients',side_effect=coefficients):reference=old.prepare(x,w,21,d,fit_cache=Work(),solver=solver,menu='finite_wealth41')
    b=result['finite41']
    for key in('A','B'):assert b['coefficients']['flat'][key].tobytes()==reference['coefficients']['flat'][key].tobytes()
    assert b['weights_by_mode']['flat'].tobytes()==reference['weights_by_mode']['flat'].tobytes()
    assert c.energy.even_allocation(b['weights_by_mode']['flat'],240).tobytes()==c.energy.even_allocation(reference['weights_by_mode']['flat'],240).tobytes()
def test_restrictions_before_each_fold_submatrix_and_no_parent_mutation():
    bank=copy.deepcopy(c.ppw.BANK);fn=c.ppw.all_chunks;calls=[]
    def track(A,B):calls.append(len(A));return solver(A,B)
    result,work=stage(solve=track)
    assert len(calls)==2 and calls[1]>calls[0]and len(work.calls)==3
    for mode,b in result.items():
        assert set(b['coefficients']['flat']['valid'])<=set(c.MENUS[mode])
        assert all(set(t['eligible_this_fold'])<=set(c.MENUS[mode])for t in b['trace'])
        assert all(i in c.MENUS[mode]for i in np.flatnonzero(b['weights_by_mode']['flat']>0))
    assert c.ppw.BANK==bank and c.ppw.all_chunks is fn
    assert set(work.calls[-1][1])=={int(i)for b in result.values()for i in np.flatnonzero(b['weights_by_mode']['flat']>0)}
def test_final_length1_alias_exact_mass_before_allocation():
    calls=[]
    def select(A,B):
        calls.append(len(A));v=np.zeros(len(A));v[0 if len(calls)==1 else -1]=1.;return {'weights':v}
    b=stage(work=Work(length=1),solve=select)[0]['finite77'];aliases=b['diagnostics']['final_IID_aliases'];assert len(aliases)==1
    derivative,base=next(iter(aliases.items()));v=b['weights_by_mode']['flat'];assert v[derivative]==0 and v[base]==1 and v.sum()==1
    counts=c.energy.even_allocation(v,240);assert counts[base]==240 and counts[derivative]==0
    assert derivative in b['coefficients']['flat']['valid']and b['fit']['flat']['weights'][-1]==1
def test_mode_solver_failure_isolation_source_and_bundle_guards():
    calls=[]
    def fail_one(A,B):
        calls.append(len(A))
        if len(calls)==2:raise ArithmeticError('fixture failure')
        return solver(A,B)
    result,_=stage(solve=fail_one)
    assert result['finite77']['unsupported_solver']and result['finite77']['weights_by_mode']['flat']is None
    c.build_model('finite41',fixture()[2],result['finite41'])
    with pytest.raises(ArithmeticError,match='no substitution'):c.build_model('finite77',fixture()[2],result['finite77'])
    with pytest.raises(ValueError,match='source-bound'):c.build_model('finite77',fixture()[2],result['finite41'])
    assert c.identity('finite41')!=c.identity('finite77')
    with patch.object(c,'sources',return_value={}):
        with pytest.raises(ValueError,match='source-bound'):c.build_model('finite41',fixture()[2],result['finite41'])
def test_selected_current_derivative_failure_does_not_alias_or_discard_control():
    class FailDerivative(Work):
        def prefix_fits(self,x,ids,w):
            prepared,failed=super().prefix_fits(x,ids,w)
            return prepared,({i:{'type':'fixture','reason':'failed current dependency'}for i in ids if i in c.DERIVATIVES}if len(x)==40 else failed)
    calls=[]
    def select(A,B):
        calls.append(len(A));v=np.zeros(len(A));v[0 if len(calls)==1 else -1]=1.;return {'weights':v}
    result,_=stage(work=FailDerivative(length=1),solve=select)
    assert result['finite77']['unsupported_selected_current']and not result['finite77']['diagnostics']['final_IID_aliases']
    assert sum(result['finite77']['weights_by_mode']['flat'][list(c.DERIVATIVES)])==1.
    c.build_model('finite41',fixture()[2],result['finite41'])
    with pytest.raises(ArithmeticError):c.build_model('finite77',fixture()[2],result['finite77'])
