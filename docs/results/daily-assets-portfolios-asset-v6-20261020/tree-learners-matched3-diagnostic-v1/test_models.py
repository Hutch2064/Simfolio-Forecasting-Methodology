"""Synthetic math, causal/grouped selection and actual library qualification."""
import copy,pickle,unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from . import models
from .objective import risk_gradient,direction_step,NumericalFailure

def records():
    from candidates.asset_additional_learners.test_learners import GateTests
    out=GateTests.records()
    for r in out:r['state_features']={'values':np.arange(15,dtype=float).reshape(5,3)/7,'columns':['M0','V0','D0'],'groups':['M','V','D'],'eligible':list(range(5))}
    return out

class TreeTests(unittest.TestCase):
    def test_exact_group_gradient_includes_offdiagonal_and_mass_once(self):
        r=records();work,ctx,z,b=models.preparation(r,());ell=np.random.default_rng(4).normal(size=(3,5))
        value,g,_=risk_gradient(ell,r,b,ctx['mass'],ctx['S'])
        fd=approx_derivative(lambda x:risk_gradient(x.reshape(3,5),r,b,ctx['mass'],ctx['S'])[0],ell.ravel(),method='3-point')
        np.testing.assert_allclose(g,fd.ravel(),atol=1e-9,rtol=1e-6)
        changed=[x.copy() for x in b];changed[0][0,1]+=.1
        self.assertNotEqual(risk_gradient(ell,r,changed,ctx['mass'],ctx['S'])[0],value)
        np.testing.assert_array_equal(np.diag(changed[0]),np.diag(b[0])) # vertex targets unchanged
        # Duplicating every represented mass leaves normalized risk unchanged.
        doubled=copy.deepcopy(r)
        for x in doubled:x['represented_origins']*=2
        _,c2,_,_=models.preparation(doubled,())
        self.assertEqual(value,risk_gradient(ell,doubled,b,c2['mass'],c2['S'])[0])

    def test_exact_descent_and_rejection_finite_checks(self):
        r=records();_,c,z,b=models.preparation(r,());ell=[np.zeros(5) for _ in r]
        value,g,_=risk_gradient(ell,r,b,c['mass'],c['S']);directions=list(-g.reshape(3,5))
        new,receipt=direction_step(ell,directions,r,b,c['mass'],c['S'],1e-13)
        self.assertLess(receipt['objective_after'],value);self.assertIsNotNone(new)
        rescaled,_=direction_step(ell,[x*1e-8 for x in directions],r,b,c['mass'],c['S'],1e-13)
        np.testing.assert_allclose(new,rescaled,atol=1e-15)
        bad,reason=direction_step(ell,[-x for x in directions],r,b,c['mass'],c['S'],1e-13)
        self.assertIsNone(bad);self.assertEqual(reason['endpoint'],'no_resolved_descent_direction')
        with self.assertRaises(NumericalFailure):risk_gradient([np.full(5,np.nan)]*3,r,b,c['mass'],c['S'])

    def test_maturity_and_current_outcome_do_not_enter_learning(self):
        r=records();self.assertEqual(len(models.matured(r,1)),1)
        changed=copy.deepcopy(r);changed[0]['validation_stop']=changed[1]['training_rows']+1
        self.assertEqual(models.matured(changed,1),[])
        current=copy.deepcopy(r[-1]);before=models.fit('cart',r,current)
        current['A']=np.full(5,np.nan);after=models.fit('cart',r,current)
        np.testing.assert_array_equal(before['weights'],after['weights']);self.assertEqual(before['candidate_labels'],after['candidate_labels'])

    def test_actual_forest_prefix_identity(self):
        r=records()
        for kind in ('random_forest','extra_trees'):
            one=models.fit_control(kind,r,(),0.,1)['model'];two=models.fit_control(kind,r,(),0.,2)['model']
            np.testing.assert_array_equal(one.estimators_[0].tree_.threshold,two.estimators_[0].tree_.threshold)
            np.testing.assert_array_equal(one.estimators_[0].tree_.value,two.estimators_[0].tree_.value)
            self.assertFalse(getattr(two,'oob_score',False))

    def test_all_five_actual_libraries_geometry_and_MVD(self):
        r=records()
        for groups in ((),('M','V','D')):
            for kind in models.KINDS:
                with self.subTest(kind=kind,groups=groups):
                    events=[];fit=models.fit(kind,r,r[-1],groups,lambda **event:events.append(event));w=fit['weights']
                    self.assertTrue(np.isfinite(w).all());self.assertGreaterEqual(w.min(),0.);self.assertAlmostEqual(w.sum(),1.)
                    self.assertEqual(len(fit['episodes']),2);self.assertFalse(fit['global_optimum_claim']);self.assertTrue(events)
                    pickle.dumps(models.export_model(fit['model']))
                    if 'forest_prefix_trace' in fit:
                        trace=fit['forest_prefix_trace'];self.assertGreaterEqual(len(trace),2);self.assertGreaterEqual(trace[-1]['pooled_validation_risk'],trace[-2]['pooled_validation_risk'])

    def test_actual_boost_direction_is_not_individual_CRPS_fit(self):
        r=records()
        for kind in ('lightgbm','xgboost'):
            path=models.BoostPath(kind,r,()).extend(1)
            self.assertEqual(len(path.trees),1)
            value,_,_=risk_gradient(path.logits,r,path.b,path.context['mass'],path.context['S'])
            self.assertLess(value,1.)
            from scipy.special import softmax
            for record,logits in zip(r,path.logits):
                np.testing.assert_allclose(path.predict(record,1),softmax(logits),atol=1e-14)

    def test_cached_control_preparation_preserves_predictions(self):
        r=records();prepared=models.control_preparation(r,())
        for kind in ('cart','random_forest','extra_trees'):
            fresh=models.fit_control(kind,r,(),0.,2);cached=models.fit_control(kind,r,(),0.,2,prepared)
            np.testing.assert_array_equal(models.predict_control(fresh,r[-1]),models.predict_control(cached,r[-1]))

    def test_tie_rule_and_explicit_selected_stage_failure(self):
        risk,label,tau=models.choose({(0.,1):[1.,0.,0.],(2.,1):[1.,0.,0.],(float('inf'),1):[1.,0.,0.]})
        self.assertEqual(label[0],float('inf'));self.assertEqual(tau,0.)
        path=models.BoostPath('xgboost',records()[:1],())
        with patch.object(models,'direction_step',return_value=(None,{'endpoint':'no_resolved_descent_direction'})):
            with self.assertRaises(NumericalFailure):path.extend(1)

if __name__=='__main__':unittest.main(verbosity=2)
