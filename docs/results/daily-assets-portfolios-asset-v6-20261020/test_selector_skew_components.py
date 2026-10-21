from pathlib import Path
import sys, unittest
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';sys.path.insert(0,str(V))
import common
from candidates.asset_selector_skew_components import bank, candidate as c
from candidates.asset_selector_skew_components.paths import group_chunks
from candidates.asset_selector_batched_paths.paths import group_chunks as gaussian_chunks
from candidates.asset_selector_universe import bank as original
from candidates.asset_selector_menu_components import candidate as control
from candidates.asset_SAS_skew import skew


class SkewComponentsTests(unittest.TestCase):
    def fixture(self):
        x=np.random.default_rng(522).normal(.0001,.01,(40,3));w=np.array([.2,.3,.5]);dates=np.datetime64('2002-01-01')+np.arange(len(x))
        return x,w,dates

    def test_Gaussian_boundary_is_exact_for_all9families_and6mean_choices(self):
        x,w,_=self.fixture();prepared,invalid=bank.prefix_fits(x,list(range(len(bank.BANK))),w);self.assertFalse(invalid)
        for family in bank.fits.FAMILIES:
            bundle={**prepared['families'][family]};bundle['SAS_fit']=[{**fit,'BIC_t':0.,'positive_second_moment':.5,'negative_second_moment':.5}for fit in bundle['SAS_fit']]
            specs=[s for s in bank.SKEW if s['family']==family]
            old_ids=[i for i,s in enumerate(original.BANK)if s.get('family')==family and s.get('innovations')=='Gaussian']
            a=np.concatenate([values for _,values in gaussian_chunks(x,w,7,10,81,old_ids,bundle,width=3)],axis=2)
            b=np.concatenate([values for _,values in group_chunks(x,w,7,10,81,specs,bundle,width=3)],axis=2)
            np.testing.assert_array_equal(a,b)

    def test_original_and_flattened_controls_remain_exact(self):
        x,w,dates=self.fixture();a=control.prepare(x,w,4,dates,inner_paths=4);b=c.prepare(x,w,4,dates,inner_paths=4)
        for name in ('original','flat','flat_PPW'):
            for key in ('A','B'):
                np.testing.assert_array_equal(a['coefficients'][name][key],b['coefficients'][name][key])
            np.testing.assert_array_equal(a['weights_by_mode'][name],b['weights_by_mode'][name][:170])
            pa=control.MenuPool({'menu':name,'inner_paths':4},dates,a);pb=c.MenuPool({'menu':name,'inner_paths':4},dates,b)
            np.testing.assert_array_equal(np.concatenate([v for _,v in pa.terminal_chunks(x,w,4,1000,61)],axis=1),np.concatenate([v for _,v in pb.terminal_chunks(x,w,4,1000,61)],axis=1))
            self.assertEqual(pa.model_id,pb.model_id)

    def test_group_and_single_option_joint_paths_match(self):
        x,w,_=self.fixture();prepared,invalid=bank.prefix_fits(x,list(range(len(bank.BANK))),w);self.assertFalse(invalid)
        for family in bank.fits.FAMILIES:
            ids=[i for i,s in enumerate(bank.BANK)if i>=170 and s['family']==family]
            specs=[bank.BANK[i]for i in ids];bundle=prepared['families'][family]
            grouped=np.concatenate([v for _,v in group_chunks(x,w,9,10,91,specs,bundle,width=3)],axis=2)
            for offset,index in enumerate(ids):
                single=np.concatenate([v for _,v in bank.model(index,prepared).terminal_chunks(x,w,9,10,91,width=3)],axis=1)
                np.testing.assert_array_equal(single,grouped[offset])

    def test_shape_fits_use_only_training_innovations_and_preserve_standardization(self):
        rng=np.random.default_rng(623);z=rng.normal(size=20000);y=skew.transform(z,.7);fitted=skew.fit(y)
        self.assertGreater(fitted['BIC_t'],.5)
        from scipy.integrate import quad
        density=lambda z:np.exp(-z*z/2)/np.sqrt(2*np.pi)
        for t in(-.8,.5,.9):
            first=quad(lambda z:float(skew.transform(z,t))*density(z),-12,12,epsabs=1e-10)[0]
            second=quad(lambda z:float(skew.transform(z,t))**2*density(z),-12,12,epsabs=1e-10)[0]
            self.assertAlmostEqual(first,0,places=10);self.assertAlmostEqual(second,1,places=10)
        x,w,dates=self.fixture();b=c.prepare(x,w,4,dates,inner_paths=2)
        for fold in b['trace']:
            self.assertLessEqual(fold['validation_stop'],len(x))
            for family,entries in fold['SAS_fits'].items():
                self.assertEqual(len(entries),3)
                self.assertTrue(all(e['training_rows']==fold['training_rows']for e in entries))

    def test_GJR_uses_actual_asymmetric_second_moments(self):
        from candidates.asset_selector_skew_components.shape import partial_moments
        from candidates.asset_selector_skew_components.paths import gjr_parameters
        p=np.zeros((10,3));p[1]=.8;p[2]=.1;p[3]=.35
        shapes=[]
        for t in(-.8,0.,.8):
            pos,neg=partial_moments(t);self.assertAlmostEqual(pos+neg,1,places=14)
            shapes.append({'positive_second_moment':pos,'negative_second_moment':neg})
        self.assertNotEqual(shapes[0]['positive_second_moment'],shapes[2]['positive_second_moment'])
        q=gjr_parameters(p,shapes,40)
        actual=q[1]+q[2]*np.array([s['positive_second_moment']for s in shapes])+q[3]*np.array([s['negative_second_moment']for s in shapes])
        self.assertTrue(np.all(actual<1));np.testing.assert_allclose(q[4],1-actual,rtol=0,atol=2e-16)

    def test_menu_membership_no_parent_and_exact_final_count(self):
        self.assertEqual(len(bank.BANK),224);self.assertEqual(len(bank.SKEW),54)
        x,w,dates=self.fixture();b=c.prepare(x,w,4,dates,inner_paths=2)
        for name,ids in bank.MENUS.items():
            weights=b['weights_by_mode'][name]
            self.assertTrue(np.all(weights[[i for i in range(len(bank.BANK))if i not in ids]]==0))
            if name!='original':self.assertEqual(weights[0],0.)
            model=c.MenuPool({'menu':name,'inner_paths':2},dates,b)
            values=np.concatenate([v for _,v in model.terminal_chunks(x,w,4,1000,33)],axis=1)
            self.assertEqual(values.shape,(1000,4));self.assertTrue(np.isfinite(values).all());self.assertEqual(sum(model.last_selection['path_counts']),1000)


if __name__=='__main__':unittest.main()
