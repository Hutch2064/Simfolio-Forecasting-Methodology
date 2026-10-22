import unittest
import score_series_buckets_v9 as s

class Tests(unittest.TestCase):
    def setUp(self):
        self.meta={k:dict(panel_branch=b,asset_class=c,family=f) for k,b,c,f in
                   [('a','assets','eq','f'),('b','assets','eq','f'),('p','portfolios','mix','p')]}
        self.rows=[dict(request_id=b+k,series_id=k,origin_date='2000-01-01',horizon_bucket=b,score=v)
                   for b,v in zip(s.BUCKETS,[1.,4.,10.]) for k in ('a','b','p')]
        self.w=s.build_weights(self.rows,self.meta)
    def test_equal_thirds_cap_and_no_quarter_dependence(self):
        out=s.aggregate_records(self.rows,self.meta,weights=self.w)
        self.assertEqual(out['score'],5.)
        for b in s.BUCKETS:
            self.assertEqual(sum(v['weight'] for v in self.w['requests'].values() if v['bucket']==b and v['series_id']=='p'),.5)
        moved=[dict(r,origin_date='1900-04-01') for r in self.rows]
        self.assertEqual(s.aggregate_records(moved,self.meta,weights=self.w)['score'],5.)
    def test_partial_never_zero_imputes_or_reweights_plan(self):
        out=s.aggregate_records(self.rows[:1],self.meta,weights=self.w)
        self.assertIsNone(out['score']);self.assertFalse(out['complete'])
        self.assertEqual(out['buckets']['short']['observed_weight_support'],.25)
        self.assertEqual(out['buckets']['short']['partial_score'],1.)
    def test_series_request_count_invariance(self):
        rows=self.rows+[dict(self.rows[0],request_id='extra',score=1.)]
        w=s.build_weights(rows,self.meta)
        self.assertEqual(s.aggregate_records(rows,self.meta,weights=w)['score'],5.)
        self.assertEqual(w['requests']['shorta']['weight'],.125)
    def test_raw_origin_score_no_scale(self):
        self.assertEqual(s.origin_score([2.,4.],[0.,0.],'2000-01-01',['2000-01-02','2000-01-03']),3.)
        with self.assertRaises(ValueError):s.origin_score([-1.],[], '2000-01-01',['2000-01-02'])
        with self.assertRaises(ValueError):s.origin_score([1.],[], '2000-01-01',['1999-01-02'])

    def test_invalid_missing_duplicate(self):
        with self.assertRaises(ValueError):s.build_weights([r for r in self.rows if r['series_id']=='p'],self.meta)
        with self.assertRaises(ValueError):s.build_weights(self.rows+self.rows[:1],self.meta)
        with self.assertRaises(ValueError):s.aggregate_records([dict(self.rows[0],score=-1)],self.meta,weights=self.w)
    def test_dates_classification_and_two_estimands(self):
        from score_horizon_buckets_v8 import horizon_bucket
        self.assertEqual(horizon_bucket('2000-02-29','2003-02-28'),'short')
        self.assertEqual(horizon_bucket('2000-02-29','2020-02-29'),'long')
        baseline=s.aggregate_records(self.rows,self.meta,weights=self.w)
        changed=[dict(r,score=r['score']*(.5 if r['horizon_bucket']=='short' else 1)) for r in self.rows]
        comparison=s.compare(s.aggregate_records(changed,self.meta,weights=self.w),baseline)
        self.assertAlmostEqual(comparison['mean_equal_bucket_percent_gain'],100/6)
        self.assertAlmostEqual(comparison['percent_gain_of_equal_bucket_loss'],100*.5/15)

if __name__=='__main__':unittest.main()
