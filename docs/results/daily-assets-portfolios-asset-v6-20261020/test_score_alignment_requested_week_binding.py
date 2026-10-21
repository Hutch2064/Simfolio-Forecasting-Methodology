"""No-science contracts for actual requested-week dates, H propagation and resume."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
import numpy as np
import run_score_alignment_requested_week1000 as week


class WeekBindingTests(unittest.TestCase):
    def test_inclusive_calendar_boundary_weekday_and_daily_targets(self):
        dates=np.array(['2020-01-02','2020-01-03','2020-01-06','2020-01-07','2020-01-08','2020-01-09','2020-01-10','2020-01-13'],dtype='datetime64[D]')
        origin={'position':1,'origin_date':'2020-01-03','max_horizon':6}
        bound=week.requested_week(dates,origin)
        self.assertEqual(bound['requested_horizon'],5)
        self.assertEqual(bound['calendar_week_boundary'],'2020-01-10')
        self.assertEqual(bound['target_last_date'],'2020-01-10')
        daily=np.arange(np.datetime64('2020-01-02'),np.datetime64('2020-01-12'))
        self.assertEqual(week.requested_week(daily,dict(origin,max_horizon=8))['requested_horizon'],7)
        with self.assertRaises(ValueError):week.requested_week(dates,dict(origin,origin_date='2020-01-04'))

    def test_actual_H_propagates_to_prepare_all36_and_both_baselines(self):
        context={'history':np.ones((30,2))*.001,'weights':np.array([.5,.5]),
                 'dates':np.arange(np.datetime64('2020-01-01'),np.datetime64('2020-01-31'))}
        bundle={'horizon':5};model=Mock();seen=[]
        def forecast(model,context_arg,h,paths,seed):
            self.assertIs(context_arg,context);self.assertEqual(h,5);self.assertEqual(paths,1000)
            seen.append((h,paths));return np.ones(h)
        with patch.object(week.core,'prepare',return_value=bundle)as prepare, \
             patch.object(week.core,'ScorePool',return_value=model)as pool, \
             patch.object(week.asset_models,'build_replacement',return_value=model)as wr, \
             patch.object(week.asset_models,'build_without_replacement',return_value=model)as wor, \
             patch.object(week.io,'forecast_loss',side_effect=forecast):
            short=week.prepare_short(context,5)
            for cfg in week.core.configurations(2):
                loss,_=week.forecast_short('candidate',cfg,context,5,918,short)
                self.assertEqual(loss.shape,(5,))
            week.forecast_short('WR',None,context,5,918)
            week.forecast_short('WOR',None,context,5,918)
        self.assertEqual(prepare.call_args.args[2],5)
        self.assertIs(prepare.call_args.args[3],context['dates'])
        self.assertEqual(pool.call_count,36);self.assertEqual(wr.call_count,1);self.assertEqual(wor.call_count,1)
        self.assertEqual(len(seen),38)

    def test_checkpoint_binds_shortH_and_preserves_original_catalogue_origin(self):
        origin={'position':29,'origin_date':'2020-01-30','max_horizon':400,'origin_label':'frozen'}
        plan={'cases':[{'series_id':'unit','origin':origin,'requested_horizon':5,
                        'calendar_week_boundary':'2020-02-06','target_dates_sha256':'dates'}],
              'configs':week.core.configurations(2),'design_digest':'unit','source_hashes':{}}
        binding=week.identity(plan,0,12)
        self.assertEqual(binding['origin']['max_horizon'],5)
        self.assertEqual(binding['catalogue_origin']['max_horizon'],400)
        with tempfile.TemporaryDirectory()as directory:
            cp=week.io.paths(directory,0,12)[0]
            week.api.original._write_checkpoint(cp,binding,np.ones(5),{'forecast_and_score_seconds':.1})
            self.assertIsNotNone(week.io.existing(directory,binding)[0])
            changed=dict(binding,requested_horizon=6,origin=dict(binding['origin'],max_horizon=6))
            with self.assertRaises(week.io.ResumeMismatch):week.io.existing(directory,changed)


if __name__=='__main__':unittest.main(verbosity=2)
