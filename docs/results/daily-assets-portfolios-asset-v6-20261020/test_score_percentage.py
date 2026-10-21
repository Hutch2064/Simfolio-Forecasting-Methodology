import unittest
from score_percentage import percent_better,normalize_model_percentages

class PercentageRegression(unittest.TestCase):
    def test_known_quarter_improvement_and_regression(self):
        self.assertEqual(percent_better(3.,4.),25.);self.assertEqual(percent_better(5.,4.),-25.);self.assertEqual(percent_better(4.,4.),0.)
        with self.assertRaises(ValueError):percent_better(1.,0.)
    def test_simulation_count_cannot_change_display_scale(self):
        for paths in (100,1000):
            report={'paths':paths,'reference_main_Q':{'fixed_reference':4.},'models':{'m':{'main_Q':3.,'percent_better_vs_fixed_reference':250.}},'complete_identical_coverage_screen_ranking':[{'model_id':'m','main_Q':3.,'percent_better_vs_fixed_reference':250.}]}
            corrected,changes=normalize_model_percentages(report);self.assertEqual(report['models']['m']['percent_better_vs_fixed_reference'],250.);self.assertEqual(corrected['models']['m']['percent_better_vs_fixed_reference'],25.);self.assertEqual(corrected['complete_identical_coverage_screen_ranking'][0]['percent_better_vs_fixed_reference'],25.);self.assertEqual(len(changes),1);self.assertEqual(normalize_model_percentages(corrected)[1],[])

if __name__=='__main__':unittest.main()
