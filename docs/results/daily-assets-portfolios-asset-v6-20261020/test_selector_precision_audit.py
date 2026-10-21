"""Reject invalid allocations and prevent reused-prepare timing undercounts."""
import copy, unittest
import verify_selector_inner_precision_batched72 as audit


class AuditTests(unittest.TestCase):
    def test_reused_prepare_is_charged_to_either_standalone_forecast(self):
        terminal = {'case': 1, 'configuration': {'criterion': 'terminal', 'inner_paths': 30, 'seed_replication': 2},
                    'cpu_seconds': 10., 'prepare_cpu_seconds': 8.}
        path = {'case': 1, 'configuration': {'criterion': 'path', 'inner_paths': 30, 'seed_replication': 2},
                'cpu_seconds': 3.001, 'prepare_cpu_seconds': .001}
        self.assertEqual(audit.standalone_cost(terminal, terminal), 10.)
        self.assertEqual(audit.standalone_cost(path, terminal), 11.)
        other = copy.deepcopy(terminal); other['configuration']['seed_replication'] = 1
        with self.assertRaises(AssertionError):
            audit.standalone_cost(path, other)

    def test_allocation_must_match_learned_weights_and_final_count(self):
        selection = {'ideal_weights': [.7, .3] + [0.] * 164, 'path_counts': [700, 300] + [0] * 164}
        self.assertEqual(audit.allocated_counts(selection).sum(), 1000)
        corrupt = copy.deepcopy(selection); corrupt['path_counts'][:2] = [500, 500]
        with self.assertRaises(AssertionError):
            audit.allocated_counts(corrupt)
        corrupt = copy.deepcopy(selection); corrupt['path_counts'][0] = 699
        with self.assertRaises(AssertionError):
            audit.allocated_counts(corrupt)
        corrupt = copy.deepcopy(selection); corrupt['ideal_weights'][2] = float('nan')
        with self.assertRaises(AssertionError):
            audit.allocated_counts(corrupt)


if __name__ == '__main__':
    unittest.main()
