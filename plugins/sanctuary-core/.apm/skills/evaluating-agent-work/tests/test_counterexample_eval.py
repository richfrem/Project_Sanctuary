"""Behavioral tests for bounded search and non-inflating mutation scoring."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[6]
SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'counterexample_eval.py'
spec = importlib.util.spec_from_file_location('counterexample_eval', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
EXPERIMENT = REPO / 'LEARNING' / 'experiments' / 'mission-041'


class SearchTests(unittest.TestCase):
    """Exercise real isolated workers, including hangs and oracle errors."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='unit-', dir=EXPERIMENT)
        cls.root = Path(cls.temp.name)
        cls.component = cls.root / 'component.py'
        cls.component.write_text('''def correct(c):
    return c * 2
def wrong(c):
    return c * 2 + 1
def hangs(c):
    while True:
        pass
def throws(c):
    raise ValueError('bad candidate')
def noisy(c):
    print('candidate debug')
    return c * 2
def mutates(c):
    c.append(99)
    return 0
''')
        cls.claim = cls.root / 'claim.py'
        cls.claim.write_text('''def generate(rng):
    return rng.randrange(100)
def example_generate(rng):
    return rng.randrange(100)
def expected(c):
    return c + c
def check(c, actual):
    return actual == expected(c)
''')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_search(self, function, **kwargs):
        return runner.search(f'{self.component}:{function}', str(self.claim), **kwargs)

    def test_reference_exhaustion_is_not_proof(self):
        result = self.run_search('correct', trials=17)
        self.assertEqual(result['status'], 'EXHAUSTED')
        self.assertEqual(result['tested'], 17)
        self.assertIn('not proof', result['note'])

    def test_reproducible_counterexample(self):
        a = self.run_search('wrong', seed=123)
        b = self.run_search('wrong', seed=123)
        self.assertEqual(a['status'], 'COUNTEREXAMPLE')
        self.assertEqual(a['case'], b['case'])
        self.assertEqual(a['tested'], b['tested'])
        replay = self.run_search('wrong', mode='witness', trials=1, cases=[a['case']])
        self.assertEqual(replay['status'], 'COUNTEREXAMPLE')

    def test_example_baseline_executes(self):
        self.assertEqual(self.run_search('wrong', mode='example')['status'], 'COUNTEREXAMPLE')

    def test_timeout_is_inconclusive(self):
        result = self.run_search('hangs', seconds=0.15)
        self.assertEqual(result['status'], 'TIMEOUT')
        self.assertIsNone(result['tested'])
        self.assertLess(result['elapsed_seconds'], 3)

    def test_candidate_exception_is_failure_on_valid_input(self):
        result = self.run_search('throws')
        self.assertEqual(result['status'], 'COUNTEREXAMPLE')
        self.assertEqual(result['reason'], 'component_exception')

    def test_property_exception_is_error(self):
        claim = self.root / 'broken.py'
        claim.write_text(self.claim.read_text().replace('return actual == expected(c)', 'raise RuntimeError("oracle")'))
        result = runner.search(f'{self.component}:correct', str(claim))
        self.assertEqual(result['status'], 'ERROR')
        self.assertIn('oracle', result['detail'])

    def test_non_boolean_property_is_error(self):
        claim = self.root / 'not_bool.py'
        claim.write_text(self.claim.read_text().replace('return actual == expected(c)', 'return 1'))
        self.assertEqual(runner.search(f'{self.component}:correct', str(claim))['status'], 'ERROR')

    def test_import_error_is_not_detection(self):
        self.assertEqual(self.run_search('missing_function')['status'], 'ERROR')

    def test_stdout_does_not_corrupt_report(self):
        self.assertEqual(self.run_search('noisy')['status'], 'EXHAUSTED')

    def test_candidate_mutation_does_not_mutate_oracle_input(self):
        claim = self.root / 'mutable.py'
        claim.write_text('''def generate(rng):
    return []
example_generate = generate
def expected(c):
    return len(c)
def check(c, actual):
    return actual == len(c)
''')
        self.assertEqual(runner.search(f'{self.component}:mutates', str(claim), trials=2)['status'], 'EXHAUSTED')

    def test_invalid_budgets_rejected(self):
        for kwargs in ({'trials': 0}, {'trials': 1.5}, {'seconds': 0},
                       {'seconds': float('nan')}, {'seconds': float('inf')},
                       {'mode': 'unknown'}, {'mode': 'witness', 'trials': 2, 'cases': [1]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_search('correct', **kwargs)


class BenchmarkTests(unittest.TestCase):
    """Use development mutants only; score held-out faults in a separate run."""

    @classmethod
    def setUpClass(cls):
        cls.report = runner.benchmark(EXPERIMENT / 'fixtures' / 'manifest.json',
                                      split='development', trials=16)

    def test_development_curation_and_detection(self):
        summary = self.report['summary']
        self.assertEqual(summary['mutants'], 6)
        self.assertEqual(summary['property_detection_percent'], 100)
        self.assertEqual(summary['reference_false_alarm_percent'], 0)
        self.assertTrue(summary['valid_measurement'])
        self.assertEqual(summary['scoped_verdict'], 'FALSIFIED')
        self.assertEqual(summary['delta_pp'], 0)

    def test_budget_parity(self):
        for component in self.report['components']:
            for mutant in component['mutants']:
                self.assertTrue(mutant['non_equivalent'])
                for key in ('trial_budget', 'seconds_budget', 'seed'):
                    self.assertEqual(mutant['property'][key], mutant['example'][key])

    def test_partition_separates_operators(self):
        data = json.loads((EXPERIMENT / 'fixtures' / 'manifest.json').read_text())
        dev, heldout = set(), set()
        heldout_count = 0
        for c in data['components']:
            for m in c['mutants']:
                (dev if m['split'] == 'development' else heldout).add(m['operator'])
                heldout_count += m['split'] == 'heldout'
        self.assertFalse(dev & heldout)
        self.assertGreaterEqual(heldout_count, 20)
        self.assertEqual(len(data['components']), 6)

    def test_timeout_invalidates_threshold_pass(self):
        report = json.loads(json.dumps(self.report))
        report['components'][0]['references']['property']['status'] = 'TIMEOUT'
        self.assertEqual(runner.summarize(report)['scoped_verdict'], 'INVALID')

    def test_failed_witness_is_not_silently_excluded(self):
        report = json.loads(json.dumps(self.report))
        report['components'][0]['mutants'][0]['non_equivalent'] = False
        summary = runner.summarize(report)
        self.assertEqual(summary['mutants'], 6)
        self.assertFalse(summary['valid_measurement'])

    def test_aggregate_component_budget(self):
        report = runner.benchmark(EXPERIMENT / 'fixtures' / 'manifest.json',
                                  split='development', component_seconds=0.001)
        self.assertEqual(report['summary']['scoped_verdict'], 'INVALID')
        self.assertGreater(report['summary']['timeouts'], 0)
        self.assertEqual(report['summary']['property_detection_percent'], 0)

    def test_empty_measurement_never_passes(self):
        result = runner.summarize({'components': []})
        self.assertFalse(result['valid_measurement'])
        self.assertEqual(result['scoped_verdict'], 'INVALID')

    def test_reference_alarm_is_counted_against_all_references(self):
        report = json.loads(json.dumps(self.report))
        report['components'][0]['references']['property']['status'] = 'COUNTEREXAMPLE'
        summary = runner.summarize(report)
        self.assertAlmostEqual(summary['reference_false_alarm_percent'], 100 / 6)
        self.assertFalse(summary['thresholds']['false_alarm_le_5'])

    def test_infrastructure_error_invalidates_measurement(self):
        report = json.loads(json.dumps(self.report))
        report['components'][0]['mutants'][0]['property']['status'] = 'ERROR'
        summary = runner.summarize(report)
        self.assertEqual(summary['errors'], 1)
        self.assertEqual(summary['mutants'], 6)
        self.assertEqual(summary['scoped_verdict'], 'INVALID')

    def test_exact_threshold_boundaries(self):
        exhausted = {'status': 'EXHAUSTED'}
        killed = {'status': 'COUNTEREXAMPLE'}
        mutants = [dict(non_equivalent=True, reference_witness=exhausted,
                        mutant_witness=killed,
                        property=killed if i < 16 else exhausted,
                        example=killed if i < 12 else exhausted)
                   for i in range(20)]
        report = {'components': [dict(references={'property': exhausted,
                                                  'example': exhausted},
                                      mutants=mutants, elapsed_seconds=600)]}
        summary = runner.summarize(report)
        self.assertEqual(summary['scoped_verdict'], 'THRESHOLDS_MET')
        self.assertEqual(summary['property_detection_percent'], 80)
        self.assertEqual(summary['delta_pp'], 20)
        report['components'][0]['elapsed_seconds'] = 600.001
        self.assertFalse(runner.summarize(report)['thresholds']['component_runtime_le_600s'])


if __name__ == '__main__':
    unittest.main()
