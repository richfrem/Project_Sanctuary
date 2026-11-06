"""Phase-1 checks only: no frozen arm ever executes a held-out mutant."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import components
import harness
import pools
import specifications as spec


def canonical(value):
    return json.dumps(value, sort_keys=True)


class PhaseOneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suites = json.loads((ROOT/'suites.json').read_text())
        cls.curation = json.loads((ROOT/'curation.json').read_text())

    def test_pre_mutant_pool_seal_intact(self):
        seal = json.loads((ROOT/'pool-seal.json').read_text())
        for filename, digest in seal['sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/filename).read_bytes()).hexdigest(), digest)
        self.assertEqual(canonical(pools.build()), canonical(self.suites))

    def test_all_twenty_have_literal_pre_mutant_unit_tests(self):
        self.assertEqual(set(self.suites), set(spec.NAMES))
        for name, s in self.suites.items():
            self.assertGreaterEqual(len(s['existing']), 5)
            self.assertEqual(s['existing_provenance'], 'authored_v3_before_mutants')
            for case, wanted in s['existing']:
                self.assertEqual(spec.expected(name, case), wanted, (name, case))

    def test_reference_probes_and_exact_call_parity(self):
        for name, s in self.suites.items():
            for arm in ['property', 'example', 'existing']:
                result = harness.execute(getattr(components, name), name, arm, s[arm])
                self.assertEqual(result['status'], 'EXHAUSTED', (name, arm, result))
                self.assertEqual(result['calls'], 16 if arm != 'existing' else len(s[arm]))

    def test_all_frozen_inputs_are_in_declared_domain(self):
        for name, s in self.suites.items():
            def semantic_case(c):
                # Migration keys are a SET, domain enumerates canonical set representatives.
                return [c[0], sorted(c[1])] if name == 'migration' else c
            legal = {canonical(semantic_case(c)) for c in spec.domain(name)}
            inputs = list(s['example']) + [p[0] for p in s['existing']]
            inputs += [case for a, b, law in s['property'] for case in [a, b]]
            self.assertTrue(all(canonical(semantic_case(c)) in legal for c in inputs), name)

    def test_independent_specification_and_generator_dependencies(self):
        for file in ['specifications.py', 'pools.py']:
            tree = ast.parse((ROOT/file).read_text())
            imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
            imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
            self.assertFalse(set(imports) & {'components', 'build_mutants', 'curate', 'mutations'})

    def test_all_candidates_are_changed_executable_source(self):
        source = (ROOT/'components.py').read_text()
        originals = {n.name: ast.get_source_segment(source, n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
        split = json.loads((ROOT/'operator-split.json').read_text())
        self.assertFalse(set(split['development']) & set(split['heldout']))
        for m in json.loads((ROOT/'mutants.json').read_text()):
            self.assertIn(m['operator'], split['heldout'])
            changed = (ROOT/m['file']).read_text()
            self.assertNotEqual(ast.dump(ast.parse(originals[m['component']])), ast.dump(ast.parse(changed)))
            self.assertEqual(changed, originals[m['component']].replace(m['old'], m['new'])+'\n')

    def test_exhaustive_curation_complete_and_diagnostic(self):
        summary = self.curation['summary']
        self.assertEqual(summary['applicable_candidates'], 85)
        self.assertEqual(summary['true_equivalents'], 5)
        self.assertEqual(summary['non_equivalents'], 80)
        densities = []
        for c in self.curation['components']:
            self.assertEqual(sum(1 for _ in spec.domain(c['component'])), c['domain_size'])
            self.assertEqual(sum(m['classification'] == 'NON_EQUIVALENT' for m in c['candidates']), 4)
            for m in c['candidates']:
                self.assertEqual(m['kill_density'], m['distinguishing_cases']/m['domain_size'])
                self.assertEqual(m['first_witness'] is None, m['classification'] == 'TRUE_EQUIVALENT')
                if m['kill_density']:
                    densities.append(m['kill_density'])
        self.assertLess(min(densities), .01)
        self.assertLessEqual(max(densities), .23)  # ~20%, explicitly reported as 22.4076% max

    def test_worker_reference_smoke(self):
        result = harness.isolated(ROOT/'components.py', 'lease', 'lease', 'property', self.suites['lease']['property'])
        self.assertEqual(result['status'], 'EXHAUSTED')
        self.assertEqual(result['calls'], 16)

    def test_oracle_errors_are_not_detections(self):
        with self.assertRaises(KeyError):
            harness.execute(lambda c: None, 'absent_contract', 'example', [[1]])

    def test_every_call_is_copied_and_charged(self):
        def mutate(c):
            c.append('a')
            return '/a'
        cases = [[]]
        saved = copy.deepcopy(cases)
        result = harness.execute(mutate, 'path', 'example', cases)
        self.assertEqual(cases, saved)
        self.assertEqual(result['calls'], 1)
        self.assertEqual(result['status'], 'COUNTEREXAMPLE')

    def test_scoring_has_not_run(self):
        self.assertFalse((ROOT/'phase2-results.json').exists())
        self.assertFalse((ROOT/'phase2-checkpoint.json').exists())


if __name__ == '__main__':
    unittest.main()
