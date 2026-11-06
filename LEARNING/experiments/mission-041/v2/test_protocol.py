"""Premeasurement protocol controls, without importing held-out mutations."""
import importlib.util
import random
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import specifications
import references


class ProtocolTests(unittest.TestCase):
    def test_twenty_components_cover_five_families(self):
        self.assertEqual(len(specifications.NAMES),20)
        self.assertEqual(len(set(specifications.NAMES)),20)
        for family in set(specifications.FAMILIES.values()):
            self.assertEqual(list(specifications.FAMILIES.values()).count(family),4)

    def test_distinct_generators_agree_with_independent_oracles(self):
        for name in specifications.NAMES:
            with self.subTest(name=name):
                streams = []
                for generate in [specifications.example_generate, specifications.property_generate]:
                    rng = random.Random(41052)
                    stream = [generate(name,rng) for _ in range(64)]
                    streams.append(stream)
                    for case in stream:
                        self.assertEqual(getattr(references,name)(case), specifications.expected(name,case))
                self.assertNotEqual(streams[0],streams[1])

    def test_property_constructor_has_no_example_alias(self):
        import inspect
        source = inspect.getsource(specifications.property_generate)
        self.assertNotIn('example_generate(',source)
        self.assertNotIn('mutations',source)
        self.assertNotIn('witness',source)

    def test_v1_six_references_are_behaviorally_preserved(self):
        v1 = ROOT.parent/'fixtures/references.py'
        spec = importlib.util.spec_from_file_location('v1_references',v1)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        for name in specifications.OLD:
            rng = random.Random(41052)
            for _ in range(64):
                case = specifications.example_generate(name,rng)
                self.assertEqual(getattr(old,name)(case),getattr(references,name)(case))


if __name__ == '__main__':
    unittest.main()
