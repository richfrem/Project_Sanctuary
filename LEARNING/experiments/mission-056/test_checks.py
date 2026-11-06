"""Pre-registered checks (MISSION-056). They must pass before any held-out evaluation."""
import itertools, math, random, sys, unittest
from fractions import Fraction
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rhythm as R
import brute


class Checks(unittest.TestCase):
    CASES = [(n, k) for n in range(5, 10) for k in range(2, min(5, n - 1) + 1)]

    def test_mask_probabilities_sum_to_one(self):
        for n, k in self.CASES:
            for t in (R.even_template(n, k), R.clustered_template(n, k), sorted(random.Random(n * 31 + k).sample(range(n), k))):
                _, d = R.expected_success(R.to_mask(t), n, (0.25,))
                self.assertAlmostEqual(d["mask_mass_at_p_0.25"], 1.0, places=12)

    def test_no_deletion_gives_success_one(self):
        for n, k in self.CASES:
            s, _ = R.expected_success(R.to_mask(R.even_template(n, k)), n, (0.0,))
            self.assertAlmostEqual(s[0.0], 1.0, places=12)

    def test_all_deletion_gives_one_over_distinct_rotations(self):
        for n, k in self.CASES:
            for t in (R.even_template(n, k), R.clustered_template(n, k)):
                s, d = R.expected_success(R.to_mask(t), n, (1.0,))
                self.assertAlmostEqual(s[1.0], 1.0 / d["R"], places=12)

    def test_rotating_a_template_preserves_expected_success(self):
        for n, k in self.CASES:
            t = sorted(random.Random(n * 7 + k).sample(range(n), k))
            base, _ = R.expected_success(R.to_mask(t), n, (0.25,))
            for s in range(1, n):
                rot = R.rotate(R.to_mask(t), n, s)
                got, _ = R.expected_success(rot, n, (0.25,))
                self.assertAlmostEqual(got[0.25], base[0.25], places=12)

    def test_independent_brute_force_agrees_on_tiny_cases(self):
        rng = random.Random(1)
        for n, k in self.CASES:
            for t in (R.even_template(n, k), R.clustered_template(n, k), sorted(rng.sample(range(n), k)), sorted(rng.sample(range(n), k))):
                for p in (Fraction(1, 4), Fraction(1, 10), Fraction(1, 2)):
                    exact = brute.expected_success_brute(t, n, p)
                    fast, _ = R.expected_success(R.to_mask(t), n, (float(p),))
                    self.assertAlmostEqual(float(exact), fast[float(p)], places=12, msg=(n, k, t, p))

    def test_hand_example(self):
        # n=4, template {0,2} (k=2): distinct rotations are {0,2} and {1,3} (R=2). p=0.25:
        # P(both kept)=9/16 -> success 1; P(one kept)=6/16 -> observed beat lies in exactly one rotation -> success 1; P(none)=1/16 -> 1/2.
        s, d = R.expected_success(R.to_mask([0, 2]), 4, (0.25,))
        self.assertEqual(d["R"], 2); self.assertAlmostEqual(s[0.25], 15 / 16 + (1 / 16) * 0.5, places=12)

    def test_sampling_rules(self):
        for n, k in ((13, 3), (13, 8), (24, 8), (14, 13 - 0) if False else (14, 8)):
            t, elig = R.irregular_templates(n, k)
            self.assertLessEqual(len(t), 40); self.assertEqual(len({tuple(x) for x in t}), len(t))
            ev = set(R.distinct_rotations(R.to_mask(R.even_template(n, k)), n))
            self.assertTrue(all(R.to_mask(x) not in ev for x in t))
            self.assertEqual(t, R.irregular_templates(n, k)[0])                     # deterministic from the seed


if __name__ == "__main__":
    unittest.main()
