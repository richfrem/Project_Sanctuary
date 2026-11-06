"""Pre-registered instrument checks (MISSION-057)."""
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import primes_lastdigit as P

PRIMES = P.sieve(10 ** 8)


class Checks(unittest.TestCase):
    def test_prime_counts(self):
        for lim, pi in ((10 ** 6, 78498), (10 ** 7, 664579), (10 ** 8, 5761455)):
            self.assertEqual(int(np.searchsorted(PRIMES, lim, "right")), pi)

    def test_sieve_agrees_with_trial_division_on_dev_window(self):
        ref = P.trial_division_primes(10 ** 4, 2 * 10 ** 4)           # a slice of the dev window is enough to catch indexing errors
        self.assertTrue(np.array_equal(P.window(PRIMES, 10 ** 4, 2 * 10 ** 4), ref))

    def test_transition_rows_sum_to_digit_counts(self):
        for lo, hi in ((10 ** 4, 10 ** 5), (10 ** 5, 10 ** 6)):
            w = P.window(PRIMES, lo, hi); C = P.transitions(w)
            counts = [int(np.sum(w[:-1] % 10 == d)) for d in P.DIGITS]
            self.assertEqual([int(r) for r in C.sum(axis=1)], counts)
            self.assertEqual(int(C.sum()), len(w) - 1)

    def test_known_tiny_example(self):
        C = P.transitions(np.array([11, 13, 17, 19, 23, 29, 31]))      # 1->3, 3->7, 7->9, 9->3, 3->9, 9->1
        self.assertEqual(int(C.sum()), 6); self.assertEqual(int(C[0, 1]), 1); self.assertEqual(int(C[3, 1]), 1); self.assertEqual(int(C[1, 3]), 1)

    def test_control_density_matches_prime_density(self):
        for lo, hi in ((10 ** 5, 10 ** 6),):
            ctrl = P.cramer_set(lo, hi, np.random.default_rng(1)); real = P.window(PRIMES, lo, hi)
            self.assertLess(abs(len(ctrl) / len(real) - 1), 0.02)

    def test_formula_limit_and_value(self):
        self.assertAlmostEqual(P.cramer_same_digit_formula(1e-6), 0.25, places=5)
        self.assertAlmostEqual(P.cramer_same_digit_formula(0.5), 0.5 * 0.125 / (1 - 0.0625), places=12)
        self.assertLess(P.cramer_same_digit_formula(0.2), 0.25)

    def test_simulated_control_matches_formula_on_a_big_window(self):
        lo, hi = 10 ** 6, 3 * 10 ** 6
        s = P.cramer_set(lo, hi, np.random.default_rng(3)); q = len(s) / (0.4 * (hi - lo))
        got = P.same_digit_probs(P.transitions(s))
        self.assertTrue(all(abs(got[d] - P.cramer_same_digit_formula(q)) < 0.003 for d in P.DIGITS))

    def test_control_is_reproducible_from_seed(self):
        a = P.cramer_set(10 ** 5, 2 * 10 ** 5, np.random.default_rng(5)); b = P.cramer_set(10 ** 5, 2 * 10 ** 5, np.random.default_rng(5))
        self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
