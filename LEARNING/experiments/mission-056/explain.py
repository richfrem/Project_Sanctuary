"""EXPLORATORY (post hoc, NOT frozen, cannot affect the verdict): why do irregular templates win? Hypothesis from the dev/held-out split:
repeated cyclic distances between beats make two surviving beats ambiguous; irregular templates have more distinct distances."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rhythm as R


def distinct_distances(pos, n):
    return len({(a - b) % n for a in pos for b in pos if a != b})


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 0 and np.std(ry) > 0 else None


rows, corrs = [], []
for n in range(13, 25):
    for k in range(3, min(8, n - 1) + 1):
        irr, _ = R.irregular_templates(n, k)
        temps = [R.even_template(n, k)] + irr
        s = [R.expected_success(R.to_mask(t), n, (0.25,))[0][0.25] for t in temps]
        d = [distinct_distances(t, n) for t in temps]
        c = spearman(d, s)
        if c is not None:
            corrs.append(c)
        rows.append({"n": n, "k": k, "even_distinct_distances": d[0], "irregular_mean_distinct_distances": float(np.mean(d[1:])), "spearman_distinct_distances_vs_success": c})
def candidates(template, n, observed):
    ob = R.to_mask(observed)
    return sum((r & ob) == ob for r in R.distinct_rotations(R.to_mask(template), n))
ex = {"n": 13, "k": 3, "even_template": R.even_template(13, 3), "irregular_example": [0, 1, 4],
      "even_template_observing_[0,4]_candidates": candidates(R.even_template(13, 3), 13, [0, 4]),
      "irregular_example_observing_[0,1]_candidates": candidates([0, 1, 4], 13, [0, 1]),
      "expected_success_p0.25": {"even": R.expected_success(R.to_mask(R.even_template(13, 3)), 13, (0.25,))[0][0.25],
                                 "irregular_example": R.expected_success(R.to_mask([0, 1, 4]), 13, (0.25,))[0][0.25]}}
out = {"label": "exploratory, post hoc; not part of the frozen protocol", "mean_spearman_over_pairs": float(np.mean(corrs)), "pairs": len(corrs),
       "share_of_pairs_with_positive_correlation": float(np.mean([c > 0 for c in corrs])), "example": ex, "rows": rows}
Path("explain-exploratory.json").write_text(json.dumps(out, indent=1))
print(round(out["mean_spearman_over_pairs"], 3), round(out["share_of_pairs_with_positive_correlation"], 3), json.dumps(ex))
