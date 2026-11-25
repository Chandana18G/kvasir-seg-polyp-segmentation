"""Is each step up the ladder really better? Paired tests over the 120 test images.

For each comparison (e.g. "U-Net pretrained vs U-Net scratch") we have two Dice values per
test image, one per method: the same images, so the test is PAIRED.
  - paired t-test: is the mean difference different from 0?
  - Wilcoxon signed-rank test: the same question without assuming normal differences
    (Dice differences are skewed: many near 0, a few large).
We test 3 comparisons at once, so the p-values are corrected with Holm's method.

A test over seeds is not possible: with 3 seeds Wilcoxon cannot reach p < 0.05 (needs >= 6).
The seed spread is reported as mean +- std instead (results/C1_results.md).

Input:  results/C1_dice_per_image.json  (from chandana/03; U-Net = mean of its 3 seeds)
Output: results/C6_stats.md
Usage:  python src/stats.py
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
COMPARISONS = [
    ("baseline", "floor", "brightness threshold vs floor (always polyp)"),
    ("scratch", "baseline", "U-Net scratch vs brightness threshold"),
    ("pretrained", "scratch", "U-Net pretrained vs U-Net scratch"),
]


def holm(p_values):
    """Holm correction: sort p-values, multiply the i-th smallest by (m - i), keep them rising."""
    m = len(p_values)
    order = np.argsort(p_values)
    corrected = np.empty(m)
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, min(1.0, (m - rank) * p_values[i]))
        corrected[i] = running_max
    return corrected


dice = json.loads((ROOT / "results" / "C1_dice_per_image.json").read_text())
rows = []
for a, b, label in COMPARISONS:
    x, y = np.array(dice[a]), np.array(dice[b])
    diff = x - y
    rows.append({
        "label": label,
        "mean_diff": diff.mean(),
        "better": int((diff > 0).sum()),
        "t_p": stats.ttest_rel(x, y).pvalue,
        "w_p": stats.wilcoxon(x, y).pvalue,
    })
t_holm = holm(np.array([r["t_p"] for r in rows]))
w_holm = holm(np.array([r["w_p"] for r in rows]))

lines = [
    "# C6 · Paired tests over the 120 test images (Holm-corrected)",
    "",
    "| Comparison | mean Dice gain | better on | paired t-test p | Wilcoxon p |",
    "|---|---|---|---|---|",
]
for r, tp, wp in zip(rows, t_holm, w_holm):
    lines.append(f"| {r['label']} | {r['mean_diff']:+.3f} | {r['better']} of 120 | {tp:.1e} | {wp:.1e} |")
lines += ["", "U-Net Dice per image = mean of its 3 seeds. p-values Holm-corrected over the 3 comparisons."]
text = "\n".join(lines) + "\n"
(ROOT / "results" / "C6_stats.md").write_text(text, encoding="utf-8")
print(text)
