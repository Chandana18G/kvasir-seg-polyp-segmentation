# C6 · Paired tests over the 120 test images (Holm-corrected)

| Comparison | mean Dice gain | better on | paired t-test p | Wilcoxon p |
|---|---|---|---|---|
| brightness threshold vs floor (always polyp) | +0.080 | 78 of 120 | 1.1e-04 | 1.3e-04 |
| U-Net scratch vs brightness threshold | +0.432 | 111 of 120 | 3.2e-28 | 2.4e-18 |
| U-Net pretrained vs U-Net scratch | +0.106 | 104 of 120 | 1.4e-08 | 5.3e-13 |

U-Net Dice per image = mean of its 3 seeds. p-values Holm-corrected over the 3 comparisons.
