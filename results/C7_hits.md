# C7 · Not miss one: images and polyps found (120 test images, threshold 0.5)

| Setting | images: polyp touched | images: IoU ≥ 0.5 | polyps found (any overlap) | polyps ≥ half covered | images with a false alarm |
|---|---|---|---|---|---|
| brightness threshold | 104 of 120 (86.7 %) | 15 of 120 (12.5 %) | 108 of 127 (85.0 %) | 79 of 127 (62.2 %) | 16 of 120 (13.3 %) |
| U-Net scratch (3 seeds) | 117.0 ± 1.7 of 120 (97.5 %) | 89.0 ± 1.7 of 120 (74.2 %) | 120.7 ± 2.1 of 127 (95.0 %) | 106.0 ± 2.6 of 127 (83.5 %) | 25.0 ± 8.5 of 120 (20.8 %) |
| U-Net pretrained (3 seeds) | 120.0 ± 0.0 of 120 (100.0 %) | 106.7 ± 1.2 of 120 (88.9 %) | 125.7 ± 0.6 of 127 (99.0 %) | 120.0 ± 2.0 of 127 (94.5 %) | 14.7 ± 8.7 of 120 (12.2 %) |

A polyp = one connected region of the doctor's mask (≥ 20 pixels at 256 × 256); 127 polyps in 120 images.
Touched = any overlap; IoU ≥ 0.5 per image; half covered = ≥ 50 % of that polyp's pixels predicted.
False alarm = a predicted region (≥ 20 pixels) that touches no polyp.
Images where at least one pretrained seed misses a polyp: 3.

## Learning-curve gap (Dice, mean of 3 seeds, best validation epoch)

| U-Net | training (200 of 779) | validation (100) | test (120) | gap training − validation |
|---|---|---|---|---|
| scratch | 0.827 | 0.788 | 0.756 | +0.039 |
| pretrained | 0.928 | 0.870 | 0.862 | +0.058 |
