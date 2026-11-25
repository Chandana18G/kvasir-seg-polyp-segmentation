"""Did we find the polyp? Counting polyps, not pixels.

Recall counts pixels: a large polyp found well can hide a small polyp missed completely.
Here we count images and polyps:

Per image (each test image has at least one polyp):
  hit_any   = the prediction touches the doctor's mask at all (overlap > 0)
  hit_iou50 = IoU of prediction and doctor's mask >= 0.5

Per polyp (a "polyp" = one connected region of the doctor's mask; some images have two):
  found_any  = the prediction touches this region
  found_half = the prediction covers at least half of this region's pixels

False alarm = a connected region of the prediction that touches no polyp at all.
Regions smaller than MIN_PIXELS are ignored (a few stray pixels are not a polyp or an alarm).
"""
import numpy as np
from scipy import ndimage

MIN_PIXELS = 20          # at 256 x 256: 20 pixels = 0.03 % of the image


def regions(mask):
    """List of boolean masks, one per connected region with at least MIN_PIXELS pixels."""
    labels, n = ndimage.label(mask)
    out = []
    for k in range(1, n + 1):
        region = labels == k
        if region.sum() >= MIN_PIXELS:
            out.append(region)
    return out


def count_hits(pred, truth):
    """Counts for one image. pred, truth: 0/1 arrays of the same size."""
    pred, truth = pred.astype(bool), truth.astype(bool)
    overlap = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    polyps = regions(truth)
    found_any = sum(1 for r in polyps if np.logical_and(pred, r).any())
    found_half = sum(1 for r in polyps if np.logical_and(pred, r).sum() >= 0.5 * r.sum())
    alarms = sum(1 for r in regions(pred) if not np.logical_and(r, truth).any())
    return {
        "hit_any": overlap > 0,
        "hit_iou50": union > 0 and overlap / union >= 0.5,
        "polyps": len(polyps),
        "found_any": found_any,
        "found_half": found_half,
        "false_alarms": alarms,
    }


def summarise(preds, truths):
    """Totals over all test images."""
    rows = [count_hits(p, t) for p, t in zip(preds, truths)]
    return {
        "images": len(rows),
        "hit_any": sum(r["hit_any"] for r in rows),
        "hit_iou50": sum(r["hit_iou50"] for r in rows),
        "polyps": sum(r["polyps"] for r in rows),
        "found_any": sum(r["found_any"] for r in rows),
        "found_half": sum(r["found_half"] for r in rows),
        "images_with_false_alarm": sum(r["false_alarms"] > 0 for r in rows),
        "false_alarms": sum(r["false_alarms"] for r in rows),
    }
