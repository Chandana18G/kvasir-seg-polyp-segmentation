"""Scores for one predicted mask against the doctor's mask (both 0/1 arrays).

All four count pixels:
  overlap   = pixels that are polyp in both
  Dice      = 2 * overlap / (predicted pixels + true pixels)     (main score)
  IoU       = overlap / pixels that are polyp in either
  precision = overlap / predicted pixels   (how much of what we drew is really polyp)
  recall    = overlap / true pixels        (how much of the polyp we found)
We score every image on its own, then take the mean over images.
"""
import numpy as np


def scores(pred, truth):
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    overlap = np.logical_and(pred, truth).sum()
    either = np.logical_or(pred, truth).sum()
    n_pred, n_true = pred.sum(), truth.sum()
    return {
        "dice": 2 * overlap / (n_pred + n_true) if n_pred + n_true > 0 else 1.0,
        "iou": overlap / either if either > 0 else 1.0,
        "precision": overlap / n_pred if n_pred > 0 else 0.0,
        "recall": overlap / n_true if n_true > 0 else 1.0,
    }


def mean_scores(preds, truths):
    """Mean of each score over all images, plus the list of Dice per image."""
    per_image = [scores(p, t) for p, t in zip(preds, truths)]
    result = {k: float(np.mean([s[k] for s in per_image])) for k in per_image[0]}
    result["dice_per_image"] = [float(s["dice"]) for s in per_image]
    return result
