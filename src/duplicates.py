"""Find near-identical images (the same photo saved twice, or almost the same frame).

Step 1, a "difference hash" (dHash): shrink the image to 9 x 8 grey pixels and, in
each row, note whether each pixel is brighter than its right neighbour. That gives
8 x 8 = 64 yes/no answers, a "fingerprint". Two copies of the same photo give
almost the same fingerprint, even after resizing or JPEG compression. We count how
many of the 64 answers differ (the "Hamming distance"): 0 = same picture.

Step 2: the black octagon border looks the same in every frame, so many different
photos also get a small distance. We therefore only call two images duplicates if
their polyp masks also overlap well (IoU >= 0.5). We chose both limits by looking at
the 25 closest pairs by eye (notebook 02).
"""
import numpy as np
from PIL import Image

MAX_DISTANCE = 3   # fingerprints differ in at most 3 of 64 places
MIN_MASK_IOU = 0.5  # and the polyps sit in the same place


def dhash(image):
    """64 True/False values for one RGB image (a numpy array)."""
    grey = Image.fromarray(image).convert("L").resize((9, 8), Image.BILINEAR)
    pixels = np.array(grey, dtype=int)
    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


def mask_iou(mask_a, mask_b):
    """Overlap of two 0/1 masks of the same size: shared area / combined area."""
    both = np.logical_and(mask_a, mask_b).sum()
    either = np.logical_or(mask_a, mask_b).sum()
    return both / either if either > 0 else 0.0


def close_pairs(names, hashes, max_distance):
    """All pairs (name_a, name_b, distance) with distance <= max_distance."""
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            distance = int((hashes[i] != hashes[j]).sum())
            if distance <= max_distance:
                pairs.append((names[i], names[j], distance))
    return pairs


def find_duplicates(names, hashes, small_masks):
    """Pairs that pass both tests. small_masks: name -> resized 0/1 mask."""
    duplicates = []
    for a, b, distance in close_pairs(names, hashes, MAX_DISTANCE):
        iou = mask_iou(small_masks[a], small_masks[b])
        if iou >= MIN_MASK_IOU:
            duplicates.append((a, b, distance, round(float(iou), 2)))
    return duplicates
