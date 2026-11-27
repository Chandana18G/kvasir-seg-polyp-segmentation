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
    # Shrink to 9 wide x 8 high, grey. Tiny on purpose: only the rough layout survives,
    # so small changes (JPEG, resizing, noise) do not change the fingerprint.
    grey = Image.fromarray(image).convert("L").resize((9, 8), Image.BILINEAR)
    pixels = np.array(grey, dtype=int)
    # Compare each pixel with its right neighbour: 8 comparisons per row x 8 rows = 64.
    # pixels[:, 1:] = columns 1..8, pixels[:, :-1] = columns 0..7.
    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


def mask_iou(mask_a, mask_b):
    """Overlap of two 0/1 masks of the same size: shared area / combined area."""
    both = np.logical_and(mask_a, mask_b).sum()      # pixels that are polyp in both masks
    either = np.logical_or(mask_a, mask_b).sum()     # pixels that are polyp in at least one
    return both / either if either > 0 else 0.0      # IoU: 1 = same place, 0 = no overlap


def close_pairs(names, hashes, max_distance):
    """All pairs (name_a, name_b, distance) with distance <= max_distance."""
    pairs = []
    # Compare every image with every later image: 1,000 x 999 / 2 = about 500,000 pairs.
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            # Hamming distance: in how many of the 64 places the fingerprints differ.
            distance = int((hashes[i] != hashes[j]).sum())
            if distance <= max_distance:
                pairs.append((names[i], names[j], distance))
    return pairs


def find_duplicates(names, hashes, small_masks):
    """Pairs that pass both tests. small_masks: name -> resized 0/1 mask."""
    duplicates = []
    for a, b, distance in close_pairs(names, hashes, MAX_DISTANCE):   # test 1: similar picture
        iou = mask_iou(small_masks[a], small_masks[b])                  # test 2: polyp in same place
        if iou >= MIN_MASK_IOU:
            duplicates.append((a, b, distance, round(float(iou), 2)))
    return duplicates
