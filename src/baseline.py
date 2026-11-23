"""Classical baseline: find the polyp by brightness, no learning.

Why brightness: on the validation images, polyps are clearly BRIGHTER than the bowel
wall around them (Lab lightness L higher in 90 % of images, by 37 on a 0-255 scale), but
hardly redder (a* only 3.8 higher). Our first idea, a redness (a*) threshold, scored
Dice 0.02 on validation, so we switched to L. Both decided on validation, never on test.

Steps for one image:
1. Convert RGB to Lab colour and take L, the lightness (0 = black, 255 = white).
2. Ignore the black border and the black corner box (very dark pixels).
3. Otsu's method picks the threshold that best splits the L values into two groups;
   the brighter group is our polyp guess.
4. Clean up: "opening" removes small specks, "closing" fills small holes.
5. Keep only the largest connected region (the polyp is usually one blob).

The size of the clean-up (radius) is chosen on the validation images only.
"""
import numpy as np
from PIL import Image, ImageCms
from scipy import ndimage

_TO_LAB = ImageCms.buildTransformFromOpenProfiles(
    ImageCms.createProfile("sRGB"), ImageCms.createProfile("LAB"), "RGB", "LAB")


def lightness(image):
    """The L channel of Lab colour, 0-255."""
    lab = ImageCms.applyTransform(Image.fromarray(image), _TO_LAB)
    return np.array(lab)[:, :, 0].astype(float)


def otsu_threshold(values):
    """The threshold that makes the two groups (below / above) as different as possible."""
    hist, edges = np.histogram(values, bins=256, range=(0, 256))
    centres = (edges[:-1] + edges[1:]) / 2
    best_t, best_score = 0, -1
    for i in range(1, 256):
        w0, w1 = hist[:i].sum(), hist[i:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0 = (hist[:i] * centres[:i]).sum() / w0
        m1 = (hist[i:] * centres[i:]).sum() / w1
        score = w0 * w1 * (m0 - m1) ** 2          # between-group variance
        if score > best_score:
            best_t, best_score = centres[i], score
    return best_t


def disk(radius):
    """A round shape for opening/closing."""
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return x ** 2 + y ** 2 <= radius ** 2


def predict(image, radius=11):
    """0/1 polyp mask for one RGB image (numpy array)."""
    light = lightness(image)
    inside = image.max(axis=2) > 30                 # not the black border / box
    mask = (light > otsu_threshold(light[inside])) & inside
    if radius > 0:
        mask = ndimage.binary_opening(mask, structure=disk(radius))
        mask = ndimage.binary_closing(mask, structure=disk(radius))
    labels, n = ndimage.label(mask)
    if n > 1:                                       # keep the largest region
        sizes = ndimage.sum(mask, labels, range(1, n + 1))
        mask = labels == (np.argmax(sizes) + 1)
    return mask.astype(np.uint8)
