"""Load Kvasir-SEG images and masks.

Every image has a mask with the same file name. A mask is white (255) where the
polyp is and black (0) elsewhere. The masks are saved as JPEG, so a few pixels near
the edge are not exactly 0 or 255; we call every pixel above 127 "polyp".
"""
import os                      # file paths and folder listings

import numpy as np             # images become numpy arrays (numbers we can compute with)
from PIL import Image          # Pillow: opens .jpg files, converts colour, resizes

# Where the data lives: <project>/data/Kvasir-SEG (made by scripts/download_data.py).
# os.path.dirname(__file__) = the folder of this file (src/), ".." = one level up.
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "Kvasir-SEG")
# Where the split lists live: <project>/splits (made by scripts/make_split.py).
SPLIT_DIR = os.path.join(os.path.dirname(__file__), "..", "splits")


def list_names():
    """All 1,000 file names without '.jpg', sorted."""
    files = sorted(os.listdir(os.path.join(DATA_DIR, "images")))   # sorted = same order every time
    return [f[:-4] for f in files if f.endswith(".jpg")]            # f[:-4] cuts off ".jpg"


def load_image(name):
    """The colonoscopy photo as a (height, width, 3) uint8 array."""
    path = os.path.join(DATA_DIR, "images", name + ".jpg")
    # convert("RGB") makes sure we always get 3 colour channels (red, green, blue), 0-255 each.
    return np.array(Image.open(path).convert("RGB"))


def load_mask(name):
    """The mask as a (height, width) array of 0 (background) and 1 (polyp)."""
    path = os.path.join(DATA_DIR, "masks", name + ".jpg")
    grey = np.array(Image.open(path).convert("L"))     # "L" = one grey channel, 0-255
    # JPEG compression blurs the edge: 0.73 % of mask pixels are grey (not 0, not 255).
    # Threshold in the middle: above 127 = polyp (1), else background (0).
    return (grey > 127).astype(np.uint8)


def load_resized(name, size=256):
    """Image and mask shrunk to size x size.

    The mask uses NEAREST so it stays 0/1 (no in-between values).
    """
    # Images come in 333 different sizes; the network needs one fixed size (256 x 256).
    # BILINEAR = smooth resizing for the photo (mixes neighbouring pixels).
    image = Image.fromarray(load_image(name)).resize((size, size), Image.BILINEAR)
    # NEAREST = copy the closest pixel, so the mask keeps only 0 and 1 (no 0.5 values).
    mask = Image.fromarray(load_mask(name)).resize((size, size), Image.NEAREST)
    return np.array(image), np.array(mask)


def read_split(split):
    """File names in one split: 'train', 'val' or 'test'."""
    path = os.path.join(SPLIT_DIR, split + ".txt")     # e.g. splits/train.txt
    with open(path) as f:
        # one name per line; strip() removes the line break; skip empty lines
        return [line.strip() for line in f if line.strip()]
