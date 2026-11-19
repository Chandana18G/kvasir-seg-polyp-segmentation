"""Load Kvasir-SEG images and masks.

Every image has a mask with the same file name. A mask is white (255) where the
polyp is and black (0) elsewhere. The masks are saved as JPEG, so a few pixels near
the edge are not exactly 0 or 255; we call every pixel above 127 "polyp".
"""
import os

import numpy as np
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "Kvasir-SEG")
SPLIT_DIR = os.path.join(os.path.dirname(__file__), "..", "splits")


def list_names():
    """All 1,000 file names without '.jpg', sorted."""
    files = sorted(os.listdir(os.path.join(DATA_DIR, "images")))
    return [f[:-4] for f in files if f.endswith(".jpg")]


def load_image(name):
    """The colonoscopy photo as a (height, width, 3) uint8 array."""
    path = os.path.join(DATA_DIR, "images", name + ".jpg")
    return np.array(Image.open(path).convert("RGB"))


def load_mask(name):
    """The mask as a (height, width) array of 0 (background) and 1 (polyp)."""
    path = os.path.join(DATA_DIR, "masks", name + ".jpg")
    grey = np.array(Image.open(path).convert("L"))
    return (grey > 127).astype(np.uint8)


def load_resized(name, size=256):
    """Image and mask shrunk to size x size.

    The mask uses NEAREST so it stays 0/1 (no in-between values).
    """
    image = Image.fromarray(load_image(name)).resize((size, size), Image.BILINEAR)
    mask = Image.fromarray(load_mask(name)).resize((size, size), Image.NEAREST)
    return np.array(image), np.array(mask)


def read_split(split):
    """File names in one split: 'train', 'val' or 'test'."""
    path = os.path.join(SPLIT_DIR, split + ".txt")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]
