"""Make the train / validation / test split and save it to splits/.

- Test (120): the dataset authors' own list (data/val.txt from their GitHub), so our
  test numbers can be compared with published ones. We never change it.
- Near-duplicates (src/duplicates.py): if a training image is a copy of a test image,
  we drop the training copy, so the test set stays unseen.
- Validation (100): drawn at random (seed 42) from the remaining training images,
  only from images that have no near-copy, so no copy sits in both train and val.
- Train: everything else.

Run from the project folder:  python scripts/make_split.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import list_names, load_image, load_resized  # noqa: E402
from duplicates import dhash, find_duplicates  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
SEED = 42
N_VAL = 100


def read_list(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def write_list(path, names):
    with open(path, "w") as f:
        f.write("\n".join(sorted(names)) + "\n")


names = list_names()
authors_train = read_list(os.path.join(ROOT, "data", "train.txt"))
authors_test = read_list(os.path.join(ROOT, "data", "val.txt"))
assert sorted(authors_train + authors_test) == names, "split files do not match the images"

print("Fingerprinting 1,000 images (takes about a minute) ...")
hashes = [dhash(load_image(n)) for n in names]
small_masks = {n: load_resized(n, 160)[1] for n in names}
duplicates = find_duplicates(names, hashes, small_masks)

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
with open(os.path.join(ROOT, "results", "duplicates.csv"), "w") as f:
    f.write("image_a,image_b,hash_distance,mask_iou\n")
    for a, b, distance, iou in duplicates:
        f.write(f"{a},{b},{distance},{iou}\n")
print(f"Near-duplicate pairs: {len(duplicates)} (saved to results/duplicates.csv)")

# 1. Drop training copies of test images.
test = set(authors_test)
dropped = set()
for a, b, _, _ in duplicates:
    if a in test and b not in test:
        dropped.add(b)
    if b in test and a not in test:
        dropped.add(a)
train = [n for n in authors_train if n not in dropped]
print(f"Dropped from training because a copy is in test: {sorted(dropped)}")

# 2. Validation only from images that have no near-copy.
has_copy = set()
for a, b, _, _ in duplicates:
    has_copy.add(a)
    has_copy.add(b)
candidates = sorted(n for n in train if n not in has_copy)
random.seed(SEED)
val = random.sample(candidates, N_VAL)
train = [n for n in train if n not in val]

os.makedirs(os.path.join(ROOT, "splits"), exist_ok=True)
write_list(os.path.join(ROOT, "splits", "train.txt"), train)
write_list(os.path.join(ROOT, "splits", "val.txt"), val)
write_list(os.path.join(ROOT, "splits", "test.txt"), authors_test)
write_list(os.path.join(ROOT, "splits", "dropped.txt"), dropped)
print(f"train {len(train)}, val {len(val)}, test {len(authors_test)}, dropped {len(dropped)}")
