"""Download Kvasir-SEG and the authors' split files into data/.

The images are NOT in this repository: the licence allows research and education
only, and the dataset must be cited (Jha et al., MMM 2020). This script gets them
from the official Simula page instead.

Run from the project folder:  python scripts/download_data.py
"""
import os
import urllib.request
import zipfile

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")

ZIP_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"  # 46 MB
SPLIT_URL = "https://raw.githubusercontent.com/DebeshJha/Kvasir-SEG/master/Data-split/"

os.makedirs(DATA, exist_ok=True)

zip_path = os.path.join(DATA, "kvasir-seg.zip")
if not os.path.exists(zip_path):
    print("Downloading", ZIP_URL)
    urllib.request.urlretrieve(ZIP_URL, zip_path)

if not os.path.exists(os.path.join(DATA, "Kvasir-SEG", "images")):
    print("Unzipping ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA)

# The authors' split: train.txt (880 names) and val.txt (120 names, our test set).
for name in ["train.txt", "val.txt"]:
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        urllib.request.urlretrieve(SPLIT_URL + name, path)

n_images = len(os.listdir(os.path.join(DATA, "Kvasir-SEG", "images")))
n_masks = len(os.listdir(os.path.join(DATA, "Kvasir-SEG", "masks")))
print(f"Done: {n_images} images, {n_masks} masks in data/Kvasir-SEG/")
