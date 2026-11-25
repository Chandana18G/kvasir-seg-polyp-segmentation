# Dineshkumar Swaminathan: the data

Everything about the images before and after the models: is the data clean, how is it split,
what is the floor, and which test images are hard? Run in this order (on Colab each notebook
clones the repository and downloads the data itself):

| Notebook | What it does | Results |
|---|---|---|
| `01_data_and_floor.ipynb` | mask check (JPEG grey pixels), polyp share, the floor, 256 × 256 cache | `results/fig_polyp_share.png` |
| `02_look_and_leakage.ipynb` | 20 images by eye, near-duplicates (fingerprint + mask overlap), split check, how the test set differs | `results/fig_20_examples.png`, `fig_closest_pairs.png` |
| `BIA_W6_Project_Template_PartA.ipynb` | the course template, Part A (A1–A5) on our data | `results/A1_*`, `A2_*`, `A5_*` |
| `hard_images.ipynb` | after the models: which test images are hard for the U-Net (size, brightness contrast, later batch); no GPU, about 1 minute | `results/D1_*` |

Code: `scripts/download_data.py`, `scripts/make_split.py` (→ `splits/`, `results/duplicates.csv`),
`src/data.py`, `src/duplicates.py`.
