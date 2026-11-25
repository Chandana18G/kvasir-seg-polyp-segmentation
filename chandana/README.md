# Chandana Gurusiddappa: data, models and evaluation

Run in this order (each notebook clones the repository and downloads the data itself on Colab):

| Notebook | What it does | Results |
|---|---|---|
| `01_data_and_floor.ipynb` | checks the masks, polyp share, the floor, 256 × 256 cache | `results/fig_polyp_share.png` |
| `02_look_and_leakage.ipynb` | 20 images by eye, near-duplicates, split check, how the test set differs | `results/fig_20_examples.png`, `fig_closest_pairs.png` |
| `BIA_W6_Project_Template_PartA.ipynb` | the course template, Part A (A1–A5) on our data | `results/A1_*`, `A2_*`, `A5_*` |
| `03_models_and_evaluation.ipynb` | **Colab, T4 GPU**: brightness baseline, U-Net from scratch vs. ImageNet-pretrained (3 seeds each), test set once, worst cases, Grad-CAM, uncertainty | `results/B1_*`, `B2_*`, `C1_*` … `C5_*` |
| `04_template_extras.ipynb` | Colab, no GPU: loads the saved U-Nets; template B5 heat-map sanity check, C3 calibration, exact versions | `results/B5_*`, `C3_calibration.*`, `requirements.txt` |

The shared code these notebooks use is in `src/` and `scripts/` (also written by Chandana);
`python src/stats.py` makes the paired statistical tests (`results/C6_stats.md`).
