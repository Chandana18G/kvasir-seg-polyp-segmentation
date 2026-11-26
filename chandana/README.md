# Chandana Gurusiddappa: models and results

The main part: does a U-Net outline polyps better than a brightness threshold, and does
ImageNet pretraining help? Uses the split and data checks from [`dinesh/`](../dinesh/).

| Notebook | What it does | Results |
|---|---|---|
| `03_models_and_evaluation.ipynb` | **Colab, T4 GPU** (about 55 min; finished runs are saved to Google Drive): brightness baseline, U-Net from scratch vs. ImageNet-pretrained (3 seeds each), test set used once, worst cases, Grad-CAM, uncertainty | `results/B1_*`, `B2_*`, `C1_*` … `C5_*` |
| `04_template_extras.ipynb` | Colab, no GPU: loads the saved U-Nets; template B5 heat-map sanity check, C3 calibration, exact package versions | `results/B5_*`, `C3_calibration.*`, `requirements.txt` |
| `05_hit_rate.ipynb` | Colab, T4 GPU (about 5 min): loads the saved U-Nets; "not miss one": in how many images and how many polyps does the U-Net hit the polyp (any overlap, IoU ≥ 0.5), the missed polyps one by one, false alarms; and the learning-curve gap (Dice on training vs. validation vs. test) | `results/C7_hits.md`, `C7_missed_polyps.png` |

Code: `src/baseline.py`, `src/unet.py`, `src/train.py`, `src/metrics.py`, `src/stats.py`, `src/hits.py`
(`python src/stats.py` → paired tests, `results/C6_stats.md`).
