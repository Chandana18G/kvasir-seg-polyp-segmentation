# Polyp outlines in colonoscopy images: U-Net vs. a colour threshold

**Biomedical Image Analysis · M.Sc. Applied Data Science and AI · SRH University Munich · WS 2026/27**
**Authors:** Chandana Gurusiddappa, Dinesh [surname] · **Last updated:** 19.11.2026

Data: Kvasir-SEG (Jha et al., MMM 2020). The images are **not** in this repository: their
licence allows research and education only. `scripts/download_data.py` fetches them.

---

## 1 · My question in one sentence  *(Session 12 · report section 1)*

Can a U-Net outline polyps in colonoscopy images well enough to help a doctor not miss one,
and how much better is it than a simple colour threshold?

## 2 · Five lines  *(Session 6 · report sections 1–3)*

| | |
|---|---|
| **Question** | Can a U-Net segment polyps in single colonoscopy frames better than a colour threshold? |
| **Images** | Kvasir-SEG: 1,000 RGB colonoscopy frames (JPEG, 332–1920 px wide), Vestre Viken Health Trust, Norway. https://datasets.simula.no/kvasir-seg/ (46 MB, unchanged since 2019, downloaded 19.11.2026) |
| **Labels** | One binary mask per image (white = polyp). Every image contains a polyp. Polyps cover on average 15.4 % of the image (median 11.4 %, from 0.5 % to 81 %); 200 images have a polyp under 5 % of the image. |
| **Metric** | Dice per image, averaged (main), plus IoU, precision and recall, threshold 0.5; mean ± std over 3 seeds. Not pixel accuracy: about 85 % of pixels are background. |
| **Baseline** | Floor: "always background" = Dice 0.000; "always polyp" = Dice 0.243 ± 0.184 on test. Classical: Lab a* (redness) channel + Otsu threshold + clean-up + largest region *(after 19.11.)*. |

## 3 · Flat images or stacks?  *(Session 10 · report section 2)*

- **2D or 3D:** 2D. One RGB video frame per file, no slices or volumes.
- **Pixel size:** not reported. Endoscopy has no fixed scale: it changes with the distance between the camera and the bowel wall. Polyp size is therefore given as a share of the image.
- **Image size:** 333 different sizes, from 332 × 352 to 1920 × 1072 px; the most common is 622 × 530. All methods use copies shrunk to 256 × 256 (masks with nearest-neighbour, so they stay 0/1).

## 4 · Who made the labels?  *(Session 11 · report section 2)*

1. **Who:** an engineer and a medical doctor outlined each polyp in Labelbox; an experienced gastroenterologist then checked the outlines (Jha et al., 2020).
2. **How many / agreement:** one final mask per image; the separate outlines are not released. Agreement between annotators: **not reported**.
3. **Ceiling:** unknown, because no second annotation exists. The masks are filled polygons with smooth edges, so a Dice close to 1 is not meaningful. The masks are stored as JPEG: 0.73 % of mask pixels are not exactly black or white, so we call every pixel above 127 "polyp".

## 5 · Leakage check  *(Session 7 · report sections 2 and 5)*

1. **Several images per patient? Split by patient?** Probably yes, and no. The file names are random IDs and there are no patient IDs, so we cannot split by patient. We found one pair that shows the same polyp on the same date from another angle; it sits in train and validation. Our numbers may therefore be a little optimistic.
2. **Copies?** No byte-identical files. Our check (fingerprint "dHash" distance ≤ 3 of 64 **and** mask IoU ≥ 0.5, limits chosen by looking at the 25 closest pairs) found **7 near-duplicate pairs** (3 identical photos, 4 almost the same frame). One pair crossed into the test set; we dropped its training copy. No mirrored copies. Details: `notebooks/02_look_and_leakage.ipynb`, `results/duplicates.csv`.
3. **Statistics and augmentation after the split, training only?** Yes. We split first. Normalisation uses ImageNet values or training-set statistics only; flips, rotations and brightness changes are applied to training images only.
4. **Epochs and thresholds on validation, never test?** Yes. Best epoch by validation Dice; threshold fixed at 0.5 in advance; the classical threshold is tuned on validation. The test set is used once, at the end.
5. **Could something in the image give the label away?** Every image has a polyp, so nothing can give away "polyp yes/no". But: a black octagon border, a black box in the bottom-left corner (the position map was blacked out), text with date and time, and sometimes instruments. **13 test images still show the coloured position box, and none of the training images do.** We report Dice on the polyp only, not pixel accuracy.

**The split** (`scripts/make_split.py`, seed 42): **train 779 / validation 100 / test 120**, 1 dropped.
Test = the authors' own list of 120 (`Data-split/val.txt` on their GitHub), so our test numbers can be compared with published ones. Validation = 100 random training images that have no near-copy. **23 test images come from a later batch** (file names `cjyz…`, `ck2…`) with no images in train or validation; we will also report their Dice separately.

## 6 · Progress: one turn of the cycle  *(Session 12)*

- [x] 0 · Question: one sentence (block 1)
- [ ] 1 · Load: my data downloads in Colab *(works on our laptops; Colab check after the first push)*
- [x] 2 · Look: 20 images with their labels, looked at by eye (`results/fig_20_examples.png`)
- [x] 3 · Count: share of foreground pixels; the floor (`notebooks/01_data_and_floor.ipynb`)
- [x] 4 · Check: twins, the same patient twice (block 5)
- [x] 5 · Table: the data table in report section 2 is filled in
- [ ] 6 · Number: a first network next to the floor and the classical baseline *(after 19.11.)*

## 7 · How to run it  *(report section 4)*

- **Environment:** Google Colab (free GPU) or our own laptops; Python 3, `pip install -r requirements.txt`.
- **Order:**
  1. `python scripts/download_data.py`: data into `data/` (46 MB).
  2. `python scripts/make_split.py`: `splits/train.txt`, `val.txt`, `test.txt` and `results/duplicates.csv` (about a minute).
  3. `notebooks/01_data_and_floor.ipynb`: checks, polyp share, floor, 256 × 256 cache.
  4. `notebooks/02_look_and_leakage.ipynb`: 20 examples, copies, split check, test-set check.
- **Which file makes which table or figure:**

| Output | Made by |
|---|---|
| `results/fig_polyp_share.png`, floor numbers | `notebooks/01_data_and_floor.ipynb` |
| `results/fig_20_examples.png`, `results/fig_closest_pairs.png` | `notebooks/02_look_and_leakage.ipynb` |
| `splits/*.txt`, `results/duplicates.csv` | `scripts/make_split.py` |

## 8 · Shared with my Deep Learning project?

No. Our Deep Learning project (same two people) reproduces LSTNet on time series. No data, paper or code is shared. If we reuse a small helper (for example a seed function), we will name it here and in both reports.

## Who did what

*(Our plan; we will update it at the end.)*

| | Chandana | Dinesh |
|---|---|---|
| Data | download script, loading, mask check, polyp share, floor, report §2 | 20-image figure, duplicate check, split, README block 5 |
| Methods | classical baseline (colour threshold) | U-Net from scratch vs. ImageNet-pretrained, augmentation, 3 seeds |
| Evaluation | metrics, final test run, worst cases | learning curves, Grad-CAM, uncertainty maps |
| Report (lead) | §1, §2, §5 | §3, §4, §6 |

## Citation

Jha D., Smedsrud P. H., Riegler M. A., Halvorsen P., de Lange T., Johansen D., Johansen H. D.
*Kvasir-SEG: A Segmented Polyp Dataset.* MultiMedia Modeling (MMM 2020), LNCS 11962,
pp. 451–462. https://doi.org/10.1007/978-3-030-37734-2_37
