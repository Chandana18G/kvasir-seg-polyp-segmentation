# A5 · Data table: Kvasir-SEG

Chandana Gurusiddappa, Dineshkumar Swaminathan · 19.11.2026
Numbers from `notebooks/01_data_and_floor.ipynb`, `notebooks/02_look_and_leakage.ipynb` and `scripts/make_split.py`.

| | |
|---|---|
| Question | Can a U-Net outline polyps in colonoscopy images well enough to help a doctor not miss one, and how much better is it than a colour threshold? |
| Source, version, download date | Kvasir-SEG, https://datasets.simula.no/kvasir-seg/ (`kvasir-seg.zip`, 46 MB, unchanged since 2019), downloaded 19.11.2026. Jha et al., MMM 2020, DOI 10.1007/978-3-030-37734-2_37 |
| Licence | Research and education only, no commercial use, cite the paper. Images not in this repo (`scripts/download_data.py`) |
| Modality | Endoscopy (colonoscopy), 2D RGB video frames |
| Size | 1,000 images, 1,000 masks; 333 sizes, 332 × 352 to 1920 × 1072 px; resized to 256 × 256. Patients: not reported. Pixel size: not reported |
| Labels: who | Engineer + medical doctor (Labelbox), checked by one gastroenterologist |
| Labels: agreement | Not reported → ceiling unknown |
| Task / target | Segmentation, polyp vs. background per pixel; every image has a polyp |
| Foreground share | Mean 15.4 %, median 11.4 %, min 0.5 %, max 81 %; 200 images < 5 % |
| Floor (test, 120 images) | Always background: Dice 0.000 · always polyp: Dice 0.243 ± 0.184 |
| Split | Train 779 / validation 100 / test 120 (authors' test list), seed 42. **Not by patient**: no patient IDs |
| Leakage | 0 identical files; 7 near-duplicate pairs (dHash ≤ 3 + mask IoU ≥ 0.5); 1 training copy of a test image removed; 0 mirrored copies; 1 same polyp in train and validation (by eye) |
| Clues in the image | Black border, black corner box, date/time text, instruments. 13 test images (0 training) still show the coloured position box |
| Test vs. train | 23 test images from a later batch with no training images → reported separately |
| Metric | Dice per image (main), IoU, precision, recall; 3 seeds, mean ± std |
| Baselines next | Colour threshold (Lab a* + Otsu); U-Net scratch vs. ImageNet-pretrained |
