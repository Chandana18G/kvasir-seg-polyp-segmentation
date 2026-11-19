# [Title: the clinical question and how you answered it]

**Biomedical Image Analysis · M.Sc. Applied Data Science and AI · SRH University Munich · Winter semester 2026/27**
**Author(s):** Chandana Gurusiddappa, Dinesh [surname] · **Repository:** [link to your GitHub repository] · **Date:** [date]

> How to use this template: copy it into your repository as `report.md` (or into LaTeX or Word with the same sections).
> Replace everything in [square brackets]. The section titles follow the eight grading criteria of Session 1 (weights in brackets).
> The remaining criterion, code understanding and technical defense (15 %), is graded in the exam conversation, not in the report.
> Same structure as the Deep Learning report, plus the biomedical parts: modality, labels, patient-level split, clinical meaning.
> Shared data or paper with Deep Learning? Allowed (decided 15.10.): different questions; data preparation is graded here only; name the shared work in both reports.

---

## Abstract

[Four or five sentences: the clinical question, the data, your classical baseline, your deep-learning model, the result with its spread, what it means and what it does not.]

## 1 · The biomedical problem (15 %, with section 2)

- **Clinical question:** [who would use the result, for what decision; for example "count nuclei to see whether a drug kills cells"]
- **Task type:** [classification, segmentation, detection, regression; what goes in, what comes out]
- **Why it is hard:** [small lesions, class imbalance, variation between scanners or sites, unclear borders …]

## 2 · The data (15 %, with section 1)

| | |
|---|---|
| Source, version, download date | Kvasir-SEG, https://datasets.simula.no/kvasir-seg/ (`kvasir-seg.zip`, 46 MB, unchanged since 18.11.2019), downloaded 19.11.2026. Paper: Jha et al., *Kvasir-SEG: A Segmented Polyp Dataset*, MMM 2020, LNCS 11962, pp. 451–462. Split lists from the authors' GitHub (`DebeshJha/Kvasir-SEG`, `Data-split/`). |
| Licence | No named licence. Research and educational use only, no commercial use without written permission, the paper must be cited. So the images are not in our repository; `scripts/download_data.py` fetches them. |
| Modality | Endoscopy (colonoscopy, Olympus endoscopes, Vestre Viken Health Trust, Norway); 2D RGB video frames. |
| Size | 1,000 images, each with one polyp mask. Number of patients: not reported (no patient IDs). 333 different image sizes, 332 × 352 to 1920 × 1072 px (most common 622 × 530). Pixel size in mm: not reported; endoscopy has no fixed scale. |
| Labels: who made them | An engineer and a medical doctor outlined the polyps in Labelbox; one experienced gastroenterologist checked them. |
| Labels: agreement | Not reported (one final mask per image; separate outlines not released). Ceiling therefore unknown. |
| Classes or targets | Two classes per pixel: polyp / background. Every image contains a polyp. Polyp share per image: mean 15.4 %, median 11.4 %, min 0.5 %, max 81 %; 200 of 1,000 images under 5 %. Floor on test: "always background" Dice 0.000; "always polyp" Dice 0.243 ± 0.184. |
| Splits | **Not by patient**: no patient IDs exist. Test = the authors' 120 images; validation = 100 random training images without near-copies (seed 42); train = 779; 1 image dropped (copy of a test image). |
| Preprocessing | Masks binarised with > 127 (0.73 % of mask pixels are grey from JPEG). All images and masks shrunk to 256 × 256 (images bilinear, masks nearest-neighbour). Normalisation: ImageNet values or training-set statistics only. |
| Quality checks | No broken files; every mask has the same size as its image; no empty mask. 20 images looked at by eye: black octagon border, black corner box (blacked-out position map), date/time text, reflections, sometimes instruments; masks look tight. Duplicates: 0 byte-identical, 7 near-duplicate pairs (fingerprint + mask overlap), 0 mirrored; one same-polyp pair across train/validation found by eye. **Test differs from train**: 13 test images (and no training image) still show the coloured position box; 23 test images come from a later batch with no training images. |
| Shared with Deep Learning? | No. The DL project (same partners) reproduces LSTNet on time series; all data preparation here is for this course only. |

![20 Kvasir-SEG images with the doctors' masks in green](results/fig_20_examples.png)

*Figure 1: 20 random images (seed 0) with their masks in green. Made by `notebooks/02_look_and_leakage.ipynb`.*

## 3 · Experimental design (15 %)

- **The floor:** [the trivial answer: largest class, or "always background" (Dice 0)]
- **Classical baseline:** [threshold, features + random forest, … (Sessions 5–6)]
- **Deep-learning model:** [architecture, pretrained or not, 2D/2.5D/3D, loss (Dice + cross-entropy?), why]
- **Augmentation:** [which changes, and why each one is plausible for this modality (Session 8)]
- **Comparisons:** [one change at a time; same splits and budget; at least three seeds]

## 4 · Implementation and reproducibility (25 %)

- **Run it:** [commands or notebook order; environment, for example `requirements.txt`]
- **Seeds, hardware, training time:** […]
- **Where the results come from:** [which notebook or script makes which table and figure]

## 5 · Evaluation, validation and critical analysis (15 %)

| Setting | [metric] mean | ± spread | seeds |
|---|---|---|---|
| floor | | | |
| classical baseline | | | |
| deep-learning model | | | |
| + [change] | | | |

- **Metrics that fit the task:** [Dice and IoU for masks; sensitivity and specificity, AUC for classes; volumes in mL …]
- **Worst cases:** [the images where the model fails most, and why (Session 9)]
- **Ceiling:** [what a second annotator would score, if known (Session 11)]
- **Test set:** used **once**, for the final models only. External data, if any: [which].
- **Limitations:** [what the results do **not** show: other scanners, other hospitals, other populations]

## 6 · Explainability, ethics and clinical responsibility (5 %)

- [Saliency maps or other explanations (Session 14), and what they can and cannot show]
- [Who could be harmed by a wrong answer; what a doctor would still have to check; privacy of the data]

## 7 · Use of AI tools (5 %)

[Which tools you used, for what (code, text, debugging), and how you **checked** their output. Be specific.]

## 8 · Conclusion (documentation and communication, 5 %)

[What you found, how sure you are, what you would do next.]

## References

[All papers, data sets, code and websites you used.]
