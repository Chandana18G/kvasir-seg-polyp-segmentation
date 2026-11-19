| | |
|---|---|
| Question | Can a U-Net outline polyps in colonoscopy images well enough to help a doctor not miss one, and how much better is it than a simple colour threshold? |
| Source, licence | Kvasir-SEG, https://datasets.simula.no/kvasir-seg/ (Jha et al., MMM 2020); research and education only, cite the paper; images not in our repository |
| Modality, size | Endoscopy (colonoscopy), 2D RGB frames; 999 images (+ 1 dropped as a copy of a test image), 332-1920 px wide, used at 256 × 256 pixels; pixel size not reported |
| Labels | one mask per image: polyp / background; who decided them: an engineer and a medical doctor (Labelbox), checked by one gastroenterologist; agreement not reported |
| Classes | polyp pixels: mean 15% of a training image (median 11%); every image has a polyp; floor: always background Dice 0.000, always polyp Dice 0.243 |
| Splits | 779 / 100 / 120 (training / validation / test); test = the authors' 120; not by patient |
| Quality checks | 0 of 120 test images have a near-twin in training (template check); patient not known from the files: cannot check; near-duplicate check found 7 pairs, 1 training copy of a test image removed (scripts/make_split.py) |
| Where my data could differ | site: one hospital group in Norway (Vestre Viken) · device: Olympus endoscopes · population: Norwegian routine colonoscopies · who labelled: one team, one reviewer |
| A possible shortcut | black corner box, date/time text, black border; 13 test images (0 training) still show the coloured position box; 23 test images come from a later batch |
