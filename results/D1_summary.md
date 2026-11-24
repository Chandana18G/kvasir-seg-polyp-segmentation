# D1 · Which test images are hard? (Dinesh Swaminathan)

Pretrained U-Net, 120 test images.
- Dice vs polyp size: Spearman r = +0.18 (p = 0.055)
- Dice vs brightness contrast: Spearman r = +0.06 (p = 0.535)
- Later batch: Dice 0.809 (n = 23) vs rest 0.874
- Hard images (Dice < 0.5): 7; median polyp size 25.3% (all: 11.7%); median contrast +48.6 (all: +30.9); 3 from the later batch
