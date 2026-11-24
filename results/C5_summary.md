Kvasir-SEG, 120 test images (the authors' test list), used once.
| Setting | Dice | IoU | Precision | Recall |
|---|---|---|---|---|
| floor: always polyp | 0.243 | 0.153 | 0.153 | 1.000 |
| floor: always background | 0.000 | 0.000 | 0.000 | 0.000 |
| brightness threshold (radius 11) | 0.323 | 0.223 | 0.304 | 0.598 |
| U-Net scratch (3 seeds) | 0.756 ± 0.009 | 0.661 ± 0.014 | 0.790 ± 0.025 | 0.807 ± 0.021 |
| U-Net pretrained (3 seeds) | 0.862 ± 0.007 | 0.788 ± 0.011 | 0.864 ± 0.029 | 0.912 ± 0.016 |

Time per epoch: 18.3 s on cuda; 30 epochs; seeds [0, 1, 2]; best epoch by validation Dice.
Classical baseline: brightness (Lab L) + Otsu + clean-up radius 11 (chosen on validation).
