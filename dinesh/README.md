# Dinesh Swaminathan: which test images are hard?

`hard_images.ipynb` looks at the 120 test images and asks whether the images the pretrained
U-Net gets wrong have something in common: polyp size, how much brighter the polyp is than
the bowel wall, or coming from the later batch. No training, no GPU, about 1 minute.

Run: open in Colab (or Jupyter) and Run all. It needs `results/C1_dice_per_image.json`
from `notebooks/03_models_and_evaluation.ipynb`.

Output: `results/D1_hard_images.png`, `results/D1_summary.md`.
