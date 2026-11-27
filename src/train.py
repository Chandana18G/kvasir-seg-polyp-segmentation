"""Train the U-Net and predict with it.

Settings (the same for both models, so pretraining is the only change):
- input 256 x 256, ImageNet mean/std normalisation
- loss = binary cross-entropy + Dice loss
- Adam, learning rate 1e-4, batch 16
- augmentation on training images only: flips, 90-degree turns, brightness +-10 %
- after every epoch: Dice on the validation images; we keep the epoch with the best one
"""
import copy                       # deepcopy: save a full copy of the best weights
import time                       # measure seconds per epoch

import numpy as np
import torch
import torch.nn.functional as F   # the loss function binary_cross_entropy_with_logits

from metrics import mean_scores   # Dice per image (src/metrics.py)
from unet import UNet             # our network (src/unet.py)

# ImageNet mean and standard deviation per colour channel (R, G, B).
# The pretrained encoder learned on images normalised like this, so we do the same.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_tensor(images):
    """uint8 (n, H, W, 3) -> float (n, 3, H, W) in 0..1."""
    # PyTorch wants the channel axis second: permute(0, 3, 1, 2). /255 scales 0-255 to 0-1.
    return torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) / 255


def normalise(x):
    # Subtract the mean, divide by the std (on the same device as x: CPU or GPU).
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def augment(x, y):
    """Random flips, 90-degree turns and brightness, the same for image and mask."""
    # Why: only 779 training images. Showing them flipped/turned makes new-looking examples,
    # so the network learns the polyp, not the exact pictures (less overfitting).
    # An endoscope can see a polyp from any side, so these changes are realistic.
    x, y = x.clone(), y.clone()                     # copy, so the original images stay unchanged
    for i in range(len(x)):
        if torch.rand(1) < 0.5:                                 # 50 % chance
            x[i], y[i] = x[i].flip(-1), y[i].flip(-1)           # left-right
        if torch.rand(1) < 0.5:
            x[i], y[i] = x[i].flip(-2), y[i].flip(-2)           # up-down
        k = int(torch.randint(0, 4, (1,)))                      # turn by 0, 90, 180 or 270 degrees
        x[i], y[i] = x[i].rot90(k, (-2, -1)), y[i].rot90(k, (-2, -1))
        # Brightness x 0.9 ... 1.1 (the light changes with distance); only the image, not the mask.
        x[i] = (x[i] * (0.9 + 0.2 * torch.rand(1))).clamp(0, 1)  # brightness
    return x, y


def dice_loss(logits, target):
    # Soft Dice: uses probabilities (not 0/1) so it can be differentiated for training.
    p = torch.sigmoid(logits)                        # logit -> probability 0..1
    overlap = (p * target).sum((1, 2, 3))            # per image: "soft" overlap
    total = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    # +1 on top and bottom ("smoothing") avoids dividing by 0. Loss = 1 - Dice, lower is better.
    return (1 - (2 * overlap + 1) / (total + 1)).mean()


@torch.no_grad()                  # no gradients needed when only predicting: faster, less memory
def predict_probs(model, images, device, batch_size=16):
    """Polyp probability per pixel, (n, H, W), for uint8 images."""
    model.eval()                  # evaluation mode: batch norm uses its stored statistics
    out = []
    for i in range(0, len(images), batch_size):     # 16 images at a time (fits in memory)
        x = normalise(to_tensor(images[i:i + batch_size]).to(device))
        # sigmoid turns logits into probabilities; [:, 0] drops the single channel axis
        out.append(torch.sigmoid(model(x))[:, 0].cpu().numpy())
    return np.concatenate(out)


def train_model(pretrained, seed, train_x, train_y, val_x, val_y, device,
                epochs=30, batch_size=16, lr=1e-4):
    """Returns the model at its best validation epoch, and the history per epoch."""
    torch.manual_seed(seed)       # the seed fixes the random start weights, batch order, augmentation
    np.random.seed(seed)
    model = UNet(pretrained).to(device)              # .to(device): move to the GPU if there is one
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)   # ALL weights train (fine-tuning, not frozen)
    images, masks = to_tensor(train_x), torch.tensor(train_y, dtype=torch.float32)[:, None]

    history, best_dice, best_state = [], -1.0, None
    for epoch in range(epochs):                      # one epoch = every training image seen once
        start = time.time()
        model.train()                                # training mode (batch norm updates its statistics)
        order = torch.randperm(len(images))          # new random order every epoch
        losses = []
        for i in range(0, len(order), batch_size):
            batch = order[i:i + batch_size]          # the next 16 images
            x, y = augment(images[batch], masks[batch])
            x, y = normalise(x.to(device)), y.to(device)
            logits = model(x)                        # forward pass: one logit per pixel
            # BCE: per-pixel error, smooth gradients. Dice loss: overlap, not swamped by the
            # 85 % background. Together they work better than either alone.
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            optimiser.zero_grad()                    # forget the gradients of the last batch
            loss.backward()                          # backpropagation: compute new gradients
            optimiser.step()                         # Adam changes the weights a little
            losses.append(loss.item())

        # After each epoch: Dice on the 100 validation images (never the test images).
        val_pred = predict_probs(model, val_x, device) > 0.5
        val_dice = mean_scores(val_pred, val_y)["dice"]
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "val_dice": val_dice, "seconds": time.time() - start})
        if val_dice > best_dice:                     # best so far: keep a copy of these weights
            best_dice, best_state = val_dice, copy.deepcopy(model.state_dict())
        print(f"epoch {epoch + 1:3d}  loss {np.mean(losses):.3f}  val Dice {val_dice:.3f}"
              f"  ({time.time() - start:.0f} s)")

    model.load_state_dict(best_state)                # early stopping: go back to the best epoch
    return model, history
