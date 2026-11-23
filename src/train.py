"""Train the U-Net and predict with it.

Settings (the same for both models, so pretraining is the only change):
- input 256 x 256, ImageNet mean/std normalisation
- loss = binary cross-entropy + Dice loss
- Adam, learning rate 1e-4, batch 16
- augmentation on training images only: flips, 90-degree turns, brightness +-10 %
- after every epoch: Dice on the validation images; we keep the epoch with the best one
"""
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from metrics import mean_scores
from unet import UNet

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_tensor(images):
    """uint8 (n, H, W, 3) -> float (n, 3, H, W) in 0..1."""
    return torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) / 255


def normalise(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def augment(x, y):
    """Random flips, 90-degree turns and brightness, the same for image and mask."""
    x, y = x.clone(), y.clone()
    for i in range(len(x)):
        if torch.rand(1) < 0.5:
            x[i], y[i] = x[i].flip(-1), y[i].flip(-1)           # left-right
        if torch.rand(1) < 0.5:
            x[i], y[i] = x[i].flip(-2), y[i].flip(-2)           # up-down
        k = int(torch.randint(0, 4, (1,)))
        x[i], y[i] = x[i].rot90(k, (-2, -1)), y[i].rot90(k, (-2, -1))
        x[i] = (x[i] * (0.9 + 0.2 * torch.rand(1))).clamp(0, 1)  # brightness
    return x, y


def dice_loss(logits, target):
    p = torch.sigmoid(logits)
    overlap = (p * target).sum((1, 2, 3))
    total = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return (1 - (2 * overlap + 1) / (total + 1)).mean()


@torch.no_grad()
def predict_probs(model, images, device, batch_size=16):
    """Polyp probability per pixel, (n, H, W), for uint8 images."""
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        x = normalise(to_tensor(images[i:i + batch_size]).to(device))
        out.append(torch.sigmoid(model(x))[:, 0].cpu().numpy())
    return np.concatenate(out)


def train_model(pretrained, seed, train_x, train_y, val_x, val_y, device,
                epochs=30, batch_size=16, lr=1e-4):
    """Returns the model at its best validation epoch, and the history per epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = UNet(pretrained).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    images, masks = to_tensor(train_x), torch.tensor(train_y, dtype=torch.float32)[:, None]

    history, best_dice, best_state = [], -1.0, None
    for epoch in range(epochs):
        start = time.time()
        model.train()
        order = torch.randperm(len(images))
        losses = []
        for i in range(0, len(order), batch_size):
            batch = order[i:i + batch_size]
            x, y = augment(images[batch], masks[batch])
            x, y = normalise(x.to(device)), y.to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(loss.item())

        val_pred = predict_probs(model, val_x, device) > 0.5
        val_dice = mean_scores(val_pred, val_y)["dice"]
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "val_dice": val_dice, "seconds": time.time() - start})
        if val_dice > best_dice:
            best_dice, best_state = val_dice, copy.deepcopy(model.state_dict())
        print(f"epoch {epoch + 1:3d}  loss {np.mean(losses):.3f}  val Dice {val_dice:.3f}"
              f"  ({time.time() - start:.0f} s)")

    model.load_state_dict(best_state)
    return model, history
