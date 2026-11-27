"""U-Net with a ResNet-34 encoder.

Encoder ("going down"): ResNet-34 shrinks the image step by step (256 -> 128 -> 64 ->
32 -> 16 -> 8 pixels) and learns more and more abstract features.
Decoder ("going up"): we enlarge again step by step; at each step we glue on
("skip connection") the encoder features of the same size, so fine details come back.
Last layer: one number per pixel (a "logit"); sigmoid(logit) > 0.5 means polyp.

pretrained=True  -> the encoder starts from ResNet-34 trained on ImageNet photos.
pretrained=False -> the encoder starts from random numbers ("from scratch").
This is the ONE change between our two models.
"""
import torch
from torch import nn
from torchvision.models import resnet34, ResNet34_Weights


def conv_block(n_in, n_out):
    """Two 3x3 convolutions, each followed by batch norm and ReLU."""
    # Conv2d(…, 3, padding=1): a 3x3 filter; padding keeps the image size the same.
    # BatchNorm2d: rescales the numbers so training is stable.
    # ReLU: sets negative numbers to 0 (the non-linearity that lets the network learn shapes).
    return nn.Sequential(
        nn.Conv2d(n_in, n_out, 3, padding=1), nn.BatchNorm2d(n_out), nn.ReLU(inplace=True),
        nn.Conv2d(n_out, n_out, 3, padding=1), nn.BatchNorm2d(n_out), nn.ReLU(inplace=True),
    )


class Up(nn.Module):
    """Double the size, glue on the skip features, then two convolutions."""

    def __init__(self, n_in, n_skip, n_out):
        super().__init__()
        # Upsample x2 with bilinear interpolation (no learned weights here).
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # After gluing, the channels are n_in + n_skip; the convolutions mix them down to n_out.
        self.conv = conv_block(n_in + n_skip, n_out)

    def forward(self, x, skip):
        # torch.cat(..., dim=1): glue along the channel axis = the skip connection.
        return self.conv(torch.cat([self.up(x), skip], dim=1))


class UNet(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        # Take a ResNet-34: with ImageNet weights (pretrained) or random weights (scratch).
        r = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        # Encoder = the ResNet's layers WITHOUT its classifier at the end (we need pixels, not a class).
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)   # 1/2 size,  64 channels
        self.pool = r.maxpool                                # 1/4
        self.layer1, self.layer2 = r.layer1, r.layer2        # 1/4 (64), 1/8 (128)
        self.layer3, self.layer4 = r.layer3, r.layer4        # 1/16 (256), 1/32 (512)
        # Decoder: Up(channels coming in, channels of the skip, channels going out).
        self.up4 = Up(512, 256, 256)     # 8 px  -> 16 px, glue layer3
        self.up3 = Up(256, 128, 128)     # 16 px -> 32 px, glue layer2
        self.up2 = Up(128, 64, 64)       # 32 px -> 64 px, glue layer1
        self.up1 = Up(64, 64, 64)        # 64 px -> 128 px, glue the stem
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),   # 128 -> 256 px
            conv_block(64, 32),
            nn.Conv2d(32, 1, 1),         # 1x1 convolution: 32 numbers per pixel -> 1 logit per pixel
        )

    def forward(self, x):
        # Encoder: keep every intermediate result (s0..s3) for the skip connections.
        s0 = self.stem(x)                # 128 x 128
        s1 = self.layer1(self.pool(s0))  # 64 x 64
        s2 = self.layer2(s1)             # 32 x 32
        s3 = self.layer3(s2)             # 16 x 16
        s4 = self.layer4(s3)             # 8 x 8, the most abstract features (Grad-CAM uses this layer)
        # Decoder: go back up, gluing on the encoder result of the same size each time.
        x = self.up4(s4, s3)
        x = self.up3(x, s2)
        x = self.up2(x, s1)
        x = self.up1(x, s0)
        return self.final(x)            # (batch, 1, H, W) logits
