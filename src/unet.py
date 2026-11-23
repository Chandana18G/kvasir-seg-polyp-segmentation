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
    return nn.Sequential(
        nn.Conv2d(n_in, n_out, 3, padding=1), nn.BatchNorm2d(n_out), nn.ReLU(inplace=True),
        nn.Conv2d(n_out, n_out, 3, padding=1), nn.BatchNorm2d(n_out), nn.ReLU(inplace=True),
    )


class Up(nn.Module):
    """Double the size, glue on the skip features, then two convolutions."""

    def __init__(self, n_in, n_skip, n_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = conv_block(n_in + n_skip, n_out)

    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], dim=1))


class UNet(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        r = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)   # 1/2 size,  64 channels
        self.pool = r.maxpool                                # 1/4
        self.layer1, self.layer2 = r.layer1, r.layer2        # 1/4 (64), 1/8 (128)
        self.layer3, self.layer4 = r.layer3, r.layer4        # 1/16 (256), 1/32 (512)
        self.up4 = Up(512, 256, 256)
        self.up3 = Up(256, 128, 128)
        self.up2 = Up(128, 64, 64)
        self.up1 = Up(64, 64, 64)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv_block(64, 32),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.layer1(self.pool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        x = self.up4(s4, s3)
        x = self.up3(x, s2)
        x = self.up2(x, s1)
        x = self.up1(x, s0)
        return self.final(x)            # (batch, 1, H, W) logits
