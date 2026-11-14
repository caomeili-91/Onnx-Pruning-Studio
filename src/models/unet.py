"""轻量 UNet 示例：用于验证通用剪枝引擎，不提供预训练权重。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """标准 U-Net（4 次下采样 + 4 次上采样）。"""

    def __init__(self, num_classes: int = 21, in_channels: int = 3, base: int = 32) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.bottleneck = DoubleConv(base * 8, base * 16)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.out_conv = nn.Conv2d(base, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        x = self.bottleneck(self.pool(s4))

        x = self.dec4(torch.cat([_match_size(self.up4(x), s4), s4], dim=1))
        x = self.dec3(torch.cat([_match_size(self.up3(x), s3), s3], dim=1))
        x = self.dec2(torch.cat([_match_size(self.up2(x), s2), s2], dim=1))
        x = self.dec1(torch.cat([_match_size(self.up1(x), s1), s1], dim=1))
        return self.out_conv(x)


def _match_size(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] != reference.shape[-2:]:
        x = F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)
    return x
