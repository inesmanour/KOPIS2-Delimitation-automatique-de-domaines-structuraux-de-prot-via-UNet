#!/usr/bin/env python3
"""
Définition l'architecture d'un petit U-Net 2D pour segmenter des cartes de contacts N×N.

Entrée  : [B, 1, N, N]  (matrice de probas de contact)
Sortie  : [B, 1, N, N]  (logits de probabilité d'être "même domaine")
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Bloc conv -> BN -> ReLU répété deux fois.
    Garde la taille spatiale (padding=1).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """
    Bloc "encoder" : max-pooling puis DoubleConv.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """
    Bloc "decoder" : upsampling (bilinear) + concat avec le skip + DoubleConv.
    Gère les tailles impaires en rognant le skip si besoin.
    """
    def __init__(self, in_channels, out_channels, bilinear=True, debug: bool = False):
        super().__init__()
        self.debug = debug

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                         kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Ajuste la taille spatiale si besoin
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        if self.debug:
            print(f"[DEBUG Up] skip={skip.shape}, x={x.shape}")

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNet2D(nn.Module):
    """
    U-Net 2D standard, taille d'entrée N×N quelconque.

    in_channels  = 1 (contact map)
    out_channels = 1 (logit par pixel : même domaine / pas même domaine)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
    ):
        super().__init__()

        # Encoder
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        # Decoder
        self.up3 = Up(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.up2 = Up(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.up1 = Up(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up0 = Up(base_channels * 2 + base_channels, base_channels)

        # Sortie
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        # Bottleneck
        xb = self.bottleneck(x3)

        # Decoder
        x = self.up3(xb, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)

        # Logits (pas de sigmoid ici, on utilise BCEWithLogitsLoss)
        logits = self.out_conv(x)
        return logits