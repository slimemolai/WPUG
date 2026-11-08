#!/usr/bin/python3
# coding=utf-8
"""Attention U-Net for medical image segmentation.
Attention gates on skip connections suppress irrelevant background regions.
Returns 5-tuple (pred, wr, ws, attmap, cls_logits) to match WPUGNet interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution: Conv → BN → ReLU → Conv → BN → ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """Attention gate: filters skip connections using gating signal from coarser scale.

    Reference: "Attention U-Net: Learning Where to Look for the Pancreas" (MIDL 2018)
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gating, skip):
        # gating: [B, F_g, H, W] — decoder feature (upsampled)
        # skip:   [B, F_l, H, W] — encoder skip connection
        g1 = self.W_g(gating)
        x1 = self.W_x(skip)

        # Align spatial sizes
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear',
                               align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # [B, 1, H, W] attention coefficients
        return skip * psi


class AttUNet(nn.Module):
    """Attention U-Net: standard U-Net with attention gates on skip connections.

    Architecture:
        Encoder: 4 stages (64→128→256→512)
        Bottleneck: 512→1024
        Decoder: 4 stages with attention-gated skip connections
        Output: 1×1 conv → single-channel logits
    """

    def __init__(self, cfg=None, in_channels=3, out_channels=1,
                 features=(64, 128, 256, 512)):
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.att_gates = nn.ModuleList()

        # Encoder
        for feature in features:
            self.encoders.append(ConvBlock(in_channels, feature))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feature

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # Decoder with attention gates
        for feature in reversed(features):
            self.up_convs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            # Attention gate: gating=feature (from decoder), skip=feature (from encoder)
            self.att_gates.append(AttentionGate(F_g=feature, F_l=feature, F_int=feature // 2))
            self.decoders.append(ConvBlock(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, shape=None, mask=None):
        # Encoder with skip connections
        skips = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with attention-gated skip connections
        skips = skips[::-1]
        for up_conv, att_gate, decoder, skip in zip(
            self.up_convs, self.att_gates, self.decoders, skips
        ):
            x = up_conv(x)
            # Apply attention gate: filter skip using decoder feature as gating signal
            skip_att = att_gate(gating=x, skip=skip)
            if x.shape[2:] != skip_att.shape[2:]:
                x = F.interpolate(x, size=skip_att.shape[2:], mode='bilinear',
                                  align_corners=True)
            x = torch.cat([skip_att, x], dim=1)
            x = decoder(x)

        pred = self.final_conv(x)

        # Resize to original input shape if needed
        if shape is not None and pred.shape[2:] != shape:
            pred = F.interpolate(pred, size=shape, mode='bilinear',
                                 align_corners=True)

        # Dummy outputs for WPUGNet 5-tuple interface compatibility
        # (minimal size — unused by training when wavelet is disabled)
        B = pred.shape[0]
        device = pred.device
        dummy_wr = torch.zeros(B, 1, 1, 1, device=device)
        dummy_ws = torch.zeros(B, 1, 1, 1, device=device)
        dummy_attmap = torch.zeros(B, 1, 1, 1, device=device)
        dummy_cls_logits = None

        return pred, dummy_wr, dummy_ws, dummy_attmap, dummy_cls_logits
