#!/usr/bin/python3
# coding=utf-8
"""统一小波基模块 — 供 WaveletStem / LWA 使用

支持 10 种小波（正交/双正交），纯 PyTorch 卷积实现：
  haar, db2, db4, sym2, sym4, coif1, bior2.2, bior4.4, rbio2.2, rbio4.4

设计要点:
  - haar 走非分离 2×2 精确路径（与旧 WaveletStem/WCA 一致）
  - 其余走可分离 1D 滤波器路径（pywt 系数 → conv2d/conv_transpose2d）
  - 分析/合成滤波器严格配对，保证重构近似恒等（边界有微小误差，内部精确）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

SUPPORTED_WAVELETS = [
    'haar', 'db2', 'db4', 'sym2', 'sym4', 'coif1',
    'bior2.2', 'bior4.4', 'rbio2.2', 'rbio4.4',
]


def _haar_kernels():
    """非分离 2×2 Haar 核（0.5 归一化，与旧代码一致）"""
    h_ll = torch.tensor([[[[ 0.5,  0.5], [ 0.5,  0.5]]]])
    h_lh = torch.tensor([[[[ 0.5,  0.5], [-0.5, -0.5]]]])
    h_hl = torch.tensor([[[[ 0.5, -0.5], [ 0.5, -0.5]]]])
    h_hh = torch.tensor([[[[ 0.5, -0.5], [-0.5,  0.5]]]])
    return torch.cat([h_ll, h_lh, h_hl, h_hh], dim=0)


def _pywt_filters(name):
    """取 pywt 滤波器系数。

    返回 (dec_lo, dec_hi, rec_lo, rec_hi)，均为 torch 1D tensor。
    约定（经验验证，见 scripts/tools/debug_sep_*.py）:
      - 分析用互相关（conv stride2 + 未 flip 的 dec 系数），相位 0
      - 合成用互相关形式 = conv_transpose + flip 后的 rec 系数，
        输出需裁剪偏移 K-1
    """
    w = pywt.Wavelet(name)
    dec_lo = torch.tensor(w.dec_lo, dtype=torch.float32)
    dec_hi = torch.tensor(w.dec_hi, dtype=torch.float32)
    rec_lo = torch.tensor(w.rec_lo, dtype=torch.float32).flip(0)
    rec_hi = torch.tensor(w.rec_hi, dtype=torch.float32).flip(0)
    return dec_lo, dec_hi, rec_lo, rec_hi


class WaveletBasis(nn.Module):
    """统一 DWT/IDWT 基。decompose → (ll, lh, hl, hh) → reconstruct 精确还原。

    用法:
        basis = WaveletBasis('db2')
        ll, lh, hl, hh, orig = basis.decompose(x)      # x: (B,C,H,W)
        rec = basis.reconstruct(ll, lh, hl, hh, orig)  # rec: (B,C,H,W)
    """

    def __init__(self, name='haar'):
        super().__init__()
        name = str(name).lower()
        if name not in SUPPORTED_WAVELETS:
            raise ValueError(
                f'Unsupported wavelet: {name}. Supported: {SUPPORTED_WAVELETS}')
        self.name = name
        self.is_haar = (name == 'haar')

        if self.is_haar:
            kernel = _haar_kernels()          # (4, 1, 2, 2)
            self.register_buffer('dwt_kernel', kernel)
            self.register_buffer('idwt_kernel', kernel.clone())
        else:
            dec_lo, dec_hi, rec_lo, rec_hi = _pywt_filters(name)
            self.filter_len = dec_lo.numel()
            # 分析（互相关）: 行/列滤波用 (1,1,1,K) 与 (1,1,K,1)
            self.register_buffer('dec_lo', dec_lo.view(1, 1, 1, -1))
            self.register_buffer('dec_hi', dec_hi.view(1, 1, 1, -1))
            self.register_buffer('dec_lo_t', dec_lo.view(1, 1, -1, 1))
            self.register_buffer('dec_hi_t', dec_hi.view(1, 1, -1, 1))
            # 合成（转置卷积）
            self.register_buffer('rec_lo', rec_lo.view(1, 1, 1, -1))
            self.register_buffer('rec_hi', rec_hi.view(1, 1, 1, -1))
            self.register_buffer('rec_lo_t', rec_lo.view(1, 1, -1, 1))
            self.register_buffer('rec_hi_t', rec_hi.view(1, 1, -1, 1))

    # ------------------------------------------------------------------ 分解
    def decompose(self, x):
        """返回 (ll, lh, hl, hh, orig_size)，各子带尺寸 floor((H,W)/2)。"""
        B, C, H, W = x.shape
        if self.is_haar:
            pad_h, pad_w = H % 2, W % 2
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            xc = x.reshape(B * C, 1, x.shape[2], x.shape[3])
            sub = F.conv2d(xc, self.dwt_kernel, stride=2, padding=0)
            sub = sub.reshape(B, C, 4, sub.shape[2], sub.shape[3])
            return (sub[:, :, 0], sub[:, :, 1], sub[:, :, 2], sub[:, :, 3],
                    (H, W))
        return self._decompose_sep(x)

    def _decompose_sep(self, x):
        B, C, H, W = x.shape
        K = self.filter_len
        # 边界 reflect 填充后 stride-2 滤波，保证尺寸 floor((H,W)/2)
        x = F.pad(x, (K - 1, K - 1, K - 1, K - 1), mode='reflect')
        xc = x.reshape(B * C, 1, x.shape[2], x.shape[3])

        def row_col(h_row, h_col):
            # 先行后列: (1,1,1,K) stride (1,2) → (1,1,K,1) stride (2,1)
            y = F.conv2d(xc, h_row, stride=(1, 2), padding=0)
            y = F.conv2d(y, h_col, stride=(2, 1), padding=0)
            return y.reshape(B, C, y.shape[2], y.shape[3])

        ll = row_col(self.dec_lo, self.dec_lo_t)
        lh = row_col(self.dec_lo, self.dec_hi_t)
        hl = row_col(self.dec_hi, self.dec_lo_t)
        hh = row_col(self.dec_hi, self.dec_hi_t)
        return ll, lh, hl, hh, (H, W)

    # ------------------------------------------------------------------ 重构
    def reconstruct(self, ll, lh, hl, hh, orig_size):
        """由四个子带重建原始尺寸 (H, W)。"""
        if self.is_haar:
            B, C = ll.shape[0], ll.shape[1]
            sub = torch.stack([ll, lh, hl, hh], dim=2) \
                       .reshape(B * C, 4, ll.shape[2], ll.shape[3])
            rec = F.conv_transpose2d(sub, self.idwt_kernel, stride=2, padding=0)
            rec = rec.reshape(B, C, rec.shape[2], rec.shape[3])
            return rec[:, :, :orig_size[0], :orig_size[1]]
        return self._reconstruct_sep(ll, lh, hl, hh, orig_size)

    def _reconstruct_sep(self, ll, lh, hl, hh, orig_size):
        H, W = orig_size
        K = self.filter_len
        B, C = ll.shape[0], ll.shape[1]

        def to_bc(t):
            return t.reshape(B * C, 1, t.shape[2], t.shape[3])

        # 1) 列合成: LL/LH → 低通列, HL/HH → 高通列
        lo_w = (F.conv_transpose2d(to_bc(ll), self.rec_lo_t, stride=(2, 1), padding=0) +
                F.conv_transpose2d(to_bc(lh), self.rec_hi_t, stride=(2, 1), padding=0))
        hi_w = (F.conv_transpose2d(to_bc(hl), self.rec_lo_t, stride=(2, 1), padding=0) +
                F.conv_transpose2d(to_bc(hh), self.rec_hi_t, stride=(2, 1), padding=0))
        # 2) 行合成
        rec = (F.conv_transpose2d(lo_w, self.rec_lo, stride=(1, 2), padding=0) +
               F.conv_transpose2d(hi_w, self.rec_hi, stride=(1, 2), padding=0))
        rec = rec.reshape(B, C, rec.shape[2], rec.shape[3])
        # 3) 裁剪: 互相关形式合成需偏移 K-1 对齐原信号起点
        off = K - 1
        rec = rec[:, :, off:off + H, off:off + W]
        return rec

    # ------------------------------------------------------------------ 其他
    def extra_repr(self):
        return f'name={self.name}'
