#!/usr/bin/python3
# coding=utf-8
"""
WPUGNet: Wavelet-Prompted Universal Grafted Network
超声图像分割基础模型 (原 BUSI-Segmentation 迁移)

统一版本 — 合并自 wpugnet.py + wpugnet_wavelet_vis.py 
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from Res import resnet18, resnet50
from Swin import Swintransformer
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import pywt
from wavelet_basis import WaveletBasis, SUPPORTED_WAVELETS
Act = nn.GELU


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _cpu_detach(x):
    return x.detach().float().cpu()


def _mean_abs_map(x):
    return _cpu_detach(x).abs().mean(dim=1, keepdim=True)


def _cfg_get(cfg, key, default):
    """Safe config read — works around Config.__getattr__ returning None for missing keys."""
    if cfg is None:
        return default
    val = cfg.kwargs.get(key, default)
    return default if val is None else val


def resolve_decoder_wca_plan(use_wavelet_decoder=True, variant='baseline'):
    default_plan = {
        'd1': False, 'd2': True,  'd3': True,
        'd4': True,  'd5': True,  'd6': True,
    }
    if not use_wavelet_decoder:
        return 'disabled', {k: False for k in default_plan}

    normalized = 'baseline' if variant is None else str(variant).strip().lower()
    alias_map = {
        'default': 'baseline', 'original': 'baseline', 'baseline': 'baseline',
        'stage1': 'baseline_plus_stage1', 'stage1_plus': 'baseline_plus_stage1',
        'baseline_plus_stage1': 'baseline_plus_stage1',
        'stage1_only': 'stage1_only',
        'stage23': 'stage23_only', 'stage2_3': 'stage23_only',
        'stage2-3': 'stage23_only', 'stage23_only': 'stage23_only',
    }
    normalized = alias_map.get(normalized, normalized)

    plans = {
        'baseline': default_plan,
        'baseline_plus_stage1': {
            'd1': True, 'd2': True, 'd3': True, 'd4': True, 'd5': True, 'd6': True,
        },
        'stage1_only': {
            'd1': True, 'd2': False, 'd3': False, 'd4': False, 'd5': False, 'd6': False,
        },
        'stage23_only': {
            'd1': False, 'd2': True, 'd3': True, 'd4': False, 'd5': False, 'd6': False,
        },
        # 单 stage 位置消融
        'd2_only': {'d1': False, 'd2': True,  'd3': False, 'd4': False, 'd5': False, 'd6': False},
        'd3_only': {'d1': False, 'd2': False, 'd3': True,  'd4': False, 'd5': False, 'd6': False},
        'd4_only': {'d1': False, 'd2': False, 'd3': False, 'd4': True,  'd5': False, 'd6': False},
        'd5_only': {'d1': False, 'd2': False, 'd3': False, 'd4': False, 'd5': True,  'd6': False},
        'd6_only': {'d1': False, 'd2': False, 'd3': False, 'd4': False, 'd5': False, 'd6': True},
    }
    if normalized not in plans:
        raise ValueError(
            f'Unknown decoder_wca_variant: {variant}. '
            'Supported values: baseline / baseline_plus_stage1 / stage1_only / stage23_only'
        )
    return normalized, plans[normalized]


# ---------------------------------------------------------------------------
# 权重初始化
# ---------------------------------------------------------------------------

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.GELU, Act, nn.AdaptiveAvgPool2d, nn.Softmax,
                            nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.ReLU6,
                            nn.MaxPool2d, nn.AvgPool2d, nn.Dropout, nn.Dropout2d,
                            nn.Upsample, nn.UpsamplingBilinear2d, nn.Identity)):
            pass
        elif hasattr(m, 'initialize'):
            m.initialize()
        else:
            pass


# ---------------------------------------------------------------------------
# WaveletStem
# ---------------------------------------------------------------------------

class WaveletStem(nn.Module):
    """固定 Haar 小波一级分解层（à-trous 全分辨率版本）

    用 stride=1 + dilation=2 的非抽样小波（à-trous），4 个子带保持输入分辨率，
    输出 12ch × H × W。修复原 stride=2 导致的"小波特征 vs 分辨率减半"消融混淆
    。输出像素总量 12×H×W（原为 12×(H/2)×(W/2)=3×H×W），
    后续 ResNet conv1 stride=2 不变，内部特征图整体上移一档分辨率。
    """

    def __init__(self):
        super(WaveletStem, self).__init__()
        self.capture_debug = False
        self.latest_debug = None
        h_lo = torch.tensor([[[[ 0.5,  0.5], [ 0.5,  0.5]]]])
        h_lh = torch.tensor([[[[ 0.5,  0.5], [-0.5, -0.5]]]])
        h_hl = torch.tensor([[[[ 0.5, -0.5], [ 0.5, -0.5]]]])
        h_hh = torch.tensor([[[[ 0.5, -0.5], [-0.5,  0.5]]]])
        kernel = torch.cat([h_lo, h_lh, h_hl, h_hh], dim=0)
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        B, C, H, W = x.shape
        bands = []
        for c in range(C):
            xc = x[:, c:c+1, :, :]
            sub = F.conv2d(xc, self.kernel, stride=1, padding=1, dilation=2)
            bands.append(sub)
        out = torch.cat(bands, dim=1)
        if self.capture_debug:
            self.latest_debug = {
                'll': _mean_abs_map(out[:, 0::4, :, :]),
                'lh': _mean_abs_map(out[:, 1::4, :, :]),
                'hl': _mean_abs_map(out[:, 2::4, :, :]),
                'hh': _mean_abs_map(out[:, 3::4, :, :]),
            }
        else:
            self.latest_debug = None
        return out

    def initialize(self):
        pass


# ---------------------------------------------------------------------------
# Cross Grafting Module (CGM) — 核心创新
# ---------------------------------------------------------------------------

class Grafting(nn.Module):
    """Cross Grafting Module: 融合 ResNet 深层语义 + Swin 浅层结构"""

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkvs = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkvr = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.conv = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.lnx = nn.LayerNorm(64)
        self.lny = nn.LayerNorm(64)
        self.bn = nn.BatchNorm2d(8)
        self.max_pool1d = nn.AdaptiveMaxPool1d(784)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.GELU()
        )

    def forward(self, x, y):
        batch_size = x.shape[0]
        chanel = x.shape[1]
        sc1 = x + y

        x = x.view(batch_size, chanel, -1).permute(0, 2, 1)
        y = y.view(batch_size, chanel, -1).permute(0, 2, 1)

        sc2 = torch.cat((x, y), 1).permute(0, 2, 1)
        sc2 = self.max_pool1d(sc2).permute(0, 2, 1)

        x = self.lnx(x)
        y = self.lny(y)

        B, N, C = x.shape
        x_qkvr = self.qkvr(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        y_qkvs = self.qkvs(y).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        x_qr, x_kr, x_vr = x_qkvr[0], x_qkvr[1], x_qkvr[2]
        y_qs, y_ks, y_vs = y_qkvs[0], y_qkvs[1], y_qkvs[2]

        q = x_qr + y_qs
        k = x_kr + y_ks
        v = x_vr + y_vs

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        y = (attn @ v).transpose(1, 2).reshape(B, N, C)
        y = self.proj(y)
        y = (y + sc2)
        y = y.permute(0, 2, 1)
        y = y.view(batch_size, chanel, *sc1.size()[2:])
        gf = self.conv2(y) + y
        ctam = self.act(self.bn(self.conv(attn + attn.transpose(-1, -2))))
        return gf, ctam

    def initialize(self):
        weight_init(self)


class FusionConcat(nn.Module):
    """CGM 消融变体 1: 简单 concat 融合（无注意力，无 attmap 监督）"""

    def __init__(self, dim=64):
        super(FusionConcat, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim), nn.GELU()
        )

    def forward(self, x, y):
        gf = self.fuse(torch.cat([x, y], dim=1))
        return gf, None

    def initialize(self):
        weight_init(self)


class FusionAdd(nn.Module):
    """CGM 消融变体 2: 逐元素相加融合（无参数，无 attmap 监督）"""

    def forward(self, x, y):
        return x + y, None

    def initialize(self):
        pass


class FusionCrossAttn(nn.Module):
    """CGM 消融变体 3: TransFuse 式单向交叉注意力 (Q=ResNet, K/V=Swin)

    返回 (gf, attn) 其中 attn 为 (B, 8, N, N) 多头注意力图，
    经模型自带的 conv(8→1) 头输出 attmap logits → att_loss 监督保持开启。
    """

    def __init__(self, dim=64, num_heads=8):
        super(FusionCrossAttn, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, y):
        B, C, H, W = x.shape
        N = H * W
        xf = x.view(B, C, N).permute(0, 2, 1)   # B, N, C
        yf = y.view(B, C, N).permute(0, 2, 1)
        q = self.wq(xf).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(yf).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(yf).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.norm(out + xf)               # 残差回接 ResNet 分支
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out, attn

    def initialize(self):
        weight_init(self)


def build_fusion(mode, dim=64, num_heads=8):
    """按 cgm_mode 构造融合模块: cgm / concat / add / crossattn"""
    normalized = str(mode).strip().lower()
    if normalized in ('cgm', 'grafting'):
        return Grafting(dim, num_heads=num_heads)
    if normalized == 'concat':
        return FusionConcat(dim)
    if normalized == 'add':
        return FusionAdd()
    if normalized == 'crossattn':
        return FusionCrossAttn(dim, num_heads=num_heads)
    raise ValueError(
        f'Unknown cgm_mode: {mode}. Supported: cgm / concat / add / crossattn'
    )


# ---------------------------------------------------------------------------
# WaveletChannelAttention (WCA) — 已弃用，保留仅供对照
# ---------------------------------------------------------------------------

class WaveletChannelAttention(nn.Module):
    """小波通道注意力（已弃用: 残差 out+x 造成 2x 放大 → 5 块串联梯度爆炸 NaN。

    已被 LWA (LearnableWaveletAttention) 替代，本类不再被任何模块引用，
    保留仅供论文中 WCA vs LWA 的对照分析。）"""

    def __init__(self, channels, reduction=4):
        super(WaveletChannelAttention, self).__init__()
        self.capture_debug = False
        self.latest_debug = None
        h_ll = torch.tensor([[[[ 0.5,  0.5], [ 0.5,  0.5]]]])
        h_lh = torch.tensor([[[[ 0.5,  0.5], [-0.5, -0.5]]]])
        h_hl = torch.tensor([[[[ 0.5, -0.5], [ 0.5, -0.5]]]])
        h_hh = torch.tensor([[[[ 0.5, -0.5], [-0.5,  0.5]]]])
        kernel = torch.cat([h_ll, h_lh, h_hl, h_hh], dim=0)
        self.register_buffer('dwt_kernel', kernel)
        self.register_buffer('idwt_kernel', kernel.clone())

        mid = max(channels // reduction, 4)

        self.se_ll = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, channels), nn.Sigmoid()
        )
        self.se_hf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels * 3, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, channels * 3), nn.Sigmoid()
        )

    def _dwt(self, x):
        B, C, H, W = x.shape
        pad_h, pad_w = H % 2, W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        xc = x.reshape(B * C, 1, x.shape[2], x.shape[3])
        sub = F.conv2d(xc, self.dwt_kernel, stride=2, padding=0)
        sub = sub.reshape(B, C, 4, sub.shape[2], sub.shape[3])
        return sub[:, :, 0], sub[:, :, 1], sub[:, :, 2], sub[:, :, 3], (H, W)

    def _idwt(self, ll, lh, hl, hh, orig_size):
        B, C = ll.shape[0], ll.shape[1]
        sub = torch.stack([ll, lh, hl, hh], dim=2).reshape(B * C, 4, ll.shape[2], ll.shape[3])
        rec = F.conv_transpose2d(sub, self.idwt_kernel, stride=2, padding=0)
        rec = rec.reshape(B, C, rec.shape[2], rec.shape[3])
        return rec[:, :, :orig_size[0], :orig_size[1]]

    def forward(self, x):
        ll, lh, hl, hh, orig_size = self._dwt(x)

        ll_weight = self.se_ll(ll)
        ll_w = ll_weight.unsqueeze(-1).unsqueeze(-1)
        ll_att = ll * ll_w

        B, C, h, w = lh.shape
        hf_cat = torch.cat([lh, hl, hh], dim=1)
        hf_weight = self.se_hf(hf_cat)
        hf_w = hf_weight.unsqueeze(-1).unsqueeze(-1)
        hf_w = hf_w.reshape(B, 3, C, 1, 1)
        lh_att = lh * hf_w[:, 0]
        hl_att = hl * hf_w[:, 1]
        hh_att = hh * hf_w[:, 2]

        if self.capture_debug:
            self.latest_debug = {
                'll': _mean_abs_map(ll), 'lh': _mean_abs_map(lh),
                'hl': _mean_abs_map(hl), 'hh': _mean_abs_map(hh),
                'll_att': _mean_abs_map(ll_att), 'lh_att': _mean_abs_map(lh_att),
                'hl_att': _mean_abs_map(hl_att), 'hh_att': _mean_abs_map(hh_att),
                'll_weight': _cpu_detach(ll_weight),
                'hf_weight': _cpu_detach(hf_weight.reshape(B, 3, C)),
            }
        else:
            self.latest_debug = None

        out = self._idwt(ll_att, lh_att, hl_att, hh_att, orig_size)
        return out + x

    def initialize(self):
        weight_init(self)


# ---------------------------------------------------------------------------
# LearnableWaveletAttention (LWA) — WCA 的稳定替代
# ---------------------------------------------------------------------------

class LearnableWaveletAttention(nn.Module):
    """可学习小波注意力 (LWA)。

    设计:
      - 固定小波基（WaveletBasis，默认 haar，可选 db2/bior4.4 等 10 种）
      - 逐通道标量门 g ∈ R^{4×C}，初始化为 1
      - 输出 = IDWT(g·子带)，无残差放大

    关键性质: 初始化时 g=1 且 DWT/IDWT 精确互逆 → 模块严格恒等映射，
    训练从恒等出发逐步学到频带重加权，从结构上杜绝 WCA 的 NaN 病态。
    参数 4×C（C=64 时 256 个），远小于 WCA 的 SE（8192 个）。
    """

    def __init__(self, channels, basis='haar'):
        super(LearnableWaveletAttention, self).__init__()
        self.capture_debug = False
        self.latest_debug = None
        self.basis_name = basis
        self.basis = WaveletBasis(basis)
        # 逐通道标量门，init=1 → 恒等起步
        self.g_ll = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.g_lh = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.g_hl = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.g_hh = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        ll, lh, hl, hh, orig_size = self.basis.decompose(x)

        ll_g = ll * self.g_ll
        lh_g = lh * self.g_lh
        hl_g = hl * self.g_hl
        hh_g = hh * self.g_hh

        if self.capture_debug:
            # debug 键与旧 WCA 兼容（train.py/export 工具直接可用）
            self.latest_debug = {
                'll': _mean_abs_map(ll), 'lh': _mean_abs_map(lh),
                'hl': _mean_abs_map(hl), 'hh': _mean_abs_map(hh),
                'll_att': _mean_abs_map(ll_g), 'lh_att': _mean_abs_map(lh_g),
                'hl_att': _mean_abs_map(hl_g), 'hh_att': _mean_abs_map(hh_g),
                'll_weight': _cpu_detach(self.g_ll.reshape(1, -1)),
                'hf_weight': _cpu_detach(torch.cat(
                    [self.g_lh, self.g_hl, self.g_hh], dim=1).reshape(1, 3, -1)),
            }
        else:
            self.latest_debug = None

        out = self.basis.reconstruct(ll_g, lh_g, hl_g, hh_g, orig_size)
        return out

    def initialize(self):
        pass  # 门控保持恒等初始化，不参与 weight_init

    def extra_repr(self):
        return f'basis={self.basis_name}'


# ---------------------------------------------------------------------------
# SE-LWA — 以 Squeeze-and-Excitation 为机制的可学习小波注意力
# ---------------------------------------------------------------------------

class _LWA_SEBase(nn.Module):
    """SE 系小波注意力的公共基类。

    机制: DWT 分解 → LL 子带与三高频子带(拼接)各过一个 SE 通道注意力
    (AdaptiveAvgPool → Linear(C→mid) → ReLU → Linear(mid→C)) → 门控各子带
    → IDWT 重建。

    恒等起步设计: 门控取 2·σ(z) 且 SE 末层零初始化 → 初始化时门=1，
    DWT/IDWT 精确互逆 ⇒ 模块严格恒等（与标量门 LWA 相同的稳定性保证）。
    历史对照: 旧 WCA (WaveletChannelAttention) 为同类 SE 结构但固定残差
    out=att(x)+x 造成 2x 放大 → 5 块串联梯度爆炸 NaN。

    参数: se_ll C·mid×2 + se_hf 3C·mid×2（C=64, r=4 → mid=16 ⇒ 约 8.5K/块）。
    """

    residual = False  # 子类覆盖

    def __init__(self, channels, basis='haar', reduction=4):
        super().__init__()
        self.capture_debug = False
        self.latest_debug = None
        self.basis_name = basis
        self.basis = WaveletBasis(basis)
        mid = max(channels // reduction, 4)

        self.se_ll = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )
        self.se_hf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels * 3, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, channels * 3),
        )
        # 恒等起步：末层零初始化 → 2·σ(0) = 1
        nn.init.zeros_(self.se_ll[-1].weight)
        nn.init.zeros_(self.se_ll[-1].bias)
        nn.init.zeros_(self.se_hf[-1].weight)
        nn.init.zeros_(self.se_hf[-1].bias)

    def _gates(self, ll, lh, hl, hh):
        B, C = ll.shape[0], ll.shape[1]
        ll_g = 2 * torch.sigmoid(self.se_ll(ll)).view(B, C, 1, 1)
        hf = torch.cat([lh, hl, hh], dim=1)
        hf_g = 2 * torch.sigmoid(self.se_hf(hf)).view(B, 3, C, 1, 1)
        return ll_g, hf_g

    def forward(self, x):
        ll, lh, hl, hh, orig_size = self.basis.decompose(x)

        ll_g, hf_g = self._gates(ll, lh, hl, hh)
        ll_att = ll * ll_g
        lh_att = lh * hf_g[:, 0]
        hl_att = hl * hf_g[:, 1]
        hh_att = hh * hf_g[:, 2]

        if self.capture_debug:
            # debug 键与标量门 LWA 兼容（train.py/export 工具直接可用）
            self.latest_debug = {
                'll': _mean_abs_map(ll), 'lh': _mean_abs_map(lh),
                'hl': _mean_abs_map(hl), 'hh': _mean_abs_map(hh),
                'll_att': _mean_abs_map(ll_att), 'lh_att': _mean_abs_map(lh_att),
                'hl_att': _mean_abs_map(hl_att), 'hh_att': _mean_abs_map(hh_att),
                'll_weight': _cpu_detach(ll_g.reshape(x.shape[0], -1)),
                'hf_weight': _cpu_detach(hf_g.reshape(x.shape[0], 3, -1)),
            }
        else:
            self.latest_debug = None

        out = self.basis.reconstruct(ll_att, lh_att, hl_att, hh_att, orig_size)
        if self.residual:
            out = out + self.res_scale * x
        return out

    def initialize(self):
        pass  # 保持零初始化恒等起步，不参与 weight_init

    def extra_repr(self):
        return f'basis={self.basis_name}, residual={self.residual}'


class LearnableWaveletAttentionSE(_LWA_SEBase):
    """SE-LWA（无残差）: SE 通道注意力门控小波子带，IDWT 重建，无残差项。

    初始化严格恒等（2σ 参数化 + 末层零初始化），与标量门 LWA 的稳定性
    性质一致，仅门控机制从标量改为 SE（可建模通道间依赖）。
    """

    residual = False


class LearnableWaveletAttentionSERes(_LWA_SEBase):
    """SE-Res-LWA（残差）: SE 门控小波子带 + IDWT 重建 + 可学习残差路径。

    out = IDWT(g·子带) + r·x，残差尺度 r ∈ R^{1×C×1×1} 初始化为 0 ⇒
    初始化仍严格恒等，训练中可学习增大残差贡献。与旧 WCA 的固定
    out=att(x)+x（初始化即 2x 放大）不同，从结构上规避其 NaN 病态。
    """

    residual = True

    def __init__(self, channels, basis='haar', reduction=4):
        super().__init__(channels, basis=basis, reduction=reduction)
        self.res_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))


class LearnableWaveletAttentionWCAPlus(nn.Module):
    """WCA+（迁移自 src/WCA+.py，2026-08-25，仅最小适配）: 频域感知小波校准模块。

    与原文件一致的设计:
      - LL: SE 通道注意力做全局结构校准（reduction=8, hidden=max(C//8,4)）
      - LH/HL: 拼接后经分组空间卷积门控做边界增强（局部空间门，非通道门）
      - HH: 逐通道标量门 sigmoid(hh_gate) 做高频噪声/细节抑制（init 0.5）
      - 稳定残差: y = x + gamma * F(x)，gamma 初始化 0 ⇒ 初始化严格恒等，
        训练中逐步学习残差贡献（与旧 WCA 固定 out=att(x)+x 的 2x 放大不同）
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        self.capture_debug = False
        self.latest_debug = None

        # -------------------------
        # Haar wavelet
        # -------------------------
        h_ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        h_lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
        h_hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
        h_hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        kernel = torch.stack([h_ll, h_lh, h_hl, h_hh]).unsqueeze(1)
        self.register_buffer("dwt_kernel", kernel)
        self.register_buffer("idwt_kernel", kernel)

        # -------------------------
        # LL global attention
        # -------------------------
        hidden = max(channels // reduction, 4)
        self.ll_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels), nn.Sigmoid()
        )

        # -------------------------
        # LH / HL boundary attention (local spatial gate)
        # -------------------------
        self.edge_attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # -------------------------
        # HH noise gate
        # -------------------------
        self.hh_gate = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.5)

        # -------------------------
        # residual scaling
        # -------------------------
        self.gamma = nn.Parameter(torch.zeros(1))

    def dwt(self, x):
        B, C, H, W = x.shape
        xc = x.reshape(B * C, 1, H, W)
        out = F.conv2d(xc, self.dwt_kernel, stride=2)
        h, w = out.shape[-2:]
        out = out.reshape(B, C, 4, h, w)
        return out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3], (H, W)

    def idwt(self, ll, lh, hl, hh, size):
        B, C, H, W = ll.shape
        x = torch.stack([ll, lh, hl, hh], dim=2)
        x = x.reshape(B * C, 4, H, W)
        out = F.conv_transpose2d(x, self.idwt_kernel, stride=2)
        out = out.reshape(B, C, out.shape[-2], out.shape[-1])
        return out[:, :, :size[0], :size[1]]

    def _wavelet_branch(self, x):
        ll, lh, hl, hh, size = self.dwt(x)

        # 1. LL global structure
        ll_weight = self.ll_attention(ll)
        ll_weight = ll_weight.unsqueeze(-1).unsqueeze(-1)
        ll_out = ll * ll_weight

        # 2. LH/HL boundary
        edge = torch.cat([lh, hl], dim=1)
        edge_weight = self.edge_attention(edge)
        lh_out = lh * edge_weight
        hl_out = hl * edge_weight

        # 3. HH detail/noise
        hh_weight = torch.sigmoid(self.hh_gate)
        hh_out = hh * hh_weight

        if self.capture_debug:
            edge_mean = edge_weight.detach().float().cpu().mean(dim=(2, 3))  # (B,C)
            hh_g = _cpu_detach(hh_weight.squeeze(-1).squeeze(-1))            # (1,C)
            self.latest_debug = {
                'll': _mean_abs_map(ll), 'lh': _mean_abs_map(lh),
                'hl': _mean_abs_map(hl), 'hh': _mean_abs_map(hh),
                'll_att': _mean_abs_map(ll_out), 'lh_att': _mean_abs_map(lh_out),
                'hl_att': _mean_abs_map(hl_out), 'hh_att': _mean_abs_map(hh_out),
                'll_weight': _cpu_detach(ll_weight.reshape(x.shape[0], -1)),
                'hf_weight': torch.stack(
                    [edge_mean, edge_mean, hh_g.expand(edge_mean.shape[0], -1)],
                    dim=1,
                ),  # (B,3,C)：lh/hl 用边界门均值，hh 用噪声门
            }
        else:
            self.latest_debug = None

        # reconstruction
        return self.idwt(ll_out, lh_out, hl_out, hh_out, size)

    def forward(self, x):
        O = self._wavelet_branch(x)
        # stable residual
        return x + self.gamma * O

    def initialize(self):
        pass  # gamma=0 恒等起步，不参与 weight_init

    def extra_repr(self):
        return f'channels={self.channels}'


class LearnableWaveletAttentionResWCA(LearnableWaveletAttentionWCAPlus):
    """ResWCA: 差模残差 Y = X + γ·(O−X) = (1−γ)·X + γ·O。

    与 WCA+ 唯一区别在残差融合形式：WCA+ 加完整重建 O，ResWCA 加差 O−X。
    性质:
      - γ=0 ⇒ Y=X（恒等起步，同 WCA+）
      - γ=1 ⇒ Y=O（完全采用小波支路）
      - O≈X 时 Y≈X —— 天然避免 WCA+ 在浅层（O≈X 且 γ→-1）出现的
        Y≈(1-|γ|)·X 幅度衰减问题（见 2026-08-26 块级统计 d5/d6 S≈0.22）
    """

    def forward(self, x):
        O = self._wavelet_branch(x)
        return x + self.gamma * (O - x)


class LearnableWaveletAttentionResWCAGCA(LearnableWaveletAttentionWCAPlus):
    """ResWCA-GCA: RWCA+ 差模残差 + 全局内容自适应。

    与 ResWCA 区别:
      - HH 门由静态标量 sigmoid(hh_gate) 改为内容自适应的 MLP(|HH|) 通道门
        —— 四个子带门全部内容自适应（LL=SE 全局、LH/HL=分组空间卷积、
        HH=MLP(|HH|)），消除唯一的非自适应部件
      - v3（08-29 凌晨，「γ 无界 + HH 部分有界」后备方案）:
        v1 无界 γ 三次 NaN（γ 漂移 0.74-0.91 + HH 放开高频）；
        v2 γ 有界+HH 下界稳定但 test 家族最弱（0.7904，表达力被束缚）
        → γ 恢复无界 init 0（父类原设计）
        → hh_weight = 0.5·σ(MLP(|HH|)) ∈ (0,0.5]（高频至多半通过，
          针对 v1 崩溃机制「HH 全通」设上限，保内容自适应）
    - 差模残差 Y = X + γ·(O−X)
    - 参数: HH 门 ≈ 2·C·(C/8)（C=64 时 ~1.2K，因 hidden=C//8=8）
    """

    def __init__(self, channels, reduction=8):
        super().__init__(channels, reduction)
        hidden = max(channels // reduction, 4)
        # 替换静态 HH 门为内容自适应 MLP 通道门（先注销父类的 Parameter 注册）
        del self.hh_gate
        self.hh_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels)
        )
        # 首层 kaiming、末层零初始化（起步 hh = 0.5·σ(0) = 0.25）
        nn.init.kaiming_normal_(self.hh_gate[2].weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.hh_gate[2].bias)
        nn.init.zeros_(self.hh_gate[4].weight)
        nn.init.zeros_(self.hh_gate[4].bias)
        # γ 保持父类无界 init 0（v3 设计，不另行定义）

    def _wavelet_branch(self, x):
        ll, lh, hl, hh, size = self.dwt(x)

        # 1. LL global structure（同 WCA+）
        ll_weight = self.ll_attention(ll)
        ll_weight = ll_weight.unsqueeze(-1).unsqueeze(-1)
        ll_out = ll * ll_weight

        # 2. LH/HL boundary（同 WCA+）
        edge = torch.cat([lh, hl], dim=1)
        edge_weight = self.edge_attention(edge)
        lh_out = lh * edge_weight
        hl_out = hl * edge_weight

        # 3. HH 内容自适应通道门（对幅值统计 |HH|，上界 0.5 防高频全通）
        hh_weight = 0.5 * torch.sigmoid(self.hh_gate(hh.abs())).unsqueeze(-1).unsqueeze(-1)
        hh_out = hh * hh_weight

        if self.capture_debug:
            edge_mean = edge_weight.detach().float().cpu().mean(dim=(2, 3))
            hh_g = _cpu_detach(hh_weight.squeeze(-1).squeeze(-1))
            self.latest_debug = {
                'll': _mean_abs_map(ll), 'lh': _mean_abs_map(lh),
                'hl': _mean_abs_map(hl), 'hh': _mean_abs_map(hh),
                'll_att': _mean_abs_map(ll_out), 'lh_att': _mean_abs_map(lh_out),
                'hl_att': _mean_abs_map(hl_out), 'hh_att': _mean_abs_map(hh_out),
                'll_weight': _cpu_detach(ll_weight.reshape(x.shape[0], -1)),
                'hf_weight': torch.stack(
                    [edge_mean, edge_mean, hh_g.expand(edge_mean.shape[0], -1)],
                    dim=1,
                ),
            }
        else:
            self.latest_debug = None

        return self.idwt(ll_out, lh_out, hl_out, hh_out, size)

    def forward(self, x):
        O = self._wavelet_branch(x)
        return x + self.gamma * (O - x)


class LearnableWaveletAttentionRWCAScalar(nn.Module):
    """RWCA-Scalar: 每子带标量门（非通道级）对照。

    与无 β RWCA 唯一区别: a_i = α·tanh(raw_i)，raw_i 为 4 个标量参数
    （非 MLP 通道门）。ΔX = IDWT(a_i·W_i)，Y = X + ΔX。
    用途: 检验 channel specificity 是否必要。
    """

    def __init__(self, channels, basis='haar', alpha=0.5):
        super().__init__()
        self.channels = channels
        self.basis_name = basis
        self.alpha = float(alpha)
        self.basis = WaveletBasis(basis)
        self.raw_ll = nn.Parameter(torch.zeros(1))
        self.raw_lh = nn.Parameter(torch.zeros(1))
        self.raw_hl = nn.Parameter(torch.zeros(1))
        self.raw_hh = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        ll, lh, hl, hh, orig_size = self.basis.decompose(x)
        a_ll = self.alpha * torch.tanh(self.raw_ll)
        a_lh = self.alpha * torch.tanh(self.raw_lh)
        a_hl = self.alpha * torch.tanh(self.raw_hl)
        a_hh = self.alpha * torch.tanh(self.raw_hh)
        delta_x = self.basis.reconstruct(
            a_ll * ll, a_lh * lh, a_hl * hl, a_hh * hh, orig_size)
        return x + delta_x


class LearnableWaveletAttentionSpatialControl(nn.Module):
    """Spatial matched control: 无频率分解的对照。

    与无 β RWCA 参数量匹配（4 个 MLP 通道门），但门直接作用于原始特征:
      a_i = α·tanh(MLP_i(GAP(|X|)))（i=1..4，同一输入统计）
      ΔX = (Σ a_i / 4)·X，Y = X + ΔX
    用途: 检验性能增益来自「小波频率分解」还是「额外通道校准容量」。
    """

    def __init__(self, channels, reduction=4, alpha=0.5):
        super().__init__()
        self.channels = channels
        self.alpha = float(alpha)
        mid = max(channels // reduction, 4)
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(channels, mid), nn.ReLU(inplace=True),
                nn.Linear(mid, channels),
            ) for _ in range(4)
        ])
        for g in self.gates:
            nn.init.kaiming_normal_(g[2].weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(g[2].bias)
            nn.init.zeros_(g[4].weight)
            nn.init.zeros_(g[4].bias)

    def forward(self, x):
        stat = x.abs()
        a_sum = 0.0
        for g in self.gates:
            a_sum = a_sum + g(stat)
        a = self.alpha * torch.tanh(a_sum / 4.0)
        a = a.unsqueeze(-1).unsqueeze(-1)
        return x + a * x


# ---------------------------------------------------------------------------
# RWCA — Residual Wavelet Channel Attention：Channel-wise Subband Residual Calibration
# ---------------------------------------------------------------------------

class _SubbandResidualGate(nn.Module):
    """
    Channel-wise signed residual gate.

    对单个 wavelet subband 生成:
        a = alpha * tanh(MLP(stat))

    其中:
        a ∈ [-alpha, alpha]

    a > 0 : 增强该频带
    a < 0 : 抑制该频带
    a = 0 : 不进行频率修正

    注意：
    这里输出的是 residual coefficient，
    而不是传统 sigmoid attention weight。
    """

    def __init__(self, channels, reduction=4, alpha=0.5, gate_fn='tanh', use_abs_stat=True):
        super().__init__()

        self.channels = channels
        self.alpha = float(alpha)
        self.gate_fn = gate_fn
        self.use_abs_stat = use_abs_stat

        mid = max(channels // reduction, 4)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(channels, mid)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mid, channels)

        # 第一层正常初始化
        nn.init.kaiming_normal_(
            self.fc1.weight,
            mode='fan_in',
            nonlinearity='relu'
        )
        nn.init.zeros_(self.fc1.bias)

        # 第二层初始化保证恒等起步：
        #   tanh:    全零 → logits=0 → a = alpha*tanh(0) = 0
        #   sigmoid: 全零 + bias=-4 → logits≈-4 → a = alpha*sigmoid(-4) ≈ 0.018*alpha ≈ 0
        nn.init.zeros_(self.fc2.weight)
        if gate_fn == 'sigmoid':
            nn.init.constant_(self.fc2.bias, -4.0)
        else:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        x: [B, C, H, W]

        return:
            a: [B, C, 1, 1]
        """

        # 对小波系数采用 magnitude statistic。
        #
        # 特别是 LH / HL / HH:
        # 直接 GAP 容易受到正负系数抵消影响。
        stat = x.abs() if self.use_abs_stat else x

        stat = self.pool(stat).flatten(1)

        logits = self.fc2(
            self.act(
                self.fc1(stat)
            )
        )

        # 零中心 + 有界（tanh）；或非负注意力（sigmoid）
        if self.gate_fn == 'sigmoid':
            a = self.alpha * torch.sigmoid(logits)
        else:
            a = self.alpha * torch.tanh(logits)

        return a.unsqueeze(-1).unsqueeze(-1)

    def initialize(self):
        # 已经在 __init__ 中显式初始化
        pass


class LearnableWaveletAttentionRWCA(nn.Module):
    """
    RWCA — Residual Wavelet Channel Attention

    ------------------------------------------------------------
    核心公式
    ------------------------------------------------------------

        W_i = DWT(X)_i

        a_i = alpha * tanh(f_i(W_i))

        DeltaW_i = a_i * W_i

        DeltaX = IDWT(DeltaW)

        Y = X + beta * DeltaX

    ------------------------------------------------------------
    与旧 ResWCA 的根本区别
    ------------------------------------------------------------

    ResWCA:

        O = F(X)

        Y = X + gamma * (O - X)

    这里 O 可以自由学习为一个新的 feature representation。

    新 RWCA:

        DeltaW_i = a_i * W_i

    不再生成自由的 O。

    wavelet branch 只能对已有的频率成分进行有界调制，
    因而更加接近:

        frequency residual calibration

    ------------------------------------------------------------
    初始化
    ------------------------------------------------------------

    a_i = 0

    beta = beta_init > 0

    因此:

        DeltaW = 0
        DeltaX = 0
        Y = X

    是严格恒等映射。

    但 beta 不初始化为 0，
    避免 beta 和 gate 同时 zero 导致 gate 学习启动困难。
    """

    def __init__(
        self,
        channels,
        basis='haar',
        reduction=4,
        alpha=0.5,
        beta_init=0.25,
        beta_max=1.0,
        gate_mode='per_subband',
        gate_fn='tanh',
        use_abs_stat=True,
        learnable_beta=True,
        use_beta=True,
    ):
        super(LearnableWaveletAttentionRWCA, self).__init__()

        self.channels = channels
        self.basis_name = basis

        self.alpha = float(alpha)
        self.beta_max = float(beta_max)
        self.gate_mode = gate_mode
        self.gate_fn = gate_fn
        self.use_abs_stat = use_abs_stat
        self.learnable_beta = learnable_beta
        self.use_beta = use_beta

        if alpha <= 0:
            raise ValueError(
                f'alpha must be > 0, got {alpha}'
            )

        if beta_max <= 0:
            raise ValueError(
                f'beta_max must be > 0, got {beta_max}'
            )

        if learnable_beta and not (0 < beta_init < beta_max):
            raise ValueError(
                f'beta_init must satisfy '
                f'0 < beta_init < beta_max, '
                f'got beta_init={beta_init}'
            )

        if gate_mode not in ('per_subband', 'shared', 'll_only', 'hf_only',
                             'lh_only', 'hl_only', 'hh_only', 'll_hf', 'shuffle'):
            raise ValueError(f'unknown gate_mode: {gate_mode}')
        if gate_fn not in ('tanh', 'sigmoid'):
            raise ValueError(f'unknown gate_fn: {gate_fn}')

        # -------------------------------------------------------
        # 使用 WPUGNet 已有 WaveletBasis
        # -------------------------------------------------------
        self.basis = WaveletBasis(basis)

        # -------------------------------------------------------
        # 子带门（消融选项，2026-08-27）:
        #   per_subband: LL/LH/HL/HH 四个独立门（默认）
        #   shared:      四子带共享一个门
        #   ll_only:     仅 LL 门（高频门冻结为 0）
        #   hf_only:     仅 LH/HL/HH 门（LL 冻结为 0）
        # gate_fn:
        #   tanh:    a = alpha * tanh(logits)（有符号残差，默认）
        #   sigmoid: a = alpha * sigmoid(logits)（非负注意力，fc2 bias init -4
        #            使 logits≈-4 → a≈0.018*alpha ≈ 0 保持恒等起步）
        # -------------------------------------------------------
        self.gate_ll = None
        self.gate_lh = None
        self.gate_hl = None
        self.gate_hh = None

        if gate_mode in ('per_subband', 'shuffle'):
            self.gate_ll = _SubbandResidualGate(channels, reduction, alpha, gate_fn, use_abs_stat)
            self.gate_lh = _SubbandResidualGate(channels, reduction, alpha, gate_fn, use_abs_stat)
            self.gate_hl = _SubbandResidualGate(channels, reduction, alpha, gate_fn, use_abs_stat)
            self.gate_hh = _SubbandResidualGate(channels, reduction, alpha, gate_fn, use_abs_stat)
        elif gate_mode == 'shared':
            self.gate_ll = _SubbandResidualGate(channels, reduction, alpha, gate_fn, use_abs_stat)
        elif gate_mode == 'll_only':
            self.gate_ll = _SubbandResidualGate(channels, reduction, alpha, gate_fn, use_abs_stat)
        elif gate_mode == 'hf_only':
            self.gate_lh = _SubbandResidualGate(channels, reduction, alpha, gate_fn, use_abs_stat)
            self.gate_hl = _SubbandResidualGate(channels, reduction, alpha, gate_fn, use_abs_stat)
            self.gate_hh = _SubbandResidualGate(channels, reduction, alpha, gate_fn, use_abs_stat)
        elif gate_mode == 'lh_only':
            self.gate_lh = _SubbandResidualGate(channels, reduction, alpha, gate_fn, use_abs_stat)
        elif gate_mode == 'hl_only':
            self.gate_hl = _SubbandResidualGate(channels, reduction, alpha, gate_fn, use_abs_stat)
        elif gate_mode == 'hh_only':
            self.gate_hh = _SubbandResidualGate(channels, reduction, alpha, gate_fn, use_abs_stat)
        elif gate_mode == 'll_hf':
            self.gate_ll = _SubbandResidualGate(channels, reduction, alpha, gate_fn, use_abs_stat)
            self.gate_hf = _SubbandResidualGate(channels, reduction, alpha, gate_fn, use_abs_stat)

        # -------------------------------------------------------
        # Global residual strength beta
        #
        # beta = beta_max * tanh(beta_raw)
        #
        # 让:
        #     |beta| < beta_max
        #
        # 同时 beta 初始为 beta_init，而不是 0。
        # learnable_beta=False（消融 beta1）: beta 固定为 1.0
        # -------------------------------------------------------
        if learnable_beta:
            beta_ratio = beta_init / beta_max
            beta_raw_init = torch.atanh(
                torch.tensor(beta_ratio, dtype=torch.float32)
            )
            self.beta_raw = nn.Parameter(beta_raw_init.clone())
        elif use_beta:
            # beta1 消融: beta 固定 ≈1.0（tanh(4.0)=0.9993），不参与训练
            self.register_buffer('beta_raw', torch.tensor(4.0))
        # use_beta=False: 无 β 结构（Y = X + ΔX），β 经 α 重标定吸收
        #

        # debug
        self.capture_debug = False
        self.latest_debug = None

    # ------------------------------------------------------------------
    # beta
    # ------------------------------------------------------------------
    def get_beta(self):
        return self.beta_max * torch.tanh(self.beta_raw) if self.use_beta else 1.0

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x):

        # ------------------------------------------------------
        # 1. DWT
        # ------------------------------------------------------
        ll, lh, hl, hh, orig_size = self.basis.decompose(x)

        # ------------------------------------------------------
        # 2. Channel-wise residual coefficient
        #
        # a_i ∈ [-alpha, alpha]
        # 消融模式: shared（四子带共享一门）/ ll_only / hf_only
        # ------------------------------------------------------
        if self.gate_mode == 'shared':
            a_ll = self.gate_ll(ll)
            a_lh = self.gate_ll(lh)
            a_hl = self.gate_ll(hl)
            a_hh = self.gate_ll(hh)
        elif self.gate_mode == 'll_only':
            a_ll = self.gate_ll(ll)
            a_lh = torch.zeros_like(ll)
            a_hl = torch.zeros_like(hl)
            a_hh = torch.zeros_like(hh)
        elif self.gate_mode == 'hf_only':
            a_ll = torch.zeros_like(ll)
            a_lh = self.gate_lh(lh)
            a_hl = self.gate_hl(hl)
            a_hh = self.gate_hh(hh)
        elif self.gate_mode == 'lh_only':
            a_ll = torch.zeros_like(ll)
            a_lh = self.gate_lh(lh)
            a_hl = torch.zeros_like(hl)
            a_hh = torch.zeros_like(hh)
        elif self.gate_mode == 'hl_only':
            a_ll = torch.zeros_like(ll)
            a_lh = torch.zeros_like(lh)
            a_hl = self.gate_hl(hl)
            a_hh = torch.zeros_like(hh)
        elif self.gate_mode == 'hh_only':
            a_ll = torch.zeros_like(ll)
            a_lh = torch.zeros_like(lh)
            a_hl = torch.zeros_like(hl)
            a_hh = self.gate_hh(hh)
        elif self.gate_mode == 'll_hf':
            a_ll = self.gate_ll(ll)
            a_lh = self.gate_hf(lh)
            a_hl = self.gate_hf(hl)
            a_hh = self.gate_hf(hh)
        elif self.gate_mode == 'shuffle':
            # 内容旋转错位: 物理子带与槽位固定错开一位，
            # 破坏频率-槽位语义对应（ll 内容进 lh 槽，依此类推）
            s_ll, s_lh, s_hl, s_hh = hh, ll, lh, hl
            a_ll = self.gate_ll(s_ll)
            a_lh = self.gate_lh(s_lh)
            a_hl = self.gate_hl(s_hl)
            a_hh = self.gate_hh(s_hh)
            ll, lh, hl, hh = s_ll, s_lh, s_hl, s_hh
        else:
            a_ll = self.gate_ll(ll)
            a_lh = self.gate_lh(lh)
            a_hl = self.gate_hl(hl)
            a_hh = self.gate_hh(hh)

        # ------------------------------------------------------
        # 3. Subband residual
        #
        # DeltaW_i = a_i * W_i
        # ------------------------------------------------------
        delta_ll = a_ll * ll
        delta_lh = a_lh * lh
        delta_hl = a_hl * hl
        delta_hh = a_hh * hh

        # ------------------------------------------------------
        # 4. IDWT
        #
        # 注意：
        # 这里 reconstruct 的不是新的 O，
        # 而是 frequency residual DeltaX。
        # ------------------------------------------------------
        delta_x = self.basis.reconstruct(
            delta_ll,
            delta_lh,
            delta_hl,
            delta_hh,
            orig_size,
        )

        # ------------------------------------------------------
        # 5. Global residual strength
        # ------------------------------------------------------
        beta = self.get_beta()

        # ------------------------------------------------------
        # 6. Final residual correction
        # ------------------------------------------------------
        # use_beta=False: 无 β 结构，out = x + delta_x（β 经 α 重标定吸收）
        out = x + (beta * delta_x if self.use_beta else delta_x)

        # ------------------------------------------------------
        # 7. Debug
        # ------------------------------------------------------
        if self.capture_debug:

            with torch.no_grad():

                # 实际频带乘数：
                #
                #   W' = (1 + a)W
                #
                # 而不是只记录 residual coefficient a。
                ll_weight = 1.0 + a_ll
                lh_weight = 1.0 + a_lh
                hl_weight = 1.0 + a_hl
                hh_weight = 1.0 + a_hh

                # correction magnitude
                x_norm = torch.sqrt(
                    torch.mean(
                        x.detach().float() ** 2
                    ) + 1e-8
                )

                dx_norm = torch.sqrt(
                    torch.mean(
                        delta_x.detach().float() ** 2
                    ) + 1e-8
                )

                correction_ratio = (
                    dx_norm / x_norm
                )

            self.latest_debug = {
                # ------------------------------------------------
                # 原始 subband
                # ------------------------------------------------
                'll': _mean_abs_map(ll),
                'lh': _mean_abs_map(lh),
                'hl': _mean_abs_map(hl),
                'hh': _mean_abs_map(hh),

                # ------------------------------------------------
                # residual 后的 subband
                # ------------------------------------------------
                'll_att': _mean_abs_map(delta_ll),
                'lh_att': _mean_abs_map(delta_lh),
                'hl_att': _mean_abs_map(delta_hl),
                'hh_att': _mean_abs_map(delta_hh),

                # ------------------------------------------------
                # 实际乘数:
                #
                # 1 + a
                #
                # 范围:
                # [1-alpha, 1+alpha]
                # ------------------------------------------------
                'll_weight': _cpu_detach(
                    ll_weight.reshape(x.shape[0], -1)
                ),

                'hf_weight': _cpu_detach(
                    torch.cat(
                        [
                            lh_weight,
                            hl_weight,
                            hh_weight
                        ],
                        dim=1
                    ).reshape(
                        x.shape[0],
                        3,
                        -1
                    )
                ),

                # ------------------------------------------------
                # 新增：真正的 residual coefficient
                # ------------------------------------------------
                'll_residual': _cpu_detach(
                    a_ll.reshape(x.shape[0], -1)
                ),

                'hf_residual': _cpu_detach(
                    torch.cat(
                        [
                            a_lh,
                            a_hl,
                            a_hh
                        ],
                        dim=1
                    ).reshape(
                        x.shape[0],
                        3,
                        -1
                    )
                ),

                # ------------------------------------------------
                # 全局 correction strength
                # ------------------------------------------------
                'beta': beta.detach().float().cpu(),

                # ------------------------------------------------
                # residual magnitude
                # ------------------------------------------------
                'correction_ratio': (
                    correction_ratio.detach()
                    .float()
                    .cpu()
                ),
            }

        else:
            self.latest_debug = None

        return out

    def initialize(self):
        # 子模块已经显式初始化；
        # beta 也已经在 __init__ 中初始化。
        pass

    def extra_repr(self):
        return (
            f'channels={self.channels}, '
            f'basis={self.basis_name}, '
            f'alpha={self.alpha}, '
            f'beta_max={self.beta_max}'
        )


def build_lwa(mode, channels, basis='haar', reduction=4):
    """按 lwa_mode 构造解码器小波注意力: gate / se / se_res / wca_plus / res_wca"""
    normalized = str(mode).strip().lower()
    if normalized in ('gate', 'lwa'):
        return LearnableWaveletAttention(channels, basis=basis)
    if normalized == 'se':
        return LearnableWaveletAttentionSE(channels, basis=basis, reduction=reduction)
    if normalized == 'se_res':
        return LearnableWaveletAttentionSERes(channels, basis=basis, reduction=reduction)
    if normalized in ('wca_plus', 'wca+'):
        return LearnableWaveletAttentionWCAPlus(channels, reduction=8)
    if normalized in ('res_wca', 'reswca', 'rwca_plus', 'rwca+'):
        # RWCA+: WCA+ 结构 + 差模残差 Y = X + gamma*(O-X)，
        # 实现见 LearnableWaveletAttentionResWCA 与独立文件 src/RWCA+.py
        return LearnableWaveletAttentionResWCA(channels, reduction=8)
    if normalized in ('rwca', 'rwca_v2', 'rwca_channel', 'subband_rwca'):
        # 新版 RWCA:
        # 逐子带通道级带符号残差门 a=alpha*tanh(MLP(|W|))，
        # Y = X + beta*IDWT(a_i*W_i)，恒等起步（a=0）且 beta 有界
        return LearnableWaveletAttentionRWCA(
            channels=channels,
            basis=basis,
            reduction=reduction,
            alpha=0.5,
            beta_init=0.25,
            beta_max=1.0,
        )
    # ---- RWCA 消融变体----
    if normalized in ('rwca_ll',):
        return LearnableWaveletAttentionRWCA(
            channels=channels, basis=basis, reduction=reduction,
            alpha=0.5, beta_init=0.25, beta_max=1.0, gate_mode='ll_only')
    if normalized in ('rwca_hf',):
        return LearnableWaveletAttentionRWCA(
            channels=channels, basis=basis, reduction=reduction,
            alpha=0.5, beta_init=0.25, beta_max=1.0, gate_mode='hf_only')
    if normalized in ('rwca_shared',):
        return LearnableWaveletAttentionRWCA(
            channels=channels, basis=basis, reduction=reduction,
            alpha=0.5, beta_init=0.25, beta_max=1.0, gate_mode='shared')
    if normalized in ('rwca_sigmoid',):
        return LearnableWaveletAttentionRWCA(
            channels=channels, basis=basis, reduction=reduction,
            alpha=0.5, beta_init=0.25, beta_max=1.0, gate_fn='sigmoid')
    if normalized in ('rwca_nostat',):
        return LearnableWaveletAttentionRWCA(
            channels=channels, basis=basis, reduction=reduction,
            alpha=0.5, beta_init=0.25, beta_max=1.0, use_abs_stat=False)
    if normalized in ('rwca_beta1',):
        return LearnableWaveletAttentionRWCA(
            channels=channels, basis=basis, reduction=reduction,
            alpha=0.5, beta_init=0.25, beta_max=1.0, learnable_beta=False)
    if normalized in ('rwca_nobeta', 'rwca_no_beta', 'rwca_beta_removed'):
        # RWCA 无 β: 与 β≡1 精确等价的重参数化——
        # β₀=tanh(4.0)=0.9993293 经 IDWT 线性性吸收进 α: α′=β₀·0.5=0.49966465
        return LearnableWaveletAttentionRWCA(
            channels=channels, basis=basis, reduction=reduction,
            alpha=0.49966465, beta_init=0.25, beta_max=1.0,
            learnable_beta=False, use_beta=False)
    # ---- RWCA 无 β 消融变体----
    _NB = dict(learnable_beta=False, use_beta=False)
    if normalized in ('rwca_ll_nb',):
        return LearnableWaveletAttentionRWCA(
            channels=channels, basis=basis, reduction=reduction,
            alpha=0.5, beta_init=0.25, beta_max=1.0, gate_mode='ll_only', **_NB)
    if normalized in ('rwca_hf_nb',):
        return LearnableWaveletAttentionRWCA(
            channels=channels, basis=basis, reduction=reduction,
            alpha=0.5, beta_init=0.25, beta_max=1.0, gate_mode='hf_only', **_NB)
    if normalized in ('rwca_shared_nb',):
        return LearnableWaveletAttentionRWCA(
            channels=channels, basis=basis, reduction=reduction,
            alpha=0.5, beta_init=0.25, beta_max=1.0, gate_mode='shared', **_NB)
    if normalized in ('rwca_sigmoid_nb',):
        return LearnableWaveletAttentionRWCA(
            channels=channels, basis=basis, reduction=reduction,
            alpha=0.5, beta_init=0.25, beta_max=1.0, gate_fn='sigmoid', **_NB)
    if normalized in ('rwca_nostat_nb',):
        return LearnableWaveletAttentionRWCA(
            channels=channels, basis=basis, reduction=reduction,
            alpha=0.5, beta_init=0.25, beta_max=1.0, use_abs_stat=False, **_NB)
    for _a, _sfx in ((0.3, 'a03'), (0.4, 'a04'), (0.6, 'a06'), (0.7, 'a07')):
        if normalized == f'rwca_{_sfx}_nb':
            return LearnableWaveletAttentionRWCA(
                channels=channels, basis=basis, reduction=reduction,
                alpha=_a, beta_init=0.25, beta_max=1.0, **_NB)
    # ---- single-band gates, alpha endpoints, shuffle, scalar and spatial controls ----
    _NB_SINGLE = {'lh_only': 'lh', 'hl_only': 'hl', 'hh_only': 'hh'}
    for _gm, _sfx in _NB_SINGLE.items():
        if normalized == f'rwca_{_sfx}_nb':
            return LearnableWaveletAttentionRWCA(
                channels=channels, basis=basis, reduction=reduction,
                alpha=0.5, beta_init=0.25, beta_max=1.0,
                gate_mode=_gm, **_NB)
    if normalized in ('rwca_llhf_nb', 'rwca_ll_hf_nb'):
        return LearnableWaveletAttentionRWCA(
            channels=channels, basis=basis, reduction=reduction,
            alpha=0.5, beta_init=0.25, beta_max=1.0,
            gate_mode='ll_hf', **_NB)
    if normalized in ('rwca_shuffle_nb',):
        return LearnableWaveletAttentionRWCA(
            channels=channels, basis=basis, reduction=reduction,
            alpha=0.5, beta_init=0.25, beta_max=1.0,
            gate_mode='shuffle', **_NB)
    if normalized in ('rwca_scalar_nb',):
        return LearnableWaveletAttentionRWCAScalar(channels, basis=basis, alpha=0.5)
    if normalized in ('rwca_spatial_ctl', 'spatial_control', 'rwca_spatial_nb'):
        return LearnableWaveletAttentionSpatialControl(channels, reduction=reduction, alpha=0.5)
    for _a, _sfx in ((0.2, 'a02'), (1.0, 'a10')):
        if normalized == f'rwca_{_sfx}_nb':
            return LearnableWaveletAttentionRWCA(
                channels=channels, basis=basis, reduction=reduction,
                alpha=_a, beta_init=0.25, beta_max=1.0, **_NB)
    if normalized in ('rwca_gca', 'rwca_plus_gca', 'reswca_gca'):
        # RWCA-GCA: RWCA+ 差模残差 + 全局内容自适应
        #（HH 静态标量门 → MLP(|HH|) 通道门；γ 无界 init 0 不变）
        return LearnableWaveletAttentionResWCAGCA(channels, reduction=8)
    if normalized in ('rwca_a07', 'rwca07', 'rwca_0.7'):
        # α 扫描: |a| 界放宽至 0.7
        return LearnableWaveletAttentionRWCA(
            channels=channels,
            basis=basis,
            reduction=reduction,
            alpha=0.7,
            beta_init=0.25,
            beta_max=1.0,
        )
    if normalized in ('rwca_a03', 'rwca03', 'rwca_0.3'):
        # α 扫描: |a| 界收紧至 0.3
        return LearnableWaveletAttentionRWCA(
            channels=channels,
            basis=basis,
            reduction=reduction,
            alpha=0.3,
            beta_init=0.25,
            beta_max=1.0,
        )
    if normalized in ('rwca_a04', 'rwca04', 'rwca_0.4'):
        # α 扫描补全: |a| 界 0.4
        return LearnableWaveletAttentionRWCA(
            channels=channels,
            basis=basis,
            reduction=reduction,
            alpha=0.4,
            beta_init=0.25,
            beta_max=1.0,
        )
    if normalized in ('rwca_a06', 'rwca06', 'rwca_0.6'):
        # α 扫描补全: |a| 界 0.6
        return LearnableWaveletAttentionRWCA(
            channels=channels,
            basis=basis,
            reduction=reduction,
            alpha=0.6,
            beta_init=0.25,
            beta_max=1.0,
        )
    raise ValueError(
        f'Unknown lwa_mode: {mode}. Supported: '
        f'gate / se / se_res / wca_plus / res_wca / rwca_plus / rwca')


# ---------------------------------------------------------------------------
# Decoder Blocks
# ---------------------------------------------------------------------------

class DB1(nn.Module):
    def __init__(self, inplanes, outplanes, use_wavelet=False, wavelet_basis='haar', lwa_mode='gate'):
        super(DB1, self).__init__()
        self.use_wavelet = use_wavelet
        self.latest_debug = None
        self.squeeze1 = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.squeeze2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.lwa = build_lwa(lwa_mode, outplanes, wavelet_basis) if use_wavelet else None

    def forward(self, x):
        z = self.squeeze2(self.squeeze1(x))
        self.latest_debug = None
        if self.use_wavelet and self.lwa is not None:
            z = self.lwa(z)
            self.latest_debug = self.lwa.latest_debug
        return z, z

    def initialize(self):
        weight_init(self)


class DB2(nn.Module):
    def __init__(self, inplanes, outplanes, use_wavelet=True, wavelet_basis='haar', lwa_mode='gate'):
        super(DB2, self).__init__()
        self.use_wavelet = use_wavelet
        self.latest_debug = None
        self.short_cut = nn.Conv2d(outplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes + outplanes, outplanes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes), nn.ReLU(inplace=True),
            nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes), nn.ReLU(inplace=True)
        )
        self.lwa = build_lwa(lwa_mode, outplanes, wavelet_basis)
        self.conv2 = nn.Sequential(
            nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes), nn.ReLU(inplace=True),
            nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes), nn.ReLU(inplace=True)
        )

    def forward(self, x, z):
        z = F.interpolate(z, size=x.size()[2:], mode='bilinear', align_corners=True)
        p = self.conv(torch.cat((x, z), 1))
        sc = self.short_cut(z)
        p = p + sc
        self.latest_debug = None
        if self.use_wavelet:
            p = self.lwa(p)
            self.latest_debug = self.lwa.latest_debug
        p2 = self.conv2(p)
        p = p + p2
        return p, p

    def initialize(self):
        weight_init(self)


class DB3(nn.Module):
    def __init__(self, use_wavelet=True, wavelet_basis='haar', lwa_mode='gate',
                 s1_channels=128, r4_channels=256):
        super(DB3, self).__init__()
        self.db2 = DB2(64, 64, use_wavelet=use_wavelet, wavelet_basis=wavelet_basis,
                       lwa_mode=lwa_mode)
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.sqz_r4 = nn.Sequential(
            nn.Conv2d(r4_channels, 64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.sqz_s1 = nn.Sequential(
            nn.Conv2d(s1_channels, 64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )

    def forward(self, s, r, up):
        up = F.interpolate(up, size=s.size()[2:], mode='bilinear', align_corners=True)
        s = self.sqz_s1(s)
        r = self.sqz_r4(r)
        sr = self.conv3x3(s + r)
        out, _ = self.db2(sr, up)
        return out, out

    def initialize(self):
        weight_init(self)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class decoder(nn.Module):
    def __init__(self, use_wavelet_decoder=True, wca_variant='baseline', wavelet_basis='haar',
                 fusion_mode='cgm', lwa_mode='gate',
                 resnet_channels=(64, 128, 256, 512),
                 swin_channels=(128, 256, 512, 512)):
        super(decoder, self).__init__()
        self.latest_debug = None
        self.wca_variant, self.wca_plan = resolve_decoder_wca_plan(use_wavelet_decoder, wca_variant)
        self.wavelet_basis = wavelet_basis
        self.fusion_mode = fusion_mode
        self.lwa_mode = lwa_mode
        r2_ch, r3_ch, r4_ch, r5_ch = resnet_channels
        s1_ch, s2_ch, s3_ch, s4_ch = swin_channels
        self.sqz_s2 = nn.Sequential(
            nn.Conv2d(s2_ch, 64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.sqz_r5 = nn.Sequential(
            nn.Conv2d(r5_ch, 64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.GF = build_fusion(fusion_mode, dim=64, num_heads=8)
        self.d1 = DB1(s4_ch, 64, use_wavelet=self.wca_plan['d1'], wavelet_basis=wavelet_basis, lwa_mode=lwa_mode)
        self.d2 = DB2(s3_ch, 64, use_wavelet=self.wca_plan['d2'], wavelet_basis=wavelet_basis, lwa_mode=lwa_mode)
        self.d3 = DB2(64, 64, use_wavelet=self.wca_plan['d3'], wavelet_basis=wavelet_basis, lwa_mode=lwa_mode)
        self.d4 = DB3(use_wavelet=self.wca_plan['d4'], wavelet_basis=wavelet_basis, lwa_mode=lwa_mode,
                      s1_channels=s1_ch, r4_channels=r4_ch)
        self.d5 = DB2(r3_ch, 64, use_wavelet=self.wca_plan['d5'], wavelet_basis=wavelet_basis, lwa_mode=lwa_mode)
        self.d6 = DB2(r2_ch, 64, use_wavelet=self.wca_plan['d6'], wavelet_basis=wavelet_basis, lwa_mode=lwa_mode)
        self.wca_blocks = {
            'd1': self.d1, 'd2': self.d2, 'd3': self.d3,
            'd4': self.d4.db2, 'd5': self.d5, 'd6': self.d6,
        }

    def set_wavelet_debug(self, enabled):
        for block in self.wca_blocks.values():
            if getattr(block, 'lwa', None) is not None:
                block.lwa.capture_debug = enabled and getattr(block, 'use_wavelet', False)
                if not enabled:
                    block.lwa.latest_debug = None
            if not enabled:
                block.latest_debug = None
        if not enabled:
            self.latest_debug = None

    def forward(self, s1, s2, s3, s4, r2, r3, r4, r5):
        r5 = F.interpolate(r5, size=s2.size()[2:], mode='bilinear', align_corners=True)
        s1 = F.interpolate(s1, size=r4.size()[2:], mode='bilinear', align_corners=True)

        s4_, _ = self.d1(s4)
        s3_, _ = self.d2(s3, s4_)

        s2_ = self.sqz_s2(s2)
        r5_ = self.sqz_r5(r5)
        graft_feature_r5, ctam = self.GF(r5_, s2_)

        graft_feature_r5_, _ = self.d3(graft_feature_r5, s3_)
        graft_feature_r4, _ = self.d4(s1, r4, graft_feature_r5_)
        r3_, _ = self.d5(r3, graft_feature_r4)
        r2_, _ = self.d6(r2, r3_)

        self.latest_debug = {
            'd1': self.d1.latest_debug, 'd2': self.d2.latest_debug,
            'd3': self.d3.latest_debug, 'd4': self.d4.db2.latest_debug,
            'd5': self.d5.latest_debug, 'd6': self.d6.latest_debug,
        }
        return r2_, ctam, r5_, s2_

    def initialize(self):
        weight_init(self)


# ---------------------------------------------------------------------------
# 骨干通道映射（实证: Swin 返回 4 路 [s1..s4]，见smoke test记录）
# ---------------------------------------------------------------------------

RESNET_OUT_CHANNELS = {
    'resnet18': (64, 128, 256, 512),
    'resnet50': (256, 512, 1024, 2048),
}
SWIN_OUT_CHANNELS = {
    'swin_base': (128, 256, 512, 512),
    'swin_tiny': (96, 192, 384, 384),   # 2026-08-17 smoke test实证确认
}


# ---------------------------------------------------------------------------
# WPUGNet — 主模型
# ---------------------------------------------------------------------------

class WPUGNet(nn.Module):
    """WPUGNet: 超声图像分割基础模型

    双骨干 (ResNet18 + Swin-Base) + Cross-Grafting 融合 + 可选小波增强。
    """

    def __init__(self, cfg=None):
        super(WPUGNet, self).__init__()
        self.cfg = cfg

        # Config 读取（使用 _cfg_get 修复 Config.__getattr__ bug）
        self.task_mode = _cfg_get(cfg, 'task_mode', 'seg')
        self.num_classes = _cfg_get(cfg, 'num_classes', 0)
        self.cls_use_mask = _cfg_get(cfg, 'cls_use_mask', True)
        self.disable_reciprocal_cls_head = _cfg_get(cfg, 'disable_reciprocal_cls_head', False)
        self.use_wavelet_stem = _cfg_get(cfg, 'use_wavelet_stem', True)
        self.use_decoder_wavelet = _cfg_get(cfg, 'use_decoder_wavelet', True)
        self.decoder_wca_variant = _cfg_get(cfg, 'decoder_wca_variant', 'baseline')
        self.decoder_wavelet_basis = _cfg_get(cfg, 'wavelet_basis', 'haar')
        self.cgm_mode = _cfg_get(cfg, 'cgm_mode', 'cgm')
        self.lwa_mode = _cfg_get(cfg, 'lwa_mode', 'gate')
        self.backbone_r = _cfg_get(cfg, 'backbone_r', 'resnet18')
        self.backbone_s = _cfg_get(cfg, 'backbone_s', 'swin_base')
        if self.backbone_r not in RESNET_OUT_CHANNELS:
            raise ValueError(f'Unknown backbone_r: {self.backbone_r}. '
                             f'Supported: {list(RESNET_OUT_CHANNELS)}')
        if self.backbone_s not in SWIN_OUT_CHANNELS:
            raise ValueError(f'Unknown backbone_s: {self.backbone_s}. '
                             f'Supported: {list(SWIN_OUT_CHANNELS)}')
        self.export_wavelet_debug = _cfg_get(cfg, 'export_wavelet_debug', False)
        self.latest_wavelet_debug = None
        self.enable_cls = (
            not self.disable_reciprocal_cls_head
            and self.num_classes is not None
            and self.num_classes > 0
            and self.task_mode in ['joint', 'cls']
        )

        # WaveletStem（可选）
        if self.use_wavelet_stem:
            self.wavelet = WaveletStem()
            self.wavelet.capture_debug = self.export_wavelet_debug
        else:
            self.wavelet = None

        # Decoder
        self.decoder = decoder(
            use_wavelet_decoder=self.use_decoder_wavelet,
            wca_variant=self.decoder_wca_variant,
            wavelet_basis=self.decoder_wavelet_basis,
            fusion_mode=self.cgm_mode,
            lwa_mode=self.lwa_mode,
            resnet_channels=RESNET_OUT_CHANNELS[self.backbone_r],
            swin_channels=SWIN_OUT_CHANNELS[self.backbone_s]
        )
        self.decoder_wca_variant = self.decoder.wca_variant
        self.decoder.set_wavelet_debug(self.export_wavelet_debug)

        # Prediction heads
        self.linear1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.conv    = nn.Conv2d(8,  1, kernel_size=3, stride=1, padding=1)
        if self.enable_cls:
            self.cls_head = nn.Linear(64, self.num_classes)

        if self.cfg is None or self.cfg.snapshot is None:
            weight_init(self)

        # Backbones
        _resnet_in = 12 if self.use_wavelet_stem else 3
        if self.backbone_r == 'resnet50':
            self.resnet = resnet50(in_channels=_resnet_in)
        else:
            self.resnet = resnet18(in_channels=_resnet_in)

        if self.backbone_s == 'swin_tiny':
            self.swin = Swintransformer(224, embed_dim=96, depths=[2, 2, 6, 2],
                                        num_heads=[3, 6, 12, 24], drop_path_rate=0.2)
        else:
            self.swin = Swintransformer(224)

        # ---- 加载预训练权重 ----
        _src_dir = os.path.dirname(os.path.abspath(__file__))
        _pre_dir = os.path.join(_src_dir, '..', 'pre')
        _swin_path = os.path.join(
            _pre_dir, 'swin_tiny224.pth' if self.backbone_s == 'swin_tiny' else 'swin224.pth')
        self.swin.load_state_dict(
            torch.load(_swin_path, map_location='cpu', weights_only=False)['model'],
            strict=False
        )

        _res_path = os.path.join(
            _pre_dir, 'resnet50.pth' if self.backbone_r == 'resnet50' else 'resnet18.pth')
        resnet_ckpt = torch.load(_res_path, map_location='cpu', weights_only=False)
        if self.use_wavelet_stem:
            w_orig = resnet_ckpt['conv1.weight']
            w_new = w_orig.repeat(1, 4, 1, 1) * 0.25
            resnet_ckpt['conv1.weight'] = w_new
        self.resnet.load_state_dict(resnet_ckpt, strict=False)

        # Checkpoint 恢复
        if self.cfg is not None and self.cfg.snapshot:
            print('load checkpoint')
            pretrain = torch.load(self.cfg.snapshot, map_location='cpu', weights_only=False)
            if isinstance(pretrain, dict) and 'model_state_dict' in pretrain:
                pretrain = pretrain['model_state_dict']
            elif isinstance(pretrain, dict) and 'state_dict' in pretrain:
                pretrain = pretrain['state_dict']
            new_state_dict = {}
            for k, v in pretrain.items():
                new_key = k[7:] if k.startswith('module.') else k
                new_state_dict[new_key] = v
            missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
            print(f'checkpoint loaded: missing={len(missing)}, unexpected={len(unexpected)}')

    def forward(self, x, shape=None, mask=None):
        shape = x.size()[2:] if shape is None else shape

        # WaveletStem（可选）
        if self.use_wavelet_stem:
            x_wav = self.wavelet(x)
        else:
            x_wav = x

        y = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)

        r2, r3, r4, r5 = self.resnet(x_wav)
        s1, s2, s3, s4 = self.swin(y)

        r2_, ctam, r5_, s2_ = self.decoder(s1, s2, s3, s4, r2, r3, r4, r5)

        pred1 = F.interpolate(self.linear1(r2_), size=shape, mode='bilinear')
        wr    = F.interpolate(self.linear2(r5_), size=(28, 28), mode='bilinear')
        ws    = F.interpolate(self.linear3(s2_), size=(28, 28), mode='bilinear')

        cls_logits = None
        if self.enable_cls:
            if self.cls_use_mask:
                att = torch.sigmoid(pred1).detach()
                denom = att.flatten(2).sum(-1, keepdim=True).clamp(min=1e-6)
                pooled = (r2_ * att).flatten(2).sum(-1) / denom
            else:
                pooled = F.adaptive_avg_pool2d(r2_, 1).flatten(1)
            cls_logits = self.cls_head(pooled)

        if self.export_wavelet_debug:
            self.latest_wavelet_debug = {
                'stem': self.wavelet.latest_debug if self.wavelet is not None else None,
                'decoder': self.decoder.latest_debug,
            }
        else:
            self.latest_wavelet_debug = None

        attmap = self.conv(ctam) if ctam is not None else None
        return pred1, wr, ws, attmap, cls_logits
