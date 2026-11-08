#!/usr/bin/python3
# coding=utf-8
# Simplified single-GPU training script using PyTorch native AMP (No Apex, No DDP)

import sys
import datetime
import argparse
import os
import json
import time
import cv2


# WPUGNet 超声图像分割基础模型
TASK_PATH = './src'

sys.path.insert(0, TASK_PATH)
sys.dont_write_bytecode = True

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler  # PyTorch native AMP
import numpy as np

# 全局 AMP dtype（train() 按 --amp_dtype 设置；bf16 用于外部数据 FT 防 fp16 激活溢出 NaN，2026-09-10）
AMP_DTYPE = torch.float16

# 导入任务相关模块
try:
    import dataset
    from wpugnet import WPUGNet
    from utils.lr_scheduler import LR_Scheduler

    print(f"[OK] 成功导入模块，使用任务路径: {TASK_PATH}")
except ImportError as e:
    print(f"[ERROR] 导入失败: {e}")
    print(f"请检查TASK_PATH是否正确: {TASK_PATH}")
    sys.exit(1)


# ============ 损失函数 ============

def flat(mask):
    """ground truth attention map"""
    batch_size = mask.shape[0]
    h = 28
    mask = F.interpolate(mask, size=(int(h), int(h)), mode='bilinear')
    x = mask.view(batch_size, 1, -1).permute(0, 2, 1)
    g = x @ x.transpose(-2, -1)
    gt_att = g.unsqueeze(1)
    return gt_att


def att_loss_fn(pred, mask, p4, p5):
    g = flat(mask)
    np4 = torch.sigmoid(p4.detach())
    np5 = torch.sigmoid(p5.detach())
    p4 = flat(np4)
    p5 = flat(np5)
    w1 = torch.abs(g - p4)
    w2 = torch.abs(g - p5)
    w = (w1 + w2) * 0.5 + 1
    attbce = F.binary_cross_entropy_with_logits(pred, g, weight=w * 1.0, reduction='mean')
    return attbce


def bce_iou_loss(pred, mask):
    """iou loss for global structure"""
    size = pred.size()[2:]
    mask = F.interpolate(mask, size=size, mode='bilinear')
    wbce = F.binary_cross_entropy_with_logits(pred, mask)
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_loss(pred, mask, eps=1e-6):
    """soft dice loss based on logits"""
    size = pred.size()[2:]
    mask = F.interpolate(mask, size=size, mode='nearest')
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def soft_erode(x, kernel_size=3):
    pad = kernel_size // 2
    return -F.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=pad)


def boundary_map(x, kernel_size=3):
    x = x.clamp(0.0, 1.0)
    return (x - soft_erode(x, kernel_size=kernel_size)).clamp(0.0, 1.0)


def boundary_iou_loss(pred, mask, kernel_size=3, eps=1e-6):
    """soft boundary IoU loss based on logits"""
    size = pred.size()[2:]
    mask = F.interpolate(mask, size=size, mode='nearest')
    pred_boundary = boundary_map(torch.sigmoid(pred), kernel_size=kernel_size)
    mask_boundary = boundary_map(mask, kernel_size=kernel_size)
    inter = (pred_boundary * mask_boundary).sum(dim=(1, 2, 3))
    union = (pred_boundary + mask_boundary - pred_boundary * mask_boundary).sum(dim=(1, 2, 3))
    return (1 - (inter + eps) / (union + eps)).mean()


def segmentation_loss(pred, mask, cfg):
    mode = getattr(cfg, 'seg_loss_mode', 'bce_iou')
    if mode == 'dice':
        return dice_loss(pred, mask)
    if mode == 'dice_biou':
        dice_w = getattr(cfg, 'dice_weight', 0.6)
        biou_w = getattr(cfg, 'boundary_iou_weight', 0.4)
        kernel = getattr(cfg, 'boundary_kernel', 3)
        return dice_w * dice_loss(pred, mask) + biou_w * boundary_iou_loss(pred, mask, kernel_size=kernel)
    return bce_iou_loss(pred, mask)


def resolve_selection_metric(args, has_val_loader, val_loss=None, val_dice=None, train_loss=None):
    if has_val_loader:
        if args.best_by == 'val_loss':
            metric_val = float(val_loss) if val_loss is not None else float('inf')
            return -metric_val, f'val_loss={metric_val:.4f}'
        metric_val = float(val_dice) if val_dice is not None else -float('inf')
        return metric_val, f'val_dice={metric_val:.4f}'
    metric_val = float(train_loss) if train_loss is not None else float('inf')
    return -metric_val, f'train_loss={metric_val:.4f}'


def tensor_to_uint8_map(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    x = np.squeeze(x)
    if x.ndim == 0:
        x = np.array([[float(x)]], dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_min, x_max = float(x.min()), float(x.max())
    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)
    else:
        x = np.zeros_like(x, dtype=np.float32)
    return (x * 255.0).clip(0, 255).astype(np.uint8)


def map_to_heatmap_bgr(x):
    return cv2.applyColorMap(tensor_to_uint8_map(x), cv2.COLORMAP_JET)


def vector_to_heatmap_bgr(x, out_hw=(96, 192)):
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    x = np.squeeze(x)
    if x.ndim == 1:
        x = x[None, :]
    x_img = tensor_to_uint8_map(x)
    x_img = cv2.resize(x_img, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(x_img, cv2.COLORMAP_JET)


def mask_to_bgr(x):
    gray = tensor_to_uint8_map(x)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def denormalize_image_bgr(image_tensor):
    if isinstance(image_tensor, torch.Tensor):
        image_tensor = image_tensor.detach().float().cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    img = image_tensor * std + mean
    img = np.clip(img, 0.0, 1.0)
    img = (img.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def resize_tiles(tiles, tile_hw=(192, 192)):
    return [cv2.resize(tile, (tile_hw[1], tile_hw[0]), interpolation=cv2.INTER_LINEAR) for tile in tiles]


def build_stage_stats(stage_debug):
    stats = {}
    for key in ['ll', 'lh', 'hl', 'hh', 'll_att', 'lh_att', 'hl_att', 'hh_att']:
        if key in stage_debug and stage_debug[key] is not None:
            arr = stage_debug[key].numpy()
            stats[f'{key}_mean'] = float(arr.mean())
            stats[f'{key}_std'] = float(arr.std())
    if stage_debug.get('ll_weight') is not None:
        ll_weight = stage_debug['ll_weight'].numpy()
        stats['ll_weight_mean'] = float(ll_weight.mean())
        stats['ll_weight_std'] = float(ll_weight.std())
    if stage_debug.get('hf_weight') is not None:
        hf_weight = stage_debug['hf_weight'].numpy()
        stats['hf_lh_weight_mean'] = float(hf_weight[:, 0, :].mean())
        stats['hf_hl_weight_mean'] = float(hf_weight[:, 1, :].mean())
        stats['hf_hh_weight_mean'] = float(hf_weight[:, 2, :].mean())
    return stats


def save_wavelet_debug_visuals(args, net, image, mask, pred_logits, stage_name, epoch_idx, step_idx, vis_counters):
    if not getattr(args, 'export_wavelet_debug', False):
        return None
    max_batches = max(0, int(getattr(args, 'vis_max_batches', 0)))
    counter_key = f'{stage_name}_ep{epoch_idx+1}'
    current_count = vis_counters.get(counter_key, 0)
    if max_batches > 0 and current_count >= max_batches:
        return None
    model_ref = net.module if hasattr(net, 'module') else net
    debug_payload = getattr(model_ref, 'latest_wavelet_debug', None)
    if not debug_payload:
        return None

    vis_root = args.vis_save_dir or os.path.join(args.savepath, 'wavelet_vis')
    os.makedirs(vis_root, exist_ok=True)
    sample_dir = os.path.join(vis_root, f'{stage_name}_ep{epoch_idx+1:03d}_step{step_idx:03d}')
    os.makedirs(sample_dir, exist_ok=True)

    image_bgr = denormalize_image_bgr(image[0])
    mask_bgr = mask_to_bgr(mask[0])
    pred_bgr = mask_to_bgr(torch.sigmoid(pred_logits[0]))

    stem_debug = debug_payload.get('stem') or {}
    stem_tiles = resize_tiles([
        image_bgr,
        mask_bgr,
        pred_bgr,
        map_to_heatmap_bgr(stem_debug.get('ll', torch.zeros(1, 1, 8, 8))[0]),
        map_to_heatmap_bgr(stem_debug.get('lh', torch.zeros(1, 1, 8, 8))[0]),
        map_to_heatmap_bgr(stem_debug.get('hl', torch.zeros(1, 1, 8, 8))[0]),
        map_to_heatmap_bgr(stem_debug.get('hh', torch.zeros(1, 1, 8, 8))[0]),
    ])
    cv2.imwrite(os.path.join(sample_dir, 'overview_stem.png'), np.hstack(stem_tiles))

    summary = {
        'stage': stage_name,
        'epoch': int(epoch_idx + 1),
        'step': int(step_idx),
        'stem': build_stage_stats(stem_debug) if stem_debug else {},
        'decoder': {}
    }

    decoder_debug = debug_payload.get('decoder') or {}
    for block_name, stage_debug in decoder_debug.items():
        if not stage_debug:
            continue
        row1 = resize_tiles([
            map_to_heatmap_bgr(stage_debug['ll'][0]),
            map_to_heatmap_bgr(stage_debug['lh'][0]),
            map_to_heatmap_bgr(stage_debug['hl'][0]),
            map_to_heatmap_bgr(stage_debug['hh'][0]),
        ])
        row2 = resize_tiles([
            map_to_heatmap_bgr(stage_debug['ll_att'][0]),
            map_to_heatmap_bgr(stage_debug['lh_att'][0]),
            map_to_heatmap_bgr(stage_debug['hl_att'][0]),
            map_to_heatmap_bgr(stage_debug['hh_att'][0]),
        ])
        row3 = resize_tiles([
            vector_to_heatmap_bgr(stage_debug['ll_weight'][0]),
            vector_to_heatmap_bgr(stage_debug['hf_weight'][0, 0]),
            vector_to_heatmap_bgr(stage_debug['hf_weight'][0, 1]),
            vector_to_heatmap_bgr(stage_debug['hf_weight'][0, 2]),
        ])
        grid = np.vstack([np.hstack(row1), np.hstack(row2), np.hstack(row3)])
        cv2.imwrite(os.path.join(sample_dir, f'{block_name}_wca.png'), grid)
        summary['decoder'][block_name] = build_stage_stats(stage_debug)

    with open(os.path.join(sample_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    vis_counters[counter_key] = current_count + 1
    print(f'[VIS] saved wavelet debug to: {sample_dir}')
    return sample_dir


# ============ 验证评估 ============



def compute_dice_f1(pred_logits, masks):
    """
    pred_logits: (B, 1, H, W) float tensor (raw logits)
    masks:       (B, 1, H, W) float tensor [0,1]
    返回 batch 平均 Dice 和 F1（像素二分类，阈值 0.5）
    """
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    masks_bin = (masks > 0.5).float()
    # resize pred to mask size if needed
    if pred.shape != masks_bin.shape:
        pred = F.interpolate(pred, size=masks_bin.shape[2:], mode='bilinear')
        pred = (pred > 0.5).float()
    tp = (pred * masks_bin).sum(dim=(1, 2, 3))
    fp = (pred * (1 - masks_bin)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * masks_bin).sum(dim=(1, 2, 3))
    dice = (2 * tp / (2 * tp + fp + fn + 1e-7)).mean().item()
    prec = (tp / (tp + fp + 1e-7)).mean().item()
    rec  = (tp / (tp + fn + 1e-7)).mean().item()
    f1   = 2 * prec * rec / (prec + rec + 1e-7)
    return dice, f1


@torch.no_grad()
def validate(net, val_loader, device, cfg, args=None, vis_counters=None, epoch_idx=0, stage_name='val'):
    """在验证集上计算 avg_loss / Dice / F1"""
    net.eval()
    total_loss = 0.0
    all_dice, all_f1 = [], []
    processed_steps = 0

    for step, (image, mask, label) in enumerate(val_loader):
        image = image.float().to(device)
        mask  = mask.float().to(device)
        label = label.to(device)

        with autocast(dtype=AMP_DTYPE):
            p1, wr, ws, attmap, cls_logits = net(image)
            seg_loss = torch.tensor(0.0, device=device)
            att_loss = torch.tensor(0.0, device=device)
            if cfg.task_mode in ['seg', 'joint']:
                use_wavelet_stem = cfg.kwargs.get('use_wavelet_stem', True)
                use_dec_wavelet = cfg.kwargs.get('use_decoder_wavelet', True)
                wavelet_on = use_wavelet_stem and use_dec_wavelet
                att_loss = (att_loss_fn(attmap, mask, wr, ws)
                            if wavelet_on and attmap is not None else torch.tensor(0.0, device=device))
                loss1    = segmentation_loss(p1, mask, cfg)
                loss2    = (segmentation_loss(wr, mask, cfg) * 0.125 + segmentation_loss(ws, mask, cfg) * 0.125
                            if wavelet_on else torch.tensor(0.0, device=device))
                seg_loss = loss1 + loss2 + att_loss

            cls_loss = torch.tensor(0.0, device=device)
            if cfg.task_mode in ['cls', 'joint'] and cls_logits is not None:
                labeled = label >= 0
                if labeled.any():
                    cls_loss = F.cross_entropy(cls_logits[labeled], label[labeled])
            loss = seg_loss + cfg.cls_weight * cls_loss

        total_loss += loss.item()
        d, f = compute_dice_f1(p1, mask)
        all_dice.append(d)
        all_f1.append(f)
        processed_steps += 1

        if args is not None and vis_counters is not None:
            save_wavelet_debug_visuals(args, net, image, mask, p1, stage_name, epoch_idx, step, vis_counters)
        if args is not None and getattr(args, 'max_val_steps', 0) > 0 and processed_steps >= args.max_val_steps:
            break

    net.train()
    avg_loss = total_loss / max(processed_steps, 1)
    avg_dice = float(np.mean(all_dice)) if all_dice else 0.0
    avg_f1   = float(np.mean(all_f1)) if all_f1 else 0.0
    return avg_loss, avg_dice, avg_f1



# ============ 参数解析 ============

def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--batchsize',   default=4,    type=int)
    p.add_argument('--resize',      default=1024, type=int,
                   help='输入图像 resize 尺寸（U-Net/AttU-Net 建议 256 或 512）')
    p.add_argument('--savepath',    default="./pretrained-models", type=str)
    p.add_argument('--datapath',    default="./data/train",        type=str)
    p.add_argument('--valpath',     default=None, type=str,
                   help='验证集路径，若为空则跳过验证（仅按 train loss 保存 best）')
    p.add_argument('--labelpath',   default=None, type=str)
    p.add_argument('--checkpoint',  default=None, type=str)
    p.add_argument('--scaler_reset', action='store_true', default=False,
                   help='恢复训练时重置 GradScaler 到 init_scale（防止恢复高 scale 溢出）')
    p.add_argument('--amp_dtype',   type=str, choices=['fp16', 'bf16'], default='fp16',
                   help='AMP dtype：fp16（默认，快但量程小）/ bf16（量程同 fp32，外部数据 FT 防激活溢出 NaN）')
    p.add_argument('--lr',          default=5e-3, type=float,
                   help='head 最大学习率（默认 5e-3，backbone 为 1/10）')
    p.add_argument('--epoch',       default=50,   type=int)
    p.add_argument('--task_mode',   choices=['seg', 'cls', 'joint'], default='seg')
    p.add_argument('--model',       choices=['wpugnet', 'unet', 'attunet'], default='wpugnet',
                   help='模型架构：wpugnet（默认）/ unet / attunet（外部基线）')
    p.add_argument('--num_classes', default=2,    type=int)
    p.add_argument('--cls_weight',  default=0.3,  type=float)
    p.add_argument('--disable_reciprocal_cls_head', action='store_true',
                   help='消融：移除互惠分类头（ROI 掩码引导分类分支），即使 joint/cls 模式也不实例化 cls_head')
    p.add_argument('--run_name',    default=None, type=str)

    p.add_argument('--patience',    default=10,   type=int,
                   help='early stopping patience（val Dice 不提升的最大 epoch 数，0 禁用）')
    p.add_argument('--val_freq',    default=1,    type=int,
                   help='每隔多少 epoch 做一次验证集评估')
    p.add_argument('--disable_wavelet_stem', action='store_true',
                   help='关闭 WaveletStem（Haar DWT 前端），ResNet 直接接收 3 通道 RGB')
    p.add_argument('--disable_wavelet_aug', action='store_true',
                   help='关闭训练集中的小波增强，其它几何/颜色增强保持不变')
    p.add_argument('--wavelet_aug_basis', default='haar', type=str,
                   help='训练侧小波增强使用的小波基，如 haar / db2 / db4')
    p.add_argument('--wavelet_aug_hf_boost', default=1.3, type=float,
                   help='训练侧小波增强的高频子带增强倍数')
    p.add_argument('--wavelet_aug_prob', default=0.5, type=float,
                   help='训练侧小波增强触发概率')

    p.add_argument('--disable_decoder_wavelet', action='store_true',
                   help='关闭解码器融合特征中的小波通道注意力WCA，保留其它结构不变')
    p.add_argument('--decoder_wca_variant', default='baseline', type=str,
                   help='解码器 WCA 位置方案：baseline / baseline_plus_stage1 / stage1_only / stage23_only / d2_only~d6_only（A17 位置消融）')
    p.add_argument('--seed', type=int, default=42,
                   help='随机种子')
    p.add_argument('--wavelet_basis', default='haar', type=str,
                   choices=['haar', 'db2', 'db4', 'sym2', 'sym4', 'coif1',
                            'bior2.2', 'bior4.4', 'rbio2.2', 'rbio4.4'],
                   help='解码器 LWA 小波基（默认 haar；10 种可选，见 wavelet_basis.py）')
    p.add_argument('--lwa_mode', choices=['gate', 'se', 'se_res', 'wca_plus', 'res_wca', 'rwca_plus', 'rwca', 'rwca_a07', 'rwca_a03', 'rwca_a04', 'rwca_a06', 'rwca_ll', 'rwca_hf', 'rwca_shared', 'rwca_sigmoid', 'rwca_nostat', 'rwca_beta1', 'rwca_gca', 'rwca_nobeta', 'rwca_ll_nb', 'rwca_hf_nb', 'rwca_shared_nb', 'rwca_sigmoid_nb', 'rwca_nostat_nb', 'rwca_a03_nb', 'rwca_a04_nb', 'rwca_a06_nb', 'rwca_a07_nb', 'rwca_a02_nb', 'rwca_a10_nb', 'rwca_lh_nb', 'rwca_hl_nb', 'rwca_hh_nb', 'rwca_llhf_nb', 'rwca_shuffle_nb', 'rwca_scalar_nb', 'rwca_spatial_ctl'], default='gate',
                   help='解码器 LWA 机制：gate（标量门，默认） / se（SE 门控，无残差） / se_res（SE 门控 + 可学习残差） / wca_plus（WCA+ 频域校准，稳定残差） / res_wca（ResWCA 差模残差 Y=X+γ(O-X)）')
    p.add_argument('--standardize_loss', action='store_true',
                   help='消融：将分割损失常规化为标准 BCE+IoU（会覆盖 seg_loss_mode 为 bce_iou）')
    p.add_argument('--seg_loss_mode', choices=['bce_iou', 'dice_biou', 'dice'], default='bce_iou',

                   help='分割损失模式：默认旧版 BCE+IoU，可切换为 Dice+BoundaryIoU')
    p.add_argument('--cgm_mode', choices=['cgm', 'concat', 'add', 'crossattn'], default='cgm',
                   help='CGM 融合方式消融：cgm（原始嫁接） / concat / add / crossattn（TransFuse式）')
    p.add_argument('--backbone_r', choices=['resnet18', 'resnet50'], default='resnet18',
                   help='ResNet 骨干消融：resnet18（默认） / resnet50')
    p.add_argument('--backbone_s', choices=['swin_base', 'swin_tiny'], default='swin_base',
                   help='Swin 骨干消融：swin_base（默认） / swin_tiny')

    p.add_argument('--dice_weight', default=0.6, type=float,
                   help='Dice loss 权重（seg_loss_mode=dice_biou 时生效）')
    p.add_argument('--boundary_iou_weight', default=0.4, type=float,
                   help='Boundary IoU loss 权重（seg_loss_mode=dice_biou 时生效）')
    p.add_argument('--boundary_kernel', default=3, type=int,
                   help='Boundary IoU 边界提取核大小（建议奇数，默认 3）')
    p.add_argument('--best_by', choices=['val_dice', 'val_loss'], default='val_dice',
                   help='有验证集时按哪个指标保存best和做early stopping')
    p.add_argument('--export_wavelet_debug', action='store_true',
                   help='导出WaveletStem与WCA的attention、LL/LH/HL/HH可视化结果')
    p.add_argument('--vis_save_dir', default=None, type=str,
                   help='可视化结果输出目录，默认保存在savepath/wavelet_vis')
    p.add_argument('--vis_max_batches', default=1, type=int,
                   help='每个阶段每个epoch最多保存多少个batch的可视化')
    p.add_argument('--vis_every_n_epochs', default=1, type=int,
                   help='训练阶段每隔多少个epoch保存一次可视化')
    p.add_argument('--max_train_steps', default=0, type=int,
                   help='仅用于快速测试时限制每个epoch最多训练多少步，0表示不限制')
    p.add_argument('--max_val_steps', default=0, type=int,
                   help='仅用于快速测试时限制验证最多多少步，0表示不限制')
    return p.parse_args()






# ============ 训练主函数 ============

def train(Dataset, Network):
    args = parser()

    global AMP_DTYPE
    AMP_DTYPE = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    if args.amp_dtype == 'bf16':
        # bf16 插值垫片: torch 1.13 的 upsample_bilinear2d 不支持 bf16，
        # 自动转 fp32 计算再转回。bf16 量程同 fp32，根治 fp16 激活溢出→BN 统计 NaN。
        _orig_interp = F.interpolate

        def _interp_bf16(input, *a, **k):
            if input.dtype == torch.bfloat16:
                return _orig_interp(input.float(), *a, **k).to(torch.bfloat16)
            return _orig_interp(input, *a, **k)

        F.interpolate = _interp_bf16

    if args.standardize_loss:
        args.seg_loss_mode = 'bce_iou'
    if args.disable_reciprocal_cls_head and args.task_mode == 'cls':
        raise ValueError('`--disable_reciprocal_cls_head` 不能与 `--task_mode cls` 同时使用；否则分类任务将没有有效输出头。')

    # 多种子重复实验
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device}')

    # ---- 训练集 ----
    cfg = Dataset.Config(
        datapath=args.datapath,
        savepath=args.savepath,
        labelpath=args.labelpath,
        mode='train',
        batch=args.batchsize,
        resize=args.resize,
        lr=args.lr,
        momen=0.9,
        decay=5e-4,
        epoch=args.epoch,
        snapshot=args.checkpoint,
        task_mode=args.task_mode,
        num_classes=args.num_classes,
        cls_weight=args.cls_weight,
        disable_reciprocal_cls_head=args.disable_reciprocal_cls_head,
        run_name=args.run_name,
        use_wavelet_stem=not args.disable_wavelet_stem,
        use_wavelet_aug=not args.disable_wavelet_aug,

        wavelet_aug_basis=args.wavelet_aug_basis,
        wavelet_aug_hf_boost=args.wavelet_aug_hf_boost,
        wavelet_aug_prob=args.wavelet_aug_prob,
        use_decoder_wavelet=not args.disable_decoder_wavelet,
        decoder_wca_variant=args.decoder_wca_variant,
        wavelet_basis=args.wavelet_basis,
        lwa_mode=args.lwa_mode,
        seg_loss_mode=args.seg_loss_mode,
        cgm_mode=args.cgm_mode,
        backbone_r=args.backbone_r,
        backbone_s=args.backbone_s,


        dice_weight=args.dice_weight,
        boundary_iou_weight=args.boundary_iou_weight,
        boundary_kernel=args.boundary_kernel,
        best_by=args.best_by,
        export_wavelet_debug=args.export_wavelet_debug,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
    )




    data   = Dataset.Data(cfg)
    loader = DataLoader(data, batch_size=args.batchsize, shuffle=True,
                        num_workers=0, pin_memory=False, drop_last=True,
                        collate_fn=data.collate)

    # ---- 验证集 ----
    val_loader = None
    if args.valpath and os.path.isdir(args.valpath):
        val_cfg = Dataset.Config(
            datapath=args.valpath,
            mode='val',            # val 模式：确定性验证，不做随机增强
            batch=args.batchsize,
            resize=args.resize,
            lr=args.lr, momen=0.9, decay=5e-4, epoch=args.epoch,
            snapshot=None, task_mode=args.task_mode,
            num_classes=args.num_classes, cls_weight=args.cls_weight,
            disable_reciprocal_cls_head=args.disable_reciprocal_cls_head,
            use_wavelet_stem=not args.disable_wavelet_stem,
            use_wavelet_aug=False,

            wavelet_aug_basis=args.wavelet_aug_basis,
            wavelet_aug_hf_boost=args.wavelet_aug_hf_boost,
            wavelet_aug_prob=0.0,
            decoder_wca_variant=args.decoder_wca_variant,
            wavelet_basis=args.wavelet_basis,
            lwa_mode=args.lwa_mode,
            seg_loss_mode=args.seg_loss_mode,
            cgm_mode=args.cgm_mode,
            backbone_r=args.backbone_r,
            backbone_s=args.backbone_s,


            dice_weight=args.dice_weight,
            boundary_iou_weight=args.boundary_iou_weight,
            boundary_kernel=args.boundary_kernel,
            best_by=args.best_by,
            export_wavelet_debug=args.export_wavelet_debug,
            max_train_steps=args.max_train_steps,
            max_val_steps=args.max_val_steps,
        )



        val_data   = Dataset.Data(val_cfg)
        val_loader = DataLoader(val_data, batch_size=args.batchsize, shuffle=False,
                                num_workers=0, pin_memory=False, drop_last=False,
                                collate_fn=val_data.collate)
        print(f'验证集已加载: {len(val_data)} 样本, {len(val_loader)} batches')
    else:
        print('未指定验证集路径，将跳过验证（按 train loss 保存 best）')

    mode_tag = args.task_mode if args.run_name is None else f"{args.task_mode}_{args.run_name}"

    # ---- 模型（支持动态选择架构）----
    _model_name = getattr(args, 'model', 'wpugnet')
    if _model_name == 'unet':
        from unet import UNet
        NetClass = UNet
    elif _model_name == 'attunet':
        from attunet import AttUNet
        NetClass = AttUNet
    else:
        NetClass = Network

    net = NetClass(cfg).to(device)
    net.train()
    print(f'Model: {_model_name} | params: {sum(p.numel() for p in net.parameters()):,}')

    # 分组学习率：backbone × 0.1，head × 1.0
    base, head = [], []
    for name, param in net.named_parameters():
        if 'swin' in name or 'resnet' in name:
            base.append(param)
        else:
            head.append(param)

    optimizer = torch.optim.AdamW(
        [{'params': base, 'lr': cfg.lr * 0.1},
         {'params': head, 'lr': cfg.lr}],
        weight_decay=cfg.decay,
    )

    # Cosine Annealing with warm-up（前 5 epoch 线性 warm-up）
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epoch - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    scaler      = GradScaler(init_scale=512, growth_interval=2000, enabled=(args.amp_dtype == 'fp16'))   # 降低初始scale+减缓增长，防止fp16溢出NaN；bf16 无需缩放
    best_metric = -float('inf')   # val Dice（有验证集）或 -train_loss（无验证集）
    patience_cnt = 0
    start_epoch = 0
    vis_counters = {}
    log_path    = os.path.join(args.savepath, f'train_log_{mode_tag}.csv')

    os.makedirs(args.savepath, exist_ok=True)

    # ---- Resume from checkpoint ----
    resume_ckpt = args.checkpoint
    if resume_ckpt and os.path.exists(resume_ckpt):
        print(f'[Resume] Loading checkpoint: {resume_ckpt}')
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)  # torch>=2.6 默认 weights_only=True 会拒载旧 checkpoint
        if 'model_state_dict' not in ckpt and 'optimizer_state_dict' not in ckpt:
            # 模型-only 权重: 初始化权重 + 新鲜优化器，从 0 起训
            _m, _u = net.load_state_dict(ckpt, strict=False)
            print(f'[FineTune] model-only weights loaded: missing={len(_m)} unexpected={len(_u)}')
            start_epoch = 0
            best_metric = 0.0
            patience_cnt = 0
        else:
            net.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scaler_state_dict' in ckpt:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            else:
                print('[Resume] Old checkpoint without scaler state — using fresh scaler')
            if getattr(args, 'scaler_reset', False):
                print('[Resume] Resetting GradScaler to init_scale=512 (avoid resumed high-scale fp16 overflow)')
                scaler = GradScaler(init_scale=512, growth_interval=2000)
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            # Restore best_metric from checkpoint, or scan existing log
            if 'best_metric' in ckpt:
                best_metric = ckpt['best_metric']
                patience_cnt = ckpt.get('patience_cnt', 0)
            elif os.path.exists(log_path):
                # Scan existing log for best val_dice
                try:
                    df = __import__('pandas').read_csv(log_path)
                    val_dice_col = [c for c in df.columns if 'val_dice' in c.lower()]
                    if val_dice_col:
                        best_metric = df[val_dice_col[0]].max()
                    # Count patience: epochs since best
                    if val_dice_col and len(df) > 0:
                        best_idx = df[val_dice_col[0]].idxmax()
                        patience_cnt = len(df) - 1 - best_idx
                    print(f'[Resume] Recovered best_metric={best_metric:.4f}, patience_cnt={patience_cnt} from log')
                except Exception:
                    pass
        print(f'[Resume] Continuing from epoch {start_epoch}')
    else:
        if resume_ckpt:
            print(f'[Warn] Checkpoint not found: {resume_ckpt}, starting from scratch')

    if start_epoch == 0:
        with open(log_path, 'w') as f:
            f.write('epoch,train_loss,val_loss,val_dice,val_f1,lr_head\n')

    print(f'Start training for {cfg.epoch} epochs (start_epoch={start_epoch})...')
    print(f'Train batches: {len(loader)} | Val batches: {len(val_loader) if val_loader else 0}')
    print(f'Seg loss mode: {cfg.seg_loss_mode} | Dice weight: {cfg.dice_weight} | BoundaryIoU weight: {cfg.boundary_iou_weight} | best_by: {args.best_by}')
    print(f'Wavelet switches | stem={"on" if not args.disable_wavelet_stem else "OFF"} | train_wavelet_aug={not args.disable_wavelet_aug} | aug_basis={args.wavelet_aug_basis} | aug_hf_boost={args.wavelet_aug_hf_boost} | aug_prob={args.wavelet_aug_prob} | decoder_wca={not args.disable_decoder_wavelet} | decoder_wca_variant={args.decoder_wca_variant} | wavelet_basis={args.wavelet_basis} | lwa_mode={args.lwa_mode} | cgm_mode={args.cgm_mode} | backbone_r={args.backbone_r} | backbone_s={args.backbone_s} | export_debug={args.export_wavelet_debug}')
    print(f'Ablation switches | reciprocal_cls_head={not args.disable_reciprocal_cls_head} | standardize_loss={args.standardize_loss}')
    if args.disable_reciprocal_cls_head and args.task_mode == 'joint':
        print('[Ablation] joint 模式下已移除互惠分类头；当前训练将只保留分割损失分支。')
    if args.disable_reciprocal_cls_head and args.task_mode == 'seg':
        print('[Ablation] 当前为 seg 模式，互惠分类头原本就未启用；该开关不会改变训练行为。')
    if args.standardize_loss and requested_seg_loss_mode == 'bce_iou':
        print('[Ablation] 当前分割损失本来就是 bce_iou；standardize_loss 在该配置下等价于 no-op。')











    # ---- Epoch 0 baseline：先评估初始/加载权重，避免微调把原best冲坏 ----
    if val_loader and start_epoch == 0:
        baseline_val_loss, baseline_val_dice, baseline_val_f1 = validate(
            net, val_loader, device, cfg, args=args, vis_counters=vis_counters, epoch_idx=0, stage_name='baseline'
        )
        best_metric, best_label = resolve_selection_metric(
            args, True, val_loss=baseline_val_loss, val_dice=baseline_val_dice
        )

        lr_head = optimizer.param_groups[1]['lr']
        with open(log_path, 'a') as f:
            f.write(f'0,0.000000,{baseline_val_loss:.6f},{baseline_val_dice:.6f},{baseline_val_f1:.6f},{lr_head:.8f}\n')
        torch.save(net.state_dict(), os.path.join(args.savepath, f'best_{mode_tag}.pth'))
        print(f'>>> Epoch 0 baseline | val_loss={baseline_val_loss:.4f} '
              f'val_dice={baseline_val_dice:.4f} val_f1={baseline_val_f1:.4f}')
        print(f'  ★ Initial model saved as best ({best_label})')


    for epoch in range(start_epoch, cfg.epoch):
        net.train()
        epoch_loss   = 0.0
        epoch_dice_l = []
        epoch_f1_l   = []
        epoch_step_times = []

        for step, (image, mask, label) in enumerate(loader):
            step_start = time.perf_counter()
            image = image.float().to(device)

            mask  = mask.float().to(device)
            label = label.to(device)

            optimizer.zero_grad()

            with autocast(dtype=AMP_DTYPE):
                p1, wr, ws, attmap, cls_logits = net(image)

                seg_loss = torch.tensor(0.0, device=device)
                att_loss = torch.tensor(0.0, device=device)
                if cfg.task_mode in ['seg', 'joint']:
                    wavelet_on = not args.disable_decoder_wavelet
                    stem_on = not args.disable_wavelet_stem
                    att_loss = (att_loss_fn(attmap, mask, wr, ws)
                                if wavelet_on and stem_on and attmap is not None
                                else torch.tensor(0.0, device=device))
                    loss1    = segmentation_loss(p1, mask, cfg)
                    loss2    = (segmentation_loss(wr, mask, cfg) * 0.125 + segmentation_loss(ws, mask, cfg) * 0.125
                                if wavelet_on and stem_on else torch.tensor(0.0, device=device))
                    seg_loss = loss1 + loss2 + att_loss


                cls_loss = torch.tensor(0.0, device=device)
                if cfg.task_mode in ['cls', 'joint'] and cls_logits is not None:
                    labeled = label >= 0
                    if labeled.any():
                        cls_loss = F.cross_entropy(cls_logits[labeled], label[labeled])

                loss = seg_loss + cfg.cls_weight * cls_loss

            # NaN 检测：出现异常值时记录信息并终止训练
            if not torch.isfinite(loss):
                nan_info = {
                    'epoch': epoch + 1,
                    'step': step,
                    'total_steps': len(loader),
                    'loss': loss.item(),
                    'seg_loss': seg_loss.item(),
                    'att_loss': att_loss.item(),
                    'cls_loss': cls_loss.item(),
                    'lr': optimizer.param_groups[1]['lr'],
                    'scaler_scale': scaler.get_scale(),
                }
                nan_log_path = os.path.join(args.savepath, 'nan_abort.json')
                import json
                with open(nan_log_path, 'w') as _f:
                    json.dump(nan_info, _f, indent=2)
                # 保存当前最新权重
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'nan_info': nan_info,
                }, os.path.join(args.savepath, f'nan_abort_{mode_tag}.pth'))
                print(f'[NaN ABORT] Ep:{epoch+1} Step:{step}/{len(loader)} | '
                      f'loss={loss.item()} seg={seg_loss.item():.4f} '
                      f'att={att_loss.item():.4f} scale={scaler.get_scale()}')
                print(f'[NaN ABORT] 已保存诊断信息到 {nan_log_path}')
                optimizer.zero_grad()
                raise RuntimeError(f'NaN loss at epoch={epoch+1} step={step}, training aborted.')

            scaler.scale(loss).backward()
            # 梯度裁剪，防止梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            with torch.no_grad():
                d, f = compute_dice_f1(p1, mask)
            epoch_dice_l.append(d)
            epoch_f1_l.append(f)

            step_time = time.perf_counter() - step_start
            epoch_step_times.append(step_time)
            if args.export_wavelet_debug and (args.vis_every_n_epochs <= 1 or (epoch + 1) % args.vis_every_n_epochs == 0):
                save_wavelet_debug_visuals(args, net, image, mask, p1, 'train', epoch, step, vis_counters)

            if step % 20 == 0:
                lr_head = optimizer.param_groups[1]['lr']
                print(f'{datetime.datetime.now()} | Ep:{epoch+1}/{cfg.epoch} Step:{step}/{len(loader)} | '
                      f'lr={lr_head:.6f} | total={loss.item():.4f} '
                      f'seg={seg_loss.item():.4f} att={att_loss.item():.4f} '
                      f'dice={d:.4f} step_time={step_time:.3f}s')

            if args.max_train_steps > 0 and (step + 1) >= args.max_train_steps:
                print(f'[TEST] max_train_steps reached at epoch {epoch+1}, stop epoch early.')
                break


        scheduler.step()

        effective_train_steps = max(len(epoch_dice_l), 1)
        avg_train_loss = epoch_loss / effective_train_steps
        avg_train_dice = float(np.mean(epoch_dice_l)) if epoch_dice_l else 0.0
        avg_train_f1   = float(np.mean(epoch_f1_l)) if epoch_f1_l else 0.0
        avg_step_time  = float(np.mean(epoch_step_times)) if epoch_step_times else 0.0
        lr_head = optimizer.param_groups[1]['lr']


        # ---- 验证集评估 ----
        val_loss, val_dice, val_f1 = 0.0, 0.0, 0.0
        cur_metric = None
        current_label = None
        if val_loader and (epoch + 1) % args.val_freq == 0:
            val_loss, val_dice, val_f1 = validate(
                net, val_loader, device, cfg, args=args, vis_counters=vis_counters, epoch_idx=epoch, stage_name='val'
            )
            print(f'>>> Epoch {epoch+1} | '
                  f'train_loss={avg_train_loss:.4f} train_dice={avg_train_dice:.4f} train_step_time={avg_step_time:.3f}s | '
                  f'val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_f1={val_f1:.4f}')
            cur_metric, current_label = resolve_selection_metric(
                args, True, val_loss=val_loss, val_dice=val_dice
            )
        elif val_loader:
            print(f'>>> Epoch {epoch+1} | '
                  f'train_loss={avg_train_loss:.4f} train_dice={avg_train_dice:.4f} train_f1={avg_train_f1:.4f} train_step_time={avg_step_time:.3f}s '
                  f'| validation skipped (val_freq={args.val_freq})')
        else:
            print(f'>>> Epoch {epoch+1} | '
                  f'train_loss={avg_train_loss:.4f} train_dice={avg_train_dice:.4f} train_f1={avg_train_f1:.4f} train_step_time={avg_step_time:.3f}s')
            cur_metric, current_label = resolve_selection_metric(
                args, False, train_loss=avg_train_loss
            )




        # ---- CSV 日志 ----
        with open(log_path, 'a') as f:
            f.write(f'{epoch+1},{avg_train_loss:.6f},{val_loss:.6f},{val_dice:.6f},{val_f1:.6f},{lr_head:.8f}\n')

        # ---- 保存检查点 ----
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_metric': best_metric,
            'patience_cnt': patience_cnt,
            'loss': avg_train_loss,
            'val_dice': val_dice,
        }, os.path.join(args.savepath, f'latest_{mode_tag}.pth'))

        if cur_metric is not None:
            if cur_metric > best_metric:
                best_metric = cur_metric
                patience_cnt = 0
                torch.save(net.state_dict(), os.path.join(args.savepath, f'best_{mode_tag}.pth'))
                print(f'  ★ Best model saved ({current_label})')
            else:
                patience_cnt += 1
                if args.patience > 0 and patience_cnt >= args.patience:
                    print(f'  Early stopping triggered (patience={args.patience}, best_by={args.best_by})')
                    break



        if (epoch + 1) % 10 == 0:
            torch.save(net.state_dict(),
                       os.path.join(args.savepath, f'{mode_tag}_epoch_{epoch+1}.pth'))

    print('Training completed!')
    print(f'训练日志已保存到: {log_path}')


if __name__ == '__main__':
    print("=" * 60)
    print("WPUGNet 超声图像分割 — 训练")
    print("=" * 60)
    print(f"任务路径: {TASK_PATH}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print("=" * 60 + "\n")

    try:
        train(dataset, WPUGNet)
    except Exception as e:
        print(f"\n[ERROR] 训练出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)  # NaN 中止必须非零退出，供 run_*.sh 的 nan_abort.json 检测续训
