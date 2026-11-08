#!/usr/bin/env python3
# coding=utf-8
"""Evaluate one or more checkpoints on a test split and export the metrics.

Writes a summary CSV, a per-sample CSV and a failure-case CSV. A sample whose
test Dice falls below --failure-dice-threshold is counted as a segmentation
failure. The checkpoint is the only required argument; the architecture flags
must match those used at training time.
"""

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(ROOT_DIR)
TASK_PATH = os.path.join(REPO_DIR, 'src')
if TASK_PATH not in sys.path:
    sys.path.insert(0, TASK_PATH)

import dataset  # noqa: E402
from wpugnet import WPUGNet  # noqa: E402
from unet import UNet  # noqa: E402
from attunet import AttUNet  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    checkpoint: str
    module_type: str
    use_decoder_wavelet: bool
    decoder_wca_variant: str = 'baseline'
    export_wavelet_debug: bool = False
    use_wavelet_stem: bool = True
    wavelet_basis: str = 'haar'
    lwa_mode: str = 'gate'
    cgm_mode: str = 'cgm'
    backbone_r: str = 'resnet18'
    backbone_s: str = 'swin_base'


def default_output_paths(output_csv: str) -> Tuple[str, str, str]:
    base, ext = os.path.splitext(output_csv)
    if not ext:
        ext = '.csv'
        output_csv = base + ext
    return output_csv, base + '_per_sample.csv', base + '_failed_cases.csv'


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate checkpoints on a test split')
    parser.add_argument(
        '--datapath',
        default=os.path.join(REPO_DIR, 'data', 'BUSI_ft', 'test'),
        type=str,
        help='test split directory containing images/ and masks/',
    )
    parser.add_argument(
        '--output',
        default=os.path.join(REPO_DIR, 'results', 'test_metrics.csv'),
        type=str,
        help='summary CSV path',
    )
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='checkpoint to evaluate')
    parser.add_argument('--per-sample-output', default=None, type=str,
                        help='per-sample metrics CSV path')
    parser.add_argument('--failed-cases-output', default=None, type=str,
                        help='failure-case CSV path')
    parser.add_argument('--notes-output', default=None, type=str,
                        help='run description TXT path')
    parser.add_argument('--batchsize', default=4, type=int, help='test batch size')
    parser.add_argument('--resize', default=1024, type=int,
                        help='input resize, must match training')
    parser.add_argument('--failure-dice-threshold', default=0.60, type=float,
                        help='samples below this Dice are recorded as failures')
    parser.add_argument('--model', choices=['wpugnet', 'unet', 'attunet'], default='wpugnet',
                        help='architecture')
    parser.add_argument('--use-wavelet-stem', action='store_true', default=True,
                        help='enable the wavelet stem (default)')
    parser.add_argument('--no-wavelet-stem', action='store_false', dest='use_wavelet_stem',
                        help='disable the wavelet stem')
    parser.add_argument('--use-decoder-wavelet', action='store_true', default=True,
                        help='enable decoder RWCA (default)')
    parser.add_argument('--no-decoder-wavelet', action='store_false', dest='use_decoder_wavelet',
                        help='disable decoder RWCA')
    parser.add_argument('--decoder-wca-variant', default='baseline', type=str,
                        help='RWCA placement: baseline / baseline_plus_stage1 / '
                             'stage1_only / stage23_only')
    parser.add_argument('--wavelet-basis', default='haar', type=str,
                        choices=['haar', 'db2', 'db4', 'sym2', 'sym4', 'coif1',
                                 'bior2.2', 'bior4.4', 'rbio2.2', 'rbio4.4'],
                        help='wavelet basis, must match training')
    parser.add_argument('--lwa-mode', default='rwca', type=str,
                        help='decoder wavelet mechanism, must match training')
    parser.add_argument('--cgm-mode', default='cgm', type=str,
                        choices=['cgm', 'concat', 'add', 'crossattn'],
                        help='fusion strategy, must match training')
    parser.add_argument('--backbone-r', default='resnet18', type=str,
                        choices=['resnet18', 'resnet50'],
                        help='ResNet backbone, must match training')
    parser.add_argument('--backbone-s', default='swin_base', type=str,
                        choices=['swin_base', 'swin_tiny'],
                        help='Swin backbone, must match training')
    parser.add_argument('--display-name', default=None, type=str,
                        help='name written to the output CSVs, defaults to the checkpoint filename')
    return parser.parse_args()


def build_cfg(datapath: str, spec: ModelSpec, resize: int = 1024):
    cfg_kwargs = dict(
        datapath=datapath,
        snapshot=spec.checkpoint,
        mode='val',
        task_mode='seg',
        num_classes=0,
        cls_weight=0.3,
        resize=resize,
        use_wavelet_stem=getattr(spec, 'use_wavelet_stem', True),
        use_wavelet_aug=False,
        use_decoder_wavelet=spec.use_decoder_wavelet,
        decoder_wca_variant=spec.decoder_wca_variant,
        wavelet_basis=getattr(spec, 'wavelet_basis', 'haar'),
        lwa_mode=getattr(spec, 'lwa_mode', 'gate'),
        cgm_mode=getattr(spec, 'cgm_mode', 'cgm'),
        backbone_r=getattr(spec, 'backbone_r', 'resnet18'),
        backbone_s=getattr(spec, 'backbone_s', 'swin_base'),
        export_wavelet_debug=spec.export_wavelet_debug,
    )
    return dataset.Config(**cfg_kwargs)


@torch.no_grad()
def align_logits_to_mask(pred_logits: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if pred_logits.shape[2:] != masks.shape[2:]:
        pred_logits = F.interpolate(pred_logits, size=masks.shape[2:], mode='bilinear', align_corners=False)
    return pred_logits, masks


@torch.no_grad()
def bce_iou_per_sample(pred_logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    pred_logits, masks = align_logits_to_mask(pred_logits, masks)
    masks = masks.float()
    wbce = F.binary_cross_entropy_with_logits(pred_logits, masks, reduction='none').mean(dim=(1, 2, 3))
    pred = torch.sigmoid(pred_logits)
    inter = (pred * masks).sum(dim=(1, 2, 3))
    union = (pred + masks).sum(dim=(1, 2, 3))
    wiou = 1 - (inter + 1.0) / (union - inter + 1.0)
    return wbce + wiou


@torch.no_grad()
def segmentation_metrics(pred_logits: torch.Tensor, masks: torch.Tensor) -> Dict[str, torch.Tensor]:
    pred_logits, masks = align_logits_to_mask(pred_logits, masks)
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    masks_bin = (masks > 0.5).float()
    tp = (pred * masks_bin).sum(dim=(1, 2, 3))
    fp = (pred * (1 - masks_bin)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * masks_bin).sum(dim=(1, 2, 3))

    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        'dice': dice,
        'f1': f1,
        'iou': iou,
        'precision': precision,
        'recall': recall,
    }


@torch.no_grad()
def evaluate_one_model(
    spec: ModelSpec,
    datapath: str,
    batchsize: int,
    device: torch.device,
    failure_dice_threshold: float,
    resize: int = 1024,
) -> Tuple[Dict[str, float], List[Dict[str, object]], List[Dict[str, object]]]:
    if not os.path.isfile(spec.checkpoint):
        raise FileNotFoundError('checkpoint not found: %s' % spec.checkpoint)

    cfg = build_cfg(datapath, spec, resize=resize)
    data = dataset.Data(cfg)
    sample_names = ['%s.png' % name for name in data.samples]
    loader = DataLoader(
        data,
        batch_size=batchsize,
        shuffle=False,
        num_workers=0,
        collate_fn=data.collate,
    )

    model_cls = {'wpugnet': WPUGNet, 'unet': UNet, 'attunet': AttUNet}[spec.module_type]
    net = model_cls(cfg).to(device)
    # WPUGNet restores its weights in __init__; the two baselines do not.
    if spec.module_type in ('unet', 'attunet') and os.path.isfile(spec.checkpoint):
        ckpt = torch.load(spec.checkpoint, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
        missing, unexpected = net.load_state_dict(ckpt, strict=False)
        print('checkpoint loaded: missing=%d, unexpected=%d' % (len(missing), len(unexpected)))
    net.eval()

    losses: List[torch.Tensor] = []
    metric_bank: Dict[str, List[torch.Tensor]] = {
        'dice': [],
        'f1': [],
        'iou': [],
        'precision': [],
        'recall': [],
    }
    per_sample_rows: List[Dict[str, object]] = []
    failed_rows: List[Dict[str, object]] = []
    cursor = 0

    for image, mask, _ in loader:
        image = image.float().to(device)
        mask = mask.float().to(device)
        batch_size = image.shape[0]
        batch_names = sample_names[cursor:cursor + batch_size]
        cursor += batch_size

        pred1, _, _, _, _ = net(image)
        loss_tensor = bce_iou_per_sample(pred1, mask).cpu()
        batch_metrics = {key: value.cpu() for key, value in segmentation_metrics(pred1, mask).items()}

        losses.append(loss_tensor)
        for key, value in batch_metrics.items():
            metric_bank[key].append(value)

        for idx, file_name in enumerate(batch_names):
            row = {
                'model': spec.display_name,
                'file_name': file_name,
                'loss': float(loss_tensor[idx].item()),
                'dice': float(batch_metrics['dice'][idx].item()),
                'f1': float(batch_metrics['f1'][idx].item()),
                'iou': float(batch_metrics['iou'][idx].item()),
                'precision': float(batch_metrics['precision'][idx].item()),
                'recall': float(batch_metrics['recall'][idx].item()),
                'is_failure': 0,
                'checkpoint': os.path.relpath(spec.checkpoint, REPO_DIR),
            }
            if row['dice'] < failure_dice_threshold:
                row['is_failure'] = 1
                failed_rows.append(dict(row))
            per_sample_rows.append(row)

    result = {
        'model': spec.display_name,
        'samples': len(data),
        'loss': float(torch.cat(losses).mean().item()),
        'checkpoint': os.path.relpath(spec.checkpoint, REPO_DIR),
    }
    for key, chunks in metric_bank.items():
        result[key] = float(torch.cat(chunks).mean().item())
    return result, per_sample_rows, failed_rows


def save_csv(rows: List[Dict[str, object]], output_path: str, fieldnames: List[str]):
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_notes(specs: List[ModelSpec], notes_path: str, failure_dice_threshold: float):
    lines = [
        'Evaluation notes',
        '====================',
        'failure case: Dice < %.2f' % failure_dice_threshold,
        '',
    ]
    for idx, spec in enumerate(specs, start=1):
        lines.append('%d. %s' % (idx, spec.display_name))
        lines.append('   checkpoint: %s' % os.path.relpath(spec.checkpoint, REPO_DIR))
        lines.append('')

    parent = os.path.dirname(notes_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(notes_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines).strip() + '\n')


def print_markdown_table(rows: List[Dict[str, object]], output_path: str, failed_cases_output: str):
    print('\nSummary (loss is BCE + IoU)\n')
    print('| model | samples | Loss | Dice | F1 | IoU | Precision | Recall |')
    print('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for row in rows:
        print(
            '| {model} | {samples} | {loss:.6f} | {dice:.6f} | {f1:.6f} | {iou:.6f} | {precision:.6f} | {recall:.6f} |'.format(
                **row
            )
        )
    print('\nsummary CSV: %s' % output_path)
    print('failure CSV: %s' % failed_cases_output)


def main():
    args = get_args()
    args.output, default_per_sample, default_failed = default_output_paths(args.output)
    if args.per_sample_output is None:
        args.per_sample_output = default_per_sample
    if args.failed_cases_output is None:
        args.failed_cases_output = default_failed
    if args.notes_output is None:
        args.notes_output = os.path.splitext(args.output)[0] + '_notes.txt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: %s' % device)
    print('test split: %s' % args.datapath)
    print('failure threshold: Dice < %.2f' % args.failure_dice_threshold)

    models = [
        ModelSpec(
            name='single',
            display_name=args.display_name or os.path.basename(args.checkpoint),
            checkpoint=args.checkpoint,
            module_type=args.model,
            use_decoder_wavelet=args.use_decoder_wavelet,
            decoder_wca_variant=args.decoder_wca_variant,
            use_wavelet_stem=args.use_wavelet_stem,
            wavelet_basis=args.wavelet_basis,
            lwa_mode=args.lwa_mode,
            cgm_mode=args.cgm_mode,
            backbone_r=args.backbone_r,
            backbone_s=args.backbone_s,
        )
    ]

    summary_rows: List[Dict[str, object]] = []
    per_sample_rows: List[Dict[str, object]] = []
    failed_rows: List[Dict[str, object]] = []

    for spec in models:
        print('\n[TEST] %s' % spec.display_name)
        summary_row, per_model_rows, per_model_failed_rows = evaluate_one_model(
            spec=spec,
            datapath=args.datapath,
            batchsize=args.batchsize,
            device=device,
            failure_dice_threshold=args.failure_dice_threshold,
            resize=args.resize,
        )
        summary_rows.append(summary_row)
        per_sample_rows.extend(per_model_rows)
        failed_rows.extend(per_model_failed_rows)

    summary_fieldnames = ['model', 'samples', 'loss', 'dice', 'f1', 'iou', 'precision', 'recall', 'checkpoint']
    per_sample_fieldnames = ['model', 'file_name', 'loss', 'dice', 'f1', 'iou', 'precision', 'recall', 'is_failure', 'checkpoint']

    failed_rows.sort(key=lambda row: (str(row['model']), float(row['dice']), str(row['file_name'])))
    save_csv(summary_rows, args.output, summary_fieldnames)
    save_csv(per_sample_rows, args.per_sample_output, per_sample_fieldnames)
    save_csv(failed_rows, args.failed_cases_output, per_sample_fieldnames)
    save_notes(models, args.notes_output, args.failure_dice_threshold)
    print_markdown_table(summary_rows, args.output, args.failed_cases_output)


if __name__ == '__main__':
    main()
