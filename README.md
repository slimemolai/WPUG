# WPUGNet

Wavelet-Prompted Universal Grafted Network for breast-ultrasound lesion segmentation.

WPUGNet pairs a ResNet-18 encoder with a three-stage Swin-Base transformer encoder and
fuses them through a cross-grafting module (CGM). Three wavelet mechanisms sit on top:
an à-trous wavelet stem that feeds a 12-channel full-resolution decomposition to the CNN
branch, signed residual wavelet channel attention (RWCA) in the five decoder blocks, and
wavelet-domain high-frequency enhancement during training. On the BUSI test split the
model reaches **Dice 0.816 / IoU 0.741** and is significantly better than TransUNet
(+3.0 points) and Swin-Unet (+7.6 points), both at p < 0.001.

The central empirical finding is that the wavelet modules do not help in isolation. The
grafting baseline scores 0.794; adding the wavelet stem alone gives 0.773 and the wavelet
enhancement alone 0.762. Only the complete combination reaches 0.816, so the gain comes
from the interaction between the frequency prior and the dual-backbone graft.

## Repository layout

```
src/                    model code
  wpugnet.py            WPUGNet, CGM, RWCA and the ablation variants
  Res.py, Swin.py       encoders
  wavelet_basis.py      DWT/IDWT for ten wavelet bases
  dataset.py            BUSI data pipeline and augmentation
  unet.py, attunet.py   from-scratch baselines
scripts/
  train.py              training
  evaluate.py           checkpoint evaluation
  significance.py       paired Wilcoxon signed-rank tests
  prepare_busi.py       raw BUSI release to a flat images/masks layout
  split_busi.py         445/89/113 stratified split
  prepare_external_datasets.py   UDIAT and BrEaST staging
results/
  tables/               every result table from the paper, as CSV
  per_sample/           per-image Dice for each configuration on the 113 test images
```

## Results

The files in `results/tables/` mirror the manuscript tables:

| file | contents |
| --- | --- |
| `table01_main_comparison.csv` | BUSI test comparison with TransUNet and Swin-Unet |
| `table02_module_synergy.csv` | stem / RWCA / enhancement on and off |
| `table03_loss_ablation.csv` | BCE+IoU, Dice+boundary-IoU, pure Dice |
| `table04_fusion_ablation.csv` | CGM vs addition, concatenation, cross-attention |
| `table05_design_variants.csv` | gate form, sub-band subset, alpha, basis, seeds |
| `table06_backbone_pairing.csv` | ResNet-18/50 with Swin-Base/Tiny |
| `table07a_cv_folds.csv` | five-fold held-out Dice |
| `table07b_pairwise_tests.csv` | delta Dice, p, r_mrb and PS against the five baselines |
| `table08_cross_dataset.csv` | UDIAT and BrEaST, zero-shot and fine-tuned |
| `table09_param_budget.csv` | parameters, FLOPs and Dice for every model measured |
| `table10_efficiency.csv` | latency and peak memory at 1024x1024 |

`results/per_sample/` holds one CSV per configuration with the Dice of each of the 113
test images, which is what `scripts/significance.py` consumes. Every file there can be
used to recompute the significance table directly.

## Environment

Python 3.7 or newer, PyTorch 1.13, CUDA 11.7. Install with:

```
pip install -r requirements.txt
```

The paper's runs used a single RTX 4060 Laptop at 1024x1024 with batch size 6.

## Data

BUSI is public; download it separately and do not expect it in this repository.

```
python scripts/prepare_busi.py --src /path/to/Dataset_BUSI_with_GT --dst data/BUSI_prepared
python scripts/split_busi.py --src data/BUSI_prepared --dst data/BUSI_ft
```

`split_busi.py` uses seed 42 and reproduces the 445/89/113 split reported in the paper.
UDIAT and BrEaST are staged the same way with `prepare_external_datasets.py`.

## Training

```
python scripts/train.py \
  --trainpath data/BUSI_ft/train \
  --valpath   data/BUSI_ft/val \
  --savepath  runs/busi \
  --resize 1024 --epochs 100 --batchsize 6 \
  --lr 3e-4 --seed 123 \
  --lwa_mode rwca_nobeta
```

`--amp_dtype bf16` is recommended for fine-tuning on external ultrasound data. Under fp16
the decoder batch-norm statistics overflowed on every transfer run; bf16 shares the fp32
range and eliminated the failure. Run `python scripts/train.py --help` for the full set of
architecture switches, which must match those used at evaluation time.

## Evaluation

```
python scripts/evaluate.py \
  --checkpoint runs/busi/best_seg.pth \
  --datapath   data/BUSI_ft/test \
  --lwa_mode   rwca_nobeta \
  --resize     1024 \
  --output     results/test_metrics.csv
```

This writes a summary CSV, a per-image CSV and a failure-case CSV, and reports a sample as
a failure when its Dice is below 0.60.

## Statistics

```
python scripts/significance.py --reference wpugnet_seed123
```

The test is a two-sided Wilcoxon signed-rank test with zero differences split evenly
(`zero_method=zsplit`), paired by filename across the 113 test images. Effect size is the
matched-pairs rank-biserial correlation r = Z / sqrt(2n); PS is the probability of
superiority, P(A > B) + 0.5 P(A = B).

## Weights

Checkpoints are distributed through the Releases page rather than the repository tree.
The flagship model is the seed-123 run reported as 0.816 in the paper.

## Reproducibility notes

- Every main-table number is a single fully specified run. Re-training the same
  architecture under different seeds gives 0.788, 0.806 and 0.816 (mean 0.803, s.d. 0.014),
  and that spread sets the scale against which one- to two-point differences should be read.
- In this implementation the two auxiliary segmentation heads and the CTAM boundary
  supervision are gated by the same switches as the wavelet mechanisms, so turning a
  wavelet module off in the synergy ablation also removes those auxiliary objectives. The
  auxiliary heads supply training-time supervision only and are absent from the inference
  graph.
- The three partial configurations in the synergy ablation come from an earlier 60-epoch
  generation: the pure-CGM baseline stopped at epoch 58 after a numerical failure and the
  enhancement-only variant was terminated at epoch 44, whereas the complete model was
  trained for 100 epochs.

## Attribution

`src/Res.py` is the ResNet implementation from torchvision (BSD-3-Clause).
`src/Swin.py` follows the Swin Transformer reference implementation (MIT, Microsoft).
`src/utils/lr_scheduler.py` is Hang Zhang's scheduler (MIT), original header retained.
Everything else is ours and released under the MIT license in `LICENSE`.

## 中文说明

本仓库提供 WPUGNet 的模型代码、训练与评估脚本、完整复现所需的划分方式，以及论文中全部
实验结果表格（`results/tables/`）和逐图 Dice 数据（`results/per_sample/`）。权重通过
Releases 页面分发，不放在仓库目录内。数据集 BUSI、UDIAT、BrEaST 均为公开数据集，需自行
下载，仓库不附带任何影像数据。

核心结论：小波模块单独使用不带来增益（纯嫁接基线 0.794，仅加小波前端 0.773，仅加小波增强
0.762），三者与双骨干嫁接同时启用才达到 0.816，说明增益来自频率先验与跨骨干融合的协同。
