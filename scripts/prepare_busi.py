#!/usr/bin/python3
# coding=utf-8
"""Convert the raw BUSI release into the layout used by the training scripts.

The source is the public Dataset_BUSI_with_GT folder, which stores images as
benign/xxx.png with the matching mask as benign/xxx_mask.png. Output is a flat
images/ and masks/ pair plus a manifest. Cases without a lesion (the normal
class) are recorded in the manifest but not written out, since segmentation
metrics are only defined on the 647 lesion cases.
"""

import argparse
import csv
import os

import cv2
import numpy as np

DEFAULT_SRC = os.path.join("data", "Dataset_BUSI_with_GT")
DEFAULT_DST = os.path.join("data", "BUSI_prepared")


def convert(src, dst, include_normal):
    os.makedirs(os.path.join(dst, "images"), exist_ok=True)
    os.makedirs(os.path.join(dst, "masks"), exist_ok=True)

    rows = []
    n_written = 0
    for cls in ["benign", "malignant", "normal"]:
        cls_dir = os.path.join(src, cls)
        if not os.path.isdir(cls_dir):
            print("[warn] missing directory %s" % cls_dir)
            continue
        files = sorted(os.listdir(cls_dir))
        images = sorted(f for f in files if not f.endswith("_mask.png") and f.endswith(".png"))
        for name in images:
            mask_name = "%s_mask.png" % name[:-4]
            if mask_name not in files:
                print("[warn] %s/%s has no mask, skipped" % (cls, name))
                continue
            img = cv2.imread(os.path.join(cls_dir, name))
            msk = cv2.imread(os.path.join(cls_dir, mask_name), cv2.IMREAD_GRAYSCALE)
            if img is None or msk is None:
                print("[warn] cannot read %s/%s, skipped" % (cls, name))
                continue
            h, w = msk.shape
            bin_mask = (msk > 127).astype(np.uint8) * 255
            has_lesion = bool(bin_mask.any())
            if not has_lesion and not include_normal:
                rows.append({"filename": name, "class": cls, "has_lesion": 0,
                             "w": w, "h": h, "written": 0})
                continue
            out_name = "busi_%s_%s" % (cls, name)
            cv2.imwrite(os.path.join(dst, "images", out_name), img)
            cv2.imwrite(os.path.join(dst, "masks", out_name), bin_mask)
            n_written += 1
            rows.append({"filename": out_name, "class": cls, "has_lesion": int(has_lesion),
                         "w": w, "h": h, "written": 1})

    with open(os.path.join(dst, "manifest.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "class", "has_lesion", "w", "h", "written"])
        writer.writeheader()
        writer.writerows(rows)

    n_lesion = sum(1 for r in rows if r["has_lesion"])
    print("cases=%d lesions=%d written=%d" % (len(rows), n_lesion, n_written))
    print("output: %s" % dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=DEFAULT_SRC, help="raw Dataset_BUSI_with_GT directory")
    ap.add_argument("--dst", default=DEFAULT_DST, help="output directory")
    ap.add_argument("--include-normal", action="store_true",
                    help="also write normal cases, with an all-zero mask")
    args = ap.parse_args()
    convert(args.src, args.dst, args.include_normal)


if __name__ == "__main__":
    main()
