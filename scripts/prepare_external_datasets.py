# -*- coding: utf-8 -*-
"""Stage the UDIAT and BrEaST datasets for the cross-dataset experiments.

UDIAT: 163 image/label pairs under <root>/img and <root>/label, same filename.
BrEaST: 252 pairs in one folder, the image as case*_tumor.png and the mask as
the same stem without the _tumor suffix.

Both are shuffled under a fixed seed and split 60/15/25 into
external_data/<dataset>/{train,val,test}/{images,masks}.
"""

import argparse
import os
import random
import shutil

SEED = 42
RATIO = (0.6, 0.15, 0.25)


def split_names(names):
    names = sorted(names)
    random.Random(SEED).shuffle(names)
    n1 = round(len(names) * RATIO[0])
    n2 = round(len(names) * RATIO[1])
    return {"train": names[:n1], "val": names[n1:n1 + n2], "test": names[n1 + n2:]}


def write_split(out_root, name, pairs):
    lookup = {out_name: (img, msk) for img, msk, out_name in pairs}
    parts = split_names(lookup.keys())
    for split, names in parts.items():
        d_img = os.path.join(out_root, name, split, "images")
        d_msk = os.path.join(out_root, name, split, "masks")
        os.makedirs(d_img, exist_ok=True)
        os.makedirs(d_msk, exist_ok=True)
        for out_name in names:
            img, msk = lookup[out_name]
            shutil.copy(img, os.path.join(d_img, out_name))
            shutil.copy(msk, os.path.join(d_msk, out_name))
    print("%s: train=%d val=%d test=%d"
          % (name, len(parts["train"]), len(parts["val"]), len(parts["test"])))


def collect_udiat(root):
    pairs = []
    for name in sorted(os.listdir(os.path.join(root, "img"))):
        if name.endswith(".png"):
            pairs.append((os.path.join(root, "img", name),
                          os.path.join(root, "label", name), name))
    return pairs


def collect_breast(root):
    pairs = []
    for name in sorted(os.listdir(root)):
        if not name.endswith("_tumor.png"):
            continue
        image_name = name[:-len("_tumor.png")] + ".png"
        image_path = os.path.join(root, image_name)
        if os.path.exists(image_path):
            pairs.append((image_path, os.path.join(root, name), image_name))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--udiat", default=os.path.join("data", "UDIAT", "UDIAT"),
                    help="UDIAT root containing img/ and label/")
    ap.add_argument("--breast", default=os.path.join("data", "BrEaST", "BrEaST-Lesions_USG-images_and_masks"),
                    help="BrEaST folder with case*_tumor.png")
    ap.add_argument("--out", default="external_data", help="output directory")
    args = ap.parse_args()

    write_split(args.out, "udiat", collect_udiat(args.udiat))
    write_split(args.out, "breast", collect_breast(args.breast))


if __name__ == "__main__":
    main()
