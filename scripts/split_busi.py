#!/usr/bin/python3
# coding=utf-8
"""Stratified 445/89/113 split of the 647 BUSI lesion cases.

Cases are shuffled within class (benign 437, malignant 210) under a fixed seed
and then apportioned at the 445:89:113 ratio, so the class balance of every
split matches the full set. Writes split_manifest.csv recording the assignment
of each case; the same seed reproduces the splits used in the paper.
"""

import argparse
import csv
import os
import random
import shutil

DEFAULT_SRC = os.path.join("data", "BUSI_prepared")
DEFAULT_DST = os.path.join("data", "BUSI_ft")
SEED = 42
SPLIT = {"train": 445, "val": 89, "test": 113}
TOTAL = sum(SPLIT.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=DEFAULT_SRC, help="output of prepare_busi.py")
    ap.add_argument("--dst", default=DEFAULT_DST, help="destination for the split")
    args = ap.parse_args()

    files = sorted(os.listdir(os.path.join(args.src, "images")))
    assert len(files) == TOTAL, "expected %d cases, found %d" % (TOTAL, len(files))

    rng = random.Random(SEED)
    by_class = {"benign": [], "malignant": []}
    for name in files:
        by_class[name.split("_")[1]].append(name)

    for split in SPLIT:
        d = os.path.join(args.dst, split)
        os.makedirs(os.path.join(d, "images"), exist_ok=True)
        os.makedirs(os.path.join(d, "masks"), exist_ok=True)

    pool = {}
    for cls, names in by_class.items():
        names = names[:]
        rng.shuffle(names)
        n_tr = round(len(names) * SPLIT["train"] / TOTAL)
        n_va = round(len(names) * SPLIT["val"] / TOTAL)
        pool[cls] = {
            "train": names[:n_tr],
            "val": names[n_tr:n_tr + n_va],
            "test": names[n_tr + n_va:],
        }

    rows = []
    counts = {s: 0 for s in SPLIT}
    for split in SPLIT:
        for cls, part in pool.items():
            for name in part[split]:
                shutil.copy(os.path.join(args.src, "images", name),
                            os.path.join(args.dst, split, "images", name))
                shutil.copy(os.path.join(args.src, "masks", name),
                            os.path.join(args.dst, split, "masks", name))
                counts[split] += 1
                rows.append({"file": name, "class": cls, "split": split})

    with open(os.path.join(args.dst, "split_manifest.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "class", "split"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["split"], r["file"])))

    print("split: train=%d val=%d test=%d seed=%d"
          % (counts["train"], counts["val"], counts["test"], SEED))
    for cls in ["benign", "malignant"]:
        print("  %s: %s" % (cls, " ".join("%s=%d" % (s, len(pool[cls][s])) for s in SPLIT)))
    print("output: %s" % args.dst)


if __name__ == "__main__":
    main()
