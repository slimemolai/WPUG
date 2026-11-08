#!/usr/bin/python3
# coding=utf-8
"""Paired significance tests over the per-sample Dice files in results/per_sample.

Every model is evaluated on the same 113 BUSI test images, so the comparison is
paired by filename. For each pair we run a two-sided Wilcoxon signed-rank test
with zero differences split evenly between the two sides (`zero_method=zsplit`),
and report the matched-pairs rank-biserial correlation r = Z / sqrt(2n) and the
probability of superiority, PS = P(A > B) + 0.5 * P(A = B).
"""

import argparse
import glob
import itertools
import os

import numpy as np
from scipy.stats import norm, wilcoxon

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "per_sample")
FAIL_THRESHOLD = 0.60


def load(path):
    import csv
    scores = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scores[row["file_name"].strip()] = float(row["dice"])
    return scores


def load_all(directory):
    models = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        models[name] = load(path)
    return models


def compare(a, b):
    """a and b are dicts keyed by filename."""
    shared = sorted(set(a) & set(b))
    x = np.array([a[k] for k in shared])
    y = np.array([b[k] for k in shared])
    diff = x - y
    _, p = wilcoxon(x, y, alternative="two-sided", zero_method="zsplit")
    z = norm.ppf(1 - p / 2)
    ps = float(np.mean((diff > 0) + 0.5 * (diff == 0)))
    return {
        "n": len(shared),
        "delta": float(np.mean(diff)),
        "p": float(p),
        "r_mrb": float(z / np.sqrt(2 * len(shared))),
        "ps": ps,
    }


def summarize(models):
    lines = ["model,n,mean_dice,std_dice,failures"]
    for name, scores in models.items():
        v = np.array(list(scores.values()))
        lines.append("%s,%d,%.4f,%.4f,%d"
                     % (name, len(v), v.mean(), v.std(ddof=1),
                        int((v < FAIL_THRESHOLD).sum())))
    return lines


def pairwise(names, models, reference=None):
    lines = ["model_a,model_b,n,delta_dice,p_value,r_mrb,ps"]
    pairs = ([(reference, n) for n in names if n != reference] if reference
             else itertools.combinations(names, 2))
    for a, b in pairs:
        r = compare(models[a], models[b])
        lines.append("%s,%s,%d,%+.4f,%.4g,%.3f,%.3f"
                     % (a, b, r["n"], r["delta"], r["p"], r["r_mrb"], r["ps"]))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=DEFAULT_DIR, help="directory of per-sample CSVs")
    ap.add_argument("--reference", help="compare every model against this one only")
    ap.add_argument("--out", help="write CSV output here instead of stdout")
    args = ap.parse_args()

    models = load_all(args.dir)
    if not models:
        raise SystemExit("no per-sample CSVs found in %s" % args.dir)
    if args.reference and args.reference not in models:
        raise SystemExit("unknown reference model: %s" % args.reference)

    lines = summarize(models) + [""] + pairwise(sorted(models), models, args.reference)
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print("written: %s" % args.out)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
