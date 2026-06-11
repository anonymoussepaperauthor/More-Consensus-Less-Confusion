"""
jaccard_batch.py — batch RQ0 structural-instability measurement.

For every .csv under a given folder (recursively), build N trees with N
different seeds, compute pairwise Jaccard on used feature sets, and append
one row per dataset to ./jaccard.csv with columns:

    dataset, n_seeds, min_jaccard, max_jaccard, mean_jaccard

Usage:
  python3 jaccard_batch.py <folder> [n_seeds] [out_csv]

Defaults: n_seeds=20, out_csv=./jaccard.csv

Notes:
  - Datasets that crash (malformed CSV, EZR error, etc.) are written with
    empty metric fields and an error tag in the dataset column suffix, so
    the run as a whole isn't aborted by one bad file.
  - "Feature used" = appears in at least one split node (strict).
"""
import sys, os, random, itertools, csv as _csv
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from math import e
from tools.ezr import Data, csv, clone, shuffle, likely, Tree, treeNodes, the


def features_used_in_tree(data, tree):
    """Set of feature NAMES (column txt) that appear in any split."""
    names = dict()
    for lvl, node in treeNodes(tree):
        if lvl == 0:
            continue
        _op, at, _y = node.how
        names[data.cols.names[at]] = names.get(data.cols.names[at], 0) + (1/(lvl+1))
    return names


def build_one_tree(data_template, all_rows, seed, settings):
    """One run of EZR's pipeline: shuffle -> active-learning labels -> tree."""
    if settings == "init":
        the.Budget  = 20
        the.acq     = "xploit"
        the.leaf     = 2
        the.Impurity = "entropy"
    else: 
        the.Budget  = 50
        the.acq     = "near"
        the.leaf     = 3
        the.Impurity = "gini"        

    random.seed(seed)
    rows = all_rows[:]
    shuffle(rows)
    half = len(rows) // 2
    train = rows[:half]
    labels = likely(clone(data_template, train))
    return Tree(clone(data_template, labels))


def jaccard(a, b):
    """|A ∩ B| / |A ∪ B|; 1.0 when both sets are empty."""
    all_keys = set(list(a.keys()) + list(b.keys()))
    mins, maxs = 0, 0
    for key in all_keys:
        mins += min(a.get(key, 0), b.get(key, 0))
        maxs += max(a.get(key, 0), b.get(key, 0))
    return mins/maxs


def jaccard_for_dataset(path, n_seeds, settings = "init"):
    """Return (min, max, mean) pairwise Jaccard over n_seeds trees on `path`."""
    data = Data(csv(path))
    all_rows = data.rows[:]

    feature_sets = []
    for seed in range(1, n_seeds + 1):
        tree = build_one_tree(data, all_rows, seed, settings)
        feature_sets.append(features_used_in_tree(data, tree))

    pairs = list(itertools.combinations(range(n_seeds), 2))
    js = [jaccard(feature_sets[i], feature_sets[j]) for i, j in pairs]
    return min(js), max(js), sum(js) / len(js)


def find_csvs(folder):
    """Yield every .csv under `folder`, recursively, sorted for stable output."""
    out = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".csv"):
                out.append(os.path.join(root, f))
    out.sort()
    return out


def main():

    folder   = sys.argv[1] if len(sys.argv) > 2 else "data/optimize/"
    n_seeds  = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    out_path = sys.argv[3] if len(sys.argv) > 3 else "rq1/results/jaccard.csv"

    csv_paths = find_csvs(folder)

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["dataset",
                    "init_min_jaccard", "init_max_jaccard", "init_mean_jaccard",
                    "rfn_min_jaccard", "rfn_max_jaccard", "rfn_mean_jaccard"])

        for path in csv_paths:
            # Use the bare filename (no extension) as the dataset id; if you
            # have name collisions across subfolders, swap to relpath here.
            name = os.path.splitext(os.path.basename(path))[0]
            try:
                print(name)
                init_lo, init_hi, init_mu = jaccard_for_dataset(path, n_seeds, settings="init")
                rfn_lo, rfn_hi, rfn_mu = jaccard_for_dataset(path, n_seeds, settings="rfn")
                w.writerow([name,
                            f"{init_lo:.2f}", f"{init_hi:.2f}", f"{init_mu:.2f}", 
                            f"{rfn_lo:.2f}", f"{rfn_hi:.2f}", f"{rfn_mu:.2f}"])
            except Exception as e:
                # Don't let one broken file kill the whole run; tag it and move on.
                w.writerow([f"{name} [ERROR: {type(e).__name__}]",
                            n_seeds, "", "", ""])
            fh.flush()  # so partial results are visible if you Ctrl-C


if __name__ == "__main__":
    main()