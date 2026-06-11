"""
4.py — K-Means vs CURE vs HDBSCAN vs EZR: Stability & Performance comparison.

All four run under identical conditions
(same splits, same seeds, fixed budget = 50).

CURE algorithm:
  1. Initialise clusters via K-Means++ seeding + Lloyd assignment.
  2. For each cluster, pick CURE_N_REPR representative points via farthest-first.
  3. Adaptive per-cluster shrink: shrink_i = 1 - (intra_spread_i / global_spread).
  4. Shrink each representative toward the cluster mean by shrink_i.
  5. Predict: nearest shrunken representative -> nearest actual cluster member.

HDBSCAN algorithm (via the `hdbscan` library):
  1. Fit HDBSCAN on the budget rows using a numeric projection of each row
     (missing values replaced by column mean/mode).
  2. Points labelled -1 (noise) are reassigned to the nearest non-noise
     cluster member using distx, so every test row has a valid prediction.
  3. Predict: assign test row to its nearest cluster (by distx) and return
     the actual cluster member nearest to the query row.

Granularity knob (shared across K-Means, CURE, HDBSCAN):
  k / min_cluster_size = budget // the.leaf     (mirrors tree leaf-size)
  HDBSCAN min_samples  = max(1, min_cluster_size // 2)

Fair comparison:
  For every seed, likely(train) selects `budget` labeled rows.
  All four treatments use those SAME rows.

Treatments: kmeans, cure, hdbscan, ezr

Part 1 — Performance (20 train/holdout splits per treatment):
  Pick top-Check holdout rows -> best win -> RMSE vs true holdout best.

Part 2 — Stability (single shared train/test split, 20 models per treatment):
  Build 20 models (different seeds).
  For each test row, 20 predictions -> sd.
  sd < 0.35 * b4_wins.sd -> stable.

Output: one CSV per dataset under results/5.3/{dataset}.csv
  Columns: trt, performance_error, stability_agreement, best_performance, best_stability
    performance_error:    RMSE of win-score gap (existing metric)
    stability_agreement:  % of test rows with sd < 0.35 * b4_wins.sd (existing)
    best_performance:     1 if treatment is in the statistically best group via top()
    best_stability:       # of test rows where treatment is statistically most stable via top()
"""
from math import *
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tools.ezr import *
from tools.stats import *
import random
import hdbscan as hdbscan_lib
import numpy as np

LLOYD_ITERS = 20       # max Lloyd iterations for K-Means and CURE
TREATMENTS  = ["kmeans", "cure", "hdbscan", "ezr"]

# CURE hyper-parameters
CURE_N_REPR = 5        # representative points per cluster
# shrink is computed adaptively per cluster (no global constant)


# =============================================================================
# K-Means (medoid update, same as 5.2.py)
# =============================================================================
def kmeans(data, rows, k=None, max_iter=LLOYD_ITERS):
    """K-Means clustering using ezr's distx for mixed-type distance.

    Returns:
        centroids: list of k medoid rows.
    """
    if k is None:
        k = max(2, len(rows) // the.leaf)
    if len(rows) <= k:
        return rows[:]

    centroids = distKpp(data, rows=rows, k=k)

    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        for row in rows:
            dists   = [distx(data, row, c) for c in centroids]
            nearest = dists.index(min(dists))
            clusters[nearest].append(row)

        new_centroids = []
        for ci, cluster in enumerate(clusters):
            if not cluster:
                new_centroids.append(centroids[ci])
                continue
            best_row = min(
                cluster,
                key=lambda r: sum(distx(data, r, other) for other in cluster)
            )
            new_centroids.append(best_row)

        if all(c1 is c2 for c1, c2 in zip(centroids, new_centroids)):
            break
        centroids = new_centroids

    return centroids

def predict_nearest(data, centroids, row):
    """Return the nearest medoid/centroid to `row`."""
    return min(centroids, key=lambda c: distx(data, row, c))


# =============================================================================
# CURE helpers
# =============================================================================
def _cluster_mean(data, cluster):
    """Numeric-aware mean row for a cluster.

    Numeric cols are averaged; symbolic cols use the most-common value.
    """
    n_cols = len(cluster[0])
    mean_row = []
    for col_idx in range(n_cols):
        col_obj = data.cols.all[col_idx] if col_idx < len(data.cols.all) else None
        vals    = [r[col_idx] for r in cluster if r[col_idx] != "?"]
        if vals and col_obj and hasattr(col_obj, "mu"):
            mean_row.append(sum(vals) / len(vals))
        else:
            mean_row.append(max(set(vals), key=vals.count) if vals else "?")
    return mean_row

def _farthest_first_repr(data, cluster, mean_row, n_repr):
    """Pick `n_repr` points from `cluster` via farthest-first traversal,
    seeded from the point farthest from the cluster mean."""
    if len(cluster) <= n_repr:
        return list(cluster)
    reprs = [max(cluster, key=lambda r: distx(data, r, mean_row))]
    while len(reprs) < n_repr:
        next_r = max(
            cluster,
            key=lambda r: min(distx(data, r, rep) for rep in reprs)
        )
        reprs.append(next_r)
    return reprs

def _cluster_spread(data, cluster, mean_row):
    """Mean distance from each cluster member to the cluster mean."""
    if not cluster:
        return 0.0
    return sum(distx(data, r, mean_row) for r in cluster) / len(cluster)

def _adaptive_shrink(intra_spread, global_spread, lo=0.05, hi=0.95):
    """Per-cluster shrink factor: shrink_i = 1 - (intra_spread_i / global_spread).

    Tight cluster -> high shrink; loose cluster -> low shrink.
    Clamped to [lo, hi].
    """
    if global_spread < 1e-12:
        return hi
    return max(lo, min(hi, 1.0 - intra_spread / global_spread))

def _shrink(data, reprs, mean_row, shrink):
    """Shrink each representative toward the cluster mean (numeric cols only)."""
    shrunk = []
    for rep in reprs:
        new_rep = []
        for col_idx, (rv, mv) in enumerate(zip(rep, mean_row)):
            col_obj = data.cols.all[col_idx] if col_idx < len(data.cols.all) else None
            if col_obj and hasattr(col_obj, "mu") and rv != "?" and mv != "?":
                new_rep.append(rv + shrink * (mv - rv))
            else:
                new_rep.append(rv)
        shrunk.append(new_rep)
    return shrunk

def cure(data, rows, k=None, n_repr=CURE_N_REPR, max_iter=LLOYD_ITERS):
    """CURE clustering with adaptive per-cluster shrink.

    Returns:
        list of (shrunken_reprs, cluster_rows) tuples — one per cluster.
    """
    if k is None:
        k = max(2, len(rows) // the.leaf)
    if len(rows) <= k:
        return [([r], [r]) for r in rows]

    global_mean   = _cluster_mean(data, rows)
    global_spread = _cluster_spread(data, rows, global_mean)
    centroids     = distKpp(data, rows=rows, k=k)

    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        for row in rows:
            dists   = [distx(data, row, c) for c in centroids]
            nearest = dists.index(min(dists))
            clusters[nearest].append(row)

        cluster_reprs = []
        new_centroids = []
        for ci, cluster in enumerate(clusters):
            if not cluster:
                cluster_reprs.append([centroids[ci]])
                new_centroids.append(centroids[ci])
                continue
            mean_row     = _cluster_mean(data, cluster)
            intra_spread = _cluster_spread(data, cluster, mean_row)
            shrink       = _adaptive_shrink(intra_spread, global_spread)
            reprs        = _farthest_first_repr(data, cluster, mean_row, n_repr)
            cluster_reprs.append(_shrink(data, reprs, mean_row, shrink))
            new_centroids.append(mean_row)

        new_clusters = [[] for _ in range(k)]
        for row in rows:
            best_ci, best_dist = 0, float("inf")
            for ci, reprs in enumerate(cluster_reprs):
                d = min(distx(data, row, rep) for rep in reprs)
                if d < best_dist:
                    best_dist = d
                    best_ci   = ci
            new_clusters[best_ci].append(row)

        if all(
            set(id(r) for r in c1) == set(id(r) for r in c2)
            for c1, c2 in zip(clusters, new_clusters)
        ):
            break
        clusters  = new_clusters
        centroids = new_centroids

    result = []
    for ci, cluster in enumerate(clusters):
        if not cluster:
            result.append(([centroids[ci]], []))
            continue
        mean_row     = _cluster_mean(data, cluster)
        intra_spread = _cluster_spread(data, cluster, mean_row)
        shrink       = _adaptive_shrink(intra_spread, global_spread)
        reprs        = _farthest_first_repr(data, cluster, mean_row, n_repr)
        result.append((_shrink(data, reprs, mean_row, shrink), cluster))
    return result

def predict_cure(data, cure_clusters, row):
    """Return the actual cluster member nearest to `row` via shrunken reps."""
    best_dist, best_cluster = float("inf"), None
    for reprs, cluster in cure_clusters:
        d = min(distx(data, row, rep) for rep in reprs)
        if d < best_dist:
            best_dist    = d
            best_cluster = cluster
    if not best_cluster:
        return row
    return min(best_cluster, key=lambda r: distx(data, row, r))


# =============================================================================
# HDBSCAN helpers
# =============================================================================
def _row_to_numeric(data, row):
    """Convert a mixed-type row to a float vector for HDBSCAN.

    Numeric columns: use the value (or column mean for missing).
    Symbolic columns: encode as 0.0 (treated as a constant — HDBSCAN
    operates on the numeric subspace; distx handles full mixed distance
    at prediction time).
    """
    vec = []
    for col_obj, val in zip(data.cols.all, row):
        if hasattr(col_obj, "mu"):          # Num column
            if val == "?":
                vec.append(col_obj.mu)
            else:
                vec.append(float(val))
        else:                               # Sym column — skip (encode as 0)
            vec.append(0.0)
    return vec

def fit_hdbscan(data, rows, min_cluster_size=None):
    """Fit HDBSCAN on `rows` and return cluster assignments.

    Args:
        data:             ezr Data object (for column metadata).
        rows:             list of rows to cluster.
        min_cluster_size: smallest cluster size HDBSCAN will form.
                          Defaults to max(2, len(rows) // the.leaf).

    Returns:
        clusters: list of lists — each inner list is one cluster's rows.
                  Noise points (label -1) are reassigned to their
                  nearest non-noise neighbour's cluster.
    """
    if min_cluster_size is None:
        min_cluster_size = max(2, len(rows) // the.leaf)

    min_samples = max(1, min_cluster_size // 2)

    X = np.array([_row_to_numeric(data, r) for r in rows], dtype=float)

    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(X)

    # Group rows by cluster label
    label_set = sorted(set(labels) - {-1})

    if not label_set:
        # All noise — treat as one single cluster
        return [list(rows)]

    cluster_map = {lbl: [] for lbl in label_set}
    noise_rows  = []
    for row, lbl in zip(rows, labels):
        if lbl == -1:
            noise_rows.append(row)
        else:
            cluster_map[lbl].append(row)

    clusters = [cluster_map[lbl] for lbl in label_set]

    # Reassign noise points to nearest non-noise cluster member
    for nr in noise_rows:
        best_ci, best_dist = 0, float("inf")
        for ci, cluster in enumerate(clusters):
            d = min(distx(data, nr, r) for r in cluster)
            if d < best_dist:
                best_dist = d
                best_ci   = ci
        clusters[best_ci].append(nr)

    return clusters

def predict_hdbscan(data, clusters, row):
    """Return the actual cluster member nearest to `row`.

    Finds the cluster whose closest member is nearest (distx), then
    returns that closest member as the prediction.
    """
    best_dist, best_member = float("inf"), None
    for cluster in clusters:
        for member in cluster:
            d = distx(data, row, member)
            if d < best_dist:
                best_dist   = d
                best_member = member
    return best_member if best_member is not None else row

# =============================================================================
# Main experiment
# =============================================================================
def run(file_directory, repeats=20):
    all_data = Data(csv(file_directory))
    if not all_data.cols.y:
        if all_data.cols.klass:
            all_data.cols.y = [all_data.cols.klass]
        else:
            print(f"Error: No objective columns found in {file_directory}")
            return

    # Global baseline stats
    ys      = [disty(all_data, row) for row in all_data.rows]
    b4      = adds(ys)
    win     = lambda v: int(100 * (1 - (v - b4.lo) / (b4.mu - b4.lo)))
    b4_wins = adds([win(k) for k in ys])

    budget           = 50
    the.Budget       = budget
    the.Check        = 10
    k                = max(2, budget // the.leaf)   # K-Means clusters
    min_cluster_size = max(2, budget // the.leaf)   # HDBSCAN granularity

    # =========================================================
    # Part 1: Performance (20 train/holdout splits per treatment)
    # =========================================================
    # For each repeat, likely(train) selects the labeled 50 rows.
    # All three treatments use those SAME rows for a fair comparison.
    performance_error = {}
    error_dist        = {}   # {trt: [per-seed errors]} for top()

    for trt in TREATMENTS:
        mse    = 0
        errors = []
        for rand_seed in range(repeats):
            random.seed(rand_seed)
            the.seed      = rand_seed
            shuffled_rows = random.sample(all_data.rows, len(all_data.rows))
            half    = int(0.5 * len(all_data.rows))
            train   = clone(all_data, shuffled_rows[:half])
            holdout = clone(all_data, shuffled_rows[half:])

            # All treatments use the same labeled rows chosen by likely()
            labels  = likely(train)        # active-learning selection, budget=50
            sampled = labels[:budget]      # same rows for all methods this seed

            if trt == "kmeans":
                centroids = kmeans(clone(all_data, sampled), sampled, k=k)
                scored    = [
                    (disty(all_data, predict_nearest(all_data, centroids, row)), row)
                    for row in holdout.rows
                ]
                top_rows = sorted(scored, key=lambda x: x[0])[:the.Check]

            elif trt == "cure":
                cure_clusters = cure(clone(all_data, sampled), sampled, k=k)
                scored        = [
                    (disty(all_data, predict_cure(all_data, cure_clusters, row)), row)
                    for row in holdout.rows
                ]
                top_rows = sorted(scored, key=lambda x: x[0])[:the.Check]

            elif trt == "hdbscan":
                clusters = fit_hdbscan(
                    clone(all_data, sampled), sampled,
                    min_cluster_size=min_cluster_size
                )
                scored   = [
                    (disty(all_data, predict_hdbscan(all_data, clusters, row)), row)
                    for row in holdout.rows
                ]
                top_rows = sorted(scored, key=lambda x: x[0])[:the.Check]

            else:  # ezr tree
                tree     = Tree(clone(train, labels))
                top_rows = sorted(
                    [(treeLeaf(tree, row).mu, row) for row in holdout.rows],
                    key=lambda x: x[0]
                )[:the.Check]

            ezr_perf = win(sorted([disty(all_data, row) for _, row in top_rows])[0])
            ref_opt  = win(min(disty(all_data, row) for row in holdout.rows))
            err      = ref_opt - ezr_perf
            errors.append(err)
            mse     += abs(err) ** 2

        error_dist[trt]        = errors
        performance_error[trt] = (mse / repeats) ** 0.5

    # Efficient, concise pooled sd calculation
    pooled_sd = adds([e for errs in error_dist.values() for e in errs]).sd
    best_performances = top(error_dist, Ks=0.9, Delta="medium", eps=pooled_sd * 0.35)

    # =========================================================
    # Part 2: Stability (single shared train/test split, Option 1)
    # =========================================================
    # Each seed calls likely(train) once; all three treatments
    # build their model from those same labeled rows.
    random.seed(42)  # deterministic split independent of Part 1 RNG state
    all_data.rows = shuffle(all_data.rows)
    half       = len(all_data.rows) // 2
    train_rows = all_data.rows[:half]
    test_pool  = all_data.rows[half:]
    tests_size = min(100, len(test_pool))
    test_rows  = test_pool[:tests_size]

    train = clone(all_data, train_rows)
    test  = clone(all_data, test_rows)

    # Pre-compute one set of models per seed (shared labels across treatments)
    kmeans_models  = []
    cure_models    = []
    hdbscan_models = []
    tree_models    = []

    for rand_seed in range(repeats):
        the.seed = rand_seed
        random.seed(rand_seed)
        labels  = likely(train)        # active-learning selection, budget=50
        sampled = labels[:budget]      # same rows for all four treatments

        kmeans_models.append(kmeans(clone(all_data, sampled), sampled, k=k))
        cure_models.append(cure(clone(all_data, sampled), sampled, k=k))
        hdbscan_models.append(
            fit_hdbscan(clone(all_data, sampled), sampled,
                        min_cluster_size=min_cluster_size)
        )
        tree_models.append(Tree(clone(train, labels)))

    stability_agreement = {}

    # Compute win-score predictions for every (treatment, model, test row)
    # Shape: {trt: list-of-lists, outer=test_rows, inner=models}
    all_win_scores = {}
    for trt in TREATMENTS:
        per_row = []
        for row in test.rows:
            if trt == "kmeans":
                win_scores = [
                    win(disty(all_data, predict_nearest(all_data, centroids, row)))
                    for centroids in kmeans_models
                ]
            elif trt == "cure":
                win_scores = [
                    win(disty(all_data, predict_cure(all_data, cure_clusters, row)))
                    for cure_clusters in cure_models
                ]
            elif trt == "hdbscan":
                win_scores = [
                    win(disty(all_data, predict_hdbscan(all_data, clusters, row)))
                    for clusters in hdbscan_models
                ]
            else:  # ezr
                win_scores = [win(treeLeaf(tree, row).mu) for tree in tree_models]
            per_row.append(win_scores)
        all_win_scores[trt] = per_row

    # Threshold-based stability agreement (existing metric)
    for trt in TREATMENTS:
        agreement = 0
        for win_scores in all_win_scores[trt]:
            if adds(win_scores).sd < 0.35 * b4_wins.sd:
                agreement += 1
        stability_agreement[trt] = agreement * 100 // tests_size

    # top()-based stability: per row, which treatments have the
    # statistically highest (best) win-score distributions?
    # top() is ascending (lowest = best), so we negate win scores.
    best_stability = {trt: 0 for trt in TREATMENTS}
    for row_idx in range(tests_size):
        row_distributions = {
            trt: [-v for v in all_win_scores[trt][row_idx]]
            for trt in TREATMENTS
        }
        pooled_sd = adds([v for vals in row_distributions.values() for v in vals]).sd
        bests_in_row = top(row_distributions,
                          Ks=0.9, Delta="medium", eps=pooled_sd * 0.35)
        for trt in bests_in_row:
            best_stability[trt] += 1

    # =========================================================
    # Output
    # =========================================================
    header = "trt, performance_error, stability_agreement, best_performance, best_stability"
    print(header)
    lines = [header]
    for trt in TREATMENTS:
        bp = 1 if trt in best_performances else 0
        bs = best_stability[trt]
        line = f"{trt}, {performance_error[trt]:.2f}, {stability_agreement[trt]}, {bp}, {bs}"
        print(line)
        lines.append(line)




if __name__ == "__main__":
    file_directory = sys.argv[1] if len(sys.argv) > 1 else "data/optimize/misc/auto93.csv"
    run(file_directory, 20)
