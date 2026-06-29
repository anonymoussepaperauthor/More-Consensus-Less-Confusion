import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tools.ezr import *
from tools.stats import *
from tools.causal_tools import causalTree, causalTreeLeaf
import random
import os

TREATMENTS = ["ezr", "causal"]


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

    the.Budget  = 50
    the.Impurity= "gini"
    the.acq     = "near"
    the.leaf    = 3
    # =========================================================
    # Part 1: Performance (20 train/holdout splits per treatment)
    # =========================================================
    performance_error = {}
    error_dist        = {}   
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
            # All treatments use the same labeled rows from likely()
            labels  = random.sample(train.rows, min(the.Budget, len(train.rows))) if trt == "random" else likely(train)

            if trt == "ezr":
                tree     = Tree(clone(train, labels))
                top_rows = sorted(
                    [(treeLeaf(tree, row).mu, row) for row in holdout.rows],
                    key=lambda x: x[0]
                )[:the.Check]
            elif trt == "causal":
                ctree, canon = causalTree(clone(train, labels))
                top_rows     = sorted(
                    [(causalTreeLeaf(canon, ctree, row).mu, row)
                     for row in holdout.rows],
                    key=lambda x: x[0]
                )[:the.Check]

            trt_perf = max( 
                win(min([disty(all_data, row) for _, row in top_rows])) ,
                win(min(disty(all_data,row) for row in labels))
            )
            ref_opt  = win(min(disty(all_data, row) for row in holdout.rows + labels))
            err      = ref_opt - trt_perf
            errors.append(err)
            mse     += abs(err) ** 2

        error_dist[trt]        = errors
        performance_error[trt] = (mse / repeats) ** 0.5

    # Statistical ranking of performance
    pooled_sd = adds([e for errs in error_dist.values() for e in errs]).sd
    best_performances = top(error_dist, Ks=0.9, Delta="smed", eps=pooled_sd * 0.35)


    # =========================================================
    # Part 2: Stability (single shared train/test split)
    # =========================================================
    all_data = Data(csv(file_directory))
    # all_data.rows = shuffle(all_data.rows)
    # half       = len(all_data.rows) // 2
    # train_rows = all_data.rows[:half]
    # test_pool  = all_data.rows[half:]
    # tests_size = min(100, len(test_pool))
    # test_rows  = test_pool[:tests_size]

    tests_size = min(100, int(len(all_data.rows) * 0.5))
    test, train = clone(all_data, all_data.rows[:tests_size]), clone(all_data, all_data.rows[tests_size:])

    # train = clone(all_data, train_rows)
    # test  = clone(all_data, test_rows)

    # Build 20 models per treatment (shared labels across treatments)
    # random_models   = []
    ezr_models      = []
    causal_models   = []

    for rand_seed in range(repeats):
        the.seed = rand_seed
        random.seed(rand_seed)
        labels  = likely(train)
        data_labels = clone(train, labels)
        # random
        # random_models.append(random.sample(train.rows, min(the.Budget, len(train.rows))))
        # ezr
        ezr_models.append(Tree(data_labels))
        # causal_ezr
        ctree, canon = causalTree(clone(train, labels))
        causal_models.append((canon, ctree))

    stability_agreement = {}
    # Compute win-score predictions for every (treatment, model, test row)
    all_win_scores = {"ezr":[],"causal":[]}
    for row in test.rows:
        for trt in TREATMENTS:
            if trt == "ezr":
                win_scores = [win(treeLeaf(tree, row).mu)
                    for tree in ezr_models]
            elif trt == "causal":
                win_scores = [win(causalTreeLeaf(sdata, ctree, row).mu)
                    for sdata, ctree in causal_models]
            all_win_scores[trt].append(win_scores)
    
    # Threshold-based stability agreement
    for trt in TREATMENTS:
        agreement = 0
        for win_scores in all_win_scores[trt]:
            if adds(win_scores).sd < 0.35 * b4_wins.sd:
                agreement += 1
        stability_agreement[trt] = agreement * 100 // tests_size
    # top()-based stability: per row, which treatment is more stable?
    best_stability = {trt: 0 for trt in TREATMENTS}
    for row_idx in range(tests_size):
        row_sd = {trt: adds(all_win_scores[trt][row_idx]).sd
                  for trt in TREATMENTS}
        pooled_sd = adds([sds for sds in row_sd.values()]).sd
        bests_in_row = top({k: [v] for k, v in row_sd.items()},
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
