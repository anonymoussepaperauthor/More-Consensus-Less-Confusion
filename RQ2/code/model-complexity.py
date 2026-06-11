import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tools.ezr import *
from tools.stats import *

file_directory = sys.argv[1] if len(sys.argv) > 1 else "data/optimize/misc/auto93.csv"
repeats = 20

all_data = Data(csv(file_directory))
ys      = [disty(all_data,row) for row in all_data.rows]
b4      = adds(ys)
win     = lambda v: int(100*(1 - (v - b4.lo)/(b4.mu - b4.lo)))
b4_wins = adds([win(k) for k in ys])

the.Check   = 10
the.Budget  = 50
the.acq     = "near"

treatments = [1,3,5,7,9]

performace_error = {}
error_dist = {}
for leaf in treatments:
    the.leaf     = leaf
    mse = 0
    error = []
    for rand_seed in range(repeats):
        the.seed    = rand_seed
        random.seed(the.seed)
        shuffled_rows = random.sample(all_data.rows, len(all_data.rows))
        half = int(0.5 * len(all_data.rows))
        train, holdout = clone(all_data, shuffled_rows[:half]), clone(all_data, shuffled_rows[half:])
        labels = likely(train)
        tree   = Tree(clone(train, labels))
        top_rows = sorted( [(treeLeaf(tree, row).mu, row) for row in holdout.rows], key=lambda x: x[0])[:the.Check]
        ezr_performace = win( sorted([disty(all_data, row) for _, row in top_rows])[0] )
        # print("Suggested row win:\t", ezr_performace)
        # print("Referenced Optimal:\t", win(min(disty(all_data, row) for row in holdout.rows)))
        referenced_optima = win(min(disty(all_data, row) for row in holdout.rows))
        mse += abs(ezr_performace - referenced_optima) ** 2
        error.append(referenced_optima - ezr_performace)
    error_dist[leaf] = error
    performace_error[leaf] = (mse / repeats) ** 0.5

pooled_sd = adds([e for errs in error_dist.values() for e in errs]).sd
best_performances = top(error_dist, Ks=0.9, Delta="medium", eps=0.35*pooled_sd)

all_data.rows = shuffle(all_data.rows)
tests_size = min(100, int(len(all_data.rows) * 0.3))
test, train = clone(all_data, all_data.rows[:tests_size]), clone(all_data, all_data.rows[tests_size:])
the.Check   = 10
the.Budget  = 50
the.acq     = "near"

stability_aggreement = {}
stability_comp = [{acq:0 for acq in treatments} for _ in range(100)]
for leaf in treatments:
    the.leaf     = leaf
    trees = []
    for rand_seed in range(repeats):
        the.seed    = rand_seed
        random.seed(the.seed)
        labels = likely(train)
        tree   = Tree(clone(train, labels))
        trees.append(tree)

    aggreement = 0
    for idx, row in enumerate(test.rows):
        outputs = [win(treeLeaf(tree,row).mu) for tree in trees]
        preds = adds( outputs )
        stability_comp[idx][leaf] = preds.sd
        if preds.sd < 0.35 * b4_wins.sd:
            aggreement += 1

    stability_aggreement[leaf] = (aggreement * 100) // tests_size
# print("Stability,", stability_metric)
best_stability = {acq:0 for acq in treatments}
for row_stability in stability_comp:
    pooled_sd = adds([sds for sds in row_stability.values()]).sd
    bests_in_row = top({k:[v] for k,v in row_stability.items()}, Ks=0.9, Delta="medium", eps=0.35*pooled_sd)
    # print(bests_in_row, row_stability)
    for m in bests_in_row:
        best_stability[m] += 1

print("trt, performance_error, stability_aggreement, best_performance, best_stability")
for trt in treatments:
    print(f"{trt}, {performace_error[trt]:.2f}, {stability_aggreement[trt]}, {1 if trt in best_performances else 0}, {best_stability[trt]}")
