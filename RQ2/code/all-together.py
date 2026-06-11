import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tools.ezr import *
from tools.stats import *

file_directory = sys.argv[1] if len(sys.argv) > 1 else "data/optimize/misc/auto93.csv"
all_data = Data(csv(file_directory))
ys      = [disty(all_data,row) for row in all_data.rows]
b4      = adds(ys)
win     = lambda v: int(100*(1 - (v - b4.lo)/(b4.mu - b4.lo)))
b4_wins = adds([win(k) for k in ys])
repeats = 20

tests_size = min(100, int(len(all_data.rows) * 0.3))
half = int(0.5 * len(all_data.rows))

shuffled_rows = random.sample(all_data.rows, len(all_data.rows))
train, holdout = clone(all_data, shuffled_rows[:half]), clone(all_data, shuffled_rows[half:half + tests_size])

## Initial Settings:
the.Budget  = 20
the.acq     = "xploit"
the.leaf     = 2
the.Impurity = "entropy"
the.check = 10
initial_trees = []
for rand_seed in range(repeats):
    the.seed    = rand_seed
    random.seed(the.seed)
    labels = likely(train)
    tree   = Tree(clone(train, labels))
    initial_trees.append(tree)

## Refined Settings:
the.Budget  = 50
the.acq     = "near"
the.leaf     = 3
the.Impurity = "gini"
the.check = 10
refined_trees = []
for rand_seed in range(repeats):
    the.seed    = rand_seed
    random.seed(the.seed)
    labels = likely(train)
    tree   = Tree(clone(train, labels))
    refined_trees.append(tree)


print(f"b4-sd,{b4_wins.sd:.2f}")
row = holdout.rows[0]
initial_sds = []
refined_sds = []
for row in holdout.rows:
    initial_outputs = [win(treeLeaf(tree,row).mu) for tree in initial_trees]
    initial_sds.append(adds(initial_outputs).sd)

    refined_outputs = [win(treeLeaf(tree,row).mu) for tree in refined_trees]
    refined_sds.append( adds(refined_outputs).sd)
refined_sds
print(f"Init-sd,{sorted(initial_sds)[tests_size//2]:.1f}")
print(f"Rfn-sd,{sorted(refined_sds)[tests_size//2]:.1f}")    

# Stability (sd)
# -----------------------------------
# Performance (md)

print(f"b4-md,{b4_wins.mu:.2f}")
## Initial Settings:
the.Budget  = min(20, len(all_data.rows) // 2)
the.acq     = "xploit"
the.leaf     = 2
the.Impurity = "entropy"
the.check = 10
initial_mds = []
for rand_seed in range(repeats):
    the.seed    = rand_seed + 12345
    random.seed(the.seed)
    shuffled_rows = random.sample(all_data.rows, len(all_data.rows))
    half = int(0.5 * len(all_data.rows))
    train, holdout = clone(all_data, shuffled_rows[:half]), clone(all_data, shuffled_rows[half:])
    labels = likely(train)
    tree   = Tree(clone(train, labels))
    top_rows = sorted( [(treeLeaf(tree, row).mu, row) for row in holdout.rows], key=lambda x: x[0])[:the.Check]
    initial_mds.append( win( sorted([disty(all_data, row) for _, row in top_rows])[0] ) )

## Refined Settings:
the.Budget  = min(50, len(all_data.rows) // 2)
the.acq     = "near"
the.leaf     = 3
the.Impurity = "gini"
the.check = 10
refined_mds = []
for rand_seed in range(repeats):
    the.seed    = rand_seed + 12345
    random.seed(the.seed)
    shuffled_rows = random.sample(all_data.rows, len(all_data.rows))
    half = int(0.5 * len(all_data.rows))
    train, holdout = clone(all_data, shuffled_rows[:half]), clone(all_data, shuffled_rows[half:])
    labels = likely(train)
    tree   = Tree(clone(train, labels))
    top_rows = sorted( [(treeLeaf(tree, row).mu, row) for row in holdout.rows], key=lambda x: x[0])[:the.Check]
    refined_mds.append( win( sorted([disty(all_data, row) for _, row in top_rows])[0] ) )
print(f"Init-md,{sorted(initial_mds)[repeats//2]:.1f}")
print(f"Rfn-md,{sorted(refined_mds)[repeats//2]:.1f}")    