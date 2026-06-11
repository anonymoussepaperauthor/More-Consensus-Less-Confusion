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

all_data.rows = shuffle(all_data.rows)
tests_size = min(100, int(len(all_data.rows) * 0.3))
test, train = clone(all_data, all_data.rows[:tests_size]), clone(all_data, all_data.rows[tests_size:])


all_sigmas = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1]

initial_aggreement = {s:0 for s in all_sigmas}
refined_aggreement = {s:0 for s in all_sigmas}

the.Budget  = 20
the.acq     = "xploit"
the.leaf     = 2
the.Impurity = "entropy"
trees = []
for rand_seed in range(repeats):
    the.seed    = rand_seed
    random.seed(the.seed)
    labels = likely(train)
    tree   = Tree(clone(train, labels))
    trees.append(tree)

for idx, row in enumerate(test.rows):
    outputs = [win(treeLeaf(tree,row).mu) for tree in trees]
    preds = adds( outputs )
    for sigma in all_sigmas:
        if preds.sd < sigma * b4_wins.sd:    initial_aggreement[sigma] += 1

the.Budget  = 50
the.acq     = "near"
the.leaf     = 3
the.Impurity = "gini"
trees = []
for rand_seed in range(repeats):
    the.seed    = rand_seed
    random.seed(the.seed)
    labels = likely(train)
    tree   = Tree(clone(train, labels))
    trees.append(tree)

for idx, row in enumerate(test.rows):
    outputs = [win(treeLeaf(tree,row).mu) for tree in trees]
    preds = adds( outputs )
    for sigma in all_sigmas:
        if preds.sd < sigma * b4_wins.sd:    refined_aggreement[sigma] += 1

print(f"Dataset, {file_directory.split("/")[-1][:-4]}")
print(f"Sigmas", end=", ")
print(*all_sigmas, sep=", ")
print(f"initial_Aggrement", end=", ")
print(*list(initial_aggreement.values()), sep=", ")
print(f"refined_Aggrement", end=", ")
print(*list(refined_aggreement.values()), sep=", ")