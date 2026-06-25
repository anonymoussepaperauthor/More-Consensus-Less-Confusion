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

half = int(0.5 * len(all_data.rows))
init_quality, rfn_quality = [], []
for rand_seed in range(repeats):
    ## New draw from data
    the.seed    = rand_seed
    random.seed(the.seed)
    shuffled_rows = random.sample(all_data.rows, len(all_data.rows)) 
    train, holdout = clone(all_data, shuffled_rows[:half]), clone(all_data, shuffled_rows[half:])
    referenced_optima = win(min(disty(all_data, row) for row in holdout.rows))

    ## Initial Setting
    the.Budget  = 20
    the.acq     = "xploit"
    the.leaf    = 2
    the.Impurity= "entropy"
    the.Check   = 10
    
    
    
    labels = likely(train)
    tree   = Tree(clone(train, labels))
    top_rows = sorted( [(treeLeaf(tree, row).mu, row) for row in holdout.rows], key=lambda x: x[0])[:the.Check]
    init_optima = win( sorted([disty(all_data, row) for _, row in top_rows])[0] )
    init_quality.append(referenced_optima - init_optima)

    ## Refined Setting
    the.Budget  = 50
    the.acq     = "near"
    the.leaf     = 3
    the.Impurity = "gini"
    the.Check   = 10
    trees = []

    labels = likely(train)
    tree   = Tree(clone(train, labels))
    top_rows = sorted( [(treeLeaf(tree, row).mu, row) for row in holdout.rows], key=lambda x: x[0])[:the.Check]
    rfn_optima = win( sorted([disty(all_data, row) for _, row in top_rows])[0] )
    rfn_quality.append(referenced_optima - rfn_optima)   
print(f"Dataset, {file_directory.split("/")[-1][:-4]}")
print(f"Init-quality, {sorted(init_quality)[10]}")
print(f"Rfn-quality, {sorted(rfn_quality)[10]}")
print(f"best-quality, ", end="")
print(*list(top({"init":init_quality, "rfn":rfn_quality})), sep=", ")
print(f"Init-stability, {adds(init_quality).sd:.2f}")
print(f"Rfn-stability, {adds(rfn_quality).sd:.2f}")