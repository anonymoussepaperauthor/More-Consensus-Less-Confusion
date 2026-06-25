# RQ2 — The Fixes:

## Research Question

> Which factors drive performance instability, and what configuration changes reduce it without sacrificing optimization quality?

## Factors Investigated

We conduct four independent single-factor experiments (varying one factor at a time):

| # | Factor | Treatments |
|---|---|---|
| 1 | **Signal Strength** (labeling budget) | 10, 20, 50, 100, 200 labeled configurations |
| 2 | **Sampling Strategy** (acquisition function) | `xploit`, `xplor`, `adapt`, `bore`, `near`, `random` |
| 3 | **Model Complexity** (min leaf size) | 1, 3, 5, 7, 9 |
| 4 | **Mathematical Instability** (splitting criterion) | `entropy` (default), `gini` |

## Recommended Configuration

| Factor | Default | **Recommended** | Key Finding |
|---|---|---|---|
| Labeling budget | 20 | **50** | Only factor that improves both stability and performance simultaneously |
| Acquisition strategy | `xploit` | **`near`** | Strong stability improvement, negligible performance impact — **zero-cost fix** |
| Min leaf size | 2 | **3** | Best balance of performance, stability, and interpretability |
| Splitting criterion | `entropy` | **`gini`** | Clear stability gain, zero performance cost — **unconditional recommendation** |

## Experiment Summaries

### 1. Signal Strength (Labeling Budget)

As budget grows from 10 → 200, performance climbs from ~24% to ~100% of datasets won and stability rises from under 10% to ~80%. Marginal gains beyond **50 labels** are modest relative to earlier increments. Considering the limitation of labeling cost, budget of **50** adopted as recommended.

### 2. Sampling Strategy (Acquisition Function)

Acquisition function has **limited impact on performance** but a **strong influence on stability**. `near` (selects candidates closest to the current best centroid) achieves markedly higher agreement than alternatives while yielding little performance change. **Switching to `near` is a cost-free reliability improvement** — requires no additional labeling, no changes to the objective, and no loss in quality.

### 3. Model Complexity (Min Leaf Size)

Reveals a three-way trade-off: deeper trees (smaller min leaf) improve identification of high-quality configurations but reduce stability and interpretability. Very deep trees also become difficult for practitioners to read and act upon, undermining the core value of decision-tree-based optimization. **Min leaf = 3** achieves the best balance across all three criteria.

### 4. Mathematical Instability (Splitting Criterion)

Entropy relies on `log(p)`, which amplifies small perturbations in probability estimates under tight labeling budgets. **Replacing entropy with Gini impurity yields no performance change but a clear stability improvement.** This is a zero-cost configuration change applicable immediately.
