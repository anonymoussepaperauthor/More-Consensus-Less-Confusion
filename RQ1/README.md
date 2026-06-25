# RQ1 — The Nature:

## Research Question

> How prevalent is structural instability, and is it the same as performance instability?

## Method

We train **20 trees per dataset** under both the initial and refined EZR configurations and compute the **weighted Jaccard similarity** between the feature sets used by each pair of trees:


$$
J_w(A,B)=
\frac{\sum_{f\in F}\min\left(W_A(f),W_B(f)\right)}
{\sum_{f\in F}\max\left(W_A(f),W_B(f)\right)},
$$

where

$$
W_T(f)=\sum_{k\in S_f}\frac{1}{d_k}.
$$


Here:
- $F$ is the union of all features appearing in either tree.
- $W_T(f)$ is the weight assigned to feature $f$ in tree $T$.
- $S_f$ is the set of nodes where feature $f$ appears.
- $d_k$ is the depth of node $k$, so features appearing closer to the root receive higher weights.
We compare weighted Jaccard scores across the 127 datasets under the initial and refined EZR configurations. Since the refined configuration substantially improves performance stability, we ask whether it also improves structural similarity.

Performance stability is evaluated separately using the agreement metric described in RQ0, allowing us to compare changes in prediction agreement against changes in tree structure.

## Key Finding

**No.** The refined configuration does not produce meaningfully higher structural similarity. The two curves (initial vs. refined) remain closely aligned across all 127 datasets, meaning:

- The gains in recommendation consistency from RQ2 are **not** explained by trees becoming structurally more alike.
- Structural instability is largely **irreducible**, reflecting the Rashomon Effect: many structurally different models achieve nearly identical performance.
- **Structural and performance instability are distinct phenomena.** Reducing one does not reduce the other.

### Practical Implications

1. **Chasing structural consistency is the wrong goal.** What practitioners should care about is whether different runs lead to the same optimization decision, not whether they produce the same tree.
2. **Any single tree structure should be treated cautiously** as an interpretive artifact. Features appearing in one tree may not reliably identify genuine configuration drivers, since different runs can select different features even when optimization recommendations are consistent.
