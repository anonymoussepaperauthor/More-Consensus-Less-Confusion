# RQ1 — The Nature: Are Structural and Performance Instability the Same Phenomenon?

## Research Question

> How prevalent is structural instability, and do structural and performance instability represent the same underlying phenomenon?

## Method

We train **20 trees per dataset** under both the initial and refined EZR configurations and compute the **weighted Jaccard similarity** between the feature sets used by each pair of trees:

```
J(A, B) = |A ∩ B| / |A ∪ B|
```

where A and B are the sets of features used in any split of two trees trained under identical settings. A score of 0 means the trees share no features; 1 means they use exactly the same features.

We compare the distribution of Jaccard scores across all 127 datasets under initial vs. refined settings and ask: does the refined configuration (which substantially improves *performance* stability) also improve *structural* similarity?

## Key Finding

**No.** The refined configuration does not produce meaningfully higher structural similarity. The two curves (initial vs. refined) remain closely aligned across all 127 datasets, meaning:

- The gains in recommendation consistency from RQ2 are **not** explained by trees becoming structurally more alike.
- Structural instability is largely **irreducible** — a direct consequence of the Rashomon Effect, where many structurally different models perform nearly equivalently.
- **Structural and performance instability are distinct phenomena.** Reducing one does not reduce the other.

### Practical Implications

1. **Chasing structural consistency is the wrong goal.** What practitioners should care about is whether different runs lead to the same optimization decision, not whether they produce the same tree.
2. **Any single tree structure should be treated cautiously** as an interpretive artifact. Features appearing in one tree may not reliably identify genuine configuration drivers, since different runs can select different features even when optimization recommendations are consistent.
