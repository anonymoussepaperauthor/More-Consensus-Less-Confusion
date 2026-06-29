# RQ3 — The Limits:

## Research Question

> Is the instability remaining after RQ2 the fault of the learner, or of the data?

## Two Sub-Experiments

### 3a: Causal Augmentation

We augment EZR's tree construction with two components:

1. **Confounder Filtering** — Removes features that appear predictive of d2h only due to confounding (i.e., they co-vary with a genuine driver rather than directly influencing it). Uses Pearl's framework to distinguish direct causation from confounding.

2. **Causal Split Selection** — After filtering, builds the tree using a causal split criterion based on information-theoretic causal inference (gain ratio), identifying the gain ratio as the most stable splitting criterion.

### 3b: Clustering Baselines

We compare EZR against three clustering methods generally considered more stable than decision trees: **HDBSCAN**, **CURE**, and **KMeans**. If instability is purely learner-driven, EZR should fall far behind these stability anchors.

## Key Results

### Causal Pipeline vs. EZR (Refined)

| Metric | EZR (refined) | Causal Pipeline |
|---|---|---|
| Performance wins (datasets) | **124** | 122 |
| Stability wins (datasets) | 59 | **69** |
| Total agreements (/ 12,700) | 1,740 | **2,390** |

Causal integration **does not harm performance** (statistically indistinguishable). Stability gains are **conditional**, not universal — most useful when instability is driven by confounded feature choices. When instability is driven by label noise, limited samples, or large Rashomon sets, causal filtering has fewer opportunities to help.

### EZR vs. Clustering Baselines

**Even HDBSCAN — the most stable-by-design method — achieves only ~51% agreement.** This provides evidence of a data-inherent floor on stability. Switching tools or algorithms alone is unlikely to resolve the instability; the fix lies upstream in data collection, labeling quality, and measurement practices.

## Practical Guidance

- **Causal augmentation** is best applied as a targeted intervention in domains where organizational or process confounders distort feature-outcome relationships (e.g., defect prediction with team-level confounders).
- **When instability mainly reflects data scarcity or measurement noise**, no learning method — causal or otherwise — can fully compensate.
- **Report agreement rates alongside optimization results.** Teams should treat instability as a measurable property of the modeling setting and use agreement rates to calibrate confidence in any single run.
