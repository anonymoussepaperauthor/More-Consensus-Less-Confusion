# RQ0 — The Problem:

## Research Question

> How prevalent is performance instability in SE optimization, and what are its consequences for practitioners?

## Method

We train **20 decision trees** per dataset under two EZR configurations (initial default settings and the RQ2-refined settings) and pass **100 randomly selected test cases** through all 20 trees. For each test case, we record whether the 20 models agree — defined as:

```
σ_models < 0.35 × σ_data
```

where `σ_models` is the standard deviation of predictions across 20 models for a given test instance, and `σ_data` is the standard deviation of true d2h values across the full dataset.

Agreement is aggregated across all **127 datasets × 100 test cases = 12,700 total test cases**.

## Key Results

| Configuration | Agreement count (/ 12,700) | Agreement rate |
|---|---|---|
| Initial (default EZR) | 364 | **2.9%** |
| Refined (RQ2 settings) | 1,740 | **13.7%** |

Even at the most permissive threshold (σ = 1.0):

| Configuration | Agreement count | Agreement rate |
|---|---|---|
| Initial | 4,020 | 31.7% |
| Refined | 8,360 | 65.8% |
