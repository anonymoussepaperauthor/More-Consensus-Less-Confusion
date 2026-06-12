# More Consensus, Less Confusion: Cheap Fixes for Unstable Recommendation in SE Analytics

> **Anonymous submission for ICSE 2027** — Author information withheld for double-blind review.

---

## Overview

Software engineering optimizers increasingly guide practical decisions such as configuration tuning, test prioritization, and project planning. Yet repeated runs of the same optimizer can produce **different recommendations**, leaving practitioners unsure which advice to trust.

This repository contains all code, data, and results for a large-scale empirical study of **recommendation instability** across **127 multi-objective SE optimization problems (12,700 test cases)**. We:

- Distinguish **structural instability** (changes in the learned model) from **performance instability** (changes in the outcomes of recommended solutions)
- Show that performance instability is the norm — under default settings, repeated runs agree on only **1.5%** of test cases
- Demonstrate that structural instability is unavoidable due to the **Rashomon Effect**
- Identify interventions that improve performance stability by around 36% on average on all datasets.
- Show that a data-inherent floor on stability exists, even for methods designed to be stable

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── LICENSE
│
├── data/
│   ├── optimize/                        # 127 MOOT benchmark datasets
│   ├── behaivior_data/                  
│   ├── binary config/                   
│   ├── config/                          
│   ├── financial_data/                  
│   ├── health_data/                     
│   ├── hpo/                             
│   ├── misc/                            
│   ├── process/                         
│   ├── rl/                              
│   ├── sales_data/                      
│   ├── systems/                         
│   ├── test/                            
│   └── dataset_summary.csv              # Some information about data
│
├── tools/
│   ├── ezr.py                           # Core module of experiments: EZR
│   ├── stats.py                         # Statistical tests used in experiments
│   ├── causal_tools.py                  # Causal interventions investigated in RQ3
│   └── aggregate_results.py             # Summarize raw results for each experiments
│
├── docs/figures/
│   ├── Background/                      # Figures in Background section of manuscript
│   ├── methods/                         # Figures in Method section of manuscript
│   └── results/                         # Figures in Results section of manuscript
│
├── RQ0/                                 # All codes and results related to RQ0 experiments
│   ├── README.md
│   ├── code/
│   │   └── sensitivity_analysis.py
│   └── results/
│
├── RQ1/                                 # All codes and results related to RQ1 experiments
│   ├── README.md
│   ├── code/
│   │   └── jaccard.py
│   └── results/
│
├── RQ2/                                 # All codes and results related to RQ2 experiments
│   ├── README.md
│   ├── code/
│   │   ├── all-together.py              # Wrap up experiment (Fig. 8 in manuscript)
│   │   ├── signal-strength.py           # Signal Strength experiment  (Fig. 7, Top-Left)
│   │   ├── sampling-strategy.py         # Sampling Strategy experiment(Fig. 7, Top-Right)
│   │   ├── model-complexity.py          # Model Complexity experiment (Fig. 7, Bottom-Left)
│   │   ├── math-instability.py          # Math instability experiment (Fig. 7, Bottom-Right)
│   │   └── plot_results.py              # For generating Fig. 8
│   └── results/
│       ├── all-together/       
│       ├── signal-strength/         
│       ├── sampling-strategy/        
│       ├── model-complexity/           
│       └── math-instability/            
│
├── RQ3/                                 # All codes and results related to RQ2 experiments
│   ├── README.md
│   ├── code/
│   │   ├── causal.py                    # Confounder filtering + causal split criterion
│   │   └── clusterings.py               # HDBSCAN, CURE, KMeans baselines
│   └── results/
│       ├── causal/          
│       └── clusterings/  
│
├── runners/                             # Reproduction Package containing all runner files
│   ├── rq0.sh
│   ├── rq2-all-together.sh
│   ├── rq2-signal-strength.sh
│   ├── rq2-sampling-strategy.sh
│   ├── rq2-math-instability.sh
│   ├── rq2-model-complexity.sh
│   ├── rq3-causal.sh
│   └── rq3-clusterings.sh
│
└── [additional files described below]
```

---

## Datasets

All experiments use the **[MOOT benchmark](https://arxiv.org/abs/2511.16882)** — 127 multi-objective SE optimization tasks spanning:

| Category | # Datasets | Examples |
|---|---|---|
| Specific Software Configurations | 25 | SS-A to SS-X, billing10k |
| PromiseTune Software Configurations | 12 | 7z, BDBC, PostgreSQL, x264 |
| Software Project Health | 35 | Health-ClosedIssues, Health-PRs |
| Software Process Models | 12 | COC1000, POM3 A–D, XOMO |
| Feature Models | 8 | FFM-*, FM-* |
| Scrum | 3 | Scrum1k, Scrum10k, Scrum100k |
| Financial | 4 | BankChurners, Loan, Telco-Churn |
| Behavioral | 4 | HR-employeeAttrition, student dropout |
| Cloud Configurations | 10 | Apache, SQL, X264, HSMGP |
| Reinforcement Learning | 2 | A2C_Acrobot, A2C_CartPole |
| Miscellaneous | 12 | auto93, Wine quality, Sales |

Datasets range from **82 to 166,975 rows** and **3 to 1,044 decision variables**, with **1 to 8 optimization objectives**.

---

## Research Questions & Results

### RQ0 — The Problem: How prevalent is performance instability?

> Under default EZR settings, 20 repeated runs agree on only **189 / 12,700 test cases (1.49%)**. With refined settings, this improves to **1,248 / 12,700 (9.83%)**. Even at the most permissive agreement threshold (σ = 1.0), agreement reaches only 24.3% and 61.2% respectively.

📁 Code: [`RQ0/code/`](RQ0/code/)  
📊 Results: [`RQ0/results/`](RQ0/results/)  
🔁 [How to run the code?](#reproducing-results)

---

### RQ1 — The Nature: Are structural and performance instability the same phenomenon?

> **No.** Structural instability (measured via weighted Jaccard similarity between feature sets) remains high even under refined settings that substantially improve performance stability. Trees can be structurally diverse yet yield consistent optimization recommendations. The **Rashomon Effect** explains why: many structurally different models perform nearly equally well, making structural consistency an unachievable and unnecessary goal.

📁 Code: [`RQ1/code/`](RQ1/code/)  
📊 Results: [`RQ1/results/`](RQ1/results/)  
🔁 [How to run the code?](#reproducing-results)

---

### RQ2 — The Fixes: What configuration changes reduce performance instability?

> Four factors were systematically investigated. The recommended configuration (below) improves stability by 36% on average and performance by 6% on average across all 127 datasets.

| Factor | Default | **Recommended** | Effect |
|---|---|---|---|
| Labeling budget | 20 | **50** | Best Balance for performance, stability, and labeling cost |
| Acquisition strategy | `xploit` | **`near`** | Large stability gain, little performance gain |
| Min leaf size | 2 | **3** | Best balance of performance, stability, readability |
| Splitting criterion | `entropy` | **`gini`** | Stability gain, zero performance cost |

📁 Code: [`RQ2/code/`](RQ2/code/)  
📊 Results: [`RQ2/results/`](RQ2/results/)  
🔁 [How to run the code?](#reproducing-results)

---

### RQ3 — The Limits: How reducible is performance instability?

> Performance instability has a **data-inherent floor**. Even clustering methods designed to be more stable than decision trees reach at most **~51% agreement** across the benchmark. Causal augmentation (confounder filtering + causal split criterion) matches EZR on optimization performance and achieves more total agreements (2,108 vs. 1,668), but with a narrower dataset-level advantage. The fix, if one exists, lies upstream: in **data collection, labeling quality, and measurement practices**.

📁 Code: [`RQ3/code/`](RQ3/code/)  
📊 Results: [`RQ3/results/`](RQ3/results/)  
🔁 [How to run the code?](#reproducing-results)

---

## Requirements

- **Python** 3.10+
- **OS**: Linux / macOS (tested on macOS 26.05)
- Core dependencies: see `requirements.txt`

Key libraries:

| Library | Version | Purpose |
|---|---|---|
| `hdbscan` | ≥ 0.8 | HDBSCAN clustering baseline |
| `numpy` | ≥ 1.24 | Numerical for Figures |
| `pandas` | ≥ 2.0 | Data Handling for Figures |
| `matplotlib` | ≥ 3.7 | Figures |

---

## Installation

```bash
# 1. Clone this repository
git clone https://github.com/[ANONYMOUS]/stability-evaluation.git
cd stability-evaluation

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # Linux/macOS
# venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Reproducing Results

### Run individual RQs

```bash
# RQ0: Measure instability prevalence (sensitivity analysis)
bash runners/rq0.sh

# RQ1: Measure Structural instability
python RQ1/code/jaccard.py

# RQ2: Four-factor experiments (runs all four sub-experiments)
bash runners/rq2-signal-strength.sh
bash runners/rq2-sampling-strategy.sh
bash runners/rq2-model-complexity.sh
bash runners/rq2-math-instability.sh
bash runners/rq2-all-together.sh

# RQ3: Limits — causal pipeline and clustering baselines
bash runners/rq3-causal.sh
bash runners/rq3-clusterings.sh
```

### Expected runtime

| Experiment | Approx. time |
|---|---|
| RQ0 | ~16 min |
| RQ1 | ~33 min |
| RQ2 - Signal Strength | ~47 mins |
| RQ2 - Sampling Strategy | ~80 mins |
| RQ2 - Model Complexity | ~50 mins |
| RQ2 - Math Instability | ~24 mins |
| RQ2 - all together | ~13 min |
| RQ3 - Causal | ~25 mins |
| RQ3 - Clustering | ~127 mins |

---

## Key Findings Summary

1. **Performance instability is pervasive.** Default SE optimizers agree on well under 10% of test cases across repeated runs.
2. **Structural and performance instability are decoupled.** Chasing structural consistency is the wrong goal — consistent recommendations matter, not consistent trees.
3. **Acquisition strategy is the highest-leverage, zero-cost fix.** Switching to `near` selection dramatically improves stability with no performance trade-off.
4. **Gini beats Entropy for free.** Replacing the entropy splitting criterion with Gini yields stability gains with no performance cost.
5. **A data-inherent stability floor exists.** Even clustering methods designed to be stable peak below 51% agreement — some instability reflects properties of the data, not the learner.
6. **Causal augmentation helps when confounders are the cause.** It is a targeted intervention, not a universal fix.

---

## License

This repository is released under the [MIT License](LICENSE).

---

## Citation
```bibtex
To be Completed ...
```

---

*For questions or issues with reproducibility, please open a GitHub Issue.*
