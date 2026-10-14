# KAN vs MLP vs Frequency-Aware KAN under Noise and Distribution Shift

> Batch F Capstone Project | Deep Learning

| Field | Details |
|---|---|
| **Name** | Aayan Ali |
| **Registration Number** | 2430010369 |
| **Branch** | Data Science> |
| **Batch** | D |
| **Project Title** | KAN vs MLP vs Frequency-Aware KAN under Noise and Distribution Shift |
| **GitHub Username** | Aayan789 |
| **Training Program** | Btech Cse Data Science |
| **Team Capstone Repo** | `<LINK>` |

## 1. Overview
This project compares three parameter-matched (~5,000 parameters) neural models on five tabular datasets:

1. **MLP**: a 2-hidden-layer ReLU network (baseline).
2. **KAN**: a FastKAN-style Kolmogorov-Arnold Network (RBF basis expansion plus a SiLU base path).
3. **FreqKAN**: the KAN preceded by a learnable Fourier-feature layer, which gives it explicit frequency information.

Each model is tested under (a) clean data, (b) Gaussian noise added to test features at three levels, and (c) a covariate distribution shift. The work extends Zeng et al., *"KAN versus MLP on Irregular or Noisy Functions"* (arXiv 2408.07906) from synthetic functions to real tabular data.

## 2. Key Results (5 seeds)
Winner counts across 25 dataset-condition pairs: **MLP 15, KAN 8, FreqKAN 2**.

| Finding | Evidence |
|---|---|
| MLP is best on clean data and on small/medium classification | wins breast_cancer, wine, covertype (noise) |
| KAN is far more robust to test noise on California Housing | RMSE at noise 0.5: MLP 4.04 vs KAN 0.93 |
| FreqKAN gives the best result under shift on California Housing | 0.655 vs MLP 0.721 / KAN 0.712 |
| FreqKAN does **not** consistently beat plain KAN | see limitations in `docs/MODEL_DOCUMENTATION.md` |

Full table: [`capstone/results/results_table.csv`](capstone/results/results_table.csv). Plot: [`capstone/results/robustness.png`](capstone/results/robustness.png).

![Robustness](capstone/results/robustness.png)

## 3. Repository Structure
```
MUJ-DS-<RegNo>/
├── README.md
├── requirements.txt
├── assignments/            # weekly training assignments
├── notebooks/              # dlproject_1.ipynb (Colab notebook)
├── code/                   # kan_vs_mlp_robustness.py
├── resources/              # paper, references, datasets notes
├── presentations/          # final presentation + outline
├── capstone/               # results, screenshots, team-repo notes
│   ├── results/
│   └── screenshots/
├── docs/                   # installation, model docs, GitHub workflow
└── .github/                # pull-request template
```

## 4. Quick Start
```bash
git clone https://github.com/<username>/MUJ-DS-<RegNo>.git
cd MUJ-DS-<RegNo>
pip install -r requirements.txt
python code/kan_vs_mlp_robustness.py 2     # quick test: 2 seeds
python code/kan_vs_mlp_robustness.py       # full run: 5 seeds (~8 min on CPU)
```
Detailed steps: [`docs/INSTALLATION.md`](docs/INSTALLATION.md).

## 5. Documentation
- [Model documentation](docs/MODEL_DOCUMENTATION.md): architecture, methodology, results, limitations
- [Installation guide](docs/INSTALLATION.md)
- [GitHub workflow](docs/GITHUB_WORKFLOW.md): issues, branches, pull requests, contribution log
- [Presentation outline](presentations/presentation_outline.md)

## 6. Individual Contributions
See the contribution table in [`docs/GITHUB_WORKFLOW.md`](docs/GITHUB_WORKFLOW.md#5-individual-contribution-log).

## 7. Reference
Zeng et al., *KAN versus MLP on Irregular or Noisy Functions*, arXiv:2408.07906.
