# Presentation Outline (about 10 to 12 slides, ~10 min + demo)

| # | Slide | Content |
|---|---|---|
| 1 | Title | Project title, name, Reg No, Batch F, team, GitHub links |
| 2 | Motivation | MLPs vs KANs; why robustness to noise and shift matters on real tabular data |
| 3 | Research question | Are KANs more robust than MLPs? Do Fourier features help? Extension of Zeng et al. (arXiv 2408.07906) |
| 4 | Models | MLP, KAN (RBF + SiLU), FreqKAN (Fourier front end); diagram of each; ~5k parameters matched |
| 5 | Datasets | table: 5 datasets, task, size, features |
| 6 | Experimental design | random vs PC1 shift split, test-noise 0.1/0.25/0.5, 5 seeds, metrics |
| 7 | Results: clean and noise | `robustness.png`; callout: California Housing MLP 4.04 vs KAN 0.93 at noise 0.5 |
| 8 | Results: distribution shift | grouped bars; FreqKAN best on California; KAN instability on breast_cancer |
| 9 | Scoreboard | MLP 15 / KAN 8 / FreqKAN 2 wins; honest takeaway: no universal winner |
| 10 | Limitations and future work | width/depth confound, test-only noise, ablations, tuning, significance tests |
| 11 | Team and GitHub process | issues, branches, PRs, contribution table, screenshots of Insights |
| 12 | Live demo and Q&A | run `python code/kan_vs_mlp_robustness.py 2`, show outputs |

## Speaker tips
- Lead with the finding the data supports: *KAN shines in noise robustness on California Housing; MLP wins most other cells; FreqKAN is not a consistent improvement.*
- Be ready for: "Why is the KAN so narrow?" (parameter matching) and "Why does the MLP explode on California under noise?" (unbounded extrapolation; hypothesis).
- Put repo screenshots (commit graph, issues, merged PRs) in `capstone/screenshots/`.
