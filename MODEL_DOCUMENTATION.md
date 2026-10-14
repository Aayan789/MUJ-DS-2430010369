# Model Documentation

## 1. Problem Statement
Kolmogorov-Arnold Networks (KANs) place learnable univariate functions on edges instead of fixed activations on nodes. Zeng et al. (arXiv 2408.07906) studied KAN vs MLP on irregular or noisy *synthetic* functions. This project asks: **on real tabular data, are KANs more robust than MLPs to input noise and distribution shift, and does adding explicit Fourier (frequency) features help?**

## 2. Models
All models are matched to ~5,000 trainable parameters (`PARAM_TARGET`). Hidden width is chosen per dataset by `matched_width()`, which searches widths 2 to 299 for the closest parameter count.

| Model | Architecture | Notes |
|---|---|---|
| **MLP** | `Linear(d,h) > ReLU > Linear(h,h) > ReLU > Linear(h,out)` | 2 hidden layers |
| **KAN** | `KANLayer(d,h) > KANLayer(h,out)` | 1 hidden layer |
| **FreqKAN** | `FourierFeatures(d,8) > KANLayer(d+16,h) > KANLayer(h,out)` | KAN with learnable Fourier front end |

### KANLayer (FastKAN-style)
```
x   = LayerNorm(x)
phi = exp(-((x - c_k) / 0.5)^2),  c = linspace(-2, 2, 8)     # 8 fixed RBF centres per input
out = Linear(SiLU(x)) + Linear_nobias(flatten(phi))           # base path + spline-like path
```

### FourierFeatures
`z = x @ B` with learnable `B` (d x 8, initialised N(0,1)); output is `[x, sin(z), cos(z)]`, which adds 16 features.

## 3. Datasets
| Dataset | Task | Samples | Features | Source |
|---|---|---|---|---|
| breast_cancer | classification | 569 | 30 | `sklearn.datasets` |
| wine | classification (3 classes) | 178 | 13 | `sklearn.datasets` |
| california_housing | regression | 20,640 | 8 | `fetch_california_housing` |
| covertype | classification (7 classes) | 20,000 (subsample) | 54 | `fetch_covtype`, stratified subsample |
| adult | classification | 20,000 (subsample) | 105 (one-hot) | OpenML `adult` v2, rows with NaN dropped |

## 4. Experimental Protocol
**Splits**
- *Random split:* stratified (classification) 80/20 train/test.
- *Distribution shift:* standardise, project onto PC1; train on the lowest 70% of PC1 values and test on the highest 30%.
- In both cases 10% of the training portion is held out for validation. The scaler is fitted on training data only; regression targets are standardised using training statistics.

**Training:** AdamW (lr 3e-3, weight decay 1e-4), batch size 256, epochs 30 to 120 depending on dataset. The checkpoint with the lowest validation loss is restored.

**Evaluation conditions**
1. `clean`: random-split test set.
2. `noise 0.1 / 0.25 / 0.5`: Gaussian noise with std = level x feature std (features are standardised) added to **test features only** of the random split.
3. `dist. shift`: the PC1 extrapolation test set.

**Metric (lower is better):** classification = 1 - accuracy; regression = RMSE on the standardised target. Results are mean ± std over 5 seeds.

## 5. Results
Parameters actually used: MLP 5,002, KAN 5,027, FreqKAN 5,082.

Winner counts over the 25 dataset-condition pairs: **MLP 15, KAN 8, FreqKAN 2** (see `capstone/results/results_table.csv` for every number).

### Selected results
| Dataset | Condition | MLP | KAN | FreqKAN |
|---|---|---|---|---|
| breast_cancer | clean | **0.0316** | 0.0614 | 0.0456 |
| breast_cancer | dist. shift | **0.0690** | 0.3789 ± 0.2935 | 0.1088 |
| california_housing | clean | **0.4546** | 0.5377 | 0.4730 |
| california_housing | noise 0.5 | 4.0381 ± 1.1324 | **0.9345** | 1.0479 |
| california_housing | dist. shift | 0.7214 | 0.7120 | **0.6552** |
| adult | noise 0.25 | 0.1632 | **0.1591** | 0.1604 |
| covertype | noise 0.5 | **0.3631** | 0.4188 | 0.4211 |

### Interpretation
- **MLP is the strongest default** on the small classification sets (breast_cancer, wine) and on covertype, in clean and noisy conditions.
- **KAN's clearest advantage is noise robustness on California Housing**, where the MLP's error grows from 0.45 to 4.04 while KAN's grows from 0.54 to 0.93. KAN also wins most adult conditions, though by margins (about 0.003 to 0.004) that are within one standard deviation.
- **FreqKAN** wins only 2 of 25 cells (adult noise 0.5 and California shift). It sits between MLP and KAN on several tasks but does not consistently improve on plain KAN, so the hypothesis that explicit Fourier features improve robustness is only weakly supported.
- **Instability:** KAN on breast_cancer under shift has a very large spread (0.379 ± 0.294), indicating some seeds fail badly.

## 6. Limitations and Threats to Validity
1. **Parameter matching forces very narrow KANs.** KAN hidden widths are 5 to 59 and FreqKAN widths 4 to 21 versus 35 to 65 for the MLP, because each KAN weight set scales with `d_in x 8`. This confounds architecture with width and depth (the MLP has two hidden layers, the KANs one).
2. **Noise is applied at test time only.** The models are not trained with noise, so this measures sensitivity to input perturbation, not noise-aware learning.
3. **Noise scale vs. heavy-tailed features.** MLP blow-up on California Housing may be driven by unbounded linear extrapolation; KAN's LayerNorm and bounded RBF basis would damp this. This explanation is plausible but untested here.
4. **Shift definition.** The PC1 split is one particular covariate shift; the shift split also trains on a different subset than the random split, so clean vs. shift numbers are not strictly paired.
5. **Single hyperparameter setting.** No per-model tuning of learning rate, grid size or epochs; the Fourier matrix `B` uses an untuned N(0,1) initialisation.
6. **Subsampled datasets** (20k rows) and a fixed number of seeds (5) limit statistical power; no significance tests are reported.
7. **Fallback behaviour:** if California Housing fails to download the script silently switches to the Diabetes dataset, which would change results.

## 7. Suggested Future Work
- Ablate hidden width and depth to separate architecture from capacity.
- Train with noise augmentation and compare.
- Tune the Fourier initialisation scale and number of frequencies.
- Add paired significance tests (e.g. Wilcoxon over seeds) and more shift types.
- Add gradient-boosted tree baselines for context on tabular data.

## 8. Reproducibility
Run `python code/kan_vs_mlp_robustness.py` (5 seeds). Seeds are controlled with `torch.manual_seed(seed)` per model and a seeded generator for the noise. Results here came from a CPU run of ~491 s.
