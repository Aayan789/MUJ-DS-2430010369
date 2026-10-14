# Installation Guide

## Requirements
- Python 3.9 or newer
- ~2 GB free disk space (datasets are downloaded and cached by scikit-learn)
- Internet access on first run (California Housing, Covertype and Adult are downloaded)
- GPU optional. The script uses CUDA automatically if available. The full run took **~491 s on CPU**.

## Option A: Google Colab (easiest)
1. Open `notebooks/dlproject_1.ipynb` in Colab (File > Upload notebook).
2. Run the cell for the full 5-seed experiment. For a quick test, save the code as `kan_vs_mlp_robustness.py` and run `!python kan_vs_mlp_robustness.py 2`.
3. Download `results_raw.csv`, `results_table.csv` and `robustness.png` from the Colab file panel.

## Option B: Local machine
```bash
git clone https://github.com/<username>/MUJ-DS-<RegNo>.git
cd MUJ-DS-<RegNo>

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Run
```bash
python code/kan_vs_mlp_robustness.py 2    # 2 seeds, quick test
python code/kan_vs_mlp_robustness.py      # 5 seeds, full experiment
```

### Outputs (written to the current directory)
| File | Content |
|---|---|
| `results_raw.csv` | one row per dataset / condition / model / seed |
| `results_table.csv` | mean ± std table with the winner per row |
| `robustness.png` | error vs noise plots per dataset |

## Configuration (top of `code/kan_vs_mlp_robustness.py`)
| Variable | Default | Meaning |
|---|---|---|
| `SEEDS` | 5 (or first CLI argument) | number of random seeds |
| `PARAM_TARGET` | 5000 | parameter budget used to match model widths |
| `NOISE_LEVELS` | 0.1, 0.25, 0.5 | test-noise std as a fraction of feature std |
| `EPOCHS` | per dataset | training epochs |
| `BIG_N` | 20000 | subsample size for Covertype and Adult |
| `BATCH`, `LR` | 256, 3e-3 | optimiser settings |

## Troubleshooting
- **`fetch_openml` / download errors:** the script skips a dataset it cannot load (and falls back to the Diabetes dataset if California Housing fails). Re-run with internet access. Note that a fallback changes the results.
- **Out of memory on GPU:** reduce `BIG_N` or `BATCH`.
- **Slow run:** use 2 seeds, or run on Colab with a GPU runtime.
