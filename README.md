# Multiscale Causal Discovery for Drought Dynamics

An exploratory causal-discovery pipeline for identifying nonlinear, lag-dependent climate drivers of drought in **Illinois, USA**. The project replaces correlation-only lag screening with **Convergent Cross Mapping (CCM)**, then decomposes the selected signals into interpretable temporal scales with causal stationary wavelet transforms.

> **Research question:** Can nonlinear dynamical dependence reveal delayed climate signals that conventional cross-correlation misses or misplaces?

![Top multiscale features for SPEI at the 6-month scale](figures/top_multiscale_features_spei_6m.png)

## Project at a glance

| Dimension | Scope |
|---|---|
| Study area | Illinois, USA |
| Targets | SPI and SPEI |
| Accumulation windows | 1, 3, 6, and 12 months |
| Candidate drivers | 7 macroclimatic indices + 6 local hydroclimatic variables |
| Lag search | 0–12 months |
| Data span | Monthly observations, Jan 1949–Jun 2025 |
| Experimental scale | 8 target–window tasks; 1,352 predictor–lag evaluations |
| Compute | Python + `pyEDM`; Slurm job arrays for the UIUC Campus Cluster |

## Why this matters

Cross-correlation is useful for screening, but climate time series are seasonal, autocorrelated, nonlinear, and dynamically coupled. A high correlation can reflect shared persistence rather than a directional relationship, and its peak may be shifted away from a physically meaningful delay.

CCM approaches the problem through state-space reconstruction. If a candidate driver contributes information to drought dynamics, the reconstructed drought manifold should increasingly recover that driver as the library size grows. This project uses that convergence pattern to screen predictor–lag pairs and connect causal discovery with multiscale feature analysis.

```mermaid
flowchart LR
    A[Monthly climate and drought series] --> B[Temporal alignment and 70/30 split]
    B --> C[Simplex selection of embedding dimension E]
    C --> D[CCM scan across predictors and lags 0–12]
    D --> E[Convergence, skill and surrogate filters]
    E --> F[Causal SWT decomposition]
    F --> G[Spearman screening by A2, D2 and D1 scale]
    G --> H[Forecast-ready multiscale matrices]
```

## Selected findings

- Local precipitation produced the strongest CCM skill across several targets. For the 6-month indices, zero-lag 6-month precipitation reached exploratory cross-map skill of **0.719 for SPEI** and **0.733 for SPI**.
- Large-scale indices showed more structured delayed behavior. In the SPEI-6M scan, ENSO-related ONI and Niño 3.4 were strongest around 5–7 months, while PDO strengthened at longer lags.
- Multiscale screening separated smooth low-frequency structure (`A2`) from medium- and high-frequency variation (`D2`, `D1`). For SPEI-6M, the leading feature was `Precipitation_1M_lag0_A2`, followed by several lagged precipitation detail components.
- The workflow was parallelized into eight Slurm array tasks covering SPI/SPEI × 1/3/6/12 months.

These findings are **exploratory dynamical-dependence results**, not proof of a complete physical causal mechanism. Pairwise CCM can still be affected by shared seasonality, indirect coupling, finite samples, and nonstationarity.

## Repository structure

```text
.
├── src/                    # CCM, selection, EDA, and wavelet utilities
├── scripts/                # SPI/SPEI scans, multiscale extraction, demo data
├── cluster/                # Slurm array jobs and environment setup
├── metadata/               # Schemas for all four accumulation windows
├── results/                # Compact exploratory result snapshots
├── figures/                # Portfolio-ready figures
├── data/README.md          # Data contract and provenance checklist
└── docs/methodology.md     # Assumptions, decisions, and limitations
```

## Reproduce the pipeline

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Use a dataset matching [the documented schema](data/README.md), or create synthetic data for a smoke test:

```bash
python scripts/generate_demo_data.py
python scripts/run_spei_ccm.py --aggregation 6m --n-surrogates 100
python scripts/run_multiscale.py --target spei --aggregation 6m
```

Run all eight scans on Slurm:

```bash
sbatch --account=<account> --partition=<partition> cluster/run_ccm_array.sbatch
sbatch --account=<account> --partition=<partition> cluster/run_multiscale_array.sbatch
```

## Statistical safeguards

The public version makes two corrections to the exploratory research snapshot:

1. A predictor–lag pair must satisfy the combined `passed` criterion (significance, minimum skill, and positive convergence slope) when that field is available; `significant` alone is not sufficient.
2. Surrogate p-values use an add-one correction, and the command-line default is 100 surrogate replicates instead of the five used during early prototyping.

The CSV files under `results/` preserve the earlier exploratory run for traceability. They should be regenerated with the stricter defaults before confirmatory scientific use.

## Tech stack

Python · pandas · NumPy · SciPy · pyEDM · PyWavelets · scikit-learn · statsmodels · Matplotlib · Seaborn · Slurm

## Author

**Ruichen Tang** — project design, CCM implementation, multiscale feature pipeline, cluster workflow, analysis, and visualization.

## Acknowledgments

Research conducted under the mentorship of **Prof. Lois Bravo de Guenni**. The project extends the drought-forecasting framework introduced by Vivas, Ji, and Bravo de Guenni by replacing correlation-based lag screening with a nonlinear dynamical-systems approach.
