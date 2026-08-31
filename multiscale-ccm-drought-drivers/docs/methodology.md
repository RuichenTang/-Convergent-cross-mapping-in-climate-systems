# Methodology and research notes

## 1. Problem formulation

The target series are the Standardized Precipitation Index (SPI) and Standardized Precipitation Evapotranspiration Index (SPEI) at 1-, 3-, 6-, and 12-month accumulation windows. Candidate drivers include macroclimatic indices (`NAO`, `PDO`, `PNA`, `NINO34`, `ONI`, `SOLAR`, `WP`) and local precipitation and temperature aggregates.

The task is feature discovery rather than final forecasting: select predictor–lag pairs that contain dynamically relevant information about a drought target, then identify which temporal-scale components are most informative.

## 2. Temporal design

- Monthly observations are sorted chronologically.
- Missing rows are removed after the required fields are selected.
- The first 70% of observations form the training period; the final 30% are held out.
- Lag and multiscale selection are learned from the training data only.

## 3. CCM screening

For each target, simplex forecasting searches embedding dimensions `E = 2, …, 11`. Each candidate predictor is shifted by 0–12 months and evaluated with CCM. The pipeline records:

- final cross-map skill (`rho`);
- convergence slope and improvement as library size increases;
- a seasonal-surrogate p-value; and
- a combined pass/fail decision.

The public implementation uses an add-one surrogate p-value estimate:

`p = (count(rho_surrogate >= rho_observed) + 1) / (n_surrogates + 1)`.

## 4. Multiscale feature extraction

Selected lagged predictors are decomposed with a causal stationary wavelet transform. Level 2 produces:

- `A2`: smooth, low-frequency structure;
- `D2`: medium-scale variation; and
- `D1`: high-frequency variation.

Candidate wavelet/component pairs are ranked by absolute Spearman correlation against the corresponding target component. A default threshold of `|rho_s| >= 0.15` creates forecast-ready component datasets.

## 5. Interpretation boundary

CCM detects evidence of dynamical dependence under state-space reconstruction assumptions. It does not, by itself, establish a complete physical mechanism. Results may be sensitive to embedding dimension, delay, library size, surrogate design, seasonality, finite samples, nonstationarity, shared drivers, and indirect paths. Multivariate or partial-CCM extensions and out-of-sample forecast comparisons are appropriate next steps.

