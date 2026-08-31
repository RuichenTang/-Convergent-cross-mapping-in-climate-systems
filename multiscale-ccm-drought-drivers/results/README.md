# Results snapshot

This directory contains compact outputs from the original exploratory experiment:

- `ccm_scans/`: macroclimatic and local lag scans plus selected lag tables;
- `selected_features/`: final multiscale feature names for all eight tasks.

The snapshot evaluated 13 predictors across lags 0–12 for each of eight SPI/SPEI target-window combinations (1,352 scan rows in total). It used five seasonal surrogate replicates during prototyping, so p-values are too coarse for confirmatory inference.

The public code now defaults to 100 surrogates, applies an add-one p-value correction, and prioritizes the combined `passed` criterion. Regenerate these files before using them for scientific claims.

