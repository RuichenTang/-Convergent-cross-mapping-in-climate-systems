import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from eda import *
from ccm import *


configure_pandas_display()
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

RESULTS_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CCM workflow on ICC.")
    parser.add_argument(
        "--aggregation",
        default="6m",
        choices=["1m", "3m", "6m", "12m"],
        help="Temporal aggregation.",
    )
    parser.add_argument(
        "--n-surrogates",
        type=int,
        default=100,
        help="Seasonal surrogate replicates used for exploratory significance testing.",
    )
    return parser.parse_args()


def resolve_metadata_path(aggregation: str) -> Path:
    metadata_dir = PROJECT_DIR / "metadata"
    candidates = [
        metadata_dir / f"metadata{aggregation}.JSON",
        metadata_dir / f"metadata{aggregation}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find metadata file for aggregation {aggregation}. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def save_current_plot(filename: str) -> None:
    out = RESULTS_DIR / filename
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out}")


args = parse_args()
AGGREGATION = args.aggregation
print(f"Loading data for aggregation {AGGREGATION}...")

path_metadata = resolve_metadata_path(AGGREGATION)
metadata = read_metadata(str(path_metadata))

time_col = str(metadata["time"][0])
feature_cols = metadata["features"]
cols_to_load = [time_col] + feature_cols

data_file_path = PROJECT_DIR / metadata["data_path"] / metadata["data_file"]
print(f"Loading dataset from: {data_file_path}")

df = pd.read_excel(data_file_path, usecols=cols_to_load)

print(f"\nDataset dimensions: {df.shape}")
print(f"Date range: {df[time_col].min()} to {df[time_col].max()}")

time_span = df[time_col].max() - df[time_col].min()
years = time_span.days / 365.25
print(f"Total time period: {years:.1f} years")

print("Preprocessing temporal variables")
df[time_col] = pd.to_datetime(df[time_col])
df["Year"] = df[time_col].dt.year
df["Month"] = df[time_col].dt.strftime("%b")
df["Month_num"] = df[time_col].dt.month
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
df = df.sort_values(time_col).reset_index(drop=True)
print("Data preprocessing completed")
print("Dataset Overview:")
print("=" * 50)
df.info()

# Train / Test split
df = df.sort_values(time_col).reset_index(drop=True)
train_ratio = 0.70
split_idx = int(len(df) * train_ratio)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()
print(f"Train Shape: {train_df.shape}")
print(f"Test Shape: {test_df.shape}")
print(f"Train period: {train_df[time_col].min().date()} -> {train_df[time_col].max().date()}")
print(f"Test period: {test_df[time_col].min().date()} -> {test_df[time_col].max().date()}")

test_csv = RESULTS_DIR / f"test_df_{AGGREGATION}.csv"
train_csv = RESULTS_DIR / f"train_df_{AGGREGATION}.csv"
test_df.to_csv(test_csv, index=False)
train_df.to_csv(train_csv, index=False)
print(f"Saved: {test_csv}")
print(f"Saved: {train_csv}")

# 2.5.2 SPI_6M results
## Macroclimatics

TARGET_COL = f"SPI_{AGGREGATION.upper()}"
train_df_ccm = train_df.copy()
train_df_ccm[time_col] = pd.to_datetime(train_df_ccm[time_col])
train_df_ccm = train_df_ccm.set_index(time_col).sort_index()
ccm_cols = [TARGET_COL, "WP", "ONI", "PNA", "NAO", "NINO34", "PDO", "SOLAR"]
train_df_ccm = train_df_ccm[ccm_cols].dropna()
print(train_df_ccm.head())
print(train_df_ccm.shape)

E = choose_embedding_dimension(train_df_ccm, target_col=TARGET_COL, tau=1)
print("Best E:", E)
predictor_cols = ["WP", "ONI", "PNA", "NAO", "NINO34", "PDO", "SOLAR"]

ccm_scan_spi = run_ccm_lag_scan(
    train_df=train_df_ccm,
    target_col=TARGET_COL,
    predictor_cols=predictor_cols,
    max_lag=12,
    tau=1,
    E=E,
    n_surrogates=args.n_surrogates,
    alpha=0.05,
    min_rho=0.05,
)

plt.figure(figsize=(8, 5))
for pred in predictor_cols:
    sub = ccm_scan_spi[ccm_scan_spi["predictor"] == pred].sort_values("lag")
    if len(sub) > 0:
        plt.plot(sub["lag"], sub["final_rho"], marker="o", label=pred)
plt.xlabel("Lag (months)")
plt.ylabel("Final cross-map skill (rho)")
plt.title(f"CCM causal strength across lags for {TARGET_COL}")
plt.legend()
plt.grid(True, alpha=0.3)
save_current_plot(f"ccm_strength_spi_{AGGREGATION}.png")

plt.figure(figsize=(8, 5))
for pred in predictor_cols:
    sub = ccm_scan_spi[ccm_scan_spi["predictor"] == pred].sort_values("lag")
    if len(sub) > 0:
        plt.plot(sub["lag"], sub["improvement"], marker="o", label=pred)
plt.xlabel("Lag (months)")
plt.ylabel("Improvement")
plt.title(f"CCM improvement across lags for {TARGET_COL}")
plt.legend()
plt.grid(True, alpha=0.3)
save_current_plot(f"ccm_improvement_spi_{AGGREGATION}.png")

spei_csv = RESULTS_DIR / f"ccm_scan_spi_{AGGREGATION}.csv"
ccm_scan_spi.to_csv(spei_csv, index=False)
print(f"Saved: {spei_csv}")

## Locals

train_df_ccm_B = train_df.copy()
train_df_ccm_B[time_col] = pd.to_datetime(train_df_ccm_B[time_col])
train_df_ccm_B = train_df_ccm_B.set_index(time_col).sort_index()
ccm_cols_B = [
    TARGET_COL,
    "Precipitation_1M", "Precipitation_3M", "Precipitation_6M",
    "Temperature_1M", "Temperature_3M", "Temperature_6M"
]
train_df_ccm_B = train_df_ccm_B[ccm_cols_B].dropna()

E_B = choose_embedding_dimension(
     train_df_ccm_B,
     target_col=TARGET_COL,
     tau=1
 )
print("Best E:", E_B)

predictor_cols_B = [
    "Precipitation_1M", "Precipitation_3M", "Precipitation_6M",
    "Temperature_1M", "Temperature_3M", "Temperature_6M"
]

ccm_scan_spi_B = run_ccm_lag_scan(
    train_df=train_df_ccm_B,
    target_col=TARGET_COL,
    predictor_cols=predictor_cols_B,
    max_lag=12,
    tau=1,
    E=E_B,
    n_surrogates=args.n_surrogates,
    alpha=0.05,
    min_rho=0.05
)

plt.figure(figsize=(9, 5))

for pred in predictor_cols_B:
    sub = ccm_scan_spi_B[ccm_scan_spi_B["predictor"] == pred].sort_values("lag")
    if len(sub) > 0:
        plt.plot(sub["lag"], sub["final_rho"], marker="o", label=pred)

plt.xlabel("Lag (months)")
plt.ylabel("Final cross-map skill (rho)")
plt.title(f"CCM causal strength for Locals across lags for {TARGET_COL}")
plt.legend()
plt.grid(True, alpha=0.3)
save_current_plot(f"ccm_strength_Locals_spi_{AGGREGATION}.png")

plt.figure(figsize=(9, 5))

for pred in predictor_cols_B:
    sub = ccm_scan_spi_B[ccm_scan_spi_B["predictor"] == pred].sort_values("lag")
    if len(sub) > 0:
        plt.plot(sub["lag"], sub["improvement"], marker="o", label=pred)

plt.xlabel("Lag (months)")
plt.ylabel("Improvement")
plt.title(f"CCM improvement for Locals across lags for {TARGET_COL}")
plt.legend()
plt.grid(True, alpha=0.3)
save_current_plot(f"ccm_improvement_Locals_spi_{AGGREGATION}.png")

spei_csv = RESULTS_DIR / f"ccm_scan_Locals_spi_{AGGREGATION}.csv"
ccm_scan_spi_B.to_csv(spei_csv, index=False)
print(f"Saved: {spei_csv}")




