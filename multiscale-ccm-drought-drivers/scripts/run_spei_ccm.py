# -*- coding: utf-8 -*-
"""ICC-friendly CCM script.

Cleaned from a Colab-exported notebook for batch execution on ICC.
Key changes:
- no input(); use --aggregation argument instead
- no notebook-only display()
- save figures instead of plt.show()
- use metadata time column consistently
- explicit imports
"""

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
from selection import select_ccm_lags


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

print("Missing Values Analysis:")
print("=" * 40)
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_summary = pd.DataFrame({
    "Missing_Count": missing_values,
    "Missing_Percentage": missing_percentage,
}).round(2)
print(missing_summary)

plt.figure(figsize=(12, 6))
missing_summary[missing_summary["Missing_Count"] > 0]["Missing_Percentage"].plot(kind="bar")
plt.title("Missing Values by Variable (%)")
plt.ylabel("Percentage Missing")
plt.xlabel("Variables")
plt.xticks(rotation=45)
save_current_plot(f"missing_values_{AGGREGATION}.png")

print("Data Cleaning:")
print("=" * 20)
initial_rows = len(df)
df_clean = df.dropna()
final_rows = len(df_clean)
print(f"Initial rows: {initial_rows}")
print(f"Final rows: {final_rows}")
print(f"Rows removed: {initial_rows - final_rows}")
df = df_clean.copy()
print("Data cleaning completed")

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


# 2.5.2 SPEI_6M results
## Macroclimatics

TARGET_COL = f"SPEI_{AGGREGATION.upper()}"
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

ccm_scan_spei = run_ccm_lag_scan(
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
    sub = ccm_scan_spei[ccm_scan_spei["predictor"] == pred].sort_values("lag")
    if len(sub) > 0:
        plt.plot(sub["lag"], sub["final_rho"], marker="o", label=pred)
plt.xlabel("Lag (months)")
plt.ylabel("Final cross-map skill (rho)")
plt.title(f"CCM causal strength across lags for {TARGET_COL}")
plt.legend()
plt.grid(True, alpha=0.3)
save_current_plot(f"ccm_strength_spei_{AGGREGATION}.png")

plt.figure(figsize=(8, 5))
for pred in predictor_cols:
    sub = ccm_scan_spei[ccm_scan_spei["predictor"] == pred].sort_values("lag")
    if len(sub) > 0:
        plt.plot(sub["lag"], sub["improvement"], marker="o", label=pred)
plt.xlabel("Lag (months)")
plt.ylabel("Improvement")
plt.title(f"CCM improvement across lags for {TARGET_COL}")
plt.legend()
plt.grid(True, alpha=0.3)
save_current_plot(f"ccm_improvement_spei_{AGGREGATION}.png")

spei_csv = RESULTS_DIR / f"ccm_scan_spei_{AGGREGATION}.csv"
ccm_scan_spei.to_csv(spei_csv, index=False)
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

ccm_scan_spei_B = run_ccm_lag_scan(
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
    sub = ccm_scan_spei_B[ccm_scan_spei_B["predictor"] == pred].sort_values("lag")
    if len(sub) > 0:
        plt.plot(sub["lag"], sub["final_rho"], marker="o", label=pred)

plt.xlabel("Lag (months)")
plt.ylabel("Final cross-map skill (rho)")
plt.title(f"CCM causal strength for Locals across lags for {TARGET_COL}")
plt.legend()
plt.grid(True, alpha=0.3)
save_current_plot(f"ccm_strength_Locals_spei_{AGGREGATION}.png")

plt.figure(figsize=(9, 5))

for pred in predictor_cols_B:
    sub = ccm_scan_spei_B[ccm_scan_spei_B["predictor"] == pred].sort_values("lag")
    if len(sub) > 0:
        plt.plot(sub["lag"], sub["improvement"], marker="o", label=pred)

plt.xlabel("Lag (months)")
plt.ylabel("Improvement")
plt.title(f"CCM improvement for Locals across lags for {TARGET_COL}")
plt.legend()
plt.grid(True, alpha=0.3)
save_current_plot(f"ccm_improvement_Locals_spei_{AGGREGATION}.png")

spei_csv = RESULTS_DIR / f"ccm_scan_Locals_spei_{AGGREGATION}.csv"
ccm_scan_spei_B.to_csv(spei_csv, index=False)
print(f"Saved: {spei_csv}")


# ============================================================
# 3. Multiscale Feature Extraction
# ============================================================
# This section takes the selected CCM lags and transforms each
# lagged predictor into multiscale wavelet components:
#   X(t-lag) -> A2 + D2 + D1
#
# Output:
#   1. multiscale_train_spei_{aggregation}.csv
#   2. multiscale_test_spei_{aggregation}.csv
#   3. multiscale_spearman_summary_spei_{aggregation}.csv
#   4. selected_multiscale_features_spei_{aggregation}.csv
# ============================================================

print("\n" + "=" * 70)
print("Step 3: Multiscale Feature Extraction")
print("=" * 70)

try:
    import pywt
except ImportError:
    raise ImportError(
        "PyWavelets is required for wavelet decomposition. "
        "Install it with: pip install PyWavelets"
    )


def infer_selected_ccm_lags(
    ccm_result: pd.DataFrame,
    rho_col: str = "final_rho",
    improvement_col: str = "improvement",
    min_rho: float = 0.05,
    min_improvement: float = 0.0,
) -> pd.DataFrame:
    """
    Select predictor-lag pairs from CCM scan results.

    The function is written to be robust because the exact column names
    may differ depending on the CCM function implementation.
    It prioritizes explicit significance columns if they exist, otherwise
    it uses final_rho and improvement thresholds.
    """

    result = ccm_result.copy()

    # Case 1: if the CCM function already returns a selected/significant column
    possible_flag_cols = [
        "passed",
        "selected",
        "is_selected",
        "significant",
        "is_significant",
        "pass_test",
    ]

    flag_col = None
    for col in possible_flag_cols:
        if col in result.columns:
            flag_col = col
            break

    if flag_col is not None:
        selected = result[result[flag_col].astype(bool)].copy()

    # Case 2: use p-value if available
    elif "p_value" in result.columns:
        selected = result[
            (result["p_value"] <= 0.05)
            & (result[rho_col] >= min_rho)
        ].copy()

    # Case 3: use rho and improvement thresholds
    else:
        selected = result[
            (result[rho_col] >= min_rho)
            & (result[improvement_col] > min_improvement)
        ].copy()

    selected = selected[["predictor", "lag", rho_col, improvement_col]].copy()
    selected["lag"] = selected["lag"].astype(int)

    return selected.sort_values(["predictor", "lag"]).reset_index(drop=True)


def pad_to_swt_length(x: np.ndarray, level: int = 2) -> tuple[np.ndarray, int]:
    """
    SWT requires the length to be divisible by 2**level.
    This function pads the series at the end and returns the padded array
    plus the original length.
    """

    original_len = len(x)
    block = 2 ** level
    remainder = original_len % block

    if remainder == 0:
        return x, original_len

    pad_len = block - remainder
    x_padded = np.pad(x, (0, pad_len), mode="edge")

    return x_padded, original_len


def swt_decompose_series(
    series: pd.Series,
    wavelet: str = "db4",
    level: int = 2,
) -> pd.DataFrame:
    """
    Apply SWT decomposition and return A2, D2, and D1 components.

    For level=2:
      A2 = low-frequency approximation
      D2 = medium-frequency detail
      D1 = high-frequency detail
    """

    x = series.astype(float).values

    # Fill remaining NaN values to avoid wavelet errors
    x = pd.Series(x).interpolate().bfill().ffill().values

    x_padded, original_len = pad_to_swt_length(x, level=level)

    coeffs = pywt.swt(
        x_padded,
        wavelet=wavelet,
        level=level,
        trim_approx=False,
    )

    # For level=2, coeffs usually returns:
    # [(cA2, cD2), (cA1, cD1)]
    cA2, cD2 = coeffs[0]
    _, cD1 = coeffs[1]

    components = pd.DataFrame(
        {
            "A2": cA2[:original_len],
            "D2": cD2[:original_len],
            "D1": cD1[:original_len],
        },
        index=series.index,
    )

    return components


def build_multiscale_features(
    base_df: pd.DataFrame,
    time_col: str,
    target_col: str,
    selected_lags: pd.DataFrame,
    wavelet: str = "db4",
    level: int = 2,
) -> pd.DataFrame:
    """
    Build multiscale lagged features from selected predictor-lag pairs.

    For each selected pair:
      predictor with lag l -> predictor_lag{l}_A2, predictor_lag{l}_D2, predictor_lag{l}_D1
    """

    df_work = base_df.copy()
    df_work[time_col] = pd.to_datetime(df_work[time_col])
    df_work = df_work.set_index(time_col).sort_index()

    output = pd.DataFrame(index=df_work.index)
    output[target_col] = df_work[target_col]

    for _, row in selected_lags.iterrows():
        pred = row["predictor"]
        lag = int(row["lag"])

        if pred not in df_work.columns:
            print(f"Warning: predictor {pred} not found in dataframe. Skipping.")
            continue

        # Predictor at t-lag for target at time t
        lagged_series = df_work[pred].shift(lag)

        components = swt_decompose_series(
            lagged_series,
            wavelet=wavelet,
            level=level,
        )

        for comp in ["A2", "D2", "D1"]:
            feature_name = f"{pred}_lag{lag}_{comp}"
            output[feature_name] = components[comp]

    output = output.dropna().reset_index()

    return output


def spearman_screening(
    feature_df: pd.DataFrame,
    time_col: str,
    target_col: str,
    min_abs_corr: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Screen multiscale features using absolute Spearman correlation
    with the target variable.
    """

    feature_cols = [
        col for col in feature_df.columns
        if col not in [time_col, target_col]
    ]

    records = []

    for col in feature_cols:
        corr = feature_df[[target_col, col]].corr(method="spearman").iloc[0, 1]
        records.append(
            {
                "feature": col,
                "spearman_corr": corr,
                "abs_spearman_corr": abs(corr),
            }
        )

    summary = pd.DataFrame(records)
    summary = summary.sort_values(
        "abs_spearman_corr",
        ascending=False,
    ).reset_index(drop=True)

    selected_features = summary[
        summary["abs_spearman_corr"] >= min_abs_corr
    ].copy()

    keep_cols = [time_col, target_col] + selected_features["feature"].tolist()
    screened_df = feature_df[keep_cols].copy()

    return screened_df, summary


# ------------------------------------------------------------
# 3.1 Combine selected CCM lags from macroclimatic and local predictors
# ------------------------------------------------------------

selected_macro_lags = select_ccm_lags(
    ccm_scan_spei,
    min_rho=0.05,
    min_improvement=0.0,
)

selected_local_lags = select_ccm_lags(
    ccm_scan_spei_B,
    min_rho=0.05,
    min_improvement=0.0,
)

selected_ccm_lags = pd.concat(
    [selected_macro_lags, selected_local_lags],
    ignore_index=True,
).drop_duplicates(subset=["predictor", "lag"])

selected_ccm_lags = selected_ccm_lags.sort_values(
    ["predictor", "lag"]
).reset_index(drop=True)

selected_lags_csv = RESULTS_DIR / f"selected_ccm_lags_spei_{AGGREGATION}.csv"
selected_ccm_lags.to_csv(selected_lags_csv, index=False)
print(f"Saved selected CCM lags: {selected_lags_csv}")

print("\nSelected CCM lags:")
print(selected_ccm_lags)


# ------------------------------------------------------------
# 3.2 Build multiscale features for train and test sets
# ------------------------------------------------------------

multiscale_train = build_multiscale_features(
    base_df=train_df,
    time_col=time_col,
    target_col=TARGET_COL,
    selected_lags=selected_ccm_lags,
    wavelet="db4",
    level=2,
)

multiscale_test = build_multiscale_features(
    base_df=test_df,
    time_col=time_col,
    target_col=TARGET_COL,
    selected_lags=selected_ccm_lags,
    wavelet="db4",
    level=2,
)

print(f"Multiscale train shape: {multiscale_train.shape}")
print(f"Multiscale test shape: {multiscale_test.shape}")


# ------------------------------------------------------------
# 3.3 Spearman screening on training set only
# ------------------------------------------------------------

screened_train, spearman_summary = spearman_screening(
    feature_df=multiscale_train,
    time_col=time_col,
    target_col=TARGET_COL,
    min_abs_corr=0.10,
)

selected_feature_names = [
    col for col in screened_train.columns
    if col not in [time_col, TARGET_COL]
]

screened_test = multiscale_test[
    [time_col, TARGET_COL] + [
        col for col in selected_feature_names
        if col in multiscale_test.columns
    ]
].copy()

print(f"Selected multiscale features: {len(selected_feature_names)}")


# ------------------------------------------------------------
# 3.4 Save outputs
# ------------------------------------------------------------

multiscale_train_csv = RESULTS_DIR / f"multiscale_train_spei_{AGGREGATION}.csv"
multiscale_test_csv = RESULTS_DIR / f"multiscale_test_spei_{AGGREGATION}.csv"
spearman_summary_csv = RESULTS_DIR / f"multiscale_spearman_summary_spei_{AGGREGATION}.csv"
selected_features_csv = RESULTS_DIR / f"selected_multiscale_features_spei_{AGGREGATION}.csv"

screened_train.to_csv(multiscale_train_csv, index=False)
screened_test.to_csv(multiscale_test_csv, index=False)
spearman_summary.to_csv(spearman_summary_csv, index=False)

pd.DataFrame(
    {
        "selected_feature": selected_feature_names
    }
).to_csv(selected_features_csv, index=False)


# ------------------------------------------------------------
# 3.5 Plot top multiscale features
# ------------------------------------------------------------

top_n = min(20, len(spearman_summary))

if top_n > 0:
    plt.figure(figsize=(10, 6))
    top_features = spearman_summary.head(top_n).sort_values("abs_spearman_corr")
    plt.barh(top_features["feature"], top_features["abs_spearman_corr"])
    plt.xlabel("Absolute Spearman correlation")
    plt.ylabel("Multiscale feature")
    plt.title(f"Top multiscale features for {TARGET_COL}")
    plt.grid(True, axis="x", alpha=0.3)
    save_current_plot(f"top_multiscale_features_spei_{AGGREGATION}.png")

print("Step 3 completed successfully.")
