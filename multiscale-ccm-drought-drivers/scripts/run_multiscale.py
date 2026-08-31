# -*- coding: utf-8 -*-
"""CCM-based BiwaveSWT-style multiscale feature extraction.

This script keeps the paper's Stage 2/3 wavelet logic as the template:
1. select the target mother wavelet by causal SWT energy preservation;
2. select predictor-specific wavelet/component pairs by Spearman correlation;
3. build component-specific modeling datasets.

The intended methodological change is only the lag source:
CCF-selected lags in the paper are replaced here by CCM-selected lags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from eda import read_metadata
from wavelet import swt_mra_causal


WAVELET_FAMILIES = [
    "bior1.1", "bior1.3", "bior1.5", "bior2.2", "bior2.4", "bior2.6", "bior2.8",
    "bior3.1", "bior3.3", "bior3.5", "bior3.7", "bior3.9", "bior4.4", "bior5.5",
    "bior6.8", "coif1", "coif2", "coif3", "coif4", "coif5", "db1", "db2", "db3",
    "db4", "db5", "db6", "db7", "db8", "db9", "db10", "db11", "db12", "db13",
    "db14", "db15", "haar", "rbio1.1", "rbio1.3", "rbio1.5", "rbio2.2",
    "rbio2.4", "rbio2.6", "rbio2.8", "rbio3.1", "rbio3.3", "rbio3.5",
    "rbio3.7", "rbio3.9", "rbio4.4", "rbio5.5", "rbio6.8", "sym2", "sym3",
    "sym4", "sym5", "sym6", "sym7", "sym8", "sym9", "sym10", "sym11", "sym12",
    "sym13", "sym14", "sym15",
]

COMPONENTS = ["Original", "A2", "D2", "D1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper-style causal SWT multiscale datasets using CCM-selected lags."
    )
    parser.add_argument("--aggregation", default="6m", choices=["1m", "3m", "6m", "12m"])
    parser.add_argument("--target", default="spei", choices=["spei", "spi"])
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument(
        "--min-abs-corr",
        type=float,
        default=0.15,
        help="Spearman absolute-correlation threshold, matching the paper's usual screening level.",
    )
    parser.add_argument(
        "--wavelets",
        nargs="*",
        default=None,
        help="Optional subset of wavelets. Omit to use the paper-style candidate list.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional testing limit on selected CCM predictor-lag pairs.",
    )
    return parser.parse_args()


def resolve_metadata_path(aggregation: str) -> Path:
    for suffix in ("JSON", "json"):
        path = PROJECT_DIR / "metadata" / f"metadata{aggregation}.{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No metadata file found for aggregation {aggregation}")


def load_processed_split(aggregation: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed = PROJECT_DIR / "data" / "processed"
    train_path = processed / f"train_df_{aggregation}.csv"
    test_path = processed / f"test_df_{aggregation}.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing train/test files for {aggregation}. Run the CCM stage first."
        )
    return pd.read_csv(train_path), pd.read_csv(test_path)


def load_selected_ccm_lags(target: str, aggregation: str) -> pd.DataFrame:
    processed = PROJECT_DIR / "data" / "processed"
    selected_path = processed / f"selected_ccm_lags_{target}_{aggregation}.csv"
    if selected_path.exists():
        selected = pd.read_csv(selected_path)
    else:
        scans = []
        for name in [
            f"ccm_scan_{target}_{aggregation}.csv",
            f"ccm_scan_Locals_{target}_{aggregation}.csv",
        ]:
            path = processed / name
            if not path.exists():
                raise FileNotFoundError(f"Missing CCM scan file: {path}")
            scans.append(pd.read_csv(path))
        selected = pd.concat(scans, ignore_index=True)
        if "passed" in selected.columns:
            passed = selected[selected["passed"].astype(bool)].copy()
            if not passed.empty:
                selected = passed
        selected = selected[["predictor", "lag", "final_rho", "improvement", "p_value"]]

    required = {"predictor", "lag"}
    if not required.issubset(selected.columns):
        raise ValueError(f"Selected CCM lag file must include {sorted(required)}")

    selected = selected.dropna(subset=["predictor", "lag"]).copy()
    selected["lag"] = selected["lag"].astype(int)
    selected = selected.drop_duplicates(subset=["predictor", "lag"])
    return selected.sort_values(["predictor", "lag"]).reset_index(drop=True)


def lagged_name(predictor: str, lag: int) -> str:
    return f"{predictor}_lag{lag}"


def build_lagged_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_lags: pd.DataFrame,
    time_col: str,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = train_df.copy()
    test = test_df.copy()
    train["_split"] = "train"
    test["_split"] = "test"
    full = pd.concat([train, test], ignore_index=True)
    full[time_col] = pd.to_datetime(full[time_col])
    full = full.sort_values(time_col).reset_index(drop=True)

    feature_cols = []
    for _, row in selected_lags.iterrows():
        pred = str(row["predictor"])
        lag = int(row["lag"])
        if pred not in full.columns:
            print(f"Warning: predictor {pred} not found. Skipping.")
            continue
        col = lagged_name(pred, lag)
        full[col] = full[pred].shift(lag)
        feature_cols.append(col)

    keep = [time_col, "_split", target_col] + feature_cols
    full = full[keep]
    train_lagged = full[full["_split"] == "train"].drop(columns="_split")
    test_lagged = full[full["_split"] == "test"].drop(columns="_split")

    train_lagged = train_lagged.dropna(subset=[target_col] + feature_cols).reset_index(drop=True)
    test_lagged = test_lagged.dropna(subset=[target_col] + feature_cols).reset_index(drop=True)

    if train_lagged.empty or test_lagged.empty:
        raise ValueError("Lagged train or test data is empty after dropping missing values.")

    return train_lagged, test_lagged, feature_cols


def causal_components(
    series: pd.Series,
    wavelet: str,
    level: int,
    history: pd.Series | None = None,
) -> dict[str, np.ndarray]:
    hist_values = None if history is None else history.to_numpy(dtype=float)
    out = swt_mra_causal(
        series.to_numpy(dtype=float),
        wavefunc=wavelet,
        level=level,
        history=hist_values,
    )
    if level != 2:
        raise ValueError("This script currently writes A2, D2, D1 datasets and expects level=2.")
    return {
        "Original": series.to_numpy(dtype=float),
        "A2": np.asarray(out[1], dtype=float),
        "D2": np.asarray(out[2], dtype=float),
        "D1": np.asarray(out[3], dtype=float),
    }


def target_energy_ranking(
    train_target: pd.Series,
    wavelets: list[str],
    level: int,
) -> pd.DataFrame:
    records = []
    var_total = np.var(train_target.to_numpy(dtype=float))
    for wavelet in wavelets:
        try:
            comps = causal_components(train_target, wavelet=wavelet, level=level)
            energy = (
                np.var(comps["A2"]) + np.var(comps["D2"]) + np.var(comps["D1"])
            ) / var_total
            records.append(
                {
                    "wavelet": wavelet,
                    "E_total": energy,
                    "E_A2": np.var(comps["A2"]) / var_total,
                    "E_D2": np.var(comps["D2"]) / var_total,
                    "E_D1": np.var(comps["D1"]) / var_total,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "wavelet": wavelet,
                    "E_total": np.nan,
                    "E_A2": np.nan,
                    "E_D2": np.nan,
                    "E_D1": np.nan,
                    "error": str(exc),
                }
            )
    return (
        pd.DataFrame(records)
        .sort_values("E_total", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def safe_abs_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    corr, p_value = spearmanr(x, y, nan_policy="omit")
    if not np.isfinite(corr):
        return np.nan, np.nan
    return float(corr), float(p_value)


def select_predictor_wavelets(
    train_lagged: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    target_wavelet: str,
    wavelets: list[str],
    level: int,
    min_abs_corr: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    target_comps = causal_components(train_lagged[target_col], target_wavelet, level)
    predictor_component_cache: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    all_records = []

    for feature in feature_cols:
        predictor_component_cache[feature] = {}
        for wavelet in wavelets:
            try:
                pred_comps = causal_components(train_lagged[feature], wavelet, level)
            except Exception as exc:
                all_records.append(
                    {
                        "Variable": feature,
                        "Target": None,
                        "Predictor": None,
                        "Wavelet": wavelet,
                        "Spearman": np.nan,
                        "abs_spearman": np.nan,
                        "p_value": np.nan,
                        "error": str(exc),
                    }
                )
                continue
            predictor_component_cache[feature][wavelet] = pred_comps

            for target_component in COMPONENTS:
                for predictor_component in COMPONENTS:
                    corr, p_value = safe_abs_spearman(
                        pred_comps[predictor_component],
                        target_comps[target_component],
                    )
                    all_records.append(
                        {
                            "Variable": feature,
                            "Target": target_component,
                            "Predictor": predictor_component,
                            "Wavelet": wavelet,
                            "Spearman": corr,
                            "abs_spearman": abs(corr) if np.isfinite(corr) else np.nan,
                            "p_value": p_value,
                        }
                    )

    all_df = pd.DataFrame(all_records)
    valid = all_df.dropna(subset=["Target", "Predictor", "abs_spearman"]).copy()
    best = (
        valid.sort_values("abs_spearman", ascending=False)
        .groupby(["Variable", "Target", "Predictor"], as_index=False)
        .first()
    )
    selected = best[best["abs_spearman"] >= min_abs_corr].copy()
    selected = selected.sort_values(
        ["Target", "abs_spearman"], ascending=[True, False]
    ).reset_index(drop=True)
    return selected, predictor_component_cache


def build_component_datasets(
    train_lagged: pd.DataFrame,
    test_lagged: pd.DataFrame,
    time_col: str,
    target_col: str,
    target_wavelet: str,
    selected_features: pd.DataFrame,
    level: int,
    output_prefix: str,
) -> None:
    processed = PROJECT_DIR / "data" / "processed"
    train_target_comps = causal_components(train_lagged[target_col], target_wavelet, level)
    test_target_comps = causal_components(
        test_lagged[target_col],
        target_wavelet,
        level,
        history=train_lagged[target_col],
    )

    for target_component in COMPONENTS:
        train_out = pd.DataFrame({time_col: train_lagged[time_col]})
        test_out = pd.DataFrame({time_col: test_lagged[time_col]})
        train_out[target_col] = train_target_comps[target_component]
        test_out[target_col] = test_target_comps[target_component]

        sub = selected_features[selected_features["Target"] == target_component]
        for _, row in sub.iterrows():
            variable = row["Variable"]
            predictor_component = row["Predictor"]
            wavelet = row["Wavelet"]
            col_name = f"{variable}_{predictor_component}_{wavelet}"

            train_pred = causal_components(train_lagged[variable], wavelet, level)
            test_pred = causal_components(
                test_lagged[variable],
                wavelet,
                level,
                history=train_lagged[variable],
            )
            train_out[col_name] = train_pred[predictor_component]
            test_out[col_name] = test_pred[predictor_component]

        train_path = processed / f"{output_prefix}_train_{target_component}.csv"
        test_path = processed / f"{output_prefix}_test_{target_component}.csv"
        train_out.to_csv(train_path, index=False)
        test_out.to_csv(test_path, index=False)
        print(f"Saved: {train_path}")
        print(f"Saved: {test_path}")


def main() -> None:
    args = parse_args()
    aggregation = args.aggregation.lower()
    target_key = args.target.lower()
    target_col = f"{target_key.upper()}_{aggregation.upper()}"
    wavelets = args.wavelets or WAVELET_FAMILIES

    metadata = read_metadata(str(resolve_metadata_path(aggregation)))
    time_col = str(metadata["time"][0])
    if target_col not in metadata["target"]:
        raise ValueError(f"{target_col} is not listed in metadata targets: {metadata['target']}")

    train_df, test_df = load_processed_split(aggregation)
    selected_lags = load_selected_ccm_lags(target_key, aggregation)
    if args.max_pairs is not None:
        selected_lags = selected_lags.head(args.max_pairs).copy()

    print("=" * 78)
    print("CCM-BiwaveSWT-style multiscale feature extraction")
    print("=" * 78)
    print(f"Target: {target_col}")
    print(f"Aggregation: {aggregation}")
    print(f"CCM predictor-lag pairs: {len(selected_lags)}")
    print(f"Candidate wavelets: {len(wavelets)}")

    train_lagged, test_lagged, feature_cols = build_lagged_splits(
        train_df=train_df,
        test_df=test_df,
        selected_lags=selected_lags,
        time_col=time_col,
        target_col=target_col,
    )
    print(f"Lagged train shape: {train_lagged.shape}")
    print(f"Lagged test shape: {test_lagged.shape}")

    processed = PROJECT_DIR / "data" / "processed"
    prefix = f"ccm_biwave_{target_key}_{aggregation}"

    ranking = target_energy_ranking(train_lagged[target_col], wavelets, args.level)
    ranking_path = processed / f"{prefix}_target_wavelet_energy.csv"
    ranking.to_csv(ranking_path, index=False)
    target_wavelet = str(ranking.loc[0, "wavelet"])
    print(f"Selected target wavelet: {target_wavelet}")
    print(f"Saved: {ranking_path}")

    selected_features, _ = select_predictor_wavelets(
        train_lagged=train_lagged,
        target_col=target_col,
        feature_cols=feature_cols,
        target_wavelet=target_wavelet,
        wavelets=wavelets,
        level=args.level,
        min_abs_corr=args.min_abs_corr,
    )
    selected_path = processed / f"{prefix}_selected_predictor_wavelets.csv"
    selected_features.to_csv(selected_path, index=False)
    print(f"Selected multiscale predictor-component rows: {len(selected_features)}")
    print(f"Saved: {selected_path}")

    build_component_datasets(
        train_lagged=train_lagged,
        test_lagged=test_lagged,
        time_col=time_col,
        target_col=target_col,
        target_wavelet=target_wavelet,
        selected_features=selected_features,
        level=args.level,
        output_prefix=prefix,
    )

    print("=" * 78)
    print("Done.")
    print("=" * 78)


if __name__ == "__main__":
    main()
