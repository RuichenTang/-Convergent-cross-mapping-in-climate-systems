"""Selection rules for CCM predictor–lag scans."""

from __future__ import annotations

import pandas as pd


def select_ccm_lags(
    scan: pd.DataFrame,
    *,
    min_rho: float = 0.05,
    min_improvement: float = 0.0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Return unique predictor–lag pairs passing the strongest available rule.

    The combined ``passed`` flag is authoritative when present. Otherwise the
    function falls back to significance and numeric convergence thresholds.
    """
    required = {"predictor", "lag"}
    missing = required.difference(scan.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = scan.copy()
    if "passed" in result.columns:
        mask = result["passed"].fillna(False).astype(bool)
    else:
        mask = pd.Series(True, index=result.index)
        if "significant" in result.columns:
            mask &= result["significant"].fillna(False).astype(bool)
        elif "p_value" in result.columns:
            mask &= result["p_value"].le(alpha)
        if "final_rho" in result.columns:
            mask &= result["final_rho"].ge(min_rho)
        if "improvement" in result.columns:
            mask &= result["improvement"].gt(min_improvement)
        if "slope" in result.columns:
            mask &= result["slope"].gt(0)

    keep = [
        column
        for column in ("predictor", "lag", "final_rho", "improvement", "p_value")
        if column in result.columns
    ]
    return (
        result.loc[mask, keep]
        .drop_duplicates(subset=["predictor", "lag"])
        .sort_values(["predictor", "lag"])
        .reset_index(drop=True)
    )

