import numpy as np
import pandas as pd
from pyEDM import Simplex, CCM

"""
Automatically determine the optimal embedding dimension (E) for the
# target time series using simplex projection forecasting.
Method:
# - Try several candidate embedding dimensions (E_values)
# - For each E, run simplex forecasting using pyEDM
# - Measure prediction skill (Pearson correlation between observed and predicted)
# - Select the E that gives the highest prediction skill (rho)
"""
def choose_embedding_dimension(
    df: pd.DataFrame,
    target_col: str,
    E_values=range(2, 12),
    tau: int = 1,
    lib_sizes=None,
) -> int:
    """
    Choose embedding dimension E for the target series using simplex forecasting.
    """
    if lib_sizes is None:
        n = len(df)
        lib_sizes = f"{max(20, n//4)} {max(30, n//2)} {max(40, n-5)}"

    best_E = None
    best_rho = -np.inf

    temp = df[[target_col]].dropna().copy()
    temp.insert(0, "Time", np.arange(1, len(temp) + 1))

    for E in E_values:
        try:
            res = Simplex(
                dataFrame=temp,
                lib=f"1 {len(temp)-10}",
                pred=f"{len(temp)-9} {len(temp)-1}",   # 注意这里也改一下
                E=E,
                Tp=1,
                tau=tau,
                columns=target_col,
                target=target_col,
                showPlot=False,
                verbose=False,
            )

            valid = res[["Observations", "Predictions"]].dropna()
            if len(valid) < 3:
                continue

            rho = np.corrcoef(
                valid["Observations"].values,
                valid["Predictions"].values
            )[0, 1]

            if np.isfinite(rho) and rho > best_rho:
                best_rho = rho
                best_E = E

        except Exception as e:
            print(f"E={E} failed: {e}")
            continue

    if best_E is None:
        raise ValueError(f"Could not determine embedding dimension for {target_col}")

    return best_E


def seasonal_surrogate(series: pd.Series, random_state=None) -> pd.Series:
    """
    Very simple surrogate generator:
    keeps month-of-year seasonality and shuffles residuals within the series.
    Assumes DatetimeIndex.
    """
    rng = np.random.default_rng(random_state)

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("seasonal_surrogate requires DatetimeIndex")

    s = series.copy()
    month_means = s.groupby(s.index.month).transform("mean")
    residuals = s - month_means
    shuffled = pd.Series(rng.permutation(residuals.values), index=residuals.index)

    return month_means + shuffled

"""
Execute a single Convergent Cross Mapping (CCM) test between two variables.
"""

def run_single_ccm(
    df: pd.DataFrame,
    cause_col: str,
    effect_col: str,
    E: int,
    tau: int = 1,
    lib_sizes=None,
    sample: int = 100,
    random: bool = False,
) -> pd.DataFrame:
    temp = df[[cause_col, effect_col]].dropna().copy()
    temp.insert(0, "Time", np.arange(1, len(temp) + 1))

    if lib_sizes is None:
        n = len(temp)
        lib_min = max(E + 2, 20)
        lib_max = max(lib_min + 10, n - 5)
        step = max(5, (lib_max - lib_min) // 6)
        lib_sizes = " ".join(map(str, range(lib_min, lib_max + 1, step)))

    res = CCM(
        dataFrame=temp,
        E=E,
        tau=tau,
        columns=effect_col,    
        target=cause_col,      
        libSizes=lib_sizes,
        sample=sample,
        showPlot=False,
        verbose=False,
    )
    return res


def ccm_convergence_score(ccm_result: pd.DataFrame) -> dict:
    """
    Summarize CCM result:
    - final rho
    - slope of rho vs library size
    """
    out = ccm_result.copy()

    rho_cols = [c for c in out.columns if c != "LibSize"]
    if len(rho_cols) == 0:
        raise ValueError("No rho column found in CCM output")

    rho_col = rho_cols[0]
    x = out["LibSize"].values.astype(float)
    y = out[rho_col].values.astype(float)

    final_rho = y[-1]
    slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else np.nan
    improvement = y[-1] - y[0] if len(y) > 1 else np.nan

    return {
        "rho_col": rho_col,
        "final_rho": final_rho,
        "slope": slope,
        "improvement": improvement,
    }


def test_ccm_significance(
    df: pd.DataFrame,
    cause_col: str,
    effect_col: str,
    E: int,
    tau: int = 1,
    n_surrogates: int = 50,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict:
    """
    Simple significance test using seasonal surrogates of the cause variable.
    Compare observed final rho against surrogate distribution.
    """
    observed = run_single_ccm(df, cause_col, effect_col, E=E, tau=tau)
    obs_stats = ccm_convergence_score(observed)
    obs_rho = obs_stats["final_rho"]

    rng = np.random.default_rng(random_state)
    surrogate_rhos = []

    for i in range(n_surrogates):
        tmp = df[[cause_col, effect_col]].dropna().copy()
        tmp[cause_col] = seasonal_surrogate(
            tmp[cause_col],
            random_state=int(rng.integers(1e9))
        )
        try:
            res_surr = run_single_ccm(tmp, cause_col, effect_col, E=E, tau=tau)
            surr_stats = ccm_convergence_score(res_surr)
            surrogate_rhos.append(surr_stats["final_rho"])
        except Exception:
            continue

    surrogate_rhos = np.array(surrogate_rhos)
    if len(surrogate_rhos) == 0:
        return {
            "observed_rho": obs_rho,
            "p_value": np.nan,
            "significant": False,
            "surrogate_mean": np.nan,
        }

    # Add-one correction avoids impossible p=0 estimates with finite surrogates.
    p_value = (np.sum(surrogate_rhos >= obs_rho) + 1) / (len(surrogate_rhos) + 1)
    significant = p_value < alpha

    return {
        "observed_rho": obs_rho,
        "p_value": p_value,
        "significant": significant,
        "surrogate_mean": surrogate_rhos.mean(),
    }


"""
Main Pipeline B function for identifying causal predictors using CCM.
For each candidate predictor variable, the function scans multiple
time lags and evaluates causal influence on the target variable.
method:
Workflow:
1. Determine embedding dimension (if not provided)
2. For each predictor variable:
       For each lag (0 → max_lag):
           - shift predictor to create lagged series
           - run CCM
           - evaluate convergence
           - test statistical significance
3. Record results for each lag

Filtering criteria:
 p-value < alpha
 final rho above threshold
 optional requirement for positive convergence slope
"""
def run_ccm_lag_scan(
    train_df: pd.DataFrame,
    target_col: str,
    predictor_cols: list,
    max_lag: int = 12,
    tau: int = 1,
    E: int = None,
    n_surrogates: int = 30,
    alpha: float = 0.05,
    min_rho: float = 0.05,
    require_positive_slope: bool = True,
) -> pd.DataFrame:
    df = train_df.copy()

    if E is None:
        E = choose_embedding_dimension(df, target_col=target_col, tau=tau)

    results = []

    for predictor in predictor_cols:
        for lag in range(0, max_lag + 1):
            temp = df[[target_col, predictor]].copy()
            temp[f"{predictor}_lag{lag}"] = temp[predictor].shift(lag)
            temp = temp[[target_col, f"{predictor}_lag{lag}"]].dropna()

            if len(temp) < 50:
                continue

            cause_col = f"{predictor}_lag{lag}"
            effect_col = target_col

            try:
                # observed CCM
                ccm_res = run_single_ccm(
                    temp,
                    cause_col=cause_col,
                    effect_col=effect_col,
                    E=E,
                    tau=tau,
                )
                stats = ccm_convergence_score(ccm_res)

                # significance
                sig = test_ccm_significance(
                    temp,
                    cause_col=cause_col,
                    effect_col=effect_col,
                    E=E,
                    tau=tau,
                    n_surrogates=n_surrogates,
                    alpha=alpha,
                )

                passed = sig["significant"] and stats["final_rho"] >= min_rho
                if require_positive_slope:
                    passed = passed and (stats["slope"] > 0)

                results.append({
                    "target": target_col,
                    "predictor": predictor,
                    "lag": lag,
                    "E": E,
                    "final_rho": stats["final_rho"],
                    "slope": stats["slope"],
                    "improvement": stats["improvement"],
                    "p_value": sig["p_value"],
                    "significant": sig["significant"],
                    "passed": passed,
                })

            except Exception as e:
                results.append({
                    "target": target_col,
                    "predictor": predictor,
                    "lag": lag,
                    "E": E,
                    "final_rho": np.nan,
                    "slope": np.nan,
                    "improvement": np.nan,
                    "p_value": np.nan,
                    "significant": False,
                    "passed": False,
                    "error": str(e),
                })

    return pd.DataFrame(results)


def select_best_ccm_predictors(ccm_scan_df: pd.DataFrame) -> pd.DataFrame:
    """
    From lag-scan results, keep the best lag for each predictor.
    """
    valid = ccm_scan_df[ccm_scan_df["passed"]].copy()
    if valid.empty:
        return valid

    best = (
        valid.sort_values(["predictor", "final_rho"], ascending=[True, False])
             .groupby("predictor", as_index=False)
             .first()
    )
    return best
