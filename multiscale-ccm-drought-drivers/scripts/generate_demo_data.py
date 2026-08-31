"""Generate synthetic monthly input data for a non-scientific smoke test."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "raw" / "data.xlsx"


def rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def main() -> None:
    rng = np.random.default_rng(42)
    time = pd.date_range("1980-01-01", periods=360, freq="MS")
    t = np.arange(len(time))
    enso = np.sin(2 * np.pi * t / 48) + rng.normal(0, 0.25, len(t))
    precipitation = 50 + 12 * np.sin(2 * np.pi * t / 12) - 4 * np.roll(enso, 5) + rng.normal(0, 5, len(t))
    temperature = 12 + 15 * np.sin(2 * np.pi * (t - 3) / 12) + rng.normal(0, 2, len(t))

    frame = pd.DataFrame({"Time": time})
    frame["ONI"] = enso
    frame["NINO34"] = enso + rng.normal(0, 0.1, len(t))
    for name, period in {"NAO": 30, "PDO": 96, "PNA": 24, "WP": 60}.items():
        frame[name] = np.sin(2 * np.pi * t / period) + rng.normal(0, 0.35, len(t))
    frame["SOLAR"] = 1361 + 0.8 * np.sin(2 * np.pi * t / 132)

    p = pd.Series(precipitation)
    temp = pd.Series(temperature)
    for window in (1, 3, 6, 12):
        frame[f"Precipitation_{window}M"] = rolling(p, window)
        frame[f"Temperature_{window}M"] = rolling(temp, window)
        water_balance = rolling(p - 1.4 * temp, window)
        frame[f"SPI_{window}M"] = (rolling(p, window) - rolling(p, window).mean()) / rolling(p, window).std()
        frame[f"SPEI_{window}M"] = (water_balance - water_balance.mean()) / water_balance.std()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(OUTPUT, index=False)
    print(f"Wrote synthetic demo data: {OUTPUT}")


if __name__ == "__main__":
    main()

