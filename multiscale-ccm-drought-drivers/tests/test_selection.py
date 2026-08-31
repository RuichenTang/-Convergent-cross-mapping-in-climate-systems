import pandas as pd

from src.selection import select_ccm_lags


def test_passed_flag_is_authoritative() -> None:
    scan = pd.DataFrame(
        {
            "predictor": ["ONI", "PDO"],
            "lag": [5, 10],
            "final_rho": [0.12, 0.20],
            "improvement": [0.08, 0.10],
            "significant": [True, True],
            "passed": [False, True],
        }
    )
    selected = select_ccm_lags(scan)
    assert selected[["predictor", "lag"]].to_dict("records") == [
        {"predictor": "PDO", "lag": 10}
    ]

