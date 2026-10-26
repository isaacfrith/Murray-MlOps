"""Scoring: MAE per horizon, the persistence baseline, and regime breakdowns."""

import numpy as np
import pandas as pd

from murray.transforms.labels import HORIZONS

# A Yarrawonga change of this many metres over 7 days marks a "release change" period
# (the Murray is regulated: dam release decisions move levels). Steady otherwise.
RELEASE_CHANGE_M = 0.30


def persistence_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """'Every future day equals the last observed day.'"""
    return pd.DataFrame({f"h{h}": df["barmah_level_m"].to_numpy() for h in HORIZONS},
                        index=df.index)


def mae_by_horizon(pred: pd.DataFrame, df: pd.DataFrame) -> dict[int, float]:
    """MAE in metres per horizon, ignoring rows whose target is missing."""
    out = {}
    for h in HORIZONS:
        err = (pred[f"h{h}"] - df[f"target_h{h}"]).abs()
        out[h] = float(err.mean()) if err.notna().any() else float("nan")
    return out


def regime(df: pd.DataFrame) -> pd.Series:
    changing = df["yarra_change_7d"].abs() >= RELEASE_CHANGE_M
    return pd.Series(np.where(changing, "release_change", "steady"), index=df.index)


def score_table(pred: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Rows: (regime, model|persistence); columns: h1..h7 MAE in metres."""
    rows = []
    reg = regime(df)
    base = persistence_forecast(df)
    for name in ["all", "steady", "release_change"]:
        mask = reg.notna() if name == "all" else reg.eq(name)
        if not mask.any():
            continue
        for label, p in (("model", pred), ("persistence", base)):
            maes = mae_by_horizon(p[mask], df[mask])
            rows.append({"regime": name, "forecaster": label, "n": int(mask.sum()),
                         **{f"h{h}": maes[h] for h in HORIZONS}})
    return pd.DataFrame(rows)
