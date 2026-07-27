"""Market disruption detection: flags abnormal volatility/price regimes.

This is a rules-based composite score, not a black box -- each component is a
well-known indicator of a market shock (return outlier, volatility regime
shift, Bollinger breakout). Combining them gives an interpretable 0-100 score.
"""

import numpy as np
import pandas as pd

BASELINE_WINDOW = 60


def compute_disruption(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    ret = close.pct_change()

    # 1. Return outlier: how many baseline std-devs is today's move?
    roll_mean = ret.rolling(BASELINE_WINDOW).mean()
    roll_std = ret.rolling(BASELINE_WINDOW).std().replace(0, np.nan)
    return_z = (ret - roll_mean) / roll_std
    out["return_zscore"] = return_z

    # 2. Volatility regime shift: short vol vs long baseline vol
    short_vol = ret.rolling(5).std()
    long_vol = ret.rolling(BASELINE_WINDOW).std().replace(0, np.nan)
    vol_ratio = short_vol / long_vol
    out["vol_ratio"] = vol_ratio

    # 3. Bollinger breakout distance (0 = inside bands, >0 = outside)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    breakout = pd.Series(
        np.where(close > upper, (close - upper) / std20.replace(0, np.nan),
                 np.where(close < lower, (lower - close) / std20.replace(0, np.nan), 0.0)),
        index=out.index,
    )
    out["bb_breakout"] = breakout

    # Normalize each component to a 0-100-ish contribution and combine.
    return_component = (return_z.abs().clip(0, 4) / 4) * 100
    vol_component = ((vol_ratio - 1).clip(0, 3) / 3) * 100
    breakout_component = (breakout.clip(0, 3) / 3) * 100

    score = (0.45 * return_component.fillna(0)
             + 0.35 * vol_component.fillna(0)
             + 0.20 * breakout_component.fillna(0))
    out["disruption_score"] = score.clip(0, 100)

    def _label(s):
        if pd.isna(s):
            return "Unknown"
        if s >= 60:
            return "High Disruption"
        if s >= 30:
            return "Elevated"
        return "Normal"

    out["disruption_label"] = out["disruption_score"].apply(_label)
    return out
