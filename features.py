"""Technical-indicator feature engineering for price-movement prediction."""

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Given raw OHLCV data, return a DataFrame of model-ready features plus targets.

    Targets:
      target_direction: 1 if next-period return > 0 else 0
      target_return:    next-period simple return
    """
    out = df.copy()
    close = out["Close"]

    out["return_1d"] = close.pct_change()
    out["log_return_1d"] = np.log(close / close.shift(1))

    for lag in (1, 2, 3, 5):
        out[f"return_lag_{lag}"] = out["return_1d"].shift(lag)

    for window in (5, 10, 20):
        out[f"volatility_{window}"] = out["return_1d"].rolling(window).std()
        out[f"momentum_{window}"] = close.pct_change(window)
        ma = close.rolling(window).mean()
        out[f"ma_dist_{window}"] = (close - ma) / ma

    out["rsi_14"] = _rsi(close, 14)
    macd_line, signal_line, hist = _macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    out["bb_pct_b"] = (close - lower) / (upper - lower).replace(0, np.nan)
    out["bb_bandwidth"] = (upper - lower) / ma20

    out["price_zscore_20"] = (close - ma20) / std20.replace(0, np.nan)

    if "Volume" in out.columns and out["Volume"].fillna(0).sum() > 0:
        out["volume_change"] = out["Volume"].pct_change()
        out["volume_zscore_20"] = (
            out["Volume"] - out["Volume"].rolling(20).mean()
        ) / out["Volume"].rolling(20).std().replace(0, np.nan)
    else:
        out["volume_change"] = 0.0
        out["volume_zscore_20"] = 0.0

    out["target_return"] = out["return_1d"].shift(-1)
    out["target_direction"] = (out["target_return"] > 0).astype(int)

    # Near-zero denominators (e.g. flat volume/volatility) can blow up ratios to
    # values that overflow float32 during model fitting -- treat as missing.
    out = out.replace([np.inf, -np.inf], np.nan)

    return out


FEATURE_COLUMNS = [
    "return_1d",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_5",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "ma_dist_5",
    "ma_dist_10",
    "ma_dist_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_pct_b",
    "bb_bandwidth",
    "price_zscore_20",
    "volume_change",
    "volume_zscore_20",
]
