import numpy as np
import pandas as pd
import pytest

from disruption import compute_disruption


def _make_df(closes):
    return pd.DataFrame({"Close": closes})


def test_disruption_score_stays_in_bounds():
    rng = np.random.default_rng(0)
    closes = 100 + np.cumsum(rng.normal(0, 1, size=200))
    out = compute_disruption(_make_df(closes))

    assert out["disruption_score"].min() >= 0
    assert out["disruption_score"].max() <= 100
    assert set(out["disruption_label"].unique()) <= {
        "Unknown",
        "Normal",
        "Elevated",
        "High Disruption",
    }


def test_flat_price_series_scores_as_normal_or_unknown():
    closes = [100.0] * 100
    out = compute_disruption(_make_df(closes))

    # zero variance -> ratios are NaN -> components fill to 0 -> score 0
    tail = out["disruption_label"].iloc[70:]
    assert set(tail.unique()) <= {"Unknown", "Normal"}


def test_price_spike_raises_disruption_score():
    closes = [100.0] * 80 + [130.0]
    out = compute_disruption(_make_df(closes))

    spike_score = out["disruption_score"].iloc[-1]
    baseline_score = out["disruption_score"].iloc[70]
    assert spike_score > baseline_score


def test_output_preserves_input_columns_and_length():
    closes = list(range(100, 170))
    df = _make_df(closes)
    out = compute_disruption(df)

    assert len(out) == len(df)
    assert "Close" in out.columns
    for col in ("return_zscore", "vol_ratio", "bb_breakout", "disruption_score", "disruption_label"):
        assert col in out.columns
