"""Interactive dashboard: forex/commodity disruption detection + ML price-movement forecasts.

Run with:  streamlit run app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from assets import ALL_ASSETS, COMMODITIES, FOREX
from data import fetch_history
from disruption import compute_disruption
from model import strategy_backtest, train_and_evaluate

st.set_page_config(page_title="Market Disruption & Price Movement Predictor", layout="wide")

DISRUPTION_COLORS = {"Normal": "#2ecc71", "Elevated": "#f39c12", "High Disruption": "#e74c3c", "Unknown": "#95a5a6"}

st.title("📉 Market Disruption & Price Movement Predictor")
st.caption(
    "Research/educational tool for forex & commodities. Combines a rules-based volatility/anomaly "
    "score with a gradient-boosted ML model trained on historical technical indicators. "
    "**Not financial advice** — predictions are probabilistic and can be wrong."
)

with st.sidebar:
    st.header("Settings")
    asset_group = st.radio("Asset class", ["Forex", "Commodities"], horizontal=True)
    options = FOREX if asset_group == "Forex" else COMMODITIES
    asset_name = st.selectbox("Instrument", list(options.keys()))
    ticker = options[asset_name]
    period = st.select_slider("History window", options=["1y", "2y", "5y", "10y"], value="5y")
    test_fraction = st.slider("Test set size (walk-forward)", 0.1, 0.4, 0.2, 0.05)
    run_button = st.button("Fetch data & run analysis", type="primary", use_container_width=True)

if "last_run" not in st.session_state:
    st.session_state.last_run = None

if run_button:
    st.session_state.last_run = (ticker, asset_name, period, test_fraction)

if st.session_state.last_run is None:
    st.info("Choose an instrument in the sidebar and click **Fetch data & run analysis** to begin.")
    st.stop()

ticker, asset_name, period, test_fraction = st.session_state.last_run

with st.spinner(f"Fetching {asset_name} ({ticker}) history..."):
    try:
        raw_df = fetch_history(ticker, period=period)
    except Exception as e:
        st.error(f"Failed to fetch data for {ticker}: {e}")
        st.stop()

disruption_df = compute_disruption(raw_df)

with st.spinner("Training model (walk-forward split, no look-ahead bias)..."):
    try:
        result = train_and_evaluate(raw_df, test_fraction=test_fraction)
    except ValueError as e:
        st.error(str(e))
        st.stop()

latest = disruption_df.iloc[-1]
pred = result.latest_prediction

# --- Top summary row ---------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Last close ({asset_name})", f"{latest['Close']:.4f}",
          f"{latest['Close'] - disruption_df.iloc[-2]['Close']:.4f}")

label = latest["disruption_label"]
c2.markdown(
    f"**Disruption status**<br>"
    f"<span style='font-size:1.6em;color:{DISRUPTION_COLORS[label]}'>{label}</span> "
    f"({latest['disruption_score']:.0f}/100)",
    unsafe_allow_html=True,
)

if pred.get("available"):
    dir_color = "#2ecc71" if pred["direction"] == "Up" else "#e74c3c"
    c3.markdown(
        f"**Next-period prediction**<br>"
        f"<span style='font-size:1.6em;color:{dir_color}'>{pred['direction']}</span> "
        f"({pred['confidence']*100:.0f}% confidence)",
        unsafe_allow_html=True,
    )
    c4.metric("Predicted return", f"{pred['predicted_return']*100:.2f}%")
else:
    c3.warning("Prediction unavailable (insufficient recent data)")

st.divider()

# --- Price chart with Bollinger bands + disruption markers --------------
st.subheader("Price history, volatility & disruption events")

plot_df = disruption_df.tail(500)
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05,
    subplot_titles=("Price", "Disruption score"),
)

fig.add_trace(go.Candlestick(
    x=plot_df.index, open=plot_df["Open"], high=plot_df["High"], low=plot_df["Low"], close=plot_df["Close"],
    name="Price", showlegend=False,
), row=1, col=1)

high_disruption = plot_df[plot_df["disruption_label"] == "High Disruption"]
if not high_disruption.empty:
    fig.add_trace(go.Scatter(
        x=high_disruption.index, y=high_disruption["Close"], mode="markers",
        marker=dict(color="#e74c3c", size=9, symbol="x"), name="High disruption",
    ), row=1, col=1)

fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df["disruption_score"], mode="lines", fill="tozeroy",
    line=dict(color="#e67e22"), name="Disruption score", showlegend=False,
), row=2, col=1)
fig.add_hline(y=60, line_dash="dot", line_color="#e74c3c", row=2, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="#f39c12", row=2, col=1)

fig.update_layout(height=650, xaxis_rangeslider_visible=False, margin=dict(t=40, b=20))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Model performance ---------------------------------------------------
st.subheader("Model performance (out-of-sample, walk-forward)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Direction accuracy", f"{result.test_accuracy*100:.1f}%",
          f"{(result.test_accuracy - result.baseline_accuracy)*100:+.1f}pp vs baseline")
m2.metric("ROC AUC", f"{result.test_auc:.3f}" if result.test_auc == result.test_auc else "n/a")
m3.metric("Return MAE", f"{result.test_mae*100:.3f}%")
m4.metric("Naive baseline acc.", f"{result.baseline_accuracy*100:.1f}%")

st.caption(
    "Baseline = always predicting the majority class in the test period. "
    "Accuracy modestly above baseline is expected for liquid FX/commodities; "
    "large outperformance claims should be treated with suspicion (possible leakage/overfitting)."
)

bt = strategy_backtest(result.test_df)
bt_fig = go.Figure()
bt_fig.add_trace(go.Scatter(x=bt.index, y=bt["strategy_cum_return"] * 100, name="Model-driven strategy"))
bt_fig.add_trace(go.Scatter(x=bt.index, y=bt["buy_hold_cum_return"] * 100, name="Buy & hold"))
bt_fig.update_layout(
    title="Backtest: cumulative return over test period (%, illustrative only)",
    height=350, margin=dict(t=40, b=20), yaxis_title="Cumulative return (%)",
)
st.plotly_chart(bt_fig, use_container_width=True)

with st.expander("Feature importance (direction model)"):
    st.bar_chart(result.feature_importances.head(12))

with st.expander("Recent disruption events (last 30 rows)"):
    st.dataframe(
        disruption_df.tail(30)[["Close", "return_zscore", "vol_ratio", "bb_breakout", "disruption_score", "disruption_label"]]
        .style.format({"Close": "{:.4f}", "return_zscore": "{:.2f}", "vol_ratio": "{:.2f}", "bb_breakout": "{:.2f}", "disruption_score": "{:.0f}"})
    )

st.divider()
st.caption(
    "Data source: Yahoo Finance via yfinance. This tool is for research/educational purposes only "
    "and does not constitute investment advice. Past performance and backtests are not indicative "
    "of future results."
)
