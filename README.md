# Market Disruption & Price Movement Predictor

Research/educational dashboard for forex and commodities. Combines:

- **Disruption detection** (`disruption.py`): a rules-based 0-100 score from return
  outliers, short-vs-long volatility regime shifts, and Bollinger Band breakouts.
- **Price-movement prediction** (`model.py`): gradient-boosted models trained on
  technical indicators (`features.py`) to forecast next-period direction and
  return magnitude, evaluated with a strict time-ordered (walk-forward) split
  to avoid look-ahead bias.
- **Dashboard** (`app.py`): Streamlit UI with price/disruption charts, live
  predictions, model performance metrics, and a simple long/short backtest.

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

## Run

```bash
./venv/bin/streamlit run app.py
```

Then open the printed local URL, pick an instrument in the sidebar, and click
**Fetch data & run analysis**.

## Notes

- Data comes from Yahoo Finance via `yfinance`, cached locally for 6 hours
  (`cache/`) to avoid rate limits.
- This is **not financial advice**. Accuracy modestly above a naive baseline
  is expected/normal for liquid FX and commodities; treat large outperformance
  claims as a sign of overfitting or leakage, not edge.
