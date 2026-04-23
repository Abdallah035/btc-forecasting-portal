<div align="center">

# ₿ Bitcoin Price Forecasting Portal

### Interactive Streamlit dashboard comparing ARIMA, Prophet, and XGBoost for cryptocurrency price forecasting

*Replicates and extends **Yenidogan et al. (2023)** — "Comparative Analysis of ARIMA and Prophet Algorithms in Bitcoin Price Forecasting"*

### [🚀 Launch Live Demo ↗](https://btc-forecasting-app-eiaig8h9pvydmwjryanx6x.streamlit.app/)

![Live Demo](https://img.shields.io/badge/demo-live-2ea043?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.39-ff4b4b.svg)
![Plotly](https://img.shields.io/badge/plotly-5.24-3f4f75.svg)
![Prophet](https://img.shields.io/badge/prophet-1.1.6-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

---

</div>

## Overview

An interactive web application for forecasting Bitcoin (BTC-USD) price trends using three different time-series algorithms. Users can upload any Kaggle-style BTC historical CSV, configure forecast parameters, and compare all three models side-by-side.

Built as a replication and extension of **Yenidogan et al. (2023)**, this project re-evaluates the paper's ARIMA-vs-Prophet comparison on a **14-year BTC-USD dataset** (vs. the paper's 2-year BTC-IDR) and adds **XGBoost** as a third algorithm for a richer comparison.

## Key Features

### Forecasting Engine
- **3 algorithms:** Prophet, ARIMA (auto-tuned with `pmdarima`), XGBoost (with engineered features)
- **Multi-granularity:** Daily · Weekly · Monthly aggregation — replicates paper Section 3.1
- **80/20 chronological split** — matches paper Section 3.3 methodology
- **Log transformation** for stable forecasts across Bitcoin's 600x price range ($200 → $120,000)
- **Recency-weighted training** — slider to train only on post-2020 data (post-ETF era)

### Evaluation — Paper's Full Metric Suite
- **MAE** (Mean Absolute Error) — in USD
- **RMSE** (Root Mean Squared Error) — in USD
- **MAPE** (Mean Absolute Percentage Error)
- **MSE** (Mean Squared Error)
- **R²** (Coefficient of Determination)
- **Processing Time** — as measured in paper Section 3.4

### Interactive Dashboard
- **Dark professional UI** with Bitcoin orange (`#f7931a`) accent
- **Interactive Plotly charts** with range selector, zoom, and tooltips
- **95% confidence intervals** visualized as shaded bands
- **Technical indicators:** SMA 20 / 50 / 200 and EMA 20 overlays
- **Model comparison view** — run all 3 algorithms in one click, see winners per metric
- **Side-by-side future predictions** with downloadable CSV

### Error Handling & UX
- Auto-detection of date columns (`Date`, `Timestamp`, `time`, etc.)
- Auto-detection of price columns (`Close`, `Open`, `High`, `Low`, `Weighted_Price`)
- Friendly error messages for malformed CSVs (empty, wrong columns, non-date values)
- Progress indicators for long-running operations (ARIMA training ~60s)
- Forecast caching via `st.session_state` — downloads don't retrigger training

---

## Try It Live

👉 **[btc-forecasting-app.streamlit.app](https://btc-forecasting-app-eiaig8h9pvydmwjryanx6x.streamlit.app/)**

No installation needed — upload a Kaggle BTC CSV directly in the browser.

## Run Locally

```bash
# Clone the repository
git clone https://github.com/Abdallah035/btc-forecasting-portal
cd btc-forecasting-portal

# Create a virtual environment
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Dataset

- **Source:** [Kaggle — Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)
- **Coverage:** 2012 – 2026 (auto-updated by dataset maintainer)
- **Columns:** `Timestamp`, `Open`, `High`, `Low`, `Close`, `Volume`
- A sample daily CSV (`data/sample_btc.csv`) is included for quick testing

---

## Architecture

```
btc-forecasting-portal/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Pinned dependencies
├── .streamlit/
│   └── config.toml            # Dark theme + 1GB upload limit
├── src/
│   ├── data_loader.py         # CSV ingestion, auto-detection, validation
│   ├── preprocessing.py       # Granularity resampling, 80/20 split, SMA/EMA
│   ├── evaluation.py          # MAE, MAPE, MSE, RMSE, R²
│   └── models/
│       ├── prophet_model.py   # Prophet with log transformation
│       ├── arima_model.py     # ARIMA + ADF test + pmdarima auto-tune
│       └── xgboost_model.py   # Lag features + rolling stats + recursive forecast
├── data/
│   └── sample_btc.csv         # Lightweight daily sample
└── README.md                  # You are here
```

---

## Methodology — How We Adapted the Paper

### What We Replicated

| Paper Feature | Paper Section | Our Implementation |
|---|---|---|
| 80/20 chronological split | 3.3 | `src/preprocessing.py::train_test_split` |
| Daily / Weekly / Monthly granularity | 3.1 | `src/preprocessing.py::resample_granularity` |
| 4 metrics (MAE, MAPE, MSE, RMSE) | 3.4 | `src/evaluation.py::compute_metrics` |
| Prophet Auto vs Manual tuning | 3.3.2 | `prophet_model.py::train_prophet_auto/manual` |
| ADF stationarity test | 3.3.1 | `arima_model.py::adf_test` |
| Auto-tuned ARIMA (p, d, q) | 3.3.1 | `pmdarima.auto_arima` (AIC criterion) |
| Processing time as metric | 3.4 | `time.time()` wrapper on all training |

### What We Extended (Adaptations for Our Dataset)

The paper used 2 years of BTC-IDR data (one bull cycle, ~$3k–$60k range). Our dataset is **14 years of BTC-USD** spanning $200 → $120,000 across multiple bull/bear cycles. Their parameters do not generalize. Our adaptations:

1. **Log transformation** — model `log(price)` instead of raw price. Reduces the effective growth from 600x to 6.4x, making the exponential trend linearly tractable.
2. **XGBoost as a third model** — paper compared only statistical models. We add gradient-boosted trees with lag features (7/14/30-day) and rolling statistics to capture non-linear patterns.
3. **Recency slider** — pre-2020 BTC had no institutional participants, no ETFs, and different market structure. Users can exclude older data to focus on the modern regime.
4. **Confidence band clipping** — ARIMA's uncertainty compounds exponentially over long horizons. We cap upper bounds at 3x historical max to prevent visual blow-up while keeping forecasts honest.

---

## Results — Honest Findings

On 14 years of BTC-USD data with the 2024 ETF rally in the test set:

| Model | MAE | RMSE | MAPE | R² | Train Time |
|---|---|---|---|---|---|
| **ARIMA** (post-2020) | **$16,540** | $19,245 | **16.9%** | **-0.28** | 4.9s |
| ARIMA (full history) | $27,044 | $47,680 | 35.0% | -1.91 | 69.6s |
| Prophet (full history) | $61,539 | $69,007 | 78.4% | -5.10 | 4.3s |

**Key finding:** On long, multi-cycle crypto data, **ARIMA significantly outperforms Prophet** — the opposite of the paper's conclusion. This aligns with the paper's own caveat: their findings held on 2 years of smooth data, not across multi-cycle regimes.

**Interpretation:** Prophet's changepoint detection overfits historical bubbles and extrapolates incorrect bearish trends when test data contains structural breaks not present in training. ARIMA's simpler autoregressive approach, while less sophisticated, proves more robust on BTC's fractured trend structure.

No model achieved positive R², illustrating the fundamental difficulty of forecasting BTC across regime changes. MAPE is the more meaningful metric here — **ARIMA's 17% MAPE on recent data is respectable for crypto forecasting**.

---

## How Each Model Handles Crypto Volatility

### Prophet (paper's winner on daily data)
Handles volatility through automatic **changepoint detection** — the trend can bend at arbitrary dates. Seasonality components (weekly, yearly) capture repeating patterns. Best suited to smooth-trending series; struggles when the test set contains structural breaks absent from training.

### ARIMA (paper's winner on monthly data)
Differencing (`d=1`) makes the series stationary by modeling price **changes** instead of prices. Autoregressive and moving-average terms capture short-term momentum. For long-horizon forecasts, ARIMA converges to a constant — a mathematically honest statement of uncertainty.

### XGBoost (our extension)
Learns non-linear rules from engineered features: 1/7/14/30-day price lags, rolling means and standard deviations, and calendar features (day-of-week, month, quarter). Forecasts recursively — each prediction feeds the next. Better suited to regime changes but requires careful feature engineering.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI Framework | Streamlit 1.39 |
| Visualization | Plotly 5.24 |
| Data | pandas 2.2, NumPy |
| Forecasting | Prophet, pmdarima, XGBoost, statsmodels |
| Evaluation | scikit-learn |
| Language | Python 3.10+ |

---

## Reference

> **Angelo, M. D., Fadhiilrahman, I., & Purnama, Y. (2023).**
> *Comparative Analysis of ARIMA and Prophet Algorithms in Bitcoin Price Forecasting.*
> Procedia Computer Science, 227, 490–499.
> Presented at the 8th International Conference on Computer Science and Computational Intelligence (ICCSCI 2023).
> [ScienceDirect — https://doi.org/10.1016/j.procs.2023.10.550](https://doi.org/10.1016/j.procs.2023.10.550)

---

## License

MIT License — see `LICENSE` file.

---

<div align="center">

Built as part of the **ITI AI Engineering curriculum** — Time Series Analysis & Forecasting module.

**Paper-backed · Rubric-focused · Portfolio-ready**

</div>
