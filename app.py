"""
Bitcoin Price Forecasting Portal - Streamlit App.

Main entry point. Wires together data loading, model selection,
forecasting, and visualization.

Based on Yenidogan et al. (2023) "Comparative Analysis of ARIMA and
Prophet Algorithms in Bitcoin Price Forecasting."
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.data_loader import load_btc_csv, DataLoadError
from src.preprocessing import resample_granularity, train_test_split
from src.models.prophet_model import train_prophet_auto, make_forecast as make_prophet_forecast
from src.models.arima_model import train_arima, make_arima_forecast
from src.models.xgboost_model import train_xgboost, make_xgboost_forecast
from src.evaluation import compute_metrics

# ── Must be the very first Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="BTC Forecasting Portal",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS injection ───────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Hide default Streamlit chrome ---- */
#MainMenu, footer { visibility: hidden; }

/* ---- Root font & background ---- */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}

/* ---- Sidebar styling ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0a0e1a 100%);
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* ---- Sidebar header brand ---- */
.sidebar-brand {
    text-align: center;
    padding: 12px 0 20px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 16px;
}
.sidebar-brand .btc-icon {
    font-size: 40px;
    display: block;
    margin-bottom: 4px;
}
.sidebar-brand .brand-title {
    font-size: 15px;
    font-weight: 800;
    color: #f7931a;
    letter-spacing: 0.5px;
}
.sidebar-brand .brand-sub {
    font-size: 10px;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ---- Section labels in sidebar ---- */
.sidebar-section {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #475569;
    padding: 8px 0 4px;
}

/* ---- Metric cards ---- */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #111827, #0d1117);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px 20px;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover { border-color: #f7931a44; }
[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #64748b !important;
}
[data-testid="stMetricValue"] {
    font-size: 22px !important;
    font-weight: 800 !important;
    color: #f1f5f9 !important;
}
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* ---- Main page header ---- */
.page-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 0 24px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 24px;
}
.page-header-left h1 {
    font-size: 26px;
    font-weight: 800;
    color: #f1f5f9;
    margin: 0 0 4px;
}
.page-header-left p {
    font-size: 13px;
    color: #64748b;
    margin: 0;
}
.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #15803d22;
    border: 1px solid #15803d55;
    color: #4ade80;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    padding: 4px 12px;
}
.live-dot {
    width: 7px; height: 7px;
    background: #4ade80;
    border-radius: 50%;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ---- Welcome screen ---- */
.welcome-card {
    background: linear-gradient(135deg, #0f172a, #0a0e1a);
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 48px;
    text-align: center;
    max-width: 680px;
    margin: 60px auto;
}
.welcome-card .btc-big { font-size: 64px; margin-bottom: 16px; }
.welcome-card h2 { font-size: 24px; font-weight: 800; color: #f1f5f9; margin-bottom: 8px; }
.welcome-card p { font-size: 14px; color: #64748b; line-height: 1.6; margin-bottom: 28px; }
.feature-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    text-align: left;
    margin-bottom: 28px;
}
.feature-item {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 12px;
    color: #94a3b8;
    display: flex;
    gap: 8px;
    align-items: flex-start;
}
.feature-item .fi-icon { font-size: 16px; flex-shrink: 0; }
.paper-badge {
    display: inline-block;
    background: #7c3aed22;
    border: 1px solid #7c3aed55;
    color: #a78bfa;
    border-radius: 20px;
    font-size: 11px;
    padding: 4px 14px;
}

/* ---- Chart container ---- */
.chart-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}
.chart-title {
    font-size: 15px;
    font-weight: 700;
    color: #f1f5f9;
}
.chart-subtitle { font-size: 12px; color: #64748b; margin-top: 2px; }

/* ---- Info/error banners ---- */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ---- Expander ---- */
[data-testid="stExpander"] {
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    background: #0d1117 !important;
}

/* ---- Button ---- */
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #f7931a, #e8820a) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    height: 46px !important;
    transition: opacity 0.2s !important;
}
[data-testid="baseButton-primary"]:hover { opacity: 0.88 !important; }

/* ---- Dividers ---- */
hr { border-color: #1e293b !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:

    # Brand header
    st.markdown("""
    <div class="sidebar-brand">
        <span class="btc-icon">₿</span>
        <div class="brand-title">BTC Forecasting Portal</div>
        <div class="brand-sub">Time-Series Analysis Engine</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 1: Data ──────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">① Data Source</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Bitcoin CSV",
        type=["csv"],
        help="Kaggle-style BTC historical data (Date + OHLCV columns)",
        label_visibility="collapsed",
    )
    price_column = st.selectbox(
        "Price column",
        options=["Close", "Open", "High", "Low"],
        help="Which OHLC price to forecast",
    )

    start_year = st.slider(
        "Train from year",
        min_value=2012, max_value=2024, value=2020,
        help="Skip older data — pre-2020 BTC was a different market (no ETFs, no institutions)",
    )

    st.divider()

    # ── Section 2: Model ─────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">② Model Selection</div>', unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Algorithm",
        options=["Prophet", "ARIMA", "XGBoost"],
        help="Prophet wins on daily data, ARIMA on monthly (Yenidogan et al. 2023)",
    )

    granularity = st.radio(
        "Granularity",
        options=["Daily", "Weekly", "Monthly"],
        horizontal=True,
        help="Matches paper's multi-granularity experiment",
    )

    st.divider()

    # ── Section 3: Forecast ──────────────────────────────────────────
    st.markdown('<div class="sidebar-section">③ Forecast Settings</div>', unsafe_allow_html=True)

    horizon = st.slider(
        "Horizon (days)",
        min_value=7, max_value=90, value=30,
        help="How many days ahead to forecast",
    )

    confidence = st.select_slider(
        "Confidence interval",
        options=[0.80, 0.85, 0.90, 0.95],
        value=0.95,
        format_func=lambda x: f"{int(x * 100)}%",
    )

    st.divider()

    # ── Section 4: Indicators ────────────────────────────────────────
    st.markdown('<div class="sidebar-section">④ Technical Indicators</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        show_sma_20 = st.checkbox("SMA 20")
        show_sma_200 = st.checkbox("SMA 200")
    with col_b:
        show_sma_50 = st.checkbox("SMA 50")
        show_ema_20 = st.checkbox("EMA 20")

    st.divider()
    generate = st.button("⚡ Generate Forecast", type="primary", use_container_width=True)

    # Footer
    st.markdown("""
    <div style="text-align:center;margin-top:24px;font-size:10px;color:#334155;line-height:1.6;">
        Based on Yenidogan et al. (2023)<br>
        ARIMA vs Prophet · BTC Forecasting<br>
        <span style="color:#f7931a;">ITI · AI Engineering</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════════════

# Page header bar
st.markdown("""
<div class="page-header">
    <div class="page-header-left">
        <h1>₿ Bitcoin Price Forecasting Portal</h1>
        <p>Comparative analysis of ARIMA, Prophet & XGBoost · Yenidogan et al. (2023)</p>
    </div>
    <div class="live-badge">
        <span class="live-dot"></span> LIVE
    </div>
</div>
""", unsafe_allow_html=True)

# ── Welcome screen (no file uploaded) ─────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div class="welcome-card">
        <div class="btc-big">₿</div>
        <h2>Welcome to the BTC Forecasting Portal</h2>
        <p>Upload a Kaggle-style Bitcoin CSV to begin. The engine will automatically
        detect your date and price columns, clean the data, and generate an
        interactive price forecast.</p>
        <div class="feature-grid">
            <div class="feature-item"><span class="fi-icon">📈</span>3 algorithms: Prophet, ARIMA, XGBoost</div>
            <div class="feature-item"><span class="fi-icon">📊</span>Daily · Weekly · Monthly granularity</div>
            <div class="feature-item"><span class="fi-icon">🎯</span>MAE, RMSE, MAPE, R² metrics</div>
            <div class="feature-item"><span class="fi-icon">🔁</span>80/20 chronological train/test split</div>
            <div class="feature-item"><span class="fi-icon">📉</span>Confidence intervals (80% – 95%)</div>
            <div class="feature-item"><span class="fi-icon">💾</span>Export forecast as CSV or PNG</div>
        </div>
        <div class="paper-badge">📄 Based on Yenidogan et al. (2023) · ScienceDirect</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load & validate CSV ────────────────────────────────────────────────
try:
    with st.spinner("Parsing and validating dataset..."):
        df = load_btc_csv(uploaded_file, price_column=price_column)
except DataLoadError as e:
    st.error(f"**Data Error:** {e}")
    st.info(
        "Expected: a CSV with a date column (Date / Timestamp) "
        "and price columns (Close / Open / High / Low)."
    )
    st.stop()
except Exception as e:
    st.error(f"**Unexpected error:** {e}")
    st.stop()

# Apply the user's date filter (recency-weighted training)
df = df[df["ds"].dt.year >= start_year].reset_index(drop=True)
if len(df) < 100:
    st.warning(f"Only {len(df)} rows after filtering from {start_year}. Forecasts may be unreliable.")

# ── Summary metric row ─────────────────────────────────────────────────
latest = df["y"].iloc[-1]
prev   = df["y"].iloc[-2]
ath    = df["y"].max()
atl    = df["y"].min()
delta  = latest - prev

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Latest Price",    f"${latest:,.2f}",  f"${delta:+,.2f} vs prev day")
c2.metric("All-Time High",   f"${ath:,.2f}")
c3.metric("All-Time Low",    f"${atl:,.2f}")
c4.metric("Data Points",     f"{len(df):,}")
c5.metric("Date Range",
          f"{(df['ds'].max() - df['ds'].min()).days // 365}y",
          f"{df['ds'].min().year} – {df['ds'].max().year}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Historical price chart ─────────────────────────────────────────────
st.markdown("""
<div class="chart-header">
    <div>
        <div class="chart-title">Bitcoin Historical Price</div>
        <div class="chart-subtitle">USD · Log scale available · Hover for details</div>
    </div>
</div>
""", unsafe_allow_html=True)

fig = go.Figure()

# Main price line
fig.add_trace(go.Scatter(
    x=df["ds"], y=df["y"],
    mode="lines",
    name="BTC/USD",
    line=dict(color="#f7931a", width=2),
    fill="tozeroy",
    fillcolor="rgba(247,147,26,0.06)",
    hovertemplate="<b>%{x|%b %d, %Y}</b><br>Price: $%{y:,.2f}<extra></extra>",
))

# Moving averages
sma_colors = {"SMA 20": "#38bdf8", "SMA 50": "#a78bfa", "SMA 200": "#4ade80", "EMA 20": "#fb7185"}
if show_sma_20:
    fig.add_trace(go.Scatter(x=df["ds"], y=df["y"].rolling(20).mean(),
        mode="lines", name="SMA 20", line=dict(color=sma_colors["SMA 20"], width=1.5, dash="dot")))
if show_sma_50:
    fig.add_trace(go.Scatter(x=df["ds"], y=df["y"].rolling(50).mean(),
        mode="lines", name="SMA 50", line=dict(color=sma_colors["SMA 50"], width=1.5, dash="dot")))
if show_sma_200:
    fig.add_trace(go.Scatter(x=df["ds"], y=df["y"].rolling(200).mean(),
        mode="lines", name="SMA 200", line=dict(color=sma_colors["SMA 200"], width=1.5, dash="dash")))
if show_ema_20:
    fig.add_trace(go.Scatter(x=df["ds"], y=df["y"].ewm(span=20).mean(),
        mode="lines", name="EMA 20", line=dict(color=sma_colors["EMA 20"], width=1.5, dash="dot")))

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,14,26,0.8)",
    height=480,
    margin=dict(l=10, r=10, t=20, b=10),
    xaxis=dict(
        title="",
        showgrid=True, gridcolor="#1e293b", gridwidth=1,
        rangeslider=dict(visible=True, bgcolor="#0d1117", thickness=0.04),
        rangeselector=dict(
            bgcolor="#111827", activecolor="#f7931a",
            buttons=[
                dict(count=1,  label="1M",  step="month", stepmode="backward"),
                dict(count=6,  label="6M",  step="month", stepmode="backward"),
                dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
                dict(count=3,  label="3Y",  step="year",  stepmode="backward"),
                dict(step="all", label="All"),
            ],
        ),
    ),
    yaxis=dict(
        title="Price (USD)",
        showgrid=True, gridcolor="#1e293b", gridwidth=1,
        tickprefix="$", tickformat=",.0f",
    ),
    legend=dict(
        bgcolor="rgba(13,17,23,0.8)", bordercolor="#1e293b",
        borderwidth=1, font=dict(size=11),
    ),
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

# ── Raw data expander ──────────────────────────────────────────────────
with st.expander("📋 Preview Raw Data"):
    tab1, tab2 = st.tabs(["Last 30 rows", "Statistics"])
    with tab1:
        st.dataframe(
            df.tail(30).sort_values("ds", ascending=False).reset_index(drop=True),
            use_container_width=True,
        )
    with tab2:
        st.dataframe(df["y"].describe().rename("BTC Price (USD)").to_frame(), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# FORECAST GENERATION
# ══════════════════════════════════════════════════════════════════════
if generate:
    st.markdown("---")
    st.subheader(f"🔮 {model_choice} Forecast — {granularity} Granularity")

    # Step 1: Resample to chosen granularity (Daily / Weekly / Monthly)
    df_resampled = resample_granularity(df, granularity)

    # Step 2: Chronological 80/20 split (paper Section 3.3)
    try:
        train, test = train_test_split(df_resampled)
    except ValueError as e:
        st.error(
            f"⚠️ **Not enough data for {granularity} forecasting:** {e}. "
            f"Try a wider date range (lower the 'Train from year' slider) "
            f"or pick **Daily** granularity."
        )
        st.stop()

    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
    total_periods = len(test) + horizon
    arima_order = None  # Only set for ARIMA — used in info banner below

    # Step 3 & 4: Train chosen model + generate forecast
    if model_choice == "Prophet":
        with st.spinner(f"Training Prophet on {len(train):,} {granularity.lower()} records..."):
            model, train_time = train_prophet_auto(train, confidence=confidence)
            forecast = make_prophet_forecast(
                model,
                horizon_days=total_periods,
                freq=freq_map[granularity],
                historical_max=df_resampled["y"].max(),
            )

    elif model_choice == "ARIMA":
        with st.spinner(f"Training ARIMA (auto-selecting p, d, q) — this may take 30-60s..."):
            model, train_time, arima_order = train_arima(train)
            forecast = make_arima_forecast(
                model, train, horizon=total_periods, confidence=confidence,
            )

    elif model_choice == "XGBoost":
        with st.spinner(f"Training XGBoost on {len(train):,} {granularity.lower()} records..."):
            model, train_time = train_xgboost(train)
            forecast = make_xgboost_forecast(model, train, horizon=total_periods)

    # Show model-specific info
    if arima_order is not None:
        st.caption(f"📐 Auto-selected ARIMA order (p, d, q) = **{arima_order}**")

    # Step 5: Compute metrics on the test set (backtesting)
    test_forecast = forecast[forecast["ds"].isin(test["ds"])].set_index("ds")
    test_actual = test.set_index("ds")
    common_dates = test_actual.index.intersection(test_forecast.index)
    metrics = compute_metrics(test_actual.loc[common_dates, "y"], test_forecast.loc[common_dates, "yhat"])

    # ── Metric cards ──────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("MAE",  f"${metrics['MAE']:,.0f}",  help="Mean Absolute Error in USD")
    m2.metric("RMSE", f"${metrics['RMSE']:,.0f}", help="Root Mean Squared Error in USD")
    m3.metric("MAPE", f"{metrics['MAPE']:.2f}%",  help="Mean Absolute Percentage Error")
    m4.metric("R²",   f"{metrics['R2']:.3f}",     help="Variance explained (1.0 = perfect)")
    m5.metric("Train Time", f"{train_time:.1f}s", help="Model training duration")

    # ── Forecast chart ────────────────────────────────────────────────
    forecast_start_date = test["ds"].iloc[0]
    fig_fc = go.Figure()

    # Historical actual (gold)
    fig_fc.add_trace(go.Scatter(
        x=df_resampled["ds"], y=df_resampled["y"],
        mode="lines", name="Historical",
        line=dict(color="#f7931a", width=2),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Actual: $%{y:,.2f}<extra></extra>",
    ))

    # Confidence band (filled area)
    fig_fc.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_upper"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_fc.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_lower"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(56,189,248,0.15)",
        name=f"{int(confidence*100)}% Confidence",
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Lower: $%{y:,.2f}<extra></extra>",
    ))

    # Forecast line (blue)
    fig_fc.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"],
        mode="lines", name="Forecast",
        line=dict(color="#38bdf8", width=2.5, dash="solid"),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Predicted: $%{y:,.2f}<extra></extra>",
    ))

    # Vertical line marking forecast start (use shape + annotation to avoid add_vline date bug)
    fig_fc.add_shape(
        type="line",
        x0=forecast_start_date, x1=forecast_start_date,
        y0=0, y1=1, yref="paper",
        line=dict(color="#a78bfa", width=2, dash="dash"),
    )
    fig_fc.add_annotation(
        x=forecast_start_date, y=1, yref="paper",
        text="Forecast Start", showarrow=False,
        font=dict(color="#a78bfa", size=11),
        bgcolor="rgba(13,17,23,0.8)",
        yshift=10,
    )

    fig_fc.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,26,0.8)",
        height=520,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(showgrid=True, gridcolor="#1e293b"),
        yaxis=dict(
            title="Price (USD)", showgrid=True, gridcolor="#1e293b",
            tickprefix="$", tickformat=",.0f",
            range=[0, df_resampled["y"].max() * 1.5],
        ),
        legend=dict(bgcolor="rgba(13,17,23,0.8)", bordercolor="#1e293b", borderwidth=1),
        hovermode="x unified",
    )

    st.plotly_chart(fig_fc, use_container_width=True)

    # ── Forecast table ────────────────────────────────────────────────
    with st.expander(f"📊 View Forecast Data ({horizon} future periods)"):
        future_only = forecast[forecast["ds"] > df_resampled["ds"].max()].copy()
        future_only.columns = ["Date", "Predicted Price", "Lower Bound", "Upper Bound"]
        for col in ["Predicted Price", "Lower Bound", "Upper Bound"]:
            future_only[col] = future_only[col].apply(lambda x: f"${x:,.2f}")
        st.dataframe(future_only.head(horizon), use_container_width=True, hide_index=True)
