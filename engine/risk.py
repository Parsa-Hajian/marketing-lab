"""Risk & Uncertainty estimation using a multi-model forecasting pipeline.

Uses multiple statistical models to generate month-over-month forecasts,
then computes risk bands from the ensemble of predictions. The DNA model
remains the primary forecast; these models provide uncertainty quantification.

Models included:
    - NNS.VAR (Non-linear Non-parametric Statistics — Fred Viole's method)
    - Exponential Smoothing (Holt-Winters additive)
    - Simple Moving Average (baseline)
    - Linear Trend + Seasonal (OLS)
    - Rolling Volatility (GARCH-lite)

All models are pure Python/NumPy/pandas — no external dependencies beyond
what's already in requirements.txt.
"""
import numpy as np
import pandas as pd
from datetime import timedelta


def _prepare_monthly(df_raw, sel_brands, metric="sales"):
    """Aggregate raw data to monthly totals for the selected brands."""
    mask = df_raw["brand"].isin(sel_brands)
    daily = (
        df_raw[mask]
        .groupby("Date")
        .agg({metric: "sum"})
        .sort_index()
        .reset_index()
    )
    daily.columns = ["Date", "value"]
    daily["YearMonth"] = daily["Date"].dt.to_period("M")
    monthly = daily.groupby("YearMonth")["value"].sum().reset_index()
    monthly["Date"] = monthly["YearMonth"].dt.to_timestamp()
    monthly = monthly.sort_values("Date").reset_index(drop=True)

    # Drop incomplete trailing month (< 15 days of data)
    if len(monthly) > 1:
        last_ym = monthly["YearMonth"].iloc[-1]
        n_days_last = daily[daily["YearMonth"] == last_ym]["Date"].nunique()
        if n_days_last < 15:
            monthly = monthly.iloc[:-1].reset_index(drop=True)

    return monthly[["Date", "value"]]


# ── NNS.VAR — Non-linear Non-parametric Statistics (Viole) ───────────────────
#
# Python implementation of Fred Viole's NNS.VAR forecasting method.
# Core idea: use partial-moment-based distance to find nearest neighbors
# in lagged embedding space, then average their "next" values as forecast.
#
# References:
#   Viole, F. (2020). "NNS: Nonlinear Nonparametric Statistics"
#   https://cran.r-project.org/package=NNS

def _partial_moment_distance(x, y, target_x, target_y):
    """Compute NNS-style partial moment distance between observation and target.

    Uses co-partial moments: how x,y jointly deviate from target in each
    quadrant (++, +−, −+, −−). This captures non-linear dependence that
    Euclidean distance misses.
    """
    dx = x - target_x
    dy = y - target_y

    # Co-partial moments in each quadrant
    if dx >= 0 and dy >= 0:
        return dx * dy          # upper-right concordance
    elif dx < 0 and dy < 0:
        return abs(dx) * abs(dy)  # lower-left concordance
    else:
        return -(abs(dx) * abs(dy))  # discordance (penalized)


def _nns_var_step(y, lags, k=5):
    """One-step-ahead NNS.VAR forecast.

    Parameters
    ----------
    y : np.ndarray
        Time series values.
    lags : list[int]
        Lag orders to use for embedding (e.g., [1, 2, 3, 12]).
    k : int
        Number of nearest neighbors.

    Returns
    -------
    forecast : float
        One-step-ahead forecast.
    """
    n = len(y)
    max_lag = max(lags)

    if n <= max_lag + 1:
        return float(np.mean(y))

    # Build lagged embedding matrix
    # Each row i: [y[i - lag1], y[i - lag2], ..., y[i - lagN]] → target y[i]
    n_embed = n - max_lag
    X_embed = np.zeros((n_embed, len(lags)))
    y_embed = np.zeros(n_embed)

    for j, lag in enumerate(lags):
        for i in range(n_embed):
            X_embed[i, j] = y[max_lag + i - lag]
        y_embed = y[max_lag:]

    # Current point: the last observation's lagged features
    x_current = np.array([y[n - lag] for lag in lags])

    # Compute NNS partial-moment-based distance to each embedded point
    distances = np.zeros(n_embed)
    for i in range(n_embed):
        d = 0.0
        for j in range(len(lags)):
            d += abs(_partial_moment_distance(
                X_embed[i, j], y_embed[i],
                x_current[j], np.mean(y_embed),
            ))
        # Add small Euclidean component for tie-breaking
        d += 0.01 * np.sqrt(np.sum((X_embed[i] - x_current) ** 2))
        distances[i] = d

    # Find k nearest neighbors (smallest distance = most similar pattern)
    k_actual = min(k, n_embed)
    nn_idx = np.argsort(distances)[:k_actual]

    # Weight by inverse distance
    nn_dists = distances[nn_idx]
    nn_dists = np.maximum(nn_dists, 1e-10)  # avoid div-by-zero
    inv_w = 1.0 / nn_dists
    inv_w = inv_w / inv_w.sum()

    # Forecast = weighted average of neighbors' observed values
    forecast = float(np.sum(inv_w * y_embed[nn_idx]))
    return max(0, forecast)


def _forecast_nns_var(history, n_ahead, seasonal_period=12):
    """Multi-step NNS.VAR forecast using iterative one-step-ahead prediction.

    Uses multiple lag specifications and averages across them for robustness.
    """
    y = history["value"].values.astype(float)
    n = len(y)

    # Determine lag specifications based on available data length
    lag_specs = [[1, 2, 3]]  # short-term patterns
    if n >= 7:
        lag_specs.append([1, 3, 6])  # medium-term
    if n >= seasonal_period + 1:
        lag_specs.append([1, 2, seasonal_period])  # seasonal
    if n >= seasonal_period + 3:
        lag_specs.append([1, 3, 6, seasonal_period])  # full

    # Number of neighbors scales with data size
    k = max(3, min(n // 4, 10))

    # Iterative multi-step forecast
    forecasts = np.zeros(n_ahead)

    for spec in lag_specs:
        y_ext = y.copy().tolist()
        spec_fc = []
        for h in range(n_ahead):
            fc = _nns_var_step(np.array(y_ext), spec, k=k)
            spec_fc.append(fc)
            y_ext.append(fc)
        forecasts += np.array(spec_fc)

    forecasts /= len(lag_specs)
    return forecasts


# ── Model 1: Exponential Smoothing (Holt-Winters additive) ───────────────────

def _forecast_exp_smoothing(history, n_ahead, seasonal_period=12):
    """Simple additive Holt-Winters implementation."""
    y = history["value"].values.astype(float)
    n = len(y)

    if n < seasonal_period + 2:
        # Not enough data for seasonal model — fall back to simple exponential
        alpha = 0.3
        level = y[0]
        forecasts = []
        for i in range(1, n):
            level = alpha * y[i] + (1 - alpha) * level
        for _ in range(n_ahead):
            forecasts.append(level)
        return np.array(forecasts)

    # Initialize
    alpha, beta, gamma = 0.2, 0.1, 0.3
    sp = seasonal_period

    # Initial level: mean of first season
    level = np.mean(y[:sp])
    # Initial trend
    trend = (np.mean(y[sp:2*sp]) - np.mean(y[:sp])) / sp if n >= 2 * sp else 0
    # Initial seasonal factors
    seasonal = np.zeros(sp)
    for i in range(sp):
        seasonal[i] = y[i] - level

    # Fit
    for i in range(sp, n):
        s_idx = i % sp
        old_level = level
        level = alpha * (y[i] - seasonal[s_idx]) + (1 - alpha) * (level + trend)
        trend = beta * (level - old_level) + (1 - beta) * trend
        seasonal[s_idx] = gamma * (y[i] - level) + (1 - gamma) * seasonal[s_idx]

    # Forecast
    forecasts = []
    for h in range(1, n_ahead + 1):
        s_idx = (n + h - 1) % sp
        forecasts.append(max(0, level + h * trend + seasonal[s_idx]))

    return np.array(forecasts)


# ── Model 2: Simple Moving Average ───────────────────────────────────────────

def _forecast_sma(history, n_ahead, window=6):
    """Forecast using simple moving average of last `window` observations."""
    y = history["value"].values.astype(float)
    if len(y) < window:
        window = max(1, len(y))
    avg = np.mean(y[-window:])
    return np.full(n_ahead, max(0, avg))


# ── Model 3: Linear Trend + Seasonal ────────────────────────────────────────

def _forecast_linear_seasonal(history, n_ahead, seasonal_period=12):
    """OLS linear trend with seasonal dummies."""
    y = history["value"].values.astype(float)
    n = len(y)
    t = np.arange(n)

    # Build design matrix: intercept + trend + seasonal dummies
    sp = min(seasonal_period, n - 1)
    X = np.zeros((n, 2 + sp))
    X[:, 0] = 1        # intercept
    X[:, 1] = t         # trend
    for i in range(sp):
        X[np.arange(n) % sp == i, 2 + i] = 1

    # Remove last dummy to avoid multicollinearity
    X = X[:, :-1]

    # OLS
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.full(n_ahead, max(0, np.mean(y)))

    # Forecast
    forecasts = []
    for h in range(1, n_ahead + 1):
        t_new = n + h - 1
        x_new = np.zeros(X.shape[1])
        x_new[0] = 1          # intercept
        x_new[1] = t_new      # trend
        s_idx = t_new % sp
        if 2 + s_idx < X.shape[1]:
            x_new[2 + s_idx] = 1  # seasonal dummy
        forecasts.append(max(0, float(x_new @ beta)))

    return np.array(forecasts)


# ── Model 4: Rolling Volatility ──────────────────────────────────────────────

def _estimate_volatility(history, window=6):
    """Estimate rolling volatility (std of month-over-month changes)."""
    y = history["value"].values.astype(float)
    if len(y) < 3:
        return float(np.std(y)) if len(y) > 0 else 0.0
    changes = np.diff(y) / np.maximum(y[:-1], 1.0)
    if len(changes) < window:
        window = len(changes)
    return float(np.std(changes[-window:]) * np.mean(y[-window:]))


# ── Ensemble Pipeline ────────────────────────────────────────────────────────

_METRIC_COL_MAP = {
    "sales":    "Sales_Sim",
    "quantity":  "Qty_Sim",
    "clicks":   "Clicks_Sim",
}


def run_risk_pipeline(df_raw, sel_brands, df_projection, metric="sales"):
    """Run risk estimation that wraps uncertainty around the DNA forecast.

    Each statistical model produces an independent forecast for the projection
    period. The model divergence from the model mean is then applied as an
    offset to the DNA forecast, so the DNA forecast remains the center line
    and the models provide uncertainty bands.

    Parameters
    ----------
    df_raw : DataFrame
        Raw super_dataset.
    sel_brands : list[str]
        Brands to aggregate.
    df_projection : DataFrame
        DNA projection DataFrame with Date and Clicks_Sim/Qty_Sim/Sales_Sim.
    metric : str
        One of 'clicks', 'quantity', 'sales'.

    Returns
    -------
    result : dict or None
        Keys:
            'history' — monthly historical DataFrame
            'forecasts' — DataFrame with Date, dna_forecast, bands, model cols
            'model_scores' — dict of model name → MAPE
            'weights' — dict of model name → weight
            'risk_summary' — dict with risk label, spread, volatility, etc.
    """
    monthly = _prepare_monthly(df_raw, sel_brands, metric)

    if len(monthly) < 6:
        return None

    # ── Aggregate DNA projection to monthly ──────────────────────────────
    sim_col = _METRIC_COL_MAP.get(metric, "Sales_Sim")
    proj = df_projection[["Date", sim_col]].copy()
    proj["YearMonth"] = proj["Date"].dt.to_period("M")
    dna_monthly = (
        proj.groupby("YearMonth")[sim_col]
        .sum()
        .reset_index()
    )
    dna_monthly["Date"] = dna_monthly["YearMonth"].dt.to_timestamp()
    dna_monthly = dna_monthly.sort_values("Date").reset_index(drop=True)
    n_ahead = len(dna_monthly)
    dna_values = dna_monthly[sim_col].values.astype(float)

    if n_ahead == 0:
        return None

    # ── Score models via leave-last-3-out cross-validation ───────────────
    n_test = min(3, len(monthly) // 3)
    train  = monthly.iloc[:-n_test].copy()
    test   = monthly.iloc[-n_test:].copy()
    test_y = test["value"].values

    models = {
        "NNS.VAR":           _forecast_nns_var,
        "Exp. Smoothing":    _forecast_exp_smoothing,
        "Moving Average":    lambda h, n: _forecast_sma(h, n, window=6),
        "Linear + Seasonal": _forecast_linear_seasonal,
    }

    scores = {}
    for name, fn in models.items():
        try:
            pred = fn(train, n_test)
            mape = float(np.mean(np.abs((test_y - pred) / np.maximum(test_y, 1.0))))
            scores[name] = mape
        except Exception:
            scores[name] = 1.0

    sorted_models = sorted(scores.items(), key=lambda x: x[1])

    # ── Full forecasts using all historical data ─────────────────────────
    forecasts = {}
    for name, fn in models.items():
        try:
            forecasts[name] = fn(monthly, n_ahead)
        except Exception:
            forecasts[name] = np.full(n_ahead, monthly["value"].mean())

    # ── Compute weights (inverse MAPE) ───────────────────────────────────
    total_inv = sum(1.0 / max(s, 0.01) for _, s in sorted_models)
    weights = {name: (1.0 / max(score, 0.01)) / total_inv
               for name, score in sorted_models}

    # ── Wrap model divergence around DNA forecast ────────────────────────
    # Each model's offset from the model mean is applied to the DNA forecast
    all_fc = np.array(list(forecasts.values()))  # (n_models, n_ahead)
    model_mean = np.mean(all_fc, axis=0)

    # Adjusted values: DNA + (model - model_mean)
    adjusted = {}
    for name, fc in forecasts.items():
        delta = fc - model_mean
        adjusted[name] = np.maximum(0, dna_values + delta)

    adj_matrix = np.array(list(adjusted.values()))  # (n_models, n_ahead)

    # Percentile bands from adjusted values
    p10 = np.percentile(adj_matrix, 10, axis=0)
    p25 = np.percentile(adj_matrix, 25, axis=0)
    p50 = np.percentile(adj_matrix, 50, axis=0)
    p75 = np.percentile(adj_matrix, 75, axis=0)
    p90 = np.percentile(adj_matrix, 90, axis=0)

    # Add volatility-based widening
    vol = _estimate_volatility(monthly)
    vol_lower = np.maximum(0, dna_values - 1.65 * vol)
    vol_upper = dna_values + 1.65 * vol

    fc_dates = dna_monthly["Date"].values

    fc_df = pd.DataFrame({
        "Date": fc_dates,
        "dna_forecast": dna_values,
        "lower_10": np.minimum(p10, vol_lower),
        "lower_25": p25,
        "median": p50,
        "upper_75": p75,
        "upper_90": np.maximum(p90, vol_upper),
    })
    # Add individual adjusted model lines
    for name, adj_vals in adjusted.items():
        fc_df[name] = adj_vals

    # Risk summary
    spread_pct = float(np.mean((fc_df["upper_90"] - fc_df["lower_10"]) /
                               np.maximum(fc_df["dna_forecast"], 1.0)))
    risk_label = (
        "Low" if spread_pct < 0.3
        else "Medium" if spread_pct < 0.6
        else "High"
    )

    return {
        "history": monthly,
        "forecasts": fc_df,
        "model_scores": dict(sorted_models),
        "weights": weights,
        "risk_summary": {
            "spread_pct": spread_pct,
            "risk_label": risk_label,
            "volatility": float(vol),
            "best_model": sorted_models[0][0] if sorted_models else "N/A",
            "best_mape": sorted_models[0][1] if sorted_models else 1.0,
        },
    }
