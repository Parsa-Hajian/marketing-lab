"""Forecasting models for the Monitor dashboard.

Seven model implementations for monthly time-series forecasting:
    1. SARIMAX — Seasonal ARIMA with exogenous variables
    2. Local Linear Trend — Unobserved components model with exogenous variables
    3. Neural Network — MLPRegressor with lagged features and exogenous variables
    4. NNS.VAR — Re-exported from engine.risk (pure numpy)
    5. Random Forest — Ensemble of decision trees with lagged + exogenous features
    6. Gradient Boosting — Sequential ensemble with residual learning
    7. Decision Tree — Single interpretable tree with lagged + exogenous features

All models operate on monthly aggregated data with columns [Date, value].
"""
import numpy as np
import pandas as pd
import warnings


# ── Exogenous feature builder ────────────────────────────────────────────────

def _build_exog(dates):
    """Build exogenous feature matrix from a DatetimeIndex or Series of dates.

    Features:
        month_sin, month_cos — circular encoding of month
        trend — linear trend (0, 1, 2, ...)
        year_idx — normalized year index
    """
    if isinstance(dates, pd.Series):
        dates = pd.DatetimeIndex(dates)
    months = dates.month
    n = len(dates)
    years = dates.year
    min_year = years.min() if n > 0 else 2020
    max_year = years.max() if n > 0 else 2020
    yr_range = max(max_year - min_year, 1)

    exog = pd.DataFrame({
        "month_sin": np.sin(2 * np.pi * months / 12),
        "month_cos": np.cos(2 * np.pi * months / 12),
        "trend": np.arange(n, dtype=float),
        "year_idx": (years - min_year) / yr_range,
    }, index=dates)
    return exog


# ── Model 1: SARIMAX ────────────────────────────────────────────────────────

def forecast_sarimax(monthly_df, n_ahead):
    """SARIMAX(1,1,1)(1,1,1,12) forecast with exogenous variables.

    Falls back to simpler orders if fitting fails.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = monthly_df["value"].values.astype(float)
    dates = pd.DatetimeIndex(monthly_df["Date"])
    exog = _build_exog(dates)

    # Generate future dates and exog
    last_date = dates[-1]
    future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=n_ahead, freq="MS")
    exog_future = _build_exog(future_dates)
    # Extend trend to continue from training
    exog_future["trend"] = np.arange(len(y), len(y) + n_ahead, dtype=float)
    min_year = dates.year.min()
    yr_range = max(dates.year.max() - min_year, 1)
    exog_future["year_idx"] = (future_dates.year - min_year) / yr_range

    # Try progressively simpler models
    configs = [
        ((1, 1, 1), (1, 1, 1, 12)),
        ((1, 1, 0), (1, 0, 0, 12)),
        ((1, 0, 0), (0, 0, 0, 0)),
    ]

    for order, seasonal_order in configs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    y, exog=exog.values, order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False, maxiter=200)
                fc = result.forecast(steps=n_ahead, exog=exog_future.values)
                return np.maximum(0, fc)
        except Exception:
            continue

    # Last resort: naive forecast
    return np.full(n_ahead, max(0, float(np.mean(y[-6:]))))


# ── Model 2: Local Linear Trend ─────────────────────────────────────────────

def forecast_local_linear_trend(monthly_df, n_ahead):
    """Local Linear Trend model via UnobservedComponents with exogenous variables."""
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    y = monthly_df["value"].values.astype(float)
    dates = pd.DatetimeIndex(monthly_df["Date"])
    exog = _build_exog(dates)

    # Future exogenous
    last_date = dates[-1]
    future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=n_ahead, freq="MS")
    exog_future = _build_exog(future_dates)
    exog_future["trend"] = np.arange(len(y), len(y) + n_ahead, dtype=float)
    min_year = dates.year.min()
    yr_range = max(dates.year.max() - min_year, 1)
    exog_future["year_idx"] = (future_dates.year - min_year) / yr_range

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = UnobservedComponents(
                y, level="local linear trend", exog=exog.values,
            )
            result = model.fit(disp=False, maxiter=200)
            fc = result.forecast(steps=n_ahead, exog=exog_future.values)
            return np.maximum(0, fc)
    except Exception:
        pass

    # Fallback without exog
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = UnobservedComponents(y, level="local linear trend")
            result = model.fit(disp=False, maxiter=200)
            fc = result.forecast(steps=n_ahead)
            return np.maximum(0, fc)
    except Exception:
        return np.full(n_ahead, max(0, float(np.mean(y[-6:]))))


# ── Model 3: Neural Network ─────────────────────────────────────────────────

def forecast_neural_network(monthly_df, n_ahead):
    """Neural network (MLPRegressor) with lagged features and exogenous variables.

    Uses 3 lags + exogenous features. Iterative multi-step forecast.
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    y = monthly_df["value"].values.astype(float)
    dates = pd.DatetimeIndex(monthly_df["Date"])
    n = len(y)
    n_lags = 3

    if n <= n_lags + 1:
        return np.full(n_ahead, max(0, float(np.mean(y))))

    # Build training features: [lag1, lag2, lag3, month_sin, month_cos, trend]
    exog = _build_exog(dates)
    X_rows = []
    y_rows = []
    for i in range(n_lags, n):
        lags = [y[i - j] for j in range(1, n_lags + 1)]
        exog_row = exog.iloc[i].values.tolist()
        X_rows.append(lags + exog_row)
        y_rows.append(y[i])

    X_train = np.array(X_rows)
    y_train = np.array(y_rows)

    # Scale features
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_train)
    y_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15 if len(X_train) > 10 else 0.0,
            )
            # If too few samples for validation, disable early stopping
            if len(X_train) <= 10:
                model.set_params(early_stopping=False)
            model.fit(X_scaled, y_scaled)
    except Exception:
        return np.full(n_ahead, max(0, float(np.mean(y[-6:]))))

    # Iterative multi-step forecast
    y_ext = list(y)
    last_date = dates[-1]
    forecasts = []
    for h in range(n_ahead):
        future_date = last_date + pd.DateOffset(months=h + 1)
        month_val = future_date.month
        trend_val = n + h
        min_year = dates.year.min()
        yr_range = max(dates.year.max() - min_year, 1)
        year_idx_val = (future_date.year - min_year) / yr_range

        lags = [y_ext[-(j + 1)] for j in range(n_lags)]
        exog_vals = [
            np.sin(2 * np.pi * month_val / 12),
            np.cos(2 * np.pi * month_val / 12),
            float(trend_val),
            year_idx_val,
        ]
        x_new = np.array([lags + exog_vals])
        x_new_scaled = scaler_x.transform(x_new)
        pred_scaled = model.predict(x_new_scaled)
        pred = float(scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0])
        pred = max(0, pred)
        forecasts.append(pred)
        y_ext.append(pred)

    return np.array(forecasts)


# ── Model 4: NNS.VAR (re-export from risk) ──────────────────────────────────

def forecast_nns_var(monthly_df, n_ahead):
    """NNS.VAR forecast — delegates to the implementation in engine.risk."""
    from engine.risk import _forecast_nns_var
    return _forecast_nns_var(monthly_df, n_ahead)


# ── Tree-based shared helper ─────────────────────────────────────────────────

def _tree_forecast(monthly_df, n_ahead, model_class, **kwargs):
    """Shared iterative forecasting logic for tree-based sklearn models."""
    y = monthly_df["value"].values.astype(float)
    dates = pd.DatetimeIndex(monthly_df["Date"])
    n = len(y)
    n_lags = 3

    if n <= n_lags + 1:
        return np.full(n_ahead, max(0, float(np.mean(y))))

    exog = _build_exog(dates)
    X_rows, y_rows = [], []
    for i in range(n_lags, n):
        lags = [y[i - j] for j in range(1, n_lags + 1)]
        X_rows.append(lags + exog.iloc[i].values.tolist())
        y_rows.append(y[i])

    X_train = np.array(X_rows)
    y_train = np.array(y_rows)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = model_class(**kwargs)
            model.fit(X_train, y_train)
    except Exception:
        return np.full(n_ahead, max(0, float(np.mean(y[-6:]))))

    y_ext = list(y)
    last_date = dates[-1]
    min_year = dates.year.min()
    yr_range = max(dates.year.max() - min_year, 1)
    forecasts = []
    for h in range(n_ahead):
        future_date = last_date + pd.DateOffset(months=h + 1)
        lags = [y_ext[-(j + 1)] for j in range(n_lags)]
        exog_vals = [
            np.sin(2 * np.pi * future_date.month / 12),
            np.cos(2 * np.pi * future_date.month / 12),
            float(n + h),
            (future_date.year - min_year) / yr_range,
        ]
        pred = max(0, float(model.predict(np.array([lags + exog_vals]))[0]))
        forecasts.append(pred)
        y_ext.append(pred)

    return np.array(forecasts)


# ── Model 5: Random Forest ──────────────────────────────────────────────────

def forecast_random_forest(monthly_df, n_ahead):
    """Random Forest with lagged features and exogenous variables."""
    from sklearn.ensemble import RandomForestRegressor
    return _tree_forecast(monthly_df, n_ahead, RandomForestRegressor,
                          n_estimators=100, random_state=42, max_depth=5)


# ── Model 6: Gradient Boosting ──────────────────────────────────────────────

def forecast_gradient_boosting(monthly_df, n_ahead):
    """Gradient Boosting with lagged features and exogenous variables."""
    from sklearn.ensemble import GradientBoostingRegressor
    return _tree_forecast(monthly_df, n_ahead, GradientBoostingRegressor,
                          n_estimators=100, random_state=42,
                          max_depth=3, learning_rate=0.1)


# ── Model 7: Decision Tree ──────────────────────────────────────────────────

def forecast_decision_tree(monthly_df, n_ahead):
    """Decision Tree with lagged features and exogenous variables."""
    from sklearn.tree import DecisionTreeRegressor
    return _tree_forecast(monthly_df, n_ahead, DecisionTreeRegressor,
                          random_state=42, max_depth=5)


# ── Model registry ──────────────────────────────────────────────────────────

MODELS = {
    "SARIMAX": forecast_sarimax,
    "Local Linear Trend": forecast_local_linear_trend,
    "Neural Network": forecast_neural_network,
    "NNS.VAR": forecast_nns_var,
    "Random Forest": forecast_random_forest,
    "Gradient Boosting": forecast_gradient_boosting,
    "Decision Tree": forecast_decision_tree,
}

MODEL_INFO = {
    "SARIMAX": {
        "strength": "Captures seasonal patterns & linear trends explicitly",
        "weakness": "Struggles with non-linear dynamics & structural breaks",
    },
    "Local Linear Trend": {
        "strength": "Decomposes trend & level shifts cleanly",
        "weakness": "Assumes linear local dynamics; weak with complex seasonality",
    },
    "Neural Network": {
        "strength": "Captures non-linear patterns & feature interactions",
        "weakness": "Prone to overfitting with limited data; less interpretable",
    },
    "NNS.VAR": {
        "strength": "Non-parametric; adapts to any distribution shape",
        "weakness": "Needs sufficient historical data; sensitive to noise",
    },
    "Random Forest": {
        "strength": "Robust to outliers; captures non-linear relationships well",
        "weakness": "Cannot extrapolate beyond historical training range",
    },
    "Gradient Boosting": {
        "strength": "Strong at learning residual patterns & heterogeneous data",
        "weakness": "Can overfit small datasets; sensitive to hyperparameters",
    },
    "Decision Tree": {
        "strength": "Fully interpretable; captures step-function patterns",
        "weakness": "High variance; poor generalization from limited data",
    },
}


def run_monitor_forecast(monthly_df, n_ahead, model_names=None):
    """Run selected models and return results dict.

    Parameters
    ----------
    monthly_df : DataFrame with [Date, value] — monthly aggregated metric.
    n_ahead : int — number of months to forecast.
    model_names : list[str] or None — models to run (default: all).

    Returns
    -------
    dict with keys:
        'forecasts' — {model_name: np.ndarray}
        'future_dates' — DatetimeIndex
        'scores' — {model_name: MAPE} from leave-last-3-out CV
    """
    if model_names is None:
        model_names = list(MODELS.keys())

    last_date = pd.Timestamp(monthly_df["Date"].iloc[-1])
    future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=n_ahead, freq="MS")

    # Cross-validation: leave-last-3-out
    n_test = min(3, len(monthly_df) // 3)
    train = monthly_df.iloc[:-n_test].copy() if n_test > 0 else monthly_df.copy()
    test = monthly_df.iloc[-n_test:].copy() if n_test > 0 else pd.DataFrame()
    test_y = test["value"].values if not test.empty else np.array([])

    forecasts = {}
    scores = {}

    for name in model_names:
        fn = MODELS.get(name)
        if fn is None:
            continue

        # Full forecast
        try:
            forecasts[name] = fn(monthly_df, n_ahead)
        except Exception:
            forecasts[name] = np.full(n_ahead, max(0, float(monthly_df["value"].mean())))

        # CV score
        if len(test_y) > 0:
            try:
                pred = fn(train, n_test)
                mape = float(np.mean(np.abs((test_y - pred) / np.maximum(test_y, 1.0))))
                scores[name] = mape
            except Exception:
                scores[name] = 1.0
        else:
            scores[name] = None

    return {
        "forecasts": forecasts,
        "future_dates": future_dates,
        "scores": scores,
    }
