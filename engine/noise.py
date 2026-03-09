"""Daily/weekly noise estimation from historical brand data.

Uses historical residuals (actual − rolling trend) to estimate forecast
variability at daily and weekly resolution. The DNA model produces smooth
monthly indices; this module adds realistic daily/weekly variance bands.
"""
import numpy as np
import pandas as pd


def estimate_daily_noise(df_raw, sel_brands, metric="clicks", window=14):
    """Estimate daily noise statistics from historical brand data.

    Parameters
    ----------
    df_raw : DataFrame
        Raw super_dataset with columns: Date, brand, clicks, quantity, sales.
    sel_brands : list[str]
        Brands to aggregate.
    metric : str
        One of 'clicks', 'quantity', 'sales'.
    window : int
        Rolling window size for trend extraction (default 14 days).

    Returns
    -------
    noise_stats : dict
        Keys: 'daily_std', 'daily_cv', 'weekly_pattern', 'monthly_std',
              'residuals' (DataFrame with Date, actual, trend, residual).
    """
    mask = df_raw["brand"].isin(sel_brands)
    daily = (
        df_raw[mask]
        .groupby("Date")
        .agg({metric: "sum"})
        .sort_index()
        .reset_index()
    )

    if daily.empty or len(daily) < window * 2:
        return None

    daily.columns = ["Date", "actual"]
    daily["trend"] = daily["actual"].rolling(window, center=True, min_periods=window // 2).mean()
    daily["trend"] = daily["trend"].bfill().ffill()
    daily["residual"] = daily["actual"] - daily["trend"]
    daily["dow"] = daily["Date"].dt.dayofweek       # 0=Mon, 6=Sun
    daily["month"] = daily["Date"].dt.month

    # Day-of-week pattern: mean residual by weekday
    dow_pattern = daily.groupby("dow")["residual"].agg(["mean", "std"]).reset_index()
    dow_pattern.columns = ["dow", "dow_mean", "dow_std"]

    # Monthly noise: std of residuals by calendar month
    monthly_std = daily.groupby("month")["residual"].std().reset_index()
    monthly_std.columns = ["month", "noise_std"]

    # Overall
    overall_std = daily["residual"].std()
    overall_mean = daily["actual"].mean()
    cv = overall_std / overall_mean if overall_mean > 0 else 0

    return {
        "daily_std": float(overall_std),
        "daily_cv": float(cv),
        "dow_pattern": dow_pattern,
        "monthly_std": monthly_std,
        "residuals": daily[["Date", "actual", "trend", "residual", "dow", "month"]],
    }


def project_daily_with_noise(df_forecast, noise_stats, metric_base="Clicks_Base",
                              n_simulations=200, seed=42):
    """Project daily forecast with noise bands using Monte Carlo.

    Takes the smooth DNA-based forecast and adds realistic daily noise
    estimated from historical data.

    Parameters
    ----------
    df_forecast : DataFrame
        Projection DataFrame with Date and the metric_base column.
    noise_stats : dict
        Output from estimate_daily_noise().
    metric_base : str
        Column name in df_forecast to add noise to.
    n_simulations : int
        Number of Monte Carlo draws for band estimation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    result : DataFrame
        With columns: Date, forecast_smooth, forecast_p10, forecast_median,
        forecast_p90, daily_noise_std.
    """
    rng = np.random.default_rng(seed)

    dates = df_forecast["Date"].values
    smooth = df_forecast[metric_base].values.astype(float)
    n_days = len(smooth)

    # Build per-day noise std using month and day-of-week
    monthly_std = noise_stats["monthly_std"]
    dow_pattern = noise_stats["dow_pattern"]
    daily_cv    = noise_stats["daily_cv"]

    noise_std = np.zeros(n_days)
    for i, (d, s) in enumerate(zip(dates, smooth)):
        dt = pd.Timestamp(d)
        # Get monthly noise std
        m_row = monthly_std[monthly_std["month"] == dt.month]
        m_std = float(m_row["noise_std"].iloc[0]) if len(m_row) > 0 else noise_stats["daily_std"]
        # Scale noise proportionally to forecast level (CV approach)
        noise_std[i] = max(s * daily_cv, m_std * 0.5)

    # Monte Carlo simulation
    simulations = np.zeros((n_simulations, n_days))
    for sim in range(n_simulations):
        noise = rng.normal(0, noise_std)
        simulations[sim] = np.maximum(0, smooth + noise)

    # Percentile bands
    p10    = np.percentile(simulations, 10, axis=0)
    median = np.percentile(simulations, 50, axis=0)
    p90    = np.percentile(simulations, 90, axis=0)

    result = pd.DataFrame({
        "Date": dates,
        "forecast_smooth": smooth,
        "forecast_p10": p10,
        "forecast_median": median,
        "forecast_p90": p90,
        "daily_noise_std": noise_std,
    })
    return result


def aggregate_to_weekly(daily_result):
    """Aggregate daily noise projection to weekly resolution.

    Parameters
    ----------
    daily_result : DataFrame
        Output from project_daily_with_noise().

    Returns
    -------
    weekly : DataFrame
        Weekly aggregation with noise bands.
    """
    df = daily_result.copy()
    df["Week"] = pd.to_datetime(df["Date"]).dt.isocalendar().week.astype(int)
    df["Year"] = pd.to_datetime(df["Date"]).dt.year

    weekly = df.groupby(["Year", "Week"]).agg({
        "Date": "first",
        "forecast_smooth": "sum",
        "forecast_p10": "sum",
        "forecast_median": "sum",
        "forecast_p90": "sum",
        "daily_noise_std": "mean",
    }).reset_index()

    return weekly


def apply_noise_bands(df, df_raw, sel_brands, n_simulations=200, seed=42):
    """Replace static +/-15% bands with Monte Carlo noise bands.

    Modifies df in-place, overwriting the *_Min/*_Max columns with
    noise-derived p10/p90 percentile bands.

    Parameters
    ----------
    df : DataFrame
        Projection DataFrame with Clicks_Base, Clicks_Sim, etc.
    df_raw : DataFrame
        Raw super_dataset for noise estimation.
    sel_brands : list[str]
        Brands to aggregate for noise statistics.
    n_simulations : int
        Monte Carlo draws.
    seed : int
        Random seed.
    """
    for metric_key, base_col, sim_col in [
        ("clicks",   "Clicks_Base", "Clicks_Sim"),
        ("quantity",  "Qty_Base",    "Qty_Sim"),
        ("sales",    "Sales_Base",  "Sales_Sim"),
    ]:
        noise_stats = estimate_daily_noise(df_raw, sel_brands, metric=metric_key)
        if noise_stats is None:
            continue  # Keep static ±15% bands as fallback

        prefix = base_col.replace("_Base", "")  # "Clicks", "Qty", "Sales"

        # Apply to Base
        base_result = project_daily_with_noise(
            df, noise_stats, metric_base=base_col,
            n_simulations=n_simulations, seed=seed,
        )
        df[f"{prefix}_Base_Min"] = base_result["forecast_p10"]
        df[f"{prefix}_Base_Max"] = base_result["forecast_p90"]

        # Apply to Sim
        sim_result = project_daily_with_noise(
            df, noise_stats, metric_base=sim_col,
            n_simulations=n_simulations, seed=seed + 1,
        )
        df[f"{prefix}_Sim_Min"] = sim_result["forecast_p10"]
        df[f"{prefix}_Sim_Max"] = sim_result["forecast_p90"]
