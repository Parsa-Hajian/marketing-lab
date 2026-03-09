import numpy as np
import pandas as pd
from engine.simulation import get_shock_multiplier, _build_reapplied_cols


def calibrate_base(df, t_start, t_end, adj_c, adj_q, adj_s):
    """Calibrate base constants using pre-trial DNA and adjusted trial values.

    Returns (base_clicks, base_cr, base_aov) or (None, None, None) if the trial
    window has no valid data.
    """
    t_mask = (df["Date"].dt.date >= t_start) & (df["Date"].dt.date <= t_end)
    t_d    = df[t_mask]

    if t_d.empty or t_d["idx_clicks_pretrial"].sum() == 0:
        return None, None, None

    base_clicks = adj_c / t_d["idx_clicks_pretrial"].sum()
    trial_cr    = adj_q / adj_c if adj_c > 0 else 0
    trial_aov   = adj_s / adj_q if adj_q > 0 else 0
    base_cr     = (trial_cr  / t_d["idx_cr_pretrial"].mean()
                   if t_d["idx_cr_pretrial"].mean()  > 0 else trial_cr)
    base_aov    = (trial_aov / t_d["idx_aov_pretrial"].mean()
                   if t_d["idx_aov_pretrial"].mean() > 0 else trial_aov)

    return base_clicks, base_cr, base_aov


def build_projections(df, base_clicks, base_cr, base_aov, event_log):
    """Add Baseline, Simulation, and ±15% Margin columns to df in-place."""
    # ── Baseline "Before" (pre-trial DNA, no shocks) ───────────────────────
    df["Clicks_Base"] = base_clicks * df["idx_clicks_pretrial"]
    df["Qty_Base"]    = df["Clicks_Base"] * (base_cr  * df["idx_cr_pretrial"])
    df["Sales_Base"]  = df["Qty_Base"]    * (base_aov * df["idx_aov_pretrial"])

    # ── Standard shock multipliers ──────────────────────────────────────────
    df["Shock"] = df["Date"].apply(lambda x: get_shock_multiplier(x, event_log))

    # ── Reapplied signature injections ──────────────────────────────────────
    abs_c, abs_q, abs_s, rel_c, rel_q, rel_s = _build_reapplied_cols(df, event_log)

    # ── Simulation "After" (full working DNA + shocks + injections) ─────────
    c_standard = (base_clicks * df["idx_clicks_work"]) * (1 + df["Shock"])
    q_standard = c_standard * (base_cr  * df["idx_cr_work"])
    s_standard = q_standard * (base_aov * df["idx_aov_work"])

    df["Clicks_Sim"] = c_standard + df["Clicks_Base"] * rel_c + abs_c
    df["Qty_Sim"]    = q_standard + df["Qty_Base"]    * rel_q + abs_q
    df["Sales_Sim"]  = s_standard + df["Sales_Base"]  * rel_s + abs_s

    # ── Confidence margins ±15% ─────────────────────────────────────────────
    _recompute_margins(df)


def apply_historical_shrinkage(df, profiles, sel_brands):
    """Shrink forecast toward historical monthly medians when deviation is extreme.

    For each month, if the forecast total deviates more than 2x from the
    historical monthly median (across all years), progressively blend the
    forecast toward the median.  Operates on df in-place.
    """
    # ── Build historical monthly medians from profiles ────────────────────
    m_prof = profiles[
        (profiles["brand"].isin(sel_brands))
        & (profiles["Level"] == "Monthly")
        & (profiles["Year"] != "Overall")
    ]
    if m_prof.empty:
        return

    # Aggregate across selected brands per (Year, TimeIdx) then median across years
    yr_agg = (
        m_prof.groupby(["Year", "TimeIdx"])
        .agg(clicks=("clicks", "sum"), quantity=("quantity", "sum"), sales=("sales", "sum"))
        .reset_index()
    )
    hist_medians = (
        yr_agg.groupby("TimeIdx")
        .agg(clicks=("clicks", "median"), quantity=("quantity", "median"), sales=("sales", "median"))
        .reset_index()
    )

    # Map profiles metric → forecast column pairs
    metric_map = [
        ("clicks",   "Clicks_Base", "Clicks_Sim"),
        ("quantity", "Qty_Base",    "Qty_Sim"),
        ("sales",    "Sales_Base",  "Sales_Sim"),
    ]

    log2 = np.log(2)

    for prof_met, base_col, sim_col in metric_map:
        for month in range(1, 13):
            med_row = hist_medians[hist_medians["TimeIdx"] == month]
            if med_row.empty:
                continue
            hist_med = float(med_row[prof_met].iloc[0])
            if hist_med <= 0:
                continue

            month_mask = df["Month"] == month
            n_days = month_mask.sum()
            if n_days == 0:
                continue

            for col in (base_col, sim_col):
                fc_total = df.loc[month_mask, col].sum()
                if fc_total <= 0:
                    continue

                deviation = fc_total / hist_med
                if deviation > 2 or deviation < 0.5:
                    log_dev = abs(np.log(deviation))
                    alpha = min(0.8, (log_dev - log2) / (2 * log2))
                    alpha = max(0.0, alpha)

                    # Distribute median across days proportionally to forecast shape
                    daily_share = df.loc[month_mask, col] / fc_total
                    blended = (1 - alpha) * df.loc[month_mask, col] + alpha * (daily_share * hist_med)
                    df.loc[month_mask, col] = blended

    # ── Recompute margin columns ──────────────────────────────────────────
    _recompute_margins(df)


def _recompute_margins(df):
    """Recompute ±15% confidence margin columns."""
    for m in ["Clicks", "Qty", "Sales"]:
        df[f"{m}_Base_Min"] = df[f"{m}_Base"] * 0.85
        df[f"{m}_Base_Max"] = df[f"{m}_Base"] * 1.15
        df[f"{m}_Sim_Min"]  = df[f"{m}_Sim"]  * 0.85
        df[f"{m}_Sim_Max"]  = df[f"{m}_Sim"]  * 1.15


def apply_trial_conservatism(df, profiles, sel_brands, t_start, t_end):
    """Apply conservative bounds when trial window is small.

    When the trial covers a small fraction of the year, base calibration
    extrapolation is highly uncertain.  This function:

    1. Blends monthly forecasts toward historical medians — the blend
       strength is proportional to how *little* of the year the trial covers.
    2. Caps the yearly forecast total so it never exceeds the historical
       range plus a small margin.
    3. If the trial period's forecast is below the *minimum* historical value
       for that same period (meaning the trial underperformed every past year),
       the yearly total is hard-capped at the historical median.

    Operates on df in-place.
    """
    trial_days = (t_end - t_start).days + 1
    trial_coverage = trial_days / 365.0

    # No extra conservatism needed once trial covers ≥50% of the year
    if trial_coverage >= 0.50:
        return

    # ── Build historical references ─────────────────────────────────────
    m_prof = profiles[
        (profiles["brand"].isin(sel_brands))
        & (profiles["Level"] == "Monthly")
        & (profiles["Year"] != "Overall")
    ]
    if m_prof.empty:
        return

    yr_month_agg = (
        m_prof.groupby(["Year", "TimeIdx"])
        .agg(clicks=("clicks", "sum"), quantity=("quantity", "sum"),
             sales=("sales", "sum"))
        .reset_index()
    )
    hist_monthly_medians = (
        yr_month_agg.groupby("TimeIdx")
        .agg(clicks=("clicks", "median"), quantity=("quantity", "median"),
             sales=("sales", "median"))
        .reset_index()
    )
    yr_totals = (
        yr_month_agg.groupby("Year")
        .agg(clicks=("clicks", "sum"), quantity=("quantity", "sum"),
             sales=("sales", "sum"))
        .reset_index()
    )

    # ── Conservatism strength (0.0 – 0.75) based on trial coverage ──────
    #   1 month  ≈ 0.085 coverage → strength ≈ 0.62
    #   3 months ≈ 0.25  coverage → strength ≈ 0.37
    #   6 months ≈ 0.50  coverage → strength = 0.00
    strength = max(0.0, min(0.75, 1.5 * (0.5 - trial_coverage)))

    metric_map = [
        ("clicks",   "Clicks_Base", "Clicks_Sim"),
        ("quantity", "Qty_Base",    "Qty_Sim"),
        ("sales",    "Sales_Base",  "Sales_Sim"),
    ]

    # ── Step 1: Monthly blending toward historical medians ──────────────
    for prof_met, base_col, sim_col in metric_map:
        for month in range(1, 13):
            med_row = hist_monthly_medians[hist_monthly_medians["TimeIdx"] == month]
            if med_row.empty:
                continue
            hist_med = float(med_row[prof_met].iloc[0])
            if hist_med <= 0:
                continue

            month_mask = df["Month"] == month
            if month_mask.sum() == 0:
                continue

            for col in (base_col, sim_col):
                fc_total = df.loc[month_mask, col].sum()
                if fc_total <= 0:
                    continue
                daily_share = df.loc[month_mask, col] / fc_total
                blended = ((1 - strength) * df.loc[month_mask, col]
                           + strength * (daily_share * hist_med))
                df.loc[month_mask, col] = blended

    # ── Step 2: Yearly cap — never exceed historical p95 + margin ───────
    for prof_met, base_col, sim_col in metric_map:
        hist_vals = yr_totals[prof_met].values
        if len(hist_vals) == 0 or float(np.max(hist_vals)) <= 0:
            continue

        cap = float(np.percentile(hist_vals, 95))
        cap *= (1 + 0.05)  # 5% margin on top of p95

        for col in (base_col, sim_col):
            fc_total = df[col].sum()
            if fc_total > cap and fc_total > 0:
                df[col] *= (cap / fc_total)

    # ── Step 3: Trial-below-historical guard ────────────────────────────
    # If the forecast for the trial months is below every historical year's
    # value for those months, the trial data underperformed all history.
    # In that case, cap the yearly total at the historical median.
    trial_months = sorted(
        pd.date_range(t_start, t_end).month.unique().tolist()
    )

    for prof_met, base_col, sim_col in metric_map:
        trial_period_by_year = (
            yr_month_agg[yr_month_agg["TimeIdx"].isin(trial_months)]
            .groupby("Year")[prof_met].sum()
        )
        if trial_period_by_year.empty:
            continue

        hist_trial_min = float(trial_period_by_year.min())
        hist_med_yr = float(yr_totals[prof_met].median())

        for col in (base_col, sim_col):
            trial_fc = df.loc[df["Month"].isin(trial_months), col].sum()

            # Forecast for trial period is below every historical year
            if trial_fc < hist_trial_min and hist_trial_min > 0:
                fc_yearly = df[col].sum()
                if fc_yearly > hist_med_yr and hist_med_yr > 0:
                    df[col] *= (hist_med_yr / fc_yearly)

    # ── Recompute margin columns ────────────────────────────────────────
    _recompute_margins(df)
