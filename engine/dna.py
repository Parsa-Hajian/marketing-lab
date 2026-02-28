import pandas as pd
import numpy as np


def _periods_from_range(start_d, end_d, t_col):
    """Return sorted unique period indices covered by [start_d, end_d]."""
    dates = pd.date_range(start=start_d, end=end_d)
    if t_col == "Month":
        return sorted(dates.month.unique().tolist())
    elif t_col == "Week":
        return sorted(set(int(w) for w in dates.isocalendar().week))
    else:  # DayOfYear
        return sorted(dates.dayofyear.unique().tolist())


def _apply_dna_ev(df, ev, suffix):
    """Apply a custom_drag or swap event to idx_*_{suffix} columns in df (in-place)."""
    lv    = ev.get("level", "Monthly")
    t_col = "Month" if lv == "Monthly" else "Week" if lv == "Weekly" else "DayOfYear"
    cols  = [f"idx_clicks_{suffix}", f"idx_cr_{suffix}", f"idx_aov_{suffix}"]

    if ev["type"] == "custom_drag":
        mask = df[t_col] == ev["target"]
        for c in cols:
            df.loc[mask, c] *= ev["lift"]

    elif ev["type"] == "swap":
        # Date-range format: a_start/a_end/b_start/b_end
        if "a_start" in ev:
            a_periods = _periods_from_range(ev["a_start"], ev["a_end"], t_col)
            b_periods = _periods_from_range(ev["b_start"], ev["b_end"], t_col)
            for pa, pb in zip(a_periods, b_periods):
                ma, mb = df[t_col] == pa, df[t_col] == pb
                for c in cols:
                    av = df.loc[ma, c].mean()
                    bv = df.loc[mb, c].mean()
                    df.loc[ma, c] = df.loc[ma, c] * (bv / av) if av > 0 else bv
                    df.loc[mb, c] = df.loc[mb, c] * (av / bv) if bv > 0 else av
        else:
            # Legacy single-index format
            ma, mb = df[t_col] == ev["a"], df[t_col] == ev["b"]
            for c in cols:
                av, bv = df.loc[ma, c].mean(), df.loc[mb, c].mean()
                df.loc[ma, c] = df.loc[ma, c] * (bv / av) if av > 0 else bv
                df.loc[mb, c] = df.loc[mb, c] * (av / bv) if bv > 0 else av


def compute_similarity_weights(profiles, sel_brands, proj_year, t_start, t_end,
                                c_val, q_val, s_val):
    """Compute dynamic 35/65 similarity weights for historical years.

    Returns a dict {year_str: normalised_weight} for the 65% historical component.
    """
    trial_days  = pd.date_range(t_start, t_end).dayofyear.tolist()
    daily_dna   = profiles[(profiles["brand"].isin(sel_brands)) & (profiles["Level"] == "Daily")]
    hist_trial  = daily_dna[daily_dna["TimeIdx"].isin(trial_days)]
    valid_hist  = hist_trial[
        (hist_trial["Year"] != "Overall") & (hist_trial["Year"] != str(proj_year))
    ]
    yrly_totals = (
        valid_hist.groupby("Year")
        .agg({"clicks": "sum", "quantity": "sum", "sales": "sum"})
        .reset_index()
    )

    weights = {}
    for _, row in yrly_totals.iterrows():
        y     = row["Year"]
        err_c = abs(c_val - row["clicks"])   / max(c_val, 1)
        err_q = abs(q_val - row["quantity"]) / max(q_val, 1)
        err_s = abs(s_val - row["sales"])    / max(s_val, 1)
        weights[y] = 1.0 / ((err_c + err_q + err_s) / 3.0 + 0.01)

    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()} if total > 0 else {}


def build_pure_dna(profiles, sel_brands, norm_weights):
    """Build the blended monthly DNA profile (35% overall + 65% historical)."""
    m_dna      = profiles[(profiles["brand"].isin(sel_brands)) & (profiles["Level"] == "Monthly")]
    m_overall  = (
        m_dna[m_dna["Year"] == "Overall"]
        .groupby("TimeIdx")
        .agg({"idx_clicks": "median", "idx_cr": "median", "idx_aov": "median"})
        .reset_index()
    )
    m_yrly_agg = (
        m_dna[m_dna["Year"] != "Overall"]
        .groupby(["Year", "TimeIdx"])
        .agg({"idx_clicks": "median", "idx_cr": "median", "idx_aov": "median"})
        .reset_index()
    )

    pure_dna = m_overall.copy()
    for col in ["idx_clicks", "idx_cr", "idx_aov"]:
        pure_dna[col] = m_overall[col] * 0.35
        for y, w in norm_weights.items():
            y_data = m_yrly_agg[m_yrly_agg["Year"] == str(y)]
            if not y_data.empty:
                merged = (
                    pure_dna[["TimeIdx"]]
                    .merge(y_data[["TimeIdx", col]], on="TimeIdx", how="left")
                    .fillna(1.0)
                )
                pure_dna[col] += merged[col] * (0.65 * w)
            else:
                pure_dna[col] += m_overall[col] * (0.65 * w)

    return pure_dna


def build_year_dataframe(proj_year):
    """Build the 365-day base DataFrame for the projection year.

    Returns (df, full_year_DatetimeIndex).
    """
    full_year = pd.date_range(start=f"{proj_year}-01-01", end=f"{proj_year}-12-31")
    df = pd.DataFrame({"Date": full_year})
    df["Month"]     = df["Date"].dt.month
    df["Week"]      = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    return df, full_year


def build_dna_layers(df, pure_dna, event_log):
    """Add idx_*_pure / idx_*_pretrial / idx_*_work columns to df in-place.

    - pure:     raw blended DNA (no modifications)
    - pretrial: pure + pre_trial events → used for base calibration
    - work:     pretrial + post_trial events → used for simulation (After)
    """
    merged = (
        df[["Month"]]
        .merge(
            pure_dna[["TimeIdx", "idx_clicks", "idx_cr", "idx_aov"]],
            left_on="Month", right_on="TimeIdx", how="left",
        )
        .fillna(1.0)
    )

    for m in ["clicks", "cr", "aov"]:
        df[f"idx_{m}_pure"]     = merged[f"idx_{m}"].values
        df[f"idx_{m}_pretrial"] = merged[f"idx_{m}"].values
        df[f"idx_{m}_work"]     = merged[f"idx_{m}"].values

    for ev in event_log:
        if ev["type"] in ["custom_drag", "swap"] and ev.get("scope") == "pre_trial":
            _apply_dna_ev(df, ev, "pretrial")

    for m in ["clicks", "cr", "aov"]:
        df[f"idx_{m}_work"] = df[f"idx_{m}_pretrial"].copy()

    for ev in event_log:
        if ev["type"] in ["custom_drag", "swap"] and ev.get("scope", "post_trial") == "post_trial":
            _apply_dna_ev(df, ev, "work")
