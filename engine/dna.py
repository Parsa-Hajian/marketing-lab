import calendar
import pandas as pd
import numpy as np


# ── Month-swap at raw-data level ─────────────────────────────────────────────

def _doy_range(year_int, month):
    """Return (first_doy, last_doy) for *month* in *year_int*."""
    first = sum(calendar.monthrange(year_int, m)[1] for m in range(1, month)) + 1
    last = first + calendar.monthrange(year_int, month)[1] - 1
    return first, last


def _ratio_scale(arr, ratio):
    """Multiply array by ratio, safe against NaN/inf."""
    if not np.isfinite(ratio) or ratio == 0:
        return arr
    return arr * ratio


def apply_month_swaps(profiles, df_raw, sel_brands, swap_events):
    """Apply swaps to profiles and df_raw at the raw historical data level.

    Supports two formats:
    - Date-range (a_start/a_end/b_start/b_end): DOY-level swap of the
      specific day ranges.  Daily profiles are ratio-scaled within those
      DOY ranges, monthly profiles for affected months are recomputed,
      and df_raw rows within those date ranges are ratio-scaled.
    - Legacy single-month (a/b): whole-month swap (TimeIdx swap for
      monthly profiles, DOY-range ratio-scale for daily profiles and
      df_raw).

    Returns (profiles_mod, df_raw_mod).
    """
    mod_p = profiles.copy()
    mod_r = df_raw.copy()
    mod_r["Date"] = pd.to_datetime(mod_r["Date"])

    for ev in swap_events:
        if ev.get("type") != "swap":
            continue

        if "a_start" in ev:
            # ── Date-range swap: DOY-level ────────────────────────────
            a_doys = sorted(pd.date_range(ev["a_start"], ev["a_end"])
                            .dayofyear.unique().tolist())
            b_doys = sorted(pd.date_range(ev["b_start"], ev["b_end"])
                            .dayofyear.unique().tolist())
            a_doys_set = set(a_doys)
            b_doys_set = set(b_doys)

            # Which months are touched (for monthly profile update)
            a_months = sorted(pd.date_range(ev["a_start"], ev["a_end"])
                              .month.unique().tolist())
            b_months = sorted(pd.date_range(ev["b_start"], ev["b_end"])
                              .month.unique().tolist())
            affected_months = sorted(set(a_months + b_months))

            for brand in sel_brands:
                years = mod_p.loc[
                    (mod_p["brand"] == brand) & (mod_p["Level"] == "Daily"),
                    "Year",
                ].unique()

                for yr_str in years:
                    yr_int = 2024 if yr_str == "Overall" else int(yr_str)

                    daily_base = (
                        (mod_p["brand"] == brand)
                        & (mod_p["Year"] == yr_str)
                        & (mod_p["Level"] == "Daily")
                    )

                    # Snapshot monthly totals BEFORE the swap
                    old_month_totals = {}
                    for m in affected_months:
                        m_s, m_e = _doy_range(yr_int, m)
                        m_mask = daily_base & mod_p["TimeIdx"].between(m_s, m_e)
                        old_month_totals[m] = {
                            met: mod_p.loc[m_mask, met].sum()
                            for met in ("clicks", "quantity", "sales")
                            if met in mod_p.columns
                        }

                    # ── Daily profiles: ratio-scale DOY ranges ────────
                    am = daily_base & mod_p["TimeIdx"].isin(a_doys_set)
                    bm = daily_base & mod_p["TimeIdx"].isin(b_doys_set)

                    for met in ("clicks", "quantity", "sales"):
                        if met not in mod_p.columns:
                            continue
                        sa = mod_p.loc[am, met].sum()
                        sb = mod_p.loc[bm, met].sum()
                        if sa > 0 and sb > 0:
                            r = sb / sa
                            orig_a = mod_p.loc[am, met].copy()
                            orig_b = mod_p.loc[bm, met].copy()
                            mod_p.loc[am, met] = _ratio_scale(orig_a, r)
                            mod_p.loc[bm, met] = _ratio_scale(orig_b, 1.0 / r)

                    # ── Monthly profiles: recompute affected months ───
                    monthly_base = (
                        (mod_p["brand"] == brand)
                        & (mod_p["Year"] == yr_str)
                        & (mod_p["Level"] == "Monthly")
                    )
                    for m in affected_months:
                        m_s, m_e = _doy_range(yr_int, m)
                        m_daily = daily_base & mod_p["TimeIdx"].between(m_s, m_e)
                        m_row = monthly_base & (mod_p["TimeIdx"] == m)

                        if mod_p.loc[m_daily].empty or mod_p.loc[m_row].empty:
                            continue

                        for met in ("clicks", "quantity", "sales"):
                            if met not in mod_p.columns:
                                continue
                            old_val = old_month_totals[m].get(met, 0)
                            new_val = mod_p.loc[m_daily, met].sum()
                            mod_p.loc[m_row, met] = new_val

                            # Scale the corresponding idx_* proportionally
                            if old_val > 0:
                                ratio = new_val / old_val
                                idx_map = {
                                    "clicks": "idx_clicks",
                                    "quantity": "idx_cr",
                                    "sales": "idx_aov",
                                }
                                idx_col = idx_map.get(met)
                                if idx_col and idx_col in mod_p.columns:
                                    mod_p.loc[m_row, idx_col] *= ratio

            # ── df_raw: ratio-scale date ranges by DOY ────────────────
            raw_doy = mod_r["Date"].dt.dayofyear
            raw_brand = mod_r["brand"].isin(sel_brands)

            for yr in mod_r.loc[raw_brand, "Date"].dt.year.unique():
                yr_mask = raw_brand & (mod_r["Date"].dt.year == yr)
                am = yr_mask & raw_doy.isin(a_doys_set)
                bm = yr_mask & raw_doy.isin(b_doys_set)

                for met in ("clicks", "quantity", "sales"):
                    sa = mod_r.loc[am, met].sum()
                    sb = mod_r.loc[bm, met].sum()
                    if sa > 0 and sb > 0:
                        r = sb / sa
                        orig_a = mod_r.loc[am, met].copy()
                        orig_b = mod_r.loc[bm, met].copy()
                        mod_r.loc[am, met] = _ratio_scale(orig_a, r)
                        mod_r.loc[bm, met] = _ratio_scale(orig_b, 1.0 / r)

        else:
            # ── Legacy: whole-month swap ──────────────────────────────
            ma, mb = ev["a"], ev["b"]
            if ma == mb:
                continue

            # Monthly profiles — swap TimeIdx
            bp = mod_p["brand"].isin(sel_brands) & (mod_p["Level"] == "Monthly")
            a_idx = mod_p.index[bp & (mod_p["TimeIdx"] == ma)]
            b_idx = mod_p.index[bp & (mod_p["TimeIdx"] == mb)]
            mod_p.loc[a_idx, "TimeIdx"] = -999
            mod_p.loc[b_idx, "TimeIdx"] = ma
            mod_p.loc[mod_p["TimeIdx"] == -999, "TimeIdx"] = mb

            # Daily profiles — ratio-scale
            for brand in sel_brands:
                years = mod_p.loc[
                    (mod_p["brand"] == brand) & (mod_p["Level"] == "Daily"),
                    "Year",
                ].unique()
                for yr_str in years:
                    yr_int = 2024 if yr_str == "Overall" else int(yr_str)
                    da_s, da_e = _doy_range(yr_int, ma)
                    db_s, db_e = _doy_range(yr_int, mb)

                    base = (
                        (mod_p["brand"] == brand)
                        & (mod_p["Year"] == yr_str)
                        & (mod_p["Level"] == "Daily")
                    )
                    am = base & mod_p["TimeIdx"].between(da_s, da_e)
                    bm = base & mod_p["TimeIdx"].between(db_s, db_e)

                    for met in ("clicks", "quantity", "sales"):
                        if met not in mod_p.columns:
                            continue
                        sa = mod_p.loc[am, met].sum()
                        sb = mod_p.loc[bm, met].sum()
                        if sa > 0 and sb > 0:
                            r = sb / sa
                            mod_p.loc[am, met] = _ratio_scale(mod_p.loc[am, met], r)
                            mod_p.loc[bm, met] = _ratio_scale(
                                mod_p.loc[bm, met], 1.0 / r)

            # df_raw — ratio-scale per year
            raw_month = mod_r["Date"].dt.month
            raw_brand = mod_r["brand"].isin(sel_brands)
            for yr in mod_r.loc[raw_brand, "Date"].dt.year.unique():
                yr_mask = raw_brand & (mod_r["Date"].dt.year == yr)
                am = yr_mask & (raw_month == ma)
                bm = yr_mask & (raw_month == mb)

                for met in ("clicks", "quantity", "sales"):
                    sa = mod_r.loc[am, met].sum()
                    sb = mod_r.loc[bm, met].sum()
                    if sa > 0 and sb > 0:
                        r = sb / sa
                        mod_r.loc[am, met] = _ratio_scale(mod_r.loc[am, met], r)
                        mod_r.loc[bm, met] = _ratio_scale(
                            mod_r.loc[bm, met], 1.0 / r)

    return mod_p, mod_r


def filter_swap_events(event_log):
    """Return event_log with pre_trial swap events removed (already applied to raw data)."""
    return [
        e for e in event_log
        if not (e.get("type") == "swap" and e.get("scope") == "pre_trial")
    ]


# ── Original helpers ─────────────────────────────────────────────────────────

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
        # Date-range format: swap ALL days in range A with ALL days in range B
        if "a_start" in ev:
            a_periods = set(_periods_from_range(ev["a_start"], ev["a_end"], t_col))
            b_periods = set(_periods_from_range(ev["b_start"], ev["b_end"], t_col))
            ma = df[t_col].isin(a_periods)
            mb = df[t_col].isin(b_periods)
            for c in cols:
                av = df.loc[ma, c].mean() if ma.any() else 0
                bv = df.loc[mb, c].mean() if mb.any() else 0
                if av > 0 and bv > 0:
                    orig_a = df.loc[ma, c].copy()
                    orig_b = df.loc[mb, c].copy()
                    df.loc[ma, c] = orig_a * (bv / av)
                    df.loc[mb, c] = orig_b * (av / bv)
        else:
            # Legacy single-index format
            ma, mb = df[t_col] == ev["a"], df[t_col] == ev["b"]
            for c in cols:
                av, bv = df.loc[ma, c].mean(), df.loc[mb, c].mean()
                if av > 0 and bv > 0:
                    orig_a = df.loc[ma, c].copy()
                    orig_b = df.loc[mb, c].copy()
                    df.loc[ma, c] = orig_a * (bv / av)
                    df.loc[mb, c] = orig_b * (av / bv)


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

    # Fallback: no "Overall" rows → synthesise from historical years
    if m_overall.empty and not m_yrly_agg.empty:
        m_overall = (
            m_yrly_agg
            .groupby("TimeIdx")
            .agg({"idx_clicks": "median", "idx_cr": "median", "idx_aov": "median"})
            .reset_index()
        )

    # Last resort: still empty (brand has no monthly profile at all)
    if m_overall.empty:
        m_overall = pd.DataFrame({
            "TimeIdx":    list(range(1, 13)),
            "idx_clicks": [1.0] * 12,
            "idx_cr":     [1.0] * 12,
            "idx_aov":    [1.0] * 12,
        })

    overall_weight = 1.0 if not norm_weights else 0.35
    pure_dna = m_overall.copy()
    for col in ["idx_clicks", "idx_cr", "idx_aov"]:
        pure_dna[col] = m_overall[col] * overall_weight
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
