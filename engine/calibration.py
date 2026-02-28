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
    for m in ["Clicks", "Qty", "Sales"]:
        df[f"{m}_Base_Min"] = df[f"{m}_Base"] * 0.85
        df[f"{m}_Base_Max"] = df[f"{m}_Base"] * 1.15
        df[f"{m}_Sim_Min"]  = df[f"{m}_Sim"]  * 0.85
        df[f"{m}_Sim_Max"]  = df[f"{m}_Sim"]  * 1.15
