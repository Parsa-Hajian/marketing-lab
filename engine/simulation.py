import numpy as np
import pandas as pd
from datetime import timedelta

from config import EVENT_MAPPING


def get_shock_multiplier(dt, shocks):
    """Return the total shock multiplier for a given date across all standard shocks."""
    total = 0.0
    for s in shocks:
        if s["type"] != "shock":
            continue
        if s["start"] <= dt.date() <= s["end"]:
            duration = (s["end"] - s["start"]).days + 1
            t = (dt.date() - s["start"]).days
            p = t / duration if duration > 0 else 0
            shape = EVENT_MAPPING.get(s["shape"], "Step")
            if shape == "Step":
                total += s["str"]
            elif shape == "Linear Fade":
                total += s["str"] * (1 - p)
            elif shape == "Front-Loaded":
                total += s["str"] * np.exp(-3.0 * p)
            elif shape == "Delayed Peak":
                total += s["str"] * np.exp(
                    -((t - duration * 0.4) ** 2) / (2 * (duration * 0.3) ** 2)
                )
    return total


def _build_reapplied_cols(df, ev_subset):
    """Compute absolute and relative addition arrays from reapplied_shock events.

    Returns six numpy arrays: abs_c, abs_q, abs_s, rel_c, rel_q, rel_s
    """
    n = len(df)
    abs_c = np.zeros(n); abs_q = np.zeros(n); abs_s = np.zeros(n)
    rel_c = np.zeros(n); rel_q = np.zeros(n); rel_s = np.zeros(n)

    for ev in ev_subset:
        if ev["type"] != "reapplied_shock":
            continue
        new_end = ev["new_start"] + timedelta(days=ev["duration"] - 1)
        mask = (df["Date"].dt.date >= ev["new_start"]) & (df["Date"].dt.date <= new_end)
        idx  = df[mask].index
        k    = min(len(idx), ev["duration"])
        if ev["mode"] == "Absolute Volume":
            abs_c[idx[:k]] += np.array(ev["daily_abs_c"][:k])
            abs_q[idx[:k]] += np.array(ev["daily_abs_q"][:k])
            abs_s[idx[:k]] += np.array(ev["daily_abs_s"][:k])
        else:
            rel_c[idx[:k]] += np.array(ev["daily_pct_c"][:k])
            rel_q[idx[:k]] += np.array(ev["daily_pct_q"][:k])
            rel_s[idx[:k]] += np.array(ev["daily_pct_s"][:k])

    return abs_c, abs_q, abs_s, rel_c, rel_q, rel_s


def eval_events(ev_subset, *, pure_dna, adj_c, adj_q, adj_s,
                t_start, t_end, tgt_start, tgt_end):
    """Full simulation rebuild for an event subset (used by Attribution Engine).

    Returns {Sales, Qty, Clicks} totals for the target period.
    """
    from engine.dna import build_year_dataframe, build_dna_layers

    df, _ = build_year_dataframe(t_start.year)
    build_dna_layers(df, pure_dna, ev_subset)

    t_mask = (df["Date"].dt.date >= t_start) & (df["Date"].dt.date <= t_end)
    t_d    = df[t_mask]

    if t_d.empty or t_d["idx_clicks_pretrial"].sum() == 0:
        return {"Sales": 0.0, "Qty": 0.0, "Clicks": 0.0}

    b_clicks = adj_c / t_d["idx_clicks_pretrial"].sum()
    t_cr     = adj_q / adj_c if adj_c > 0 else 0
    t_aov    = adj_s / adj_q if adj_q > 0 else 0
    b_cr     = t_cr  / t_d["idx_cr_pretrial"].mean()  if t_d["idx_cr_pretrial"].mean()  > 0 else t_cr
    b_aov    = t_aov / t_d["idx_aov_pretrial"].mean() if t_d["idx_aov_pretrial"].mean() > 0 else t_aov

    df["Clicks_Base_"] = b_clicks * df["idx_clicks_pretrial"]
    df["Qty_Base_"]    = df["Clicks_Base_"] * (b_cr  * df["idx_cr_pretrial"])
    df["Sales_Base_"]  = df["Qty_Base_"]    * (b_aov * df["idx_aov_pretrial"])

    df["Shock_"] = df["Date"].apply(lambda x: get_shock_multiplier(x, ev_subset))
    ac, aq, as_, rc, rq, rs = _build_reapplied_cols(df, ev_subset)

    c_std = (b_clicks * df["idx_clicks_work"]) * (1 + df["Shock_"])
    q_std = c_std * (b_cr  * df["idx_cr_work"])
    s_std = q_std * (b_aov * df["idx_aov_work"])

    c_sim = c_std + df["Clicks_Base_"] * rc + ac
    q_sim = q_std + df["Qty_Base_"]    * rq + aq
    s_sim = s_std + df["Sales_Base_"]  * rs + as_

    tgt = (df["Date"].dt.date >= tgt_start) & (df["Date"].dt.date <= tgt_end)
    return {
        "Sales":  float(s_sim[tgt].sum()),
        "Qty":    float(q_sim[tgt].sum()),
        "Clicks": float(c_sim[tgt].sum()),
    }
