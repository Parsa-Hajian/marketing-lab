import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
import time

from config import EVENT_MAPPING
from engine.dna import (
    compute_similarity_weights, _apply_dna_ev,
    build_pure_dna, build_year_dataframe, build_dna_layers,
    apply_month_swaps, filter_swap_events,
)
from engine.calibration import (
    calibrate_base, build_projections, apply_historical_shrinkage,
    apply_trial_conservatism,
)
from engine.noise import apply_noise_bands
from engine.simulation import get_shock_multiplier, eval_events
from engine.activity_log import log_action
from engine.settings_store import load_settings, get_campaign_default
from utils.export import build_excel_report

_C_BASE = "#94a3b8"
_C_SIM  = "#1a1a6b"
_TMPL   = "plotly_white"

# ── Event type styling ─────────────────────────────────────────────────────────
_EV_STYLE = {
    "shock":          {"icon": "📣", "label": "Campaign",    "color": "#fef3c7", "border": "#f59e0b"},
    "custom_drag":    {"icon": "🖱️", "label": "DNA Drag",    "color": "#ede9fe", "border": "#7c3aed"},
    "swap":           {"icon": "🔄", "label": "DNA Swap",    "color": "#e0f2fe", "border": "#0284c7"},
    "reapplied_shock":{"icon": "💉", "label": "Re-Injection","color": "#dcfce7", "border": "#16a34a"},
}


def _ensure_lab_state():
    """Defensive session state init for lab functions."""
    if "event_log"        not in st.session_state: st.session_state.event_log        = []
    if "shift_target_idx" not in st.session_state: st.session_state.shift_target_idx = None
    if "tgt_start"        not in st.session_state: st.session_state.tgt_start        = None
    if "tgt_end"          not in st.session_state: st.session_state.tgt_end          = None
    if "target_metric"    not in st.session_state: st.session_state.target_metric    = "Sales"
    if "target_val"       not in st.session_state: st.session_state.target_val       = 0.0


_STEP_ORDER = [
    "nav_brand_select", "nav_edit_dna", "nav_trial_data",
    "nav_goal_tracker", "nav_campaigns", "nav_risk",
    "nav_audit", "nav_download",
]


def _invalidate_from(step_key):
    """Mark the given step and all subsequent steps as incomplete."""
    idx = _STEP_ORDER.index(step_key) if step_key in _STEP_ORDER else -1
    if idx < 0:
        return
    for k in _STEP_ORDER[idx:]:
        st.session_state.step_completed[k] = False
    # Clear pipeline cache when invalidating from step 3 or earlier
    if idx <= 2:
        for ck in list(st.session_state.pipeline_cache.keys()):
            st.session_state.pipeline_cache[ck] = None


def _render_step_toolbar(step_key, on_undo, on_reset):
    """Render Undo / Reset inline buttons for a workflow step."""
    c_undo, c_reset, _ = st.columns([1, 1, 6])
    if c_undo.button("Undo", key=f"tb_undo_{step_key}", use_container_width=True):
        on_undo()
    if c_reset.button("Reset", key=f"tb_reset_{step_key}", use_container_width=True):
        on_reset()
    st.markdown("---")


def _clear_snapshot(step_key):
    """Remove snapshot so a fresh one is taken on next visit."""
    sk = f"_step_snapshot_{step_key}"
    if sk in st.session_state:
        del st.session_state[sk]


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Brand Selection
# ═══════════════════════════════════════════════════════════════════════════════

def _save_brand_select(profiles, sel_brands):
    """Save brand selection: build cache + mark complete (no navigation)."""
    st.session_state.ui_sel_brands = sel_brands
    pure_dna = build_pure_dna(profiles, sel_brands, {})
    proj_year = str(st.session_state.ui_t_start.year)
    df_base, _ = build_year_dataframe(int(proj_year))

    _invalidate_from("nav_edit_dna")
    st.session_state.step_completed["nav_brand_select"] = True
    st.session_state.pipeline_cache["pure_dna"] = pure_dna
    st.session_state.pipeline_cache["df_base"] = df_base
    _clear_snapshot("nav_brand_select")

    log_action(
        name=st.session_state.get("_user_name", ""),
        username=st.session_state.get("_username", ""),
        action="Brand Selection Saved",
        details=f"Brands: {', '.join(sel_brands)}",
    )


def render_brand_select(profiles, all_brands):
    """Step 1: Select brands for DNA analysis."""

    # Snapshot for undo
    snap_key = "_step_snapshot_nav_brand_select"
    if snap_key not in st.session_state:
        st.session_state[snap_key] = {
            "ui_sel_brands": list(st.session_state.ui_sel_brands),
        }

    def _undo():
        snap = st.session_state.get(snap_key, {})
        st.session_state.ui_sel_brands = snap.get("ui_sel_brands", [])
        st.toast("Reverted to previous brand selection.")
        st.rerun()

    def _reset():
        st.session_state.ui_sel_brands = []
        st.session_state.brands_select_all = False
        st.session_state.step_completed["nav_brand_select"] = False
        st.toast("Brand selection reset.")
        st.rerun()

    _render_step_toolbar("nav_brand_select", _undo, _reset)

    st.markdown("##### DNA Brands")
    st.caption("Select the brands whose historical DNA profiles will be blended for calibration.")

    # Recompute default when key was deleted (user navigated away and back)
    if "brands_select_all" not in st.session_state:
        if st.session_state.ui_sel_brands:
            st.session_state.brands_select_all = (
                set(st.session_state.ui_sel_brands) >= set(all_brands)
            )
        else:
            st.session_state.brands_select_all = True

    select_all = st.checkbox("All brands", key="brands_select_all")
    brand_cols = st.columns(min(len(all_brands), 4))
    sel_brands = []
    for i, b in enumerate(all_brands):
        col = brand_cols[i % len(brand_cols)]
        chk = col.checkbox(
            b.title(),
            value=(select_all or b in st.session_state.ui_sel_brands),
            disabled=select_all,
            key=f"chk_{b}",
        )
        if select_all or chk:
            sel_brands.append(b)

    if not sel_brands:
        st.warning("Select at least one brand.")
        return

    st.markdown("---")

    if st.button("Confirm Selection →", type="primary", use_container_width=True):
        _save_brand_select(profiles, sel_brands)
        st.session_state.nav_page = "nav_edit_dna"
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Edit DNA (Swap + Sculpt + Visualization)
# ═══════════════════════════════════════════════════════════════════════════════

def render_edit_dna(profiles, df_raw, sel_brands, res_level, time_col, settings=None):
    """Step 2: Swap, sculpt, and visualize DNA modifications."""
    _ensure_lab_state()
    if settings is None:
        settings = load_settings()
    _user_name = st.session_state.get("_user_name", "Unknown")
    _username  = st.session_state.get("_username", "")

    cache = st.session_state.pipeline_cache
    pure_dna = cache.get("pure_dna")
    if pure_dna is None:
        st.warning("Complete Step 1 first.")
        return

    # ── Toolbar ──
    def _undo():
        pre = [e for e in st.session_state.event_log if e.get("scope") == "pre_trial"]
        if not pre:
            st.toast("Nothing to undo.")
            return
        # Remove last pre_trial event
        for i in range(len(st.session_state.event_log) - 1, -1, -1):
            if st.session_state.event_log[i].get("scope") == "pre_trial":
                st.session_state.event_log.pop(i)
                break
        st.toast("Last DNA edit undone.")
        st.rerun()

    def _reset():
        st.session_state.event_log = [
            e for e in st.session_state.event_log
            if e.get("scope") != "pre_trial"
        ]
        st.session_state.step_completed["nav_edit_dna"] = False
        st.toast("All DNA edits cleared.")
        st.rerun()

    _render_step_toolbar("nav_edit_dna", _undo, _reset)

    # ── Section A: DNA Swap ──────────────────────────────────────────────
    st.markdown("##### 🔄 DNA Swap")
    st.caption(
        "Swap the seasonal DNA pattern between two date ranges. "
        "For example, swap a weak month with a strong month to test what-if scenarios."
    )

    sw1, sw2 = st.columns(2)
    with sw1:
        st.markdown("**Range A**")
        a_s1, a_s2 = st.columns(2)
        swap_a_start = a_s1.date_input("A Start", date(2026, 1, 1), key="swap_a_start")
        swap_a_end   = a_s2.date_input("A End",   date(2026, 1, 31), key="swap_a_end")
    with sw2:
        st.markdown("**Range B**")
        b_s1, b_s2 = st.columns(2)
        swap_b_start = b_s1.date_input("B Start", date(2026, 6, 1), key="swap_b_start")
        swap_b_end   = b_s2.date_input("B End",   date(2026, 6, 30), key="swap_b_end")

    if st.button("Execute DNA Swap", key="exec_swap"):
        if swap_a_start > swap_a_end or swap_b_start > swap_b_end:
            st.error("Start must be before End in both ranges.")
        else:
            st.session_state.event_log.append({
                "type": "swap", "level": res_level,
                "a_start": swap_a_start, "a_end": swap_a_end,
                "b_start": swap_b_start, "b_end": swap_b_end,
                "scope": "pre_trial",
            })
            log_action(
                name=_user_name, username=_username,
                action="DNA Swap",
                details=(
                    f"Brands: {', '.join(sel_brands)} | "
                    f"A: {swap_a_start} → {swap_a_end} | "
                    f"B: {swap_b_start} → {swap_b_end} | "
                    f"Level: {res_level} | Scope: pre_trial"
                ),
            )
            st.rerun()

    st.markdown("---")

    # ── Section B: DNA Sculpting ─────────────────────────────────────────
    st.markdown("##### 🖱️ DNA Sculpting")
    st.caption("Apply a multiplier to a target time period to reshape the DNA curve.")

    col_a, col_b = st.columns(2)
    cd_target = col_a.number_input(
        f"Target {res_level} Index", min_value=1, value=1, key="sculpt_target")
    cd_lift = col_b.slider("Multiplier (×)", 0.0, 5.0, 1.0, step=0.05, key="sculpt_lift")

    if st.button("Apply Sculpt", key="apply_sculpt"):
        st.session_state.event_log.append({
            "type": "custom_drag", "level": res_level,
            "target": cd_target, "lift": cd_lift, "scope": "pre_trial",
        })
        log_action(
            name=_user_name, username=_username,
            action="DNA Sculpt",
            details=(
                f"Brands: {', '.join(sel_brands)} | "
                f"Level: {res_level} | Target: {cd_target} | "
                f"Multiplier: ×{cd_lift:.2f} | Scope: pre_trial"
            ),
        )
        st.rerun()

    st.markdown("---")

    # ── Section C: Before / After DNA Visualization ──────────────────────
    st.markdown("##### 🧬 DNA Visualization — Before & After")
    proj_year = str(st.session_state.ui_t_start.year)
    df_before, _ = build_year_dataframe(int(proj_year))
    df_after,  _ = build_year_dataframe(int(proj_year))

    # Build DNA layers: before (no events, original DNA)
    build_dna_layers(df_before, pure_dna, [])

    # Build DNA layers: after (swaps applied at raw-data level, sculpts via layers)
    pre_trial_events = [
        e for e in st.session_state.event_log
        if e.get("scope") == "pre_trial"
    ]
    swap_events = [e for e in pre_trial_events if e["type"] == "swap"]
    non_swap_events = [e for e in pre_trial_events if e["type"] != "swap"]

    if swap_events:
        profiles_swapped, _ = apply_month_swaps(profiles, df_raw, sel_brands, swap_events)
        pure_dna_swapped = build_pure_dna(profiles_swapped, sel_brands, {})
    else:
        pure_dna_swapped = pure_dna

    build_dna_layers(df_after, pure_dna_swapped, non_swap_events)

    # Aggregate to selected resolution
    agg_before = df_before.groupby(time_col).agg({
        "idx_clicks_pretrial": "mean", "idx_cr_pretrial": "mean", "idx_aov_pretrial": "mean",
    }).reset_index()
    agg_after = df_after.groupby(time_col).agg({
        "idx_clicks_pretrial": "mean", "idx_cr_pretrial": "mean", "idx_aov_pretrial": "mean",
    }).reset_index()

    _DNA_LABELS = {"idx_clicks_pretrial": "Clicks", "idx_cr_pretrial": "CR", "idx_aov_pretrial": "AOV"}
    _BEFORE_COLORS = {"idx_clicks_pretrial": "#FCD34D", "idx_cr_pretrial": "#6EE7B7", "idx_aov_pretrial": "#C4B5FD"}
    _AFTER_COLORS  = {"idx_clicks_pretrial": "#C2410C", "idx_cr_pretrial": "#065F46", "idx_aov_pretrial": "#5B21B6"}

    fig_dna = go.Figure()
    for col_name in _DNA_LABELS:
        label = _DNA_LABELS[col_name]
        fig_dna.add_trace(go.Scatter(
            x=agg_before[time_col], y=agg_before[col_name],
            mode="lines", line=dict(dash="dot", width=2, color=_BEFORE_COLORS[col_name]),
            name=f"{label} (Before)"))
        fig_dna.add_trace(go.Scatter(
            x=agg_after[time_col], y=agg_after[col_name],
            mode="lines+markers",
            line=dict(width=3, color=_AFTER_COLORS[col_name]),
            marker=dict(size=4),
            name=f"{label} (After)"))

    fig_dna.update_layout(
        template=_TMPL,
        title=dict(text=f"DNA Profile at {res_level} Resolution — Before vs After", font=dict(color="#12124a")),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_dna, use_container_width=True)

    n_events = len(pre_trial_events)
    if n_events:
        st.info(f"**{n_events} DNA edit(s)** active.")

    # ── Confirm ──
    st.markdown("---")
    if st.button("Confirm DNA Edits →", type="primary", use_container_width=True):
        _invalidate_from("nav_trial_data")
        st.session_state.step_completed["nav_edit_dna"] = True
        log_action(
            name=_user_name, username=_username,
            action="Edit DNA Completed",
            details=(
                f"Brands: {', '.join(sel_brands)} | "
                f"Events: {n_events}"
            ),
        )
        st.session_state.nav_page = "nav_trial_data"
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Trial Data
# ═══════════════════════════════════════════════════════════════════════════════

def _run_trial_calibration(profiles, sel_brands, df_raw):
    """Run the full calibration pipeline. Returns True on success, False on error."""
    t_start = st.session_state.ui_t_start
    t_end = st.session_state.ui_t_end
    c_val = st.session_state.ui_c_val
    q_val = st.session_state.ui_q_val
    s_val = st.session_state.ui_s_val

    if c_val <= 0 and q_val <= 0 and s_val <= 0:
        st.error("Enter at least one trial metric (Clicks, Quantity, or Sales).")
        return False

    # Apply pre-trial swaps at the raw data level
    event_log = st.session_state.event_log
    swap_events = [
        e for e in event_log
        if e.get("type") == "swap" and e.get("scope") == "pre_trial"
    ]
    if swap_events:
        profiles_use, df_raw_use = apply_month_swaps(
            profiles, df_raw, sel_brands, swap_events)
    else:
        profiles_use, df_raw_use = profiles, df_raw

    # Filter swap events out — they are already applied to raw data
    event_log_no_swaps = filter_swap_events(event_log)

    proj_year = str(t_start.year)
    norm_weights = compute_similarity_weights(
        profiles_use, sel_brands, proj_year, t_start, t_end, c_val, q_val, s_val)
    pure_dna = build_pure_dna(profiles_use, sel_brands, norm_weights)
    df, _ = build_year_dataframe(int(proj_year))
    build_dna_layers(df, pure_dna, event_log_no_swaps)

    adj_c = c_val / (1 + st.session_state.ui_adj_c / 100) if (1 + st.session_state.ui_adj_c / 100) != 0 else c_val
    adj_q = q_val / (1 + st.session_state.ui_adj_q / 100) if (1 + st.session_state.ui_adj_q / 100) != 0 else q_val
    adj_s = s_val / (1 + st.session_state.ui_adj_s / 100) if (1 + st.session_state.ui_adj_s / 100) != 0 else s_val

    base_clicks, base_cr, base_aov = calibrate_base(df, t_start, t_end, adj_c, adj_q, adj_s)
    if base_clicks is None:
        st.error("Trial date range yields zero DNA sum. Widen the trial period.")
        return False

    build_projections(df, base_clicks, base_cr, base_aov, event_log)
    apply_historical_shrinkage(df, profiles_use, sel_brands)
    apply_trial_conservatism(df, profiles_use, sel_brands, t_start, t_end)
    apply_noise_bands(df, df_raw_use, sel_brands)

    _invalidate_from("nav_goal_tracker")
    st.session_state.step_completed["nav_trial_data"] = True
    _clear_snapshot("nav_trial_data")

    cache = st.session_state.pipeline_cache
    cache["pure_dna"] = pure_dna
    cache["pure_dna_weighted"] = pure_dna
    cache["df"] = df
    cache["norm_weights"] = norm_weights
    cache["base_clicks"] = base_clicks
    cache["base_cr"] = base_cr
    cache["base_aov"] = base_aov
    cache["adj_c"] = adj_c
    cache["adj_q"] = adj_q
    cache["adj_s"] = adj_s
    cache["proj_year"] = proj_year
    cache["t_start"] = t_start
    cache["t_end"] = t_end
    cache["df_raw_mod"] = df_raw_use
    cache["profiles_mod"] = profiles_use

    # Persist trial inputs so they survive Streamlit widget-key cleanup
    st.session_state._persisted_inputs.update({
        "ui_t_start":    t_start,
        "ui_t_end":      t_end,
        "ui_c_val":      c_val,
        "ui_q_val":      q_val,
        "ui_s_val":      s_val,
        "ui_adj_c":      st.session_state.ui_adj_c,
        "ui_adj_q":      st.session_state.ui_adj_q,
        "ui_adj_s":      st.session_state.ui_adj_s,
        "ui_trial_mode": st.session_state.get("ui_trial_mode", "enter"),
    })
    return True


def _compute_historical_autofill(df_raw, sel_brands):
    """Find the most recent complete year for selected brands and sum metrics.

    Returns (year, clicks_total, qty_total, sales_total) or None.
    """
    brand_data = df_raw[df_raw["brand"].isin(sel_brands)].copy()
    if brand_data.empty:
        return None
    brand_data["_year"] = brand_data["Date"].dt.year
    # A "complete year" must have data in all 12 months
    month_counts = (
        brand_data.groupby("_year")["Date"]
        .apply(lambda s: s.dt.month.nunique())
    )
    complete_years = month_counts[month_counts == 12].index.tolist()
    if not complete_years:
        return None
    best_year = max(complete_years)
    yd = brand_data[brand_data["_year"] == best_year]
    return (
        best_year,
        float(yd["clicks"].sum()),
        float(yd["quantity"].sum()),
        float(yd["sales"].sum()),
    )


def render_trial_data(profiles, all_brands, min_data_yr, max_data_yr, df_raw):
    """Step 3: Enter trial metrics and calibrate the pipeline."""
    _ensure_lab_state()
    _user_name = st.session_state.get("_user_name", "Unknown")
    _username  = st.session_state.get("_username", "")
    sel_brands = st.session_state.ui_sel_brands

    # Snapshot for undo
    snap_key = "_step_snapshot_nav_trial_data"
    if snap_key not in st.session_state:
        st.session_state[snap_key] = {
            "ui_t_start": st.session_state.ui_t_start,
            "ui_t_end": st.session_state.ui_t_end,
            "ui_c_val": st.session_state.ui_c_val,
            "ui_q_val": st.session_state.ui_q_val,
            "ui_s_val": st.session_state.ui_s_val,
            "ui_adj_c": st.session_state.ui_adj_c,
            "ui_adj_q": st.session_state.ui_adj_q,
            "ui_adj_s": st.session_state.ui_adj_s,
            "ui_trial_mode": st.session_state.ui_trial_mode,
        }

    def _undo():
        snap = st.session_state.get(snap_key, {})
        for k, v in snap.items():
            st.session_state[k] = v
        st.toast("Reverted trial data inputs.")
        st.rerun()

    def _reset():
        from datetime import date as _d
        st.session_state.ui_t_start = _d(2026, 1, 1)
        st.session_state.ui_t_end = _d(2026, 1, 31)
        st.session_state.ui_c_val = 5_000.0
        st.session_state.ui_q_val = 250.0
        st.session_state.ui_s_val = 12_500.0
        st.session_state.ui_adj_c = 0.0
        st.session_state.ui_adj_q = 0.0
        st.session_state.ui_adj_s = 0.0
        st.session_state.ui_trial_mode = "enter"
        st.session_state.step_completed["nav_trial_data"] = False
        # Clear persisted trial values
        _pi = st.session_state.get("_persisted_inputs", {})
        for k in ["ui_t_start", "ui_t_end", "ui_c_val", "ui_q_val", "ui_s_val",
                   "ui_adj_c", "ui_adj_q", "ui_adj_s", "ui_trial_mode"]:
            _pi.pop(k, None)
        st.toast("Trial data reset to defaults.")
        st.rerun()

    _render_step_toolbar("nav_trial_data", _undo, _reset)

    # ── Trial mode selector ──
    _mode_options = ["Enter Trial Data", "Skip — Use Last Year's Data"]
    _mode_idx = 1 if st.session_state.ui_trial_mode == "skip" else 0
    trial_mode = st.radio(
        "Calibration Mode", _mode_options, index=_mode_idx,
        horizontal=True, key="ui_trial_mode_radio",
    )
    is_skip = trial_mode.startswith("Skip")
    st.session_state.ui_trial_mode = "skip" if is_skip else "enter"

    st.markdown("---")

    if is_skip:
        # ── Auto-fill from last complete year ──
        result = _compute_historical_autofill(df_raw, sel_brands)
        if result is None:
            st.error(
                "No complete year of data found for the selected brands. "
                "Please enter trial data manually."
            )
            return

        hist_year, hist_clicks, hist_qty, hist_sales = result
        st.info(
            f"Calibrating with **{hist_year}** full-year historical data. "
            "No trial input needed."
        )
        col_v1, col_v2, col_v3 = st.columns(3)
        col_v1.metric("Clicks", f"{hist_clicks:,.0f}")
        col_v2.metric("Quantity", f"{hist_qty:,.0f}")
        col_v3.metric("Sales", f"\u20ac{hist_sales:,.0f}")

        st.caption(
            f"Trial period: **Jan 1 \u2013 Dec 31, {hist_year}**. "
            "The pipeline will calibrate using this full year of observed data."
        )

        # Auto-fill session state
        from datetime import date as _d
        st.session_state.ui_t_start = _d(hist_year, 1, 1)
        st.session_state.ui_t_end   = _d(hist_year, 12, 31)
        st.session_state.ui_c_val   = hist_clicks
        st.session_state.ui_q_val   = hist_qty
        st.session_state.ui_s_val   = hist_sales
        st.session_state.ui_adj_c   = 0.0
        st.session_state.ui_adj_q   = 0.0
        st.session_state.ui_adj_s   = 0.0

        t_start = st.session_state.ui_t_start
        t_end   = st.session_state.ui_t_end
        c_val   = hist_clicks
        q_val   = hist_qty
        s_val   = hist_sales
    else:
        # ── Manual trial entry (existing UI) ──
        st.markdown("##### Trial Reality")
        st.caption("Enter the observed metrics from your trial period for calibration.")

        col_d1, col_d2 = st.columns(2)
        t_start = col_d1.date_input("Start Date", key="ui_t_start")
        t_end   = col_d2.date_input("End Date",   key="ui_t_end")

        if t_start.year < min_data_yr or t_end.year > max_data_yr + 2:
            st.warning(f"Outside data range ({min_data_yr}\u2013{max_data_yr}).")

        col_m1, col_m2, col_m3 = st.columns(3)
        c_val = col_m1.number_input("Clicks",   min_value=0.0, key="ui_c_val")
        q_val = col_m2.number_input("Quantity", min_value=0.0, key="ui_q_val")
        s_val = col_m3.number_input("Sales",    min_value=0.0, key="ui_s_val")

        # ── Pre-Adjustment ──
        with st.expander("Pre-Adjustment", expanded=False):
            st.caption("+ % = boosted trial (strip lift).  \u2212 % = suppressed (add lift back).")
            ac1, ac2, ac3 = st.columns(3)
            ac1.number_input("Clicks adj (%)",   -100.0, 500.0, key="ui_adj_c", step=5.0)
            ac2.number_input("Quantity adj (%)", -100.0, 500.0, key="ui_adj_q", step=5.0)
            ac3.number_input("Sales adj (%)",    -100.0, 500.0, key="ui_adj_s", step=5.0)

    st.markdown("---")

    # ── DNA Weights Preview ──
    if c_val > 0 or q_val > 0 or s_val > 0:
        proj_year = str(t_start.year)
        _swap_evs = [
            e for e in st.session_state.event_log
            if e.get("type") == "swap" and e.get("scope") == "pre_trial"
        ]
        _profiles_preview = (
            apply_month_swaps(profiles, df_raw, sel_brands, _swap_evs)[0]
            if _swap_evs else profiles
        )
        norm_weights = compute_similarity_weights(
            _profiles_preview, sel_brands, proj_year, t_start, t_end, c_val, q_val, s_val)

        st.markdown("##### DNA Weights")
        st.caption("35% \u2014 All-time overall")
        w_cols = st.columns(min(len(norm_weights), 5))
        for i, (y, w) in enumerate(norm_weights.items()):
            w_cols[i % len(w_cols)].metric(f"Year {y}", f"{w * 65.0:.1f}%")

    st.markdown("---")

    # ── Confirm & Calibrate ──
    if st.button("Confirm & Calibrate \u2192", type="primary", use_container_width=True):
        if _run_trial_calibration(profiles, sel_brands, df_raw):
            _mode_label = "Skip (Historical)" if is_skip else "Manual Entry"
            log_action(
                name=_user_name, username=_username,
                action="Trial Data Completed",
                details=(
                    f"Brands: {', '.join(sel_brands)} | "
                    f"Mode: {_mode_label} | "
                    f"Trial: {t_start} \u2192 {t_end} | "
                    f"Clicks: {c_val:,.0f} | Qty: {q_val:,.0f} | Sales: {s_val:,.0f}"
                ),
            )
            st.session_state.nav_page = "nav_goal_tracker"
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Campaigns
# ═══════════════════════════════════════════════════════════════════════════════

def render_campaigns(df, df_raw, sel_brands, t_start,
                     base_clicks, base_cr, base_aov, settings=None,
                     profiles=None):
    """Render the Campaign injection page with re-projection."""
    _ensure_lab_state()
    if settings is None:
        settings = load_settings()
    _user_name = st.session_state.get("_user_name", "Unknown")
    _username  = st.session_state.get("_username", "")

    # ── Toolbar ──
    def _undo():
        # Remove last shock event
        for i in range(len(st.session_state.event_log) - 1, -1, -1):
            if st.session_state.event_log[i].get("type") == "shock":
                st.session_state.event_log.pop(i)
                break
        else:
            st.toast("No campaigns to undo.")
            return
        # Re-run projections
        cache = st.session_state.pipeline_cache
        if cache.get("df") is not None:
            _dr = cache.get("df_raw_mod") if cache.get("df_raw_mod") is not None else df_raw
            build_projections(cache["df"], cache["base_clicks"], cache["base_cr"],
                              cache["base_aov"], st.session_state.event_log)
            if profiles is not None:
                _pr = cache.get("profiles_mod") if cache.get("profiles_mod") is not None else profiles
                apply_historical_shrinkage(cache["df"], _pr, sel_brands)
            apply_noise_bands(cache["df"], _dr, sel_brands)
        st.toast("Last campaign undone.")
        st.rerun()

    def _reset():
        st.session_state.event_log = [
            e for e in st.session_state.event_log
            if e.get("type") != "shock"
        ]
        st.session_state.step_completed["nav_campaigns"] = False
        # Re-run projections without shocks
        cache = st.session_state.pipeline_cache
        if cache.get("df") is not None:
            _dr = cache.get("df_raw_mod") if cache.get("df_raw_mod") is not None else df_raw
            build_projections(cache["df"], cache["base_clicks"], cache["base_cr"],
                              cache["base_aov"], st.session_state.event_log)
            if profiles is not None:
                _pr = cache.get("profiles_mod") if cache.get("profiles_mod") is not None else profiles
                apply_historical_shrinkage(cache["df"], _pr, sel_brands)
            apply_noise_bands(cache["df"], _dr, sel_brands)
        st.toast("All campaigns cleared.")
        st.rerun()

    _render_step_toolbar("nav_campaigns", _undo, _reset)

    c_start = st.date_input("Start Date", date(t_start.year, 6, 1),  key="ev_start")
    c_end   = st.date_input("End Date",   date(t_start.year, 6, 15), key="ev_end")

    c_shape = st.selectbox("Campaign Shape", list(EVENT_MAPPING.keys()), key="ev_shape")

    _brand_for_default = sel_brands[0] if len(sel_brands) == 1 else "__all__"
    _default_pct = get_campaign_default(settings, _brand_for_default, c_shape)

    c_str_pct = st.slider(
        "Traffic Lift (%)",
        min_value=-100, max_value=300,
        value=_default_pct,
        step=5,
        key=f"ev_str_{c_shape}",
        help="Default loaded from Settings. Adjust as needed.",
    )
    c_str = c_str_pct / 100

    if c_str_pct != _default_pct:
        st.caption(f"Settings default for this shape: **{_default_pct}%**")

    st.markdown("##### Shape Preview")
    sim_d = (c_end - c_start).days + 1
    if sim_d > 0:
        p_df = pd.DataFrame({"Date": pd.date_range(c_start, c_end)})
        p_df["Multiplier"] = p_df["Date"].apply(
            lambda d: get_shock_multiplier(
                d, [{"type": "shock", "start": c_start, "end": c_end,
                      "str": c_str, "shape": c_shape}]))
        fig_p = px.area(
            p_df, x="Date", y="Multiplier",
            title=f"{c_shape} — {c_str*100:.0f}% Lift Profile",
            color_discrete_sequence=[_C_SIM],
        )
        fig_p.update_layout(height=220, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_p, use_container_width=True)

    if st.button("Inject Campaign"):
        duration = (c_end - c_start).days + 1
        est_clicks = round(base_clicks * c_str * duration)
        est_sales  = round(base_clicks * c_str * duration * base_cr * base_aov, 2)
        st.session_state.event_log.append({
            "type": "shock", "start": c_start, "end": c_end,
            "str": c_str, "shape": c_shape,
        })

        # Re-run projections with new campaign
        cache = st.session_state.pipeline_cache
        if cache.get("df") is not None:
            _dr = cache.get("df_raw_mod") if cache.get("df_raw_mod") is not None else df_raw
            build_projections(cache["df"], cache["base_clicks"], cache["base_cr"],
                              cache["base_aov"], st.session_state.event_log)
            if profiles is not None:
                _pr = cache.get("profiles_mod") if cache.get("profiles_mod") is not None else profiles
                apply_historical_shrinkage(cache["df"], _pr, sel_brands)
            apply_noise_bands(cache["df"], _dr, sel_brands)

        log_action(
            name=_user_name, username=_username,
            action="Campaign Injected",
            details=(
                f"Brand: {', '.join(sel_brands)} | "
                f"Shape: {c_shape} | Lift: {c_str_pct}% | "
                f"Period: {c_start} → {c_end} ({duration}d) | "
                f"Est. additional clicks: {est_clicks:,} | "
                f"Est. additional sales: +{est_sales:,.0f}"
            ),
        )
        st.rerun()

    # ── Confirm step complete ──
    st.markdown("---")
    if st.button("Confirm Campaigns →", type="primary", use_container_width=True):
        st.session_state.step_completed["nav_campaigns"] = True
        log_action(
            name=_user_name, username=_username,
            action="Campaigns Step Completed",
            details=f"Brands: {', '.join(sel_brands)} | Events: {len(st.session_state.event_log)}",
        )
        st.session_state.nav_page = "nav_risk"
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Download Strategy
# ═══════════════════════════════════════════════════════════════════════════════

def render_download(df, event_log, sel_brands, t_start, t_end,
                    adj_c, adj_q, adj_s, base_clicks, base_cr, base_aov,
                    proj_year):
    """Step 8: Export the complete strategy report."""

    def _undo():
        st.toast("Nothing to undo — this is the download page.")

    def _reset():
        st.toast("Nothing to reset — this is the download page.")

    _render_step_toolbar("nav_download", _undo, _reset)

    st.subheader("📥 Download Strategy Report")

    st.markdown("##### Report Summary")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Brands", ", ".join(b.title() for b in sel_brands))
    sc2.metric("Trial Period", f"{t_start} → {t_end}")
    sc3.metric("Active Events", str(len(event_log)))

    if df is not None and base_clicks is not None:
        st.markdown("---")
        excel_bytes = build_excel_report(
            df, event_log, sel_brands,
            t_start, t_end, adj_c, adj_q, adj_s,
            base_clicks, base_cr, base_aov,
        )
        st.download_button(
            label="Download Strategy Report (.xlsx)",
            data=excel_bytes,
            file_name=f"strategy_{proj_year}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary",
        )
        st.session_state.step_completed["nav_download"] = True
        log_action(
            name=st.session_state.get("_user_name", ""),
            username=st.session_state.get("_username", ""),
            action="Strategy Report Downloaded",
            details=f"Year: {proj_year} | Brands: {', '.join(sel_brands)}",
        )
    else:
        st.warning("Complete the workflow (Steps 1–7) to generate the report.")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Audit & Gap Attribution
# ═══════════════════════════════════════════════════════════════════════════════

def render_audit(df, pure_dna, adj_c, adj_q, adj_s, t_start, t_end):
    _user_name = st.session_state.get("_user_name", "Unknown")
    _username  = st.session_state.get("_username", "")
    sel_brands = st.session_state.get("ui_sel_brands", [])

    # ── Toolbar ──
    def _undo():
        if not st.session_state.event_log:
            st.toast("Nothing to undo.")
            return
        st.session_state.event_log.pop()
        st.toast("Last event removed.")
        st.rerun()

    def _reset():
        st.toast("Use the individual event delete buttons to manage events.")

    _render_step_toolbar("nav_audit", _undo, _reset)

    st.subheader("Simulation Audit & Gap Attribution")

    if not st.session_state.event_log:
        st.info("No events active.")
        return

    tgt_met  = ("Sales"
                 if st.session_state.target_metric in ["CR", "AOV"]
                 else st.session_state.target_metric)
    base_vol = eval_events(
        [], pure_dna=pure_dna, adj_c=adj_c, adj_q=adj_q, adj_s=adj_s,
        t_start=t_start, t_end=t_end,
        tgt_start=st.session_state.tgt_start, tgt_end=st.session_state.tgt_end,
    )[tgt_met]
    needed_vol = (
        st.session_state.target_val
        if tgt_met == st.session_state.target_metric
        else df[
            (df["Date"].dt.date >= st.session_state.tgt_start) &
            (df["Date"].dt.date <= st.session_state.tgt_end)
        ][f"{tgt_met}_Base"].sum()
    )
    total_gap = needed_vol - base_vol if (needed_vol - base_vol) != 0 else 1.0

    st.markdown(
        f"**Metric:** {tgt_met}  |  **Organic Base:** {base_vol:,.0f}  "
        f"|  **Target:** {needed_vol:,.0f}  |  **Gap:** {needed_vol - base_vol:,.0f}"
    )
    st.markdown("---")

    for i, ev in enumerate(st.session_state.event_log):
        sty = _EV_STYLE.get(ev["type"], {"icon": "•", "label": ev["type"],
                                          "color": "#f1f5f9", "border": "#64748b"})

        vol_prev = eval_events(
            st.session_state.event_log[:i],
            pure_dna=pure_dna, adj_c=adj_c, adj_q=adj_q, adj_s=adj_s,
            t_start=t_start, t_end=t_end,
            tgt_start=st.session_state.tgt_start, tgt_end=st.session_state.tgt_end,
        )[tgt_met]
        vol_curr = eval_events(
            st.session_state.event_log[:i + 1],
            pure_dna=pure_dna, adj_c=adj_c, adj_q=adj_q, adj_s=adj_s,
            t_start=t_start, t_end=t_end,
            tgt_start=st.session_state.tgt_start, tgt_end=st.session_state.tgt_end,
        )[tgt_met]
        added   = vol_curr - vol_prev
        pct_gap = (added / total_gap) * 100 if total_gap else 0

        if ev["type"] == "shock":
            desc = f"{ev.get('shape','?')} | {ev['str']*100:.0f}% | {ev['start']} → {ev['end']}"
            scope_txt = "Post-Trial"
        elif ev["type"] == "custom_drag":
            desc = f"{ev.get('level','?')} {ev['target']} × {ev['lift']:.2f}"
            scope_txt = ev.get("scope", "post_trial").replace("_", " ").title()
        elif ev["type"] == "swap":
            if "a_start" in ev:
                desc = f"{ev.get('level','?')}  {ev['a_start']}–{ev['a_end']} ↔ {ev['b_start']}–{ev['b_end']}"
            else:
                desc = f"{ev.get('level','?')} {ev['a']} ↔ {ev['b']}"
            scope_txt = ev.get("scope", "post_trial").replace("_", " ").title()
        elif ev["type"] == "reapplied_shock":
            desc = f"{ev['name']} | {ev['mode']} | from {ev['new_start']}"
            scope_txt = "Post-Trial"
        else:
            desc = str(ev); scope_txt = ""

        delta_color = "#16a34a" if added >= 0 else "#dc2626"
        sign        = "+" if added >= 0 else ""

        st.markdown(
            f"""<div style="
                background:{sty['color']};
                border-left:4px solid {sty['border']};
                border-radius:8px;
                padding:10px 14px;
                margin-bottom:8px;
                display:flex;
                align-items:center;
                gap:12px;
            ">
              <span style="font-size:1.3em">{sty['icon']}</span>
              <div style="flex:1">
                <strong>{sty['label']}</strong>
                <span style="color:#475569;font-size:0.9em;margin-left:8px">{desc}</span>
                <span style="color:#94a3b8;font-size:0.82em;margin-left:8px">({scope_txt})</span>
              </div>
              <span style="color:{delta_color};font-weight:700;font-size:1.05em">
                {sign}{added:,.0f} ({pct_gap:.1f}%)
              </span>
            </div>""",
            unsafe_allow_html=True,
        )

        act1, act2, act3 = st.columns([1, 1, 6])
        if ev["type"] == "shock":
            if act1.button("↔ Shift", key=f"shift_{i}"):
                st.session_state.shift_target_idx = i
                st.rerun()
        else:
            act1.write("")
        if act2.button("❌", key=f"del_{i}"):
            _ev_del = st.session_state.event_log[i]
            _ev_type = _ev_del.get("type", "unknown")
            _ev_det  = (
                f"shape={_ev_del.get('shape','')}, "
                f"start={_ev_del.get('start','')}, "
                f"end={_ev_del.get('end','')}"
            ) if _ev_type == "shock" else str(_ev_del)
            log_action(
                name=_user_name, username=_username,
                action="Event Deleted",
                details=f"Type: {_ev_type} | {_ev_det}",
            )
            st.session_state.event_log.pop(i)
            st.rerun()
        act3.write("")

    # Shift UI
    if st.session_state.shift_target_idx is not None:
        si = st.session_state.shift_target_idx
        if si < len(st.session_state.event_log):
            sev = st.session_state.event_log[si]
            st.markdown("---")
            st.markdown(f"##### Shift: {sev.get('shape','')} campaign")
            ns = st.date_input("New Start Date", sev["start"], key="shift_new_start")
            if st.button("Apply Shift", key="shift_apply"):
                delta_d = (ns - sev["start"]).days
                sev["start"] = ns
                sev["end"]   = sev["end"] + timedelta(days=delta_d)
                st.session_state.shift_target_idx = None
                log_action(
                    name=_user_name, username=_username,
                    action="Campaign Shifted",
                    details=f"Shape: {sev.get('shape','')} | New: {sev['start']} → {sev['end']}",
                )
                st.rerun()

    # ── Confirm step complete ──
    st.markdown("---")
    if st.button("Confirm Audit →", type="primary", use_container_width=True):
        st.session_state.step_completed["nav_audit"] = True
        log_action(
            name=_user_name, username=_username,
            action="Audit Step Completed",
            details=f"Brands: {', '.join(sel_brands)} | Events: {len(st.session_state.event_log)}",
        )
        st.session_state.nav_page = "nav_download"
        st.rerun()
