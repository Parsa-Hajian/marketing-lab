import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.fmt import _fmt, color_neg
from engine.activity_log import log_action
from engine.monitor_models import MODELS as FORECAST_MODELS, MODEL_INFO

# ── Consistent palette ─────────────────────────────────────────────────────────
_C_BASE   = "#94a3b8"          # slate-grey baseline
_C_SIM    = "#1a1a6b"          # navy simulation
_C_BAND   = "rgba(26,26,107,0.12)"
_C_TARGET = "#dc2626"          # red target
_C_SHOCK  = "rgba(220,38,38,0.10)"   # shock window band
_TEMPLATE = "plotly_white"


def _add_shock_markers(fig, event_log, y_ref="y"):
    """Add semi-transparent vrect for every shock/reapplied_shock event."""
    for ev in event_log:
        if ev["type"] == "shock":
            fig.add_vrect(
                x0=str(ev["start"]), x1=str(ev["end"]),
                fillcolor=_C_SHOCK, layer="below", line_width=0,
                annotation_text=f"📣 {ev.get('shape','')[:8]}",
                annotation_position="top left",
                annotation=dict(font_size=10, font_color="#dc2626"),
            )
        elif ev["type"] == "reapplied_shock":
            from datetime import timedelta
            end_d = ev["new_start"] + timedelta(days=ev["duration"] - 1)
            fig.add_vrect(
                x0=str(ev["new_start"]), x1=str(end_d),
                fillcolor="rgba(16,185,129,0.10)", layer="below", line_width=0,
                annotation_text="💉",
                annotation_position="top left",
                annotation=dict(font_size=10),
            )


def _ensure_state():
    """Defensive session state init."""
    if "event_log"     not in st.session_state: st.session_state.event_log     = []
    if "tgt_start"     not in st.session_state: st.session_state.tgt_start     = None
    if "tgt_end"       not in st.session_state: st.session_state.tgt_end       = None
    if "target_metric" not in st.session_state: st.session_state.target_metric = "Sales"
    if "target_val"    not in st.session_state: st.session_state.target_val    = 0.0
    if "gt_hist_year"   not in st.session_state: st.session_state.gt_hist_year   = None
    if "gt_hist_metric" not in st.session_state: st.session_state.gt_hist_metric = "sales"
    if "gt_growth_pct"  not in st.session_state: st.session_state.gt_growth_pct  = 5.0
    if "gt_vol_driver"  not in st.session_state: st.session_state.gt_vol_driver  = "Traffic (Clicks)"


# ── Goal Tracker (Step 4) ──────────────────────────────────────────────────────

def _render_step_toolbar(step_key, on_undo, on_reset):
    """Render Undo / Reset inline buttons for a workflow step."""
    c_undo, c_reset, _ = st.columns([1, 1, 6])
    if c_undo.button("Undo", key=f"tb_undo_{step_key}", use_container_width=True):
        on_undo()
    if c_reset.button("Reset", key=f"tb_reset_{step_key}", use_container_width=True):
        on_reset()
    st.markdown("---")


def render_goal_tracker(df, df_raw, profiles, yearly_kpis, sel_brands,
                        res_level, time_col, base_cr, base_aov,
                        trial_start, trial_end):
    """Render the Goal Tracker with embedded forecast overview and noise bands."""
    _ensure_state()
    event_log  = st.session_state.event_log
    has_events = bool(event_log)

    # Snapshot for undo
    from datetime import date as _d
    snap_key = "_step_snapshot_nav_goal_tracker"
    if snap_key not in st.session_state:
        st.session_state[snap_key] = {
            "tgt_start": st.session_state.tgt_start,
            "tgt_end": st.session_state.tgt_end,
            "target_metric": st.session_state.target_metric,
            "target_val": st.session_state.target_val,
            "gt_hist_year": st.session_state.gt_hist_year,
            "gt_hist_metric": st.session_state.gt_hist_metric,
            "gt_growth_pct": st.session_state.gt_growth_pct,
            "gt_vol_driver": st.session_state.gt_vol_driver,
        }

    def _undo():
        snap = st.session_state.get(snap_key, {})
        for k, v in snap.items():
            st.session_state[k] = v
        st.toast("Reverted goal tracker settings.")
        st.rerun()

    def _reset():
        st.session_state.tgt_start = _d(2026, 1, 1)
        st.session_state.tgt_end = _d(2026, 12, 31)
        st.session_state.target_metric = "Sales"
        st.session_state.target_val = 200_000.0
        st.session_state.gt_hist_year = None
        st.session_state.gt_hist_metric = "sales"
        st.session_state.gt_growth_pct = 5.0
        st.session_state.gt_vol_driver = "Traffic (Clicks)"
        st.session_state.step_completed["nav_goal_tracker"] = False
        st.toast("Goal tracker reset to defaults.")
        st.rerun()

    _render_step_toolbar("nav_goal_tracker", _undo, _reset)

    # ══════════════════════════════════════════════════════════════════════
    # Section A: Forecast Overview (embedded from old render_projection_overview)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("##### Forecast Overview")

    agg_cols = {
        "Date": "first",
        **{
            f"{m}_{v}": "sum"
            for m in ["Clicks", "Qty", "Sales"]
            for v in ["Base", "Sim", "Base_Min", "Base_Max", "Sim_Min", "Sim_Max"]
        },
    }
    agg_df = df.groupby(time_col).agg(agg_cols).reset_index()

    for pfx in ["_Base", "_Sim", "_Base_Min", "_Base_Max", "_Sim_Min", "_Sim_Max"]:
        agg_df[f"CR{pfx}"] = (
            (agg_df[f"Qty{pfx}"] / agg_df[f"Clicks{pfx}"])
            .replace([float("inf"), float("-inf")], 0).fillna(0)
        )
        agg_df[f"AOV{pfx}"] = (
            (agg_df[f"Sales{pfx}"] / agg_df[f"Qty{pfx}"])
            .replace([float("inf"), float("-inf")], 0).fillna(0)
        )

    met = st.selectbox("Select View Metric", ["Sales", "Clicks", "Qty", "CR", "AOV"],
                        key="fc_overview_met")
    fig = go.Figure()

    if has_events:
        fig.add_trace(go.Scatter(
            x=agg_df["Date"], y=agg_df[f"{met}_Base_Max"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=agg_df["Date"], y=agg_df[f"{met}_Base_Min"],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(148,163,184,0.15)",
            name="Baseline Noise Band"))
        fig.add_trace(go.Scatter(
            x=agg_df["Date"], y=agg_df[f"{met}_Base"],
            mode="lines", line=dict(color=_C_BASE, dash="dot", width=2),
            name="Baseline (Before)"))
        fig.add_trace(go.Scatter(
            x=agg_df["Date"], y=agg_df[f"{met}_Sim_Max"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=agg_df["Date"], y=agg_df[f"{met}_Sim_Min"],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=_C_BAND,
            name="Forecast Noise Band"))
        fig.add_trace(go.Scatter(
            x=agg_df["Date"], y=agg_df[f"{met}_Sim"],
            mode="lines+markers", line=dict(color=_C_SIM, width=3),
            name="Forecast (After Events)"))
        _add_shock_markers(fig, event_log)
    else:
        fig.add_trace(go.Scatter(
            x=agg_df["Date"], y=agg_df[f"{met}_Base_Max"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=agg_df["Date"], y=agg_df[f"{met}_Base_Min"],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(148,163,184,0.15)",
            name="Noise Band"))
        fig.add_trace(go.Scatter(
            x=agg_df["Date"], y=agg_df[f"{met}_Base"],
            mode="lines+markers", line=dict(color=_C_BASE, width=3),
            name="Baseline (No Events)"))

    # Mark trial period on forecast chart
    fig.add_vrect(
        x0=str(trial_start), x1=str(trial_end),
        fillcolor="rgba(244,121,32,0.08)", layer="below", line_width=0,
        annotation_text="Trial Period",
        annotation_position="top left",
        annotation=dict(font_size=9, font_color="#F47920"),
    )

    fig.update_layout(
        template=_TEMPLATE,
        title=dict(text=f"{res_level} Forecast: {met}", font=dict(size=18, color="#12124a")),
        xaxis_title="Date", yaxis_title=met,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    if has_events:
        st.caption(
            f"Showing **{len(event_log)} event(s)** — shaded bands show campaign windows.")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════
    # Section B: Goal Tracker
    # ══════════════════════════════════════════════════════════════════════

    calculated_target = st.session_state.target_val
    target_metric_raw = "sales"
    metric_map = {
        "sales": "Sales", "quantity": "Quantity", "clicks": "Clicks",
        "cr": "CR", "aov": "AOV",
    }

    _gt_hist_year   = st.session_state.gt_hist_year
    _gt_hist_metric = st.session_state.gt_hist_metric
    _gt_growth_pct  = st.session_state.gt_growth_pct

    if len(sel_brands) == 1:
        st.info(f"📈 **Single Brand Mode:** {sel_brands[0].title()}")
        brand_hist = yearly_kpis[yearly_kpis["brand"] == sel_brands[0]].sort_values("Year")

        year_opts = brand_hist["Year"].unique().tolist()
        saved_yr  = st.session_state.gt_hist_year
        yr_idx    = year_opts.index(saved_yr) if saved_yr in year_opts else 0

        met_opts  = ["sales", "quantity", "clicks", "cr", "aov"]
        saved_met = st.session_state.gt_hist_metric
        met_idx   = met_opts.index(saved_met) if saved_met in met_opts else 0

        h1, h2, h3 = st.columns(3)
        hist_year         = h1.selectbox("Base Year", year_opts, index=yr_idx)
        target_metric_raw = h2.selectbox("Historical Metric", met_opts, index=met_idx)
        growth_pct        = h3.number_input(
            "Growth vs Year (%)", value=float(st.session_state.gt_growth_pct), step=1.0)

        _gt_hist_year   = hist_year
        _gt_hist_metric = target_metric_raw
        _gt_growth_pct  = growth_pct

        fig_h = px.bar(
            brand_hist, x="Year", y=target_metric_raw,
            title=f"Historical {metric_map[target_metric_raw]}",
            text_auto=".2s", color_discrete_sequence=["#1a1a6b"],
        )
        fig_h.update_layout(template=_TEMPLATE, height=300)
        c_ch, c_tb = st.columns([2, 1])
        with c_ch:
            st.plotly_chart(fig_h, use_container_width=True)
        with c_tb:
            fmt_s = (
                "{:.2%}" if target_metric_raw == "cr"
                else "€{:,.2f}" if target_metric_raw == "aov"
                else "{:,.0f}"
            )
            st.dataframe(
                brand_hist[["Year", target_metric_raw]]
                .style.format({target_metric_raw: fmt_s}),
                use_container_width=True, hide_index=True,
            )

        base_hist = brand_hist[brand_hist["Year"] == hist_year][target_metric_raw].values[0]
        calculated_target = base_hist * (1 + growth_pct / 100)
        if target_metric_raw == "cr":
            st.success(f"Target: **{calculated_target:.2%}** {metric_map[target_metric_raw]}")
        elif target_metric_raw == "aov":
            st.success(f"Target: **€{calculated_target:,.2f}** {metric_map[target_metric_raw]}")
        else:
            st.success(f"Target: **{calculated_target:,.0f}** {metric_map[target_metric_raw]}")
        st.markdown("---")

    # ── Target period selection (trial dates prohibited) ──
    col_d1, col_d2 = st.columns(2)
    st.session_state.tgt_start = col_d1.date_input(
        "Target Period Start", st.session_state.tgt_start)
    st.session_state.tgt_end   = col_d2.date_input(
        "Target Period End",   st.session_state.tgt_end)

    # Trial exclusion check
    if (st.session_state.tgt_start <= trial_end and
        st.session_state.tgt_end >= trial_start):
        st.error(
            f"Target period overlaps with trial period ({trial_start} → {trial_end}). "
            "The trial period should not be included in the forecast target because "
            "trial data is used for calibration, not prediction. "
            "Please adjust the target period."
        )

    col_m1, col_m2, col_m3 = st.columns(3)
    m_opts = ["Sales", "Quantity", "Clicks", "CR", "AOV"]
    default_idx = (
        m_opts.index(metric_map[target_metric_raw])
        if len(sel_brands) == 1 and metric_map[target_metric_raw] in m_opts
        else m_opts.index(st.session_state.target_metric)
        if st.session_state.target_metric in m_opts else 0
    )
    st.session_state.target_metric = col_m1.selectbox(
        "Final Target Metric", m_opts, index=default_idx, key="dash_met")

    volume_driver = st.session_state.gt_vol_driver
    if st.session_state.target_metric in ["Sales", "Quantity"]:
        d_opts = (
            ["Traffic (Clicks)", "Conversion Rate (CR)", "Average Order Value (AOV)"]
            if st.session_state.target_metric == "Sales"
            else ["Traffic (Clicks)", "Conversion Rate (CR)"]
        )
        saved_vd = st.session_state.gt_vol_driver
        vd_idx   = d_opts.index(saved_vd) if saved_vd in d_opts else 0
        volume_driver = col_m2.selectbox("Scale via:", d_opts, index=vd_idx)

    if st.session_state.target_metric == "CR":
        st.session_state.target_val = col_m3.number_input(
            "Desired CR",
            value=float(calculated_target if len(sel_brands) == 1 else st.session_state.target_val),
            step=0.01, format="%.4f",
        )
    elif st.session_state.target_metric == "AOV":
        st.session_state.target_val = col_m3.number_input(
            "Desired AOV (€)",
            value=float(calculated_target if len(sel_brands) == 1 else st.session_state.target_val),
            step=1.0,
        )
    else:
        st.session_state.target_val = col_m3.number_input(
            f"Desired {st.session_state.target_metric}",
            value=float(calculated_target if len(sel_brands) == 1 else st.session_state.target_val),
            step=1000.0,
        )

    sv_col, _ = st.columns([1, 3])
    if sv_col.button("💾 Save settings", key="gt_save", use_container_width=True):
        st.session_state.gt_hist_year   = _gt_hist_year
        st.session_state.gt_hist_metric = _gt_hist_metric
        st.session_state.gt_growth_pct  = _gt_growth_pct
        st.session_state.gt_vol_driver  = volume_driver
        log_action(
            name=st.session_state.get("_user_name", "Unknown"),
            username=st.session_state.get("_username", ""),
            action="Goal Tracker: Settings Saved",
            details=(
                f"Brand: {', '.join(sel_brands)} | "
                f"Base year: {_gt_hist_year} | "
                f"Historical metric: {_gt_hist_metric} | "
                f"Growth %: {_gt_growth_pct:.1f}% | "
                f"Target metric: {st.session_state.target_metric} | "
                f"Target value: {st.session_state.target_val:,.2f} | "
                f"Period: {st.session_state.tgt_start} → {st.session_state.tgt_end} | "
                f"Volume driver: {volume_driver}"
            ),
        )
        st.toast("Goal Tracker settings saved.", icon="✅")

    # ── Forecast Model Selection ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("##### Forecast Model")
    st.caption("Select one or more models. Multiple selections are averaged. "
               "Hover the **ⓘ** icon for each model's strengths & weaknesses.")

    _ALL_MODEL_OPTS = {
        "DNA Pipeline": {
            "strength": "Brand-specific seasonal DNA calibrated from trial data",
            "weakness": "Sensitive to trial window size & extrapolation",
        },
        **MODEL_INFO,
    }

    _model_names = list(_ALL_MODEL_OPTS.keys())
    _selected_models = []

    # Row 1: DNA Pipeline + first 3 statistical models
    _r1 = st.columns(4)
    for i, col in enumerate(_r1):
        if i < len(_model_names):
            nm = _model_names[i]
            info = _ALL_MODEL_OPTS[nm]
            if col.checkbox(
                nm, value=(nm == "DNA Pipeline"),
                key=f"gt_mdl_{nm}",
                help=(f"**Strength:** {info['strength']}\n\n"
                      f"**Weakness:** {info['weakness']}"),
            ):
                _selected_models.append(nm)

    # Row 2: remaining models
    _r2 = st.columns(4)
    for i, col in enumerate(_r2):
        idx = i + 4
        if idx < len(_model_names):
            nm = _model_names[idx]
            info = _ALL_MODEL_OPTS[nm]
            if col.checkbox(
                nm, value=False,
                key=f"gt_mdl_{nm}",
                help=(f"**Strength:** {info['strength']}\n\n"
                      f"**Weakness:** {info['weakness']}"),
            ):
                _selected_models.append(nm)

    if not _selected_models:
        _selected_models = ["DNA Pipeline"]
        st.info("No model selected — defaulting to DNA Pipeline.")

    # ── Build df_tgt ──────────────────────────────────────────────────────
    df_tgt = df[
        (df["Date"].dt.date >= st.session_state.tgt_start) &
        (df["Date"].dt.date <= st.session_state.tgt_end)
    ].copy()

    if df_tgt.empty:
        st.warning("Target period has no data in the projection year.")
        return

    # ── Run statistical model forecasts & blend into df_tgt ───────────────
    _stat_models = [m for m in _selected_models if m != "DNA Pipeline"]
    _use_dna = "DNA Pipeline" in _selected_models

    if _stat_models:
        # Build historical monthly series from raw data
        raw_brand = df_raw[df_raw["brand"].isin(sel_brands)].copy()
        raw_brand["Date"] = pd.to_datetime(raw_brand["Date"])
        _hist_monthly = (
            raw_brand.groupby(pd.Grouper(key="Date", freq="MS"))
            .agg(clicks=("clicks", "sum"), quantity=("quantity", "sum"),
                 sales=("sales", "sum"))
            .reset_index().sort_values("Date")
        )
        _hist_monthly = _hist_monthly[_hist_monthly["clicks"] > 0]

        if _hist_monthly.empty:
            st.warning("No historical monthly data found — model forecasts skipped.")
        else:
            _last_hist = pd.Timestamp(_hist_monthly["Date"].max())
            _tgt_end_ts = pd.Timestamp(st.session_state.tgt_end)
            _n_ahead = max(1,
                           (_tgt_end_ts.year - _last_hist.year) * 12
                           + _tgt_end_ts.month - _last_hist.month)
            _future_dates = pd.date_range(
                _last_hist + pd.DateOffset(months=1),
                periods=_n_ahead, freq="MS",
            )

            # Map (year, month) → index for robust cross-year mapping
            _fc_ym_map = {(d.year, d.month): i
                          for i, d in enumerate(_future_dates)}

            _ok_models = []
            _fail_models = []

            for met_raw, sim_col in [("clicks", "Clicks_Sim"),
                                      ("quantity", "Qty_Sim"),
                                      ("sales", "Sales_Sim")]:
                _mdf = pd.DataFrame({
                    "Date": _hist_monthly["Date"].values,
                    "value": _hist_monthly[met_raw].values,
                })
                _preds = []
                _pred_names = []
                for mdl_name in _stat_models:
                    fn = FORECAST_MODELS.get(mdl_name)
                    if fn is None:
                        continue
                    try:
                        result = fn(_mdf.copy(), _n_ahead)
                        _preds.append(result)
                        _pred_names.append(mdl_name)
                    except Exception as exc:
                        if mdl_name not in _fail_models:
                            _fail_models.append(mdl_name)

                if _preds:
                    for pn in _pred_names:
                        if pn not in _ok_models:
                            _ok_models.append(pn)

                if not _preds:
                    continue

                _avg_pred = np.mean(_preds, axis=0)

                # Apply per-month scaling to target period
                _seen_ym = set()
                for _, row in df_tgt[["Date"]].iterrows():
                    ym = (row["Date"].year, row["Date"].month)
                    if ym in _seen_ym:
                        continue
                    _seen_ym.add(ym)
                    fi = _fc_ym_map.get(ym)
                    if fi is None or fi >= len(_avg_pred):
                        continue
                    model_total = float(_avg_pred[fi])
                    month_mask = (df_tgt["Date"].dt.year == ym[0]) & \
                                 (df_tgt["Date"].dt.month == ym[1])
                    dna_total = df_tgt.loc[month_mask, sim_col].sum()
                    if dna_total <= 0 or model_total <= 0:
                        continue

                    if _use_dna:
                        n_models = len(_preds) + 1
                        blended = (dna_total + model_total * len(_preds)) / n_models
                    else:
                        blended = model_total

                    scale = blended / dna_total
                    df_tgt.loc[month_mask, sim_col] *= scale
                    for sfx in ("_Sim_Min", "_Sim_Max"):
                        c = sim_col.replace("_Sim", sfx)
                        if c in df_tgt.columns:
                            df_tgt.loc[month_mask, c] *= scale

            # User feedback
            if _ok_models:
                st.success(f"Models ran: {', '.join(_ok_models)}")
            if _fail_models:
                st.warning(f"Models failed: {', '.join(_fail_models)}")

    _model_label = ", ".join(_selected_models)
    st.caption(f"**Active model(s):** {_model_label}")
    st.markdown("---")

    tgt_sales_base  = df_tgt["Sales_Base"].sum()
    tgt_qty_base    = df_tgt["Qty_Base"].sum()
    tgt_clicks_base = df_tgt["Clicks_Base"].sum()
    tgt_eff_aov     = tgt_sales_base / tgt_qty_base    if tgt_qty_base    > 0 else base_aov
    tgt_eff_cr      = tgt_qty_base   / tgt_clicks_base if tgt_clicks_base > 0 else base_cr

    needed_sales = needed_qty = needed_clicks = 0.0
    tm = st.session_state.target_metric
    tv = st.session_state.target_val

    if tm == "Sales":
        if volume_driver == "Traffic (Clicks)":
            needed_sales  = tv
            needed_qty    = needed_sales  / tgt_eff_aov if tgt_eff_aov > 0 else 0
            needed_clicks = needed_qty    / tgt_eff_cr  if tgt_eff_cr  > 0 else 0
        elif volume_driver == "Conversion Rate (CR)":
            needed_clicks = tgt_clicks_base; needed_sales = tv
            needed_qty    = needed_sales / tgt_eff_aov if tgt_eff_aov > 0 else 0
        else:
            needed_clicks = tgt_clicks_base; needed_qty = tgt_qty_base; needed_sales = tv
    elif tm == "Quantity":
        if volume_driver == "Traffic (Clicks)":
            needed_qty    = tv
            needed_clicks = needed_qty / tgt_eff_cr  if tgt_eff_cr  > 0 else 0
            needed_sales  = needed_qty * tgt_eff_aov
        else:
            needed_clicks = tgt_clicks_base; needed_qty = tv; needed_sales = tv * tgt_eff_aov
    elif tm == "Clicks":
        needed_clicks = tv; needed_qty = tv * tgt_eff_cr; needed_sales = needed_qty * tgt_eff_aov
    elif tm == "CR":
        needed_clicks = tgt_clicks_base; needed_qty = tgt_clicks_base * tv
        needed_sales  = needed_qty * tgt_eff_aov
    else:  # AOV
        needed_qty    = tgt_qty_base; needed_clicks = tgt_clicks_base
        needed_sales  = needed_qty * tv

    needed_cr  = needed_qty / needed_clicks if needed_clicks > 0 else 0
    needed_aov = needed_sales / needed_qty  if needed_qty    > 0 else 0

    s_c   = df_tgt["Clicks_Sim"].sum()
    s_q   = df_tgt["Qty_Sim"].sum()
    s_s   = df_tgt["Sales_Sim"].sum()
    s_cr  = s_q / s_c if s_c > 0 else 0
    s_aov = s_s / s_q if s_q > 0 else 0

    # Show forecast line whenever events exist OR a statistical model is active
    _show_forecast = has_events or bool(_stat_models)
    _fc_label = "Forecast" if bool(_stat_models) else "After"

    st.markdown(
        "<div style='margin:16px 0 6px;font-family:Inter,sans-serif;"
        "font-size:0.7rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.12em;color:#AAAAAA'>Needed</div>",
        unsafe_allow_html=True,
    )
    kpi_cols = st.columns(5)
    labels = ["Traffic", "Orders", "Revenue", "CR", "AOV"]
    needed = [needed_clicks, needed_qty, needed_sales, needed_cr, needed_aov]
    before = [tgt_clicks_base, tgt_qty_base, tgt_sales_base, tgt_eff_cr, tgt_eff_aov]
    after  = [s_c, s_q, s_s, s_cr, s_aov]

    for col, lbl, n_v, b_v, a_v in zip(kpi_cols, labels, needed, before, after):
        col.metric(lbl, _fmt(lbl, n_v))
        col.caption(f"**Before:** {_fmt(lbl, b_v)}")
        if _show_forecast:
            delta = a_v - b_v
            sign  = "+" if delta >= 0 else ""
            col.caption(f"**{_fc_label}:** {_fmt(lbl, a_v)} *(Δ {sign}{_fmt(lbl, delta)})*")

    agg_tgt = df_tgt.groupby(time_col).agg({
        "Date": "first",
        "Clicks_Sim": "sum", "Qty_Sim": "sum", "Sales_Sim": "sum",
        "Clicks_Base": "sum", "Qty_Base": "sum", "Sales_Base": "sum",
        "Clicks_Sim_Min": "sum", "Clicks_Sim_Max": "sum",
        "Qty_Sim_Min": "sum", "Qty_Sim_Max": "sum",
        "Sales_Sim_Min": "sum", "Sales_Sim_Max": "sum",
    }).reset_index()

    for col_n, total in [("Needed_Sales", needed_sales),
                          ("Needed_Qty",   needed_qty),
                          ("Needed_Clicks", needed_clicks)]:
        base_c = col_n.replace("Needed_", "") + "_Base"
        bt     = agg_tgt[base_c].sum()
        agg_tgt[col_n] = total * (agg_tgt[base_c] / bt) if bt > 0 else 0

    for sfx in ["Base", "Sim"]:
        agg_tgt[f"Gap_Sales_{sfx}"]  = agg_tgt[f"Sales_{sfx}"]  - agg_tgt["Needed_Sales"]
        agg_tgt[f"Gap_Qty_{sfx}"]    = agg_tgt[f"Qty_{sfx}"]    - agg_tgt["Needed_Qty"]
        agg_tgt[f"Gap_Clicks_{sfx}"] = agg_tgt[f"Clicks_{sfx}"] - agg_tgt["Needed_Clicks"]

    st.markdown("---")
    gap_m = st.radio("Visualize Tracking For:", ["Sales", "Qty", "Clicks"], horizontal=True)

    # ── Needed vs Forecasted chart with noise bands ──
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=agg_tgt["Date"], y=agg_tgt[f"Needed_{gap_m}"],
        mode="lines", line=dict(color=_C_TARGET, dash="dash", width=2), name="Target"))
    fig_ts.add_trace(go.Scatter(
        x=agg_tgt["Date"], y=agg_tgt[f"{gap_m}_Base"],
        mode="lines", line=dict(color=_C_BASE, dash="dot", width=2), name="Before"))
    # Noise band
    fig_ts.add_trace(go.Scatter(
        x=agg_tgt["Date"], y=agg_tgt[f"{gap_m}_Sim_Max"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig_ts.add_trace(go.Scatter(
        x=agg_tgt["Date"], y=agg_tgt[f"{gap_m}_Sim_Min"],
        mode="lines", line=dict(width=0),
        fill="tonexty",
        fillcolor=_C_BAND if _show_forecast else "rgba(148,163,184,0.15)",
        name="Noise Band"))
    if _show_forecast:
        fig_ts.add_trace(go.Scatter(
            x=agg_tgt["Date"], y=agg_tgt[f"{gap_m}_Sim"],
            mode="lines+markers", line=dict(color=_C_SIM, width=2), name=_fc_label))
    if has_events:
        _add_shock_markers(fig_ts, event_log)

    fig_ts.update_layout(
        template=_TEMPLATE, height=350,
        title=dict(text=f"Needed vs Forecasted — {gap_m}", font=dict(color="#12124a")),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    fig_gap = go.Figure()
    fig_gap.add_trace(go.Bar(
        x=agg_tgt["Date"], y=agg_tgt[f"Gap_{gap_m}_Base"],
        name="Gap (Before)", marker_color="#cbd5e1"))
    if _show_forecast:
        colors = [
            "#f87171" if v < 0 else "#4ade80"
            for v in agg_tgt[f"Gap_{gap_m}_Sim"]
        ]
        fig_gap.add_trace(go.Bar(
            x=agg_tgt["Date"], y=agg_tgt[f"Gap_{gap_m}_Sim"],
            name=f"Gap ({_fc_label})", marker_color=colors))
    fig_gap.update_layout(
        template=_TEMPLATE, barmode="group", height=280,
        title=dict(text=f"{gap_m} Surplus / Shortfall", font=dict(color="#12124a")),
    )
    st.plotly_chart(fig_gap, use_container_width=True)

    if _show_forecast:
        disp_cols = [
            "Date",
            "Needed_Sales",  "Sales_Base",  "Sales_Sim",  "Gap_Sales_Base",  "Gap_Sales_Sim",
            "Needed_Qty",    "Qty_Base",    "Qty_Sim",    "Gap_Qty_Base",    "Gap_Qty_Sim",
            "Needed_Clicks", "Clicks_Base", "Clicks_Sim", "Gap_Clicks_Base", "Gap_Clicks_Sim",
        ]
        gap_cols = [
            "Gap_Sales_Base", "Gap_Sales_Sim",
            "Gap_Qty_Base",   "Gap_Qty_Sim",
            "Gap_Clicks_Base","Gap_Clicks_Sim",
        ]
    else:
        disp_cols = [
            "Date",
            "Needed_Sales", "Sales_Base",  "Gap_Sales_Base",
            "Needed_Qty",   "Qty_Base",    "Gap_Qty_Base",
            "Needed_Clicks","Clicks_Base", "Gap_Clicks_Base",
        ]
        gap_cols = ["Gap_Sales_Base", "Gap_Qty_Base", "Gap_Clicks_Base"]

    disp_df = agg_tgt[disp_cols].copy()

    # Clean date: date-only, no time component
    disp_df["Date"] = pd.to_datetime(disp_df["Date"]).dt.date

    # Round all numeric columns to integers (except CR columns if any)
    num_cols = [c for c in disp_df.columns if c != "Date"]
    for c in num_cols:
        disp_df[c] = disp_df[c].round(0).astype(int)

    try:
        st.dataframe(disp_df.style.map(color_neg, subset=gap_cols), use_container_width=True)
    except AttributeError:
        st.dataframe(
            disp_df.style.applymap(color_neg, subset=gap_cols), use_container_width=True)

    # ── Confirm step complete ──
    st.markdown("---")
    if st.button("Confirm Goal Tracker →", type="primary", use_container_width=True):
        st.session_state.step_completed["nav_goal_tracker"] = True
        log_action(
            name=st.session_state.get("_user_name", "Unknown"),
            username=st.session_state.get("_username", ""),
            action="Goal Tracker Completed",
            details=(
                f"Brands: {', '.join(sel_brands)} | "
                f"Target: {st.session_state.target_metric} = {st.session_state.target_val:,.2f} | "
                f"Period: {st.session_state.tgt_start} → {st.session_state.tgt_end}"
            ),
        )
        st.session_state.nav_page = "nav_campaigns"
        st.rerun()
