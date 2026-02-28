import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
import time

from config import EVENT_MAPPING
from engine.dna import _apply_dna_ev
from engine.simulation import get_shock_multiplier, eval_events


def render_lab(df, df_raw, sel_brands, res_level, time_col,
               base_clicks, base_cr, base_aov, adj_c, adj_q, adj_s,
               t_start, t_end, pure_dna):
    """Render the Event Simulation Lab page (4 tabs)."""
    st.header("Simulation Lab & DNA Editor")

    t_custom, t_events, t_deshock, t_log = st.tabs([
        "🖱️ Visual DNA Drag",
        "🚀 Events",
        "🧹 De-Shock Tool & Signature Library",
        "📋 Audit & Gap Attribution",
    ])

    # ── Tab 1: Visual DNA Drag ──────────────────────────────────────────────
    with t_custom:
        st.subheader("Interactive DNA Sculpting")
        dna_plot = df.groupby(time_col).agg({
            "idx_clicks_pure": "mean",
            "idx_clicks_work": "mean",
        }).reset_index()

        fig_i = go.Figure()
        fig_i.add_trace(go.Scatter(
            x=dna_plot[time_col], y=dna_plot["idx_clicks_pure"],
            mode="lines", line=dict(color="gray", dash="dot"),
            name="Pure DNA (Before)"))
        fig_i.add_trace(go.Scatter(
            x=dna_plot[time_col], y=dna_plot["idx_clicks_work"],
            mode="lines+markers", line=dict(color="navy"),
            name="Sculpted DNA (After)"))
        fig_i.update_layout(
            template="plotly_white",
            title=f"Select & Sculpt {res_level} DNA",
            hovermode="x unified")

        target_idx = None
        try:
            sel = st.plotly_chart(
                fig_i, on_select="rerun", selection_mode="points", key="dna_plot")
            if sel and sel.get("selection", {}).get("points"):
                target_idx = sel["selection"]["points"][0]["x"]
        except TypeError:
            st.warning("Update Streamlit for visual click dragging.")

        st.markdown("---")
        col_a, col_b, col_sc = st.columns(3)
        cd_target = col_a.number_input(
            f"Target {res_level}", min_value=1,
            value=int(target_idx) if target_idx else 1)
        cd_lift = col_b.slider("Multiplier (×)", 0.0, 5.0, 1.0, step=0.05)
        scope_c = col_sc.radio(
            "When to apply",
            ["Pre-Trial (affects calibration)", "Post-Trial (projection only)"],
            key="drag_scope")
        scope_val = "pre_trial" if "Pre" in scope_c else "post_trial"

        if st.button("🔨 Apply Structural Customization"):
            st.session_state.event_log.append({
                "type": "custom_drag", "level": res_level,
                "target": cd_target, "lift": cd_lift, "scope": scope_val,
            })
            st.rerun()

    # ── Tab 2: Events ───────────────────────────────────────────────────────
    with t_events:
        col_c, col_s = st.columns(2)

        with col_c:
            st.subheader("Add Time-Bound Campaign")
            c_start = st.date_input("Start Date", date(t_start.year, 6, 1),  key="ev_start")
            c_end   = st.date_input("End Date",   date(t_start.year, 6, 15), key="ev_end")
            c_str   = st.slider("Traffic Lift (%)", -100, 300, 25, step=5) / 100
            c_shape = st.selectbox("Campaign Shape", list(EVENT_MAPPING.keys()))

            # Shape preview
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
                    title=f"{c_shape} — {c_str*100:.0f}% Lift Profile")
                fig_p.update_layout(height=220, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_p, use_container_width=True)

            if st.button("➕ Inject Campaign"):
                st.session_state.event_log.append({
                    "type": "shock", "start": c_start, "end": c_end,
                    "str": c_str, "shape": c_shape,
                })
                st.rerun()

        with col_s:
            st.subheader("Swap Time Periods")
            swap_a = st.number_input(f"{res_level} A", min_value=1, value=1)
            swap_b = st.number_input(f"{res_level} B", min_value=1, value=2)
            swap_sc = st.radio(
                "When to apply",
                ["Pre-Trial (affects calibration)", "Post-Trial (projection only)"],
                key="swap_scope")
            swap_scope = "pre_trial" if "Pre" in swap_sc else "post_trial"
            if st.button("🔄 Execute DNA Swap"):
                st.session_state.event_log.append({
                    "type": "swap", "level": res_level,
                    "a": swap_a, "b": swap_b, "scope": swap_scope,
                })
                st.rerun()

    # ── Tab 3: De-Shock Tool & Signature Library ────────────────────────────
    with t_deshock:
        _render_deshock(df, df_raw, sel_brands, t_start)

    # ── Tab 4: Audit & Gap Attribution ─────────────────────────────────────
    with t_log:
        _render_audit(df, pure_dna, adj_c, adj_q, adj_s, t_start, t_end)


# ── De-Shock Tool ───────────────────────────────────────────────────────────

def _render_deshock(df, df_raw, sel_brands, t_start):
    st.subheader("🧹 Isolate & Extract Historical Shocks")

    # Data coverage info
    available_start = df_raw[df_raw["brand"].isin(sel_brands)]["Date"].min()
    available_end   = df_raw[df_raw["brand"].isin(sel_brands)]["Date"].max()
    if pd.isna(available_start):
        st.warning("No raw data available for the selected brands.")
        return

    st.info(
        f"📅 **Available data range for selected brands:** "
        f"{available_start.date()} → {available_end.date()}"
    )

    col1, col2 = st.columns(2)
    # Sensible defaults: last November within data range
    default_yr   = available_end.year if available_end.month >= 11 else available_end.year - 1
    default_yr   = max(default_yr, available_start.year)
    ds_start_def = date(default_yr, 11, 20)
    ds_end_def   = date(default_yr, 11, 30)

    ds_start = col1.date_input("Shock Window Start", ds_start_def, key="ds_start")
    ds_end   = col2.date_input("Shock Window End",   ds_end_def,   key="ds_end")

    if ds_start > ds_end:
        st.error("Start must be before End.")
        return

    # Validate against available data
    if pd.Timestamp(ds_start) > available_end or pd.Timestamp(ds_end) < available_start:
        st.warning(
            f"Selected window ({ds_start} → {ds_end}) is outside available data range "
            f"({available_start.date()} → {available_end.date()})."
        )
        return

    # Context window ±14 calendar days
    ctx_start = ds_start - timedelta(days=14)
    ctx_end   = ds_end   + timedelta(days=14)

    # Filter from super_dataset.csv (real daily observed values)
    ctx_mask = (
        df_raw["brand"].isin(sel_brands) &
        (df_raw["Date"] >= pd.Timestamp(ctx_start)) &
        (df_raw["Date"] <= pd.Timestamp(ctx_end))
    )
    ctx_raw = (
        df_raw[ctx_mask]
        .groupby("Date")
        .agg({"clicks": "sum", "quantity": "sum", "sales": "sum"})
        .reset_index()
    )

    if ctx_raw.empty:
        st.warning("No raw data found for the selected period. Try a different date range.")
        return

    shock_raw = ctx_raw[
        (ctx_raw["Date"] >= pd.Timestamp(ds_start)) &
        (ctx_raw["Date"] <= pd.Timestamp(ds_end))
    ].copy()

    if shock_raw.empty:
        st.warning("No data in the shock window itself (context window has data).")
        return

    # Organic floor — 10th percentile of shock window (per spec)
    floor_c = shock_raw["clicks"].quantile(0.10)
    floor_q = shock_raw["quantity"].quantile(0.10)
    floor_s = shock_raw["sales"].quantile(0.10)

    shock_raw["delta_c"] = np.maximum(0, shock_raw["clicks"]   - floor_c)
    shock_raw["delta_q"] = np.maximum(0, shock_raw["quantity"] - floor_q)
    shock_raw["delta_s"] = np.maximum(0, shock_raw["sales"]    - floor_s)

    # ── Context window chart ──
    fig_ds = go.Figure()
    fig_ds.add_trace(go.Scatter(
        x=ctx_raw["Date"], y=ctx_raw["clicks"],
        mode="lines", name="Historical Traffic", line=dict(color="gray")))
    # Shade shock delta area
    fig_ds.add_trace(go.Scatter(
        x=shock_raw["Date"], y=shock_raw["clicks"],
        mode="lines", showlegend=False, line=dict(color="rgba(0,0,0,0)")))
    fig_ds.add_trace(go.Scatter(
        x=shock_raw["Date"], y=[floor_c] * len(shock_raw),
        mode="lines", fill="tonexty",
        fillcolor="rgba(33,195,84,0.4)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Extracted Shock Delta"))
    # Organic floor line
    fig_ds.add_shape(
        type="line",
        x0=str(ds_start), x1=str(ds_end),
        y0=floor_c, y1=floor_c,
        line=dict(color="red", dash="dash"),
    )
    fig_ds.add_trace(go.Scatter(
        x=[ds_start, ds_end], y=[floor_c, floor_c],
        mode="lines", name="Organic Floor (10th pct)",
        line=dict(color="red", dash="dash")))
    fig_ds.update_layout(
        template="plotly_white", height=320,
        title="De-Shock Isolation View (+/- 14-day context window)")
    st.plotly_chart(fig_ds, use_container_width=True)

    tot_delta_c = shock_raw["delta_c"].sum()
    tot_delta_q = shock_raw["delta_q"].sum()
    tot_delta_s = shock_raw["delta_s"].sum()
    ds_dur      = (ds_end - ds_start).days + 1

    if tot_delta_c <= 0:
        st.warning("No significant shock detected above the organic floor in this window.")
        return

    # ── Forensic Details ──
    organic_cr = floor_q / floor_c if floor_c > 0 else 0
    event_cr   = tot_delta_q / tot_delta_c if tot_delta_c > 0 else 0
    cr_delta   = event_cr - organic_cr

    st.markdown("#### 🔬 Forensic Details")
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Δ Clicks (Artificial)",       f"+{tot_delta_c:,.0f}")
    fc1.metric("Δ Orders (Artificial)",        f"+{tot_delta_q:,.0f}")
    fc2.metric("Δ Sales (Artificial)",         f"+€{tot_delta_s:,.0f}")
    fc2.metric("Organic CR",                   f"{organic_cr:.2%}")
    fc3.metric("Event CR",                     f"{event_cr:.2%}")
    fc3.metric("CR Delta (Event − Organic)",   f"{cr_delta:+.2%}")

    st.markdown("---")
    sig_name = st.text_input("Name this Signature", f"Shock {ds_start}→{ds_end}")
    if st.button("💾 Save Signature to Library"):
        st.session_state.shock_library.append({
            "id":          str(time.time()),
            "name":        sig_name,
            "duration":    ds_dur,
            "orig_start":  ds_start,
            "orig_end":    ds_end,
            "organic_cr":  organic_cr,
            "event_cr":    event_cr,
            "cr_delta":    cr_delta,
            "tot_delta_c": tot_delta_c,
            "tot_delta_q": tot_delta_q,
            "tot_delta_s": tot_delta_s,
            "daily_abs_c": shock_raw["delta_c"].tolist(),
            "daily_abs_q": shock_raw["delta_q"].tolist(),
            "daily_abs_s": shock_raw["delta_s"].tolist(),
            "daily_pct_c": (shock_raw["delta_c"] / floor_c).fillna(0).tolist()
                           if floor_c > 0 else [0] * len(shock_raw),
            "daily_pct_q": (shock_raw["delta_q"] / floor_q).fillna(0).tolist()
                           if floor_q > 0 else [0] * len(shock_raw),
            "daily_pct_s": (shock_raw["delta_s"] / floor_s).fillna(0).tolist()
                           if floor_s > 0 else [0] * len(shock_raw),
        })
        st.success(f"'{sig_name}' saved to library.")
        st.rerun()

    # ── Signature Library ──
    st.markdown("---")
    st.subheader("📚 Signature Library & Re-Injection")
    if not st.session_state.shock_library:
        st.info("Library is empty. Extract a shock above to get started.")
        return

    for sig in st.session_state.shock_library:
        with st.expander(
                f"📦 {sig['name']}  |  {sig['duration']} days  "
                f"|  +{sig['tot_delta_c']:,.0f} Clicks"):
            d1, d2 = st.columns(2)
            with d1:
                st.markdown(f"**Original:** {sig['orig_start']} → {sig['orig_end']}")
                st.metric("Δ Clicks",   f"+{sig['tot_delta_c']:,.0f}")
                st.metric("Δ Orders",   f"+{sig['tot_delta_q']:,.0f}")
                st.metric("Δ Sales",    f"+€{sig['tot_delta_s']:,.0f}")
                st.metric("Organic CR", f"{sig['organic_cr']:.2%}")
                st.metric("Event CR",   f"{sig['event_cr']:.2%}")
                st.metric("CR Delta",   f"{sig['cr_delta']:+.2%}")

                inj_date = st.date_input(
                    "Inject at new start date",
                    date(2026, 11, 20), key=f"inj_d_{sig['id']}")
                inj_mode = st.radio(
                    "Injection Mode",
                    ["Absolute Volume (exact historical units)",
                     "Relative Baseline Multiplier (scale to new base)"],
                    key=f"inj_m_{sig['id']}")
                actual_mode = "Absolute Volume" if "Absolute" in inj_mode else "Relative"

                if st.button("💉 Inject Signature", key=f"inj_{sig['id']}"):
                    st.session_state.event_log.append({
                        "type":       "reapplied_shock",
                        "name":       sig["name"],
                        "mode":       actual_mode,
                        "new_start":  inj_date,
                        "duration":   sig["duration"],
                        "daily_abs_c": sig["daily_abs_c"],
                        "daily_abs_q": sig["daily_abs_q"],
                        "daily_abs_s": sig["daily_abs_s"],
                        "daily_pct_c": sig["daily_pct_c"],
                        "daily_pct_q": sig["daily_pct_q"],
                        "daily_pct_s": sig["daily_pct_s"],
                    })
                    st.rerun()

            with d2:
                new_end  = inj_date + timedelta(days=sig["duration"] - 1)
                pmask    = (df["Date"].dt.date >= inj_date) & (df["Date"].dt.date <= new_end)
                if pmask.sum() > 0:
                    v_n   = min(pmask.sum(), sig["duration"])
                    df_pv = df[pmask].iloc[:v_n].copy()
                    if actual_mode == "Absolute Volume":
                        df_pv["Inj"] = np.array(sig["daily_abs_c"][:v_n])
                    else:
                        df_pv["Inj"] = df_pv["Clicks_Base"] * np.array(sig["daily_pct_c"][:v_n])

                    fig_pv = go.Figure()
                    fig_pv.add_trace(go.Bar(
                        x=df_pv["Date"], y=df_pv["Inj"],
                        name="Injected Clicks", marker_color="navy"))
                    fig_pv.add_trace(go.Scatter(
                        x=df_pv["Date"], y=df_pv["Clicks_Base"],
                        mode="lines", name="Baseline",
                        line=dict(color="gray", dash="dot")))
                    fig_pv.update_layout(
                        barmode="overlay", height=250,
                        margin=dict(l=0, r=0, t=30, b=0),
                        title="Injection Preview")
                    st.plotly_chart(fig_pv, use_container_width=True)


# ── Audit & Gap Attribution ──────────────────────────────────────────────────

def _render_audit(df, pure_dna, adj_c, adj_q, adj_s, t_start, t_end):
    st.subheader("Simulation Audit & Gap Attribution")

    if not st.session_state.event_log:
        st.info("No events logged yet.")
        return

    tgt_met    = ("Sales"
                  if st.session_state.target_metric in ["CR", "AOV"]
                  else st.session_state.target_metric)
    base_vol   = eval_events(
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
        cols = st.columns([0.5, 1.5, 3.5, 2, 2, 0.7, 0.7])

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
        pct_gap = (added / total_gap) * 100

        if ev["type"] == "shock":
            cols[0].write("🚀"); cols[1].write("Campaign")
            cols[2].write(f"{ev['shape']} | {ev['str']*100:.0f}% | {ev['start']} → {ev['end']}")
            cols[3].write("Post-Trial")
        elif ev["type"] == "custom_drag":
            cols[0].write("🖱️"); cols[1].write("DNA Drag")
            cols[2].write(f"{ev.get('level','?')} {ev['target']} × {ev['lift']:.2f}")
            cols[3].write(ev.get("scope", "post_trial").replace("_", " ").title())
        elif ev["type"] == "swap":
            cols[0].write("🔄"); cols[1].write("DNA Swap")
            cols[2].write(f"{ev.get('level','?')} {ev['a']} ↔ {ev['b']}")
            cols[3].write(ev.get("scope", "post_trial").replace("_", " ").title())
        elif ev["type"] == "reapplied_shock":
            cols[0].write("💉"); cols[1].write("Re-Injection")
            cols[2].write(f"{ev['name']} | {ev['mode']} | from {ev['new_start']}")
            cols[3].write("Post-Trial")

        if added >= 0:
            cols[4].success(f"+{added:,.0f} ({pct_gap:.1f}%)")
        else:
            cols[4].error(f"{added:,.0f} ({pct_gap:.1f}%)")

        # Shift button for standard shocks
        if ev["type"] == "shock":
            if cols[5].button("↔", key=f"shift_{i}", help="Shift dates"):
                st.session_state.shift_target_idx = i
                st.rerun()
        else:
            cols[5].write("")

        if cols[6].button("❌", key=f"del_{i}"):
            st.session_state.event_log.pop(i)
            st.session_state.shift_target_idx = None
            st.rerun()

    # ── Inline shift UI ──
    if st.session_state.shift_target_idx is not None:
        idx_s = st.session_state.shift_target_idx
        if idx_s < len(st.session_state.event_log):
            ev_s  = st.session_state.event_log[idx_s]
            dur_s = (ev_s["end"] - ev_s["start"]).days
            st.markdown("---")
            st.markdown(
                f"**Shifting:** {ev_s['shape']} shock ({ev_s['start']} → {ev_s['end']})")
            new_start_s = st.date_input(
                "New Start Date", ev_s["start"], key="shift_new_start")
            sc1, sc2 = st.columns(2)
            if sc1.button("✅ Confirm Shift"):
                st.session_state.event_log[idx_s]["start"] = new_start_s
                st.session_state.event_log[idx_s]["end"]   = new_start_s + timedelta(days=dur_s)
                st.session_state.shift_target_idx = None
                st.rerun()
            if sc2.button("✖ Cancel"):
                st.session_state.shift_target_idx = None
                st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear Entire Event Log"):
        st.session_state.event_log        = []
        st.session_state.shift_target_idx = None
        st.rerun()
