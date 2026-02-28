import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.fmt import _fmt, color_neg


def render_dashboard(df, profiles, yearly_kpis, sel_brands, res_level, time_col,
                     base_cr, base_aov):
    """Render the Main Dashboard page (3 tabs)."""
    tab1, tab2, tab3 = st.tabs([
        "📈 Projection Overview",
        "🎯 Goal Tracker",
        "🧬 Market DNA Profile",
    ])

    # ── Tab 1: Projection Overview ──────────────────────────────────────────
    with tab1:
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
                .replace([float("inf"), float("-inf")], 0)
                .fillna(0)
            )
            agg_df[f"AOV{pfx}"] = (
                (agg_df[f"Sales{pfx}"] / agg_df[f"Qty{pfx}"])
                .replace([float("inf"), float("-inf")], 0)
                .fillna(0)
            )

        met = st.selectbox("Select View Metric", ["Sales", "Clicks", "Qty", "CR", "AOV"])
        fig = go.Figure()

        if st.session_state.affect_future and st.session_state.event_log:
            fig.add_trace(go.Scatter(
                x=agg_df["Date"], y=agg_df[f"{met}_Base"],
                mode="lines", line=dict(color="gray", dash="dot"),
                name="Baseline (Before)"))
            fig.add_trace(go.Scatter(
                x=agg_df["Date"], y=agg_df[f"{met}_Sim_Max"],
                mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=agg_df["Date"], y=agg_df[f"{met}_Sim_Min"],
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(0,0,128,0.15)",
                name="+/- 15% Range (After)"))
            fig.add_trace(go.Scatter(
                x=agg_df["Date"], y=agg_df[f"{met}_Sim"],
                mode="lines+markers", line=dict(color="navy", width=3),
                name="Forecast (After Events)"))
        else:
            fig.add_trace(go.Scatter(
                x=agg_df["Date"], y=agg_df[f"{met}_Base_Max"],
                mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=agg_df["Date"], y=agg_df[f"{met}_Base_Min"],
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(128,128,128,0.15)",
                name="+/- 15% Range (Baseline)"))
            fig.add_trace(go.Scatter(
                x=agg_df["Date"], y=agg_df[f"{met}_Base"],
                mode="lines+markers", line=dict(color="gray", width=3),
                name="Baseline (No Events)"))

        fig.update_layout(
            template="plotly_white",
            title=f"{res_level} Forecast: {met}",
            xaxis_title="Date", yaxis_title=met,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Goal Tracker ─────────────────────────────────────────────────
    with tab2:
        st.subheader("🎯 Target Translation")
        calculated_target = st.session_state.target_val
        target_metric_raw = "sales"
        metric_map = {
            "sales": "Sales", "quantity": "Quantity", "clicks": "Clicks",
            "cr": "CR", "aov": "AOV",
        }

        if len(sel_brands) == 1:
            st.info(f"📈 **Single Brand Mode:** {sel_brands[0].title()}")
            brand_hist = yearly_kpis[yearly_kpis["brand"] == sel_brands[0]].sort_values("Year")

            h1, h2, h3 = st.columns(3)
            hist_year         = h1.selectbox("Base Year", brand_hist["Year"].unique())
            target_metric_raw = h2.selectbox(
                "Historical Metric", ["sales", "quantity", "clicks", "cr", "aov"])
            growth_pct        = h3.number_input("Growth vs Year (%)", value=5.0, step=1.0)

            fig_h = px.bar(
                brand_hist, x="Year", y=target_metric_raw,
                title=f"Historical {metric_map[target_metric_raw]}",
                text_auto=".2s", color_discrete_sequence=["indigo"],
            )
            fig_h.update_layout(template="plotly_white", height=300)
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

        col_d1, col_d2 = st.columns(2)
        st.session_state.tgt_start = col_d1.date_input(
            "Target Period Start", st.session_state.tgt_start)
        st.session_state.tgt_end   = col_d2.date_input(
            "Target Period End",   st.session_state.tgt_end)

        col_m1, col_m2, col_m3 = st.columns(3)
        m_opts = ["Sales", "Quantity", "Clicks", "CR", "AOV"]
        default_idx = (
            m_opts.index(metric_map[target_metric_raw])
            if len(sel_brands) == 1 and metric_map[target_metric_raw] in m_opts
            else m_opts.index(st.session_state.target_metric)
            if st.session_state.target_metric in m_opts else 0
        )
        st.session_state.target_metric = col_m1.selectbox(
            "Final Target Metric", m_opts, index=default_idx)

        volume_driver = "Traffic (Clicks)"
        if st.session_state.target_metric in ["Sales", "Quantity"]:
            d_opts = (
                ["Traffic (Clicks)", "Conversion Rate (CR)", "Average Order Value (AOV)"]
                if st.session_state.target_metric == "Sales"
                else ["Traffic (Clicks)", "Conversion Rate (CR)"]
            )
            volume_driver = col_m2.selectbox("Scale via:", d_opts)

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

        df_tgt = df[
            (df["Date"].dt.date >= st.session_state.tgt_start) &
            (df["Date"].dt.date <= st.session_state.tgt_end)
        ].copy()

        if df_tgt.empty:
            st.warning("Target period has no data in the projection year.")
            return

        tgt_sales_base  = df_tgt["Sales_Base"].sum()
        tgt_qty_base    = df_tgt["Qty_Base"].sum()
        tgt_clicks_base = df_tgt["Clicks_Base"].sum()
        tgt_eff_aov     = tgt_sales_base / tgt_qty_base    if tgt_qty_base    > 0 else base_aov
        tgt_eff_cr      = tgt_qty_base   / tgt_clicks_base if tgt_clicks_base > 0 else base_cr

        # ── Compute needed volumes by driver ────────────────────────────────
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

        # ── KPI Matrix 5 × (Needed / Before / After) ────────────────────────
        kpi_cols = st.columns(5)
        labels = ["Traffic", "Orders", "Revenue", "CR", "AOV"]
        needed = [needed_clicks, needed_qty, needed_sales, needed_cr, needed_aov]
        before = [tgt_clicks_base, tgt_qty_base, tgt_sales_base, tgt_eff_cr, tgt_eff_aov]
        after  = [s_c, s_q, s_s, s_cr, s_aov]

        for col, lbl, n_v, b_v, a_v in zip(kpi_cols, labels, needed, before, after):
            col.metric(f"Needed {lbl}", _fmt(lbl, n_v))
            col.caption(f"**Before:** {_fmt(lbl, b_v)}")
            if st.session_state.affect_future:
                col.caption(f"**After:** {_fmt(lbl, a_v)}")

        # ── Period-by-period chart + table ──────────────────────────────────
        agg_tgt = df_tgt.groupby(time_col).agg({
            "Date": "first",
            "Clicks_Sim": "sum", "Qty_Sim": "sum", "Sales_Sim": "sum",
            "Clicks_Base": "sum", "Qty_Base": "sum", "Sales_Base": "sum",
        }).reset_index()

        for col_n, total in [("Needed_Sales", needed_sales),
                              ("Needed_Qty",   needed_qty),
                              ("Needed_Clicks",needed_clicks)]:
            base_c = col_n.replace("Needed_", "") + "_Base"
            bt     = agg_tgt[base_c].sum()
            agg_tgt[col_n] = total * (agg_tgt[base_c] / bt) if bt > 0 else 0

        for sfx in ["Base", "Sim"]:
            agg_tgt[f"Gap_Sales_{sfx}"]  = agg_tgt[f"Sales_{sfx}"]  - agg_tgt["Needed_Sales"]
            agg_tgt[f"Gap_Qty_{sfx}"]    = agg_tgt[f"Qty_{sfx}"]    - agg_tgt["Needed_Qty"]
            agg_tgt[f"Gap_Clicks_{sfx}"] = agg_tgt[f"Clicks_{sfx}"] - agg_tgt["Needed_Clicks"]

        st.markdown("---")
        gap_m = st.radio("Visualize Tracking For:", ["Sales", "Qty", "Clicks"], horizontal=True)

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=agg_tgt["Date"], y=agg_tgt[f"Needed_{gap_m}"],
            mode="lines", line=dict(color="red", dash="dash"), name="Target"))
        fig_ts.add_trace(go.Scatter(
            x=agg_tgt["Date"], y=agg_tgt[f"{gap_m}_Base"],
            mode="lines", line=dict(color="gray", dash="dot"), name="Before"))
        if st.session_state.affect_future:
            fig_ts.add_trace(go.Scatter(
                x=agg_tgt["Date"], y=agg_tgt[f"{gap_m}_Sim"],
                mode="lines+markers", line=dict(color="navy", width=2), name="After"))
        fig_ts.update_layout(
            template="plotly_white", height=350,
            title=f"Needed vs Forecasted — {gap_m}")
        st.plotly_chart(fig_ts, use_container_width=True)

        fig_gap = go.Figure()
        fig_gap.add_trace(go.Bar(
            x=agg_tgt["Date"], y=agg_tgt[f"Gap_{gap_m}_Base"],
            name="Gap (Before)", marker_color="lightgray"))
        if st.session_state.affect_future:
            colors = [
                "#ff4b4b" if v < 0 else "#21c354"
                for v in agg_tgt[f"Gap_{gap_m}_Sim"]
            ]
            fig_gap.add_trace(go.Bar(
                x=agg_tgt["Date"], y=agg_tgt[f"Gap_{gap_m}_Sim"],
                name="Gap (After)", marker_color=colors))
        fig_gap.update_layout(
            template="plotly_white", barmode="group", height=280,
            title=f"{gap_m} Surplus / Shortfall")
        st.plotly_chart(fig_gap, use_container_width=True)

        # Data table — volumes only (no CR/AOV per spec)
        if st.session_state.affect_future:
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
        try:
            st.dataframe(disp_df.style.map(color_neg, subset=gap_cols), use_container_width=True)
        except AttributeError:
            st.dataframe(
                disp_df.style.applymap(color_neg, subset=gap_cols), use_container_width=True)

    # ── Tab 3: Market DNA Profile ────────────────────────────────────────────
    with tab3:
        st.subheader("Current Operational DNA")
        dna_plot = df.groupby(time_col).agg({
            "idx_clicks_pure": "mean", "idx_cr_pure": "mean", "idx_aov_pure": "mean",
            "idx_clicks_pretrial": "mean", "idx_cr_pretrial": "mean", "idx_aov_pretrial": "mean",
            "idx_clicks_work": "mean", "idx_cr_work": "mean", "idx_aov_work": "mean",
        }).reset_index()

        fig_dna = go.Figure()
        for m, name in [("idx_clicks_pure", "Clicks"),
                         ("idx_cr_pure", "CR"), ("idx_aov_pure", "AOV")]:
            fig_dna.add_trace(go.Scatter(
                x=dna_plot[time_col], y=dna_plot[m],
                mode="lines", line=dict(dash="dot", width=1),
                name=f"{name} (Pure)"))

        if st.session_state.affect_future and st.session_state.event_log:
            for m, name in [("idx_clicks_pretrial", "Clicks"),
                             ("idx_cr_pretrial", "CR"), ("idx_aov_pretrial", "AOV")]:
                fig_dna.add_trace(go.Scatter(
                    x=dna_plot[time_col], y=dna_plot[m],
                    mode="lines", line=dict(dash="dash"),
                    name=f"{name} (Pre-Trial)"))
            for m, name in [("idx_clicks_work", "Clicks"),
                             ("idx_cr_work", "CR"), ("idx_aov_work", "AOV")]:
                fig_dna.add_trace(go.Scatter(
                    x=dna_plot[time_col], y=dna_plot[m],
                    mode="lines", name=f"{name} (Full Work)"))

        fig_dna.update_layout(
            template="plotly_white",
            title=f"DNA Profile at {res_level} Resolution")
        st.plotly_chart(fig_dna, use_container_width=True)
